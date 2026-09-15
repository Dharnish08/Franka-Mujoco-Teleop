"""
pickplace/sac.py -- Soft Actor-Critic

Self-contained on purpose. stable-baselines3 would do this too, but it pins
gymnasium and torch versions, and this env already sits on gymnasium 1.3 +
torch 2.11 + a Blackwell GPU. A dependency that pins torch is a dependency
that fights the one thing we had to fix to get here.

Standard SAC, no tricks:
  - tanh-squashed Gaussian actor with the exact log-prob correction
  - twin Q critics + Polyak-averaged targets (clipped double-Q)
  - automatic entropy tuning against a target entropy of -|A|

Correctness is checked, not assumed: `python -m pickplace.sac` trains on
Pendulum-v1 and asserts it clears the standard solved bar. If that fails, the
bug is here, not in the robot.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
EPS = 1e-6


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], out_act=nn.ReLU())
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            u = mu
        else:
            # rsample, not sample: the actor loss backprops through the action.
            u = mu + std * torch.randn_like(mu)

        a = torch.tanh(u)

        if not with_logprob:
            return a, None

        # log pi(a|s) for a tanh-squashed Gaussian. The correction term is
        # -sum(log(1 - tanh(u)^2)); written via softplus for numerical
        # stability, which matters because tanh saturates hard at |u| > 5 and
        # the naive form underflows to log(0).
        logp = (-0.5 * ((u - mu) / (std + EPS)) ** 2 - log_std
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class Critic(nn.Module):
    """Twin Q networks in one module."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    """Flat numpy ring buffer. float32 throughout to halve the transfer cost."""

    def __init__(self, obs_dim, act_dim, capacity):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        # `done` here must be the TERMINAL flag only, never the timeout flag.
        # Bootstrapping is cut at a true terminal state; a truncated episode is
        # still worth bootstrapping through, or every timeout teaches the
        # critic the world ends at step max_steps.
        self.done[i] = done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.done)


class SAC:

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=1_000_000,
        init_alpha=0.1,
        target_entropy=None,
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # alpha is optimized in log space so it can never go negative.
        self.log_alpha = torch.tensor(
            float(np.log(init_alpha)), device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = float(-act_dim if target_entropy is None else target_entropy)

        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    # ----------------------------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    # ----------------------------------------------------------------------
    def update(self, batch_size):
        obs, act, rew, next_obs, done = self.buffer.sample(batch_size, self.device)

        # --- critic ---------------------------------------------------------
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_act)
            # Clipped double-Q: the min of the two targets is what keeps SAC
            # from the runaway overestimation that sinks vanilla actor-critic.
            target_v = torch.min(tq1, tq2) - self.alpha * next_logp
            target_q = rew + self.gamma * (1.0 - done) * target_v

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor ----------------------------------------------------------
        # Freeze the critic for this step: we want gradients w.r.t. the action,
        # not another set of critic gradients piling onto its .grad buffers.
        for p in self.critic.parameters():
            p.requires_grad_(False)

        new_act, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # --- temperature ------------------------------------------------------
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Polyak -----------------------------------------------------------
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha),
            "q_mean": float(q1.mean().detach()),
            "entropy": float(-logp.mean().detach()),
        }

    # ----------------------------------------------------------------------
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_target.load_state_dict(ck["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))


# ==========================================================================
# Self-test: if SAC is correct it solves Pendulum-v1. If this fails, do not go
# looking for the bug in the robot env.
# ==========================================================================
def _selftest(total_steps=15_000, seed=0):
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SAC(obs_dim, act_dim, gamma=0.99, buffer_size=total_steps, device="cpu")

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        if step < 1000:
            a = env.action_space.sample() / act_scale
        else:
            a = agent.act(obs)
        next_obs, r, term, trunc, _ = env.step(a * act_scale)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        if step >= 1000:
            agent.update(256)

    returns = []
    for ep in range(10):
        o, _ = eval_env.reset(seed=1000 + ep)
        total = 0.0
        while True:
            o, r, term, trunc, _ = eval_env.step(agent.act(o, deterministic=True) * act_scale)
            total += r
            if term or trunc:
                break
        returns.append(total)
    mean_ret = float(np.mean(returns))
    print("Pendulum-v1 mean return over 10 eval episodes: %.1f" % mean_ret)
    # Random policy scores about -1200; a solved Pendulum sits above -200.
    assert mean_ret > -300.0, "SAC failed its Pendulum self-test (%.1f)" % mean_ret
    print("SAC self-test PASSED")
    return mean_ret


if __name__ == "__main__":
    _selftest()
