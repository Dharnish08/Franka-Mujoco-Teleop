"""
pickplace/train_reach.py -- train the goal-conditioned SAC reach policy.

    python -m pickplace.train_reach
    python -m pickplace.train_reach --steps 150000 --device cuda

This trains phases 1, 3 and 4 of the hybrid in one policy. Evaluation is
reported PER GOAL CLUSTER, not just as an aggregate, because the clusters map
onto different phases and an aggregate hides a phase that is quietly broken:
a policy at 95% overall could be 100% on cube and 60% on drop, which would
show up later as a mysterious transfer failure.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .reach_env import GoalReachEnv
from .sac import SAC
from . import config as C


# The clusters evaluated separately, and the phase each one stands in for.
EVAL_CLUSTERS = {
    "cube":    "phase 1 reach",
    "drop":    "phase 4 transfer",
    "home":    "phase 3 retract",
    "scatter": "in-between coverage",
}


def sample_cluster_goal(env, kind, rng):
    if kind == "cube":
        return np.array([rng.uniform(*C.CUBE_X_RANGE),
                         rng.uniform(*C.CUBE_Y_RANGE),
                         rng.uniform(0.02, 0.10)])
    if kind == "drop":
        return C.DROP_POINT + np.array([rng.uniform(-.04, .04),
                                        rng.uniform(-.04, .04),
                                        rng.uniform(.05, .25)])
    if kind == "home":
        return env.home_ee_pos + rng.uniform(-0.05, 0.05, size=3)
    return np.array([rng.uniform(0.25, 0.60),
                     rng.uniform(-0.30, 0.40),
                     rng.uniform(0.05, 0.45)])


def evaluate(agent, env, episodes_per_cluster, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for kind in EVAL_CLUSTERS:
        succ, steps, dists = 0, [], []
        for ep in range(episodes_per_cluster):
            goal = sample_cluster_goal(env, kind, rng)
            obs, _ = env.reset(goal=goal, hold_gripper=float(rng.random() < 0.4))
            for t in range(env.max_steps):
                obs, r, term, trunc, info = env.step(agent.act(obs, deterministic=True))
                if term or trunc:
                    break
            succ += int(info["is_success"])
            steps.append(t + 1)
            dists.append(info["dist"])
        out[kind] = {
            "success": succ / episodes_per_cluster,
            "mean_steps": float(np.mean(steps)),
            "mean_dist": float(np.mean(dists)),
        }
    out["overall"] = {
        "success": float(np.mean([v["success"] for k, v in out.items() if k in EVAL_CLUSTERS])),
        "mean_steps": float(np.mean([v["mean_steps"] for k, v in out.items() if k in EVAL_CLUSTERS])),
        "mean_dist": float(np.mean([v["mean_dist"] for k, v in out.items() if k in EVAL_CLUSTERS])),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=C.SAC_TOTAL_STEPS)
    p.add_argument("--warmup", type=int, default=C.SAC_WARMUP_STEPS)
    p.add_argument("--batch-size", type=int, default=C.SAC_BATCH_SIZE)
    p.add_argument("--eval-every", type=int, default=C.SAC_EVAL_EVERY)
    p.add_argument("--eval-episodes", type=int, default=C.SAC_EVAL_EPISODES)
    p.add_argument("--updates-per-step", type=int, default=C.SAC_UPDATES_PER_STEP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(C.SAC_CKPT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = GoalReachEnv(seed=args.seed)
    eval_env = GoalReachEnv(seed=args.seed + 10_000)

    agent = SAC(
        obs_dim=C.SAC_OBS_DIM,
        act_dim=C.SAC_ACTION_DIM,
        hidden=C.SAC_HIDDEN,
        lr=C.SAC_LR,
        gamma=C.SAC_GAMMA,
        tau=C.SAC_TAU,
        buffer_size=C.SAC_BUFFER_SIZE,
        init_alpha=C.SAC_INIT_ALPHA,
        target_entropy=C.SAC_TARGET_ENTROPY,
        device=args.device,
    )
    print("SAC reach: obs %d act %d on %s" % (C.SAC_OBS_DIM, C.SAC_ACTION_DIM, agent.device))
    print("handoff radius %.3f m, budget %d steps" % (C.HANDOFF_RADIUS, C.REACH_MAX_STEPS))

    log_path = out_dir / "train_log.csv"
    log_f = open(log_path, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["step", "ep_return", "ep_success", "critic_loss", "actor_loss",
                    "alpha", "q_mean", "entropy",
                    *["eval_%s" % k for k in EVAL_CLUSTERS], "eval_overall"])

    obs, _ = env.reset()
    ep_return, ep_len = 0.0, 0
    recent_returns, recent_success = [], []
    best_overall = -1.0
    stats = {}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1

        # Only `terminated` cuts the bootstrap. Passing `truncated` here is the
        # classic time-limit bug: the critic learns the episode really ends at
        # the horizon and the value function collapses near it.
        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        if terminated or truncated:
            recent_returns.append(ep_return)
            recent_success.append(float(info["is_success"]))
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        if step > args.warmup:
            for _ in range(args.updates_per_step):
                stats = agent.update(args.batch_size)

        if step % 2_000 == 0:
            sps = step / (time.time() - t0)
            print("[%7d] ret %7.2f  succ %.2f  alpha %.3f  q %7.2f  ent %6.2f  %.0f steps/s"
                  % (step,
                     float(np.mean(recent_returns[-50:])) if recent_returns else 0.0,
                     float(np.mean(recent_success[-50:])) if recent_success else 0.0,
                     stats.get("alpha", 0.0), stats.get("q_mean", 0.0),
                     stats.get("entropy", 0.0), sps))

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(agent, eval_env, args.eval_episodes, seed=args.seed)
            line = "  ".join("%s %.0f%%" % (k, 100 * ev[k]["success"]) for k in EVAL_CLUSTERS)
            print("[eval @%d] %s  |  OVERALL %.1f%%  mean_steps %.1f"
                  % (step, line, 100 * ev["overall"]["success"], ev["overall"]["mean_steps"]))

            log_w.writerow([step,
                            float(np.mean(recent_returns[-50:])) if recent_returns else 0.0,
                            float(np.mean(recent_success[-50:])) if recent_success else 0.0,
                            stats.get("critic_loss", 0.0), stats.get("actor_loss", 0.0),
                            stats.get("alpha", 0.0), stats.get("q_mean", 0.0),
                            stats.get("entropy", 0.0),
                            *[ev[k]["success"] for k in EVAL_CLUSTERS],
                            ev["overall"]["success"]])
            log_f.flush()

            agent.save(out_dir / "sac_reach_last.pt")
            if ev["overall"]["success"] > best_overall:
                best_overall = ev["overall"]["success"]
                agent.save(out_dir / "sac_reach_best.pt")
                print("           new best (%.1f%%) -> sac_reach_best.pt" % (100 * best_overall))

    log_f.close()
    env.close()
    eval_env.close()
    print("\nDone in %.1f min. Best overall success %.1f%%. Checkpoints in %s"
          % ((time.time() - t0) / 60, 100 * best_overall, out_dir))


if __name__ == "__main__":
    main()
