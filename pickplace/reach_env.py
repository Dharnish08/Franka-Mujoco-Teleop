"""
pickplace/reach_env.py -- GoalReachEnv

Gymnasium env for the SAC half of the hybrid. One goal-conditioned reach
policy serves three of the five phases:

    phase 1 REACH     goal = cube position,        gripper open
    phase 3 RETRACT   goal = home EE position,     gripper closed (holding)
    phase 4 TRANSFER  goal = above the drop point, gripper closed (holding)

Training one goal-conditioned policy instead of three specialists is not just
tidiness. Phases 3 and 4 begin from wherever the grasp happened to end, which
is a state distribution we cannot enumerate in advance -- so the start state is
randomized here rather than always being the home pose. A policy trained only
from home would be out of distribution the moment it is handed a post-grasp
pose, which is precisely when it matters.

OBSERVATION (25-D, privileged -- no pixels)
    [cos(q) x7, sin(q) x7, ee_pos x3, goal x3, goal-ee x3, dist, gripper]

    cos/sin rather than raw angles: it removes the wrap discontinuity and
    bounds every input feature to [-1, 1] without a running normalizer.

ACTION (3-D)
    [dx, dy, dz] in [-1, 1], passed straight through to PickPlaceEnv with the
    held gripper value appended. Same units as every other consumer.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import PickPlaceEnv
from . import config as C


# Goal sampling mixture. Weights must sum to 1.0.
#   cube     -- phase 1 targets, matching the teleop cube spawn band
#   drop     -- phase 4 targets, above the fixed plate
#   home     -- phase 3 targets, the rest pose the arm returns to
#   scatter  -- generic workspace filler, so the value function does not become
#               a lookup table over three clusters and fall apart in between
GOAL_MIX = {"cube": 0.40, "drop": 0.20, "home": 0.15, "scatter": 0.25}


class GoalReachEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, seed=None, env=None, max_steps: int = C.REACH_MAX_STEPS):
        super().__init__()
        # enable_rendering=False: this env never looks at pixels, and two
        # MuJoCo Renderers per worker is pure overhead during RL.
        self.sim = env if env is not None else PickPlaceEnv(enable_rendering=False)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(C.SAC_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.SAC_ACTION_DIM,), dtype=np.float32
        )

        self._goal = np.zeros(3)
        self._hold_gripper = 0.0
        self._t = 0
        self._prev_dist = 0.0

        # Home EE position, resolved once from the model rather than hardcoded,
        # so it cannot drift from PANDA_HOME_QPOS.
        self.sim.reset()
        self.home_ee_pos = self.sim.get_ee_pose()[0].copy()

    # ======================================================================
    # Goal + start-state sampling
    # ======================================================================
    def sample_goal(self):
        r = self._rng.random()
        cum = 0.0
        for kind, w in GOAL_MIX.items():
            cum += w
            if r < cum:
                break

        if kind == "cube":
            return np.array([
                self._rng.uniform(*C.CUBE_X_RANGE),
                self._rng.uniform(*C.CUBE_Y_RANGE),
                self._rng.uniform(0.02, 0.10),
            ])
        if kind == "drop":
            return C.DROP_POINT + np.array([
                self._rng.uniform(-0.04, 0.04),
                self._rng.uniform(-0.04, 0.04),
                self._rng.uniform(0.05, 0.25),
            ])
        if kind == "home":
            return self.home_ee_pos + self._rng.uniform(-0.05, 0.05, size=3)
        # scatter. z is capped at 0.45, well below the home EE height of 0.52,
        # on purpose: apply_action holds the wrist quaternion fixed at the
        # home downward-pointing orientation, and goals that are both high and
        # close to the base cannot be hit under that constraint -- the position
        # solve fights the orientation term. Measured, not assumed: a scripted
        # P-controller misses (0.25,-0.30,0.55) by 0.49m at radius 0.68 while
        # hitting (0.60,+0.40,0.55) at radius 0.91. Sampling goals no policy
        # can reach only teaches the critic that some states are hopeless.
        return np.array([
            self._rng.uniform(0.25, 0.60),
            self._rng.uniform(-0.30, 0.40),
            self._rng.uniform(0.05, 0.45),
        ])

    def _in_start_workspace(self, ee):
        """Is this a start pose the arm can actually recover from?

        The radius cap is the load-bearing term. Measured: a scripted
        controller commanded straight back to home gets stuck 5/120 times, and
        every single stuck case had the EE at radius > 0.84 from the base --
        the arm stretched nearly straight, where the Jacobian is near-singular
        radially and damped least-squares (damping=0.05) cannot pull it back
        in. Home itself sits at radius 0.755, so the cap has to sit above that
        and below 0.84.

        Those poses are also unreachable in the real task -- phases 3 and 4
        start from a post-grasp pose over the cube band, x <= 0.55 -- so
        excluding them costs no coverage and removes a ~4% unrecoverable floor
        that SAC would otherwise have to eat.
        """
        x, y, z = ee
        return (
            0.20 <= x <= 0.65
            and -0.35 <= y <= 0.45
            and 0.03 <= z <= 0.60
            and float(np.linalg.norm(ee)) <= 0.80
        )

    def _randomize_start(self):
        """Walk the arm away from home along a random direction.

        Phases 3 and 4 hand this policy a post-grasp pose, never the home pose.
        Training only from home would make every one of those starts
        off-distribution. A short random walk through the SAME action interface
        the policy uses guarantees the start states are reachable and
        physically consistent -- which teleporting qpos would not.

        The walk stops early the moment it leaves the recoverable workspace,
        leaving the arm at the last good pose.
        """
        n = int(self._rng.integers(0, 40))
        if n == 0:
            return
        direction = self._rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-9
        for _ in range(n):
            jitter = self._rng.normal(scale=0.3, size=3)
            a = np.clip(direction + jitter, -1.0, 1.0)
            self.sim.apply_action(np.concatenate([a, [self._hold_gripper]]))
            if not self._in_start_workspace(self.sim.get_ee_pose()[0]):
                # Back off one step so we end inside the envelope, not on its
                # far side. The EE target is persistent, so it must be pulled
                # back too or the next action resumes from outside.
                self.sim.apply_action(np.concatenate([-a, [self._hold_gripper]]))
                return

    # ======================================================================
    def reset(self, seed=None, options=None, goal=None, hold_gripper=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        self.sim.reset(seed=int(self._rng.integers(0, 2**31 - 1)))

        # Gripper is held for the whole episode: open for a cube approach,
        # closed for a transport. Sampled so the policy sees both.
        if hold_gripper is None:
            hold_gripper = options.get("hold_gripper")
        self._hold_gripper = (
            float(hold_gripper) if hold_gripper is not None
            else float(self._rng.random() < 0.4)
        )

        self._randomize_start()

        g = goal if goal is not None else options.get("goal")
        self._goal = np.asarray(g, dtype=np.float64) if g is not None else self.sample_goal()

        self._t = 0
        self._prev_dist = self.sim.ee_to(self._goal)
        return self._obs(), {"goal": self._goal.copy()}

    # ======================================================================
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.sim.apply_action(np.concatenate([a, [self._hold_gripper]]))
        self._t += 1

        dist = self.sim.ee_to(self._goal)
        success = dist < C.HANDOFF_RADIUS

        # Potential-based progress term plus a small absolute-distance term.
        # Progress alone is scale-free and learns fast; the absolute term stops
        # the agent parking just outside the radius to farm the time penalty
        # into a smaller negative than committing would cost.
        reward = (
            C.REACH_DIST_WEIGHT * (self._prev_dist - dist) * 10.0
            - 0.1 * dist
            - C.REACH_TIME_PENALTY
            - C.REACH_ACTION_PENALTY * float(np.sum(a ** 2))
        )
        if success:
            reward += C.REACH_SUCCESS_BONUS

        self._prev_dist = dist
        truncated = self._t >= self.max_steps

        info = {
            "dist": dist,
            "is_success": bool(success),
            "ee_pos": self.sim.get_ee_pose()[0].copy(),
            "goal": self._goal.copy(),
        }
        # terminated vs truncated kept distinct on purpose: SAC must bootstrap
        # through a timeout but not through a real terminal state.
        return self._obs(), float(reward), bool(success), bool(truncated), info

    # ======================================================================
    def _obs(self):
        q = self.sim.get_joint_positions()
        ee = self.sim.get_ee_pose()[0]
        delta = self._goal - ee
        return np.concatenate([
            np.cos(q), np.sin(q),
            ee, self._goal, delta,
            [np.linalg.norm(delta)],
            [self._hold_gripper],
        ]).astype(np.float32)

    def close(self):
        self.sim.close()
