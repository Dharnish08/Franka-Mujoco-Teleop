"""
pickplace/orchestrator.py -- the hybrid controller.

This is the piece that makes it a hybrid rather than two separate projects.
It runs a five-phase state machine and hands the robot back and forth:

    phase      controller  goal / task                        advances when
    ---------  ----------  ---------------------------------  ------------------------
    reach      SAC         goal = cube position               ee within HANDOFF_RADIUS
    grasp      SmolVLA     "pick up the red cube"             cube held and lifted
    retract    SAC         goal = home EE pose, holding       ee within HANDOFF_RADIUS
    transfer   SAC         goal = above the drop plate        ee within HANDOFF_RADIUS
    place      SmolVLA     "place the red cube ..."           cube on plate, released

WHY THE GRIPPER IS NOT SAC's TO CHOOSE
    The reach policy emits 3-D deltas; the orchestrator appends the gripper
    command. During reach it is open, during retract and transfer it is
    closed because the cube is in hand. Letting a reach policy decide would
    let it drop the cube mid-transfer for no benefit -- there is no decision
    to make, so it is not modelled as one.

WHY EACH PHASE HAS A BUDGET
    A phase that cannot finish must fail loudly and be attributable. Without
    budgets a stalled grasp looks identical to a slow one, and an evaluation
    run reports "0% success" with nothing to say about where it went wrong.
    Every rollout returns the phase it died in.

FAILURE DETECTION MID-PHASE
    Dropping the cube during retract or transfer is checked explicitly rather
    than discovered at the end. Carrying an empty gripper to the drop point
    and carefully releasing nothing wastes the budget and, worse, pollutes the
    statistics with a "place failure" that was really a grasp failure.
"""

import numpy as np

from .env import PickPlaceEnv
from .scripted_expert import PLACE_HOVER
from . import config as C


PHASES = ["reach", "grasp", "retract", "transfer", "place"]


class HybridController:

    def __init__(self, env: PickPlaceEnv, reach_agent, vla, verbose=False):
        """
        reach_agent : anything with .act(obs_vector, deterministic=True) -> (3,)
        vla         : SmolVLAController, or None to substitute the scripted
                      expert (used by the ablation in evaluate.py)
        """
        self.env = env
        self.reach = reach_agent
        self.vla = vla
        self.verbose = verbose
        self.home_ee = None

    # ----------------------------------------------------------------------
    def _reach_obs(self, goal, hold_gripper):
        """Rebuild the 25-D vector GoalReachEnv trains on.

        Kept identical to GoalReachEnv._obs by construction. If these two ever
        drift, SAC receives a permuted observation and fails in a way that
        looks like a policy problem.
        """
        q = self.env.get_joint_positions()
        ee = self.env.get_ee_pose()[0]
        delta = np.asarray(goal) - ee
        return np.concatenate([
            np.cos(q), np.sin(q),
            ee, np.asarray(goal), delta,
            [np.linalg.norm(delta)],
            [hold_gripper],
        ]).astype(np.float32)

    def _run_reach_phase(self, name, goal, hold_gripper, trace, on_step=None):
        budget = C.PHASE_STEP_BUDGET[name]
        for t in range(budget):
            if self.env.ee_to(goal) < C.HANDOFF_RADIUS:
                return True, t
            a3 = self.reach.act(self._reach_obs(goal, hold_gripper), deterministic=True)
            self.env.apply_action(np.concatenate([a3, [hold_gripper]]))
            trace.append(name)
            if on_step is not None:
                on_step(name)

            # Dropped the cube? Stop now; everything after this is wasted.
            if hold_gripper >= 0.5 and not self.env.is_grasped():
                return False, t
        return self.env.ee_to(goal) < C.HANDOFF_RADIUS, budget

    def _run_vla_phase(self, name, task, done_fn, trace, on_step=None):
        budget = C.PHASE_STEP_BUDGET[name]
        self.vla.set_task(task)
        for t in range(budget):
            if done_fn():
                return True, t
            obs = self.env.get_observation(with_images=True)
            a = self.vla.act(obs)
            self.env.apply_action(a)
            trace.append(name)
            if on_step is not None:
                on_step(name)
        return done_fn(), budget

    # ======================================================================
    def run_episode(self, cube_xy=None, seed=None, on_step=None):
        env = self.env
        env.reset(cube_xy=cube_xy, seed=seed)
        if self.home_ee is None:
            self.home_ee = env.get_ee_pose()[0].copy()
        if self.vla is not None:
            self.vla.reset()

        trace = []
        timings = {}
        result = {"success": False, "failed_phase": None, "phases": timings}

        # --- 1 REACH -------------------------------------------------------
        ok, t = self._run_reach_phase("reach", env.get_object_position(), 0.0,
                                      trace, on_step)
        timings["reach"] = t
        if not ok:
            result["failed_phase"] = "reach"
            return self._finish(result, trace)

        # --- 2 GRASP -------------------------------------------------------
        ok, t = self._run_vla_phase("grasp", C.TASK_PICK, env.is_picked,
                                    trace, on_step)
        timings["grasp"] = t
        if not ok:
            result["failed_phase"] = "grasp"
            return self._finish(result, trace)

        # --- 3 RETRACT -----------------------------------------------------
        ok, t = self._run_reach_phase("retract", self.home_ee, 1.0, trace, on_step)
        timings["retract"] = t
        if not ok:
            result["failed_phase"] = "retract"
            return self._finish(result, trace)

        # --- 4 TRANSFER ----------------------------------------------------
        hover = env.get_drop_position() + np.array([0.0, 0.0, PLACE_HOVER])
        ok, t = self._run_reach_phase("transfer", hover, 1.0, trace, on_step)
        timings["transfer"] = t
        if not ok:
            result["failed_phase"] = "transfer"
            return self._finish(result, trace)

        # --- 5 PLACE -------------------------------------------------------
        ok, t = self._run_vla_phase(
            "place", C.TASK_PLACE,
            lambda: env.is_placed() and env.cube_is_static(),
            trace, on_step,
        )
        timings["place"] = t
        if not ok:
            result["failed_phase"] = "place"
            return self._finish(result, trace)

        result["success"] = True
        return self._finish(result, trace)

    # ----------------------------------------------------------------------
    def _finish(self, result, trace):
        env = self.env
        result["total_steps"] = len(trace)
        result["cube_pos"] = env.get_object_position().tolist()
        result["drop_pos"] = env.get_drop_position().tolist()
        result["place_error"] = float(np.linalg.norm(
            env.get_object_position()[:2] - env.get_drop_position()[:2]))
        result["is_grasped"] = env.is_grasped()
        result["is_placed"] = env.is_placed()
        if self.verbose:
            print("  %-8s steps %3d  place_err %.3f  %s"
                  % ("SUCCESS" if result["success"] else "FAIL",
                     result["total_steps"], result["place_error"],
                     result["failed_phase"] or ""))
        return result


# ==========================================================================
class ScriptedVLAStandIn:
    """Drop-in replacement for SmolVLAController that runs the scripted expert.

    This exists so evaluate.py can measure the pipeline with a PERFECT
    manipulation policy. If the hybrid scores 95% with this and 40% with the
    real VLA, the gap is the VLA. If it scores 50% with this too, the problem
    is the handoff or the reach, and no amount of VLA training will fix it.
    That separation is worth the twenty lines.
    """

    def __init__(self, env):
        from .scripted_expert import ScriptedExpert
        self.env = env
        self.expert = ScriptedExpert(env, noise_std=0.0)
        self._task = None
        self._queue = []

    def set_task(self, task):
        if task != self._task:
            self._task = task
            self._queue = []

    def reset(self):
        self._queue = []

    def act(self, obs):
        if not self._queue:
            recorded = []
            if self._task == C.TASK_PICK:
                self._replay_into(self.expert.run_grasp, recorded)
            else:
                self._replay_into(self.expert.run_place, recorded)
            self._queue = recorded or [np.zeros(C.ACTION_DIM, dtype=np.float32)]
        return self._queue.pop(0)

    def _replay_into(self, fn, out):
        """Run the scripted routine on a snapshot, collecting its actions.

        The routine actually steps the simulator, so the state is saved and
        restored around it -- otherwise planning would double-advance physics.
        """
        import mujoco
        env = self.env
        snap_qpos = env.data.qpos.copy()
        snap_qvel = env.data.qvel.copy()
        snap_ctrl = env.data.ctrl.copy()
        snap_target = env._ee_target_pos.copy()
        snap_grip = env._gripper_closed

        fn(record=out.append)

        env.data.qpos[:] = snap_qpos
        env.data.qvel[:] = snap_qvel
        env.data.ctrl[:] = snap_ctrl
        env._ee_target_pos = snap_target
        env._gripper_closed = snap_grip
        mujoco.mj_forward(env.model, env.data)
