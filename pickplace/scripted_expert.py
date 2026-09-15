"""
pickplace/scripted_expert.py -- privileged scripted pick-and-place.

Generates the demonstration data SmolVLA trains on. It reads the cube pose
straight out of the simulator, which is exactly the privileged information the
VLA will NOT have at rollout time -- the expert sees state, the student sees
only pixels plus proprioception. That asymmetry is the whole point: it is
cheap supervision for a policy that has to work without it.

Every action goes through PickPlaceEnv.apply_action, so the demonstrated
actions carry identical semantics to the ones SAC produces and the ones the
teleop episodes recorded. Nothing here reaches into MuJoCo directly.

THE TWO SEGMENTS
    This expert is written as two independently-recordable segments, because
    the hybrid hands SmolVLA control twice and only twice:

      GRASP  starts within HANDOFF_RADIUS of the cube (where SAC drops it),
             ends with the cube lifted clear.
      PLACE  starts above the drop plate (where SAC drops it again),
             ends with the cube released and the gripper retreated.

    The transport between them belongs to SAC, so it is deliberately NOT
    recorded. Training the VLA on motion it will never be asked to perform
    would just dilute the two skills that matter.
"""

import numpy as np

from .env import PickPlaceEnv
from . import config as C


# Waypoint offsets, in metres, relative to the cube or the drop point.
APPROACH_HEIGHT = 0.11     # hover here before descending onto the cube
GRASP_HEIGHT = 0.015       # EE site sits this far above the cube centre
LIFT_HEIGHT = 0.18         # how high to carry after a successful grasp
PLACE_HOVER = 0.14         # hover here before lowering onto the plate
RELEASE_HEIGHT = 0.065     # cube bottom just clears the plate at release

POS_TOL = 0.012            # waypoint considered hit within this
GRIPPER_SETTLE_STEPS = 8   # ticks to let the fingers actually close/open


class ScriptedExpert:
    """A waypoint-following state machine over the shared action interface."""

    def __init__(self, env: PickPlaceEnv, noise_std: float = C.SCRIPTED_NOISE_STD, rng=None):
        self.env = env
        self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------------------------------------------------
    def _action_toward(self, target, gripper_closed):
        """Proportional controller in EE space, in normalized action units."""
        ee = self.env.get_ee_pose()[0]
        delta = (np.asarray(target) - ee) / C.STEP_SIZE
        a = np.clip(delta, -1.0, 1.0)
        if self.noise_std > 0:
            a = np.clip(a + self.rng.normal(scale=self.noise_std, size=3), -1.0, 1.0)
        return np.concatenate([a, [1.0 if gripper_closed else 0.0]]).astype(np.float32)

    def _goto(self, target, gripper_closed, budget, record, tol=POS_TOL):
        """Drive to a waypoint. Returns True if it arrived within budget."""
        for _ in range(budget):
            if self.env.ee_to(target) < tol:
                return True
            a = self._action_toward(target, gripper_closed)
            record(a)
            self.env.apply_action(a)
        return self.env.ee_to(target) < tol

    def _hold(self, gripper_closed, steps, record):
        """Stay put while the gripper opens or closes.

        A zero delta rather than a re-derived one: the persistent EE target
        already holds position, and commanding motion here would fight the
        fingers as they close on the cube.
        """
        for _ in range(steps):
            a = np.array([0.0, 0.0, 0.0, 1.0 if gripper_closed else 0.0], dtype=np.float32)
            record(a)
            self.env.apply_action(a)

    # ======================================================================
    # Segment 1: GRASP  (what SmolVLA does after SAC's phase-1 reach)
    # ======================================================================
    def run_grasp(self, record=None):
        # `is None`, never `or`: a recorder object that defines __len__ is
        # FALSY while it is still empty, so `record or noop` silently swaps a
        # perfectly good recorder for a no-op on the very first episode and
        # every segment comes back with zero frames.
        record = (lambda a: None) if record is None else record
        cube = self.env.get_object_position()

        above = cube + np.array([0.0, 0.0, APPROACH_HEIGHT])
        at = cube + np.array([0.0, 0.0, GRASP_HEIGHT])

        self._goto(above, False, 60, record)
        self._goto(at, False, 60, record, tol=0.008)
        self._hold(True, GRIPPER_SETTLE_STEPS, record)       # close on the cube

        lift = np.array([cube[0], cube[1], C.TABLE_TOP_Z + LIFT_HEIGHT])
        self._goto(lift, True, 60, record)
        return self.env.is_picked()

    # ======================================================================
    # Segment 2: PLACE  (what SmolVLA does after SAC's phase-4 transfer)
    # ======================================================================
    def run_place(self, record=None):
        # `is None`, never `or`: a recorder object that defines __len__ is
        # FALSY while it is still empty, so `record or noop` silently swaps a
        # perfectly good recorder for a no-op on the very first episode and
        # every segment comes back with zero frames.
        record = (lambda a: None) if record is None else record
        drop = self.env.get_drop_position()

        hover = drop + np.array([0.0, 0.0, PLACE_HOVER])
        down = drop + np.array([0.0, 0.0, RELEASE_HEIGHT])

        self._goto(hover, True, 60, record)
        self._goto(down, True, 60, record, tol=0.010)
        self._hold(False, GRIPPER_SETTLE_STEPS, record)      # release

        retreat = drop + np.array([0.0, 0.0, PLACE_HOVER])
        self._goto(retreat, False, 40, record)
        # Let the cube settle before the success test, so a mid-bounce frame
        # cannot be scored as a place.
        self._hold(False, 10, record)
        return self.env.is_placed()

    # ======================================================================
    # Transport: SAC owns this at rollout time. Here it only exists to put the
    # arm in the right place to record the PLACE segment, and is NOT recorded.
    # ======================================================================
    def transport_to_drop(self):
        home_ish = np.array([0.50, 0.10, 0.45])
        self._goto(home_ish, True, 80, lambda a: None, tol=0.05)
        above_drop = self.env.get_drop_position() + np.array([0.0, 0.0, 0.22])
        self._goto(above_drop, True, 80, lambda a: None, tol=0.05)
        return self.env.is_grasped()
