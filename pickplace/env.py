"""
pickplace/env.py -- PickPlaceEnv

Extends teleop.env.FrankaMujocoEnv with everything the pick-and-place task
needs that pure teleoperation did not: the drop zone, a contact-based grasp
test, place success, and -- most importantly -- ONE shared action interface.

THE ONE SHARED ACTION INTERFACE
    apply_action(a) is the only way anything drives this robot. SAC, the
    scripted expert, SmolVLA and the orchestrator all call it. That is
    deliberate and load-bearing: it reproduces the control loop in
    record_episode.py exactly (persistent EE target, leash clamp, damped
    least-squares IK), so an action means the identical thing to every
    consumer. If the VLA were trained against one action semantics and rolled
    out under another, it would fail in ways that look like a training problem
    and are not.

ACTION CONVENTION
    a = [dx, dy, dz, gripper], all float32.
      dx, dy, dz : normalized to [-1, 1], multiplied by config.STEP_SIZE (1cm)
                   to get the metre-space EE delta.
      gripper    : >= 0.5 closed, < 0.5 open.
    The recorded teleop episodes store dx..dz in METRES, so the converter
    divides by STEP_SIZE. See to_lerobot.py.
"""

import numpy as np
import mujoco

from teleop.env import FrankaMujocoEnv
from teleop.diff_ik import DiffIKSolver
from . import config as C


class PickPlaceEnv(FrankaMujocoEnv):

    def __init__(
        self,
        xml_path: str = C.SCENE_XML,
        camera_names=None,
        render_size=C.RENDER_SIZE,
        control_hz: int = C.CONTROL_HZ,
        enable_rendering: bool = True,
        hide_sites: bool = True,
        seed=None,
    ):
        # enable_rendering=False skips renderer construction entirely. The SAC
        # reach policy is privileged/state-only, and building two MuJoCo
        # Renderers per env costs GPU memory and ~1.5ms/step we never use.
        self.enable_rendering = enable_rendering
        cams = list(camera_names if camera_names is not None else C.CAMERA_NAMES)

        super().__init__(
            xml_path=xml_path,
            camera_names=cams if enable_rendering else [],
            render_size=render_size,
            control_hz=control_hz,
            table_top_z=C.TABLE_TOP_Z,
            lift_threshold=C.LIFT_THRESHOLD,
        )
        self.camera_names = cams   # remember the real list even when not rendering

        self._ik = DiffIKSolver(damping=C.IK_DAMPING, max_dq=C.IK_MAX_DQ)

        # --- drop zone -----------------------------------------------------
        self._drop_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "drop_site")

        # --- finger bodies, for the contact-based grasp test ----------------
        self._finger_body_ids = {
            self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self._object_geom_id = self._name2id(
            mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom"
        )
        self._object_dof_adr = self.model.jnt_dofadr[
            self._name2id(mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)
        ]

        # --- keep the IK marker out of the camera -----------------------------
        # ee_site is declared rgba="0 1 0 1" in panda.xml, and it sits 10cm
        # down the tool axis -- directly in front of wrist_cam. Measured on the
        # existing recordings: it fills 13.5% of every wrist frame, dead centre,
        # exactly where the cube belongs during a grasp. The site is needed for
        # IK, so it cannot simply be deleted; hiding its visualization group
        # keeps the kinematics and clears the view.
        self._scene_option = None
        if hide_sites:
            self._scene_option = mujoco.MjvOption()
            self._scene_option.sitegroup[:] = 0

        self._rng = np.random.default_rng(seed)

        # persistent EE target -- see the module docstring in record_episode.py
        self._ee_target_pos = None
        self._ee_target_quat = None
        self._gripper_closed = False

    # ======================================================================
    # Reset
    # ======================================================================
    def reset(self, randomize_object: bool = True, cube_xy=None, seed=None):
        """Reset the scene and re-seed the persistent EE target.

        cube_xy overrides the random spawn -- used by the evaluation harness to
        replay an identical set of cube positions across policies.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self._arm_qpos_ids] = C.PANDA_HOME_QPOS
        self.data.ctrl[self._arm_actuator_ids] = C.PANDA_HOME_QPOS
        self.data.ctrl[self._gripper_actuator_id] = self.gripper_ctrlrange[1]
        self._gripper_closed = False

        if cube_xy is not None:
            x, y = float(cube_xy[0]), float(cube_xy[1])
        elif randomize_object:
            x = self._rng.uniform(*C.CUBE_X_RANGE)
            y = self._rng.uniform(*C.CUBE_Y_RANGE)
        else:
            x, y = 0.45, 0.0

        a = self._object_qpos_adr
        self.data.qpos[a : a + 3] = [x, y, C.TABLE_TOP_Z + C.CUBE_HALF_SIZE]
        self.data.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):          # let the cube settle before frame 0
            mujoco.mj_step(self.model, self.data)

        # Seed the target from the settled pose. Skipping this makes the arm
        # lunge at a stale target on the first step of every episode.
        self._ee_target_pos, self._ee_target_quat = self.get_ee_pose()
        return self.get_observation()

    # ======================================================================
    # The one shared action interface
    # ======================================================================
    def apply_action(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != C.ACTION_DIM:
            raise ValueError(
                "action must be %d-D, got %d" % (C.ACTION_DIM, a.shape[0])
            )

        delta = np.clip(a[:3], -1.0, 1.0) * C.STEP_SIZE
        self._gripper_closed = bool(a[3] >= 0.5)

        current_pos, current_quat = self.get_ee_pose()

        # Accumulate onto the PERSISTENT target rather than re-anchoring to the
        # measured pose. The Panda actuators are pure PD with no gravity comp,
        # so re-anchoring compounds the steady-state sag without bound.
        self._ee_target_pos = self._ee_target_pos + delta

        # Leash: stop the target running away while the arm is blocked (e.g.
        # pressing into the table or into the cube).
        gap = self._ee_target_pos - current_pos
        gap_norm = float(np.linalg.norm(gap))
        if gap_norm > C.MAX_LEASH:
            self._ee_target_pos = current_pos + gap * (C.MAX_LEASH / gap_norm)

        q_target = self._ik.solve(
            current_pos, current_quat,
            self._ee_target_pos, self._ee_target_quat,
            self.get_jacobian(), self.get_joint_positions(),
            self.get_joint_limits(),
        )
        g_lo, g_hi = self.gripper_ctrlrange
        self.step(q_target, g_lo if self._gripper_closed else g_hi)

    # ======================================================================
    # Observations
    # ======================================================================
    def get_observation(self, with_images=None):
        """Everything any consumer might want. Cheap fields always; pixels on
        request, because rendering is ~1.5ms and SAC never needs it."""
        if with_images is None:
            with_images = self.enable_rendering

        ee_pos, ee_quat = self.get_ee_pose()
        obs = {
            "joints": self.get_joint_positions(),
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper": 1.0 if self._gripper_closed else 0.0,
            "cube_pos": self.get_object_position(),
            "drop_pos": self.get_drop_position(),
        }
        # The 8-D state vector the teleop episodes recorded, verbatim.
        obs["state"] = np.concatenate(
            [obs["joints"], [obs["gripper"]]]
        ).astype(np.float32)

        if with_images:
            if not self.enable_rendering:
                raise RuntimeError(
                    "images requested but this env was built with "
                    "enable_rendering=False"
                )
            obs["images"] = self.render_all_cameras()
        return obs

    # ======================================================================
    # Rendering
    # ======================================================================
    def render_all_cameras(self):
        """Render every camera with the site markers suppressed.

        Overrides the base class purely to pass scene_option; see the note on
        _scene_option in __init__.
        """
        images = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=name,
                                  scene_option=self._scene_option)
            images[name] = renderer.render().copy()
        return images

    # ======================================================================
    # Task predicates
    # ======================================================================
    def get_drop_position(self):
        return self.data.site_xpos[self._drop_site_id].copy()

    def is_grasped(self):
        """True when BOTH fingers touch the cube and the gripper is commanded
        closed.

        Overrides the lift-only heuristic in the base class. For pick-and-place
        that one is not good enough: the cube riding up on top of a closed
        gripper, or being flicked into the air, both read as grasped under a
        pure height test and then silently corrupt the place phase.
        """
        if not self._gripper_closed:
            return False
        touching = set()
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == self._object_geom_id:
                other = self.model.geom_bodyid[g2]
            elif g2 == self._object_geom_id:
                other = self.model.geom_bodyid[g1]
            else:
                continue
            if other in self._finger_body_ids:
                touching.add(other)
        return len(touching) == 2

    def is_lifted(self):
        return bool(
            self.get_object_position()[2] > C.TABLE_TOP_Z + C.LIFT_THRESHOLD
        )

    def is_picked(self):
        """Grasp phase success: held AND clear of the table."""
        return self.is_grasped() and self.is_lifted()

    def is_placed(self):
        """Place phase success: cube resting on the plate, gripper released.

        The gripper check matters -- hovering the cube over the plate while
        still holding it is not a place, and without this term a policy learns
        exactly that shortcut.
        """
        if self._gripper_closed:
            return False
        cube = self.get_object_position()
        drop = self.get_drop_position()
        xy_ok = float(np.linalg.norm(cube[:2] - drop[:2])) < C.PLACE_SUCCESS_RADIUS
        # Cube resting on the plate: its centre sits at plate_top + half-size.
        z_ok = cube[2] < drop[2] + C.CUBE_HALF_SIZE + 0.02
        return bool(xy_ok and z_ok)

    def cube_is_static(self, tol: float = 0.02):
        """Cube has stopped moving -- guards against scoring a place mid-bounce."""
        vel = self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6]
        return bool(np.linalg.norm(vel) < tol)

    # ======================================================================
    def ee_to(self, target):
        return float(np.linalg.norm(self.get_ee_pose()[0] - np.asarray(target)))

    @property
    def gripper_closed(self):
        return self._gripper_closed
