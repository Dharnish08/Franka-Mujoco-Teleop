"""
pickplace/generate_demos.py -- generate scripted pick-and-place demonstrations.

    python -m pickplace.generate_demos --n 300
    python -m pickplace.generate_demos --n 20 --video outputs/demo_preview.mp4

Writes one .npz per SEGMENT, not per episode. SmolVLA is handed control twice
in the hybrid, so it is trained on exactly those two situations:

    <idx>_grasp.npz   task = "pick up the red cube"
    <idx>_place.npz   task = "place the red cube on the green drop point"

The transport between them is SAC's job and is deliberately not recorded.

WHERE EACH SEGMENT STARTS -- AND WHO PUTS IT THERE
    A demonstration is only useful if it starts where the policy will actually
    be handed control. Recording therefore begins at the handoff condition,
    not at the start of the scripted routine:

      grasp -- recording starts once the EE is within HANDOFF_RADIUS of the
               cube, the same predicate SAC terminates its reach on.
      place -- recording starts once the EE is within HANDOFF_RADIUS of the
               point above the drop plate that SAC transfers to.

    Everything before those points is simulated but discarded.

    That alone is NOT enough, and getting it wrong is silent. The approach has
    to be flown by the SAME controller that will fly it at rollout. Measured
    over 120 episodes with the scripted flyover doing the approach:

        EE offset from cube at handoff     SAC            scripted flyover
          dx                               -0.0262±0.0167  -0.0001±0.0011

    a gap of 23 TRAINING STANDARD DEVIATIONS. The scripted expert climbs to a
    hover waypoint directly over the cube and descends, so it crosses the
    radius dead-centre. SAC arrives from the home pose and crosses it 2.6cm
    short in x, with 17mm of spread. On a 4cm cube that is most of the cube's
    width, and the demonstrations contain almost no lateral variation anywhere
    to generalize from.

    So by default the approach is driven by the trained SAC policy and the
    scripted expert takes over exactly where it hands off. The demonstrations
    then begin in the state distribution the VLA will actually meet. Pass
    --approach scripted to reproduce the old (mismatched) behaviour.

EVERY SEGMENT IS VERIFIED
    A grasp segment is written only if the episode ends with the cube actually
    held and lifted; a place segment only if the cube ends resting on the
    plate with the gripper open. A demonstration of a failure is worse than no
    demonstration at all.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .env import PickPlaceEnv
from .scripted_expert import ScriptedExpert, PLACE_HOVER
from . import config as C


class SegmentRecorder:
    """Records (image, state, action) triples, but only once armed.

    `arm_when` is a predicate on the env; recording starts the first time it
    returns True. That is what aligns the demonstration with the handoff.
    """

    def __init__(self, env, arm_when=None):
        self.env = env
        self.arm_when = arm_when
        self.armed = arm_when is None
        self.images = {cam: [] for cam in env.camera_names}
        self.states = []
        self.actions = []

    def __call__(self, action):
        if not self.armed:
            if not self.arm_when(self.env):
                return
            self.armed = True
        obs = self.env.get_observation(with_images=True)
        for cam, img in obs["images"].items():
            self.images[cam].append(img)
        self.states.append(obs["state"])
        self.actions.append(np.asarray(action, dtype=np.float32))

    def __len__(self):
        return len(self.states)

    def arrays(self):
        return {
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            **{("images_%s" % k): np.asarray(v, dtype=np.uint8)
               for k, v in self.images.items()},
        }


class SACApproach:
    """Flies the approach with the trained reach policy.

    This is the piece that aligns the demonstration start states with rollout.
    It rebuilds the 25-D observation exactly as GoalReachEnv and
    HybridController do -- tests.py asserts all three agree, because a
    permuted observation here would silently poison every demonstration.
    """

    def __init__(self, checkpoint, device="cpu"):
        from .sac import SAC
        self.agent = SAC(obs_dim=C.SAC_OBS_DIM, act_dim=C.SAC_ACTION_DIM,
                         hidden=C.SAC_HIDDEN, buffer_size=1, device=device)
        self.agent.load(str(checkpoint), map_location=device)
        self.agent.actor.eval()

    def _obs(self, env, goal, hold):
        q = env.get_joint_positions()
        ee = env.get_ee_pose()[0]
        d = np.asarray(goal) - ee
        return np.concatenate([np.cos(q), np.sin(q), ee, np.asarray(goal), d,
                               [np.linalg.norm(d)], [hold]]).astype(np.float32)

    def run(self, env, goal, hold_gripper, budget):
        """Drive until within HANDOFF_RADIUS of goal. True if it arrived."""
        for _ in range(budget):
            if env.ee_to(goal) < C.HANDOFF_RADIUS:
                return True
            a = self.agent.act(self._obs(env, goal, hold_gripper), deterministic=True)
            env.apply_action(np.concatenate([a, [hold_gripper]]))
        return env.ee_to(goal) < C.HANDOFF_RADIUS


def save_segment(out_dir, idx, name, recorder, task, extra=None):
    np.savez_compressed(
        out_dir / ("%04d_%s.npz" % (idx, name)),
        success=np.array(True),
        task_instruction=np.array(task),
        segment=np.array(name),
        **recorder.arrays(),
        **(extra or {}),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=C.N_SCRIPTED_DEMOS)
    p.add_argument("--out", default=str(C.SCRIPTED_DEMO_DIR))
    p.add_argument("--seed", type=int, default=C.DEMO_SEED)
    p.add_argument("--noise", type=float, default=C.SCRIPTED_NOISE_STD)
    p.add_argument("--video", default=None,
                   help="also write an mp4 of the first episode, for eyeballing")
    p.add_argument("--approach", default="sac", choices=["sac", "scripted"],
                   help="who flies the approach. 'sac' matches rollout; "
                        "'scripted' reproduces the 23-sigma mismatch")
    p.add_argument("--sac-checkpoint", default=str(C.SAC_CKPT_DIR / "sac_reach_best.pt"))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PickPlaceEnv(enable_rendering=True, hide_sites=True)
    rng = np.random.default_rng(args.seed)

    approach = None
    if args.approach == "sac":
        ckpt = Path(args.sac_checkpoint)
        if not ckpt.exists():
            raise SystemExit(
                "No SAC checkpoint at %s. Train the reach policy first "
                "(python -m pickplace.train_reach), or pass "
                "--approach scripted to accept the distribution mismatch." % ckpt)
        approach = SACApproach(ckpt)
        print("approach flown by SAC (%s) -- demo start states match rollout" % ckpt.name)
    else:
        print("approach flown by the scripted expert -- WARNING: this puts the "
              "demonstration start states ~23 sigma off where SAC hands over")

    # The home EE pose, resolved from the model once -- the same value the
    # orchestrator retracts to.
    env.reset()
    home_ee = env.get_ee_pose()[0].copy()

    n_grasp = n_place = 0
    fail_grasp = fail_transport = fail_place = fail_approach = 0
    grasp_frames = place_frames = 0
    video_frames = []
    t0 = time.time()

    for idx in range(args.n):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        expert = ScriptedExpert(env, noise_std=args.noise, rng=rng)
        cube_xy = env.get_object_position()[:2].copy()

        # --- fly the approach the way rollout will ---------------------------
        if approach is not None:
            if not approach.run(env, env.get_object_position(), 0.0,
                                C.PHASE_STEP_BUDGET["reach"]):
                fail_approach += 1
                continue

        # --- GRASP: arm the recorder at the SAC handoff condition ------------
        # Already inside the radius when SAC flew the approach, so this arms on
        # the very first call -- which is exactly the intent: the segment
        # begins at the pose the VLA is handed.
        grasp_rec = SegmentRecorder(
            env, arm_when=lambda e: e.ee_to(e.get_object_position()) < C.HANDOFF_RADIUS
        )
        if args.video and idx == 0:
            base = grasp_rec.__call__

            def base_with_video(a, _b=base):
                _b(a)
                video_frames.append(env.render_all_cameras()["front_cam"])
            grasp_call = base_with_video
        else:
            grasp_call = grasp_rec

        ok = expert.run_grasp(record=grasp_call)
        if not ok or len(grasp_rec) == 0:
            fail_grasp += 1
            continue
        save_segment(out_dir, idx, "grasp", grasp_rec, C.TASK_PICK,
                     extra={"cube_xy": cube_xy.astype(np.float32)})
        n_grasp += 1
        grasp_frames += len(grasp_rec)

        # --- transport (SAC's job at rollout; not recorded) ------------------
        hover_target = env.get_drop_position() + np.array([0.0, 0.0, PLACE_HOVER])
        if approach is not None:
            # Mirror the orchestrator exactly: retract to home, then transfer.
            # Flying this with SAC matters for the same reason it did for the
            # grasp -- it sets the pose the place segment starts from.
            ok = approach.run(env, home_ee, 1.0, C.PHASE_STEP_BUDGET["retract"])
            ok = ok and approach.run(env, hover_target, 1.0,
                                     C.PHASE_STEP_BUDGET["transfer"])
            if not ok or not env.is_grasped():
                fail_transport += 1
                continue
        elif not expert.transport_to_drop():
            fail_transport += 1
            continue

        # --- PLACE: arm at the point SAC transfers to ------------------------
        place_rec = SegmentRecorder(
            env, arm_when=lambda e, t=hover_target: e.ee_to(t) < C.HANDOFF_RADIUS
        )
        if args.video and idx == 0:
            base2 = place_rec.__call__

            def place_with_video(a, _b=base2):
                _b(a)
                video_frames.append(env.render_all_cameras()["front_cam"])
            place_call = place_with_video
        else:
            place_call = place_rec

        ok = expert.run_place(record=place_call)
        if not ok or len(place_rec) == 0:
            fail_place += 1
            continue
        save_segment(out_dir, idx, "place", place_rec, C.TASK_PLACE,
                     extra={"cube_xy": cube_xy.astype(np.float32)})
        n_place += 1
        place_frames += len(place_rec)

        if (idx + 1) % 25 == 0:
            print("  %3d/%d  grasp %d  place %d  (%.1fs)"
                  % (idx + 1, args.n, n_grasp, n_place, time.time() - t0))

    print("\nscripted demos written to %s" % out_dir)
    print("  grasp segments : %d  (%d frames, mean %.0f)"
          % (n_grasp, grasp_frames, grasp_frames / max(1, n_grasp)))
    print("  place segments : %d  (%d frames, mean %.0f)"
          % (n_place, place_frames, place_frames / max(1, n_place)))
    print("  failures       : approach %d, grasp %d, transport %d, place %d"
          % (fail_approach, fail_grasp, fail_transport, fail_place))
    print("  elapsed        : %.1fs" % (time.time() - t0))

    if args.video and video_frames:
        import imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, video_frames, fps=C.CONTROL_HZ)
        print("  preview video  : %s (%d frames)" % (args.video, len(video_frames)))

    env.close()


if __name__ == "__main__":
    main()
