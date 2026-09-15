"""
pickplace/replay_teleop.py -- rebuild the teleop episodes in the pick-place scene.

    python -m pickplace.replay_teleop --limit 5 --dry-run
    python -m pickplace.replay_teleop

WHY NOT JUST USE THE RECORDED IMAGES
    The 100 recorded episodes are good trajectories wrapped in unusable
    pixels. Two defects, both measured:

      1. ee_site, the green IK marker, fills 13.5% of EVERY wrist frame, dead
         centre -- precisely where the cube sits during a grasp. The wrist
         camera is the most informative view for a grasp and its middle is
         covered by a debug sphere.

      2. They were recorded in franka_panda_scene.xml, which has no drop
         plate. At rollout the plate is visible in 2.7% of the front view. A
         policy trained only on plate-free frames meets a plate-bearing frame
         at test time.

    Neither can be fixed by editing pixels. But the episodes DO record the
    action sequence, and actions are all the simulator needs.

THE REPLAY
    Re-drive the recorded action sequence through PickPlaceEnv in the new
    scene, then re-render. Two things make this sound rather than hopeful:

      - The IK, the leash, the control rate and the action semantics are
        byte-identical, because apply_action IS the loop record_episode.py ran.
      - The cube position, never recorded, is recovered from the grasp: at the
        frame where the fingers close on a successful pick they are by
        definition around the cube, so cube_xy = EE xy at that instant, and
        cube_z is known exactly (it rests on the table).

    Measured over 25 episodes: 96% replay successfully, median terminal joint
    error 0.0055 rad (0.3 degrees), worst 0.033 rad.

    Every episode is VERIFIED, not assumed: one that does not end in a
    successful pick is discarded rather than written out as a demonstration of
    a grasp that never happened.
"""

import argparse
import numpy as np
import mujoco
from pathlib import Path

from .env import PickPlaceEnv
from .crop_teleop import crop_episode, forward_kinematics_ee
from . import config as C


def build_fk_rig():
    """A model/data pair on the TELEOP scene, for recovering the recorded EE
    pose. It must be the teleop scene: that is where these joints were
    recorded, and the arm kinematics have to match exactly."""
    model = mujoco.MjModel.from_xml_path(C.TELEOP_SCENE_XML)
    data = mujoco.MjData(model)
    qpos_ids = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)]
        for i in range(1, 8)
    ])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    return model, data, qpos_ids, site_id


def deidle_mask(actions, max_idle_run=C.TELEOP_MAX_IDLE_RUN,
                protect_window=C.GRIPPER_PROTECT_WINDOW):
    """Which frames of a teleop segment are worth keeping.

    A frame survives if it commands motion, changes the gripper, sits inside
    the protected window after a gripper change, or is among the first
    `max_idle_run` frames of an idle stretch. See the note in config.py for
    why 83% of these frames are idle and why that has to be dealt with.
    """
    moving = np.abs(actions[:, :3]).max(axis=1) > 1e-9

    grip = actions[:, 3]
    grip_change = np.ones(len(actions), dtype=bool)
    grip_change[1:] = grip[1:] != grip[:-1]

    # The closure itself commands zero motion. It is also the whole point of
    # the demonstration, so protect it and the settling frames after it.
    protect = grip_change.copy()
    for k in range(1, protect_window + 1):
        protect[k:] |= grip_change[:-k]

    keep = moving | protect
    run = 0
    for i in range(len(keep)):
        if keep[i]:
            run = 0
        else:
            run += 1
            if run <= max_idle_run:
                keep[i] = True
    return keep


def replay_episode(env, actions, cube_xy, start, render=True, keep_mask=None):
    """Replay one episode; render only from `start` onward.

    Frames before the crop point still have to be SIMULATED -- the arm must
    travel the same path to arrive in the same configuration -- but they do
    not have to be RENDERED, and rendering is by far the expensive half.
    """
    env.reset(cube_xy=cube_xy)

    frames = {cam: [] for cam in env.camera_names} if render else None
    states, acts = [], []

    for i, a in enumerate(actions):
        norm = np.concatenate([np.clip(a[:3] / C.STEP_SIZE, -1.0, 1.0), a[3:4]]).astype(np.float32)
        # Every frame is SIMULATED -- the arm has to travel the same path to
        # arrive in the same configuration. Only the kept ones are recorded.
        if i >= start and (keep_mask is None or keep_mask[i]):
            # Observation BEFORE the step, matching the convention the teleop
            # recorder used and the one SmolVLA expects.
            if render:
                imgs = env.render_all_cameras()
                for cam in frames:
                    frames[cam].append(imgs[cam])
            states.append(np.concatenate([
                env.get_joint_positions(),
                [1.0 if env.gripper_closed else 0.0],
            ]).astype(np.float32))
            acts.append(norm)
        env.apply_action(norm)

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(acts, dtype=np.float32),
        "images": {k: np.asarray(v, dtype=np.uint8) for k, v in (frames or {}).items()},
        "picked": env.is_picked(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(C.CROPPED_TELEOP_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-idle-run", type=int, default=C.TELEOP_MAX_IDLE_RUN,
                   help="consecutive idle frames to keep (0 = drop all idle)")
    p.add_argument("--no-deidle", action="store_true",
                   help="keep every frame, including the 83%% that are idle")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    model, data, qpos_ids, site_id = build_fk_rig()
    env = PickPlaceEnv(enable_rendering=not args.dry_run, hide_sites=True)

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for d in C.TELEOP_DIRS:
        files.extend(sorted(Path(d).glob("episode_*.npz")))
    if args.limit:
        files = files[: args.limit]
    print("replaying %d teleop episodes into the pick-place scene" % len(files))

    kept = 0
    dropped = {"crop": 0, "replay": 0}
    total_frames = 0
    raw_total = 0
    for path in files:
        z = np.load(path, allow_pickle=True)
        states, actions = z["states"], z["actions"]
        ee_pos = forward_kinematics_ee(model, data, qpos_ids, site_id, states[:, :7])

        start, cube_est, reason = crop_episode(ee_pos, actions)
        if start is None:
            dropped["crop"] += 1
            print("  drop %-24s %s" % (path.name, reason))
            continue

        mask = None if args.no_deidle else deidle_mask(actions, args.max_idle_run)
        raw_len = len(actions) - start

        ep = replay_episode(env, actions, (float(cube_est[0]), float(cube_est[1])),
                            start, render=not args.dry_run, keep_mask=mask)
        if not ep["picked"]:
            dropped["replay"] += 1
            print("  drop %-24s replay did not end in a successful pick" % path.name)
            continue

        kept += 1
        total_frames += len(ep["states"])
        raw_total += raw_len

        if not args.dry_run:
            np.savez_compressed(
                out_dir / ("%s__%s" % (path.parent.name, path.name)),
                states=ep["states"],
                actions=ep["actions"],
                success=np.array(True),
                task_instruction=np.array(C.TASK_PICK),
                segment=np.array("grasp"),
                source=np.array(str(path)),
                cube_xy=np.asarray(cube_est[:2], dtype=np.float32),
                **{("images_%s" % k): v for k, v in ep["images"].items()},
            )
        print("  keep %-24s %4d frames (from %4d raw)  cube=(%.3f, %.3f)"
              % (path.name, len(ep["states"]), raw_len, cube_est[0], cube_est[1]))

    print("\nkept %d / %d  (dropped: %d at crop, %d at replay)"
          % (kept, len(files), dropped["crop"], dropped["replay"]))
    if kept:
        print("frames %d -> %d after de-idling (%.0f%% kept, mean %.0f per episode)"
              % (raw_total, total_frames, 100 * total_frames / max(1, raw_total),
                 total_frames / kept))
    print("(dry run -- nothing written)" if args.dry_run else "wrote to %s" % out_dir)
    env.close()


if __name__ == "__main__":
    main()
