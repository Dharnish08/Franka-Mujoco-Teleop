"""
pickplace/crop_teleop.py -- crop the recorded teleop episodes to the GRASP segment.

    python -m pickplace.crop_teleop
    python -m pickplace.crop_teleop --dry-run

WHY CROP AT ALL
    In the hybrid, SAC owns the approach and hands over only once the EE is
    within HANDOFF_RADIUS of the cube. A teleop episode averages ~500 frames,
    most of which are that approach. Training SmolVLA on the full episode
    teaches it a skill it will never be asked to perform, and -- worse --
    biases it toward the long-range motions that dominate the frame count.
    The VLA should see the distribution it will actually be handed.

RECONSTRUCTING WHAT WAS NOT RECORDED
    The episodes store 7 joint angles + a gripper flag per frame. They do NOT
    store the EE pose or the cube position, both of which the crop needs.

      EE pose  -- recovered exactly, by writing the recorded joints into qpos
                  and running mj_forward. This is forward kinematics on data
                  we have, not an estimate.

      Cube xy  -- estimated as the EE position at the first frame where the
                  gripper closes. Every one of these episodes is a SUCCESSFUL
                  pick, so at the instant the fingers close they are by
                  definition around the cube. The estimate is validated below
                  against the known spawn band: any episode whose inferred
                  cube lands outside CUBE_X_RANGE/CUBE_Y_RANGE is rejected
                  rather than silently mis-cropped.
"""

import argparse
import numpy as np
import mujoco
from pathlib import Path

from . import config as C


def forward_kinematics_ee(model, data, arm_qpos_ids, ee_site_id, joints_seq):
    """EE position for every frame, by replaying the recorded joint angles."""
    out = np.zeros((len(joints_seq), 3))
    for i, q in enumerate(joints_seq):
        data.qpos[arm_qpos_ids] = q
        mujoco.mj_forward(model, data)
        out[i] = data.site_xpos[ee_site_id]
    return out


def crop_episode(ee_pos, actions, handoff_radius=C.HANDOFF_RADIUS):
    """Return (start_index, cube_xy_estimate, reason) or (None, ..., reason)."""
    grip = actions[:, 3]
    closed = grip >= 0.5
    if not closed.any():
        return None, None, "gripper never closed"

    # The FINAL sustained closure, not the first close. Operators toggle the
    # gripper early -- testing it, or a stray q -- and taking the first close
    # puts the inferred cube wherever the EE happened to be, often still near
    # the home pose. Every one of these episodes ends in a successful lift, so
    # the real grasp is the closure that runs to the last frame.
    if not closed[-1]:
        return None, None, "episode does not end with the gripper closed"
    first_close = int(len(closed) - 1)
    while first_close > 0 and closed[first_close - 1]:
        first_close -= 1

    cube_est = ee_pos[first_close].copy()

    # Sanity-check the inference against the known spawn band before trusting
    # it. A silently wrong cube position produces a silently wrong crop.
    if not (C.CUBE_X_RANGE[0] - 0.08 <= cube_est[0] <= C.CUBE_X_RANGE[1] + 0.08):
        return None, cube_est, "inferred cube x=%.3f outside spawn band" % cube_est[0]
    if not (C.CUBE_Y_RANGE[0] - 0.08 <= cube_est[1] <= C.CUBE_Y_RANGE[1] + 0.08):
        return None, cube_est, "inferred cube y=%.3f outside spawn band" % cube_est[1]
    # The height check is what actually catches a bad inference: a cube sits on
    # the table at z ~ 0.02 and the EE site closes on it a few cm above. An
    # inferred "cube" up at z=0.5 is the home pose, not a cube.
    if not (0.0 <= cube_est[2] <= 0.12):
        return None, cube_est, "inferred cube z=%.3f not on the table" % cube_est[2]

    dist = np.linalg.norm(ee_pos - cube_est[None, :], axis=1)
    inside = np.flatnonzero(dist < handoff_radius)
    if inside.size == 0:
        return None, cube_est, "never entered handoff radius"

    # First entry that is not immediately left again: take the first index of
    # the run that contains the grasp, so a brief early fly-by does not start
    # the segment 200 frames too soon.
    before_close = inside[inside <= first_close]
    if before_close.size == 0:
        return None, cube_est, "entered radius only after the grasp"
    start = int(before_close[0])
    for i in range(before_close.size - 1, 0, -1):
        if before_close[i] - before_close[i - 1] > 1:
            start = int(before_close[i])
            break
    return start, cube_est, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=float, default=C.HANDOFF_RADIUS)
    p.add_argument("--out", default=str(C.CROPPED_TELEOP_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="report the crop statistics without writing anything")
    args = p.parse_args()

    # The teleop scene, because that is what these episodes were recorded in.
    model = mujoco.MjModel.from_xml_path(C.TELEOP_SCENE_XML)
    data = mujoco.MjData(model)
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)
                     for i in range(1, 8)]
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids])
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for d in C.TELEOP_DIRS:
        files.extend(sorted(Path(d).glob("episode_*.npz")))
    print("found %d teleop episodes" % len(files))

    kept = 0
    orig_frames = crop_frames = 0
    rejects = {}
    for path in files:
        z = np.load(path, allow_pickle=True)
        states, actions = z["states"], z["actions"]
        ee_pos = forward_kinematics_ee(model, data, arm_qpos_ids, ee_site_id, states[:, :7])

        start, cube_est, reason = crop_episode(ee_pos, actions, args.radius)
        orig_frames += len(states)
        if start is None:
            rejects[reason.split(" outside")[0].split(" x=")[0].split(" y=")[0]] = \
                rejects.get(reason.split(" outside")[0].split(" x=")[0].split(" y=")[0], 0) + 1
            print("  REJECT %-28s %s" % (path.name, reason))
            continue

        n = len(states) - start
        crop_frames += n
        kept += 1

        if not args.dry_run:
            np.savez_compressed(
                out_dir / ("%s__%s" % (path.parent.name, path.name)),
                states=states[start:],
                # Recorded deltas are in METRES; every consumer downstream
                # expects normalized [-1, 1] units. Convert once, here.
                actions=np.concatenate([
                    np.clip(actions[start:, :3] / C.STEP_SIZE, -1.0, 1.0),
                    actions[start:, 3:4],
                ], axis=1).astype(np.float32),
                success=np.array(True),
                task_instruction=np.array(C.TASK_PICK),
                segment=np.array("grasp"),
                cube_xy_estimate=cube_est[:2].astype(np.float32),
                source=np.array(str(path)),
                **{("images_%s" % cam): z["images_%s" % cam][start:]
                   for cam in C.CAMERA_NAMES if ("images_%s" % cam) in z},
            )

    print("\nkept %d / %d episodes" % (kept, len(files)))
    for r, n in rejects.items():
        print("  rejected (%s): %d" % (r, n))
    if kept:
        print("frames: %d -> %d  (%.1f%% retained, mean %.0f frames/episode)"
              % (orig_frames, crop_frames, 100 * crop_frames / orig_frames, crop_frames / kept))
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        print("wrote to %s" % out_dir)


if __name__ == "__main__":
    main()
