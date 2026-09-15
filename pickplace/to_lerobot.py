"""
pickplace/to_lerobot.py -- build the LeRobotDataset SmolVLA trains on.

    python -m pickplace.to_lerobot
    python -m pickplace.to_lerobot --push          # after `hf auth login`

Merges two sources into ONE language-conditioned dataset:

    outputs/scripted_demos/*_grasp.npz  -> "pick up the red cube"
    outputs/scripted_demos/*_place.npz  -> "place the red cube on the green drop point"
    outputs/cropped_teleop/*.npz        -> "pick up the red cube"   (human)

One dataset and one policy, not two. SmolVLA is language-conditioned, so the
task string is what selects the skill at rollout time -- that is precisely the
capability being bought by using a VLA here rather than two behaviour-cloned
MLPs. Training separate models per skill would throw it away.

ACTION UNITS
    Everything written here is in NORMALIZED units, [-1, 1] scaled by
    STEP_SIZE, matching PickPlaceEnv.apply_action. The raw teleop .npz files
    store metres; replay_teleop.py already converted them. The scripted demos
    were recorded in normalized units to begin with. This file asserts the
    range rather than trusting it, because a silent unit mismatch between
    training and rollout is the single most expensive bug available here.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from . import config as C


def build_features():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (C.STATE_DIM,),
            "names": ["joint%d" % i for i in range(1, 8)] + ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (C.ACTION_DIM,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }
    for cam in C.CAMERA_NAMES:
        features["observation.images.%s" % cam] = {
            "dtype": "video",
            "shape": (C.RENDER_SIZE[0], C.RENDER_SIZE[1], 3),
            "names": ["height", "width", "channel"],
        }
    return features


def iter_source_episodes(include_teleop=True):
    """Yield (path, kind) for every episode file, scripted first."""
    for p in sorted(Path(C.SCRIPTED_DEMO_DIR).glob("*_grasp.npz")):
        yield p, "scripted_grasp"
    for p in sorted(Path(C.SCRIPTED_DEMO_DIR).glob("*_place.npz")):
        yield p, "scripted_place"
    if include_teleop:
        for p in sorted(Path(C.CROPPED_TELEOP_DIR).glob("*.npz")):
            yield p, "teleop_grasp"


def load_episode(path):
    z = np.load(path, allow_pickle=True)
    ep = {
        "states": z["states"].astype(np.float32),
        "actions": z["actions"].astype(np.float32),
        "task": str(z["task_instruction"]),
        "images": {cam: z["images_%s" % cam]
                   for cam in C.CAMERA_NAMES if ("images_%s" % cam) in z},
    }
    if ep["states"].shape[1] != C.STATE_DIM:
        raise ValueError("%s: state dim %d, expected %d"
                         % (path.name, ep["states"].shape[1], C.STATE_DIM))
    if ep["actions"].shape[1] != C.ACTION_DIM:
        raise ValueError("%s: action dim %d, expected %d"
                         % (path.name, ep["actions"].shape[1], C.ACTION_DIM))
    # The unit check. Normalized deltas live in [-1, 1]; metres would be ~0.01
    # and would sail through a naive max() test, so check the SCALE too.
    amax = float(np.abs(ep["actions"][:, :3]).max())
    if amax > 1.0 + 1e-4:
        raise ValueError("%s: |action| max %.4f > 1 -- not normalized" % (path.name, amax))
    if amax < 0.05:
        raise ValueError(
            "%s: |action| max %.4f is suspiciously small -- these look like "
            "metres, not normalized units" % (path.name, amax))
    return ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=C.HF_REPO_ID)
    p.add_argument("--root", default=str(C.LEROBOT_DIR))
    p.add_argument("--no-teleop", action="store_true",
                   help="scripted demonstrations only")
    p.add_argument("--push", action="store_true", help="push to the Hugging Face Hub")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vcodec", default="h264",
                   help="lerobot's default is libsvtav1 (AV1). Valid names "
                        "here are h264 / hevc / libsvtav1 and the hardware "
                        "variants (h264_nvenc etc). Measured on this machine, "
                        "h264 and h264_nvenc are indistinguishable -- the "
                        "speedup comes from the parallel image writers below.")
    p.add_argument("--writer-threads", type=int, default=8)
    p.add_argument("--writer-processes", type=int, default=2)
    # Must stay 1 on lerobot 0.4.4. Anything larger turns on the batched
    # encoding path, which reads self.meta.episodes[start_episode] for
    # episodes the 10-slot metadata buffer has not flushed yet and dies on
    # "'NoneType' object is not subscriptable" as soon as the first batch
    # closes. The speedup came from the parallel image writers anyway.
    p.add_argument("--batch-encoding-size", type=int, default=1)
    args = p.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(
                "%s already exists. Pass --overwrite to rebuild it." % root)
        shutil.rmtree(root)

    include_teleop = C.INCLUDE_TELEOP_DATA and not args.no_teleop
    sources = list(iter_source_episodes(include_teleop))
    if not sources:
        raise SystemExit(
            "No episodes found. Run `python -m pickplace.generate_demos` and "
            "`python -m pickplace.replay_teleop` first.")

    print("building LeRobotDataset at %s" % root)
    print("  %d source episodes (teleop %s)"
          % (len(sources), "included" if include_teleop else "EXCLUDED"))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=C.CONTROL_HZ,
        root=root,
        features=build_features(),
        robot_type="franka_panda",
        use_videos=True,
        # Encoding settings, not cosmetics. At lerobot's defaults (libsvtav1,
        # single-threaded image writing, encode-per-episode) this dataset
        # builds at ~7 episodes/min -- about 100 minutes for 695 episodes.
        # With parallel writers and batched encoding it runs at ~1.4s per
        # episode, roughly 16 minutes. Nothing downstream cares which codec
        # the frames arrived in.
        vcodec=args.vcodec,
        image_writer_threads=args.writer_threads,
        image_writer_processes=args.writer_processes,
        batch_encoding_size=args.batch_encoding_size,
    )

    counts, frames_by_kind = {}, {}
    for path, kind in sources:
        ep = load_episode(path)
        for i in range(len(ep["states"])):
            frame = {
                "observation.state": ep["states"][i],
                "action": ep["actions"][i],
                "task": ep["task"],
            }
            for cam, arr in ep["images"].items():
                frame["observation.images.%s" % cam] = arr[i]
            dataset.add_frame(frame)
        # save_episode is what actually encodes the video and writes parquet.
        # Omit it and you get an empty dataset with no error whatsoever.
        dataset.save_episode()
        counts[kind] = counts.get(kind, 0) + 1
        frames_by_kind[kind] = frames_by_kind.get(kind, 0) + len(ep["states"])

    print("\nepisodes written:")
    total = 0
    for kind in sorted(counts):
        print("  %-16s %4d episodes  %6d frames" % (kind, counts[kind], frames_by_kind[kind]))
        total += frames_by_kind[kind]
    print("  %-16s %4d episodes  %6d frames" % ("TOTAL", sum(counts.values()), total))

    tasks = sorted({load_episode(p)["task"] for p, _ in sources[:1]} |
                   {C.TASK_PICK, C.TASK_PLACE})
    print("\ntasks: %s" % tasks)

    if args.push:
        print("\npushing to %s ..." % args.repo_id)
        dataset.push_to_hub()
        print("done -- check the dataset viewer before training.")
    else:
        print("\nlocal only. Pass --push to upload (after `hf auth login`).")


if __name__ == "__main__":
    main()
