"""
pickplace/dataset.py -- the training set, straight from the .npz episodes.

    python -m pickplace.dataset --pack          # build the memmap pack
    python -m pickplace.dataset --inspect       # check what came out

WHY NOT LeRobotDataset
    Fine-tuning SmolVLA needs exactly two things a plain array does not give
    you for free:

      1. normalization statistics (mean/std over state and action), which the
         processor pipeline applies, and
      2. action chunks -- the next `chunk_size` actions per sample, with a pad
         mask for the frames that run off the end of an episode.

    Both are a few lines. LeRobotDataset supplies them, but it also re-encodes
    every frame to video and decodes it back at training time. That round trip
    is lossy -- these are pristine synthetic renders going through h264 -- it
    takes the better part of an hour to build 695 episodes, and on lerobot
    0.4.4 the batched-encoding path is simply broken (see to_lerobot.py).

    So training reads the episodes directly. to_lerobot.py still exists and
    still works; it is now an EXPORT step for sharing on the Hub, not a
    prerequisite for training.

THE PACK
    27,159 frames x 2 cameras of uint8 224x224x3 is 8.2 GB -- too much to hold
    in RAM next to a 450M-parameter model, and too slow to decompress from
    .npz on every access, since shuffled batches defeat any per-episode cache.

    So it is packed once into flat .npy files and read back with mmap_mode="r".
    The OS page cache then does the work, reads are lossless, and random
    access across 695 episodes costs a page fault instead of a zlib inflate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C


PACK_DIR = C.OUT_ROOT / "packed"


# ==========================================================================
# Packing
# ==========================================================================
def iter_source_episodes(include_teleop=True):
    """(path, kind) for every episode, scripted first. Mirrors to_lerobot.py."""
    for p in sorted(Path(C.SCRIPTED_DEMO_DIR).glob("*_grasp.npz")):
        yield p, "scripted_grasp"
    for p in sorted(Path(C.SCRIPTED_DEMO_DIR).glob("*_place.npz")):
        yield p, "scripted_place"
    if include_teleop:
        for p in sorted(Path(C.CROPPED_TELEOP_DIR).glob("*.npz")):
            yield p, "teleop_grasp"


def _validate(path, states, actions):
    if states.shape[1] != C.STATE_DIM:
        raise ValueError("%s: state dim %d, expected %d"
                         % (path.name, states.shape[1], C.STATE_DIM))
    if actions.shape[1] != C.ACTION_DIM:
        raise ValueError("%s: action dim %d, expected %d"
                         % (path.name, actions.shape[1], C.ACTION_DIM))
    # Normalized deltas live in [-1, 1]. Metres would be ~0.01 and would pass
    # a naive max() test, so the SCALE is checked too. A silent unit mismatch
    # between training and rollout is the most expensive bug available here.
    amax = float(np.abs(actions[:, :3]).max())
    if amax > 1.0 + 1e-4:
        raise ValueError("%s: |action| max %.4f > 1 -- not normalized"
                         % (path.name, amax))
    if amax < 0.05:
        raise ValueError("%s: |action| max %.4f looks like metres, not "
                         "normalized units" % (path.name, amax))


def pack(out_dir=PACK_DIR, include_teleop=True, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = list(iter_source_episodes(include_teleop))
    if not sources:
        raise SystemExit("No episodes found. Run generate_demos and replay_teleop first.")

    # Pass 1: sizes and metadata, without holding any pixels.
    lengths, tasks, kinds = [], [], []
    for path, kind in sources:
        z = np.load(path, allow_pickle=True)
        lengths.append(len(z["states"]))
        tasks.append(str(z["task_instruction"]))
        kinds.append(kind)
    total = int(sum(lengths))

    task_names = sorted(set(tasks))
    task_to_id = {t: i for i, t in enumerate(task_names)}

    if verbose:
        print("packing %d episodes, %d frames -> %s" % (len(sources), total, out_dir))
        print("  tasks: %s" % task_names)
        print("  image pack: %.1f GB per camera"
              % (total * C.RENDER_SIZE[0] * C.RENDER_SIZE[1] * 3 / 1e9))

    states = np.zeros((total, C.STATE_DIM), dtype=np.float32)
    actions = np.zeros((total, C.ACTION_DIM), dtype=np.float32)
    episode_index = np.zeros(total, dtype=np.int32)
    task_index = np.zeros(total, dtype=np.int16)

    h, w = C.RENDER_SIZE
    image_files = {
        cam: np.lib.format.open_memmap(
            out_dir / ("images_%s.npy" % cam), mode="w+",
            dtype=np.uint8, shape=(total, h, w, 3))
        for cam in C.CAMERA_NAMES
    }

    # Pass 2: stream the pixels in, one episode at a time.
    cursor = 0
    for ep_i, ((path, kind), n) in enumerate(zip(sources, lengths)):
        z = np.load(path, allow_pickle=True)
        s, a = z["states"].astype(np.float32), z["actions"].astype(np.float32)
        _validate(path, s, a)

        sl = slice(cursor, cursor + n)
        states[sl] = s
        actions[sl] = a
        episode_index[sl] = ep_i
        task_index[sl] = task_to_id[tasks[ep_i]]
        for cam, mm in image_files.items():
            mm[sl] = z["images_%s" % cam]
        cursor += n

        if verbose and (ep_i + 1) % 100 == 0:
            print("  %d/%d episodes" % (ep_i + 1, len(sources)), flush=True)

    for mm in image_files.values():
        mm.flush()

    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "episode_index.npy", episode_index)
    np.save(out_dir / "task_index.npy", task_index)

    # Episode start/end, so a chunk never runs across an episode boundary.
    ends = np.cumsum(lengths).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    np.save(out_dir / "episode_starts.npy", starts)
    np.save(out_dir / "episode_ends.npy", ends)

    # The stats SmolVLA's normalizer needs. Computed here because this is the
    # only place that has seen all the data at once.
    stats = {
        "observation.state": {
            "mean": states.mean(0).tolist(), "std": (states.std(0) + 1e-8).tolist(),
        },
        "action": {
            "mean": actions.mean(0).tolist(), "std": (actions.std(0) + 1e-8).tolist(),
        },
    }
    meta = {
        "total_frames": total,
        "total_episodes": len(sources),
        "fps": C.CONTROL_HZ,
        "tasks": task_names,
        "cameras": C.CAMERA_NAMES,
        "image_shape": [h, w, 3],
        "episodes_by_kind": {k: kinds.count(k) for k in sorted(set(kinds))},
        "frames_by_kind": {
            k: int(sum(n for kk, n in zip(kinds, lengths) if kk == k))
            for k in sorted(set(kinds))
        },
        "stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print("\npacked:")
        for k in sorted(meta["episodes_by_kind"]):
            print("  %-16s %4d episodes  %6d frames"
                  % (k, meta["episodes_by_kind"][k], meta["frames_by_kind"][k]))
        print("  %-16s %4d episodes  %6d frames" % ("TOTAL", len(sources), total))
    return meta


# ==========================================================================
# Dataset
# ==========================================================================
class PickPlaceDataset:
    """Frames with action chunks, shaped the way SmolVLA's processor expects.

    Returns tensors-as-numpy; the training loop's collate turns them into a
    batch. Images stay uint8 here and are converted to float CHW in [0, 1] at
    collate time -- moving 8 GB of float32 through the loader instead of uint8
    would be four times the bandwidth for no gain.
    """

    def __init__(self, root=PACK_DIR, chunk_size=C.VLA_CHUNK_SIZE):
        root = Path(root)
        if not (root / "meta.json").exists():
            raise SystemExit(
                "No pack at %s. Run `python -m pickplace.dataset --pack` first." % root)

        self.root = root
        self.chunk_size = chunk_size
        self.meta = json.loads((root / "meta.json").read_text())

        self.states = np.load(root / "states.npy")
        self.actions = np.load(root / "actions.npy")
        self.task_index = np.load(root / "task_index.npy")
        self.starts = np.load(root / "episode_starts.npy")
        self.ends = np.load(root / "episode_ends.npy")
        self.episode_index = np.load(root / "episode_index.npy")
        self.tasks = self.meta["tasks"]

        # Memmaps are opened LAZILY, per process, and deliberately kept out of
        # __getstate__. Windows DataLoader workers use spawn, which pickles
        # the whole dataset object into each one -- and a numpy memmap does
        # not pickle as a reference to the file, it materializes. Holding them
        # as attributes here would copy 8.2 GB into every worker.
        self._images = None

    @property
    def images(self):
        if self._images is None:
            self._images = {
                cam: np.load(self.root / ("images_%s.npy" % cam), mmap_mode="r")
                for cam in self.meta["cameras"]
            }
        return self._images

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None      # reopened in the child process
        return state

    def __len__(self):
        return int(self.meta["total_frames"])

    @property
    def stats(self):
        import torch
        return {
            k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()}
            for k, v in self.meta["stats"].items()
        }

    def __getitem__(self, i):
        ep = int(self.episode_index[i])
        ep_end = int(self.ends[ep])

        # Chunk clipped at the episode boundary, then padded. Without the
        # clip a chunk would run into the next episode and teach the model
        # transitions that never happened; without the pad mask the padding
        # itself becomes a training target.
        hi = min(i + self.chunk_size, ep_end)
        n_real = hi - i
        chunk = np.zeros((self.chunk_size, C.ACTION_DIM), dtype=np.float32)
        chunk[:n_real] = self.actions[i:hi]
        if n_real < self.chunk_size:
            # Hold the last real action through the pad region. The mask means
            # the loss ignores it, but leaving zeros here would still be an
            # odd input to anything that looks at the raw chunk.
            chunk[n_real:] = self.actions[hi - 1]
        pad = np.zeros(self.chunk_size, dtype=bool)
        pad[n_real:] = True

        sample = {
            "observation.state": self.states[i],
            "action": chunk,
            "action_is_pad": pad,
            "task": self.tasks[int(self.task_index[i])],
        }
        for cam, mm in self.images.items():
            sample["observation.images.%s" % cam] = np.asarray(mm[i])
        return sample


def collate(batch):
    """uint8 HWC -> float32 CHW in [0, 1], which is what SmolVLA wants.

    The VISUAL normalization mode is IDENTITY, meaning the policy does NOT
    rescale for you. Handing it raw uint8 puts every pixel 255x out of range
    and the model emits garbage while looking perfectly healthy.
    """
    import torch

    out = {
        "observation.state": torch.from_numpy(
            np.stack([b["observation.state"] for b in batch])),
        "action": torch.from_numpy(np.stack([b["action"] for b in batch])),
        "action_is_pad": torch.from_numpy(np.stack([b["action_is_pad"] for b in batch])),
        "task": [b["task"] for b in batch],
    }
    for key in batch[0]:
        if key.startswith("observation.images."):
            arr = np.stack([b[key] for b in batch])          # B H W C uint8
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            out[key] = t.to(torch.float32).div_(255.0)
    return out


# ==========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", action="store_true")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--out", default=str(PACK_DIR))
    p.add_argument("--no-teleop", action="store_true")
    args = p.parse_args()

    if args.pack:
        pack(args.out, include_teleop=C.INCLUDE_TELEOP_DATA and not args.no_teleop)

    if args.inspect or not args.pack:
        ds = PickPlaceDataset(args.out)
        print("\nframes %d, episodes %d, tasks %s"
              % (len(ds), ds.meta["total_episodes"], ds.tasks))
        s = ds[0]
        for k, v in s.items():
            print("  %-34s %s" % (k, getattr(v, "shape", v)))
        print("\naction stats  mean %s"
              % np.round(ds.meta["stats"]["action"]["mean"], 3))
        print("              std  %s"
              % np.round(ds.meta["stats"]["action"]["std"], 3))

        # A chunk must never cross an episode boundary.
        import random
        bad = 0
        for _ in range(2000):
            i = random.randrange(len(ds))
            ep = int(ds.episode_index[i])
            hi = min(i + ds.chunk_size, int(ds.ends[ep]))
            if hi > int(ds.ends[ep]):
                bad += 1
        print("chunk-boundary check: %d violations in 2000 samples" % bad)


if __name__ == "__main__":
    main()
