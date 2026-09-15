"""
pickplace/snapshot_checkpoints.py -- keep every checkpoint, not just the last.

    python -m pickplace.snapshot_checkpoints            # run alongside training

train_smolvla.py writes to a single `checkpoint/` directory and overwrites it
every save. That is fine for resuming and useless for answering the question
that actually matters: HOW MANY STEPS DOES THIS TASK NEED?

Training loss cannot answer it. In imitation learning a loss of 0.04 versus
0.03 says almost nothing about whether the gripper closes on the cube -- the
only meaningful measure is rollout success, and that has to be evaluated per
checkpoint. So the checkpoints have to survive.

This watcher copies each save aside as it appears. It keys on step.txt, which
train_smolvla.py writes LAST, after the weights and both processors -- so a
changed step.txt means the save is complete and the copy cannot catch a
half-written checkpoint.

Run it in the background next to training; it costs nothing but disk.
"""

import argparse
import shutil
import time
from pathlib import Path

from . import config as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(C.VLA_CKPT_DIR / "checkpoint"))
    p.add_argument("--out", default=str(C.VLA_CKPT_DIR / "snapshots"))
    p.add_argument("--poll", type=float, default=20.0)
    p.add_argument("--timeout", type=float, default=6 * 3600,
                   help="give up if nothing new appears for this long")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    for d in out.glob("step_*"):
        seen.add(d.name.split("_")[1])

    last_new = time.time()
    print("watching %s -> %s" % (ckpt, out), flush=True)

    while True:
        step_file = ckpt / "step.txt"
        if step_file.exists():
            try:
                step = step_file.read_text().strip()
            except OSError:
                step = None
            if step and step not in seen:
                dest = out / ("step_%06d" % int(step))
                tmp = out / ("_partial_step_%06d" % int(step))
                try:
                    # Copy to a temp name and rename, so an interrupted copy
                    # never leaves a directory that looks like a valid
                    # snapshot to the evaluator.
                    if tmp.exists():
                        shutil.rmtree(tmp)
                    shutil.copytree(ckpt, tmp)
                    if dest.exists():
                        shutil.rmtree(dest)
                    tmp.rename(dest)
                    seen.add(step)
                    last_new = time.time()
                    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                    print("snapshot step %s  (%.2f GB)" % (step, size / 1e9), flush=True)
                except OSError as exc:
                    print("copy failed at step %s: %s" % (step, exc), flush=True)
                    shutil.rmtree(tmp, ignore_errors=True)

        if time.time() - last_new > args.timeout:
            print("no new checkpoint for %.0f min, stopping"
                  % (args.timeout / 60), flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
