"""
pickplace/train_smolvla.py -- fine-tune SmolVLA on the two manipulation segments.

    python -m pickplace.train_smolvla
    python -m pickplace.train_smolvla --steps 20000 --batch-size 16

ONE POLICY, TWO SKILLS
    The dataset mixes "pick up the red cube" and "place the red cube on the
    green drop point" episodes, and a single set of weights learns both. The
    task string selects the behaviour at rollout time. That is the reason a
    VLA is worth its cost here at all -- two behaviour-cloned MLPs would need
    two checkpoints and an external switch, and would share nothing.

WHAT IS ACTUALLY TRAINED
    SmolVLA is ~450M parameters, but with the SigLIP vision tower frozen and
    train_expert_only set, only the ~100M-parameter action expert gets
    gradients. Measured on this machine: 1.6 GB of VRAM at batch 2, so a batch
    of 16 fits inside 12.8 GB with room to spare.

ACTION CHUNKING
    Each sample carries the next chunk_size actions, which is what the
    flow-matching head regresses. Chunks are clipped at the episode boundary
    and the remainder flagged in action_is_pad so the loss ignores it --
    without that the model is trained to predict padding as a real action.

WHERE THE DATA COMES FROM
    pickplace.dataset, reading the packed .npz episodes directly. NOT
    LeRobotDataset: fine-tuning needs only normalization stats and action
    chunks, both of which are a few lines, whereas LeRobotDataset re-encodes
    every frame to h264 and decodes it back -- lossy, on pristine synthetic
    renders, and the better part of an hour to build. to_lerobot.py still
    exists for publishing to the Hub; it is not on the training path.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature

from .dataset import PickPlaceDataset, collate, PACK_DIR
from . import config as C


def build_config(device, chunk_size, n_action_steps, pretrained=True):
    return SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(C.STATE_DIM,)),
            **{
                "observation.images.%s" % cam: PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *C.RENDER_SIZE)
                )
                for cam in C.CAMERA_NAMES
            },
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(C.ACTION_DIM,))
        },
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        device=device,
        freeze_vision_encoder=C.VLA_FREEZE_VISION,
        train_expert_only=C.VLA_TRAIN_EXPERT_ONLY,
        load_vlm_weights=not pretrained,
        optimizer_lr=C.VLA_LR,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=str(PACK_DIR))
    p.add_argument("--out", default=str(C.VLA_CKPT_DIR))
    p.add_argument("--steps", type=int, default=C.VLA_STEPS)
    p.add_argument("--batch-size", type=int, default=C.VLA_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.VLA_LR)
    p.add_argument("--chunk-size", type=int, default=C.VLA_CHUNK_SIZE)
    p.add_argument("--n-action-steps", type=int, default=C.VLA_N_ACTION_STEPS)
    p.add_argument("--save-every", type=int, default=2_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-scratch", action="store_true",
                   help="skip the lerobot/smolvla_base weights (much worse; for ablation)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args.device, args.chunk_size, args.n_action_steps,
                       pretrained=not args.from_scratch)

    # --- dataset ----------------------------------------------------------
    dataset = PickPlaceDataset(args.dataset_root, chunk_size=args.chunk_size)
    print("dataset: %d frames, %d episodes, fps %d"
          % (len(dataset), dataset.meta["total_episodes"], dataset.meta["fps"]))
    print("tasks: %s" % dataset.tasks)
    for k, v in dataset.meta["episodes_by_kind"].items():
        print("  %-16s %4d episodes  %6d frames"
              % (k, v, dataset.meta["frames_by_kind"][k]))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
        collate_fn=collate,
    )

    # --- policy -----------------------------------------------------------
    if args.from_scratch:
        policy = SmolVLAPolicy(cfg)
    else:
        print("loading pretrained %s ..." % C.SMOLVLA_BASE)
        policy = SmolVLAPolicy.from_pretrained(C.SMOLVLA_BASE, config=cfg)
    policy.to(args.device)
    policy.train()

    n_all = sum(p_.numel() for p_ in policy.parameters())
    n_train = sum(p_.numel() for p_ in policy.parameters() if p_.requires_grad)
    print("policy: %.0fM params, %.0fM trainable (%.0f%%)"
          % (n_all / 1e6, n_train / 1e6, 100 * n_train / n_all))

    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset.stats)

    optim = torch.optim.AdamW(
        [p_ for p_ in policy.parameters() if p_.requires_grad],
        lr=args.lr, betas=C.VLA_BETAS, weight_decay=C.VLA_WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
        anneal_strategy="cos", final_div_factor=40.0,
    )

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a")

    step = 0
    t0 = time.time()
    running = []
    print("\ntraining for %d steps, batch %d, lr %.1e" % (args.steps, args.batch_size, args.lr))
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            processed = pre(batch)
            loss, _ = policy.forward(processed)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in policy.parameters() if p_.requires_grad],
                C.VLA_GRAD_CLIP,
            )
            optim.step()
            sched.step()

            step += 1
            running.append(float(loss.detach()))

            if step % args.log_every == 0:
                mean_loss = float(np.mean(running[-args.log_every:]))
                rec = {
                    "step": step,
                    "loss": mean_loss,
                    "grad_norm": float(grad_norm),
                    "lr": float(sched.get_last_lr()[0]),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                print("[%6d/%d] loss %.4f  grad %.2f  lr %.2e  %.2f it/s  vram %.1fGB"
                      % (step, args.steps, mean_loss, float(grad_norm),
                         sched.get_last_lr()[0], step / (time.time() - t0),
                         torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0),
                      flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            if step % args.save_every == 0 or step == args.steps:
                ckpt = out_dir / "checkpoint"
                policy.save_pretrained(ckpt)
                pre.save_pretrained(ckpt)
                post.save_pretrained(ckpt)
                (ckpt / "step.txt").write_text(str(step))
                print("  saved checkpoint at step %d -> %s" % (step, ckpt), flush=True)

    log_f.close()
    print("\ndone in %.1f min. final loss %.4f"
          % ((time.time() - t0) / 60, float(np.mean(running[-100:]))))


if __name__ == "__main__":
    main()
