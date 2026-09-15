"""
pickplace/eval_checkpoints.py -- how many training steps does this task need?

    python -m pickplace.eval_checkpoints --episodes 20

Runs the full hybrid with each saved snapshot and reports SUCCESS vs STEPS.
That curve is the only honest answer to "why 12,000 steps". Training loss is
not: a flow-matching loss of 0.04 versus 0.03 says nothing about whether the
gripper closes on the cube.

Every checkpoint is evaluated on the SAME seeded cube positions, so the
comparison measures the checkpoint and not the luck of the draw.

Reads the curve for you at the end: the point after which success stops
improving is where training could have stopped.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PickPlaceEnv
from .orchestrator import HybridController, PHASES
from .sac import SAC
from .evaluate import fixed_cube_positions, summarize
from . import config as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", default=str(C.VLA_CKPT_DIR / "snapshots"))
    p.add_argument("--sac-checkpoint", default=str(C.SAC_CKPT_DIR / "sac_reach_best.pt"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default=str(C.EVAL_DIR / "checkpoint_sweep.json"))
    p.add_argument("--device", default=None)
    p.add_argument("--min-step", type=int, default=0,
                   help="skip snapshots below this step")
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="seeds torch too: SmolVLA samples noise per chunk, "
                        "so the policy itself is stochastic at inference")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    snaps = sorted(Path(args.snapshots).glob("step_*"),
                   key=lambda d: int(d.name.split("_")[1]))
    snaps = [d for d in snaps
             if int(d.name.split("_")[1]) >= args.min_step
             and (args.max_step is None or int(d.name.split("_")[1]) <= args.max_step)]
    if not snaps:
        raise SystemExit("No snapshots in %s. Is snapshot_checkpoints running?"
                         % args.snapshots)
    print("evaluating %d checkpoints x %d episodes on %s\n"
          % (len(snaps), args.episodes, device))

    env = PickPlaceEnv(enable_rendering=True, hide_sites=True)
    agent = SAC(obs_dim=C.SAC_OBS_DIM, act_dim=C.SAC_ACTION_DIM,
                hidden=C.SAC_HIDDEN, buffer_size=1, device=device)
    agent.load(args.sac_checkpoint, map_location=device)
    agent.actor.eval()

    positions = fixed_cube_positions(args.episodes)
    from .vla_policy import SmolVLAController

    rows = []
    print("%8s %9s %8s %8s   %s" % ("step", "success", "grasp", "place", "failures"))
    print("-" * 74)
    for snap in snaps:
        step = int(snap.name.split("_")[1])
        # Re-seed torch before EVERY checkpoint. SmolVLA's flow-matching head
        # samples noise on each chunk prediction, so the policy is stochastic
        # at inference -- the same checkpoint on the same cube positions scored
        # 29/30 and 21/30 on two runs that differed only in prior RNG state.
        # Without this the sweep measures the noise draw as much as the
        # checkpoint, and an apparent dip between adjacent steps means nothing.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        vla = SmolVLAController(str(snap), device=device)
        ctrl = HybridController(env, agent, vla)
        results = [ctrl.run_episode(cube_xy=xy) for xy in positions]
        s = summarize(results)

        # Completion of the two VLA-owned phases, the ones this checkpoint
        # is actually responsible for.
        grasp = s["completed"]["grasp"] / s["episodes"]
        place = s["completed"]["place"] / s["episodes"]
        fails = ", ".join("%s %d" % kv
                          for kv in s["failed_phase_counts"].items() if kv[1])
        print("%8d %8.0f%% %7.0f%% %7.0f%%   %s"
              % (step, 100 * s["success_rate"], 100 * grasp, 100 * place, fails or "-"))
        rows.append({"step": step, "summary": s,
                     "grasp_rate": grasp, "place_rate": place})
        del vla, ctrl

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # --- where does it saturate? ------------------------------------------
    steps = [r["step"] for r in rows]
    succ = [r["summary"]["success_rate"] for r in rows]
    best = max(succ)
    # First checkpoint within one evaluation standard error of the best. With
    # n episodes the SE on a proportion is at most 0.5/sqrt(n), so anything
    # inside that band is statistically indistinguishable from the peak.
    se = 0.5 / np.sqrt(args.episodes)
    enough = next(s for s, v in zip(steps, succ) if v >= best - se)

    print("\nbest success %.0f%% at step %d" % (100 * best, steps[int(np.argmax(succ))]))
    print("within 1 SE (+/-%.0f%%) from step %d onward" % (100 * se, enough))
    if enough < max(steps):
        print("=> %d steps would have been enough; %d was %.1fx more than needed."
              % (enough, max(steps), max(steps) / max(1, enough)))
    else:
        print("=> success was still improving at the last checkpoint -- "
              "training longer may help.")
    print("\nwrote %s" % out)
    env.close()


if __name__ == "__main__":
    main()
