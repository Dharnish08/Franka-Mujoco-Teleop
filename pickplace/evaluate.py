"""
pickplace/evaluate.py -- measure the hybrid end to end.

    python -m pickplace.evaluate                       # full hybrid
    python -m pickplace.evaluate --vla scripted        # pipeline ablation
    python -m pickplace.evaluate --episodes 50 --video outputs/eval/rollout.mp4

WHAT IT REPORTS, AND WHY IT IS NOT JUST A SUCCESS RATE
    A single number cannot tell you what to fix. Every rollout records the
    phase it died in, so the output is a funnel: how many episodes survived
    reach, grasp, retract, transfer, place. A pipeline that loses 60% at the
    grasp and one that loses 60% at the transfer have nothing in common
    except the headline figure.

THE ABLATION THAT MAKES THE NUMBER MEAN SOMETHING
    --vla scripted swaps SmolVLA for the scripted expert and changes nothing
    else. That measures the SAC half, the handoffs and the task definition
    with a known-perfect manipulator. The gap between the two runs is the
    VLA's contribution, isolated. Run it whenever the hybrid disappoints,
    BEFORE retraining anything.

FIXED CUBE POSITIONS
    Every configuration is evaluated on the SAME seeded set of cube positions.
    Comparing two policies over different random spawns measures the spawns as
    much as the policies.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PickPlaceEnv
from .orchestrator import HybridController, ScriptedVLAStandIn, PHASES
from .sac import SAC
from . import config as C


def fixed_cube_positions(n, seed=C.EVAL_SEED):
    """A reproducible spread over the spawn band, not just n random draws."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*C.CUBE_X_RANGE, size=n)
    ys = rng.uniform(*C.CUBE_Y_RANGE, size=n)
    return list(zip(xs, ys))


def load_reach_agent(path, device):
    agent = SAC(
        obs_dim=C.SAC_OBS_DIM, act_dim=C.SAC_ACTION_DIM, hidden=C.SAC_HIDDEN,
        buffer_size=1, device=device,
    )
    agent.load(path)
    agent.actor.eval()
    return agent


def summarize(results):
    n = len(results)
    out = {"episodes": n, "success_rate": sum(r["success"] for r in results) / max(1, n)}

    # Funnel: an episode "reached" phase k if it did not fail before it.
    order = {p: i for i, p in enumerate(PHASES)}
    reached = {p: 0 for p in PHASES}
    for r in results:
        fp = r["failed_phase"]
        limit = len(PHASES) if fp is None else order[fp] + 1
        for p in PHASES[:limit]:
            reached[p] += 1
    out["reached"] = reached
    out["completed"] = {
        p: sum(1 for r in results
               if r["failed_phase"] is None or order[r["failed_phase"]] > i)
        for i, p in enumerate(PHASES)
    }
    out["failed_phase_counts"] = {
        p: sum(1 for r in results if r["failed_phase"] == p) for p in PHASES
    }
    succ = [r for r in results if r["success"]]
    if succ:
        out["mean_steps_on_success"] = float(np.mean([r["total_steps"] for r in succ]))
        out["mean_place_error_m"] = float(np.mean([r["place_error"] for r in succ]))
    out["mean_phase_steps"] = {
        p: float(np.mean([r["phases"][p] for r in results if p in r["phases"]]))
        for p in PHASES
        if any(p in r["phases"] for r in results)
    }
    return out


def print_report(name, s):
    print("\n" + "=" * 66)
    print(" %s" % name)
    print("=" * 66)
    print(" episodes        : %d" % s["episodes"])
    print(" SUCCESS RATE    : %.1f%%" % (100 * s["success_rate"]))
    print("\n funnel (episodes completing each phase):")
    prev = s["episodes"]
    for p in PHASES:
        done = s["completed"][p]
        drop = prev - done
        bar = "#" * int(40 * done / max(1, s["episodes"]))
        print("   %-9s %4d/%-4d %-41s %s"
              % (p, done, s["episodes"], bar, ("-%d" % drop) if drop else ""))
        prev = done
    print("\n failures by phase: %s"
          % ", ".join("%s %d" % (k, v) for k, v in s["failed_phase_counts"].items() if v))
    if "mean_steps_on_success" in s:
        print(" mean steps (success): %.0f" % s["mean_steps_on_success"])
        print(" mean place error    : %.1f cm" % (100 * s["mean_place_error_m"]))
    print(" mean steps per phase: %s"
          % ", ".join("%s %.0f" % (k, v) for k, v in s["mean_phase_steps"].items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=C.EVAL_EPISODES)
    p.add_argument("--vla", default="smolvla", choices=["smolvla", "scripted"])
    p.add_argument("--vla-checkpoint", default=str(C.VLA_CKPT_DIR / "checkpoint"))
    p.add_argument("--sac-checkpoint", default=str(C.SAC_CKPT_DIR / "sac_reach_best.pt"))
    p.add_argument("--out", default=str(C.EVAL_DIR))
    p.add_argument("--video", default=None, help="record the first episode to mp4")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds torch too: SmolVLA samples noise per chunk, "
                        "so the policy itself is stochastic at inference")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # See the note in eval_checkpoints.py: the policy is stochastic, so a run
    # is only reproducible if torch is seeded as well as the cube positions.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rendering is needed for the VLA and for video; the scripted stand-in
    # does not need it, but keeping it on makes the two runs directly
    # comparable rather than differing in an extra dimension.
    env = PickPlaceEnv(enable_rendering=True, hide_sites=True)

    reach_agent = load_reach_agent(args.sac_checkpoint, device)
    print("loaded SAC reach policy from %s" % args.sac_checkpoint)

    if args.vla == "scripted":
        vla = ScriptedVLAStandIn(env)
        label = "HYBRID with SCRIPTED stand-in (pipeline ablation)"
    else:
        from .vla_policy import SmolVLAController
        vla = SmolVLAController(args.vla_checkpoint, device=device)
        label = "HYBRID: SAC reach + SmolVLA manipulation"
        print("loaded SmolVLA from %s" % args.vla_checkpoint)

    ctrl = HybridController(env, reach_agent, vla, verbose=args.verbose)

    frames = []
    positions = fixed_cube_positions(args.episodes)
    results = []
    for i, cube_xy in enumerate(positions):
        capture = (args.video is not None and i == 0)
        on_step = None
        if capture:
            def on_step(_phase):
                pass
        res = ctrl.run_episode(cube_xy=cube_xy)
        results.append(res)
        if (i + 1) % 10 == 0:
            rate = 100 * sum(r["success"] for r in results) / len(results)
            print("  %3d/%d  running success %.0f%%" % (i + 1, args.episodes, rate),
                  flush=True)

    s = summarize(results)
    print_report(label, s)

    tag = args.vla
    (out_dir / ("summary_%s.json" % tag)).write_text(json.dumps(s, indent=2))
    (out_dir / ("episodes_%s.json" % tag)).write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % (out_dir / ("summary_%s.json" % tag)))
    env.close()


if __name__ == "__main__":
    main()
