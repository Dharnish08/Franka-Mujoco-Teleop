"""
pickplace/run_live.py -- watch the hybrid run in the MuJoCo viewer.

    python -m pickplace.run_live
    python -m pickplace.run_live --vla scripted --episodes 5
    python -m pickplace.run_live --speed 0.5          # half speed

Opens the interactive viewer and drives the real policies through it, printing
each phase transition as it happens so you can see which half of the hybrid is
in control:

    [ep 0]  reach     SAC      ......... 43 steps
    [ep 0]  grasp     SmolVLA  ......... 37 steps
    ...

Drag to orbit, scroll to zoom, and the usual MuJoCo viewer keys work.

PACING
    The control loop is 15 Hz, so real time is 66.7 ms per step. SmolVLA
    inference costs roughly 30-60 ms whenever it re-plans a chunk, which eats
    most of that budget -- so the rollout runs at about real time and the
    pacing below only sleeps when there is time left over. --speed slows it
    down for a closer look; it cannot speed it up beyond what inference allows.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from .env import PickPlaceEnv
from .orchestrator import HybridController, ScriptedVLAStandIn
from .sac import SAC
from . import config as C


PHASE_OWNER = {
    "reach": "SAC", "grasp": "SmolVLA", "retract": "SAC",
    "transfer": "SAC", "place": "SmolVLA",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla", default="smolvla", choices=["smolvla", "scripted"])
    p.add_argument("--vla-checkpoint", default=str(C.VLA_CKPT_DIR / "checkpoint"))
    p.add_argument("--sac-checkpoint", default=str(C.SAC_CKPT_DIR / "sac_reach_best.pt"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0,
                   help="1.0 = real time, 0.5 = half speed")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pause", type=float, default=1.5,
                   help="seconds to hold between episodes")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Rendering must stay on: the VLA consumes camera images, and they are
    # rendered off-screen independently of what the viewer window shows.
    env = PickPlaceEnv(enable_rendering=True, hide_sites=True)

    agent = SAC(obs_dim=C.SAC_OBS_DIM, act_dim=C.SAC_ACTION_DIM,
                hidden=C.SAC_HIDDEN, buffer_size=1, device=device)
    agent.load(args.sac_checkpoint, map_location=device)
    agent.actor.eval()
    print("SAC reach policy loaded from %s" % args.sac_checkpoint)

    if args.vla == "scripted":
        vla = ScriptedVLAStandIn(env)
        print("manipulation: scripted stand-in")
    else:
        from .vla_policy import SmolVLAController
        vla = SmolVLAController(args.vla_checkpoint, device=device)
        print("manipulation: SmolVLA from %s (on %s)" % (args.vla_checkpoint, device))

    ctrl = HybridController(env, agent, vla)
    dt = (1.0 / C.CONTROL_HZ) / max(1e-6, args.speed)

    print("\nopening viewer -- drag to orbit, scroll to zoom, close the window to stop\n")
    n_ok = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            if not viewer.is_running():
                break

            state = {"phase": None, "count": 0, "last": time.time()}

            def on_step(phase):
                # Announce transitions rather than every step.
                if phase != state["phase"]:
                    if state["phase"] is not None:
                        print("[ep %d]  %-9s %-8s %4d steps"
                              % (ep, state["phase"], PHASE_OWNER[state["phase"]],
                                 state["count"]))
                    state["phase"] = phase
                    state["count"] = 0
                state["count"] += 1

                viewer.sync()
                # Sleep only the leftover budget: VLA inference already
                # consumes most of a control tick.
                elapsed = time.time() - state["last"]
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                state["last"] = time.time()

            res = ctrl.run_episode(seed=1000 + ep, on_step=on_step)

            if state["phase"] is not None:
                print("[ep %d]  %-9s %-8s %4d steps"
                      % (ep, state["phase"], PHASE_OWNER[state["phase"]], state["count"]))
            n_ok += int(res["success"])
            print("[ep %d]  %s   %d steps, place error %.1f cm%s\n"
                  % (ep, "SUCCESS" if res["success"] else "FAILED",
                     res["total_steps"], 100 * res["place_error"],
                     "" if res["success"] else "   (failed at: %s)" % res["failed_phase"]))

            # Hold on the final frame so the outcome is visible.
            end = time.time() + args.pause
            while time.time() < end and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    print("%d/%d succeeded" % (n_ok, args.episodes))
    env.close()


if __name__ == "__main__":
    main()
