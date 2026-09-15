"""
pickplace/record_rollout.py -- record the hybrid doing a full pick-and-place.

    python -m pickplace.record_rollout --vla scripted
    python -m pickplace.record_rollout --vla smolvla --episodes 3

Writes an mp4 with the phase and controller burned into each frame, so it is
obvious which half of the hybrid is driving at any moment. That labelling is
the point: a silent video of an arm moving tells you almost nothing about a
system whose whole design is who-controls-what-and-when.
"""

import argparse
from pathlib import Path

import numpy as np

from .env import PickPlaceEnv
from .orchestrator import HybridController, ScriptedVLAStandIn
from .sac import SAC
from . import config as C


# Which controller owns each phase. Drives the on-screen label.
PHASE_OWNER = {
    "reach": "SAC", "grasp": "VLA", "retract": "SAC",
    "transfer": "SAC", "place": "VLA",
}


def annotate(frame, phase, owner, step, ok=None):
    """Draw a caption bar. cv2 if available, otherwise a plain colour strip.

    The strip alone still carries the signal -- blue for SAC, orange for the
    VLA -- so the video stays readable even without OpenCV.
    """
    img = np.ascontiguousarray(frame)
    colour = (60, 120, 220) if owner == "SAC" else (240, 150, 40)
    bar = 22
    img[:bar] = colour
    try:
        import cv2
        text = "%s | %s | t=%d" % (owner, phase, step)
        if ok is not None:
            text += " | %s" % ("SUCCESS" if ok else "FAIL")
        cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla", default="scripted", choices=["scripted", "smolvla"])
    p.add_argument("--vla-checkpoint", default=str(C.VLA_CKPT_DIR / "checkpoint"))
    p.add_argument("--sac-checkpoint", default=str(C.SAC_CKPT_DIR / "sac_reach_best.pt"))
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--out", default=str(C.EVAL_DIR / "rollout.mp4"))
    p.add_argument("--camera", default="front_cam", choices=C.CAMERA_NAMES + ["both"])
    p.add_argument("--fps", type=int, default=C.CONTROL_HZ)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    env = PickPlaceEnv(enable_rendering=True, hide_sites=True)

    agent = SAC(obs_dim=C.SAC_OBS_DIM, act_dim=C.SAC_ACTION_DIM,
                hidden=C.SAC_HIDDEN, buffer_size=1, device=args.device)
    agent.load(args.sac_checkpoint, map_location=args.device)
    agent.actor.eval()

    if args.vla == "scripted":
        vla = ScriptedVLAStandIn(env)
    else:
        from .vla_policy import SmolVLAController
        vla = SmolVLAController(args.vla_checkpoint, device=args.device)

    ctrl = HybridController(env, agent, vla)
    frames = []
    results = []

    for ep in range(args.episodes):
        step = [0]

        def on_step(phase):
            imgs = env.render_all_cameras()
            if args.camera == "both":
                frame = np.concatenate([imgs[c] for c in C.CAMERA_NAMES], axis=1)
            else:
                frame = imgs[args.camera]
            frames.append(annotate(frame, phase, PHASE_OWNER[phase], step[0]))
            step[0] += 1

        res = ctrl.run_episode(seed=1000 + ep, on_step=on_step)
        results.append(res)
        print("episode %d: %s  steps %d  place_err %.3f m  %s"
              % (ep, "SUCCESS" if res["success"] else "FAIL", res["total_steps"],
                 res["place_error"], res["failed_phase"] or ""))

        # Hold the final frame so the outcome is readable rather than a flash.
        if frames:
            for _ in range(args.fps):
                frames.append(annotate(frames[-1][:, :, :].copy(), "done",
                                       "SAC" if res["success"] else "VLA",
                                       step[0], ok=res["success"]))

    import imageio
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)

    n_ok = sum(r["success"] for r in results)
    print("\n%d/%d succeeded. wrote %s (%d frames, %.1fs)"
          % (n_ok, len(results), out, len(frames), len(frames) / args.fps))
    env.close()


if __name__ == "__main__":
    main()
