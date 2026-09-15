"""
pickplace/status.py -- what is the pipeline doing right now?

    python -m pickplace.status
    python -m pickplace.status --watch          # refresh every 30s

Reads state off disk rather than tracking processes, so it works from any
terminal, survives a session restart, and does not care who started the job.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import config as C


BLOCKS = " .:-=+*#%@"


def sparkline(values, width=48):
    """A crude loss curve. Good enough to see a plateau or a divergence."""
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return BLOCKS[0] * len(values)
    return "".join(
        BLOCKS[min(len(BLOCKS) - 1, int((v - lo) / (hi - lo) * (len(BLOCKS) - 1)))]
        for v in values
    )


def human_time(seconds):
    seconds = int(max(0, seconds))
    if seconds < 90:
        return "%ds" % seconds
    if seconds < 5400:
        return "%dm" % (seconds // 60)
    return "%.1fh" % (seconds / 3600)


def bar(frac, width=28):
    n = int(round(width * max(0.0, min(1.0, frac))))
    return "[" + "#" * n + "." * (width - n) + "]"


# ==========================================================================
def vla_status():
    log = C.VLA_CKPT_DIR / "train_log.jsonl"
    if not log.exists():
        return ["SmolVLA fine-tune   not started"]

    rows = []
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return ["SmolVLA fine-tune   log is empty"]

    last = rows[-1]
    total = C.VLA_STEPS
    frac = last["step"] / total
    rate = last["step"] / max(1e-9, last["elapsed_s"])
    eta = (total - last["step"]) / max(1e-9, rate)

    # Stale check: if the newest record is old, the job probably died.
    age = time.time() - log.stat().st_mtime
    state = "RUNNING" if age < 180 else "STALLED or finished (log idle %s)" % human_time(age)

    out = [
        "SmolVLA fine-tune   %s" % state,
        "  %s %d/%d  (%.1f%%)" % (bar(frac), last["step"], total, 100 * frac),
        "  loss %.4f   grad %.2f   lr %.2e   %.2f it/s   elapsed %s   ETA %s"
        % (last["loss"], last["grad_norm"], last["lr"], rate,
           human_time(last["elapsed_s"]), human_time(eta)),
        "  loss %.3f -> %.3f  %s" % (rows[0]["loss"], last["loss"],
                                     sparkline([r["loss"] for r in rows])),
    ]
    ckpt = C.VLA_CKPT_DIR / "checkpoint"
    if (ckpt / "step.txt").exists():
        out.append("  checkpoint at step %s" % (ckpt / "step.txt").read_text().strip())
    else:
        out.append("  no checkpoint yet (first save at step 1000)")
    return out


def sac_status():
    csv = C.SAC_CKPT_DIR / "train_log.csv"
    if not csv.exists():
        return ["SAC reach policy    not trained"]
    lines = [l for l in csv.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return ["SAC reach policy    no evaluations logged yet"]
    header = lines[0].split(",")
    last = dict(zip(header, lines[-1].split(",")))
    best = (C.SAC_CKPT_DIR / "sac_reach_best.pt").exists()
    return [
        "SAC reach policy    trained, checkpoint %s"
        % ("present" if best else "MISSING"),
        "  last eval @step %s   cube %.0f%%  drop %.0f%%  home %.0f%%  scatter %.0f%%"
        % (last["step"], 100 * float(last["eval_cube"]), 100 * float(last["eval_drop"]),
           100 * float(last["eval_home"]), 100 * float(last["eval_scatter"])),
        "  overall %.1f%%" % (100 * float(last["eval_overall"])),
    ]


def data_status():
    out = ["Data"]
    n_scripted = len(list(Path(C.SCRIPTED_DEMO_DIR).glob("*.npz"))) \
        if Path(C.SCRIPTED_DEMO_DIR).exists() else 0
    n_teleop = len(list(Path(C.CROPPED_TELEOP_DIR).glob("*.npz"))) \
        if Path(C.CROPPED_TELEOP_DIR).exists() else 0
    out.append("  scripted segments %d   replayed teleop %d" % (n_scripted, n_teleop))

    meta_path = C.OUT_ROOT / "packed" / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        out.append("  packed: %d episodes, %d frames, tasks %d"
                   % (m["total_episodes"], m["total_frames"], len(m["tasks"])))
        for k in sorted(m["episodes_by_kind"]):
            out.append("    %-16s %4d episodes  %6d frames"
                       % (k, m["episodes_by_kind"][k], m["frames_by_kind"][k]))
    else:
        out.append("  packed: NOT BUILT -- run `python -m pickplace.dataset --pack`")
    return out


def eval_status():
    out = []
    for tag, label in (("scripted", "scripted ablation"), ("smolvla", "real VLA")):
        f = C.EVAL_DIR / ("summary_%s.json" % tag)
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        line = "  %-18s %5.1f%% over %d episodes" % (
            label, 100 * s["success_rate"], s["episodes"])
        fails = {k: v for k, v in s["failed_phase_counts"].items() if v}
        if fails:
            line += "   failures: " + ", ".join("%s %d" % kv for kv in fails.items())
        out.append(line)

    # The checkpoint sweep -- success vs training steps. This is the most
    # informative artifact the pipeline produces and it was missing here:
    # the headline eval files only ever hold the LAST run, so a sweep showing
    # the VLA going 40% -> 93% was invisible on the dashboard.
    sweeps = sorted(C.EVAL_DIR.glob("*sweep*.json"),
                    key=lambda p: p.stat().st_mtime)
    if sweeps:
        rows = json.loads(sweeps[-1].read_text())
        out.append("")
        out.append("  checkpoint sweep (%s, %d episodes each)"
                   % (sweeps[-1].stem, rows[0]["summary"]["episodes"]))
        out.append("     %7s %9s %8s %8s   %s"
                   % ("step", "success", "grasp", "place", "failures"))
        for r in rows:
            s = r["summary"]
            fails = ", ".join("%s %d" % kv
                              for kv in s["failed_phase_counts"].items() if kv[1])
            out.append("     %7d %8.0f%% %7.0f%% %7.0f%%   %s"
                       % (r["step"], 100 * s["success_rate"],
                          100 * r["grasp_rate"], 100 * r["place_rate"],
                          fails or "-"))
        best = max(rows, key=lambda r: r["summary"]["success_rate"])
        out.append("     best %.0f%% at step %d"
                   % (100 * best["summary"]["success_rate"], best["step"]))

    return ["Evaluations"] + out if out else ["Evaluations          none run yet"]


def gpu_status():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
             "memory.total,temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return []
        name, util, used, total, temp = [x.strip() for x in r.stdout.strip().split(",")]
        return ["GPU                 %s   %s util   %s / %s   %sC"
                % (name, util, used, total, temp)]
    except Exception:
        return []


def render():
    lines = ["", "=" * 72, " pickplace pipeline status   %s"
             % time.strftime("%H:%M:%S"), "=" * 72]
    for block in (vla_status, sac_status, data_status, eval_status, gpu_status):
        out = block()
        if out:
            lines.extend(out)
            lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="refresh until interrupted")
    p.add_argument("--interval", type=int, default=30)
    args = p.parse_args()

    if not args.watch:
        print(render())
        return
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
