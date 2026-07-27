"""Unsupervised weekend sweep for the temperature-conditioning fix.

Diagnosis (docs/emulator_technical_report.tex, residual_fft.py): the failing-band
elements are underfit in the T-range where their dominant line complex turns
on (sharpest d(emissivity)/dT) -- cold for Na, mid for Ca, hot for Cr -- as a
smooth low-frequency continuum misfit, NOT energy-axis jitter. This sweep
tests the mechanism-matched fix (a Fourier T-embedding on the trunk
conditioning, --film_t_freqs) against a convergence control (learning rate),
with the primary comparison replicated across the cold->hot trend.

Design:
  * PRIMARY  baseline vs +T-embedding on Na(11, cold) / Ca(20, mid) / Cr(24,
             hot), all at lr 1e-3 -- does the fix work AND generalise?
  * SECONDARY convergence / sampling knobs on Na only.
  All arms: from-scratch (no warm start), full --steps, early stopping OFF
  (controlled experiment -> comparable curves), reweight mode on the full
  grid. Fully pre-specified, so it runs unattended as a sequential queue.

Resumable: an arm whose result.json exists is skipped, so re-running the
same command after an interruption continues where it stopped. A crashing
arm is logged and does NOT stop the sweep. After every arm the ranked
summary (sweep_summary.{json,md}) is rebuilt, so partial results are always
available.

    nohup python scripts/run_weekend_sweep.py \
        --dataroot ~/data/spexai_data \
        --runroot  ~/data/spexai_data/runs_sweep \
        > ~/data/spexai_data/runs_sweep/sweep.log 2>&1 &

Enable --compile 1 (after a quick timing check) to roughly halve wall time;
left OFF by default so a torch.compile incompatibility cannot fail the whole
weekend at once.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from benchmark_operator import load_model, predict_all  # noqa: E402
from spexai.train.operator import FixedGridMLP  # noqa: E402
from spexai.train.train_operator import FLOOR, SpectrumData  # noqa: E402

# flags shared by every arm: from-scratch, full grid, controlled (no early
# stop), WSD schedule matching the original runs.
COMMON = ["--mode", "reweight", "--n_train", "0", "--schedule", "wsd",
          "--batch", "256", "--early_stop_patience", "0",
          "--eval_every", "2000", "--tag", "sweep"]

# hard-T regime of each probed element (from residual_fft.py --stratify)
REGIME = {11: "cold", 20: "mid", 24: "hot"}

# the program. `flags` are appended to COMMON; every arm trains from scratch.
ARMS: List[Dict] = [
    # --- PRIMARY: baseline vs T-embedding across the cold->hot trend ---
    dict(tag="na_base",   z=11, flags=["--lr", "1e-3"]),
    dict(tag="na_tembed", z=11, flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    dict(tag="ca_base",   z=20, flags=["--lr", "1e-3"]),
    dict(tag="ca_tembed", z=20, flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    dict(tag="cr_base",   z=24, flags=["--lr", "1e-3"]),
    dict(tag="cr_tembed", z=24, flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    # --- SECONDARY: convergence / sampling knobs on Na only ---
    dict(tag="na_lr3e3",  z=11, flags=["--lr", "3e-3"]),
    dict(tag="na_lr5e4",  z=11, flags=["--lr", "5e-4"]),
    dict(tag="na_prmix",  z=11, flags=["--lr", "1e-3", "--pr_mix", "0.8"]),
    dict(tag="na_curr01", z=11, flags=["--lr", "1e-3", "--curriculum_frac", "0.1"]),
    dict(tag="na_anneal", z=11, flags=["--lr", "1e-3", "--wsd_decay_frac", "0.45",
                                       "--lr_min_frac", "0.005"]),
]

# temperature-band edges (keV) for stratified test metrics: <1, 1-5, >5
T_EDGES = [0.0, 1.0, 5.0, np.inf]
BAND_LABELS = ["cold(<1)", "mid(1-5)", "hot(>5)"]


def steps_to_yield(history: List[Dict], thresh: float = 50.0) -> Optional[int]:
    """First eval step whose val yield@0.1% reached `thresh` (None if never)."""
    for r in history:
        if r.get("val_yield_01pct", 0.0) >= thresh:
            return int(r["step"])
    return None


@torch.no_grad()
def test_band_metrics(ckpt: str, cachedir: str, device: str) -> Dict:
    """Overall + per-T-band test-set MRE and yield@0.1% for a checkpoint."""
    data = SpectrumData(cachedir)
    model, _ = load_model(ckpt, data)
    model = model.to(device)
    fixed = isinstance(model, FixedGridMLP)
    idx = data.test_idx
    temps = data.temps[idx].numpy()
    truth = data.logflux[idx].numpy()
    pred = predict_all(model, data, idx, device, fixed)
    valid = truth > FLOOR
    d = np.clip(pred - np.clip(truth, FLOOR, None), -4, 4)
    eps = np.abs(10.0 ** d - 1.0)
    mre = (np.where(valid, eps, 0.0).sum(1)
           / np.maximum(valid.sum(1), 1))               # (S,) per-spectrum MRE

    def stats(sub: np.ndarray) -> Dict:
        if sub.sum() == 0:
            return {"n": 0}
        m = mre[sub]
        return {"n": int(sub.sum()), "mre_mean": float(m.mean()),
                "mre_median": float(np.median(m)),
                "yield_01pct": float((m <= 1e-3).mean() * 100),
                "yield_1pct": float((m <= 1e-2).mean() * 100)}

    out = {"overall": stats(np.ones(len(mre), bool))}
    for lab, lo, hi in zip(BAND_LABELS, T_EDGES[:-1], T_EDGES[1:]):
        out[lab] = stats((temps >= lo) & (temps < hi))
    return out


def run_arm(arm: Dict, args: argparse.Namespace) -> Dict:
    """Train one arm from scratch, then evaluate it. Resumable + crash-safe."""
    outdir = os.path.join(args.runroot, arm["tag"])
    os.makedirs(outdir, exist_ok=True)
    result_path = os.path.join(outdir, "result.json")
    if os.path.isfile(result_path) and not args.force:
        with open(result_path) as f:
            return json.load(f)

    cachedir = os.path.join(args.dataroot, "processed", f"element{arm['z']}")
    ckpt = os.path.join(outdir, "sweep.pt")
    hist_path = os.path.join(outdir, "sweep_history.json")
    # per-arm --steps override (arm["steps"]) falls back to the sweep default;
    # lets a follow-on mix budgets (e.g. 200k arms) in one program.
    cmd = [sys.executable, "-m", "spexai.train.train_adaptive",
           "--cachedir", cachedir, "--outdir", outdir,
           "--steps", str(arm.get("steps", args.steps)),
           "--compile", str(args.compile),
           "--diag_plots", str(args.diag_plots),
           *COMMON, *arm["flags"]]
    result = {"tag": arm["tag"], "z": arm["z"], "regime": REGIME.get(arm["z"], ""),
              "flags": arm["flags"], "cmd": " ".join(cmd)}
    t0 = time.time()
    print(f"\n=== ARM {arm['tag']} (Z{arm['z']}, {REGIME.get(arm['z'], '')}) ===\n"
          f"    {' '.join(arm['flags'])}", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=REPO)
    except subprocess.CalledProcessError as e:
        result.update(status="train_failed", error=str(e),
                      elapsed_min=(time.time() - t0) / 60)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    ARM {arm['tag']} FAILED in training: {e}", flush=True)
        return result

    try:
        with open(hist_path) as f:
            hist = json.load(f)
        result["best_val"] = hist.get("best", {})
        result["steps_to_yield50"] = steps_to_yield(hist["history"], 50.0)
        result["steps_to_yield_any"] = steps_to_yield(
            [r for r in hist["history"] if r.get("val_yield_01pct", 0) > 0], 0.0)
        result["test"] = test_band_metrics(ckpt, cachedir, args.device)
        result["status"] = "ok"
    except Exception as e:  # eval must never kill the sweep
        result.update(status="eval_failed", error=repr(e))
    result["elapsed_min"] = (time.time() - t0) / 60
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    ARM {arm['tag']} done in {result['elapsed_min']:.0f} min "
          f"(status={result['status']})", flush=True)
    return result


def collect(results: List[Dict], runroot: str) -> None:
    """Rebuild the ranked summary (best test yield@0.1% first)."""
    with open(os.path.join(runroot, "sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    def key(r: Dict) -> float:
        return -(r.get("test", {}).get("overall", {}).get("yield_01pct", -1))

    rows = ["| arm | Z | regime | overall y@0.1% | MRE | cold | mid | hot "
            "| steps→50% | min |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([x for x in results if x.get("status") == "ok"], key=key):
        t = r["test"]
        o = t["overall"]

        def y(lab: str) -> str:
            b = t.get(lab, {})
            return f"{b.get('yield_01pct', float('nan')):.0f}" if b.get("n") else "-"
        s2y = r.get("steps_to_yield50")
        rows.append(
            f"| {r['tag']} | {r['z']} | {r['regime']} | "
            f"{o['yield_01pct']:.1f} | {o['mre_mean']:.4f} | "
            f"{y('cold(<1)')} | {y('mid(1-5)')} | {y('hot(>5)')} | "
            f"{s2y if s2y else '-'} | {r.get('elapsed_min', 0):.0f} |")
    bad = [x for x in results if x.get("status") not in ("ok", None)]
    if bad:
        rows.append("")
        rows.append("Non-ok arms: " + ", ".join(
            f"{x['tag']}({x['status']})" for x in bad))
    with open(os.path.join(runroot, "sweep_summary.md"), "w") as f:
        f.write("\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=os.path.expanduser("~/data/spexai_data"),
                    help="holds processed/element<Z>/ caches")
    ap.add_argument("--runroot", default=os.path.expanduser("~/data/spexai_data/runs_sweep"),
                    help="per-arm outputs + sweep_summary.{json,md}")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--compile", type=int, default=0,
                    help="torch.compile the forward hot path; ~2x faster but "
                         "left OFF by default so a compile incompatibility "
                         "cannot fail every arm. Verify with a short run first.")
    ap.add_argument("--diag_plots", type=int, default=1)
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these arm tags (default: the whole program)")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms even if result.json exists")
    ap.add_argument("--device", default=None,
                    help="eval device (default: cuda if available else cpu)")
    args = ap.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.runroot, exist_ok=True)

    arms = ARMS if not args.only else [a for a in ARMS if a["tag"] in args.only]
    print(f"weekend sweep: {len(arms)} arms, steps={args.steps}, "
          f"compile={args.compile}, device={args.device}\n"
          f"  runroot={args.runroot}", flush=True)
    results = []
    for arm in arms:
        results.append(run_arm(arm, args))
        collect(results, args.runroot)   # refresh summary after every arm
    print(f"\nsweep complete. ranked summary: "
          f"{os.path.join(args.runroot, 'sweep_summary.md')}", flush=True)


if __name__ == "__main__":
    main()
