"""Do any bias-campaign temperatures sit near a training-cache step?

Background: `find_cache_steps.py` shows the coarse generation `var gacc` stamped
step discontinuities (up to +45%) into 10 of 30 elements' training spectra, 16
of them above 2 keV. This asks whether the campaign actually evaluates near one.

**Why proximity is the right question, and what it does NOT mean.** The bias
campaign compares the emulator against `SpexTruthModel`, which PCHIPs the *same*
training rows. Both therefore inherit the same steps, and away from a step they
inherit them identically, so the defect largely cancels and the measured bias is
still a fair emulator-vs-truth number. Right AT a step the cancellation fails:
a discontinuity is precisely where a monotone-cubic interpolator and a smooth
network diverge -- PCHIP flattens against it while the network smooths across
it. So a campaign point near a step can show inflated bias that is an artefact
of the data, not a property of the emulator.

Separately, and NOT measured here: every result is expressed relative to a truth
standard that itself carries the steps, so absolute fidelity to real SPEX is a
different and larger question (see the gacc systematic in the P3 summary).

Temperatures checked:
  * the Perseus fiducial kT, and any single-T sweep points found in the jsonl;
  * the DEM temperature grid, which is FIXED at 48 log-spaced points over
    0.7-10 keV regardless of the sampled T_mean -- so the DEM forward touches
    far more temperatures than the sweep's T_mean values alone suggest.

Usage:
    python scripts/inference/crosscheck_steps.py \
        --steps ~/work/data/spexai/results/cache_audit/steps_all.json \
        [--sweep <bias_single_n20_s3.jsonl>] [--tol 0.01]
"""
import argparse
import json
import os

import numpy as np

SYMBOL = {11: "Na", 13: "Al", 18: "Ar", 20: "Ca", 24: "Cr", 25: "Mn",
          26: "Fe", 27: "Co", 28: "Ni", 29: "Cu"}
PERSEUS_KT = 3.9
DEM_LO, DEM_HI, DEM_N = 0.7, 10.0, 48


def campaign_temps(sweep):
    """{label: array of temperatures the campaign evaluates the emulator at}."""
    out = {"Perseus fiducial (single-T)": np.array([PERSEUS_KT])}
    # gaussian_dem's default grid: campaign.gaussian_dem -> td.TempGrid(lo, hi, n)
    out[f"DEM grid ({DEM_N} pts, fixed)"] = np.logspace(
        np.log10(DEM_LO), np.log10(DEM_HI), DEM_N)
    if sweep and os.path.exists(sweep):
        kt = []
        with open(sweep) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # "point" is the integer index; the sampled values are under
                # "params" (older records may inline them at the top level)
                p = rec.get("params", rec)
                if not isinstance(p, dict):
                    continue
                for k in ("kT", "T_mean"):
                    if k in p:
                        kt.append(float(p[k]))
        if kt:
            out[f"sweep points ({len(kt)})"] = np.array(sorted(kt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="flag when |T - T_step|/T_step is below this")
    args = ap.parse_args()

    with open(args.steps) as fh:
        steps = {int(k): v for k, v in json.load(fh).items() if v}
    allsteps = [(z, s) for z, v in steps.items() for s in v]
    print(f"{len(allsteps)} steps across {len(steps)} elements; "
          f"flagging within {args.tol:.1%} in temperature\n")

    temps = campaign_temps(args.sweep)
    worst = []
    for label, tt in temps.items():
        print(f"--- {label}: {len(tt)} temperature(s), "
              f"{tt.min():.3f}-{tt.max():.3f} keV")
        hits = []
        for z, s in allsteps:
            # a step spans [T_lo, T_hi]; distance is 0 if a point lands inside
            d = np.where((tt >= s["T_lo"]) & (tt <= s["T_hi"]), 0.0,
                         np.minimum(np.abs(tt - s["T_lo"]),
                                    np.abs(tt - s["T_hi"])) / s["T_lo"])
            j = int(np.argmin(d))
            if d[j] <= args.tol:
                hits.append((d[j], z, s, tt[j]))
        for d, z, s, t in sorted(hits):
            inside = " INSIDE THE STEP" if d == 0.0 else ""
            print(f"    {SYMBOL.get(z, z):>2}  step {s['step']:+8.2%} at "
                  f"{s['T_lo']:.5f} keV   campaign T={t:.5f}  "
                  f"({d:.2%} away){inside}")
            worst.append((abs(s["step"]), z, s, t, label))
        if not hits:
            print("    none within tolerance")

    print()
    if worst:
        worst.sort(reverse=True)
        print(f"{len(worst)} (step, campaign-point) pairs within "
              f"{args.tol:.1%}. Largest by step size:")
        for mag, z, s, t, label in worst[:8]:
            print(f"  {SYMBOL.get(z, z):>2} {s['step']:+8.2%} at "
                  f"{s['T_lo']:.4f} keV -- {label} T={t:.4f}")
    else:
        print(f"NO campaign temperature falls within {args.tol:.1%} of any "
              f"step.\nThe published bias numbers are not contaminated by this "
              f"mechanism.")

    # how close does the campaign get, regardless of tolerance?
    print("\nclosest approach per affected element (any tolerance):")
    for z, v in sorted(steps.items()):
        best = None
        for label, tt in temps.items():
            for s in v:
                d = np.where((tt >= s["T_lo"]) & (tt <= s["T_hi"]), 0.0,
                             np.minimum(np.abs(tt - s["T_lo"]),
                                        np.abs(tt - s["T_hi"])) / s["T_lo"])
                j = int(np.argmin(d))
                if best is None or d[j] < best[0]:
                    best = (d[j], s, tt[j], label)
        d, s, t, label = best
        print(f"  {SYMBOL.get(z, z):>2}  {d:7.2%} away  "
              f"(step {s['step']:+8.2%} at {s['T_lo']:.4f} keV, "
              f"nearest {label} T={t:.4f})")


if __name__ == "__main__":
    main()
