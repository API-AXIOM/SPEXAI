"""Comparison of the adaptive-data training arms (train_adaptive).

Reads <mode>_history.json for the three arms from --rundir and produces:
  * learning curves (val MRE and yield@1% vs step, one line per arm)
  * the acquisition map for the adaptive arm: histogram of acquired
    temperatures over the training grid, with gate-rejected intervals
    (the shortlist for real SPEX runs) marked
  * a summary table (printed and written to adaptive_summary.md)

    python scripts/plot_adaptive.py [--rundir .../runs/element26/adaptive]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
COLORS = {"baseline": "#52514e", "reweight": "#2a78d6",
          "adaptive": "#c23b2a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/adaptive")
    ap.add_argument("--tags", nargs="+",
                    default=["baseline", "reweight", "adaptive"])
    args = ap.parse_args()

    runs = {}
    for tag in args.tags:
        path = os.path.join(args.rundir, f"{tag}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                runs[tag] = json.load(f)
        else:
            print(f"missing {path}, skipping")
    if not runs:
        raise SystemExit("no history files found")

    fig = plt.figure(figsize=(11, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8],
                          hspace=0.35, wspace=0.28)

    ax_mre = fig.add_subplot(gs[0, 0])
    ax_yld = fig.add_subplot(gs[0, 1])
    for tag, r in runs.items():
        h = r["history"]
        steps = [x["step"] for x in h]
        c = COLORS.get(tag, INK)
        ax_mre.plot(steps, [x["val_mre_mean"] for x in h],
                    color=c, lw=1.4, label=tag)
        ax_yld.plot(steps, [x["val_yield_1pct"] for x in h],
                    color=c, lw=1.4, label=tag)
    ax_mre.set(yscale="log", xlabel="step", ylabel="val MRE (SPEX only)")
    ax_yld.set(xlabel="step", ylabel="val yield@1% (%)")
    ax_mre.legend(frameon=False, fontsize=9)

    ax_acq = fig.add_subplot(gs[1, :])
    adapt = runs.get("adaptive")
    if adapt and adapt.get("acquired"):
        lt = np.array([a["logT"] for a in adapt["acquired"]])
        ax_acq.hist(lt, bins=60, color=COLORS["adaptive"], alpha=0.85,
                    label=f"acquired synthetic T ({len(lt)})")
    if adapt:
        for j, iv in enumerate(adapt.get("gate_rejected_intervals", [])):
            ax_acq.axvspan(iv["logT_lo"], iv["logT_hi"], color="#eda100",
                           alpha=0.35, lw=0,
                           label="gate-rejected" if j == 0 else None)
    ax_acq.set(xlabel="log10 T (keV)", ylabel="count")
    ax_acq.set_title("adaptive arm: where data was added "
                     "(orange: interpolation untrusted -> run real SPEX)",
                     loc="left", fontsize=10, color=INK2)
    if adapt:
        ax_acq.legend(frameon=False, fontsize=9)

    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        for s in ax.spines.values():
            s.set_visible(False)

    out = os.path.join(args.rundir, "adaptive_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)

    md = ["| arm | best step | val MRE | val yield1% | val yield10% | "
          "synthetic spectra |", "|---|---|---|---|---|---|"]
    for tag, r in runs.items():
        b = r["best"]
        md.append(f"| {tag} | {b.get('step')} | {b['val_mre_mean']:.4f} | "
                  f"{b['val_yield_1pct']:.2f} | {b['val_yield_10pct']:.2f} | "
                  f"{len(r.get('acquired', []))} |")
    table = "\n".join(md)
    print(table)
    with open(os.path.join(args.rundir, "adaptive_summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
