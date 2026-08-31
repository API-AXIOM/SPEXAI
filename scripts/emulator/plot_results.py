"""Summary figures for the ablation + baseline study.

Currently produces:
  figures/data_efficiency.png - test mean relative error vs training-set
      size for the classical interpolation baselines, with the neural
      variants (trained on the full set) overplotted for comparison.

Reads the JSON outputs of scripts/baseline_interpolation.py and
scripts/benchmark_operator.py from --rundir.

    python scripts/plot_results.py [--rundir DIR]
"""

import argparse
import glob
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
COLORS = {"linear": "#2a78d6", "pchip": "#1baf7a", "pca_gp": "#eda100",
          "neural": "#4a3aa7"}
LABELS = {"linear": "linear interp.", "pchip": "PCHIP interp.",
          "pca_gp": "PCA + GP"}


def load_baseline_curves(rundir, split="test"):
    curves = {m: [] for m in LABELS}
    for f in glob.glob(os.path.join(rundir, f"baselines_{split}*.json")):
        r = json.load(open(f))
        for m in curves:
            if m in r:
                curves[m].append((r[m]["n_train"], r[m]["overall"]["mre_mean"]))
    return {m: np.array(sorted(v)) for m, v in curves.items() if v}


def plot_data_efficiency(rundir, split="test"):
    curves = load_baseline_curves(rundir, split)
    bench = json.load(open(os.path.join(rundir, f"benchmark_{split}.json")))
    nn = {v: r["overall"]["mre_mean"] for v, r in bench.items()}
    n_full = max(c[:, 0].max() for c in curves.values())

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # vertical nudge (points) per label; linear and PCHIP end nearly on top
    # of each other, so push them apart
    nudge = {"linear": 8, "pchip": -8, "pca_gp": 0}
    for m, c in curves.items():
        ax.plot(c[:, 0], c[:, 1], "-o", color=COLORS[m], lw=2, ms=5,
                markerfacecolor=COLORS[m], markeredgecolor=SURFACE,
                markeredgewidth=1)
        ax.annotate(LABELS[m], (c[-1, 0], c[-1, 1]),
                    xytext=(8, nudge.get(m, 0)), textcoords="offset points",
                    color=COLORS[m], fontsize=9, va="center")

    # neural variants trained on the full set; no_fourier is off-scale
    best = min(nn, key=nn.get)
    for v, mre in nn.items():
        if v == "no_fourier":
            continue
        ax.plot([n_full], [mre], marker="D", ls="none",
                ms=7 if v == best else 5, color=COLORS["neural"],
                alpha=1.0 if v == best else 0.45,
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.annotate(f"neural variants\n(best: {best.replace('_', ' ')})",
                (n_full, nn[best]), xytext=(-14, -4),
                textcoords="offset points", ha="right", va="top",
                color=COLORS["neural"], fontsize=9)

    ax.axhline(0.01, color=INK2, lw=0.8, ls=":")
    ax.annotate("1% target", (110, 0.0115), color=INK2, fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="training spectra",
           ylabel=f"mean relative error ({split})")
    ax.set_title("Element 26 (Fe): emulator error vs training-set size",
                 fontsize=11, color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out = os.path.join(rundir, "figures", "data_efficiency.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    plot_data_efficiency(args.rundir, args.split)
