"""Headline figure: systematic emulator bias vs Poisson error as a function of
in-band counts, per published parameter, for the literature-strategy Perseus
fit. Reads results/bias_{single,dem}.npz written by fisher_bias.py.

The ratio |b_sys| / sigma_stat(N) = (|b_sys|/sigma_ref) * sqrt(N/N_ref) is a
straight slope-1/2 line in log-log space, crossing 1 (bias = noise) at the
crossover N*. Below the line the emulator floor is irrelevant; above it, the
emulator biases that parameter beyond the statistical error.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Results live in the shared data tree written by fisher_bias.py, NOT beside
# this script -- the local ./results path predates the scripts/ refactor and
# was the reason the figures referenced from docs/ did not exist.
from spexai.config import RESULTS as _RESULTS_BASE
RESULTS = os.path.join(_RESULTS_BASE, "hot_floor")

# parameter -> (group, colour); groups match the science framing.
SCIENCE = {"Cr", "Mn", "Ni"}
ALPHA = {"Si", "S", "Ar", "Ca"}
NUIS = {"kT", "T_mean", "T_sigma", "sigma_v", "n_h", "log_norm"}
COLOR = {"Fe": "#111111", "Cr": "#d62728", "Mn": "#ff7f0e", "Ni": "#9467bd",
         "Si": "#7fb0d0", "S": "#7fb0d0", "Ar": "#7fb0d0", "Ca": "#7fb0d0"}

# realistic XRISM Perseus in-band count anchors.
XRISM_REGION = 4e4        # per-region minimum from arXiv:2606.17141
XRISM_DEEP = 1e6          # deep stacked core (order of magnitude)


def load(mode, tag=""):
    suffix = f"_{tag}" if tag else ""
    p = os.path.join(RESULTS, f"bias_{mode}{suffix}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    # response provenance: refuse to plot a run built with a flat effective
    # area as if it were instrument-folded (see check_truth_response).
    if "arf" not in d:
        print(f"WARNING: {os.path.basename(p)} records no ARF -- it predates "
              f"the response fix and its N* values are not valid.")
    return d


def panel(ax, data, mode, counts):
    names = list(data["names"])
    b, s, nref = data["b_sys"], data["sigma_ref"], float(data["n_ref"])
    for i, nm in enumerate(names):
        ratio0 = abs(b[i]) / s[i]                 # at N_REF
        y = ratio0 * np.sqrt(counts / nref)
        grp = ("science" if nm in SCIENCE else "Fe" if nm == "Fe"
               else "alpha" if nm in ALPHA else "nuisance")
        lw = 2.4 if grp in ("science", "Fe") else 1.2
        ls = "--" if grp == "nuisance" else "-"
        c = COLOR.get(nm, "#9aa0a6")
        ax.plot(counts, y, ls, color=c, lw=lw, label=nm, zorder=3 if lw > 2 else 2)
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.text(counts[0], 1.15, "bias = noise", color="0.4", fontsize=8)
    for x, lab in ((XRISM_REGION, "XRISM/region"), (XRISM_DEEP, "deep core")):
        ax.axvline(x, color="0.75", lw=1.0)
        ax.text(x, ax.get_ylim()[0] if False else 3e-4, lab, rotation=90,
                va="bottom", ha="right", color="0.55", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("in-band counts")
    ax.set_title(f"{mode.upper()}  (|$b_{{sys}}$| / $\\sigma_{{stat}}$)")
    ax.grid(True, which="both", alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="suffix of the bias_{mode}<_tag>.npz to plot")
    args = ap.parse_args()
    counts = np.logspace(3.5, 8.5, 200)
    modes = [m for m in ("single", "dem") if load(m, args.tag) is not None]
    if not modes:
        sys.exit(f"no {RESULTS}/bias_*{'_' + args.tag if args.tag else ''}.npz "
                 f"found -- run fisher_bias.py first")
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 5.0),
                             squeeze=False)
    for ax, m in zip(axes[0], modes):
        panel(ax, load(m, args.tag), m, counts)
    axes[0][0].set_ylabel(r"|systematic bias| / statistical error")
    # one shared legend
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(RESULTS,
                       f"bias_vs_counts{'_' + args.tag if args.tag else ''}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
