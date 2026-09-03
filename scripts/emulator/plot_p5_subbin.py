"""Figure: sub-bin placement is absent, against a calibrated control.

Plots the folded flux-weighted error as a function of injected coherent
displacement (the control curve) and overlays the observed spread of the
grid-phase sweep, at both turbulent velocities. The point of the figure is
the gap between the two: the control rises steeply, the phase sweep does not
move, so the metric is sensitive to the effect and the effect is not there.

    python scripts/emulator/plot_p5_subbin.py
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from spexai.config import RESULTS

VEL = {"subbin_ctl": (180, "tab:blue", "o"), "subbin_v25": (25, "tab:red", "s")}


def load(tag: str) -> dict:
    path = os.path.join(RESULTS, "p5", f"benchmark_instruments30_{tag}.json")
    with open(path) as f:
        return json.load(f)["results"]


def series(res: dict, folded: bool):
    """(control shifts, control errors, phase-sweep errors) in percent."""
    suf = "_folded" if folded else ""
    ctl = sorted((float(k.split("control")[1].split("_")[0]), v)
                 for k, v in res.items()
                 if k.startswith("subbin_control") and k.endswith(suf)
                 and (folded or "_folded" not in k))
    ph = [v["mre_flux_mean"] * 100 for k, v in res.items()
          if k.startswith("subbin_phase") and k.endswith(suf)
          and (folded or "_folded" not in k)]
    dx = [0.0] + [c[0] for c in ctl]
    err = [np.median(ph)] + [c[1]["mre_flux_mean"] * 100 for c in ctl]
    return np.array(dx), np.array(err), np.array(ph)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "docs", "figures", "p5_subbin.pdf"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for tag, (v, colour, marker) in VEL.items():
        res = load(tag)
        dx, err, ph = series(res, folded=True)
        ax.plot(dx, err, marker=marker, color=colour, lw=1.6, ms=5,
                label=rf"injected shift, $\sigma_v={v}$ km/s")
        # observed phase sweep: plotted at zero displacement as a range
        ax.errorbar([0.0], [np.median(ph)],
                    yerr=[[np.median(ph) - ph.min()], [ph.max() - np.median(ph)]],
                    color=colour, marker=marker, ms=9, capsize=6, lw=2.5,
                    mfc="white", mew=1.8, zorder=5)
    ax.annotate("grid-phase sweep\n(full 0-1 bin range)", xy=(0.0, 0.125),
                xytext=(0.13, 0.09), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"))
    ax.set_xlabel("line displacement (fractions of a 0.4914 eV training bin)")
    ax.set_ylabel(r"folded flux-weighted error [\%]" if
                  matplotlib.rcParams["text.usetex"] else
                  "folded flux-weighted error [%]")
    ax.set_xlim(-0.03, 0.55)
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=200)
    print(f"saved {args.outfile}", flush=True)


if __name__ == "__main__":
    main()
