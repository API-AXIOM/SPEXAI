"""Plot simulated observations: one figure per instrument, one subplot per
temperature, showing the Poisson-drawn counts and the underlying expected
(mean) model. Solar abundances (every element at 1.0x solar).

    python scripts/plot_simulated.py --outdir <dir>
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, ScalarFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.response import Response
from spexai.inference.simulate import simulate_grid

RESP = os.path.expanduser("~/work/data/spexai/responses")
PLOTS = os.path.expanduser("~/work/repositories/emulator_plots/ResponseFiles")
REPRO = os.path.expanduser("~/work/repositories/reproduction_package/responses")

# Okabe-Ito, colour-blind-safe: observed = blue (primary), model = orange
C_OBS, C_MODEL, INK, GRID = "#0072B2", "#E69F00", "#222222", "#dddddd"
TEMPS = [0.7, 2.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sim_plots"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    responses = {
        "Chandra ACIS-S": Response(f"{RESP}/aciss_aimpt_cy28.rmf",
                                   f"{RESP}/aciss_aimpt_cy28.arf"),
        "XMM-Newton EPIC-PN": Response(f"{PLOTS}/PN.rmf", f"{PLOTS}/PN.arf"),
        "XRISM Resolve": Response(f"{REPRO}/xrt.rmf", f"{REPRO}/xrt.arf"),
    }
    model = JointOperatorModel(device="cpu")   # all 16 available elements
    obs = simulate_grid(model, responses, TEMPS,
                        base_params={"norm": 1e10, "velocity": 200.0,
                                     "logz": -10.0},               # solar abund
                        exposure=1e5, target_counts=5e4, seed=42)
    by_inst = {}
    for o in obs:
        by_inst.setdefault(o.instrument, []).append(o)

    for inst, olist in by_inst.items():
        olist.sort(key=lambda o: o.true_params["temp"])
        fig, axes = plt.subplots(len(olist), 1, figsize=(9.5, 8.5),
                                 sharex=True)
        for ax, o in zip(np.atleast_1d(axes), olist):
            e = o.response.chan_e_cent.numpy()
            width = (o.response.chan_e_max - o.response.chan_e_min).numpy()
            # counts per keV so different channel widths compare fairly
            obs_rate = o.counts / np.where(width > 0, width, np.nan)
            mod_rate = o.expected / np.where(width > 0, width, np.nan)
            band = o.expected > (0.02 * o.expected.max())   # occupied channels
            ax.step(e, np.where(obs_rate > 0, obs_rate, np.nan), where="mid",
                    color=C_OBS, lw=1.0, label="simulated counts")
            ax.plot(e, np.where(mod_rate > 0, mod_rate, np.nan),
                    color=C_MODEL, lw=1.6, alpha=0.9, label="expected model")
            ax.set(xscale="log", yscale="log")
            peak = float(np.nanmax(mod_rate))
            ax.set_ylim(peak / 3e3, peak * 4)          # focus on the signal range
            if band.any():
                lo, hi = max(0.2, e[band].min() * 0.9), e[band].max() * 1.1
                ax.set_xlim(lo, hi)
                ticks = [t for t in (0.2, 0.3, 0.5, 1, 2, 5, 10) if lo <= t <= hi]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, which="major", color=GRID, lw=0.6)
            ax.set_ylabel("counts keV$^{-1}$", color=INK)
            ax.text(0.985, 0.93, f"$T = {o.true_params['temp']}$ keV",
                    transform=ax.transAxes, ha="right", va="top", fontsize=12,
                    color=INK)
            ax.text(0.985, 0.83, f"{o.total_counts:,} counts",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color="#666666")
        np.atleast_1d(axes)[-1].set_xlabel("Energy (keV)", color=INK)
        np.atleast_1d(axes)[0].legend(loc="upper left", frameon=False,
                                      fontsize=10)
        fig.suptitle(f"{inst} — simulated solar-abundance plasma "
                     f"(exposure $10^5$ s, $\\sigma_v=200$ km/s)",
                     x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fname = os.path.join(args.outdir,
                             f"sim_{inst.split()[0].lower()}.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
