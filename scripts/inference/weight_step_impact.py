"""How much flux does the DEM actually draw from near a cache step?

`crosscheck_steps.py` shows the DEM's fixed 48-point temperature grid passes
close to several large steps. Proximity alone overstates the problem: a Gaussian
DEM weights its grid, and the cool points that sit near the worst steps may carry
almost no weight. This turns proximity into a flux-weighted number.

For each affected element it reports the fraction of the DEM's total weight that
sits on grid points within `--tol` of a step, and the implied worst-case flux
error (that weight fraction times the step size) -- an upper bound, since being
near a step is not the same as being wrong by its full size.

Usage:
    python scripts/inference/weight_step_impact.py \
        --steps ~/work/data/spexai/results/cache_audit/steps_all.json \
        [--tmean 3.9] [--tsigma 1.0] [--tol 0.01]
"""
import argparse
import json

import numpy as np

SYMBOL = {11: "Na", 13: "Al", 18: "Ar", 20: "Ca", 24: "Cr", 25: "Mn",
          26: "Fe", 27: "Co", 28: "Ni", 29: "Cu"}
DEM_LO, DEM_HI, DEM_N = 0.7, 10.0, 48


def dem_weights(t_mean, t_sigma):
    """Gaussian-in-T DEM on the campaign's fixed grid, normalised to sum 1.

    Mirrors campaign.gaussian_dem -> tempdist.gaussian_T: the weights carry the
    DEM shape times the grid spacing, so wide bins are not over-counted.
    """
    grid = np.logspace(np.log10(DEM_LO), np.log10(DEM_HI), DEM_N)
    edges = np.sqrt(grid[:-1] * grid[1:])
    dt = np.diff(np.concatenate([[grid[0] - (edges[0] - grid[0])], edges,
                                 [grid[-1] + (grid[-1] - edges[-1])]]))
    w = np.exp(-0.5 * ((grid - t_mean) / t_sigma) ** 2) * dt
    return grid, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--tmean", type=float, default=3.9, help="Perseus default")
    ap.add_argument("--tsigma", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    with open(args.steps) as fh:
        steps = {int(k): v for k, v in json.load(fh).items() if v}
    grid, w = dem_weights(args.tmean, args.tsigma)
    print(f"Gaussian DEM T_mean={args.tmean} keV, T_sigma={args.tsigma} keV, "
          f"on the fixed {DEM_N}-point grid {DEM_LO}-{DEM_HI} keV")
    print(f"grid points within {args.tol:.1%} of a step carry the weight below."
          f"\n\n{'el':>3} {'pts':>4} {'weight near a step':>19} "
          f"{'max step there':>15} {'upper-bound flux err':>21}")

    rows = []
    for z, v in sorted(steps.items()):
        near = np.zeros(len(grid), dtype=bool)
        biggest = 0.0
        for s in v:
            d = np.where((grid >= s["T_lo"]) & (grid <= s["T_hi"]), 0.0,
                         np.minimum(np.abs(grid - s["T_lo"]),
                                    np.abs(grid - s["T_hi"])) / s["T_lo"])
            hit = d <= args.tol
            if hit.any():
                near |= hit
                biggest = max(biggest, abs(s["step"]))
        wf = float(w[near].sum())
        bound = wf * biggest
        rows.append((bound, z, int(near.sum()), wf, biggest))
        print(f"{SYMBOL.get(z, z):>3} {int(near.sum()):4d} {wf:19.4%} "
              f"{biggest:15.2%} {bound:21.4%}")

    rows.sort(reverse=True)
    print(f"\nworst element: {SYMBOL.get(rows[0][1], rows[0][1])} at "
          f"{rows[0][0]:.3%} upper-bound flux error from this mechanism.")
    print("Upper bound because a grid point NEAR a step is not wrong by the "
          "full step size;\nit bounds how much of the DEM's emission is drawn "
          "from a region where the\ninterpolator and the network can disagree "
          "about a discontinuity.")

    # where the DEM's weight actually is, for context
    k = np.argsort(-w)[:5]
    print(f"\nfor context, the DEM's 5 heaviest grid points: "
          + ", ".join(f"{grid[i]:.2f} keV ({w[i]:.1%})" for i in sorted(k)))


if __name__ == "__main__":
    main()
