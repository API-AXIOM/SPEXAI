"""Turn bake-off results into an SBC campaign budget.

The bake-off measures, per sampler, how much wall-clock and how many forwards
buy how much effective sample size. SBC needs far less than a publication-grade
posterior per simulation -- only ``L`` (~100) near-independent draws, because
the rank of the truth among ``L`` draws is all a rank histogram consumes. This
script does the one division that turns those two facts into a decision:

    time per SBC sim  =  burn-in  +  sampling time * L / minESS

The burn-in term is why this is not a one-liner. A chain cannot be shortened
below the length it takes to reach the typical set, so the campaign cost floors
at ``n_sims * burn_in`` no matter how generous the ESS is. For emcee/zeus that
floor is ``discard_frac`` of the measured runtime; for NUTS it is the warmup;
nested sampling and VI have no separable burn-in and are reported as-is.

    python scripts/inference/sbc_cost_model.py --n_sims 100 --n_draws 100 --gpus 1

Read the output as a *ranking*, not a promise: it extrapolates one measured run
linearly in chain length, which is right for the sampling phase and optimistic
if the sampler was still equilibrating when it was measured.
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from spexai.config import RESULTS                              # noqa: E402

SAMPLERS = ("emcee", "zeus", "ultranest", "nuts", "svi")

# fraction of a measured run that is burn-in and therefore does NOT shrink when
# the campaign asks for fewer draws (matches the sampler defaults)
BURN_IN_FRAC = {"emcee": 0.4, "zeus": 0.4, "nuts": 0.5,
                "ultranest": 0.0, "svi": 0.0}
# samplers whose cost is not meaningfully reducible by asking for fewer draws:
# nested sampling runs to an evidence tolerance, VI to convergence
IRREDUCIBLE = {"ultranest", "svi"}


def load(results_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "bakeoff_*.npz"))):
        name = os.path.basename(path)[len("bakeoff_"):-len(".npz")]
        if name not in SAMPLERS:
            continue
        z = np.load(path, allow_pickle=True)
        ess = np.asarray(z["ess"], dtype=float)
        out[name] = {
            "runtime_s": float(z["runtime_s"]),
            "n_eval": int(z["n_eval"]),
            "min_ess": float(np.nanmin(ess)) if ess.size else float("nan"),
        }
    return out


def project(rec, name, n_draws, n_sims, gpus):
    """One sampler's projected campaign cost."""
    runtime, min_ess = rec["runtime_s"], rec["min_ess"]
    if not np.isfinite(min_ess) or min_ess <= 0:
        return None
    if name in IRREDUCIBLE:
        per_sim = runtime
        shrink = 1.0
    else:
        burn = BURN_IN_FRAC.get(name, 0.0) * runtime
        sample = runtime - burn
        # asking for n_draws instead of min_ess scales only the sampling phase,
        # and never lengthens it beyond what was measured
        shrink = min(1.0, n_draws / min_ess)
        per_sim = burn + sample * shrink
    total_h = per_sim * n_sims / 3600.0
    return {"per_sim_h": per_sim / 3600.0, "total_gpu_h": total_h,
            "wall_days": total_h / max(1, gpus) / 24.0, "shrink": shrink,
            "min_ess": min_ess, "measured_h": runtime / 3600.0,
            "evals_per_sim": rec["n_eval"] * shrink}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(RESULTS, "bakeoff"),
                    help="directory holding bakeoff_<sampler>.npz")
    ap.add_argument("--n_sims", type=int, default=100)
    ap.add_argument("--n_draws", type=int, default=100,
                    help="independent draws per sim (the SBC L)")
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    recs = load(args.results)
    if not recs:
        raise SystemExit(f"no bakeoff_*.npz in {args.results} -- run the "
                         f"bake-off first, or point --results elsewhere")

    print(f"SBC campaign: {args.n_sims} sims x {args.n_draws} draws, "
          f"{args.gpus} GPU(s)\n")
    print(f"{'sampler':>10} {'measured':>9} {'minESS':>8} {'shrink':>7} "
          f"{'h/sim':>7} {'GPU-h':>9} {'wall days':>10}")
    rows = []
    for name in SAMPLERS:
        if name not in recs:
            continue
        p = project(recs[name], name, args.n_draws, args.n_sims, args.gpus)
        if p is None:
            print(f"{name:>10} {'--':>9} {'no ESS':>8}   (cannot project)")
            continue
        rows.append((p["total_gpu_h"], name, p))
        print(f"{name:>10} {p['measured_h']:>8.2f}h {p['min_ess']:>8.0f} "
              f"{p['shrink']:>7.2f} {p['per_sim_h']:>7.2f} "
              f"{p['total_gpu_h']:>9.0f} {p['wall_days']:>10.1f}")

    missing = [s for s in SAMPLERS if s not in recs]
    if missing:
        print(f"\nnot yet run: {', '.join(missing)}")
    if rows:
        rows.sort()
        best_h, best, p = rows[0]
        print(f"\ncheapest: {best} at {best_h:.0f} GPU-h "
              f"({p['wall_days']:.1f} days on {args.gpus} GPU(s))")
        if p["shrink"] >= 1.0:
            print("  NOTE: its measured minESS is already at or below the "
                  "requested draws, so there is no chain-shortening left -- "
                  "the only levers are fewer sims or more GPUs.")
        else:
            print(f"  chain shortens to {p['shrink']:.0%} of the measured run; "
                  f"the burn-in floor is what remains.")


if __name__ == "__main__":
    main()
