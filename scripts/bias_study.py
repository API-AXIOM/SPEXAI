"""Staged bias / calibration study for the operator emulator.

Inject simulated observations from the INDEPENDENT ``SpexTruthModel`` (PCHIP over
SPEX + exact broadening [+ optional absorption]), fit them with the emulator
(``JointOperatorModel``) via emcee, and quantify bias:

  --stage point : median-minus-truth *pulls* and central-interval *coverage*
                  over a grid of realistic parameters. An emulator-self
                  round-trip (``--self-test``) is the control that isolates
                  emulator bias from sampler/statistical scatter.
  --stage sbc   : simulation-based calibration -- draw params from the prior,
                  fit, and record the *rank* of the truth in the posterior; a
                  uniform rank histogram => calibrated.

Realistic and extreme parameter ranges follow the plan's Context table. Runs on
CPU; start small (laptop-safe):

    conda run -n spexai python scripts/bias_study.py --n_sims 4 --nsteps 150
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.simulate import simulate_observation
from spexai.inference.fitting import Param, run_emcee, SIGMA_V_PRIOR

# realistic cluster ranges (temp keV, velocity km/s, log10 norm); abundances
# fixed solar and logz fixed here to keep the smoke small -- extend as needed.
# "realistic" velocity is the literature-grounded Perseus prior (see
# spexai.inference.fitting.SIGMA_V_PRIOR); "extreme" stays deliberately wider so
# SBC also probes dispersions no cluster in the sample actually shows.
REALISTIC = {"temp": (1.0, 8.0), "velocity": SIGMA_V_PRIOR, "log_norm": (9.0, 12.0)}
EXTREME = {"temp": (0.3, 10.0), "velocity": (10.0, 1000.0), "log_norm": (8.0, 13.0)}


def sample_truth(rng, ranges):
    return {k: float(rng.uniform(*v)) for k, v in ranges.items()}


def fit_params(truth, ranges):
    """Fit temp/velocity/log_norm with box priors.

    log_norm is centred on the (target-counts-rescaled) truth +/- 1.5 dex, so
    the prior stays consistent with the actual normalisation and the walker
    cloud is not degenerate (a fixed norm box collapses under target-counts
    rescaling)."""
    return [
        Param("temp", *ranges["temp"], truth=truth["temp"]),
        Param("velocity", *ranges["velocity"], truth=truth["velocity"]),
        Param("log_norm", truth["log_norm"] - 1.5, truth["log_norm"] + 1.5,
              truth=truth["log_norm"]),
    ]


def run_one(truth_model, fit_model, response, ranges, rng, exposure,
            target_counts, nwalkers, nsteps):
    """Simulate one observation from truth_model, fit with fit_model.

    Returns per-parameter dicts of pull = (median-truth)/sigma, covered (truth
    in the 16-84 interval), and SBC rank (fraction of posterior below truth).
    """
    truth = sample_truth(rng, ranges)
    p_sim = {"temp": truth["temp"], "velocity": truth["velocity"],
             "norm": 10.0 ** truth["log_norm"], "logz": -10.0, "abundances": {}}
    obs = simulate_observation(truth_model, response, p_sim, exposure,
                               target_counts=target_counts, rng=int(rng.integers(1 << 30)))
    truth["log_norm"] = float(np.log10(obs.true_params["norm"]))  # after rescale
    params = fit_params(truth, ranges)
    res = run_emcee(obs, fit_model, params, {"abundances": {}, "logz": -10.0},
                    nwalkers=nwalkers, nsteps=nsteps, seed=int(rng.integers(1 << 30)))
    out = {}
    for i, name in enumerate(res.names):
        s = res.samples[:, i]
        q16, q50, q84 = np.percentile(s, [16, 50, 84])
        sigma = 0.5 * (q84 - q16) or np.std(s) or 1e-9
        t = truth[name]
        out[name] = {"truth": t, "median": float(q50),
                     "pull": float((q50 - t) / sigma),
                     "covered": bool(q16 <= t <= q84),
                     "rank": float(np.mean(s < t))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=4)
    ap.add_argument("--stage", choices=["point", "sbc"], default="point")
    ap.add_argument("--elements", nargs="+", default=["26"],
                    help="element Zs, or 'all' for every element in the manifest")
    ap.add_argument("--extreme", action="store_true", help="use extreme ranges")
    ap.add_argument("--self-test", action="store_true",
                    help="inject with the emulator (control) instead of SPEX truth")
    ap.add_argument("--exposure", type=float, default=1e5)
    ap.add_argument("--target-counts", type=float, default=5e4)
    ap.add_argument("--nwalkers", type=int, default=16)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rmf", default=os.path.expanduser(
        "~/work/data/spexai/responses/aciss_aimpt_cy28.rmf"))
    ap.add_argument("--arf", default=os.path.expanduser(
        "~/work/data/spexai/responses/aciss_aimpt_cy28.arf"))
    ap.add_argument("--out", default="bias_study.json")
    args = ap.parse_args()

    if args.stage == "sbc":
        raise SystemExit(
            "--stage sbc is superseded by scripts/sbc_campaign.py.\n"
            "This path is emcee-only on CPU, and its ranks were taken over the\n"
            "raw correlated chain, which fails the uniformity test even for a\n"
            "calibrated sampler. --stage point (emulator-vs-SPEX bias) is\n"
            "unaffected and still lives here.")

    elements = None if "all" in args.elements else [int(z) for z in args.elements]
    from spexai.inference.response import Response
    response = Response(args.rmf, args.arf if os.path.exists(args.arf) else None)
    fit_model = JointOperatorModel(device="cpu", elements=elements)
    truth_model = fit_model if args.self_test else SpexTruthModel(
        device="cpu", elements=elements)
    ranges = EXTREME if args.extreme else REALISTIC
    rng = np.random.default_rng(args.seed)

    results = [run_one(truth_model, fit_model, response, ranges, rng,
                       args.exposure, args.target_counts, args.nwalkers,
                       args.nsteps) for _ in range(args.n_sims)]

    names = list(results[0])
    summary = {}
    for name in names:
        pulls = np.array([r[name]["pull"] for r in results])
        cov = np.mean([r[name]["covered"] for r in results])
        ranks = np.array([r[name]["rank"] for r in results])
        summary[name] = {"pull_mean": float(pulls.mean()),
                         "pull_std": float(pulls.std()),
                         "coverage_68": float(cov),
                         "rank_mean": float(ranks.mean())}
        print(f"{name:10s} pull={pulls.mean():+.2f}+-{pulls.std():.2f}  "
              f"cov68={cov:.2f}  rank_mean={ranks.mean():.2f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary,
                   "results": results}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
