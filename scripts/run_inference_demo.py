"""Run emcee + UltraNest inference on a simulated spectrum and produce all
diagnostic / corner / posterior-predictive plots.

Defaults use a reduced element set (O, Si, Fe) + Chandra ACIS so it is
tractable on a laptop CPU; the same code runs the full 16-element set on a
GPU by passing --elements 1 2 ... 26 (and a bigger --nwalkers/--nlive).
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.inference.fit_plots import (plot_corner_overlay, plot_emcee_trace,
                                        plot_posterior_predictive,
                                        plot_ultranest_diagnostics)
from spexai.inference.fitting import Param, run_emcee, run_ultranest
from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.response import Response
from spexai.inference.simulate import simulate_observation

RESP = os.path.expanduser("~/work/data/spexai/responses")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", nargs="+", type=int, default=[8, 14, 26])
    ap.add_argument("--outdir", default=os.path.join(REPO, "inference_demo"))
    ap.add_argument("--nwalkers", type=int, default=16)
    ap.add_argument("--nsteps", type=int, default=250)
    ap.add_argument("--nlive", type=int, default=150)
    ap.add_argument("--exposure", type=float, default=1e5)
    ap.add_argument("--target_counts", type=float, default=2e4)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--velocity", type=float, default=200.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t_all = time.time()

    resp = Response(f"{RESP}/aciss_aimpt_cy28.rmf", f"{RESP}/aciss_aimpt_cy28.arf")
    model = JointOperatorModel(device="cpu", elements=args.elements)
    print("model:", model, flush=True)

    truth = {"temp": args.temp, "norm": 1e10, "velocity": args.velocity,
             "logz": -10.0}
    obs = simulate_observation(model, resp, truth, exposure=args.exposure,
                               target_counts=args.target_counts,
                               instrument="Chandra ACIS-S", rng=0)
    true_lognorm = float(np.log10(obs.true_params["norm"]))
    print(f"simulated {obs.total_counts:,} counts; true log_norm="
          f"{true_lognorm:.3f}", flush=True)

    params = [Param("temp", 0.5, 6.0, "T [keV]", args.temp),
              Param("log_norm", true_lognorm - 1.5, true_lognorm + 1.5,
                    r"$\log_{10}$ norm", true_lognorm),
              Param("velocity", 0.0, 600.0, "v [km/s]", args.velocity)]
    fixed = {"abundances": {}, "logz": -10.0}

    print("running emcee ...", flush=True)
    er = run_emcee(obs, model, params, fixed, nwalkers=args.nwalkers,
                   nsteps=args.nsteps)
    print(f"  emcee: {er.runtime_s:.0f}s, {er.n_eval:,} evals, tau={er.tau}",
          flush=True)

    print("running ultranest ...", flush=True)
    er_un_dir = os.path.join(args.outdir, "un")
    ur = run_ultranest(obs, model, params, fixed,
                       min_num_live_points=args.nlive, logdir=er_un_dir)
    print(f"  ultranest: {ur.runtime_s:.0f}s, {ur.n_eval:,} calls, "
          f"logZ={ur.logz:.2f}+-{ur.logzerr:.2f}, ESS={ur.ess:.0f}", flush=True)

    with open(os.path.join(args.outdir, "results.pkl"), "wb") as f:
        pickle.dump(dict(obs_counts=obs.counts, expected=obs.expected,
                         true=obs.true_params, er=er, ur=ur,
                         elements=args.elements), f)

    plot_emcee_trace(er, os.path.join(args.outdir, "emcee_trace.png"))
    plot_ultranest_diagnostics(ur, os.path.join(args.outdir, "ultranest_diag.png"))
    plot_corner_overlay(er, ur, os.path.join(args.outdir, "corner.png"))
    plot_posterior_predictive(obs, model, er, ur, fixed,
                              os.path.join(args.outdir,
                                           "posterior_predictive.png"))

    print("\nrecovery (truth | emcee median +/-1sigma | ultranest median +/-1sigma):",
          flush=True)
    for i, p in enumerate(params):
        em = np.percentile(er.samples[:, i], [16, 50, 84])
        un = np.percentile(ur.samples[:, i], [16, 50, 84])
        print(f"  {p.name:10} {p.truth:>8.4g} | "
              f"{em[1]:.4g} (+{em[2]-em[1]:.2g}/-{em[1]-em[0]:.2g}) | "
              f"{un[1]:.4g} (+{un[2]-un[1]:.2g}/-{un[1]-un[0]:.2g})", flush=True)
    print(f"\nwrote plots to {args.outdir}  (total {time.time()-t_all:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
