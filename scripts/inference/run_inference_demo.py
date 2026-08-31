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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spexai.inference.fit_plots import (plot_corner_overlay, plot_emcee_trace,
                                        plot_posterior_predictive,
                                        plot_ultranest_diagnostics)
from spexai.inference.abundances import AbundanceModel
from spexai.inference.fitting import (Param, run_emcee, run_ultranest,
                                      SIGMA_V_PRIOR)
from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.response import Response
from spexai.inference.simulate import simulate_observation
from spexai.config import RESP_DIR, RESULTS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _AbundWithH:
    """Wrap an AbundanceModel to also sample H (Z=1), which AbundanceModel
    excludes by design (it manages only metals). Exposes the same
    ``param_names`` / ``to_abundances`` interface the fitter expects."""

    def __init__(self, inner, h_name="H"):
        self.inner = inner
        self.h_name = h_name

    @property
    def param_names(self):
        return list(self.inner.param_names) + [self.h_name]

    def to_abundances(self, params):
        ab = dict(self.inner.to_abundances(params))
        ab[1] = float(params[self.h_name])   # H, solar-relative
        return ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", nargs="+", type=int, default=[8, 14, 26])
    ap.add_argument("--outdir", default=os.path.join(RESULTS, "inference_demo"))
    ap.add_argument("--nwalkers", type=int, default=16)
    ap.add_argument("--nsteps", type=int, default=250)
    ap.add_argument("--nlive", type=int, default=150)
    ap.add_argument("--exposure", type=float, default=1e5)
    ap.add_argument("--target_counts", type=float, default=2e4)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--velocity", type=float, default=200.0)
    ap.add_argument("--sampler", choices=["both", "emcee", "ultranest"],
                    default="both", help="skip the slow nested-sampling pass "
                    "with 'emcee' for a quick laptop recovery check")
    ap.add_argument("--metallicity", action="store_true",
                    help="free one global metal parameter (all metals Z>=3 tied "
                    "to it, i.e. solar-ratio to Fe) instead of fixing at solar")
    ap.add_argument("--free_h", action="store_true",
                    help="also sample the H abundance separately")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t_all = time.time()

    resp = Response(f"{RESP_DIR}/aciss_aimpt_cy28.rmf", f"{RESP_DIR}/aciss_aimpt_cy28.arf")
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
              Param("velocity", *SIGMA_V_PRIOR, "v [km/s]", args.velocity)]
    fixed = {"abundances": {}, "logz": -10.0}

    # Optional free abundances. `--metallicity` frees one global metal parameter
    # (every metal tied to it => solar-ratio to Fe); `--free_h` adds H on top.
    # Truth is solar (=1.0), since the spectrum was simulated at solar.
    abundance_model = None
    if args.metallicity or args.free_h:
        am = AbundanceModel(args.elements)
        if args.metallicity:
            am = am.global_metallicity("Z_Fe")           # all metals share Z_Fe
            params.append(Param("Z_Fe", 0.1, 5.0,
                                r"$Z$ (all metals $=$ Fe)", 1.0))
        abundance_model = _AbundWithH(am) if args.free_h else am
        if args.free_h:
            params.append(Param("H", 0.1, 5.0, "H", 1.0))

    er = ur = None
    if args.sampler in ("both", "emcee"):
        print("running emcee ...", flush=True)
        er = run_emcee(obs, model, params, fixed, nwalkers=args.nwalkers,
                       nsteps=args.nsteps, abundance_model=abundance_model)
        print(f"  emcee: {er.runtime_s:.0f}s, {er.n_eval:,} evals, tau={er.tau}",
              flush=True)

    if args.sampler in ("both", "ultranest"):
        print("running ultranest ...", flush=True)
        er_un_dir = os.path.join(args.outdir, "un")
        ur = run_ultranest(obs, model, params, fixed,
                           min_num_live_points=args.nlive, logdir=er_un_dir,
                           abundance_model=abundance_model)
        print(f"  ultranest: {ur.runtime_s:.0f}s, {ur.n_eval:,} calls, "
              f"logZ={ur.logz:.2f}+-{ur.logzerr:.2f}, ESS={ur.ess:.0f}", flush=True)

    with open(os.path.join(args.outdir, "results.pkl"), "wb") as f:
        pickle.dump(dict(obs_counts=obs.counts, expected=obs.expected,
                         true=obs.true_params, er=er, ur=ur,
                         elements=args.elements), f)

    if er is not None:
        plot_emcee_trace(er, os.path.join(args.outdir, "emcee_trace.png"))
    if ur is not None:
        plot_ultranest_diagnostics(ur,
                                   os.path.join(args.outdir, "ultranest_diag.png"))
    if er is not None and ur is not None:
        plot_corner_overlay(er, ur, os.path.join(args.outdir, "corner.png"))
    plot_posterior_predictive(obs, model, er, ur, fixed,
                              os.path.join(args.outdir,
                                           "posterior_predictive.png"),
                              abundance_model=abundance_model)

    print("\nrecovery (truth | emcee median +/-1sigma | ultranest median +/-1sigma):",
          flush=True)
    for i, p in enumerate(params):
        cols = [f"  {p.name:10} {p.truth:>8.4g}"]
        for res in (er, ur):
            if res is None:
                cols.append("        -")
                continue
            q = np.percentile(res.samples[:, i], [16, 50, 84])
            cols.append(f"{q[1]:.4g} (+{q[2]-q[1]:.2g}/-{q[1]-q[0]:.2g})")
        print(" | ".join(cols), flush=True)
    print(f"\nwrote plots to {args.outdir}  (total {time.time()-t_all:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
