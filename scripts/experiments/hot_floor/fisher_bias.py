"""Phase 1: systematic bias (b_sys) + Fisher statistical error -> crossover N*.

For the literature-strategy fit of a realistic Perseus spectrum, quantify, per
published parameter:

  * b_sys  -- the emulator's systematic bias in the infinite-count limit =
              argmin Poisson-deviance of the emulator vs the *noise-free* SPEX
              truth. Shape params (T, abundances, N_H, sigma_v) are independent
              of the overall count scale (log_norm absorbs it), so this is a
              fixed property of {emulator, strategy, T-model}.
  * sigma_stat(N) -- Poisson statistical error from the Fisher matrix at the
              truth; scales as 1/sqrt(N).
  * N*     -- the counts where |b_sys| = sigma_stat(N*). Below N* the emulator
              floor is invisible (noise-dominated); above it, it biases science.

Run (laptop, single-T):
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai \
      python scripts/experiments/hot_floor/fisher_bias.py --mode single

The literature-strategy fit parametrisation (``N_REF``, ``Par``, ``Forward``,
``build_params``) and the campaign config (``PERSEUS``, ``FREE_Z``, ...) live
in ``scripts/inference/campaign.py`` -- this module is the general Fisher/
deviance engine on top of them.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))
from campaign import (                                      # noqa: E402
    FREE_Z, HOT_SCIENCE, HOT_WEAK, N_REF, Par, Forward, build_params,
    injected_abundances, find_xrism_response, band_mask, EXCLUDE_NONE, TruthConfig,
    stream_truth_counts, gaussian_dem, resolve_perseus)
from spexai.config import STORE, RESULTS                      # noqa: E402
from spexai.inference.abundances import SYMBOL                # noqa: E402
from spexai.inference.response import Response               # noqa: E402
from spexai.inference.absorption import Absorption           # noqa: E402
from spexai.inference.operator_model import JointOperatorModel  # noqa: E402


# cond(F) above this is reported as near-singular: the Fisher solve is then
# dominated by round-off in the smallest eigendirection, so per-parameter
# b_sys/sigma_ref are not trustworthy even where their ratio looks sane.
COND_F_WARN = 1e10


def poisson_deviance(mu: np.ndarray, d: np.ndarray) -> float:
    """2 * sum[ mu - d + d ln(d/mu) ], the Poisson deviance (0 at mu=d)."""
    mu = np.clip(mu, 1e-30, None)
    term = mu - d + np.where(d > 0, d * np.log(d / mu), 0.0)
    return 2.0 * float(np.sum(term))


def jacobian(fwd, pars, verbose=True):
    """Central-difference dmu/dtheta at the truth, plus mu0 = emulator@truth.

    Returns (mu0 (n_keep,), J (n, n_keep)). This is the only place the emulator
    is evaluated -- 2*n+1 forwards -- shared by both the Fisher matrix and the
    linearised bias.
    """
    x0 = np.array([p.truth for p in pars])
    t0 = time.time()
    mu0 = np.clip(fwd(x0), 1e-30, None)                 # (n_keep,) at N_REF
    J = np.zeros((len(pars), mu0.size))
    for i, p in enumerate(pars):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += p.step
        xm[i] -= p.step
        J[i] = (fwd(xp) - fwd(xm)) / (2.0 * p.step)     # dmu/dtheta_i
    if verbose:
        print(f"  Jacobian: {2*len(pars)+1} forwards in {time.time()-t0:.1f}s",
              flush=True)
    return mu0, J


def linear_bias_fisher(fwd, pars, d, verbose=True):
    """First-order systematic bias and Fisher statistical error at the truth.

    b_sys = F^{-1} s (one Newton step of the expected-log-likelihood MLE), with
    score s_i = sum_c (d_c/mu0_c - 1) J_ic and Fisher F_ij = sum_c J_ic J_jc/mu0_c
    (d ~ mu0). Valid because the emulator error (d/mu0 - 1) is small. Returns
    (b_sys (n,), sigma_ref (n,), cond_F) at N_REF in-band counts.

    ``cond_F`` is the 2-norm condition number of F and must be checked, not
    ignored. A near-singular F makes b_sys and sigma_ref individually
    meaningless while leaving their *ratio* -- and hence N* -- finite and
    plausible-looking, so a degenerate parameter direction produces a
    publishable-looking number with nothing to flag it. This is not
    hypothetical: with a flat (unit) effective area the n_h direction went
    near-singular, because absorption is constrained almost entirely in the
    soft band, and the published table carried b_sys = -3.8e19 with
    sigma_ref = 5.1e20 as a finite N* = 1.8e7. Anything above ~1e10 is
    approaching float64's limit and should be treated as unconstrained.
    """
    mu0, J = jacobian(fwd, pars, verbose=verbose)
    F = (J / mu0) @ J.T                                 # (n,n)
    cond_F = float(np.linalg.cond(F))
    cov = np.linalg.inv(F)
    sigma_ref = np.sqrt(np.diag(cov))                   # ~ 1/sqrt(N)
    resid = d / mu0 - 1.0                               # emulator error, in-band
    score = J @ resid                                   # (n,)
    b_sys = cov @ score
    if verbose:
        rms = np.sqrt(np.mean(resid ** 2))
        print(f"  in-band emulator residual RMS = {rms:.3e} "
              f"(max |{np.abs(resid).max():.3e}|)", flush=True)
        flag = "  <-- NEAR-SINGULAR, b_sys/sigma_ref unreliable" \
            if cond_F > COND_F_WARN else ""
        print(f"  cond(F) = {cond_F:.3e}{flag}", flush=True)
    return b_sys, sigma_ref, cond_F


def _rebin(a: np.ndarray, g: int) -> np.ndarray:
    """Sum a 1-D array into non-overlapping groups of g channels."""
    n = (a.size // g) * g
    return a[:n].reshape(-1, g).sum(1)


def _diagnose_residual(fwd, pars, d, mu0):
    """Characterise the truth-vs-emulator residual: is the big native-resolution
    RMS a sub-resolution lineshape artifact (averages down with binning) or a
    coherent emissivity error (survives binning)? Reports unweighted and
    counts-weighted RMS + deviance/dof vs channel grouping, and at sigma_v=0.
    """
    x0 = np.array([p.truth for p in pars])
    print("\nRESIDUAL DIAGNOSTIC (truth d vs emulator mu0, N_REF in-band):")
    print(f"{'group':>6} {'eV/bin~':>8} {'unw.RMS':>9} {'cw.RMS':>9} "
          f"{'dev/dof':>9}")
    for g in (1, 2, 4, 8, 16, 32):
        dg, mg = _rebin(d, g), np.clip(_rebin(mu0, g), 1e-30, None)
        r = dg / mg - 1.0
        unw = np.sqrt(np.mean(r ** 2))
        cw = np.sqrt(np.sum(mg * r ** 2) / np.sum(mg))
        dev = poisson_deviance(mg, dg) / (dg.size - len(pars))
        print(f"{g:>6} {0.5*g:>8.1f} {unw:>9.3e} {cw:>9.3e} {dev:>9.2e}",
              flush=True)
    # velocity off: isolates whether velocity broadening drives the mismatch
    iv = fwd.names.index("sigma_v")
    xv = x0.copy(); xv[iv] = 0.0
    muv = np.clip(fwd(xv), 1e-30, None)
    # rescale truth stays same d; compare shape via counts-weighted RMS
    rv = d / muv - 1.0
    cwv = np.sqrt(np.sum(muv * rv ** 2) / np.sum(muv))
    print(f"counts-weighted RMS at sigma_v=0 (native): {cwv:.3e} "
          f"(vs {np.sqrt(np.sum(mu0*(d/mu0-1)**2)/np.sum(mu0)):.3e} at 180)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--diag", action="store_true",
                    help="characterise the residual vs binning, then exit")
    ap.add_argument("--counts", type=float, nargs="+",
                    default=[1e4, 4e4, 2e5, 1e6])
    # physical overrides for the quiet/cool stress case (defaults = Perseus).
    ap.add_argument("--sigma_v", type=float, default=None, help="km/s")
    ap.add_argument("--kT", type=float, default=None, help="single-T keV")
    ap.add_argument("--dem_mean", type=float, default=None)
    ap.add_argument("--dem_sigma", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--out", default=os.path.join(RESULTS, "hot_floor"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    perseus = resolve_perseus({"vel": args.sigma_v, "kT": args.kT,
                               "dem_mean": args.dem_mean,
                               "dem_sigma": args.dem_sigma})
    suffix = (f"_{args.tag}" if args.tag else "")
    print(f"config: kT={perseus['kT']} sigma_v={perseus['vel']} "
          f"dem=({perseus['dem_mean']},{perseus['dem_sigma']}) tag='{args.tag}'",
          flush=True)

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response, exclude=EXCLUDE_NONE)
    emu = JointOperatorModel(models_dir=STORE, device="cpu")
    print(f"emulator elements: {emu.elements}")
    dem, dem_p = (gaussian_dem(mean=perseus["dem_mean"], sigma=perseus["dem_sigma"])
                  if args.mode == "dem" else (None, {}))

    # --- noise-free truth, scaled to N_REF in-band counts ---
    els = emu.elements
    ab = injected_abundances(els)
    cfg = TruthConfig(elements=els, abundances=ab, exposure=1.0,
                      dem=dem, dem_params=dem_p)
    t0 = time.time()
    d_ref = stream_truth_counts(cfg, response, absorption, perseus=perseus)
    assert np.isfinite(d_ref[keep]).all(), (
        "non-finite truth counts -- a DEM grid point is outside the per-element "
        "training temperature range (PCHIP extrapolation blow-up)")
    s = N_REF / d_ref[keep].sum()
    d = d_ref[keep] * s
    log_norm_truth = float(np.log10(cfg.norm_ref * s))
    print(f"truth streamed in {time.time()-t0:.1f}s; in-band counts scaled to "
          f"{d.sum():.3e}; log_norm_truth={log_norm_truth:.3f}")

    fwd = Forward(emu, response, absorption, keep, args.mode, dem=dem,
                 perseus=perseus)
    pars = build_params(fwd, log_norm_truth)
    t0 = time.time()
    mu0 = fwd(np.array([p.truth for p in pars]))
    print(f"one emulator forward: {time.time()-t0:.2f}s over {len(pars)} params",
          flush=True)

    if args.diag:
        _diagnose_residual(fwd, pars, d, mu0)
        return

    b_sys, sigma_ref, cond_F = linear_bias_fisher(fwd, pars, d)

    counts = np.array(args.counts)
    print(f"\n{args.mode.upper()} | b_sys, sigma_stat, and crossover N* "
          f"(in-band counts):")
    print(f"{'param':>9} {'truth':>10} {'b_sys':>11} {'sig@Nref':>10} "
          f"{'N*':>10}  group")
    rows = {}
    for i, p in enumerate(pars):
        z = {v: k for k, v in SYMBOL.items()}.get(p.name)
        grp = ("science" if z in HOT_SCIENCE else "weak" if z in HOT_WEAK
               else "Fe" if p.name == "Fe" else "other")
        nstar = (N_REF * (sigma_ref[i] / abs(b_sys[i])) ** 2
                 if b_sys[i] != 0 else np.inf)
        rows[p.name] = dict(truth=p.truth, b_sys=b_sys[i],
                            sigma_ref=sigma_ref[i], nstar=nstar, group=grp)
        print(f"{p.name:>9} {p.truth:>10.4g} {b_sys[i]:>+11.3e} "
              f"{sigma_ref[i]:>10.3e} {nstar:>10.2e}  {grp}")

    outp = os.path.join(args.out, f"bias_{args.mode}{suffix}.npz")
    np.savez(outp, names=[p.name for p in pars], truth=[p.truth for p in pars],
             b_sys=b_sys, sigma_ref=sigma_ref, counts=counts, n_ref=N_REF,
             kT=perseus["kT"], sigma_v=perseus["vel"], cond_F=cond_F,
             # response provenance: an ARF changes neither n_keep nor the
             # element set but rescales the fit channel by channel, so a
             # result that does not record it cannot be told apart from a
             # flat-effective-area run. See check_truth_response().
             rmf=os.path.basename(rmf), arf=os.path.basename(arf))
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
