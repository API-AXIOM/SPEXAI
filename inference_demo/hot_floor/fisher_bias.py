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
      python inference_demo/hot_floor/fisher_bias.py --mode single
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (                                    # noqa: E402
    PERSEUS, STORE, FREE_Z, HOT_SCIENCE, HOT_WEAK, injected_abundances,
    find_xrism_response, band_mask, TruthConfig, stream_truth_counts,
    gaussian_dem)
from spexai.inference.response import Response               # noqa: E402
from spexai.inference.absorption import Absorption           # noqa: E402
from spexai.inference.operator_model import JointOperatorModel  # noqa: E402
from spexai.inference.abundances import AbundanceModel       # noqa: E402
from spexai.inference.fitting import SIGMA_V_PRIOR           # noqa: E402

SYMBOL = {14: "Si", 16: "S", 18: "Ar", 20: "Ca", 24: "Cr", 25: "Mn",
          26: "Fe", 28: "Ni", 22: "Ti", 27: "Co", 29: "Cu", 30: "Zn"}
N_REF = 1e5                    # reference in-band counts for Fisher/deviance


@dataclass
class Par:
    name: str
    truth: float
    step: float                # finite-difference step
    low: float
    high: float


class Forward:
    """theta (free-parameter vector) -> in-band emulator counts at N_REF scale."""

    def __init__(self, emu, response, absorption, keep, mode, dem=None):
        self.emu, self.resp, self.absn = emu, response, absorption
        self.keep, self.mode, self.dem = keep, mode, dem
        self.logz = float(np.log10(PERSEUS["z"]))
        self.ld = PERSEUS["dist_m"]
        # literature abundance parametrisation: free FREE_Z, tie the rest to Fe.
        ab = AbundanceModel(emu.elements)
        for z in FREE_Z:
            ab.free_element(z, SYMBOL[z])
        others = [z for z in emu.elements if z >= 3 and z not in FREE_Z]
        ab.tie_const(others, 1.0, 26)
        self.abmodel = ab
        self.abnames = ab.param_names
        # parameter order: [abundances...] + thermal + [sigma_v, n_h, log_norm]
        self.thermal = ["kT"] if mode == "single" else ["T_mean", "T_sigma"]
        self.names = self.abnames + self.thermal + ["sigma_v", "n_h", "log_norm"]

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        p = dict(zip(self.names, theta))
        abund = self.abmodel.to_abundances(p)
        norm = 10.0 ** p["log_norm"]
        common = dict(luminosity_distance=self.ld, absorption=self.absn,
                      n_h=p["n_h"] * 1e21)          # n_h sampled in 1e21 cm^-2
        if self.mode == "single":
            mu = self.emu.predict_counts(
                torch.tensor([p["kT"]]), abund, self.logz, norm, p["sigma_v"],
                self.resp, 1.0, **common)
        else:
            w = self.dem.weights({"T_mean": p["T_mean"], "T_sigma": p["T_sigma"]})
            mu = self.emu.predict_counts_dem(
                self.dem.temp_grid, w, abund, self.logz, norm, p["sigma_v"],
                self.resp, 1.0, **common)
        return mu.squeeze(0).cpu().numpy()[self.keep]


def build_params(fwd: Forward, log_norm_truth: float) -> List[Par]:
    """Truth values, steps and bounds for every free parameter."""
    inj = injected_abundances(fwd.emu.elements)
    out: List[Par] = []
    for z in FREE_Z:
        out.append(Par(SYMBOL[z], inj[z], 1e-3, 0.02, 3.0))
    if fwd.mode == "single":
        out.append(Par("kT", PERSEUS["kT"], 5e-3, 1.5, 7.5))
    else:
        out.append(Par("T_mean", PERSEUS["dem_mean"], 5e-3, 1.5, 7.5))
        out.append(Par("T_sigma", PERSEUS["dem_sigma"], 5e-3, 0.15, 3.0))
    out.append(Par("sigma_v", PERSEUS["vel"], 1.0, *SIGMA_V_PRIOR))
    out.append(Par("n_h", PERSEUS["n_h"] / 1e21, 1e-2, 0.0, 5.0))  # 1e21 cm^-2
    out.append(Par("log_norm", log_norm_truth, 2e-3,
                   log_norm_truth - 1.0, log_norm_truth + 1.0))
    return out


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
    (b_sys (n,), sigma_ref (n,)) at N_REF in-band counts.
    """
    mu0, J = jacobian(fwd, pars, verbose=verbose)
    F = (J / mu0) @ J.T                                 # (n,n)
    cov = np.linalg.inv(F)
    sigma_ref = np.sqrt(np.diag(cov))                   # ~ 1/sqrt(N)
    resid = d / mu0 - 1.0                               # emulator error, in-band
    score = J @ resid                                   # (n,)
    b_sys = cov @ score
    if verbose:
        rms = np.sqrt(np.mean(resid ** 2))
        print(f"  in-band emulator residual RMS = {rms:.3e} "
              f"(max |{np.abs(resid).max():.3e}|)", flush=True)
    return b_sys, sigma_ref


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
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # apply physical overrides in-place (experiment.* reads PERSEUS at call time)
    for key, val in (("vel", args.sigma_v), ("kT", args.kT),
                     ("dem_mean", args.dem_mean), ("dem_sigma", args.dem_sigma)):
        if val is not None:
            PERSEUS[key] = val
    suffix = (f"_{args.tag}" if args.tag else "")
    print(f"config: kT={PERSEUS['kT']} sigma_v={PERSEUS['vel']} "
          f"dem=({PERSEUS['dem_mean']},{PERSEUS['dem_sigma']}) tag='{args.tag}'",
          flush=True)

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response)
    emu = JointOperatorModel(models_dir=STORE, device="cpu")
    print(f"emulator elements: {emu.elements}")
    dem, dem_p = (gaussian_dem() if args.mode == "dem" else (None, {}))

    # --- noise-free truth, scaled to N_REF in-band counts ---
    els = emu.elements
    ab = injected_abundances(els)
    cfg = TruthConfig(elements=els, abundances=ab, exposure=1.0,
                      dem=dem, dem_params=dem_p)
    t0 = time.time()
    d_ref = stream_truth_counts(cfg, response, absorption)
    assert np.isfinite(d_ref[keep]).all(), (
        "non-finite truth counts -- a DEM grid point is outside the per-element "
        "training temperature range (PCHIP extrapolation blow-up)")
    s = N_REF / d_ref[keep].sum()
    d = d_ref[keep] * s
    log_norm_truth = float(np.log10(cfg.norm_ref * s))
    print(f"truth streamed in {time.time()-t0:.1f}s; in-band counts scaled to "
          f"{d.sum():.3e}; log_norm_truth={log_norm_truth:.3f}")

    fwd = Forward(emu, response, absorption, keep, args.mode, dem=dem)
    pars = build_params(fwd, log_norm_truth)
    t0 = time.time()
    mu0 = fwd(np.array([p.truth for p in pars]))
    print(f"one emulator forward: {time.time()-t0:.2f}s over {len(pars)} params",
          flush=True)

    if args.diag:
        _diagnose_residual(fwd, pars, d, mu0)
        return

    b_sys, sigma_ref = linear_bias_fisher(fwd, pars, d)

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
             kT=PERSEUS["kT"], sigma_v=PERSEUS["vel"])
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
