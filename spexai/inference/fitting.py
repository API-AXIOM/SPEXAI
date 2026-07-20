"""Bayesian inference on (simulated) spectra with the operator emulator.

One shared Poisson likelihood over `JointOperatorModel.predict_counts`, run
two ways: MCMC (`emcee`) and nested sampling (`ultranest`). Both fit the same
parameter set so their posteriors are directly comparable.

Parameters are sampled in natural units except the normalisation, which is
sampled as ``log_norm`` = log10(norm). `fixed` carries anything not fit
(abundances, logz, and velocity if it is not a free parameter).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Param:
    name: str
    low: float
    high: float
    label: str = ""
    truth: float = None


def make_loglike(obs, model, param_names, fixed):
    """Poisson log-likelihood (constant dropped) for one Observation."""
    counts = np.asarray(obs.counts, dtype=np.float64)
    resp, expo = obs.response, obs.exposure
    abund = fixed.get("abundances", {})
    logz = float(fixed.get("logz", -10.0))
    vfix = float(fixed.get("velocity", 0.0))

    def loglike(theta):
        p = dict(zip(param_names, theta))
        vel = p.get("velocity", vfix)
        mu = model.predict_counts(
            torch.tensor([float(p["temp"])]), abund, logz,
            10.0 ** float(p["log_norm"]), float(vel), resp, expo
        ).squeeze(0).cpu().numpy().astype(np.float64)
        mu = np.clip(mu, 1e-30, None)
        return float(np.sum(counts * np.log(mu) - mu))
    return loglike


# --- emcee -----------------------------------------------------------------

@dataclass
class EmceeResult:
    names: list
    labels: list
    chain: np.ndarray        # (nsteps, nwalkers, ndim)
    samples: np.ndarray      # (N, ndim) post-burn-in, flattened
    log_prob: np.ndarray     # (nsteps, nwalkers)
    tau: np.ndarray          # autocorrelation time per parameter
    discard: int
    truths: np.ndarray
    runtime_s: float
    n_eval: int

    @property
    def median(self):
        return np.median(self.samples, axis=0)


def run_emcee(obs, model, params, fixed, nwalkers=16, nsteps=400,
              discard_frac=0.4, seed=0, progress=False):
    import emcee
    names = [p.name for p in params]
    labels = [p.label or p.name for p in params]
    ndim = len(params)
    lo = np.array([p.low for p in params])
    hi = np.array([p.high for p in params])
    truths = np.array([p.truth if p.truth is not None else np.nan for p in params])
    loglike = make_loglike(obs, model, names, fixed)

    def logprob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return loglike(theta)

    rng = np.random.default_rng(seed)
    center = np.where(np.isfinite(truths), truths, 0.5 * (lo + hi))
    p0 = center + 0.02 * (hi - lo) * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
    sampler.run_mcmc(p0, nsteps, progress=progress)
    runtime = time.time() - t0

    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(ndim, np.nan)
    discard = int(discard_frac * nsteps)
    return EmceeResult(names, labels, sampler.get_chain(),
                       sampler.get_chain(discard=discard, flat=True),
                       sampler.get_log_prob(), tau, discard, truths,
                       runtime, nwalkers * nsteps)


# --- ultranest -------------------------------------------------------------

@dataclass
class UltranestResult:
    names: list
    labels: list
    samples: np.ndarray      # equal-weighted posterior (N, ndim)
    logz: float
    logzerr: float
    ess: float
    truths: np.ndarray
    result: dict = field(repr=False, default=None)
    runtime_s: float = 0.0
    n_eval: int = 0

    @property
    def median(self):
        return np.median(self.samples, axis=0)


def run_ultranest(obs, model, params, fixed, min_num_live_points=200,
                  frac_remain=0.01, seed=0, logdir=None):
    import ultranest
    names = [p.name for p in params]
    labels = [p.label or p.name for p in params]
    lo = np.array([p.low for p in params])
    span = np.array([p.high - p.low for p in params])
    truths = np.array([p.truth if p.truth is not None else np.nan for p in params])
    loglike1 = make_loglike(obs, model, names, fixed)

    def loglike(thetas):                          # vectorized wrapper (loops)
        thetas = np.atleast_2d(thetas)
        return np.array([loglike1(t) for t in thetas])

    def ptform(cubes):
        return lo + np.atleast_2d(cubes) * span

    t0 = time.time()
    sampler = ultranest.ReactiveNestedSampler(
        names, loglike, ptform, vectorized=True,
        log_dir=logdir, resume="overwrite" if logdir else None)
    res = sampler.run(min_num_live_points=min_num_live_points,
                      frac_remain=frac_remain, show_status=progress_flag())
    runtime = time.time() - t0
    return UltranestResult(names, labels, np.asarray(res["samples"]),
                           float(res["logz"]), float(res["logzerr"]),
                           float(res.get("ess", len(res["samples"]))),
                           truths, res, runtime, int(res.get("ncall", 0)))


def progress_flag():
    return False
