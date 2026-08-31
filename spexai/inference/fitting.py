"""Bayesian inference on (simulated) spectra with the operator emulator.

One Poisson likelihood, run two ways: MCMC (`emcee`) and nested sampling
(`ultranest`). Both fit the same parameter set so their posteriors are directly
comparable.

Parameters are sampled in natural units except the normalisation, which is
sampled as ``log_norm`` = log10(norm). `fixed` carries anything not fit
(abundances, logz, and velocity if it is not a free parameter).

**Two implementations of that likelihood, deliberately.**

``make_loglike`` is the *scalar reference*: one parameter set per call, straight
through ``JointOperatorModel.predict_counts``. Simple enough to read and check
by eye.

:class:`~spexai.inference.posterior.PoissonPosterior` over
:class:`~spexai.inference.vector_forward.VectorForward` is the *production*
path: it evaluates the whole walker ensemble in one batched forward, which is
where the element-batched trunk's speedup actually lands. ``vectorized=True``
(the default) selects it.

They must agree, and ``tests/test_fitting.py`` asserts that they do. Keeping the
scalar version is not redundancy for its own sake -- it is the independent
reference that catches the kind of silent, walker-axis-alignment bug the batched
forward has already produced once (a per-walker ``n_h`` broadcast against the
element axis). Since this machinery exists to *establish* calibration, a
quietly wrong likelihood would invalidate the very thing it measures.

The vectorised path falls back to the scalar one automatically, with a reason,
whenever it does not apply -- see :func:`build_posterior`.
"""
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from spexai.inference.units import D_REF_M


# Line-of-sight velocity dispersion prior (km/s), literature-grounded on Perseus
# and then widened for robustness. Measured values:
#   * Hitomi 2016 (Nature 535, 117): sigma_v = 164 +/- 10 km/s, 30-60 kpc from
#     the nucleus -- the canonical "quiescent core" number.
#   * Hitomi 2018 (PASJ 70, 9): ~100 km/s over most of the mapped region, rising
#     to ~200 km/s toward the central AGN and the NW ghost bubble; a ~100 km/s
#     line-of-sight velocity gradient across the core from sloshing.
#   * XRISM 2025 (arXiv:2510.12782): ~300 km/s in the eastern region and a
#     dipole of +/-200-300 km/s from a recent merger, out to ~500 kpc.
# So Perseus itself spans ~100-300 km/s. The prior is widened either side: the
# floor sits well below the quiescent value (and below XRISM Resolve's own
# resolution, ~100 km/s equivalent at Fe-K, where the likelihood goes flat), and
# the ceiling covers merger-driven dispersions above anything measured here.
# Kept strictly positive: at sigma_v -> 0 the line profile collapses below the
# response width and the parameter stops being identifiable.
SIGMA_V_PRIOR = (30.0, 600.0)


@dataclass
class Param:
    name: str
    low: float
    high: float
    label: str = ""
    truth: float = None


def make_loglike(obs, model, param_names, fixed, abundance_model=None, dem=None,
                 absorption=None):
    """Poisson log-likelihood (constant dropped) for one Observation.

    Single-temperature by default. Optional hooks:

    * ``abundance_model``: an :class:`~spexai.inference.abundances.AbundanceModel`
      whose ``to_abundances(p)`` maps the sampled parameters to ``{Z: value}``
      (merged over any ``fixed["abundances"]``). Without it, abundances are
      whatever ``fixed`` carries.
    * ``dem``: a temperature-distribution model (see
      :mod:`spexai.inference.tempdist`) exposing ``temp_grid`` and
      ``weights(p)``; when given, the likelihood uses ``predict_counts_dem``
      instead of a single ``temp``.

    ``logz`` and ``velocity`` are read from the sampled parameters when present,
    otherwise from ``fixed``.
    """
    counts = np.asarray(obs.counts, dtype=np.float64)
    resp, expo = obs.response, obs.exposure
    fixed_abund = fixed.get("abundances", {})
    logz_fix = float(fixed.get("logz", -10.0))
    vfix = float(fixed.get("velocity", 0.0))
    nh_fix = float(fixed.get("n_h", 0.0))
    ld_fix = float(fixed.get("luminosity_distance", D_REF_M))   # metres; fixed

    def loglike(theta):
        p = dict(zip(param_names, theta))
        vel = float(p.get("velocity", vfix))
        logz = float(p.get("logz", logz_fix))
        n_h = float(p.get("n_h", nh_fix))
        ld = float(p.get("luminosity_distance", ld_fix))
        norm = 10.0 ** float(p["log_norm"])   # Y = emission measure (1e64 m^-3)
        abund = ({**fixed_abund, **abundance_model.to_abundances(p)}
                 if abundance_model is not None else fixed_abund)
        if dem is not None:
            mu = model.predict_counts_dem(
                dem.temp_grid, dem.weights(p), abund, logz, norm, vel, resp,
                expo, luminosity_distance=ld, absorption=absorption, n_h=n_h)
        else:
            mu = model.predict_counts(
                torch.tensor([float(p["temp"])]), abund, logz, norm, vel,
                resp, expo, luminosity_distance=ld, absorption=absorption, n_h=n_h)
        mu = mu.squeeze(0).cpu().numpy().astype(np.float64)
        mu = np.clip(mu, 1e-30, None)
        return float(np.sum(counts * np.log(mu) - mu))
    return loglike


# --- vectorised posterior ---------------------------------------------------

# Reasons the batched forward cannot serve a given fit. Each one is a genuine
# structural limitation, not a missing feature flag.
_NO_VECTOR = {
    "dem": "the DEM shape has no weights_batch() -- it wraps a scipy "
           "distribution with no torch equivalent, so it cannot be evaluated "
           "per walker or differentiated",
    "logz": "redshift is a sampled parameter (the batched forward bakes the "
            "rest-frame energy grid in at construction)",
    "luminosity_distance": "the distance is a sampled parameter (it is "
                           "folded into a constant scale factor)",
}


def vectorization_blocker(param_names, dem):
    """Why the vectorised posterior cannot be used here, or ``None``.

    DEMs are supported as long as the shape provides the batched
    ``weights_batch`` contract; the closed-form ones here all do."""
    if dem is not None and not hasattr(dem, "weights_batch"):
        return _NO_VECTOR["dem"]
    for name in ("logz", "luminosity_distance"):
        if name in param_names:
            return _NO_VECTOR[name]
    return None


def build_posterior(obs, model, params, fixed, abundance_model=None,
                    absorption=None, keep=None, dem=None, **forward_kwargs):
    """Vectorised :class:`PoissonPosterior` for one observation.

    Mirrors :func:`make_loglike`'s semantics exactly -- same parameter roles,
    same ``fixed`` fallbacks, same physical scaling -- but evaluates a whole
    ensemble per call. Returns ``None`` if the fit needs something the batched
    forward cannot express (see :func:`vectorization_blocker`).
    """
    from spexai.inference.posterior import BoxPrior, PoissonPosterior
    from spexai.inference.vector_forward import VectorForward
    from spexai.inference.abundances import AbundanceModel

    names = [p.name for p in params]
    counts = np.asarray(obs.counts, dtype=np.float64)
    keep = np.ones(len(counts), dtype=bool) if keep is None else np.asarray(keep)
    # velocity: sampled if it is a parameter, otherwise pinned at its fixed value
    velocity = None if "velocity" in names else float(fixed.get("velocity", 0.0))
    ab = abundance_model if abundance_model is not None else AbundanceModel([])
    forward = VectorForward(
        model, obs.response, keep, names, ab, absorption=absorption,
        redshift=10.0 ** float(fixed.get("logz", -10.0)),
        luminosity_distance=float(fixed.get("luminosity_distance", D_REF_M)),
        velocity=velocity, fixed=fixed, n_h_scale=1.0,
        device=model.device, exposure=float(obs.exposure),
        temp_name="temp", norm_name="log_norm", velocity_name="velocity",
        nh_name="n_h", dem=dem, **forward_kwargs)
    # abundances not managed by the abundance model still have to be applied;
    # they are constants, so they ride along as the forward's fixed set
    forward.fixed.setdefault("abundances", fixed.get("abundances", {}))
    prior = BoxPrior.from_params(params, device=model.device)
    return PoissonPosterior(forward, counts[keep], prior)


def _resolve_posterior(obs, model, params, fixed, abundance_model, dem,
                       absorption, vectorized):
    """The posterior to sample, or ``None`` to use the scalar path."""
    if not vectorized:
        return None
    blocker = vectorization_blocker([p.name for p in params], dem)
    if blocker is not None:
        warnings.warn(f"falling back to the scalar likelihood: {blocker}",
                      RuntimeWarning, stacklevel=3)
        return None
    return build_posterior(obs, model, params, fixed, abundance_model,
                           absorption, dem=dem)


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
              discard_frac=0.4, seed=0, progress=False,
              abundance_model=None, dem=None, absorption=None,
              vectorized=True):
    """Ensemble MCMC. ``vectorized`` batches the whole ensemble into one
    forward per step; set it False to force the scalar reference likelihood."""
    import emcee
    names = [p.name for p in params]
    labels = [p.label or p.name for p in params]
    ndim = len(params)
    lo = np.array([p.low for p in params])
    hi = np.array([p.high for p in params])
    truths = np.array([p.truth if p.truth is not None else np.nan for p in params])
    post = _resolve_posterior(obs, model, params, fixed, abundance_model, dem,
                              absorption, vectorized)
    if post is None:
        loglike = make_loglike(obs, model, names, fixed, abundance_model, dem,
                               absorption)

        def logprob(theta):
            if np.any(theta < lo) or np.any(theta > hi):
                return -np.inf
            return loglike(theta)
    else:
        logprob = post.logp

    rng = np.random.default_rng(seed)
    center = np.where(np.isfinite(truths), truths, 0.5 * (lo + hi))
    p0 = center + 0.02 * (hi - lo) * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob,
                                    vectorize=post is not None)
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
                  frac_remain=0.01, seed=0, logdir=None,
                  abundance_model=None, dem=None, absorption=None,
                  vectorized=True):
    """Nested sampling. ``vectorized`` evaluates UltraNest's whole live-point
    block in one batched forward -- this used to loop the likelihood one row at
    a time, which threw the batching away entirely."""
    import ultranest
    names = [p.name for p in params]
    labels = [p.label or p.name for p in params]
    lo = np.array([p.low for p in params])
    span = np.array([p.high - p.low for p in params])
    truths = np.array([p.truth if p.truth is not None else np.nan for p in params])
    post = _resolve_posterior(obs, model, params, fixed, abundance_model, dem,
                              absorption, vectorized)
    if post is None:
        loglike1 = make_loglike(obs, model, names, fixed, abundance_model, dem,
                               absorption)

        def loglike(thetas):                      # scalar fallback (loops)
            thetas = np.atleast_2d(thetas)
            return np.array([loglike1(t) for t in thetas])
    else:
        # ptform guarantees points inside the box, so the bounds check in
        # `logp` would only waste work here
        loglike = post.loglike

    def ptform(cubes):
        return lo + np.atleast_2d(cubes) * span

    t0 = time.time()
    sampler = ultranest.ReactiveNestedSampler(
        names, loglike, ptform, vectorized=True,
        log_dir=logdir, resume="overwrite")   # valid even when log_dir is None
    res = sampler.run(min_num_live_points=min_num_live_points,
                      frac_remain=frac_remain, show_status=progress_flag())
    runtime = time.time() - t0
    return UltranestResult(names, labels, np.asarray(res["samples"]),
                           float(res["logz"]), float(res["logzerr"]),
                           float(res.get("ess", len(res["samples"]))),
                           truths, res, runtime, int(res.get("ncall", 0)))


def progress_flag():
    return False
