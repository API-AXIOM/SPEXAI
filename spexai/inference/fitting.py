"""The scalar reference likelihood, and the parameter dataclass.

**This module no longer runs fits.** Its ``run_emcee``/``run_ultranest`` front
ends and their ``EmceeResult``/``UltranestResult`` types were a second, weaker
copy of what :mod:`spexai.inference.samplers` already did -- two samplers
instead of nine, no ESS, no checkpointing -- and were removed on 2026-09-22.
Fits are assembled with :class:`~spexai.inference.spectral_fit.SpectralFit` and
run with ``.sample(<name>)``. What remains here is the part that was never
duplicated:

``make_loglike`` is the **scalar reference likelihood**: one parameter set per
call, straight through ``JointOperatorModel.predict_counts``, simple enough to
read and check by eye. The production path -- a
:class:`~spexai.inference.posterior.PoissonPosterior` over a
:class:`~spexai.inference.vector_forward.VectorForward` -- evaluates a whole
walker ensemble in one batched forward instead.

The two must agree, and ``tests/test_fitting.py`` asserts that they do at every
construction. Keeping the scalar version is not redundancy for its own sake: it
is the independent reference that catches silent walker-axis-alignment bugs, of
which the batched forward has already produced one (a per-walker ``n_h``
broadcast against the element axis). Since this machinery exists to *establish*
calibration, a quietly wrong likelihood would invalidate the very thing it
measures.

``Param`` and ``SIGMA_V_PRIOR`` stay because the simulation studies specify
parameters as boxes with a truth attached;
:meth:`~spexai.inference.priors.PriorSet.from_params` turns a ``Param`` list
into the prior a fit actually uses.
"""
import warnings
from dataclasses import dataclass

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

    **Deprecated.** This is now a thin adapter over
    :class:`~spexai.inference.spectral_fit.SpectralFit`, which is the single
    assembly point; it survives only so existing callers and the scalar-vs-
    vectorised reference tests keep working unchanged. New code should build a
    ``SpectralFit`` directly -- it reaches all nine samplers, whereas this
    returns a bare posterior.

    Returns ``None`` if the fit needs something the batched forward cannot
    express (see :func:`vectorization_blocker`), which ``SpectralFit`` would
    instead raise over.

    ``n_h_scale=1.0`` is pinned here deliberately: this entry point has always
    taken ``n_h`` in absolute cm^-2, and the campaign's 1e21 convention would
    silently change every existing caller's absorption by that factor.
    """
    from spexai.inference.priors import PriorSet
    from spexai.inference.spectral_fit import SpectralFit

    warnings.warn(
        "fitting.build_posterior is deprecated; build a "
        "spexai.inference.SpectralFit instead. It takes the same pieces, gives "
        "you .posterior, and adds .sample(<any of nine samplers>). Note "
        "build_posterior pins n_h_scale=1.0 (n_h in absolute cm^-2); "
        "SpectralFit makes you state the convention.",
        DeprecationWarning, stacklevel=2)

    if vectorization_blocker([p.name for p in params], dem) is not None:
        return None
    return SpectralFit(
        emulator=model, response=obs.response, counts=obs.counts,
        exposure=obs.exposure, priors=PriorSet.from_params(params),
        abundances=abundance_model, absorption=absorption, dem=dem,
        redshift=10.0 ** float(fixed.get("logz", -10.0)),
        luminosity_distance=float(fixed.get("luminosity_distance", D_REF_M)),
        keep=keep, fixed=fixed, n_h_scale=1.0, device=model.device,
        **forward_kwargs).posterior
