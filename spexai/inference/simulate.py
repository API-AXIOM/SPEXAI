"""Simulate detector-level observations from the joint operator model.

Given a `JointOperatorModel`, an instrument `Response`, and physical
parameters, compute the expected channel counts and draw a Poisson
realisation. `simulate_grid` produces the standard test matrix of several
temperatures x several instruments used for the inference round-trip tests.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from spexai.config import STORE, DATADIR
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.units import D_REF_M


@dataclass
class Observation:
    """A simulated observation and everything a fit needs to score it."""
    counts: np.ndarray           # observed counts per detector channel (int)
    response: object             # the Response it was folded through
    exposure: float              # seconds
    true_params: dict            # generating params (incl. the norm actually used)
    instrument: str = ""
    expected: np.ndarray = field(default=None, repr=False)  # Poisson mean

    @property
    def n_channels(self):
        return len(self.counts)

    @property
    def total_counts(self):
        return int(self.counts.sum())


def expected_counts(model, response, params, exposure, absorption=None):
    """Poisson-mean counts per channel for one parameter set (no draw).

    Optional Galactic absorption: pass an ``absorption`` screen and carry the
    column as ``params["n_h"]`` (cm^-2)."""
    mu = model.predict_counts(
        torch.tensor([float(params["temp"])]),
        params.get("abundances", {}),
        float(params.get("logz", -10.0)),
        float(params["norm"]),
        float(params.get("velocity", 0.0)),
        response, exposure=exposure,
        luminosity_distance=float(params.get("luminosity_distance", D_REF_M)),
        absorption=absorption, n_h=float(params.get("n_h", 0.0))
    ).squeeze(0).cpu().numpy()
    return np.clip(mu, 0.0, None)


def simulate_observation(model, response, params, exposure,
                         target_counts=None, instrument="", rng=None,
                         absorption=None):
    """Simulate one observation.

    params: {temp, norm, [abundances], [logz], [velocity], [n_h]}.
    target_counts: if given, `norm` is rescaled so the *expected* total counts
        equal this (handy while absolute flux units are still placeholder);
        the rescaled norm is stored in the returned Observation's true_params.
    rng: int seed or numpy Generator for the Poisson draw.
    absorption: optional Galactic absorption screen (with ``params["n_h"]``).
    """
    p = dict(params)
    p.setdefault("abundances", {})
    p.setdefault("logz", -10.0)
    p.setdefault("velocity", 0.0)
    mu = expected_counts(model, response, p, exposure, absorption=absorption)
    if target_counts is not None:
        scale = float(target_counts) / max(float(mu.sum()), 1e-30)
        p["norm"] = float(p["norm"]) * scale
        mu = mu * scale
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    counts = gen.poisson(mu).astype(np.int64)
    return Observation(counts=counts, response=response, exposure=exposure,
                       true_params=p, instrument=instrument, expected=mu)


def simulate_grid(model, responses, temps, base_params, exposure,
                  target_counts=None, seed=0):
    """Simulate every (temperature x instrument) combination.

    responses: {instrument_name: Response}. temps: iterable of temperatures.
    base_params: shared params (norm, abundances, logz, velocity); `temp` is
        overridden per grid point. Returns a list of Observations, seeded
        reproducibly (each point gets its own seed derived from `seed`).
    """
    obs = []
    for i, T in enumerate(temps):
        for j, (name, resp) in enumerate(responses.items()):
            p = {**base_params, "temp": float(T)}
            obs.append(simulate_observation(
                model, resp, p, exposure, target_counts=target_counts,
                instrument=name, rng=seed + 1000 * i + j))
    return obs


def stream_truth_counts(elements, abundances, z, kT, vel, n_h, dist_m,
                        response, absorption, exposure=1e5, norm_ref=1e11,
                        dem=None, dem_params=None, store=None, datadir=None,
                        device="cpu", verbose=False) -> np.ndarray:
    """Noise-free channel counts from the independent PCHIP SPEX truth model
    (``SpexTruthModel``), summed one element at a time.

    Unlike ``expected_counts`` (which scores the *emulator* being fit), this
    generates an injected truth from the training data directly -- the
    round-trip that measures the emulator's systematic bias. Elements are
    streamed one at a time and discarded (~0.4 GB peak instead of ~11 GB for
    a full store) because abundance weighting, velocity broadening, Galactic
    absorption and folding are all LINEAR in each element's native flux, so
    the total is just their sum.

    ``dem``/``dem_params``: pass a ``tempdist`` model + its params for a
    multi-temperature DEM; leave ``dem=None`` for a single temperature
    ``kT``. Returns the Poisson-mean counts (N_channels,) at ``norm_ref``;
    rescale linearly to any target total counts.
    """
    store = STORE if store is None else store
    datadir = DATADIR if datadir is None else datadir
    dem_params = dem_params or {}
    logz = float(np.log10(z))
    total = None
    for el in elements:
        el = int(el)
        a = abundances.get(el, 1.0)
        if a == 0.0 or el in (1, 2):
            # H/He carry the continuum at solar -> include them at a=1.0
            if el not in (1, 2):
                continue
            a = 1.0
        m = SpexTruthModel(models_dir=store, datadir=datadir,
                           elements=[el], device=device)
        if dem is not None:
            w = dem.weights(dem_params)
            c = m.predict_counts_dem(
                dem.temp_grid, w, {el: a}, logz, norm_ref, vel, response,
                exposure, luminosity_distance=dist_m,
                absorption=absorption, n_h=n_h)
        else:
            c = m.predict_counts(
                torch.tensor([kT]), {el: a}, logz, norm_ref, vel, response,
                exposure, luminosity_distance=dist_m,
                absorption=absorption, n_h=n_h)
        c = c.squeeze(0).cpu().numpy()
        total = c if total is None else total + c
        if verbose:
            print(f"  +Z{el:02d} a={a:.3f}  running total counts={total.sum():.3e}")
        del m
    return np.clip(total, 0.0, None)
