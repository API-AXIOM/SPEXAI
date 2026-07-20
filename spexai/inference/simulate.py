"""Simulate detector-level observations from the joint operator model.

Given a `JointOperatorModel`, an instrument `Response`, and physical
parameters, compute the expected channel counts and draw a Poisson
realisation. `simulate_grid` produces the standard test matrix of several
temperatures x several instruments used for the inference round-trip tests.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


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


def expected_counts(model, response, params, exposure):
    """Poisson-mean counts per channel for one parameter set (no draw)."""
    mu = model.predict_counts(
        torch.tensor([float(params["temp"])]),
        params.get("abundances", {}),
        float(params.get("logz", -10.0)),
        float(params["norm"]),
        float(params.get("velocity", 0.0)),
        response, exposure=exposure).squeeze(0).cpu().numpy()
    return np.clip(mu, 0.0, None)


def simulate_observation(model, response, params, exposure,
                         target_counts=None, instrument="", rng=None):
    """Simulate one observation.

    params: {temp, norm, [abundances], [logz], [velocity]}.
    target_counts: if given, `norm` is rescaled so the *expected* total counts
        equal this (handy while absolute flux units are still placeholder);
        the rescaled norm is stored in the returned Observation's true_params.
    rng: int seed or numpy Generator for the Poisson draw.
    """
    p = dict(params)
    p.setdefault("abundances", {})
    p.setdefault("logz", -10.0)
    p.setdefault("velocity", 0.0)
    mu = expected_counts(model, response, p, exposure)
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
