"""Unit tests for observation simulation."""
import numpy as np

from spexai.inference.simulate import (Observation, expected_counts,
                                       simulate_grid, simulate_observation)


def test_expected_counts_shape_nonneg(fe_model, acis_response):
    mu = expected_counts(fe_model, acis_response,
                         {"temp": 2.0, "norm": 1e10}, exposure=1e4)
    assert mu.shape == (acis_response.n_channels,)
    assert (mu >= 0).all() and np.isfinite(mu).all()


def test_simulate_observation_basic(fe_model, acis_response):
    o = simulate_observation(fe_model, acis_response, {"temp": 2.0, "norm": 1e10},
                             exposure=1e4, target_counts=1e5,
                             instrument="ACIS", rng=0)
    assert isinstance(o, Observation)
    assert o.counts.shape == (acis_response.n_channels,)
    assert o.counts.dtype.kind in "iu"              # integer counts
    assert (o.counts >= 0).all()
    assert o.instrument == "ACIS"
    assert o.n_channels == acis_response.n_channels


def test_target_counts_sets_expected_total(fe_model, acis_response):
    o = simulate_observation(fe_model, acis_response, {"temp": 2.0, "norm": 1e10},
                             exposure=1e4, target_counts=1e5, rng=1)
    assert np.isclose(o.expected.sum(), 1e5, rtol=1e-5)      # exact after rescale
    assert abs(o.total_counts - 1e5) < 5 * np.sqrt(1e5)      # Poisson ~ target


def test_reproducible_and_varies_with_seed(fe_model, acis_response):
    p = {"temp": 2.0, "norm": 1e10}
    a = simulate_observation(fe_model, acis_response, p, 1e4, 1e5, rng=7)
    b = simulate_observation(fe_model, acis_response, p, 1e4, 1e5, rng=7)
    c = simulate_observation(fe_model, acis_response, p, 1e4, 1e5, rng=8)
    assert np.array_equal(a.counts, b.counts)
    assert not np.array_equal(a.counts, c.counts)


def test_rescaled_norm_is_recorded(fe_model, acis_response):
    # the norm actually used (rescaled to hit target_counts) is the round-trip
    # truth the fit must recover, so it must be recorded, not the input norm
    o = simulate_observation(fe_model, acis_response, {"temp": 2.0, "norm": 1e10},
                             exposure=1e4, target_counts=1e5, rng=0)
    assert o.true_params["norm"] != 1e10
    assert np.isclose(expected_counts(fe_model, acis_response,
                                      o.true_params, 1e4).sum(), 1e5, rtol=1e-5)


def test_simulate_grid(fe_model, acis_response):
    temps = [0.7, 2.0, 8.0]
    responses = {"ACIS": acis_response}
    obs = simulate_grid(fe_model, responses, temps, {"norm": 1e10},
                        exposure=1e4, target_counts=1e5, seed=0)
    assert len(obs) == len(temps) * len(responses)
    assert [o.true_params["temp"] for o in obs] == temps
    assert all(o.instrument == "ACIS" for o in obs)
