"""SBC primitives: ranks, thinning and the uniformity verdict.

The load-bearing test is ``test_correlated_chain_fails_without_thinning``: it
pins the bug the old ``bias_study.py --stage sbc`` had, where a perfectly
calibrated sampler was reported as miscalibrated because the ranks were taken
over a raw correlated chain.
"""
import numpy as np
import pytest

from spexai.inference import calibration


def _calibrated_chain(rng, n_steps, n_walkers, ndim, rho=0.0):
    """AR(1) chain around 0 with unit marginal variance.

    ``rho`` is the step-to-step correlation, so the integrated autocorrelation
    time is ``(1 + rho) / (1 - rho)`` and ``rho=0`` gives independent draws.
    """
    chain = np.empty((n_steps, n_walkers, ndim))
    x = rng.standard_normal((n_walkers, ndim))
    scale = np.sqrt(1.0 - rho ** 2)
    for t in range(n_steps):
        x = rho * x + scale * rng.standard_normal((n_walkers, ndim))
        chain[t] = x
    return chain


def test_sbc_rank_counts_draws_below_truth():
    draws = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert calibration.sbc_rank(draws, np.array([2.5])) == np.array([3])
    assert calibration.sbc_rank(draws, np.array([-1.0])) == np.array([0])
    assert calibration.sbc_rank(draws, np.array([9.0])) == np.array([4])


def test_sbc_rank_rejects_1d_draws():
    with pytest.raises(ValueError):
        calibration.sbc_rank(np.zeros(4), np.array([0.0]))


def test_autocorr_thin_matches_tau():
    # ess = walkers * steps / tau, so tau = 4 here
    assert calibration.autocorr_thin(1000, 8, ess=2000.0) == 4
    # independent chain: no thinning
    assert calibration.autocorr_thin(1000, 8, ess=8000.0) == 1
    # a NaN ESS must not silently pretend the chain is independent
    assert calibration.autocorr_thin(1000, 8, ess=np.nan) == 1000


def test_thin_to_independent_respects_max_draws():
    rng = np.random.default_rng(0)
    chain = _calibrated_chain(rng, 400, 16, 3)
    draws, thin = calibration.thin_to_independent(
        chain, ess=np.array([400.0 * 16]), max_draws=100, rng=rng)
    assert draws.shape == (100, 3)
    assert thin == 1


def test_thin_to_independent_rejects_flat_chain():
    with pytest.raises(ValueError):
        calibration.thin_to_independent(np.zeros((10, 3)), ess=None)


def test_uniform_ranks_pass():
    rng = np.random.default_rng(1)
    ranks = rng.integers(0, 101, size=400)
    out = calibration.rank_uniformity(ranks, n_draws=100)
    assert out["p_value"] > 0.05
    assert out["shape"] == "uniform"


def test_overconfident_posterior_gives_u_shape():
    """Truth falls in the tails too often => ranks pile up at 0 and L."""
    rng = np.random.default_rng(2)
    ranks = np.where(rng.random(400) < 0.5,
                     rng.integers(0, 12, 400), rng.integers(89, 101, 400))
    out = calibration.rank_uniformity(ranks, n_draws=100)
    assert out["p_value"] < 0.05
    assert "too narrow" in out["shape"]


def test_biased_posterior_shifts_mean_rank():
    rng = np.random.default_rng(3)
    ranks = rng.integers(60, 101, size=400)     # truth usually above the draws
    out = calibration.rank_uniformity(ranks, n_draws=100)
    assert out["p_value"] < 0.05
    assert out["mean_rank"] > 0.5


def test_correlated_chain_fails_without_thinning():
    """The regression this module exists for.

    A calibrated-by-construction setup: the 'posterior' is standard normal, the
    'truth' is drawn from the same normal, so ranks must be uniform. Taking them
    from the raw rho=0.9 chain concentrates them and the test rejects; thinning
    by the autocorrelation time recovers uniformity.
    """
    rng = np.random.default_rng(4)
    n_sims, n_steps, n_walkers = 600, 600, 8
    rho = 0.9
    tau = (1 + rho) / (1 - rho)
    ess = np.array([n_steps * n_walkers / tau])

    raw_ranks, thin_ranks = [], []
    for _ in range(n_sims):
        chain = _calibrated_chain(rng, n_steps, n_walkers, 1, rho=rho)
        truth = rng.standard_normal(1)
        # untitled draws straight off one walker: 100 samples but only ~5
        # independent ones, which is what skipping the thinning buys you
        raw_ranks.append(calibration.sbc_rank(chain[:100, 0, :], truth)[0])
        draws, _ = calibration.thin_to_independent(
            chain, ess, max_draws=100, rng=rng)
        thin_ranks.append(calibration.sbc_rank(draws, truth)[0])

    raw = calibration.rank_uniformity(np.array(raw_ranks), n_draws=100)
    thinned = calibration.rank_uniformity(np.array(thin_ranks), n_draws=100)
    assert raw["p_value"] < 0.05, "correlated ranks should look miscalibrated"
    assert thinned["p_value"] > 0.05, "thinned ranks should look calibrated"
    # the mechanism, asserted directly rather than through a p-value threshold:
    # 100 correlated draws are only ~5 independent ones, so the sample cloud is
    # too narrow and the truth falls OUTSIDE it too often -- ranks pile up at 0
    # and L, giving a U-shaped (over-dispersed) histogram. That is why an
    # under-thinned SBC run is misread as "posterior too narrow".
    assert raw["var_ratio"] > 1.0
    assert raw["var_ratio"] > thinned["var_ratio"]
    assert "too narrow" in raw["shape"]


def test_summarise_lists_every_parameter():
    rng = np.random.default_rng(5)
    ranks = {"kT": rng.integers(0, 101, 50), "Fe": rng.integers(0, 101, 50)}
    text = calibration.summarise(ranks, n_draws=100)
    assert "kT" in text and "Fe" in text
