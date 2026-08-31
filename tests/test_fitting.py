"""The vectorised and scalar likelihoods must be the same likelihood.

``fitting`` now has two implementations: ``make_loglike`` (one parameter set per
call, through ``predict_counts``) and the batched ``PoissonPosterior`` over
``VectorForward``. The batched one is what production uses; the scalar one is
kept precisely so this file can check it.

That check is not ceremony. The batched forward has already produced exactly
this class of bug once -- a per-walker ``n_h`` broadcast against the element
axis instead of the walker axis, invisible to every test that used a scalar
column density. These tests exercise the walker axis with *distinct* values per
walker, which is the only way such a bug shows up.
"""
import os

import numpy as np
import pytest
import torch

from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR
from spexai.inference.abundances import AbundanceModel
from spexai.inference.fitting import (Param, build_posterior, make_loglike,
                                      run_emcee, vectorization_blocker)
from spexai.inference.simulate import simulate_observation
from spexai.inference.response import Response

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
    reason="model store not present")

ELEMENTS = [2, 8, 26]
RMF = os.path.expanduser("~/work/data/spexai/responses/aciss_aimpt_cy28.rmf")


@pytest.fixture(scope="module")
def model():
    return JointOperatorModel(device="cpu", elements=ELEMENTS)


@pytest.fixture(scope="module")
def obs(model):
    if not os.path.exists(RMF):
        pytest.skip("ACIS response not available")
    resp = Response(RMF, None)   # folding mechanics only; no effective area
    p = {"temp": 3.0, "velocity": 200.0, "norm": 1e10, "logz": -10.0,
         "abundances": {}}
    return simulate_observation(model, resp, p, exposure=1e4,
                               target_counts=2e4, rng=0)


def _params():
    return [Param("temp", 1.0, 8.0, truth=3.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0)]


FIXED = {"abundances": {}, "logz": -10.0}


def test_vectorised_matches_scalar_loglike(model, obs):
    # distinct values per walker on every axis: a mis-aligned broadcast cannot
    # survive this even though it survives identical rows
    params = _params()
    names = [p.name for p in params]
    post = build_posterior(obs, model, params, FIXED)
    assert post is not None
    scalar = make_loglike(obs, model, names, FIXED)

    theta = np.array([[2.5, 120.0, 9.8],
                      [3.0, 200.0, 10.0],
                      [4.2, 480.0, 10.3]])
    got = post.loglike(theta)
    ref = np.array([scalar(t) for t in theta])
    assert np.allclose(got, ref, rtol=1e-4), f"{got} vs {ref}"


def test_vectorised_matches_scalar_with_absorption(model, obs):
    from spexai.inference.absorption import Absorption
    absn = Absorption.default()
    params = _params() + [Param("n_h", 0.0, 5e21, truth=1e21)]
    names = [p.name for p in params]
    post = build_posterior(obs, model, params, FIXED, absorption=absn)
    scalar = make_loglike(obs, model, names, FIXED, absorption=absn)
    # per-walker n_h specifically: the axis that was silently wrong before
    theta = np.array([[3.0, 200.0, 10.0, 2e20],
                      [3.0, 200.0, 10.0, 3e21]])
    got = post.loglike(theta)
    ref = np.array([scalar(t) for t in theta])
    assert np.allclose(got, ref, rtol=1e-4), f"{got} vs {ref}"


def test_vectorised_matches_scalar_with_abundance_model(model, obs):
    ab = AbundanceModel(model.elements).free_element(26, "Fe")
    ab.tie_const([z for z in model.elements if z >= 3 and z != 26], 1.0, 26)
    params = _params() + [Param("Fe", 0.1, 2.0, truth=1.0)]
    names = [p.name for p in params]
    post = build_posterior(obs, model, params, FIXED, abundance_model=ab)
    scalar = make_loglike(obs, model, names, FIXED, abundance_model=ab)
    theta = np.array([[3.0, 200.0, 10.0, 0.6],
                      [3.4, 260.0, 10.1, 1.4]])
    got = post.loglike(theta)
    ref = np.array([scalar(t) for t in theta])
    assert np.allclose(got, ref, rtol=1e-4), f"{got} vs {ref}"


def test_fixed_abundances_are_applied(model, obs):
    # a constant abundance in `fixed` must reach the forward; dropping it would
    # look like a mild flux offset, not an error
    fixed = {"abundances": {26: 0.3}, "logz": -10.0}
    params = _params()
    post = build_posterior(obs, model, params, fixed)
    scalar = make_loglike(obs, model, [p.name for p in params], fixed)
    theta = np.array([[3.0, 200.0, 10.0]])
    assert np.allclose(post.loglike(theta), [scalar(theta[0])], rtol=1e-4)
    # and it must actually change the answer vs solar
    solar = build_posterior(obs, model, params, FIXED)
    assert not np.isclose(post.loglike(theta)[0], solar.loglike(theta)[0])


def test_dem_without_batched_weights_blocks_vectorisation():
    # DEMs themselves are fine now; only a shape with no weights_batch (a scipy
    # distribution with no torch equivalent) still forces the scalar path
    assert vectorization_blocker(["temp"], dem=object()) is not None
    assert "weights_batch" in vectorization_blocker(["temp"], object())


def test_sampled_redshift_blocks_vectorisation():
    assert vectorization_blocker(["temp", "logz"], None) is not None
    assert vectorization_blocker(["temp", "log_norm"], None) is None


def test_run_emcee_warns_and_falls_back_on_dem(model, obs):
    class _DEM:
        temp_grid = torch.tensor([2.0, 3.0])

        def weights(self, p):
            return torch.tensor([0.5, 0.5])

    with pytest.warns(RuntimeWarning, match="scalar likelihood"):
        res = run_emcee(obs, model, _params(), FIXED, nwalkers=8, nsteps=3,
                        dem=_DEM())
    assert res.samples.shape[1] == 3


def test_run_emcee_vectorised_and_scalar_agree(model, obs):
    # same seed, same walkers, same proposals -> the chains must coincide, which
    # is a far stricter check than the posteriors merely looking similar
    kw = dict(nwalkers=8, nsteps=6, seed=3)
    vec = run_emcee(obs, model, _params(), FIXED, vectorized=True, **kw)
    sca = run_emcee(obs, model, _params(), FIXED, vectorized=False, **kw)
    assert np.allclose(vec.chain, sca.chain, rtol=1e-3, atol=1e-6)


# --- DEM ---------------------------------------------------------------------

def _dem():
    from spexai.inference import tempdist as td
    return td.gaussian_T(td.TempGrid(1.0, 8.0, n=10))


def _dem_params():
    return [Param("T_mean", 1.0, 8.0, truth=3.0),
            Param("T_sigma", 0.05, 4.0, truth=1.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0)]


def test_dem_no_longer_blocks_vectorisation():
    assert vectorization_blocker(["T_mean"], _dem()) is None


def test_vectorised_dem_matches_scalar(model, obs):
    # distinct DEM shapes per walker: _flux_dem flattens (B, G) into one batch,
    # so a repeat instead of repeat_interleave would pair each walker's weights
    # with another walker's temperatures -- wrong, and perfectly plausible-looking
    dem = _dem()
    params = _dem_params()
    names = [p.name for p in params]
    post = build_posterior(obs, model, params, FIXED, dem=dem)
    assert post is not None
    scalar = make_loglike(obs, model, names, FIXED, dem=dem)
    theta = np.array([[2.0, 0.5, 150.0, 9.9],
                      [3.5, 1.2, 240.0, 10.1],
                      [5.5, 2.5, 400.0, 10.2]])
    got = post.loglike(theta)
    ref = np.array([scalar(t) for t in theta])
    assert np.allclose(got, ref, rtol=1e-4), f"{got} vs {ref}"


def test_vectorised_dem_matches_scalar_with_absorption(model, obs):
    from spexai.inference.absorption import Absorption
    absn = Absorption.default()
    dem = _dem()
    params = _dem_params() + [Param("n_h", 0.0, 5e21, truth=1e21)]
    names = [p.name for p in params]
    post = build_posterior(obs, model, params, FIXED, absorption=absn, dem=dem)
    scalar = make_loglike(obs, model, names, FIXED, dem=dem, absorption=absn)
    theta = np.array([[3.0, 1.0, 200.0, 10.0, 5e20],
                      [4.0, 0.6, 300.0, 10.1, 3e21]])
    got = post.loglike(theta)
    ref = np.array([scalar(t) for t in theta])
    assert np.allclose(got, ref, rtol=1e-4), f"{got} vs {ref}"


def test_dem_walker_chunk_shrinks_by_grid_size(model, obs):
    # a G-point grid makes each walker G emulator rows; the walker chunk must
    # shrink accordingly or peak memory is G times what `chunk` promised
    dem = _dem()
    post = build_posterior(obs, model, _dem_params(), FIXED, dem=dem)
    g = dem.temp_grid.numel()
    assert post.forward.walker_chunk == max(1, post.forward.chunk // g)
    plain = build_posterior(obs, model, _params(), FIXED)
    assert plain.forward.walker_chunk == plain.forward.chunk


def test_dem_chunking_is_numerically_transparent(model, obs):
    # more walkers than the (shrunken) chunk, so the chunked path is exercised
    post = build_posterior(obs, model, _dem_params(), FIXED, dem=_dem())
    rng = np.random.default_rng(0)
    theta = np.column_stack([rng.uniform(2.0, 5.0, 7),
                             rng.uniform(0.3, 2.0, 7),
                             rng.uniform(100.0, 400.0, 7),
                             rng.uniform(9.9, 10.1, 7)])
    together = post.loglike(theta)
    apart = np.concatenate([post.loglike(theta[i:i + 1]) for i in range(7)])
    assert np.allclose(together, apart, rtol=1e-5)


def test_dem_run_emcee_does_not_warn(model, obs):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = run_emcee(obs, model, _dem_params(), FIXED, nwalkers=10,
                        nsteps=3, dem=_dem())
    assert res.samples.shape[1] == 4
