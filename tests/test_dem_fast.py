"""The fixed-grid trunk table must be a pure cache, not a second forward.

``BatchedJointForward.build_temp_table`` tabulates the trunk continuum and the
line amplitudes at the DEM's (fixed) temperature grid, so they are evaluated
once per run instead of once per forward -- the ~83% of the DEM forward the
coordinate-operator trunk was costing. The table is on by default for a DEM
(``VectorForward(dem_fast=True)``), and ``dem_fast=False`` restores the
uncached forward, which is the reference every test here compares against.

Two things can go wrong and would be invisible to the rest of the suite:

* the table is served where it must not be -- to a *single-T* fit whose kT
  carries ``requires_grad``, where a gather has no derivative and would
  silently zero ``d(counts)/d(kT)``;
* the table goes stale -- served after the grid, the band restriction or the
  models have changed underneath it.

The positive controls at the bottom are for exactly those two.
"""
import os

import numpy as np
import pytest
import torch

from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR
from spexai.inference.fitting import Param, build_posterior
from spexai.inference.simulate import simulate_observation
from spexai.inference.response import Response
from spexai.inference import tempdist as td

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
    reason="model store not present")

ELEMENTS = [2, 8, 26]
RMF = os.path.expanduser("~/work/data/spexai/responses/aciss_aimpt_cy28.rmf")
FIXED = {"abundances": {}, "logz": -10.0}


@pytest.fixture(scope="module")
def model():
    return JointOperatorModel(device="cpu", elements=ELEMENTS)


@pytest.fixture(scope="module")
def obs(model):
    if not os.path.exists(RMF):
        pytest.skip("ACIS response not available")
    resp = Response(RMF, None)
    p = {"temp": 3.0, "velocity": 200.0, "norm": 1e10, "logz": -10.0,
         "abundances": {}}
    return simulate_observation(model, resp, p, exposure=1e4,
                                target_counts=2e4, rng=0)


def _dem():
    return td.gaussian_T(td.TempGrid(1.0, 8.0, n=10))


def _dem_params():
    return [Param("T_mean", 1.0, 8.0, truth=3.0),
            Param("T_sigma", 0.05, 4.0, truth=1.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0)]


def _pair(obs, model, **kw):
    """(fast, reference) forwards over the same DEM and parameters."""
    fast = build_posterior(obs, model, _dem_params(), FIXED, dem=_dem(),
                           dem_fast=True, **kw).forward
    ref = build_posterior(obs, model, _dem_params(), FIXED, dem=_dem(),
                          dem_fast=False, **kw).forward
    return fast, ref


def _theta(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(2.0, 5.0, n),      # T_mean
                            rng.uniform(0.3, 2.0, n),      # T_sigma
                            rng.uniform(80.0, 500.0, n),   # sigma_v
                            rng.uniform(9.9, 10.1, n)])    # log_norm


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_table_is_used_at_all(obs, model):
    """Guard against the whole file passing because the fast path never fires:
    every comparison below is vacuous if no table was built."""
    fast, _ = _pair(obs, model)
    fast(_theta(1))
    tab = fast.emu.batched.temp_table
    assert tab is not None and tab.matches(_dem().temp_grid)
    fast.emu.batched.temp_table = None            # leave the module fixture clean


def test_fast_matches_uncached_values(obs, model):
    fast, ref = _pair(obs, model)
    th = _theta(5)
    got, want = fast(th), ref(th)
    assert _rel(got, want) < 1e-5, _rel(got, want)


def test_fast_matches_uncached_values_logT(obs, model):
    """The bias campaign's DEM is a Gaussian in log10 T. The table and the
    contraction never see the shape family -- only its (B, G) weights -- so
    the log-T parametrisation must agree with the uncached forward exactly as
    the linear-T one does."""
    def dem():
        return td.gaussian_logT(td.TempGrid(1.0, 8.0, n=10))
    params = [Param("logT_mean", 0.0, 0.9, truth=0.5),
              Param("logT_sigma", 0.03, 0.5, truth=0.15),
              Param("velocity", 30.0, 600.0, truth=200.0),
              Param("log_norm", 9.0, 11.0, truth=10.0)]
    fast = build_posterior(obs, model, params, FIXED, dem=dem(),
                           dem_fast=True).forward
    ref = build_posterior(obs, model, params, FIXED, dem=dem(),
                          dem_fast=False).forward
    rng = np.random.default_rng(1)
    th = np.column_stack([rng.uniform(0.3, 0.7, 5),       # logT_mean
                          rng.uniform(0.08, 0.3, 5),      # logT_sigma
                          rng.uniform(80.0, 500.0, 5),    # sigma_v
                          rng.uniform(9.9, 10.1, 5)])     # log_norm
    got, want = fast(th), ref(th)
    fast.emu.batched.temp_table = None
    assert _rel(got, want) < 1e-5, _rel(got, want)


def test_fast_matches_uncached_with_absorption(obs, model):
    """Absorption is applied on the fine grid and at the line energies, i.e.
    downstream of everything the table holds -- so it must be untouched by the
    cache, including a per-walker n_h."""
    from spexai.inference.absorption import Absorption
    params = _dem_params() + [Param("n_h", 0.0, 1e22, truth=3e21)]
    absn = Absorption.default()
    out = []
    for flag in (True, False):
        fwd = build_posterior(obs, model, params, FIXED, absorption=absn,
                              dem=_dem(), dem_fast=flag).forward
        th = np.column_stack([_theta(4), np.array([0.0, 1e21, 3e21, 8e21])])
        out.append(fwd(th))
    assert _rel(out[0], out[1]) < 1e-5, _rel(out[0], out[1])


def test_fast_matches_uncached_gradients(obs, model):
    """The DEM gradient flows through the weights, the broadening and the
    abundances -- never through the trunk, whose input is the fixed grid. The
    table must therefore leave every gradient the fit uses unchanged."""
    fast, ref = _pair(obs, model)
    grads = []
    for fwd in (fast, ref):
        th = torch.tensor(_theta(2), dtype=torch.float32,
                          requires_grad=True)
        bt = fwd.emu.batched
        with bt.grad_enabled(True):
            fwd.counts_torch(th, grad=True).sum().backward()
        grads.append(th.grad.detach().clone())
    assert torch.isfinite(grads[0]).all()
    assert (grads[0].abs() > 0).all()            # every column is live
    rel = (grads[0] - grads[1]).abs().max() / grads[1].abs().max()
    assert float(rel) < 1e-4, float(rel)


def test_table_not_served_to_a_differentiable_temperature(obs, model):
    """THE trap. A single-T fit whose kT lands exactly on a tabulated
    temperature would get a correct value from the table and a zero gradient,
    because a gather does not differentiate w.r.t. the temperature. Build the
    table, then ask a single-T forward for d(counts)/d(kT) AT a grid node."""
    grid = _dem().temp_grid
    kt = float(grid[4])                           # exactly a tabulated node
    params = [Param("temp", 1.0, 8.0, truth=kt),
              Param("velocity", 30.0, 600.0, truth=200.0),
              Param("log_norm", 9.0, 11.0, truth=10.0)]
    fwd = build_posterior(obs, model, params, FIXED).forward
    fwd.emu.batched.build_temp_table(grid)
    try:
        th = torch.tensor([[kt, 200.0, 10.0]], requires_grad=True)
        with fwd.emu.batched.grad_enabled(True):
            fwd.counts_torch(th, grad=True).sum().backward()
        assert float(th.grad[0, 0].abs()) > 0.0, "d(counts)/d(kT) was zeroed"
    finally:
        fwd.emu.batched.temp_table = None

    # and the VALUE at a node must be the cached one, i.e. the cache is a cache
    fwd.emu.batched.build_temp_table(grid)
    try:
        with torch.no_grad():
            cached = fwd.counts_torch(torch.tensor([[kt, 200.0, 10.0]]))
        fwd.emu.batched.temp_table = None
        with torch.no_grad():
            plain = fwd.counts_torch(torch.tensor([[kt, 200.0, 10.0]]))
    finally:
        fwd.emu.batched.temp_table = None
    assert _rel(cached.numpy(), plain.numpy()) < 1e-6


def test_table_is_rebuilt_when_the_grid_changes(obs, model):
    """A stale table is the failure mode with no symptom: it would serve one
    grid's continua for another grid's temperatures, and every spectrum would
    still look like a spectrum."""
    fwd = build_posterior(obs, model, _dem_params(), FIXED,
                          dem=td.gaussian_T(td.TempGrid(1.0, 8.0, n=10))).forward
    th = _theta(2)
    fwd(th)
    assert fwd.emu.batched.temp_table.temps.numel() == 10
    fwd.dem = td.gaussian_T(td.TempGrid(1.0, 8.0, n=14))   # grid swapped
    got = fwd(th)
    assert fwd.emu.batched.temp_table.temps.numel() == 14
    ref = build_posterior(obs, model, _dem_params(), FIXED, dem_fast=False,
                          dem=td.gaussian_T(td.TempGrid(1.0, 8.0, n=14))).forward(th)
    assert _rel(got, ref) < 1e-5
    fwd.emu.batched.temp_table = None


def test_sigma_v_still_moves_the_spectrum(obs, model):
    """Positive control: the table holds only temperature-dependent pieces, so
    the kinematics must still be live. If the table ever swallowed the
    broadening, every comparison above would still pass."""
    fast, _ = _pair(obs, model)
    mu = fast(np.array([[3.5, 0.8, 100.0, 10.0],
                        [3.5, 0.8, 600.0, 10.0]]))
    assert _rel(mu[0], mu[1]) > 1e-3
    fast.emu.batched.temp_table = None
