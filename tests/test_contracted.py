"""Contracting the element (and DEM grid) axis before the broadening tail.

Broadening, absorption, rebinning and the line deposit are linear in the flux
and depend on the walker only through ``(sigma_v, n_h)``, so summing over
elements -- and over a DEM's temperature grid -- BEFORE that tail is the same
arithmetic in a different order, with ``B`` fine-grid rows instead of ``N*B``
(``N*B*G`` for a DEM). ``contract=False`` / ``contract_first=False`` keeps the
element-stacked path, which is what everything here compares against.

The reassociation is a sum of positive terms with no cancellation, so the two
orders agree to float32 noise; the tolerances below are set from the measured
agreement, not from hope.

The per-walker tests matter more than they look. The contraction is a GEMM
over ``(B, N)`` abundances and ``(B, G)`` weights, and getting an axis wrong
there gives every walker some other walker's abundances -- a spectrum that is
still a spectrum, and identical for the scalar-abundance case that most tests
use.
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
def joint():
    return JointOperatorModel(device="cpu", elements=ELEMENTS)


@pytest.fixture(scope="module")
def edges(joint):
    return torch.linspace(0.5, 9.0, 4001)


@pytest.fixture(scope="module")
def obs(joint):
    if not os.path.exists(RMF):
        pytest.skip("ACIS response not available")
    resp = Response(RMF, None)
    p = {"temp": 3.0, "velocity": 200.0, "norm": 1e10, "logz": -10.0,
         "abundances": {}}
    return simulate_observation(joint, resp, p, exposure=1e4,
                                target_counts=2e4, rng=0)


def _rel(a, b):
    a, b = (x.detach().numpy() if torch.is_tensor(x) else np.asarray(x)
            for x in (a, b))
    return float(np.abs(a - b).max() / np.abs(b).max())


def _dem():
    return td.gaussian_T(td.TempGrid(1.0, 8.0, n=10))


def _dem_params():
    return [Param("T_mean", 1.0, 8.0, truth=3.0),
            Param("T_sigma", 0.05, 4.0, truth=1.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0)]


def _theta(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(2.0, 5.0, n),
                            rng.uniform(0.3, 2.0, n),
                            rng.uniform(80.0, 500.0, n),
                            rng.uniform(9.9, 10.1, n)])


# --- single temperature ------------------------------------------------------

def test_contracted_matches_stacked(joint, edges):
    T = torch.tensor([0.9, 3.0, 7.5])
    ab = {2: 1.0, 8: 0.6, 26: 1.3}
    got = joint.batched.flux(T, ab, 150.0, edges, contract=True)
    ref = joint.batched.flux(T, ab, 150.0, edges, contract=False)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-5, _rel(got, ref)


def test_contracted_matches_stacked_per_walker(joint, edges):
    """Per-walker abundances, sigma_v and n_h at once: the axis-mixing bug the
    class docstring warns about shows up here and nowhere else."""
    from spexai.inference.absorption import Absorption
    T = torch.tensor([1.5, 3.0, 6.0])
    ab = {2: 1.0,
          8: torch.tensor([0.2, 0.9, 1.6]),
          26: torch.tensor([1.4, 0.7, 0.3])}
    v = torch.tensor([80.0, 300.0, 550.0])
    nh = torch.tensor([1e20, 2e21, 7e21])
    kw = dict(absorption=Absorption.default(), n_h=nh, redshift=0.01)
    got = joint.batched.flux(T, ab, v, edges, contract=True, **kw)
    ref = joint.batched.flux(T, ab, v, edges, contract=False, **kw)
    assert _rel(got, ref) < 1e-5, _rel(got, ref)
    # positive control: the walkers must not be interchangeable
    assert _rel(got[0], got[1]) > 1e-2


def test_contracted_matches_serial(joint, edges):
    """Against the serial ``JointOperatorModel.flux`` -- a different
    implementation, not just a different summation order."""
    T = torch.tensor([2.0, 5.0])
    ab = {8: 0.7, 26: 1.1}
    got = joint.batched.flux(T, ab, 220.0, edges, contract=True)
    assert _rel(got, joint.flux(T, ab, 220.0, edges)) < 1e-4


def test_contracted_respects_zero_abundance(joint, edges):
    """The contracted path drops ``abundance_weight``'s skip shortcut, so a
    zero abundance now multiplies instead of eliding. It must still be zero."""
    T = torch.tensor([3.0])
    got = joint.batched.flux(T, {26: 0.0}, 150.0, edges, contract=True)
    ref = joint.batched.flux(T, {26: 0.0}, 150.0, edges, contract=False)
    assert _rel(got, ref) < 1e-5
    # and Fe really is switched off
    on = joint.batched.flux(T, {26: 1.0}, 150.0, edges, contract=True)
    assert _rel(got, on) > 1e-2


# --- DEM ---------------------------------------------------------------------

def test_dem_contracted_matches_stacked(obs, joint):
    th = _theta(5)
    got = build_posterior(obs, joint, _dem_params(), FIXED, dem=_dem(),
                          contract_first=True).forward(th)
    ref = build_posterior(obs, joint, _dem_params(), FIXED, dem=_dem(),
                          contract_first=False).forward(th)
    assert _rel(got, ref) < 1e-5, _rel(got, ref)


def test_dem_contracted_matches_stacked_gradients(obs, joint):
    grads = []
    for flag in (True, False):
        fwd = build_posterior(obs, joint, _dem_params(), FIXED, dem=_dem(),
                              contract_first=flag).forward
        th = torch.tensor(_theta(2), dtype=torch.float32, requires_grad=True)
        with fwd.emu.batched.grad_enabled(True):
            fwd.counts_torch(th, grad=True).sum().backward()
        grads.append(th.grad.detach().clone())
    assert (grads[0].abs() > 0).all()
    rel = float((grads[0] - grads[1]).abs().max() / grads[1].abs().max())
    assert rel < 1e-4, rel


def test_dem_contracted_keeps_walkers_distinct(obs, joint):
    """The DEM contraction is a GEMM over (B, N) abundances x (B, G) weights;
    a transposed or repeated axis silently hands every walker the first
    walker's DEM. Walkers with clearly different shapes must stay different,
    and each must equal what it gets when evaluated alone."""
    fwd = build_posterior(obs, joint, _dem_params(), FIXED,
                          dem=_dem(), contract_first=True).forward
    th = np.array([[2.0, 0.3, 120.0, 10.0],
                   [6.0, 1.8, 450.0, 10.0]])
    together = fwd(th)
    assert _rel(together[0], together[1]) > 1e-2
    apart = np.concatenate([fwd(th[i:i + 1]) for i in range(2)])
    assert _rel(together, apart) < 1e-5


def test_contracted_dem_skips_grouping_and_keeps_the_walker_budget(obs, joint):
    """Both exist only to survive B*G emulator rows, which the contraction
    removes; if they were left on, the walker budget would still be divided by
    G and the cheap path would run G times too few walkers at a time."""
    fwd = build_posterior(obs, joint, _dem_params(), FIXED, dem=_dem(),
                          contract_first=True).forward
    assert fwd.walker_chunk == fwd.chunk
    stacked = build_posterior(obs, joint, _dem_params(), FIXED, dem=_dem(),
                              contract_first=False).forward
    assert stacked.walker_chunk == max(1, stacked.chunk
                                       // _dem().temp_grid.numel())
