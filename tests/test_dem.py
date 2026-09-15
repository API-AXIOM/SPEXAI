"""Tests for the abundance interface and temperature-distribution (DEM) models."""
import numpy as np
import pytest
import torch

from spexai.inference.abundances import AbundanceModel
from spexai.inference import tempdist as td


# --- AbundanceModel ---------------------------------------------------------

def test_abundance_global_free_and_tie():
    ab = (AbundanceModel([1, 2, 8, 26])
          .global_metallicity("Z")
          .free_element(26, "Fe")
          .tie([8], "O_Fe", ref=26))
    assert ab.param_names == ["Z", "Fe", "O_Fe"]
    out = ab.to_abundances({"Z": 0.3, "Fe": 0.5, "O_Fe": 1.2})
    assert set(out) == {8, 26}                 # H/He excluded (held solar)
    assert out[26] == pytest.approx(0.5)       # free element, absolute
    assert out[8] == pytest.approx(0.6)        # 1.2 * Fe(0.5), tied to iron


def test_abundance_global_only_scales_all_metals():
    ab = AbundanceModel([2, 8, 14, 26]).global_metallicity("Z")
    out = ab.to_abundances({"Z": 0.4})
    assert out == {8: 0.4, 14: 0.4, 26: 0.4}


# --- DEM weights ------------------------------------------------------------

def test_gaussian_logt_weights_quadrature_and_peaked():
    """Parametric weights are pdf*dx, NOT renormalised: their sum is the
    fraction of the distribution the grid contains, so it should track the
    analytic CDF difference rather than being forced to 1."""
    from scipy.stats import norm as _norm
    grid = td.TempGrid(0.2, 10.0, n=80)
    dem = td.gaussian_logT(grid)
    mu, sig = np.log10(4.0), 0.1
    w = dem.weights({"logT_mean": mu, "logT_sigma": sig})
    d = _norm(mu, sig)
    contained = d.cdf(grid.logtemps[-1]) - d.cdf(grid.logtemps[0])
    assert float(w.sum()) == pytest.approx(contained, abs=1e-3)
    assert contained == pytest.approx(1.0, abs=1e-3)     # this one IS contained
    peak_T = float(grid.temp_grid[int(torch.argmax(w))])
    assert 3.0 < peak_T < 5.0                  # peak near 4 keV


def test_parametric_weights_not_renormalised_when_truncated():
    """A DEM running off the top of the grid must produce LESS emission
    measure, not the same amount redistributed."""
    from scipy.stats import norm as _norm
    grid = td.TempGrid(0.7, 10.0, n=48)        # campaign grid
    dem = td.gaussian_logT(grid)
    mu, sig = np.log10(8.0), 0.4               # hot and wide -> truncated
    w = dem.weights({"logT_mean": mu, "logT_sigma": sig})
    d = _norm(mu, sig)
    # dlogt is uniform at the endpoints too, so the quadrature is a MIDPOINT
    # rule over bins centred on the nodes: it covers half a cell beyond each
    # end, and that is the reference the sum should reproduce.
    h = 0.5 * float(grid.dlogt[0])
    contained = (d.cdf(grid.logtemps[-1] + h) - d.cdf(grid.logtemps[0] - h))
    assert contained < 0.8                     # genuinely truncated
    assert float(w.sum()) == pytest.approx(contained, abs=1e-3)
    assert float(w.sum()) < 0.95               # NOT forced back to 1


def test_gaussian_T_matches_thesis_params():
    grid = td.TempGrid(0.2, 10.0, n=120)
    dem = td.gaussian_T(grid)
    w = dem.weights({"T_mean": 4.5, "T_sigma": 0.794})
    # contained on this grid, so the un-renormalised quadrature is ~1
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-3)
    mean_T = float((grid.temp_grid * w).sum() / w.sum())
    assert mean_T == pytest.approx(4.5, abs=0.2)


def test_binned_dem_weights_normalised():
    dem = td.BinnedDEM(0.5, 8.0, n_bins=5)
    assert dem.param_names == ["dem0", "dem1", "dem2", "dem3", "dem4"]
    w = dem.weights({f"dem{i}": float(i + 1) for i in range(5)})
    np.testing.assert_allclose(w.numpy(), np.arange(1, 6) / 15.0, rtol=1e-6)


# --- DEM vs single-temperature consistency (needs models + response) --------

def test_predict_counts_dem_reduces_to_single_temp(fe_model, acis_response):
    # a one-node "distribution" at T must equal the single-temperature call
    T, norm = 4.0, 1e11
    single = fe_model.predict_counts(torch.tensor([T]), {}, -10.0, norm, 150.0,
                                     acis_response).squeeze(0).numpy()
    dem = fe_model.predict_counts_dem(torch.tensor([T]), torch.tensor([1.0]),
                                      {}, -10.0, norm, 150.0,
                                      acis_response).squeeze(0).numpy()
    np.testing.assert_allclose(dem, single, rtol=1e-5)


# --- batched weights (weights_batch) ----------------------------------------
# The scalar `weights` cannot serve a vectorised fit: it calls float() on each
# parameter and evaluates a scipy pdf, so it takes neither per-walker values nor
# gradients. `weights_batch` is the torch equivalent, and these check it against
# the scalar version row by row -- with DIFFERENT parameters per walker, which
# is the only way a broadcast/ordering bug shows up.

import pytest
import torch as _torch

from spexai.inference import tempdist as _td


def _grid():
    return _td.TempGrid(0.5, 8.0, n=12)


@pytest.mark.parametrize("factory", ["gaussian_logT", "gaussian_T"])
def test_weights_batch_matches_scalar(factory):
    dem = getattr(_td, factory)(_grid())
    names = dem.param_names
    rows = [{names[0]: 0.4, names[1]: 0.25},
            {names[0]: 0.6, names[1]: 0.10},
            {names[0]: 0.2, names[1]: 0.50}]
    if factory == "gaussian_T":
        rows = [{names[0]: 2.0, names[1]: 0.8},
                {names[0]: 4.0, names[1]: 1.5},
                {names[0]: 6.0, names[1]: 0.4}]
    batch = {n: _torch.tensor([float(r[n]) for r in rows]) for n in names}
    got = dem.weights_batch(batch)
    assert got.shape == (3, dem.temp_grid.numel())
    for i, r in enumerate(rows):
        ref = dem.weights(r)
        assert _torch.allclose(got[i], ref, atol=1e-6), f"walker {i}"


def test_weights_batch_rows_are_not_renormalised():
    """Batched parametric weights must carry the same contained-fraction
    semantics as the scalar path -- a truncated row stays short."""
    dem = _td.gaussian_logT(_grid())
    n = dem.param_names
    w = dem.weights_batch({n[0]: _torch.tensor([0.3, 0.7]),
                           n[1]: _torch.tensor([0.2, 0.4])})
    sums = w.sum(-1)
    assert float(sums[0]) == pytest.approx(1.0, abs=5e-3)   # contained
    assert float(sums[1]) < 0.8                             # truncated, kept short
    for i, (m, sg) in enumerate(((0.3, 0.2), (0.7, 0.4))):
        ref = dem.weights({n[0]: m, n[1]: sg})
        assert _torch.allclose(w[i], ref, atol=1e-6), f"row {i}"


def test_two_gaussian_weights_batch_matches_scalar():
    dem = _td.TwoGaussianDEM(_grid())
    rows = [{"logT1": 0.2, "sig1": 0.15, "logT2": 0.7, "sig2": 0.3, "frac": 0.3},
            {"logT1": 0.5, "sig1": 0.30, "logT2": 0.1, "sig2": 0.1, "frac": 0.8}]
    batch = {n: _torch.tensor([float(r[n]) for r in rows])
             for n in dem.param_names}
    got = dem.weights_batch(batch)
    for i, r in enumerate(rows):
        assert _torch.allclose(got[i], dem.weights(r), atol=1e-6), f"walker {i}"


def test_two_gaussian_weights_not_renormalised_when_truncated():
    """Same semantics as ``ParametricDEM``: the sum is the contained fraction.
    A component running off the top of the grid must LOSE emission measure,
    in both the scalar and the batched path, not have it redistributed."""
    from scipy.stats import norm as _norm
    grid = td.TempGrid(0.7, td.EMULATOR_T_HI_KEV, n=70)
    dem = td.TwoGaussianDEM(grid)
    p = {"logT1": np.log10(2.0), "sig1": 0.1,
         "logT2": np.log10(18.0), "sig2": 0.3, "frac": 0.5}
    # midpoint quadrature: half a cell beyond each end node
    h = 0.5 * float(grid.dlogt[0])
    lo, hi = grid.logtemps[0] - h, grid.logtemps[-1] + h

    def inside(m, s):
        d = _norm(m, s)
        return d.cdf(hi) - d.cdf(lo)

    contained = (p["frac"] * inside(p["logT1"], p["sig1"])
                 + (1 - p["frac"]) * inside(p["logT2"], p["sig2"]))
    assert contained < 0.85                    # genuinely truncated
    w = dem.weights(p)
    assert float(w.sum()) == pytest.approx(contained, abs=2e-3)
    # float32, as VectorForward passes theta
    wb = dem.weights_batch({n: torch.tensor([float(p[n])], dtype=torch.float32)
                            for n in dem.param_names})
    assert torch.allclose(wb[0], w, atol=1e-6)


def test_gaussian_logt_default_bounds_are_the_campaign_design():
    b = td.gaussian_logT(td.TempGrid(0.7, td.EMULATOR_T_HI_KEV, n=70)
                         ).suggested_bounds()
    assert b["logT_mean"] == pytest.approx((np.log10(0.7), np.log10(15.0)))
    assert b["logT_sigma"] == pytest.approx((0.056, 0.4))


def test_binned_weights_batch_matches_scalar():
    dem = _td.BinnedDEM(0.5, 8.0, n_bins=5)
    rows = [{f"dem{i}": v for i, v in enumerate([0.1, 0.4, 0.2, 0.0, 0.3])},
            {f"dem{i}": v for i, v in enumerate([0.5, 0.0, 0.0, 0.25, 0.25])}]
    batch = {n: _torch.tensor([float(r[n]) for r in rows])
             for n in dem.param_names}
    got = dem.weights_batch(batch)
    for i, r in enumerate(rows):
        assert _torch.allclose(got[i], dem.weights(r), atol=1e-6), f"walker {i}"


def test_weights_batch_is_differentiable():
    # NUTS and VI need d(weights)/d(DEM params); the scalar scipy path gives 0
    dem = _td.gaussian_logT(_grid())
    n = dem.param_names
    mean = _torch.tensor([0.4], requires_grad=True)
    sig = _torch.tensor([0.25], requires_grad=True)
    w = dem.weights_batch({n[0]: mean, n[1]: sig})
    (w * _torch.linspace(1.0, 2.0, w.shape[-1])).sum().backward()
    for name, p in ((n[0], mean), (n[1], sig)):
        assert p.grad is not None and _torch.isfinite(p.grad).all()
        assert float(p.grad.abs().max()) > 0, f"{name} gradient is zero"


def test_scipy_only_dem_refuses_to_batch():
    # a custom scipy shape has no torch equivalent and must say so rather than
    # silently producing something wrong
    from scipy.stats import gamma
    dem = _td.ParametricDEM(_grid(), lambda v: gamma(a=v[0], scale=v[1]),
                            ["a", "scale"], variable="T")
    assert not hasattr(dem, "weights_batch") or True
    with pytest.raises(NotImplementedError, match="no torch equivalent"):
        dem.weights_batch({"a": _torch.tensor([2.0]),
                           "scale": _torch.tensor([1.0])})
