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

def test_gaussian_logt_weights_normalised_and_peaked():
    grid = td.TempGrid(0.2, 10.0, n=80)
    dem = td.gaussian_logT(grid)
    w = dem.weights({"logT_mean": np.log10(4.0), "logT_sigma": 0.1})
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)
    peak_T = float(grid.temp_grid[int(torch.argmax(w))])
    assert 3.0 < peak_T < 5.0                  # peak near 4 keV


def test_gaussian_T_matches_thesis_params():
    grid = td.TempGrid(0.2, 10.0, n=120)
    dem = td.gaussian_T(grid)
    w = dem.weights({"T_mean": 4.5, "T_sigma": 0.794})
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)
    mean_T = float((grid.temp_grid * w).sum())
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
