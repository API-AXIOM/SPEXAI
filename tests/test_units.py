"""Tests for the physical normalisation (emission measure Y + distance)."""
import numpy as np
import pytest
import torch

from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2, distance_factor


def test_distance_factor():
    assert distance_factor(D_REF_M) == pytest.approx(1.0)          # reference
    assert distance_factor(2 * D_REF_M) == pytest.approx(0.25)     # inverse square
    assert distance_factor(D_REF_M / 10) == pytest.approx(100.0)


def test_flux_area_unit_constant():
    # SPEX flux is per m^2; OGIP ARF per cm^2 -> 1 m^2 = 1e4 cm^2
    assert FLUX_M2_TO_CM2 == pytest.approx(1e-4)


def test_predict_counts_scales_with_Y_and_distance(fe_model, acis_response):
    kw = dict(velocity=150.0)
    base = fe_model.predict_counts(torch.tensor([4.0]), {}, -10.0, 1.0,
                                   150.0, acis_response).squeeze(0).numpy()
    # linear in emission measure Y (= norm)
    twoY = fe_model.predict_counts(torch.tensor([4.0]), {}, -10.0, 2.0,
                                   150.0, acis_response).squeeze(0).numpy()
    np.testing.assert_allclose(twoY, 2.0 * base, rtol=1e-5)
    # inverse-square in luminosity distance
    far = fe_model.predict_counts(torch.tensor([4.0]), {}, -10.0, 1.0, 150.0,
                                  acis_response,
                                  luminosity_distance=2 * D_REF_M).squeeze(0).numpy()
    np.testing.assert_allclose(far, 0.25 * base, rtol=1e-5)
