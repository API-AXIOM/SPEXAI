"""Tests for the Galactic absorption screen."""
import numpy as np
import pytest
import torch

from spexai.inference.absorption import Absorption, wabs_sigma


def test_wabs_sigma_matches_published_coefficients():
    # Morrison & McCammon (1983): sigma(1 keV) = (120.6 + 169.3 - 47.7)e-24 cm^2
    assert wabs_sigma(1.0) == pytest.approx(242.2e-24, rel=1e-6)
    # negligible above 10 keV (outside the fitted bands)
    assert wabs_sigma(11.0) == 0.0


def test_transmission_bounds_and_monotonicity():
    ab = Absorption.wabs()
    E = np.array([0.3, 0.5, 1.0, 2.0, 6.0])
    t0 = ab.transmission(E, 0.0).numpy()
    t1 = ab.transmission(E, 1e21).numpy()
    t2 = ab.transmission(E, 5e21).numpy()
    assert np.allclose(t0, 1.0)                 # no column -> full transmission
    assert np.all((t1 > 0) & (t1 <= 1.0))
    assert np.all(t2 <= t1 + 1e-12)             # more column -> more absorption
    # soft X-rays absorbed far more than hard
    assert t1[0] < t1[-1]


def test_transmission_softer_energies_more_absorbed():
    ab = Absorption.wabs()
    t = ab.transmission(np.array([0.3, 1.0, 6.0]), 1e21).numpy()
    assert t[0] < t[1] < t[2]


def test_tbabs_table_if_present():
    import os
    from spexai.inference.absorption import DEFAULT_TBABS_PATH
    if not os.path.exists(DEFAULT_TBABS_PATH):
        pytest.skip("tbabs table not built (needs HEASoft; see build_tbabs_table.py)")
    tb = Absorption.tbabs()
    assert tb.name == "tbabs"
    E = np.array([0.3, 1.0, 6.0])
    assert np.allclose(tb.transmission(E, 0.0).numpy(), 1.0)
    t1 = tb.transmission(E, 1e21).numpy()
    assert np.all((t1 > 0) & (t1 <= 1.0)) and t1[0] < t1[1] < t1[2]
    # tbabs (Wilms abundances) differs from the wabs fallback
    assert abs(tb.sigma(1.0) - Absorption.wabs().sigma(1.0)) > 1e-24
    # Absorption.default() prefers the table when present
    assert Absorption.default().name == "tbabs"


def test_absorption_reduces_soft_counts(fe_model, acis_response):
    T, norm = 4.0, 1e11
    ab = Absorption.wabs()
    base = fe_model.predict_counts(torch.tensor([T]), {}, -10.0, norm, 150.0,
                                   acis_response).squeeze(0).numpy()
    absd = fe_model.predict_counts(torch.tensor([T]), {}, -10.0, norm, 150.0,
                                   acis_response, absorption=ab,
                                   n_h=3e21).squeeze(0).numpy()
    # n_h=0 must be a no-op vs no absorption argument at all
    noop = fe_model.predict_counts(torch.tensor([T]), {}, -10.0, norm, 150.0,
                                   acis_response, absorption=ab,
                                   n_h=0.0).squeeze(0).numpy()
    np.testing.assert_allclose(noop, base, rtol=1e-6)
    # absorbed spectrum has strictly fewer total counts, driven by the soft band
    cen = acis_response.chan_e_cent.numpy()
    soft = cen < 1.0
    assert absd.sum() < base.sum()
    assert absd[soft].sum() < 0.99 * base[soft].sum()
