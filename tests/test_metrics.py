"""Unit tests for the shared held-out metrics (spexai.train.metrics)."""
import numpy as np
import pytest

from spexai.train.metrics import (FLOOR, abs_rel_error, floor_violations,
                                   line_continuum_masks, metrics_from_eps,
                                   spectrum_metrics)


def _synthetic_targets(n_bins=600):
    """One spectrum: flat continuum at -3, two bright line bins at 0, and a
    block of empty (floor) bins at -12."""
    t = np.full((1, n_bins), -3.0, dtype=np.float32)
    t[0, 100] = 0.0            # line bin (>log10 2 above continuum)
    t[0, 300] = 0.0            # line bin
    t[0, 400:450] = -12.0      # empty (below FLOOR)
    return t


def test_abs_rel_error_zero_on_valid_identity():
    # Identity is exact on valid bins; floor bins are intentionally nonzero
    # (target is clamped to FLOOR, pred is not) because they are masked out.
    t = _synthetic_targets()
    eps = abs_rel_error(t, t)
    valid = t > FLOOR
    assert np.allclose(eps[valid], 0.0)


def test_abs_rel_error_known_offset():
    t = _synthetic_targets()
    pred = t + 0.1                       # +0.1 dex everywhere
    eps = abs_rel_error(pred, t)
    # valid bins: |10^0.1 - 1| (float32 tolerance)
    assert eps[0, 0] == pytest.approx(10.0 ** 0.1 - 1.0, rel=1e-4)


def test_masks_classify_lines_and_floor():
    t = _synthetic_targets()
    valid, line_mask, cont_mask = line_continuum_masks(t)
    assert line_mask[0, 100] and line_mask[0, 300]
    assert not valid[0, 425]                     # floor bin is not valid
    assert cont_mask[0, 200]                     # plain continuum bin
    # a bin is exactly one of {line, continuum, empty}
    assert np.all(valid == (line_mask | cont_mask))


def test_spectrum_metrics_identity_is_perfect():
    t = _synthetic_targets()
    m = spectrum_metrics(t, t)
    assert m["overall"]["mre_mean"] == pytest.approx(0.0)
    assert m["overall"]["yield_1pct"] == pytest.approx(100.0)
    assert m["overall"]["yield_01pct"] == pytest.approx(100.0)
    assert m["floor"]["violation_pct"] == pytest.approx(0.0)


def test_spectrum_metrics_offset_degrades():
    t = _synthetic_targets()
    pred = t + 0.1
    m = spectrum_metrics(pred, t)
    assert m["overall"]["mre_mean"] == pytest.approx(10.0 ** 0.1 - 1.0, rel=1e-4)
    assert m["overall"]["yield_1pct"] == pytest.approx(0.0)


def test_floor_violation_detected():
    t = _synthetic_targets()
    pred = t.copy()
    pred[0, 425] = -5.0                          # empty bin predicted well above FLOOR
    fv = floor_violations(pred, t)
    assert fv["n_floor_bins"] == 50
    assert fv["violation_pct"] > 0.0
    assert fv["max_excess_dex"] == pytest.approx(FLOOR * -1 - 5.0)  # -5 - (-10) = 5


def test_metrics_from_eps_empty_mask_is_safe():
    eps = np.zeros((1, 10))
    mask = np.zeros((1, 10), dtype=bool)
    out = metrics_from_eps(eps, mask)
    assert out["n_spectra"] == 0
