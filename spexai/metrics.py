"""Shared, reusable accuracy metrics for the operator emulator.

The mean-relative-error (MRE) machinery, the line/continuum bin split, the
per-point statistics of Matthijsse's thesis (Eq. 6.4) and the empty-bin
(floor) contract check were previously re-implemented across several
benchmark scripts. They live here once so every caller agrees bit-for-bit.

All functions operate on NumPy arrays of log10 flux, shape ``(n_spectra,
n_bins)`` (the offline benchmark convention). The training loop keeps its own
torch-native ``evaluate`` in :mod:`spexai.train.train_operator`; this module is
its numpy counterpart for held-out evaluation.
"""
from typing import Dict

import numpy as np

from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,
                                         continuum_estimate)


def abs_rel_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-bin absolute relative error in linear flux, ``|10^(pred-target)-1|``.

    pred, target: (n_spectra, n_bins) log10 flux. The target is clamped at the
    empty-bin floor and the exponent is clamped to +/-4 dex, matching the
    training-time ``evaluate`` and the benchmark scripts exactly.
    """
    d = np.clip(pred - np.clip(target, FLOOR, None), -4.0, 4.0)
    return np.abs(10.0 ** d - 1.0)


def line_continuum_masks(target: np.ndarray):
    """Split bins into (valid, line, continuum) boolean masks.

    A bin is *valid* (non-empty) when its true log flux exceeds ``FLOOR``; a
    valid bin is a *line* bin when it sits more than ``LINE_THRESHOLD_DEX``
    above the running-median continuum estimate, otherwise it is *continuum*.
    Matches ``scripts/benchmark_operator.py`` (clip before the comparison).
    """
    valid = target > FLOOR
    cont = continuum_estimate(target)
    line_mask = valid & (np.clip(target, FLOOR, None) - cont > LINE_THRESHOLD_DEX)
    cont_mask = valid & ~line_mask
    return valid, line_mask, cont_mask


def metrics_from_eps(eps: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Per-spectrum mean rel. error over ``mask`` bins -> summary dict.

    Also reports the per-POINT statistic of Matthijsse's thesis (Eq. 6.4):
    the percentage of masked points with relative error above 1e-3 / 1e-2.
    """
    cnt = mask.sum(axis=1)
    ok = cnt > 0
    mre = np.where(ok, (eps * mask).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    mre = mre[ok]
    eps_pts = eps[mask]
    if mre.size == 0:                       # no bins of this class anywhere
        nan = float("nan")
        return {"n_spectra": 0, "mre_mean": nan, "mre_median": nan,
                "yield_01pct": nan, "yield_1pct": nan, "yield_10pct": nan,
                "points_above_01pct": nan, "points_above_1pct": nan}
    return {
        "n_spectra": int(ok.sum()),
        "mre_mean": float(np.mean(mre)),
        "mre_median": float(np.median(mre)),
        "yield_01pct": float((mre <= 0.001).mean() * 100),
        "yield_1pct": float((mre <= 0.01).mean() * 100),
        "yield_10pct": float((mre <= 0.10).mean() * 100),
        "points_above_01pct": float((eps_pts > 1e-3).mean() * 100),
        "points_above_1pct": float((eps_pts > 1e-2).mean() * 100),
    }


def floor_violations(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Check the empty-bin contract on unseen data: wherever the true flux
    is at/below FLOOR (invisible with a real telescope), the emulator must
    also predict below FLOOR. These bins are excluded from the error
    metrics, so violations would otherwise go unnoticed."""
    floor_mask = target <= FLOOR
    n = int(floor_mask.sum())
    if n == 0:
        return {"n_floor_bins": 0, "violation_pct": 0.0,
                "violation_gt1dex_pct": 0.0, "max_excess_dex": 0.0,
                "spectra_with_violation_pct": 0.0}
    excess = np.where(floor_mask, pred - FLOOR, -np.inf)
    return {
        "n_floor_bins": n,
        "violation_pct": float((excess > 0).sum() / n * 100),
        "violation_gt1dex_pct": float((excess > 1).sum() / n * 100),
        "max_excess_dex": float(excess.max()),
        "spectra_with_violation_pct": float((excess > 0).any(axis=1).mean() * 100),
    }


def spectrum_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, dict]:
    """Full held-out evaluation for one model: overall / line / continuum MRE
    dicts plus the floor-violation report. ``pred``/``target`` are (n_spectra,
    n_bins) log10 flux on the same energy grid."""
    eps = abs_rel_error(pred, target)
    valid, line_mask, cont_mask = line_continuum_masks(target)
    return {
        "overall": metrics_from_eps(eps, valid),
        "lines": metrics_from_eps(eps, line_mask),
        "continuum": metrics_from_eps(eps, cont_mask),
        "floor": floor_violations(pred, target),
    }
