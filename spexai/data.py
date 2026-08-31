"""Shared data plumbing used by both training and inference.

Neither ``SpectrumData`` (the in-memory preprocessed-cache loader) nor
``pchip_generate`` (per-bin PCHIP interpolation on a local stencil) is
training logic -- they're consumed by ~10 evaluation/inference scripts too.
Moved out of ``spexai.train.train_operator``/``train_adaptive`` (2026-08-21
restructure) so importing them no longer drags in a training driver.
"""
import os

import numpy as np
import torch


class SpectrumData:
    """Holds the preprocessed cache in memory (float32)."""

    def __init__(self, cachedir):
        self.energy = torch.from_numpy(np.load(os.path.join(cachedir, "energy.npy")))
        self.temps = torch.from_numpy(np.load(os.path.join(cachedir, "temps.npy")))
        self.logflux = torch.from_numpy(
            np.load(os.path.join(cachedir, "logflux.npy"), mmap_mode=None))
        splits = np.load(os.path.join(cachedir, "splits.npz"))
        self.train_idx = torch.from_numpy(splits["train"]).long()
        self.val_idx = torch.from_numpy(splits["val"]).long()
        self.test_idx = torch.from_numpy(splits["test"]).long()
        self.n_bins = self.logflux.shape[1]


def pchip_generate(lt_grid, Y, lt_new, half=4):
    """Per-bin PCHIP log-flux spectra at new log-temperatures, fit on a
    local stencil of the training grid (train rows only, never val/test)."""
    from scipy.interpolate import PchipInterpolator
    out = np.empty((len(lt_new), Y.shape[1]), dtype=np.float32)
    for i, lt in enumerate(lt_new):
        j = int(np.searchsorted(lt_grid, lt))
        lo, hi = max(0, j - half), min(len(lt_grid), j + half)
        out[i] = PchipInterpolator(lt_grid[lo:hi], Y[lo:hi],
                                   axis=0)(lt).astype(np.float32)
    return out
