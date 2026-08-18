"""Diagnose the NaN in the DEM truth: which grid temperatures / elements blow up.

Prints the Gaussian-DEM grid range, each probed element's valid training-temp
range, and where native_flux is non-finite.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

from experiment import STORE, DATADIR, gaussian_dem                 # noqa: E402
from spexai.inference.spex_truth import ElementTruth                  # noqa: E402

dem, _ = gaussian_dem()
grid = dem.temp_grid.numpy()
print(f"DEM grid: {grid.min():.4f}..{grid.max():.3f} keV, n={grid.size}")

manifest = json.load(open(os.path.join(STORE, "manifest.json")))
for zstr in sorted(manifest["elements"], key=int):
    z = int(zstr)
    cache = os.path.join(DATADIR, f"element{z}")
    if not os.path.isdir(cache):
        continue
    el = ElementTruth.from_cache(cache)
    tmin, tmax = 10 ** el.lt_grid.min(), 10 ** el.lt_grid.max()
    f = el.native_flux(grid).numpy()                                  # (n_grid, n_bins)
    bad = ~np.isfinite(f).all(axis=1)
    fmax = np.nanmax(np.abs(f))
    flag = f"  <-- non-finite at T={grid[bad]}" if bad.any() else ""
    print(f"Z{z:02d}: train T {tmin:.4f}..{tmax:.3f}; max|f|={fmax:.2e}{flag}")
