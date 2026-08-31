"""Independent SPEX ground truth via per-element PCHIP interpolation.

``SpexTruthModel`` mirrors :class:`~spexai.inference.operator_model.
JointOperatorModel`'s API so it is a drop-in generator (for ``simulate.py``)
and injector (for the bias study), but is **independent of the trained
emulator**. Per element it PCHIP-interpolates the preprocessed SPEX training
spectra in log10 T -- the exact classical baseline of
``scripts/baseline_interpolation.py`` -- sums elements linearly by abundance,
broadens with the **exact** ``direct_broadening_matrix`` on the native SPEX
grid, and folds through the instrument :class:`~spexai.inference.response.
Response`.

This is a documented approximation to a live SPEX call: the cache has no
abundance dimension (abundances enter only through the linear per-element sum,
which is exact for CIE emissivities), broadening is at the native SPEX
resolution, and off-grid temperatures come from PCHIP rather than SPEX itself.
Galactic absorption is threaded in by Workstream G via the shared screen so the
same operator multiplies truth and emulator.
"""
import json
import os

import numpy as np
import torch

from spexai.operator import edges_from_centers
from spexai.broadening import direct_broaden, rebin_flux
from spexai.data import SpectrumData, pchip_generate
from spexai.inference.operator_model import MODELS_DIR
from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2, distance_factor


class ElementTruth:
    """One element's PCHIP truth on the native SPEX grid (train rows only).

    Independent of any trained model: it needs only the sorted training-row
    log10 temperatures ``lt_grid`` (n_train,), their log10 flux **density**
    ``Y`` (n_train, n_bins), and the native energy-bin ``centers`` (n_bins,).
    """

    def __init__(self, lt_grid, Y, centers, half: int = 4):
        order = np.argsort(np.asarray(lt_grid, dtype=np.float64))
        self.lt_grid = np.asarray(lt_grid, dtype=np.float64)[order]
        self.Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float32)[order])
        self.half = int(half)
        cen = torch.as_tensor(np.asarray(centers), dtype=torch.float32)
        self.edges = edges_from_centers(cen)                  # (n_bins+1,)
        self.widths = self.edges[1:] - self.edges[:-1]        # (n_bins,)

    @classmethod
    def from_cache(cls, cachedir: str, half: int = 4) -> "ElementTruth":
        data = SpectrumData(cachedir)
        tr = data.train_idx.numpy()                           # train rows only
        lt = np.log10(data.temps.numpy()[tr])
        Y = data.logflux[tr].numpy()
        return cls(lt, Y, data.energy.numpy(), half=half)

    def native_flux(self, temp_kev) -> torch.Tensor:
        """Integrated flux per native bin at solar abundance, (B, n_bins)."""
        t = np.atleast_1d(np.asarray(temp_kev, dtype=np.float64))
        # (B, n_bins) log10 flux density via per-bin monotone-cubic in log10 T
        dens = pchip_generate(self.lt_grid, self.Y, np.log10(t), half=self.half)
        return torch.pow(10.0, torch.from_numpy(dens)) * self.widths


class SpexTruthModel:
    """All available per-element PCHIP truths combined into one emitter.

    Same ``flux`` / ``predict_counts`` signatures as ``JointOperatorModel``.
    Loads the elements listed in the model-store manifest (so the truth and the
    emulator cover the same set) from the preprocessed caches in ``datadir``.

    Note: holds each element's training-row spectra in memory (log10 flux, ~0.4
    GB/element). Use ``elements=`` to restrict on a laptop.
    """

    def __init__(self, models_dir: str = MODELS_DIR, datadir: str = None,
                 elements=None, device: str = "cpu", half: int = 4):
        self.device = device
        with open(os.path.join(models_dir, "manifest.json")) as f:
            manifest = json.load(f)
        if datadir is None:
            from spexai.eval import _default_datadir
            datadir = _default_datadir(models_dir, manifest)
        self.truths = {}
        for zstr in manifest["elements"]:
            z = int(zstr)
            if elements is not None and z not in elements:
                continue
            cache = os.path.join(datadir, f"element{z}")
            if os.path.isdir(cache):
                self.truths[z] = ElementTruth.from_cache(cache, half=half)
        if not self.truths:
            raise FileNotFoundError(
                f"no element caches found under {datadir}")
        self.elements = sorted(self.truths)
        self.fixed_solar = {1, 2}                    # H, He held solar
        self.native_edges = next(iter(self.truths.values())).edges.to(device)

    def __repr__(self):
        return (f"SpexTruthModel({len(self.truths)} elements {self.elements}, "
                f"device={self.device})")

    @torch.no_grad()
    def flux(self, temp_kev, abundances, velocity, bin_edges,
             absorption=None, n_h=0.0, redshift=0.0) -> torch.Tensor:
        """Abundance-weighted, exact-broadened integrated flux on ``bin_edges``.

        temp_kev: (B,) tensor. abundances: {Z: solar-relative}; missing loaded
        elements default to 1.0. velocity: km/s. Optional Galactic absorption
        (``absorption``, ``n_h`` cm^-2) is applied on the native grid after
        broadening, in the observed frame (``redshift`` = source z). Returns
        (B, M)."""
        temp_kev = torch.as_tensor(temp_kev, dtype=torch.float32).view(-1)
        bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32,
                                    device=self.device)
        tnp = temp_kev.cpu().numpy()
        total = None
        for z, el in self.truths.items():
            a = float(abundances.get(z, 1.0)) if abundances else 1.0
            if a == 0.0:
                continue
            f = el.native_flux(tnp).to(self.device)           # (B, n_bins)
            total = a * f if total is None else total + a * f
        if total is None:
            return torch.zeros((temp_kev.numel(), bin_edges.numel() - 1),
                               device=self.device)
        if velocity and float(velocity) > 0.0:
            total = direct_broaden(total, self.native_edges, float(velocity))
        if absorption is not None and float(n_h) > 0.0:
            native_cent = torch.sqrt(self.native_edges[:-1] * self.native_edges[1:])
            total = total * absorption.transmission(
                native_cent / (1.0 + redshift), n_h, device=self.device)
        return rebin_flux(total, self.native_edges, bin_edges)   # (B, M)

    @torch.no_grad()
    def predict_counts(self, temp_kev, abundances, logz, norm, velocity,
                       response, exposure=1.0, luminosity_distance=D_REF_M,
                       absorption=None, n_h=0.0):
        """Predicted detector-channel counts (mirrors JointOperatorModel).

        ``norm`` = emission measure Y (1e64 m^-3); ``luminosity_distance`` in
        metres. Redshifts the response grid by (1+z), applies optional Galactic
        absorption, folds the exact-broadened flux through ARF+RMF, and scales
        to physical counts. Returns (B, N_channels)."""
        z = 10.0 ** float(logz)
        edges_rest = response.energy_edges * (1.0 + z)
        flux = self.flux(temp_kev, abundances, velocity, edges_rest,
                         absorption=absorption, n_h=n_h, redshift=z)
        rate = response.fold(flux).to(self.device)
        scale = (float(norm) * float(exposure)
                 * distance_factor(luminosity_distance) * FLUX_M2_TO_CM2)
        return scale * rate

    @torch.no_grad()
    def predict_counts_dem(self, temp_grid, weights, abundances, logz, norm,
                           velocity, response, exposure=1.0,
                           luminosity_distance=D_REF_M, absorption=None,
                           n_h=0.0):
        """DEM counts (mirrors JointOperatorModel.predict_counts_dem)."""
        temp_grid = torch.as_tensor(temp_grid, dtype=torch.float32).view(-1)
        weights = torch.as_tensor(weights, dtype=torch.float32,
                                  device=self.device).view(-1, 1)
        z = 10.0 ** float(logz)
        edges_rest = response.energy_edges * (1.0 + z)
        flux = self.flux(temp_grid, abundances, velocity, edges_rest,
                         absorption=absorption, n_h=n_h, redshift=z)  # (G, M)
        dem_flux = (weights * flux).sum(dim=0, keepdim=True)
        rate = response.fold(dem_flux).to(self.device)
        scale = (float(norm) * float(exposure)
                 * distance_factor(luminosity_distance) * FLUX_M2_TO_CM2)
        return scale * rate
