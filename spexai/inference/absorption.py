"""Galactic photoelectric absorption as a multiplicative transmission screen.

Absorption multiplies the source flux by ``T(E) = exp(-N_H * sigma(E))`` in the
**observed frame**. The only physics is the cross-section per hydrogen atom,
``sigma(E)``; we do not invent it:

* ``Absorption.tbabs()`` loads a cached ``sigma(E)`` table tabulated once from
  XSPEC/sherpa's ``tbabs`` (Wilms, Allen & McCray 2000) -- the community
  standard for cluster work. Runtime never calls XSPEC; it reads the cached
  tensor. See ``scripts/build_tbabs_table.py``.
* ``Absorption.wabs()`` is a dependency-free fallback: the Morrison & McCammon
  (1983) analytic polynomial cross-section, validated against its published
  coefficients.

Applied on the native fine grid **before** rebinning (see
``operator_model.element_broadened_flux``), so it is instrument-resolution
independent and needs no operator retraining.
"""
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_TBABS_PATH = os.path.join(DATA_DIR, "tbabs_sigma.npz")

# Morrison & McCammon (1983) band edges (keV) and polynomial coefficients:
# sigma(E) = (c0 + c1 E + c2 E^2) * 1e-24 / E^3  cm^2 per H atom.
_MM_EDGES = np.array([0.03, 0.10, 0.284, 0.400, 0.532, 0.707, 0.867, 1.303,
                      1.840, 2.471, 3.210, 4.038, 7.111, 8.331, 10.0])
_MM_C = np.array([
    [17.3, 608.1, -2150.0], [34.6, 267.9, -476.1], [78.1, 18.8, 4.3],
    [71.4, 66.8, -51.4], [95.5, 145.8, -61.1], [308.9, -380.6, 294.0],
    [120.6, 169.3, -47.7], [141.3, 146.8, -31.5], [202.7, 104.7, -17.0],
    [342.7, 18.7, 0.0], [352.2, 18.7, 0.0], [433.9, -2.4, 0.75],
    [629.0, 30.9, 0.0], [701.2, 25.2, 0.0]])


def wabs_sigma(energy_kev) -> np.ndarray:
    """Morrison & McCammon (1983) cross-section (cm^2 per H) at ``energy_kev``.

    Zero outside 0.03-10 keV (absorption is negligible above ~10 keV; the
    emulator band starts near 0.1 keV)."""
    e = np.asarray(energy_kev, dtype=np.float64)
    idx = np.clip(np.searchsorted(_MM_EDGES, e, side="right") - 1,
                  0, len(_MM_C) - 1)
    c = _MM_C[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = (c[..., 0] + c[..., 1] * e + c[..., 2] * e * e) * 1e-24 / e ** 3
    return np.where((e >= _MM_EDGES[0]) & (e <= _MM_EDGES[-1]), sig, 0.0)


class Absorption:
    """A ``sigma(E)`` cross-section table + the transmission it implies.

    ``energy_ref`` (K,) keV ascending, ``sigma_ref`` (K,) cm^2 per H. Transmission
    is interpolated in log-energy (sigma is smooth between edges)."""

    def __init__(self, energy_ref, sigma_ref, name: str = ""):
        self.energy_ref = np.asarray(energy_ref, dtype=np.float64)
        self.sigma_ref = np.asarray(sigma_ref, dtype=np.float64)
        self.log_e_ref = np.log(self.energy_ref)
        self.name = name

    @classmethod
    def wabs(cls, n_grid: int = 4096) -> "Absorption":
        e = np.logspace(np.log10(0.03), np.log10(12.0), n_grid)
        return cls(e, wabs_sigma(e), name="wabs")

    @classmethod
    def default(cls) -> "Absorption":
        """The best available screen: cached ``tbabs`` if present, else ``wabs``."""
        try:
            return cls.tbabs()
        except FileNotFoundError:
            return cls.wabs()

    @classmethod
    def tbabs(cls, path: str = DEFAULT_TBABS_PATH) -> "Absorption":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"tbabs table {path} not found; generate it with "
                f"scripts/build_tbabs_table.py (needs sherpa/xspec), or use "
                f"Absorption.wabs().")
        z = np.load(path)
        return cls(z["energy"], z["sigma"], name="tbabs")

    def sigma(self, energy_kev) -> np.ndarray:
        """Interpolated cross-section (cm^2 per H) at ``energy_kev``."""
        e = np.asarray(energy_kev, dtype=np.float64)
        return np.interp(np.log(np.clip(e, 1e-6, None)), self.log_e_ref,
                         self.sigma_ref, left=self.sigma_ref[0], right=0.0)

    def transmission(self, energy_kev, n_h, device=None) -> torch.Tensor:
        """``exp(-N_H * sigma(E))`` as a float32 tensor. ``n_h`` in cm^-2."""
        t = np.exp(-float(n_h) * self.sigma(energy_kev))
        return torch.as_tensor(t, dtype=torch.float32, device=device)

    def transmission_torch(self, energy, n_h, device=None) -> torch.Tensor:
        """GPU/torch-native ``exp(-N_H * sigma(E))`` for the batched forward.

        Same physics as :meth:`transmission` (sigma linearly interpolated in
        log-energy, held at ``sigma_ref[0]`` below the table and 0 above) but
        stays on-device, and accepts ``n_h`` as a scalar (-> ``(K,)``) or a
        ``(B,)`` tensor of per-walker columns (-> ``(B, K)``)."""
        e = torch.as_tensor(energy, dtype=torch.float32, device=device)
        dev = e.device
        loge = torch.log(e.clamp(min=1e-6))
        xp = torch.as_tensor(self.log_e_ref, dtype=torch.float32, device=dev)
        fp = torch.as_tensor(self.sigma_ref, dtype=torch.float32, device=dev)
        j = torch.searchsorted(xp, loge).clamp(1, xp.numel() - 1)
        w = ((loge - xp[j - 1]) / (xp[j] - xp[j - 1])).clamp(0.0, 1.0)
        sig = fp[j - 1] + w * (fp[j] - fp[j - 1])
        sig = torch.where(loge <= xp[0], fp[0], sig)
        sig = torch.where(loge >= xp[-1], torch.zeros_like(sig), sig)
        n = torch.as_tensor(n_h, dtype=torch.float32, device=dev)
        return torch.exp(-(n[:, None] if n.ndim else n) * sig)
