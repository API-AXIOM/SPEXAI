"""Walker-batched, all-GPU forward for the hot-floor MCMC (Tier 1: sigma_v
fixed).

emcee with ``vectorized=True`` hands the log-prob the whole walker ensemble at
once; ``EnsembleForward`` turns that ``(B, ndim)`` block into ``(B, n_keep)``
in-band counts in a single batched pass on ``device`` -- one GPU process, no CPU
pool. The batch dimension is the walkers: per-walker temperature (native to the
operator), per-walker abundances (weight applied outside the element flux),
per-walker N_H (``transmission_torch`` broadcasts on-device), and the response
fold as a ``torch.sparse`` CSR mat-mul on the GPU. Velocity is held fixed so the
scalar-velocity line deposition is reused unchanged; freeing sigma_v needs the
per-walker line-broadening vectorisation (the required Tier-2 step).

Correctness is checked against the trusted serial forward
(``fisher_bias.Forward`` -> ``JointOperatorModel.predict_counts``) on CPU:
``python inference_demo/hot_floor/gpu_forward.py`` (device cpu).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import PERSEUS, STORE28, find_xrism_response, band_mask  # noqa: E402
from fisher_bias import FREE_Z, SYMBOL, build_params, Forward             # noqa: E402
from spexai.inference.operator_model import (                             # noqa: E402
    JointOperatorModel, element_broadened_flux)
from spexai.inference.response import Response                            # noqa: E402
from spexai.inference.absorption import Absorption                        # noqa: E402
from spexai.inference.units import distance_factor, FLUX_M2_TO_CM2        # noqa: E402


class EnsembleForward:
    """theta ``(B, ndim)`` -> in-band counts ``(B, n_keep)`` on ``device``.

    ``velocity`` (km/s) is fixed (Tier 1); the sampled parameters are the free
    abundances, the thermal parameter(s), ``n_h`` (in 1e21 cm^-2) and
    ``log_norm``.
    """

    def __init__(self, emu, response, absorption, keep, mode, velocity, device,
                 chunk=32):
        if mode != "single":
            raise NotImplementedError("EnsembleForward: single-T only for now")
        self.emu, self.absn, self.device = emu, absorption, device
        self.mode = mode
        self.velocity = float(velocity)
        # walkers are processed in sub-batches of `chunk` so peak GPU memory is
        # bounded (the forward's embedding/FFT grids scale with the batch); the
        # emcee ensemble size is then independent of GPU memory.
        self.chunk = int(chunk)
        self.z = 10.0 ** float(np.log10(PERSEUS["z"]))
        self.abnames = [SYMBOL[z] for z in FREE_Z]
        self.names = self.abnames + ["kT", "n_h", "log_norm"]
        self.col = {n: i for i, n in enumerate(self.names)}
        self.fe_idx = self.col["Fe"]
        # GPU sparse fold: R^T (C, N) so counts = (R^T @ eff^T)^T = eff @ R.
        Rt = response.R.T.tocsr()
        self.Rt = torch.sparse_csr_tensor(
            torch.from_numpy(Rt.indptr.astype(np.int64)),
            torch.from_numpy(Rt.indices.astype(np.int64)),
            torch.from_numpy(Rt.data.astype(np.float32)),
            size=Rt.shape, device=device)
        self.arf = torch.as_tensor(response.arf, dtype=torch.float32, device=device)
        self.edges_rest = (response.energy_edges * (1.0 + self.z)).to(device)
        self.keep_idx = torch.as_tensor(np.where(keep)[0], device=device)
        self.scale_const = distance_factor(PERSEUS["dist_m"]) * FLUX_M2_TO_CM2

    def params(self, log_norm_truth):
        """Par list (truth/bounds/step) for ``self.names`` -- build_params minus
        the (now fixed) sigma_v."""
        return [p for p in build_params(self, log_norm_truth)
                if p.name != "sigma_v"]

    def _flux(self, theta):
        """Abundance-weighted, broadened flux on the response grid -> (B, M).

        The emissivity/broadening stage (per-element operator nets, FFT
        continuum broadening, on-device absorption, rebin, analytic lines) --
        i.e. everything except the fold. Split out so it can be timed alone."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))   # (B, ndim)
        B, dev = th.shape[0], self.device
        temps = torch.as_tensor(th[:, self.col["kT"]], dtype=torch.float32,
                                device=dev)                        # (B,)
        n_h = torch.as_tensor(th[:, self.col["n_h"]] * 1e21, dtype=torch.float32,
                              device=dev)                          # (B,)
        fe = torch.as_tensor(th[:, self.fe_idx], dtype=torch.float32, device=dev)
        total = None
        for z, model in self.emu.models.items():
            if z in (1, 2):
                a = torch.ones(B, device=dev)                      # H/He solar
            elif z in FREE_Z:
                a = torch.as_tensor(th[:, self.col[SYMBOL[z]]],
                                    dtype=torch.float32, device=dev)
            else:
                a = fe                                             # tied to Fe
            ef = element_broadened_flux(                           # (B, M)
                model, temps, self.velocity, self.edges_rest,
                absorption=self.absn, n_h=n_h, redshift=self.z,
                use_torch_absorption=True)
            total = a[:, None] * ef if total is None else total + a[:, None] * ef
        return total

    def _fold(self, total, theta):
        """Fold flux (B, M) through ARF+RMF on the GPU, scale by norm."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        norm = torch.as_tensor(10.0 ** th[:, self.col["log_norm"]],
                               dtype=torch.float32, device=self.device)
        eff = (total * self.arf).transpose(0, 1).contiguous()      # (N, B)
        counts = torch.sparse.mm(self.Rt, eff).transpose(0, 1)     # (B, C)
        return counts[:, self.keep_idx] * norm[:, None] * self.scale_const

    def __call__(self, theta):
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if th.shape[0] <= self.chunk:
            return self._fold(self._flux(th), th).detach().cpu().numpy()
        parts = [self._fold(self._flux(th[i:i + self.chunk]),
                            th[i:i + self.chunk]).detach().cpu().numpy()
                 for i in range(0, th.shape[0], self.chunk)]
        return np.concatenate(parts, axis=0)


def _validate():
    """Batched EnsembleForward vs serial fisher_bias.Forward, on CPU."""
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response)
    emu = JointOperatorModel(models_dir=STORE28, device="cpu")

    ens = EnsembleForward(emu, response, absorption, keep, "single",
                          velocity=PERSEUS["vel"], device="cpu")
    ser = Forward(emu, response, absorption, keep, "single")       # has sigma_v
    pars = ens.params(15.685)
    rng = np.random.default_rng(0)
    truth = np.array([p.truth for p in pars])
    lo = np.array([p.low for p in pars]); hi = np.array([p.high for p in pars])
    B = 4
    theta = np.clip(truth + 0.02 * (hi - lo) * rng.standard_normal((B, len(pars))),
                    lo, hi)
    ens_out = ens(theta)                                          # (B, n_keep)
    # serial: insert fixed sigma_v into each walker's parameter vector
    vi = ser.names.index("sigma_v")
    ser_out = np.stack([
        ser(np.insert(theta[b], vi, PERSEUS["vel"])) for b in range(B)])
    rel = np.abs(ens_out - ser_out) / np.clip(np.abs(ser_out), 1e-30, None)
    print(f"batched vs serial: max rel diff {rel.max():.2e}, "
          f"median {np.median(rel):.2e} over {ens_out.shape} (B x n_keep)")
    print("PASS" if rel.max() < 1e-3 else "FAIL (investigate)")


if __name__ == "__main__":
    _validate()
