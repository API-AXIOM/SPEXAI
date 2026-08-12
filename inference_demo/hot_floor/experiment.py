"""Hot-element floor experiment: does the emulator's sub-0.1% miss on the hot
Fe-peak elements bias the quantities XRISM actually publishes, and above what
count level?

Design (agreed with D.H., 2026-08-12): one headline experiment -- a realistic
Perseus spectrum generated from the independent SPEX truth, fit back with the
emulator using the *literature* modelling strategy (XRISM/Resolve Perseus-core
enrichment paper, arXiv:2606.17141), across a scan of total counts, in both a
single-temperature and a Gaussian-DEM flavour. The verdict is the crossover
count N* where the emulator's systematic bias equals the Poisson error, per
published parameter.

This module holds the shared pieces: physical + abundance configuration, the
XRISM response and fit-band channel mask, and a *memory-frugal* truth-counts
generator. Because abundance weighting, velocity broadening, Galactic
absorption and folding are all LINEAR in each element's native flux, the total
counts are the sum of per-element counts -- so we stream one element at a time
(reusing the validated ``SpexTruthModel`` path) and never hold more than a
single element's cache (~0.4 GB) in memory instead of ~11 GB for all 28.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from spexai.inference.response import Response
from spexai.inference.absorption import Absorption
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference import tempdist as td

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Paths are overridable by environment variable for cluster deployment.
STORE28 = os.environ.get(
    "SPEXAI_STORE", os.path.join(REPO, "inference_demo", "hot_floor", "store28"))
DATADIR = os.environ.get(
    "SPEXAI_PROCESSED", os.path.expanduser("~/work/data/spexai/processed"))
RESP_DIR = os.environ.get(
    "SPEXAI_RESPONSES", os.path.expanduser("~/work/data/spexai/responses"))
MPC_M = 3.0857e22

# --- Perseus fiducials (arXiv:2606.17141) ----------------------------------
# z, N_H (cm^-2), core kT (keV), turbulence sigma_v (km/s); Gaussian-DEM mean
# and width for the multi-T flavour.
PERSEUS = dict(z=0.017284, dist_mpc=75.0, n_h=1.36e21, kT=3.9, vel=180.0,
               dem_mean=3.9, dem_sigma=1.0)
PERSEUS["dist_m"] = PERSEUS["dist_mpc"] * MPC_M

# Injected abundance pattern (Asplund proto-solar relative), from the paper's
# SSM analysis: free Fe and X/Fe for the alpha + iron-peak elements; every other
# metal is tied to Fe at ratio 1.0 (that includes the hot weak failers Ti, Co,
# Cu, Zn). H/He stay solar and are never managed here.
FE_ABUND = 0.71
XFE_RATIO: Dict[int, float] = {14: 0.96, 16: 0.93, 18: 0.83, 20: 0.86,
                               24: 0.95, 25: 0.67, 28: 1.02}
# Elements freed in the literature strategy (also the fit's free abundances):
FREE_Z: List[int] = sorted([26] + list(XFE_RATIO))     # Si,S,Ar,Ca,Cr,Mn,Fe,Ni
# The emulator's problem children, split by whether the literature frees them.
HOT_SCIENCE = [24, 25, 28]                 # Cr, Mn, Ni -- weak *and* fitted
HOT_WEAK = [22, 27, 29, 30]                # Ti, Co, Cu, Zn -- tied to Fe, unfit

# Fit band and the excluded Fe XXV region (keV), matching the paper.
BAND = (1.9, 12.0)
EXCLUDE = (6.567, 6.620)


def injected_abundances(elements) -> Dict[int, float]:
    """{Z: solar-relative} for every metal in ``elements`` (H/He omitted)."""
    ab = {}
    for z in elements:
        z = int(z)
        if z in (1, 2):
            continue
        ab[z] = XFE_RATIO[z] * FE_ABUND if z in XFE_RATIO else FE_ABUND
    return ab


# --- response + fit-band mask ----------------------------------------------

def find_xrism_response():
    """XRISM/Resolve RMF (+ ARF if present) from the responses dir."""
    import glob
    for pat in ("rsl_*.rmf", "*resolve*.rmf", "*xrism*.rmf"):
        hits = sorted(glob.glob(os.path.join(RESP_DIR, pat)))
        if hits:
            rmf = hits[0]
            arf = rmf.replace(".rmf", ".arf")
            if not os.path.exists(arf):
                cand = sorted(glob.glob(os.path.join(RESP_DIR, "rsl_*.arf")))
                arf = cand[0] if cand else None
            return rmf, arf
    raise FileNotFoundError(f"no XRISM/Resolve RMF under {RESP_DIR}")


def band_mask(response, band=BAND, exclude=EXCLUDE) -> np.ndarray:
    """Boolean channel mask: within ``band`` and outside ``exclude`` (keV)."""
    cent = response.chan_e_cent.cpu().numpy()          # (N_channels,)
    keep = (cent >= band[0]) & (cent <= band[1])
    if exclude is not None:
        keep &= ~((cent >= exclude[0]) & (cent <= exclude[1]))
    return keep


# --- memory-frugal truth generator -----------------------------------------

@dataclass
class TruthConfig:
    """Everything that defines the injected (noise-free) truth spectrum."""
    elements: List[int]
    abundances: Dict[int, float]
    exposure: float = 1e5
    norm_ref: float = 1e11                 # reference emission measure (1e64 m^-3)
    dem: Optional[object] = None           # a tempdist model, or None for single-T
    dem_params: dict = field(default_factory=dict)


def gaussian_dem(mean=None, sigma=None, lo=0.5, hi=10.0, n=48):
    """Gaussian-in-T DEM model + its (mean,sigma) params, on a fixed grid."""
    grid = td.TempGrid(lo, hi, n=n)
    model = td.gaussian_T(grid)
    p = {"T_mean": PERSEUS["dem_mean"] if mean is None else mean,
         "T_sigma": PERSEUS["dem_sigma"] if sigma is None else sigma}
    return model, p


def stream_truth_counts(cfg: TruthConfig, response, absorption,
                        store=STORE28, datadir=DATADIR, device="cpu",
                        verbose=False) -> np.ndarray:
    """Noise-free channel counts for ``cfg``, summed one element at a time.

    Returns the Poisson-mean counts (N_channels,) at ``cfg.norm_ref``; rescale
    linearly to any target total counts.
    """
    logz = float(np.log10(PERSEUS["z"]))
    total = None
    for z in cfg.elements:
        z = int(z)
        a = cfg.abundances.get(z, 1.0)
        if a == 0.0 or z in (1, 2):
            # H/He carry the continuum at solar -> include them at a=1.0
            if z not in (1, 2):
                continue
            a = 1.0
        m = SpexTruthModel(models_dir=store, datadir=datadir,
                           elements=[z], device=device)
        if cfg.dem is not None:
            w = cfg.dem.weights(cfg.dem_params)
            c = m.predict_counts_dem(
                cfg.dem.temp_grid, w, {z: a}, logz, cfg.norm_ref,
                PERSEUS["vel"], response, cfg.exposure,
                luminosity_distance=PERSEUS["dist_m"],
                absorption=absorption, n_h=PERSEUS["n_h"])
        else:
            c = m.predict_counts(
                torch.tensor([PERSEUS["kT"]]), {z: a}, logz, cfg.norm_ref,
                PERSEUS["vel"], response, cfg.exposure,
                luminosity_distance=PERSEUS["dist_m"],
                absorption=absorption, n_h=PERSEUS["n_h"])
        c = c.squeeze(0).cpu().numpy()
        total = c if total is None else total + c
        if verbose:
            print(f"  +Z{z:02d} a={a:.3f}  running total counts={total.sum():.3e}")
        del m
    return np.clip(total, 0.0, None)


# --- smoke test ------------------------------------------------------------

def _smoke():
    """Phase-0 sanity: 3-element truth vs emulator through the XRISM response."""
    from spexai.inference.operator_model import JointOperatorModel

    rmf, arf = find_xrism_response()
    print(f"response: {os.path.basename(rmf)}  arf={arf is not None}")
    response = Response(rmf, arf)
    print(response)
    absorption = Absorption.default()
    print(f"absorption: {absorption.name}")
    keep = band_mask(response)
    print(f"band {BAND} minus Fe XXV {EXCLUDE}: "
          f"{keep.sum()}/{len(keep)} channels kept")

    # H (continuum) + Fe + Cr, a quick end-to-end check.
    els = [1, 26, 24]
    ab = injected_abundances(els)
    cfg = TruthConfig(elements=els, abundances=ab, exposure=1e5)
    mu = stream_truth_counts(cfg, response, absorption, verbose=True)
    print(f"truth total counts (norm_ref, full band): {mu.sum():.3e}; "
          f"in-band: {mu[keep].sum():.3e}")

    emu = JointOperatorModel(models_dir=STORE28, device="cpu", elements=els)
    logz = float(np.log10(PERSEUS["z"]))
    ce = emu.predict_counts(
        torch.tensor([PERSEUS["kT"]]), ab, logz, cfg.norm_ref, PERSEUS["vel"],
        response, cfg.exposure, luminosity_distance=PERSEUS["dist_m"],
        absorption=absorption, n_h=PERSEUS["n_h"]).squeeze(0).cpu().numpy()
    frac = (ce[keep].sum() - mu[keep].sum()) / mu[keep].sum()
    print(f"emulator in-band counts: {ce[keep].sum():.3e}  "
          f"(fractional diff vs truth: {frac:+.4%})")


if __name__ == "__main__":
    _smoke()
