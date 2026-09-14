"""Hot-element floor experiment: campaign-specific configuration + scaffolding.

Does the emulator's sub-0.1% miss on the hot Fe-peak elements bias the
quantities XRISM actually publishes, and above what count level?

Design (agreed with D.H., 2026-08-12): one headline experiment -- a realistic
Perseus spectrum generated from the independent SPEX truth, fit back with the
emulator using the *literature* modelling strategy (XRISM/Resolve Perseus-core
enrichment paper, arXiv:2606.17141), across a scan of total counts, in both a
single-temperature and a Gaussian-DEM flavour. The verdict is the crossover
count N* where the emulator's systematic bias equals the Poisson error, per
published parameter.

This module holds the campaign-specific pieces (the Perseus fiducials, the
literature abundance-tying scheme, which XRISM/Resolve response to use, the
fit band) plus thin wrappers that bind the general package capabilities in
``spexai.inference.*`` to them. The general capabilities themselves --
``band_mask``, response lookup, the memory-frugal truth streamer, the
PCHIP-safe DEM grid floor -- live in the package (``spexai.inference.response``,
``spexai.inference.simulate``, ``spexai.inference.tempdist``) since they have
no Perseus-specific default baked in.

Former home: ``inference_demo/hot_floor/experiment.py`` + the campaign-specific
half of ``fisher_bias.py`` (``N_REF``, ``Par``, ``Forward``, ``build_params``),
reached by 8 ``sys.path.insert(..., "hot_floor")`` sites across scripts/ and
inference_demo/hot_floor/. Those are gone: everything importing this now does
``from campaign import ...`` (or, for scripts/ outside this directory, adds
this directory to ``sys.path`` once) with the dependency direction strictly
scripts/ -> spexai/, never the reverse.
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from spexai.broadening import C_KMS
from spexai.config import RESP_DIR
from spexai.inference.abundances import AbundanceModel, SYMBOL
from spexai.inference.fitting import SIGMA_V_PRIOR
from spexai.inference.response import band_mask as _band_mask
from spexai.inference.response import find_response
from spexai.inference.simulate import stream_truth_counts as _stream_truth_counts
from spexai.inference import tempdist as td

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

BAND = (1.9, 12.0)

# The Fe XXV region excluded by the XRISM/Resolve Perseus analysis (keV,
# OBSERVED frame -- band_mask keys off channel centres and applies no redshift
# correction). At Perseus's z=0.017284 this is 6.681-6.734 keV rest, which
# covers the He-alpha resonance line w (6.7004) and intercombination x
# (6.6824) while keeping y (6.6675) and the forbidden line z (6.6366). Masking
# w while keeping z is a RESONANCE SCATTERING cut: w is optically thick in the
# Perseus core, so its flux is redistributed and an optically-thin CIE model
# cannot reproduce it.
#
# CHOOSE THIS DELIBERATELY -- it used to be a module default that every
# experiment inherited silently, which is how a source-specific astrophysical
# decision ended up inside general emulator-characterisation measurements.
# Two regimes:
#
#   EXCLUDE_PERSEUS_LITERATURE -- only when the goal is to REPRODUCE the
#       published Perseus analysis (the showcase). The mask is part of that
#       analysis's strategy.
#
#   EXCLUDE_NONE -- everywhere the goal is to MEASURE EMULATOR ERROR against a
#       self-consistent CIE injection. Resonance scattering is absent from both
#       the SPEX truth and the emulator, so the mask corrects nothing there; it
#       only deletes the Fe-K channels where the emulator's line-head floor is
#       worst, which biases the result optimistic. Note the direction: dropping
#       the mask raises |b_sys| and lowers sigma_ref, so it LOWERS N* -- the
#       unmasked number is the conservative bound.
#
# WARNING: this window is observed-frame and is only meaningful because every
# experiment currently pins z to PERSEUS["z"]. If redshift is ever varied, a
# fixed observed-frame window is incoherent (at z=0.1 it masks arbitrary
# continuum while Fe XXV sits elsewhere) and must be made rest-frame and
# shifted per point, or dropped.
EXCLUDE_PERSEUS_LITERATURE = (6.567, 6.620)
EXCLUDE_NONE = None

N_REF = 1e5                    # reference in-band counts for Fisher/deviance


def injected_abundances(elements) -> Dict[int, float]:
    """{Z: solar-relative} for every metal in ``elements`` (H/He omitted)."""
    ab = {}
    for z in elements:
        z = int(z)
        if z in (1, 2):
            continue
        ab[z] = XFE_RATIO[z] * FE_ABUND if z in XFE_RATIO else FE_ABUND
    return ab


def resolve_perseus(overrides: Optional[Dict[str, float]] = None) -> Dict:
    """A local copy of PERSEUS with CLI-style overrides applied.

    Never mutates the shared module dict -- callers used to do
    ``PERSEUS[key] = val`` in place, which meant every subsequent reader saw
    the override (fine for a single-process CLI run) but made the module
    state itself the channel carrying it. Threading the returned dict through
    explicitly makes that channel visible at every call site instead."""
    p = dict(PERSEUS)
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    return p


# --- response + fit-band mask ----------------------------------------------

# ARF choice: the 5' flat-field, gate-valve-closed ARF from the Cycle 2/3
# canned set. Perseus is extended, so a point-source ARF would carry the wrong
# aperture correction. The flat field is still an approximation for a
# cool-core cluster's peaked surface brightness -- it is recorded here as an
# explicit assumption. For emulator testing it does not need to be exact: the
# same ARF folds both the injected truth and the fitted model, so it cancels.
# Any absolute-flux claim would need an xaarfgen ARF for the real pointing.
RESOLVE_RMF = os.environ.get("SPEXAI_RESOLVE_RMF", "rsl_Hp_L_2025.rmf")
RESOLVE_ARF = os.environ.get("SPEXAI_RESOLVE_ARF", "rsl_extflat5_GVC_2025.arf")


def find_xrism_response():
    """(RMF, ARF) paths for XRISM/Resolve. Both must exist -- no fallback."""
    try:
        return find_response(RESP_DIR, RESOLVE_RMF, RESOLVE_ARF)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\nSet $SPEXAI_RESPONSES to the responses dir, or "
            f"$SPEXAI_RESOLVE_RMF/$SPEXAI_RESOLVE_ARF to different filenames. "
            f"The canned Cycle 3 files are at https://heasarc.gsfc.nasa.gov"
            f"/docs/xrism/proposals/responses.html") from None


def check_truth_response(tz, rmf, arf):
    """Fail if a truth npz was built against a different response than the
    one this run uses.

    The channel-count and element-set guards cannot catch this: an ARF changes
    neither ``n_keep`` (band_mask keys off channel centres only) nor the
    element list, but it rescales ``d_inband`` channel by channel. A truth
    built with a flat effective area is silently wrong against an ARF-folded
    fit, which is exactly the failure this exists to stop."""
    want = (os.path.basename(rmf), os.path.basename(arf))
    if "arf" not in tz.files:
        raise SystemExit(
            "truth npz predates response recording: it was built with a flat "
            "(unit) effective area and is NOT valid for an ARF-folded fit.\n"
            "Regenerate it with scripts/inference/dump_truth.py.")
    got = (str(tz["rmf"]), str(tz["arf"]))
    if got != want:
        raise SystemExit(
            f"truth npz was built with RMF/ARF {got} but this run uses {want}."
            f"\nRegenerate the truth against THIS response, or set "
            f"$SPEXAI_RESOLVE_RMF / $SPEXAI_RESOLVE_ARF to match it.")


# Reach of the RMF's UPWARD redistribution above the incident energy, keV.
# Measured on rsl_Hp_L_2025.rmf: 10-12 eV at every incident energy from 1.5 to
# 11 keV (the Gaussian core's upper side). The RMF's long tail runs the other
# way -- thousands of eV DOWNWARD -- which is why the cut below is safe at all:
# energy-loss processes scatter photons down in recorded energy, never up.
RMF_UP_REACH_KEV = 0.012
# fft_broaden zero-pads to 8 sigma, so that is the reach the continuum can be
# lifted by velocity broadening.
FFT_SIGMA_REACH = 8.0
# The three physical margins alone land ~10 eV short: scatter_to_grid's
# delta-at-centre deposit and rebin_flux both lose a little at the very first
# cells of the truncated grid. Measured against an untruncated reference (Fe,
# kT=5 keV, sigma_v=600 km/s, all 20200 in-band channels): the physical cut of
# 1.8901 keV leaves 10 channels above 1e-4 with a worst of 7.9e-3, while
# 1.88 keV and below sit at 2.4e-7, i.e. float noise. 1% buys ~19 eV of margin
# for ~20 extra grid bins out of 24212.
EDGE_SAFETY_FRAC = 0.01


def band_low_cut(band=BAND, sigma_v_max=None, z=None,
                 rmf_up_reach=RMF_UP_REACH_KEV):
    """Lowest REST-frame energy the emulator must still produce, in keV.

    Below this, continuum flux cannot reach an in-band channel by any route,
    so evaluating the trunk and broadening there is wasted work (see
    ``operator_model.restrict_band``). Three margins, applied in order:

    1. ``band[0]`` is a channel (observed-frame) edge. Incident photons as low
       as ``band[0] - rmf_up_reach`` can still be redistributed into it.
    2. Velocity broadening can lift continuum from lower still, by up to
       ``FFT_SIGMA_REACH`` sigma at the widest sigma_v the prior allows.
    3. The model grid is the RESPONSE grid redshifted, ``edges * (1 + z)``, so
       the threshold converts to the rest frame by the same factor.

    A fourth, ``EDGE_SAFETY_FRAC``, covers the grid-edge losses the first three
    do not describe; see the constant for the measurement that sets it.

    At the campaign defaults the margins nearly cancel: 1.9 keV -> 1.888 (RMF)
    -> 1.858 (broadening) -> 1.890 (rest frame at Perseus's z) -> 1.871 keV.
    """
    if sigma_v_max is None:
        sigma_v_max = SIGMA_V_PRIOR[1]
    if z is None:
        z = PERSEUS["z"]
    e_rmf = band[0] - rmf_up_reach
    e_broad = e_rmf * math.exp(-FFT_SIGMA_REACH * sigma_v_max / C_KMS)
    return e_broad * (1.0 + z) * (1.0 - EDGE_SAFETY_FRAC)


def restrict_to_band(emu, band=BAND, verbose=True, **kw):
    """Apply ``band_low_cut`` to a freshly built ``JointOperatorModel``.

    Call once, immediately after construction and before anything touches
    ``emu.batched`` (which caches the grid). Verified lossless in-band: worst
    relative change 4.9e-6 over H/He/Si/Fe at kT = 1-9 keV, sigma_v = 30-600
    km/s, n_H = 0-5e21, against an untruncated reference.
    """
    from spexai.inference.operator_model import restrict_band
    e_lo = band_low_cut(band=band, **kw)
    cut = restrict_band(emu, e_lo)
    if verbose:
        if cut is None:
            print(f"band restriction: nothing below {e_lo:.4f} keV to cut",
                  flush=True)
        else:
            print(f"band restriction: e_lo={e_lo:.4f} keV rest frame, "
                  f"training grid {cut[0]:,} -> {cut[1]:,} bins "
                  f"({1 - cut[1] / cut[0]:.1%} dropped)", flush=True)
    return e_lo


def band_mask(response, band=BAND, *, exclude) -> np.ndarray:
    """Boolean channel mask: within ``band`` and outside ``exclude`` (keV).

    ``exclude`` is KEYWORD-ONLY AND REQUIRED, deliberately. It previously
    defaulted to the Perseus resonance-scattering cut, so all ten call sites
    inherited a source-specific astrophysical choice without anyone deciding
    it -- including the experiments whose whole purpose is to measure emulator
    error, where that cut removes precisely the Fe-K channels the emulator is
    worst at. Pass ``EXCLUDE_PERSEUS_LITERATURE`` or ``EXCLUDE_NONE``
    explicitly; see their definitions above for which applies."""
    return _band_mask(response, band, exclude)


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


def gaussian_dem(mean=None, sigma=None, lo=td.PCHIP_TRUTH_SAFE_LO_KEV,
                 hi=td.EMULATOR_T_HI_KEV, n=70):
    """Gaussian-in-T DEM model + its (mean,sigma) params, on a fixed grid.

    ``lo`` defaults to the package's PCHIP-safe floor (see
    ``spexai.inference.tempdist.PCHIP_TRUTH_SAFE_LO_KEV``): a grid point below
    the per-element training minimum (~0.501 keV) makes the PCHIP SPEX truth
    extrapolate and blow up (Ar -> inf at 0.5 keV), and the Gaussian carries
    negligible weight there anyway.

    ``hi`` is the top of the emulator's trained temperature range. It was
    previously 10 keV, which was an unrevisited default rather than a derived
    limit, and it threw away the hot 28% of the usable span -- precisely where
    a Gaussian DEM runs off the grid. At T_mean = 8 keV and sigma = 0.2 dex the
    old grid contained only 71% of the emission measure; since the weights are
    no longer renormalised (see ``tempdist.ParametricDEM``) that truncation is
    now visible in the spectrum instead of being silently redistributed.

    ``n`` sets the resolution. Over 0.7-19.94 keV, 70 nodes give 0.0211 dex per
    cell, which resolves any DEM down to sigma ~ 0.056 dex at SPEX's own
    standard of 16 nodes across the distribution. Note this is a FLOOR, not a
    limit that more nodes remove: the grid is anchored in absolute temperature
    rather than to T_mean, so as sigma -> 0 the weight lands on nodes offset
    from T_mean by up to half a cell instead of collapsing onto it. Near-
    isothermal DEMs are outside what a fixed grid can represent, however fine.
    """
    grid = td.TempGrid(lo, hi, n=n)
    model = td.gaussian_T(grid)
    p = {"T_mean": PERSEUS["dem_mean"] if mean is None else mean,
         "T_sigma": PERSEUS["dem_sigma"] if sigma is None else sigma}
    return model, p


def stream_truth_counts(cfg: TruthConfig, response, absorption,
                        store=None, datadir=None, device="cpu",
                        perseus: Optional[Dict] = None,
                        verbose=False) -> np.ndarray:
    """Adapts ``TruthConfig`` + the Perseus fiducials to the package's
    general ``spexai.inference.simulate.stream_truth_counts``.

    ``perseus`` defaults to the module ``PERSEUS``; pass
    ``resolve_perseus(overrides)`` to apply CLI overrides without mutating it.
    """
    p = PERSEUS if perseus is None else perseus
    kwargs = {}
    if store is not None:
        kwargs["store"] = store
    if datadir is not None:
        kwargs["datadir"] = datadir
    return _stream_truth_counts(
        cfg.elements, cfg.abundances, p["z"], p["kT"], p["vel"], p["n_h"],
        p["dist_m"], response, absorption, exposure=cfg.exposure,
        norm_ref=cfg.norm_ref, dem=cfg.dem, dem_params=cfg.dem_params,
        device=device, verbose=verbose, **kwargs)


# --- literature-strategy fit parametrisation (from fisher_bias.py) ---------

@dataclass
class Par:
    name: str
    truth: float
    step: float                # finite-difference step
    low: float
    high: float


class Forward:
    """theta (free-parameter vector) -> in-band emulator counts at N_REF scale."""

    def __init__(self, emu, response, absorption, keep, mode, dem=None,
                 perseus=None):
        self.emu, self.resp, self.absn = emu, response, absorption
        self.keep, self.mode, self.dem = keep, mode, dem
        self.perseus = PERSEUS if perseus is None else perseus
        self.logz = float(np.log10(self.perseus["z"]))
        self.ld = self.perseus["dist_m"]
        # literature abundance parametrisation: free FREE_Z, tie the rest to Fe.
        ab = AbundanceModel(emu.elements)
        for z in FREE_Z:
            ab.free_element(z, SYMBOL[z])
        others = [z for z in emu.elements if z >= 3 and z not in FREE_Z]
        ab.tie_const(others, 1.0, 26)
        self.abmodel = ab
        self.abnames = ab.param_names
        # parameter order: [abundances...] + thermal + [sigma_v, n_h, log_norm]
        self.thermal = ["kT"] if mode == "single" else ["T_mean", "T_sigma"]
        self.names = self.abnames + self.thermal + ["sigma_v", "n_h", "log_norm"]

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        p = dict(zip(self.names, theta))
        abund = self.abmodel.to_abundances(p)
        norm = 10.0 ** p["log_norm"]
        common = dict(luminosity_distance=self.ld, absorption=self.absn,
                      n_h=p["n_h"] * 1e21)          # n_h sampled in 1e21 cm^-2
        if self.mode == "single":
            mu = self.emu.predict_counts(
                torch.tensor([p["kT"]]), abund, self.logz, norm, p["sigma_v"],
                self.resp, 1.0, **common)
        else:
            w = self.dem.weights({"T_mean": p["T_mean"], "T_sigma": p["T_sigma"]})
            mu = self.emu.predict_counts_dem(
                self.dem.temp_grid, w, abund, self.logz, norm, p["sigma_v"],
                self.resp, 1.0, **common)
        return mu.squeeze(0).cpu().numpy()[self.keep]


def build_params(fwd: Forward, log_norm_truth: float) -> List[Par]:
    """Truth values, steps and bounds for every free parameter.

    Reads the physical fiducials from ``fwd.perseus`` (falls back to module
    ``PERSEUS`` for objects that don't set it), so a caller's overrides
    (via ``resolve_perseus``) flow through as long as they built ``fwd`` with
    the same dict.
    """
    p = getattr(fwd, "perseus", None) or PERSEUS
    inj = injected_abundances(fwd.emu.elements)
    out: List[Par] = []
    for z in FREE_Z:
        out.append(Par(SYMBOL[z], inj[z], 1e-3, 0.02, 3.0))
    if fwd.mode == "single":
        out.append(Par("kT", p["kT"], 5e-3, 1.5, 7.5))
    else:
        out.append(Par("T_mean", p["dem_mean"], 5e-3, 1.5, 7.5))
        out.append(Par("T_sigma", p["dem_sigma"], 5e-3, 0.15, 3.0))
    out.append(Par("sigma_v", p["vel"], 1.0, *SIGMA_V_PRIOR))
    out.append(Par("n_h", p["n_h"] / 1e21, 1e-2, 0.0, 5.0))  # 1e21 cm^-2
    out.append(Par("log_norm", log_norm_truth, 2e-3,
                   log_norm_truth - 1.0, log_norm_truth + 1.0))
    return out
