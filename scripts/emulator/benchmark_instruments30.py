"""Instrument-level validation of the 30-element emulator (agenda item P5).

Supersedes ``benchmark_instruments.py``, which covered iron only, 16 spectra,
and folded without effective area. Three passes, all on the abundance-summed
30-element spectrum:

(1) ``grids``   -- resolution-matched instrument grids (bin width = FWHM /
                   --oversample, no redistribution).
(2) ``fold``    -- the real response: prediction evaluated directly on the
                   RMF's incident-energy grid (the production path), truth
                   rebinned onto it, both folded through RMF x ARF and scored
                   per detector channel inside the instrument band. Run twice,
                   with and without the ARF, because the effective area
                   reweights channels by ~60x across the Resolve band and so
                   changes every counts-weighted number.
(3) ``subbin``  -- sub-bin line placement. The training grid is linear at
                   0.4914 eV, essentially Resolve's channel width, so a target
                   grid offset by a fraction of a bin is the regime where the
                   delta-at-bin-centre line deposit costs accuracy. Sweeps the
                   phase of a 0.4914 eV target grid over [0, 1) bins and
                   reports the peak-to-peak swing, unfolded and (with
                   --subbin_fold) after rebinning to the RMF grid and folding.

Truth is the SPEX cache itself -- no interpolation. The 30 element caches do
not share a temperature grid (they come in several groups), so the evaluation
temperatures are the intersection of all 30 grids; ``--offgrid`` adds a
cross-check against ``SpexTruthModel``'s PCHIP truth at temperatures on no
cache grid at all. Note that a cache temperature is in the *test* split of
only ~10% of the elements: see ``--report_split`` for the train/test breakdown
that bounds what that costs.

    python scripts/emulator/benchmark_instruments30.py --ntemp 16
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.emulator.benchmark_instruments import (BANDS, RMF_FILES,
                                                    instrument_grids)
from spexai.broadening import direct_broaden, rebin_flux
from spexai.config import DATADIR, RESP_DIR, RESULTS
from spexai.data import SpectrumData
from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR
from spexai.inference.response import Response

# ARF for each instrument; Resolve uses the campaign default (5' extended
# flat field, gate valve closed) -- see docs/inference_methodology.tex.
ARF_FILES = {"resolve": "rsl_extflat5_GVC_2025.arf",
             "acis": "aciss_aimpt_cy28.arf",
             "heg": "aciss_heg1_cy28.garf",
             "meg": "aciss_meg1_cy28.garf"}

TRAIN_BIN_KEV = 0.4914e-3     # native training grid spacing (linear)


# --------------------------------------------------------------------------
# truth: the cache itself
# --------------------------------------------------------------------------

def common_temperatures(datadir: str, elements: Sequence[int]
                        ) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """Temperatures present in every element cache, per-element test masks,
    and the union of all cache grids.

    Returns ``(common (T,), {Z: bool (T,) "in Z's test split"}, union (U,))``.
    Element caches were generated in groups and do not share a temperature
    grid, so the intersection is strictly smaller than any one cache; the
    union is what an "off-grid" temperature has to avoid.
    """
    grids, test_sets = {}, {}
    for z in elements:
        d = SpectrumData(os.path.join(datadir, f"element{z}"))
        t = d.temps.numpy().astype(np.float64)
        grids[z] = t
        test_sets[z] = set(t[d.test_idx.numpy()].tolist())
    common = grids[elements[0]]
    union = grids[elements[0]]
    for z in elements[1:]:
        common = np.intersect1d(common, grids[z])
        union = np.union1d(union, grids[z])
    common = np.sort(common)
    masks = {z: np.array([float(t) in test_sets[z] for t in common])
             for z in elements}
    return common, masks, np.sort(union)


def select_temperatures(common: np.ndarray, ntemp: int) -> np.ndarray:
    """``ntemp`` temperatures spread log-uniformly over the common grid."""
    want = np.logspace(np.log10(common[0]), np.log10(common[-1]), ntemp)
    idx = np.unique(np.abs(np.log10(common)[None, :]
                           - np.log10(want)[:, None]).argmin(axis=1))
    return common[idx]


def cache_truth_flux(datadir: str, elements: Sequence[int],
                     temps: np.ndarray, abundances: Optional[Dict[int, float]]
                     = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact SPEX cache spectra, abundance-summed. (B, n_bins) integrated flux.

    One element is held in memory at a time (each cache is ~0.4 GB).
    """
    total, edges = None, None
    for z in elements:
        d = SpectrumData(os.path.join(datadir, f"element{z}"))
        t = d.temps.numpy().astype(np.float64)
        order = np.argsort(t)
        pos = order[np.searchsorted(t[order], temps)]
        assert np.allclose(t[pos], temps), f"Z={z} missing an evaluation temp"
        if edges is None:
            from spexai.operator import edges_from_centers
            edges = edges_from_centers(d.energy).double()      # (n_bins+1,)
            widths = edges[1:] - edges[:-1]                    # (n_bins,)
        # (B, n_bins) integrated flux at solar abundance
        f = (torch.pow(10.0, d.logflux[pos].double().clamp(min=-30)) * widths)
        a = float((abundances or {}).get(z, 1.0))
        total = f * a if total is None else total + f * a
        del d, f
    return total, edges


def offgrid_truth_flux(datadir: str, elements: Sequence[int],
                       temps: np.ndarray,
                       abundances: Optional[Dict[int, float]] = None,
                       half: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCHIP truth at temperatures on no cache grid. (B, n_bins), native edges.

    Same interpolation as ``SpexTruthModel``/``ElementTruth`` -- per-bin
    monotone cubic in log10 T over the *training* rows only -- but streaming
    one element at a time, because holding all 30 caches costs ~12 GB.
    P3 bounds this interpolation's own error at ~1e-5, so a cross-check
    against it isolates whether the on-grid evaluation temperatures flatter
    the emulator.
    """
    from spexai.data import pchip_generate
    from spexai.operator import edges_from_centers
    total, edges, widths = None, None, None
    for z in elements:
        d = SpectrumData(os.path.join(datadir, f"element{z}"))
        tr = d.train_idx.numpy()
        lt = np.log10(d.temps.numpy()[tr].astype(np.float64))
        order = np.argsort(lt)
        if edges is None:
            edges = edges_from_centers(d.energy).double()
            widths = edges[1:] - edges[:-1]
        # (B, n_bins) log10 flux density -> integrated flux
        dens = pchip_generate(lt[order],
                              np.ascontiguousarray(d.logflux[tr].numpy()[order]),
                              np.log10(temps), half=half)
        f = torch.pow(10.0, torch.from_numpy(dens).double()) * widths
        a = float((abundances or {}).get(z, 1.0))
        total = f * a if total is None else total + f * a
        del d, f, dens
    return total, edges


def offgrid_temperatures(union: np.ndarray, ntemp: int) -> np.ndarray:
    """Temperatures maximally far (in log10 T) from any cache row.

    Picks the ``ntemp`` widest gaps in the union of all 30 cache grids and
    returns their midpoints, so PCHIP is doing the most work it ever does and
    no evaluation point is any element's training row. Note that the widest
    gaps are not uniformly distributed in T, so this set is *not* matched to
    the on-grid set -- use ``offgrid_paired`` for a controlled comparison.
    """
    lt = np.log10(union)
    gaps = np.diff(lt)
    pick = np.argsort(gaps)[-ntemp:]
    mid = 10.0 ** (0.5 * (lt[pick] + lt[pick + 1]))
    return np.sort(mid)


def offgrid_paired(union: np.ndarray, on_grid: np.ndarray) -> np.ndarray:
    """Off-grid temperatures paired one-to-one with an on-grid selection.

    For each on-grid temperature, returns the midpoint of the union-grid
    interval immediately above it. The two sets then differ only in whether
    the evaluation point is a cache row, not in where they sit in T -- which
    is what isolates "is the emulator flattered by on-grid evaluation?" from
    "is it worse at the temperatures the gaps happen to be at?".
    """
    lt = np.log10(union)
    j = np.searchsorted(lt, np.log10(on_grid))                 # (n,)
    j = np.clip(j, 0, len(lt) - 2)
    return 10.0 ** (0.5 * (lt[j] + lt[j + 1]))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    """Scores for (B, C) integrated flux, aggregated over B spectra.

    ``mre_flux`` is the flux-weighted error of Sec. sec:instruments, i.e. the
    misplaced-flux fraction sum|pred-truth| / sum truth; it is the headline
    number because it cannot be driven by near-empty channels. ``cw_rms`` is
    the counts-weighted RMS used by the bias campaign. ``mre_median`` is the
    median per-channel relative error over occupied channels, reported for
    continuity with the Fe-only table. Unweighted per-channel *means* are
    deliberately not reported (ill-conditioned; see the section).
    """
    pred, truth = pred.double(), truth.double()
    valid = truth > 0
    eps = torch.where(valid, (pred / truth.clamp(min=1e-300) - 1.0).abs(),
                      torch.zeros_like(truth))                 # (B, C)
    mre_flux = ((pred - truth).abs().sum(1)
                / truth.sum(1).clamp(min=1e-300))              # (B,)
    w = truth / truth.sum(1, keepdim=True).clamp(min=1e-300)   # (B, C)
    cw_rms = torch.sqrt((w * eps ** 2).sum(1))                 # (B,)
    med = torch.stack([e[v].median() if int(v.sum()) else torch.zeros(())
                       for e, v in zip(eps, valid)])           # (B,)
    return {"mre_flux_mean": float(mre_flux.mean()),
            "mre_flux_median": float(mre_flux.median()),
            "mre_flux_max": float(mre_flux.max()),
            "cw_rms_mean": float(cw_rms.mean()),
            "cw_rms_median": float(cw_rms.median()),
            "mre_median": float(med.median()),
            "yield_1pct": float((mre_flux <= 0.01).double().mean() * 100),
            "norm_ratio": float(pred.sum() / truth.sum()),
            "n_channels": int(valid.shape[1]),
            "n_spectra": int(valid.shape[0])}


def fmt(tag: str, r: Dict[str, float]) -> str:
    return (f"  {tag:28s} fluxMRE={r['mre_flux_mean'] * 100:7.4f}% "
            f"(med {r['mre_flux_median'] * 100:7.4f}%, "
            f"max {r['mre_flux_max'] * 100:7.4f}%)  "
            f"cwRMS={r['cw_rms_mean'] * 100:7.4f}%  "
            f"chanMed={r['mre_median'] * 100:7.4f}%  "
            f"norm={r['norm_ratio']:.5f}")


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------

def run_grids(joint, truth_native, native_edges, temps, velocity, oversample,
              results):
    """Pass 1: resolution-matched grids, no redistribution."""
    for gname, ge in instrument_grids(oversample).items():
        t0 = time.time()
        truth = rebin_flux(truth_native, native_edges, ge)
        pred = joint.flux(temps, {}, velocity, ge.float()).double().cpu()
        r = metrics(pred, truth)
        r["seconds"] = time.time() - t0
        results[f"grid_{gname}"] = r
        print(fmt(f"grid {gname} ({len(ge) - 1} bins)", r), flush=True)


def run_fold(joint, truth_native, native_edges, temps, velocity, resp_dir,
             results, arf_modes: Sequence[bool] = (True, False)):
    """Pass 2: real RMF (x ARF) folding, scored per detector channel.

    The incident-grid flux does not depend on the ARF, so it is computed once
    per instrument and folded through each requested effective-area choice.
    """
    for gname, rmf in RMF_FILES.items():
        rmf_path = os.path.join(resp_dir, rmf)
        arf_path = os.path.join(resp_dir, ARF_FILES[gname])
        if not os.path.exists(rmf_path):
            print(f"  {gname}: RMF missing, skipped", flush=True)
            continue
        t0 = time.time()
        base = Response(rmf_path, None)
        ge = base.energy_edges.double()                        # (N_e+1,)
        lo, hi = BANDS[gname]
        chan = ((base.chan_e_cent >= lo) & (base.chan_e_cent <= hi)).numpy()
        # (B, N_e) on the incident grid -- the production evaluation path
        pred_e = joint.flux(temps, {}, velocity, ge.float()).cpu()
        truth_e = rebin_flux(truth_native, native_edges, ge)
        t_fwd = time.time() - t0
        for with_arf in arf_modes:
            if with_arf and not os.path.exists(arf_path):
                print(f"  {gname}: ARF missing, skipped", flush=True)
                continue
            resp = Response(rmf_path, arf_path) if with_arf else base
            r = metrics(resp.fold(pred_e)[:, chan], resp.fold(truth_e)[:, chan])
            r["seconds"] = t_fwd
            r["arf"] = ARF_FILES[gname] if with_arf else None
            r["rmf"] = rmf
            key = f"fold_{gname}" + ("" if with_arf else "_noarf")
            results[key] = r
            print(fmt(f"fold {gname}{'' if with_arf else ' (no ARF)'} "
                      f"({int(chan.sum())} ch)", r), flush=True)


def run_subbin(joint, truth_native, native_edges, temps, velocity, phases,
               band, resp_dir, results, fold_too: bool,
               controls: Sequence[float] = ()):
    """Pass 3: phase sweep of a training-width target grid (Resolve band).

    The target grid has the training bin width, so phase 0 aligns its edges
    with the training grid and phase 0.5 puts every training bin centre on a
    target edge -- the worst case for a delta-at-bin-centre deposit.

    ``controls`` injects a deliberate misplacement: the truth is rebinned from
    native edges shifted by that fraction of a training bin, so its flux sits
    where a sub-bin placement error would have put it. These runs are the
    positive control for the sweep -- they establish that the metric responds
    to the effect at all, without which a flat phase curve proves nothing.
    """
    lo, hi = band
    n = int((hi - lo) / TRAIN_BIN_KEV)
    resp = None
    if fold_too:
        rmf_path = os.path.join(resp_dir, RMF_FILES["resolve"])
        arf_path = os.path.join(resp_dir, ARF_FILES["resolve"])
        resp = Response(rmf_path, arf_path)
        rge = resp.energy_edges.double()
        chan = ((resp.chan_e_cent >= lo) & (resp.chan_e_cent <= hi)).numpy()
    for ph in phases:
        ge = torch.arange(n + 1, dtype=torch.float64) * TRAIN_BIN_KEV \
            + lo + ph * TRAIN_BIN_KEV                           # (n+1,)
        truth = rebin_flux(truth_native, native_edges, ge)
        pred = joint.flux(temps, {}, velocity, ge.float()).double().cpu()
        r = metrics(pred, truth)
        results[f"subbin_phase{ph:.2f}"] = r
        print(fmt(f"subbin phase={ph:.2f}", r), flush=True)
        if resp is not None:
            # same two spectra, rebinned onto the RMF grid and folded: the
            # phase dependence that survives redistribution
            tf = resp.fold(rebin_flux(truth, ge, rge))[:, chan]
            pf = resp.fold(rebin_flux(pred, ge, rge))[:, chan]
            rf = metrics(pf, tf)
            results[f"subbin_phase{ph:.2f}_folded"] = rf
            print(fmt(f"subbin phase={ph:.2f} folded", rf), flush=True)

    # positive control: same comparison, truth displaced by a known sub-bin
    # amount. Only the phase-0 grid is needed.
    ge = torch.arange(n + 1, dtype=torch.float64) * TRAIN_BIN_KEV + lo
    pred = joint.flux(temps, {}, velocity, ge.float()).double().cpu()
    for dx in controls:
        shifted = native_edges + dx * TRAIN_BIN_KEV             # (n_bins+1,)
        truth = rebin_flux(truth_native, shifted, ge)
        r = metrics(pred, truth)
        results[f"subbin_control{dx:+.2f}"] = r
        print(fmt(f"CONTROL truth shifted {dx:+.2f} bin", r), flush=True)
        if resp is not None:
            # the control must be folded too: a flat *folded* phase sweep only
            # means the response erased the effect if an injected
            # misplacement of known size still survives the same fold
            rf = metrics(resp.fold(rebin_flux(pred, ge, rge))[:, chan],
                         resp.fold(rebin_flux(truth, ge, rge))[:, chan])
            results[f"subbin_control{dx:+.2f}_folded"] = rf
            print(fmt(f"CONTROL {dx:+.2f} bin folded", rf), flush=True)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=None,
                    help="preprocessed element caches (default $SPEXAI_PROCESSED)")
    ap.add_argument("--responses_dir", default=None,
                    help="default $SPEXAI_RESPONSES")
    ap.add_argument("--outdir", default=None, help="default $SPEXAI_RESULTS/p5")
    ap.add_argument("--models_dir", default=MODELS_DIR)
    ap.add_argument("--ntemp", type=int, default=16)
    ap.add_argument("--velocity", type=float, default=180.0,
                    help="turbulent sigma_v, km/s (Perseus fiducial)")
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--elements", nargs="+", type=int, default=None)
    ap.add_argument("--passes", nargs="+",
                    default=["grids", "fold", "subbin"],
                    choices=["grids", "fold", "subbin"])
    ap.add_argument("--no_arf_pass", action="store_true",
                    help="skip the ARF-free comparison run")
    ap.add_argument("--subbin_phases", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--subbin_control", nargs="+", type=float,
                    default=[0.1, 0.25, 0.5],
                    help="positive control: fractions of a training bin by "
                         "which to displace the truth (0 disables)")
    ap.add_argument("--subbin_fold", action="store_true",
                    help="also fold the phase sweep through the Resolve response")
    ap.add_argument("--offgrid", action="store_true",
                    help="cross-check against PCHIP truth at temperatures in "
                         "the widest gaps of the union cache grid, instead of "
                         "at exact cache rows")
    ap.add_argument("--offgrid_paired", action="store_true",
                    help="off-grid temperatures paired one-to-one with the "
                         "on-grid selection (controls for temperature)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="p5")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    datadir = args.datadir or DATADIR
    resp_dir = args.responses_dir or RESP_DIR
    outdir = args.outdir or os.path.join(RESULTS, "p5")
    os.makedirs(outdir, exist_ok=True)

    joint = JointOperatorModel(args.models_dir, device=args.device,
                               elements=args.elements)
    elements = joint.elements
    print(f"{joint}", flush=True)

    common, test_masks, union = common_temperatures(datadir, elements)
    t0 = time.time()
    if args.offgrid or args.offgrid_paired:
        # the union of the cache grids runs slightly past the joint model's
        # validity box (which is the intersection of the per-element boxes),
        # and check_temperature rightly refuses to extrapolate -- so pick
        # off-grid points only from the part both agree on
        t_lo, t_hi = joint.temp_range
        usable = union[(union >= t_lo) & (union <= t_hi)]
        temps_np = (offgrid_paired(usable, select_temperatures(common,
                                                               args.ntemp))
                    if args.offgrid_paired
                    else offgrid_temperatures(usable, args.ntemp))
        held = np.full(len(temps_np), len(elements))   # on no training grid
        gap = np.abs(np.log10(union)[None, :]
                     - np.log10(temps_np)[:, None]).min(axis=1)
        print(f"off-grid mode: {len(temps_np)} temperatures in the widest "
              f"gaps of the {len(union)}-point union grid; nearest cache row "
              f"is {gap.min():.2e}-{gap.max():.2e} in log10 T away",
              flush=True)
        print(f"T = {np.array2string(temps_np, precision=3)}", flush=True)
        truth_native, native_edges = offgrid_truth_flux(datadir, elements,
                                                        temps_np)
    else:
        temps_np = select_temperatures(common, args.ntemp)
        sel = np.searchsorted(common, temps_np)                # (ntemp,)
        # how many of the 30 elements hold each evaluation temperature out
        held = np.sum([test_masks[z][sel] for z in elements], axis=0)
        print(f"{len(common)} temperatures common to all {len(elements)} "
              f"caches; using {len(temps_np)} of them, "
              f"{float(np.mean(held)):.1f}/{len(elements)} elements held out "
              f"on average", flush=True)
        print(f"T = {np.array2string(temps_np, precision=3)}", flush=True)
        truth_native, native_edges = cache_truth_flux(datadir, elements,
                                                      temps_np)
    print(f"truth: {tuple(truth_native.shape)} in "
          f"{time.time() - t0:.0f}s", flush=True)
    truth_native = direct_broaden(truth_native.float(), native_edges.float(),
                                  args.velocity).double()      # (B, n_bins)
    temps = torch.from_numpy(temps_np).float()

    results: Dict[str, dict] = {}
    if "grids" in args.passes:
        print("--- pass 1: resolution-matched grids", flush=True)
        run_grids(joint, truth_native, native_edges, temps, args.velocity,
                  args.oversample, results)
    if "fold" in args.passes:
        print("--- pass 2: RMF x ARF folding", flush=True)
        run_fold(joint, truth_native, native_edges, temps, args.velocity,
                 resp_dir, results,
                 arf_modes=(True,) if args.no_arf_pass else (True, False))
    if "subbin" in args.passes:
        print("--- pass 3: sub-bin phase sweep", flush=True)
        run_subbin(joint, truth_native, native_edges, temps, args.velocity,
                   args.subbin_phases, BANDS["resolve"], resp_dir, results,
                   args.subbin_fold,
                   [c for c in args.subbin_control if c != 0.0])

    out = os.path.join(outdir, f"benchmark_instruments30_{args.tag}.json")
    with open(out, "w") as f:
        json.dump({"results": results,
                   "temps": temps_np.tolist(),
                   "elements": elements,
                   "held_out_count": held.tolist(),
                   "args": vars(args)}, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
