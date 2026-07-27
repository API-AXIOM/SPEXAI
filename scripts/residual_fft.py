"""Residual and residual-power-spectrum diagnostic across elements.

The failing-band diagnosis blames a broadband high-frequency "jitter" floor
from the Fourier trunk. That jitter lives in the NORMALISED log-energy
coordinate x in [-1, 1] the trunk actually sees (FourierFeatures uses
phase = 2 pi f x, with f in [f_min, f_max]), not in keV. So we:

  1. predict the full training grid for a T-spread subset of the held-out
     test spectra,
  2. form the log10 residual r = pred - truth over the valid (non-floor)
     bins,
  3. resample r onto a UNIFORM x grid (the SPEX grid is adaptive, not
     log-uniform) and take its power spectrum, averaged over spectra,
  4. mark the model's [f_min, f_max] Fourier band.

If a failing element's residual power reaches up toward f_max while the
HIT elements' does not, the floor is trunk jitter (architectural). If the
excess is low-frequency / broadband, it is amplitude (under-convergence),
not jitter. A per-element figure plus a cross-element comparison figure
are written, so HIT vs OFF spectra can be read side by side.

    python scripts/residual_fft.py \
        --dataroot ~/work/data/spexai --runroot ~/work/data/spexai/runs \
        --elements 6 7 8 9 10 11 26 --out docs/development_plots/residual_fft \
        --device cpu
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from benchmark_operator import load_model, predict_all  # noqa: E402
from spexai.train.operator import FixedGridMLP  # noqa: E402
from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,  # noqa: E402
                                         SpectrumData, continuum_estimate)

# Z -> symbol, for titles only
SYMBOLS = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
           8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
           15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc",
           22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe"}

# HIT / OFF grouping from the failing-band survey (see the technical report,
# docs/emulator_technical_report.tex), for colour coding only
HIT = {2, 3, 4, 6, 7, 8, 9, 10, 15, 17}
OFF = {11, 13, 18, 20, 24}


def element_residual_spectrum(
    z: int, cachedir: str, ckpt: str, device: str, n_spec: int, n_fft: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Per-element residual + resampled power spectrum over a T-spread test
    subset. Returns None if inputs are missing."""
    if not (os.path.isdir(cachedir) and os.path.isfile(ckpt)):
        print(f"  Z{z:02d}: skip (cache or ckpt missing)", flush=True)
        return None
    data = SpectrumData(cachedir)
    model, _ = load_model(ckpt, data)
    model = model.to(device)
    fixed_grid = isinstance(model, FixedGridMLP)

    # T-spread subset of the frozen test split (deterministic)
    idx_all = data.test_idx.numpy()
    order = idx_all[np.argsort(data.temps.numpy()[idx_all])]
    sel = order[np.linspace(0, len(order) - 1, min(n_spec, len(order))).astype(int)]
    sel_t = torch.from_numpy(sel).long()

    energy = data.energy.numpy().astype(np.float64)          # (M,) keV, ascending
    # normalised x coordinate the Fourier trunk sees; (M,), ascending in E
    x = model.norm_energy(data.energy.to(device)).cpu().numpy().astype(np.float64)

    pred = predict_all(model, data, sel_t, device, fixed_grid)  # (S, M) log10
    truth = data.logflux[sel_t].numpy()                          # (S, M) log10

    # per-bin line/continuum mask from the SPEX truth (union over the subset)
    cont = continuum_estimate(truth)
    valid = truth > FLOOR
    line_mask = valid & (np.clip(truth, FLOOR, None) - cont > LINE_THRESHOLD_DEX)

    resid = pred - np.clip(truth, FLOOR, None)                   # (S, M) log10
    xu = np.linspace(x.min(), x.max(), n_fft)                    # uniform x grid
    dx = xu[1] - xu[0]
    win = np.hanning(n_fft)
    win_norm = np.sum(win ** 2)
    freqs = np.fft.rfftfreq(n_fft, d=dx)                        # cycles per unit x

    # per-spectrum power spectra (nan row where too few valid/continuum bins),
    # kept so the caller can aggregate over any temperature subset
    pw_full_s = np.full((len(sel), len(freqs)), np.nan)
    pw_cont_s = np.full((len(sel), len(freqs)), np.nan)
    for s in range(len(sel)):
        v = valid[s]
        if v.sum() < 16:
            continue
        # full residual, resampled over valid bins (floor gaps interpolated)
        r_full = np.interp(xu, x[v], resid[s, v])
        ru = (r_full - r_full.mean()) * win
        pw_full_s[s] = np.abs(np.fft.rfft(ru)) ** 2 / win_norm
        # continuum-only: also drop line bins so line spikes don't dominate
        c = v & ~line_mask[s]
        if c.sum() >= 16:
            r_cont = np.interp(xu, x[c], resid[s, c])
            rc = (r_cont - r_cont.mean()) * win
            pw_cont_s[s] = np.abs(np.fft.rfft(rc)) ** 2 / win_norm
    if np.isnan(pw_full_s).all():
        return None

    cfg = model.config
    d = {
        "z": z, "energy": energy, "freqs": freqs,
        "temps": data.temps[sel_t].numpy(),                     # (S,)
        "pw_full_s": pw_full_s, "pw_cont_s": pw_cont_s,          # (S, F)
        "resid": resid, "valid": valid,                         # (S, M)
        "f_min": float(cfg.f_min), "f_max": float(cfg.f_max),
        "n_freqs": int(cfg.n_freqs),
        "line_frac": float(line_mask.any(axis=0).mean()),
    }
    d.update(aggregate(d))          # add whole-subset resid_rms/pw_full/pw_cont
    return d


def aggregate(d: Dict[str, np.ndarray],
              sub: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Aggregate per-spectrum arrays over a subset `sub` of the spectra
    (default all): mean power spectra + per-bin RMS residual."""
    if sub is None:
        sub = np.arange(len(d["temps"]))
    resid, valid = d["resid"][sub], d["valid"][sub]
    bin_valid = valid.sum(axis=0)
    resid_rms = np.sqrt(np.where(valid, resid ** 2, 0.0).sum(axis=0)
                        / np.maximum(bin_valid, 1))
    return {
        "n": int(len(sub)),
        "resid_rms": resid_rms,
        "pw_full": np.nanmean(d["pw_full_s"][sub], axis=0),
        "pw_cont": np.nanmean(d["pw_cont_s"][sub], axis=0),
    }


def _band_lines(ax: plt.Axes, res: Dict[str, np.ndarray]) -> None:
    ax.axvline(res["f_min"], color="#888", ls=":", lw=1)
    ax.axvline(res["f_max"], color="#888", ls="--", lw=1)
    ax.text(res["f_max"], ax.get_ylim()[1], " f_max", color="#888",
            fontsize=8, va="top", ha="left")


def plot_element(res: Dict[str, np.ndarray], out: str) -> None:
    z = res["z"]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 8))
    a0.plot(res["energy"], res["resid_rms"], lw=0.7, color="#0072B2")
    a0.axhline(4e-4, ls=":", color="#D55E00", lw=1)
    a0.text(res["energy"][0], 4.2e-4, "0.1% target (~4e-4 log10)",
            color="#D55E00", fontsize=8, va="bottom")
    a0.set(xscale="log", yscale="log", xlabel="Energy (keV)",
           ylabel="RMS log10 residual",
           title=f"Z{z:02d} {SYMBOLS.get(z, '')}: residual vs energy")
    a0.grid(True, which="major", color="#eee", lw=0.6)

    a1.loglog(res["freqs"][1:], res["pw_full"][1:], lw=0.9, color="#222",
              label="full residual")
    a1.loglog(res["freqs"][1:], res["pw_cont"][1:], lw=0.9, color="#0072B2",
              label="continuum only")
    _band_lines(a1, res)
    a1.set(xlabel="frequency (cycles per unit x)  [x = normalised log10 E]",
           ylabel="mean residual power",
           title=f"Z{z:02d} {SYMBOLS.get(z, '')}: residual power spectrum "
                 f"(n_freqs={res['n_freqs']})")
    a1.legend(frameon=False, loc="upper right")
    a1.grid(True, which="major", color="#eee", lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def plot_stratified(d: Dict[str, np.ndarray], t_edges: List[float],
                    out: str) -> None:
    """Overlay the residual (vs energy) and continuum power spectrum for a
    few temperature bands, to see whether the smooth low-frequency misfit is
    uniform in T or concentrated (e.g. at the cold end)."""
    z = d["z"]
    temps = d["temps"]
    edges = [0.0] + list(t_edges) + [np.inf]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(edges) - 1))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sub = np.nonzero((temps >= lo) & (temps < hi))[0]
        if len(sub) < 3:
            continue
        agg = aggregate(d, sub)
        hs = "" if np.isinf(hi) else f"{hi:g}"
        lab = f"{lo:g}-{hs} keV  (n={agg['n']}, med {np.median(agg['resid_rms']):.1e})"
        a0.plot(d["energy"], agg["resid_rms"], lw=0.8, color=cmap[i], label=lab)
        a1.loglog(d["freqs"][1:], agg["pw_cont"][1:], lw=1.0, color=cmap[i],
                  label=lab)
    a0.axhline(4e-4, ls=":", color="#D55E00", lw=1)
    a0.set(xscale="log", yscale="log", xlabel="Energy (keV)",
           ylabel="RMS log10 residual",
           title=f"Z{z:02d} {SYMBOLS.get(z, '')}: residual vs energy by T")
    a0.grid(True, which="major", color="#eee", lw=0.6)
    a0.legend(frameon=False, fontsize=8)
    _band_lines(a1, d)
    a1.set(xlabel="frequency (cycles per unit x)",
           ylabel="continuum residual power",
           title=f"Z{z:02d} {SYMBOLS.get(z, '')}: continuum power spectrum by T")
    a1.grid(True, which="major", color="#eee", lw=0.6)
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def plot_comparison(results: List[Dict[str, np.ndarray]], out: str) -> None:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
    for res in results:
        z = res["z"]
        col = "#D55E00" if z in OFF else "#0072B2" if z in HIT else "#777"
        lab = f"Z{z:02d} {SYMBOLS.get(z, '')}"
        a0.plot(res["energy"], res["resid_rms"], lw=0.8, color=col, alpha=0.8,
                label=lab)
        a1.loglog(res["freqs"][1:], res["pw_cont"][1:], lw=1.0, color=col,
                  alpha=0.8, label=lab)
    a0.axhline(4e-4, ls=":", color="k", lw=1)
    a0.set(xscale="log", yscale="log", xlabel="Energy (keV)",
           ylabel="RMS log10 residual",
           title="Residual vs energy  (orange = OFF, blue = HIT)")
    a0.grid(True, which="major", color="#eee", lw=0.6)
    a0.legend(frameon=False, fontsize=8, ncol=2)
    # all runs share the Fourier band; take it from the first result
    _band_lines(a1, results[0])
    a1.set(xlabel="frequency (cycles per unit x)", ylabel="continuum residual power",
           title="Continuum residual power spectrum")
    a1.grid(True, which="major", color="#eee", lw=0.6)
    a1.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=os.path.expanduser("~/work/data/spexai"))
    ap.add_argument("--runroot",
                    default=os.path.expanduser("~/work/data/spexai/runs"))
    ap.add_argument("--elements", type=int, nargs="+",
                    default=[6, 7, 8, 9, 10, 11, 26])
    ap.add_argument("--ckpt_name", default="tier1/reweight_full.pt")
    ap.add_argument("--out", default="docs/development_plots/residual_fft")
    ap.add_argument("--n_spec", type=int, default=120,
                    help="T-spread test spectra used per element")
    ap.add_argument("--n_fft", type=int, default=8192,
                    help="uniform-x resample length for the power spectrum")
    ap.add_argument("--device", default="cpu",
                    help="cpu (laptop-safe) | cuda | mps")
    ap.add_argument("--stratify", type=int, default=1,
                    help="also write a per-element residual/FFT split into "
                         "temperature bands (see --t_edges)")
    ap.add_argument("--t_edges", default="1,5",
                    help="comma-separated interior T-band edges in keV "
                         "(default '1,5' -> <1, 1-5, >5 keV)")
    args = ap.parse_args()
    t_edges = [float(v) for v in args.t_edges.split(",") if v.strip()]
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    results = []
    summary = []
    for z in args.elements:
        print(f"Z{z:02d} ...", flush=True)
        cachedir = os.path.join(args.dataroot, "processed", f"element{z}")
        ckpt = os.path.join(args.runroot, f"element{z}", args.ckpt_name)
        res = element_residual_spectrum(z, cachedir, ckpt, args.device,
                                        args.n_spec, args.n_fft)
        if res is None:
            continue
        plot_element(res, os.path.join(args.out, f"Z{z:02d}_residual_fft.png"))
        if args.stratify:
            plot_stratified(res, t_edges,
                            os.path.join(args.out, f"Z{z:02d}_by_temperature.png"))
        # scalar summary: total RMS residual, and high-frequency power share
        hi = res["freqs"] > 0.5 * res["f_max"]
        summary.append({
            "z": z, "symbol": SYMBOLS.get(z, ""),
            "group": "OFF" if z in OFF else "HIT" if z in HIT else "mid",
            "rms_resid_median": float(np.median(res["resid_rms"])),
            "hi_freq_power_frac": float(res["pw_cont"][hi].sum()
                                        / max(res["pw_cont"][1:].sum(), 1e-30)),
        })
        # drop the heavy per-spectrum arrays before retaining for comparison
        for k in ("resid", "valid", "pw_full_s", "pw_cont_s"):
            res.pop(k, None)
        results.append(res)

    if results:
        plot_comparison(results, os.path.join(args.out, "comparison.png"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary (median RMS residual | hi-freq power share):", flush=True)
    for s in summary:
        print(f"  Z{s['z']:02d} {s['symbol']:>2} [{s['group']}]  "
              f"rms={s['rms_resid_median']:.2e}  "
              f"hi_freq={s['hi_freq_power_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
