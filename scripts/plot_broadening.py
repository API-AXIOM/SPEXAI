"""Visual comparison of the broadening pipelines.

One test temperature, one velocity. Four curves:
  exact     - true SPEX spectrum, exact erf-integral broadening (reference)
  exact+FFT - true SPEX spectrum, FFT broadening (isolates the convolution)
  emu+FFT   - line-head emulator prediction, FFT broadening (option 1 as
              deployed end-to-end)
  hybrid    - FFT-broadened emulator trunk + analytic Gaussian lines
              (option 3)

Top panel: full band. Bottom: zooms on the strongest line in each of
three sub-bands so line cores/wings can be compared.

    python scripts/plot_broadening.py [--temp 3.0] [--velocity 300]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_operator import load_model, predict_full_grid
from spexai.train.broadening import (broaden_native, direct_broaden,
                                     hybrid_broadened_on_grid, rebin_flux,
                                     uniform_log_edges)
from spexai.train.operator import OperatorConfig, SpectralOperator, \
    edges_from_centers
from spexai.train.train_operator import SpectrumData

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
STYLE = {  # color, linestyle, linewidth, zorder
    "exact":     (INK,       "-",  1.8, 4),
    "exact+FFT": ("#2a78d6", "--", 1.3, 5),
    "emu+FFT":   ("#eda100", "-",  1.1, 3),
    "hybrid":    ("#4a3aa7", "-",  1.1, 2),
    "emu(T,v)":  ("#c23b2a", "-",  1.1, 1),
}


def load_broadened(path, edges):
    """(T, v) emulator of broadened spectra (train_broadened checkpoint);
    returns model plus its uniform grid and velocity normalisation."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OperatorConfig(**b["config"])
    uni = uniform_log_edges(float(edges[0]), float(edges[-1]),
                            b["args"]["dlx"])
    uni_cen = torch.sqrt(uni[:-1] * uni[1:])
    stats = ((b["state_dict"]["bn_mu"], b["state_dict"]["bn_sigma"])
             if cfg.use_binnorm else None)
    model = SpectralOperator(cfg, energy_grid=uni_cen, bin_stats=stats)
    model.load_state_dict(b["state_dict"])
    return model.eval(), uni, uni_cen, b["args"]["vmin"], b["args"]["vmax"]


def predict_broadened(model, uni, uni_cen, vmin, vmax, T, v,
                      edges, chunk=8192):
    """Native-grid integrated flux predicted by the (T, v) emulator."""
    tn = model.norm_temp(T)
    lv = torch.log10(torch.tensor(float(v)))
    vn = 2.0 * (lv - np.log10(vmin)) / np.log10(vmax / vmin) - 1.0
    theta = torch.stack([tn, vn.expand(len(T))], dim=1)
    K = len(uni_cen)
    x = model.norm_energy(uni_cen).view(1, -1, 1)
    dens_u = torch.cat(
        [torch.pow(10.0, model.forward_norm(
            theta, x[:, lo:lo + chunk].expand(len(T), -1, -1),
            bins=torch.arange(lo, min(lo + chunk, K))))
         for lo in range(0, K, chunk)], dim=1)
    return rebin_flux(dens_u * (uni[1:] - uni[:-1]), uni, edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--ckpt", default=None,
                    help="line-head checkpoint (default <rundir>/line_head.pt)")
    ap.add_argument("--broadened_ckpt", default=None,
                    help="(T,v) emulator checkpoint (default "
                         "<rundir>/broadened/broadened.pt; skipped if absent)")
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--velocity", type=float, default=300.0)
    args = ap.parse_args()

    data = SpectrumData(args.cachedir)
    edges = edges_from_centers(data.energy)
    widths = edges[1:] - edges[:-1]
    energy = data.energy.numpy()

    # test spectrum closest to the requested temperature
    i = data.test_idx[torch.argmin(
        (data.temps[data.test_idx] - args.temp).abs())]
    T = data.temps[i:i + 1]
    flux_true = torch.pow(10.0, torch.clamp(data.logflux[i:i + 1],
                                            min=-30)) * widths
    v = args.velocity

    model, _ = load_model(args.ckpt or os.path.join(args.rundir,
                                                    "line_head.pt"), data)

    with torch.no_grad():
        curves = {}
        curves["exact"] = direct_broaden(flux_true, edges, v)
        curves["exact+FFT"] = broaden_native(flux_true, edges, v)
        pred = predict_full_grid(model, T, data.energy, fixed_grid=False)
        flux_emu = torch.pow(10.0, pred) * widths
        curves["emu+FFT"] = broaden_native(flux_emu, edges, v)
        curves["hybrid"] = hybrid_broadened_on_grid(model, T, v, edges)
        bckpt = args.broadened_ckpt or os.path.join(args.rundir, "broadened",
                                                    "broadened.pt")
        if os.path.exists(bckpt):
            curves["emu(T,v)"] = predict_broadened(
                *load_broadened(bckpt, edges), T, v, edges)
    dens = {k: (c / widths).squeeze(0).numpy() for k, c in curves.items()}

    # strongest line (by head amplitude at this T) in each sub-band
    lh = model.line_head
    with torch.no_grad():
        tnorm = model.norm_temp(T).view(-1, model.config.n_params)
        amp = lh.all_line_amplitudes(tnorm).squeeze(0)
    line_e = lh.line_energies.numpy()
    zoom_lines = []
    for lo, hi in [(0.5, 1.2), (1.2, 4.0), (5.5, 9.0)]:
        band = (line_e >= lo) & (line_e <= hi)
        j = np.flatnonzero(band)[amp[band].argmax()]
        zoom_lines.append(float(line_e[j]))

    fig = plt.figure(figsize=(11, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.9, 0.55],
                          hspace=0.42, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, :])
    for name, d in dens.items():
        c, ls, lw, z = STYLE[name]
        ax0.plot(energy, np.clip(d, 1e-12, None), color=c, ls=ls, lw=0.7,
                 zorder=z, label=name)
    ax0.set(xscale="log", yscale="log", xlim=(0.1, 12),
            xlabel="Energy (keV)", ylabel="flux density")
    ax0.set_title(f"Fe, T = {float(T):.2f} keV, v = {v:.0f} km/s",
                  loc="left", fontsize=11, color=INK)
    ax0.legend(loc="lower left", fontsize=9, ncols=4, frameon=False)

    sigma_frac = v / 299792.458
    ref = dens["exact"]
    for k, e0 in enumerate(zoom_lines):
        axp = fig.add_subplot(gs[1, k])
        axr = fig.add_subplot(gs[2, k], sharex=axp)
        half = 12.0 * sigma_frac * e0
        sel = (energy > e0 - half) & (energy < e0 + half)
        for name, d in dens.items():
            c, ls, lw, z = STYLE[name]
            axp.plot(energy[sel], np.clip(d[sel], 1e-12, None),
                     color=c, ls=ls, lw=lw, zorder=z)
            if name != "exact":
                ratio = 100.0 * (d[sel] / np.clip(ref[sel], 1e-30, None) - 1.0)
                axr.plot(energy[sel], ratio, color=c, ls=ls, lw=lw, zorder=z)
        axp.set(yscale="log")
        axp.set_title(f"line at {e0:.3f} keV", fontsize=9, color=INK2)
        axp.tick_params(labelsize=8, labelbottom=False)
        axr.axhline(0.0, color=INK2, lw=0.6, ls=":")
        axr.set(xlabel="Energy (keV)")
        span = np.percentile(np.abs(np.concatenate(
            [100.0 * (d[sel] / np.clip(ref[sel], 1e-30, None) - 1.0)
             for n_, d in dens.items() if n_ != "exact"])), 98)
        axr.set_ylim(-1.2 * span, 1.2 * span)
        axr.tick_params(labelsize=8)
        if k == 0:
            axp.set_ylabel("flux density", fontsize=9)
            axr.set_ylabel("vs exact (%)", fontsize=9)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        for s in ax.spines.values():
            s.set_visible(False)

    out = os.path.join(args.rundir, "figures",
                       f"broadening_T{float(T):.2f}_v{v:.0f}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
