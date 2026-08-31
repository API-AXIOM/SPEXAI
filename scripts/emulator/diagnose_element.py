"""Temperature-resolved error diagnostic for a trained element.

Computes per-spectrum line / continuum / overall MRE on the held-out set and
plots each against temperature, so you can see whether the error is in the
lines, the continuum, or overall, and whether it is worse at particular
temperatures.

    python scripts/diagnose_element.py \
        --cachedir ~/data/spexai_data/processed/element11 \
        --ckpt     ~/data/spexai_data/runs/.../element11/tier1/reweight_full.pt \
        --out      element11_diag.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "emulator"))  # for benchmark_operator

from benchmark_operator import load_model, predict_all
from spexai.operator import FixedGridMLP
from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,
                                         continuum_estimate)
from spexai.data import SpectrumData

COLORS = {"overall": "#222222", "continuum": "#0072B2", "lines": "#D55E00"}


def per_spectrum_mre(eps, mask):
    return np.where(mask, eps, 0.0).sum(1) / np.maximum(mask.sum(1), 1)


def binned_median(x, y, nbins=20):
    edges = np.geomspace(x.min(), x.max(), nbins + 1)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 3:
            cx.append(np.sqrt(lo * hi))
            cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default="element_diag.png")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    data = SpectrumData(args.cachedir)
    model, _ = load_model(args.ckpt, data)
    model = model.to(device)
    fixed_grid = isinstance(model, FixedGridMLP)
    idx = data.test_idx if args.split == "test" else data.val_idx
    temps = data.temps[idx].numpy()
    truth = data.logflux[idx].numpy()
    pred = predict_all(model, data, idx, device, fixed_grid)

    # per-bin line/continuum masks from the SPEX truth
    cont = continuum_estimate(truth)
    valid = truth > FLOOR
    line_mask = valid & (np.clip(truth, FLOOR, None) - cont > LINE_THRESHOLD_DEX)
    cont_mask = valid & ~line_mask

    d = np.clip(pred - np.clip(truth, FLOOR, None), -4, 4)
    eps = np.abs(10.0 ** d - 1.0)
    mre = {"overall": per_spectrum_mre(eps, valid),
           "lines": per_spectrum_mre(eps, line_mask),
           "continuum": per_spectrum_mre(eps, cont_mask)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in ("continuum", "lines", "overall"):
        y = np.clip(mre[k], 1e-6, None)
        ax.scatter(temps, y, s=4, alpha=0.12, color=COLORS[k], linewidths=0)
        cx, cy = binned_median(temps, y)
        ax.plot(cx, cy, color=COLORS[k], lw=2.2,
                label=f"{k}  (median {np.median(mre[k]):.4f})")
    ax.axhline(0.01, ls="--", color="#999", lw=1)
    ax.axhline(0.001, ls=":", color="#999", lw=1)
    ax.text(temps.min(), 0.0105, "1%", color="#999", fontsize=9)
    ax.text(temps.min(), 0.00105, "0.1%", color="#999", fontsize=9)
    ax.set(xscale="log", yscale="log", xlabel="Temperature (keV)",
           ylabel="per-spectrum MRE",
           title=f"{os.path.basename(args.cachedir)}: error vs temperature "
                 f"(median lines over faint per-spectrum points)")
    ax.grid(True, which="major", color="#eee", lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    for k in ("overall", "lines", "continuum"):
        y = mre[k]
        worst_T = temps[y > np.percentile(y, 90)]
        print(f"  {k:10}: median={np.median(y):.4f}  p90={np.percentile(y, 90):.4f}"
              f"  worst-10% spectra at T=[{worst_T.min():.2f}, {worst_T.max():.2f}] keV")


if __name__ == "__main__":
    main()
