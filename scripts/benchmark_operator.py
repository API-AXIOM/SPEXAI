"""Benchmark trained emulator variants on the held-out test set.

Overall metrics (mean/median relative error, 1% / 10% yields), separate
line vs continuum metrics, residual-vs-energy distribution, example
spectra, and evaluation-speed timings.

Line/continuum split: per spectrum, the continuum is estimated with a
running median of log10-flux (window ~501 bins on the log-energy grid,
evaluated every 50 bins and interpolated). A bin is a "line" bin when its
flux exceeds twice the local continuum estimate; remaining non-empty bins
count as continuum.

    python scripts/benchmark_operator.py --rundir .../runs/element26
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spexai.train.operator import FixedGridMLP
from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,
                                         SpectrumData, continuum_estimate,
                                         make_variant)


def load_model(ckpt_path, data):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    model = make_variant(ckpt["variant"], data, args)
    # older checkpoints predate the line_energies/line_widths buffers; the
    # freshly built model already computed them from the cache, so it is
    # safe to leave them at their constructed values
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    allowed = {"line_head.line_energies", "line_head.line_widths", "train_energy", "train_edges"}
    if unexpected or not set(missing) <= allowed:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    model.eval()
    return model, ckpt["variant"]


@torch.no_grad()
def predict_full_grid(model, temps, energy, fixed_grid, echunk=4096):
    """Predict all bins for a batch of temperatures, chunking the energy
    grid so peak memory stays bounded (the coordinate embedding is
    batch x points x embed_dim and grows very large otherwise)."""
    if fixed_grid:
        return model(temps)
    parts = [model(temps, energy[lo:lo + echunk], bins=torch.arange(
        lo, min(lo + echunk, len(energy)), device=energy.device))
        for lo in range(0, len(energy), echunk)]
    return torch.cat(parts, dim=1)


@torch.no_grad()
def predict_all(model, data, idx, device, fixed_grid, batch=64):
    energy = data.energy.to(device)
    preds = []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        temps = data.temps[sel].to(device)
        p = predict_full_grid(model, temps, energy, fixed_grid)
        preds.append(p.cpu())
    return torch.cat(preds).numpy()


def metrics_from_eps(eps, mask):
    """Per-spectrum mean rel. error over `mask` bins -> summary dict."""
    cnt = mask.sum(axis=1)
    ok = cnt > 0
    mre = np.where(ok, (eps * mask).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    mre = mre[ok]
    return {
        "n_spectra": int(ok.sum()),
        "mre_mean": float(np.mean(mre)),
        "mre_median": float(np.median(mre)),
        "yield_1pct": float((mre <= 0.01).mean() * 100),
        "yield_10pct": float((mre <= 0.10).mean() * 100),
    }


def speed_benchmark(model, data, device, fixed_grid, nrep=50):
    energy = data.energy.to(device)
    temps = data.temps[:256].to(device)
    sync = (torch.mps.synchronize if device == "mps"
            else torch.cuda.synchronize if device == "cuda" else lambda: None)
    with torch.no_grad():
        for batch in (1, 256):
            t = temps[:batch]
            for _ in range(5):
                _ = predict_full_grid(model, t, energy, fixed_grid)
            sync()
            t0 = time.time()
            for _ in range(nrep):
                _ = predict_full_grid(model, t, energy, fixed_grid)
            sync()
            dt = (time.time() - t0) / nrep
            yield batch, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    idx = data.test_idx if args.split == "test" else data.val_idx
    target = data.logflux[idx].numpy()
    energy = data.energy.numpy()

    print("estimating continua for line/continuum split ...", flush=True)
    cont = continuum_estimate(target)
    valid = target > FLOOR
    line_mask = valid & (np.clip(target, FLOOR, None) - cont > LINE_THRESHOLD_DEX)
    cont_mask = valid & ~line_mask
    print(f"line bins: {line_mask.mean()*100:.1f}%  continuum bins: "
          f"{cont_mask.mean()*100:.1f}%  empty: {(~valid).mean()*100:.1f}%")

    results = {}
    figdir = os.path.join(args.rundir, "figures")
    os.makedirs(figdir, exist_ok=True)

    for ckpt_path in sorted(glob.glob(os.path.join(args.rundir, "*.pt"))):
        model, _ = load_model(ckpt_path, data)
        # key results by file stem so tagged runs (e.g. HPO trials) don't collide
        variant = os.path.splitext(os.path.basename(ckpt_path))[0]
        model = model.to(device)
        fixed_grid = isinstance(model, FixedGridMLP)
        pred = predict_all(model, data, idx, device, fixed_grid)

        d = np.clip(pred - np.clip(target, FLOOR, None), -4, 4)
        eps = np.abs(10.0 ** d - 1.0)

        res = {
            "overall": metrics_from_eps(eps, valid),
            "lines": metrics_from_eps(eps, line_mask),
            "continuum": metrics_from_eps(eps, cont_mask),
            "params": model.count_parameters(),
        }
        for batch, dt in speed_benchmark(model, data, device, fixed_grid):
            res[f"eval_time_batch{batch}_ms"] = dt * 1e3
        results[variant] = res
        print(f"{variant}: overall MRE={res['overall']['mre_mean']:.4f} "
              f"lines={res['lines']['mre_mean']:.4f} "
              f"cont={res['continuum']['mre_mean']:.4f}", flush=True)

        # residual-vs-energy percentile bands (cf. Fig. 6 of Ricketts et al.)
        eps_masked = np.where(valid, eps, np.nan)
        pct = np.nanpercentile(eps_masked, [1, 5, 25, 50, 75, 95, 99], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(energy, pct[0], pct[6], alpha=0.2, label="1-99%")
        ax.fill_between(energy, pct[1], pct[5], alpha=0.3, label="5-95%")
        ax.fill_between(energy, pct[2], pct[4], alpha=0.5, label="25-75%")
        ax.plot(energy, pct[3], lw=0.7, label="median")
        ax.set(xscale="log", yscale="log", xlabel="Energy (keV)",
               ylabel="relative error", title=f"{variant}: residuals vs energy")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"residuals_{variant}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(args.rundir, f"benchmark_{args.split}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # summary table
    rows = ["| Variant | params | overall MRE | line MRE | cont MRE | "
            "yield1% | yield10% | t(1 spec) ms |",
            "|---|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        rows.append(
            f"| {v} | {r['params']:,} | {r['overall']['mre_mean']:.4f} "
            f"| {r['lines']['mre_mean']:.4f} | {r['continuum']['mre_mean']:.4f} "
            f"| {r['overall']['yield_1pct']:.2f} | {r['overall']['yield_10pct']:.2f} "
            f"| {r['eval_time_batch1_ms']:.1f} |")
    md = "\n".join(rows)
    with open(os.path.join(args.rundir, f"benchmark_{args.split}.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
