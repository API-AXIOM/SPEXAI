"""Classical interpolation baselines for the single-element emulator benchmark.

Because the CIE single-element model has a single parameter (temperature),
per-bin interpolation across the training temperatures is the natural
classical competitor to any neural emulator:

  * linear  - per-bin linear interpolation in (log10 T, log10 flux);
              what SPEX/APEC-style codes do between tabulated temperatures
  * pchip   - per-bin monotone cubic (PCHIP), avoids overshoot at sharp
              emissivity peaks
  * pca_gp  - Speculator/Coyote-style classical emulator: PCA on log flux,
              Gaussian process (Matern-5/2) on the component amplitudes

Metrics identical to scripts/benchmark_operator.py (overall / line /
continuum mean relative error and 1% / 10% yields). Use --n-train to
subsample the training temperatures (evenly in log T) for data-efficiency
comparisons.

    python scripts/baseline_interpolation.py [--methods linear pchip pca_gp]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,
                                         SpectrumData, continuum_estimate)
# metrics_from_eps is re-exported here for scripts/benchmark_offgrid.py, which
# imports it from this module; the single definition lives in spexai.train.metrics.
from spexai.train.metrics import metrics_from_eps


def predict_linear(lt_train, Y, lt_test):
    """Per-bin linear interpolation in log10 T (vectorised over bins)."""
    j = np.clip(np.searchsorted(lt_train, lt_test), 1, len(lt_train) - 1)
    w = ((lt_test - lt_train[j - 1]) /
         (lt_train[j] - lt_train[j - 1]))[:, None].astype(np.float32)
    return Y[j - 1] * (1 - w) + Y[j] * w


def predict_pchip(lt_train, Y, lt_test, chunk=2000):
    from scipy.interpolate import PchipInterpolator
    out = np.empty((len(lt_test), Y.shape[1]), dtype=np.float32)
    for lo in range(0, Y.shape[1], chunk):
        hi = min(lo + chunk, Y.shape[1])
        interp = PchipInterpolator(lt_train, Y[:, lo:hi], axis=0, extrapolate=True)
        out[:, lo:hi] = interp(lt_test).astype(np.float32)
    return out


def predict_pca_gp(lt_train, Y, lt_test, ncomp=64, max_gp=2000, seed=42):
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel, Matern

    pca = PCA(n_components=min(ncomp, len(lt_train) - 1, Y.shape[1]),
              svd_solver="randomized", random_state=seed)
    A = pca.fit_transform(Y)
    scale = A.std(axis=0)
    scale[scale == 0] = 1.0
    if len(lt_train) > max_gp:
        sub = np.linspace(0, len(lt_train) - 1, max_gp).astype(int)
    else:
        sub = np.arange(len(lt_train))
    kernel = (Matern(length_scale=0.05, nu=2.5,
                     length_scale_bounds=(1e-3, 1e1)) +
              WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
    gp.fit(lt_train[sub, None], (A[sub] / scale))
    A_pred = gp.predict(lt_test[:, None]) * scale
    return pca.inverse_transform(A_pred).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--methods", nargs="+", default=["linear", "pchip", "pca_gp"])
    ap.add_argument("--n-train", type=int, default=0,
                    help="subsample training temperatures (0 = use all)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    data = SpectrumData(args.cachedir)
    temps = data.temps.numpy()
    idx = (data.test_idx if args.split == "test" else data.val_idx).numpy()

    tr = data.train_idx.numpy()
    tr = tr[np.argsort(temps[tr])]
    if args.n_train and args.n_train < len(tr):
        tr = tr[np.linspace(0, len(tr) - 1, args.n_train).astype(int)]
    lt_train = np.log10(temps[tr].astype(np.float64))
    Y = np.clip(data.logflux[torch.from_numpy(tr)].numpy(), FLOOR, None)
    lt_test = np.log10(temps[idx].astype(np.float64))
    target = data.logflux[torch.from_numpy(idx)].numpy()

    print(f"train temps: {len(tr)}, {args.split} spectra: {len(idx)}")
    cont = continuum_estimate(target)
    valid = target > FLOOR
    line_mask = valid & (np.clip(target, FLOOR, None) - cont > LINE_THRESHOLD_DEX)
    cont_mask = valid & ~line_mask

    predictors = {"linear": predict_linear, "pchip": predict_pchip,
                  "pca_gp": predict_pca_gp}
    results = {}
    for m in args.methods:
        t0 = time.time()
        pred = predictors[m](lt_train, Y, lt_test)
        fit_predict_s = time.time() - t0
        d = np.clip(pred - np.clip(target, FLOOR, None), -4, 4)
        eps = np.abs(10.0 ** d - 1.0)
        results[m] = {
            "overall": metrics_from_eps(eps, valid),
            "lines": metrics_from_eps(eps, line_mask),
            "continuum": metrics_from_eps(eps, cont_mask),
            "n_train": len(tr),
            "fit_predict_s": fit_predict_s,
        }
        r = results[m]
        print(f"{m}: overall MRE={r['overall']['mre_mean']:.5f} "
              f"median={r['overall']['mre_median']:.5f} "
              f"lines={r['lines']['mre_mean']:.5f} "
              f"cont={r['continuum']['mre_mean']:.5f} "
              f"yield1%={r['overall']['yield_1pct']:.2f} "
              f"yield10%={r['overall']['yield_10pct']:.2f} "
              f"({fit_predict_s:.1f}s)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_n{args.n_train}" if args.n_train else ""
    path = os.path.join(args.outdir, f"baselines_{args.split}{tag}.json")
    if os.path.exists(path):
        with open(path) as f:
            results = {**json.load(f), **results}
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
