"""Confirmation benchmark on fresh, unseen off-grid spectra.

Samples temperatures log-uniformly inside the training range (a NEW set
on every invocation via --seed), generates reference spectra by per-bin
PCHIP interpolation over the TRAINING split only (interpolation error
~0.003%, two orders of magnitude below emulator error, so it serves as
truth), and scores checkpoints with the same metrics as
benchmark_operator. Use this instead of the frozen SPEX test split for
repeated confirmation runs, so the test set is never consulted during
development.

    python scripts/benchmark_offgrid.py --ckpts <a>.pt <b>.pt --seed 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.baseline_interpolation import metrics_from_eps
from scripts.benchmark_operator import load_model, predict_all
from spexai.train.train_operator import (FLOOR, LINE_THRESHOLD_DEX,
                                         SpectrumData, continuum_estimate)


def pchip_truth(lt_train, Y, lt_new, chunk=2000):
    from scipy.interpolate import PchipInterpolator
    out = np.empty((len(lt_new), Y.shape[1]), dtype=np.float32)
    for lo in range(0, Y.shape[1], chunk):
        hi = min(lo + chunk, Y.shape[1])
        out[:, lo:hi] = PchipInterpolator(
            lt_train, Y[:, lo:hi], axis=0)(lt_new).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--n", type=int, default=0,
                    help="number of probe spectra (default: test-split size)")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: drawn from the clock (fresh set each run)")
    ap.add_argument("--outdir", default=None,
                    help="default: directory of the first checkpoint")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | mps (default: auto; use cpu on the "
                         "laptop -- full-grid MPS evals can freeze it)")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = np.random.default_rng(seed)
    data = SpectrumData(args.cachedir)
    n = args.n or len(data.test_idx)

    tr = data.train_idx.numpy()
    temps_np = data.temps.numpy().astype(np.float64)
    tr = tr[np.argsort(temps_np[tr])]
    lt = np.log10(temps_np[tr])
    Y = np.clip(data.logflux[torch.from_numpy(tr)].numpy(), FLOOR, None)

    lt_new = np.sort(rng.uniform(lt[0] + 1e-6, lt[-1] - 1e-6, size=n))
    print(f"off-grid probe: n={n} seed={seed} "
          f"T in [{10**lt_new[0]:.3f}, {10**lt_new[-1]:.3f}] keV", flush=True)
    t0 = time.time()
    truth = pchip_truth(lt, Y, lt_new)
    print(f"PCHIP truth generated ({time.time()-t0:.0f}s)", flush=True)

    valid = truth > FLOOR
    cont = continuum_estimate(truth)
    line_mask = valid & (truth - cont > LINE_THRESHOLD_DEX)
    cont_mask = valid & ~line_mask
    probe_temps = torch.from_numpy((10.0 ** lt_new).astype(np.float32))

    results = {"seed": seed, "n": n, "models": {}}
    for ck in args.ckpts:
        tag = os.path.basename(ck).replace(".pt", "")
        model, variant = load_model(ck, data)
        model = model.to(device)

        class _Probe:  # duck-typed container for predict_all
            energy, logflux, temps = data.energy, None, probe_temps
        with torch.no_grad():
            pred = predict_all(model, _Probe, torch.arange(n), device,
                               fixed_grid=variant == "fixed_grid")
        d = np.clip(pred - truth, -4, 4)
        eps = np.abs(10.0 ** d - 1.0)
        r = {"overall": metrics_from_eps(eps, valid),
             "lines": metrics_from_eps(eps, line_mask),
             "continuum": metrics_from_eps(eps, cont_mask)}
        results["models"][tag] = r
        o = r["overall"]
        print(f"{tag}: MRE={o['mre_mean']:.5f} median={o['mre_median']:.5f} "
              f"yield1%={o['yield_1pct']:.2f} yield0.1%={o['yield_01pct']:.2f} "
              f"lines={r['lines']['mre_mean']:.5f} "
              f"cont={r['continuum']['mre_mean']:.5f}", flush=True)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.ckpts[0]))
    path = os.path.join(outdir, f"offgrid_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
