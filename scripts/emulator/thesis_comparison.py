"""Compare emulators to the Matthijsse MSc thesis on its own metric.

Thesis (Table 6.1, iron): percentage of unmasked temperature-energy POINTS
whose error fraction |F_NN - F_SPEX| / F_SPEX exceeds 1e-3 and 1e-2, with
flux clamped/masked at log10 flux = -10. Best iron model there
(FFN(3x250), "Non Lin." activation): 17.7% above 1e-3, 1.48% above 1e-2.

This script computes the same per-point statistic on our held-out test set
for every checkpoint in --rundir plus the linear/PCHIP interpolation
baselines.

    python scripts/thesis_comparison.py [--rundir DIR] [--cachedir DIR]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.emulator.baseline_interpolation import predict_linear, predict_pchip
from scripts.emulator.benchmark_operator import load_model, predict_all
from spexai.operator import FixedGridMLP
from spexai.train.train_operator import FLOOR
from spexai.data import SpectrumData

THESIS_ROWS = [
    ("thesis Fe FFN(3x250) NonLin", 17.7, 1.48),
    ("thesis combined model (all elements, solar)", 3.12, 0.0813),
]


def per_point_stats(pred, target):
    valid = target > FLOOR
    d = np.clip(pred - np.clip(target, FLOOR, None), -4, 4)
    eps = np.abs(10.0 ** d - 1.0)[valid]
    return (float((eps > 1e-3).mean() * 100),
            float((eps > 1e-2).mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    idx = data.test_idx
    temps = data.temps.numpy()
    target = data.logflux[idx].numpy()

    rows = []

    # interpolation baselines
    tr = data.train_idx.numpy()
    tr = tr[np.argsort(temps[tr])]
    lt_train = np.log10(temps[tr].astype(np.float64))
    Y = np.clip(data.logflux[torch.from_numpy(tr)].numpy(), FLOOR, None)
    lt_test = np.log10(temps[idx.numpy()].astype(np.float64))
    for name, fn in [("linear interp.", predict_linear),
                     ("PCHIP interp.", predict_pchip)]:
        rows.append((name, *per_point_stats(fn(lt_train, Y, lt_test), target)))

    # neural checkpoints
    for ckpt in sorted(glob.glob(os.path.join(args.rundir, "*.pt"))):
        model, _ = load_model(ckpt, data)
        model = model.to(device)
        name = os.path.splitext(os.path.basename(ckpt))[0]
        pred = predict_all(model, data, idx, device,
                           isinstance(model, FixedGridMLP))
        rows.append((name, *per_point_stats(pred, target)))

    print("\nPer-point error-fraction statistics (thesis metric, Eq. 6.4):")
    print("| model | % points > 0.1% err | % points > 1% err |")
    print("|---|---|---|")
    for name, p3, p2 in THESIS_ROWS:
        print(f"| {name} | {p3:.4g} | {p2:.4g} |")
    for name, p3, p2 in rows:
        print(f"| {name} | {p3:.4g} | {p2:.4g} |")

    out = {n: {"pct_above_1e-3": a, "pct_above_1e-2": b} for n, a, b in rows}
    with open(os.path.join(args.rundir, "thesis_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
