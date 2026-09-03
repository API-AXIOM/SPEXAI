"""Inspect a run of consecutive cache rows around one temperature.

Written to answer: when two rows disagree, is it a bad row (defect), a step
(generation-batch boundary), or a steep-but-smooth feature (physics that the
training split simply undersamples)? The three look identical if you only
compare two rows, and they have completely different consequences:

  DEFECT    one row is an outlier against BOTH neighbours -- bad data, drop or
            regenerate it.
  STEP      every row below sits at one level and every row above at another --
            a change of generation settings partway through the grid.
  RAMP      the rows in between trace a smooth path -- real temperature
            dependence, and any interpolation error is a statement about grid
            density, not data quality.

Crucially this walks the FULL cache, including val/test rows. Two rows adjacent
in the training split are usually NOT adjacent in the cache, so a gap that looks
like a discontinuity to the PCHIP truth may be well sampled underneath.

Usage:
    python scripts/inference/inspect_cache_neighbourhood.py --z 28 --temp 1.194 \
        [--rows 24] [--band 1.9,12] [--datadir DIR]
"""
import argparse
import os

import numpy as np

SYMBOL = {1: "H", 8: "O", 26: "Fe", 28: "Ni", 29: "Cu", 25: "Mn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--rows", type=int, default=24,
                    help="consecutive cache rows to show, centred on --temp")
    ap.add_argument("--band", default="1.9,12")
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    args = ap.parse_args()

    d = os.path.join(args.datadir, f"element{args.z}")
    t = np.load(os.path.join(d, "temps.npy")).astype(np.float64)
    tr = set(np.load(os.path.join(d, "splits.npz"))["train"].tolist())
    lf = np.load(os.path.join(d, "logflux.npy"), mmap_mode="r")
    cen = np.load(os.path.join(d, "energy.npy")).astype(np.float64)
    lo, hi = (float(x) for x in args.band.split(","))
    m = (cen >= lo) & (cen < hi)

    order = np.argsort(t)
    ts = t[order]
    k0 = int(np.searchsorted(ts, args.temp))
    a = max(0, k0 - args.rows // 2)
    b = min(len(ts), a + args.rows)

    print(f"Z={args.z} ({SYMBOL.get(args.z, '?')}), {b - a} consecutive CACHE "
          f"rows around {args.temp} keV, band {lo}-{hi} keV")
    print(f"{'row':>7} {'T [keV]':>10} {'split':>6} {'band flux':>12} "
          f"{'ratio to prev':>14} {'dlnF/dlnT':>10}")
    prev_f = prev_t = None
    fluxes, temps_used, splits = [], [], []
    for k in range(a, b):
        idx = int(order[k])
        row = np.asarray(lf[idx], dtype=np.float64)
        f = float(np.sum(10.0 ** row[m]))
        split = "train" if idx in tr else "val/test"
        cell = slope = ""
        if prev_f is not None and prev_f > 0:
            r = f / prev_f
            cell = f"{r:14.5f}"
            dlnt = np.log(ts[k]) - np.log(prev_t)
            slope = f"{np.log(r) / dlnt:10.1f}" if dlnt else f"{'inf':>10}"
        print(f"{idx:7d} {ts[k]:10.5f} {split:>6} {f:12.5e} {cell} {slope}")
        fluxes.append(f)
        temps_used.append(ts[k])
        splits.append(split)
        prev_f, prev_t = f, ts[k]

    f = np.asarray(fluxes)
    tt = np.asarray(temps_used)
    # A defect shows up as a row far from the LOCAL TREND. Fit a straight line
    # in log-log through the window (robust to a steep but smooth ramp) and
    # look at residuals: a ramp leaves small residuals, a bad row leaves one
    # large isolated residual, a step leaves two blocks of opposite sign.
    ok = f > 0
    if ok.sum() > 3:
        p = np.polyfit(np.log(tt[ok]), np.log(f[ok]), 1)
        res = np.log(f[ok]) - np.polyval(p, np.log(tt[ok]))
        print(f"\nlocal power-law fit: F ~ T^{p[0]:.2f}; "
              f"residual RMS {np.std(res):.3%}, max |residual| "
              f"{np.max(np.abs(res)):.3%}")
        j = int(np.argmax(np.abs(res)))
        print(f"largest deviation: row at T={tt[ok][j]:.5f} keV "
              f"({splits[int(np.flatnonzero(ok)[j])]}), {res[j]:+.2%}")
        sign = np.sign(res)
        flips = int(np.sum(sign[1:] != sign[:-1]))
        print(f"residual sign changes across the window: {flips} "
              f"(a ramp gives many, a step gives one, a spike gives two)")


if __name__ == "__main__":
    main()
