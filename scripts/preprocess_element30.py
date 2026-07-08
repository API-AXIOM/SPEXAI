"""Parse raw SPEX spectra for one element into a compact training cache.

Reads all Z<elem>_*keV.txt files (columns: energy [keV], log10 flux),
restricts to [emin, emax], and stores float32 arrays:

    energy.npy   (nbins,)            energy grid in keV
    temps.npy    (nspec,)            temperature in keV (from filename)
    logflux.npy  (nspec, nbins)      log10 flux
    splits.npz   train/val/test index arrays (81/9/10, seeded)

Usage:
    python scripts/preprocess_element30.py [--datadir DIR] [--outdir DIR]
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="/Users/danielahuppenkothen/work/data/spexai/element30")
    ap.add_argument("--outdir", default="/Users/danielahuppenkothen/work/data/spexai/processed/element30")
    ap.add_argument("--emin", type=float, default=0.1)
    ap.add_argument("--emax", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.datadir, "Z*_*keV.txt")))
    pattern = re.compile(r"Z\d+_([0-9.]+)keV\.txt$")
    files = [f for f in files if pattern.search(f)]
    temps = np.array([float(pattern.search(f).group(1)) for f in files], dtype=np.float32)
    print(f"{len(files)} spectra, T in [{temps.min():.4g}, {temps.max():.4g}] keV")

    # shared energy grid from the first file
    d0 = pd.read_csv(files[0], sep=r"\s+", names=["e", "f"], engine="c", dtype=np.float64)
    energy = d0["e"].values
    band = (energy >= args.emin) & (energy <= args.emax)
    lo, hi = np.flatnonzero(band)[[0, -1]]
    energy = energy[lo:hi + 1].astype(np.float32)
    nbins = len(energy)
    print(f"{nbins} bins in [{args.emin}, {args.emax}] keV")

    os.makedirs(args.outdir, exist_ok=True)
    logflux = np.lib.format.open_memmap(
        os.path.join(args.outdir, "logflux.npy"), mode="w+",
        dtype=np.float32, shape=(len(files), nbins))

    t0 = time.time()
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep=r"\s+", names=["e", "f"], engine="c",
                        dtype=np.float64, skiprows=lo, nrows=nbins)
        logflux[i] = d["f"].values
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    logflux.flush()

    np.save(os.path.join(args.outdir, "energy.npy"), energy)
    np.save(os.path.join(args.outdir, "temps.npy"), temps)

    # 81/9/10 split as in Ricketts et al.: 90/10 train/test, train split 90/10 again
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files))
    ntest = int(round(0.1 * len(files)))
    test = perm[:ntest]
    rest = perm[ntest:]
    nval = int(round(0.1 * len(rest)))
    val, train = rest[:nval], rest[nval:]
    np.savez(os.path.join(args.outdir, "splits.npz"), train=train, val=val, test=test)
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")
    print(f"done in {time.time()-t0:.0f}s -> {args.outdir}")


if __name__ == "__main__":
    main()
