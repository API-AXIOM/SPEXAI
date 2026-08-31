"""Parse raw SPEX spectra for one element into a compact training cache.

Reads all Z<elem>_*keV.txt files (columns: energy [keV], log10 flux),
restricts to [emin, emax], and stores float32 arrays:

    energy.npy   (nbins,)            energy grid in keV
    temps.npy    (nspec,)            temperature in keV (from filename)
    logflux.npy  (nspec, nbins)      log10 flux
    splits.npz   train/val/test index arrays (81/9/10, seeded)

Usage:
    python scripts/preprocess_spectra.py [--datadir DIR] [--outdir DIR]
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
    ap.add_argument("--datadir", default="/Users/danielahuppenkothen/work/data/spexai/element_26")
    ap.add_argument("--outdir", default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--emin", type=float, default=0.1)
    ap.add_argument("--emax", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-bad", action="store_true",
                    help="drop files whose grid does not match file 0 "
                         "(empty/truncated spectra) instead of aborting")
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
    w = 0            # write cursor: valid rows are packed at the front
    bad = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep=r"\s+", names=["e", "f"], engine="c",
                        dtype=np.float64, skiprows=lo, nrows=nbins)
        vals = d["f"].values
        if len(vals) != nbins:
            # empty or truncated spectrum: its grid does not match file 0
            bad.append((os.path.basename(f), len(vals)))
            continue
        logflux[w] = vals
        temps[w] = temps[i]
        w += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    logflux.flush()

    if bad:
        preview = ", ".join(f"{n}({r} rows)" for n, r in bad[:6])
        msg = (f"{len(bad)} file(s) do not match the {nbins}-bin grid of "
               f"{os.path.basename(files[0])}: {preview}"
               f"{' ...' if len(bad) > 6 else ''}")
        if not args.skip_bad:
            raise SystemExit(
                "ERROR: " + msg + "\n  These are empty/truncated spectra. "
                "Regenerate them, or rerun with --skip-bad to drop them.")
        print(f"WARNING: dropping {len(bad)} bad file(s): {preview}", flush=True)

    # keep only the valid rows (packed at the front)
    temps = temps[:w]
    if w != len(files):
        keep = np.array(logflux[:w])                 # w x nbins into RAM
        np.save(os.path.join(args.outdir, "logflux.npy"), keep)
    np.save(os.path.join(args.outdir, "energy.npy"), energy)
    np.save(os.path.join(args.outdir, "temps.npy"), temps)

    # 81/9/10 split as in Ricketts et al.: 90/10 train/test, train split 90/10 again
    # (over the valid spectra only, after any bad files were dropped)
    nspec = len(temps)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(nspec)
    ntest = int(round(0.1 * nspec))
    test = perm[:ntest]
    rest = perm[ntest:]
    nval = int(round(0.1 * len(rest)))
    val, train = rest[:nval], rest[nval:]
    np.savez(os.path.join(args.outdir, "splits.npz"), train=train, val=val, test=test)
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")
    print(f"done in {time.time()-t0:.0f}s -> {args.outdir}")


if __name__ == "__main__":
    main()
