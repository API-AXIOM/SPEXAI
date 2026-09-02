"""Check the per-element TRAINING spectra against today's SPEX, element by element.

The script that generated the training spectra is lost, so the SPEX settings
they were made with (SPEXACT version, abundance table, how a single element was
isolated) are not recorded anywhere. They can be recovered empirically: dump
single-element CIE spectra with ``validate_spex_norm.py --mode spex --elements
... --keep-h`` and compare them here against the preprocessed cache rows.

Recovered 2026-09-01 (P2), and assumed by this script:
  * the cache holds **log10 flux density** (ph/s/m^2/keV), not per-bin flux --
    the per-bin ratio is 1.00000 under that reading and ~2000 under the other;
  * element Z alone = ``CIE(H=1, Z=1) - CIE(H=1)``, since zeroing hydrogen is a
    fatal SPEX error (no electrons);
  * SPEX 3.08.01 / SPEXACT 2.07.00 with Lodders et al. (2009) abundances
    reproduces oxygen and sulphur bin for bin to 1e-5.

This is a provenance check, NOT an emulator check: no trained model is loaded.
A mismatch here means the training data itself cannot be regenerated with the
current SPEX, which is a stronger problem than emulator error and has to be
settled before any independent SPEX truth (P3) is trusted.

Usage (either conda env -- numpy only):
    python scripts/inference/check_training_provenance.py \
        --spex /tmp/spex_elem.npz --temp 4.0 --datadir ~/work/data/spexai/processed
"""
import argparse
import os

import numpy as np

SYMBOL = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
          9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
          16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
          23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
          30: "Zn"}


def element_alone(dump, ti, j, z):
    """SPEX flux for element z alone at temperature index ti, ph/bin.

    Hydrogen's own row is already element-alone; every other row carries H's
    continuum and has the H-only run at the SAME temperature subtracted.
    """
    f = dump["flux"][ti, j]
    return f if z == 1 else f - dump["flux_h"][ti, j]


def cache_row(datadir, z, temp):
    """(density, T) of the training row nearest ``temp`` for element z."""
    t = np.load(os.path.join(datadir, f"element{z}", "temps.npy"))
    row = int(np.argmin(np.abs(t - temp)))
    lf = np.load(os.path.join(datadir, f"element{z}", "logflux.npy"),
                 mmap_mode="r")
    return 10.0 ** np.asarray(lf[row], dtype=np.float64), float(t[row]), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spex", required=True,
                    help="npz from validate_spex_norm.py --mode spex --elements")
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    ap.add_argument("--temp", type=float, default=4.0,
                    help="which dumped temperature to check")
    ap.add_argument("--band", default="3,8", help="comparison band in keV")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="flag elements whose median bin ratio deviates by more")
    ap.add_argument("--top", type=int, default=0,
                    help="also list the N bins contributing most of the "
                         "band-flux difference, for flagged elements")
    args = ap.parse_args()

    d = np.load(args.spex)
    if not bool(d["keep_h"]):
        raise SystemExit("--spex dump must have been made with --keep-h")
    zs = [int(z) for z in d["elements"]]
    cen, edges = d["centers"], d["edges"]
    widths = np.diff(edges)
    ti = int(np.argmin(np.abs(d["temps"] - args.temp)))
    lo, hi = (float(x) for x in args.band.split(","))
    m = (cen >= lo) & (cen < hi)
    print(f"SPEX dump T={d['temps'][ti]:.5f} keV, spexact={d['spexact']}, "
          f"abun={d['abun']}; band {lo}-{hi} keV; cache={args.datadir}")
    print(f"{'Z':>3} {'el':>3} {'T_cache':>9} {'band':>9} {'p16':>9} "
          f"{'median':>9} {'p84':>9} {'net/H':>9}")

    flagged = []
    for j, z in enumerate(zs):
        spx = element_alone(d, ti, j, z)                     # ph/bin
        dens, tc, _ = cache_row(args.datadir, z, d["temps_used"][ti, j])
        if abs(tc - d["temps_used"][ti, j]) > 1e-9:
            # the SPEX run and the cache row must be at the same temperature:
            # at kT=0.7 keV a 2.6e-4 relative offset already moves the
            # 1.9-12 keV continuum by 0.2%, more than the effect being measured
            print(f"{z:3d} {SYMBOL[z]:>3}   SKIPPED: dump ran at "
                  f"{d['temps_used'][ti, j]:.5f} keV, cache row is at "
                  f"{tc:.5f} keV (rerun the dump with --match-cache)")
            continue
        model = dens * widths                                # ph/bin
        ok = m & (spx > 0)
        if not ok.sum():
            print(f"{z:3d} {SYMBOL[z]:>3} {tc:9.5f}   (no positive SPEX flux "
                  f"in band -- H-subtraction cancellation)")
            continue
        r = model[ok] / spx[ok]
        med = float(np.median(r))
        h = d["flux_h"][ti, j]
        print(f"{z:3d} {SYMBOL[z]:>3} {tc:9.5f} "
              f"{model[ok].sum() / spx[ok].sum():9.5f} "
              f"{np.percentile(r, 16):9.5f} {med:9.5f} "
              f"{np.percentile(r, 84):9.5f} {spx[m].sum() / h[m].sum():9.2e}")
        if abs(med - 1.0) > args.tol:
            flagged.append((z, med, model, spx, ok))

    if flagged:
        print(f"\nFLAGGED (median bin ratio off by > {args.tol:g}): "
              + ", ".join(f"{SYMBOL[z]} {med:.4f}" for z, med, *_ in flagged))
    else:
        print(f"\nAll {len(zs)} elements reproduce to within {args.tol:g}.")

    for z, med, model, spx, ok in flagged[:args.top and len(flagged)]:
        diff = np.where(ok, model - spx, 0.0)
        idx = np.argsort(-np.abs(diff))[:args.top]
        tot = diff.sum()
        print(f"\n{SYMBOL[z]}: bins driving the band-flux difference "
              f"(total {tot:+.3e} ph/bin/s/m2)")
        for j in sorted(idx, key=lambda j: cen[j]):
            print(f"   {cen[j]:8.4f} keV  cache {model[j]:.4e}  "
                  f"spex {spx[j]:.4e}  ratio {model[j] / spx[j]:8.4f}  "
                  f"{100 * diff[j] / tot:6.1f}% of the difference")


if __name__ == "__main__":
    main()
