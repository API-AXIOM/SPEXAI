"""Print the P2 normalisation summary from the saved comparison npz files.

Reads what ``validate_spex_norm.py --mode compare/--mode compare-elements``
saved and reports the three numbers the paper quotes: the full-CIE agreement in
the fit band, the per-bin (shape) agreement, and the composition error of the
linear per-element sum.
"""
import argparse
import os

import numpy as np

from spexai.config import RESULTS

FIT_BAND = (1.9, 12.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=os.path.join(
        RESULTS, "spex_norm", "norm_30elem_spexact2_lodders.npz"))
    ap.add_argument("--elements", default=os.path.join(
        RESULTS, "spex_norm", "elements_30_spexact2_lodders.npz"))
    args = ap.parse_args()

    d = np.load(args.full)
    cen, temps, spx, emu = d["centers"], d["temps"], d["flux_spex"], d["flux_emu"]
    lo, hi = FIT_BAND
    m = (cen >= lo) & (cen < hi)
    print(f"full CIE, {len(d['elements'])} elements, "
          f"spexact={d['spexact']} abun={d['abun']}")
    print(f"{'T [keV]':>9} {'integ ratio':>12} {'median bin':>11} "
          f"{'p16':>9} {'p84':>9}   ({lo}-{hi} keV)")
    for i, t in enumerate(temps):
        ok = m & (spx[i] > 0)
        r = emu[i][ok] / spx[i][ok]
        print(f"{t:9.3f} {emu[i][ok].sum() / spx[i][ok].sum():12.5f} "
              f"{np.median(r):11.5f} {np.percentile(r, 16):9.5f} "
              f"{np.percentile(r, 84):9.5f}")

    if os.path.exists(args.elements):
        e = np.load(args.elements)
        cen_e = e["centers"]
        me = (cen_e >= lo) & (cen_e < hi)
        print(f"\ncomposition error of the linear per-element sum "
              f"(SPEX only, {lo}-{hi} keV)")
        for i, t in enumerate(e["temps"]):
            # sum of SPEX's own single-element runs vs the interpolated full CIE
            j = int(np.argmin(np.abs(temps - t)))
            print(f"{t:9.3f} {e['flux_spex'][i].sum(axis=0)[me].sum() / spx[j][me].sum():12.5f}")


if __name__ == "__main__":
    main()
