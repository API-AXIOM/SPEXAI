"""Row-to-row consistency of the training cache.

CIE emissivity is smooth in temperature, so two adjacent training rows a few
tenths of a percent apart in T should agree to well under a percent. Where they
do not, the cache is internally inconsistent, and NO interpolator -- PCHIP
truth or trained network -- can do better than that inconsistency. This is
therefore a floor on the achievable accuracy, and it is a property of the data
rather than of any model.

Found 2026-09-02: Ni has adjacent rows near 1.19 keV that disagree by 24% over
a 0.3% step in temperature, which is what makes Ni's apparent "interpolation
error" the largest in the P3 comparison.

Usage:
    python scripts/inference/scan_cache_jumps.py [--datadir DIR] [--top 12]
"""
import argparse
import os

import numpy as np

BAND = (1.9, 12.0)
SYMBOL = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
          9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
          16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
          23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
          30: "Zn"}


def scan_element(datadir, z, m, windows, width):
    """Consistency between TRULY ADJACENT rows, sampled in contiguous windows.

    Striding through the grid would compare rows far apart in temperature and
    measure real emissivity variation instead. Each jump is normalised by its
    own temperature step, giving a dimensionless d(ln F)/d(ln T): smooth CIE
    emissivity is O(1), so a value of 50+ means the cache disagrees with itself
    between neighbouring rows.
    """
    t = np.load(os.path.join(datadir, f"element{z}", "temps.npy"))
    order = np.argsort(t)
    ts = t[order].astype(np.float64)
    lf = np.load(os.path.join(datadir, f"element{z}", "logflux.npy"),
                 mmap_mode="r")
    out = []
    for centre in windows:
        k0 = int(np.searchsorted(ts, centre))
        lo = max(1, k0 - width // 2)
        prev = np.asarray(lf[int(order[lo - 1])], dtype=np.float64)[m]
        for k in range(lo, min(lo + width, len(ts))):
            row = np.asarray(lf[int(order[k])], dtype=np.float64)[m]
            ok = (row > -300) & (prev > -300)
            dlnt = abs(np.log(ts[k]) - np.log(ts[k - 1]))
            if ok.sum() > 200 and dlnt > 0:
                jump = float(np.median(np.abs(
                    10.0 ** (row[ok] - prev[ok]) - 1.0)))
                out.append((jump / dlnt, jump, ts[k], dlnt))
            prev = row
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    ap.add_argument("--width", type=int, default=40,
                    help="adjacent rows per window")
    ap.add_argument("--windows", default="0.6,1.2,2.5,5.4,8.0,14.6",
                    help="window centres in keV")
    args = ap.parse_args()
    windows = [float(x) for x in args.windows.split(",")]

    cen = np.load(os.path.join(args.datadir, "element26", "energy.npy"))
    m = (cen >= BAND[0]) & (cen < BAND[1])
    print(f"ADJACENT-row |dF/F| / |dlnT| in {BAND[0]}-{BAND[1]} keV, "
          f"{args.width} rows around each of {windows} keV.")
    print("Smooth CIE emissivity gives O(1). Large values = the cache "
          "disagrees with itself.")
    print(f"\n{'Z':>3} {'el':>3} {'median':>9} {'p99':>9} {'worst':>10} "
          f"{'at T':>9} {'raw jump':>10}")
    rows = []
    for z in range(1, 31):
        r = scan_element(args.datadir, z, m, windows, args.width)
        if r is None:
            print(f"{z:3d} {SYMBOL[z]:>3}   (no comparable rows in band)")
            continue
        norm = np.array([x[0] for x in r])
        i = int(np.argmax(norm))
        rows.append((z, float(np.median(norm)), norm[i], r[i][1], r[i][2]))
        print(f"{z:3d} {SYMBOL[z]:>3} {np.median(norm):9.2f} "
              f"{np.percentile(norm, 99):9.2f} {norm[i]:10.1f} "
              f"{r[i][2]:9.4f} {r[i][1]:10.2%}")
    if rows:
        bad = sorted(rows, key=lambda r: -r[2])[:6]
        print("\nworst adjacent-row inconsistencies (normalised, raw jump):")
        for z, med, w, raw, tw in bad:
            print(f"  {SYMBOL[z]:>2}  {w:8.1f}  ({raw:6.2%} across one step "
                  f"at {tw:.4f} keV; element median {med:.1f})")
        print("\nNo interpolator or trained network can be more accurate "
              "than the cache is self-consistent.")


if __name__ == "__main__":
    main()
