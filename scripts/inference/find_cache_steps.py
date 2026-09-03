"""Sweep the training cache for flux discontinuities in temperature.

CIE emissivity is smooth in T, so the band flux of consecutive cache rows should
trace a slowly-varying power law. This finds pairs that do not: a large flux
change across a negligible temperature step, which no physics produces.

Two conditions must hold together, and both are needed:

  |dlnF/dlnT| > SLOPE_MAX   -- the implied logarithmic derivative is unphysical.
                               Real slopes in 1.9-12 keV run ~2 (Fe) to ~9 (Zn),
                               so 50 is far outside the physical range.
  |flux step|  > STEP_MIN   -- and the jump is actually large. Without this,
                               a rounding-level change across two near-duplicate
                               temperatures divides by ~0 and flags everything.

Established 2026-09-02 (see the cache-step-defects note): the steps are real,
element-specific, concentrated below ~2 keV, and NOT reproduced by SPEX at any
free-bound accuracy -- so they are generation-time defects rather than a
settings artefact. Ni is the worst affected and is an element this work fits.

Usage:
    python scripts/inference/find_cache_steps.py [--datadir DIR] \
        [--band 1.9,12] [--out steps.json] [--elements 1-30]
"""
import argparse
import json
import os

import numpy as np

SLOPE_MAX = 50.0        # |dlnF/dlnT| beyond which no CIE physics applies
STEP_MIN = 0.02         # and the flux must move by at least this fraction
SYMBOL = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
          9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
          16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
          23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
          30: "Zn"}


def band_flux(datadir, z, m):
    """(sorted temperatures, band-summed flux density per row).

    Bin widths are deliberately omitted: every quantity here is a RATIO between
    two rows on the same grid, so the widths cancel exactly.
    """
    d = os.path.join(datadir, f"element{z}")
    t = np.load(os.path.join(d, "temps.npy")).astype(np.float64)
    lf = np.load(os.path.join(d, "logflux.npy"), mmap_mode="r")
    order = np.argsort(t)
    f = np.empty(len(order))
    for i, k in enumerate(order):
        f[i] = np.sum(10.0 ** np.asarray(lf[int(k)], dtype=np.float64)[m])
    return t[order], f, order


def find_steps(ts, f):
    """Indices i where the pair (i, i+1) is an unphysical jump."""
    ok = f > 0
    if ok.sum() < 10:
        return np.empty(0, dtype=int), np.zeros(0), np.zeros(0)
    lnf = np.log(np.clip(f, 1e-300, None))
    dlnt = np.diff(np.log(ts))
    dlnf = np.diff(lnf)
    good = (dlnt > 0) & ok[:-1] & ok[1:]
    slope = np.where(good, dlnf / np.where(good, dlnt, 1.0), 0.0)
    step = np.where(good, np.expm1(dlnf), 0.0)
    hit = good & (np.abs(slope) > SLOPE_MAX) & (np.abs(step) > STEP_MIN)
    return np.flatnonzero(hit), step, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    ap.add_argument("--band", default="1.9,12")
    ap.add_argument("--elements", default="1-30")
    ap.add_argument("--out", default=None, help="write full detail as JSON")
    args = ap.parse_args()

    if "-" in args.elements:
        a, b = args.elements.split("-")
        zs = list(range(int(a), int(b) + 1))
    else:
        zs = [int(x) for x in args.elements.split(",")]

    cen = np.load(os.path.join(args.datadir, "element26",
                               "energy.npy")).astype(np.float64)
    lo, hi = (float(x) for x in args.band.split(","))
    m = (cen >= lo) & (cen < hi)

    print(f"discontinuities in {lo}-{hi} keV: |dlnF/dlnT| > {SLOPE_MAX:g} "
          f"AND |flux step| > {STEP_MIN:.0%}")
    print(f"\n{'Z':>3} {'el':>3} {'rows':>6} {'steps':>6} {'worst':>9} "
          f"{'at T':>9} {'T range affected':>20} {'>2keV':>6}")
    detail = {}
    for z in zs:
        try:
            ts, f, order = band_flux(args.datadir, z, m)
        except FileNotFoundError:
            print(f"{z:3d} {SYMBOL[z]:>3}   (no cache)")
            continue
        idx, step, slope = find_steps(ts, f)
        if not len(ts) or (f > 0).sum() < 10:
            print(f"{z:3d} {SYMBOL[z]:>3} {len(ts):6d}   "
                  f"(no in-band flux -- element does not emit here)")
            continue
        if not len(idx):
            print(f"{z:3d} {SYMBOL[z]:>3} {len(ts):6d} {0:6d} "
                  f"{'--':>9} {'--':>9} {'clean':>20} {0:6d}")
            detail[z] = []
            continue
        w = idx[int(np.argmax(np.abs(step[idx])))]
        hot = int(np.sum(ts[idx] > 2.0))
        print(f"{z:3d} {SYMBOL[z]:>3} {len(ts):6d} {len(idx):6d} "
              f"{step[w]:+9.2%} {ts[w]:9.4f} "
              f"{f'{ts[idx].min():.2f}-{ts[idx].max():.2f} keV':>20} {hot:6d}")
        detail[z] = [{"T_lo": float(ts[i]), "T_hi": float(ts[i + 1]),
                      "step": float(step[i]), "slope": float(slope[i]),
                      "row_lo": int(order[i]), "row_hi": int(order[i + 1])}
                     for i in idx]

    if detail:
        n_aff = sum(1 for v in detail.values() if v)
        tot = sum(len(v) for v in detail.values())
        above = sum(1 for v in detail.values() for s in v if s["T_lo"] > 2.0)
        print(f"\n{n_aff}/{len(detail)} elements with at least one step; "
              f"{tot} steps in total, {above} of them above 2 keV.")
        worst = sorted(((abs(s["step"]), z, s) for z, v in detail.items()
                        for s in v), reverse=True)[:8]
        print("largest steps overall:")
        for _, z, s in worst:
            print(f"  {SYMBOL[z]:>2} {s['step']:+8.2%} at "
                  f"{s['T_lo']:.5f}->{s['T_hi']:.5f} keV "
                  f"(dT/T = {(s['T_hi']/s['T_lo'] - 1):.1e})")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({str(k): v for k, v in detail.items()}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
