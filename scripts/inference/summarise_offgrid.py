"""Reduce the P3 off-grid comparison to the numbers that are actually measured.

The raw per-element table mixes three very different kinds of cell, and reading
it without separating them produces nonsense (an "emulator/SPEX = 74" for Cu is
not a 7400% error, it is a near-zero denominator):

  MEASURED    the cache and SPEX both carry flux over most of the band. Only
              here is a ratio meaningful.
  SPARSE      the cache is zero across most of the band where SPEX is positive
              -- the missing sub-threshold continuum (see the P/Cl/Mn note).
              The ratio measures that mismatch, not interpolation.
  ABSENT      neither carries in-band flux (Li, Be, B, F): the element does not
              emit above ~1.9 keV at all, so there is nothing to compare.

Within MEASURED cells the quantity of interest is the PCHIP **interpolation**
error: (PCHIP/SPEX at the off-grid T) / (cache/SPEX at that element's nearest
training node). Dividing by the on-node ratio removes the training-data-vs-SPEX
offset, which is the larger term and would otherwise be read as interpolation
error.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/inference/summarise_offgrid.py \
        --offgrid off_sa3_g001.npz --nodes nodes_sa3_g001.npz \
        [--gdef off_sa3_gdef.npz] [--datadir <processed>]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BAND = (1.9, 12.0)
# A line-dominated trace element legitimately carries flux in only a few
# percent of the band -- its continuum sits below the generation gacc threshold
# and SPEX writes exact zeros (see the training-cache-zero-bins note). So the
# criterion is an absolute count of comparable bins, enough for a stable
# median, NOT a fraction of the band: a fraction test throws away Sc, Ti and V,
# which are precisely the elements the bias campaign cares about.
MIN_BINS = 200
SYMBOL = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
          9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
          16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
          23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
          30: "Zn"}


def element_alone(d, i, j, z):
    """SPEX flux for element z alone: every run carries H's continuum."""
    return d["flux"][i, j] if z == 1 else d["flux"][i, j] - d["flux_h"][i, j]


def classify(cache_off, spx_off, m):
    """(state, comparable_mask) for one element/temperature cell.

    SPARSE is reserved for a real disagreement: SPEX carries flux over a good
    part of the band while the cache is zero there. An element that is simply
    line-dominated has few live bins in BOTH, and is MEASURED on those bins.
    """
    live_spx = m & (spx_off > 0)
    both = live_spx & (cache_off > 0)
    if both.sum() >= MIN_BINS:
        return "MEASURED", both
    if live_spx.sum() >= MIN_BINS:      # SPEX emits, the cache does not
        return "SPARSE", both
    return "ABSENT", both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offgrid", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--gdef", default=None,
                    help="same off-grid temps at SPEX's DEFAULT gacc; gives "
                         "the atomic-data-configuration systematic")
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    args = ap.parse_args()

    from spexai.inference.spex_truth import ElementTruth

    def assert_training_nodes(nod, elements):
        """Every on-node reference must sit on a TRAINING row.

        The correction assumes PCHIP is exact at the node. On a val/test row it
        is not, and the "offset" then carries interpolation error -- which is
        precisely the quantity being corrected for. Silent when right, loud
        when wrong; this failed for 63/180 cells before nearest_cache_temp
        learned to restrict to the training split.
        """
        bad = []
        for j, z in enumerate(elements):
            d = os.path.join(args.datadir, f"element{z}")
            t = np.load(os.path.join(d, "temps.npy"))
            tr = set(np.load(os.path.join(d, "splits.npz"))["train"].tolist())
            for tu in nod["temps_used"][:, j]:
                if int(np.argmin(np.abs(t - tu))) not in tr:
                    bad.append((z, float(tu)))
        if bad:
            raise SystemExit(
                f"{len(bad)} of {nod['temps_used'].size} on-node references "
                f"are val/test rows, where PCHIP is not exact "
                f"(e.g. Z={bad[0][0]} at {bad[0][1]:.6f} keV). Regenerate the "
                f"--nodes dump; nearest_cache_temp now defaults to train-only.")

    off, nod = np.load(args.offgrid), np.load(args.nodes)
    assert np.allclose(off["temps"], nod["temps"]), "temperature grids differ"
    assert str(off["gacc"]) == str(nod["gacc"]), "gacc differs"
    assert str(off["spexact"]) == str(nod["spexact"]), "spexact differs"
    cen, edges, temps = off["centers"], off["edges"], off["temps"]
    elements = [int(z) for z in off["elements"]]
    widths = np.diff(edges)
    lo, hi = BAND
    m = (cen >= lo) & (cen < hi)
    gdef = np.load(args.gdef) if args.gdef else None
    assert_training_nodes(nod, elements)

    print(f"off-grid truth: spexact={off['spexact']} gacc={off['gacc']}; "
          f"band {lo}-{hi} keV; {len(temps)} temperatures")
    print(f"\n{'Z':>3} {'el':>3} {'state':>9} {'n':>3}  "
          f"{'interp err (PCHIP, offset removed)':>36}   {'gacc syst':>10}")
    print(f"{'':>3} {'':>3} {'':>9} {'':>3}  {'median':>11} {'worst':>11} "
          f"{'@T':>11}   {'median':>10}")

    stats = []
    for j, z in enumerate(elements):
        truth = ElementTruth.from_cache(os.path.join(args.datadir,
                                                     f"element{z}"))
        pch_off = truth.native_flux(np.asarray(temps, float)).numpy()
        pch_nod = truth.native_flux(
            np.asarray(nod["temps_used"][:, j], float)).numpy()
        del truth

        errs, states, gsys = [], [], []
        for i in range(len(temps)):
            spx_off = element_alone(off, i, j, z)
            state, live = classify(pch_off[i], spx_off, m)
            states.append(state)
            if state != "MEASURED":
                continue
            # on-node reference: the SAME quantity where PCHIP is exact by
            # construction, so it isolates the training-data-vs-SPEX offset
            spx_nod = element_alone(nod, i, j, z)
            ok_nod = m & (spx_nod > 0) & (pch_nod[i] > 0)
            if not ok_nod.sum():
                continue
            r_off = np.median(pch_off[i][live] / spx_off[live])
            r_nod = np.median(pch_nod[i][ok_nod] / spx_nod[ok_nod])
            if r_nod > 0:
                errs.append((abs(r_off / r_nod - 1.0), temps[i]))
            if gdef is not None:
                g = element_alone(gdef, i, j, z)
                okg = live & (g > 0)
                if okg.sum():
                    gsys.append(abs(np.median(spx_off[okg] / g[okg]) - 1.0))

        n = sum(s == "MEASURED" for s in states)
        nsp = sum(s == "SPARSE" for s in states)
        state = ("MEASURED" if n == len(temps) else
                 "ABSENT" if all(s == "ABSENT" for s in states) else
                 f"SPARSE:{nsp}" if nsp else "mixed")
        if errs:
            med = float(np.median([e for e, _ in errs]))
            worst, wt = max(errs)
            stats.append((z, med, worst, wt))
            gcell = f"{np.median(gsys):10.2%}" if gsys else f"{'-':>10}"
            print(f"{z:3d} {SYMBOL[z]:>3} {state:>9} {n:3d}  {med:10.4%} "
                  f"{worst:10.4%} {wt:11.3f}   {gcell}")
        else:
            print(f"{z:3d} {SYMBOL[z]:>3} {state:>9} {n:3d}  "
                  f"{'(no measurable cell)':>36}")

    if stats:
        meds = np.array([s[1] for s in stats])
        worst = max(stats, key=lambda s: s[2])
        print(f"\n{len(stats)} elements with at least one measurable cell.")
        print(f"PCHIP interpolation error, median over elements: "
              f"{np.median(meds):.4%}; 90th pct {np.percentile(meds, 90):.4%}")
        print(f"worst single element: {SYMBOL[worst[0]]} {worst[2]:.3%} "
              f"at T={worst[3]:.3f} keV")
        good = [s for s in stats if s[2] < 0.01]
        print(f"{len(good)}/{len(stats)} elements stay under 1% at every "
              f"measurable temperature.")


if __name__ == "__main__":
    main()
