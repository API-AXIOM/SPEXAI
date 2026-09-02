"""P3: an independent SPEX truth at temperatures the emulator never trained on.

``SpexTruthModel`` is the current ground truth for the bias campaign, but it
PCHIP-interpolates the very rows the emulator trained on, so at an off-grid
temperature "truth" and emulator share an interpolation. This script removes
that caveat by calling SPEX itself at off-grid temperatures, **inside** the
training box (0.50-19.94 keV -- no extrapolation testing, D.H.'s call), on the
native training grid.

Three modes:

``plan``     (spexai env or any env with numpy) choose the test temperatures.
             The worst case for interpolation is the middle of the largest gap
             in the **training-split** temperature grid -- val/test rows were
             held out, so the nodes actually available to both the emulator and
             the PCHIP truth are sparser than the raw cache. Also reports, for
             every element, where each chosen temperature falls within its local
             gap, since the three generation batches have different grids.

``generate`` (SPEX env) dump single-element spectra at those temperatures.
             Runs at TWO free-bound accuracies by default: the setting the
             training data was made with (``var gacc 0.01``, see
             ``check_training_provenance.py``) and SPEX's own default. The first
             is the emulator test; the difference between them is the
             atomic-data-configuration systematic, which would otherwise be
             misreported as emulator error.

``compare``  (spexai env) emulator vs SPEX and PCHIP truth vs SPEX, at the same
             temperatures. Neither ratio is the interpolation error on its own:
             both also carry the training-data-vs-SPEX offset documented in
             ``check_training_provenance.py``, which is the larger term. Pass
             ``--nodes`` (a ``--match-cache`` dump at the same temperatures and
             the same gacc) to divide that offset out; the resulting ``interp``
             row is the number P3 exists to produce.

Usage:
    python scripts/inference/spex_offgrid_truth.py --mode plan --n 6
    # SPEX env, then, for each gacc:
    python scripts/inference/spex_offgrid_truth.py --mode generate \
        --edges /tmp/egrid.npz --temps <from plan> --gacc 0.01 --out truth_g01.npz
    # the on-node reference, same temps and gacc, via validate_spex_norm:
    python scripts/inference/validate_spex_norm.py --mode spex --elements all \
        --keep-h --match-cache <processed> --gacc 0.01 --temps <same> \
        --edges /tmp/egrid.npz --out nodes_g01.npz
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/inference/spex_offgrid_truth.py \
        --mode compare --in truth_g01.npz --nodes nodes_g01.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# the emulator's validity box in temperature (checkpoint t_lo/t_hi, log10 keV)
T_LO, T_HI = 0.501, 19.94
BAND = (1.9, 12.0)


def train_temps(datadir, z):
    """Sorted training-split temperatures for element z (val/test excluded)."""
    t = np.load(os.path.join(datadir, f"element{z}", "temps.npy"))
    tr = np.load(os.path.join(datadir, f"element{z}", "splits.npz"))["train"]
    return np.sort(np.asarray(t[tr], dtype=np.float64))


def plan(datadir, elements, n, ref_z=26):
    """Pick n off-grid temperatures at the widest training-grid gaps."""
    t = train_temps(datadir, ref_z)
    t = t[(t >= T_LO) & (t <= T_HI)]
    gaps = np.diff(np.log10(t))
    order = np.argsort(-gaps)[:n]
    temps = np.sort(10.0 ** (0.5 * (np.log10(t[order]) + np.log10(t[order + 1]))))
    print(f"element {ref_z} training grid: {len(t)} rows in "
          f"[{t[0]:.3f}, {t[-1]:.3f}] keV; log10-T gap median "
          f"{np.median(gaps):.2e}, max {gaps.max():.2e}")
    print(f"\nchosen temperatures (midpoints of the {n} widest gaps):")
    print("  " + ", ".join(f"{x:.6f}" for x in temps))

    print(f"\nposition within each element's own training grid "
          f"(0 = on a node, 0.5 = worst case):")
    print(f"{'Z':>3}  " + "  ".join(f"{x:8.4f}" for x in temps))
    for z in elements:
        tz = train_temps(datadir, z)
        cells = []
        for x in temps:
            j = int(np.searchsorted(tz, x))
            lo, hi = tz[max(0, j - 1)], tz[min(len(tz) - 1, j)]
            gap = np.log10(hi) - np.log10(lo)
            frac = (np.log10(x) - np.log10(lo)) / gap if gap > 0 else 0.0
            cells.append(f"{min(frac, 1 - frac):8.3f}")
        print(f"{z:3d}  " + "  ".join(cells))
    return temps


def node_ratio(nodes, datadir, j, z, m, widths):
    """(cache / SPEX) at each element's own nearest TRAINING NODE, per temp.

    This is the reference the off-grid ratio has to be divided by. At a training
    node the PCHIP truth is exact by construction, so whatever it shows there is
    the training-data-vs-SPEX offset alone (see ``check_training_provenance``).
    Dividing the off-grid ratio by it leaves the interpolation error, which is
    what P3 set out to measure -- otherwise the offset, which is far larger,
    would be read as interpolation error.
    """
    from check_training_provenance import cache_row, element_alone
    out = []
    for i in range(nodes["flux"].shape[0]):
        spx = element_alone(nodes, i, j, z)
        dens, _, _ = cache_row(datadir, z, nodes["temps_used"][i, j])
        ok = m & (spx > 0)
        out.append(np.median((dens * widths)[ok] / spx[ok]) if ok.sum()
                   else np.nan)
    return np.asarray(out)


def compare(inp, datadir, device="cpu", save=None, nodes=None):
    """Emulator and PCHIP truth vs the off-grid SPEX truth, per element."""
    import torch
    from spexai.inference.abundances import SYMBOL
    from spexai.inference.operator_model import (JointOperatorModel,
                                                 element_broadened_flux)
    from spexai.inference.spex_truth import ElementTruth

    d = np.load(inp)
    edges, cen, temps = d["edges"], d["centers"], d["temps"]
    elements = [int(z) for z in d["elements"]]
    print(f"SPEX truth: gacc={d['gacc']} spexact={d['spexact']} "
          f"abun={d['abun']} ibal={d['ibal']}")
    lo, hi = BAND
    m = (cen >= lo) & (cen < hi)
    et = torch.as_tensor(edges, dtype=torch.float32)
    tt = torch.as_tensor(temps, dtype=torch.float32)
    widths = np.diff(edges)

    nd = np.load(nodes) if nodes else None
    if nd is not None:
        assert np.allclose(nd["temps"], temps), \
            "the on-node reference must target the same temperatures"
        assert str(nd["gacc"]) == str(d["gacc"]), \
            f"gacc mismatch: off-grid {d['gacc']} vs nodes {nd['gacc']}"

    model = JointOperatorModel(device=device, elements=elements)
    print(f"\nmedian per-bin ratio over {lo}-{hi} keV, at off-grid T")
    print(f"{'Z':>3} {'el':>3} " + " ".join(f"{t:>8.3f}" for t in temps)
          + "   (emulator/SPEX; PCHIP/SPEX below)")
    rows = {}
    for j, z in enumerate(elements):
        # element alone: H's continuum is in every single-element SPEX run
        spx = d["flux"][:, j] - (0.0 if z == 1 else d["flux_h"][:, j])
        emu = element_broadened_flux(model.models[z], tt, 0.0, et).cpu().numpy()
        # the PCHIP truth is loaded one element at a time: SpexTruthModel holds
        # ~0.4 GB of training rows per element, which will not fit for 30 at once
        truth = ElementTruth.from_cache(os.path.join(datadir, f"element{z}"))
        pch = truth.native_flux(np.asarray(temps, dtype=np.float64)).numpy()
        cells_e, cells_p = [], []
        for i in range(len(temps)):
            ok = m & (spx[i] > 0)
            if not ok.sum():
                cells_e.append("       -")
                cells_p.append("       -")
                continue
            cells_e.append(f"{np.median(emu[i][ok] / spx[i][ok]):8.4f}")
            cells_p.append(f"{np.median(pch[i][ok] / spx[i][ok]):8.4f}")
        rows[z] = (cells_e, cells_p)
        print(f"{z:3d} {SYMBOL.get(z, '?'):>3} " + " ".join(cells_e))
        print(f"{'':3} {'pchip':>3} " + " ".join(cells_p))
        if nd is not None:
            # interpolation error alone: the off-grid ratio with the on-node
            # training-data offset divided out
            ref = node_ratio(nd, datadir, j, z, m, widths)
            cells_i = []
            for i in range(len(temps)):
                v = cells_p[i]
                cells_i.append("       -" if v.strip() == "-" or
                               not np.isfinite(ref[i]) or ref[i] == 0
                               else f"{float(v) / ref[i]:8.5f}")
            print(f"{'':3} {'interp':>6} " + " ".join(cells_i))
        del truth, pch

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez(save, temps=temps, elements=np.asarray(elements),
                 centers=cen, gacc=d["gacc"])
        print(f"\nwrote {save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["plan", "generate", "compare"])
    ap.add_argument("--datadir",
                    default=os.environ.get("SPEXAI_PROCESSED",
                                           os.path.expanduser(
                                               "~/data/spexai_data/processed")))
    ap.add_argument("--elements", default="all")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temps", default=None)
    ap.add_argument("--edges", default="egrid.npz")
    ap.add_argument("--out", default="spex_offgrid.npz")
    ap.add_argument("--in", dest="inp", default="spex_offgrid.npz")
    ap.add_argument("--gacc", type=float, default=None)
    ap.add_argument("--spexact", type=int, choices=[0, 1, 2], default=None,
                    help="SPEXACT version. Full 3 (=2) is REQUIRED to get the "
                         "15 non-MEKAL elements at all: under SPEXACT 2 they "
                         "emit nothing whatever their abundance")
    ap.add_argument("--match-cache", default=None,
                    help="generate mode: run each element at its own nearest "
                         "training node instead of the requested temperature "
                         "(this builds the --nodes reference)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None)
    ap.add_argument("--nodes", default=None,
                    help="compare mode: a --match-cache dump at the same "
                         "temperatures and gacc, giving the on-node reference")
    args = ap.parse_args()

    zs = (list(range(1, 31)) if args.elements == "all"
          else [int(z) for z in args.elements.split(",")])
    if args.mode == "plan":
        plan(args.datadir, zs, args.n)
    elif args.mode == "generate":
        from validate_spex_norm import dump_spex_elements
        temps = [float(x) for x in args.temps.split(",")]
        for t in temps:
            if not T_LO <= t <= T_HI:
                raise SystemExit(f"T={t} is outside the training box "
                                 f"[{T_LO}, {T_HI}] keV -- P3 tests inside only")
        dump_spex_elements(args.out, args.edges, temps, zs, args.spexact, None,
                           keep_h=True, gacc=args.gacc,
                           match_cache=args.match_cache)
    else:
        compare(args.inp, args.datadir, device=args.device, save=args.save,
                nodes=args.nodes)


if __name__ == "__main__":
    main()
