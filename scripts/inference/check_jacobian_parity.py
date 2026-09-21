"""Compare two ``bias_sweep --stage bias`` runs point by point.

Built for one job: confirming that ``--jacobian batched`` reproduces
``--jacobian serial``. The two paths differ in three ways that could each move
a number -- a batched ``VectorForward`` instead of ``campaign.Forward``, float32
on the device instead of float64 on the CPU, and (in DEM mode) the shape
parameters carried in the vector instead of pinned per point -- so the claim
that they agree has to be measured rather than asserted.

The metric is deliberately in sigma units, not relative: ``b_sys`` is published
only ever divided by ``sigma_ref`` (and ``N* = N_REF (sigma/b)^2`` is the same
ratio squared), so a large relative disagreement on a component whose bias is a
thousandth of an error bar changes no result, while a small one on a binding
component does. Relative differences are printed too, for diagnosis.

    python scripts/inference/check_jacobian_parity.py \
        --a results/bias_sweep/bias_single_n20_s3.jsonl \
        --b results/bias_sweep/bias_single_n20_s3_batched.jsonl

Exit status is 1 if any shared point exceeds the tolerances, so it can gate a
production run.
"""
import argparse
import json
import sys

import numpy as np


def load(path):
    """-> {point: record}. Keeps the LAST record for a repeated point.

    ``stage_bias`` opens its jsonl in append mode and only skips finished work
    under ``--resume``, so a rerun without it leaves two records per point in
    one file. Deduplicating here rather than refusing keeps the check usable on
    files that already have the duplicates.
    """
    recs, dupes = {}, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["point"]) in recs:
                dupes += 1
            recs[int(r["point"])] = r
    if dupes:
        print(f"note: {path} has {dupes} duplicate record(s); kept the last "
              f"of each", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference jsonl (serial)")
    ap.add_argument("--b", required=True, help="jsonl under test (batched)")
    ap.add_argument("--tol_sigma", type=float, default=1e-2,
                    help="max allowed |b_a - b_b| / sigma_a, per component")
    ap.add_argument("--tol_sigma_rel", type=float, default=1e-2,
                    help="max allowed relative difference in sigma_ref")
    ap.add_argument("--n_worst", type=int, default=10)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("no points in common")
    only = (set(A) ^ set(B))
    if only:
        print(f"note: {len(only)} point(s) in only one file: "
              f"{sorted(only)[:10]}{'...' if len(only) > 10 else ''}")

    names = A[shared[0]]["names"]
    if names != B[shared[0]]["names"]:
        raise SystemExit(f"parameter names differ: {names} vs "
                         f"{B[shared[0]]['names']}")

    rows = []                      # (dsig, rel_sig, point, param)
    dcond = []
    for i in shared:
        a, b = A[i], B[i]
        if a["names"] != names or b["names"] != names:
            raise SystemExit(f"point {i}: parameter order changed")
        ba = np.array(a["b_sys"])
        bb = np.array(b["b_sys"])
        sa = np.array(a["sigma_ref"])
        sb = np.array(b["sigma_ref"])
        dsig = np.abs(ba - bb) / sa                   # the published units
        rel_sig = np.abs(sb - sa) / sa
        for j, nm in enumerate(names):
            rows.append((dsig[j], rel_sig[j], i, nm))
        dcond.append(abs(b["cond_F"] - a["cond_F"]) / a["cond_F"])

    dsig = np.array([r[0] for r in rows])
    rel_sig = np.array([r[1] for r in rows])
    dcond = np.array(dcond)

    print(f"\n{len(shared)} shared points x {len(names)} parameters\n")
    print(f"{'quantity':>28} {'median':>10} {'p90':>10} {'max':>10}")
    for label, v in (("|b_a - b_b| / sigma_a", dsig),
                     ("rel. diff in sigma_ref", rel_sig),
                     ("rel. diff in cond(F)", dcond)):
        print(f"{label:>28} {np.median(v):>10.2e} "
              f"{np.percentile(v, 90):>10.2e} {v.max():>10.2e}")

    print(f"\nworst {args.n_worst} components by |b_a - b_b| / sigma_a:")
    print(f"{'point':>6} {'param':>12} {'d_sigma':>10} {'rel_sigma':>10}")
    for d, rs, i, nm in sorted(rows, reverse=True)[:args.n_worst]:
        print(f"{i:>6} {nm:>12} {d:>10.2e} {rs:>10.2e}")

    bad_b = dsig.max() > args.tol_sigma
    bad_s = rel_sig.max() > args.tol_sigma_rel
    if bad_b or bad_s:
        print(f"\nFAIL: b_sys to {dsig.max():.2e} sigma "
              f"(tol {args.tol_sigma:.1e}), sigma_ref to {rel_sig.max():.2e} "
              f"relative (tol {args.tol_sigma_rel:.1e})")
        return 1
    print(f"\nPASS: b_sys agrees to {dsig.max():.2e} sigma and sigma_ref to "
          f"{rel_sig.max():.2e} relative, over every shared component")
    return 0


if __name__ == "__main__":
    sys.exit(main())
