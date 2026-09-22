"""Integrity + conditioning diagnostics for a P7 bias-sweep jsonl.

``bias_sweep --summarise`` reports medians, percentiles and N* per parameter
but says nothing about two things that can invalidate them:

* **Duplicate records.** ``stage_bias`` opens its jsonl in append mode and only
  skips finished work under ``--resume``, so a rerun without it leaves two
  records for the same point and every median here is silently reweighted.
* **cond(F).** A near-singular Fisher matrix makes ``b_sys`` and ``sigma_ref``
  individually meaningless while leaving their ratio finite and plausible, so a
  degenerate direction produces a publishable-looking outlier with nothing in
  the summary to flag it (see ``fisher_bias.linear_bias_fisher``).

This reports both, then anatomises the tail: for each parameter, which points
exceed 1 sigma at ``--target_counts`` and what their conditioning looks like
relative to the bulk. It reads the jsonl only -- nothing is recomputed.

    python scripts/inference/p7_screen_diag.py \
        --jsonl ~/work/data/spexai/results/bias_sweep/bias_single_n1000_s39235.jsonl
"""
import argparse
import json
from collections import Counter

import numpy as np

COND_F_WARN = 1e10


def load(path):
    """-> (records in point order, duplicate counter). Keeps the LAST record."""
    recs, seen = {}, Counter()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            seen[int(r["point"])] += 1
            recs[int(r["point"])] = r
    return recs, seen


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--n_points", type=int, default=1000,
                    help="what the sweep was asked for, to detect a short file")
    ap.add_argument("--target_counts", type=float, default=1e6)
    ap.add_argument("--n_show", type=int, default=8,
                    help="worst points to anatomise per flagged parameter")
    ap.add_argument("--confinement", action="store_true",
                    help="KS test per design axis: is the tail confined?")
    ap.add_argument("--drivers", default="",
                    help="comma-separated parameters to rank-correlate "
                         "against the design axes, e.g. sigma_v,Mn")
    args = ap.parse_args()

    recs, seen = load(args.jsonl)
    names = recs[min(recs)]["names"]

    print("=== integrity ===")
    dupes = {p: c for p, c in seen.items() if c > 1}
    print(f"records read      : {sum(seen.values())}")
    print(f"distinct points   : {len(recs)} (asked for {args.n_points})")
    print(f"duplicated points : {len(dupes)}"
          + (f"  {sorted(dupes)[:10]}" if dupes else ""))
    missing = sorted(set(range(args.n_points)) - set(recs))
    print(f"missing points    : {len(missing)}"
          + (f"  {missing[:10]}" if missing else ""))

    pts = sorted(recs)
    b = np.array([recs[i]["b_sys"] for i in pts])            # (P, n)
    sig = np.array([recs[i]["sigma_ref"] for i in pts])      # (P, n)
    cond = np.array([recs[i]["cond_F"] for i in pts])        # (P,)
    n_ref = recs[pts[0]]["n_ref"]
    # sigma scales as 1/sqrt(N); b_sys is count-independent
    ratio = np.abs(b) / (sig * np.sqrt(n_ref / args.target_counts))

    print("\n=== cond(F) over the design ===")
    for q in (50, 90, 99, 100):
        print(f"  p{q:<3d} {np.percentile(cond, q):.3e}")
    n_bad = int((cond > COND_F_WARN).sum())
    print(f"  above COND_F_WARN={COND_F_WARN:.0e}: {n_bad} point(s)"
          + (f"  {[pts[j] for j in np.where(cond > COND_F_WARN)[0]][:10]}"
             if n_bad else ""))

    print(f"\n=== tail anatomy at {args.target_counts:.1e} counts ===")
    print("cond(F) percentile is the point's rank within the design, so a tail "
          "\nsitting at high percentiles is conditioning, not physics.\n")
    cond_pct = 100.0 * np.argsort(np.argsort(cond)) / (len(cond) - 1)
    for j, nm in enumerate(names):
        over = np.where(ratio[:, j] > 1.0)[0]
        if not len(over):
            continue
        order = over[np.argsort(-ratio[over, j])][:args.n_show]
        print(f"{nm}: {len(over)} point(s) above 1 sigma "
              f"(median {np.median(ratio[:, j]):.3f})")
        print(f"{'point':>7} {'b/sigma':>9} {'cond(F)':>10} {'cond pct':>9}")
        for k in order:
            print(f"{pts[k]:>7} {ratio[k, j]:>9.3f} {cond[k]:>10.2e} "
                  f"{cond_pct[k]:>9.1f}")
        print()

    if args.confinement:
        confinement([recs[i] for i in pts], ratio)

    for param in [x for x in args.drivers.split(",") if x]:
        drivers(pts, recs, ratio, cond, names, param, len(names))


def confinement(recs, ratio, thresh=1.0):
    """Is the >1 sigma population confined to a region of the design?

    Two-sample KS between the tail's marginal and the bulk's, per design axis,
    plus the share of the tail in that axis's lowest design quartile (which
    gives the KS statistic a direction -- a confined tail and an inverted one
    both reject). kT is tested in log10 because the design samples it
    log-uniform. This is the number behind the corner plot.
    """
    from scipy.stats import ks_2samp
    bad = ratio.max(axis=1) > thresh
    rows = []
    for k in list(recs[0]["params"]):
        v = np.array([r["params"][k] for r in recs])
        if k == "kT":
            v = np.log10(v)
        ks, p = ks_2samp(v[bad], v[~bad])
        rows.append((p, k, ks, (v[bad] <= np.percentile(v, 25)).mean()))
    rows.sort()
    print(f"=== confinement of the {int(bad.sum())} points above "
          f"{thresh:g} sigma ===")
    print(f"{'axis':>12} {'KS':>7} {'p':>10} {'low-quartile share':>20}")
    for p, k, ks, frac in rows:
        flag = "  <== confined" if p < 1e-3 else ""
        print(f"{k:>12} {ks:>7.3f} {p:>10.1e} {100 * frac:>19.0f}%{flag}")
    print()


def drivers(pts, recs, ratio, cond, names, param, n_show):
    """Rank-correlate one parameter's bias/sigma against the design axes.

    Spearman rather than Pearson: the question is which coordinates ORDER the
    tail, and b/sigma spans two decades with a heavy tail, so a rank statistic
    is the honest one. cond(F) is included as an axis because a tail that
    tracks conditioning is a numerical story and one that tracks a physical
    coordinate is not.
    """
    from scipy.stats import spearmanr
    j = names.index(param)
    y = ratio[:, j]
    axes = {k: np.array([recs[i]["params"][k] for i in pts])
            for k in recs[pts[0]]["params"]}
    axes["cond_F"] = cond
    rows = []
    for k, x in axes.items():
        rho, p = spearmanr(x, y)
        rows.append((abs(rho), rho, p, k))
    rows.sort(reverse=True)
    print(f"=== what orders {param}'s bias/sigma (Spearman, n={len(y)}) ===")
    print(f"{'axis':>12} {'rho':>8} {'p':>10}")
    for _, rho, p, k in rows[:n_show]:
        print(f"{k:>12} {rho:>8.3f} {p:>10.2e}")
    print()


if __name__ == "__main__":
    main()
