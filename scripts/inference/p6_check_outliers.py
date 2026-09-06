"""Is a large k real, or a small-denominator artefact?

``k = mean_delta / b_sys`` is a ratio. In NOISELESS mode ``se_d`` is numerical
scatter between starts rather than statistical error, and Gauss-Newton drove
that to ~0.02 sigma -- so the sweep's ``|b_sys| > 3 se_d`` detectability rule
now passes at |b_sys| > 0.06 sigma and marks essentially every parameter
measurable. That makes the rule weakest exactly where the optimiser is best,
so a headline k needs its denominator inspected directly.

For every (point, parameter) above ``--k_min`` this prints |b_sys| in sigma
units, the implied mean_delta, and se_k -- the three numbers that decide
whether the ratio means anything.

    python scripts/inference/p6_check_outliers.py p6_gn_smoke.jsonl
"""
import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--k_min", type=float, default=1.3,
                    help="report every |k| at or above this")
    ap.add_argument("--b_min_sigma", type=float, default=0.5,
                    help="flag denominators thinner than this many sigma")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.jsonl) if ln.strip()]
    print(f"{len(rows)} points from {args.jsonl}\n")
    print(f"{'point':>5} {'param':>9} {'k':>8} {'se_k':>8} "
          f"{'|b|/sig':>9} {'delta/sig':>10} {'spread/sig':>11}  flag")

    worst_trusted = (0.0, None)
    for r in sorted(rows, key=lambda x: x["point"]):
        names = r["names"]
        k = np.asarray(r["k"])                                # (ndim,)
        se_k = np.asarray(r["se_k"])
        b = np.asarray(r["b_sys"])
        sig = np.asarray(r["sigma"])
        delta = np.asarray(r["mean_delta"])
        meas = np.asarray(r["measurable"], dtype=bool)
        spread = float(r.get("start_spread_sigma", np.nan))

        for j in np.argsort(-np.abs(k)):
            if not meas[j] or abs(k[j]) < args.k_min:
                continue
            b_sig = abs(b[j]) / sig[j]
            thin = b_sig < args.b_min_sigma
            # a ratio is only as good as its denominator: se_k/|k| is the
            # fractional error, and it blows up as b_sys -> 0
            frac = se_k[j] / max(abs(k[j]), 1e-30)
            flag = []
            if thin:
                flag.append(f"THIN b_sys ({b_sig:.2f} sigma)")
            if frac > 0.2:
                flag.append(f"se_k/k = {frac:.0%}")
            if spread == spread and abs(delta[j]) < 3 * spread:
                flag.append("delta < 3x start spread")
            print(f"{r['point']:>5} {names[j]:>9} {k[j]:>+8.2f} "
                  f"{se_k[j]:>8.2f} {b_sig:>9.3f} {delta[j]/sig[j]:>10.3f} "
                  f"{spread:>11.3f}  {'; '.join(flag) or 'ok'}")
            if not flag and abs(k[j]) > worst_trusted[0]:
                worst_trusted = (abs(k[j]), (r["point"], names[j]))

    print()
    if worst_trusted[1] is None:
        print("No unflagged k above the threshold: the headline number is "
              "carried entirely by thin denominators. Report the flagged "
              "values as limits, not measurements.")
    else:
        pt, nm = worst_trusted[1]
        print(f"Worst UNFLAGGED k: {worst_trusted[0]:.2f} ({nm} at point {pt})"
              f"  =>  N* optimistic by {worst_trusted[0] ** 2:.1f}x")


if __name__ == "__main__":
    main()
