"""Which k values are real, and which parameter actually sets N*?

Two distinct questions, and conflating them is the trap this script exists to
avoid.

1. **Is a given k a measurement?** ``k = mean_delta / b_sys`` is a ratio. In
   NOISELESS mode ``se_d`` is numerical scatter between starts rather than
   statistical error, and Gauss-Newton drove that to ~0.02 sigma -- so the
   sweep's ``|b_sys| > 3 se_d`` rule now passes at |b_sys| > 0.06 sigma and
   marks essentially every parameter measurable. The rule got MORE permissive
   because the optimiser got better. So the honest error on k in noiseless mode
   is the start spread over the measured offset, not ``se_k`` (which collapses
   to ~0 when the starts agree, and looks reassuring while meaning nothing).

2. **Does it matter?** N* is where systematic bias meets statistical error.
   Since b/sigma ~ sqrt(N), the parameter that BINDS N* is the one with the
   largest |mean_delta|/sigma -- not the one with the largest k. A parameter
   whose bias is 0.06 sigma can carry a huge k and still be irrelevant, because
   0.15 sigma of bias does not threaten anything either. Reporting the largest
   k anywhere as "N* is optimistic by k^2" is therefore wrong unless that
   parameter is also the binding one.

    python scripts/inference/p6_check_outliers.py p6_gn_sweep.jsonl
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
                    help="call a denominator THIN below this many sigma")
    ap.add_argument("--max_frac_err", type=float, default=0.2,
                    help="flag k whose fractional error exceeds this")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in open(args.jsonl) if ln.strip()]
    print(f"{len(rows)} points from {args.jsonl}\n")
    print("== large k, and whether each is a measurement ==")
    print(f"{'point':>5} {'param':>9} {'k':>8} {'+-':>6} {'|b|/sig':>9} "
          f"{'delta/sig':>10} {'spread/sig':>11}  flag")

    worst_any = (0.0, None)
    worst_trusted = (0.0, None)
    binding = (0.0, None, 0.0)          # (|delta|/sig, (point, name), k)

    for r in sorted(rows, key=lambda x: x["point"]):
        names = r["names"]
        k = np.asarray(r["k"])                                # (ndim,)
        b = np.asarray(r["b_sys"])
        sig = np.asarray(r["sigma"])
        delta = np.asarray(r["mean_delta"])
        meas = np.asarray(r["measurable"], dtype=bool)
        # start spread is ALREADY in sigma units; delta and b_sys are in raw
        # parameter units. Everything below is converted to sigma first --
        # mixing the two silently flagged a clean -2.0 sigma offset as
        # unresolved in the first version of this script.
        spread = float(r.get("start_spread_sigma", np.nan))
        d_sig = delta / sig                                   # (ndim,)
        b_sig = np.abs(b) / sig                               # (ndim,)

        for j in range(len(names)):
            if not meas[j]:
                continue
            if abs(d_sig[j]) > binding[0]:
                binding = (abs(d_sig[j]), (r["point"], names[j]), k[j])
            if abs(k[j]) > worst_any[0]:
                worst_any = (abs(k[j]), (r["point"], names[j]))

        for j in np.argsort(-np.abs(k)):
            if not meas[j] or abs(k[j]) < args.k_min:
                continue
            # in noiseless mode the numerator's error IS the start spread
            frac = spread / max(abs(d_sig[j]), 1e-30)
            flag = []
            if b_sig[j] < args.b_min_sigma:
                flag.append(f"THIN b_sys ({b_sig[j]:.3f} sigma)")
            if frac > args.max_frac_err:
                flag.append(f"delta known to only {frac:.0%}")
            print(f"{r['point']:>5} {names[j]:>9} {k[j]:>+8.2f} "
                  f"{frac * abs(k[j]):>6.2f} {b_sig[j]:>9.3f} "
                  f"{d_sig[j]:>10.3f} {spread:>11.3f}  "
                  f"{'; '.join(flag) or 'ok'}")
            if not flag and abs(k[j]) > worst_trusted[0]:
                worst_trusted = (abs(k[j]), (r["point"], names[j]))

    print("\n== what this means for N* ==")
    kw, nw = worst_any
    print(f"largest k anywhere      : {kw:.2f} ({nw[1]} at point {nw[0]})")
    if worst_trusted[1] is not None:
        kt, nt = worst_trusted
        print(f"largest UNFLAGGED k     : {kt:.2f} ({nt[1]} at point {nt[0]})")
    else:
        print("largest UNFLAGGED k     : none above the threshold")

    db, nb, kb = binding
    print(f"binding parameter       : {nb[1]} at point {nb[0]}, "
          f"|delta| = {db:.2f} sigma, k = {kb:+.2f}")
    print(f"\n=> N* is optimistic by {kb ** 2:.1f}x, set by the BINDING "
          f"parameter.\n   A larger k on a parameter with a thin bias does "
          f"not move N*: it is\n   a large ratio between two small numbers, "
          f"and P7 should carry it as a\n   per-parameter limit, not as the "
          f"headline.")


if __name__ == "__main__":
    main()
