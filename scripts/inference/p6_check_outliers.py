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
    ap.add_argument("--big_sigma", type=float, default=1.0,
                    help="a bias this many sigma or more is big enough to "
                         "move N*; k is reported for every such parameter")
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

    # The single global binding parameter is one number from one point. The
    # claim worth making is stronger and needs this table: k ~ 1 for EVERY
    # parameter whose bias is large enough to move N*, with departures confined
    # to thin biases that cannot. If a big-bias parameter shows k far from 1
    # anywhere, the headline above is hiding it.
    print(f"\n== k where the bias is big enough to matter "
          f"(|delta| > {args.big_sigma:g} sigma) ==")
    print(f"{'point':>5} {'param':>9} {'delta/sig':>10} {'k':>8} {'+-':>6}")
    worst_big = (0.0, None)
    for r in sorted(rows, key=lambda x: x["point"]):
        names, k = r["names"], np.asarray(r["k"])
        sig = np.asarray(r["sigma"])
        d_sig = np.asarray(r["mean_delta"]) / sig
        meas = np.asarray(r["measurable"], dtype=bool)
        spread = float(r.get("start_spread_sigma", np.nan))
        b_sig = np.abs(np.asarray(r["b_sys"])) / sig
        idx = [j for j in np.argsort(-np.abs(d_sig))
               if meas[j] and abs(d_sig[j]) > args.big_sigma]
        for j in idx:
            frac = spread / max(abs(d_sig[j]), 1e-30)
            # A big MEASURED bias does not make k trustworthy: k = delta/b_sys,
            # so it is the DENOMINATOR that has to be thick. Ar at point 8 has
            # |delta| = 1.23 sigma (passes this table's filter) on |b_sys| =
            # 0.067 sigma, and reports k = 18.5. Filtering on the numerator
            # alone let that set the headline at 342x.
            thin = b_sig[j] < args.b_min_sigma
            print(f"{r['point']:>5} {names[j]:>9} {d_sig[j]:>10.2f} "
                  f"{k[j]:>+8.3f} {frac * abs(k[j]):>6.3f}"
                  f"{'   THIN b_sys' if thin else ''}")
            if not thin and abs(k[j] - 1.0) > worst_big[0]:
                worst_big = (abs(k[j] - 1.0), (r["point"], names[j], k[j]))
        if not idx:
            print(f"{r['point']:>5} {'--':>9} {'':>10} "
                  f"{'(no bias above threshold)':>8}")
    if worst_big[1] is not None:
        pt, nm, kv = worst_big[1]
        print(f"\nlargest departure from k=1, thin denominators excluded: "
              f"{kv:+.3f} ({nm} at point {pt})  =>  that parameter's N* is "
              f"off by {kv ** 2:.2f}x")

    # N* is a PER-POINT quantity -- each point is a different source model --
    # so a single global binding parameter is the wrong summary. It answers
    # "which parameter anywhere has the biggest bias", when the question is
    # "for each source, which parameter binds ITS N*, and is that parameter's
    # k equal to 1?". The worst point is then the headline.
    print("\n== per-point: which parameter binds THIS point's N* ==")
    print(f"{'point':>5} {'binds on':>9} {'delta/sig':>10} {'k':>8} "
          f"{'+-':>6} {'N* factor':>10}")
    worst_point = (0.0, None)
    for r in sorted(rows, key=lambda x: x["point"]):
        names, k = r["names"], np.asarray(r["k"])
        sig = np.asarray(r["sigma"])
        d_sig = np.asarray(r["mean_delta"]) / sig
        meas = np.asarray(r["measurable"], dtype=bool)
        spread = float(r.get("start_spread_sigma", np.nan))
        cand = [j for j in range(len(names)) if meas[j]]
        if not cand:
            continue
        j = max(cand, key=lambda i: abs(d_sig[i]))     # this point's binder
        frac = spread / max(abs(d_sig[j]), 1e-30)
        print(f"{r['point']:>5} {names[j]:>9} {d_sig[j]:>10.2f} "
              f"{k[j]:>+8.3f} {frac * abs(k[j]):>6.3f} "
              f"{k[j] ** 2:>9.2f}x")
        if k[j] ** 2 > worst_point[0]:
            worst_point = (k[j] ** 2, (r["point"], names[j], k[j]))

    print("\n== what this means for N* ==")
    kw, nw = worst_any
    print(f"largest k anywhere        : {kw:.2f} ({nw[1]} at point {nw[0]}) "
          f"-- IGNORE unless it also binds a point")
    if worst_trusted[1] is not None:
        kt, nt = worst_trusted
        print(f"largest UNFLAGGED k       : {kt:.2f} "
              f"({nt[1]} at point {nt[0]})")
    db, nb, kb = binding
    print(f"largest bias anywhere     : {db:.2f} sigma ({nb[1]} at point "
          f"{nb[0]}), k = {kb:+.3f}")
    if worst_point[1] is not None:
        pt, nm, kv = worst_point[1]
        print(f"\n=> WORST POINT is {pt}, binding on {nm}: k = {kv:+.3f}, so "
              f"ITS N* is optimistic by {kv ** 2:.2f}x.")
        print("   Report P6 per point. A large k on a parameter with a thin "
              "bias is a\n   ratio between two small numbers and moves no N*; "
              "carry those as limits.")


if __name__ == "__main__":
    main()
