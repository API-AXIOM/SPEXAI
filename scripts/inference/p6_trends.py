"""Where is the bias worst, and where does the linearisation break down?

Two separate questions with different answers, so they are reported apart:

* **bias** = |mean_delta| / sigma -- how badly the emulator shifts a parameter.
* **linearisation error** -- how badly ``b_sys`` PREDICTS that shift, as both
  the ratio |k - 1| and the absolute offset |delta - b_sys| in sigma.

A parameter can carry a huge bias that linearises perfectly (sigma_v) or a
modest bias that does not (log_norm), so neither number substitutes for the
other.

STATISTICS WARNING, read before quoting any regional correlation: the rows are
NOT independent. All rows from one point share that point's kT, sigma_v and
n_h, so a 20-point sweep gives ~20 effective samples for any trend against a
POINT-level quantity, not the ~160 rows. The printed p-values for those are
badly anti-conservative; the cluster-aware ones (computed per point) are the
ones to read. Per-PARAMETER trends do not have this problem -- each parameter
gets one row per point.

    python scripts/inference/p6_trends.py p6_gn_sweep.jsonl
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--big_sigma", type=float, default=1.0,
                    help="only rows whose |delta| exceeds this many sigma")
    ap.add_argument("--b_min_sigma", type=float, default=0.5,
                    help="|b_sys|/sigma below this makes k a thin ratio and is "
                         "excluded from every linearisation statistic")
    ap.add_argument("--max_spread", type=float, default=0.10,
                    help="drop points whose multi-start spread exceeds this "
                         "fraction of the bias. The spread is the only real "
                         "convergence certificate (drift is not), and an "
                         "unconverged fit biases k LOW -- the direction that "
                         "falsely exonerates. Set to 0 to disable the cut")
    args = ap.parse_args()

    allrows = [json.loads(ln) for ln in open(args.jsonl) if ln.strip()]
    rows = allrows
    if args.max_spread > 0:
        rows = [r for r in allrows
                if float(r.get("start_spread_frac_of_bias", 0.0))
                <= args.max_spread]
        drop = sorted(int(r["point"]) for r in allrows if r not in rows)
        if drop:
            print(f"EXCLUDED {len(drop)} of {len(allrows)} points on "
                  f"start spread > {args.max_spread:.0%} of bias: {drop}")
            print("  (their k is biased LOW; rerun them before quoting any "
                  "number that depends on them)\n")
    pt, nm, d, k, b = [], [], [], [], []
    phys = {}
    for r in rows:
        sig = np.asarray(r["sigma"])
        d_s = np.asarray(r["mean_delta"]) / sig
        b_s = np.asarray(r["b_sys"]) / sig
        kk = np.asarray(r["k"])
        meas = np.asarray(r["measurable"], dtype=bool)
        p = int(r["point"])
        # p6_sweep stores params as a dict; mle_reseed's own writer json.dumps
        # it to a string. Accept either rather than silently reporting
        # "not in the jsonl" for every regional trend.
        par = r.get("params", {})
        if isinstance(par, str):
            par = json.loads(par)
        phys[p] = (par.get("kT", np.nan), par.get("sigma_v", np.nan),
                   par.get("n_h", np.nan), float(r.get("cond_F", np.nan)))
        for j, name in enumerate(r["names"]):
            if meas[j] and abs(d_s[j]) > args.big_sigma:
                pt.append(p), nm.append(name)
                d.append(d_s[j]), k.append(kk[j]), b.append(b_s[j])

    pt, nm = np.array(pt), np.array(nm)
    d, k, b = np.array(d), np.array(k), np.array(b)
    thick = np.abs(b) >= args.b_min_sigma
    kerr = np.abs(k - 1.0)
    off = np.abs(d - b)                       # absolute error, in sigma
    print(f"{len(d)} rows over |delta| > {args.big_sigma:g} sigma from "
          f"{len(rows)} points; {thick.sum()} with a thick denominator\n")

    print("== per parameter ==")
    print(f"{'param':>9} {'n':>3} | {'BIAS |delta|/sig':>21} | "
          f"{'LINEARISATION (thick)':>34}")
    print(f"{'':>9} {'':>3} | {'med':>6} {'max':>7} {'@pt':>5} | "
          f"{'n':>3} {'med|k-1|':>9} {'max|k-1|':>9} {'med|d-b|':>9}")
    for s in sorted(set(nm), key=lambda s: -np.median(np.abs(d[nm == s]))):
        m, mt = nm == s, (nm == s) & thick
        j = int(np.argmax(np.abs(d[m])))
        line = (f"{s:>9} {m.sum():>3} | {np.median(np.abs(d[m])):>6.2f} "
                f"{np.abs(d[m]).max():>7.2f} {pt[m][j]:>5d} | ")
        line += (f"{mt.sum():>3} {np.median(kerr[mt]):>9.3f} "
                 f"{kerr[mt].max():>9.3f} {np.median(off[mt]):>9.3f}"
                 if mt.sum() else f"{0:>3} {'--':>9} {'--':>9} {'--':>9}")
        print(line)

    # The key structural question. If |delta - b_sys| is independent of
    # |b_sys|, the linearisation error is a fixed ADDITIVE offset in sigma,
    # so k = 1 + offset/b: it tends to 1 exactly where the bias is large
    # enough to set N*, and diverges only where the bias cannot matter.
    print("\n== is the error additive (fixed sigma) or multiplicative? ==")
    rr, pp = spearmanr(np.abs(b[thick]), off[thick])
    print(f"  |b_sys| vs |delta - b_sys|: rho = {rr:+.3f}, p = {pp:.4f}"
          f"   (rho ~ 0 => ADDITIVE)")
    rr2, pp2 = spearmanr(np.abs(b[thick]), kerr[thick])
    print(f"  |b_sys| vs |k - 1|        : rho = {rr2:+.3f}, p = {pp2:.4f}"
          f"   (negative => k -> 1 for big biases)")
    print(f"  |delta - b_sys|: median {np.median(off[thick]):.3f}, "
          f"90th {np.percentile(off[thick], 90):.3f}, "
          f"max {off[thick].max():.3f} sigma")
    # A sign test on rho only says "not zero". The exponent says how much:
    # |delta - b| ~ |b|^alpha, with alpha = 0 a fixed additive offset (k -> 1
    # for large bias, the safe case) and alpha = 1 a fixed FRACTIONAL error
    # (k constant, so a big bias stays proportionally wrong). Anything in
    # between means the error grows sublinearly and k still improves with
    # bias, just more slowly than a pure offset would give.
    m = thick & (off > 0) & (np.abs(b) > 0)
    alpha, c = np.polyfit(np.log(np.abs(b[m])), np.log(off[m]), 1)
    print(f"  log-log slope alpha = {alpha:+.2f}   "
          f"(0 = purely additive, 1 = purely multiplicative)")

    # Cluster-aware regional trends: collapse to ONE number per point first,
    # so n = number of points rather than number of rows.
    print("\n== regional trends, aggregated PER POINT (n = points) ==")
    print(f"{'against':>10} {'BIAS (max/pt)':>22} {'|k-1| (median/pt)':>24}")
    agg_b, agg_k = defaultdict(list), defaultdict(list)
    for p in sorted(set(pt.tolist())):
        m, mt = pt == p, (pt == p) & thick
        agg_b[p] = np.abs(d[m]).max()
        agg_k[p] = np.median(kerr[mt]) if mt.sum() else np.nan
    pts = sorted(agg_b)
    yb = np.array([agg_b[p] for p in pts])
    yk = np.array([agg_k[p] for p in pts])
    ok = ~np.isnan(yk)
    for i, label in enumerate(("kT", "sigma_v", "n_h", "cond(F)")):
        x = np.array([phys[p][i] for p in pts], dtype=float)
        if np.all(np.isnan(x)):
            print(f"{label:>10}   (not in the jsonl)")
            continue
        rb, pb = spearmanr(x, yb)
        rl, pl = spearmanr(x[ok], yk[ok])
        print(f"{label:>10} {rb:>+15.3f} p={pb:>5.3f} "
              f"{rl:>+16.3f} p={pl:>5.3f}")
    print(f"  n = {len(pts)} points; |rho| > ~{1.96 / np.sqrt(len(pts) - 1):.2f}"
          f" is needed to clear p < 0.05 at this n.")

    print("\n== worst linearisation, thick denominators only ==")
    for i in np.argsort(-kerr)[:15]:
        if not thick[i]:
            continue
        kt, sv, nh, _ = phys[int(pt[i])]
        print(f"  point {pt[i]:>2} {nm[i]:>9}  k={k[i]:>+7.3f}  "
              f"|b|={abs(b[i]):>5.2f}  |delta|={abs(d[i]):>5.2f}  "
              f"|d-b|={off[i]:>5.2f} sigma   kT={kt:>5.2f} "
              f"sigma_v={sv:>5.1f} n_h={nh:>4.2f}")


if __name__ == "__main__":
    main()
