"""Where is the BIAS worst, from a p6_sweep log alone (no jsonl needed).

The log is small and easy to move off a cluster mid-run, and it carries every
point's physical parameters plus ``worst |b|/sig`` -- enough to map where in
parameter space the emulator systematic is large.

**It cannot answer the linearisation question.** Per-parameter ``k``,
``b_sys`` and ``mean_delta`` are written only to the jsonl; the log's k table
appears at the end of a completed sweep. For "where is the linearisation less
trustworthy", use ``p6_trends.py`` on the jsonl. This script deliberately does
not guess at that from ``drift``/``spread``, which measure the optimiser, not
the linearisation.

    python scripts/inference/p6_log_trends.py logs/p6_overnight_gn_*.log
"""
import argparse
import re

import numpy as np
from scipy.stats import spearmanr

HEAD = re.compile(
    r"^point (\d+): kT=([\d.]+) sigma_v=([\d.]+) n_h=([\d.]+)\s+"
    r"cond\(F\)=([\d.e+]+)\s+worst \|b\|/sig@1e\+05=([\d.]+)")
SUMM = re.compile(
    r"^\s+(\d+)s\s+drift ([\d.]+) sigma \(([\d.]+)% of bias\)\s+"
    r"start spread ([\d.]+) sigma \(([\d.]+)% of bias\)\s+(\d+)/(\d+) measurable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--counts", type=float, default=1e9,
                    help="report |b|/sigma at this many counts; the log states "
                         "it at 1e5 and b/sigma scales as sqrt(N)")
    args = ap.parse_args()

    scale = np.sqrt(args.counts / 1e5)
    pts, cur = {}, None
    for line in open(args.log):
        m = HEAD.match(line)
        if m:
            cur = int(m.group(1))
            # the header is printed twice for the first point (the repeat
            # check echoes it); keep the first and never overwrite a point
            # that already has a summary attached
            pts.setdefault(cur, dict(
                kT=float(m.group(2)), sigma_v=float(m.group(3)),
                n_h=float(m.group(4)), cond=float(m.group(5)),
                b=float(m.group(6)) * scale))
            continue
        m = SUMM.match(line)
        if m and cur is not None and "drift" not in pts.get(cur, {}):
            pts[cur].update(secs=int(m.group(1)), drift=float(m.group(3)),
                            spread=float(m.group(5)),
                            meas=int(m.group(6)), ndim=int(m.group(7)),
                            conv="NOT CONVERGED" not in line)

    done = {p: v for p, v in pts.items() if "drift" in v}
    print(f"{len(pts)} points seen, {len(done)} completed, from {args.log}\n")
    if len(done) < 3:
        raise SystemExit("too few completed points to correlate")

    g = lambda k: np.array([done[p][k] for p in sorted(done)])   # noqa: E731
    b, kT, sv, nh, cond = g("b"), g("kT"), g("sigma_v"), g("n_h"), g("cond")

    print("== optimiser health (context, not the science) ==")
    print(f"  start spread : median {np.median(g('spread')):.2f}%, "
          f"max {g('spread').max():.2f}% of bias  (criterion 10%)")
    print(f"  measurable   : {int(g('meas').min())}-{int(g('meas').max())} "
          f"of {int(g('ndim')[0])}")
    # Split the flag by WHICH criterion fired. Only the multi-start spread is a
    # convergence certificate; drift measures the last step's movement, which
    # point 10 of the 2026-09-05 sweep showed can be 2e-4 sigma while the fit
    # is 3.4 sigma out, and which at n=20 produced two pure false alarms by
    # dividing a normal step by a thin bias.
    sp, dr, conv = g("spread"), g("drift"), g("conv").astype(bool)
    real = sp > 10.0
    drift_only = (~conv) & ~real
    print(f"  flagged NOT CONVERGED: {int((~conv).sum())} of {len(conv)}")
    print(f"    spread > 10%  (REAL non-convergence)     : {int(real.sum())}")
    print(f"    drift only    (thin-bias false alarm)    : "
          f"{int(drift_only.sum())}")
    if real.any():
        bad_pts = [p for p in sorted(done) if done[p]["spread"] > 10.0]
        print(f"    points with a real failure: {bad_pts}")
        print(f"    their spreads: "
              f"{[round(done[p]['spread'], 1) for p in bad_pts]}")
    print(f"  wall time    : {g('secs').sum() / 3600:.1f} h, "
          f"median {np.median(g('secs')):.0f} s/pt")

    print(f"\n== BIAS: worst |b|/sigma per point at {args.counts:.0e} counts ==")
    print(f"  median {np.median(b):.1f}, mean {b.mean():.1f}, "
          f"range {b.min():.1f} - {b.max():.1f} sigma")
    crit = 1.96 / np.sqrt(len(b) - 1)
    print(f"\n  Spearman vs the point's physical parameters "
          f"(n={len(b)}, |rho| > ~{crit:.2f} for p<0.05):")
    for lab, x in (("kT", kT), ("sigma_v", sv), ("n_h", nh), ("cond(F)", cond)):
        r, p = spearmanr(x, b)
        star = "*" if p < 0.05 else " "
        print(f"    {lab:>8}  rho = {r:>+6.3f}{star}  p = {p:.4f}")

    print("\n  terciles (median worst |b|/sigma):")
    for lab, x in (("kT", kT), ("sigma_v", sv), ("n_h", nh)):
        lo, hi = np.percentile(x, [33.3, 66.7])
        grp = [b[x <= lo], b[(x > lo) & (x <= hi)], b[x > hi]]
        cells = "  ".join(f"{np.median(q):6.1f} (n={len(q):2d})" for q in grp)
        print(f"    {lab:>8}  low/mid/high: {cells}   "
              f"[splits {lo:.2f}, {hi:.2f}]")

    print("\n  worst 8 points:")
    for p in sorted(done, key=lambda p: -done[p]["b"])[:8]:
        v = done[p]
        print(f"    point {p:>3}  |b|/sig={v['b']:>6.1f}  kT={v['kT']:>5.2f}  "
              f"sigma_v={v['sigma_v']:>6.1f}  n_h={v['n_h']:>5.2f}  "
              f"cond={v['cond']:.1e}")
    print("\n  best 8 points:")
    for p in sorted(done, key=lambda p: done[p]["b"])[:8]:
        v = done[p]
        print(f"    point {p:>3}  |b|/sig={v['b']:>6.1f}  kT={v['kT']:>5.2f}  "
              f"sigma_v={v['sigma_v']:>6.1f}  n_h={v['n_h']:>5.2f}  "
              f"cond={v['cond']:.1e}")

    print("\nNOTE: this is the LINEAR b_sys prediction, not the measured "
          "offset. At\n  n=20 the binding parameter had k ~ 1, so it is a good "
          "proxy there --\n  but confirm with p6_trends.py on the jsonl.")


if __name__ == "__main__":
    main()
