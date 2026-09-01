"""Compare Tier B sweep results with and without the Fe XXV mask.

Companion to ``scripts/experiments/hot_floor/compare_masking.py``, which does
the same for the single-point hot-floor Fisher runs. Reads two ``bias_*.jsonl``
files written by ``bias_sweep.py --stage bias`` and reports, per parameter, how
the distribution of |b_sys|/sigma over the sampled volume moves.

The specific question this exists to answer: the Perseus-fiducial hot-floor run
showed iron's bias almost completely cancelling when the Fe XXV complex is
restored (N* 2.1e7 -> 3.0e12). Is that a general property of the emulator, or a
coincidence of that one abundance pattern? A sweep over 20 points answers it,
and a per-point spread in N* is the diagnostic: a general cancellation would
lift every point, a fine-tuned one lifts a few.

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python \\
        scripts/inference/compare_sweep_masking.py
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from spexai.config import RESULTS                                 # noqa: E402


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    names = recs[0]["names"]
    for r in recs:
        if r["names"] != names:
            raise SystemExit("parameter lists differ between records")
    b = np.array([r["b_sys"] for r in recs])            # (n_pts, n_par)
    s = np.array([r["sigma_ref"] for r in recs])
    nref = float(recs[0]["n_ref"])
    return names, b, s, nref, recs


def ratios(b, s, nref, counts):
    """|b_sys| / sigma_stat(counts), per point and parameter."""
    return np.abs(b) / s * np.sqrt(counts / nref)


def main():
    ap = argparse.ArgumentParser()
    d = os.path.join(RESULTS, "bias_sweep")
    ap.add_argument("--unmasked", default=os.path.join(d, "bias_single_n20_s3.jsonl"))
    ap.add_argument("--masked", default=os.path.join(d, "bias_single_n20_s3_masked.jsonl"))
    ap.add_argument("--counts", type=float, default=1e6)
    args = ap.parse_args()

    nm, bm, sm, refm, recm = load(args.masked)
    nu, bu, su, refu, recu = load(args.unmasked)
    if nm != nu:
        raise SystemExit("parameter lists differ between the two runs")
    rm, ru = ratios(bm, sm, refm, args.counts), ratios(bu, su, refu, args.counts)
    nsm = refm * (sm / np.abs(bm)) ** 2                 # (n_pts, n_par)
    nsu = refu * (su / np.abs(bu)) ** 2

    print(f"Tier B, {len(recm)} points, |b_sys|/sigma at {args.counts:.0e} "
          f"in-band counts")
    print(f"{'param':>9} | {'masked':^23} | {'unmasked':^23}")
    print(f"{'':>9} | {'median':>7}{'max':>8}{'medN*':>9} | "
          f"{'median':>7}{'max':>8}{'medN*':>9}")
    for i, p in enumerate(nm):
        print(f"{p:>9} | {np.median(rm[:,i]):>7.3f}{rm[:,i].max():>8.3f}"
              f"{np.median(nsm[:,i]):>9.2e} | "
              f"{np.median(ru[:,i]):>7.3f}{ru[:,i].max():>8.3f}"
              f"{np.median(nsu[:,i]):>9.2e}")

    print(f"\nfrac > 1 sigma: masked {np.mean(rm > 1):.3f}, "
          f"unmasked {np.mean(ru > 1):.3f}")
    for lab, r, n in (("masked", rm, nm), ("unmasked", ru, nm)):
        j = np.unravel_index(np.argmax(r), r.shape)
        print(f"worst single instance {lab:>8}: {r[j]:.3f} sigma on "
              f"{n[j[1]]} (point {j[0]})")

    # Does the Perseus Fe cancellation generalise?
    fe = nm.index("Fe")
    print(f"\nFe N* across the {len(recu)} sweep points (unmasked): "
          f"min {nsu[:,fe].min():.2e}, median {np.median(nsu[:,fe]):.2e}, "
          f"max {nsu[:,fe].max():.2e}")
    print(f"  (Perseus fiducial, unmasked, single-T: 2.98e+12)")
    print(f"  ratio unmasked/masked per point: median "
          f"{np.median(nsu[:,fe]/nsm[:,fe]):.2f}, max "
          f"{(nsu[:,fe]/nsm[:,fe]).max():.2f}")


if __name__ == "__main__":
    main()
