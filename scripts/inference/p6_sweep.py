"""P6 across the whole Tier B sweep: the linearisation factor k at every point.

One point told us k <= ~1.1 at point 14 and that k is count-independent over
three decades. Neither settles parameter space: k already varies from 0.11
(n_h) to 1.10 (Ca) BETWEEN PARAMETERS at that single point, so assuming it is
constant BETWEEN POINTS would be wishful. This runs every point and reports the
distribution.

What P7 needs from this is a single defensible statement -- "no point in the
sampled space has k above X" -- because Tier B's N* scales as k^-2 and P7 is
about to run ~1000 points through that screen. A k <= 1 everywhere means the
screen is honest-to-conservative and needs no correction at all.

Structure mirrors bias_sweep.py: checkpointed per point into a jsonl, --resume
skips what is already done, and --summarise reads the jsonl back. The emulator
and its VectorForward are built ONCE and reused for every point -- only the
truth vector and the prior box change from point to point -- which is the whole
reason this exists as a driver rather than a shell loop over mle_reseed.py.

    # the sweep (GPU, exclusive card; ~6 min/point):
    python -u scripts/inference/p6_sweep.py --device cuda --deterministic \\
        --counts 1e9 --n_seeds 8 --seed_chunk 2 --resume

    # read it back:
    python scripts/inference/p6_sweep.py --summarise
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from mle_reseed import (                                          # noqa: E402
    lbfgs_batch, tierb_forward, tierb_point, tierb_response, worst_ratio)
from spexai.config import RESULTS, STORE                          # noqa: E402
from spexai.inference.posterior import BoxPrior                   # noqa: E402


def repeat_check(forward, truth, n=3):
    """Evaluate the forward repeatedly at ONE fixed theta and report the spread.

    ``--deterministic`` is necessary but cannot be sufficient:
    ``torch.use_deterministic_algorithms`` only governs the ops torch KNOWS are
    nondeterministic (``index_add`` on CUDA is on that list, so the line
    deposit is covered). It says nothing about cuFFT plan selection, and
    nothing at all about the forward being float32 internally. So the flag
    being ON is not evidence that identical parameters give identical
    likelihoods -- and P6's L-BFGS assumes exactly that, since a line search
    comparing two evaluations of the same point is what decides every step.

    Symptom this exists to catch: a pass reporting 0.00e+00 movement whose
    -logL still differs from the previous pass's. That is only possible if the
    forward is not reproducible, and it silently sets a floor on how far any
    optimiser can converge.
    """
    th = torch.as_tensor(np.atleast_2d(truth), dtype=torch.float64,
                         device=forward.device)
    with torch.no_grad():
        mus = [forward.counts_torch(th, grad=False).double() for _ in range(n)]
    spread = max(float((m - mus[0]).abs().max()) for m in mus[1:])
    scale = float(mus[0].abs().max())
    rel = spread / scale
    print(f"repeat check: {n} evaluations at one fixed theta differ by at most "
          f"{spread:.3e} counts ({rel:.2e} relative)", flush=True)
    if spread > 0.0:
        print("  the forward is not bit-reproducible. Every L-BFGS line search "
              "compares evaluations of the same objective, so this is a floor "
              "on convergence: it shows up as passes that move 0.00e+00 yet "
              "report a changed -logL. --deterministic does not cover it "
              "(index_add IS covered; this is elsewhere). At ~1 float32 ulp "
              "(1.2e-07) it is the forward's own precision and there is "
              "nothing to fix -- float64 would not help, since the emulator's "
              "accuracy is ~1e-3. Much above that, look at cuFFT.", flush=True)

    # Is the objective L-BFGS minimises the same one the pass line prints?
    # The closure uses grad=True -> fold(flux(th)) inside grad_enabled(); the
    # printed -logL uses grad=False -> _counts_chunked(th). Same math on paper.
    # If they differ by more than the jitter above, the -logL trace is NOT the
    # optimised objective and its non-monotonicity across passes means nothing.
    mu_g = forward.counts_torch(th, grad=True).detach().double()
    path = float((mu_g - mus[0]).abs().max())
    print(f"  grad=True vs grad=False at the same theta: {path:.3e} counts "
          f"({path / scale:.2e} relative)", flush=True)
    if path > 5.0 * max(spread, 1e-30):
        print("  WARNING: the two paths disagree by more than run-to-run "
              "jitter, so the printed -logL is not the quantity being "
              "minimised. Read the drift diagnostic, not the -logL trace.",
              flush=True)
    return spread


def run_point(args, rec, counts_row, forward, keep):
    """One point: K reseeded MLEs -> k per parameter, plus diagnostics."""
    pars, names, truth, mu_true = tierb_point(args, rec, counts_row, keep)
    prior = BoxPrior.from_params(pars, device=args.device)

    data_batch = np.stack([
        np.random.default_rng(args.seed0 + i).poisson(mu_true)
        for i in range(args.n_seeds)]).astype(np.float64)

    b_sys = np.asarray(rec["b_sys"])
    # b_sys is count-independent (F and the residual term both scale with N);
    # sigma ~ N^-1/2. Verified empirically: k agrees at 1e7/1e8/1e9.
    sigma = np.asarray(rec["sigma_ref"]) * np.sqrt(
        float(rec["n_ref"]) / args.counts)

    sc = args.seed_chunk or args.n_seeds
    mle_parts, move_parts = [], []
    for lo_k in range(0, args.n_seeds, sc):
        hi_k = min(lo_k + sc, args.n_seeds)
        m, mv = lbfgs_batch(forward, prior, data_batch[lo_k:hi_k], truth,
                            args.max_iter, n_restarts=args.n_restarts,
                            sigma_ref=sigma, objective="mle",
                            tol_change=args.tol_change,
                            tol_grad=args.tol_grad,
                            precondition=args.precondition,
                            max_eval=args.max_eval,
                            ls_debug=args.ls_debug)
        mle_parts.append(m)
        move_parts.append(mv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mle = np.concatenate(mle_parts, axis=0)
    move = np.concatenate(move_parts, axis=0)

    delta = mle - truth[None, :]
    mean_d = delta.mean(0)
    se_d = delta.std(0, ddof=1) / np.sqrt(args.n_seeds)
    k = mean_d / b_sys
    se_k = se_d / np.abs(b_sys)
    # k is a ratio: only meaningful where b_sys clears this run's noise floor
    measurable = np.abs(b_sys) > 3.0 * se_d
    # convergence judged against the bias, for parameters whose bias is resolved
    bias_sig = np.abs(mean_d / sigma)
    resolved = bias_sig > 3.0 * (se_d / sigma)
    frac = np.where(resolved, move.max(axis=0) / np.maximum(bias_sig, 1e-12), 0.0)

    return {
        "point": int(rec["point"]), "params": rec["params"], "names": names,
        "counts": args.counts, "n_seeds": args.n_seeds,
        "truth": truth.tolist(), "b_sys": b_sys.tolist(),
        "sigma": sigma.tolist(), "k": k.tolist(), "se_k": se_k.tolist(),
        "mean_delta": mean_d.tolist(), "se_delta": se_d.tolist(),
        "measurable": measurable.tolist(),
        "cond_F": rec["cond_F"], "worst_b_over_sig": worst_ratio(rec),
        "drift_sigma": float(move.max()),
        "drift_frac_of_bias": float(frac.max()),
        "converged": bool(frac.max() <= 0.10),
    }


def summarise(path):
    """k per parameter across points -- the distribution P7 needs."""
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if not recs:
        raise SystemExit(f"{path} is empty")
    names = recs[0]["names"]
    print(f"{len(recs)} points from {path}")
    bad = [r["point"] for r in recs if not r["converged"]]
    if bad:
        print(f"NOT CONVERGED at points {bad} -- their k is biased LOW and "
              f"must not be read as reassurance; rerun with more restarts")
    print(f"\n{'param':>9} {'n_meas':>7} {'median k':>9} {'min':>7} "
          f"{'max':>7} {'max@point':>10}  (measurable points only)")
    worst_overall, worst_where = 0.0, None
    for j, n in enumerate(names):
        ks, pts = [], []
        for r in recs:
            if r["measurable"][j] and r["converged"]:
                ks.append(r["k"][j])
                pts.append(r["point"])
        if not ks:
            print(f"{n:>9} {0:>7}        --      --      --          --")
            continue
        ks = np.array(ks)
        imax = int(np.argmax(ks))
        if ks[imax] > worst_overall:
            worst_overall, worst_where = float(ks[imax]), (n, pts[imax])
        print(f"{n:>9} {len(ks):>7} {np.median(ks):>+9.2f} {ks.min():>+7.2f} "
              f"{ks.max():>+7.2f} {pts[imax]:>10}")
    if worst_where:
        print(f"\nlargest k anywhere: {worst_overall:+.2f} "
              f"({worst_where[0]} at point {worst_where[1]})")
        if worst_overall <= 1.0:
            print("=> no measurable k exceeds 1 anywhere in the sampled space: "
                  "the linearised b_sys never underpredicts the real bias, so "
                  "Tier B's N* table needs no k^-2 correction and is "
                  "conservative where k < 1.")
        else:
            print(f"=> N* at the worst point is optimistic by "
                  f"{worst_overall ** 2:.1f}x. P7 must carry this per "
                  f"parameter, not as one scalar.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    R = os.path.join(RESULTS, "bias_sweep")
    ap.add_argument("--bias_jsonl", default=os.path.join(
        R, "bias_single_n20_s3.jsonl"))
    ap.add_argument("--truth_npz", default=os.path.join(
        R, "truth_single_n20_s3_stamped.npz"))
    ap.add_argument("--out", default=os.path.join(
        RESULTS, "mle_reseed", "p6_sweep_single.jsonl"))
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--counts", type=float, default=1e9)
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--seed_chunk", type=int, default=2)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--n_restarts", type=int, default=4)
    ap.add_argument("--tol_change", type=float, default=0.0)
    # See mle_reseed.lbfgs_batch for why each of these exists; all four target
    # the same root cause, that torch's LBFGS thresholds are ABSOLUTE and this
    # problem's coordinates span three decades in sigma.
    ap.add_argument("--tol_grad", type=float, default=1e-6)
    ap.add_argument("--max_eval", type=int, default=None,
                    help="torch's default (max_iter*5//4) caps FUNCTION EVALS "
                         "and starves the late line searches; set several "
                         "times --max_iter")
    ap.add_argument("--precondition", action="store_true",
                    help="optimise in units of sigma_ref")
    ap.add_argument("--ls_debug", action="store_true",
                    help="per-pass line-search trace")
    ap.add_argument("--points", default=None,
                    help="comma-separated subset, e.g. 0,5,9; default all")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--echunk", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="REQUIRED in practice: without it the line deposit's "
                         "atomicAdd jitter stalls the line search (point 14 "
                         "went from 2.63 sigma drift to 0.021 sigma with it)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--check_only", action="store_true",
                    help="build the forward, run the reproducibility repeat "
                         "check, and exit without fitting anything")
    args = ap.parse_args()

    if args.summarise:
        return summarise(args.out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("deterministic algorithms ON", flush=True)
    else:
        print("WARNING: running WITHOUT --deterministic. The line deposit's "
              "index_add_ has repeated indices, so CUDA sums in a varying "
              "order and identical parameters return different likelihoods; "
              "at point 14 that inflated the residual drift 125x.", flush=True)

    with open(args.bias_jsonl) as f:
        recs = {int(json.loads(l)["point"]): json.loads(l)
                for l in f if l.strip()}
    tz = np.load(args.truth_npz, allow_pickle=True)
    counts = tz["counts"]

    wanted = ([int(p) for p in args.points.split(",")] if args.points
              else sorted(recs))
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {int(json.loads(l)["point"]) for l in f if l.strip()}
        print(f"resuming: {len(done)} points already done", flush=True)
    todo = [p for p in wanted if p not in done]
    if not todo:
        print("nothing to do")
        return summarise(args.out)
    print(f"{len(todo)} points to run at {args.counts:.1e} counts, "
          f"{args.n_seeds} seeds each", flush=True)

    response, keep, rmf, arf = tierb_response(tz)
    # Built once: every single-T point shares the emulator, response, band and
    # abundance scheme. Only the truth vector and prior box differ.
    forward = tierb_forward(args, recs[todo[0]]["names"], response, keep)
    _, _, truth0, _ = tierb_point(args, recs[todo[0]], counts[todo[0]], keep)
    repeat_check(forward, truth0)
    if args.check_only:
        return

    for i, p in enumerate(todo):
        t0 = time.time()
        print(f"\n===== [{i + 1}/{len(todo)}] point {p} =====", flush=True)
        out = run_point(args, recs[p], counts[p], forward, keep)
        out["runtime_s"] = time.time() - t0
        out["rmf"], out["arf"] = os.path.basename(rmf), os.path.basename(arf)
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")
            f.flush()
            os.fsync(f.fileno())          # a long GPU run must not lose points
        flag = "" if out["converged"] else "  !! NOT CONVERGED"
        print(f"  {out['runtime_s']:.0f}s  drift {out['drift_sigma']:.3f} "
              f"sigma ({out['drift_frac_of_bias']:.1%} of bias)  "
              f"{int(np.sum(out['measurable']))}/{len(out['names'])} "
              f"measurable{flag}", flush=True)

    summarise(args.out)


if __name__ == "__main__":
    main()
