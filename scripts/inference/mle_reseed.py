"""Noise-vs-bias check for the sampler bake-off's recovery pulls.

The bake-off (emcee/UltraNest/SVI, all three agree) saw Ni +2.0 sigma, Mn
-1.3/-1.5 sigma, S -1.3 sigma, sigma_v +1.3/+1.6 sigma on ONE simulated
Poisson realization. Fisher predicts a bias far too small to explain that
(Ni -0.11 sigma, wrong sign) -- but three samplers agreeing only rules out a
*sampler* bug, since all three fit the identical draw and necessarily agree on
whatever that draw's fluctuation did.

This redraws the Poisson realization at the SAME truth/counts with many seeds
and computes a quick point estimate for each -- Fisher scoring (iteratively
reweighted least squares for the Poisson likelihood), NOT a full posterior.
If Ni's pull stays pinned near +2 sigma across redraws, that is bias. If it
scatters around 0 (Fisher's predicted -0.11 sigma), the original run was
noise.

Two point-estimate methods, picked with ``--method``:

``fisher`` (default, CPU-safe) -- iteratively reweighted least squares
(Newton-like) reusing the exact batched central-difference Jacobian trick
``fisher_bias.py`` / ``bias_sweep.py`` already use, batching ALL seeds'
stencils into one non-autograd, CHUNKED ``forward()`` call per round. This is
what laptop runs should use: ``VectorForward.counts_torch(grad=True)`` is
UNCHUNKED (the whole graph has to survive for a gradient step), and on CPU
that made even 4 seeds x 60 LBFGS iterations run past 10 minutes without
converging.

``lbfgs`` (GPU only) -- batched L-BFGS directly on the autograd path (all K
seeds as one batch dimension, one closure per line-search evaluation). The
unchunked graph that kills this on CPU is exactly what a GPU has memory
for, and it does not need the box-clipped Newton steps ``fisher`` uses to
stay numerically safe. Needs ``--device cuda``.

    # laptop (already run 2026-08-31, 8 seeds x 3 rounds, ~27 min CPU):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python -u \\
        scripts/inference/mle_reseed.py --method fisher --n_seeds 8 --n_iter 3

    # cluster GPU:
    python -u scripts/inference/mle_reseed.py --method lbfgs --device cuda \\
        --n_seeds 20 --max_iter 60 --compile

P6 MODE -- the linearisation factor
-----------------------------------
Passing ``--bias_jsonl``/``--truth_npz`` switches the problem from the
bake-off's single Perseus point to a *Tier B sweep point* (``--point i``), and
the report from sigma-unit pulls to the P6 estimator

    k_j = mean_k(theta_hat[k,j] - theta_true[j]) / b_sys[j]

the factor by which ``fisher_bias.linear_bias_fisher``'s LINEARISED bias
underpredicts the bias a real fit actually incurs. Tier B's whole ``N*`` table
scales as ``k^-2``, so this is the calibration of the screen, not a bias
measurement in its own right -- which is why it is run at several points: one
value of ``k`` cannot distinguish "uniformly 2x optimistic" from "fine except
in one corner", and those imply different things for P7.

Three things make the estimator cheap. The bias is a fixed offset in parameter
units while sigma ~ N^-1/2, so bias/noise grows as sqrt(N) and ~20-40 seeds
suffice at 1e8 counts; the Poisson likelihood costs the same to evaluate at
1e8 as at 1e6; and the autograd graph is batch-size independent, so K seeds
cost the memory of one. ``b_sys`` is count-independent and ``sigma_ref``
scales as N^-1/2, so both rescale analytically from the jsonl's ``n_ref``.

The failure mode to guard against is under-convergence: L-BFGS starts AT the
truth, so a fit that stops early leaves theta_hat near theta_true and reports
a spuriously SMALL k -- i.e. it fails in the reassuring direction. Two guards,
both on by default: ``--n_restarts 2`` reruns L-BFGS from its own solution and
reports the second pass's movement in sigma units (converged => ~0), and
``--start_sigma`` starts every seed displaced from the truth to confirm the
solutions do not depend on where they began.

    # P6, one Tier B point:
    python -u scripts/inference/mle_reseed.py --method lbfgs --device cuda \\
        --bias_jsonl $R/bias_sweep/bias_single_n20_s3.jsonl \\
        --truth_npz  $R/bias_sweep/truth_single_n20_s3.npz \\
        --point 14 --counts 1e8 --n_seeds 40 --max_iter 200
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

from bake_off import build_problem                                # noqa: E402
from bias_sweep import NORM_REF, build_pars                       # noqa: E402
from campaign import (                                            # noqa: E402
    PERSEUS, FREE_Z, find_xrism_response, band_mask, EXCLUDE_NONE,
    check_truth_response)
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import AbundanceModel, SYMBOL    # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.posterior import BoxPrior                   # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.vector_forward import VectorForward         # noqa: E402


def load_tierb(bias_jsonl, truth_npz, point):
    """(record, truth counts row) for one Tier B sweep point.

    The two files must come from the SAME sweep run: the jsonl carries the
    fitted-parameter side (``b_sys``, ``sigma_ref``, the point's truth vector)
    and the npz carries the noise-free SPEX truth spectrum, which the jsonl
    does not store.
    """
    with open(bias_jsonl) as f:
        recs = {int(json.loads(l)["point"]): json.loads(l)
                for l in f if l.strip()}
    if point not in recs:
        raise SystemExit(f"point {point} not in {bias_jsonl} "
                         f"(have {sorted(recs)[:5]}...{sorted(recs)[-1]})")
    tz = np.load(truth_npz, allow_pickle=True)
    counts = tz["counts"]                              # (n_points, n_channels)
    if point >= len(counts):
        raise SystemExit(f"truth npz has {len(counts)} points, asked for "
                         f"{point} -- the jsonl and npz are from different runs")
    return recs[point], counts[point], tz


def worst_ratio(rec):
    """max_j |b_sys_j| / sigma_ref_j -- the sweep's own severity ranking."""
    b = np.abs(np.asarray(rec["b_sys"]))
    return float((b / np.asarray(rec["sigma_ref"])).max())


def build_tierb_problem(args, rec, counts_row, tz):
    """Forward/prior/truth for one Tier B point, matching the sweep exactly.

    Mirrors ``bake_off.build_problem`` but on the sweep's terms: this point's
    LHS parameters rather than the Perseus fiducials, and ``EXCLUDE_NONE``
    rather than the Perseus resonance-scattering mask -- the mask is what
    ``bias_sweep`` used to produce the ``b_sys`` we are calibrating, and it
    deletes exactly the Fe-K channels the emulator is worst at (see
    ``campaign.EXCLUDE_NONE``). Using the other mask here would compare two
    different measurements.
    """
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    check_truth_response(tz, rmf, arf)
    keep = band_mask(response, exclude=EXCLUDE_NONE)

    d_ref = counts_row[keep]
    if d_ref.sum() <= 0:
        raise SystemExit(f"point has zero in-band truth counts")
    scale = args.counts / d_ref.sum()
    mu_true = d_ref * scale
    log_norm_truth = float(np.log10(NORM_REF * scale))

    # identical bounds/steps to the sweep's own Fisher solve
    pars = build_pars(None, rec["params"], log_norm_truth, "single")
    names = [p.name for p in pars]
    if names != list(rec["names"]):
        raise SystemExit(f"parameter order changed: jsonl has {rec['names']}, "
                         f"build_pars gives {names}")

    emu = JointOperatorModel(models_dir=args.store, device=args.device,
                             accelerate=False)
    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z],
                 1.0, 26)

    forward = VectorForward(
        emu, response, keep, names, ab, absorption=Absorption.default(),
        redshift=PERSEUS["z"], luminosity_distance=PERSEUS["dist_m"],
        velocity=None, device=args.device, chunk=args.chunk,
        batched=True, compile_trunk=args.compile, mem_gb=args.mem_gb,
        echunk=args.echunk)
    prior = BoxPrior.from_params(pars, device=args.device)
    truth = np.array([p.truth for p in pars])
    print(f"point {rec['point']}: kT={rec['params']['kT']:.3f} "
          f"sigma_v={rec['params']['sigma_v']:.1f} n_h={rec['params']['n_h']:.3f}"
          f"  cond(F)={rec['cond_F']:.1e}  worst |b|/sig@{rec['n_ref']:.0e}="
          f"{worst_ratio(rec):.3f}", flush=True)
    print(f"  {args.counts:.1e} in-band counts on {int(keep.sum())} channels; "
          f"log_norm_truth={log_norm_truth:.4f}", flush=True)
    return forward, prior, pars, truth, names, mu_true, (rmf, arf)


def batched_jacobian(forward, pars, theta):
    """Central-difference Jacobian at each of K rows of ``theta``, one call.

    ``theta``: (K, ndim). Returns (mu0 (K, n_keep), J (K, ndim, n_keep)) --
    all 2*ndim+1 stencil points for all K rows are folded into a single
    ``forward()`` call so the per-forward CPU cost (~3s/row) is paid once,
    not once per seed per iteration.
    """
    K, ndim = theta.shape
    steps = np.array([p.step for p in pars])
    stencil = np.repeat(theta, 2 * ndim + 1, axis=0)          # (K*(2n+1), ndim)
    stencil = stencil.reshape(K, 2 * ndim + 1, ndim)
    for i in range(ndim):
        stencil[:, 2 * i + 1, i] += steps[i]
        stencil[:, 2 * i + 2, i] -= steps[i]
    flat = stencil.reshape(-1, ndim)
    t0 = time.time()
    mu = forward(flat).reshape(K, 2 * ndim + 1, -1)
    print(f"    batched stencil: {flat.shape[0]} forwards (1 call) in "
          f"{time.time() - t0:.1f}s", flush=True)
    mu0 = np.clip(mu[:, 0, :], 1e-30, None)                    # (K, n_keep)
    J = np.zeros((K, ndim, mu0.shape[1]))
    for i in range(ndim):
        J[:, i, :] = (mu[:, 2 * i + 1, :] - mu[:, 2 * i + 2, :]) / (2.0 * steps[i])
    return mu0, J


def fisher_scoring(forward, prior, pars, truth, data_batch, n_iter):
    """K-way batched Fisher scoring: theta <- theta + F^-1 score, per seed.

    Equivalent to ``fisher_bias.linear_bias_fisher`` but iterated (relinearise
    at the current estimate each round) and solved for K seeds' data at once.
    Steps are clipped to the prior box so a wild early Newton step cannot
    walk a parameter permanently off the model's valid range.
    """
    K = data_batch.shape[0]
    ndim = len(pars)
    lo = np.array([p.low for p in pars])
    hi = np.array([p.high for p in pars])
    theta = np.tile(truth, (K, 1))
    sigma_ref = None
    for it in range(n_iter):
        print(f"  iteration {it + 1}/{n_iter}", flush=True)
        mu0, J = batched_jacobian(forward, pars, theta)
        deltas = np.zeros((K, ndim))
        for k in range(K):
            F = (J[k] / mu0[k]) @ J[k].T                       # (ndim, ndim)
            cov = np.linalg.inv(F)
            if it == 0:
                if sigma_ref is None:
                    sigma_ref = np.zeros((K, ndim))
                sigma_ref[k] = np.sqrt(np.diag(cov))            # at the truth
            resid = data_batch[k] / mu0[k] - 1.0
            score = J[k] @ resid
            deltas[k] = cov @ score
            theta[k] = np.clip(theta[k] + deltas[k], lo, hi)
        print(f"    max |step| this round: {np.abs(deltas).max():.3e}",
              flush=True)
    return theta, sigma_ref.mean(axis=0)          # sigma_ref ~constant across
                                                    # seeds (property of the
                                                    # truth, tiny K-scatter)


def lbfgs_batch(forward, prior, data_batch, truth, max_iter, tol_grad=1e-6,
                n_restarts=1, start=None, sigma_ref=None, objective="mle"):
    """K-way batched L-BFGS MLE on the autograd path. GPU only in practice.

    Rows are independent (row i's loss depends only on theta_i), so the true
    Hessian is block-diagonal; LBFGS's shared low-rank curvature approximation
    is an expedient, not exact per-row preconditioning, but it converges fine
    here because the optimum is close to the flat start (truth) and the
    posterior is close to Gaussian at high counts.

    ``objective`` picks what is maximised. ``mle`` is the likelihood alone,
    which is the quantity ``b_sys`` predicts a shift in and therefore what P6
    must measure. ``map`` adds the box prior's ``log|det J|``, i.e. the mode of
    the density in the unconstrained coordinates -- the bake-off's convention.
    The two differ by O(sigma^2 / box width), negligible at 1e8 counts, but
    they are not the same estimator and the choice should be explicit.

    ``n_restarts`` > 1 reruns L-BFGS from its own solution with a fresh
    optimizer (dropping the accumulated curvature history). The movement in
    the final pass, reported in units of ``sigma_ref``, is the convergence
    diagnostic that matters here: this optimiser is started at the truth, so
    stopping early biases the measured bias toward zero -- it fails in the
    direction that would let us declare the linearisation fine.
    """
    device = forward.device
    K = data_batch.shape[0]
    data = torch.as_tensor(data_batch, dtype=torch.float32, device=device)
    th0 = truth if start is None else start                  # (ndim,) or (K, ndim)
    th0 = torch.as_tensor(np.atleast_2d(th0), dtype=torch.float64, device=device)
    z0 = prior.to_unconstrained(th0)                      # (1, ndim) or (K, ndim)
    z = z0.expand(K, z0.shape[-1]).clone().detach().requires_grad_(True)

    def make_closure(opt):
        def closure():
            opt.zero_grad()
            theta, logdet = prior.to_constrained(z.double())
            mu = forward.counts_torch(theta, grad=True).clamp_min(1e-30)
            ll = (data.double() * torch.log(mu) - mu).sum(-1)
            if objective == "map":
                ll = ll + logdet
            loss = -ll.sum()
            loss.backward()
            return loss
        return closure

    prev = prior.to_constrained(z.detach().double())[0].cpu().numpy()
    move = None
    for r in range(n_restarts):
        opt = torch.optim.LBFGS([z], max_iter=max_iter,
                                tolerance_grad=tol_grad,
                                line_search_fn="strong_wolfe")
        t0 = time.time()
        loss = opt.step(make_closure(opt))
        cur = prior.to_constrained(z.detach().double())[0].cpu().numpy()
        move = np.abs(cur - prev)
        if sigma_ref is not None:
            move = move / sigma_ref[None, :]
        prev = cur
        print(f"  LBFGS pass {r + 1}/{n_restarts}: {time.time() - t0:.1f}s, "
              f"-logL={loss.item():.6e}, max move this pass "
              f"{move.max():.2e} sigma", flush=True)
    return prev, move


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--truth", default=os.path.join(
        RESULTS, "hot_floor", "truth_single.npz"))
    ap.add_argument("--counts", type=float, default=1e6)
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1000,
                    help="first reseed value; deliberately far from the "
                         "bake-off's --seed 0 so no draw duplicates it")
    ap.add_argument("--n_iter", type=int, default=3,
                    help="Fisher-scoring rounds (--method fisher)")
    ap.add_argument("--method", choices=["fisher", "lbfgs"], default="fisher",
                    help="fisher = CPU-safe batched Newton scoring (default); "
                         "lbfgs = batched autograd L-BFGS, GPU only")
    ap.add_argument("--device", default="cpu",
                    help="cpu for --method fisher; cuda required in practice "
                         "for --method lbfgs")
    ap.add_argument("--max_iter", type=int, default=60,
                    help="L-BFGS iterations (--method lbfgs)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--echunk", type=int, default=None)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the batched trunk (--method lbfgs, GPU)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "mle_reseed",
                                                   "reseed.npz"))
    # --- P6 mode ---
    ap.add_argument("--bias_jsonl", default=None,
                    help="Tier B bias jsonl; switches to P6 mode (measure the "
                         "linearisation factor k = bias / b_sys)")
    ap.add_argument("--truth_npz", default=None,
                    help="truth npz from the SAME Tier B sweep run")
    ap.add_argument("--point", type=int, default=None,
                    help="which sweep point; omit to take the worst by "
                         "max |b_sys|/sigma_ref")
    ap.add_argument("--n_restarts", type=int, default=2,
                    help="L-BFGS passes; the last pass's movement in sigma "
                         "units is the convergence diagnostic")
    ap.add_argument("--start_sigma", type=float, default=0.0,
                    help="start each seed displaced by this many sigma from "
                         "the truth (control against a truth-anchored optimum)")
    ap.add_argument("--objective", choices=["mle", "map"], default="mle",
                    help="mle = likelihood only, what b_sys predicts")
    ap.add_argument("--seed_chunk", type=int, default=None,
                    help="seeds per L-BFGS call (--method lbfgs). THE memory "
                         "lever: counts_torch(grad=True) does not chunk "
                         "walkers, so the graph scales with the seeds in one "
                         "call. Rows are independent optimisations, so "
                         "grouping changes no result. Default: all at once")
    args = ap.parse_args()
    args.seed = 0        # build_problem's own draw is unused (mu_true is
                          # recomputed below); kept only so the arg exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    p6 = args.bias_jsonl is not None

    if p6:
        if args.truth_npz is None:
            raise SystemExit("--bias_jsonl needs --truth_npz from the same run")
        if args.point is None:
            with open(args.bias_jsonl) as f:
                allrecs = [json.loads(l) for l in f if l.strip()]
            args.point = int(max(allrecs, key=worst_ratio)["point"])
            print(f"no --point given; taking the worst by |b_sys|/sigma_ref: "
                  f"point {args.point}", flush=True)
        rec, counts_row, tz = load_tierb(args.bias_jsonl, args.truth_npz,
                                         args.point)
        (forward, prior, pars, truth, names, mu_true,
         (rmf, arf)) = build_tierb_problem(args, rec, counts_row, tz)
        # b_sys is count-independent (F and the residual term both scale with
        # N, so F^-1 J^T diag(1/mu) r does not); sigma ~ N^-1/2.
        b_sys = np.asarray(rec["b_sys"])
        sigma_jsonl = np.asarray(rec["sigma_ref"]) * np.sqrt(
            float(rec["n_ref"]) / args.counts)
    else:
        print(f"building the bake-off problem (literature Perseus fit, "
              f"{args.device})...", flush=True)
        post0, pars, truth, names = build_problem(args)
        forward, prior = post0.forward, post0.prior
        b_sys = sigma_jsonl = None

        # recompute mu_true exactly as build_problem does internally, so every
        # reseed draws from the SAME independent-SPEX-truth mean the bake-off
        # used
        tz = np.load(args.truth)
        scale = args.counts / tz["d_inband"].sum()
        mu_true = tz["d_inband"] * scale

    data_batch = np.stack([
        np.random.default_rng(args.seed0 + k).poisson(mu_true)
        for k in range(args.n_seeds)]).astype(np.float64)

    move = None
    if args.method == "fisher":
        print(f"\ndrawing {args.n_seeds} independent Poisson realizations at "
              f"{args.counts:.1e} counts (seeds {args.seed0}.."
              f"{args.seed0 + args.n_seeds - 1}) and running {args.n_iter}"
              f"-round batched Fisher scoring...", flush=True)
        mle, sigma_ref = fisher_scoring(forward, prior, pars, truth,
                                        data_batch, args.n_iter)
    else:
        if args.device == "cpu":
            print("WARNING: --method lbfgs on CPU was measured >10min for "
                  "just 4 seeds x 60 iterations without converging; this is "
                  "meant for --device cuda.", flush=True)
        print(f"\ndrawing {args.n_seeds} independent Poisson realizations at "
              f"{args.counts:.1e} counts (seeds {args.seed0}.."
              f"{args.seed0 + args.n_seeds - 1}) and running batched "
              f"L-BFGS ({args.max_iter} iterations)...", flush=True)
        if sigma_jsonl is not None:
            sigma_ref = sigma_jsonl          # the sigma b_sys is paired with
        else:
            mu0, J = batched_jacobian(forward, pars, truth[None, :])
            F = (J[0] / mu0[0]) @ J[0].T
            sigma_ref = np.sqrt(np.diag(np.linalg.inv(F)))  # at the truth only
        start = truth
        if args.start_sigma > 0:
            # per-seed displaced starts, clipped inside the box: if the fits
            # converge to the same place from here, the result is not an
            # artefact of starting the optimiser at the answer
            rng = np.random.default_rng(args.seed0 - 1)
            start = truth[None, :] + args.start_sigma * sigma_ref[None, :] * \
                rng.standard_normal((args.n_seeds, len(truth)))
            lo = np.array([p.low for p in pars])
            hi = np.array([p.high for p in pars])
            start = np.clip(start, lo + 1e-9, hi - 1e-9)
            print(f"  starts displaced by {args.start_sigma} sigma "
                  f"(max {np.abs(start - truth[None, :]).max():.3e} absolute)",
                  flush=True)
        # Seed groups run sequentially: each row is its own optimisation
        # problem (row i's loss depends only on theta_i), so the split is
        # invisible to the result and only bounds the graph.
        sc = args.seed_chunk or args.n_seeds
        mle_parts, move_parts = [], []
        for lo_k in range(0, args.n_seeds, sc):
            hi_k = min(lo_k + sc, args.n_seeds)
            if sc < args.n_seeds:
                print(f"\nseeds {lo_k}-{hi_k - 1} of {args.n_seeds}",
                      flush=True)
            st = start if start.ndim == 1 else start[lo_k:hi_k]
            m, mv = lbfgs_batch(forward, prior, data_batch[lo_k:hi_k], truth,
                                args.max_iter, n_restarts=args.n_restarts,
                                start=st, sigma_ref=sigma_ref,
                                objective=args.objective)
            mle_parts.append(m)
            move_parts.append(mv)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        mle = np.concatenate(mle_parts, axis=0)
        move = np.concatenate(move_parts, axis=0)
        if args.n_restarts < 2:
            print("  WARNING: --n_restarts 1 gives no convergence diagnostic; "
                  "the reported movement is just the distance from the start.",
                  flush=True)
        elif move.max() > 0.05:
            print(f"  WARNING: final pass still moved {move.max():.2f} sigma "
                  f"-- NOT converged. An under-converged fit sits near its "
                  f"truth start and reports k too SMALL.", flush=True)

    if p6:
        # always report against the sigma b_sys was computed with, so k is a
        # ratio of two quantities from the same linearisation
        sigma_ref = sigma_jsonl
    pulls = (mle - truth[None, :]) / sigma_ref[None, :]
    if p6:
        # --- the P6 estimator ---------------------------------------------
        K = mle.shape[0]
        delta = mle - truth[None, :]                       # (K, ndim)
        mean_d = delta.mean(0)
        se_d = delta.std(0, ddof=1) / np.sqrt(K)
        k = mean_d / b_sys
        se_k = se_d / np.abs(b_sys)
        print(f"\nlinearisation factor at point {args.point}, {K} seeds, "
              f"{args.counts:.1e} counts")
        print(f"{'param':>10} {'b_sys/sig':>10} {'meas/sig':>10} "
              f"{'k':>8} {'+-':>7}   (k = measured bias / linear b_sys)")
        for j, n in enumerate(names):
            # a parameter the linear estimate says is unbiased has no ratio to
            # measure: k is 0/0 and its error bar diverges. Flag, do not hide.
            flag = "  (b_sys < 0.01 sigma: k undefined)" \
                if abs(b_sys[j]) < 0.01 * sigma_ref[j] else ""
            print(f"{n:>10} {b_sys[j] / sigma_ref[j]:>+10.3f} "
                  f"{mean_d[j] / sigma_ref[j]:>+10.3f} {k[j]:>+8.2f} "
                  f"{se_k[j]:>7.2f}{flag}")
        np.savez(args.out, names=names, truth=truth, sigma_ref=sigma_ref,
                 seeds=np.arange(args.seed0, args.seed0 + args.n_seeds),
                 mle=mle, pulls=pulls, counts=args.counts, point=args.point,
                 b_sys=b_sys, k=k, se_k=se_k, mean_delta=mean_d, se_delta=se_d,
                 cond_F=rec["cond_F"], n_ref=rec["n_ref"],
                 params=json.dumps(rec["params"]), method=args.method,
                 start_sigma=args.start_sigma, objective=args.objective,
                 final_move_sigma=(np.nan if move is None else move),
                 rmf=os.path.basename(rmf), arf=os.path.basename(arf))
        print(f"\nsaved {args.out}")
        print("\nk ~ 1 means the linearised b_sys is honest and Tier B's N* "
              "table stands. k > 1 means every N* is optimistic by k^2. Read "
              "the SE before either: a k whose error bar spans 1 measures "
              "nothing, and a parameter with b_sys ~ 0 cannot yield a ratio "
              "at all.")
        return

    print(f"\n{'param':>10} {'mean pull':>10} {'std':>7} {'min':>7} "
          f"{'max':>7}   (original bake-off pull)")
    bakeoff_pulls = {"Ni": 2.0, "Mn": -1.4, "S": -1.3, "sigma_v": 1.45}
    for j, n in enumerate(names):
        ref = f"  (bake-off ~{bakeoff_pulls[n]:+.1f})" if n in bakeoff_pulls else ""
        print(f"{n:>10} {pulls[:, j].mean():>+10.2f} {pulls[:, j].std():>7.2f} "
              f"{pulls[:, j].min():>+7.2f} {pulls[:, j].max():>+7.2f}{ref}")

    # Propagate the response from the truth npz rather than re-deriving it:
    # that file IS this run's provenance chain (build_problem has already
    # validated it via check_truth_response), so copying it forward keeps the
    # result attributable to the exact response the truth was built against.
    np.savez(args.out, names=names, truth=truth, sigma_ref=sigma_ref,
             seeds=np.arange(args.seed0, args.seed0 + args.n_seeds),
             mle=mle, pulls=pulls, counts=args.counts,
             rmf=str(tz["rmf"]) if "rmf" in tz.files else "unrecorded",
             arf=str(tz["arf"]) if "arf" in tz.files else "unrecorded")
    print(f"\nsaved {args.out}")
    print("\nIf a parameter's reseed pulls scatter around 0 with std~1 and no "
          "consistent sign, the bake-off's observed pull was that one draw's "
          "noise. If they cluster away from 0 in the same direction as the "
          "bake-off pull, that is real emulator bias the linear Fisher "
          "estimate underpredicts (as the hot-floor 1e8 cross-check also "
          "found).")


if __name__ == "__main__":
    main()
