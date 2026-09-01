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
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from bake_off import build_problem                                # noqa: E402
from spexai.config import STORE, RESULTS                          # noqa: E402


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


def lbfgs_batch(forward, prior, data_batch, truth, max_iter, tol_grad=1e-6):
    """K-way batched L-BFGS MLE on the autograd path. GPU only in practice.

    Rows are independent (row i's loss depends only on theta_i), so the true
    Hessian is block-diagonal; LBFGS's shared low-rank curvature approximation
    is an expedient, not exact per-row preconditioning, but it converges fine
    here because the optimum is close to the flat start (truth) and the
    posterior is close to Gaussian at high counts.
    """
    device = forward.device
    K = data_batch.shape[0]
    data = torch.as_tensor(data_batch, dtype=torch.float32, device=device)
    z0 = prior.to_unconstrained(
        torch.as_tensor(truth, dtype=torch.float64, device=device))
    z = z0.unsqueeze(0).repeat(K, 1).clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=max_iter, tolerance_grad=tol_grad,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        theta, logdet = prior.to_constrained(z.double())
        mu = forward.counts_torch(theta, grad=True).clamp_min(1e-30)
        ll = (data.double() * torch.log(mu) - mu).sum(-1) + logdet
        loss = -ll.sum()
        loss.backward()
        return loss

    t0 = time.time()
    opt.step(closure)
    print(f"  LBFGS: {time.time() - t0:.1f}s", flush=True)
    theta_mle, _ = prior.to_constrained(z.detach().double())
    return theta_mle.cpu().numpy()


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
    args = ap.parse_args()
    args.seed = 0        # build_problem's own draw is unused (mu_true is
                          # recomputed below); kept only so the arg exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"building the bake-off problem (literature Perseus fit, "
          f"{args.device})...", flush=True)
    post0, pars, truth, names = build_problem(args)
    forward, prior = post0.forward, post0.prior

    # recompute mu_true exactly as build_problem does internally, so every
    # reseed draws from the SAME independent-SPEX-truth mean the bake-off used
    tz = np.load(args.truth)
    scale = args.counts / tz["d_inband"].sum()
    mu_true = tz["d_inband"] * scale

    data_batch = np.stack([
        np.random.default_rng(args.seed0 + k).poisson(mu_true)
        for k in range(args.n_seeds)]).astype(np.float64)

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
        mu0, J = batched_jacobian(forward, pars, truth[None, :])
        F = (J[0] / mu0[0]) @ J[0].T
        sigma_ref = np.sqrt(np.diag(np.linalg.inv(F)))     # at the truth only
        mle = lbfgs_batch(forward, prior, data_batch, truth, args.max_iter)

    pulls = (mle - truth[None, :]) / sigma_ref[None, :]
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
