"""Cluster MCMC cross-check of the linearised bias (Phase 2).

Runs a real emcee fit of a simulated Perseus spectrum with the SAME masked,
literature-strategy model used by the Fisher engine (`fisher_bias.Forward`), so
the posterior offset (median - truth) is directly comparable to the analytic
`b_sys`. Because the Poisson likelihood cost is independent of the total counts,
we can simulate near `N*` -- where the systematic bias reaches ~1 sigma and is
directly detectable -- to confirm the linearisation.

Local smoke (a few minutes, just checks it runs):
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python -u \
      inference_demo/hot_floor/mcmc_check.py --smoke

Cluster (scale up; use a process pool):
  MKL_THREADING_LAYER=GNU python -u inference_demo/hot_floor/mcmc_check.py \
      --mode single --counts 1e8 --nwalkers 48 --nsteps 4000 --pool 24
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (                                    # noqa: E402
    STORE, find_xrism_response, band_mask,
    TruthConfig, stream_truth_counts, gaussian_dem, resolve_perseus)
from fisher_bias import Forward, build_params, N_REF        # noqa: E402
from spexai.inference.response import Response               # noqa: E402
from spexai.inference.absorption import Absorption           # noqa: E402
from spexai.inference.operator_model import JointOperatorModel  # noqa: E402

torch.set_num_threads(1)          # one thread/worker; the pool provides parallelism

# module-level so a multiprocessing pool can pickle the closure targets
_FWD = None
_D = None
_LO = None
_HI = None


def _logprob(theta):
    if np.any(theta < _LO) or np.any(theta > _HI):
        return -np.inf
    mu = np.clip(_FWD(theta), 1e-30, None)
    return float(np.sum(_D * np.log(mu) - mu))


def _profile(fwd, response, n, device):
    """Per-eval timing of the full forward and of the (always-CPU) fold, to see
    how much of the cost a GPU can actually remove (stage 2 is fold-bound)."""
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    x0 = np.array([p.truth for p in build_params(fwd, 15.0)])
    fwd(x0); sync()                                        # warm up
    t = time.time()
    for _ in range(n):
        fwd(x0)
    sync()
    per = (time.time() - t) / n
    flux = torch.ones((1, response.n_energy))              # fold-only (CPU sparse)
    t = time.time()
    for _ in range(n):
        response.fold(flux)
    foldper = (time.time() - t) / n
    print(f"device={device}: forward {per*1e3:.0f} ms/eval; "
          f"fold {foldper*1e3:.0f} ms/eval = {100*foldper/per:.0f}% of forward "
          f"(the fold stays on CPU, so that fraction is your GPU speedup ceiling)",
          flush=True)


def _run_vectorized(emu, response, absorption, keep, args, log_norm_truth, d,
                    perseus):
    """Walker-batched all-GPU forward + emcee vectorized=True (single process)."""
    import emcee
    from gpu_forward import EnsembleForward
    # --fix_sigma_v pins sigma_v (the old Tier-1 behaviour); otherwise it is a
    # free per-walker parameter like the rest.
    ens = EnsembleForward(emu, response, absorption, keep, args.mode,
                          perseus["vel"] if args.fix_sigma_v else None,
                          args.device, chunk=args.chunk,
                          compile_nets=args.compile,
                          batched=not args.serial_forward,
                          mem_gb=args.mem_gb, perseus=perseus)
    pars = ens.params(log_norm_truth)
    names = [p.name for p in pars]
    lo = np.array([p.low for p in pars])
    hi = np.array([p.high for p in pars])
    x0 = np.array([p.truth for p in pars])
    ndim = len(pars)

    def logprob(thetas):                                  # (n, ndim) -> (n,)
        thetas = np.atleast_2d(thetas)
        ok = np.all((thetas >= lo) & (thetas <= hi), axis=1)
        out = np.full(thetas.shape[0], -np.inf)
        if ok.any():
            mu = np.clip(ens(thetas[ok]), 1e-30, None)    # (n_ok, n_keep)
            out[ok] = (d[None, :] * np.log(mu) - mu).sum(1)
        return out

    rng = np.random.default_rng(args.seed)
    p0 = x0 + np.array([p.step * 5 for p in pars]) * rng.standard_normal(
        (args.nwalkers, ndim))
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    # One-off torch.compile stall: the benchmarks all warm up before timing, so
    # this cost never appeared in the ladder. Amortised over nwalkers*nsteps it
    # should be negligible, but it is worth knowing for short runs (e.g. an SBC
    # campaign of many short chains, where it is paid once per chain).
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (
        lambda: None)
    t0 = time.time()
    ens(p0[:1])
    sync()
    t_first = time.time() - t0
    t0 = time.time()
    ens(p0[:1])
    sync()
    t_warm = time.time() - t0
    print(f"forward: first call {t_first:.1f}s, warm {t_warm:.2f}s "
          f"(one-off overhead {t_first - t_warm:.1f}s)", flush=True)

    t0 = time.time()
    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, logprob, vectorize=True)
    every = max(1, args.nsteps // 20)                    # ~20 progress lines
    for i, _ in enumerate(sampler.sample(p0, iterations=args.nsteps)):
        if i == 0 or (i + 1) % every == 0:               # first step -> early ETA
            el = time.time() - t0
            print(f"  step {i+1}/{args.nsteps}  {el:.0f}s elapsed  "
                  f"(~{el/(i+1)*args.nsteps:.0f}s projected)", flush=True)
    print(f"emcee(vectorized,{args.device}): {args.nwalkers}x{args.nsteps} in "
          f"{time.time()-t0:.0f}s", flush=True)

    chain = sampler.get_chain(discard=int(0.4 * args.nsteps), flat=True)
    med = np.median(chain, axis=0)
    sig = 0.5 * (np.percentile(chain, 84, axis=0)
                 - np.percentile(chain, 16, axis=0))
    print(f"\n{'param':>9} {'truth':>10} {'median':>10} {'post.bias':>11} "
          f"{'post.sigma':>10}  bias/sigma")
    for i, p in enumerate(pars):
        print(f"{p.name:>9} {p.truth:>10.4g} {med[i]:>10.4g} "
              f"{med[i]-p.truth:>+11.3e} {sig[i]:>10.3e}  "
              f"{(med[i]-p.truth)/sig[i]:>+.2f}")
    # posterior-predictive band: fold a random subset of posterior draws
    rng2 = np.random.default_rng(1)
    idx = rng2.choice(chain.shape[0], size=min(64, chain.shape[0]), replace=False)
    ppc = ens(chain[idx])                                # (Nppc, n_keep)
    chan_e = response.chan_e_cent.cpu().numpy()[keep]
    suffix = f"_{args.tag}" if args.tag else ""
    outp = os.path.join(args.out, f"mcmc_{args.mode}_vec{suffix}.npz")
    np.savez(outp, names=names, truth=x0, chain=chain,
             full_chain=sampler.get_chain(), log_prob=sampler.get_log_prob(),
             nwalkers=args.nwalkers, discard=int(0.4 * args.nsteps),
             obs_data=d, chan_e=chan_e, ppc=ppc,
             counts=args.counts, n_ref=N_REF)
    print(f"saved {outp}", flush=True)


def main():
    global _FWD, _D, _LO, _HI
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--counts", type=float, default=1e8,
                    help="in-band total counts to simulate (near N* by default)")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=0, help="process-pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_v", type=float, default=None)
    ap.add_argument("--kT", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--truth_npz", default=None,
                    help="precomputed truth from dump_truth.py; skips the "
                         "cache-dependent streaming (use this on the cluster)")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; the operator nets run here (the response "
                         "fold is always CPU/scipy-sparse)")
    ap.add_argument("--profile", type=int, default=0,
                    help="time this many forwards (and the fold's share) on "
                         "--device, then exit; no truth/sampling needed")
    ap.add_argument("--vectorized", action="store_true",
                    help="walker-batched all-GPU forward + emcee vectorized "
                         "(single process); use with --device cuda")
    ap.add_argument("--fix_sigma_v", action="store_true",
                    help="pin sigma_v at the truth instead of sampling it "
                         "(the original Tier-1 behaviour; --vectorized only)")
    ap.add_argument("--chunk", type=int, default=32,
                    help="walker sub-batch size for the GPU forward (bounds "
                         "memory); lower if you hit CUDA OOM")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 tensor-core matmul (Ampere+)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the trunk (the grouped vmap forward on "
                         "the batched path, the per-element MLP on the serial "
                         "one); this is where the batched path's 2.19x lives")
    ap.add_argument("--serial_forward", action="store_true",
                    help="use the old per-element serial loop instead of the "
                         "element-batched trunk (reference / debugging)")
    ap.add_argument("--mem_gb", type=float, default=2.0,
                    help="memory budget steering the batched forward's energy "
                         "and row chunking (numerically transparent)")
    ap.add_argument("--fft32", action="store_true",
                    help="float32 FFT continuum broadening on CUDA (~4x)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (overrides nwalkers/nsteps/counts)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.fft32:
        import spexai.train.broadening as _br
        _br.USE_FLOAT32_FFT = True
    if args.smoke:
        # 24 >= 2*ndim: emcee's red-blue move refuses fewer walkers than twice
        # the dimension, and ndim is 11 here (8 abundances + kT + sigma_v + n_h
        # + log_norm, less one when sigma_v is pinned).
        args.nwalkers, args.nsteps, args.counts = 24, 40, 1e6
    perseus = resolve_perseus({"vel": args.sigma_v, "kT": args.kT})

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response)
    # accelerate=False: this cross-check drives the accel knobs itself
    # (--tf32/--compile/--fft32, and the vectorized EnsembleForward compile) so
    # ACCEL="" gives a genuine float64 reference; the model must not auto-enable.
    emu = JointOperatorModel(models_dir=STORE, device=args.device,
                             accelerate=False)
    dem, dem_p = (gaussian_dem() if args.mode == "dem" else (None, {}))

    if args.profile:                                     # measure per-eval cost
        _profile(Forward(emu, response, absorption, keep, args.mode, dem=dem,
                         perseus=perseus),
                 response, args.profile, args.device)
        return

    if args.truth_npz:                                   # cluster path: no caches
        tz = np.load(args.truth_npz)
        assert int(tz["n_keep"]) == int(keep.sum()), (
            "truth npz was built with a different response/band")
        # element set too: a 28-element truth has the identical channel grid,
        # so the check above cannot tell it from a 30-element one
        if "elements" in tz:
            assert [int(z) for z in tz["elements"]] == list(emu.elements), (
                f"truth npz has {len(tz['elements'])} elements but the store "
                f"has {len(emu.elements)}; regenerate with dump_truth.py "
                f"against this store")
        else:
            print("WARNING: truth npz predates element-set recording; "
                  "regenerate it to be sure it matches this store", flush=True)
        d_inband, norm_ref = tz["d_inband"], float(tz["norm_ref"])
    else:                                                # local path: stream truth
        from experiment import injected_abundances
        cfg = TruthConfig(elements=emu.elements,
                          abundances=injected_abundances(emu.elements),
                          exposure=1.0, dem=dem, dem_params=dem_p)
        d_inband = stream_truth_counts(cfg, response, absorption,
                                       perseus=perseus)[keep]
        norm_ref = cfg.norm_ref
    s = args.counts / d_inband.sum()
    mu_true = d_inband * s                                 # in-band truth mean
    log_norm_truth = float(np.log10(norm_ref * s))
    rng = np.random.default_rng(args.seed)
    _D = rng.poisson(mu_true).astype(np.float64)
    print(f"simulated {args.counts:.1e} in-band counts (drawn {_D.sum():.3e})",
          flush=True)

    if args.vectorized:                                   # GPU walker-batched path
        _run_vectorized(emu, response, absorption, keep, args, log_norm_truth, _D,
                        perseus)
        return

    _FWD = Forward(emu, response, absorption, keep, args.mode, dem=dem,
                   perseus=perseus)
    pars = build_params(_FWD, log_norm_truth)
    _LO = np.array([p.low for p in pars])
    _HI = np.array([p.high for p in pars])
    x0 = np.array([p.truth for p in pars])
    ndim = len(pars)

    import emcee
    p0 = x0 + np.array([p.step * 5 for p in pars]) * rng.standard_normal(
        (args.nwalkers, ndim))
    p0 = np.clip(p0, _LO + 1e-9, _HI - 1e-9)

    t0 = time.time()
    pool = None
    if args.pool > 0:
        # fork so workers inherit the module globals set above (_FWD/_D/_LO/_HI);
        # spawn (macOS default) would re-import the module and lose them.
        from multiprocessing import get_context
        pool = get_context("fork").Pool(args.pool)
    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, _logprob, pool=pool)
    sampler.run_mcmc(p0, args.nsteps, progress=True)
    if pool is not None:
        pool.close()
    print(f"emcee: {args.nwalkers}x{args.nsteps} in {time.time()-t0:.0f}s",
          flush=True)

    discard = int(0.4 * args.nsteps)
    chain = sampler.get_chain(discard=discard, flat=True)
    print(f"\n{'param':>9} {'truth':>10} {'median':>10} {'post.bias':>11} "
          f"{'post.sigma':>10}  b_sys/sigma")
    med = np.median(chain, axis=0)
    sig = 0.5 * (np.percentile(chain, 84, axis=0)
                 - np.percentile(chain, 16, axis=0))
    for i, p in enumerate(pars):
        print(f"{p.name:>9} {p.truth:>10.4g} {med[i]:>10.4g} "
              f"{med[i]-p.truth:>+11.3e} {sig[i]:>10.3e}  "
              f"{(med[i]-p.truth)/sig[i]:>+.2f}")
    suffix = f"_{args.tag}" if args.tag else ""
    outp = os.path.join(args.out, f"mcmc_{args.mode}{suffix}.npz")
    np.savez(outp, names=[p.name for p in pars], truth=x0, chain=chain,
             counts=args.counts, n_ref=N_REF)
    print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
