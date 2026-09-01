"""Sampler bake-off: five samplers, one posterior, one comparison table.

Fits the same simulated Perseus spectrum with emcee, zeus, UltraNest, NUTS and
VI over the identical :class:`PoissonPosterior`, so what is being compared is
the algorithms and nothing else. Scores each on:

* **wall-clock** -- what you actually wait for;
* **forward evaluations**, counted in *walkers* pushed through the emulator;
* **ESS per evaluation** -- the metric that matters, because it is what decides
  whether the SBC campaign is affordable at all;
* **recovery**, median-minus-truth in units of the posterior sigma, against the
  Fisher ``b_sys`` prediction from ``fisher_bias.py``;
* **agreement** with a reference chain, per parameter, in reference sigmas;
* **log Z**, where the sampler produces one (UltraNest only).

Read ESS/eval alongside wall-clock, never instead of it. The ensemble samplers
and VI amortise a whole batch into each forward; Pyro's NUTS evaluates one
point per gradient. NUTS can therefore need far fewer evaluations and still
lose badly on wall-clock -- that is a fact about batching, not about mixing,
and the table is built to show both.

Samplers run **one at a time** and each writes its own result file, so they can
be launched as separate jobs and re-run individually without redoing the rest.
``--summarise`` then builds the table from whatever has landed.

Typical use (see also docs/tutorial_inference.md):

    # one sampler per job, detached
    nohup python -u scripts/bake_off.py --sampler emcee --device cuda \\
        > logs/bakeoff_emcee.log 2>&1 &

    # table over everything finished so far
    python scripts/bake_off.py --summarise
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
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from campaign import (                                            # noqa: E402
    PERSEUS, FREE_Z, injected_abundances, find_xrism_response,
    band_mask, EXCLUDE_PERSEUS_LITERATURE, check_truth_response, build_params)
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import SYMBOL                    # noqa: E402
from spexai.inference.abundances import AbundanceModel            # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.posterior import BoxPrior, PoissonPosterior  # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.vector_forward import VectorForward         # noqa: E402
from spexai.inference import samplers                             # noqa: E402

SAMPLERS = ("emcee", "zeus", "ultranest", "nautilus", "pocomc", "inessai",
            "nuts", "hmc", "svi")

# samplers whose ESS is a Kish effective size over importance weights rather
# than a draw count -- flagged in the table because the two are not the same
# quantity and a reader will otherwise compare them as if they were
WEIGHTED = {"nautilus", "pocomc", "inessai"}

# samplers whose ESS is fixed by construction rather than measured. VI draws
# are iid by definition, so run_svi sets ESS = n_posterior and ESS/eval is then
# just n_posterior/evals -- an artefact of two settings, not a mixing result.
# Left in the table because the wall-clock and recovery columns are real, but
# it must never be read as "31x better than nested sampling".
BY_CONSTRUCTION = {"svi"}


# --- problem -----------------------------------------------------------------

def build_problem(args):
    """The literature-strategy Perseus fit: posterior, parameters, truth."""
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response, exclude=EXCLUDE_PERSEUS_LITERATURE)
    emu = JointOperatorModel(models_dir=args.store, device=args.device,
                             accelerate=False)
    print(f"store: {args.store}\n{len(emu.models)} elements: {emu.elements}\n"
          f"response: {os.path.basename(rmf)} + {os.path.basename(arf)}",
          flush=True)

    tz = np.load(args.truth)
    check_truth_response(tz, rmf, arf)
    if int(tz["n_keep"]) != int(keep.sum()):
        raise SystemExit(
            f"truth npz has {int(tz['n_keep'])} in-band channels but this "
            f"response/band gives {int(keep.sum())}. Regenerate it with "
            f"dump_truth.py against THIS store and response -- a truth built "
            f"for the 28-element store is not valid for a 30-element one.")
    # the element set matters as much as the channel count: a 28-element truth
    # has the identical channel grid, so only this catches a stale one
    if "elements" in tz:
        truth_els = [int(z) for z in tz["elements"]]
        if truth_els != list(emu.elements):
            missing = sorted(set(emu.elements) - set(truth_els))
            raise SystemExit(
                f"truth npz was built from {len(truth_els)} elements but the "
                f"store has {len(emu.elements)} (missing from truth: "
                f"{missing}). Regenerate it against THIS store:\n"
                f"  SPEXAI_STORE={args.store} python -u "
                f"scripts/inference/dump_truth.py --mode single")
    else:
        print("WARNING: truth npz predates element-set recording; it may have "
              "been built from a different store. Regenerate to be sure.",
              flush=True)
    d_inband, norm_ref = tz["d_inband"], float(tz["norm_ref"])
    scale = args.counts / d_inband.sum()
    mu_true = d_inband * scale
    log_norm_truth = float(np.log10(norm_ref * scale))
    data = np.random.default_rng(args.seed).poisson(mu_true).astype(np.float64)
    print(f"simulated {args.counts:.1e} in-band counts "
          f"(drawn {data.sum():.3e}); log_norm_truth={log_norm_truth:.4f}",
          flush=True)

    # literature abundance scheme: free FREE_Z, every other metal tied to Fe
    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z],
                 1.0, 26)

    class _Spec:                       # build_params only needs these fields
        mode = "single"
        emu = None
    spec = _Spec()
    spec.emu = emu
    pars = build_params(spec, log_norm_truth)
    names = [p.name for p in pars]

    forward = VectorForward(
        emu, response, keep, names, ab, absorption=absorption,
        redshift=PERSEUS["z"], luminosity_distance=PERSEUS["dist_m"],
        velocity=None, device=args.device, chunk=args.chunk,
        batched=True, compile_trunk=args.compile, mem_gb=args.mem_gb,
        echunk=args.echunk)
    prior = BoxPrior.from_params(pars, device=args.device)
    post = PoissonPosterior(forward, data, prior)
    truth = np.array([p.truth for p in pars])
    return post, pars, truth, names


def _model_for(post, names):
    """Pyro model over the same forward -- what NUTS and VI consume."""
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    return SpectrumModel(post.forward, post.data_np,
                         uniform_priors(names, lo, hi,
                                        device=post.forward.device))


# --- running -----------------------------------------------------------------

def run_sampler(name, post, pars, names, args):
    center = np.array([p.truth for p in pars])
    if name == "emcee":
        # chain streamed to HDF5 as it advances: a long GPU job must not be
        # able to lose everything to a late failure
        return samplers.run_emcee(post, nwalkers=args.nwalkers,
                                  nsteps=args.nsteps, seed=args.seed,
                                  center=center,
                                  backend_path=os.path.join(
                                      args.out, "bakeoff_emcee_chain.h5"),
                                  resume=args.resume)
    if name == "zeus":
        return samplers.run_zeus(post, nwalkers=args.nwalkers,
                                 nsteps=args.zeus_steps or args.nsteps // 4,
                                 seed=args.seed, center=center)
    if name == "ultranest":
        # its own checkpoint dir, so an interrupted run can be resumed rather
        # than restarted from scratch
        return samplers.run_ultranest(post, min_num_live_points=args.live,
                                      seed=args.seed, show_status=True,
                                      logdir=os.path.join(args.out,
                                                          "ultranest"),
                                      resume=args.resume)
    if name == "nautilus":
        # its own HDF5 checkpoint, same protection as UltraNest's log_dir
        return samplers.run_nautilus(post, n_live=args.n_live,
                                     n_eff=args.n_eff, seed=args.seed,
                                     filepath=os.path.join(
                                         args.out, "nautilus.hdf5"),
                                     resume=args.resume, verbose=True)
    if name == "pocomc":
        # save_every is deliberately OFF, unlike UltraNest/nautilus above.
        # pocoMC checkpoints by dill-ing Sampler.__dict__ wholesale (it strips
        # only `pbar` and `pool`), so our likelihood -- a bound method of
        # PoissonPosterior holding VectorForward, its torch modules and, with
        # --compile, a dynamo-compiled trunk -- goes into the pickle and dies
        # on a pybind11 function record. UltraNest and nautilus are unaffected
        # because they write their own formats and never serialise the
        # likelihood. This cost us nothing: resume_state_path was never wired
        # up here, so the state files were written and never read.
        if args.resume:
            print("WARNING: --resume does not apply to pocomc (no usable "
                  "checkpoint); starting from scratch.", flush=True)
        return samplers.run_pocomc(post, n_effective=args.n_effective,
                                   n_active=args.n_active,
                                   n_total=args.n_total, seed=args.seed,
                                   output_dir=os.path.join(args.out, "pocomc"),
                                   save_every=None, progress=True)
    if name == "inessai":
        return samplers.run_inessai(post, nlive=args.n_live, seed=args.seed,
                                    target_ess=args.target_ess,
                                    output=os.path.join(args.out, "inessai"),
                                    resume=args.resume)
    if name == "nuts":
        # init_values=center puts NUTS on the same footing as every other
        # sampler here, all of which start at the truth. Pyro's default
        # (init_to_uniform) starts from a random draw of the prior box, which
        # in a posterior this sharp both handicaps it and blows up the early
        # trajectories.
        return samplers.run_nuts(_model_for(post, names),
                                 n_samples=args.nuts_samples,
                                 n_warmup=args.nuts_warmup, seed=args.seed,
                                 progress=True, full_mass=args.nuts_full_mass,
                                 max_tree_depth=args.nuts_tree_depth,
                                 init_values=center)
    if name == "hmc":
        # batched, multi-chain HMC directly on `post` -- no Pyro/_model_for
        # wrapper, unlike nuts/svi above. See samplers.run_hmc's docstring:
        # this exists because NUTS's B=1-per-gradient forward is a structural
        # GPU tax that a dense mass matrix (--nuts_full_mass) did not fix.
        return samplers.run_hmc(post, nwalkers=args.nwalkers,
                                n_samples=args.hmc_samples,
                                n_warmup=args.hmc_warmup,
                                n_leapfrog=args.hmc_leapfrog,
                                step_size_init=args.hmc_step_size,
                                target_accept=args.hmc_target_accept,
                                seed=args.seed, center=center,
                                walker_chunk=args.hmc_walker_chunk)
    if name == "svi":
        return samplers.run_svi(_model_for(post, names), steps=args.svi_steps,
                                num_particles=args.svi_particles,
                                lr=args.svi_lr, seed=args.seed,
                                guide=args.svi_guide,
                                guide_transforms=args.svi_transforms,
                                particle_chunk=args.svi_particle_chunk)
    raise SystemExit(f"unknown sampler {name!r}; choose from {SAMPLERS}")


def save(res, truth, path):
    np.savez(path, name=res.name, names=list(res.names), samples=res.samples,
             truth=truth, runtime_s=res.runtime_s, n_eval=res.n_eval,
             ess=res.ess if res.ess is not None else np.array([]),
             logz=np.nan if res.logz is None else res.logz,
             logzerr=np.nan if res.logzerr is None else res.logzerr)
    print(f"saved {path}", flush=True)


# --- reporting ---------------------------------------------------------------

def _load_all(outdir):
    out = {}
    for name in SAMPLERS:
        p = os.path.join(outdir, f"bakeoff_{name}.npz")
        if os.path.exists(p):
            out[name] = np.load(p, allow_pickle=True)
    return out


def summarise(outdir, reference="emcee"):
    got = _load_all(outdir)
    if not got:
        raise SystemExit(f"no results in {outdir}; run some samplers first")
    print(f"\n{'sampler':>10} {'wall(s)':>10} {'evals':>12} {'minESS':>9} "
          f"{'ESS/eval':>10} {'logZ':>12}")
    for name, z in got.items():
        ess = z["ess"]
        m = float(np.min(ess)) if ess.size else np.nan
        per = m / float(z["n_eval"]) if z["n_eval"] else np.nan
        lz = float(z["logz"])
        print(f"{name:>10} {float(z['runtime_s']):>10.1f} "
              f"{int(z['n_eval']):>12d} {m:>9.0f} {per:>10.2e} "
              f"{('%.2f' % lz) if np.isfinite(lz) else '-':>12}"
              f"{'  *' if name in WEIGHTED else '  !' if name in BY_CONSTRUCTION else ''}")
    if any(n in WEIGHTED for n in got):
        print("  * minESS is a Kish effective size over importance weights, "
              "not a draw count.\n    Comparable to the others as an ESS, but "
              "the raw sample array is larger.")
    if any(n in BY_CONSTRUCTION for n in got):
        print("  ! minESS is FIXED BY CONSTRUCTION (n_posterior), not "
              "measured. ESS/eval for this\n    row is an artefact of the "
              "settings and is NOT comparable -- judge it on the recovery\n"
              "    and agreement tables instead.")
    print("  Read ESS/eval with wall-clock: the ensembles amortise a batch "
          "into each\n  forward, NUTS evaluates one point per gradient.")

    ref = got.get(reference)
    if ref is None:
        print(f"\n(no {reference} result, skipping the agreement table)")
        return
    names = [str(n) for n in ref["names"]]
    rmed = np.median(ref["samples"], axis=0)
    rsig = 0.5 * (np.percentile(ref["samples"], 84, axis=0)
                  - np.percentile(ref["samples"], 16, axis=0))

    print(f"\nRECOVERY: (median - truth) / posterior sigma")
    print(f"{'param':>10} " + " ".join(f"{n:>10}" for n in got))
    truth = ref["truth"]
    for i, pname in enumerate(names):
        row = f"{pname:>10} "
        for name, z in got.items():
            s = z["samples"][:, i]
            sig = 0.5 * (np.percentile(s, 84) - np.percentile(s, 16))
            row += f"{(np.median(s) - truth[i]) / max(sig, 1e-30):>+10.2f} "
        print(row)

    print(f"\nAGREEMENT with {reference}: |median - ref| / ref sigma, and "
          f"width ratio")
    for name, z in got.items():
        if name == reference:
            continue
        med = np.median(z["samples"], axis=0)
        sig = 0.5 * (np.percentile(z["samples"], 84, axis=0)
                     - np.percentile(z["samples"], 16, axis=0))
        dmed = np.abs(med - rmed) / np.clip(rsig, 1e-30, None)
        print(f"  {name:>10}: max shift {dmed.max():.2f} sigma "
              f"({names[int(np.argmax(dmed))]}), width ratio "
              f"{np.min(sig/np.clip(rsig,1e-30,None)):.2f}-"
              f"{np.max(sig/np.clip(rsig,1e-30,None)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=SAMPLERS,
                    help="run exactly one sampler (the usual mode: one job each)")
    ap.add_argument("--all", action="store_true",
                    help="run every sampler serially in this process")
    ap.add_argument("--summarise", action="store_true",
                    help="build the table from saved results and exit")
    ap.add_argument("--resume", action="store_true",
                    help="continue emcee from its HDF5 backend instead of "
                         "restarting (no-op for the other samplers)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--truth", default=os.path.join(
        RESULTS, "hot_floor", "truth_single.npz"))
    ap.add_argument("--counts", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "bakeoff"))
    # forward
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--echunk", type=int, default=None,
                    help="energy-chunk size; also the gradient-checkpoint "
                         "segment, so it is the memory lever for nuts/svi "
                         "(try 2048 or 1024 if they OOM)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the batched trunk (recommended on GPU)")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--fft32", action="store_true")
    # per-sampler budgets
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=800)
    ap.add_argument("--n_live", type=int, default=2000,
                    help="live points for nautilus / i-nessai")
    ap.add_argument("--n_eff", type=int, default=10000,
                    help="nautilus target effective sample size")
    ap.add_argument("--target_ess", type=float, default=2000.0,
                    help="i-nessai stops when the effective sample size "
                         "reaches this (its 'ratio' default stops far too "
                         "early on a peaked likelihood)")
    ap.add_argument("--n_effective", type=int, default=512,
                    help="pocoMC effective particles")
    ap.add_argument("--n_active", type=int, default=256,
                    help="pocoMC active particles (its forward batch size)")
    ap.add_argument("--n_total", type=int, default=4096,
                    help="pocoMC total posterior samples")
    ap.add_argument("--zeus_steps", type=int, default=0,
                    help="0 = nsteps//4 (zeus mixes faster per step)")
    ap.add_argument("--live", type=int, default=400)
    ap.add_argument("--nuts_samples", type=int, default=500)
    ap.add_argument("--nuts_warmup", type=int, default=500)
    ap.add_argument("--nuts_full_mass", action="store_true",
                    help="dense mass matrix. Pyro's diagonal default cannot "
                         "represent the abundance/norm correlation, which is "
                         "what makes the trajectories saturate")
    ap.add_argument("--nuts_tree_depth", type=int, default=10,
                    help="max doublings: 10 allows 1023 gradients/iteration, "
                         "7 allows 127. Break-even vs the batched ensembles "
                         "is around 50 steps")
    ap.add_argument("--hmc_samples", type=int, default=500,
                    help="post-warmup HMC iterations")
    ap.add_argument("--hmc_warmup", type=int, default=500,
                    help="dual-averaging step-size warmup iterations")
    ap.add_argument("--hmc_leapfrog", type=int, default=20,
                    help="fixed leapfrog steps/iteration (no tree-doubling); "
                         "tune this if mixing is poor")
    ap.add_argument("--hmc_step_size", type=float, default=0.01,
                    help="dual-averaging starting step size")
    ap.add_argument("--hmc_target_accept", type=float, default=0.8)
    ap.add_argument("--hmc_walker_chunk", type=int, default=None,
                    help="chunk the batched gradient over this many walkers "
                         "at a time (default: all nwalkers at once). "
                         "counts_torch(grad=True) is unchunked, so peak "
                         "memory scales linearly with nwalkers and can OOM; "
                         "lower this before lowering --nwalkers")
    ap.add_argument("--svi_steps", type=int, default=2000)
    ap.add_argument("--svi_particles", type=int, default=64)
    ap.add_argument("--svi_lr", type=float, default=1e-2)
    ap.add_argument("--svi_guide", default="mvn",
                    choices=["mvn", "iaf", "lowrank", "normal"],
                    help="variational family: mvn = full-rank Gaussian "
                         "(default), iaf = normalizing flow, normal = "
                         "mean-field control")
    ap.add_argument("--svi_transforms", type=int, default=2,
                    help="IAF only: number of autoregressive transforms")
    ap.add_argument("--svi_particle_chunk", type=int, default=0,
                    help="particles per backward pass (0 = all at once). "
                         "THE fix for SVI OOM: memory scales with this, "
                         "gradient quality with --svi_particles")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets, just to prove the wiring runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.summarise:
        summarise(args.out)
        return
    if not args.sampler and not args.all:
        raise SystemExit("pass --sampler <name>, --all, or --summarise")
    if args.smoke:
        args.nwalkers, args.nsteps, args.counts = 24, 20, 1e5
        args.live, args.nuts_samples, args.nuts_warmup = 50, 20, 20
        args.hmc_samples, args.hmc_warmup, args.hmc_leapfrog = 20, 20, 5
        # counts_torch(grad=True) is unchunked over its batch dimension (see
        # run_hmc's docstring), so the shared --nwalkers=24 above would OOM a
        # gradient-based sampler that a forward-only one handles fine;
        # --hmc_walker_chunk lets the user raise this back up deliberately
        args.hmc_walker_chunk = args.hmc_walker_chunk or 2
        args.svi_steps, args.svi_particles = 50, 8
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.fft32:
        import spexai.broadening as _br
        _br.USE_FLOAT32_FFT = True

    post, pars, truth, names = build_problem(args)
    todo = list(SAMPLERS) if args.all else [args.sampler]
    for name in todo:
        print(f"\n=== {name} ===", flush=True)
        post.n_eval = 0
        t0 = time.time()
        res = run_sampler(name, post, pars, names, args)
        print(res.summary(truths=truth), flush=True)
        print(f"({name} finished in {time.time()-t0:.0f}s)", flush=True)
        save(res, truth, os.path.join(args.out, f"bakeoff_{name}.npz"))
    summarise(args.out)


if __name__ == "__main__":
    main()
