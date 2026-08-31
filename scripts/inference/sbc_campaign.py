"""Simulation-based calibration campaign on the batched inference stack.

The SBC half of ``bias_study.py``, rebuilt on the stack the bake-off uses
(``VectorForward`` / ``PoissonPosterior`` / ``spexai.inference.samplers``) so it
can run **whichever sampler the bake-off shows to be affordable** rather than
being hardwired to emcee on CPU.

What each simulation does:

1. draw ``theta ~ prior`` (the *same* box the fit uses -- this is what makes the
   rank test valid);
2. push it through the emulator to get expected counts, and Poisson-sample;
3. fit with the chosen sampler;
4. thin the chain to near-independent draws and record the rank of the truth.

Three things differ from the old ``--stage sbc`` and each was a correctness
problem, not a refactor:

* **Ranks are computed on thinned draws.** The old code used ``mean(s < t)``
  over the raw correlated chain, which fails the uniformity test even for a
  perfectly calibrated sampler. See ``spexai.inference.calibration``.
* **The prior is fixed across simulations.** ``fisher_bias.build_params``
  centres the ``log_norm`` box on the truth; under SBC that makes the prior a
  function of the draw, and the ranks are then uniform by construction for
  ``log_norm`` no matter how badly the sampler behaves.
* **Injection is the emulator itself, always.** SBC asks whether the posterior
  is self-consistent for the model being fitted. Injecting SPEX truth instead
  measures emulator bias, which is ``bias_study.py --stage point``'s job, and
  mixing the two makes any non-uniformity unattributable.

The forward and the emulator are built **once** and reused for every
simulation; only the data changes, so per-sim setup is free.

Every simulation appends to a JSONL file the moment it finishes, and ``--resume``
skips simulations already in it. A campaign is days long and the non-ensemble
samplers have no checkpointing of their own, so the protection has to live here.

    python -u scripts/sbc_campaign.py --sampler emcee --n_sims 100 \\
        --device cuda --compile --tf32 --fft32 --resume
    python scripts/sbc_campaign.py --summarise --out sbc_emcee
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
    PERSEUS, FREE_Z, injected_abundances, find_xrism_response, band_mask)
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import AbundanceModel, SYMBOL    # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.fitting import SIGMA_V_PRIOR                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.posterior import PoissonPosterior           # noqa: E402
from spexai.inference.priors import PriorSet                      # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.vector_forward import VectorForward         # noqa: E402
from spexai.inference import calibration, samplers                # noqa: E402

SAMPLERS = ("emcee", "zeus", "ultranest", "nuts", "svi")


# --- the fixed SBC prior -----------------------------------------------------

def sbc_prior(elements, log_norm_ref: float, abund_range=(0.2, 2.0)):
    """Parameter names and a box that does **not** depend on the drawn truth.

    Deliberately narrower than the bake-off's single-fit boxes on the
    abundances: SBC draws truths from this box and must simulate spectra that
    are physically sensible at every corner of it, and a 0.02-3.0 abundance box
    puts most of its volume on spectra no cluster resembles. Widen it only with
    a reason -- a wide box makes the campaign slower (walkers start further out)
    without making the calibration test stronger.
    """
    names, lo, hi = [], [], []
    for z in FREE_Z:
        names.append(SYMBOL[z])
        lo.append(abund_range[0])
        hi.append(abund_range[1])
    names += ["kT", "sigma_v", "n_h", "log_norm"]
    lo += [1.5, SIGMA_V_PRIOR[0], 0.0, log_norm_ref - 1.0]
    hi += [7.5, SIGMA_V_PRIOR[1], 5.0, log_norm_ref + 1.0]
    return names, np.array(lo, dtype=float), np.array(hi, dtype=float)


def build_forward(args):
    """Emulator, response and forward -- built once, reused by every sim."""
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    keep = band_mask(response)
    emu = JointOperatorModel(models_dir=args.store, device=args.device,
                             accelerate=False)
    # SBC draws its own truths through this same forward, so the ARF cancels
    # by construction -- it is printed only so a run's provenance is on record.
    print(f"store: {args.store}\n{len(emu.models)} elements: {emu.elements}\n"
          f"response: {os.path.basename(rmf)} + {os.path.basename(arf)}",
          flush=True)

    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z],
                 1.0, 26)

    names, lo, hi = sbc_prior(emu.elements, args.log_norm_ref)
    forward = VectorForward(
        emu, response, keep, names, ab, absorption=Absorption.default(),
        redshift=PERSEUS["z"], luminosity_distance=PERSEUS["dist_m"],
        velocity=None, device=args.device, chunk=args.chunk,
        batched=True, compile_trunk=args.compile, mem_gb=args.mem_gb,
        echunk=args.echunk)
    # a PriorSet, not a BoxPrior: SBC must *draw* from the prior, and going
    # through prior.sample keeps this script correct if the box is later
    # swapped for informative priors
    prior = PriorSet.uniform(names, lo, hi, device=args.device)
    return forward, prior, names


# --- one simulation ----------------------------------------------------------

def simulate(forward, prior, theta_true, rng):
    """Emulator-injected Poisson counts at ``theta_true``. -> (n_keep,)"""
    mu = np.clip(forward(theta_true[None, :])[0], 1e-30, None)   # (n_keep,)
    return rng.poisson(mu).astype(np.float64), mu


def run_one(i, forward, prior, names, args):
    """One SBC replicate. Returns the record appended to the JSONL."""
    rng = np.random.default_rng(args.seed + i)
    # drawn from the prior itself, not its bounding box -- with an informative
    # prior those differ, and SBC is only valid if the truth comes from the
    # same distribution the fit assumes
    theta_true = prior.sample(rng, 1)[0]                         # (ndim,)
    data, mu = simulate(forward, prior, theta_true, rng)
    post = PoissonPosterior(forward, data, prior)

    t0 = time.time()
    res = run_sampler(args.sampler, post, names, theta_true, args, seed=i)
    # walkers are separate chains, so thin the STEP axis and keep all of them
    if res.chain is not None:
        draws, thin = calibration.thin_to_independent(
            res.chain, res.ess, max_draws=args.n_draws, rng=rng)
    else:
        # nested sampling / VI already return decorrelated draws
        draws, thin = res.samples, 1
        if len(draws) > args.n_draws:
            draws = draws[rng.choice(len(draws), args.n_draws, replace=False)]
    ranks = calibration.sbc_rank(draws, theta_true)

    med = np.median(draws, axis=0)
    q16, q84 = np.percentile(draws, [16, 84], axis=0)
    sigma = np.where((q84 - q16) > 0, 0.5 * (q84 - q16), 1e-30)
    return {
        "sim": i,
        "sampler": args.sampler,
        "names": list(names),
        "truth": theta_true.tolist(),
        "rank": ranks.tolist(),
        "n_draws": int(len(draws)),
        "thin": int(thin),
        "median": med.tolist(),
        "sigma": sigma.tolist(),
        "pull": ((med - theta_true) / sigma).tolist(),
        "covered": ((q16 <= theta_true) & (theta_true <= q84)).tolist(),
        "total_counts": float(data.sum()),
        "min_ess": float(res.min_ess),
        "n_eval": int(res.n_eval),
        "runtime_s": float(time.time() - t0),
    }


def run_sampler(name, post, names, center, args, seed=0):
    """Dispatch, mirroring bake_off.run_sampler but per-simulation.

    Walkers start at the truth: SBC measures calibration, not the sampler's
    ability to find the mode from a cold start, and a lost chain would show up
    as a rank pathology that has nothing to do with calibration.
    """
    if name == "emcee":
        return samplers.run_emcee(post, nwalkers=args.nwalkers,
                                  nsteps=args.nsteps, seed=seed, center=center)
    if name == "zeus":
        return samplers.run_zeus(post, nwalkers=args.nwalkers,
                                 nsteps=args.zeus_steps or args.nsteps // 4,
                                 seed=seed, center=center)
    if name == "ultranest":
        return samplers.run_ultranest(post, min_num_live_points=args.live,
                                      seed=seed, show_status=False)
    model = _model_for(post, names)
    if name == "nuts":
        return samplers.run_nuts(model, n_samples=args.nuts_samples,
                                 n_warmup=args.nuts_warmup, seed=seed,
                                 progress=False)
    if name == "svi":
        return samplers.run_svi(model, steps=args.svi_steps,
                                num_particles=args.svi_particles,
                                lr=args.svi_lr, seed=seed)
    raise SystemExit(f"unknown sampler {name!r}; choose from {SAMPLERS}")


def _model_for(post, names):
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    return SpectrumModel(post.forward, post.data_np,
                         uniform_priors(names, post.prior.lo.cpu().numpy(),
                                        post.prior.hi.cpu().numpy(),
                                        device=post.forward.device))


# --- resume ------------------------------------------------------------------

def done_sims(path):
    """Indices already on disk. A truncated final line (killed mid-write) is
    dropped rather than crashing the resume."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            try:
                out.add(int(json.loads(line)["sim"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())        # a node that dies must not lose the record


# --- reporting ---------------------------------------------------------------

def summarise(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        raise SystemExit(f"no records in {path}")
    names = recs[0]["names"]
    n_draws = int(np.min([r["n_draws"] for r in recs]))
    ranks = {n: [r["rank"][i] for r in recs] for i, n in enumerate(names)}
    print(f"{len(recs)} simulations from {path}")
    print(calibration.summarise(ranks, n_draws, names))
    pulls = np.array([r["pull"] for r in recs])           # (n_sims, ndim)
    cov = np.array([r["covered"] for r in recs]).mean(axis=0)
    print(f"\n{'param':>12} {'pull mean':>10} {'pull std':>9} {'cov68':>7}")
    for i, n in enumerate(names):
        print(f"{n:>12} {pulls[:, i].mean():>+10.3f} "
              f"{pulls[:, i].std():>9.3f} {cov[i]:>7.2f}")
    rt = np.array([r["runtime_s"] for r in recs])
    print(f"\nruntime/sim: {rt.mean() / 3600:.2f} h mean, "
          f"{rt.sum() / 3600:.1f} h total; "
          f"median independent draws/sim {np.median([r['n_draws'] for r in recs]):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampler", choices=SAMPLERS, default="emcee")
    ap.add_argument("--n_sims", type=int, default=100)
    ap.add_argument("--n_draws", type=int, default=100,
                    help="independent posterior draws per sim (the SBC L)")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--log_norm_ref", type=float, default=11.0,
                    help="centre of the FIXED log_norm prior box")
    ap.add_argument("--out", default=os.path.join(RESULTS, "sbc", "sbc_run"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # forward
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--fft32", action="store_true")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--echunk", type=int, default=0)
    ap.add_argument("--mem_gb", type=float, default=8.0)
    # samplers
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=800)
    ap.add_argument("--zeus_steps", type=int, default=0)
    ap.add_argument("--live", type=int, default=400)
    ap.add_argument("--nuts_samples", type=int, default=1000)
    ap.add_argument("--nuts_warmup", type=int, default=1000)
    ap.add_argument("--svi_steps", type=int, default=2000)
    ap.add_argument("--svi_particles", type=int, default=4)
    ap.add_argument("--svi_lr", type=float, default=1e-2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, f"sbc_{args.sampler}.jsonl")
    if args.summarise:
        return summarise(jsonl)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.fft32:
        os.environ.setdefault("SPEXAI_FFT32", "1")

    forward, prior, names = build_forward(args)
    already = done_sims(jsonl) if args.resume else set()
    if already:
        print(f"resuming: {len(already)} simulations already done", flush=True)
    elif os.path.exists(jsonl) and not args.resume:
        raise SystemExit(f"{jsonl} exists; pass --resume to continue it or "
                         f"choose another --out (refusing to mix two runs)")

    todo = [i for i in range(args.n_sims) if i not in already]
    for k, i in enumerate(todo):
        rec = run_one(i, forward, prior, names, args)
        append(jsonl, rec)
        print(f"[{k + 1}/{len(todo)}] sim {i}: {rec['runtime_s'] / 60:.1f} min, "
              f"{rec['n_draws']} draws (thin {rec['thin']}), "
              f"min ESS {rec['min_ess']:.0f}", flush=True)
    summarise(jsonl)


if __name__ == "__main__":
    main()
