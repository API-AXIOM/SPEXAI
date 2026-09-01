"""Tier C: MCMC pull/coverage check at Tier B's worst offenders (+ controls).

Tier B (``bias_sweep.py``) screens hundreds of points with a linearised
Fisher estimate -- cheap, but first-order, and the hot-floor cross-check
found real MLE bias running 1.5-3x the linear b_sys at high counts (though in
the predicted direction). This validates the screen with real posteriors at a
much smaller set of points: the worst Tier B offenders (where the screen says
the emulator floor should bite) PLUS a few points Tier B calls safe (a
false-negative check -- does the screen miss anything?).

Reuses the exact literature-strategy parametrisation the sampler bake-off
validated (``AbundanceModel`` free/tied scheme, ``VectorForward``,
``PoissonPosterior``) and the exact point definitions + cached noise-free
truth Tier B already computed, so nothing here duplicates Tier B's physics --
only the *evaluation* (a real posterior instead of one Newton step) is new.

This is a PREP script: it selects points and runs the fit end-to-end, but at
the small default budget it is meant to be smoke-tested on a laptop, not run
at production scale. Production (higher --nwalkers/--nsteps, --device cuda)
belongs on the GPU cluster, at ~1.5-2h/point per the bake-off's emcee timing.

    # laptop smoke: 1 worst + 1 safe point, tiny budget
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python -u \\
        scripts/inference/tier_c_mcmc.py --n_worst 1 --n_safe 1 \\
        --nwalkers 32 --nsteps 60 --device cpu

    # cluster production, per selected point ~1.5-2h (emcee, bake-off timing)
    python -u scripts/inference/tier_c_mcmc.py --n_worst 10 --n_safe 5 \\
        --nwalkers 64 --nsteps 800 --device cuda
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

from campaign import PERSEUS, FREE_Z, find_xrism_response, band_mask, EXCLUDE_NONE  # noqa: E402
from bias_sweep import build_pars, abundance_map                  # noqa: E402
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import AbundanceModel, SYMBOL    # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.posterior import BoxPrior, PoissonPosterior # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.vector_forward import VectorForward         # noqa: E402
from spexai.inference import samplers                             # noqa: E402


def select_points(bias_jsonl, n_worst, n_safe):
    """Tier B's ranked points: worst |b_sys/sigma| offenders + a safe sample.

    The safe sample is a false-negative check on the screen itself, not a
    convenience -- if a "safe" point's real posterior also shows a pull,
    Tier B's linear approximation is missing something at that point.
    """
    recs = [json.loads(l) for l in open(bias_jsonl) if l.strip()]
    ratio = np.array([np.max(np.abs(r["b_sys"]) / np.array(r["sigma_ref"]))
                      for r in recs])
    order = np.argsort(-ratio)
    worst = [recs[i] for i in order[:n_worst]]
    safe = [recs[i] for i in order[-n_safe:]] if n_safe else []
    return ([(r, "worst", ratio[j]) for j, r in
             zip(order[:n_worst], worst)]
            + [(r, "safe", ratio[j]) for j, r in
               zip(order[-n_safe:] if n_safe else [], safe)])


def build_point_problem(store, response, absorption, keep, rec, d_ref, args):
    """One Tier B point's cached truth -> a fit-ready (post, pars, truth,
    names), reusing bias_sweep's own truth-vector builder. ``d_ref`` is the
    point's noise-free in-band truth counts at N_REF, pulled from Tier B's
    truth_<tag>.npz (bias_sweep records b_sys/sigma_ref in the jsonl, not the
    truth spectrum itself -- that lives in the separate npz the jsonl was
    built from).
    """
    pt = rec["params"]
    emu = JointOperatorModel(models_dir=store, device=args.device)
    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z],
                1.0, 26)

    log_norm_truth = rec["log_norm_truth"]
    pars = build_pars(None, pt, log_norm_truth, args.mode)
    names = [p.name for p in pars]
    truth = np.array([p.truth for p in pars])

    forward = VectorForward(
        emu, response, keep, names, ab, absorption=absorption,
        redshift=PERSEUS["z"], luminosity_distance=PERSEUS["dist_m"],
        velocity=None, device=args.device, chunk=args.chunk, batched=True,
        compile_trunk=False, mem_gb=args.mem_gb)

    # rescale the cached truth (stored at N_REF) to the injected target counts,
    # exactly mirroring stage_bias's own d_ref -> d rescale
    scale = args.target_counts / d_ref.sum()
    mu_true = d_ref * scale
    rng = np.random.default_rng(args.seed + rec["point"])
    data = rng.poisson(mu_true).astype(np.float64)

    prior = BoxPrior.from_params(pars, device=args.device)
    post = PoissonPosterior(forward, data, prior)
    return post, pars, truth, names


def run_point(store, response, absorption, keep, rec, d_ref, tag, ratio, args):
    print(f"\n=== point {rec['point']} ({tag}, Tier B ratio {ratio:.2f}) ===",
          flush=True)
    for k, v in rec["params"].items():
        print(f"  {k:>10} = {v:.4g}")
    post, pars, truth, names = build_point_problem(
        store, response, absorption, keep, rec, d_ref, args)
    center = truth
    res = samplers.run_emcee(post, nwalkers=args.nwalkers, nsteps=args.nsteps,
                             seed=args.seed, center=center)
    s = res.samples                                    # already discard-applied
    q16, q50, q84 = np.percentile(s, [16, 50, 84], axis=0)
    sigma = np.clip(0.5 * (q84 - q16), 1e-30, None)
    pull = (q50 - truth) / sigma
    covered = (q16 <= truth) & (truth <= q84)
    print(f"{'param':>10} {'pull':>8} {'covered':>8}  (Tier B b/sigma)")
    for j, n in enumerate(names):
        b_over_sig = np.abs(rec["b_sys"][j]) / rec["sigma_ref"][j]
        print(f"{n:>10} {pull[j]:>+8.2f} {str(bool(covered[j])):>8}  "
              f"{b_over_sig:.2f}")
    return dict(point=rec["point"], tag=tag, tier_b_ratio=float(ratio),
               names=names, truth=truth.tolist(), median=q50.tolist(),
               sigma=sigma.tolist(), pull=pull.tolist(),
               covered=covered.tolist(), runtime_s=res.runtime_s,
               n_eval=res.n_eval)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--bias_jsonl", required=True,
                    help="Tier B's bias_<tag>.jsonl")
    ap.add_argument("--truth_npz", required=True,
                    help="Tier B's truth_<tag>.npz for the SAME sweep run "
                         "(same --n_points/--seed/--mode) -- carries the "
                         "noise-free per-point truth the jsonl's b_sys was "
                         "computed against")
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--n_worst", type=int, default=10)
    ap.add_argument("--n_safe", type=int, default=5)
    ap.add_argument("--target_counts", type=float, default=1e6)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(RESULTS, "tier_c",
                                                   "pulls.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response, exclude=EXCLUDE_NONE)

    selected = select_points(args.bias_jsonl, args.n_worst, args.n_safe)
    print(f"selected {len(selected)} points from {args.bias_jsonl} "
          f"({args.n_worst} worst + {args.n_safe} safe)", flush=True)

    tz = np.load(args.truth_npz, allow_pickle=True)
    truth_counts = tz["counts"]                    # (n_points, n_channels)

    with open(args.out, "a") as f:
        for rec, tag, ratio in selected:
            t0 = time.time()
            d_ref = truth_counts[rec["point"]][keep]
            out = run_point(args.store, response, absorption, keep, rec,
                            d_ref, tag, ratio, args)
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
