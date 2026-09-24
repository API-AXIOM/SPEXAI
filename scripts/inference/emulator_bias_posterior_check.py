"""Check the linearised emulator-bias screen against full posteriors.

``bias_sweep.py`` screens a thousand parameter points with a linearised Fisher
estimate of the bias the emulator's own error imprints on the recovered
parameters. That is cheap but first-order, and the hot-floor cross-check found
real MLE bias running 1.5-3x the linear ``b_sys`` at high counts (though in the
predicted direction). This script re-evaluates a small set of those points with
a real posterior instead of one Newton step, and reports, per parameter:

  pull        (posterior median - truth) / posterior sigma, the measured bias
  screen      the same quantity the screen predicts, rescaled to the counts
              actually injected here
  k           pull / screen -- the nonlinearity factor, the same ratio the
              Gauss-Newton sweep estimated, now measured where it matters

Points come in two kinds. The **worst** ones are where the screen says the
emulator floor should bite; once the Mn design artifacts are excluded they are
the cold, narrow corner (kT < 1.5 keV, sigma_v < 150 km/s) the sweep flagged.
The **random** ones are drawn from the rest of the space as a false-negative
check: if the screen calls a point unremarkable and its posterior still pulls,
the linear approximation is missing something.

Counts: the sweep stores ``b_sys``/``sigma_ref`` at its own reference level
``n_ref`` (1e5), while the campaign quotes and fits at 1e6. ``b_sys`` scales
linearly with counts and sigma as its square root, so every screen number
printed or recorded here is rescaled by sqrt(target_counts/n_ref) -- a factor
3.162 at the default. The raw stored ratio is kept in the output as
``screen_ratio_raw`` so the provenance is not lost.

A caveat on k: the data is one Poisson realisation, so at 1e6 counts a ~2-3
sigma effect carries ~1 sigma of realisation scatter, i.e. 35-50% on a
single point's k. Compare the *set* of points against the Gauss-Newton k, not
any one of them.

Nothing here duplicates the sweep's physics -- it reuses the sweep's own point
definitions, cached noise-free truth and parameter builder, and changes only
the evaluation.

This is a PREP script at its default budget: smoke-testable on a laptop, but
production belongs on the GPU cluster.

    # laptop smoke: 1 worst + 1 random, tiny budget
    conda run -n spexai python -u \\
        scripts/inference/emulator_bias_posterior_check.py \\
        --bias_jsonl .../bias_single_n1000_s39235.jsonl \\
        --truth_npz  .../truth_single_n1000_s39235.npz \\
        --n_worst 1 --n_random 1 --exclude_points 629 155 \\
        --sampler emcee --nwalkers 32 --nsteps 60 --device cpu

    # cluster production, single-T: 4 worst + 4 random
    python -u scripts/inference/emulator_bias_posterior_check.py \\
        --bias_jsonl .../bias_single_n1000_s39235.jsonl \\
        --truth_npz  .../truth_single_n1000_s39235.npz \\
        --mode single --exclude_points 629 155 \\
        --sampler nautilus --n_eff 1000 --device cuda
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

from campaign import (
    PERSEUS,
    FREE_Z,
    find_xrism_response,
    band_mask,  # noqa: E402
    EXCLUDE_NONE,
    gaussian_logT_dem,
)
from bias_sweep import build_pars, abundance_map  # noqa: E402
from spexai.config import STORE, RESULTS  # noqa: E402
from spexai.inference.abundances import AbundanceModel, SYMBOL  # noqa: E402
from spexai.inference.absorption import Absorption  # noqa: E402
from spexai.inference.operator_model import JointOperatorModel  # noqa: E402
from spexai.inference.response import Response  # noqa: E402
from spexai.inference.priors import PriorSet  # noqa: E402
from spexai.inference.spectral_fit import SpectralFit  # noqa: E402
from spexai.inference import samplers  # noqa: E402


def screen_ratio(rec):
    """The screen's worst |b_sys|/sigma over the point's parameters, as stored.

    This is at the sweep's own reference level ``rec["n_ref"]``, NOT at the
    level this script injects at -- see ``scale_to_counts``.
    """
    return float(np.max(np.abs(rec["b_sys"]) / np.array(rec["sigma_ref"])))


def scale_to_counts(rec, target_counts):
    """Rescale a screen ratio from the sweep's N_REF to ``target_counts``.

    ``b_sys`` grows linearly with the counts while sigma grows as sqrt, so the
    ratio scales as sqrt(N_target/N_REF). The sweep stores b_sys/sigma_ref at
    N_REF = 1e5 while the campaign quotes -- and this script injects at -- 1e6,
    a factor sqrt(10) = 3.162. Comparing a posterior pull measured at 1e6
    against the raw stored ratio understates the screen by exactly that factor.
    """
    return np.sqrt(target_counts / rec["n_ref"])


def select_points(bias_jsonl, n_worst, n_random, exclude=(), seed=0):
    """The screen's worst offenders, plus a random sample of the rest.

    ``exclude`` drops points by their ``point`` id before ranking. The campaign
    uses it for the two single-T points whose worst parameter is Mn: Mn/Fe ~ 4x
    solar is not an ICM composition, so they are a design artifact of the sweep
    rather than a physical target. Once they are dropped the global ranking IS
    the cold + narrow corner, so no region cut is needed. NOTE the ids are
    per-flavour -- single-T point 155 is excluded, DEM point 155 is a target.

    The random sample is drawn from every point that is neither excluded nor
    already selected as worst, and is the false-negative check on the screen:
    if a point the screen calls unremarkable shows a pull anyway, the linear
    approximation is missing something.
    """
    recs = [json.loads(l) for l in open(bias_jsonl) if l.strip()]
    ratio = np.array([screen_ratio(r) for r in recs])
    ids = np.array([r["point"] for r in recs])

    excluded = np.isin(ids, np.asarray(list(exclude), dtype=int))
    missing = set(exclude) - set(ids[excluded].tolist())
    if missing:
        raise SystemExit(
            f"--exclude_points {sorted(missing)} not in "
            f"{bias_jsonl}; check the flavour (the ids differ "
            f"between the single-T and DEM sweeps)"
        )

    order = np.argsort(-np.where(excluded, -np.inf, ratio))
    worst_idx = list(order[:n_worst])

    pool = [i for i in range(len(recs)) if not excluded[i] and i not in set(worst_idx)]
    rng = np.random.default_rng(seed)
    rand_idx = (
        list(rng.choice(pool, size=min(n_random, len(pool)), replace=False))
        if n_random
        else []
    )

    return [(recs[i], "worst", ratio[i]) for i in worst_idx] + [
        (recs[i], "random", ratio[i]) for i in rand_idx
    ]


def build_point_problem(store, response, absorption, keep, rec, d_ref, args):
    """One screened point's cached truth -> a fit-ready (post, pars, truth,
    names), reusing bias_sweep's own truth-vector builder. ``d_ref`` is the
    point's noise-free in-band truth counts at N_REF, pulled from the sweep's
    truth_<tag>.npz (bias_sweep records b_sys/sigma_ref in the jsonl, not the
    truth spectrum itself -- that lives in the separate npz the jsonl was
    built from).
    """
    pt = rec["params"]
    emu = JointOperatorModel(models_dir=store, device=args.device)
    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z], 1.0, 26)

    # log_norm MUST be rescaled with the data. The sweep recorded
    # ``log_norm_truth`` for a spectrum normalised to ``n_ref`` counts; we inject
    # ``target_counts``, and ``norm = 10**log_norm`` (fitting.py), so the truth
    # the data actually implies is shifted by log10(target_counts/n_ref).
    #
    # Leaving it unshifted is not a cosmetic error in one number: build_pars
    # gives log_norm the box [truth-1, truth+1], so at the campaign's default
    # 10x rescale (n_ref 1e5 -> 1e6) the required value lands EXACTLY on the
    # upper bound. The 2026-09-23 probe of point 310 saturated there -- median
    # 1.01 sigma below the ceiling -- and the fit absorbed the missing
    # normalisation into the abundances, inflating Si/S/Ar/Ca/Fe to +1.2..+3.3
    # sigma and dropping coverage to 2/12. Shifting the truth also re-centres
    # the box, so it cannot bind.
    count_shift = float(np.log10(args.target_counts / rec["n_ref"]))
    log_norm_truth = rec["log_norm_truth"] + count_shift
    pars = build_pars(None, pt, log_norm_truth, args.mode)
    names = [p.name for p in pars]
    truth = np.array([p.truth for p in pars])

    # rescale the cached truth (stored at N_REF) to the injected target counts,
    # exactly mirroring stage_bias's own d_ref -> d rescale
    scale = args.target_counts / d_ref.sum()
    mu_true = d_ref * scale
    rng = np.random.default_rng(args.seed + rec["point"])
    data = rng.poisson(mu_true).astype(np.float64)

    # DEM mode: build_pars emits logT_mean/logT_sigma, so the forward MUST carry
    # the matching DEM. Until 2026-09-22 this script built a VectorForward with
    # no dem= at all while passing those names, so --mode dem could not even
    # construct (VectorForward then requires a kT column, which DEM pars lack).
    # It is impossible to reintroduce now: SpectralFit is the only assembly
    # point, and the dem travels with the parameters.
    dem = gaussian_logT_dem()[0] if args.mode == "dem" else None

    # n_h_scale=1e21: build_pars emits Par("n_h", point["n_h"], 1e-2, 0.0, 6.0),
    # i.e. n_h in units of 1e21 cm^-2. `data` is already in-band.
    fit = SpectralFit(
        emulator=emu,
        response=response,
        counts=data,
        exposure=1.0,
        priors=PriorSet.from_params(pars),
        keep=keep,
        abundances=ab,
        absorption=absorption,
        dem=dem,
        redshift=PERSEUS["z"],
        luminosity_distance=PERSEUS["dist_m"],
        n_h_scale=1e21,
        device=args.device,
        chunk=args.chunk,
        batched=True,
        compile_trunk=False,
        mem_gb=args.mem_gb,
    )
    return fit.posterior, pars, truth, names


def run_sampler(post, center, args):
    """Dispatch to the requested sampler.

    emcee is asked for a step count; nautilus is asked for an effective sample
    size (``n_eff``) and runs to it. They are not interchangeable knobs: emcee's
    cost is linear in what you ask for, while nautilus pays the full
    prior-to-posterior compression first and only then tops up to ``n_eff``, so
    lowering ``n_eff`` buys much less than proportionally.
    """
    if args.sampler == "emcee":
        return samplers.run_emcee(
            post,
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            seed=args.seed,
            center=center,
        )
    if args.sampler == "nautilus":
        return samplers.run_nautilus(
            post, n_live=args.n_live, n_eff=args.n_eff, seed=args.seed
        )
    raise SystemExit(f"unknown --sampler {args.sampler}")


def run_point(store, response, absorption, keep, rec, d_ref, tag, ratio, args):
    fac = scale_to_counts(rec, args.target_counts)
    print(
        f"\n=== point {rec['point']} ({tag}, screen {ratio:.2f} raw "
        f"@N_REF={rec['n_ref']:.3g} -> {ratio * fac:.2f} "
        f"@{args.target_counts:.3g}) ===",
        flush=True,
    )
    for k, v in rec["params"].items():
        print(f"  {k:>10} = {v:.4g}")
    post, pars, truth, names = build_point_problem(
        store, response, absorption, keep, rec, d_ref, args
    )
    res = run_sampler(post, truth, args)
    s = res.samples  # already discard-applied
    q16, q50, q84 = np.percentile(s, [16, 50, 84], axis=0)
    sigma = np.clip(0.5 * (q84 - q16), 1e-30, None)
    pull = (q50 - truth) / sigma
    covered = (q16 <= truth) & (truth <= q84)

    # The screen's own prediction, brought to the level we actually injected at:
    # b_sys scales linearly with counts, sigma as sqrt. Printing the raw stored
    # ratio next to a pull measured at target_counts understates it by `fac`.
    b_screen = np.asarray(rec["b_sys"]) * (args.target_counts / rec["n_ref"])
    sig_screen = np.asarray(rec["sigma_ref"]) * np.sqrt(
        args.target_counts / rec["n_ref"]
    )
    pull_screen = b_screen / sig_screen

    # k = measured bias / linearised bias, the same ratio P6 estimated -- but
    # here from one Poisson realisation, so each point carries ~1 sigma of
    # realisation scatter on a ~2-3 sigma effect (~35-50% per point). It is the
    # set of points, not any single one, that is comparable to P6's k.
    k = np.where(np.abs(pull_screen) > 1e-12, pull / pull_screen, np.nan)

    print(f"{'param':>10} {'pull':>8} {'covered':>8} {'screen':>8} {'k':>7}")
    for j, n in enumerate(names):
        print(
            f"{n:>10} {pull[j]:>+8.2f} {str(bool(covered[j])):>8} "
            f"{pull_screen[j]:>+8.2f} {k[j]:>7.2f}"
        )
    if args.save_samples:  # default True; --no_save_samples opts out
        # Named after --out, not just the mode: samples are saved by default
        # now, so two runs of the same point (a re-run after a fix, say) would
        # otherwise silently overwrite each other's draws.
        stem = os.path.splitext(os.path.basename(args.out))[0]
        npz = os.path.join(
            os.path.dirname(args.out),
            f"samples_{stem}_pt{rec['point']}_" f"{args.sampler}.npz",
        )
        np.savez_compressed(
            npz,
            samples=s,
            names=np.array(names),
            truth=truth,
            median=q50,
            sigma=sigma,
            q16=q16,
            q84=q84,
            pull=pull,
            pull_screen=pull_screen,
            k=k,
            target_counts=args.target_counts,
            n_ref=rec["n_ref"],
            point=rec["point"],
        )
        print(f"  samples -> {npz}  {s.shape}", flush=True)

    return dict(
        point=rec["point"],
        tag=tag,
        sampler=args.sampler,
        screen_ratio_raw=float(ratio),
        screen_ratio_scaled=float(ratio * fac),
        n_ref=float(rec["n_ref"]),
        target_counts=float(args.target_counts),
        log_norm_count_shift=float(np.log10(args.target_counts / rec["n_ref"])),
        names=names,
        truth=truth.tolist(),
        median=q50.tolist(),
        sigma=sigma.tolist(),
        q16=q16.tolist(),
        q84=q84.tolist(),
        pull=pull.tolist(),
        pull_screen=pull_screen.tolist(),
        k=k.tolist(),
        covered=covered.tolist(),
        runtime_s=res.runtime_s,
        n_eval=res.n_eval,
        ess=None if res.ess is None else np.asarray(res.ess).tolist(),
        logz=res.logz,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--bias_jsonl", required=True, help="bias_sweep's bias_<tag>.jsonl")
    ap.add_argument(
        "--truth_npz",
        required=True,
        help="bias_sweep's truth_<tag>.npz for the SAME sweep run "
        "(same --n_points/--seed/--mode) -- carries the "
        "noise-free per-point truth the jsonl's b_sys was "
        "computed against",
    )
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument(
        "--n_worst",
        type=int,
        default=4,
        help="top-ranked screen points to fit (after exclusions)",
    )
    ap.add_argument(
        "--n_random",
        type=int,
        default=4,
        help="points drawn uniformly from the rest of the space, "
        "as the screen's false-negative check",
    )
    ap.add_argument(
        "--exclude_points",
        type=int,
        nargs="*",
        default=[],
        help="point ids to drop before ranking. PER-FLAVOUR: the "
        "single-T sweep needs `--exclude_points 629 155` (Mn "
        "design artifacts); the DEM sweep needs none, and its "
        "point 155 is a genuine target",
    )
    ap.add_argument(
        "--target_counts",
        type=float,
        default=1e6,
        help="in-band counts to inject. The sweep's b_sys/sigma "
        "are stored at n_ref (1e5); the campaign quotes and "
        "fits at 1e6",
    )
    ap.add_argument("--sampler", choices=["emcee", "nautilus"], default="emcee")
    ap.add_argument(
        "--no_save_samples",
        dest="save_samples",
        action="store_false",
        help="do NOT write the posterior draws. The draws are "
        "saved by default, for every sampler: the jsonl "
        "carries summary statistics only, so discarding them "
        "throws away the run's actual product -- corner "
        "plots, re-derived intervals, convergence checks, "
        "anything not anticipated when the run was launched. "
        "A 17h nautilus run was lost this way on 2026-09-23. "
        "Use this only when disk is genuinely the constraint",
    )
    ap.set_defaults(save_samples=True)
    ap.add_argument(
        "--n_eff", type=int, default=1000, help="nautilus: target effective sample size"
    )
    ap.add_argument("--n_live", type=int, default=2000, help="nautilus")
    ap.add_argument("--nwalkers", type=int, default=64, help="emcee")
    ap.add_argument("--nsteps", type=int, default=800, help="emcee")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out", default=os.path.join(RESULTS, "bias_posterior_check", "pulls.jsonl")
    )
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response, exclude=EXCLUDE_NONE)

    selected = select_points(
        args.bias_jsonl,
        args.n_worst,
        args.n_random,
        exclude=args.exclude_points,
        seed=args.seed,
    )
    print(
        f"selected {len(selected)} points from {args.bias_jsonl} "
        f"({args.n_worst} worst + {args.n_random} random, "
        f"excluding {args.exclude_points or 'nothing'}); "
        f"sampler={args.sampler}, injecting {args.target_counts:.3g} counts",
        flush=True,
    )

    tz = np.load(args.truth_npz, allow_pickle=True)
    truth_counts = tz["counts"]  # (n_points, n_channels)

    with open(args.out, "a") as f:
        for rec, tag, ratio in selected:
            t0 = time.time()
            d_ref = truth_counts[rec["point"]][keep]
            out = run_point(
                args.store, response, absorption, keep, rec, d_ref, tag, ratio, args
            )
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
