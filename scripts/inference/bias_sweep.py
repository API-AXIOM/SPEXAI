"""Tier B: systematic-bias sweep over cluster parameter space.

Everything the package knows about emulator bias at the *science* level is
currently anchored at one point -- Perseus, with a couple of single-axis
variants. This sweeps it. At each of many parameter combinations it reports the
linearised systematic bias ``b_sys`` and the Fisher statistical error, and
therefore the crossover count ``N*`` where the emulator floor starts to matter:

    N* = N_REF * (sigma_ref / |b_sys|)^2

The output is a map of *where in cluster parameter space the emulator is safe
for science, and at what exposure it stops being safe* -- per published
parameter, in units the science cares about, rather than a goodness score.

Why the Fisher route rather than fitting: ``b_sys`` costs ``2n+1`` emulator
forwards (a Jacobian) plus one truth spectrum, no MCMC at all. That is what
makes a sweep of hundreds of points affordable when a single posterior costs
hours. It is a **screen**, not a final number: the linearisation is first
order, and the hot-floor cross-check found MCMC offsets running 1.5-3x the
linear ``b_sys`` at 1e8 counts, though in the predicted direction (10/11
parameter signs matched). Rank points by this, then validate the interesting
ones with real posteriors (Tier C).

**Combinations are the point.** Abundance enters truth and emulator linearly,
so varying it cannot create per-element error -- but it changes which elements
dominate which channels, which reweights the per-element errors and moves
``b_sys``. Per-element benchmarks cannot see that; ``F^{-1}`` can, because it
carries the parameter covariance.

Two stages, because they want different machines:

``--stage truth``  CPU + the 40 GB SPEX caches. Loops **elements outermost**:
                   each element's cache is loaded once and evaluated at every
                   sweep point, so the caches are read 30 times, not 30*N.
                   Measured 2.4 s to load an element and ~0.4 s per evaluation,
                   which makes this ~7x cheaper than the natural point-outer
                   loop and keeps peak memory at one element (~0.4 GB).
``--stage bias``   The emulator Jacobian + Fisher solve. GPU-friendly, needs no
                   caches, and consumes the npz the truth stage wrote.
                   ``--jacobian batched`` (what P7 runs) folds every stencil
                   point of ``--point_chunk`` sweep points into ONE emulator
                   call via the batched forward P6 used; ``serial`` is the
                   original call-per-stencil-point path, kept as the default
                   so earlier commands reproduce. Check one against the other
                   with ``check_jacobian_parity.py``, not by assumption.

Both stages checkpoint per unit of work and skip what is already on disk.

    # cluster (matches the package defaults, so the exports are optional there)
    SPEXAI_PROCESSED=~/data/spexai_data/processed \\
    SPEXAI_RESPONSES=~/data/spexai_data/responses \\
        python -u scripts/bias_sweep.py --stage truth --n_points 200 --mode both
    # laptop, where the data lives elsewhere and the exports ARE required
    SPEXAI_PROCESSED=~/work/data/spexai/processed \\
    SPEXAI_RESPONSES=~/work/data/spexai/responses \\
        python -u scripts/bias_sweep.py --stage truth --n_points 200 --mode both
    python -u scripts/bias_sweep.py --stage bias --n_points 200 --mode both
    python scripts/bias_sweep.py --summarise
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

from campaign import (                                            # noqa: E402
    PERSEUS, FREE_Z, HOT_SCIENCE, HOT_WEAK, find_xrism_response,
    band_mask, EXCLUDE_NONE, gaussian_logT_dem, check_truth_dem_param,
    DEM_PARAM, N_REF, Par, Forward, restrict_to_band)
from fisher_bias import linear_bias_fisher, COND_F_WARN           # noqa: E402
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import SYMBOL                    # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.spex_truth import SpexTruthModel            # noqa: E402

# --- the cluster science range (revised 2026-09-15) -------------------------
# Temperature: 0.7-15 keV, groups through the most massive clusters, for BOTH
# modes. 0.7 keV is the PCHIP truth's safe floor (tempdist.PCHIP_TRUTH_SAFE_LO_KEV;
# below it Ar -> inf at 0.5 keV), 15 keV stays inside the emulator's trained
# 19.94 keV. The 2026-08-19 range (1.5-8 keV) excluded groups and exactly the
# hot clusters where the hot line-rich metals matter.
#
# The DEM is a Gaussian in log10 T (SPEX gdem). Width 0.056-0.4 dex: the floor
# keeps sigma resolved on the 70-node grid, the ceiling is the wide end of
# SPEX-typical widths. NO width cap tied to the centre -- a DEM running off the
# grid is truncated identically in truth and emulator, so b_sys stays a fair
# comparison; the truncation is recorded per point (``contained_fraction``)
# and read at interpretation time, not cut at design time.
RANGES = {
    "kT": (0.7, 15.0),             # keV, single-T; drawn UNIFORM IN log10 kT
    "logT_mean": (float(np.log10(0.7)), float(np.log10(15.0))),  # log10 keV
    "logT_sigma": (0.056, 0.4),    # dex, DEM width
    "abundance": (0.2, 2.0),       # x solar, each free element independently
    "sigma_v": (30.0, 600.0),      # km/s
    "n_h": (0.0, 5.0),             # 1e21 cm^-2
}
# axes whose RANGES are in linear units but are drawn uniform in log10, so a
# temperature axis samples the same way in both modes
LOG_AXES = {"kT"}
NORM_REF = 1e11


def sample_points(n, mode, seed):
    """Latin hypercube over the science range. -> list of dicts.

    LHS rather than a grid: with 12+ axes a grid is impossible, and LHS gives
    every axis full marginal coverage at any sample size, which is what makes
    a partial run still interpretable. Axes in ``LOG_AXES`` are stratified in
    log10 and returned in linear units.
    """
    from scipy.stats import qmc
    axes = (["logT_mean", "logT_sigma"] if mode == "dem" else ["kT"])
    axes = axes + [f"a_{SYMBOL[z]}" for z in FREE_Z] + ["sigma_v", "n_h"]
    sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
    u = sampler.random(n)                                   # (n, d)
    lo, hi = [], []
    for a in axes:
        key = ("abundance" if a.startswith("a_") else a)
        a_lo, a_hi = RANGES[key]
        if a in LOG_AXES:
            a_lo, a_hi = np.log10(a_lo), np.log10(a_hi)
        lo.append(a_lo)
        hi.append(a_hi)
    x = qmc.scale(u, lo, hi)                                # (n, d)
    for j, a in enumerate(axes):
        if a in LOG_AXES:
            x[:, j] = 10.0 ** x[:, j]
    return [{k: float(v) for k, v in zip(axes, row)} for row in x]


def contained_fraction(point, mode):
    """Fraction of the DEM's emission measure on the fixed grid; 1 for single-T.

    The parametric weights are not renormalised, so their sum IS this fraction.
    Recorded per point rather than used as a design cut: a truncated DEM is
    truncated identically in truth and emulator, so it is a property of the
    point, not an invalid one."""
    if mode != "dem":
        return 1.0
    dem, p = gaussian_logT_dem(mean=point["logT_mean"],
                               sigma=point["logT_sigma"])
    return float(dem.weights(p).sum())


def abundance_map(point, elements):
    """Per-element abundance for one sweep point.

    Mirrors the literature strategy the fits use: ``FREE_Z`` free, every other
    metal tied to Fe, H/He at solar (they carry the continuum).
    """
    a_fe = point[f"a_{SYMBOL[26]}"]
    out = {}
    for z in elements:
        z = int(z)
        if z in (1, 2):
            out[z] = 1.0
        elif z in FREE_Z:
            out[z] = point[f"a_{SYMBOL[z]}"]
        else:
            out[z] = a_fe
    return out


# --- stage 1: truth ---------------------------------------------------------

def stage_truth(args, points, outp):
    """Noise-free SPEX truth at every sweep point, elements outermost.

    Accumulates into one ``(n_points, n_channels)`` buffer. The inversion
    versus ``experiment.stream_truth_counts`` is the whole point: that function
    rebuilds a ``SpexTruthModel`` per element per call, which at N points would
    re-read every cache N times.
    """
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    logz = float(np.log10(PERSEUS["z"]))
    emu_elements = JointOperatorModel(models_dir=args.store,
                                      device="cpu").elements
    n_chan = response.n_channels if hasattr(response, "n_channels") else None

    dem_cache = {}
    contained = np.array([contained_fraction(pt, args.mode) for pt in points])
    total = None
    done = set()
    if os.path.exists(outp) and args.resume:
        z = np.load(outp, allow_pickle=True)
        total = z["counts"]
        done = set(int(v) for v in z["elements_done"])
        print(f"resuming truth: {len(done)} elements already summed",
              flush=True)

    for z_el in emu_elements:
        z_el = int(z_el)
        if z_el in done:
            continue
        t0 = time.time()
        # datadir explicitly: without it SpexTruthModel derives the cache path
        # from the manifest's TRAINING runroot, which is an absolute path on
        # whichever machine trained the model. None keeps the old behaviour
        # (SPEXAI_PROCESSED, then the manifest).
        m = SpexTruthModel(models_dir=args.store, elements=[z_el],
                           datadir=args.datadir or None, device="cpu")
        load_s = time.time() - t0
        t0 = time.time()
        for i, pt in enumerate(points):
            a = abundance_map(pt, emu_elements)[z_el]
            common = dict(luminosity_distance=PERSEUS["dist_m"],
                          absorption=absorption, n_h=pt["n_h"] * 1e21)
            if args.mode == "dem":
                key = (round(pt["logT_mean"], 6), round(pt["logT_sigma"], 6))
                if key not in dem_cache:
                    dem_cache[key] = gaussian_logT_dem(mean=key[0],
                                                       sigma=key[1])
                dem, dp = dem_cache[key]
                c = m.predict_counts_dem(
                    dem.temp_grid, dem.weights(dp), {z_el: a}, logz, NORM_REF,
                    pt["sigma_v"], response, 1.0, **common)
            else:
                c = m.predict_counts(
                    torch.tensor([pt["kT"]]), {z_el: a}, logz, NORM_REF,
                    pt["sigma_v"], response, 1.0, **common)
            c = c.squeeze(0).cpu().numpy()
            if total is None:
                total = np.zeros((len(points), c.size))
            total[i] += c
        done.add(z_el)
        # record the response: an ARF changes neither the channel count nor
        # the element set, but rescales the truth channel by channel, so a
        # truth built against a different response is silently wrong and no
        # other field in this file can reveal it. ``dem_param`` does the same
        # job for the DEM parametrisation (see campaign.check_truth_dem_param).
        np.savez(outp, counts=total, elements_done=sorted(done),
                 points=json.dumps(points), mode=args.mode,
                 rmf=os.path.basename(rmf), arf=os.path.basename(arf),
                 dem_param=DEM_PARAM if args.mode == "dem" else "none",
                 contained=contained)
        print(f"Z={z_el:>2}: load {load_s:.1f}s, {len(points)} points in "
              f"{time.time() - t0:.1f}s ({len(done)}/{len(emu_elements)} done)",
              flush=True)

    if not np.isfinite(total).all():
        bad = int((~np.isfinite(total)).any(axis=1).sum())
        raise SystemExit(
            f"{bad} sweep points have non-finite truth counts -- almost "
            f"certainly a DEM grid point below the per-element training "
            f"minimum (~0.501 keV), where the PCHIP truth extrapolates and "
            f"blows up. The campaign grid starts at the PCHIP-safe 0.7 keV, so "
            f"check campaign.gaussian_logT_dem's `lo` and RANGES['kT'].")
    print(f"truth complete -> {outp}", flush=True)


# --- stage 2: bias ----------------------------------------------------------

# fit-box padding beyond the temperature design range, dex
LOGT_PAD_DEX = 0.1
# finite-difference step for every temperature axis, in DEX. Relative, not
# absolute: the central difference's truncation error grows as h^2 times the
# local curvature, and above 1.9 keV a cold plasma's in-band counts fall away
# exponentially with kT, so one absolute step cannot serve 0.7 and 15 keV at
# once. The inherited 5e-3 keV was 0.036% of kT at 14 keV but 0.67% at 0.75
# keV, where it biased the Jacobian by ~8% (measured: halving the step moved
# the Fisher-weighted column by 6.0e-2, against 3.1e-3 at 14 keV).
# 5e-4 dex sits in the valley between truncation and float32 round-off at
# every temperature tested (7.7e-5 to 8.3e-4 for the DEM parameters).
STEP_DEX = 5e-4


def build_pars(fwd, point, log_norm_truth, mode):
    """Truth vector for this sweep point, in ``fwd.names`` order.

    The Fisher solve never leaves the truth point, but the P6 fits (L-BFGS,
    Gauss-Newton) clip to these bounds, so they are chosen to NEVER BIND at a
    truth point rather than from physics: the temperature design range padded
    by ``LOGT_PAD_DEX`` (kT 0.556-18.9 keV, still inside the emulator's
    0.5013-19.94 keV), and a width box of 0.045-0.5 dex whose floor sits just
    below the 0.056 dex resolution limit -- much lower and the width's Fisher
    information comes from grid interpolation, not the spectrum.

    Steps: ``STEP_DEX`` for every temperature axis. The DEM parameters are
    already in dex, so they take it directly; ``kT`` is in keV, so it takes the
    same step CONVERTED AT THE POINT, ``kT ln10 STEP_DEX`` -- 8.6e-4 keV at
    0.75 keV, 0.016 keV at 14 keV. See ``STEP_DEX`` for the measurement that
    retired the old absolute 5e-3 keV.
    """
    out = []
    for z in FREE_Z:
        out.append(Par(SYMBOL[z], point[f"a_{SYMBOL[z]}"], 1e-3, 0.02, 3.0))
    if mode == "single":
        k_lo, k_hi = np.log10(RANGES["kT"])
        out.append(Par("kT", point["kT"],
                       float(point["kT"] * np.log(10.0) * STEP_DEX),
                       float(10.0 ** (k_lo - LOGT_PAD_DEX)),
                       float(10.0 ** (k_hi + LOGT_PAD_DEX))))
    else:
        m_lo, m_hi = RANGES["logT_mean"]
        out.append(Par("logT_mean", point["logT_mean"], STEP_DEX,
                       m_lo - LOGT_PAD_DEX, m_hi + LOGT_PAD_DEX))
        out.append(Par("logT_sigma", point["logT_sigma"], STEP_DEX,
                       0.045, 0.5))
    out.append(Par("sigma_v", point["sigma_v"], 1.0, 10.0, 700.0))
    out.append(Par("n_h", point["n_h"], 1e-2, 0.0, 6.0))
    out.append(Par("log_norm", log_norm_truth, 2e-3,
                   log_norm_truth - 1.0, log_norm_truth + 1.0))
    return out


def _point_inputs(args, pt, counts_row, keep):
    """Per-point pieces shared by both Jacobian paths.

    Returns ``(d, log_norm_truth, pars)`` or ``None`` when the point has no
    in-band truth counts. ``d`` is the SPEX truth rescaled to ``N_REF``
    in-band counts; ``log_norm_truth`` carries that rescaling, so the emulator
    is evaluated at the same normalisation the truth was rescaled to.
    """
    d_ref = counts_row[keep]
    if d_ref.sum() <= 0:
        return None
    s = N_REF / d_ref.sum()
    d = d_ref * s
    log_norm_truth = float(np.log10(NORM_REF * s))
    pars = build_pars(None, pt, log_norm_truth, args.mode)
    return d, log_norm_truth, pars


def _bias_record(i, pt, pars, b_sys, sigma_ref, cond_F, contained_i,
                 log_norm_truth, rmf, arf, runtime_s):
    # contained: fraction of the DEM on the grid (1 for single-T), kept
    # next to b_sys so truncated points can be read as such, not excluded
    return {"point": i, "params": pt, "names": [p.name for p in pars],
            "truth": [p.truth for p in pars], "b_sys": b_sys.tolist(),
            "sigma_ref": sigma_ref.tolist(), "n_ref": N_REF,
            "contained": float(contained_i),
            "log_norm_truth": log_norm_truth, "cond_F": cond_F,
            "rmf": os.path.basename(rmf), "arf": os.path.basename(arf),
            "runtime_s": runtime_s}


def _write_bias(outp, rec, n_points):
    with open(outp, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    b_sys = np.array(rec["b_sys"])
    sigma_ref = np.array(rec["sigma_ref"])
    worst = int(np.argmax(np.abs(b_sys) / sigma_ref))
    # cond(F) is printed per point because this loop runs unattended over
    # hundreds of points: a near-singular F yields a finite, plausible N*
    # from two meaningless numbers, and nothing else in the output would
    # reveal it. See linear_bias_fisher's docstring for the n_h case.
    warn = "  !! NEAR-SINGULAR F" if rec["cond_F"] > COND_F_WARN else ""
    print(f"[{rec['point'] + 1}/{n_points}] {rec['runtime_s']:.1f}s  worst "
          f"{rec['names'][worst]} b/sig@Nref="
          f"{b_sys[worst] / sigma_ref[worst]:+.3f}  "
          f"cond(F)={rec['cond_F']:.1e}{warn}", flush=True)


def _serial_forward(args, response, keep):
    """The original point-at-a-time forward: ``campaign.Forward``."""
    emu = JointOperatorModel(models_dir=args.store, device=args.device)
    restrict_to_band(emu)
    dem = gaussian_logT_dem()[0] if args.mode == "dem" else None
    return Forward(emu, response, Absorption.default(), keep, args.mode,
                   dem=dem)


def stage_bias(args, points, truth_path, outp):
    tz = np.load(truth_path, allow_pickle=True)
    check_truth_dem_param(tz)
    counts = tz["counts"]
    if len(counts) != len(points):
        raise SystemExit(f"truth has {len(counts)} points but the sweep asks "
                         f"for {len(points)} -- regenerate with the same "
                         f"--n_points/--seed/--mode")
    contained = (tz["contained"] if "contained" in tz.files else
                 np.array([contained_fraction(pt, args.mode) for pt in points]))

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    keep = band_mask(response, exclude=EXCLUDE_NONE)

    done = {}
    if os.path.exists(outp) and args.resume:
        with open(outp) as f:
            done = {int(json.loads(l)["point"]): 1 for l in f if l.strip()}
        print(f"resuming bias: {len(done)} points already done", flush=True)
    todo = [i for i in range(len(points)) if i not in done]

    if args.jacobian == "serial":
        _stage_bias_serial(args, points, counts, contained, keep, response,
                           rmf, arf, todo, outp)
    else:
        _stage_bias_batched(args, points, counts, contained, keep, response,
                            rmf, arf, todo, outp)


def _stage_bias_serial(args, points, counts, contained, keep, response,
                       rmf, arf, todo, outp):
    fwd = _serial_forward(args, response, keep)
    for i in todo:
        pt = points[i]
        got = _point_inputs(args, pt, counts[i], keep)
        if got is None:
            print(f"point {i}: zero in-band truth, skipped", flush=True)
            continue
        d, log_norm_truth, pars = got
        # the DEM model has to carry THIS point's shape, not the fiducial's
        if args.mode == "dem":
            fwd.dem = gaussian_logT_dem(mean=pt["logT_mean"],
                                        sigma=pt["logT_sigma"])[0]
        t0 = time.time()
        b_sys, sigma_ref, cond_F = linear_bias_fisher(fwd, pars, d,
                                                      verbose=False)
        rec = _bias_record(i, pt, pars, b_sys, sigma_ref, cond_F, contained[i],
                           log_norm_truth, rmf, arf, time.time() - t0)
        _write_bias(outp, rec, len(points))


def _stage_bias_batched(args, points, counts, contained, keep, response,
                        rmf, arf, todo, outp):
    """Same Fisher solve, but every point's 2n+1 stencil in ONE emulator call.

    The serial path pays the per-forward overhead 2n+1 times per point on a
    forward built for one row at a time (262 s/point measured on laptop CPU,
    which is 145 CPU-h for P7's 1000 points x 2 flavours). This path reuses the
    primitive P6 already ran on the GPU: ``mle_reseed.tierb_forward`` (a
    batched ``VectorForward``) driven by ``mle_reseed.batched_jacobian``, which
    folds ``point_chunk * (2n+1)`` stencil rows into a single call. The algebra
    afterwards is ``fisher_bias.fisher_from_jacobian`` -- the same six lines
    the serial path runs, not a reimplementation.

    Two differences from the serial path, both deliberate:

    * **The DEM shape is not pinned per point.** ``VectorForward`` carries the
      log-T Gaussian as a shape FAMILY with ``logT_mean``/``logT_sigma`` as
      ordinary vector components, so the per-point ``fwd.dem`` rebuild the
      serial path needs disappears and those Jacobian columns come for free.
    * **The forward runs in float32** (``VectorForward.__call__`` casts), where
      the serial path is float64 on CPU. The central differences are ~1e-4-1e-3
      of ``mu``, so ~1e-7 relative forward noise lands as ~1e-3 relative noise
      in J. Check it, do not assume it: ``check_jacobian_parity.py`` compares
      the two paths' ``b_sys``/``sigma_ref``/``cond(F)`` point by point.
    """
    from mle_reseed import tierb_forward, batched_jacobian
    from fisher_bias import fisher_from_jacobian

    # names are a property of the mode, not the point, so the forward is built
    # ONCE for the whole sweep -- see tierb_forward's docstring
    probe = build_pars(None, points[todo[0] if todo else 0], 0.0, args.mode)
    names = [p.name for p in probe]
    fwd = tierb_forward(args, names, response, keep)

    for lo in range(0, len(todo), args.point_chunk):
        idx = todo[lo:lo + args.point_chunk]
        rows = []
        for i in idx:
            got = _point_inputs(args, points[i], counts[i], keep)
            if got is None:
                print(f"point {i}: zero in-band truth, skipped", flush=True)
                continue
            rows.append((i,) + got)
        if not rows:
            continue
        theta = np.array([[p.truth for p in r[3]] for r in rows])   # (K, ndim)
        steps = np.array([[p.step for p in r[3]] for r in rows])    # (K, ndim)
        t0 = time.time()
        mu0, J = batched_jacobian(fwd, None, theta, steps=steps,
                                  verbose=False)
        per_point = (time.time() - t0) / len(rows)
        for k, (i, d, log_norm_truth, pars) in enumerate(rows):
            b_sys, sigma_ref, cond_F = fisher_from_jacobian(mu0[k], J[k], d,
                                                            verbose=False)
            rec = _bias_record(i, points[i], pars, b_sys, sigma_ref, cond_F,
                               contained[i], log_norm_truth, rmf, arf,
                               per_point)
            _write_bias(outp, rec, len(points))


# --- reporting --------------------------------------------------------------

def group_of(name):
    z = {v: k for k, v in SYMBOL.items()}.get(name)
    if z in HOT_SCIENCE:
        return "science"
    if z in HOT_WEAK:
        return "weak"
    return "Fe" if name == "Fe" else "other"


def summarise(outp, target_counts):
    recs = [json.loads(l) for l in open(outp) if l.strip()]
    if not recs:
        raise SystemExit(f"no records in {outp}")
    names = recs[0]["names"]
    b = np.array([r["b_sys"] for r in recs])                  # (P, n)
    sig = np.array([r["sigma_ref"] for r in recs])            # (P, n)
    n_ref = recs[0]["n_ref"]
    # sigma scales as 1/sqrt(N); b_sys is count-independent
    ratio = np.abs(b) / (sig * np.sqrt(n_ref / target_counts))
    nstar = np.where(b != 0, n_ref * (sig / np.abs(b)) ** 2, np.inf)

    print(f"{len(recs)} sweep points; bias/sigma evaluated at "
          f"{target_counts:.1e} in-band counts\n")
    print(f"{'param':>10} {'group':>8} {'median':>9} {'p90':>9} {'max':>9} "
          f"{'frac>1':>8} {'median N*':>11}")
    for j, nm in enumerate(names):
        print(f"{nm:>10} {group_of(nm):>8} {np.median(ratio[:, j]):>9.3f} "
              f"{np.percentile(ratio[:, j], 90):>9.3f} "
              f"{ratio[:, j].max():>9.3f} "
              f"{np.mean(ratio[:, j] > 1):>8.2f} "
              f"{np.median(nstar[:, j]):>11.2e}")

    worst_pt = int(np.argmax(ratio.max(axis=1)))
    print(f"\nworst point ({ratio[worst_pt].max():.2f} sigma on "
          f"{names[int(np.argmax(ratio[worst_pt]))]}):")
    for k, v in recs[worst_pt]["params"].items():
        print(f"    {k:>10} = {v:.4g}")
    print("\nThis is a linearised screen. Points above ~1 sigma are candidates "
          "for a Tier C MCMC check, where the offset may run 1.5-3x larger.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["truth", "bias"], default="truth")
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--datadir", default="",
                    help="preprocessed per-element caches (processed/elementZ) "
                         "for --stage truth. Default: SPEXAI_PROCESSED, else "
                         "the sibling of the manifest's TRAINING runroot -- "
                         "which is an absolute path on the training machine "
                         "and will not exist elsewhere. Set this (or "
                         "SPEXAI_PROCESSED) when store and caches come from "
                         "different machines")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(RESULTS, "bias_sweep"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--target_counts", type=float, default=1e6,
                    help="in-band counts at which to report bias/sigma")
    # --- Jacobian path (--stage bias) ---------------------------------------
    ap.add_argument("--jacobian", choices=["serial", "batched"],
                    default="serial",
                    help="serial = campaign.Forward, one stencil point per "
                         "emulator call (262 s/point on laptop CPU); it is "
                         "the default only so pre-2026-09 runs reproduce "
                         "command-for-command. batched = one call per "
                         "--point_chunk points via mle_reseed.tierb_forward, "
                         "which is what P7 runs")
    ap.add_argument("--point_chunk", type=int, default=4,
                    help="sweep points per batched emulator call. Each costs "
                         "(2n+1) rows (25 single-T, 27 DEM), and a DEM row "
                         "costs the temperature grid on top, so raise this on "
                         "a GPU and leave it small on a laptop")
    # forward knobs, spelled as in mle_reseed/p6_sweep so one command carries
    # across the three scripts; ignored by --jacobian serial
    ap.add_argument("--chunk", type=int, default=32,
                    help="emulator rows per sub-batch (batched only)")
    ap.add_argument("--gchunk", type=int, default=None,
                    help="DEM only: temperature-grid points per emulator call")
    ap.add_argument("--echunk", type=int, default=None)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_table", action="store_true",
                    help="DEM only: disable the fixed-grid trunk/line table")
    ap.add_argument("--no_contract", action="store_true",
                    help="DEM only: broaden per element instead of "
                         "contracting first")
    ap.add_argument("--deterministic", action="store_true",
                    help="force deterministic CUDA kernels. The line "
                         "deposit's index_add_ has repeated indices, so "
                         "identical parameters otherwise return "
                         "different counts -- and "
                         "a central difference is a DIFFERENCE of two such "
                         "calls, which is where that jitter lands")
    args = ap.parse_args()

    if args.stage == "bias" and args.jacobian == "batched":
        # must precede the first CUDA allocation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")
        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            print("deterministic algorithms ON", flush=True)
        elif args.device != "cpu":
            print("WARNING: batched Jacobian on CUDA without --deterministic; "
                  "the line deposit's index_add_ sums in a varying order, so "
                  "the two arms of each central difference carry independent "
                  "jitter", flush=True)
    if (args.stage == "bias" and args.jacobian == "serial"
            and args.n_points > 50):
        print(f"WARNING: --jacobian serial at {args.n_points} points is "
              f"~{args.n_points * 262 / 3600:.0f} CPU-h at the measured "
              f"262 s/point. P7 runs --jacobian batched.", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.mode}_n{args.n_points}_s{args.seed}"
    truth_path = os.path.join(args.out, f"truth_{tag}.npz")
    bias_path = os.path.join(args.out, f"bias_{tag}.jsonl")

    if args.summarise:
        return summarise(bias_path, args.target_counts)

    points = sample_points(args.n_points, args.mode, args.seed)
    if args.stage == "truth":
        stage_truth(args, points, truth_path)
    else:
        if not os.path.exists(truth_path):
            raise SystemExit(f"{truth_path} missing -- run --stage truth first")
        stage_bias(args, points, truth_path, bias_path)
        summarise(bias_path, args.target_counts)


if __name__ == "__main__":
    main()
