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

Both stages checkpoint per unit of work and skip what is already on disk.

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
    band_mask, gaussian_dem, N_REF, Par, Forward)
from fisher_bias import linear_bias_fisher                        # noqa: E402
from spexai.config import STORE, RESULTS                          # noqa: E402
from spexai.inference.abundances import SYMBOL                    # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.response import Response                    # noqa: E402
from spexai.inference.spex_truth import SpexTruthModel            # noqa: E402

# --- the cluster science range (agreed 2026-08-19) --------------------------
# Deliberately the range real cluster fits live in, not the full training box:
# the question is "is the emulator safe for the clusters we fit", and filling
# the map with cold/hot extremes no cluster shows would swamp that answer.
RANGES = {
    "kT": (1.5, 8.0),              # keV, single-T
    "T_mean": (1.5, 8.0),          # keV, DEM centroid
    "T_sigma": (0.15, 3.0),        # keV, DEM width
    "abundance": (0.2, 2.0),       # x solar, each free element independently
    "sigma_v": (30.0, 600.0),      # km/s
    "n_h": (0.0, 5.0),             # 1e21 cm^-2
}
NORM_REF = 1e11


def sample_points(n, mode, seed):
    """Latin hypercube over the science range. -> list of dicts.

    LHS rather than a grid: with 12+ axes a grid is impossible, and LHS gives
    every axis full marginal coverage at any sample size, which is what makes
    a partial run still interpretable.
    """
    from scipy.stats import qmc
    axes = (["T_mean", "T_sigma"] if mode == "dem" else ["kT"])
    axes = axes + [f"a_{SYMBOL[z]}" for z in FREE_Z] + ["sigma_v", "n_h"]
    sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
    u = sampler.random(n)                                   # (n, d)
    lo, hi = [], []
    for a in axes:
        key = ("abundance" if a.startswith("a_") else a)
        lo.append(RANGES[key][0])
        hi.append(RANGES[key][1])
    x = qmc.scale(u, lo, hi)                                # (n, d)
    return [dict(zip(axes, row)) for row in x]


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
        m = SpexTruthModel(models_dir=args.store, elements=[z_el], device="cpu")
        load_s = time.time() - t0
        t0 = time.time()
        for i, pt in enumerate(points):
            a = abundance_map(pt, emu_elements)[z_el]
            common = dict(luminosity_distance=PERSEUS["dist_m"],
                          absorption=absorption, n_h=pt["n_h"] * 1e21)
            if args.mode == "dem":
                key = (round(pt["T_mean"], 6), round(pt["T_sigma"], 6))
                if key not in dem_cache:
                    dem_cache[key] = gaussian_dem(mean=key[0], sigma=key[1])
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
        np.savez(outp, counts=total, elements_done=sorted(done),
                 points=json.dumps(points), mode=args.mode)
        print(f"Z={z_el:>2}: load {load_s:.1f}s, {len(points)} points in "
              f"{time.time() - t0:.1f}s ({len(done)}/{len(emu_elements)} done)",
              flush=True)

    if not np.isfinite(total).all():
        bad = int((~np.isfinite(total)).any(axis=1).sum())
        raise SystemExit(
            f"{bad} sweep points have non-finite truth counts -- almost "
            f"certainly a DEM grid point below the per-element training "
            f"minimum (~0.501 keV), where the PCHIP truth extrapolates and "
            f"blows up. Narrow RANGES['T_sigma'] or raise the DEM grid floor.")
    print(f"truth complete -> {outp}", flush=True)


# --- stage 2: bias ----------------------------------------------------------

def build_pars(fwd, point, log_norm_truth, mode):
    """Truth vector for this sweep point, in ``fwd.names`` order.

    Bounds are only used for finite differences here, so they are the science
    range widened slightly -- the Fisher solve never leaves the truth point.
    """
    out = []
    for z in FREE_Z:
        out.append(Par(SYMBOL[z], point[f"a_{SYMBOL[z]}"], 1e-3, 0.02, 3.0))
    if mode == "single":
        out.append(Par("kT", point["kT"], 5e-3, 1.0, 9.0))
    else:
        out.append(Par("T_mean", point["T_mean"], 5e-3, 1.0, 9.0))
        out.append(Par("T_sigma", point["T_sigma"], 5e-3, 0.1, 3.5))
    out.append(Par("sigma_v", point["sigma_v"], 1.0, 10.0, 700.0))
    out.append(Par("n_h", point["n_h"], 1e-2, 0.0, 6.0))
    out.append(Par("log_norm", log_norm_truth, 2e-3,
                   log_norm_truth - 1.0, log_norm_truth + 1.0))
    return out


def stage_bias(args, points, truth_path, outp):
    tz = np.load(truth_path, allow_pickle=True)
    counts = tz["counts"]
    if len(counts) != len(points):
        raise SystemExit(f"truth has {len(counts)} points but the sweep asks "
                         f"for {len(points)} -- regenerate with the same "
                         f"--n_points/--seed/--mode")

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response)
    emu = JointOperatorModel(models_dir=args.store, device=args.device)
    dem = gaussian_dem()[0] if args.mode == "dem" else None
    fwd = Forward(emu, response, absorption, keep, args.mode, dem=dem)

    done = {}
    if os.path.exists(outp) and args.resume:
        with open(outp) as f:
            done = {int(json.loads(l)["point"]): 1 for l in f if l.strip()}
        print(f"resuming bias: {len(done)} points already done", flush=True)

    for i, pt in enumerate(points):
        if i in done:
            continue
        d_ref = counts[i][keep]
        if d_ref.sum() <= 0:
            print(f"point {i}: zero in-band truth, skipped", flush=True)
            continue
        s = N_REF / d_ref.sum()
        d = d_ref * s
        log_norm_truth = float(np.log10(NORM_REF * s))
        # the DEM model has to carry THIS point's shape, not the fiducial's
        if args.mode == "dem":
            fwd.dem = gaussian_dem(mean=pt["T_mean"], sigma=pt["T_sigma"])[0]
        pars = build_pars(fwd, pt, log_norm_truth, args.mode)
        t0 = time.time()
        b_sys, sigma_ref = linear_bias_fisher(fwd, pars, d, verbose=False)
        rec = {"point": i, "params": pt, "names": [p.name for p in pars],
               "truth": [p.truth for p in pars], "b_sys": b_sys.tolist(),
               "sigma_ref": sigma_ref.tolist(), "n_ref": N_REF,
               "log_norm_truth": log_norm_truth,
               "runtime_s": time.time() - t0}
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        worst = int(np.argmax(np.abs(b_sys) / sigma_ref))
        print(f"[{i + 1}/{len(points)}] {rec['runtime_s']:.1f}s  worst "
              f"{rec['names'][worst]} b/sig@Nref="
              f"{b_sys[worst] / sigma_ref[worst]:+.3f}", flush=True)


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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(RESULTS, "bias_sweep"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--target_counts", type=float, default=1e6,
                    help="in-band counts at which to report bias/sigma")
    args = ap.parse_args()

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
