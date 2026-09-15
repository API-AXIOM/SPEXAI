"""CPU checks for the log-T DEM campaign design (2026-09-15).

Two independent checks, neither of which runs a SPEX truth:

``--table``  Resolution and containment over a design draw. For every DEM
             point: how many grid nodes fall within +-1 sigma of the centre
             (the resolution the width actually gets), and the contained
             fraction (sum of the un-renormalised weights). No emulator.

``--steps``  Finite-difference step check. The Tier B/P6 Jacobian is a central
             difference with ``Par.step``; halve the step and compare the
             Fisher-weighted Jacobian column, ||(J_h - J_h/2)/sqrt(mu)|| /
             ||J_h/2/sqrt(mu)||. Truncation error falls as h^2 and float32
             round-off grows as 1/h, so a small number means the step sits in
             the flat part. Points: the fiducial, the cold-narrow and hot-wide
             DEM corners, and single-T kT at both ends of 0.7-15 keV.

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python \\
        scripts/inference/check_logT_design.py --table --steps
"""
import argparse
import dataclasses
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from bias_sweep import build_pars, contained_fraction, sample_points  # noqa: E402
from campaign import (                                            # noqa: E402
    EXCLUDE_NONE, FREE_Z, PERSEUS, band_mask, find_xrism_response,
    gaussian_logT_dem, restrict_to_band)


def table(n: int, seed: int) -> None:
    pts = sample_points(n, "dem", seed)
    dem, _ = gaussian_logT_dem()
    x = np.log10(dem.temp_grid.numpy().astype(np.float64))         # (G,)
    mu = np.array([p["logT_mean"] for p in pts])                  # (n,)
    sg = np.array([p["logT_sigma"] for p in pts])                 # (n,)
    # (n, G) -> nodes within one sigma of each centre
    nodes = (np.abs(x[None, :] - mu[:, None]) < sg[:, None]).sum(axis=1)
    cf = np.array([contained_fraction(p, "dem") for p in pts])
    cell = float(np.diff(x).mean())
    print(f"grid: {x.size} nodes, {cell:.4f} dex/cell; design draw n={n} "
          f"seed={seed}")
    print(f"  nodes within +-1 sigma : min {nodes.min()}  median "
          f"{int(np.median(nodes))}  max {nodes.max()}")
    print(f"  sigma / cell           : min {sg.min() / cell:.2f}  max "
          f"{sg.max() / cell:.2f}")
    print(f"  contained fraction     : min {cf.min():.3f}  median "
          f"{np.median(cf):.3f}  max {cf.max():.3f}")
    for thr in (0.99, 0.9, 0.5):
        print(f"  points below {thr:.2f}      : {int((cf < thr).sum())}/{n}")
    worst = np.argsort(cf)[:3]
    for i in worst:
        print(f"    least contained: T={10 ** mu[i]:6.2f} keV  "
              f"sigma={sg[i]:.3f} dex  contained={cf[i]:.3f}")


def _forward(names: List[str], dem):
    """Fast-path VectorForward, built as mle_reseed.tierb_forward builds it."""
    from spexai.config import STORE
    from spexai.inference.abundances import AbundanceModel, SYMBOL
    from spexai.inference.absorption import Absorption
    from spexai.inference.operator_model import JointOperatorModel
    from spexai.inference.response import Response
    from spexai.inference.vector_forward import VectorForward
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    keep = band_mask(response, exclude=EXCLUDE_NONE)
    emu = JointOperatorModel(models_dir=STORE, device="cpu", accelerate=False)
    restrict_to_band(emu, verbose=False)
    ab = AbundanceModel(emu.elements)
    for z in FREE_Z:
        ab.free_element(z, SYMBOL[z])
    ab.tie_const([z for z in emu.elements if z >= 3 and z not in FREE_Z],
                 1.0, 26)
    return VectorForward(emu, response, keep, names, ab,
                         absorption=Absorption.default(),
                         redshift=PERSEUS["z"],
                         luminosity_distance=PERSEUS["dist_m"],
                         velocity=None, device="cpu", batched=True,
                         mem_gb=2.0, dem=dem)


def _step_ratio(fwd, pars, keys: List[str]) -> Dict[str, float]:
    from mle_reseed import batched_jacobian
    theta = np.array([[p.truth for p in pars]])                   # (1, ndim)
    cols = {}
    for scale in (1.0, 0.5):
        pp = [dataclasses.replace(p, step=p.step * scale) for p in pars]
        mu0, J = batched_jacobian(fwd, pp, theta)    # (1, n_keep), (1, ndim, n_keep)
        w = 1.0 / np.sqrt(np.clip(mu0[0], 1e-30, None))           # (n_keep,)
        cols[scale] = {k: J[0, [p.name for p in pars].index(k)] * w
                       for k in keys}
    return {k: float(np.linalg.norm(cols[1.0][k] - cols[0.5][k])
                     / np.linalg.norm(cols[0.5][k])) for k in keys}


def steps() -> None:
    base_dem = sample_points(1, "dem", 0)[0]
    base_one = sample_points(1, "single", 0)[0]
    fid = gaussian_logT_dem()[1]
    dem_pts = {"fiducial 3.9 keV, 0.111 dex": fid,
               "cold-narrow 0.75 keV, 0.056 dex":
                   {"logT_mean": np.log10(0.75), "logT_sigma": 0.056},
               "hot-wide 14 keV, 0.4 dex":
                   {"logT_mean": np.log10(14.0), "logT_sigma": 0.4}}
    keys = ["logT_mean", "logT_sigma"]
    fwd = None
    print("\nstep check (Fisher-weighted relative change of J column, h -> h/2)")
    for label, thermal in dem_pts.items():
        pt = dict(base_dem, **{k: float(v) for k, v in thermal.items()})
        pars = build_pars(None, pt, 10.0, "dem")
        if fwd is None:
            fwd = _forward([p.name for p in pars], gaussian_logT_dem()[0])
        r = _step_ratio(fwd, pars, keys)
        print(f"  DEM {label:32s} " + "  ".join(f"{k} {v:.2e}"
                                                 for k, v in r.items()))
    fwd = None
    for kt in (0.75, 14.0):
        pt = dict(base_one, kT=kt)
        pars = build_pars(None, pt, 10.0, "single")
        if fwd is None:
            fwd = _forward([p.name for p in pars], None)
        r = _step_ratio(fwd, pars, ["kT"])
        print(f"  single-T kT={kt:5.2f} keV (step 5e-3 keV)       "
              f"kT {r['kT']:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    np.random.seed(args.seed)
    if args.table:
        table(args.n, args.seed)
    if args.steps:
        steps()


if __name__ == "__main__":
    main()
