"""Freeze reference values for the inference-API unification refactor.

This script must be run against the **pre-refactor** code. It records what the
old ``spexai.inference.fitting`` path produces, so the new ``SpectralFit`` path
can be checked against it long after the old path has been deleted.

That is the point of freezing to a file rather than writing an old-vs-new test:
a live A/B comparison dies the moment the old code is removed, which is exactly
when the guarantee stops being checkable. A golden file outlives the deletion.

Two cases, chosen to cover the paths the refactor actually touches:

``single``
    single-temperature, 3 elements, no absorption -- the plain path.
``dem_abs``
    a Gaussian DEM *and* Galactic absorption, with ``n_h`` sampled per walker.
    This is the configuration whose wiring was broken in ``tier_c_mcmc.py``
    (renamed ``emulator_bias_posterior_check.py``)
    (a ``VectorForward`` built with no ``dem=``), and per-walker ``n_h`` is the
    axis that carried a real broadcast bug before. A single-T golden case alone
    would pass while proving nothing about either.

Both are deliberately tiny (3 elements, 10-point DEM grid, 50 steps) so the
whole file regenerates in well under a minute on a laptop CPU.

    conda run -n spexai python scripts/inference/make_refactor_golden.py

Re-running must reproduce every array exactly; ``--check`` asserts that against
the existing file instead of overwriting it. (The .npz itself is not byte-stable
-- zip entries carry mtimes -- so the contract is array equality, not file
hash.)
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from spexai.inference.abundances import AbundanceModel
from spexai.inference.absorption import Absorption
from spexai.inference.fitting import Param, build_posterior

try:                       # removed 2026-09-22 -- see the note in main()
    from spexai.inference.fitting import run_emcee
except ImportError:
    run_emcee = None
from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR
from spexai.inference.response import Response
from spexai.inference.simulate import simulate_observation
from spexai.inference import tempdist as td

ELEMENTS = [2, 8, 26]
RMF = os.path.expanduser("~/work/data/spexai/responses/aciss_aimpt_cy28.rmf")
OUT = os.path.join(REPO, "tests", "data", "refactor_golden.npz")

FIXED = {"abundances": {}, "logz": -10.0}
GRID_SEED = 12345          # the theta grid; fixed so the file is reproducible
EMCEE_SEED = 0
NWALKERS, NSTEPS = 16, 50
NGRID = 64


def _obs(model):
    """The same simulated ACIS observation the unit tests already use."""
    resp = Response(RMF, None)        # folding mechanics only; no effective area
    p = {"temp": 3.0, "velocity": 200.0, "norm": 1e10, "logz": -10.0,
         "abundances": {}}
    return simulate_observation(model, resp, p, exposure=1e4,
                                target_counts=2e4, rng=0)


def _single_params():
    return [Param("temp", 1.0, 8.0, truth=3.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0)]


def _dem_params():
    return [Param("T_mean", 1.0, 8.0, truth=3.0),
            Param("T_sigma", 0.05, 4.0, truth=1.0),
            Param("velocity", 30.0, 600.0, truth=200.0),
            Param("log_norm", 9.0, 11.0, truth=10.0),
            Param("n_h", 0.0, 5e21, truth=1e21)]


def _theta_grid(params):
    """(NGRID, ndim) spanning the prior box -- deterministic given GRID_SEED."""
    rng = np.random.default_rng(GRID_SEED)
    lo = np.array([p.low for p in params])
    hi = np.array([p.high for p in params])
    return lo + (hi - lo) * rng.random((NGRID, len(params)))


def _p0(params):
    """The walker cloud ``run_emcee`` starts from, recomputed here.

    Stored in its own right because it is the one part of the sampler path that
    IS deterministic from ``seed``, and because the refactor reconciles the two
    implementations' final clip (``lo + 1e-6`` vs ``lo + 1e-9``). Checking p0
    separately pins that reconciliation without depending on emcee's internals.
    """
    lo = np.array([p.low for p in params])
    hi = np.array([p.high for p in params])
    truths = np.array([p.truth if p.truth is not None else np.nan
                       for p in params])
    rng = np.random.default_rng(EMCEE_SEED)
    center = np.where(np.isfinite(truths), truths, 0.5 * (lo + hi))
    p0 = center + 0.02 * (hi - lo) * rng.standard_normal((NWALKERS, len(params)))
    return np.clip(p0, lo + 1e-6, hi - 1e-6)


def _case(name, model, obs, params, **kw):
    """One frozen case: logp over a fixed grid + a seeded emcee chain."""
    post = build_posterior(obs, model, params, FIXED, **kw)
    if post is None:
        raise RuntimeError(f"{name}: build_posterior refused to vectorise")
    theta = _theta_grid(params)
    logp = np.asarray(post.logp(theta), dtype=np.float64)
    if not np.all(np.isfinite(logp)):
        raise RuntimeError(f"{name}: {np.sum(~np.isfinite(logp))}/{NGRID} "
                           f"grid points gave a non-finite logp")
    # `seed` now covers the chain as well as the walker init: run_emcee pins
    # NumPy's global legacy RNG across emcee's construction, which is where
    # emcee snapshots its own generator. Before 2026-09-22 it did not, and this
    # file could not be frozen at all.
    res = run_emcee(obs, model, params, FIXED, nwalkers=NWALKERS,
                    nsteps=NSTEPS, seed=EMCEE_SEED, **kw)
    return {
        f"{name}__p0": _p0(params),
        f"{name}__theta": theta,
        f"{name}__logp": logp,
        f"{name}__chain": np.asarray(res.chain, dtype=np.float64),
        f"{name}__names": np.array([p.name for p in params]),
        f"{name}__lo": np.array([p.low for p in params]),
        f"{name}__hi": np.array([p.high for p in params]),
        f"{name}__truth": np.array([p.truth for p in params]),
        f"{name}__counts": np.asarray(obs.counts),
    }


def build():
    if run_emcee is None:
        raise SystemExit(
            "fitting.run_emcee no longer exists: it was deleted on 2026-09-22, "
            "at the end of the refactor this file's output certifies.\n\n"
            "That is expected, and it is why the reference values were FROZEN "
            "to a file rather than compared live -- tests/test_refactor_"
            "equivalence.py checks the current code against "
            "tests/data/refactor_golden.npz and needs nothing from here.\n\n"
            "This script is kept as the provenance record: it documents exactly "
            "how those numbers were produced. To re-run it, check out a commit "
            "from before that deletion.")
    if not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")):
        raise SystemExit(f"model store not found under {MODELS_DIR}; set "
                         f"SPEXAI_STORE")
    if not os.path.exists(RMF):
        raise SystemExit(f"ACIS response not found at {RMF}")

    model = JointOperatorModel(device="cpu", elements=ELEMENTS)
    obs = _obs(model)

    out = {}
    print("case 'single' ...", flush=True)
    out.update(_case("single", model, obs, _single_params()))

    print("case 'dem_abs' ...", flush=True)
    ab = AbundanceModel(model.elements)
    out.update(_case("dem_abs", model, obs, _dem_params(),
                     dem=td.gaussian_T(td.TempGrid(1.0, 8.0, n=10)),
                     absorption=Absorption.default(),
                     abundance_model=ab))

    out["meta__elements"] = np.array(ELEMENTS)
    out["meta__exposure"] = np.array(obs.exposure)
    out["meta__rmf"] = np.array(os.path.basename(RMF))
    out["meta__emcee_seed"] = np.array(EMCEE_SEED)
    out["meta__grid_seed"] = np.array(GRID_SEED)
    out["meta__nwalkers"] = np.array(NWALKERS)
    out["meta__nsteps"] = np.array(NSTEPS)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare against the existing file "
                         "instead of overwriting it")
    args = ap.parse_args()

    built = build()

    if args.check:
        ref = np.load(args.out, allow_pickle=False)
        bad = []
        for k, v in built.items():
            if k not in ref:
                bad.append(f"{k}: absent from {args.out}")
            elif not np.array_equal(ref[k], v):
                bad.append(f"{k}: differs")
        for k in ref.files:
            if k not in built:
                bad.append(f"{k}: in the file but no longer generated")
        if bad:
            raise SystemExit("REGENERATION IS NOT REPRODUCIBLE:\n  "
                             + "\n  ".join(bad))
        print(f"reproducible: {len(built)} arrays identical")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **built)
    print(f"\nwrote {args.out}")
    for name in ("single", "dem_abs"):
        print(f"  {name}: logp[0]={built[name + '__logp'][0]:.8f}  "
              f"chain{built[name + '__chain'].shape}")


if __name__ == "__main__":
    main()
