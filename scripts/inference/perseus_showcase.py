"""Perseus cluster showcase: recover known parameters from simulated spectra.

Simulates realistic Perseus observations from the INDEPENDENT ``SpexTruthModel``
(PCHIP over SPEX + exact broadening + Galactic absorption), folded through an
XRISM/Resolve response (falling back to Chandra ACIS if Resolve is absent), with
Poisson noise, then fits them back with the emulator.

  --mode single : single-temperature core (kT, Z, velocity, log_norm)
  --mode dem    : a Gaussian temperature distribution (mean T, width) -- the
                  DEM injection-recovery demonstration

Literature fiducials (see docs/inference_methodology): z=0.0179, N_H~1.4e21 cm^-2,
core kT~3.9 keV, Z_Fe~0.55 Zsun, sigma_v~180 km/s; DEM mean~4.27, width~1.11 keV
(Matthijsse/XRISM). Compute is modest by default; scale up on a cluster.

    conda run -n spexai python scripts/perseus_showcase.py --mode single
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.response import Response
from spexai.inference.absorption import Absorption
from spexai.inference.abundances import AbundanceModel
from spexai.inference.simulate import Observation, simulate_observation
from spexai.inference.fitting import Param, run_emcee, SIGMA_V_PRIOR
from spexai.inference import tempdist as td

RESP_DIR = os.environ.get(
    "SPEXAI_RESPONSES", os.path.expanduser("~/data/spexai_data/responses"))
RESULTS = os.environ.get(
    "SPEXAI_RESULTS", os.path.expanduser("~/data/spexai_data/results"))
MPC_M = 3.0857e22          # 1 Megaparsec in metres
# Perseus: z=0.0179 -> luminosity distance ~75 Mpc; emission measure Y in the
# SPEX unit of 1e64 m^-3 (n_H n_e V). Distance is FIXED (degenerate with Y).
PERSEUS = dict(z=0.0179, dist_mpc=75.0, n_h=1.4e21, kT=3.9, Z=0.55, vel=180.0,
               dem_mean=4.27, dem_sigma=1.11)
PERSEUS["dist_m"] = PERSEUS["dist_mpc"] * MPC_M


def find_response():
    """XRISM/Resolve RMF+ARF; fall back to Chandra ACIS if Resolve is absent.

    Delegates to ``campaign.find_xrism_response`` so there is one definition
    of which Resolve files we use. A local copy once globbed in a different
    order than that module's, so the two could silently disagree about which
    RMF they picked."""
    sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))
    from campaign import find_xrism_response
    try:
        rmf, arf = find_xrism_response()
        return rmf, arf, "XRISM/Resolve"
    except FileNotFoundError:
        rmf = os.path.join(RESP_DIR, "aciss_aimpt_cy28.rmf")
        arf = os.path.join(RESP_DIR, "aciss_aimpt_cy28.arf")
        if not (os.path.exists(rmf) and os.path.exists(arf)):
            raise FileNotFoundError(
                f"no Resolve response, and no ACIS RMF+ARF under {RESP_DIR}")
        return rmf, arf, "Chandra ACIS"


def metal_abundances(elements, z):
    return {el: z for el in elements if el not in (1, 2)}


def run_single(truth, emu, response, absorption, args):
    logz = float(np.log10(PERSEUS["z"]))
    ab = metal_abundances(truth.elements, PERSEUS["Z"])
    p = {"temp": PERSEUS["kT"], "velocity": PERSEUS["vel"], "norm": 1e11,
         "logz": logz, "n_h": PERSEUS["n_h"], "abundances": ab,
         "luminosity_distance": PERSEUS["dist_m"]}
    obs = simulate_observation(truth, response, p, args.exposure,
                               target_counts=args.target_counts,
                               absorption=absorption, rng=args.seed)
    # with the distance fixed at Perseus's, norm IS the physical emission measure
    ln = float(np.log10(obs.true_params["norm"]))
    abmodel = AbundanceModel(emu.elements).global_metallicity("Z")
    params = [Param("temp", 1.0, 8.0, truth=PERSEUS["kT"]),
              Param("Z", 0.1, 1.5, truth=PERSEUS["Z"]),
              Param("velocity", *SIGMA_V_PRIOR, truth=PERSEUS["vel"]),
              Param("log_norm", ln - 1.5, ln + 1.5, truth=ln)]
    res = run_emcee(obs, emu, params,
                    {"abundances": {}, "logz": logz, "n_h": PERSEUS["n_h"],
                     "luminosity_distance": PERSEUS["dist_m"]},
                    nwalkers=args.nwalkers, nsteps=args.nsteps, seed=args.seed,
                    abundance_model=abmodel, absorption=absorption)
    return res, obs


def run_dem(truth, emu, response, absorption, args):
    logz = float(np.log10(PERSEUS["z"]))
    ab = metal_abundances(truth.elements, PERSEUS["Z"])
    # 0.5 keV sits just BELOW the per-element training floor (~0.5013 keV), so
    # it now trips the emulator's validity guard; use the same PCHIP-safe floor
    # the campaign uses (campaign.gaussian_dem), where the Gaussian carries
    # negligible weight anyway.
    grid = td.TempGrid(td.PCHIP_TRUTH_SAFE_LO_KEV, 10.0, n=48)
    dem = td.gaussian_T(grid)
    tp = {"T_mean": PERSEUS["dem_mean"], "T_sigma": PERSEUS["dem_sigma"]}
    w = dem.weights(tp)
    # inject: DEM counts from the truth model + Poisson draw (norm via target)
    norm0 = 1e11
    ld = PERSEUS["dist_m"]
    mu = truth.predict_counts_dem(dem.temp_grid, w, ab, logz, norm0,
                                  PERSEUS["vel"], response, args.exposure,
                                  luminosity_distance=ld, absorption=absorption,
                                  n_h=PERSEUS["n_h"]).squeeze(0).cpu().numpy()
    mu = np.clip(mu, 0.0, None)
    scale = args.target_counts / max(mu.sum(), 1e-30)
    mu = mu * scale
    ln = float(np.log10(norm0 * scale))     # physical emission measure at fixed D
    gen = np.random.default_rng(args.seed)
    obs = Observation(counts=gen.poisson(mu).astype(np.int64), response=response,
                      exposure=args.exposure, true_params={**tp, "norm": norm0 * scale},
                      instrument="dem", expected=mu)
    abmodel = AbundanceModel(emu.elements).global_metallicity("Z")
    params = [Param("T_mean", 1.0, 8.0, truth=PERSEUS["dem_mean"]),
              Param("T_sigma", 0.1, 3.0, truth=PERSEUS["dem_sigma"]),
              Param("Z", 0.1, 1.5, truth=PERSEUS["Z"]),
              Param("velocity", *SIGMA_V_PRIOR, truth=PERSEUS["vel"]),
              Param("log_norm", ln - 1.5, ln + 1.5, truth=ln)]
    res = run_emcee(obs, emu, params,
                    {"abundances": {}, "logz": logz, "n_h": PERSEUS["n_h"],
                     "luminosity_distance": ld},
                    nwalkers=args.nwalkers, nsteps=args.nsteps, seed=args.seed,
                    abundance_model=abmodel, dem=dem, absorption=absorption)
    return res, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--elements", nargs="+", default=["26"],
                    help="element Zs, or 'all' for every element in the manifest")
    ap.add_argument("--exposure", type=float, default=1e5)
    ap.add_argument("--target-counts", type=float, default=1e5)
    ap.add_argument("--nwalkers", type=int, default=24)
    ap.add_argument("--nsteps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "showcase"))
    args = ap.parse_args()

    elements = None if "all" in args.elements else [int(z) for z in args.elements]
    rmf, arf, inst = find_response()
    print(f"response: {inst} ({os.path.basename(rmf)})")
    response = Response(rmf, arf)
    emu = JointOperatorModel(device="cpu", elements=elements)
    truth = SpexTruthModel(device="cpu", elements=elements)
    absorption = Absorption.default()      # cached tbabs if present, else wabs
    print(f"absorption: {absorption.name}")

    runner = run_single if args.mode == "single" else run_dem
    res, obs = runner(truth, emu, response, absorption, args)

    print(f"\nPerseus {args.mode} recovery ({obs.total_counts} counts, "
          f"D = {PERSEUS['dist_mpc']:.0f} Mpc fixed):")
    for i, name in enumerate(res.names):
        q16, q50, q84 = np.percentile(res.samples[:, i], [16, 50, 84])
        t = res.truths[i]
        flag = "" if (q16 <= t <= q84) else "  <-- truth outside 68%"
        print(f"  {name:10s} truth={t:8.4f}  fit={q50:8.4f} "
              f"(-{q50-q16:.3f}/+{q84-q50:.3f}){flag}")

    # log_norm IS the physical emission measure Y (1e64 m^-3) at the fixed distance
    if "log_norm" in res.names:
        i = res.names.index("log_norm")
        y16, y50, y84 = 10.0 ** np.percentile(res.samples[:, i], [16, 50, 84])
        print(f"\n  physical emission measure Y = n_H n_e V (at {PERSEUS['dist_mpc']:.0f} Mpc):")
        print(f"    Y = {y50:.3e} (-{y50-y16:.2e}/+{y84-y50:.2e}) x 1e64 m^-3"
              f"  =  {y50*1e64:.3e} m^-3")

    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        fig = corner.corner(res.samples, labels=res.labels, truths=res.truths)
        fig.savefig(f"{args.out}_{args.mode}_corner.png", dpi=130)
        print(f"wrote {args.out}_{args.mode}_corner.png")
    except Exception as e:
        print("corner plot skipped:", e)


if __name__ == "__main__":
    main()
