"""Reconstruct posterior-predictive inputs for MCMC runs made before the
mcmc_check save-fix (which now stores obs_data/chan_e/ppc directly).

The observed spectrum is reproduced deterministically (same truth npz + counts +
seed as the run), and a random subset of posterior draws is folded through the
forward. The npz is augmented in place so plot_mcmc.py can draw the PPC panel.

  SPEXAI_RESPONSES=~/work/data/spexai/responses \
  python scripts/experiments/hot_floor/ppc_reconstruct.py <mcmc_*_c*.npz ...> \
      --truth_npz <truth_single.npz>
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))
from campaign import find_xrism_response, band_mask, EXCLUDE_NONE, PERSEUS               # noqa: E402
from spexai.config import STORE                                            # noqa: E402
from gpu_forward import EnsembleForward                                    # noqa: E402
from spexai.inference.operator_model import JointOperatorModel             # noqa: E402
from spexai.inference.response import Response                             # noqa: E402
from spexai.inference.absorption import Absorption                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--truth_npz", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="Poisson seed the run used (run_cluster default 0)")
    ap.add_argument("--nppc", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    keep = band_mask(response, exclude=EXCLUDE_NONE)
    absn = Absorption.default()
    # accelerate=False: deterministic (seed=0) reference reconstruction, kept
    # on the unaccelerated float64 path for exact reproducibility.
    emu = JointOperatorModel(models_dir=STORE, device=args.device,
                             accelerate=False)
    ens = EnsembleForward(emu, response, absn, keep, "single", PERSEUS["vel"],
                          args.device)
    chan_e = response.chan_e_cent.cpu().numpy()[keep]
    d_inband = np.load(args.truth_npz)["d_inband"]

    for path in args.npz:
        z = dict(np.load(path, allow_pickle=True))
        if "ppc" in z:
            print(f"{os.path.basename(path)}: already has ppc; skipping")
            continue
        counts, chain = float(z["counts"]), z["chain"]
        mu_true = d_inband * (counts / d_inband.sum())   # same as mcmc_check
        obs = np.random.default_rng(args.seed).poisson(mu_true).astype(float)
        idx = np.random.default_rng(1).choice(
            chain.shape[0], size=min(args.nppc, chain.shape[0]), replace=False)
        ppc = ens(chain[idx])                            # (nppc, n_keep) CPU forward
        assert len(obs) == len(chan_e) == ppc.shape[1], \
            (len(obs), len(chan_e), ppc.shape)
        z.update(obs_data=obs, chan_e=chan_e, ppc=ppc)
        np.savez(path, **z)
        print(f"{os.path.basename(path)}: counts={counts:.1e} obs_sum={obs.sum():.3e} "
              f"ppc={ppc.shape} -> augmented", flush=True)


if __name__ == "__main__":
    main()
