"""Precompute the noise-free Perseus truth spectrum LOCALLY (needs the 40 GB
per-element caches) and save a tiny in-band vector, so the cluster MCMC never
needs the caches or SpexTruthModel -- only the 388 MB store + the response.

  KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python -u \
      scripts/inference/dump_truth.py --mode single

Writes results/truth_<mode><tag>.npz with the in-band truth counts at norm_ref,
the reference norm, and the total channel count (a guard against a mismatched
response on the cluster). Transfer that npz with the store + response.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))
from campaign import (                                       # noqa: E402
    injected_abundances, find_xrism_response, band_mask, TruthConfig,
    stream_truth_counts, gaussian_dem, resolve_perseus)
from spexai.config import STORE, RESULTS                      # noqa: E402
from spexai.inference.response import Response               # noqa: E402
from spexai.inference.absorption import Absorption           # noqa: E402
from spexai.inference.operator_model import JointOperatorModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dem"], default="single")
    ap.add_argument("--sigma_v", type=float, default=None)
    ap.add_argument("--kT", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "hot_floor"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    perseus = resolve_perseus({"vel": args.sigma_v, "kT": args.kT})

    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absorption = Absorption.default()
    keep = band_mask(response)
    emu = JointOperatorModel(models_dir=STORE, device="cpu")   # for element set
    dem, dem_p = (gaussian_dem() if args.mode == "dem" else (None, {}))

    ab = injected_abundances(emu.elements)
    cfg = TruthConfig(elements=emu.elements, abundances=ab, exposure=1.0,
                      dem=dem, dem_params=dem_p)
    d_ref = stream_truth_counts(cfg, response, absorption, perseus=perseus,
                                verbose=True)
    d_inband = d_ref[keep]
    suffix = f"_{args.tag}" if args.tag else ""
    outp = os.path.join(args.out, f"truth_{args.mode}{suffix}.npz")
    # record the element set, not just the channel count: a truth built from a
    # 28-element store has the same number of channels as a 30-element one, so
    # without this a stale truth passes every shape check and silently biases
    # the whole campaign
    # rmf/arf are recorded for the same reason as `elements`: an ARF rescales
    # d_inband channel by channel while leaving n_keep and the element list
    # untouched, so nothing else here can detect a response mismatch.
    np.savez(outp, d_inband=d_inband, norm_ref=cfg.norm_ref,
             n_channels=len(keep), n_keep=int(keep.sum()),
             elements=np.asarray(emu.elements, dtype=np.int64),
             store=os.path.abspath(STORE),
             rmf=os.path.basename(rmf), arf=os.path.basename(arf),
             kT=perseus["kT"], sigma_v=perseus["vel"], mode=args.mode)
    print(f"in-band counts at norm_ref={cfg.norm_ref:.1e}: {d_inband.sum():.3e} "
          f"over {len(d_inband)} channels\nsaved {outp}")


if __name__ == "__main__":
    main()
