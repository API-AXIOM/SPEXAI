"""Does the emulator forward give the same answer on GPU as on CPU?

This matters for one specific reason. ``b_sys`` is the residual between a SPEX
truth spectrum and the emulator's prediction. Every truth npz was generated on
CPU. The single-T Tier B sweep also ran its bias stage on CPU, so truth and
model shared an arithmetic. The DEM bias stage is far too slow for that (~50x
single-T, ~70 h) and has to run on the GPU -- at which point any CPU-vs-GPU
numerical difference in the FORWARD lands in the residual and is
indistinguishable from emulator error.

The scale that matters: the emulator's own error is ~1e-3 relative. A device
discrepancy at 1e-6 is irrelevant; one at 1e-4 is a tenth of the signal and
should be recorded as a caveat; one at 1e-3 would invalidate a GPU-computed
b_sys against a CPU-generated truth.

Runs one sweep point's forward at the truth vector on both devices and compares,
weighting by counts as well as reporting the raw maximum -- a huge relative
difference in a channel with no counts changes no fit.

    python -u scripts/inference/device_parity.py --mode dem --point 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from bias_sweep import build_pars, sample_points                  # noqa: E402
from campaign import (                                            # noqa: E402
    EXCLUDE_NONE, Forward, band_mask, find_xrism_response, gaussian_dem)
from spexai.config import RESULTS, STORE                          # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.response import Response                    # noqa: E402


def forward_on(device, args, point, response, keep, log_norm_truth):
    emu = JointOperatorModel(models_dir=args.store, device=device)
    dem = gaussian_dem()[0] if args.mode == "dem" else None
    fwd = Forward(emu, response, Absorption.default(), keep, args.mode, dem=dem)
    if args.mode == "dem":
        fwd.dem = gaussian_dem(mean=point["T_mean"], sigma=point["T_sigma"])[0]
    pars = build_pars(fwd, point, log_norm_truth, args.mode)
    return fwd(np.array([p.truth for p in pars]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["single", "dem"], default="dem")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--n_points", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--device", default="cuda", help="the device to test CPU against")
    args = ap.parse_args()

    if args.device != "cpu" and not (
            torch.cuda.is_available() if args.device == "cuda"
            else torch.backends.mps.is_available()):
        raise SystemExit(f"{args.device} not available here")

    points = sample_points(args.n_points, args.mode, args.seed)
    point = points[args.point]
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    keep = band_mask(response, exclude=EXCLUDE_NONE)

    truth_npz = os.path.join(
        RESULTS, "bias_sweep",
        f"truth_{args.mode}_n{args.n_points}_s{args.seed}.npz")
    log_norm_truth = 11.0
    if os.path.exists(truth_npz):
        tz = np.load(truth_npz, allow_pickle=True)
        d_ref = tz["counts"][args.point][keep]
        log_norm_truth = float(np.log10(1e11 * (1e5 / d_ref.sum())))

    print(f"{args.mode} point {args.point}, comparing cpu vs {args.device}",
          flush=True)
    mu_cpu = forward_on("cpu", args, point, response, keep, log_norm_truth)
    mu_dev = forward_on(args.device, args, point, response, keep,
                        log_norm_truth)

    m = mu_cpu > 0
    rel = np.abs(mu_dev[m] - mu_cpu[m]) / mu_cpu[m]
    # counts-weighted: a big relative error where there are no counts is
    # invisible to a fit, and the fit is what b_sys is about
    w = mu_cpu[m] / mu_cpu[m].sum()
    print(f"  channels: {m.sum()} of {mu_cpu.size}")
    print(f"  max relative difference        : {rel.max():.3e}")
    print(f"  median relative difference     : {np.median(rel):.3e}")
    print(f"  counts-weighted mean difference: {float((rel * w).sum()):.3e}")
    print(f"  total counts cpu {mu_cpu.sum():.6e} vs {args.device} "
          f"{mu_dev.sum():.6e} (ratio {mu_dev.sum() / mu_cpu.sum():.8f})")

    wm = float((rel * w).sum())
    print("\nagainst the emulator's own ~1e-3 error:")
    if wm < 1e-5:
        print("  NEGLIGIBLE -- a GPU-computed b_sys is comparable with a "
              "CPU-generated truth and with the CPU-run single-T sweep.")
    elif wm < 1e-4:
        print("  small but worth recording as a caveat: the device difference "
              "is ~1% of the emulator error it sits inside.")
    else:
        print("  TOO LARGE. A GPU b_sys against a CPU truth would attribute "
              "device arithmetic to the emulator. Run the bias stage on the "
              "same device as the truth, or regenerate the truth on GPU.")


if __name__ == "__main__":
    main()
