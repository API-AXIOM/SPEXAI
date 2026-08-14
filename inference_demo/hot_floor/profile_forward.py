"""Profile one walker-batched forward pass and locate the time.

Reports, on ``--device`` at batch ``--nwalkers``:
  1. stage breakdown -- full forward vs the flux stage (nets + FFT broadening +
     absorption + rebin + lines) vs the fold (sparse mat-mul) vs the GPU->CPU
     copy, each CUDA-synchronised;
  2. a raw rfft/irfft micro-benchmark at float32 vs float64 on the actual fine
     grid size -- the broadening runs float64, which is often 1/8-1/64 speed on
     a GPU, the usual reason "the NN is slow on the GPU";
  3. (--detailed) a torch.profiler op-level table, so you see exactly which
     kernels (fft, mm, sparse mm, exp, ...) dominate.

  python inference_demo/hot_floor/profile_forward.py --device cuda --nwalkers 200
  python inference_demo/hot_floor/profile_forward.py --device cuda --detailed
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (PERSEUS, STORE28, find_xrism_response,             # noqa: E402
                        band_mask)
from gpu_forward import EnsembleForward                                    # noqa: E402
from spexai.inference.operator_model import JointOperatorModel             # noqa: E402
from spexai.inference.response import Response                             # noqa: E402
from spexai.inference.absorption import Absorption                        # noqa: E402
from spexai.train.broadening import uniform_log_edges                     # noqa: E402


def make_sync(device):
    return (torch.cuda.synchronize if device == "cuda" else (lambda: None))


def timeit(fn, iters, sync):
    fn(); sync()                                          # warm up + settle
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nwalkers", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=32,
                    help="walker sub-batch size (bounds GPU memory)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 tensor-core matmul (Ampere+)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the per-element coordinate-MLP")
    ap.add_argument("--fft32", action="store_true",
                    help="float32 FFT continuum broadening on CUDA (~4x)")
    ap.add_argument("--detailed", action="store_true",
                    help="torch.profiler op-level table")
    args = ap.parse_args()
    dev = args.device
    sync = make_sync(dev)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.fft32:
        import spexai.train.broadening as _br
        _br.USE_FLOAT32_FFT = True
    print(f"tf32={args.tf32}  compile={args.compile}  fft32={args.fft32}")

    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    rmf, arf = find_xrism_response()
    response = Response(rmf, arf)
    absn = Absorption.default()
    keep = band_mask(response)
    # accelerate=False: this script drives the accel knobs itself (--tf32/
    # --compile/--fft32) for a clean A/B, so the model must not auto-enable them.
    emu = JointOperatorModel(models_dir=STORE28, device=dev, accelerate=False)
    ens = EnsembleForward(emu, response, absn, keep, "single", PERSEUS["vel"],
                          dev, chunk=args.chunk, compile_nets=args.compile)
    pars = ens.params(15.685)
    theta = np.tile([p.truth for p in pars], (args.nwalkers, 1))
    B = args.nwalkers
    Bs = min(B, ens.chunk)                               # stage timers use one chunk
    theta_s = theta[:Bs]

    # fine-grid size K actually used by the FFT broadening
    m0 = next(iter(emu.models.values()))
    uni = uniform_log_edges(float(m0.train_edges[0]), float(m0.train_edges[-1]),
                            1e-5)
    K = uni.numel() - 1
    print(f"device={dev}  batch={B}  chunk={ens.chunk}  elements={len(emu.models)}"
          f"  fine-grid K={K}  n_energy={response.n_energy}  iters={args.iters}\n")

    # --- stage breakdown (flux/fold timed on ONE chunk to fit in memory;
    #     full forward runs the whole batch, chunked internally) ---
    total = ens._flux(theta_s)                            # for isolated fold timing
    t_full = timeit(lambda: ens(theta), args.iters, sync)
    t_flux = timeit(lambda: ens._flux(theta_s), args.iters, sync)
    t_fold = timeit(lambda: ens._fold(total, theta_s), args.iters, sync)
    pw_full, pw_flux, pw_fold = t_full / B, t_flux / Bs, t_fold / Bs
    print("STAGE BREAKDOWN (per walker, ms):")
    print(f"  full (incl. GPU->CPU) : {pw_full*1e3:7.3f}")
    print(f"  flux (nets+broaden+..): {pw_flux*1e3:7.3f}   {100*pw_flux/pw_full:4.0f}%")
    print(f"  fold (sparse mat-mul) : {pw_fold*1e3:7.3f}   {100*pw_fold/pw_full:4.0f}%")
    print(f"  full forward for B={B}: {t_full*1e3:.0f} ms total\n")

    # --- float64 vs float32 FFT on the real fine grid ---
    for tag, dt in (("float32", torch.float32), ("float64", torch.float64)):
        x = torch.randn(Bs, K, dtype=dt, device=dev)
        fft = lambda: torch.fft.irfft(torch.fft.rfft(x, dim=1), n=K, dim=1)
        try:
            t = timeit(fft, max(5, args.iters // 2), sync)
            print(f"  rfft+irfft ({tag}) on ({Bs},{K}): {t*1e3:8.1f} ms")
        except Exception as e:
            print(f"  rfft+irfft ({tag}) failed: {e}")
    print("  (broadening runs float64 by default off-MPS; if the gap is large, "
          "switching CUDA broadening to float32+noise-floor is the fix)\n")

    if args.detailed:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if dev == "cuda" else [])
        ens(theta); sync()
        with profile(activities=acts) as prof:
            for _ in range(5):
                ens(theta)
            sync()
        key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=20))


if __name__ == "__main__":
    main()
