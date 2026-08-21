"""Fast GPU check for the CUFFT_INTERNAL_ERROR fix -- minutes, not hours.

The failure was never about MCMC: it was that cuFFT caches one plan per distinct
transform shape, each pinning a GPU workspace, and the batched forward's shapes
moved on every call. Two things drove that -- the zero-padding tracked the
batch's maximum velocity (so a free ``sigma_v`` changed the transform length
every step) and the row-chunk followed ``N*B`` (which changes whenever the
sampler rejects a different number of walkers).

So the mechanism can be provoked directly: hammer the forward with the varying
velocities and walker counts an MCMC run would produce, and watch the plan cache
and the GPU allocator. A couple of minutes reproduces what took ~800 sampler
steps to kill.

Run the fixed path, then the old one, and compare:

    python scripts/check_cufft_stability.py --device cuda
    python scripts/check_cufft_stability.py --device cuda --emulate_old

``--emulate_old`` restores the un-quantised padding and the unbounded cache. It
is expected to show the plan count and reserved memory climbing with iteration,
and may well end in CUFFT_INTERNAL_ERROR -- that is the reproduction, not a
regression. The fixed run should hold both roughly flat.

Also useful on CPU: the "distinct FFT shapes" count is device-independent, so
``--device cpu`` still shows shape stability (the memory columns stay zero).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

import spexai.train.broadening as br                              # noqa: E402
from spexai.inference.absorption import Absorption                # noqa: E402
from spexai.inference.fitting import SIGMA_V_PRIOR                # noqa: E402
from spexai.inference.operator_model import JointOperatorModel    # noqa: E402
from spexai.inference.response import Response                    # noqa: E402


def plan_cache_size():
    try:
        return int(torch.backends.cuda.cufft_plan_cache.size)
    except (AttributeError, RuntimeError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--store", default=os.environ.get(
        "SPEXAI_STORE", os.path.join(REPO, "spexai", "models")))
    ap.add_argument("--iters", type=int, default=60,
                    help="forward calls; the leak showed up well inside 100")
    ap.add_argument("--max_walkers", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emulate_old", action="store_true",
                    help="un-quantised padding + unbounded plan cache, i.e. the "
                         "behaviour that failed; expected to leak")
    ap.add_argument("--rmf", default=None,
                    help="response; defaults to the XRISM one via experiment.py")
    args = ap.parse_args()

    if args.emulate_old:
        br.FFT_PAD_QUANTUM = 1                      # pad follows max velocity
        if torch.cuda.is_available():
            torch.backends.cuda.cufft_plan_cache.max_size = 4096
        print("EMULATING THE OLD BEHAVIOUR (expect the plan count to climb)\n")

    if args.rmf:
        response = Response(args.rmf, None)   # FFT-plan stability only; area irrelevant
    else:
        from campaign import find_xrism_response
        rmf, arf = find_xrism_response()
        response = Response(rmf, arf)
    emu = JointOperatorModel(models_dir=args.store, device=args.device,
                             accelerate=False)
    absn = Absorption.default()
    batched = emu.batched
    if args.emulate_old and torch.cuda.is_available():
        torch.backends.cuda.cufft_plan_cache.max_size = 4096   # undo the cap
    edges = (response.energy_edges * 1.017284).to(args.device)
    print(f"{len(emu.models)} elements, device={args.device}, "
          f"{args.iters} iterations, walkers 1-{args.max_walkers}\n")

    # every call gets a different walker count AND different per-walker
    # velocities -- exactly what an ensemble sampler produces, and precisely
    # what used to mint a new cuFFT plan each time
    rng = np.random.default_rng(args.seed)
    shapes, rows = set(), []
    orig_rfft = torch.fft.rfft

    def spy(x, *a, **kw):
        shapes.add(tuple(x.shape))
        return orig_rfft(x, *a, **kw)

    torch.fft.rfft = spy
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        for i in range(args.iters):
            b = int(rng.integers(1, args.max_walkers + 1))
            temps = torch.as_tensor(rng.uniform(1.5, 7.5, b),
                                    dtype=torch.float32, device=args.device)
            vel = torch.as_tensor(rng.uniform(*SIGMA_V_PRIOR, b),
                                  dtype=torch.float32, device=args.device)
            n_h = torch.as_tensor(rng.uniform(0.5e21, 3e21, b),
                                  dtype=torch.float32, device=args.device)
            ab = {26: torch.as_tensor(rng.uniform(0.3, 1.5, b),
                                      dtype=torch.float32, device=args.device)}
            batched.flux(temps, ab, vel, edges, absorption=absn, n_h=n_h,
                         redshift=0.017284, mem_gb=args.mem_gb)
            if (i + 1) % max(1, args.iters // 10) == 0:
                res = (torch.cuda.memory_reserved() / 1e9
                       if args.device == "cuda" else 0.0)
                rows.append((i + 1, len(shapes), plan_cache_size(), res))
                print(f"  iter {i+1:>4}  distinct FFT shapes {len(shapes):>4}  "
                      f"cuFFT plans {plan_cache_size():>5}  "
                      f"reserved {res:>6.2f} GB", flush=True)
    except RuntimeError as e:
        print(f"\nFAILED at iteration {i+1}: {str(e)[:120]}")
        print("(with --emulate_old this IS the reproduction)")
        torch.fft.rfft = orig_rfft
        raise SystemExit(1 if not args.emulate_old else 0)
    finally:
        torch.fft.rfft = orig_rfft

    peak = (torch.cuda.max_memory_allocated() / 1e9
            if args.device == "cuda" else 0.0)
    print(f"\ncompleted {args.iters} calls in {time.time()-t0:.0f}s")
    print(f"distinct FFT shapes : {len(shapes)}  -> {sorted(shapes)[:4]}"
          f"{' ...' if len(shapes) > 4 else ''}")
    print(f"cuFFT plans cached  : {plan_cache_size()}")
    print(f"peak allocated      : {peak:.2f} GB")
    if len(rows) >= 2:
        grew = rows[-1][1] - rows[len(rows) // 2][1]
        print(f"\nshape growth over the second half: {grew:+d}")
        print("PASS -- shapes are stable, nothing is accumulating"
              if grew == 0 and not args.emulate_old else
              "leaking (expected with --emulate_old)" if args.emulate_old else
              "FAIL -- shapes still growing; find what else varies")


if __name__ == "__main__":
    main()
