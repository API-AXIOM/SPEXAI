"""Measure the two DEM forward accelerations: agreement, and what they buy.

Two independent changes, reported as a ladder against the previous production
forward:

* **table** (``dem_fast``) -- the trunk continuum and the line amplitudes at
  the DEM's fixed temperature grid are constants, so they are tabulated once
  per run instead of recomputed every forward
  (``BatchedJointForward.build_temp_table``).
* **contract** (``contract_first``) -- elements and grid nodes are summed
  BEFORE the fine-grid broadening tail, which is linear, so the tail runs B
  times instead of N*B*G (``BatchedJointForward._contracted_flux``).

Correctness is asserted in ``tests/test_dem_fast.py`` and
``tests/test_contracted.py``; this reports the NUMBERS -- float32 agreement,
wall clock, table size -- for a configuration close to production.

    python scripts/inference/bench_dem_fast.py --elements 30 --grid 48 --walkers 1

The reference rung is the forward as it stood before either change, so on a
laptop keep ``--elements``/``--grid`` modest: its cost is the point.
"""
import argparse
import time
from typing import Dict, List

import numpy as np
import torch

from spexai.inference import tempdist as td
from spexai.inference.abundances import AbundanceModel
from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.response import Response
from spexai.inference.vector_forward import VectorForward

NAMES = ["logT_mean", "logT_sigma", "sigma_v", "log_norm"]
THETA0 = [np.log10(3.5), 0.15, 200.0, 10.0]


def _forward(joint, resp, dem, device, fast, mem_gb, contract=True):
    keep = np.ones(resp.n_channels if hasattr(resp, "n_channels")
                   else resp.fold(torch.zeros(1, resp.energy_edges.numel() - 1)
                                  ).shape[1], dtype=bool)
    return VectorForward(
        joint, resp, keep, NAMES, AbundanceModel([]), device=device,
        velocity=None, temp_name="kT", norm_name="log_norm",
        velocity_name="sigma_v", nh_name="n_h", dem=dem, dem_fast=fast,
        contract_first=contract, mem_gb=mem_gb, exposure=1.0)


def _theta(b: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    th = np.repeat(np.array(THETA0)[None, :], b, axis=0)
    th[:, 0] += rng.uniform(-0.05, 0.05, b)          # distinct walkers
    th[:, 1] *= rng.uniform(0.8, 1.2, b)
    th[:, 2] += rng.uniform(-40.0, 40.0, b)
    return torch.as_tensor(th, dtype=torch.float32)


def _time(fn, n: int) -> float:
    fn()                                              # warm-up, excluded
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmf", default="~/work/data/spexai/responses/"
                                     "aciss_aimpt_cy28.rmf")
    ap.add_argument("--elements", type=int, default=8,
                    help="use the first N elements of the store")
    ap.add_argument("--grid", type=int, default=48, help="DEM grid nodes")
    ap.add_argument("--t-lo", type=float, default=0.7)
    ap.add_argument("--t-hi", type=float, default=19.94)
    ap.add_argument("--walkers", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=2.0)
    ap.add_argument("--single-t", action="store_true",
                    help="also time the single-temperature forward")
    ap.add_argument("--grad", action="store_true",
                    help="also time the gradient path")
    args = ap.parse_args()

    import os
    resp = Response(os.path.expanduser(args.rmf), None)
    joint = JointOperatorModel(device=args.device,
                               elements=list(range(1, args.elements + 1)))
    dem = td.gaussian_logT(td.TempGrid(args.t_lo, args.t_hi, n=args.grid))
    th = _theta(args.walkers).to(args.device)

    rungs = [("reference (neither)", False, False),
             ("+ table", True, False),
             ("+ table + contract", True, True)]
    fwds = {name: _forward(joint, resp, dem, args.device, f, args.mem_gb, c)
            for name, f, c in rungs}
    thn = th.cpu().numpy()

    print(f"elements={args.elements} G={args.grid} B={args.walkers} "
          f"device={args.device}")

    # --- agreement, every rung against the reference -----------------------
    ref_counts = None
    for name, *_ in rungs:
        joint.batched.temp_table = None          # each rung builds what it needs
        c = fwds[name](thn)
        if ref_counts is None:
            ref_counts = c
        else:
            rel = np.abs(c - ref_counts).max() / np.abs(ref_counts).max()
            print(f"  {name:22s} max|d| / max|ref| = {rel:.3e}")

    # --- table build cost + size ------------------------------------------
    joint.batched.temp_table = None
    t0 = time.perf_counter()
    tab = joint.batched.build_temp_table(dem.temp_grid, mem_gb=args.mem_gb)
    t_build = time.perf_counter() - t0
    mb = tab.dens.numel() * tab.dens.element_size() / 1e6
    print(f"  table: {tuple(tab.dens.shape)} = {mb:.0f} MB, "
          f"built in {t_build:.1f} s (once per run)")

    # --- wall clock --------------------------------------------------------
    t_ref = None
    for name, f, c in rungs:
        joint.batched.temp_table = None
        if f:                                    # exclude the one-off build
            joint.batched.build_temp_table(dem.temp_grid, mem_gb=args.mem_gb)
        t = _time(lambda: fwds[name](thn), args.repeats)
        t_ref = t if t_ref is None else t_ref
        print(f"  {name:22s} {t:8.2f} s/forward   {t_ref / t:6.2f}x")

    if args.grad:
        joint.batched.temp_table = None
        for name, f, c in rungs:
            joint.batched.temp_table = None
            if f:
                joint.batched.build_temp_table(dem.temp_grid, mem_gb=args.mem_gb)

            def run(fwd=fwds[name]):
                x = th.clone().requires_grad_(True)
                with joint.batched.grad_enabled(True):
                    fwd.counts_torch(x, grad=True).sum().backward()
            t = _time(run, 1)
            print(f"  grad {name:17s} {t:8.2f} s/forward+backward")

    # --- single temperature: what `contract` alone buys the production fit --
    if args.single_t:
        joint.batched.temp_table = None
        edges = resp.energy_edges
        T = torch.full((args.walkers,), 3.5, device=args.device)
        ab = {z: 1.0 for z in joint.elements}
        for c in (False, True):
            t = _time(lambda c=c: joint.batched.flux(
                T, ab, 200.0, edges, mem_gb=args.mem_gb, contract=c),
                args.repeats)
            print(f"  single-T contract={str(c):5s} {t:8.2f} s/forward")
        a = joint.batched.flux(T, ab, 200.0, edges, mem_gb=args.mem_gb,
                               contract=True)
        b = joint.batched.flux(T, ab, 200.0, edges, mem_gb=args.mem_gb,
                               contract=False)
        rel = float((a - b).abs().max() / b.abs().max())
        print(f"  single-T contract agreement = {rel:.3e}")


if __name__ == "__main__":
    main()
