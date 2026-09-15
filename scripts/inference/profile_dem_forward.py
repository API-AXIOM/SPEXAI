"""Stage-split timing of ONE DEM forward (and its gradient) on the fast path.

The ladder in ``bench_dem_fast.py`` says what the trunk table and the
contraction bought; this says where the REMAINING time goes. Every stage of
``VectorForward.counts_torch`` is wrapped by a nesting-aware timer, so each
line is *exclusive* (children subtracted) and the lines sum to the total.

    python scripts/inference/profile_dem_forward.py --elements 30 --grid 70

Stages, in call order: table lookup (``_density``), the (B, N*G) @ (N*G, P)
contraction, scatter/FFT/rebin on the fine grid, the analytic line deposit,
and the sparse RMF fold.
"""
import argparse
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import torch

from spexai.inference import batched as bt
from spexai.inference import tempdist as td
from spexai.inference import vector_forward as vf
from spexai.inference.abundances import AbundanceModel
from spexai.inference.operator_model import JointOperatorModel, restrict_band
from spexai.inference.response import Response
from spexai.inference.vector_forward import VectorForward

NAMES = ["logT_mean", "logT_sigma", "sigma_v", "log_norm"]
THETA0 = [np.log10(3.5), 0.15, 200.0, 10.0]

_incl: Dict[str, float] = defaultdict(float)
_excl: Dict[str, float] = defaultdict(float)
_calls: Dict[str, int] = defaultdict(int)
_stack: List[List[float]] = []
_order: List[str] = []
_sync = False


def _tick() -> float:
    if _sync:
        torch.cuda.synchronize()
    return time.perf_counter()


def _wrap(obj, name: str, label: str) -> Callable:
    """Wrap ``obj.name`` with a nesting-aware timer; returns the original."""
    orig = getattr(obj, name)

    def timed(*a, **kw):
        if label not in _order:
            _order.append(label)
        _stack.append([0.0])
        t0 = _tick()
        try:
            return orig(*a, **kw)
        finally:
            dt = _tick() - t0
            child = _stack.pop()[0]
            if _stack:
                _stack[-1][0] += dt
            _incl[label] += dt
            _excl[label] += dt - child
            _calls[label] += 1

    setattr(obj, name, timed)
    return orig


def _instrument() -> None:
    for n, lab in [("_density", "emulator (table lookup)"),
                   ("_abundance_matrix", "abundance matrix"),
                   ("_contracted_flux", "contraction (B,NG)@(NG,P)"),
                   ("_continuum", "continuum glue (widths/reshape)"),
                   ("_deposit_all", "line amplitudes + weights@amp")]:
        _wrap(bt.BatchedJointForward, n, lab)
    for n, lab in [("scatter_to_grid", "scatter to fine grid"),
                   ("fft_broaden", "FFT broadening"),
                   ("rebin_flux", "rebin to RMF grid"),
                   ("deposit_gaussian_lines", "line deposit")]:
        _wrap(bt, n, lab)
    for n, lab in [("fold", "RMF fold"),
                   ("flux", "forward glue (unpack/weights)")]:
        _wrap(VectorForward, n, lab)


def _report(total: float, n: int) -> None:
    print(f"  {'stage':34s} {'ms/fwd':>9s} {'%':>6s}  calls")
    acc = 0.0
    for lab in _order:
        ms, pct = _excl[lab] / n * 1e3, 100 * _excl[lab] / total
        acc += _excl[lab]
        print(f"  {lab:34s} {ms:9.1f} {pct:6.1f}  {_calls[lab] // n}")
    rest = total - acc
    print(f"  {'UNATTRIBUTED':34s} {rest / n * 1e3:9.1f} "
          f"{100 * rest / total:6.1f}")
    print(f"  {'TOTAL':34s} {total / n * 1e3:9.1f}  100.0")


def _clear() -> None:
    _incl.clear(), _excl.clear(), _calls.clear(), _order.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmf", default="~/work/data/spexai/responses/"
                                     "aciss_aimpt_cy28.rmf")
    ap.add_argument("--elements", type=int, default=30)
    ap.add_argument("--grid", type=int, default=70)
    ap.add_argument("--t-lo", type=float, default=0.7)
    ap.add_argument("--t-hi", type=float, default=19.9415)
    ap.add_argument("--band-lo", type=float, default=1.87,
                    help="restrict the emulator grid below this (keV); 0 = off")
    ap.add_argument("--walkers", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=2.0)
    ap.add_argument("--grad", action="store_true")
    args = ap.parse_args()

    global _sync
    _sync = args.device.startswith("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    resp = Response(os.path.expanduser(args.rmf), None)
    joint = JointOperatorModel(device=args.device,
                               elements=list(range(1, args.elements + 1)))
    if args.band_lo > 0:
        restrict_band(joint, args.band_lo)
    dem = td.gaussian_logT(td.TempGrid(args.t_lo, args.t_hi, n=args.grid))

    keep = np.ones(resp.n_channels, dtype=bool)
    fwd = VectorForward(joint, resp, keep, NAMES, AbundanceModel([]),
                        device=args.device, velocity=None, temp_name="kT",
                        norm_name="log_norm", velocity_name="sigma_v",
                        nh_name="n_h", dem=dem, dem_fast=True,
                        contract_first=True, mem_gb=args.mem_gb, exposure=1.0)

    th = torch.as_tensor(np.array(THETA0)[None, :].repeat(args.walkers, 0),
                         dtype=torch.float32).to(args.device)

    t0 = _tick()
    fwd._ensure_temp_table()
    tab = joint.batched.temp_table
    mb = tab.dens.numel() * tab.dens.element_size() / 1e6
    print(f"elements={args.elements} G={args.grid} B={args.walkers} "
          f"device={args.device} band_lo={args.band_lo}")
    print(f"  table {tuple(tab.dens.shape)} = {mb:.0f} MB, "
          f"built in {_tick() - t0:.1f} s (once per run)")

    _instrument()
    fwd.counts_torch(th)                                   # warm-up, discarded
    _clear()
    t0 = _tick()
    for _ in range(args.repeats):
        fwd.counts_torch(th)
    total = _tick() - t0
    print(f"\nvalue forward, {args.repeats} calls")
    _report(total, args.repeats)

    if args.grad:
        _clear()
        def run():
            x = th.clone().requires_grad_(True)
            with joint.batched.grad_enabled(True):
                fwd.counts_torch(x, grad=True).sum().backward()
        run()
        _clear()
        t0 = _tick()
        run()
        total = _tick() - t0
        print("\ngradient forward+backward, 1 call "
              "(backward lands in UNATTRIBUTED)")
        _report(total, 1)


if __name__ == "__main__":
    main()
