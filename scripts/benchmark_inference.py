"""Cluster benchmark of the full 30-element inference forward step.

Times one walker-batched forward (the cost of one vectorised MCMC step) for the
whole store, laddering the current improvements so each one's contribution is
visible:

  1. serial-fp64   : per-element loop, float64 FFT, no accel  (reference)
  2. serial-accel  : per-element loop + TF32 + torch.compile + float32 FFT
  3. batched-accel : element nets vmapped over the same-shape groups
                     (JointOperatorModel.batched) + TF32 + float32 FFT   <-- all improvements

For each it reports the per-forward and per-walker time, the cumulative speedup,
and projects the cost of a full MCMC chain and an SBC campaign. The forward
folds through a real RMF if one is given (GPU sparse-CSR fold, as in the
production EnsembleForward); otherwise it runs on a synthetic log grid so the
benchmark is self-contained.

  python scripts/benchmark_inference.py --device cuda --nwalkers 96 --wchunk 16
  python scripts/benchmark_inference.py --device cuda --response rsl_Hp_L.rmf --arf rsl.arf

Notes / current limits:
- Velocity is a scalar (per-walker sigma_v -- freeing it -- needs the Tier-2
  line-deposition vectorisation; the net cost we benchmark is unchanged).
- Abundances are applied as a post-net multiply, so scalar-vs-per-walker
  abundances do not change the forward cost; ndim below is reported for the
  projection only.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.inference.operator_model import (JointOperatorModel,
                                             enable_inference_acceleration)


def make_sync(device):
    return torch.cuda.synchronize if device == "cuda" else (lambda: None)


def timeit(fn, iters, sync):
    fn(); sync()                                  # warm up (absorbs compile stall)
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters


def build_grid(args, device):
    """Target incident-energy edges (M+1,) and an optional GPU sparse fold."""
    if args.response:
        from spexai.inference.response import Response
        resp = Response(args.response, args.arf)
        edges = resp.energy_edges.to(device)
        Rt = resp.R.T.tocsr()                     # counts = eff @ R  == (R^T eff^T)^T
        Rt = torch.sparse_csr_tensor(
            torch.from_numpy(Rt.indptr.astype(np.int64)),
            torch.from_numpy(Rt.indices.astype(np.int64)),
            torch.from_numpy(Rt.data.astype(np.float32)),
            size=Rt.shape, device=device)
        arf = resp.arf.to(device)

        def fold(flux):                           # (B, M) -> (B, C)
            eff = (flux * arf).transpose(0, 1).contiguous()   # (M, B)
            return torch.sparse.mm(Rt, eff).transpose(0, 1)   # (B, C)
        return edges, fold, resp.n_channels
    # synthetic log grid over the training band
    edges = torch.logspace(np.log10(0.3), np.log10(12.0), args.n_energy + 1,
                           device=device)
    return edges, None, args.n_energy


def run_config(name, joint, batched, edges, fold, temps, ab, vel, absn, n_h,
               args, sync):
    """Time the full nwalkers forward, processed in wchunk sub-batches."""
    flux_fn = (joint.batched.flux if batched else joint.flux)
    kw = dict(absorption=absn, n_h=n_h, redshift=args.redshift)
    if batched:
        kw["echunk"] = args.echunk

    def step():
        for lo in range(0, args.nwalkers, args.wchunk):    # bound GPU memory
            t = temps[lo:lo + args.wchunk]                 # (wc,)
            f = flux_fn(t, ab, vel, edges, **kw)           # (wc, M)
            if fold is not None:
                f = fold(f)                                # (wc, C)

    t_step = timeit(step, args.iters, sync)
    return t_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nwalkers", type=int, default=96,
                    help="ensemble size (one vectorised MCMC step = one forward)")
    ap.add_argument("--wchunk", type=int, default=16,
                    help="walker sub-batch (bounds GPU memory)")
    ap.add_argument("--echunk", type=int, default=8192,
                    help="energy-axis chunk for the batched trunk")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--response", default=None, help="RMF path (GPU fold to counts)")
    ap.add_argument("--arf", default=None)
    ap.add_argument("--n_energy", type=int, default=6000,
                    help="synthetic target grid size when no --response")
    ap.add_argument("--absorption", action="store_true")
    ap.add_argument("--n_h", type=float, default=1e21)
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--velocity", type=float, default=180.0)
    ap.add_argument("--nsteps", type=int, default=2000, help="for the projection")
    ap.add_argument("--n_sims", type=int, nargs="+", default=[200, 1000],
                    help="SBC simulation counts for the projection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    sync = make_sync(dev)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")

    # load once with accel OFF so the fp64 reference is genuine; accel is
    # enabled in place before the accelerated configs.
    joint = JointOperatorModel(device=dev, accelerate=False)
    edges, fold, n_out = build_grid(args, dev)
    nel = len(joint.elements)
    print(f"elements: {nel}   target bins: {int(edges.numel() - 1)}   "
          f"fold: {'RMF->%d chan' % n_out if fold else 'none (flux only)'}")

    # in-band temperatures, one per walker; all elements freed (solar values)
    temps = torch.empty(args.nwalkers, device=dev).uniform_(0.7, 9.0)
    ab = {z: 1.0 for z in joint.elements}
    absn = None
    if args.absorption:
        from spexai.inference.absorption import Absorption
        absn = Absorption.default()
    n_h = args.n_h if args.absorption else 0.0

    ndim = nel + 3        # 30 abundances + kT + n_h + log_norm (sigma_v fixed)
    print(f"ndim (projection): {ndim} (all {nel} abundances + kT + n_h + "
          f"log_norm; sigma_v fixed)\n")

    B = args.nwalkers
    results = []

    # 1) serial, no accel (reference)
    t = run_config("serial-fp64", joint, False, edges, fold, temps, ab,
                   args.velocity, absn, n_h, args, sync)
    results.append(("serial-fp64", t))

    # enable the promoted accelerations in place (TF32 + compile + float32 FFT)
    knobs = enable_inference_acceleration(joint.models.values(), dev)
    print(f"enabled accelerations: {knobs or 'none (not CUDA)'}\n")

    # 2) serial + accel
    t = run_config("serial-accel", joint, False, edges, fold, temps, ab,
                   args.velocity, absn, n_h, args, sync)
    results.append(("serial-accel", t))

    # 3) batched + accel (all improvements)
    t = run_config("batched-accel", joint, True, edges, fold, temps, ab,
                   args.velocity, absn, n_h, args, sync)
    results.append(("batched-accel", t))
    gsz = sorted(len(g.zs) for g in joint.batched.groups)
    print(f"batched groups (sizes): {gsz}\n")

    base = results[0][1]
    print(f"{'config':<16}{'ms/step':>10}{'ms/walker':>12}{'vs fp64':>10}"
          f"{'vs serial':>11}")
    sacc = results[1][1]
    for name, t in results:
        print(f"{name:<16}{t*1e3:>10.1f}{t/B*1e3:>12.3f}"
              f"{base/t:>9.2f}x{sacc/t:>10.2f}x")

    # projections off the fully-improved config
    best = results[-1][1]
    print(f"\n=== projection (batched-accel, {B} walkers) ===")
    chain = best * args.nsteps
    print(f"  1 MCMC chain ({args.nsteps} steps): {chain/60:.1f} min "
          f"({chain:.0f}s)")
    for ns in args.n_sims:
        tot = chain * ns
        print(f"  SBC {ns:>4} sims x chain: {tot/3600:.1f} GPU-h")


if __name__ == "__main__":
    main()
