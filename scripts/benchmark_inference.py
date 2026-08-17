"""Unified benchmark for the operator inference forward step.

One coherent tool for timing the whole 30-element inference forward (the cost of
one vectorised MCMC step), superseding the demo-specific
``inference_demo/hot_floor/profile_forward.py``. It ladders the current
improvements so each one's contribution is visible, and reuses the shared
inference path (``JointOperatorModel`` serial loop + ``.batched`` vmap groups):

  serial-fp64     : per-element loop, float64 FFT, no accel        (reference)
  serial-accel    : + TF32 + torch.compile + float32 FFT
  batched-accel   : element nets vmapped over shape-groups + TF32 + float32 FFT
  batched-compile : batched-accel + torch.compile(vmap)  (compile ∘ vmap)
  batched-bf16    : batched-compile + bfloat16 autocast on the trunk

Options:
  --configs      which of the above to run (default: all)
  --accuracy     relative error of every config vs the float64 reference, so a
                 speedup is never read without its precision cost
  --stages       per-stage breakdown (trunk / broaden+rebin / lines / fold)
  --detailed     torch.profiler op-level CUDA table on the fastest config
  --fft-bench    float32-vs-float64 rfft micro-benchmark on the fine grid
  --response/--arf   fold to counts through a real RMF (GPU sparse fold);
                     otherwise a synthetic log grid is used

  python scripts/benchmark_inference.py --device cuda --nwalkers 96 --wchunk 16
  python scripts/benchmark_inference.py --device cuda --stages --detailed --fft-bench

Current limits (documented, not bugs): velocity is scalar (per-walker sigma_v
needs the Tier-2 line-deposition vectorisation); abundances are a post-net
multiply, so scalar-vs-per-walker abundances do not change the forward cost.
"""
import argparse
import os
import sys
import time

# reduce CUDA fragmentation before torch initialises its allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.inference.operator_model import (JointOperatorModel,
                                             enable_inference_acceleration)

# name -> (batched?, compile_trunk?, bf16?); serial-fp64 also means "accel off".
# Cumulative ladder: each rung adds one thing to the rung above it.
CONFIGS = {
    "serial-fp64": (False, False, False),
    "serial-accel": (False, False, False),
    "batched-accel": (True, False, False),
    "batched-compile": (True, True, False),
    "batched-bf16": (True, True, True),
}


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
        Rt = resp.R.T.tocsr()                     # counts = eff @ R == (R^T eff^T)^T
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
    edges = torch.logspace(np.log10(0.3), np.log10(12.0), args.n_energy + 1,
                           device=device)
    return edges, None, args.n_energy


def forward_kwargs(cfg, args, absn, n_h):
    batched, compile_trunk, bf16 = CONFIGS[cfg]
    kw = dict(absorption=absn, n_h=n_h, redshift=args.redshift)
    if batched:
        kw.update(echunk=args.echunk, mem_gb=args.mem_gb,
                  compile_trunk=compile_trunk, bf16=bf16)
    return batched, kw


def run_config(cfg, joint, edges, fold, temps, ab, args, absn, n_h, sync):
    """Time the full nwalkers forward, processed in wchunk sub-batches."""
    batched, kw = forward_kwargs(cfg, args, absn, n_h)
    flux_fn = joint.batched.flux if batched else joint.flux

    def step():
        for lo in range(0, args.nwalkers, args.wchunk):    # bound GPU memory
            f = flux_fn(temps[lo:lo + args.wchunk], ab, args.velocity, edges, **kw)
            if fold is not None:
                fold(f)

    return timeit(step, args.iters, sync)


def stage_breakdown(cfg, joint, edges, fold, temps, ab, args, absn, n_h, sync):
    """Per-stage per-walker timing for a batched config (one wchunk sub-batch,
    like profile_forward): trunk vmap / broaden+rebin / lines+combine / fold."""
    _, compile_trunk, bf16 = CONFIGS[cfg]
    b = joint.batched
    t = temps[:args.wchunk]
    B = t.numel()
    absorb = n_h > 0.0 and absn is not None
    tfun = absn.transmission_torch if absorb else None
    ech = args.echunk or b._echunk(B, args.mem_gb)

    dens, zs = b._density(t, ech, compile_trunk, bf16)     # materialise once
    cont = b._continuum(dens, edges, args.velocity, absorb, tfun, n_h,
                        args.redshift, args.mem_gb)
    t_den = timeit(lambda: b._density(t, ech, compile_trunk, bf16),
                   args.iters, sync)
    t_con = timeit(lambda: b._continuum(dens, edges, args.velocity, absorb,
                                        tfun, n_h, args.redshift, args.mem_gb),
                   args.iters, sync)
    t_com = timeit(lambda: b._combine(cont, zs, ab, t, edges, args.velocity,
                                      absorb, tfun, n_h, args.redshift),
                   args.iters, sync)
    print(f"\n=== stage breakdown [{cfg}] (per walker, ms; B={B}) ===")
    rows = [("trunk (vmap nets)", t_den), ("broaden+rebin", t_con),
            ("lines+combine", t_com)]
    if fold is not None:
        full = b.flux(t, ab, args.velocity, edges, absorption=absn, n_h=n_h,
                      redshift=args.redshift, echunk=args.echunk,
                      mem_gb=args.mem_gb, compile_trunk=compile_trunk, bf16=bf16)
        t_fold = timeit(lambda: fold(full), args.iters, sync)
        rows.append(("fold (sparse mm)", t_fold))
    tot = sum(t for _, t in rows)
    for name, t in rows:
        print(f"  {name:<20} {t/B*1e3:8.3f}   {100*t/tot:4.0f}%")


def _only(joint, z):
    """Abundances isolating one element (the rest zeroed)."""
    return {zz: (1.0 if zz == z else 0.0) for zz in joint.elements}


def config_flux(cfg, joint, edges, temps, ab, args, absn, n_h):
    """One flux() call under a config's settings -- the accuracy probe."""
    batched, kw = forward_kwargs(cfg, args, absn, n_h)
    flux_fn = joint.batched.flux if batched else joint.flux
    return flux_fn(temps, ab, args.velocity, edges, **kw)


def rel_errors(got, ref, edges, sig_frac=1e-3):
    """Error of ``got`` vs the float64 reference, restricted to bins that carry
    real flux.

    Statistics over *all* bins are useless on a real RMF grid: most incident-
    energy bins sit outside the model band and are exactly zero in both arrays,
    so the median is 0.0 no matter how wrong the config is, and a peak-clamped
    max cannot say whether the worst bin is a bright one or an empty one.

    So: keep only bins with ref >= sig_frac * peak, and report the true relative
    error (ref is its own denominator there, no clamp) plus *where* the worst one
    is and how bright that bin is, which is what decides whether the error
    matters. Returns (max, p99, median, worst_energy_keV, worst_bin_frac_of_peak).
    """
    ref, got = ref.double().reshape(-1), got.double().reshape(-1)
    peak = ref.abs().max()
    sig = ref.abs() >= sig_frac * peak
    if not bool(sig.any()):
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0
    e = (got[sig] - ref[sig]).abs() / ref[sig].abs()
    i = int(e.argmax())
    # bin centres, tiled over the walker axis so the flat index maps back
    cen = torch.sqrt(edges[:-1] * edges[1:]).double()
    cen = cen.repeat(ref.numel() // cen.numel())[sig]
    return (e.max().item(), e.quantile(0.99).item(), e.median().item(),
            cen[i].item(), (ref[sig][i].abs() / peak).item())


def detailed_profile(cfg, joint, edges, fold, temps, ab, args, absn, n_h, sync):
    from torch.profiler import ProfilerActivity, profile
    batched, kw = forward_kwargs(cfg, args, absn, n_h)
    flux_fn = joint.batched.flux if batched else joint.flux
    t = temps[:args.wchunk]

    def step():
        f = flux_fn(t, ab, args.velocity, edges, **kw)
        if fold is not None:
            fold(f)

    step(); sync()                                        # warm up / compile
    acts = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if args.device == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(5):
            step()
        sync()
    key = "cuda_time_total" if args.device == "cuda" else "cpu_time_total"
    print(f"\n=== torch.profiler op table [{cfg}] ===")
    print(prof.key_averages().table(sort_by=key, row_limit=20))


def fft_bench(joint, args, sync):
    K = joint.batched._K
    B = args.wchunk
    print(f"\n=== rfft+irfft micro-bench on the fine grid (B={B}, K={K}) ===")
    for tag, dt in (("float32", torch.float32), ("float64", torch.float64)):
        x = torch.randn(B, K, dtype=dt, device=args.device)
        fn = lambda: torch.fft.irfft(torch.fft.rfft(x, dim=1), n=K, dim=1)
        try:
            t = timeit(fn, max(5, args.iters // 2), sync)
            print(f"  {tag}: {t*1e3:8.1f} ms")
        except Exception as e:                            # e.g. no f64 on MPS
            print(f"  {tag}: failed ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--nwalkers", type=int, default=96)
    ap.add_argument("--wchunk", type=int, default=16,
                    help="walker sub-batch (bounds GPU memory; also the stage/"
                         "detailed batch)")
    ap.add_argument("--echunk", type=int, default=None,
                    help="energy-axis chunk for the batched trunk (default: auto)")
    ap.add_argument("--mem_gb", type=float, default=2.0,
                    help="soft per-intermediate GPU budget for the batched path")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--response", default=None, help="RMF path (GPU fold)")
    ap.add_argument("--arf", default=None)
    ap.add_argument("--n_energy", type=int, default=6000)
    ap.add_argument("--absorption", action="store_true")
    ap.add_argument("--n_h", type=float, default=1e21)
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--velocity", type=float, default=180.0)
    ap.add_argument("--accuracy", action="store_true",
                    help="relative error of each config vs the float64 reference")
    ap.add_argument("--acc_walkers", type=int, default=8,
                    help="walkers used for the --accuracy comparison")
    ap.add_argument("--acc_sig_frac", type=float, default=1e-3,
                    help="accuracy stats use bins >= this fraction of the peak "
                         "(most RMF bins are exactly zero and would swamp them)")
    ap.add_argument("--acc_by_element", action="store_true",
                    help="per-element error table for the most aggressive "
                         "config (one fp64 reference per element; slow)")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--detailed", action="store_true")
    ap.add_argument("--fft-bench", dest="fft_bench", action="store_true")
    ap.add_argument("--nsteps", type=int, default=2000)
    ap.add_argument("--n_sims", type=int, nargs="+", default=[200, 1000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    sync = make_sync(dev)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")

    # canonical order so the accel ladder reads correctly
    order = [c for c in CONFIGS if c in args.configs]

    joint = JointOperatorModel(device=dev, accelerate=False)   # accel enabled below
    edges, fold, n_out = build_grid(args, dev)
    nel = len(joint.elements)
    n_h = args.n_h if args.absorption else 0.0
    absn = None
    if args.absorption:
        from spexai.inference.absorption import Absorption
        absn = Absorption.default()
    print(f"elements: {nel}   target bins: {int(edges.numel() - 1)}   "
          f"fold: {'RMF->%d chan' % n_out if fold else 'none (flux only)'}   "
          f"batched groups: {sorted(len(g.zs) for g in joint.batched.groups)}")

    temps = torch.empty(args.nwalkers, device=dev).uniform_(0.7, 9.0)
    ab = {z: 1.0 for z in joint.elements}
    ndim = nel + 3
    print(f"ndim (projection): {ndim} (all {nel} abundances + kT + n_h + "
          f"log_norm; sigma_v fixed)\n")

    # accuracy reference: captured BEFORE any acceleration is enabled in place,
    # so it is the genuine float64 / no-TF32 path. Flux only (pre-fold), so the
    # numbers are instrument-independent.
    acc_ref, t_acc, el_refs = None, None, {}
    if args.accuracy:
        t_acc = temps[:args.acc_walkers]
        acc_ref = joint.flux(t_acc, ab, args.velocity, edges, absorption=absn,
                             n_h=n_h, redshift=args.redshift).double()
        if args.acc_by_element:               # one fp64 reference per element
            for z in joint.elements:
                el_refs[z] = joint.flux(t_acc, _only(joint, z), args.velocity,
                                        edges, absorption=absn, n_h=n_h,
                                        redshift=args.redshift).double()

    results, errs, accel_on = [], {}, False
    for cfg in order:
        if cfg != "serial-fp64" and not accel_on:
            knobs = enable_inference_acceleration(joint.models.values(), dev)
            print(f"enabled accelerations: {knobs or 'none (not CUDA)'}")
            accel_on = True
        t = run_config(cfg, joint, edges, fold, temps, ab, args, absn, n_h, sync)
        results.append((cfg, t))
        if acc_ref is not None:
            errs[cfg] = rel_errors(
                config_flux(cfg, joint, edges, t_acc, ab, args, absn, n_h),
                acc_ref, edges, args.acc_sig_frac)
        if args.stages and CONFIGS[cfg][0]:               # batched configs only
            stage_breakdown(cfg, joint, edges, fold, temps, ab, args, absn,
                            n_h, sync)

    B = args.nwalkers
    base = dict(results).get("serial-fp64", results[0][1])
    sacc = dict(results).get("serial-accel", results[0][1])
    print(f"\n{'config':<17}{'ms/step':>10}{'ms/walker':>12}{'vs fp64':>10}"
          f"{'vs serial':>11}")
    for name, t in results:
        print(f"{name:<17}{t*1e3:>10.1f}{t/B*1e3:>12.3f}"
              f"{base/t:>9.2f}x{sacc/t:>10.2f}x")

    if acc_ref is not None:
        print(f"\n=== accuracy vs float64 reference ({args.acc_walkers} walkers, "
              f"flux only; bins >= {args.acc_sig_frac:g} of peak) ===")
        print(f"{'config':<17}{'max rel':>10}{'p99':>10}{'median':>10}"
              f"{'worst @ keV':>13}{'that bin/peak':>15}")
        for name, _ in results:
            mx, p99, md, ekev, frac = errs[name]
            print(f"{name:<17}{mx:>10.2e}{p99:>10.2e}{md:>10.2e}"
                  f"{ekev:>13.3f}{frac:>15.2e}")
        print("  context: the emulator's own test MRE is ~3e-3. A worst bin that "
              "is a small fraction of peak matters far less than one at the peak.")
        if el_refs:
            cfg = order[-1]                   # the most aggressive rung run
            print(f"\n=== per-element accuracy [{cfg}] (worst 12) ===")
            print(f"{'Z':>4}{'max rel':>12}{'p99':>10}{'worst @ keV':>13}")
            rows = []
            for z, ref_z in el_refs.items():
                got = config_flux(cfg, joint, edges, t_acc, _only(joint, z),
                                  args, absn, n_h)
                mx, p99, _, ekev, _ = rel_errors(got, ref_z, edges,
                                                 args.acc_sig_frac)
                rows.append((mx, p99, ekev, z))
            for mx, p99, ekev, z in sorted(rows, reverse=True)[:12]:
                print(f"{z:>4}{mx:>12.2e}{p99:>10.2e}{ekev:>13.3f}")

    best = min(results, key=lambda r: r[1])
    chain = best[1] * args.nsteps
    print(f"\n=== projection (fastest: {best[0]}, {B} walkers) ===")
    print(f"  1 MCMC chain ({args.nsteps} steps): {chain/60:.1f} min")
    for ns in args.n_sims:
        print(f"  SBC {ns:>4} sims x chain: {chain*ns/3600:.1f} GPU-h")

    if args.detailed:
        detailed_profile(best[0], joint, edges, fold, temps, ab, args, absn,
                         n_h, sync)
    if args.fft_bench:
        fft_bench(joint, args, sync)


if __name__ == "__main__":
    main()
