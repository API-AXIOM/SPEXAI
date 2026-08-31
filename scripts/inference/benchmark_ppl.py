"""Does the Pyro layer cost enough to matter? And what does B=1 forfeit?

Two questions decide how the sampler layer gets built.

**1. Pyro overhead.** If wrapping the likelihood in a probabilistic program adds
a negligible slice of a forward, the Pyro interface wins on merits that have
nothing to do with speed: declarative priors, transforms and their Jacobians
from tested upstream code, one model shared by SVI and NUTS. Overhead is
isolated with a deliberately *cheap* forward -- against the real emulator any
framework cost would vanish into the noise and prove nothing. The cheap number
is then compared against the emulator's measured cost per walker.

**2. Batch scaling.** The ensemble samplers (emcee, zeus, UltraNest) amortise
every walker into one batched forward; a single-point sampler like Pyro's NUTS
evaluates at B=1. If the forward is far cheaper per walker at B=96 than at B=1,
NUTS pays a structural penalty that has nothing to do with its mixing -- which
is the argument for a vectorised NUTS over batched chains.

Laptop-safe by default (cheap forward + a small element subset). The absolute
batch-scaling curve that matters is the GPU one:

    python scripts/benchmark_ppl.py --device cuda --elements 26 14 8 2 --real
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spexai.inference.posterior import BoxPrior, PoissonPosterior  # noqa: E402
from spexai.inference.ppl import SpectrumModel, uniform_priors     # noqa: E402


class CheapForward:
    """A forward with the right interface and almost no cost, so the timing
    isolates framework overhead rather than measuring the emulator again."""

    def __init__(self, ndim, n_chan, device="cpu"):
        g = torch.Generator().manual_seed(0)
        self.w = torch.rand(ndim, n_chan, generator=g, dtype=torch.float64)
        self.names = [f"p{i}" for i in range(ndim)]
        self.device = device

    def counts_torch(self, th, grad=False):
        return torch.exp(th.double() @ self.w * 0.1 + 1.0)

    def __call__(self, theta):
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float64)
        return self.counts_torch(th).detach().cpu().numpy()


def _time(fn, n, warmup=3, sync=lambda: None):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t0) / n


def bench_overhead(args):
    """Hand-rolled potential vs the same density through a Pyro model."""
    import pyro
    from pyro.infer import Trace_ELBO
    from pyro.infer.autoguide import AutoMultivariateNormal

    ndim, n_chan = args.ndim, args.n_chan
    fwd = CheapForward(ndim, n_chan)
    lo = np.zeros(ndim)
    hi = np.ones(ndim)
    rng = np.random.default_rng(0)
    data = rng.poisson(fwd(np.full((1, ndim), 0.5))[0]).astype(np.float64)

    prior = BoxPrior(lo, hi, fwd.names)
    post = PoissonPosterior(fwd, data, prior)
    model = SpectrumModel(fwd, data, uniform_priors(fwd.names, lo, hi))

    print(f"\nFRAMEWORK OVERHEAD  (cheap forward, ndim={ndim}, "
          f"channels={n_chan}, {args.reps} reps)")
    print(f"{'B':>5} {'hand-rolled':>13} {'pyro ELBO':>13} {'ratio':>8} "
          f"{'overhead/call':>14}")
    rows = []
    for B in args.batches:
        z = torch.as_tensor(rng.standard_normal((B, ndim)), dtype=torch.float64)
        t_hand = _time(lambda: post.potential_and_grad(z), args.reps)

        pyro.clear_param_store()
        guide = AutoMultivariateNormal(model)
        elbo = Trace_ELBO(num_particles=B, vectorize_particles=True)
        opt = torch.optim.Adam(
            list(guide.parameters()) or [torch.zeros(1, requires_grad=True)],
            lr=1e-3)

        def step():
            opt.zero_grad()
            loss = elbo.differentiable_loss(model, guide)
            loss.backward()
            opt.step()

        t_pyro = _time(step, args.reps)
        rows.append((B, t_hand, t_pyro))
        print(f"{B:>5} {t_hand*1e3:>11.3f}ms {t_pyro*1e3:>11.3f}ms "
              f"{t_pyro/t_hand:>8.2f}x {(t_pyro-t_hand)*1e3:>12.3f}ms")
    return rows


def bench_batch_scaling(args):
    """Cost per walker of the real emulator forward vs batch size."""
    from spexai.inference.operator_model import JointOperatorModel

    dev = args.device
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (
        lambda: None)
    els = [int(z) for z in args.elements] if args.elements else None
    joint = JointOperatorModel(device=dev, elements=els)
    edges = torch.logspace(np.log10(2.0), np.log10(10.0), args.n_chan + 1,
                           device=dev)
    print(f"\nBATCH SCALING  (real forward, {len(joint.models)} elements, "
          f"device={dev}, {args.reps_real} reps)")
    print(f"{'B':>5} {'total':>11} {'per walker':>12} {'vs B=1':>8}")
    base = None
    for B in args.batches:
        T = torch.linspace(2.0, 6.0, B, device=dev)
        ab = {26: torch.full((B,), 1.0, device=dev)}
        v = torch.full((B,), 180.0, device=dev)

        def call():
            joint.batched.flux(T, ab, v, edges)

        t = _time(call, args.reps_real, warmup=2, sync=sync)
        per = t / B
        base = per if base is None else base
        print(f"{B:>5} {t*1e3:>9.1f}ms {per*1e3:>10.2f}ms {base/per:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 4, 16, 64])
    ap.add_argument("--ndim", type=int, default=11,
                    help="matches the hot-floor parameter count")
    ap.add_argument("--n_chan", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps_real", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--elements", nargs="+", default=["2", "8", "14", "26"],
                    help="element subset for the real forward (laptop-safe)")
    ap.add_argument("--real", action="store_true",
                    help="also run the real-emulator batch scaling")
    ap.add_argument("--skip_overhead", action="store_true")
    args = ap.parse_args()

    if not args.skip_overhead:
        bench_overhead(args)
    if args.real:
        bench_batch_scaling(args)


if __name__ == "__main__":
    main()
