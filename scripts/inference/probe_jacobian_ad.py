"""Can forward-mode AD replace the finite-difference Jacobian? (probe, 2026-09-16)

``mle_reseed.batched_jacobian`` builds the Tier B / P6 Jacobian by central
differences, which is why every parameter carries a ``step`` and why the step
has to sit in the valley between truncation error (grows as h^2) and float32
round-off (grows as 1/h). The forward is differentiable, so in principle the
Jacobian needs no step at all -- but REVERSE mode is the wrong tool: it costs
one pass per OUTPUT and the Jacobian has ~20k of them (one per channel).
FORWARD mode costs one pass per INPUT, i.e. one per parameter, which is the
shape this problem has.

This probe answers three questions before any code is replaced:

1. Does ``torch.func.jvp`` survive the whole fast path -- trunk table,
   contraction, FFT broadening, analytic line deposit, and the sparse CSR
   response fold? The sparse fold is the one expected to complain.
2. Does the AD column agree with the central difference (at the smaller step,
   which is the more accurate one)?
3. **The table trap.** ``batched._density`` bypasses the temperature table when
   ``temp_kev.requires_grad`` (batched.py:341, :658), because a gather has no
   derivative w.r.t. temperature. A forward-mode dual tensor does NOT set
   ``requires_grad``, so a single-T JVP can be served from the table and come
   back with a silently ZERO tangent. A near-zero AD column is therefore
   reported as the trap, not as disagreement.

Read-only: nothing here changes the estimator.

    python -u scripts/inference/probe_jacobian_ad.py
"""
import argparse
import dataclasses
import os
import sys
from typing import List

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from bias_sweep import build_pars, sample_points                  # noqa: E402
from campaign import gaussian_logT_dem                            # noqa: E402
from check_logT_design import _forward                            # noqa: E402


def _fd_column(fwd, pars, key: str, scale: float):
    """Central-difference J column for ``key`` with the step scaled."""
    from mle_reseed import batched_jacobian
    pp = [dataclasses.replace(p, step=p.step * scale) for p in pars]
    theta = np.array([[p.truth for p in pars]])
    mu0, J = batched_jacobian(fwd, pp, theta)
    j = [p.name for p in pars].index(key)
    return mu0[0], J[0, j]


def _ad_column(fwd, pars, key: str):
    """Forward-mode J column for ``key``: one JVP with a unit tangent.

    Returns ``(column, table_live)`` or raises whatever torch raises -- a
    failure here IS the result, so it is not caught.
    """
    names = [p.name for p in pars]
    th = torch.tensor([[p.truth for p in pars]], dtype=torch.float64,
                      device=fwd.device)
    v = torch.zeros_like(th)
    v[0, names.index(key)] = 1.0
    table_live = getattr(fwd.emu.batched, "temp_table", None) is not None

    def f(t):
        return fwd.counts_torch(t, grad=True)

    _, tangent = torch.func.jvp(f, (th,), (v,))
    return tangent.detach().reshape(-1).double().cpu().numpy(), table_live


def _rel(a, b, mu):
    w = 1.0 / np.sqrt(np.clip(mu, 1e-30, None))
    return float(np.linalg.norm((a - b) * w) / np.linalg.norm(b * w))


def probe(mode: str, keys: List[str], seed: int) -> None:
    pt = sample_points(1, mode, seed)[0]
    if mode == "dem":
        pt.update(gaussian_logT_dem()[1])
    pars = build_pars(None, pt, 10.0, mode)
    fwd = _forward([p.name for p in pars], gaussian_logT_dem()[0]
                   if mode == "dem" else None)
    print(f"\n=== {mode} ===", flush=True)
    for key in keys:
        mu0, j_h = _fd_column(fwd, pars, key, 1.0)
        _, j_half = _fd_column(fwd, pars, key, 0.5)
        print(f"  {key}: FD(h) vs FD(h/2) {_rel(j_h, j_half, mu0):.2e}",
              flush=True)
        try:
            j_ad, table_live = _ad_column(fwd, pars, key)
        except Exception as e:                       # the answer to question 1
            print(f"  {key}: forward-mode AD FAILED ({type(e).__name__}): "
                  f"{str(e)[:300]}", flush=True)
            continue
        scale = np.linalg.norm(j_ad) / max(np.linalg.norm(j_half), 1e-300)
        if scale < 1e-6:
            print(f"  {key}: AD column is ZERO (norm ratio {scale:.1e}) -- "
                  f"the table trap, temp_table live={table_live}", flush=True)
            continue
        print(f"  {key}: AD vs FD(h/2) {_rel(j_ad, j_half, mu0):.2e}  "
              f"AD vs FD(h) {_rel(j_ad, j_h, mu0):.2e}  "
              f"(temp_table live={table_live})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["dem", "single", "both"], default="both")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"torch {torch.__version__}; forward-mode jvp available: "
          f"{hasattr(torch.func, 'jvp')}")
    if args.mode in ("dem", "both"):
        probe("dem", ["logT_mean", "logT_sigma", "Fe"], args.seed)
    if args.mode in ("single", "both"):
        probe("single", ["kT", "Fe"], args.seed)


if __name__ == "__main__":
    main()
