"""Locate non-determinism in the log-likelihood (the i-nessai ModelError).

nessai's ``verify_model`` calls the log-likelihood 16 times on one point and
demands **bit-identical** results (``all(logl == logl[0])``, not ``allclose``),
so any last-bit jitter aborts the run before sampling starts:

    ModelError: Repeated calls to the log-likelihood with the same parameters
    return different values.

The prime suspect is CUDA-only, which is why it never reproduces on a laptop:
``index_add_`` with DUPLICATE indices (``spexai.broadening.
deposit_gaussian_lines``, lines 262 and 284) uses floating-point ``atomicAdd``
on CUDA, so the summation order -- and therefore the last bits -- varies
between otherwise identical calls. Many emission lines share a target bin, so
duplicate indices are guaranteed.

Note that ``bake_off.build_problem`` already passes ``accelerate=False``, so
TF32 and the float32 FFT are NOT involved; ``--compile`` is off by default too.
This script confirms the cause rather than assuming it, by re-running one point
under several configurations and counting distinct results.

Usage (on the GPU machine, same env as the bake-off):
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/inference/diagnose_nondeterminism.py \
        --truth <truth.npz> --device cuda [--n 16] [--compile]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def spread(vals):
    """(n_distinct, absolute spread, relative spread) of repeated calls."""
    v = np.asarray(vals, dtype=np.float64)
    rng = float(v.max() - v.min())
    return len(np.unique(v)), rng, rng / max(abs(float(np.median(v))), 1e-300)


def report(label, vals):
    n, rng, rel = spread(vals)
    verdict = "DETERMINISTIC" if n == 1 else f"{n} distinct values"
    print(f"  {label:<38} {verdict:>22}   spread {rng:.3e} "
          f"({rel:.2e} relative)")
    return n == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="truth npz, as bake_off uses")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=16,
                    help="repeats per configuration (nessai uses 16)")
    # mirrored from bake_off's parser so build_problem sees what it expects
    ap.add_argument("--store", default=None)
    ap.add_argument("--counts", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mem_gb", type=float, default=2.0)
    ap.add_argument("--echunk", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    import bake_off
    from spexai.config import STORE
    if args.store is None:
        args.store = STORE

    print(f"torch {torch.__version__}, device={args.device}, "
          f"cuda={torch.cuda.is_available()}, compile={args.compile}")
    post, pars, names = bake_off.build_problem(args)[:3]
    th = np.asarray([pars[n] for n in names], dtype=float)[None, :]

    print(f"\nEach configuration calls the SAME point {args.n} times "
          f"(nessai requires all {args.n} to be bit-identical):")

    # 1. exactly what the bake-off runs
    ok_prod = report("as the bake-off runs it",
                     [float(post.loglike(th)[0]) for _ in range(args.n)])

    # 2. repeat, discarding the first batch. If (1) failed and this passes, the
    #    jitter is a one-off warm-up (lazy init, autotuning) rather than a
    #    per-call effect, and one throwaway evaluation before sampling fixes it.
    ok_warm = report("repeated, after warm-up",
                     [float(post.loglike(th)[0]) for _ in range(args.n)])

    # 3. deterministic algorithms: forces a deterministic index_add_ kernel.
    #    This is the discriminating test for the atomics hypothesis.
    ok_det = None
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        ok_det = report("with deterministic algorithms",
                        [float(post.loglike(th)[0]) for _ in range(args.n)])
    except Exception as e:                      # not supported on every build
        print(f"  deterministic algorithms unavailable: "
              f"{type(e).__name__}: {e}")
    finally:
        torch.use_deterministic_algorithms(False)

    # 4. CPU: no atomics at all. Slow, so only a few repeats, and it is a
    #    control rather than a proposed fix.
    ok_cpu = None
    if args.device != "cpu":
        try:
            args_cpu = argparse.Namespace(**vars(args))
            args_cpu.device = "cpu"
            post_c, pars_c, names_c = bake_off.build_problem(args_cpu)[:3]
            th_c = np.asarray([pars_c[n] for n in names_c], float)[None, :]
            ok_cpu = report("CPU control (no atomics)",
                            [float(post_c.loglike(th_c)[0])
                             for _ in range(min(args.n, 4))])
        except Exception as e:
            print(f"  CPU control failed: {type(e).__name__}: {e}")

    print("\nreading:")
    if ok_prod:
        print("  Deterministic here -- the failure is elsewhere. Re-run with "
              "the SAME truth\n  file, --counts and --device the failing "
              "bake-off used.")
    elif ok_det:
        print("  CONFIRMED: index_add_ atomics in deposit_gaussian_lines.\n"
              "  Fix: run i-nessai under torch.use_deterministic_algorithms"
              "(True), or memoise\n  the likelihood so repeated identical "
              "points return the cached value.")
    elif ok_warm:
        print("  One-off warm-up jitter, not a per-call effect.\n"
              "  Fix: evaluate the likelihood once before constructing "
              "FlowSampler.")
    elif ok_cpu:
        print("  GPU-specific but NOT fixed by deterministic algorithms -- "
              "look for another\n  atomic/reduction kernel in the forward, or "
              "run this sampler on CPU.")
    else:
        print("  Non-determinism survives everything, including CPU: look for "
              "a stochastic\n  term in the likelihood itself (a noise or "
              "jitter draw), not in the forward.")


if __name__ == "__main__":
    main()
