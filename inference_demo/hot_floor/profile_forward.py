"""DEPRECATED — merged into ``scripts/benchmark_inference.py``.

This profiled the hot-floor demo's ``EnsembleForward`` (Tier-1: single-T,
sigma_v fixed). Its capabilities — stage breakdown, ``--detailed`` torch.profiler
table, and the float32-vs-float64 FFT micro-bench — now live in the unified
inference benchmark, which runs on the shared ``JointOperatorModel`` path
(serial loop + ``.batched`` vmap groups) and ladders every current acceleration:

    python scripts/benchmark_inference.py --device cuda --stages --detailed --fft-bench

See that script's ``--help`` for the full interface (config selection, RMF fold,
memory knobs, MCMC/SBC projections).
"""
import os
import sys

_MSG = __doc__

if __name__ == "__main__":
    sys.stderr.write(_MSG + "\n")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.stderr.write(f"\nrun: python {os.path.join(repo, 'scripts', 'benchmark_inference.py')} "
                     "--help\n")
    sys.exit(1)
