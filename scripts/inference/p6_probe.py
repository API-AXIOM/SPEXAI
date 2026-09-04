"""How many P6 seeds fit in one autograd graph on this GPU? Measure, don't guess.

``VectorForward.counts_torch(grad=True)`` does not chunk walkers, so the L-BFGS
graph scales with the seeds in one call: 40 seeds OOMed a 22 GB card at
20.84 GB. ``--seed_chunk`` bounds it, and ``--echunk`` is the other lever
(under gradients it is the checkpoint segment size, so peak memory falls
roughly linearly with it, paid for in recompute).

The two trade off, and their product sets P6's wall-clock: 40 seeds at chunk 2
is 20 sequential L-BFGS runs, at chunk 8 it is 5. Finding the largest workable
chunk is worth a few GPU-minutes before committing to a long run -- and on a
different card than the one this was sized on, guessing costs a whole booking
per wrong guess.

Runs 2 L-BFGS iterations per (seed_chunk, echunk) combination, catching OOM,
and reports peak allocated memory and seconds per iteration. Nothing it prints
is a science result; it is purely a sizing measurement.

    python -u scripts/inference/p6_probe.py [POINT]
"""
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from mle_reseed import build_tierb_problem, load_tierb, lbfgs_batch  # noqa: E402
from spexai.config import RESULTS, STORE                             # noqa: E402

R = os.path.join(RESULTS, "bias_sweep")
POINT = int(sys.argv[1]) if len(sys.argv) > 1 else 14
COUNTS = 1e9                      # the memory-hungriest level P6 runs at
SEED_CHUNKS = (2, 4, 8, 16)
ECHUNKS = (None, 512, 256)
N_SEEDS_TARGET = 8                # what the projection below assumes


class Args:
    """``build_tierb_problem`` reads attributes only, so a namespace suffices."""
    store = STORE
    device = "cuda"
    counts = COUNTS
    chunk = 32
    mem_gb = 2.0
    echunk = None
    compile = False


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- this probe only means anything on "
                         "the GPU it is sizing")
    free, total = torch.cuda.mem_get_info()
    print(f"card: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
    if (total - free) / 2**30 > 1.0:
        print("WARNING: something else already holds >1 GiB on this card. P6 "
              "needs it exclusive; the numbers below will understate what fits "
              "on an empty card.")

    rec, counts_row, tz = load_tierb(
        os.path.join(R, "bias_single_n20_s3.jsonl"),
        os.path.join(R, "truth_single_n20_s3_stamped.npz"), POINT)
    sigma = np.asarray(rec["sigma_ref"]) * np.sqrt(
        float(rec["n_ref"]) / COUNTS)

    print(f"\n{'echunk':>8} {'seeds':>6} {'peak GiB':>10} {'s/iter':>8}  result")
    best = None
    for ec in ECHUNKS:
        Args.echunk = ec
        forward, prior, pars, truth, names, mu_true, _ = build_tierb_problem(
            Args, rec, counts_row, tz)
        for k in SEED_CHUNKS:
            data = np.stack([np.random.default_rng(9000 + i).poisson(mu_true)
                             for i in range(k)]).astype(np.float64)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                lbfgs_batch(forward, prior, data, truth, max_iter=2,
                            n_restarts=1, sigma_ref=sigma, tol_change=0.0)
                peak = torch.cuda.max_memory_allocated() / 2**30
                dt = (time.time() - t0) / 2.0
                print(f"{str(ec):>8} {k:>6} {peak:>10.2f} {dt:>8.1f}  ok")
                if best is None or k > best[1]:
                    best = (ec, k, dt)
            except torch.OutOfMemoryError:
                print(f"{str(ec):>8} {k:>6} {'--':>10} {'--':>8}  OOM")
                torch.cuda.empty_cache()
                break                      # larger chunks cannot fit either
        del forward, prior
        torch.cuda.empty_cache()

    if best is None:
        raise SystemExit("nothing fit, not even 2 seeds -- lower --echunk "
                         "further, or the card is not actually exclusive")
    ec, k, dt = best
    n_groups = int(np.ceil(N_SEEDS_TARGET / k))
    print(f"\nlargest workable: --seed_chunk {k} --echunk {ec} "
          f"({dt:.1f} s/iter)")
    print(f"{N_SEEDS_TARGET} seeds => {n_groups} sequential groups; at 400 "
          f"iterations that is ~{n_groups * 400 * dt / 3600:.1f} h/point.")
    print("If that is too slow, the levers in order: fewer seeds (SE grows as "
          "1/sqrt(K), and counts buy the same precision for free), then a "
          "lower --max_iter ONLY if the convergence line still reports drift "
          "well under 10% of the measured bias.")


if __name__ == "__main__":
    main()
