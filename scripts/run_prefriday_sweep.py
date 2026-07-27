"""Pre-Friday probes: does the temperature embedding help the mid group, and
does adding a line-head T-embedding help line-limited iron?

The weekend + follow-on sweeps established the trunk Fourier temperature
embedding (--film_t_freqs) as the fix for the FAILING band. Two questions
remain before committing the week-long production re-run:

  * does the embedding also help the MID group (elements at 4-9e-4 MRE, not
    fully at target)?  -> `fe_film` (Fe), `si_film` (Si)
  * does the same Fourier trick on the LINE head (--line_t_freqs) help an
    element whose limiter is the line forest (Fe)?  -> `fe_film_line`

All arms: from scratch, 100k, lr 1e-3, early stopping off (via COMMON) --
a direction probe, not a full budget. Compare each arm's held-out yield /
MRE against the element's EXISTING baseline benchmark, which already lives
at `runs/element{26,14}/tier1/benchmark_test.json` (no need to retrain a
baseline here).

Reuses the weekend-sweep harness verbatim (run_arm / collect / per-T-band
metrics), like run_followon_sweep.py. Resumable, crash-isolated, ranked
summary rebuilt after every arm.

    export MKL_THREADING_LAYER=GNU
    python scripts/run_prefriday_sweep.py \
        --dataroot ~/data/spexai_data --runroot ~/data/spexai_data/runs_prefriday
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# reuse the weekend harness (single source of truth for the arm runner, the
# per-T-band test metrics, and the ranked collector)
import run_weekend_sweep as ws  # noqa: E402

# label these probe elements in the ranked summary (ws.REGIME only knows the
# failing-band elements otherwise)
ws.REGIME.update({26: "mid-Fe", 14: "mid-Si"})

# all arms: from scratch, 100k (default), lr 1e-3, film_t_freqs 16.
ARMS = [
    dict(tag="fe_film",      z=26, flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    dict(tag="fe_film_line", z=26, flags=["--lr", "1e-3", "--film_t_freqs", "16",
                                          "--line_t_freqs", "16"]),
    dict(tag="si_film",      z=14, flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=os.path.expanduser("~/data/spexai_data"))
    ap.add_argument("--runroot",
                    default=os.path.expanduser("~/data/spexai_data/runs_prefriday"))
    ap.add_argument("--steps", type=int, default=100000,
                    help="direction probe; per-arm 'steps' overrides it")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--diag_plots", type=int, default=1)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.runroot, exist_ok=True)

    arms = ARMS if not args.only else [a for a in ARMS if a["tag"] in args.only]
    print(f"pre-Friday sweep: {len(arms)} arms, steps={args.steps}, "
          f"compile={args.compile}, device={args.device}\n"
          f"  runroot={args.runroot}", flush=True)
    results = []
    for arm in arms:
        results.append(ws.run_arm(arm, args))
        ws.collect(results, args.runroot)   # refresh summary after every arm
    print(f"\npre-Friday sweep complete. ranked summary: "
          f"{os.path.join(args.runroot, 'sweep_summary.md')}\n"
          f"compare fe_film/si_film vs runs/element{{26,14}}/tier1/"
          f"benchmark_test.json; fe_film_line vs fe_film.", flush=True)


if __name__ == "__main__":
    main()
