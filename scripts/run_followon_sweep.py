"""Follow-on sweep: confirm the T-embedding fix is budget-limited, and push Cr.

The weekend sweep (scripts/run_weekend_sweep.py) established that a Fourier
T-embedding on the trunk conditioning (--film_t_freqs 16) is THE fix for the
failing-band elements, and that no learning-rate / sampling / schedule knob
substitutes for it. Two open questions it left:

  * Na and Ca +T-embed were still CONVERGING at the 100k cap (yield@0.1%
    crossed 50% only at 68k / 88k and was still climbing) -> their true
    ceiling is higher than 100k showed. Are they budget-limited?
  * Cr +T-embed halved MRE (0.0065 -> 0.0030) but crossed no yield threshold
    and had a noisy late trajectory -> does the hottest/hardest element need
    MORE frequencies (film_t_freqs 32) and/or more budget?

This program answers both with from-scratch 200k runs (the clean confirmation:
one schedule, anneal at the end, no stitched LR cycles). An OPTIONAL
continuation set (--with_continuation) A/Bs the cheaper resume path, which is
now viable because the weekend checkpoints DO carry opt_state_dict (the
missing piece that made every earlier finetune degrade the model).

    export MKL_THREADING_LAYER=GNU        # MKL-vs-libgomp abort guard
    nohup python scripts/run_followon_sweep.py \
        --dataroot ~/data/spexai_data \
        --runroot  ~/data/spexai_data/runs_followon \
        --prev_runroot ~/data/spexai_data/runs_sweep \
        > ~/data/spexai_data/runs_followon/followon.log 2>&1 &

Reuses the weekend harness (run_arm / collect / metrics): resumable (an arm
with result.json is skipped), crash-isolated, ranked summary rebuilt after
every arm. Enable --compile 1 only after the Monday timing check.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# reuse the weekend harness verbatim (single source of truth for the arm
# runner, the per-T-band test metrics, and the ranked collector)
import run_weekend_sweep as ws  # noqa: E402

# --- primary: from-scratch 200k confirmation of the budget hypothesis ---
# same recipe as the winning weekend arms, only --steps doubled.
CONFIRM = [
    dict(tag="na_tembed_200k", z=11, steps=200000,
         flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    dict(tag="ca_tembed_200k", z=20, steps=200000,
         flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    dict(tag="cr_tembed_200k", z=24, steps=200000,
         flags=["--lr", "1e-3", "--film_t_freqs", "16"]),
    # Cr push: more T-frequencies (sharper hot-T high-Z edges) + a longer
    # low-LR anneal to settle the noisy late trajectory seen at 100k.
    dict(tag="cr_tembed32_200k", z=24, steps=200000,
         flags=["--lr", "1e-3", "--film_t_freqs", "32",
                "--wsd_decay_frac", "0.45", "--lr_min_frac", "0.005"]),
]


def continuation_arms(prev_runroot: str):
    """OPTIONAL: resume the weekend 100k checkpoints for +100k instead of
    restarting. --finetune skips warmup and re-opens the LR (a fresh anneal
    from a moderate lr down to near-zero over the added budget);
    --resume_optimizer restores AdamW moments (present in these checkpoints).
    A/B against the from-scratch arms above: same final budget (200k), so a
    difference isolates continuation-vs-restart, not budget."""
    def prev(tag):
        return os.path.join(prev_runroot, tag, "sweep.pt")
    return [
        dict(tag="na_tembed_cont", z=11, steps=100000,
             flags=["--lr", "5e-4", "--lr_min_frac", "0.01",
                    "--film_t_freqs", "16", "--init_from", prev("na_tembed"),
                    "--finetune", "1", "--resume_optimizer", "1"]),
        dict(tag="ca_tembed_cont", z=20, steps=100000,
             flags=["--lr", "5e-4", "--lr_min_frac", "0.01",
                    "--film_t_freqs", "16", "--init_from", prev("ca_tembed"),
                    "--finetune", "1", "--resume_optimizer", "1"]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=os.path.expanduser("~/data/spexai_data"))
    ap.add_argument("--runroot",
                    default=os.path.expanduser("~/data/spexai_data/runs_followon"))
    ap.add_argument("--prev_runroot",
                    default=os.path.expanduser("~/data/spexai_data/runs_sweep"),
                    help="weekend sweep dir holding <arm>/sweep.pt for "
                         "--with_continuation")
    ap.add_argument("--steps", type=int, default=200000,
                    help="default budget; per-arm 'steps' overrides it")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--diag_plots", type=int, default=1)
    ap.add_argument("--with_continuation", action="store_true",
                    help="also run the resume-from-100k A/B arms")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.runroot, exist_ok=True)

    arms = list(CONFIRM)
    if args.with_continuation:
        arms += continuation_arms(args.prev_runroot)
    if args.only:
        arms = [a for a in arms if a["tag"] in args.only]

    print(f"follow-on sweep: {len(arms)} arms, default steps={args.steps}, "
          f"compile={args.compile}, device={args.device}\n"
          f"  runroot={args.runroot}", flush=True)
    results = []
    for arm in arms:
        results.append(ws.run_arm(arm, args))
        ws.collect(results, args.runroot)   # refresh summary after every arm
    print(f"\nfollow-on complete. ranked summary: "
          f"{os.path.join(args.runroot, 'sweep_summary.md')}", flush=True)


if __name__ == "__main__":
    main()
