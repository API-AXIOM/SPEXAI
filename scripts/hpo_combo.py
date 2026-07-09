"""Hyperparameter search for the `combo` variant (line head, no Sobolev).

Two-stage random search (no external HPO dependency):

  stage 1: --trials random configurations, each trained for --stage1_steps;
           ranked by validation yield@1% (ties broken by mean rel. error)
  stage 2: the --top best configurations retrained for --stage2_steps

Search space: learning rate, batch size, points per spectrum, trunk
width/depth, activation (gelu/silu/sine), Fourier embedding size and
bandwidth, line-embedding size, trend head on/off, log-space stabiliser
weight, curriculum length.

All trials write tagged checkpoints/histories into --outdir plus a running
`hpo_results.json`; rank any time with

    python scripts/hpo_combo.py --report --outdir ...

Run (GPU machine):

    python scripts/hpo_combo.py --trials 24 --cachedir $CACHE --outdir $RUNS/hpo
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spexai.train.train_operator import build_parser, train

SPACE = {
    "lr": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    "batch": [64, 128, 256],
    "points": [2048, 4096, 8192],
    "hidden": [256, 384, 512],
    "layers": [5, 6, 8],
    "activation": ["gelu", "silu", "sine"],
    "n_freqs": [256, 384, 512],
    "f_max": [4000.0, 8000.0, 16000.0],
    "line_dim": [16, 32],
    "use_trend": [0, 1],
    "w_log": [0.0, 0.1],
    "curriculum_frac": [0.15, 0.3],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def run_trial(tag, cfg, steps, args):
    argv = ["--variant", "combo", "--steps", str(steps), "--tag", tag,
            "--cachedir", args.cachedir, "--outdir", args.outdir,
            "--eval_every", str(max(500, steps // 6))]
    for k, v in cfg.items():
        argv += [f"--{k}", str(v)]
    _, best = train(build_parser().parse_args(argv))
    return {"tag": tag, "config": cfg, "steps": steps,
            "val_yield_1pct": best["val_yield_1pct"],
            "val_mre_mean": best.get("val_mre_mean"),
            "best_step": best.get("step")}


def rank(records):
    return sorted(records, key=lambda r: (-r["val_yield_1pct"],
                                          r["val_mre_mean"] or 1e9))


def report(outdir):
    with open(os.path.join(outdir, "hpo_results.json")) as f:
        res = json.load(f)
    rows = ["| tag | steps | yield1% | MRE | lr | batch | pts | HxL | act "
            "| K | f_max | line_dim | trend | w_log | curr |",
            "|" + "---|" * 15]
    for r in rank(res):
        c = r["config"]
        rows.append(
            f"| {r['tag']} | {r['steps']} | {r['val_yield_1pct']:.2f} "
            f"| {r['val_mre_mean']:.4f} | {c['lr']:g} | {c['batch']} "
            f"| {c['points']} | {c['hidden']}x{c['layers']} "
            f"| {c['activation']} | {c['n_freqs']} | {c['f_max']:g} "
            f"| {c['line_dim']} | {c['use_trend']} | {c['w_log']:g} "
            f"| {c['curriculum_frac']:g} |")
    md = "\n".join(rows)
    with open(os.path.join(outdir, "hpo_results.md"), "w") as f:
        f.write(md + "\n")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/hpo")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--stage1_steps", type=int, default=3000)
    ap.add_argument("--stage2_steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", action="store_true",
                    help="only print the ranking of finished trials")
    args = ap.parse_args()

    if args.report:
        report(args.outdir)
        return

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "hpo_results.json")
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    done = {r["tag"] for r in results}

    rng = random.Random(args.seed)
    configs = [sample_config(rng) for _ in range(args.trials)]

    # stage 1
    for i, cfg in enumerate(configs):
        tag = f"t{i:02d}"
        if tag in done:
            continue
        print(f"=== stage 1 trial {tag}: {cfg}", flush=True)
        results.append(run_trial(tag, cfg, args.stage1_steps, args))
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # stage 2: retrain the top configurations with the full budget
    stage1 = [r for r in results if r["steps"] == args.stage1_steps]
    for r in rank(stage1)[:args.top]:
        tag = r["tag"] + "_long"
        if tag in done:
            continue
        print(f"=== stage 2 {tag}: {r['config']}", flush=True)
        results.append(run_trial(tag, r["config"], args.stage2_steps, args))
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    report(args.outdir)


if __name__ == "__main__":
    main()
