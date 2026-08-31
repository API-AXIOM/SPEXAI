"""Run the ablation study: base model plus one-component-removed variants
and the fixed-grid baseline, all with the same training budget.

    python scripts/run_ablation.py --steps 6000 [--variants base no_film ...]

Writes checkpoints/history per variant into --outdir and a summary table
(ablation_summary.json / .md) when done.
"""

import argparse
import json
import os

from spexai.train.train_operator import build_parser, train

ALL_VARIANTS = ["base", "no_sobolev", "no_trend", "no_film", "no_fourier",
                "fixed_grid", "hash_grid", "line_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    args, extra = ap.parse_known_args()

    results = {}
    for variant in args.variants:
        targs = build_parser().parse_args(
            ["--variant", variant, "--steps", str(args.steps),
             "--outdir", args.outdir] + extra)
        _, best = train(targs)
        results[variant] = best
        with open(os.path.join(args.outdir, "ablation_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    base = results.get("base", {})
    lines = ["| Variant | val MRE (mean) | yield 1% | Δ1% | yield 10% | Δ10% |",
             "|---|---|---|---|---|---|"]
    for v in args.variants:
        r = results[v]
        d1 = base.get("val_yield_1pct", 0) - r["val_yield_1pct"]
        d10 = base.get("val_yield_10pct", 0) - r["val_yield_10pct"]
        lines.append(f"| {v} | {r['val_mre_mean']:.4f} | {r['val_yield_1pct']:.2f} "
                     f"| {d1:+.2f} | {r['val_yield_10pct']:.2f} | {d10:+.2f} |")
    md = "\n".join(lines)
    with open(os.path.join(args.outdir, "ablation_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
