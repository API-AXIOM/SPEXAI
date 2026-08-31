"""Line-head investigation sweep for the failing hot, line-rich metals.

The week-long production run trained every element with the SMALLEST line head
(``--line_dim 16 --line_hidden 128 --line_t_freqs 0``). Diagnosis
(scripts/residual_fft.py --stratify) showed the failing elements
(Ti, Mn, Co, Cu, Zn; Cr/Ni marginal) have a fine CONTINUUM but a line head
that is not accurate enough at the 0.1%% level. This driver sweeps line-head
CAPACITY and TEMPERATURE-CONDITIONING knobs -- everything else is held at the
production recipe -- so any change is attributable to the line head.

It is a thin, RESUMABLE wrapper over scripts/run_all_elements.py: each variant
trains into its own ``<runroot>/<variant>`` subtree, and every stage
(preprocess / train / benchmark) is skipped if its output already exists. So if
the job dies (e.g. you lose the ssh connection), just launch the SAME command
again and it continues where it stopped -- nothing is retrained.

Launch it DETACHED so an ssh disconnect cannot kill it:

    cd ~/spexai
    setsid nohup python scripts/run_linehead_sweep.py \
        --dataroot ~/data/spexai_data \
        --runroot  ~/data/spexai_data/runs_linehead \
        --elements 22 --steps 100000 \
        < /dev/null > ~/data/spexai_data/runs_linehead/sweep.log 2>&1 &
    echo $! > ~/data/spexai_data/runs_linehead/sweep.pid   # note the PID
    tail -f ~/data/spexai_data/runs_linehead/sweep.log     # safe to Ctrl-C / disconnect

``setsid`` puts the driver in its own session so SIGHUP on ssh close never
reaches it; ``</dev/null`` detaches stdin. (``tmux``/``screen`` work too.)

Phase A (screen levers, cheap):   --elements 22 --steps 100000   (Ti only)
Phase B (confirm winner, full):   --elements 22 24 25 27 28 29 30 --steps 300000 \
                                  --variants <winner>
Aggregate only (no training):     --report_only
"""
import argparse
import os

# On Linux clusters mkl-service defaults MKL_THREADING_LAYER=INTEL, which
# clashes with the GNU OpenMP (libgomp) that PyTorch loads ("MKL_THREADING_
# LAYER=INTEL is incompatible with libgomp.so.1"). Force the GNU layer before
# any numpy/torch import; setdefault so an explicit override still wins. This
# env is inherited by the run_all_elements children we spawn.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import subprocess
import sys
from collections import OrderedDict

# scripts/ is a sibling of spexai/; run everything from the repo root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "scripts", "emulator", "run_all_elements.py")

SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
           "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V",
           "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
SYMBOL = {z: SYMBOLS[z - 1] for z in range(1, 31)}

# Production recipe MINUS --steps (added per invocation). use_linehead is forced
# ON so the sweep is meaningful even if an element sits near the auto threshold.
BASE_FLAGS = (
    "--mode reweight --n_train 0 --tag reweight_full "
    "--lr 1e-3 --lr_min_frac 0.005 --f_max 4000 --curriculum_frac 0.3 "
    "--points 2048 --signal_frac 0.15 --signal_frac_final -1.0 "
    "--batch 128 --hidden 384 --layers 5 --n_freqs 512 "
    "--film_t_freqs 32 --film_t_fmax 64 --use_linehead on --min_line_bins 32"
)

# variant name -> line-head override flags (appended last, so they win).
# NOTE: line_t_freqs=16 HURT already-good Fe in an earlier probe; 8 is the
# gentler probe here. If the T-conditioned variants (v3/v4) lose to the pure-
# capacity ones (v1/v2), drop line_t_freqs and pursue line_dim/line_hidden.
VARIANTS = OrderedDict([
    ("v0_base",   "--line_dim 16 --line_hidden 128 --line_t_freqs 0"),  # repro
    ("v1_dim64",  "--line_dim 64 --line_hidden 128 --line_t_freqs 0"),
    ("v2_hid256", "--line_dim 16 --line_hidden 256 --line_t_freqs 0"),
    ("v3_tfreq8", "--line_dim 16 --line_hidden 128 --line_t_freqs 8"),
    ("v4_combo",  "--line_dim 64 --line_hidden 256 --line_t_freqs 8"),
])


def _benchmark_path(runroot: str, variant: str, z: int) -> str:
    return os.path.join(runroot, variant, f"element{z}", "tier1",
                        "benchmark_test.json")


def _all_done(runroot: str, variant: str, elements) -> bool:
    return all(os.path.exists(_benchmark_path(runroot, variant, z))
               for z in elements)


def run_variant(python: str, dataroot: str, runroot: str, variant: str,
                elements, steps: int, extra: str) -> int:
    """Train+benchmark one variant over all elements (resumably)."""
    override = VARIANTS[variant]
    train_flags = f"{BASE_FLAGS} --steps {steps} {override} {extra}".strip()
    out = os.path.join(runroot, variant)
    os.makedirs(out, exist_ok=True)
    cmd = [python, RUNNER, "--dataroot", dataroot, "--runroot", out,
           "--elements", *[str(z) for z in elements],
           "--train_flags", train_flags]
    print(f"\n=== variant {variant}: {override} (steps={steps}) ===", flush=True)
    print("  " + " ".join(cmd), flush=True)
    # inherit stdout/stderr so everything lands in the single detached sweep.log
    return subprocess.run(cmd, cwd=REPO).returncode


def aggregate(runroot: str, variants, elements) -> None:
    """Collect every variant's benchmark_test.json into one comparison table."""
    import json

    rows = []
    for v in variants:
        for z in elements:
            bp = _benchmark_path(runroot, v, z)
            if not os.path.exists(bp):
                rows.append({"variant": v, "Z": z, "symbol": SYMBOL[z],
                             "status": "missing"})
                continue
            m = next(iter(json.load(open(bp)).values()))
            rows.append({
                "variant": v, "Z": z, "symbol": SYMBOL[z], "status": "done",
                "test_mre": m["overall"]["mre_mean"],
                "line_mre": m["lines"]["mre_mean"],
                "cont_mre": m["continuum"]["mre_mean"],
                "yield_1pct": m["overall"]["yield_1pct"],
                "yield_01pct": m["overall"]["yield_01pct"],
                "floor_viol_pct": m.get("floor", {}).get("violation_pct"),
            })

    # per-element baseline (v0_base) line-MRE, for a relative-improvement column
    base = {r["Z"]: r.get("line_mre") for r in rows
            if r["variant"] == "v0_base" and r["status"] == "done"}

    def fmt(x, spec):
        return format(x, spec) if isinstance(x, (int, float)) else "-"

    # sort by element, then best yield@0.1% first
    rows.sort(key=lambda r: (r["Z"], -(r.get("yield_01pct") or -1)))
    header = ("| Z | el | variant | test MRE | line MRE | dLineMRE vs base "
              "| cont MRE | yield1% | yield0.1% | floor% |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        if r["status"] != "done":
            lines.append(f"| {r['Z']} | {r['symbol']} | {r['variant']} | "
                         f"(missing) | | | | | | |")
            continue
        b = base.get(r["Z"])
        dl = (f"{(r['line_mre'] - b) / b * 100:+.0f}%"
              if b and r.get("line_mre") is not None else "-")
        lines.append(
            f"| {r['Z']} | {r['symbol']} | {r['variant']} | "
            f"{fmt(r['test_mre'], '.4f')} | {fmt(r['line_mre'], '.4f')} | {dl} | "
            f"{fmt(r['cont_mre'], '.4f')} | {fmt(r['yield_1pct'], '.1f')} | "
            f"{fmt(r['yield_01pct'], '.1f')} | {fmt(r['floor_viol_pct'], '.2f')} |")
    table = "\n".join(lines)

    os.makedirs(runroot, exist_ok=True)
    with open(os.path.join(runroot, "linehead_sweep_summary.json"), "w") as f:
        json.dump({"base_flags": BASE_FLAGS, "variants": dict(VARIANTS),
                   "rows": rows}, f, indent=2)
    with open(os.path.join(runroot, "linehead_sweep_summary.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table, flush=True)
    print(f"\nwrote {os.path.join(runroot, 'linehead_sweep_summary.md')}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Resumable line-head capacity/conditioning sweep.")
    ap.add_argument("--dataroot", required=True,
                    help="holds element_<Z>/ (raw) and processed/element<Z>/")
    ap.add_argument("--runroot", required=True,
                    help="base output dir; each variant -> <runroot>/<variant>")
    ap.add_argument("--elements", nargs="+", type=int, default=[22],
                    help="default 22 (Ti): the purest line-head-only failure")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=100000,
                    help="100k screens levers; rerun the winner at 300k")
    ap.add_argument("--extra", default="",
                    help="extra flags appended to every variant's train_flags")
    ap.add_argument("--report_only", action="store_true",
                    help="skip training, just (re)build the summary table")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    os.makedirs(args.runroot, exist_ok=True)

    if not args.report_only:
        for v in args.variants:
            if _all_done(args.runroot, v, args.elements):
                print(f"=== variant {v}: already complete, skipping ===",
                      flush=True)
                continue
            try:
                rc = run_variant(args.python, args.dataroot, args.runroot, v,
                                 args.elements, args.steps, args.extra)
                if rc != 0:
                    print(f"[warn] variant {v} runner exited {rc}; "
                          "continuing to next variant", flush=True)
            except Exception as e:            # keep the sweep alive
                print(f"[warn] variant {v} raised {e!r}; continuing", flush=True)

    aggregate(args.runroot, args.variants, args.elements)


if __name__ == "__main__":
    main()
