"""Screening program for the training-recipe candidates (Section
"Related work" of the technical report). Four experiments, run
sequentially at t04 architecture on the full grid, short (screening)
budget. Selection is on validation and a FRESH off-grid PCHIP probe --
the frozen SPEX test set is never consulted here.

  1. InfoBatch A/B: does unbiased importance-sampling correction
     (--pr_correct 1) remove the low-T grating regression that biased
     prioritization (--pr_correct 0) causes at full strength (pr_mix
     0.5)? Paired seeds.
  2. Optimizer bake-off: WSD+EMA AdamW (current) vs Schedule-Free vs
     Muon.
  3. muP width sweep: learning rate tuned only at the base width,
     transferred to wider trunks -- does the widest run stay stable and
     improve monotonically?
  4. FINER x curriculum 2x2: variable-periodic activations, with and
     without the coarse-to-fine Fourier curriculum (FINER may subsume
     it).

Each arm trains into <progdir>/<tag>/ and is confirmed on one shared
off-grid probe (--confirm_seed, same set for every arm so the numbers
are comparable). Resumable: an arm whose history exists is skipped.
Winners are composed by hand and confirmed later at big_trunk scale on a
held-out ELEMENT, not the Fe test set.

    nohup python scripts/run_recipe_program.py --cachedir $CACHE \\
        --progdir $RUNS/recipe > $RUNS/recipe/program.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# base flags shared by every arm (t04 architecture, full grid, screening)
BASE = ["--mode", "reweight", "--n_train", "0",
        "--hidden", "384", "--layers", "5", "--n_freqs", "512",
        "--f_max", "4000", "--lr", "3e-3", "--diag_plots", "0"]


def arms(steps, eval_every):
    common = BASE + ["--steps", str(steps), "--eval_every", str(eval_every)]
    A = []
    # 1. InfoBatch A/B at full prioritization strength, paired seeds
    for seed in (0, 1):
        A.append(("ib_biased_s%d" % seed, common + [
            "--pr_mix", "0.5", "--pr_correct", "0", "--seed", str(seed),
            "--schedule", "wsd", "--ema_decay", "0.999"]))
        A.append(("ib_unbiased_s%d" % seed, common + [
            "--pr_mix", "0.5", "--pr_correct", "1", "--seed", str(seed),
            "--schedule", "wsd", "--ema_decay", "0.999"]))
    # 2. optimizer bake-off (pr_mix 0.4, the current recipe setting)
    A.append(("opt_wsd_ema", common + [
        "--pr_mix", "0.4", "--optimizer", "adamw",
        "--schedule", "wsd", "--ema_decay", "0.999"]))
    A.append(("opt_schedulefree", common + [
        "--pr_mix", "0.4", "--optimizer", "schedulefree", "--ema_decay", "0"]))
    A.append(("opt_muon", common + [
        "--pr_mix", "0.4", "--optimizer", "muon", "--lr_muon", "0.008",
        "--schedule", "wsd", "--ema_decay", "0.999"]))
    # 3. muP width sweep: lr tuned at base width 192, transferred upward
    for w in (192, 384, 768):
        flags = list(common)
        flags[flags.index("--hidden") + 1] = str(w)  # override trunk width
        A.append(("mup_w%d" % w, flags + [
            "--pr_mix", "0.4", "--mup", "1", "--mup_base_width", "192",
            "--schedule", "wsd", "--ema_decay", "0.999"]))
    # 4. FINER x curriculum 2x2
    for act in ("gelu", "finer"):
        for cf, cname in (("0.3", "curr"), ("0.0001", "nocurr")):
            A.append(("fin_%s_%s" % (act, cname), common + [
                "--pr_mix", "0.4", "--activation", act,
                "--curriculum_frac", cf,
                "--schedule", "wsd", "--ema_decay", "0.999"]))
    return A


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=REPO, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def summarize(progdir, tags, confirm_seed):
    rows = ["| arm | val MRE | val yield1% | off-grid MRE | off-grid "
            "yield1% | status |", "|---|---|---|---|---|---|"]
    out = {}
    for tag in tags:
        d = os.path.join(progdir, tag)
        hp = os.path.join(d, tag + "_history.json")
        op = os.path.join(d, "offgrid_seed%d.json" % confirm_seed)
        if not os.path.exists(hp):
            rows.append(f"| {tag} | | | | | pending/failed |")
            out[tag] = {"status": "pending/failed"}
            continue
        best = json.load(open(hp))["best"]
        r = {"val_mre": best["val_mre_mean"],
             "val_yield1": best["val_yield_1pct"], "status": "done"}
        og = ""
        if os.path.exists(op):
            m = json.load(open(op))["models"][tag]["overall"]
            r["offgrid_mre"] = m["mre_mean"]
            r["offgrid_yield1"] = m["yield_1pct"]
            og = f"{m['mre_mean']:.4f} | {m['yield_1pct']:.1f}"
        else:
            og = " | "
        out[tag] = r
        rows.append(f"| {tag} | {r['val_mre']:.4f} | {r['val_yield1']:.1f} "
                    f"| {og} | done |")
    with open(os.path.join(progdir, "recipe_summary.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(progdir, "recipe_summary.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir", required=True)
    ap.add_argument("--progdir", required=True)
    ap.add_argument("--steps", type=int, default=20000,
                    help="screening budget per arm")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--confirm_seed", type=int, default=7,
                    help="off-grid probe seed shared by all arms")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these arm tags")
    args = ap.parse_args()
    py = sys.executable

    os.makedirs(args.progdir, exist_ok=True)
    program = arms(args.steps, args.eval_every)
    if args.only:
        program = [a for a in program if a[0] in args.only]
    tags = [t for t, _ in program]
    log = os.path.join(args.progdir, "program.log")
    t00 = time.time()

    for tag, flags in program:
        outdir = os.path.join(args.progdir, tag)
        os.makedirs(outdir, exist_ok=True)
        hp = os.path.join(outdir, tag + "_history.json")
        if os.path.exists(hp):
            print(f"=== {tag}: already done, skipping", flush=True)
            continue
        print(f"=== {tag}: training", flush=True)
        t0 = time.time()
        rc = run([py, "-m", "spexai.train.train_adaptive",
                  *flags, "--tag", tag,
                  "--cachedir", args.cachedir, "--outdir", outdir], log)
        if rc:
            print(f"    {tag} FAILED (see program.log)", flush=True)
            summarize(args.progdir, tags, args.confirm_seed)
            continue
        # confirmation on a fresh, shared off-grid probe (never the test set)
        run([py, "scripts/benchmark_offgrid.py",
             "--ckpts", os.path.join(outdir, tag + ".pt"),
             "--cachedir", args.cachedir, "--seed", str(args.confirm_seed),
             "--outdir", outdir], log)
        summarize(args.progdir, tags, args.confirm_seed)
        print(f"    {tag} done in {(time.time()-t0)/60:.0f} min "
              f"(total {(time.time()-t00)/3600:.1f} h)", flush=True)

    print(f"program complete; summary in {args.progdir}/recipe_summary.md",
          flush=True)


if __name__ == "__main__":
    main()
