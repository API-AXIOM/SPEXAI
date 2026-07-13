"""Train and benchmark the emulator for every element, sequentially.

For each element Z the pipeline runs four stages, each skipped if its
output already exists (so the script is resumable after interruptions --
rerun the same command and it continues where it stopped):

  1. preprocess   raw <dataroot>/element_<Z>/Z<Z>_*keV.txt
                  -> <dataroot>/processed/element<Z>/
  2. train        Tier-1 recipe (train_adaptive --mode reweight on the
                  full grid: error-prioritized sampling + EMA weight
                  averaging + WSD schedule; per-run diagnostics figures
                  are produced automatically)
                  -> <runroot>/element<Z>/tier1/reweight_full.pt
  3. benchmark    held-out test set (benchmark_operator)
                  -> <runroot>/element<Z>/tier1/benchmark_test.{json,md}
  4. baselines    classical interpolation for context
                  -> <runroot>/element<Z>/baselines_test.json

After every element the cross-element summary table is rebuilt at
<runroot>/elements_summary.{json,md}. A failing element is recorded there
and does not stop the sweep (e.g. very low-Z elements may have too few
line bins for the line head; check the per-element pipeline.log).

Update TRAIN_FLAGS below (or pass --train_flags) when a better recipe
wins, e.g. the Tier-2 configs.

    nohup python scripts/run_all_elements.py \
        --dataroot ~/data/spexai_data --runroot ~/data/spexai_data/runs \
        > all_elements.log 2>&1 &
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tier-1 recipe; --train_flags appends/overrides
TRAIN_FLAGS = ["--mode", "reweight", "--n_train", "0", "--pr_mix", "0.4",
               "--schedule", "wsd", "--steps", "100000",
               "--eval_every", "2000", "--tag", "reweight_full"]


def run(cmd, log, cwd=REPO):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def summarize(runroot, elements):
    rows, table = {}, [
        "| Z | status | test MRE | line MRE | cont MRE | yield1% | "
        "yield0.1% | floor viol % |",
        "|---|---|---|---|---|---|---|---|"]
    for z in elements:
        path = os.path.join(runroot, f"element{z}", "tier1",
                            "benchmark_test.json")
        if not os.path.exists(path):
            rows[z] = {"status": "pending/failed"}
            table.append(f"| {z} | pending/failed | | | | | | |")
            continue
        with open(path) as f:
            d = json.load(f)
        m = next(iter(d.values()))
        rows[z] = {"status": "done", **m}
        table.append(
            f"| {z} | done | {m['overall']['mre_mean']:.4f} | "
            f"{m['lines']['mre_mean']:.4f} | "
            f"{m['continuum']['mre_mean']:.4f} | "
            f"{m['overall']['yield_1pct']:.1f} | "
            f"{m['overall']['yield_01pct']:.1f} | "
            f"{m.get('floor', {}).get('violation_pct', float('nan')):.2f} |")
    with open(os.path.join(runroot, "elements_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(runroot, "elements_summary.md"), "w") as f:
        f.write("\n".join(table) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True,
                    help="holds element_<Z>/ raw dirs and processed/")
    ap.add_argument("--runroot", required=True)
    ap.add_argument("--elements", nargs="+", type=int,
                    default=list(range(1, 31)))
    ap.add_argument("--train_flags", default="",
                    help="extra flags appended to the training command, "
                         'e.g. "--steps 200000 --line_t_freqs 32"')
    ap.add_argument("--baseline_methods", nargs="+",
                    default=["linear", "pchip"])
    args = ap.parse_args()
    py = sys.executable

    os.makedirs(args.runroot, exist_ok=True)
    t00 = time.time()
    for z in args.elements:
        raw = os.path.join(args.dataroot, f"element_{z}")
        cache = os.path.join(args.dataroot, "processed", f"element{z}")
        outdir = os.path.join(args.runroot, f"element{z}", "tier1")
        eldir = os.path.dirname(outdir)
        os.makedirs(eldir, exist_ok=True)
        log = os.path.join(eldir, "pipeline.log")
        t0 = time.time()
        print(f"=== element {z}", flush=True)

        # 1. preprocess
        if not os.path.exists(os.path.join(cache, "logflux.npy")):
            if not glob.glob(os.path.join(raw, f"Z{z}_*keV.txt")):
                print(f"  no raw data in {raw}, skipping", flush=True)
                summarize(args.runroot, args.elements)
                continue
            print("  preprocessing ...", flush=True)
            if run([py, "scripts/preprocess_spectra.py",
                    "--datadir", raw, "--outdir", cache], log):
                print("  PREPROCESS FAILED (see pipeline.log)", flush=True)
                summarize(args.runroot, args.elements)
                continue

        # 2. train
        if not os.path.exists(os.path.join(outdir,
                                           "reweight_full_history.json")):
            print("  training ...", flush=True)
            cmd = [py, "-m", "spexai.train.train_adaptive",
                   *TRAIN_FLAGS, *args.train_flags.split(),
                   "--cachedir", cache, "--outdir", outdir]
            if run(cmd, log):
                print("  TRAINING FAILED (see pipeline.log)", flush=True)
                summarize(args.runroot, args.elements)
                continue

        # 3. benchmark
        if not os.path.exists(os.path.join(outdir, "benchmark_test.json")):
            print("  benchmarking ...", flush=True)
            if run([py, "scripts/benchmark_operator.py",
                    "--rundir", outdir, "--cachedir", cache], log):
                print("  BENCHMARK FAILED (see pipeline.log)", flush=True)
                summarize(args.runroot, args.elements)
                continue

        # 4. interpolation baselines
        if not os.path.exists(os.path.join(eldir, "baselines_test.json")):
            run([py, "scripts/baseline_interpolation.py",
                 "--cachedir", cache, "--outdir", eldir,
                 "--methods", *args.baseline_methods], log)

        summarize(args.runroot, args.elements)
        print(f"  element {z} done in {(time.time()-t0)/60:.0f} min "
              f"(total {(time.time()-t00)/3600:.1f} h)", flush=True)

    print("all elements processed; summary in "
          f"{args.runroot}/elements_summary.md", flush=True)


if __name__ == "__main__":
    main()
