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
sys.path.insert(0, REPO)

# Tier-1 recipe; --train_flags and the size preset append/override
TRAIN_FLAGS = ["--mode", "reweight", "--n_train", "0", "--pr_mix", "0.4",
               "--schedule", "wsd", "--steps", "100000",
               "--eval_every", "2000", "--tag", "reweight_full"]

# --size auto presets, keyed to spectral complexity. Elements with edges
# but no lines KEEP a high n_freqs (edges are sharp, high-frequency
# content) while shrinking the trunk; only truly smooth elements (H, He,
# whose recombination edges lie below the band) drop n_freqs. Step count
# is NOT set here: --steps in TRAIN_FLAGS is only an upper bound, and
# early stopping (on by default in train_adaptive) ends each element when
# its validation MRE plateaus -- so easy elements finish early on their
# own without a hand-tuned budget.
SIZE_PRESETS = {
    "standard": ["--hidden", "384", "--layers", "5", "--n_freqs", "512",
                 "--use_linehead", "auto"],
    "edged":    ["--hidden", "192", "--layers", "4", "--n_freqs", "384",
                 "--use_linehead", "off"],
    "smooth":   ["--hidden", "128", "--layers", "3", "--n_freqs", "128",
                 "--use_linehead", "off"],
}


def classify_size(cache, min_line_bins=32):
    """Pick a size preset from the element's spectral complexity: lines ->
    standard; edges but no lines -> edged (keep n_freqs high); neither ->
    smooth."""
    from spexai.train.train_operator import (SpectrumData, find_edge_bins,
                                             find_line_bins)
    data = SpectrumData(cache)
    n_lines = int((find_line_bins(data) >= 0).sum())
    n_edges = int(len(find_edge_bins(data)))
    name = ("standard" if n_lines >= min_line_bins
            else "edged" if n_edges > 0 else "smooth")
    return name, n_lines, n_edges


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
    ap.add_argument("--size", default="auto", choices=["auto", "fixed"],
                    help="auto: pick model size per element from spectral "
                         "complexity (smooth/edged/standard); fixed: use the "
                         "same TRAIN_FLAGS (plus --train_flags) for all")
    ap.add_argument("--train_flags", default="",
                    help="extra flags appended to the training command "
                         "(override the preset), e.g. \"--steps 200000\"")
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
        # create every output dir up front so no step depends on a
        # hand-made folder (outdir also creates its eldir parent)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(cache, exist_ok=True)
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
            # --skip-bad: scattered SPEX generation failures (~1% per
            # element) should not abort an unattended sweep; the count of
            # dropped files is recorded in the element's pipeline.log
            if run([py, "scripts/preprocess_spectra.py", "--skip-bad",
                    "--datadir", raw, "--outdir", cache], log):
                print("  PREPROCESS FAILED (see pipeline.log)", flush=True)
                summarize(args.runroot, args.elements)
                continue

        # 2. train
        if not os.path.exists(os.path.join(outdir,
                                           "reweight_full_history.json")):
            preset = []
            if args.size == "auto":
                name, nl, ne = classify_size(cache)
                preset = SIZE_PRESETS[name]
                print(f"  size={name} ({nl} lines, {ne} edges)", flush=True)
            print("  training ...", flush=True)
            # preset overrides TRAIN_FLAGS; user --train_flags overrides both
            cmd = [py, "-m", "spexai.train.train_adaptive",
                   *TRAIN_FLAGS, *preset, *args.train_flags.split(),
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
