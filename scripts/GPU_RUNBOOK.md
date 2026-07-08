# Running the element30 ablation study on a GPU machine

Everything below assumes a Linux box with an NVIDIA GPU and CUDA drivers.
The code auto-selects `cuda` when available (falls back to `mps`/`cpu`).

## 1. Copy code and data

Two things need to be transferred:

```bash
# on the GPU machine
git clone <your-spexai-remote> spexai && cd spexai
git checkout fix_small_bugs   # or whichever branch has scripts/ + spexai/train/operator.py
```

The preprocessed data cache (1.4 GB, much cheaper to copy than the 21 GB of
raw text files):

```bash
# from your Mac
rsync -avP /Users/danielahuppenkothen/work/data/spexai/processed/element30/ \
      <gpu-host>:~/spexai_data/processed/element30/
```

(Alternatively copy the raw `element30/` directory and rerun
`python scripts/preprocess_element30.py --datadir ... --outdir ...` there.)

## 2. Environment

```bash
conda create -n spexai-ml python=3.11 -y
conda activate spexai-ml
pip install torch numpy pandas scipy scikit-learn matplotlib
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

No `pip install -e .` needed as long as you run from the repo root.

## 3. Run the ablation (8 variants)

```bash
cd spexai
CACHE=~/spexai_data/processed/element30
RUNS=~/spexai_data/runs/element30

nohup python scripts/run_ablation.py \
    --steps 20000 --lr 1e-3 \
    --cachedir $CACHE --outdir $RUNS \
    > $RUNS/ablation.log 2>&1 &
tail -f $RUNS/ablation.log
```

Notes:
- Variants run sequentially: `base, no_sobolev, no_trend, no_film,
  no_fourier, fixed_grid, hash_grid, line_head`.
- On an A100/4090-class GPU expect very roughly 30–60 min per variant at
  20k steps (it was ~1 s/step on Apple MPS; CUDA should be several times
  faster). Use `--steps 6000` for a quick first pass.
- `--lr 1e-3` was validated in a 2000-step probe on the full data
  (val mean-relative-error 0.11 and falling); the default 3e-4 learns
  noticeably slower.
- Each variant writes `<variant>.pt` (best-on-val checkpoint) and
  `<variant>_history.json`; the driver writes `ablation_summary.{json,md}`.
- If a run dies partway, rerun with e.g.
  `--variants no_film fixed_grid hash_grid line_head` to resume the rest;
  the summary table only needs the finished variants' history files.

## 4. Benchmark on the held-out test set

```bash
python scripts/benchmark_operator.py --rundir $RUNS --cachedir $CACHE
python scripts/baseline_interpolation.py --cachedir $CACHE --outdir $RUNS
# data-efficiency curve for the classical baselines:
for n in 100 300 1000 3000; do
  python scripts/baseline_interpolation.py --cachedir $CACHE --outdir $RUNS --n-train $n
done
```

This produces `benchmark_test.{json,md}` (overall + line vs continuum
metrics, timing, per-variant residual plots in `figures/`) and
`baselines_test*.json`.

## 5. Copy results back

```bash
# from your Mac
rsync -avP <gpu-host>:~/spexai_data/runs/element30/ \
      /Users/danielahuppenkothen/work/data/spexai/runs/element30/
```

Everything needed for analysis (checkpoints, histories, metrics, figures)
lives in that one directory.
