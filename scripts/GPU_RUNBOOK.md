# Running the element 26 (iron) ablation study on a GPU machine

Everything below assumes a Linux box with an NVIDIA GPU and CUDA drivers.
The code auto-selects `cuda` when available (falls back to `mps`/`cpu`).

## 1. Copy code and data

Two things need to be transferred:

```bash
# on the GPU machine
git clone <your-spexai-remote> spexai && cd spexai
git checkout fix_small_bugs   # or whichever branch has scripts/ + spexai/train/operator.py
```

The preprocessed data cache (1.4 GB, much cheaper to copy than the ~21 GB of
raw text files):

```bash
# from your Mac
rsync -avP /Users/danielahuppenkothen/work/data/spexai/processed/element26/ \
      <gpu-host>:~/spexai_data/processed/element26/
```

(Alternatively copy the raw `element_26/` directory and rerun
`python scripts/preprocess_spectra.py --datadir ... --outdir ...` there.)

## 2. Environment

```bash
# python 3.11-3.13 are zero-risk; 3.14 works with torch >= 2.10 but verify
# the CUDA (not CPU-only) wheel gets installed -- see check below
conda create -n spexai-ml python=3.12 -y
conda activate spexai-ml
pip install torch numpy pandas scipy scikit-learn matplotlib
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

No `pip install -e .` needed as long as you run from the repo root.

## 3. Run the ablation (8 variants)

```bash
cd spexai
CACHE=/home/dhuppenkot2/data/spexai_data/processed/element26
RUNS=/home/dhuppenkot2/data/spexai_data/runs/element26
HPO=/home/dhuppenkot2/data/spexai_data/runs

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

## 3b. Hyperparameter search on the winning combination

The `combo` variant is the ablation winner: line head on, Sobolev off
(trend head switchable via `--use_trend 0/1`). Two-stage random search:

```bash
mkdir -p $RUNS/hpo
nohup python scripts/hpo_combo.py \
    --trials 24 --stage1_steps 3000 --stage2_steps 20000 --top 4 \
    --cachedir $CACHE --outdir $RUNS/hpo \
    > $RUNS/hpo/hpo.log 2>&1 &
```

- Stage 1 (24 x 3000 steps) takes roughly as long as ~4 full runs; stage 2
  retrains the top 4 configs at 20k steps. Total on the order of a day.
- Fully resumable: rerun the same command after an interruption and it
  skips finished trials (state in `hpo_results.json`).
- `python scripts/hpo_combo.py --report --outdir $RUNS/hpo` prints the
  ranking table at any time (`hpo_results.md`).
- The search covers: lr, batch, points/spectrum, trunk width x depth,
  activation (gelu / silu / sine), Fourier n_freqs & f_max, line-embedding
  size, trend head on/off, log-space stabiliser, curriculum length.
- Benchmark the finished trials like any checkpoint:
  `python scripts/benchmark_operator.py --rundir $RUNS/hpo --cachedir $CACHE`

## 3c. Broadened (T, v) emulator + broadening comparison

Train the option-2 emulator of velocity-broadened spectra (targets are
generated on the fly from the original spectra; the first run builds a
~4.4 GB uniform-grid flux cache next to the preprocessed data). The
script defaults are the t04 HPO winner (hidden 384 x 5 layers, n_freqs
512, f_max 4000, lr 3e-3, batch 128, 2048 points/spectrum), so no
hyperparameter flags are needed:

```bash
nohup python -m spexai.train.train_broadened \
    --steps 20000 \
    --cachedir $CACHE --outdir $RUNS/broadened \
    > $RUNS/broadened.log 2>&1 &
```

Then compare all broadening options (exact erf reference vs the current
sparse-matrix implementation, FFT convolution, the (T,v) emulator, and
the hybrid analytic-line scheme; the emulator methods are picked up
automatically if their checkpoints exist):

```bash
python scripts/benchmark_broadening.py --cachedir $CACHE --rundir $RUNS \
    --nspec 32
```

Results land in `$RUNS/benchmark_broadening.json`.

## 3d. Adaptive-data test run (unbroadened emulator)

Three-arm experiment on a subsampled training grid: does dynamically
generating extra training spectra (per-bin PCHIP interpolation, gated by
leave-one-out interpolation error) beat plain training and
error-prioritized sampling of the existing grid? Model is the t04 HPO
winner (script defaults); validation is always the frozen SPEX val split.

```bash
mkdir -p $RUNS/adaptive
for mode in baseline reweight adaptive; do
  nohup python -m spexai.train.train_adaptive \
      --mode $mode --n_train 300 \
      --cachedir $CACHE --outdir $RUNS/adaptive \
      > $RUNS/adaptive/$mode.log 2>&1 &
  wait
done
```

- Each `<mode>_history.json` records the acquired temperatures and the
  trust-gate-rejected intervals (the shortlist for real new SPEX runs).
- Interpretation: adaptive > reweight means the synthetic data genuinely
  adds information; reweight > baseline means it was attention, not data.
- Checkpoints benchmark like any other:
  `python scripts/benchmark_operator.py --rundir $RUNS/adaptive --cachedir $CACHE`
- Repeat with `--n_train 100/1000` for the data-efficiency version
  (`--tag <mode>_n100` to keep the outputs apart).

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
rsync -avP dhuppenkot2@spexaitrain.spexcalculation.src.surf-hosted.nl:/home/dhuppenkot2/data/spexai_data/runs/element26/ \
      /Users/danielahuppenkothen/work/data/spexai/runs/element26/
```

Everything needed for analysis (checkpoints, histories, metrics, figures)
lives in that one directory.
