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

The revised option-2 model (Gaussian line head with analytic sigma(v),
line-aware point sampling, full-velocity-range validation, wing masking
-- see the train_broadened2 docstring) trains the same way:

```bash
nohup python -m spexai.train.train_broadened2 \
    --steps 20000 \
    --cachedir $CACHE --outdir $RUNS/broadened2 \
    > $RUNS/broadened2.log 2>&1 &
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
nohup bash -c 'for mode in baseline reweight adaptive; do
  python -m spexai.train.train_adaptive \
      --mode $mode --n_train 300 \
      --cachedir '"$CACHE"' --outdir '"$RUNS"'/adaptive \
      > '"$RUNS"'/adaptive/$mode.log 2>&1
done' > /dev/null 2>&1 &
```

(The loop itself must live inside the detached process: with a plain
`for ... do nohup ... & wait; done` the running arm survives an SSH
drop but the loop dies with the login shell and the remaining arms
never start. Arms are independent -- if the sequence is interrupted,
relaunch only the modes without a `<mode>_history.json`.)

- Each `<mode>_history.json` records the acquired temperatures and the
  trust-gate-rejected intervals (the shortlist for real new SPEX runs).
- Interpretation: adaptive > reweight means the synthetic data genuinely
  adds information; reweight > baseline means it was attention, not data.
- Checkpoints benchmark like any other:
  `python scripts/benchmark_operator.py --rundir $RUNS/adaptive --cachedir $CACHE`
- Repeat with `--n_train 100/1000` for the data-efficiency version
  (`--tag <mode>_n100` to keep the outputs apart).

## 3e. Tier-1 optimisation run (full grid)

Best-known recipe for pushing the full-grid emulator below t04_long's
0.31% MRE: t04 architecture + error-prioritized sampling of the training
grid + Polyak weight averaging (on by default, `--ema_decay 0.999`) +
a 5x longer cosine schedule with a floor (`--lr_min_frac`):

```bash
nohup python -m spexai.train.train_adaptive \
    --mode reweight --n_train 0 --pr_mix 0.3 \
    --steps 100000 --eval_every 2000 --tag reweight_full \
    --cachedir $CACHE --outdir $RUNS/tier1 \
    > $RUNS/tier1.log 2>&1 &
```

Every training run (train_operator and train_adaptive) now also writes
standard diagnostics to `<outdir>/figures/`: `<tag>_history.png`
(training loss, train-vs-val MRE, val yield) and three
`<tag>_spectra_T*.png` (SPEX vs emulator with residuals, full band +
three line zooms, at low/mid/high test temperatures). For older
checkpoints or comparison experiments, generate the same figures with
`python scripts/plot_model_diagnostics.py --ckpt <file>.pt --cachedir $CACHE`.
Note: with EMA enabled, early evals (first ~3k steps) lag the live
weights; judge convergence from step ~5k onward.

## 3f. Tier-2 capacity runs (full grid)

Tier 1 ended with train MRE == val MRE (0.15%): capacity-limited, not
data-limited. Two capacity hypotheses, both on the Tier-1 recipe
(reweighted sampling, EMA, `--pr_mix 0.4` to protect the faint low-T
band the gratings exposed). The `wsd` schedule holds the LR flat and
only anneals over the last 15% -- if a run is still improving, restart
it with a larger `--steps` and it re-pays only the decay leg:

```bash
# (a) bigger trunk at a stable LR (the HPO never gave large models a
#     fair run: t16 diverged at lr 3e-3)
nohup python -m spexai.train.train_adaptive \
    --mode reweight --n_train 0 --pr_mix 0.4 \
    --hidden 512 --layers 6 --n_freqs 1024 --lr 1e-3 \
    --schedule wsd --steps 100000 --eval_every 2000 --tag big_trunk \
    --cachedir $CACHE --outdir $RUNS/tier2 > $RUNS/tier2_a.log 2>&1 &

# (b) line-head capacity: wider embeddings + Fourier-embedded T
#     conditioning (per-line emissivity curves have sharp ionisation
#     features), trunk unchanged from t04
nohup python -m spexai.train.train_adaptive \
    --mode reweight --n_train 0 --pr_mix 0.4 \
    --line_dim 48 --line_hidden 256 --line_t_freqs 32 \
    --schedule wsd --steps 100000 --eval_every 2000 --tag line_heavy \
    --cachedir $CACHE --outdir $RUNS/tier2 > $RUNS/tier2_b.log 2>&1 &
```

(~1.9M params for (b), ~3.3M for (a); run sequentially if GPU memory is
tight.) Compare against `$RUNS/tier1/reweight_full` (0.16% test MRE,
99.0% yield@1%) with `benchmark_operator.py --rundir $RUNS/tier2`; check
the low-T grating regression specifically with
`benchmark_instruments.py --linehead_ckpt $RUNS/tier2/<tag>.pt`.

**Verdict (2026-07-13): big_trunk won** - 0.13% test MRE, 99.2%/52.6%
yield@1%/@0.1%, and it fixes the Tier-1 grating regression (HEG/MEG RMF
means 0.20%/0.09%, better than t04_long). line_heavy regressed (0.21%)
and is rejected: line error is not amplitude-head capacity. Use
big_trunk (`--hidden 512 --layers 6 --n_freqs 1024 --lr 1e-3`) as the
backbone where accuracy matters (21.5 ms/spec, 1.7x t04); keep the t04
architecture for cheap sweeps. Scaling is ~25% MRE per parameter
doubling and both runs were still improving at 100k steps - the cheap
next move is the same command with `--steps 200000` (WSD re-pays only
the decay leg).

## 3g. All-element sweep

Trains the current best recipe (Tier 1: reweighted sampling + EMA + WSD,
t04 architecture) for every element sequentially, then benchmarks each on
its held-out test set and runs the interpolation baselines. Resumable:
finished stages are detected by their outputs, so rerun the same command
after any interruption. Expects raw data in `<dataroot>/element_<Z>/`
(`Z<Z>_*keV.txt`); preprocessing runs automatically on first use.

```bash
nohup python scripts/run_all_elements.py \
    --dataroot ~/data/spexai_data --runroot $RUNS_ROOT \
    > $RUNS_ROOT/all_elements.log 2>&1 &
```

- Per-element outputs land in `<runroot>/element<Z>/tier1/` (checkpoint,
  history, diagnostics figures, `benchmark_test.{json,md}`); the
  cross-element table is rebuilt after every element at
  `<runroot>/elements_summary.{json,md}`.
- A failing element is recorded in the summary and skipped (check its
  `pipeline.log`).
- **Per-element sizing (`--size auto`, default).** Each element is probed
  for spectral complexity and given an *architecture* preset: `standard`
  (has lines -> full t04 model + line head), `edged` (no lines but in-band
  recombination edges, e.g. Li/Be/B -> smaller trunk but **n_freqs kept
  at 384**, since edges are sharp high-frequency features; line head off),
  or `smooth` (H/He, edges below band -> small trunk, n_freqs 128, line
  head off). `--size fixed` uses one config for all.
- **Early stopping governs the step count** (not the preset). `--steps`
  (100k) is only an upper bound; training monitors the smoothed
  validation MRE and, once it plateaus, triggers the WSD decay early and
  finishes. So each element runs exactly as long as it needs -- H stops
  itself in ~25k steps, iron runs longer -- with no hand-tuned budgets.
  The log line reports both `yield1%` and `yield0.1%`; checkpoints are
  selected on lowest val MRE.
- **Edge-aware training** is automatic: recombination edges are detected
  empirically (`find_edge_bins`) and ~15% of loss points are drawn from
  edge regions so the sharp RRC steps are supervised. Off for edge-free
  elements. Every run also writes `<tag>_edges.png` (each edge shown at
  its own peak temperature) and logs the edge/overall MRE ratio.
- When a Tier-2 config wins, update `SIZE_PRESETS["standard"]` in the
  script or pass `--train_flags "..."` (overrides the preset).
- Early stopping + auto sizing make the sweep much cheaper than a uniform
  100k-step pass. Use `--elements 8 14 26 ...` to prioritise.

## 3h. Training-recipe screening program

Screens the candidate improvements from the technical report's related-work
section (InfoBatch-style unbiased sampling, optimizer bake-off, muP width
transfer, FINER activations) in 14 short runs at t04 architecture. Needs
`pip install schedulefree` for the Schedule-Free arm (that arm errors out
otherwise; the rest are unaffected). Selection is on validation and a
shared fresh off-grid PCHIP probe -- the Fe test set is never touched.

```bash
mkdir -p $RUNS/recipe
nohup python scripts/run_recipe_program.py \
    --cachedir $CACHE --progdir $RUNS/recipe --steps 20000 \
    > $RUNS/recipe/program.log 2>&1 &
```

- Resumable (per-arm; skips arms whose history exists). ~20k steps/arm
  on the full grid, so ~1--2 GPU-days total.
- Live table at `$RUNS/recipe/recipe_summary.md` (val and off-grid MRE
  per arm), rebuilt after every arm.
- Compose winners by hand and confirm once at big_trunk scale on a
  DIFFERENT element (e.g. Z=8 or 14), not the Fe test set -- this both
  avoids test reuse and checks cross-element transfer.
- Interpretation: exp1 (ib_*) -- does the grating regression vanish with
  `pr_correct 1`? Rerun `benchmark_instruments.py` on the two winners to
  check. exp2 (opt_*) -- lowest val MRE at equal budget. exp3 (mup_w*) --
  does w768 stay stable and beat w384? exp4 (fin_*) -- does FINER match
  or beat gelu, and does it remove the need for the curriculum?

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

Emulator accuracy on instrument energy grids (XRISM Resolve, Chandra
ACIS, HETG/HEG and MEG; resolution-matched binning, truth = SPEX rebinned
to the same grid, erf-broadened for the (T,v) emulators):

```bash
python scripts/benchmark_instruments.py --cachedir $CACHE --rundir $RUNS \
    --nspec 16 --linehead_ckpt $RUNS/hpo/t04_long.pt \
    --responses_dir <dir with RMFs>
```

The RMF-folded pass needs the instrument response files (see the script
docstring for sources); rsync `~/work/data/spexai/responses/` from the
Mac, or let the pass skip if they are absent. Writes
`benchmark_instruments_test.{json,md}`.

## 5. Copy results back

```bash
# from your Mac
rsync -avP dhuppenkot2@spexaitrain.spexcalculation.src.surf-hosted.nl:/home/dhuppenkot2/data/spexai_data/runs/element26/ \
      /Users/danielahuppenkothen/work/data/spexai/runs/element26/
```

Everything needed for analysis (checkpoints, histories, metrics, figures)
lives in that one directory.
