# Inference & evaluation infrastructure — status & resume notes

Built the evaluation, simulation, inference, temperature-distribution, Galactic
absorption, and bias-testing infrastructure on top of the operator emulator, in
preparation for the week-long 30-element production run. All new code is on the
current **operator stack** and iterates the model-store manifest, so it scales
from the present 16 elements to the full 30 automatically. **64 tests pass.**
No commits were made — the working tree is yours to inspect.

**Read next / resume from:** this file (status + what's left), the approved
design `~/.claude/plans/snazzy-finding-matsumoto.md`, the methodology report
`docs/inference_methodology.tex`, the how-to `docs/tutorial_inference.md`
(incl. §5 cluster runs), and `tutorials/inference_walkthrough.ipynb`.

## What was built

| Area | Files | Purpose |
|---|---|---|
| **Evaluation** | `spexai/train/metrics.py`, `spexai/eval.py` | Shared held-out metrics (consolidated 3× dup); `evaluate_manifest` (whole-store table, CLI `python -m spexai.eval`) + `evaluate_joint` (emulator vs truth, flux & channel space). |
| **SPEX ground truth** | `spexai/inference/spex_truth.py` | `SpexTruthModel` — independent PCHIP-over-SPEX + exact broadening, mirrors `JointOperatorModel`'s API. |
| **Abundances** | `spexai/inference/abundances.py` | `AbundanceModel` — global metallicity, free elements, tie-to-fraction-of-iron. |
| **DEM** | `spexai/inference/tempdist.py` | `TempGrid`, Gaussian(logT/T), two-Gaussian, lognormal (scipy), non-parametric `BinnedDEM`; `predict_counts_dem` on both models. |
| **Absorption** | `spexai/inference/absorption.py`, `scripts/data/build_tbabs_table.py`, `spexai/inference/data/tbabs_sigma.npz` | Screen on the native fine grid pre-rebin (observed frame); tbabs (cached from XSPEC) + wabs fallback; `Absorption.default()`. |
| **Normalisation** | `spexai/inference/units.py`, `scripts/inference/validate_spex_norm.py` | Physical `norm` = emission measure Y (1e64 m⁻³) + `luminosity_distance`; m²→cm² factor; validated vs SPEX. |
| **Bias study** | `scripts/inference/bias_study.py` | Staged SBC/pulls: point bias + coverage, then rank statistics; SPEX-truth inject with emulator-self control. |
| **Perseus showcase** | `scripts/inference/perseus_showcase.py` | Single-T & Gaussian-DEM injection-recovery at Perseus's physical distance; reports the emission measure. |
| **Inference glue** | `spexai/inference/{fitting,simulate,operator_model}.py` | `make_loglike`/`run_emcee`/`run_ultranest` gained `abundance_model=`, `dem=`, `absorption=`, `luminosity_distance` (fixed); simulate is absorption/distance-aware; fixed a `run_ultranest` `resume=None` crash. |
| **Wiring** | `spexai/__init__.py`, `spexai/inference/__init__.py`, legacy modules | `import spexai` → operator stack; legacy CNN modules emit `DeprecationWarning`. |
| **Docs/tests** | `docs/*`, `tutorials/*`, `tests/test_{metrics,spex_truth,dem,absorption,units}.py` | Report, tutorial, notebook, unit + real-data-gated tests. |

## Key decisions (as agreed)

- **SPEX truth** = PCHIP over the per-element caches + **exact**
  `direct_broadening_matrix`, independent of the emulator's FFT hybrid path.
- **Absorption** on the native fine grid **before** rebinning (continuum) and at
  exact line energies (lines), observed frame → resolution-independent, no
  retraining. tbabs (cached from XSPEC) + wabs fallback. **Background out of
  scope** (left to sherpa/NDSpec).
- **Normalisation** = SPEX emission measure Y (1e64 m⁻³) + explicit luminosity
  distance (SPEX convention); D fixed (degenerate with Y), Y fitted.
- **DEM** shapes from `scipy.stats` + a free binned option (no regulariser).
  **Abundance tying** reproduces the thesis scheme (global Z + free ratios).
- **Bias** via SBC + pulls, staged; literature-grounded ranges + Perseus showcase.

## Validation status (all green)

- `evaluate_manifest` reproduces the manifest's per-element test MRE exactly.
- Emulator vs independent SPEX truth agree to ~0.1–0.7% in flux (Fe).
- **Absolute normalisation validated against a real SPEX install** (SPEXACT 2.07,
  native Apple-Silicon conda): emulator absolute flux matches SPEX `cie` at
  T=4 keV to ~2.5% across 3–8 keV (→5% at 8 keV from the missing elements). This
  confirms flux units (ph/s/m²/bin), Y_ref=1, D_ref=1e22 m, and the 1e-4
  m²→cm² factor — no order-of-magnitude error. (`scripts/inference/validate_spex_norm.py`.)
- Absorption: n_h=0 is an exact no-op; tbabs suppresses the soft band correctly.
- Perseus single-T & DEM smokes recover all injected params within ~1σ; the
  single-T run reports a physical **Y ≈ 4×10⁷² m⁻³ at 75 Mpc** — consistent with
  a ~44 kpc emitting region at Perseus core densities.

## Outstanding / to resume here

**Pending the production run (not blocking — harness auto-scales):**
- The **30-element training run hasn't happened**. When it lands: re-run
  `python -m spexai.eval`; re-run `validate_spex_norm.py` (the ~2.5→5% absolute
  offset should shrink); Na (11) & Al (13), currently flagged undertrained, get
  replaced; 14 missing elements (16–25, 27–30) fill in.
- **Only smoke-tested, not run at scale.** SBC bias study and Perseus have run at
  tiny scale (short chains, 1–2 elements, ACIS). Real runs — `--elements all`,
  long chains, XRISM/Resolve, hundreds of sims — are cluster jobs
  (`docs/tutorial_inference.md` §5).

**Optional builds (not yet done):**
- **Systematic DEM injection-recovery grid** (sweep several mean-T × width ×
  abundance points). Perseus DEM mode does one point only.
- **Vectorise `run_ultranest`** (it loops the likelihood → slow); worth it before
  heavy nested sampling on the cluster.
- **Multi-instrument joint-fit example** — now *correct* (consistent physical Y
  across instruments) but no driver/demo exists.

**Data / responses:**
- **XRISM/Resolve response IS present** as `rsl_Hp_L_2025.rmf` in
  `~/work/data/spexai/responses/` (60000×60000, 0–30 keV); `find_response()` now
  matches the `rsl_*` name. **But there is no Resolve ARF** there — fits work for
  *parameter recovery*, but the absolute emission measure needs the real
  `rsl_*.arf` (drop it in that dir; the showcase auto-uses it).
- Commit or `.gitignore` the new data assets: `spexai/inference/data/tbabs_sigma.npz`
  (~130 KB) and `spexai/models/`.

**Docs housekeeping:** `inference_methodology.tex` is a source file (not compiled
to PDF here); the notebook is ready-to-run but was validated via a proxy script,
not executed with saved outputs.

## Environment gotchas (also in memory: `macos-omp-and-xspec`)

- Prefix torch scripts with `KMP_DUPLICATE_LIB_OK=TRUE` on macOS (OpenMP crash).
- **XSPEC/tbabs:** conda-forge has no XSPEC for macOS; HEASoft installs via the
  **HEASARC conda channel** (used in the `heasoft-test` env) — that built
  `tbabs_sigma.npz` (PyXspec, `abund wilm`).
- **SPEX:** installs natively (Apple Silicon) from `https://var.sron.nl/spexconda`
  (env `spex`). It links X11 (`conda install -c conda-forge xorg-libxext …`), and
  `conda run` strips `DYLD_*`, so `conda activate spex` +
  `export DYLD_FALLBACK_LIBRARY_PATH=$CONDA_PREFIX/lib` and run the env python
  directly. Used for `validate_spex_norm.py --mode spex`.

## Changed/added files

**New:** `spexai/eval.py`, `spexai/train/metrics.py`,
`spexai/inference/{spex_truth,abundances,tempdist,absorption,units}.py`,
`spexai/inference/data/tbabs_sigma.npz`,
`scripts/{bias_study,perseus_showcase,build_tbabs_table,validate_spex_norm}.py`,
`tests/test_{metrics,spex_truth,dem,absorption,units}.py`,
`docs/{inference_methodology.tex,tutorial_inference.md,session_summary.md}`,
`tutorials/inference_walkthrough.ipynb`.

**Modified:** `spexai/__init__.py`, `spexai/inference/__init__.py`,
`spexai/inference/{fitting,simulate,operator_model}.py`,
`spexai/inference/{fit,fit_old,model,model_old,write_tensors}.py` (deprecation),
`scripts/{benchmark_operator,baseline_interpolation}.py`.

---

# Forward-step optimisation campaign (2026-08-14 → 08-17)

Goal: make one vectorised MCMC forward step cheap enough for a 30-element SBC
campaign. Measured on the cluster GPU (22 GiB), full 30-element store,
`nwalkers=96`, via `scripts/inference/benchmark_inference.py`.

## Result: 2.19x over the production path, and the ladder is now closed

Full ladder, 30 elements, 96 walkers, real RMF fold, after the recompile-limit
fix (bug 4 below); accuracy is vs the float64 reference on bins >= 0.1% of peak:

| config | ms/walker | vs fp64 | vs serial-accel | max rel | p99 |
|---|---|---|---|---|---|
| serial-fp64 | 411.7 | 1.00x | 0.60x | reference | — |
| serial-accel (TF32 + compile + fp32 FFT) | 246.2 | 1.67x | 1.00x | 9.68e-04 | 2.83e-04 |
| batched-accel (grouped vmap) | 208.6 | 1.97x | 1.18x | 6.24e-04 | 2.35e-04 |
| batched-compile (`compile ∘ vmap`) | 112.7 | 3.65x | **2.19x** | 6.22e-04 | 2.35e-04 |

**`serial-accel` is the current production path** (`run_cluster.sh`), so 2.19x is
the real margin for switching production to the batched forward. The grouped
vmap alone is worth only 1.18x; compiling the grouped GEMMs is what pays. The
compile step is numerically free (6.24e-04 -> 6.22e-04), and every config sits
well below the emulator's own ~3e-3 MRE with worst bins at 0.8-1.6% of peak.

Caveat: `serial-accel` was never measured *before* the recompile-limit fix, so
the fix's contribution is unquantified. Its 1.67x is below the ~3.0x cumulative
quoted in the methodology report for the same three knobs -- but that figure was
measured on the hot-floor `EnsembleForward` (store28, single-T, chunk 32), not
this serial loop, so the two are not directly comparable.

Stage split at B=16: **trunk 95%**, broaden+rebin 2%, lines+combine 2%. Every
remaining option is therefore Amdahl-bound by the trunk.

`torch.compile` of the batched trunk is a genuine win because
`_TrunkWrap.forward` deliberately calls `SpectralOperator.forward_norm` as an
unbound class method to bypass the compiled instance attribute — so
`enable_inference_acceleration`'s per-element compile never reached the batched
path. `batched-accel` had only TF32 + float32 FFT.

`--wchunk` 16 → 48 changed nothing (111.3 → 109.7): the GPU is already
saturated at 16.

## Four optimisations tried and rejected

**bf16 autocast on the trunk — rejected on accuracy.** 1.69x (113.2 → 66.9
ms/walker), but max relative error 2.4% on a bin at 97.5% of peak, and
p99 = 8.7e-3 ≈ 3x the emulator's own ~3e-3 MRE. Not acceptable for SBC, where a
forward-model systematic invalidates the calibration it is meant to establish.

**Trunk energy-grid coarsening — rejected; the premise is false.** The trunk is
**not** a smooth continuum. One-bin curvature |d² log10 density| on the native
P=24212 grid (T=2 keV):

| Z | max | 99.9 pct |
|---|---|---|
| 2 (He) | 0.0016 | 0.0010 |
| 8 (O) | 1.75 | 0.78 |
| 14 (Si) | 2.45 | 1.21 |
| 26 (Fe) | 4.91 | 1.76 |

Only H/He are smooth; every metal carries line structure in the *trunk* — and
Si/Fe have line heads and still show it. Stride-2 subsampling costs Fe ~10x
relative error in density and **43.5%** in broadened flux, worst bin at 70% of
peak. Broadening does not wash it out: real narrow lines are lost, not noise.
This also rules out the "precompute a dense T-grid and interpolate" idea, which
rests on the same false premise. Coarsening only H/He is ~2/30 of the trunk.

**CUDA graphs — rejected on evidence.** The `--detailed` profile shows
`Command Buffer Full` at 76% of self-CPU across 11,966 calls, i.e. the CPU is
blocked because it has run ahead and filled the GPU queue. Self CPU (25.905s) ≈
self CUDA (25.934s). Launch overhead is not the constraint.

**Larger walker batch — no effect** (see `--wchunk` above).

## Where the time actually goes (`batched-compile`, self CUDA 25.93s)

| work | self CUDA | share |
|---|---|---|
| `aten::bmm` (incl. `cutlass_80_tensorop_s1688gemm` 15.73s) | 16.49s | 63.6% |
| `triton_poi_fused_add_gelu_mul_*` (activations) | ~4.6s | ~17.8% |
| `triton_poi_fused_bmm_0/1` | ~2.7s | ~10.4% |

`s1688gemm` is the **TF32** tensor-core kernel, so the GEMMs already use tensor
cores. The trunk is GEMM-bound at TF32, and anything further must preserve
per-bin structure exactly.

## Bugs found and fixed along the way

1. **CUDA OOM in `--stages`** (21.34 GiB genuinely allocated). The staged
   refactor moved trunk code out of `flux()`, whose `@torch.no_grad()` was the
   only one — `stack_module_state` preserves `requires_grad`, so calling
   `_density`/`_continuum`/`_combine` directly built and pinned a full autograd
   graph. Fixed: `@torch.no_grad()` on each stage, plus a `requires_grad` guard
   in the tests.
2. **`torch.compile(vmap, dynamic=True)` crashes**: symbolic shapes make
   functorch's `BatchedTensor` fail `sizes()` inside `linear`. Fixed with an
   explicit `dynamic=False` (the default `dynamic=None` auto-promotes to
   symbolic on the second shape and hits the same crash), and `_echunk` now
   spreads bins evenly so static shapes stay down to one or two graphs.
3. **Silent compile fallback**: all `_TrunkGroup`s compile the *same* functorch
   `wrapped` code object and so share one dynamo cache bucket; 4 groups × chunk
   shapes hits the default `recompile_limit` of 8, past which dynamo falls back
   to eager and the 1.88x silently disappears. Fixed by
   `_ensure_recompile_limit`.
4. **The recompile-limit trap also hit the serial production path.** Dynamo
   caches per *code object*, so `enable_inference_acceleration`'s loop —
   `torch.compile(m.forward_norm)` once per element, all 30 sharing
   `SpectralOperator.forward_norm` — exhausted the default budget of 8 and
   dynamo then suppressed compilation for the whole code object, leaving ~22 of
   30 elements running eager. This is the documented factory-pattern trap
   ("if you dynamically create multiple copies of a function, they will all
   share the same code cache"). Fixed by calling
   `operator_model.ensure_recompile_limit(8 * n_models)` before either compile
   loop (batched groups and per-element), and the same guard was added to
   `gpu_forward.py`'s own loop.

   **Consequence for earlier numbers:** every serial-path timing taken before
   this fix — including the `serial-accel` rung and the "~2.6x from
   torch.compile" figure quoted in the methodology report — was measured with
   most elements uncompiled. Both should be re-measured; the serial baseline may
   improve, narrowing the batched path's margin.

5. **Broken accuracy metric** (mine): a median over all bins reads 0.0 on any
   RMF grid because most bins are exactly zero in both arrays, and a
   peak-clamped max cannot distinguish a bright bin from an empty one.
   `rel_errors` now masks to bins ≥ `--acc_sig_frac` of peak and reports max,
   p99, median, the worst bin's energy, and its brightness relative to peak.

## Capability added: per-walker sigma_v

`deposit_gaussian_lines` accepted only a scalar velocity, which is what pinned
`sigma_v` in the Tier-1 MCMC runs. (`fft_broaden` already supported `(B,)`.)
It now takes scalar *or* `(B,)`: per-walker widths make each line's ±nsigma
window ragged across the batch, which `index_add_(1, …)` cannot express, so that
branch scatters into a flat `(B*M,)` buffer with walker-offset indices. The
scalar branch is unchanged and still bit-identical.

`BatchedJointForward._continuum` tiles the velocity to match its element-major
`(N*B, K)` row order — without that a `(B,)` velocity would silently broadcast
against the element axis.

**Still open:** no sampler exposes `sigma_v` yet. `run_cluster.sh` /
`mcmc_check.py` remain Tier-1 (velocity fixed at truth); freeing it needs the
parameter and a prior added to the fitting glue.

## Recommendation

The forward is done for now: 2.19x over production banked, every other
kernel-level lever closed on measurement. At **6.0 h/chain** and **~1200 GPU-h
for 200 SBC sims**, the
productive direction is fewer forwards — fewer sims, shorter chains, or better
sampler efficiency — not a faster one.

**Next action:** port the production forward (`EnsembleForward`, used by
`run_cluster.sh`) from the per-element serial loop onto `BatchedJointForward`
with `compile_trunk=True` -- measured at 2.19x with no accuracy cost. The
batched path already supports the per-walker `sigma_v` and `n_h` that
`EnsembleForward` needs. Still unmeasured: the one-off compile stall (`timeit`
warms up before timing).

> **DONE 2026-08-18** — see the next section. The port landed, and the compile
> stall is now instrumented in `mcmc_check.py` (printed as
> `forward: first call Xs, warm Ys`) but still needs one GPU run to record.

---

# Sampler bake-off infrastructure (2026-08-18 → 08-19)

Goal: put the emulator through the SBC/evaluation machinery with several
samplers, so the campaign can be run with whichever is actually affordable.
Everything below is committed (through `4525dfd`); **142 tests pass**. The
bake-off itself has NOT been run — that is the next action.

## Read next / resume from

This section, then `scripts/inference/bake_off.py --help`. The run procedure is at the
bottom. Design decisions that were explicitly agreed with the user are marked
**[agreed]** — do not silently revisit them.

## What was built

| Area | Files | Purpose |
|---|---|---|
| **Vectorised forward** | `spexai/inference/vector_forward.py` | `VectorForward`: `EnsembleForward` promoted out of `inference_demo/hot_floor` into the package. Redshift, distance, abundance scheme, parameter names and exposure are all injected instead of hardcoded, so one class serves the hot-floor check, the bake-off and SBC. |
| **Shared posterior** | `spexai/inference/posterior.py` | `BoxPrior` + `PoissonPosterior`: one likelihood for every sampler. `logp` (numpy, batched) for the gradient-free ones, `ptform` for nested sampling, `potential`/`potential_and_grad` in unconstrained space for NUTS/VI. |
| **Probabilistic model** | `spexai/inference/ppl.py` | `SpectrumModel`: the same density as a Pyro program, priors declared as distributions. Written batch-first so `vectorize_particles=True` sends `num_particles` to the forward as one batched call. |
| **Samplers** | `spexai/inference/samplers.py` | `run_emcee`, `run_zeus`, `run_ultranest`, `run_nuts` (Pyro), `run_svi` (Pyro, full-rank Gaussian). All return `SamplerResult` with ESS and `ess_per_eval`. |
| **Driver** | `scripts/inference/bake_off.py` | One sampler per invocation, own result file, `--summarise` builds the table. Scores wall-clock, evals, ESS/eval, recovery vs truth, agreement vs a reference chain, log Z. |
| **DEM batching** | `spexai/inference/tempdist.py` | `weights_batch(params)`: `{name: (B,)}` → `(B, G)`, pure torch and differentiable, alongside the scalar scipy `weights`. Implemented for the Gaussian presets, `TwoGaussianDEM` and `BinnedDEM`. |
| **Refactor** | `spexai/inference/fitting.py` | `run_emcee`/`run_ultranest` default to `vectorized=True` over the shared posterior; the scalar `make_loglike` is kept as a tested reference. `run_ultranest`'s row-by-row likelihood loop is gone. |
| **Checks** | `scripts/data/check_deps.py`, `scripts/inference/check_cufft_stability.py` | Dependency audit (exits non-zero); fast GPU reproduction of the cuFFT failure with an `--emulate_old` A/B. |

## Key decisions [agreed]

- **Pyro over hand-rolled**, decided on measurement: Pyro's overhead is a *fixed*
  ~2.2–3.3 ms/call, not a multiplicative factor — ~0.05% of a batched forward at
  B=64. Declarative priors and upstream-tested transforms come free.
- **Pyro's NUTS, not a hand-rolled vectorised one.** A chain-vectorised NUTS was
  attempted and abandoned (see "Rejected" below).
- **VI = full-rank Gaussian** (`AutoMultivariateNormal`); mean-field and
  normalizing flows are one-line swaps kept for a tutorial.
- **Testbed = the full 30-element store**, literature parametrisation.
- **The scalar likelihood stays** as an automated cross-check, not just for
  tutorials — the batched path has produced a silent walker-axis bug once
  already.

## Bugs found and fixed

1. **Per-walker `n_h` was mis-aligned in the batched forward.** `_continuum`
   built the absorption screen as `(B, K)` but its rows are element-major
   `(N*B, K)`. Every existing test used a scalar `n_h`, which broadcasts either
   way — only the port's real per-walker values exposed it. Now gathered by
   row→walker index (tiling would be gigabytes at K~2e5).
2. **`CUFFT_INTERNAL_ERROR` partway through GPU runs** — a plan-cache leak, not
   a cuFFT bug. See `[[cufft-plan-cache-blowup]]`; fixed by `FFT_PAD_QUANTUM`,
   a fixed row-chunk with zero-padding, and `limit_cufft_plan_cache()`.
   **Confirmed on the GPU**: fixed run stable, `--emulate_old` leaks as predicted.
3. **A missing `arviz` destroyed a completed multi-hour emcee run.** `_ess` was
   evaluated inside `run_emcee`'s `return` expression, so an optional diagnostic
   sat between the sampling and the first write to disk. `_ess` can no longer
   raise, and the emcee chain now streams to HDF5 as it advances.
4. **`--resume` was broken on the path it exists for.** `p0=None` only continues
   a sampler that ran in the same process; a fresh one needs
   `backend.get_last_sample()`.
5. **`mcmc_check.py --smoke` had been broken since `sigma_v` was freed** — 16
   walkers against `ndim=11`, below emcee's `2*ndim` floor. Now 24.
6. **Gradient sampling OOM-killed at realistic scale.** Energy chunking bounds
   the *forward* but not the autograd graph. Fixed with gradient checkpointing
   in `_density` (identical gradients, ~1.2–2.5x runtime).

## Measurements

**The forward is differentiable end-to-end** w.r.t. temperature, `sigma_v` and
`n_h`; autodiff matches central differences to 0.6% / 4% / 0.01%, and backward
costs **0.78x** the forward. This is what makes NUTS/VI viable at all.

**Gradient memory** (peak RSS, one value+grad step, CPU, `echunk=4096`):

| elements | particles | plain | checkpointed |
|---|---|---|---|
| 2 | 1 | 485 MB | 365 MB |
| 4 | 4 | 7297 MB | 1035 MB |
| 8 | 4 | 11815 MB | 5936 MB |
| 12 | 8 | **OOM-killed** | 16346 MB |

Extrapolating ~linearly in elements×particles, **30 elements × 64 particles is
~300 GB** — far beyond a 22 GiB card. Practically: NUTS runs at B=1 (~5 GB, fine);
**SVI is capped at ~4 particles**, so its "particles are free batching" advantage
is largely lost to the memory ceiling. `--echunk` is the lever (it is the
checkpoint segment size); halving it roughly halves peak memory.

**Pyro overhead** (cheap forward, isolating framework cost): fixed 2.2–3.3 ms per
call across B=1..64. Negligible against a real forward.

## Rejected / abandoned

**Hand-rolled vectorised NUTS** — built, then dropped **[agreed]**. Six of seven
tests passed including agreement with Pyro's NUTS on a correlated Gaussian, so
the trajectory machinery was sound; it failed the ill-conditioned target twice.
Diagnosis: NaN leaking from the divergence path (`logaddexp(-inf,-inf)` then
`-inf - (-inf)`) into dual averaging, freezing the chains — `eps=nan` and
`inv_mass` pinned at its clamp floor, while Pyro solved the same target to 8.5%.
The un-applied fix would be to sanitise the divergence path. Consequence for the
bake-off: **Pyro NUTS evaluates at B=1** while every other sampler amortises a
batch, a structural handicap pinned by a test and reported in the table.

## Store / truth migration

- `STORE28` renamed to **`STORE`** across 8 files (the name asserted something
  false once pointing at 30 elements), defaulting to `spexai/models`.
  `SPEXAI_STORE` still overrides; `run_cluster.sh` follows.
- **Truth regenerated** against the 30-element store, and now records the
  **element set + store path**. A 28-element truth has an identical channel
  grid, so the old `n_keep` check could not tell them apart — guards added to
  both `bake_off.py` and `mcmc_check.py`.
- `store28` is now unreferenced by any default. Its files are byte-identical to
  their `spexai/models` counterparts (a strict subset, lacking Cl 17 and Sc 21).
  Kept only as provenance for the published hot-floor results.
- **Neither the store nor the truth npz is in git** — both must be rsync'd to
  the cluster, and the truth *cannot* be regenerated there (it needs the 40 GB
  SPEX caches).

## Outstanding

**Blocking the SBC campaign (not the bake-off):**
- **Sc (21) is still the provisional 2026-08-14 checkpoint.** It is the only
  manifest entry with a `status` flag and the only production element with no
  `benchmark_test.json`. The final model is not on the laptop (the store copy is
  byte-identical to the only local candidate); it must be rsync'd from the
  cluster and re-collected with `scripts/data/collect_models.py`. Sc is *tied to Fe*,
  not fitted, so it lands as a systematic in `b_sys` — fine for comparing
  samplers, not fine for calibration.

**Needs a GPU (nothing else does):**
- The **bake-off** itself.
- `python scripts/inference/benchmark_ppl.py --device cuda --real --skip_overhead` — the
  forward's B-scaling curve, which is what makes the NUTS row interpretable.
- The **compile stall**, printed by any `--vectorized` run as
  `forward: first call Xs, warm Ys`.

**Deferred [agreed]:** user-facing prior specification for every sampler
(currently uniform boxes only). Required before the package is usable by
outsiders; with Pyro chosen it is mostly `biject_to` plumbing.

**Not covered by checkpointing:** zeus, NUTS, SVI. emcee (HDF5) and UltraNest
(`log_dir`) are protected; the others are short enough that it was judged not
worth it.

## How to run the bake-off

```bash
# transfer (neither is in git; rsync is just faster than rebuilding the truth)
# CORRECTION 2026-08-19: the truth CAN be rebuilt on the cluster. The
# preprocessing that produced ~/work/data/spexai/processed was itself run
# there, so the SPEX caches exist on both machines.
rsync -av spexai/models/ REMOTE:$DEST/spexai/models/
rsync -av $SPEXAI_RESULTS/hot_floor/truth_single.npz \
    REMOTE:$DEST/data/spexai_data/results/hot_floor/

# on the cluster
export MKL_THREADING_LAYER=GNU          # or the torch import dies
export SPEXAI_RESPONSES=...             # dir holding rsl_Hp_L_2025.rmf
python scripts/data/check_deps.py            # exits non-zero if anything is missing
python -u scripts/inference/check_cufft_stability.py --device cuda    # ~1-3 min

COMMON="--device cuda --compile --tf32 --fft32 --counts 1e6"
nohup python -u scripts/inference/bake_off.py --sampler emcee $COMMON > logs/bo_emcee.log 2>&1 &
# then, one at a time (they share one GPU):
#   --sampler zeus / ultranest
#   --sampler nuts --echunk 2048
#   --sampler svi --svi_particles 4 --echunk 2048
python scripts/inference/bake_off.py --summarise
```

If SVI or NUTS OOMs, drop `--echunk` to 1024 then 512 **before** cutting
particles — the chunk is the checkpoint segment, so it buys memory without
costing gradient quality. If SVI cannot exceed 1–2 particles, that is a finding
about VI's viability here, not a tuning failure.

---

# SBC campaign infrastructure + pluggable priors (2026-08-19)

Built while the bake-off was running on the cluster, so none of it needed a
GPU. **All new code is tested; nothing here has been run at scale.**

## Why the 200-sim plan had to be rescoped

At the measured ~5–6 h/chain, 200 SBC sims is ~1100 GPU-h ≈ 46 days on one
card. The levers, in the order they should be pulled:

1. **SBC needs ESS, not chain length.** A rank only consumes `L`≈100
   near-independent draws. If the bake-off reports minESS ≫ 100, the chain
   shortens proportionally. This is the dominant lever and the bake-off is
   measuring it right now.
2. **Sampler choice.** If SVI survives, per-sim cost drops to minutes. If VI
   *fails* SBC, that is a result, not a wasted run.
3. **Fewer sims.** Sensitivity goes as 1/√N, so N=100 costs 1.4× sensitivity
   for 2× compute.
4. **Dimensionality** — SBC is per-parameter and need not free every element.

**Dead lever:** batching sims into larger forwards. The GPU is already
saturated at B=16–48 (wchunk 16→48 changed nothing), so there is no throughput
left to win.

`scripts/inference/sbc_cost_model.py` does this arithmetic from the bake-off npz files:

    python scripts/inference/sbc_cost_model.py --results <bakeoff_dir> --n_sims 100 --gpus 1

It models `t_sim = burn_in + sampling * L/minESS`, because **burn-in does not
shrink**. That floor dominates: at emcee's `discard_frac=0.4`, even a 0.06
shrink factor only takes 5.5 h/sim down to ~2.4 h/sim. Nested sampling and VI
run to a tolerance rather than a step count and are reported unshrunk.

## `scripts/inference/sbc_campaign.py` — the ported driver

Replaces `bias_study.py --stage sbc`, which now exits with a pointer. It runs
on the bake-off's stack (`VectorForward` / `PoissonPosterior` /
`spexai.inference.samplers`) and so accepts **any** of the five samplers.
`--stage point` (emulator-vs-SPEX bias) is a different question and stays in
`bias_study.py` unchanged.

Three things the old path got wrong, each a correctness bug rather than a
refactor:

- **Ranks were taken over the raw correlated chain** (`mean(s < t)`). Ranks are
  uniform only over *independent* draws; correlated ones make a perfectly
  calibrated sampler look miscalibrated. `spexai/inference/calibration.py`
  thins by the autocorrelation time implied by the ESS first. The failure mode
  is counterintuitive and worth knowing: too few effective draws makes the
  sample cloud too narrow, so the truth falls *outside* it too often and the
  rank histogram comes out **U-shaped** — which reads as "posterior too
  narrow". `tests/test_calibration.py::test_correlated_chain_fails_without_thinning`
  pins this end to end.
- **The prior depended on the drawn truth.** `fisher_bias.build_params` centres
  the `log_norm` box on the truth; under SBC that makes `log_norm`'s ranks
  uniform by construction no matter how the sampler behaves. The campaign uses
  a fixed box centred on `--log_norm_ref`.
- **Injection must be the emulator itself.** SBC asks whether the posterior is
  self-consistent for the fitted model. Injecting SPEX truth measures emulator
  bias instead, and mixing the two makes any non-uniformity unattributable.
  (It is also impossible at scale on the cluster — SPEX truth needs the 40 GB
  caches.)

Emulator and forward are built **once** and reused across sims; only the data
changes. Every sim appends to `<out>/sbc_<sampler>.jsonl` with `fsync` the
moment it lands, and `--resume` skips what is already there — the campaign is
days long and zeus/NUTS/SVI have no sampler-level checkpointing, so the
protection lives at the sim level. Without `--resume` it refuses to touch an
existing file rather than mixing two runs.

## `spexai/inference/priors.py` — the deferred prior API

`Uniform`, `LogUniform`, `Normal` (truncated; defaults to ±6σ support) and
`PriorSet`, a drop-in `BoxPrior` replacement. The obstacle was never Pyro —
`SpectrumModel` has always taken arbitrary distributions — it was the
gradient-free samplers, which reach the prior through three different
interfaces that one declaration now serves: `logpdf` (emcee/zeus), `ptform`
(UltraNest), `to_pyro` (NUTS/VI).

**The trap, now documented and tested:** `PoissonPosterior.logp` adds the prior
density, and `loglike` deliberately does **not**. UltraNest draws points
through `ptform`, where the prior is already encoded in the sampling — adding
`logpdf` there too would apply the prior twice. `BoxPrior.logpdf` returns
zeros (unnormalised on purpose) so every existing study reproduces bit-for-bit.

Priors are independent by construction; correlated priors would not survive the
`ptform` interface.

## Still outstanding

- **Sc (21) is unchanged** and still blocks SBC — needs an rsync from the
  cluster, which cannot be done from the laptop. See below.
- Nothing here has been run at scale; the campaign wants one short
  `--n_sims 2` GPU smoke before a real launch.

---

# Tier B: parameter-space bias sweep (`scripts/inference/bias_sweep.py`, 2026-08-19)

**Correction to the section above:** making SBC self-injected removed the
campaign's actual purpose. The original `bias_study.py` injected SPEX truth in
*both* stages to measure **emulator bias**; SBC-the-diagnostic can only
validate the sampler. Both are needed, and they are now separate tools rather
than one confused one.

## Why ranks are the wrong instrument for emulator bias

- **No units.** A non-uniform rank histogram says "something is wrong", not
  "Fe is off by X" or "this matters above N counts". The science question is
  bias *relative to statistical error at a real exposure*; ranks have no
  exposure axis.
- **No attribution.** Emulator error, sampler error and prior mismatch all
  produce the same non-uniformity.
- **Worst cost per bit.** A rank costs a full posterior (~5 h).
  `linear_bias_fisher` gives `b_sys` *and* `sigma_stat` from 2n+1 = 25
  forwards.

SBC's job is therefore the **control**: self-injected, ~100 cheap sims,
establishing that the sampler is calibrated so pull studies are attributable to
the emulator. SPEX-injected rank analysis is still useful as a *screen* over
the whole prior volume (it catches corners a grid would miss) but should be
labelled a misspecification test, not SBC.

## The gap this closes

Every science-level bias number was anchored at **one point** — Perseus, plus a
couple of single-axis `--kT`/`--sigma_v` variants. And `bias_study.py --stage
point` varies only `temp`/`velocity`/`log_norm`, with **abundances pinned
solar**, so the abundance dimension had never been probed at the science level.

That matters because of *combinations*: abundance enters truth and emulator
linearly and so cannot create per-element error, but it changes which elements
dominate which channels, reweighting the per-element errors and moving
`b_sys`. Per-element benchmarks structurally cannot see this; `F^{-1}` can.

## Design

Latin hypercube over the **cluster science range** (not the full training box):
kT / T_mean 1.5–8 keV, T_sigma 0.15–3.0, 8 free abundances 0.2–2× solar,
sigma_v 30–600 km/s, n_H 0–5e21. Reports `b_sys`, `sigma_ref` and the crossover
`N* = N_REF (sigma_ref/|b_sys|)^2` per parameter.

Two resumable stages, because they want different machines:

- `--stage truth` — CPU + the 40 GB SPEX caches. These exist on **both** the
  laptop (`~/work/data/spexai/processed`) and the GPU cluster, because the
  preprocessing that built them was run on the cluster — so this stage can run
  either place and the whole sweep can be one cluster job. Loops **elements
  outermost** so each cache is read once, not once per point. **Measured:** 1–2.6 s to load an element,
  ~0.6 s per element-point → **~60 min for 200 points**, peak memory one
  element (~0.4 GB). The natural point-outer loop (what
  `experiment.stream_truth_counts` does) would be ~7× worse.
- `--stage bias` — Jacobian + Fisher solve, no caches needed. **Measured 262
  s/point on laptop CPU** = ~15 h for 200 points, so this one belongs on the
  GPU. It still uses the serial `fisher_bias.Forward`; porting its 25 Jacobian
  points onto `BatchedJointForward` is the obvious next speedup.

## Status

Smoke-tested end to end at 3 points, single-T. **3 points is not a result** —
but the machinery, checkpointing and reporting all work. At 1e6 in-band counts
nothing exceeded 1 sigma (worst: sigma_v at 0.85), and sigma_v showed by far
the widest spread across points (median 0.06, max 0.85), which is exactly the
parameter-dependence the sweep exists to map. Fe median N* ~7e7, consistent
with the hot-floor verdict.

## Remaining campaign

- **Tier A** — composition check: does the 30-element sum's error follow from
  the per-element errors, or do they add coherently? Not built.
- **Tier C** — MCMC pull/coverage at ~10–20 points chosen from the Tier B map;
  needs `--stage point` extended to vary abundances. Necessary because the
  linearisation under-predicts (1.5–3× at 1e8 counts, right direction).
- **Tier D** — the shrunken self-injected SBC control (`sbc_campaign.py`).

Dependency order D → C → B → A: D validates the instrument C uses, C validates
the approximation B relies on, B covers the space.

**Caveat on record:** `SpexTruthModel` PCHIPs over `data.train_idx` — the rows
the emulator trained on. On-grid that is exactly SPEX; off-grid it is
independent of the emulator's *function* but not its *training data*. With
PCHIP interpolation error ~0.003% vs the emulator's ~0.3%
(`benchmark_offgrid.py`) that is the right trade, but it measures
interpolation fidelity, not extrapolation.

---

# Bake-off results + new samplers (2026-08-19 → 08-20)

## Read this first if resuming

The bake-off has RESULTS. Three new samplers are built and validated but not
yet GPU-run. **There is an unsynced cluster fix — see "Must sync" below.**

## MUST SYNC to the cluster before any further sampler run

`spexai/inference/{vector_forward,ppl,priors,samplers}.py` and
`scripts/inference/bake_off.py`. Plus, on the cluster:
`pip install nautilus-sampler pocomc nessai` (verified with `--dry-run` to
leave torch at 2.13.0, so the tuned compile∘vmap path is untouched).

Without the first three files, **NUTS and SVI both die** with
`Expected all tensors to be on the same device` inside `VectorForward.fold`.

## Results (one 1e6-count Perseus spectrum, 12 free params, A10-class GPU)

| sampler | wall | evals | minESS | ESS/eval | ms/eval | agreement vs emcee |
|---|---|---|---|---|---|---|
| emcee | 1.65 h | 51,020 | 144 | 2.8e-3 | 117 | reference |
| zeus | 5.38 h | 70,800 | 194 | 2.7e-3 | 274 | 0.21σ, width 1.04–1.24 |
| UltraNest | 10.83 h | 218,711 | 3507 | **1.6e-2** | 178 | 0.14σ, width 0.94–1.11 |
| SVI (4 particles) | 1.25 h | 8,006 | 4000† | –† | 564 | **2.57σ**, width 0.43–2.96 |

† VI ESS is the *requested draw count*, not measured. Its ESS/eval of 0.50 is
an artefact (`n_posterior/evals`) and made SVI look 31x better than UltraNest.
`summarise()` now flags it `!` via the `BY_CONSTRUCTION` set; the importance
samplers are flagged `*` for Kish-weighted ESS.

**Written up in `docs/inference_methodology.tex`, Sec. `sec:bakeoff`.**

**Decisions:** emcee for the SBC campaign (~165 GPU-h for 100 sims), UltraNest
as the reference and only logZ source. **Drop zeus** — its mixing is fine
(~5.5 evals/walker/iter) but `vectorize=True` hands it a *shrinking* active
subset, so each iteration's tail runs at B=8,4,2 on an idle GPU: 274 ms/eval
vs emcee's 117, plus 4–24% wider posteriors.

**Recovery offsets are NOISE, not emulator bias.** All samplers agree
(Ar ≈ −2σ, Ca ≈ +1.4σ). emcee pulls: mean −0.23, sd 0.98, χ²=11.1/12,
**p=0.52**. Fisher predicts max bias 0.23σ at this exposure — 20x smaller than
the largest pull. One fit cannot measure emulator bias.

## NUTS — diagnosed, options added, NOT yet re-run

Observed **958 s/iteration**, 21/1000 warmup in 3 h → 532 h projected. Stock
Pyro (not hand-rolled). Cause: **saturating `max_tree_depth=10`** = 1023
leapfrog steps/iteration, each a B=1 gradient at 937 ms (implying a B=1 forward
of 526 ms = **4.0x** the batched 131 ms/walker).

**NUTS is NOT disqualified** (I claimed that first and was wrong). A NUTS
gradient costs 7.1x an amortised emcee eval, so it needs ESS/eval > 2.0e-2;
since a well-adapted NUTS gives ~1 independent draw per iteration,
ESS/eval ≈ 1/(leapfrog steps). **Break-even ≈ 50 steps** — at 31 steps NUTS
would beat both emcee and UltraNest. It is 21x too long, which is
adaptation/geometry, not a fundamental limit.

Two likely causes, both now addressable:
- `full_mass=False` (Pyro default) — a **diagonal** mass matrix cannot
  represent correlation, and this posterior has the ρ≈−0.85 abundance/norm
  degeneracy. Checked: marginal scales are NOT the problem (unconstrained
  condition number only **79**, ~9 steps). It is the correlations.
- `init_to_uniform` (Pyro default) started NUTS at a **random prior draw**
  while every other sampler starts at truth — unfair *and* it guarantees huge
  early trajectories in a 44-nat-sharp posterior.

Added `--nuts_full_mass`, `--nuts_tree_depth`; `init_values=center` is now
automatic. `run_nuts` prints and records **`steps_per_iter`** from `n_eval` —
a direct measurement of trajectory length, not a wall-clock inference. Read it
against ~50. Note Pyro's first mass-matrix window doesn't close until ~iter 75.

**Next:** `--nuts_full_mass --nuts_tree_depth 7 --nuts_warmup 150
--nuts_samples 150` (~10 h worst case). If it still saturates, the answer is an
**ensemble gradient sampler** (ChEES-HMC, MEADS) — `potential_and_grad` already
accepts `(B, ndim)`, and at B=64 a gradient costs 233 ms/chain instead of 936,
projecting ESS/wall-s ≈ 0.086 vs emcee 0.024, UltraNest 0.090.

## SVI — OOM fixed, guides added, economics understood

**The OOM:** `echunk` cannot help — it bounds the trunk's *energy*-axis
activations, but `vectorize_particles=True` puts all particles in one batch, so
peak memory scales linearly with `num_particles`. Added
**`--svi_particle_chunk`**: particles are split into sub-batches, each calls
`backward()`, gradients accumulate into the same `.grad` before one step. Pyro's
`Trace_ELBO` averages over particles, so each chunk is weighted by its share and
the accumulated gradient is **identical in expectation**. Memory scales with the
chunk, gradient quality with the total. Pinned by
`test_particle_chunking_matches_unchunked_gradient` (magnitude + direction).
Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Guide families** via `--svi_guide {mvn,iaf,lowrank,normal}` and
`samplers.make_guide`. All four verified to work with
`vectorize_particles=True` (a guide that silently broke it would just look
slow). Parameter counts at 12 latents: mean-field 24, **full-rank 168**,
IAF-2 2,786, IAF-4 5,572.

**SVI cost = steps × particles.** Nothing else. The original run (2000 × 4 =
8,000 evals) was the *fastest* sampler in the table at 0.16x emcee's evals; it
was fast **because** it was under-resourced, and it was wrong. To stay cheaper
than emcee needs `steps × particles < 51,020` — so 2000 steps allows ≤25
particles. A 4000 × 32 configuration is 128,000 evals = 2.5x emcee ≈ 27 h.

**A flow guide is probably NOT the fix.** VI minimises reverse KL, which is
mode-seeking and systematically *under*-estimates variance; the observed widths
span 0.43–2.96, and ~3x **too wide** is not a family-mismatch signature, it is
an unconverged optimiser. Also, at 1e6 counts with 44 nats the posterior should
be near-Gaussian (Bernstein–von Mises), which is exactly what the whole Fisher
framework already assumes and validates. A full-rank Gaussian can represent
ρ≈−0.85 fine.

**Recommended next (not yet implemented): initialise the guide from the Fisher
matrix.** `F⁻¹` *is* the posterior covariance — set `loc`=truth,
`scale_tril`=`cholesky(F⁻¹)` instead of `init_to_median`/`init_scale=0.1`. VI
then starts at the answer with the correct correlation structure instead of
discovering it by SGD; should cut the step count sharply and is diagnostic (if
a Fisher-initialised, well-resourced Gaussian *still* fails on n_h, the family
really is inadequate and IAF becomes justified).

Suggested order: (1) Gaussian 2000 steps × 16 particles, chunk 8; (2) Gaussian
+ Fisher init; (3) IAF only if both fail.

## New samplers: built + validated, NOT GPU-run

`run_nautilus`, `run_pocomc`, `run_inessai` in `spexai/inference/samplers.py`,
wired into `bake_off.py`. **12 tests in `tests/test_samplers_flow.py`; full
suite 179+ passing.** Validated on an analytic Poisson problem, including
cross-checks against UltraNest (medians <0.5σ, width 0.6–1.6, nautilus logZ
within 1.0 nat) — the test that catches mishandled importance weights, since a
broken importance sampler still returns a confident, wrong posterior.

**i-nessai trap:** its default `stopping_criterion="ratio"` with
`tolerance=0.0` stops once `log(Z_live/Z_all) <= 0`, true within a couple of
iterations for a peaked likelihood. Result: log-weights spanning 8.7e5 nats,
Kish ESS **exactly 1.0**, one surviving draw — while the raw point cloud looked
healthy. Fixed by `stopping_criterion="ess"` + `--target_ess` (default 2000).
Before/after: n_eval 2,037→16,037; ESS 1.0→1,486. Also: `fs.posterior_samples`
rejection-samples and collapses, so use `ns.samples` + `ns.log_posterior_weights`;
and the importance sampler needs `to_unit_hypercube`/`from_unit_hypercube`,
unimplemented in the base `Model`.

**Commands** (one at a time — they share the GPU):
```bash
COMMON="--device cuda --compile --tf32 --fft32 --counts 1e6"
python -u scripts/inference/bake_off.py --sampler nautilus $COMMON --n_live 2000 --n_eff 10000
python -u scripts/inference/bake_off.py --sampler pocomc  $COMMON --n_effective 512 --n_active 256
python -u scripts/inference/bake_off.py --sampler inessai $COMMON --n_live 2000 --target_ess 2000
```
**Log GPU memory for these three** — all call the likelihood with *varying*
batch sizes, the pattern behind the earlier `CUFFT_INTERNAL_ERROR` from leaked
FFT plans. emcee/NUTS have fixed shapes and are safe; these are not.

Run **nautilus first** — it is the direct challenger to UltraNest's 1.6e-2 and
also returns logZ, so it is like-for-like.

## Outstanding

- Sync the device fix + pip installs to the cluster (blocks NUTS and SVI).
- Re-run NUTS with `--nuts_full_mass`; re-run SVI per the order above.
- GPU-run nautilus / pocoMC / i-nessai.
- Implement the Fisher-initialised VI guide (recommended, not built).
- Consider `--svi_progress_every` — the progress interval is `steps//20`, so a
  4000-step run prints only every 200 steps (~80 min of silence at 24 s/step).

## Evaluation campaign: Tier A and Tier C are still UNBUILT

The four-tier design is in the "Bias across parameter space" section above and
in `docs/inference_methodology.tex` §`sec:biassweep`. Dependency order is
**D → C → B → A**: D validates the instrument C uses, C validates the
approximation B relies on, B covers the space. Status:

| tier | what | status |
|---|---|---|
| A | joint composition check | **NOT BUILT** |
| B | Fisher `b_sys` sweep (`scripts/inference/bias_sweep.py`) | built, smoke-tested at 3 points only |
| C | MCMC pull/coverage at points chosen from B | **NOT BUILT** (needs `--stage point` extended) |
| D | self-injected SBC control (`scripts/inference/sbc_campaign.py`) | built, tested, not run at scale |

### Tier A — joint composition check (not built)

**Question:** does the 30-element sum's error follow from the per-element
errors, or do they add *coherently*? Every per-element benchmark is 1-D in
temperature and single-element; nothing tests whether errors cancel or
reinforce when 30 elements are summed at a realistic abundance pattern.

**Why it matters:** it gates interpretation of every joint result. If errors
add coherently the joint error is ~N× a single element's; if they add in
quadrature it is ~√N×. `b_sys` is a joint quantity, so this sets the scale of
what Tier B is measuring.

**Shape of the work:** pure emulator-vs-truth in *spectrum* space, no fitting
and no sampling — compare `JointOperatorModel` against a streamed
`SpexTruthModel` sum over a handful of (kT, abundance-pattern) points, and
compare the joint counts-weighted residual against the per-element residuals
combined under both assumptions. Cheapest tier; hours, not days. Reuse the
element-outer streaming loop already written in
`scripts/inference/bias_sweep.py::stage_truth`.

### Tier C — MCMC pull/coverage with free abundances (not built)

**Blocker to fix first:** `scripts/inference/bias_study.py --stage point` varies only
`temp`, `velocity`, `log_norm`. **Abundances are pinned solar and `logz` is
fixed** (`REALISTIC`/`EXTREME` dicts near the top of the file, with the comment
"fixed here to keep the smoke small -- extend as needed"). So the abundance
dimension has never been probed at the science level, which is exactly the
dimension Tier B says reweights `b_sys`.

**Why it is needed at all:** the Fisher estimator of Tier B is first order in
the residual. The hot-floor cross-check found converged MCMC offsets running
**1.5–3× the linear `b_sys`** at 1e8 counts, though in the predicted direction
(10/11 parameter signs matched). So Tier B is a calibrated *screen and
ranking*, not a final number, and its flagged points need confirmation with
full posteriors.

**Shape of the work:** (1) unpin abundances/`logz` in `bias_study.py --stage
point`, ideally by reusing `PriorSet` rather than the bespoke range dicts;
(2) run ~10–20 points **chosen from the Tier B map** — the worst offenders plus
a few Tier B calls "safe", to check for false negatives; (3) report pulls and
central-interval coverage, not ranks (ranks are Tier D's job — see
`spexai/inference/calibration.py` and the thinning trap documented there).

**Cost:** at emcee's 1.65 h/posterior, 20 points ≈ 33 GPU-h. Cheap next to the
SBC campaign, and it is what makes Tier B publishable.
