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
| **Absorption** | `spexai/inference/absorption.py`, `scripts/build_tbabs_table.py`, `spexai/inference/data/tbabs_sigma.npz` | Screen on the native fine grid pre-rebin (observed frame); tbabs (cached from XSPEC) + wabs fallback; `Absorption.default()`. |
| **Normalisation** | `spexai/inference/units.py`, `scripts/validate_spex_norm.py` | Physical `norm` = emission measure Y (1e64 m⁻³) + `luminosity_distance`; m²→cm² factor; validated vs SPEX. |
| **Bias study** | `scripts/bias_study.py` | Staged SBC/pulls: point bias + coverage, then rank statistics; SPEX-truth inject with emulator-self control. |
| **Perseus showcase** | `scripts/perseus_showcase.py` | Single-T & Gaussian-DEM injection-recovery at Perseus's physical distance; reports the emission measure. |
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
  m²→cm² factor — no order-of-magnitude error. (`scripts/validate_spex_norm.py`.)
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
`nwalkers=96`, via `scripts/benchmark_inference.py`.

## Result: 1.88x, and the ladder is now closed

| config | ms/walker | vs previous rung |
|---|---|---|
| batched-accel (TF32 + float32 FFT) | 209.1 | — |
| batched-compile (`compile ∘ vmap`) | 111.3 | **1.88x** |

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

The forward is done for now: 1.88x banked, every other kernel-level lever closed
on measurement. At **5.9 h/chain** and **~1170 GPU-h for 200 SBC sims**, the
productive direction is fewer forwards — fewer sims, shorter chains, or better
sampler efficiency — not a faster one.

**Not yet measured:** `batched-compile` vs `serial-accel` (the current
production path in `run_cluster.sh`, where the per-element compile *does*
apply). That is the comparison that decides whether the batched path should
replace the serial production forward. Also unmeasured: the one-off compile
stall (`timeit` warms up before timing).
