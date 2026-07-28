# SpexAI: evaluation & inference tutorial

Step-by-step instructions for evaluating the emulator, fitting spectra, running
the various scripts, and running the simulation-based-calibration (SBC) bias
study. For a hands-on, cell-by-cell walkthrough of a single fit, see the
notebook `tutorials/inference_walkthrough.ipynb`.

---

## 0. Setup

All commands run in the `spexai` conda env. On macOS you **must** export one
environment variable or torch scripts abort with an OpenMP error:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS only; harmless elsewhere
conda activate spexai                      # or prefix each call with: conda run -n spexai
```

Two data locations are assumed (override with flags where noted):

- **Model store** — `spexai/models/` (per-element `Z<zz>_<sym>.pt` + `manifest.json`), shipped in the repo.
- **Preprocessed SPEX caches** — `~/work/data/spexai/processed/element<Z>/` (energy/temps/logflux/splits), used for evaluation and as ground truth.
- **Instrument responses** — `~/work/data/spexai/responses/` (RMF/ARF).

Everything below runs on **CPU** and is laptop-safe (small examples). Scale up
element count, chain length, and simulations on a GPU/cluster.

---

## 1. Evaluating the emulator

### 1a. Whole-manifest table (all elements at once)

Scores every element in `manifest.json` against its held-out SPEX test split and
prints an aggregate markdown table (overall / line / continuum MRE, yields,
floor-violation %):

```bash
python -m spexai.eval --split test
# options: --split {test,val}  --datadir <caches>  --models-dir <store>  --device cpu
```

It also writes `spexai/models/manifest_eval_test.{json,md}`. This scales from the
current 16 elements to the full 30 automatically as the model store grows.

Programmatic use:

```python
from spexai.eval import evaluate_manifest, manifest_table
res = evaluate_manifest(split="test")        # {Z: {overall, lines, continuum, floor, ...}}
print(manifest_table(res))
```

### 1b. One element in depth

`benchmark_operator.py` benchmarks every checkpoint in a training rundir on the
held-out set: overall/line/continuum MRE, floor violations, evaluation speed,
plus residual-vs-energy and example-spectrum figures.

```bash
python scripts/benchmark_operator.py \
    --rundir  ~/work/data/spexai/runs/element26 \
    --cachedir ~/work/data/spexai/processed/element26 \
    --split test
```

Off-grid confirmation (fresh temperatures never in any split, PCHIP pseudo-truth):

```bash
python scripts/benchmark_offgrid.py --rundir ... --cachedir ... --seed 1
```

Temperature-resolved error for one element (finds the worst temperature bands):

```bash
python scripts/diagnose_element.py --rundir ... --cachedir ...
```

### 1c. Emulator vs. independent SPEX truth (assembled model)

`evaluate_joint` compares the *assembled* `JointOperatorModel` against the
independent `SpexTruthModel` (PCHIP over SPEX + exact broadening), in flux space
and in detector-channel space after folding:

```python
import torch
from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.spex_truth import SpexTruthModel
from spexai.inference.response import Response
from spexai.eval import evaluate_joint

resp  = Response("~/work/data/spexai/responses/aciss_aimpt_cy28.rmf",
                 "~/work/data/spexai/responses/aciss_aimpt_cy28.arf")
joint = JointOperatorModel(device="cpu", elements=[26])
truth = SpexTruthModel(device="cpu", elements=[26])
cases = [{"temp": T, "abundances": {}} for T in (2.0, 4.0, 8.0)]
print(evaluate_joint(joint, truth, resp, cases, velocity=150.0))
# -> {"flux_mre_median": ..., "channel_mre_median": ..., "per_case": [...]}
```

---

## 2. Running inference

### 2a. The turnkey demo

The fastest way to see a full fit + all diagnostic plots:

```bash
python scripts/run_inference_demo.py --elements 8 14 26 --nsteps 250 --nlive 150
# writes emcee_trace / ultranest_diag / corner / posterior_predictive PNGs to ./inference_demo/
```

### 2b. Programmatic fit (single temperature)

```python
import numpy as np, torch
from spexai.inference.operator_model import JointOperatorModel
from spexai.inference.response import Response
from spexai.inference.simulate import simulate_observation
from spexai.inference.fitting import Param, run_emcee, run_ultranest

resp  = Response(f"{RESP}/aciss_aimpt_cy28.rmf", f"{RESP}/aciss_aimpt_cy28.arf")
model = JointOperatorModel(device="cpu", elements=[8, 14, 26])

# simulate data with a known truth (or load a real observation into an Observation)
truth = {"temp": 3.0, "norm": 1e10, "velocity": 200.0, "logz": -10.0}
obs   = simulate_observation(model, resp, truth, exposure=1e5,
                             target_counts=2e4, rng=0)
ln = float(np.log10(obs.true_params["norm"]))

params = [Param("temp", 0.5, 6.0, "T [keV]", truth["temp"]),
          Param("log_norm", ln-1.5, ln+1.5, r"$\log_{10}$ norm", ln),
          Param("velocity", 0.0, 600.0, "v [km/s]", truth["velocity"])]
fixed  = {"abundances": {}, "logz": -10.0}

er = run_emcee(obs, model, params, fixed, nwalkers=16, nsteps=400)
ur = run_ultranest(obs, model, params, fixed, min_num_live_points=200)
print("emcee median :", dict(zip(er.names, er.median)))
print("ultranest lnZ:", ur.logz, "+-", ur.logzerr)
```

### 2c. Fitting abundances (and tying elements)

`AbundanceModel` maps fit parameters to a `{Z: value}` dict. Free a global
metallicity, free individual elements, or tie a group to a fraction of iron:

```python
from spexai.inference.abundances import AbundanceModel

ab = (AbundanceModel(model.elements)
      .global_metallicity("Z")        # one param scales all metals
      .free_element(26, "Fe")         # iron on its own (absolute solar)
      .tie([8, 14], "alpha_Fe", ref=26))   # O, Si tied to alpha_Fe * Fe

params = [Param("temp", 0.5, 6.0, truth=3.0),
          Param("Z", 0.1, 1.5, truth=0.3),
          Param("Fe", 0.1, 1.5, truth=0.5),
          Param("alpha_Fe", 0.5, 2.0, truth=1.2),
          Param("velocity", 0.0, 600.0, truth=200.0),
          Param("log_norm", ln-1.5, ln+1.5, truth=ln)]
er = run_emcee(obs, model, params, fixed, abundance_model=ab)
```

`ab.param_names` lists exactly the parameters you must supply as `Param`s.

### 2d. Fitting a temperature distribution (DEM)

Build a DEM model (from `spexai.inference.tempdist`) and pass it as `dem=`; the
DEM's `param_names` become fit parameters:

```python
from spexai.inference import tempdist as td

grid = td.TempGrid(0.5, 10.0, n=48)
dem  = td.gaussian_T(grid)                 # Gaussian in linear T: T_mean, T_sigma
# other shapes: td.gaussian_logT, td.lognormal_T, td.TwoGaussianDEM, td.BinnedDEM

params = [Param("T_mean", 1.0, 8.0, truth=4.0),
          Param("T_sigma", 0.1, 3.0, truth=1.0),
          Param("velocity", 0.0, 400.0, truth=180.0),
          Param("log_norm", ln-1.5, ln+1.5, truth=ln)]
er = run_emcee(obs, model, params, fixed, dem=dem)   # combine with abundance_model=... if desired
```

Note: the corner plot works for any parameter set, but
`plot_posterior_predictive` is single-temperature only (it calls
`predict_counts`). For DEM predictive checks, evaluate `model.predict_counts_dem`
with the posterior-median parameters yourself.

### 2e. Galactic absorption

Pass an absorption screen and a column (`n_h`, cm⁻²). Fix it via `fixed["n_h"]`
or fit it by adding `Param("n_h", ...)`:

```python
from spexai.inference.absorption import Absorption
absn = Absorption.default()                 # cached tbabs if built, else wabs
fixed = {"abundances": {}, "logz": np.log10(0.0179), "n_h": 1.4e21}
er = run_emcee(obs, model, params, fixed, absorption=absn)
```

The `tbabs` cross-sections (Wilms+2000) are the default once the table
`spexai/inference/data/tbabs_sigma.npz` exists; build it in a HEASoft env
(installable via the [HEASARC conda channel](https://heasarc.gsfc.nasa.gov/docs/software/conda.html))
with `conda run -n <heasoft-env> python scripts/build_tbabs_table.py`. Without
it, `Absorption.default()` falls back to the dependency-free `wabs`.

Simulate absorbed data with `simulate_observation(..., absorption=absn)` and
`params["n_h"] = 1.4e21`.

### 2f. Physical normalization and distance

`norm` (fit as `log_norm` = log₁₀ norm) is the **SPEX emission measure**
Y = n_H n_e V in units of 10⁶⁴ m⁻³. Counts scale as Y·(D_ref/D)² with the SPEX
reference distance D_ref = 10²² m. Pass the source distance in **metres** via
`luminosity_distance=` to `predict_counts`/`simulate_observation`, or
`fixed["luminosity_distance"]` in a fit (Y and distance are degenerate in one
spectrum, so fix the distance from the known redshift and fit Y). Handy
conversions: 1 Mpc = 3.086×10²² m; the SPEX default D_ref = 10²² m ≈ 0.324 Mpc.
The absolute scale is validated against a SPEX install to ~2.5%
(`scripts/validate_spex_norm.py`).

### 2g. Diagnostic plots

```python
from spexai.inference.fit_plots import (plot_emcee_trace, plot_ultranest_diagnostics,
                                        plot_corner_overlay, plot_posterior_predictive)
plot_emcee_trace(er, "emcee_trace.png")                 # traces + autocorr time
plot_ultranest_diagnostics(ur, "ultranest_diag.png")    # NS trace + lnZ/ESS
plot_corner_overlay(er, ur, "corner.png")               # emcee + UltraNest overlaid
plot_posterior_predictive(obs, model, er, ur, fixed, "ppc.png")   # single-T only
```

---

## 3. Script reference

| Script | Purpose |
|---|---|
| `python -m spexai.eval` | Whole-manifest evaluation table (all elements). |
| `scripts/benchmark_operator.py` | Per-element held-out benchmark + figures. |
| `scripts/benchmark_offgrid.py` | Off-grid-temperature confirmation vs PCHIP pseudo-truth. |
| `scripts/diagnose_element.py` | Temperature-resolved error for one element. |
| `scripts/run_inference_demo.py` | Turnkey emcee+UltraNest fit + all plots. |
| `scripts/bias_study.py` | Staged SBC / pulls bias study (Section 4). |
| `scripts/perseus_showcase.py` | Perseus single-T & DEM injection-recovery. |
| `scripts/build_tbabs_table.py` | Tabulate tbabs σ(E) (run where HEASoft/sherpa exists). |

Every script takes `--help`. `Absorption.default()` uses the cached `tbabs`
table when present, else `wabs`; build the table in a HEASoft env (HEASARC conda
channel — conda-forge has no XSPEC for macOS) with `build_tbabs_table.py`.

---

## 4. The SBC / bias study

The bias study asks: **are the emulator's posteriors unbiased and calibrated?**
It injects spectra from the *independent* `SpexTruthModel` (or the emulator
itself, as a control), fits them with the emulator, and measures how the
recovered posteriors relate to the injected truth.

### 4a. Stage 1 — pulls & coverage

```bash
python scripts/bias_study.py --stage point --n_sims 50 --elements 8 14 26 \
    --nwalkers 24 --nsteps 800 --exposure 1e5 --target-counts 5e4 --out bias_point.json
```

For each simulation and parameter it records:

- **pull** = (posterior median − truth) / posterior σ. Averaged over sims, the
  mean pull should be ≈ 0 (no bias) with spread ≈ 1 (errors not mis-sized).
- **coverage_68** = fraction of sims whose 16–84% interval contains the truth;
  should be ≈ 0.68 if calibrated.

Interpretation printed per parameter, e.g. `temp pull=+0.06+-0.49 cov68=0.66`.
A mean pull well away from 0 flags an **emulator bias**; run `--self-test`
(inject with the emulator instead of SPEX) as the control — if the self-test
pulls are ≈ 0 but the SPEX-truth pulls are not, the bias is the emulator's, not
the sampler's.

### 4b. Stage 2 — simulation-based calibration (rank statistic)

```bash
python scripts/bias_study.py --stage sbc --n_sims 200 --elements 8 14 26 \
    --nsteps 1000 --out bias_sbc.json
```

Each sim records the **rank** = fraction of posterior samples below the truth,
per parameter. Across many sims from the prior, a calibrated inference gives a
**uniform** rank histogram; systematic deviations (∪ or ∩ shapes) reveal
over/under-confidence. `rank_mean` near 0.5 is the first-order check.

### 4c. Ranges: realistic vs extreme

The default ranges are literature-realistic (kT 1–8 keV, v 0–300 km/s). Add
`--extreme` to stress-test the edges of the training domain (kT 0.3–10, v 0–600).
Run the realistic set first, then the extreme set as a robustness check.

### 4d. Scaling & caveats

- Start tiny to validate the pipeline (`--n_sims 4 --nsteps 150`), then scale
  `--n_sims`, `--nsteps`, and `--elements` on a GPU/cluster. Each sim is an
  independent MCMC, so this parallelises trivially over `--seed`.
- Absolute normalisation is still a placeholder: `--target-counts` rescales the
  norm to control S/N, so the fitted `log_norm` prior is centred on the rescaled
  value.
- To include abundances / DEM / absorption in the bias study, extend the
  parameter set in `bias_study.py` following the patterns in Section 2.

### 4e. The Perseus showcase

A concrete, literature-anchored recovery demonstration (single-temperature core
and a Gaussian DEM), simulated from SPEX truth through XRISM/Resolve (ACIS
fallback) with realistic Poisson noise and Galactic absorption:

```bash
python scripts/perseus_showcase.py --mode single --elements 8 14 26 --nsteps 800
python scripts/perseus_showcase.py --mode dem    --elements 8 14 26 --nsteps 800
# prints truth-vs-recovery per parameter and writes a corner plot
```

---

## 5. Running the SBC & Perseus tests on a remote/cluster machine

These are the compute-heavy runs. Each simulation is an independent MCMC, so
they parallelise trivially. Below, fill in the placeholders for your machine.

### 5.0 One-time setup on the remote

```bash
# get the code + environment
git clone <repo-url> spexai && cd spexai      # or: git pull
conda env create -f environment.yml           # creates the `spexai` env
conda activate spexai
pip install -e .                               # so `import spexai` resolves

# set convenient variables (edit to your paths)
export ENV=spexai
export DATADIR=$HOME/work/data/spexai/processed          # per-element SPEX caches
export RESP=$HOME/work/data/spexai/responses             # RMF/ARF files
export KMP_DUPLICATE_LIB_OK=TRUE                          # harmless off-macOS
```

The model store (`spexai/models/`) ships with the repo. The manifest's
`runroot` is laptop-specific, so **pass `--datadir $DATADIR`** where the scripts
need the caches (bias study / truth injection). If your cluster has GPUs the
fits still run on CPU here; the emulator is fast enough that MCMC is
CPU-bound on the likelihood.

### 5.1 (Optional) build the tbabs absorption table

If the cluster has HEASoft (PyXspec or sherpa) in some env, build the validated
`tbabs` table once; otherwise the runs use the `wabs` fallback automatically.

```bash
conda run -n <heasoft-env> python scripts/build_tbabs_table.py
# writes spexai/inference/data/tbabs_sigma.npz; Absorption.default() then uses it
```

### 5.2 The SBC / bias study

Run the **realistic** ranges first, then the **extreme** stress test, over the
full available element set (`--elements all`). Production-scale settings:

```bash
# realistic ranges — Stage 1 (pulls & coverage). Omit --extreme for realistic.
python scripts/bias_study.py --stage point \
    --elements all --n_sims 300 --nwalkers 48 --nsteps 3000 \
    --exposure 1e5 --target-counts 5e4 \
    --rmf $RESP/aciss_aimpt_cy28.rmf --arf $RESP/aciss_aimpt_cy28.arf \
    --seed 0 --out bias_point_realistic.json

# Stage 2 (SBC rank statistic)
python scripts/bias_study.py --stage sbc --elements all --n_sims 300 \
    --nsteps 3000 --out bias_sbc_realistic.json

# extreme stress test
python scripts/bias_study.py --stage point --extreme --elements all \
    --n_sims 200 --nsteps 3000 --out bias_point_extreme.json

# control: inject with the emulator itself (isolates emulator bias)
python scripts/bias_study.py --stage point --self-test --elements all \
    --n_sims 200 --nsteps 3000 --out bias_point_selftest.json
```

**Launching long jobs.** For a single long run, use `tmux`/`screen` or `nohup`:

```bash
tmux new -s sbc
python scripts/bias_study.py --stage sbc --elements all --n_sims 500 --nsteps 4000 \
    --out bias_sbc.json  2>&1 | tee bias_sbc.log
# detach: Ctrl-b d ; reattach: tmux attach -t sbc
```

**SLURM array (recommended at scale).** Because sims are independent, shard by
`--seed` and merge the JSONs afterward:

```bash
# sbc_array.sh
#SBATCH --array=0-19 --cpus-per-task=8 --time=12:00:00 --mem=16G
conda run -n $ENV python scripts/bias_study.py --stage sbc --elements all \
    --n_sims 25 --nsteps 3000 --seed $SLURM_ARRAY_TASK_ID \
    --out bias_sbc_${SLURM_ARRAY_TASK_ID}.json
# -> 20 x 25 = 500 sims; concatenate the "results" arrays from each JSON to
#    rebuild the rank histogram / pull distribution.
```

Interpretation (per Section 4): **pull mean ≈ 0, spread ≈ 1; coverage_68 ≈
0.68; rank histogram uniform.** Compare SPEX-truth vs `--self-test` to attribute
any bias to the emulator vs the sampler. Set threads with
`export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` (and `MKL_NUM_THREADS`) to avoid
oversubscription.

### 5.3 The Perseus test cases

The showcase fixes the **luminosity distance to Perseus's (~75 Mpc)**, so the
fitted `log_norm` is the *physical* emission measure Y = n_H n_e V (in
10⁶⁴ m⁻³), which the script reports alongside the parameter recovery.
`--target-counts` only sets the S/N (and hence the implied Y). Point at an
XRISM/Resolve response if the cluster has one (the script auto-discovers
`*resolve*`/`*xrism*` in `$RESP`, else falls back to ACIS):

```bash
python scripts/perseus_showcase.py --mode single --elements all \
    --nwalkers 48 --nsteps 3000 --exposure 3e5 --target-counts 2e5 \
    --out perseus_single  2>&1 | tee perseus_single.log

python scripts/perseus_showcase.py --mode dem --elements all \
    --nwalkers 48 --nsteps 4000 --exposure 3e5 --target-counts 2e5 \
    --out perseus_dem  2>&1 | tee perseus_dem.log
```

Each prints a truth-vs-recovery table and writes
`perseus_<mode>_corner.png`. Bump `--nsteps` until the emcee autocorrelation
warning clears (chain length ≳ 50 τ).

### 5.4 Retrieving & memory notes

- Outputs are JSON (`--out`) + PNG corner plots in the working directory; copy
  back with `scp`/`rsync`.
- `SpexTruthModel` holds each element's training spectra in RAM (~0.4 GB/element);
  `--elements all` on 16 elements needs ~8 GB — size `--mem` accordingly, or
  restrict `--elements`.
- Use **emcee** for the bulk study (fast); reserve **UltraNest** (evidence, and a
  fully independent cross-check) for a handful of spot-check fits — it loops the
  likelihood and is slow.

