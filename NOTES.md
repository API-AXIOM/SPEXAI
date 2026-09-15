# spexai working notes

Running log of context, progress and open questions. Decisions and their
reasoning go in `DECISIONS.tex`.

## Current work: DEM log-T switch (branch `dem-logT-switch`, uncommitted)

Started 2026-09-15. Baseline on `main` (9113853): 265 tests pass.

**Why:** the bias campaign's DEM was a Gaussian in linear T; SPEX's `gdem` is a
Gaussian in log10 T. The linear-T width had a 34x spread in grid resolution
across one design, which produced the 1.642 outlier. Cost is no longer a
blocker: one GN iteration is ~12 s on the A10 (was 980 s).

**Scope (in):** `tempdist` (`gaussian_logT` bounds, `TwoGaussianDEM` no longer
renormalised), `campaign` (`gaussian_logT_dem`, `check_truth_dem_param`,
`Forward` names from the DEM, `build_params` guard), `bias_sweep` (ranges,
log-uniform kT, contained fraction, truth stamp), `mle_reseed` (legacy guard),
`device_parity`, tests, tex.

### Verification status (2026-09-15)

- Full suite: **289 passed** (265 baseline + 24 new). `black`/`flake8` are not
  installed in the `spexai` env, so lint did not run.
- Tex (`docs/inference_methodology.tex`, `DECISIONS.tex`) compile clean.
- `check_logT_design.py --table` (30-point DEM draw, seed 3): nodes within
  +-1 sigma min 6 / median 20 / max 37 (sigma = 2.99-18.7 cells, every point
  resolved); contained fraction min 0.572 / median 0.960; 22/30 below 0.99,
  8/30 below 0.90, 0/30 below 0.50. **The least-contained points are at the
  COLD end** (0.74 keV, 0.178 dex: 0.572), where the grid floor is 0.7 keV --
  not the hot end, which the containment discussion had assumed.
- **Not run on the laptop:** `--steps` was killed for low memory (system swap
  already ~10/11 GB); the truth/bias smoke runs are handed to the cluster.

## Frozen scripts: how they now DIFFER from the campaign

Left unchanged on purpose, so their already-produced results stay
reproducible. If any of them is needed again, reconcile first.

| Script | What it still uses | Campaign now uses |
|---|---|---|
| `scripts/experiments/hot_floor/fisher_bias.py` | `campaign.gaussian_dem` (linear T, keV), `campaign.build_params` (kT 1.5-7.5, T_mean 1.5-7.5, T_sigma 0.15-3.0 keV), `--dem_mean/--dem_sigma` in keV | `gaussian_logT_dem`, `bias_sweep.build_pars` |
| `scripts/experiments/hot_floor/mcmc_check.py` | same as above | same |
| `scripts/experiments/hot_floor/check_dem.py`, `gpu_forward.py`, `ppc_reconstruct.py` | `campaign.gaussian_dem` / `build_params` / `PERSEUS` linear-T keys | -- |
| `scripts/inference/dump_truth.py` | linear-T `gaussian_dem`; feeds the hot_floor MCMC and the bake-off, writes `results/hot_floor` with the Perseus literature mask | not a campaign truth; `bias_sweep --stage truth` is |
| `scripts/inference/bake_off.py` | `campaign.build_params` single-T (kT 1.5-7.5 keV) | -- |
| `scripts/inference/crosscheck_steps.py`, `weight_step_impact.py` | hard-coded DEM grid 0.7-10 keV, 48 nodes, linear-T weights in numpy | 0.7-19.94 keV, 70 nodes, log-T |
| `scripts/inference/tier_a_composition.py` | imports `bias_sweep.RANGES` / `sample_points` (single-T) | a RERUN would now draw kT log-uniform over 0.7-15 keV, not 1.5-8 keV |

`campaign.Forward` still produces bit-identical names and weights for the
linear-T DEM (pinned by `test_forward_linear_T_hot_floor_path_unchanged`).

## Flags for later

- **Perseus showcase** (`perseus_showcase.py`): stays linear T to match the XRISM
  analysis (XSPEC `bvvgadem`, arXiv:2606.17141). Its DEM grid (0.7-10 keV, 48
  nodes) and fiducial (thesis 4.27/1.11 keV; XRISM SW region 3.41/1.0 keV is the
  candidate) are UNRESOLVED -- decide when the showcase is reached. Table 3
  values came from an automated fetch; verify against the PDF.
- **GN bugs** (vacuous convergence, not descending, unreachable start spread):
  fix right after the log-T switch lands.
- **`tier_c_mcmc.py --mode dem`** builds `VectorForward` without `dem=`; fix
  when Tier C is reached.
- **Line deposit** is 85% of the GPU DEM forward; revisit only if inference
  becomes computationally infeasible.
- **Cache-step scripts**: update to read the grid from the campaign and rerun
  when interpreting the regenerated DEM screen.
