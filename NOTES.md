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

- Full suite: **293 passed** (265 baseline + 24 log-T + 1 relative step + 3 D1),
  re-run after the D3 changes. `black`/`flake8` are not installed in the
  `spexai` env, so lint did not run.
- **NOT verified: no stage has run end to end under the new parametrisation.**
  The tests pin names, ordering, bounds, guards and the frozen hot_floor path;
  no truth stage, bias stage or GN sweep has executed. The 2-point smoke runs
  are what closes that gap.
- Tex (`docs/inference_methodology.tex`, `DECISIONS.tex`) compile clean.
- `check_logT_design.py --table` (30-point DEM draw, seed 3): nodes within
  +-1 sigma min 6 / median 20 / max 37 (sigma = 2.99-18.7 cells, every point
  resolved); contained fraction min 0.572 / median 0.960; 22/30 below 0.99,
  8/30 below 0.90, 0/30 below 0.50. **The least-contained points are at the
  COLD end** (0.74 keV, 0.178 dex: 0.572), where the grid floor is 0.7 keV --
  not the hot end, which the containment discussion had assumed.
- **Not run on the laptop:** `--steps` was killed for low memory (system swap
  already ~10/11 GB); the truth/bias smoke runs are handed to the cluster.

### CLOSED 2026-09-16 (accepted, not fully explained)

Rerun with the relative step: single-T kT 0.75 keV went 6.83e-2 -> 3.75e-2,
14 keV went 3.12e-3 -> 6.64e-3, DEM values bit-identical (their step did not
change). **The truncation hypothesis is REFUTED**: shrinking the step 5.8x at
0.75 keV should have cut the difference ~34x and cut it 1.8x; growing it 3.2x
at 14 keV should have grown it ~10x and grew it 2.1x. Step size is not the
controlling factor at either end, so the earlier "~8% Jacobian bias" inference
was wrong.

**The relative step is KEPT** (D.H., 2026-09-16), now justified on units --
one step in dex serves 0.7-15 keV where an absolute keV step cannot -- NOT on
the refuted bias measurement.

**Open, deliberately not chased:** at 0.75 keV almost every channel in
1.9-12 keV has mu ~ 0, and both the check and `linear_bias_fisher` weight
channels by ~1/mu, so empty channels can carry huge weight on float32 noise.
Unresolved whether the cold-end number means (a) J is unreliable below
~1.5 keV, (b) the check misreports it, or (c) the band carries no information
there at all. Affects only cold Tier B/P7 points; cond(F) is printed per point
and a near-singular F is flagged. P6's k is insulated: the GN fixed point is
set by the autograd score, and F only preconditions.
Deferred diagnostic (3 step sizes + signal-restricted metric + empty-channel
count, ~25 min cluster). **Known bug in `check_logT_design.steps()`: the
single-T label hardcodes "(step 5e-3 keV)" and no longer states the step used.**

### Superseded analysis: relative step taken, AD probed and deferred

Fallback (2) below was implemented: `bias_sweep.STEP_DEX = 5e-4` for every
temperature axis, with kT taking `kT ln10 STEP_DEX`. The AD route (1) FAILED
the probe on both paths -- `Sparse CSR tensors do not have strides` (the
response fold, DEM) and `You must implement the jvp function for custom
autograd.Function` (the trunk's gradient checkpointing, single-T). Both are
fixable (the fold is linear, so forward mode need only reach `flux`;
checkpointing is redundant under forward mode) but amount to a hot-path
restructure -- deferred, probe kept. The table trap was never reached, so it
remains unverified and still applies if AD is revisited.
Pending: rerun `--steps` to confirm kT at 0.75 keV drops from 6.8e-2.

### Original analysis: finite-difference step vs forward-mode AD

Step check (cluster, 2026-09-16): DEM steps are fine everywhere -- 5e-4 dex
gives 7.7e-5 to 8.3e-4 relative change under h -> h/2 at the fiducial,
cold-narrow and hot-wide points. **Single-T kT fails at the new cold end**:
6.0e-2 at 0.75 keV (3.1e-3 at 14 keV) with the inherited absolute 5e-3 keV
step, which is 0.67% of kT there versus 0.036% at 14 keV. Truncation error
scales as h^2, so this implies the Jacobian at h is biased ~8%, at h/2 ~2%.

Two routes, in order of preference:

1. **Retire the step: forward-mode AD.** The forward is differentiable.
   Reverse mode is wrong here (one pass per OUTPUT, ~20k channels), but
   forward mode costs one pass per INPUT -- 13 params vs 27 forwards, cheaper
   AND exact. Probe written: `scripts/inference/probe_jacobian_ad.py`
   (read-only, NOT yet run; needs the cluster). torch is 2.13.0 and
   `torch.func.jvp` exists. Three things it settles: does jvp survive the
   sparse CSR fold; does the AD column match FD(h/2); and **the table trap** --
   `batched._density` bypasses the temperature table on
   `temp_kev.requires_grad` (batched.py:341, :658), but a dual tensor has
   `requires_grad == False` (verified), so a single-T JVP could be served from
   the table and return a silently ZERO tangent. The probe reports a near-zero
   AD column as the trap, not as disagreement.
2. **Fallback if jvp fails:** make the single-T step relative,
   `step = kT * ln(10) * 5e-4` (the same 5e-4 dex the DEM uses), instead of
   absolute 5e-3 keV. `campaign.build_params` keeps 5e-3 keV -- hot_floor is
   frozen.

A third option if the cold end turns out to be physically awkward rather than
numerically: raise the single-T design floor above 0.7 keV. Above 1.9 keV a
0.75 keV plasma contributes only its exponential tail.

## Frozen scripts: how they now DIFFER from the campaign

Left unchanged on purpose, so their already-produced results stay
reproducible. If any of them is needed again, reconcile first.

| Script | What it still uses | Campaign now uses |
|---|---|---|
| `scripts/experiments/hot_floor/fisher_bias.py` | `campaign.gaussian_dem` (linear T, keV), `campaign.build_params` (kT 1.5-7.5, T_mean 1.5-7.5, T_sigma 0.15-3.0 keV; **step 5e-3 keV ABSOLUTE**), `--dem_mean/--dem_sigma` in keV | `gaussian_logT_dem`, `bias_sweep.build_pars`, step 5e-4 dex (relative) |
| `scripts/experiments/hot_floor/mcmc_check.py` | same as above | same |
| `scripts/experiments/hot_floor/check_dem.py`, `gpu_forward.py`, `ppc_reconstruct.py` | `campaign.gaussian_dem` / `build_params` / `PERSEUS` linear-T keys | -- |
| `scripts/inference/dump_truth.py` | linear-T `gaussian_dem`; feeds the hot_floor MCMC and the bake-off, writes `results/hot_floor` with the Perseus literature mask | not a campaign truth; `bias_sweep --stage truth` is |
| `scripts/inference/bake_off.py` | `campaign.build_params` single-T (kT 1.5-7.5 keV) | -- |
| `scripts/inference/crosscheck_steps.py`, `weight_step_impact.py` | hard-coded DEM grid 0.7-10 keV, 48 nodes, linear-T weights in numpy | 0.7-19.94 keV, 70 nodes, log-T |
| `scripts/inference/tier_a_composition.py` | imports `bias_sweep.RANGES` / `sample_points` (single-T) | a RERUN would now draw kT log-uniform over 0.7-15 keV, not 1.5-8 keV |

`campaign.Forward` still produces bit-identical names and weights for the
linear-T DEM (pinned by `test_forward_linear_T_hot_floor_path_unchanged`).

## GN bugs (D1-D3), 2026-09-16

- **D1 vacuous convergence: FIXED.** `p6_sweep.convergence_verdict` (extracted
  from `run_point` so it is testable) requires `resolved.any()`; the jsonl now
  carries `n_resolved`. 3 tests in `tests/test_gauss_newton.py` (18 passed);
  full-suite verification pending at the time of writing.
- **D2 not descending: HELD** by D.H. until the log-T switch + D1 have been
  through a real screen. Fix if still needed: accept only on decrease, else
  halve along the Newton direction.
- **D3 spread criterion: more iterations (option 1). IMPLEMENTED.**
  `--gn_iter` default 8 -> 20, with the arithmetic in its help text, plus a
  start-up check in `run_point` (noiseless + `--method gn`): it computes the
  actual start spread in sigma, divides by `--gn_max_step`, and warns at the
  TOP of the log if `--gn_iter` is below that. Rejected: bigger clamp
  (overshoot into the region where GN's dropped residual term is not small),
  tighter `--start_bsys_scale` (closer starts agree more easily, so the
  multi-start certificate would certify less). Only helps if the iteration
  descends -- see D2.

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
