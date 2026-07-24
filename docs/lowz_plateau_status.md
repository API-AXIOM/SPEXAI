# Low-Z accuracy plateau: status (2026-07-24)

## Target
0.1% relative error (MRE) on held-out test spectra, per element, continuum
ionization-equilibrium emulator.

## Survey: elements 2-20, 22-24, 26 (23 with benchmark data; Z=21 still
training; Z=1,30 have no benchmark output yet)

- **HIT** (yield@0.1% >= 95%): Z 2,3,4,6,7,8,9,10,15,17 (10 elements)
- **Close** (partial yield / MRE 0.05-0.2%): Z 5,12,14,16,19,22,23,26 (8)
- **OFF** (yield@0.1% = 0%, MRE 0.3-0.7%): Z 11,13,18,20,24 (5)

Failure band clusters in Na-Cr (Z 11-24), heaviest at Z11 (Na), Z13 (Al),
Z18 (Ar), Z20 (Ca), and now Z24 (Cr, worst yet: mean MRE 0.70%, median 0.45%).

## Diagnosis (elements 11/13/16, generalizes to the rest of the failing band)

- LOO T-interpolation baseline hits ~1e-6 MRE -> 0.1% target is achievable;
  difficulty is purely the NN's energy-axis representation, not data coverage.
- train_mre ~= val_mre everywhere -> representation/optimization-limited,
  not overfitting.
- Residual-vs-energy is a **flat broadband floor** (~0.2-0.3% for the
  original checkpoints) across 0.1-12 keV, not concentrated at specific
  lines/edges. Strong lines and edges are fit well; the floor comes from
  high-frequency jitter in otherwise-smooth continuum regions (e.g. visible
  at 1.333 keV in element 11).
- Root cause: Fourier trunk (n_freqs=512, f_max=4000) jitter. Predicting
  log10-flux at 0.1% relative needs ~4e-4 log10 precision, ~10x tighter than
  the observed few-1e-3 jitter.
- These failing elements sit in the same 1-3 keV spectral region where their
  L/M-shell line forests are densest, and where trunk jitter was visually
  confirmed -- plausibly a joint trunk+line-head degradation in that band,
  not an isolated line-head capacity issue. Line-bin *fraction* of the grid
  does not by itself separate HIT from OFF elements (e.g. Z17 has 9% line
  bins and HITs; Z11 has 1.9% and is OFF) -- what's systematic is that OFF
  elements show *both* line MRE and continuum MRE elevated together
  (3-5x the HIT-group level), not lines breaking while continuum stays fine.

## Fixes tried so far -- none have improved results

### 1. Code changes to `spexai/train/train_adaptive.py` (implemented, verified
   syntactically, run on cluster)

- `--finetune`: freezes LR warmup (warmup=1 step) and curriculum alpha at 1.0
  instead of re-ramping from scratch, so continuation phases don't reset to
  from-scratch behavior.
- `--resume_optimizer`: reloads AdamW moment state from checkpoint if present
  (plain adamw/no mup only).
- `--points_final` / `--signal_frac_final`: linear ramp of point-sampling
  count and signal-bin loss share over training, to reduce late-training
  sampling noise.
- Root problem these fixes were meant to address: `--init_from` alone
  reloads weights correctly but silently resets LR warmup, curriculum alpha,
  optimizer moments, and sampling EMA, so a "continuation" phase behaves
  like a fresh run (confirmed via history logs: val MRE jumping back to
  ~0.3 at the start of each "warm start" phase). The new flags fix this
  mechanically -- confirmed by smooth, monotonic loss curves from step 0
  with no jump-back -- but do not fix the underlying accuracy floor.

### 2. Finetune attempt #1 on element 11: `--finetune 1 --resume_optimizer 1
   --lr 1e-4 --f_max 2000 --curriculum_frac 0.6 --points_final 4096
   --signal_frac_final 0.05` (batch defaulted to 128, not 256)

- Result: **worse** than the source checkpoint. Test overall MRE mean 0.48%
  (best val, in-run) vs 0.36% original; residual floor visibly widened to
  ~1% in spectra plots. No spectrum reached 0.1%.

### 3. Finetune attempt #2 on element 11: same as above but `--lr 1e-5`,
   `--f_max 4000` (matches original), `--batch 256` (matches original) --
   i.e. everything matched to the source run except allowing the new
   finetune/resume-optimizer machinery to run.

- Result: **still worse**. Test benchmark: mean MRE 0.52%, median 0.35%,
  vs original 0.34%/0.27%. Residual floor ~0.5-1%, worse than the ~0.2-0.3%
  original. yield@0.1% still 0%.

### Interpretation of the two negative finetune results

Two attempts at very different LRs (1e-4, 1e-5), one of them matching every
other hyperparameter to the original run exactly, both **degraded** the
checkpoint rather than improving it. This suggests the checkpoint already
sits at/near this architecture's representational ceiling for element 11,
and further gradient steps push it away from that point rather than refine
it -- most likely because the source checkpoint has no `opt_state_dict`, so
AdamW's moments start at zero and the first several thousand steps act as an
uncalibrated kick on an already-converged model even at low LR.

**Conclusion: fine-tuning on top of an existing checkpoint is not a valid way
to test hyperparameter/architecture changes** (e.g. lower `f_max`, slower
curriculum) -- the weights are already shaped around the old basis. Any such
change needs a **from-scratch run** (no `--init_from`) to be tested fairly.

## Not yet tried

- From-scratch run (no warm start) with `f_max` lowered (~1500-2000) and
  slower curriculum (`curriculum_frac` 0.3 -> 0.6), on element 11, to test
  whether the trunk basis change itself helps once trained natively rather
  than fine-tuned into.
- Increasing model capacity (`n_freqs`, `hidden`) for the failing band
  specifically.
- Longer/lower final LR anneal (`wsd_decay_frac` 0.4-0.5) from scratch.
- Line head capacity check for the densest-line elements (only ~2-11% of
  grid is line bins across the failing band).
