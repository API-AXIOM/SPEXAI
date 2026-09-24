# spexai working notes

Running log of context, progress and open questions. Decisions and their
reasoning go in `DECISIONS.tex`.

## IN PROGRESS 2026-09-22: inference API unification (before P8)

Refactor to collapse the two parallel inference stacks into one user-facing
API, so P8 does not add a sixth copy of the posterior-assembly block. Design:
`docs/superpowers/specs/2026-09-22-inference-api-unification-design.md`.
Worktree `.claude/worktrees/inference-api-unification`, branch
`worktree-inference-api-unification`, **uncommitted**.

**COMPLETE. Baseline 296 passed -> now 305 passed, 0 failed.** The count
reconciles exactly: `296 + 11` new equivalence tests `- 2` tests whose subject
(`run_emcee`'s scalar fallback) no longer exists.

Phase status: **0-6 all DONE.** Left uncommitted for D.H. to review.

> **Do not commit `spexai/models`** in this worktree -- it is a SYMLINK to the
> main checkout's model store, created only so the tests would run here. See
> the `MODELS_DIR` finding below for why it was needed.

### What a user writes now

```python
from spexai.inference import SpectralFit, PriorSet, Uniform, Normal

fit = SpectralFit(
    emulator=emu, response=resp,
    counts=counts, exposure=1.2e5,        # loaded externally; no loader here
    priors=PriorSet({"kT":       Uniform(1.5, 7.5),
                     "Fe":       Normal(0.55, 0.05, low=0.0),
                     "sigma_v":  Uniform(30.0, 600.0),
                     "log_norm": Uniform(10.0, 12.0)}),
    abundances=ab, absorption=absn, n_h_scale=1e21,
    redshift=0.0173, band=(1.9, 12.0), device="cuda")

res = fit.sample("nautilus", n_live=2000)   # any of the nine
print(res.summary())
```

`.posterior` still hands out the bare `PoissonPosterior` for the campaign
scripts and the gradient paths.

Five findings, all outside the refactor proper:

1. **macOS OpenMP root-caused; `KMP_DUPLICATE_LIB_OK=TRUE` is withdrawn.** The
   cause is import order: NumPy claims the env's `libomp` via its BLAS, and if
   torch is imported first its bundled copy wins, so the next library to load
   the env's copy aborts. `spexai/inference/__init__.py` now imports NumPy
   first; the flag is removed from 15 scripts, 2 docs and the notebook, where
   it was already unnecessary (every script imports NumPy before torch). Do not
   reintroduce it -- it suppresses a "may silently produce incorrect results"
   diagnostic rather than fixing the conflict.

2. **`operator_model.MODELS_DIR` ignores `SPEXAI_STORE`.** It was hardcoded
   relative to the package, unlike `spexai.config.STORE` which does honour the
   variable. Consequence: in any checkout without `spexai/models/` (e.g. a
   fresh worktree) **all 25 model-dependent tests skip silently** and the suite
   still reports success -- a green run that tested none of the inference code.
   **FIXED 2026-09-23.** Three changes:
   - `operator_model.MODELS_DIR` is now `spexai.config.STORE`, so `SPEXAI_STORE`
     moves it. The default is unchanged (`spexai/models/` inside the package),
     so an unset variable behaves exactly as before. This reaches all 16 call
     sites at once -- they all import `MODELS_DIR` from `operator_model`.
   - The silent skip is now loud: `tests/conftest.py` ends a run with a red
     `INCOMPLETE RUN: artifacts missing` banner naming the missing artifact and
     where it looked, for the model store and for the ACIS response. Skipping
     is still the default so a fresh checkout can run the suite.
   - `SPEXAI_REQUIRE_MODELS=1` escalates a missing model store to a hard
     `pytest.UsageError` (exit 4) before collection. Use it for CI and for any
     campaign run that must not quietly test nothing.
   The worktree symlink workaround is removed; run the suite here with
   `SPEXAI_STORE=/Users/danielahuppenkothen/work/repositories/spexai/spexai/models`.

3. **`run_emcee` is NOT reproducible from its `seed` argument.** The seed only
   controls walker initialisation (`np.random.default_rng(seed)` for `p0`);
   emcee's own proposals draw from NumPy's **global** RNG, which neither
   `fitting.run_emcee` nor `samplers.run_emcee` seeds. Of the 9 samplers only
   `ultranest` and `pocomc` seed anything. Two identical `seed=0` runs give
   different chains. The golden generator works around it by calling
   `np.random.seed()` immediately before the run. **This affects reproducibility
   of every emcee result in the campaign, including the planned P8 runs.**
   **FIXED 2026-09-22 on D.H.'s instruction:** both `run_emcee`s now pin
   NumPy's global legacy RNG across emcee's construction (the point at which
   emcee snapshots its own generator) and restore it afterwards, so a caller's
   own RNG stream -- including the Poisson draw that makes the data -- does not
   become a function of which sampler ran first. `seed` now reproduces the
   chain. `run_zeus` and the rest are still unseeded; only emcee was asked for.

4. **The two stacks disagreed on `n_h` UNITS, silently.**
   `fitting.build_posterior` passed `n_h_scale=1.0`, so `n_h` was sampled in
   absolute cm^-2 (`Uniform(0, 5e21)`). The campaign relied on
   `VectorForward`'s default of `1e21` and sampled it in units of 1e21
   (`Par("n_h", p["n_h"] / 1e21, ..., 0.0, 5.0)`). Either default in the shared
   builder mis-scales the other convention's absorption by 10^21 -- which
   raises nothing, it just yields an absurd spectrum. Measured on the golden
   `dem_abs` case: the wrong scale gives a 0.999 relative deviation in logp.
   **`SpectralFit` therefore has no default**: if `n_h` is a fitted parameter
   or a non-zero fixed value, `n_h_scale` must be passed explicitly, or it
   raises. Phase 5 must state the convention at every campaign call site.

5. **`logp` is now offset by the prior normalisation.** `BoxPrior.logpdf`
   returned 0 inside the box -- an unnormalised prior -- while `PriorSet`
   returns a real density, so `logp` gains exactly `-sum(log(hi - lo))`
   (-8.9847 for the `single` case, -60.3221 for `dem_abs`). Constant, so it
   cancels in every acceptance ratio and moves no posterior sample; UltraNest
   is untouched because it scores through `loglike`. Pinned by
   `test_prior_is_now_normalised` so it stays deliberate. Consequence for the
   gate: the equivalence test compares **`loglike`**, not `logp` -- and the
   frozen `logp` IS the bare likelihood, precisely because the old prior
   contributed zero.

### Phase 1 artefact

`tests/data/refactor_golden.npz`, written by
`scripts/inference/make_refactor_golden.py` from the **pre-refactor** code.
Two cases -- `single` (3 elements, single-T) and `dem_abs` (Gaussian DEM +
absorption + per-walker `n_h`, the configuration `tier_c_mcmc.py` had wired
wrong). Each stores `theta` (64 pts), `logp`, `p0`, `chain` (50x16), plus
names/bounds/truth/counts. `--check` regenerates and asserts array equality:
currently **25 arrays identical**.

### Phase 3 artefact

`spexai/inference/spectral_fit.py` -- `SpectralFit`, the single assembly point,
exported from `spexai.inference` along with `PriorSet`/`Uniform`/`Normal`/
`LogUniform`. Counts + exposure + response in as arrays (no `Observation`, no
loader); `PriorSet` for the prior; `dem=` first class; `band=(lo,hi)` or
`keep=`; `.posterior` for the bare object; `.sample(name, **kw)` over all nine
samplers via the new `samplers.SAMPLERS` registry (gradient samplers get a
Pyro `SpectrumModel` instead, via `GRADIENT_SAMPLERS`). `kT`/`sigma_v` are
canonical; `temp`/`velocity` still work and emit a `DeprecationWarning`.

Gate: `tests/test_refactor_equivalence.py`, **11 passed** --
`loglike` matches the frozen reference at rtol 1e-12 in both cases, the seeded
emcee chain matches at rtol 1e-10 in both, walker init is bit-identical, the
prior offset is exactly the normalisation constant, legacy names warn, and
`n_h_scale` is required when `n_h` is fitted.

The walker-init clip difference the spec flagged (`lo + 1e-6` in `fitting` vs
`lo + 1e-9` in `samplers`) turned out to be **untriggered**: no walker starts
outside the box for either case, so both give identical `p0`. `samplers` was
therefore left alone rather than changing campaign behaviour for no effect.

### Phase 4: user-facing consumers migrated

* `fit_plots.py` now takes `SamplerResult`, so it works for all nine samplers
  rather than two. **Signature change: `truths=` is passed in, not read off the
  result.** A posterior does not know the true answer; making the result carry
  `truths` forced every real fit to invent one. `plot_corner_overlay` and
  `plot_posterior_predictive` also take a `labels=` pair, so they are no longer
  hardwired to the emcee/UltraNest combination.
* `perseus_showcase.py`, `bias_study.py`, `run_inference_demo.py` build a
  `SpectralFit` and call `.sample(...)`. The showcase needed `n_h_scale=1.0`
  (its `PERSEUS["n_h"]` is absolute, 1.4e21, reaching the forward via `fixed`).
* `tutorials/inference_walkthrough.ipynb` migrated (11 lines changed), including
  the DEM section, and now documents non-uniform priors via `PriorSet` and the
  `n_h_scale` requirement.
* **`fitting.build_posterior` is now a thin adapter over `SpectralFit`** rather
  than a second assembly. This was the high-leverage move: it converts all 27
  `build_posterior` call sites in `test_fitting.py`, `test_contracted.py` and
  `test_dem_fast.py` into tests of the new path **without editing them**, so
  the scalar-vs-vectorised reference checks now guard `SpectralFit` directly.
  Those 33 tests pass unchanged, across DEM, absorption, abundance models and
  the contracted forward -- 27 independent constructions agreeing with the
  scalar reference.

### Phase 5: campaign scripts migrated

* **`tier_c_mcmc.py` -- the P8 blocker is GONE.** It builds a `SpectralFit`
  with `dem=gaussian_logT_dem()[0]` under `--mode dem`. The old bug (a
  `VectorForward` with no `dem=` while passing `logT_mean`/`logT_sigma`) is now
  unrepresentable: there is one assembly point and the DEM travels with the
  parameters. **Not yet run against real data** -- needs the jsonl + npz pair.
* **`bake_off.py`** migrated (D.H. approved). Its numbers must not move; the
  change is a like-for-like substitution of the same assembly.
* Mechanical `BoxPrior` -> `PriorSet` swap in `benchmark_ppl.py`,
  `mle_reseed.py`, `p6_sweep.py` and 5 test files (`PriorSet.box` reproduces
  `BoxPrior`'s exact constructor, names defaulting to `p0, p1, ...`).
* **`n_h_scale=1e21` stated explicitly** at every campaign call site, since
  `build_pars` emits `Par("n_h", point["n_h"], 1e-2, 0.0, 6.0)` -- units of
  1e21, not absolute.
* `SpectralFit` now accepts counts either full-length or already band-restricted
  (the campaign's cached truths are `d_ref[keep]`); the length picks the
  meaning and anything else raises.

### Phase 6: duplicated machinery deleted

Gone: `fitting.run_emcee`, `fitting.run_ultranest`, `EmceeResult`,
`UltranestResult`, `fitting._resolve_posterior`, `fitting.progress_flag`,
`posterior.BoxPrior` (154 + 74 lines).

**Kept, on D.H.'s instruction:** `fitting.build_posterior`, now a ~15-line
adapter over `SpectralFit` that emits a `DeprecationWarning`. It keeps 27 test
call sites and any external caller working, and pins `n_h_scale=1.0` because
that entry point has always taken `n_h` in absolute cm^-2.

Also kept: `make_loglike` (the scalar reference), `Param`, `SIGMA_V_PRIOR`,
`vectorization_blocker`.

**Coverage changes.** Two tests lost their subject and were removed:
`test_run_emcee_vectorised_and_scalar_agree` and
`test_run_emcee_warns_and_falls_back_on_dem`, both of which tested
`run_emcee`'s automatic fallback to the scalar likelihood -- `SpectralFit` is
batched only. The per-walker `loglike` agreement tests are strictly more
sensitive than a 6-step chain comparison at rtol 1e-3, so the real loss is
small. `test_dem_run_emcee_does_not_warn` was **rewritten**, not deleted, as
`test_dem_fit_runs_vectorised_end_to_end`. In `test_priors.py`, the two tests
that existed to prove `PriorSet == BoxPrior` were rewritten against the
analytic uniform expressions, which is a stronger statement than agreeing with
a second implementation.

`scripts/inference/make_refactor_golden.py` can no longer run (it needs the
deleted `run_emcee`). That is by design -- it is kept as the provenance record
for `tests/data/refactor_golden.npz`, and exits with an explanation pointing at
`tests/test_refactor_equivalence.py`, which needs nothing from it.

## PROJECT RULES

Standing conventions, to be applied as code is touched and swept up in full at
the next repo refactor / clean-up.

### 1. Name code for what it does

Scripts and modules are named for their function, not for our internal campaign
numbering (no `tier_a_*`, `tier_c_*`, `p6_*`, `p8_*`). The tier/phase labels stay
in the prose, where they mean something; a filename has to be readable by
someone who has never seen the agenda. Renamed 2026-09-23:

| was | is |
|---|---|
| `scripts/inference/tier_c_mcmc.py` | `scripts/inference/emulator_bias_posterior_check.py` |
| `scripts/inference/tier_a_composition.py` | `scripts/inference/emulator_error_composition.py` |

Historical mentions of the old names were deliberately NOT rewritten where they
record a past bug (the design spec, `spectral_fit.py`, `make_refactor_golden.py`,
`test_refactor_equivalence.py`) -- the bug really was in a file with that name.
Those carry a pointer to the new name instead.

Still to do under this rule: `p6_sweep.py`, `p6_probe.py`, `p7_screen_diag.py`,
`plot_p7_sweep.py` and friends. Not renamed yet -- they are referenced from the
written-up P6/P7 sections and a rename should happen in one pass with the tex.

### 2. Docstrings are numpy-style, with full parameter and return sections

Every public function, method and class gets a numpy-style (numpydoc) docstring
with explicit `Parameters` and `Returns` sections. For each entry, give:

* the **name**,
* its **type**,
* its **default**, where it has one (write it as `optional` with the value, e.g.
  ``n_eff : int, optional`` ... ``Default is 1000.``),
* a **short description of what the variable actually is** -- not a restatement
  of its name. `keep : ndarray of bool` is not documentation; "in-band channel
  mask, as returned by `band_mask`" is.

`Raises` where the function raises deliberately, and `Notes`/`Examples` where
the reasoning needs somewhere to live.

    def scale_to_counts(rec, target_counts):
        """Rescale a screen ratio from the sweep's reference level.

        Parameters
        ----------
        rec : dict
            One record from a `bias_sweep` jsonl. Must carry ``n_ref``, the
            count level its ``b_sys``/``sigma_ref`` were computed at.
        target_counts : float
            In-band counts the posterior check actually injects.

        Returns
        -------
        float
            Multiplicative factor taking a stored ratio to `target_counts`.
            ``b_sys`` grows linearly with counts and sigma as its square root,
            so the ratio scales as ``sqrt(target_counts / n_ref)``.
        """

**Why this and not the current style.** The existing docstrings are discursive
-- they explain *why* the code is the way it is, often very usefully, and that
prose should be kept (move it to `Notes`). What they systematically do not do is
say what the arguments are, so a caller has to read the body to find out that
`keep` is a boolean mask, that `n_h` is in units of 1e21, or that `counts` may
be full-length or band-restricted and the length silently picks the meaning.
That last one is exactly the class of bug that cost us the n_h scale-convention
split. Types and units in the signature documentation are the cheap guard.

State of play: **no module in the package currently complies.** This is a
whole-repo sweep, not a per-file fix, and it wants a linter pinning it
(`pydocstyle --convention=numpy`, or ruff's `D` rules with
`convention = "numpy"`) so it cannot silently rot afterwards. Decide the linter
at the same time as the sweep, or the sweep is a one-off.

### 3. Formatting: black, 88 columns, applied per file as it is touched

`black` is installed in the `spexai` env and configured in `pyproject.toml`
(line length 88, black's default; `spexai/deprecated/` and `fit_old.py`
excluded). `pyproject.toml` carries **tool config only** -- there is
deliberately no `[build-system]` table, so `setup.py` remains the build backend
and the editable install is unaffected (verified 2026-09-23).

**The repo is NOT formatted and is not being swept now.** Format a file when you
are editing it for some other reason; do not reformat files you are not
otherwise touching.

Why not now: black would rewrite **157 of 159 files, ~25,900 lines** at 88
columns. That flattens `git blame` across the whole campaign at exactly the
moment P8 starts producing results whose provenance we will want to trace, and
it would mean the code running on the cluster differs from what was reviewed
and tested. The files touched in the 2026-09-23 session were therefore left
unformatted on purpose -- including the P8 driver, which is about to be run.

**Deferred: the whole-repo sweep happens once the scientific results are
finished, before the library is released.** Do it together with rule 2's
docstring pass and its linter, in one commit that touches nothing else, and add
the commit SHA to a `.git-blame-ignore-revs` file so `git blame` stays useful.

## RESUME HERE: P8 -- posterior confirmation of the bias screen

**P7 is COMPLETE and written up** (both flavours, 1000 points, seed 39235);
P6 before it. Everything below the P8 block is history, kept for reference.

**The old DEM blocker is GONE** (the `VectorForward` built with no `dem=`);
`SpectralFit` made it unrepresentable. The driver has still never been RUN in
DEM mode against real data, so the first DEM invocation is also its first test.

### PROBE RESULT 2026-09-24: INVALID, and why (log_norm bug)

The nautilus probe of single-T point 310 ran, but **its parameter results are
not usable**. Two findings, one fatal and one about cost.

**1. `log_norm`'s truth was never rescaled with the data. FIXED.**
`build_point_problem` rescaled the injected spectrum to `--target_counts` but
passed the sweep's `log_norm_truth` through untouched. Since `norm =
10**log_norm` and the campaign injects 1e6 against a screen recorded at
`n_ref` 1e5, the log_norm the data implies is `truth + 1.000` -- and
`build_pars` gives log_norm the box `[truth-1, truth+1]`, so the required value
lands **exactly on the upper bound**. Measured: truth 15.911002, ceiling
16.911002, posterior median 16.907575 with sigma 0.003386, i.e. pinned
**1.01 sigma below the wall**; offset 0.9966 = log10(9.92). The fit could not
raise the normalisation, so it **absorbed the shortfall into the abundances**:

| | |
|---|---|
| inflated abundances | Si +1.98, S +1.64, Ar +3.31, Ca +1.19, Fe +1.46 sigma, all high |
| coverage | 2/12 parameters contain truth |
| also prior-saturated | Ni (1.01 sigma from the 3.0 ceiling), Mn (1.49 sigma from the 0.02 floor) |
| log_norm pull | +294 sigma (meaningless) |

Fixed by shifting the truth with the data,
`log_norm_truth += log10(target_counts / n_ref)`, which also re-centres the box
so it cannot bind. **Every point must be re-run.** This bug predates the rename
(it was in `tier_c_mcmc.py`) and would have silently corrupted the whole of P8.

**The one result that survives:** `sigma_v`, the binding parameter and the only
one with a meaningful screen prediction, came in at pull **+2.85 vs screen
+2.81, k = 1.01**. sigma_v is a line-width parameter, largely orthogonal to
normalisation, which is why it escaped. Encouraging for P8's actual question,
but it is one parameter at one point from an otherwise invalid fit.

**2. `--n_eff 1000` did NOT throttle nautilus.** It returned **ESS 24,998** --
the same as the bake-off's 24,729 at `n_eff=10000` -- in **17.4 h** (62,646 s,
295,600 evals). `n_eff` is a floor, not a target: the prior-to-posterior
compression alone already yields ~25k effective samples, so the run is governed
entirely by `f_live` and n_eff is satisfied long before it can stop.
**Correction to the 2026-09-23 estimate**, which guessed "below 12.64 h": it
came in *above* the bake-off's time, not below.

Consequence for the sampler choice at ESS ~1000: nautilus cannot be bought
cheaply. emcee at ~11.6 h/point is now the cheaper route for a summary-only
campaign; nautilus costs ~17 h and delivers 25x more ESS than needed, plus
logZ. Either is ~190-280 GPU-h for 16 points. To actually throttle nautilus,
`f_live` is the knob, not `n_eff` -- untested.

**Diagnostics.** `scripts/inference/plot_bias_posterior_check.py` builds
pull / k / interval plots from the jsonl (figures in `docs/figures/biascheck_*`).
It flags any parameter whose posterior sits within 3 sigma of a known prior
bound -- that check is what caught this bug, and it should be read before any
pull is believed. Corner plots need `--save_samples` (added 2026-09-24); the
jsonl carries summary statistics only, so the probe has no corner plot.

### Settled 2026-09-23

| question | decision |
|---|---|
| points | 4 worst + 4 random per flavour, 8 per flavour, 16 total |
| exclusions | single-T drops 629 and 155 (Mn design artifacts). **Per-flavour** -- DEM point 155 is a genuine target |
| region cut | NOT needed. With 629/155 dropped the global ranking IS the cold+narrow corner |
| count level | inject at **1e6**; the sweep stores b_sys/sigma at `n_ref`=1e5, ratio scales as sqrt(N), factor 3.162 |
| noise | Poisson, one realisation per point |
| k | compare RAW: record `pull`, `pull_screen` and `k = pull/pull_screen`, then compare the set against P6's Gauss-Newton k |
| sampler | target ESS ~1000; `--sampler {emcee,nautilus}` both wired |

The selected points (verified against the screens, ratios shown at 1e6):

* single-T worst: **310** (2.81), **243** (2.73), **799** (2.67), **63** (2.56)
  -- all kT 0.74-0.95 keV, sigma_v 34-97 km/s, all binding on `sigma_v`.
* DEM worst: **456** (2.05), **155** (1.98), **82** (1.94), **734** (1.79)
  -- all kT 0.72-1.09 keV, sigma_v 42-117 km/s, all binding on `sigma_v`.
* random halves are drawn with `--seed`; seed 0 gives single-T 271/511/848/636
  and DEM 270/511/848/636. The same index is a DIFFERENT physical point in the
  two sweeps. DEM 848 happens to land at 1.49 sigma, near the region -- an
  honest draw, and a useful intermediate case.

### Sampler: the cost is a near-tie at ESS 1000

From the bake-off table (one 1e6-count Perseus spectrum, 12 params, A10):

* `emcee` delivers 24 ESS/10^3 s, so ESS 1000 costs **~11.6 h/point**.
* `nautilus` delivered ESS 24,729 in 12.64 h at its default `n_eff`=10000. It
  DOES have a throttle (`run_nautilus(..., n_eff=)`), contrary to the
  methodology text's claim that nested samplers "cannot be asked for fewer
  draws" -- **that sentence needs softening for nautilus.** But lowering
  `n_eff` does not buy proportionally: nested sampling still pays the full
  prior-to-posterior compression (~44 nats) and only the top-up shrinks. The
  n_eff=1000 cost is therefore **unmeasured, bounded above by 12.64 h**.

So the two are within ~10% at worst, and nautilus additionally returns logZ and
is the showcase sampler. **Run one point as a timing probe before committing
the other 15** (~190 GPU-h at 12 h/point).

### What P8 needs on disk

| what | where |
|---|---|
| screens | `~/work/data/spexai/results/bias_sweep/bias_{single,dem}_n1000_s39235.jsonl` |
| truths (BOTH jsonl + npz from the SAME run) | `.../truth_{single,dem}_n1000_s39235.npz` |
| driver | `scripts/inference/emulator_bias_posterior_check.py` |
| screening/diagnostic helper | `scripts/inference/p7_screen_diag.py` |
| figures | `docs/figures/p7_*.png`, written by `scripts/inference/plot_p7_sweep.py` |

Both p7 helpers ARE tracked in git (an earlier note here said untracked; wrong).

### Still open

1. **Perseus showcase wants both MCMC and nested sampling.**
   `scripts/inference/perseus_showcase.py` has no `--sampler` at all -- it is
   hardwired. Needs the same treatment as the driver. Not done.
2. **One Poisson realisation gives ~35-50% on a single point's k** (a ~1 sigma
   scatter on a ~2-3 sigma effect). Four points per flavour averages that to
   ~20%. If k needs to be tighter, the fix is an Asimov (noise-free) companion
   run, which was considered and deliberately not taken.

## P6 open threads (history, not P8 blockers)

Open, deliberately not chased, in the order they would matter:

1. **The scatter in k is unattributed** (+-0.11 DEM, +-0.20 single-T) between
   b_sys's own error (FD Jacobian + Fisher solve -- random, averages away over
   a sweep) and genuine second-order curvature (structured, would need carrying
   per parameter). Discriminator: does |k-1| shrink as |b_sys|/sigma grows? It
   does for single-T (rho = -0.38, p < 1e-4, log-log slope +0.39, from
   `p6_trends`); not run for DEM.
2. **The optimiser fixes D2 + D5** (see below). Required before any rerun of
   this stage; not required for the current result.
3. **The cold end**: whether b_sys itself is trustworthy below ~1.5 keV is
   still open (2026-09-16 section). It did NOT affect convergence -- the three
   coldest points (0.765, 0.878, 0.955 keV) all converged cleanly, and the
   three failures sit at 0.851, 2.043 and 6.187 keV, i.e. scattered.

To rerun either screen (truth -> bias -> GN, resumable per stage, ~3.4 h for 30
single-T points on an A10):

```bash
export MKL_THREADING_LAYER=GNU
NPOINTS=30 SEED=3 MODE=single GN_ITER=20 COUNTS=1e9 NSEEDS=8 \
    nohup bash scripts/inference/p6_overnight.sh > logs/p6_single.log 2>&1 &
```

`p6_overnight.sh` defaults `GN_ITER` to 12 and silently overrides p6_sweep's
default of 20, so pass it explicitly. A fresh NPOINTS is a DIFFERENT Latin
hypercube, not an extension.



**Both P6 screens are DONE, analysed and written up.** The answer is that the
linearised screen is trustworthy: **k = 1 to within ~10%, with a real tail to
~1.5x.**

| screen | points | k median | sd | central 90% |
|---|---|---|---|---|
| DEM log-T | 30 (0 failed) | 0.992 | 0.11 | 0.84-1.17 |
| single-T 0.7-15 keV | 30 (3 failed) | 0.992 | 0.20 | 0.75-1.27 |

Write-ups: `docs/inference_methodology.tex` sec:kwide (single-T widened),
sec:kdem (DEM), and sec:kcaveats. Choices: `DECISIONS.tex` 2026-09-17 and
2026-09-18. Results: `~/work/data/spexai/results/` (`p6_gn_single_n30_s3.jsonl`,
`p6_gn_dem_logT_n30_s3.jsonl`) and `logs/`.

**Next: P7.** Design FIXED 2026-09-18: **1000 points per flavour, seed 39235**
(drawn at random, recorded in `DECISIONS.tex`). `sample_points(n, mode, seed)`
gives a fresh hypercube per (n, seed), so both numbers must be quoted together;
result files carry `n1000_s39235`. Carry k PER PARAMETER, not one scalar. Quote the worst case as
~1.5x in magnitude (k^2 ~ 2.4 at single-T point 11), which REPLICATES the old
100-point design's 2.75 rather than exceeding it -- the tail is a stable
property of the linearisation, and no design coordinate predicts it in advance.

### P7 step 1 DONE 2026-09-18: batched Jacobian in the bias stage

Branch `p7-batched-jacobian`, uncommitted. `bias_sweep.py --stage bias` now
takes `--jacobian {serial,batched}`; `batched` folds `--point_chunk` points'
2n+1 stencils into ONE `mle_reseed.tierb_forward` call and runs the SAME
algebra afterwards (`fisher_bias.fisher_from_jacobian`, extracted from
`linear_bias_fisher` so there is one implementation, not two). `serial` stays
the default so pre-2026-09 commands reproduce; it prints a cost warning above
50 points. `mle_reseed.batched_jacobian` now accepts per-ROW steps, which the
sweep needs because a single-T point's kT step is `kT ln10 STEP_DEX` and so
varies point to point.

**Parity PASSES in both flavours** (2 fresh points each, current design,
laptop CPU, `scripts/inference/check_jacobian_parity.py`):

| flavour | max db/sigma | max rel d sigma_ref | max rel d cond(F) |
|---|---|---|---|
| single-T | 5.68e-4 | 1.49e-4 | 3.26e-4 |
| DEM log-T | 4.99e-4 | 2.45e-4 | 5.96e-5 |

That is the float32-vs-float64 level and nowhere near anything that moves an
N*. 296 tests pass (unchanged).

**The measured cost, CPU to CPU (laptop, 1 point per call):**

| flavour | serial | batched | ratio |
|---|---|---|---|
| single-T | 46.8 s/pt | 45.7 s/pt | 1.02x |
| DEM log-T | 3705 / 5785 s/pt | 333 s (first, builds the table) then 8.7 s | ~665x steady state |

Single-T sees nothing on CPU: it is compute-bound, not call-bound, so the
batching only removes per-call overhead and the real win has to come from GPU
parallelism (unmeasured here -- that is the 5-10 point timing run, still to do).
**The DEM number is the story.** The serial path goes through
`JointOperatorModel.predict_counts_dem`, which has NO trunk/line table, NO
contract-before-broadening and NO kinematic grouping -- all of those live in
`BatchedJointForward`/`VectorForward` and were never wired into the Fisher
stage, which is why P6 saw 980 s -> 12 s per iteration and this stage did not.
The old 235-285 s/point DEM figure is also not comparable: it was the retired
linear-T 48-node grid, and the grid is now 70 nodes.

Consequence for the P7 cost model: on the serial path the DEM flavour alone
would be ~1600 h at 1000 points, not the ~73 h the agenda assumed. The port is
load-bearing for DEM, not an optimisation.

### P7 step 2 DONE 2026-09-21/22: truth stage, both flavours, 1000 points

`truth_single_n1000_s39235.npz` (laptop, 6 threads, 2.25 h) and
`truth_dem_n1000_s39235.npz` (cluster) both complete. The single-T npz
validated: counts (1000, 60000), all finite, no zero-in-band rows, 30 elements,
`rsl_Hp_L_2025.rmf` + `rsl_extflat5_GVC_2025.arf`. It lives on the LAPTOP and
must be rsynced to `~/data/spexai_data/results/bias_sweep/` for the bias stage.

Measured truth-stage rates (per element, 1000 points), for future costing:

| run | s/element-point | 30-element total |
|---|---|---|
| laptop single-T, 6 threads, idle | 0.267 (flat 266-272 s/element) | 2.25 h |
| cluster DEM, CONTENDED | 2.95 | 24.6 h |

The DEM/single-T intrinsic ratio is ~1.6x (2-point runs), so most of the 11x
gap between those rows was contention from unrelated code on the node, not the
flavour. Per-element cost is flat across all 30 elements, so the first element's
line predicts the total.

### P7 step 3: single-T bias stage COMPLETE (1000 points), screened 2026-09-22

`bias_single_n1000_s39235.jsonl` (on the laptop too). At 1e6 in-band counts
every parameter's MEDIAN bias is <=0.23 sigma; Fe binds in the median
(0.233 sigma, median N* 1.8e7), then kT (0.205) and log_norm (0.184).

**File is clean** (`scripts/inference/p7_screen_diag.py`, NEW + untracked):
1000 records, 0 duplicates, 0 missing.

**Conditioning is NOT the story.** cond(F) p50 2.7e7, p99 4.9e8, max 3.4e9 --
ZERO points above COND_F_WARN=1e10. Every tail below is physics.

**The tail is sigma_v at COLD temperatures, and nothing else.** 77/1000 points
exceed 1 sigma on sigma_v (max 2.81). Spearman: kT rho=-0.705 (p~1e-151),
sigma_v itself rho=-0.533. The sign is NEGATIVE -- cold and narrow-lined is
worse, not hot and broad:

| kT band | points | >1 sigma | max b/sigma |
|---|---|---|---|
| 0.7-1.0 | 117 | 44 (37.6%) | 2.81 |
| 1.0-1.5 | 132 | 28 (21.2%) | 2.36 |
| 1.5-2.5 | 167 | 5 (3.0%) | 1.30 |
| 2.5-5.0 | 225 | 0 | 0.81 |
| 5.0-15  | 359 | 0 | 0.71 |

The whole tail sits below 1.89 keV. Above 2.5 keV NOTHING exceeds 1 sigma on
any parameter. Physically consistent: the band starts at 1.87 keV, so a sub-2
keV plasma is in-band only through its exponential tail and sigma_v rests on a
few narrow lines, where a line-shape error is large in sigma units.

**This is the cold-end question (open item 3) made concrete** -- but P6 already
covers the ratio there: k = 0.96-1.02 at 0.765/0.878/0.955 keV, and N* depends
only on b/sigma, which is what k validates. So the tail is most likely real.

**Other tails, all tiny:** Mn 2 points (5.29, 4.11 sigma) ordered by a_Mn
(rho +0.204) and low a_Fe (-0.123), NOT by cond(F) (rho 0.028, p=0.37) --
a genuine second mode, high Mn against a weak Fe anchor. Fe 4 points, S 2, kT 1.

### P7 step 4: DEM bias stage COMPLETE (1000 points), screened 2026-09-22

`bias_dem_n1000_s39235.jsonl`. Clean file (1000/0 dupes/0 missing), cond(F)
p50 5.9e7, max 2.3e9, ZERO above 1e10.

**The DEM flavour is uniformly BETTER than single-T**, which was not the prior
expectation (the P1a ARF redo had made DEM the tighter case):

| param | DEM median | single-T median | DEM max | single-T max |
|---|---|---|---|---|
| Fe | 0.105 | 0.233 | 0.624 | 1.279 |
| thermal | 0.099 / 0.102 (logT_mean/sigma) | 0.205 (kT) | 0.498 / 0.383 | 1.182 |
| sigma_v | 0.083 | 0.131 | 2.047 | 2.813 |
| Mn | 0.015 | 0.030 | 0.375 | 5.292 |

Averaging over a temperature distribution smooths the emulator's
temperature-local error; there is no DEM analogue of single-T's Mn outlier.

**Same tail structure, weaker.** Only sigma_v exceeds 1 sigma anywhere:
31/1000 points, max 2.047 (the global worst over BOTH flavours and all
parameters). Spearman: sigma_v rho=-0.613, logT_mean rho=-0.542 -- cold and
narrow again.

| kT_mean band | points | >1 sigma | max |
|---|---|---|---|
| 0.7-1.0 | 117 | 17 (14.5%) | 2.05 |
| 1.0-1.5 | 132 | 13 (9.8%) | 1.94 |
| 1.5-2.5 | 166 | 1 (0.6%) | 1.06 |
| 2.5-5.0 | 227 | 0 | 0.76 |
| 5.0-15  | 358 | 0 | 0.53 |

**Containment confound, named not chased:** the tail points are also the least
contained (median 0.844 vs the design's 0.956, min 0.543) because a cold
Gaussian in log T runs off the 0.7 keV grid floor. Truncation is identical in
truth and emulator so b_sys stays a fair comparison, but the cold tail points
are partly a different object from the well-contained bulk.

**P7 headline, both flavours:** above ~2.5 keV NOTHING exceeds 1 sigma at 1e6
counts, on any parameter, in either flavour. All residual risk is cold
(<1.9 keV single-T, <1.7 keV DEM) and concentrated in sigma_v.

### P7 figures BUILT 2026-09-22

`scripts/inference/plot_p7_sweep.py` (NEW, untracked) -> `docs/figures/`:

| fig | file | label | what it carries |
|---|---|---|---|
| A | `p7_safety_map.png` | fig:p7safety | worst b/sigma vs kT, coloured by who binds |
| B | `p7_bias_by_param.png` | fig:p7params | all 12/13 distributions, >1 sigma called out |
| C | `p7_nstar_vs_temperature.png` | fig:p7nstar | the same as an exposure limit |
| D | `p7_mn_outliers.png` | fig:p7mn | Mn binned medians in (a_Mn, a_Fe) + the 2 points |

Colour is by ROLE and capped at 3 hues (sigma_v blue, Fe orange, thermal aqua,
everything else muted grey) because scatter puts all pairs on screen and only
the first three palette slots validate under that condition. Same mapping in
all four figures. Bias is RAW, not k-corrected -- figure C carries the
worst-case k=1.54 (N*/2.4) as an annotated arrow instead.

Figure D started as a per-point scatter and was REBUILT as binned medians: 1000
dots of nearly the same blue read as noise and the rho=+0.20 ordering was
invisible one point at a time.

### P7 figure E 2026-09-22: the tail IS confined, in 2-3 axes only

`p7_corner_single.png` / `p7_corner_dem.png` -- full design-space corner, all
1000 points grey, the >1 sigma points red, marginals on the diagonal.
Quantified by `p7_screen_diag.py --confinement` (2-sample KS, tail vs bulk,
per axis, + the tail's share of the lowest design quartile):

| axis | single-T | DEM |
|---|---|---|
| temperature | KS 0.684, p 2e-36, 86% in lowest quartile | KS 0.753, p 5e-18, 97% |
| sigma_v | KS 0.566, p 4e-24, 70% | KS 0.650, p 7e-13, 81% |
| logT_sigma | -- | KS 0.461, p 2e-06, 58% (narrow DEMs) |
| a_S | KS 0.218, p 1e-03, 8% (INVERTED: high S) | KS 0.405, p 6e-05, 6% |
| the other 8 abundances + n_h | p 0.03-0.95, share ~20-30% = no confinement | same |

So the answer is yes and it is narrow: **cold + kinematically narrow**, plus a
weak preference for HIGH sulphur, and the DEM adds narrow width. Every other
abundance and n_h is consistent with the design's own uniform marginal, i.e.
the tail is not an abundance-corner effect. The single-T Mn pair is a separate
thing and does NOT live in this region (it is a_Mn-high/a_Fe-low, see fig D).

### P7 CLOSED 2026-09-22 -- written up

`docs/inference_methodology.tex` has a new `sec:p7sweep` inside `sec:biassweep`:
design + seed, the batched-Jacobian cost argument and its parity check, both
1000-point tables (`tab:p7single`, `tab:p7dem`), the confinement table
(`tab:p7confine`), the physical-realism discussion, and all five figures
(`fig:p7safety`, `fig:p7params`, `fig:p7nstar`, `fig:p7corner`, `fig:p7mn`).
`DECISIONS.tex` has two new entries (raw-not-k-corrected figures; the two
failure modes treated differently). Both compile; 0 undefined references on the
second pass. NOTE the methodology PDF only builds from INSIDE `docs/`, since
the figure paths are relative -- the old P1c issue, still open.

**Left undone on purpose:** the per-parameter k correction is still not applied
in code (`--apply_k` never written). The text carries the worst case ~1.5x
instead. If P8 comes back consistent with the screen, that may be all it needs.

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
| `scripts/inference/emulator_error_composition.py` | imports `bias_sweep.RANGES` / `sample_points` (single-T) | a RERUN would now draw kT log-uniform over 0.7-15 keV, not 1.5-8 keV |

`campaign.Forward` still produces bit-identical names and weights for the
linear-T DEM (pinned by `test_forward_linear_T_hot_floor_path_unchanged`).

### Smoke runs PASSED 2026-09-16 (2 DEM points, cluster)

Both stages ran end to end under the log-T parametrisation; output downloaded
to `~/work/data/spexai/results/bias_sweep_logT_smoke/`. Verified locally:
`dem_param='logT'` stamped; `contained` in BOTH npz and jsonl and equal to an
independent recomputation (0.7619, 0.9990); names/order carry
`logT_mean`/`logT_sigma`; the drawn points reproduce `sample_points(2,'dem',0)`
exactly; 30 elements; correct Resolve RMF/ARF; counts finite. Numbers are
unremarkable: worst |b|/sigma at N_REF=1e5 is +0.057 (Fe, pt0) and +0.109
(sigma_v, pt1); cond(F) 9.4e7 and 6.0e6, far below COND_F_WARN=1e10. Bias
stage ~235-285 s/point. Point 0 (1.22 keV, 0.354 dex) is 76% contained -- the
cold-end truncation the design allows, behaving as predicted.

**GOTCHA (pre-existing, not introduced here): the bias jsonl has 4 records for
2 points.** `stage_bias` opens the jsonl with `"a"` and only skips work when
`--resume` is passed, so a second invocation without `--resume` APPENDS a
duplicate set, and `summarise()` does not deduplicate -- it would double-count
every repeated point. **Always pass `--resume` on a rerun**, and check
`len(records)` against `--n_points` before reading a summary.

## GN bugs D1-D4: status 2026-09-18

Read against two real screens: DEM log-T (30 points) and single-T 0.7-15 keV
(30 points), 120 fits each.

- **D1 vacuous convergence: SUPERSEDED by D4.** The `resolved.any()` guard was
  correct for the 2026-09-10 failure but inert against the real problem, since
  `resolved` uses `se_delta`, which in noiseless mode is optimiser
  repeatability (~1.5e-5 sigma), not an error bar -- so it was true 13/13 at
  every point. D4's absolute floor subsumes it, and the precondition has been
  REMOVED from the criterion (`n_resolved` is kept as a diagnostic).
- **D2 not descending: REOPENED 2026-09-18.** Closed on 2026-09-17 on DEM
  evidence; the single-T screen refutes that closure. Three points exit through
  the stall detector while pinned at the 20 sigma clamp, and at two of them
  -logL is RISING at exit (pt 0: -2.149e4 -> -2.141e4 -> -2.119e4). GN does go
  uphill, but only when clamp-saturated -- which the DEM screen never was.
  Fix identified, NOT implemented: accept a step only on decrease, else halve
  along the Newton direction.
- **D3 spread criterion: RETIRED in practice.** Start spread is 0.003-0.048
  sigma at converged points and the start-up warning never fires. NOTE the
  warning is now nearly toothless anyway: at `--gn_iter 20` with
  `--gn_max_step 20` it only triggers above a 400 sigma start spread.
- **D5 (NEW): the trust region is isotropic.** `over = |d/sig|.max() /
  max_step_sigma; d /= over` rescales the WHOLE step vector by its worst
  component, so one runaway direction throttles every other direction by the
  same factor. Signature: direction-dependence within one fit -- at pt 0 the
  sigma_v direction converged to 1.7% of its initial across-start scatter while
  S never moved at all (99.4%), same eight starts. At pt 18 several
  thick-denominator directions ended FURTHER apart than they began. Fix
  identified, NOT implemented: clamp per component.

**D2 and D5 are the two to apply before any rerun of this stage.** They were
deferred because k replicates across three independent samples and recovering
3 points in 30 would not change it.

### D4: every threshold divided by something that vanishes -- FIXED 2026-09-18

- `measurable_mask(b_sys, se_d, sigma)` (extracted from `run_point` so it is
  testable) requires `|b_sys| > max(3 se_d, K_FLOOR_SIGMA * sigma)`,
  K_FLOOR_SIGMA = 0.5.
- `convergence_verdict` judges drift and start spread against
  `max(bias, CONV_FLOOR_SIGMA)`, CONV_FLOOR_SIGMA = 1.0, same 10% convention.
  The `np.where(resolved, ..., 0.0)` masks -- the mechanism of the vacuous
  score -- are gone.
- Verified: 296 tests (293 + 3 new in `tests/test_gauss_newton.py`). On the
  stored single-T run the new rule gives 3 NOT CONVERGED where the old gave 9,
  no indeterminate cases, and `measurable` 347/360 -> 301/360.
- The two `*_frac_of_bias` key names are KEPT although their denominator is now
  floored, because `p6_trends` and `p6_check_outliers` consume them to exclude
  points and the floor makes those exclusions strictly better.

**The jsonls on disk were written by the OLD code**, so their `measurable` and
`converged` fields are pre-D4. Every number in the write-ups comes from
applying the current rule downstream of the stored `b_sys`, `sigma`,
`drift_sigma` and `start_spread_sigma`. Re-running the sweep is the only way to
make the stored flags agree with the text; not worth it for this result.


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
