"""Every sampler, over one posterior, reporting one comparable result.

The bake-off compares *samplers*, so they must differ in nothing else: all of
these consume the same :class:`~spexai.inference.posterior.PoissonPosterior`
(or, for the gradient-based pair, the same
:class:`~spexai.inference.ppl.SpectrumModel` built over the same forward), and
all return a :class:`SamplerResult` carrying the same fields.

The currency of the comparison is **effective samples per forward evaluation**,
not wall-clock. Wall-clock conflates two things -- how well an algorithm mixes,
and whether it can use the batched forward. The ensemble samplers here (emcee,
zeus, UltraNest) amortise their whole walker set into one batched call, and so
does SVI, whose Monte-Carlo particles are its batch. A single-point sampler
pays for a B=1 forward per gradient, which is a structural handicap unrelated
to its mixing -- hence ``n_eval`` is counted in *walkers evaluated*, so the two
effects can be told apart.
"""
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class SamplerResult:
    """One sampler's output, in the shape the bake-off scores."""

    name: str
    names: Sequence[str]
    samples: np.ndarray            # (N, ndim), constrained space, post burn-in
    runtime_s: float
    n_eval: int                    # walkers pushed through the forward
    ess: Optional[np.ndarray] = None        # per-parameter effective samples
    logz: Optional[float] = None            # nested sampling only
    logzerr: Optional[float] = None
    chain: Optional[np.ndarray] = None      # (nsteps, nwalkers, ndim) if any
    extra: dict = field(default_factory=dict)

    @property
    def median(self) -> np.ndarray:
        return np.median(self.samples, axis=0)

    @property
    def sigma(self) -> np.ndarray:
        """Half the 16-84 interval -- robust and directly comparable to the
        Fisher sigma the bias study predicts."""
        q16, q84 = np.percentile(self.samples, [16, 84], axis=0)
        return 0.5 * (q84 - q16)

    @property
    def min_ess(self) -> float:
        """The worst-mixing parameter governs how long a run must be."""
        return float(np.nan) if self.ess is None else float(np.min(self.ess))

    @property
    def ess_per_eval(self) -> float:
        """THE bake-off metric: effective samples bought per forward walker."""
        if self.ess is None or not self.n_eval:
            return float(np.nan)
        return self.min_ess / self.n_eval

    def summary(self, truths=None) -> str:
        lines = [f"{self.name}: {len(self.samples)} samples, "
                 f"{self.runtime_s:.1f}s, {self.n_eval} evals, "
                 f"min ESS {self.min_ess:.0f}, "
                 f"ESS/eval {self.ess_per_eval:.2e}"]
        if self.logz is not None:
            lines[0] += f", logZ {self.logz:.2f} +- {self.logzerr:.2f}"
        med, sig = self.median, self.sigma
        head = f"{'param':>10} {'median':>12} {'sigma':>11}"
        if truths is not None:
            head += f" {'truth':>12} {'bias/sigma':>11}"
        lines.append(head)
        for i, n in enumerate(self.names):
            row = f"{n:>10} {med[i]:>12.5g} {sig[i]:>11.4g}"
            if truths is not None:
                t = truths[i]
                row += f" {t:>12.5g} {(med[i]-t)/sig[i]:>+11.2f}"
            lines.append(row)
        return "\n".join(lines)


def _ess(chain: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Per-parameter ESS from an ``(nsteps, nwalkers, ndim)`` chain.

    arviz wants (chain, draw, ...), so the walker axis becomes the chain axis --
    correct for ensemble samplers, whose walkers are genuinely separate (if
    correlated) chains.

    **Never raises.** ESS is a diagnostic; the chain is the result. This used to
    be evaluated inside the caller's ``return`` expression, so a missing arviz
    destroyed a completed multi-hour run at the last instant, before anything
    reached disk. A diagnostic must not be able to do that -- on any failure it
    reports NaN and says why, and the sampling still lands."""
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz not installed: ESS reported as NaN (the chain is "
                      "unaffected; `pip install arviz` to get the diagnostic)",
                      RuntimeWarning, stacklevel=2)
        return np.full(len(names), np.nan)
    try:
        posterior = {n: chain[:, :, i].T for i, n in enumerate(names)}
        return np.array([float(az.ess(az.convert_to_dataset({n: v}))[n])
                         for n, v in posterior.items()])
    except Exception as exc:                                     # noqa: BLE001
        warnings.warn(f"ESS computation failed ({exc}); reporting NaN. The "
                      f"chain is unaffected.", RuntimeWarning, stacklevel=2)
        return np.full(len(names), np.nan)


def _init_walkers(prior, nwalkers, rng, center=None, scatter=0.02):
    """Walker cloud around ``center`` (or the box middle), clipped inside."""
    lo = prior.lo.cpu().numpy()
    hi = prior.hi.cpu().numpy()
    c = 0.5 * (lo + hi) if center is None else np.asarray(center, dtype=float)
    p0 = c + scatter * (hi - lo) * rng.standard_normal((nwalkers, prior.ndim))
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


# --- gradient-free ensembles ------------------------------------------------

def run_emcee(post, nwalkers=64, nsteps=800, discard_frac=0.4, seed=0,
              center=None, progress_every=None, backend_path=None,
              resume=False) -> SamplerResult:
    """Affine-invariant ensemble MCMC, the established baseline.

    ``vectorize=True`` hands the whole (half-)ensemble to the posterior at once,
    which is what makes the batched forward pay off.

    ``backend_path`` writes the chain to HDF5 *as it advances*, so a run that
    dies at step 700 of 800 -- for any reason, including something downstream of
    the sampling itself -- leaves a usable chain behind instead of nothing.
    With ``resume=True`` an existing file is continued rather than truncated.
    Strongly recommended for multi-hour GPU jobs."""
    import emcee
    if nwalkers < 2 * post.prior.ndim:
        raise ValueError(f"emcee needs nwalkers >= 2*ndim = "
                         f"{2 * post.prior.ndim}, got {nwalkers}")
    rng = np.random.default_rng(seed)
    p0 = _init_walkers(post.prior, nwalkers, rng, center)

    backend = None
    if backend_path is not None:
        backend = emcee.backends.HDFBackend(backend_path)
        if resume and backend.iteration > 0:
            print(f"  resuming from {backend_path} at step "
                  f"{backend.iteration}", flush=True)
            # the walker state has to be read back explicitly: passing None
            # only continues a sampler that ran in *this* process (emcee looks
            # at its own _previous_state), which a fresh object does not have
            p0 = backend.get_last_sample()
            nsteps = max(0, nsteps - backend.iteration)
        else:
            backend.reset(nwalkers, post.prior.ndim)

    post.n_eval = 0
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, post.prior.ndim, post.logp,
                                    vectorize=True, backend=backend)
    every = progress_every or max(1, nsteps // 20)
    for i, _ in enumerate(sampler.sample(p0, iterations=nsteps)):
        if i == 0 or (i + 1) % every == 0:
            el = time.time() - t0
            print(f"  emcee step {i+1}/{nsteps}  {el:.0f}s "
                  f"(~{el/(i+1)*nsteps:.0f}s projected)", flush=True)
    runtime = time.time() - t0
    discard = int(discard_frac * nsteps)
    chain = sampler.get_chain()
    return SamplerResult(
        "emcee", post.prior.names, sampler.get_chain(discard=discard, flat=True),
        runtime, post.n_eval, _ess(chain[discard:], post.prior.names),
        chain=chain, extra={"acceptance": float(
            np.mean(sampler.acceptance_fraction))})


def run_zeus(post, nwalkers=64, nsteps=800, discard_frac=0.4, seed=0,
             center=None) -> SamplerResult:
    """Slice-sampling ensemble: gradient-free like emcee but typically much
    lower autocorrelation time.

    This is the control that separates two explanations of a NUTS win -- 'the
    gradients help' from 'emcee just mixes badly in this geometry'."""
    import zeus
    rng = np.random.default_rng(seed)
    p0 = _init_walkers(post.prior, nwalkers, rng, center)
    post.n_eval = 0
    t0 = time.time()
    sampler = zeus.EnsembleSampler(nwalkers, post.prior.ndim, post.logp,
                                   vectorize=True, verbose=False)
    sampler.run_mcmc(p0, nsteps)
    runtime = time.time() - t0
    discard = int(discard_frac * nsteps)
    chain = sampler.get_chain()
    return SamplerResult(
        "zeus", post.prior.names, sampler.get_chain(discard=discard, flat=True),
        runtime, post.n_eval, _ess(chain[discard:], post.prior.names),
        chain=chain)


def run_ultranest(post, min_num_live_points=400, frac_remain=0.01,
                  logdir=None, seed=0, show_status=False,
                  resume=False) -> SamplerResult:
    """Reactive nested sampling: the only sampler here that returns log Z.

    Genuinely vectorised -- the previous ``fitting.run_ultranest`` wrapper
    looped the likelihood one row at a time, which threw away the batched
    forward entirely. Since ``ptform`` guarantees points inside the box, the
    likelihood is called directly and skips the bounds check.

    ``logdir`` turns on UltraNest's own checkpointing: it writes live points and
    results as the run advances, so a job that dies partway leaves recoverable
    state instead of nothing -- the same protection the emcee HDF5 backend
    gives. ``resume=True`` continues such a run; the default overwrites, which
    is what you want for a fresh fit and is why it cannot be the default the
    other way round (a stale checkpoint silently resumed would mix two runs).
    Nested sampling has no fixed step count, so an interrupted run has no
    meaningful partial posterior -- resuming, not salvaging, is the point."""
    import ultranest
    np.random.seed(seed)
    post.n_eval = 0
    t0 = time.time()
    # 'resume' needs an existing log_dir; 'overwrite' is valid even when
    # log_dir is None (passing resume=None there used to crash)
    resume_mode = "resume" if (resume and logdir) else "overwrite"
    sampler = ultranest.ReactiveNestedSampler(
        list(post.prior.names), post.loglike, post.prior.ptform,
        vectorized=True, log_dir=logdir, resume=resume_mode)
    res = sampler.run(min_num_live_points=min_num_live_points,
                      frac_remain=frac_remain, show_status=show_status)
    runtime = time.time() - t0
    samples = np.asarray(res["samples"])
    # nested sampling's equal-weighted posterior is already decorrelated, so its
    # effective size is the sample count (arviz's autocorrelation ESS would be
    # meaningless on it)
    ess = np.full(post.prior.ndim, float(res.get("ess", len(samples))))
    return SamplerResult("ultranest", post.prior.names, samples, runtime,
                         post.n_eval, ess, float(res["logz"]),
                         float(res["logzerr"]), extra={"result": res})


# --- flow / neural-network accelerated ---------------------------------------

def _kish_ess(log_w: np.ndarray) -> float:
    """Kish effective sample size of importance weights, from log weights.

    ``(sum w)^2 / sum(w^2)``. This is the honest currency for the
    importance-based samplers: they return *weighted* draws, and quoting the
    raw draw count would flatter them enormously -- a run can return 100k
    points whose weight is concentrated on a few hundred. Computed in log space
    via a max-subtraction so the huge log-likelihoods here (order 1e6) do not
    overflow.
    """
    lw = np.asarray(log_w, dtype=float)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return float("nan")
    lw = lw - lw.max()
    w = np.exp(lw)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _resample_equal(samples: np.ndarray, log_w: np.ndarray,
                    rng: np.random.Generator, n: Optional[int] = None):
    """Weighted draws -> equal-weight draws, so the downstream median/percentile
    scoring (which is unweighted) is correct."""
    lw = np.asarray(log_w, dtype=float)
    ok = np.isfinite(lw)
    samples, lw = samples[ok], lw[ok]
    w = np.exp(lw - lw.max())
    w = w / w.sum()
    n = len(samples) if n is None else n
    return samples[rng.choice(len(samples), size=n, replace=True, p=w)]


def _uniform_box_scipy(prior):
    """Prior -> list of scipy uniform dists, for samplers that demand them.

    Raises rather than silently approximating: pocoMC takes the prior as
    distribution objects, so a non-uniform :class:`PriorSet` would have to be
    translated term by term. Quietly substituting a uniform would change the
    posterior without any error, which is exactly the kind of failure that is
    invisible in the output.
    """
    from scipy import stats
    lo = prior.lo.cpu().numpy()
    hi = prior.hi.cpu().numpy()
    priors = getattr(prior, "priors", None)
    if priors is not None:
        from .priors import Uniform
        bad = [n for n, p in zip(prior.names, priors)
               if not isinstance(p, Uniform)]
        if bad:
            raise NotImplementedError(
                f"pocoMC needs the prior as scipy distributions and this "
                f"wrapper only translates uniform boxes; {bad} are not "
                f"uniform. Extend _uniform_box_scipy rather than letting a "
                f"wrong prior through silently.")
    return [stats.uniform(loc=lo[i], scale=hi[i] - lo[i])
            for i in range(prior.ndim)]


def run_nautilus(post, n_live=2000, f_live=0.01, n_eff=10000, seed=0,
                 filepath=None, resume=False, n_networks=4,
                 verbose=False) -> SamplerResult:
    """Importance nested sampling with neural-network-regressed bounds.

    The reason to try this against UltraNest: it is *importance* nested
    sampling, so every likelihood evaluation contributes to both the posterior
    and log Z, where rejection-based nested sampling throws away everything
    below the current contour. The neural network regresses the iso-likelihood
    boundary to build proposals that track it, which is what pushes the
    sampling efficiency up.

    As with UltraNest, the prior enters through ``ptform`` and the likelihood
    is called *bare* -- adding the prior density here would apply it twice.

    ``filepath`` (an HDF5 file) enables nautilus's own checkpointing;
    ``resume=True`` continues it. Same rationale as UltraNest's ``log_dir``.
    """
    from nautilus import Sampler
    post.n_eval = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    sampler = Sampler(post.prior.ptform, post.loglike,
                      n_dim=post.prior.ndim, n_live=n_live,
                      n_networks=n_networks, vectorized=True, pass_dict=False,
                      seed=seed, filepath=filepath, resume=resume)
    sampler.run(f_live=f_live, n_eff=n_eff, verbose=verbose)
    runtime = time.time() - t0

    points, log_w, log_l = sampler.posterior()
    points = np.asarray(points)
    ess = np.full(post.prior.ndim, _kish_ess(log_w))
    samples = _resample_equal(points, log_w, rng)
    logz = float(sampler.log_z)
    return SamplerResult("nautilus", post.prior.names, samples, runtime,
                         post.n_eval, ess, logz, float("nan"),
                         extra={"points": points, "log_w": np.asarray(log_w),
                                "log_l": np.asarray(log_l),
                                "n_raw": len(points)})


def run_pocomc(post, n_effective=512, n_active=256, n_total=4096,
               n_evidence=4096, seed=0, output_dir=None, resume_path=None,
               save_every=None, flow="nsf6", progress=False) -> SamplerResult:
    """Preconditioned Monte Carlo: normalizing-flow SMC.

    A normalizing flow learns a change of variables that decorrelates the
    parameters, and the SMC moves are made in that preconditioned space. That
    targets this posterior's known weakness directly -- the strong
    abundance/normalisation degeneracy that costs the affine-invariant and
    slice ensembles so much of their efficiency.

    The prior is passed as distribution objects (pocoMC samples from it
    directly), so the likelihood is again called bare.

    ``output_dir`` + ``save_every`` write state periodically;
    ``resume_path`` continues from one.
    """
    import pocomc
    post.n_eval = 0
    rng = np.random.default_rng(seed)
    prior = pocomc.Prior(_uniform_box_scipy(post.prior))
    t0 = time.time()
    sampler = pocomc.Sampler(prior, post.loglike, n_dim=post.prior.ndim,
                             n_effective=n_effective, n_active=n_active,
                             vectorize=True, flow=flow, random_state=seed,
                             output_dir=output_dir)
    sampler.run(n_total=n_total, n_evidence=n_evidence, progress=progress,
                resume_state_path=resume_path, save_every=save_every)
    runtime = time.time() - t0

    res = sampler.posterior(return_logw=True)
    points, log_w = np.asarray(res[0]), np.asarray(res[-1])
    ess = np.full(post.prior.ndim, _kish_ess(log_w))
    samples = _resample_equal(points, log_w, rng)
    logz, logzerr = sampler.evidence()
    return SamplerResult("pocomc", post.prior.names, samples, runtime,
                         post.n_eval, ess, float(logz), float(logzerr),
                         extra={"points": points, "log_w": log_w,
                                "n_raw": len(points)})


def run_inessai(post, nlive=2000, seed=0, output=None, resume=False,
                flow_config=None, target_ess=2000.0,
                stopping_criterion="ess", **kwargs) -> SamplerResult:
    """i-nessai: importance nested sampling with normalizing flows (PyTorch).

    The only one of the accelerated samplers written in the same framework as
    the forward, so its flow trains on the *same* GPU the likelihood runs on
    with no extra device juggling.

    nessai drives the model through structured ("live point") arrays with one
    named field per parameter, so the adapter converts to the plain ``(B,
    ndim)`` block the batched forward wants. ``allow_vectorised`` is what makes
    nessai hand over a whole array at once rather than one point at a time --
    without it the batched forward is wasted, exactly as the old
    ``fitting.run_ultranest`` wrapper wasted it.

    **The stopping criterion is overridden on purpose.** nessai defaults to
    ``stopping_criterion="ratio"`` with ``tolerance=0.0``, which stops once
    ``log(Z_live / Z_all) <= 0``. For a spectrum likelihood that is true almost
    immediately -- the accumulated evidence outruns the live points within a
    couple of iterations -- so the run terminated after ~2k evaluations with
    its flow still close to the prior. The weighted output was then dominated
    by a single point: log-weights spanning 8.7e5 nats and a Kish ESS of
    **1.0**, which nessai's own ESS estimate agreed with. Stopping on ``ess``
    instead (comparison ``>=``) runs the sampler until it has actually
    accumulated ``target_ess`` effective samples, which is the quantity the
    bake-off scores anyway. Pass ``stopping_criterion="ratio"`` explicitly to
    get the old behaviour back.
    """
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model
    from nessai.livepoint import live_points_to_array

    names = list(post.prior.names)
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    span = hi - lo

    # i-nessai works in the unit hypercube, so it needs the map both ways.
    # For a uniform box that is the linear rescaling below; a general prior
    # would need its CDF, which PriorSet does not expose (it has ppf but not
    # cdf). Refuse rather than silently sampling the wrong prior.
    priors = getattr(post.prior, "priors", None)
    if priors is not None:
        from .priors import Uniform
        bad = [n for n, p in zip(names, priors) if not isinstance(p, Uniform)]
        if bad:
            raise NotImplementedError(
                f"i-nessai needs to_unit_hypercube/from_unit_hypercube and "
                f"this adapter implements only the uniform-box map; {bad} are "
                f"not uniform. Implementing it needs each prior's CDF.")

    class _Adapter(Model):
        """post -> nessai. Vectorised on both prior and likelihood."""

        allow_vectorised = True
        allow_vectorised_prior = True

        def __init__(self):
            super().__init__()
            self.names = names
            self.bounds = {n: [float(lo[i]), float(hi[i])]
                           for i, n in enumerate(names)}

        def _block(self, x):
            return np.atleast_2d(live_points_to_array(x, self.names))

        # --- unit-hypercube maps, required by the importance sampler ---------
        # Structured arrays carry bookkeeping fields (logP, logL, it, logW,
        # logQ, logU) alongside the parameters, so these copy and rewrite only
        # the named parameter fields.

        def from_unit_hypercube(self, x):
            out = x.copy()
            for i, n in enumerate(names):
                out[n] = lo[i] + x[n] * span[i]
            return out

        def to_unit_hypercube(self, x):
            out = x.copy()
            for i, n in enumerate(names):
                out[n] = (x[n] - lo[i]) / span[i]
            return out

        def log_prior_unit_hypercube(self, x):
            """Flat inside the cube -- the box prior's density is constant and
            the volume factor is already carried by the transform."""
            th = self._block(x)
            inside = np.all((th >= 0.0) & (th <= 1.0), axis=1)
            return np.where(inside, 0.0, -np.inf)

        def log_prior(self, x):
            th = self._block(x)
            out = np.full(th.shape[0], -np.inf)
            ok = post.prior.inside(th)
            # the prior DENSITY, not just the bounds check: nessai adds
            # log_prior to log_likelihood itself, so this is the one place a
            # non-uniform prior can enter for this sampler
            if ok.any():
                out[ok] = post.prior.logpdf(th[ok])
            return out

        def log_likelihood(self, x):
            return post.loglike(self._block(x))

    post.n_eval = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    fs = FlowSampler(_Adapter(), output=output, importance_nested_sampler=True,
                     resume=resume, seed=seed, nlive=nlive,
                     flow_config=flow_config,
                     stopping_criterion=stopping_criterion,
                     tolerance=target_ess, **kwargs)
    fs.run()
    runtime = time.time() - t0

    ns = fs.ns
    # Deliberately NOT fs.posterior_samples. That property rejection-samples
    # the weighted set, and with log-likelihoods spanning ~1e6 (as they do for
    # a real spectrum) the acceptance collapses -- it returned a *single* draw
    # on the analytic test problem. The weighted set itself is intact, so take
    # it and reweight explicitly, the same way nautilus and pocoMC are handled.
    points = np.atleast_2d(live_points_to_array(ns.samples, names))
    log_w = np.asarray(ns.log_posterior_weights, dtype=float)
    ess = _kish_ess(log_w)
    own = float(getattr(ns, "posterior_effective_sample_size", np.nan) or np.nan)
    if np.isfinite(own) and own > 0:
        ess = own                      # prefer the sampler's own estimate
    samples = _resample_equal(points, log_w, rng)
    logz = float(getattr(fs, "log_evidence", np.nan))
    logzerr = float(getattr(fs, "log_evidence_error", np.nan))
    return SamplerResult("inessai", names, samples, runtime, post.n_eval,
                         np.full(post.prior.ndim, ess), logz, logzerr,
                         extra={"points": points, "log_w": log_w,
                                "n_raw": len(points), "kish_ess": _kish_ess(log_w)})


# --- variational -------------------------------------------------------------

def make_guide(name, model, hidden_dim=None, num_transforms=2, rank=None):
    """Named variational family -> a Pyro autoguide.

    ``mvn`` (full-rank Gaussian) is the default and can represent a linear
    degeneracy but nothing beyond it. ``iaf`` is a normalizing flow (inverse
    autoregressive), the escalation when a Gaussian family is not enough --
    which is the live question here, since the full-rank guide recovered
    ``n_h`` 8 sigma from truth with a posterior 0.43x too narrow while eleven
    other parameters looked healthy. ``normal`` is mean-field and is included
    only as a deliberate control: it cannot represent correlation at all, so it
    brackets how much of the error is the family and how much the optimisation.
    """
    from pyro.infer import autoguide as ag
    name = (name or "mvn").lower()
    if name in ("mvn", "full_rank", "multivariate"):
        return ag.AutoMultivariateNormal(model)
    if name in ("iaf", "flow", "normalizing_flow"):
        # hidden_dim defaults to ~2x latent dim inside Pyro; num_transforms > 1
        # is what buys non-Gaussian shape
        return ag.AutoIAFNormal(model, hidden_dim=hidden_dim,
                                num_transforms=num_transforms)
    if name in ("lowrank", "low_rank"):
        return ag.AutoLowRankMultivariateNormal(model, rank=rank)
    if name in ("normal", "meanfield", "diagonal"):
        return ag.AutoNormal(model)
    raise ValueError(f"unknown guide {name!r}; choose from "
                     f"mvn, iaf, lowrank, normal")


def run_svi(model, steps=2000, num_particles=64, lr=1e-2, seed=0,
            n_posterior=4000, guide=None, progress_every=None,
            guide_hidden=None, guide_transforms=2,
            particle_chunk=None) -> SamplerResult:
    """Full-rank Gaussian variational inference on a :class:`SpectrumModel`.

    Full-rank rather than mean-field deliberately: this posterior has a known
    strong degeneracy (norm against the abundances), and a mean-field guide
    cannot represent a correlation at all -- it would report confidently wrong
    error bars. ``AutoNormal`` and ``AutoNormalizingFlow`` are drop-in swaps via
    ``guide=`` when a posterior turns out to need them.

    ``num_particles`` with ``vectorize_particles=True`` is the batch dimension
    of the forward, so a VI step costs one batched forward, not ``num_particles``
    of them.

    **``particle_chunk`` decouples memory from particle count.** That same
    batching is why VI OOMs: peak memory scales linearly with
    ``num_particles``, and ``echunk`` cannot help because it bounds the trunk's
    activations, not the number of particles held at once. Lowering
    ``num_particles`` to fit is the wrong trade -- it degrades the ELBO
    gradient estimate, and a noisy gradient is the leading suspect for a
    variational posterior that lands in the wrong place. Instead this splits
    the particles into sub-batches, calls backward on each, and accumulates
    into the same ``.grad`` before stepping. Since Pyro's ``Trace_ELBO``
    averages over particles, each chunk is weighted by its share, so the
    accumulated gradient is *identical in expectation* to the unchunked one --
    memory scales with ``particle_chunk``, statistical quality with
    ``num_particles``. The cost is one extra kernel launch sequence per chunk,
    which is negligible against the forward."""
    import pyro
    import torch
    from pyro.infer import Trace_ELBO
    from pyro.infer.autoguide import AutoMultivariateNormal

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    if guide is None:
        guide = AutoMultivariateNormal(model)
    elif isinstance(guide, str):
        guide = make_guide(guide, model, hidden_dim=guide_hidden,
                           num_transforms=guide_transforms)
    # particle sub-batches: sizes sum to num_particles, each weighted by its
    # share so the accumulated gradient matches the unchunked one
    pc = int(particle_chunk or num_particles)
    pc = max(1, min(pc, num_particles))
    chunks = [pc] * (num_particles // pc)
    if num_particles % pc:
        chunks.append(num_particles % pc)
    elbos = {n: Trace_ELBO(num_particles=n, vectorize_particles=True)
             for n in set(chunks)}
    if len(chunks) > 1:
        print(f"  svi: {num_particles} particles in {len(chunks)} chunks of "
              f"<= {pc} (memory scales with the chunk, gradient quality with "
              f"the total)", flush=True)

    model.n_eval = 0
    every = progress_every or max(1, steps // 20)

    t0 = time.time()
    # one-off: instantiates the guide's parameters before the optimiser sees them
    elbos[chunks[0]].differentiable_loss(model, guide)
    opt = torch.optim.Adam(guide.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        opt.zero_grad()
        total = 0.0
        for n in chunks:
            # weight by share: Trace_ELBO averages over its particles, so
            # sum_c (n_c/P) * ELBO_c == ELBO over all P particles
            loss = elbos[n].differentiable_loss(model, guide) * (n / num_particles)
            loss.backward()                # accumulates into guide.grad
            total += float(loss.detach())
        opt.step()
        losses.append(total)
        if i == 0 or (i + 1) % every == 0:
            el = time.time() - t0
            print(f"  svi step {i+1}/{steps}  ELBO {-losses[-1]:.4g}  "
                  f"{el:.0f}s (~{el/(i+1)*steps:.0f}s projected)", flush=True)
    runtime = time.time() - t0

    with torch.no_grad():
        draws = [guide() for _ in range(n_posterior)]
    samples = np.stack([[float(d[n]) for n in model.names] for d in draws])
    # a variational posterior is exact-iid by construction: every draw is
    # independent, so ESS is the draw count. That does NOT mean it is accurate
    # -- the approximation error lives in the shape, not the correlation, which
    # is exactly why it is scored against the gold chain and not on ESS alone.
    ess = np.full(len(model.names), float(n_posterior))
    return SamplerResult("svi", model.names, samples, runtime, model.n_eval,
                         ess, extra={"losses": np.array(losses),
                                     "final_elbo": -losses[-1]})


def run_nuts(model, n_samples=1000, n_warmup=1000, target_accept=0.8,
             max_tree_depth=10, seed=0, progress=False, full_mass=False,
             init_values=None, jit_compile=False) -> SamplerResult:
    """No-U-Turn Sampler, via Pyro, on a :class:`SpectrumModel`.

    Pyro handles the constrained-to-unconstrained transforms and their
    Jacobians from the priors declared on the model, so nothing about the
    geometry is hand-rolled here.

    **Read its ``ess_per_eval`` with the batching in mind.** Pyro's NUTS is a
    single-point sampler: every leapfrog step is one gradient at ``B=1``, while
    emcee, zeus and UltraNest amortise a whole ensemble into each batched
    forward, and SVI does the same with its particles. On a GPU a ``B=1``
    forward costs nowhere near 1/96 of a ``B=96`` one, so NUTS can need far
    fewer evaluations and still lose on wall-clock. That is a property of how
    it is run, not of how it mixes -- which is exactly why both numbers are
    reported. (A chain-vectorised NUTS would close the gap; one was attempted
    and abandoned as too error-prone to trust for calibration work.)

    Three settings matter far more here than the defaults suggest:

    ``full_mass`` -- Pyro defaults to a **diagonal** mass matrix, which cannot
    represent parameter correlation at all. This posterior has a strong
    abundance/normalisation degeneracy, and the marginal scales are *not* the
    problem (their unconstrained-space condition number is only ~79, worth
    ~9 leapfrog steps). With a diagonal matrix NUTS has no way to correct the
    ridge and the trajectories blow up: a production run was observed
    saturating ``max_tree_depth=10`` (1023 leapfrog steps) on every iteration,
    ~21x longer than the ~50 steps at which gradients start paying for
    themselves against the batched ensembles. For 12 parameters a dense mass
    matrix is essentially free.

    ``init_values`` -- Pyro's default ``init_to_uniform`` starts from a random
    draw of the prior box. Every other sampler in the bake-off starts at the
    truth, so leaving this unset both handicaps NUTS and, in a posterior this
    sharp (~44 nats of information gain), guarantees enormous early
    trajectories. Pass the truth vector.

    ``max_tree_depth`` -- the cost ceiling. Each unit is a doubling, so 10
    allows 1023 gradients per iteration and 7 allows 127. Worth capping while
    the mass matrix adapts; note Pyro's first adaptation window does not close
    until roughly iteration 75, so a run judged before then is being read at
    its worst moment.
    """
    import pyro
    import torch
    from pyro.infer import MCMC, NUTS

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    model.n_eval = 0

    kernel_kw = {}
    if init_values is not None:
        from pyro.infer.autoguide.initialization import init_to_value
        if not isinstance(init_values, dict):
            init_values = dict(zip(model.names,
                                   np.asarray(init_values, dtype=float).ravel()))
        device = getattr(model.forward, "device", None)
        # float64 to match the Uniform priors' dtype; a float32 init makes Pyro
        # promote inconsistently and the transforms then disagree
        kernel_kw["init_strategy"] = init_to_value(values={
            n: torch.tensor(float(v), dtype=torch.float64, device=device)
            for n, v in init_values.items()})

    kernel = NUTS(model, target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, full_mass=full_mass,
                  jit_compile=jit_compile, **kernel_kw)
    mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup,
                num_chains=1, disable_progbar=not progress)
    t0 = time.time()
    mcmc.run()
    runtime = time.time() - t0

    draws = mcmc.get_samples()
    samples = np.stack([np.asarray(draws[n]).reshape(-1) for n in model.names],
                       axis=-1)                              # (N, ndim)
    chain = samples[:, None, :]                              # one chain
    diag = mcmc.diagnostics() or {}
    # The number that decides whether NUTS is viable here: gradients per
    # iteration. n_eval counts every theta row pushed through the forward, so
    # this is a direct measurement of the trajectory length rather than an
    # inference from wall-clock. Break-even against the batched ensembles is
    # ~50; saturating max_tree_depth means adaptation has not taken hold.
    steps_per_iter = model.n_eval / max(1, n_samples + n_warmup)
    ceiling = 2 ** max_tree_depth - 1
    print(f"  nuts: {steps_per_iter:.0f} leapfrog steps/iteration "
          f"(ceiling {ceiling}"
          f"{', SATURATED' if steps_per_iter > 0.9 * ceiling else ''})",
          flush=True)
    return SamplerResult("nuts", model.names, samples, runtime, model.n_eval,
                         _ess(chain, model.names), chain=chain,
                         extra={"divergences": int(np.sum(
                             diag.get("divergences", {}).get("chain 0", []))
                             ) if diag else 0,
                             "steps_per_iter": float(steps_per_iter),
                             "tree_depth_ceiling": int(ceiling),
                             "full_mass": bool(full_mass)})
