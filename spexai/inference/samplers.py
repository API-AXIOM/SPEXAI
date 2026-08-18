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
    correlated) chains."""
    import arviz as az
    posterior = {n: chain[:, :, i].T for i, n in enumerate(names)}
    return np.array([float(az.ess(az.convert_to_dataset({n: v}))[n])
                     for n, v in posterior.items()])


def _init_walkers(prior, nwalkers, rng, center=None, scatter=0.02):
    """Walker cloud around ``center`` (or the box middle), clipped inside."""
    lo = prior.lo.cpu().numpy()
    hi = prior.hi.cpu().numpy()
    c = 0.5 * (lo + hi) if center is None else np.asarray(center, dtype=float)
    p0 = c + scatter * (hi - lo) * rng.standard_normal((nwalkers, prior.ndim))
    return np.clip(p0, lo + 1e-9, hi - 1e-9)


# --- gradient-free ensembles ------------------------------------------------

def run_emcee(post, nwalkers=64, nsteps=800, discard_frac=0.4, seed=0,
              center=None, progress_every=None) -> SamplerResult:
    """Affine-invariant ensemble MCMC, the established baseline.

    ``vectorize=True`` hands the whole (half-)ensemble to the posterior at once,
    which is what makes the batched forward pay off."""
    import emcee
    if nwalkers < 2 * post.prior.ndim:
        raise ValueError(f"emcee needs nwalkers >= 2*ndim = "
                         f"{2 * post.prior.ndim}, got {nwalkers}")
    rng = np.random.default_rng(seed)
    p0 = _init_walkers(post.prior, nwalkers, rng, center)
    post.n_eval = 0
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, post.prior.ndim, post.logp,
                                    vectorize=True)
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
                  logdir=None, seed=0, show_status=False) -> SamplerResult:
    """Reactive nested sampling: the only sampler here that returns log Z.

    Genuinely vectorised -- the previous ``fitting.run_ultranest`` wrapper
    looped the likelihood one row at a time, which threw away the batched
    forward entirely. Since ``ptform`` guarantees points inside the box, the
    likelihood is called directly and skips the bounds check."""
    import ultranest
    np.random.seed(seed)
    post.n_eval = 0
    t0 = time.time()
    sampler = ultranest.ReactiveNestedSampler(
        list(post.prior.names), post.loglike, post.prior.ptform,
        vectorized=True, log_dir=logdir, resume="overwrite")
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


# --- variational -------------------------------------------------------------

def run_svi(model, steps=2000, num_particles=64, lr=1e-2, seed=0,
            n_posterior=4000, guide=None, progress_every=None) -> SamplerResult:
    """Full-rank Gaussian variational inference on a :class:`SpectrumModel`.

    Full-rank rather than mean-field deliberately: this posterior has a known
    strong degeneracy (norm against the abundances), and a mean-field guide
    cannot represent a correlation at all -- it would report confidently wrong
    error bars. ``AutoNormal`` and ``AutoNormalizingFlow`` are drop-in swaps via
    ``guide=`` when a posterior turns out to need them.

    ``num_particles`` with ``vectorize_particles=True`` is the batch dimension
    of the forward, so a VI step costs one batched forward, not ``num_particles``
    of them."""
    import pyro
    import torch
    from pyro.infer import Trace_ELBO
    from pyro.infer.autoguide import AutoMultivariateNormal

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    guide = guide if guide is not None else AutoMultivariateNormal(model)
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    model.n_eval = 0
    every = progress_every or max(1, steps // 20)

    t0 = time.time()
    loss = elbo.differentiable_loss(model, guide)      # one-off: builds params
    opt = torch.optim.Adam(guide.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        opt.zero_grad()
        loss = elbo.differentiable_loss(model, guide)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
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
             max_tree_depth=10, seed=0, progress=False) -> SamplerResult:
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
    """
    import pyro
    from pyro.infer import MCMC, NUTS

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    model.n_eval = 0
    kernel = NUTS(model, target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, jit_compile=False)
    mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup,
                num_chains=1, disable_progbar=not progress)
    t0 = time.time()
    mcmc.run()
    runtime = time.time() - t0

    draws = mcmc.get_samples()
    samples = np.stack([np.asarray(draws[n]).reshape(-1) for n in model.names],
                       axis=-1)                              # (N, ndim)
    chain = samples[:, None, :]                              # one chain
    return SamplerResult("nuts", model.names, samples, runtime, model.n_eval,
                         _ess(chain, model.names), chain=chain,
                         extra={"divergences": int(np.sum(
                             mcmc.diagnostics().get("divergences", {}).get(
                                 "chain 0", []))) if mcmc.diagnostics() else 0})
