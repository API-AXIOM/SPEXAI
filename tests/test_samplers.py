"""Every sampler must recover the same posterior on a problem with a known one.

A bake-off is only meaningful if the samplers agree when they should. These run
all four on a small Poisson problem whose posterior is tight and near-Gaussian,
and check they land on the same answer -- so a later *disagreement* on the real
spectrum is a property of the geometry, not of the plumbing.

Deliberately tiny and analytic: no emulator, so these stay fast enough to run
on every commit.
"""
import numpy as np
import pytest
import torch

from spexai.inference.posterior import BoxPrior, PoissonPosterior
from spexai.inference import samplers

NDIM, NCHAN = 3, 40
TRUTH = np.array([0.6, 0.35, 0.75])


class _SmoothForward:
    """counts = exp(A @ theta) with a well-conditioned A: a smooth, strictly
    positive forward giving a unimodal Poisson posterior that every sampler
    should nail."""

    device = "cpu"
    names = [f"p{i}" for i in range(NDIM)]

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.A = (0.5 + torch.rand(NDIM, NCHAN, generator=g,
                                   dtype=torch.float64))

    def counts_torch(self, th, grad=False):
        return torch.exp(th.double() @ self.A + 3.0)

    def __call__(self, theta):
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float64)
        return self.counts_torch(th).detach().cpu().numpy()


@pytest.fixture(scope="module")
def post():
    fwd = _SmoothForward()
    mu = fwd(TRUTH[None, :])[0]
    data = np.random.default_rng(0).poisson(mu).astype(np.float64)
    prior = BoxPrior(np.zeros(NDIM), np.ones(NDIM), fwd.names)
    return PoissonPosterior(fwd, data, prior)


@pytest.fixture(scope="module")
def reference(post):
    """A long emcee run: the gold posterior the others are scored against."""
    return samplers.run_emcee(post, nwalkers=32, nsteps=1500, seed=1,
                              progress_every=10**9)


def _agrees(res, ref, n_sigma=0.5):
    """Medians within a fraction of the reference sigma, widths within 40%."""
    dmed = np.abs(res.median - ref.median) / ref.sigma
    dsig = np.abs(res.sigma - ref.sigma) / ref.sigma
    return dmed.max() < n_sigma, dsig.max() < 0.4, dmed, dsig


def test_emcee_recovers_the_truth(reference):
    # the injected truth must sit inside the posterior it generated
    pull = np.abs(reference.median - TRUTH) / reference.sigma
    assert pull.max() < 3.0, f"pulls {pull}"


def test_result_metrics_are_populated(reference):
    assert reference.ess is not None and reference.min_ess > 0
    assert reference.n_eval > 0
    assert np.isfinite(reference.ess_per_eval)
    assert "emcee" in reference.summary(truths=TRUTH)


def test_out_of_box_walkers_are_rejected_not_evaluated(post):
    # emcee proposes outside the box constantly; those rows must never reach
    # the forward, which is the entire cost on the real problem
    post.n_eval = 0
    lp = post.logp(np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5]]))
    assert lp[1] == -np.inf and post.n_eval == 1


def test_zeus_agrees_with_emcee(post, reference):
    res = samplers.run_zeus(post, nwalkers=32, nsteps=600, seed=2)
    med_ok, sig_ok, dmed, dsig = _agrees(res, reference)
    assert med_ok, f"median differs by {dmed} sigma"
    assert sig_ok, f"width differs by {dsig}"


def test_ultranest_agrees_with_emcee_and_returns_logz(post, reference):
    pytest.importorskip("ultranest")
    res = samplers.run_ultranest(post, min_num_live_points=200)
    med_ok, sig_ok, dmed, dsig = _agrees(res, reference)
    assert med_ok, f"median differs by {dmed} sigma"
    assert sig_ok, f"width differs by {dsig}"
    assert res.logz is not None and np.isfinite(res.logz)
    assert res.logzerr is not None and res.logzerr > 0


def test_svi_agrees_with_emcee(post, reference):
    pytest.importorskip("pyro")
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    model = SpectrumModel(post.forward, post.data_np,
                          uniform_priors(post.forward.names, lo, hi))
    res = samplers.run_svi(model, steps=600, num_particles=16, lr=2e-2,
                           seed=3, progress_every=10**9)
    # VI is an approximation, so it gets a looser bar than the exact samplers:
    # a full-rank Gaussian should still track a near-Gaussian posterior well
    med_ok, sig_ok, dmed, dsig = _agrees(res, reference, n_sigma=1.0)
    assert med_ok, f"median differs by {dmed} sigma"
    assert dsig.max() < 0.6, f"width differs by {dsig}"


def test_svi_elbo_improves(post):
    pytest.importorskip("pyro")
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    model = SpectrumModel(post.forward, post.data_np,
                          uniform_priors(post.forward.names, lo, hi))
    res = samplers.run_svi(model, steps=300, num_particles=16, lr=2e-2,
                           seed=4, progress_every=10**9)
    losses = res.extra["losses"]
    assert losses[-20:].mean() < losses[:20].mean(), "ELBO did not improve"


def test_nuts_agrees_with_emcee(post, reference):
    pytest.importorskip("pyro")
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    model = SpectrumModel(post.forward, post.data_np,
                          uniform_priors(post.forward.names, lo, hi))
    res = samplers.run_nuts(model, n_samples=400, n_warmup=400, seed=5)
    med_ok, sig_ok, dmed, dsig = _agrees(res, reference)
    assert med_ok, f"median differs by {dmed} sigma"
    assert sig_ok, f"width differs by {dsig}"
    assert res.extra["divergences"] == 0, "divergent transitions"


def test_nuts_evaluates_one_point_at_a_time(post):
    # documents the structural handicap the bake-off has to report: NUTS gets
    # no benefit from the batched forward, unlike every other sampler here
    pytest.importorskip("pyro")
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    lo = post.prior.lo.cpu().numpy()
    hi = post.prior.hi.cpu().numpy()
    seen = []
    fwd = post.forward
    orig = fwd.counts_torch

    def spy(th, grad=False):
        seen.append(int(th.shape[0]))
        return orig(th, grad=grad)

    fwd.counts_torch = spy
    try:
        model = SpectrumModel(fwd, post.data_np,
                              uniform_priors(fwd.names, lo, hi))
        samplers.run_nuts(model, n_samples=20, n_warmup=20, seed=6)
    finally:
        fwd.counts_torch = orig
    assert seen and set(seen) == {1}, f"expected all B=1, saw {set(seen)}"


def test_emcee_rejects_too_few_walkers(post):
    with pytest.raises(ValueError, match="nwalkers >="):
        samplers.run_emcee(post, nwalkers=4, nsteps=10)
