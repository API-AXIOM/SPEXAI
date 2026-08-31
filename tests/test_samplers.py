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


def test_hmc_agrees_with_emcee(post, reference):
    res = samplers.run_hmc(post, nwalkers=16, n_samples=400, n_warmup=400,
                           n_leapfrog=15, seed=7)
    med_ok, sig_ok, dmed, dsig = _agrees(res, reference)
    assert med_ok, f"median differs by {dmed} sigma"
    assert sig_ok, f"width differs by {dsig}"
    assert 0.0 < res.extra["accept_rate"] < 1.0


def test_hmc_walker_chunk_matches_unchunked(post):
    # chunking a batch of independent walkers is EXACT, unlike run_svi's
    # particle_chunk (which must reweight since particles are averaged into
    # one shared guide gradient) -- same seed, same result bit-for-bit
    kwargs = dict(nwalkers=12, n_samples=15, n_warmup=15, n_leapfrog=6, seed=11)
    unchunked = samplers.run_hmc(post, **kwargs)
    chunked = samplers.run_hmc(post, walker_chunk=4, **kwargs)
    np.testing.assert_allclose(chunked.chain, unchunked.chain, rtol=1e-10)
    assert chunked.extra["step_size"] == unchunked.extra["step_size"]


def test_hmc_walker_chunk_bounds_the_batch_size(post):
    # regression guard for the OOM fix: with walker_chunk set, no single
    # counts_torch call should ever see more rows than the chunk
    nwalkers, chunk = 12, 5
    seen = []
    fwd = post.forward
    orig = fwd.counts_torch

    def spy(th, grad=False):
        seen.append(int(th.shape[0]))
        return orig(th, grad=grad)

    fwd.counts_torch = spy
    try:
        samplers.run_hmc(post, nwalkers=nwalkers, n_samples=5, n_warmup=5,
                         n_leapfrog=3, walker_chunk=chunk, seed=12)
    finally:
        fwd.counts_torch = orig
    assert seen and max(seen) <= chunk, (
        f"expected every batch <= {chunk}, saw max {max(seen)}")


def test_hmc_batches_the_forward(post):
    # the regression guard this sampler exists to have: the mirror image of
    # test_nuts_evaluates_one_point_at_a_time -- every leapfrog gradient must
    # be one batched call over ALL chains, never B=1
    nwalkers = 12
    seen = []
    fwd = post.forward
    orig = fwd.counts_torch

    def spy(th, grad=False):
        seen.append(int(th.shape[0]))
        return orig(th, grad=grad)

    fwd.counts_torch = spy
    try:
        samplers.run_hmc(post, nwalkers=nwalkers, n_samples=10, n_warmup=10,
                         n_leapfrog=5, seed=8)
    finally:
        fwd.counts_torch = orig
    assert seen and set(seen) == {nwalkers}, (
        f"expected every gradient batched at B={nwalkers}, saw {set(seen)}")


def test_hmc_n_eval_matches_formula(post):
    # n_eval must equal (n_warmup + n_samples) * (n_leapfrog + 1) * nwalkers --
    # confirms the accounting claim in run_hmc's docstring directly rather
    # than trusting it
    n_warmup, n_samples, n_leapfrog = 5, 5, 4
    for nwalkers in (8, 16):
        post.n_eval = 0
        samplers.run_hmc(post, nwalkers=nwalkers, n_samples=n_samples,
                         n_warmup=n_warmup, n_leapfrog=n_leapfrog, seed=9)
        expected = (n_warmup + n_samples) * (n_leapfrog + 1) * nwalkers
        assert post.n_eval == expected, (nwalkers, post.n_eval, expected)


def test_hmc_rejects_non_finite_proposals_without_crashing(post):
    # a deliberately huge step size drives divergent trajectories; these must
    # be rejected, not raise
    res = samplers.run_hmc(post, nwalkers=8, n_samples=20, n_warmup=20,
                           n_leapfrog=10, step_size_init=50.0, seed=10)
    assert res.samples.shape[0] > 0
    assert 0.0 <= res.extra["accept_rate"] <= 1.0
    assert np.isfinite(res.samples).all()


def test_emcee_rejects_too_few_walkers(post):
    with pytest.raises(ValueError, match="nwalkers >="):
        samplers.run_emcee(post, nwalkers=4, nsteps=10)


def test_ultranest_writes_checkpoints(post, tmp_path):
    # a log_dir must actually produce recoverable state on disk: the point of
    # checkpointing is that an interrupted multi-hour run is not a total loss
    pytest.importorskip("ultranest")
    d = tmp_path / "un"
    res = samplers.run_ultranest(post, min_num_live_points=100, logdir=str(d))
    assert d.exists(), "no log_dir created"
    written = [p for p in d.rglob("*") if p.is_file()]
    assert written, "log_dir is empty -- nothing was checkpointed"
    assert res.logz is not None and np.isfinite(res.logz)


def test_ultranest_resume_continues_an_existing_run(post, tmp_path):
    # resume=True must pick the existing checkpoint up rather than silently
    # starting over (which would look identical from the outside)
    pytest.importorskip("ultranest")
    d = tmp_path / "un_resume"
    first = samplers.run_ultranest(post, min_num_live_points=100, logdir=str(d))
    again = samplers.run_ultranest(post, min_num_live_points=100, logdir=str(d),
                                   resume=True)
    # a resumed run reuses the finished state, so it costs far fewer new
    # likelihood evaluations than the original
    assert again.n_eval < first.n_eval, (
        f"resume evaluated {again.n_eval} vs {first.n_eval} fresh -- it "
        f"restarted instead of resuming")
    assert np.isclose(again.logz, first.logz, atol=0.5)


def test_ultranest_without_logdir_still_runs(post):
    # log_dir=None must stay valid (resume='overwrite' with no dir used to crash)
    pytest.importorskip("ultranest")
    res = samplers.run_ultranest(post, min_num_live_points=100, logdir=None)
    assert np.isfinite(res.logz)


def test_emcee_backend_streams_the_chain(post, tmp_path):
    # THE protection that was missing when a completed run was lost: the chain
    # must be on disk as the run advances, not only after it returns
    h5 = tmp_path / "chain.h5"
    res = samplers.run_emcee(post, nwalkers=16, nsteps=40, seed=0,
                             backend_path=str(h5), progress_every=10**9)
    assert h5.exists(), "no HDF5 backend written"
    import emcee
    back = emcee.backends.HDFBackend(str(h5), read_only=True)
    assert back.iteration == 40
    # what is on disk must be the chain that was returned, not a stub
    assert np.allclose(back.get_chain(), res.chain)


def test_emcee_backend_resume_continues(post, tmp_path):
    h5 = tmp_path / "chain_resume.h5"
    samplers.run_emcee(post, nwalkers=16, nsteps=20, seed=0,
                       backend_path=str(h5), progress_every=10**9)
    res = samplers.run_emcee(post, nwalkers=16, nsteps=50, seed=0,
                             backend_path=str(h5), resume=True,
                             progress_every=10**9)
    import emcee
    back = emcee.backends.HDFBackend(str(h5), read_only=True)
    assert back.iteration == 50, (
        f"resumed run ended at {back.iteration}, expected 50 -- it restarted "
        f"or double-counted instead of continuing")
    assert res.chain.shape[0] == 50


def test_ess_survives_missing_arviz(post, monkeypatch):
    # the failure that destroyed a real run: a missing diagnostic dependency
    # must degrade to NaN, never propagate out and take the chain with it
    import builtins
    real_import = builtins.__import__

    def no_arviz(name, *a, **kw):
        if name == "arviz":
            raise ImportError("simulated missing arviz")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_arviz)
    with pytest.warns(RuntimeWarning, match="arviz not installed"):
        res = samplers.run_emcee(post, nwalkers=16, nsteps=20, seed=0,
                                 progress_every=10**9)
    assert np.all(np.isnan(res.ess))          # diagnostic degraded
    assert res.samples.shape[0] > 0           # chain intact
    assert np.isnan(res.ess_per_eval)
