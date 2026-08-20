"""The Pyro model must target the same density as the hand-rolled posterior,
and must actually batch its particles.

The batching claim is the load-bearing one. SVI's whole advantage over a
single-point sampler here is that ``vectorize_particles=True`` turns
``num_particles`` into the forward's batch dimension. If the particle plate
silently failed to reach the forward, the ELBO would evaluate one particle
``num_particles`` times, cost the same, and still converge -- just to a noisier
gradient. Nothing in the loss curve would show it.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("pyro")

from spexai.inference.posterior import BoxPrior, PoissonPosterior  # noqa: E402
from spexai.inference.ppl import SpectrumModel, uniform_priors     # noqa: E402

NDIM, NCHAN = 4, 25


class _RecordingForward:
    """Cheap forward that records the batch shape it is called with."""

    device = "cpu"

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.w = torch.rand(NDIM, NCHAN, generator=g, dtype=torch.float64)
        self.names = [f"p{i}" for i in range(NDIM)]
        self.seen = []

    def counts_torch(self, th, grad=False):
        self.seen.append(tuple(th.shape))
        return torch.exp(th.double() @ self.w * 0.1 + 1.0)

    def __call__(self, theta):
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float64)
        return self.counts_torch(th).detach().cpu().numpy()


@pytest.fixture
def setup():
    fwd = _RecordingForward()
    lo, hi = np.zeros(NDIM), np.ones(NDIM)
    rng = np.random.default_rng(0)
    data = rng.poisson(fwd(np.full((1, NDIM), 0.5))[0]).astype(np.float64)
    model = SpectrumModel(fwd, data, uniform_priors(fwd.names, lo, hi))
    post = PoissonPosterior(fwd, data, BoxPrior(lo, hi, fwd.names))
    return fwd, model, post


def test_particles_reach_the_forward_as_one_batch(setup):
    # THE test: num_particles must arrive as the forward's batch dimension, in
    # ONE call -- not as P separate calls, and not silently collapsed to 1.
    # AutoMultivariateNormal traces the model twice at B=1 when it is first
    # built, to discover the latent structure; that is a one-off, so the assert
    # is on the steady state after a warm-up step (verified: steps 1..n issue
    # exactly one batched call each).
    import pyro
    from pyro.infer import Trace_ELBO
    from pyro.infer.autoguide import AutoMultivariateNormal

    fwd, model, _ = setup
    pyro.clear_param_store()
    guide = AutoMultivariateNormal(model)
    elbo = Trace_ELBO(num_particles=8, vectorize_particles=True)
    elbo.differentiable_loss(model, guide)          # warm-up: guide setup
    fwd.seen.clear()
    elbo.differentiable_loss(model, guide)
    assert len(fwd.seen) == 1, f"expected one batched call, got {fwd.seen}"
    assert fwd.seen[0] == (8, NDIM), f"particle dim lost: {fwd.seen[0]}"


def test_missing_prior_is_rejected(setup):
    fwd, _, _ = setup
    lo, hi = np.zeros(NDIM), np.ones(NDIM)
    priors = uniform_priors(fwd.names, lo, hi)
    priors.pop(fwd.names[1])
    with pytest.raises(ValueError, match="no prior supplied"):
        SpectrumModel(fwd, np.ones(NCHAN), priors)


def test_uniform_priors_rejects_length_mismatch():
    with pytest.raises(ValueError, match="matching lengths"):
        uniform_priors(["a", "b"], [0.0], [1.0, 2.0])


def test_pyro_log_joint_matches_hand_rolled_posterior(setup):
    # the two paths must be the same density up to an additive constant: the
    # hand-rolled likelihood drops log(d!), and the uniform prior contributes a
    # constant log-volume. A constant offset is fine; a varying one is a bug.
    from pyro import poutine

    fwd, model, post = setup
    rng = np.random.default_rng(1)
    offsets = []
    for _ in range(5):
        theta = rng.uniform(0.05, 0.95, size=(1, NDIM))
        cond = {n: torch.tensor(theta[0, i]) for i, n in enumerate(fwd.names)}
        tr = poutine.trace(poutine.condition(model, data=cond)).get_trace()
        pyro_lp = float(tr.log_prob_sum())
        offsets.append(pyro_lp - float(post.loglike(theta)[0]))
    assert np.allclose(offsets, offsets[0], atol=1e-6), (
        f"offset varies with theta, so the densities differ: {offsets}")


# --- NUTS options: full_mass, init at truth, tree-depth accounting ----------

class _TinyForward:
    """Cheap differentiable forward so NUTS options can be exercised."""
    device = "cpu"
    names = ["a", "b"]

    def counts_torch(self, th, grad=False):
        import torch
        amp = torch.pow(10.0, th[..., 0:1])
        tilt = th[..., 1:2]
        x = torch.linspace(0, 1, 8, dtype=th.dtype, device=th.device)
        return amp * torch.exp(tilt * x)


def _tiny_model():
    import numpy as np
    from spexai.inference.ppl import SpectrumModel, uniform_priors
    fwd = _TinyForward()
    data = np.array([300.0, 320.0, 350.0, 380.0, 410.0, 440.0, 480.0, 520.0])
    return SpectrumModel(fwd, data,
                         uniform_priors(["a", "b"], [1.0, -1.0], [4.0, 2.0]))


def test_nuts_accepts_full_mass_and_init_values():
    pytest.importorskip("pyro")
    from spexai.inference import samplers
    model = _tiny_model()
    res = samplers.run_nuts(model, n_samples=8, n_warmup=8, max_tree_depth=4,
                            full_mass=True, init_values=[2.5, 0.4], seed=0)
    assert res.name == "nuts"
    assert res.extra["full_mass"] is True
    assert res.samples.shape[1] == 2
    assert np.isfinite(res.samples).all()


def test_nuts_records_steps_per_iteration():
    """The number that decides viability: gradients per iteration, measured
    from n_eval rather than inferred from wall-clock."""
    pytest.importorskip("pyro")
    from spexai.inference import samplers
    model = _tiny_model()
    res = samplers.run_nuts(model, n_samples=8, n_warmup=8, max_tree_depth=4,
                            init_values=[2.5, 0.4], seed=0)
    assert res.extra["tree_depth_ceiling"] == 15
    assert 0 < res.extra["steps_per_iter"] <= 15


def test_nuts_init_values_accepts_a_dict():
    pytest.importorskip("pyro")
    from spexai.inference import samplers
    model = _tiny_model()
    res = samplers.run_nuts(model, n_samples=4, n_warmup=4, max_tree_depth=3,
                            init_values={"a": 2.5, "b": 0.4}, seed=0)
    assert np.isfinite(res.samples).all()


def test_nuts_init_values_actually_move_the_chain_start():
    """Weak assertions would pass even if init_strategy were silently ignored.

    Two very different starting points, no warmup and a tiny tree depth, so the
    chain cannot travel far: the draws must reflect where each started.
    """
    pytest.importorskip("pyro")
    from spexai.inference import samplers
    lo = samplers.run_nuts(_tiny_model(), n_samples=8, n_warmup=0,
                           max_tree_depth=1, init_values=[1.2, -0.8], seed=0)
    hi = samplers.run_nuts(_tiny_model(), n_samples=8, n_warmup=0,
                           max_tree_depth=1, init_values=[3.8, 1.8], seed=0)
    assert np.abs(hi.samples[0] - lo.samples[0]).max() > 0.5, (
        f"init_values appears ignored: {lo.samples[0]} vs {hi.samples[0]}")
