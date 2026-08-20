"""User-facing prior specification across all three sampler interfaces."""
import numpy as np
import pytest
import torch

from spexai.inference.posterior import BoxPrior, PoissonPosterior
from spexai.inference.priors import (LogUniform, Normal, Prior, PriorSet,
                                     Uniform)


# --- individual priors -------------------------------------------------------

def test_uniform_density_and_support():
    p = Uniform(2.0, 6.0)
    assert p.logpdf(np.array([4.0]))[0] == pytest.approx(-np.log(4.0))
    assert p.logpdf(np.array([1.0]))[0] == -np.inf
    assert p.logpdf(np.array([7.0]))[0] == -np.inf


def test_uniform_ppf_spans_the_box():
    p = Uniform(2.0, 6.0)
    assert p.ppf(np.array([0.0, 0.5, 1.0])) == pytest.approx([2.0, 4.0, 6.0])


def test_uniform_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Uniform(6.0, 2.0)


def test_loguniform_is_flat_in_log():
    p = LogUniform(1.0, 100.0)
    # equal log-intervals must carry equal probability
    assert p.ppf(np.array([0.5]))[0] == pytest.approx(10.0)
    # density falls as 1/x
    assert (p.logpdf(np.array([1.0]))[0] - p.logpdf(np.array([10.0]))[0]
            == pytest.approx(np.log(10.0)))


def test_loguniform_rejects_nonpositive():
    with pytest.raises(ValueError):
        LogUniform(0.0, 10.0)


def test_normal_defaults_to_six_sigma_support():
    p = Normal(1.0, 0.1)
    assert p.low == pytest.approx(0.4)
    assert p.high == pytest.approx(1.6)


def test_normal_truncation_is_respected():
    p = Normal(0.0, 1.0, low=0.0)          # half-normal
    assert p.logpdf(np.array([-0.5]))[0] == -np.inf
    assert np.isfinite(p.logpdf(np.array([0.5]))[0])
    # truncated density is renormalised: it integrates to 1 over [0, 6]
    x = np.linspace(0, 6, 20001)
    assert np.trapezoid(np.exp(p.logpdf(x)), x) == pytest.approx(1.0, abs=1e-3)


def test_normal_ppf_inverts_the_cdf():
    p = Normal(3.0, 0.5)
    assert p.ppf(np.array([0.5]))[0] == pytest.approx(3.0)
    u = np.array([0.1, 0.3, 0.9])
    assert np.all(np.diff(p.ppf(u)) > 0)


def test_normal_rejects_bad_sigma():
    with pytest.raises(ValueError):
        Normal(1.0, 0.0)


def test_prior_base_is_abstract():
    with pytest.raises(NotImplementedError):
        Prior().logpdf(np.array([0.0]))


# --- PriorSet ----------------------------------------------------------------

def _set():
    return PriorSet({"kT": Uniform(1.5, 7.5), "Fe": Normal(0.55, 0.05, low=0.0)},
                    names=["kT", "Fe"])


def test_priorset_orders_by_names_not_dict_order():
    ps = PriorSet({"Fe": Normal(0.55, 0.05), "kT": Uniform(1.5, 7.5)},
                  names=["kT", "Fe"])
    assert ps.names == ["kT", "Fe"]
    assert ps.lo[0].item() == pytest.approx(1.5)


def test_priorset_rejects_missing_and_extra():
    with pytest.raises(ValueError, match="no prior supplied"):
        PriorSet({"kT": Uniform(1, 2)}, names=["kT", "Fe"])
    with pytest.raises(ValueError, match="unknown parameters"):
        PriorSet({"kT": Uniform(1, 2), "zz": Uniform(1, 2)}, names=["kT"])


def test_priorset_logpdf_sums_independent_terms():
    ps = _set()
    th = np.array([[3.0, 0.55]])
    expected = (Uniform(1.5, 7.5).logpdf(np.array([3.0]))[0]
                + Normal(0.55, 0.05, low=0.0).logpdf(np.array([0.55]))[0])
    assert ps.logpdf(th)[0] == pytest.approx(expected)


def test_priorset_ptform_maps_the_unit_cube():
    ps = _set()
    out = ps.ptform(np.array([[0.5, 0.5]]))
    assert out[0, 0] == pytest.approx(4.5)          # middle of the kT box
    assert out[0, 1] == pytest.approx(0.55)         # median of the gaussian


def test_priorset_sample_lands_in_support():
    ps = _set()
    rng = np.random.default_rng(0)
    draws = ps.sample(rng, 500)
    assert draws.shape == (500, 2)
    assert ps.inside(draws).all()
    # the Fe draws must actually follow the prior, not the box
    assert draws[:, 1].mean() == pytest.approx(0.55, abs=0.01)
    assert draws[:, 1].std() == pytest.approx(0.05, abs=0.01)


def test_priorset_matches_boxprior_on_uniforms():
    """The compatibility contract: uniform PriorSet == the old BoxPrior."""
    names, lo, hi = ["a", "b"], [0.0, -1.0], [2.0, 5.0]
    ps = PriorSet.uniform(names, lo, hi)
    bp = BoxPrior(lo, hi, names)
    cube = np.array([[0.25, 0.75], [0.0, 1.0]])
    assert ps.ptform(cube) == pytest.approx(bp.ptform(cube))
    th = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert ps.inside(th).tolist() == bp.inside(th).tolist()
    z = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    t_ps, ld_ps = ps.to_constrained(z)
    t_bp, ld_bp = bp.to_constrained(z)
    assert torch.allclose(t_ps, t_bp) and torch.allclose(ld_ps, ld_bp)


def test_priorset_round_trips_unconstrained():
    ps = _set()
    th = torch.tensor([[3.0, 0.55]], dtype=torch.float64)
    assert torch.allclose(ps.to_constrained(ps.to_unconstrained(th))[0], th,
                          atol=1e-9)


def test_from_params_reads_bounds():
    class P:
        def __init__(self, name, low, high):
            self.name, self.low, self.high = name, low, high
    ps = PriorSet.from_params([P("kT", 1.0, 8.0), P("n_h", 0.0, 5.0)])
    assert ps.names == ["kT", "n_h"]
    assert ps.hi[1].item() == pytest.approx(5.0)


# --- integration with the posterior -----------------------------------------

class _FlatForward:
    """Forward whose likelihood is constant, so only the prior shapes logp."""
    device = "cpu"
    names = ["kT", "Fe"]

    def __call__(self, th):
        return np.ones((np.atleast_2d(th).shape[0], 3))


def test_posterior_applies_a_nonuniform_prior():
    """The bug this API exists to prevent: a prior that is silently ignored."""
    fwd = _FlatForward()
    data = np.array([1.0, 1.0, 1.0])
    ps = PriorSet({"kT": Uniform(1.5, 7.5), "Fe": Normal(0.55, 0.05, low=0.0)},
                  names=["kT", "Fe"])
    post = PoissonPosterior(fwd, data, ps)
    # likelihood is identical at both points, so any difference is the prior
    at_peak = post.logp(np.array([[3.0, 0.55]]))[0]
    off_peak = post.logp(np.array([[3.0, 0.65]]))[0]
    assert at_peak > off_peak
    assert at_peak - off_peak == pytest.approx(2.0, abs=0.05)   # 2 sigma^2/2


def test_posterior_still_flat_under_boxprior():
    """Bit-compatibility: the old uniform path is unchanged by the new sum."""
    fwd = _FlatForward()
    data = np.array([1.0, 1.0, 1.0])
    post = PoissonPosterior(fwd, data, BoxPrior([1.5, 0.0], [7.5, 3.0],
                                                ["kT", "Fe"]))
    a = post.logp(np.array([[3.0, 0.55]]))[0]
    b = post.logp(np.array([[6.0, 2.0]]))[0]
    assert a == pytest.approx(b)


def test_posterior_rejects_outside_support():
    post = PoissonPosterior(_FlatForward(), np.ones(3), _set())
    assert post.logp(np.array([[9.9, 0.55]]))[0] == -np.inf


def test_loglike_stays_pure_so_ultranest_does_not_double_count():
    """UltraNest draws through ptform; if loglike also carried the prior
    density the prior would be applied twice."""
    post = PoissonPosterior(_FlatForward(), np.ones(3), _set())
    a = post.loglike(np.array([[3.0, 0.55]]))[0]
    b = post.loglike(np.array([[3.0, 0.95]]))[0]
    assert a == pytest.approx(b)


def test_to_pyro_covers_every_parameter():
    pytest.importorskip("pyro")
    ps = _set()
    d = ps.to_pyro()
    assert set(d) == {"kT", "Fe"}
    assert float(d["Fe"].mean) == pytest.approx(0.55)


# --- device placement (the NUTS crash) ---------------------------------------

def test_to_pyro_places_parameters_on_the_requested_device():
    """Regression: CPU-built Pyro priors put theta on CPU while the forward ran
    on CUDA, and the mismatch surfaced only at the last multiply in
    VectorForward.fold."""
    pytest.importorskip("pyro")
    ps = PriorSet({"kT": Uniform(1.5, 7.5), "Fe": Normal(0.55, 0.05)},
                  names=["kT", "Fe"], device="cpu")
    d = ps.to_pyro(device="cpu")
    assert d["kT"].low.device.type == "cpu"
    assert d["Fe"].loc.device.type == "cpu"


def test_to_pyro_defaults_to_the_priorset_device():
    pytest.importorskip("pyro")
    ps = PriorSet({"kT": Uniform(1.5, 7.5)}, names=["kT"], device="cpu")
    assert ps.to_pyro()["kT"].low.device.type == "cpu"


def test_loguniform_to_pyro_accepts_device():
    pytest.importorskip("pyro")
    assert LogUniform(1.0, 100.0).to_pyro(device="cpu") is not None


def test_uniform_priors_helper_accepts_device():
    pytest.importorskip("pyro")
    from spexai.inference.ppl import uniform_priors
    d = uniform_priors(["a", "b"], [0.0, 1.0], [2.0, 3.0], device="cpu")
    assert d["a"].low.device.type == "cpu"
