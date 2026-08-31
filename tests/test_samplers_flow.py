"""The flow / NN-accelerated samplers: nautilus, pocoMC, i-nessai.

Each is checked on a cheap analytic Poisson problem where the answer is known,
so a wiring mistake (wrong prior handling, un-vectorised likelihood, weights
misread) shows up as a wrong posterior rather than as a slow GPU run that looks
plausible.

The load-bearing check is ``test_*_recovers_truth``: these samplers are only
worth adding if they get the *same* answer as the established ones, and an
importance sampler whose weights are mishandled will still return a
confident-looking, wrong posterior.
"""
import numpy as np
import pytest

from spexai.inference.posterior import BoxPrior, PoissonPosterior
from spexai.inference import samplers


class _LinearForward:
    """mu(theta) = exp(a) * shape, with a second parameter the data constrains
    weakly. Poisson-exact and cheap, so the true posterior is well defined."""

    device = "cpu"
    names = ["log_amp", "tilt"]

    def __init__(self, n=64):
        self.x = np.linspace(0.0, 1.0, n)

    def __call__(self, th):
        th = np.atleast_2d(np.asarray(th, dtype=float))
        amp = np.exp(th[:, 0])[:, None]                    # (B, 1)
        tilt = th[:, 1][:, None]                           # (B, 1)
        return amp * np.exp(tilt * self.x[None, :])        # (B, n)


TRUTH = np.array([np.log(300.0), 0.7])


def _problem(seed=0):
    fwd = _LinearForward()
    mu = fwd(TRUTH[None, :])[0]
    data = np.random.default_rng(seed).poisson(mu).astype(float)
    prior = BoxPrior([np.log(30.0), -2.0], [np.log(3000.0), 3.0],
                     ["log_amp", "tilt"])
    return PoissonPosterior(fwd, data, prior)


def _check(res, post, name, tol=4.0):
    assert res.name == name
    assert res.samples.ndim == 2 and res.samples.shape[1] == 2
    assert np.isfinite(res.samples).all()
    assert res.n_eval > 0, "n_eval not counted -- ESS/eval would be wrong"
    assert res.runtime_s > 0
    # the actual test: the truth must sit inside the posterior
    med, sig = res.median, res.sigma
    assert np.all(sig > 0), f"{name} returned a degenerate posterior"
    pull = (med - TRUTH) / sig
    assert np.abs(pull).max() < tol, f"{name} pulls {pull} exceed {tol} sigma"


# --- weighted-ESS helpers ----------------------------------------------------

def test_kish_ess_equal_weights_is_the_count():
    lw = np.zeros(500)
    assert samplers._kish_ess(lw) == pytest.approx(500.0)


def test_kish_ess_collapses_on_one_dominant_weight():
    lw = np.full(500, -np.inf)
    lw[0] = 0.0
    assert samplers._kish_ess(lw) == pytest.approx(1.0)


def test_kish_ess_survives_huge_log_weights():
    """Log-likelihoods here are O(1e6); a naive exp() would overflow to inf."""
    lw = 3.7e6 + np.random.default_rng(0).normal(size=1000)
    ess = samplers._kish_ess(lw)
    assert np.isfinite(ess) and 0 < ess <= 1000


def test_resample_equal_reproduces_the_weighted_mean():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4000, 1))
    lw = -0.5 * ((pts[:, 0] - 1.0) ** 2)          # reweight toward +1
    eq = samplers._resample_equal(pts, lw, rng, n=40000)
    w = np.exp(lw - lw.max())
    target = float((w * pts[:, 0]).sum() / w.sum())
    assert eq.mean() == pytest.approx(target, abs=0.05)


def test_uniform_box_scipy_rejects_nonuniform():
    from spexai.inference.priors import Normal, PriorSet, Uniform
    ps = PriorSet({"a": Uniform(0, 1), "b": Normal(0.0, 1.0)}, names=["a", "b"])
    with pytest.raises(NotImplementedError, match="not"):
        samplers._uniform_box_scipy(ps)


def test_uniform_box_scipy_matches_bounds():
    prior = BoxPrior([0.0, -1.0], [2.0, 5.0], ["a", "b"])
    d = samplers._uniform_box_scipy(prior)
    assert d[0].ppf(0.0) == pytest.approx(0.0)
    assert d[1].ppf(1.0) == pytest.approx(5.0)


# --- the samplers themselves -------------------------------------------------

def test_nautilus_recovers_truth():
    pytest.importorskip("nautilus")
    post = _problem()
    res = samplers.run_nautilus(post, n_live=300, n_eff=1500, n_networks=1,
                                seed=0)
    _check(res, post, "nautilus")
    assert np.isfinite(res.logz), "nautilus must report log Z"
    assert res.min_ess > 100


def test_pocomc_recovers_truth():
    pytest.importorskip("pocomc")
    post = _problem()
    res = samplers.run_pocomc(post, n_effective=128, n_active=64, n_total=1000,
                              n_evidence=1000, seed=0)
    _check(res, post, "pocomc")
    assert np.isfinite(res.logz), "pocoMC must report log Z"


def test_inessai_recovers_truth(tmp_path):
    pytest.importorskip("nessai")
    post = _problem()
    res = samplers.run_inessai(post, nlive=500, seed=0, target_ess=1000.0,
                               output=str(tmp_path / "inessai"))
    _check(res, post, "inessai")
    assert res.min_ess > 500, "i-nessai should reach its target ESS"


def test_inessai_default_criterion_does_not_collapse(tmp_path):
    """Regression for the weight collapse.

    nessai's default ``stopping_criterion="ratio"`` with ``tolerance=0.0``
    stops once log(Z_live/Z_all) <= 0, which for a peaked likelihood happens
    within a couple of iterations. The run then returned a weighted set whose
    Kish ESS was exactly 1.0 -- one point carrying all the weight -- while
    still looking superficially fine, because the raw point cloud was not
    degenerate. This pins the fixed default.
    """
    pytest.importorskip("nessai")
    post = _problem()
    res = samplers.run_inessai(post, nlive=500, seed=0, target_ess=1000.0,
                               output=str(tmp_path / "inessai_def"))
    assert res.extra["kish_ess"] > 100, (
        f"weights collapsed: Kish ESS {res.extra['kish_ess']:.2f}")
    # the distinct failure signature was a single surviving draw
    assert len(np.unique(res.samples, axis=0)) > 50


def test_inessai_agrees_with_ultranest(tmp_path):
    """Same cross-check nautilus gets: disagreement means wiring, not
    algorithm."""
    pytest.importorskip("nessai")
    pytest.importorskip("ultranest")
    post = _problem()
    ref = samplers.run_ultranest(post, min_num_live_points=200)
    ine = samplers.run_inessai(post, nlive=500, seed=0, target_ess=1000.0,
                               output=str(tmp_path / "inessai_x"))
    shift = np.abs(ine.median - ref.median) / ref.sigma
    assert shift.max() < 0.5, f"i-nessai vs ultranest shift {shift} sigma"
    width = ine.sigma / ref.sigma
    assert np.all((width > 0.6) & (width < 1.6)), f"width ratio {width}"


def test_flow_samplers_agree_with_ultranest():
    """Cross-check on one problem: if these disagree with an established
    nested sampler, the wiring is wrong, not the algorithm."""
    pytest.importorskip("nautilus")
    pytest.importorskip("ultranest")
    post = _problem()
    ref = samplers.run_ultranest(post, min_num_live_points=200)
    nau = samplers.run_nautilus(post, n_live=300, n_eff=1500, n_networks=1,
                                seed=0)
    shift = np.abs(nau.median - ref.median) / ref.sigma
    assert shift.max() < 0.5, f"nautilus vs ultranest shift {shift} sigma"
    width = nau.sigma / ref.sigma
    assert np.all((width > 0.6) & (width < 1.6)), f"width ratio {width}"
    # log Z is the other thing both claim to measure
    assert abs(nau.logz - ref.logz) < 1.0
