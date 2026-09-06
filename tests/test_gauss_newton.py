"""Gauss-Newton / Fisher scoring for the P6 estimator (``gauss_newton_batch``).

L-BFGS stalled on this problem with max|grad| ~ O(1) while reporting tiny
movement: it has to LEARN curvature by differencing function values the forward
only reproduces to ~2e-7, and one failed line search discarded the history.
The 2026-09-06 noiseless sweep returned 0/20 converged with the multi-start
spread at 90-280% of the bias, and removing the premature ``tolerance_change``
break bought 19 real steps in 2002 s where the old settings got 18 in 245 s.

Fisher scoring replaces both mechanisms: F comes from structure rather than
from differences, and there is no line search at all. These tests pin the
properties that make that true, on the same toy Poisson problem as
``test_lbfgs_precondition`` (seconds, CPU, no emulator).
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from spexai.inference.posterior import BoxPrior              # noqa: E402
from mle_reseed import gauss_newton_batch, gn_score          # noqa: E402
from test_lbfgs_precondition import NDIM, _ToyForward        # noqa: E402


class _GNToy(_ToyForward):
    """``_ToyForward`` plus the numpy ``__call__`` that ``batched_jacobian``
    uses to build the central-difference J."""

    def __call__(self, theta):
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float64,
                             device=self.device)
        return self.counts_torch(th, grad=False).cpu().numpy()


class _P:
    """Duck-types the parameter record: bounds plus a difference step."""

    def __init__(self, lo, hi, step):
        self.low, self.high, self.step = float(lo), float(hi), float(step)


@pytest.fixture(scope="module")
def gtoy():
    truth = np.zeros(NDIM)
    fwd = _GNToy(truth)
    sigma = fwd.sigma_at_truth()                              # (NDIM,)
    prior = BoxPrior(truth - 1e4 * sigma, truth + 1e4 * sigma)
    # a step of 0.1 sigma is small against curvature and large against the
    # toy's float64 exactness, matching how p.step is chosen for the real run
    pars = [_P(truth[j] - 1e4 * sigma[j], truth[j] + 1e4 * sigma[j],
               0.1 * sigma[j]) for j in range(NDIM)]
    return fwd, prior, truth, sigma, pars


def _planted(fwd, truth, sigma, K, offset=3.0, seed=3):
    """Noiseless data from a displaced theta_star, plus K scattered starts."""
    theta_star = truth + offset * sigma                       # (NDIM,)
    row = fwd.counts_torch(torch.as_tensor(theta_star[None, :]),
                           grad=False).cpu().numpy()          # (1, NBIN)
    data = np.repeat(row, K, axis=0)                          # (K, NBIN)
    rng = np.random.default_rng(seed)
    start = truth[None, :] + offset * sigma[None, :] * rng.normal(
        size=(K, NDIM))                                       # (K, NDIM)
    return theta_star, data, start


# --------------------------------------------------------------------------
# the score: the ONLY quantity that sets the fixed point
# --------------------------------------------------------------------------

def test_autograd_score_matches_finite_differences(gtoy):
    """The autograd score is the gradient of the same log-likelihood.

    This is the load-bearing test of the whole method. The fixed point of
    Fisher scoring is s = 0, so an error here moves the answer; an error in F
    only changes the path taken to it (pinned separately below).
    """
    fwd, _, truth, sigma, _ = gtoy
    _, data, _ = _planted(fwd, truth, sigma, K=1)
    theta = (truth + 1.0 * sigma)[None, :]                    # (1, NDIM)
    dat = torch.as_tensor(data, dtype=torch.float64)
    mu_ref = fwd.counts_torch(torch.as_tensor(truth[None, :]), grad=False)
    log_mu_ref = torch.log(mu_ref)

    s, _ = gn_score(fwd, theta, dat, mu_ref, log_mu_ref)      # (1, NDIM)

    def loglike(th):
        mu = fwd.counts_torch(torch.as_tensor(th[None, :]),
                              grad=False).cpu().numpy()[0]
        return float((data[0] * np.log(mu) - mu).sum())

    fd = np.zeros(NDIM)
    for j in range(NDIM):
        h = 1e-4 * sigma[j]
        tp, tm = theta[0].copy(), theta[0].copy()
        tp[j] += h
        tm[j] -= h
        fd[j] = (loglike(tp) - loglike(tm)) / (2 * h)
    # compare in sigma units: the raw components span six decades, so a plain
    # rtol would be dominated by whichever parameter happens to be largest
    np.testing.assert_allclose(s[0] * sigma, fd * sigma, rtol=2e-3)


def test_score_vanishes_at_the_planted_optimum(gtoy):
    """s(theta_star) = 0 for noiseless data, to the level the test can see."""
    fwd, _, truth, sigma, _ = gtoy
    theta_star, data, _ = _planted(fwd, truth, sigma, K=1)
    dat = torch.as_tensor(data, dtype=torch.float64)
    mu_ref = fwd.counts_torch(torch.as_tensor(truth[None, :]), grad=False)
    s_star, _ = gn_score(fwd, theta_star[None, :], dat, mu_ref,
                         torch.log(mu_ref))
    s_off, _ = gn_score(fwd, (theta_star + 0.5 * sigma)[None, :], dat, mu_ref,
                        torch.log(mu_ref))
    assert np.abs(s_star * sigma).max() < 1e-6 * np.abs(s_off * sigma).max()


# --------------------------------------------------------------------------
# convergence
# --------------------------------------------------------------------------

def test_recovers_a_planted_theta_star(gtoy):
    """End-to-end: noiseless data from theta_star must return theta_star."""
    fwd, prior, truth, sigma, pars = gtoy
    theta_star, data, start = _planted(fwd, truth, sigma, K=4)
    mle, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=12,
                                start=start, sigma_ref=sigma, verbose=False)
    err = np.abs(mle - theta_star[None, :]) / sigma[None, :]
    assert err.max() < 0.02, f"worst {err.max():.3f} sigma from theta_star"


def test_scattered_starts_agree(gtoy):
    """The multi-start spread -- the diagnostic the sweep failed 0/20 on."""
    fwd, prior, truth, sigma, pars = gtoy
    _, data, start = _planted(fwd, truth, sigma, K=8, seed=11)
    mle, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=12,
                                start=start, sigma_ref=sigma, verbose=False)
    spread = (mle.max(0) - mle.min(0)) / sigma
    assert spread.max() < 0.02, f"spread {spread.max():.3f} sigma"


def test_converges_in_a_few_rounds_without_any_tuning(gtoy):
    """From ~7 sigma out, with no preconditioning, restarts or line search.

    NOT a head-to-head win over L-BFGS: on this toy L-BFGS also converges (to
    ~1e-5 sigma), so a comparative assertion here would be comparing two
    converged answers and would pass for the wrong reason. What is being pinned
    is that Fisher scoring needs none of the three things that had to be tuned
    for L-BFGS, and gets there in a handful of forwards.
    """
    fwd, prior, truth, sigma, pars = gtoy
    theta_star, data, start = _planted(fwd, truth, sigma, K=2, seed=5)
    assert (np.abs(start - theta_star[None, :]) / sigma[None, :]).max() > 5.0
    mle, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=4,
                                start=start, sigma_ref=sigma, verbose=False)
    assert (np.abs(mle - theta_star[None, :]) / sigma[None, :]).max() < 0.02


def test_toy_flatters_gauss_newton_and_cannot_predict_the_real_run(gtoy):
    """GUARD THE GUARD, and record a limitation of this whole file.

    ``_ToyForward`` is ``mu = base * exp(A d)``, so log mu is LINEAR in theta:
    a canonical-link Poisson GLM. There the Fisher information equals the
    observed information exactly, Fisher scoring IS Newton's method, and
    convergence is quadratic and global. That is the most favourable possible
    setting for this method.

    The real emulator is not log-linear, so the dropped second-order term is
    only suppressed by the emulator residual (~1e-3), convergence is linear at
    that rate rather than quadratic, and the observed information is only
    approximately F. Everything in this file therefore establishes that the
    IMPLEMENTATION is correct; none of it predicts that Fisher scoring will
    beat L-BFGS on the real problem. Only the cluster run can say that.
    """
    fwd, _, truth, sigma, pars = gtoy
    from mle_reseed import batched_jacobian

    theta = truth + 2.0 * sigma                                  # off-optimum
    th = torch.tensor(theta[None, :], dtype=torch.float64, requires_grad=True)
    logmu = torch.log(fwd.counts_torch(th, grad=True)).sum()
    g = torch.autograd.grad(logmu, th, create_graph=True)[0]
    d2 = torch.autograd.grad(g.sum(), th)[0]
    assert float(d2.abs().max()) < 1e-6, "toy stopped being log-linear"

    # and therefore: F (expected) == -Hessian (observed), exactly
    _, data, _ = _planted(fwd, truth, sigma, K=1)
    mu0, J = batched_jacobian(fwd, pars, theta[None, :])
    F = (J[0] / mu0[0]) @ J[0].T                                 # (NDIM, NDIM)

    def negll(t):
        mu = fwd.counts_torch(t.reshape(1, NDIM), grad=True).clamp_min(1e-30)
        return -(torch.as_tensor(data[0]) * torch.log(mu) - mu).sum()

    H = torch.autograd.functional.hessian(
        negll, torch.tensor(theta, dtype=torch.float64)).numpy()  # (NDIM,NDIM)
    scale = np.outer(sigma, sigma)          # compare in sigma units, not raw
    np.testing.assert_allclose(F * scale, H * scale, rtol=1e-6)


def test_decrement_falls_and_stops_the_iteration(gtoy, capsys):
    """lambda^2/2 is the stopping rule, and it must actually be reached."""
    fwd, prior, truth, sigma, pars = gtoy
    _, data, start = _planted(fwd, truth, sigma, K=2, seed=9)
    gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=15, start=start,
                       sigma_ref=sigma, tol_decrement=1e-3, verbose=True)
    out = capsys.readouterr().out
    assert "GN converged" in out
    lam = [float(ln.split("lambda^2/2")[1].split("nats")[0])
           for ln in out.splitlines() if "lambda^2/2" in ln]
    assert lam[-1] < 1e-3
    assert lam[-1] < lam[0] / 1e3          # it descended, not started there


def test_more_iterations_do_not_move_the_answer(gtoy):
    """The fixed point is a fixed point: extra rounds change nothing.

    Guards the failure mode [[p6-estimator-guards]] warns about -- a result
    that looks stable only because the iteration stopped early.
    """
    fwd, prior, truth, sigma, pars = gtoy
    _, data, start = _planted(fwd, truth, sigma, K=2, seed=13)
    kw = dict(start=start, sigma_ref=sigma, tol_decrement=0.0, verbose=False)
    a, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=8, **kw)
    b, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=16, **kw)
    assert (np.abs(a - b) / sigma[None, :]).max() < 1e-3


# --------------------------------------------------------------------------
# structural properties that L-BFGS did not have
# --------------------------------------------------------------------------

def test_batched_rows_are_independent(gtoy):
    """K rows in one call == K separate calls.

    ``lbfgs_batch`` shares one strong-Wolfe step length across the chunk, so a
    single t=0 terminated every row at once and the rows' answers were coupled
    -- which contaminates exactly the multi-start spread being measured. Fisher
    scoring solves each row's own linear system, so batching is pure
    bookkeeping. This is the test that says so.
    """
    fwd, prior, truth, sigma, pars = gtoy
    _, data, start = _planted(fwd, truth, sigma, K=3, seed=17)
    kw = dict(sigma_ref=sigma, tol_decrement=0.0, verbose=False)
    both, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=6,
                                 start=start, **kw)
    for k in range(3):
        one, _ = gauss_newton_batch(fwd, prior, data[k:k + 1], truth, pars,
                                    n_iter=6, start=start[k:k + 1], **kw)
        np.testing.assert_allclose(both[k], one[0], rtol=0, atol=1e-12)


def test_fisher_matrix_is_positive_definite(gtoy):
    """F = J diag(1/mu) J^T is a Gram matrix, so the step is always ascent.

    This is why no line search is needed: the indefinite-Hessian case that
    makes strong Wolfe return t=0 cannot arise.
    """
    from mle_reseed import batched_jacobian
    fwd, _, truth, sigma, pars = gtoy
    theta = np.stack([truth + 3.0 * sigma, truth - 7.0 * sigma])   # (2, NDIM)
    mu0, J = batched_jacobian(fwd, pars, theta)     # (2,NBIN), (2,NDIM,NBIN)
    for k in range(2):
        F = (J[k] / mu0[k]) @ J[k].T                               # (NDIM,NDIM)
        np.testing.assert_allclose(F, F.T, rtol=1e-10)
        assert np.linalg.eigvalsh(F).min() > 0


def test_an_inexact_F_still_finds_the_same_optimum(gtoy):
    """F sets the path, not the fixed point -- the premise of the design.

    A deliberately wrong F (a Levenberg ridge big enough to distort the step)
    must reach the same answer, just in more iterations. If this fails, the
    central-difference J is load-bearing after all and would need forward-mode
    autograd instead.
    """
    fwd, prior, truth, sigma, pars = gtoy
    theta_star, data, start = _planted(fwd, truth, sigma, K=2, seed=23)
    kw = dict(start=start, sigma_ref=sigma, tol_decrement=0.0, verbose=False)
    exact, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=12,
                                  ridge=0.0, **kw)
    ridged, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=60,
                                   ridge=0.5, **kw)
    assert (np.abs(exact - theta_star[None, :]) / sigma[None, :]).max() < 0.02
    assert (np.abs(ridged - theta_star[None, :]) / sigma[None, :]).max() < 0.05


# --------------------------------------------------------------------------
# safeguards
# --------------------------------------------------------------------------

def test_trust_region_caps_the_first_step(gtoy, capsys):
    """A wild start must not produce a wild step. The cap is in sigma units
    and never compares two loss values, so the noise floor cannot affect it."""
    fwd, prior, truth, sigma, pars = gtoy
    _, data, _ = _planted(fwd, truth, sigma, K=1)
    start = (truth + 500.0 * sigma)[None, :]
    gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=1, start=start,
                       sigma_ref=sigma, max_step_sigma=2.0, verbose=True)
    moves = [float(ln.split("max move")[1].split("sigma")[0])
             for ln in capsys.readouterr().out.splitlines() if "max move" in ln]
    assert moves[0] <= 2.0 + 1e-9


def test_steps_stay_inside_the_box(gtoy):
    """Clipping keeps a early step from walking a parameter off the model's
    valid range, where the forward is not defined."""
    fwd, _, truth, sigma, _ = gtoy
    lo, hi = truth - 0.5 * sigma, truth + 0.5 * sigma           # a tight box
    pars = [_P(lo[j], hi[j], 0.1 * sigma[j]) for j in range(NDIM)]
    prior = BoxPrior(lo, hi)
    _, data, _ = _planted(fwd, truth, sigma, K=2, offset=3.0)
    mle, _ = gauss_newton_batch(fwd, prior, data, truth, pars, n_iter=5,
                                sigma_ref=sigma, max_step_sigma=50.0,
                                verbose=False)
    assert np.all(mle >= lo[None, :] - 1e-12)
    assert np.all(mle <= hi[None, :] + 1e-12)
