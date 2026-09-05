"""L-BFGS conditioning guards for the P6 estimator (``mle_reseed.lbfgs_batch``).

P6's 20-point sweep left 12 of 20 points non-converged while ``-logL`` was
still descending, which biases the measured linearisation factor ``k`` LOW --
i.e. in the direction that falsely exonerates the linearisation. The cause is
that torch's LBFGS thresholds are all ABSOLUTE while this problem's parameters
span three decades in sigma, so cond(F) ~ 1e7 comes mostly from the choice of
units.

These tests reproduce that pathology in a toy Poisson problem (seconds, CPU,
no emulator) and pin the three specific mechanisms, so a future edit that
removes the preconditioning or restores a torch default fails here rather than
silently in a 3-hour cluster sweep.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

from spexai.inference.posterior import BoxPrior          # noqa: E402
from mle_reseed import lbfgs_batch                       # noqa: E402

NDIM, NBIN = 6, 400


class _ToyForward:
    """``mu(theta) = base * exp(A @ (theta - truth))``, positive by construction.

    ``A``'s columns are scaled over 1e3..1e-3 so sigma_j spans ~6 decades and
    cond(F) ~ 1e12 -- the same unit-driven ill-conditioning as the real P6
    problem, exaggerated so the failure is unambiguous, and with none of the
    cost. Duck-types the part of ``VectorForward`` that ``lbfgs_batch`` uses.
    """

    def __init__(self, truth, device="cpu"):
        self.device = device
        g = torch.Generator().manual_seed(1)
        A = torch.randn(NBIN, NDIM, generator=g, dtype=torch.float64)
        col = torch.logspace(3, -3, NDIM, dtype=torch.float64)
        self.A = (A * col[None, :]).to(device)                  # (NBIN, NDIM)
        self.base = torch.full((NBIN,), 1e5, dtype=torch.float64, device=device)
        self.truth = torch.as_tensor(truth, dtype=torch.float64, device=device)

    def counts_torch(self, theta, grad=False):
        # theta (K, NDIM) -> (K, NBIN)
        d = theta.double() - self.truth[None, :]
        out = self.base[None, :] * torch.exp(d @ self.A.T).clamp(1e-8, 1e8)
        return out if grad else out.detach()

    def sigma_at_truth(self):
        # F = A^T diag(mu) A at the truth; sigma_j = sqrt(diag(F^-1))
        F = (self.A.T * self.base[None, :]) @ self.A            # (NDIM, NDIM)
        return torch.sqrt(torch.diag(torch.linalg.inv(F))).cpu().numpy()


@pytest.fixture(scope="module")
def toy():
    truth = np.zeros(NDIM)
    fwd = _ToyForward(truth)
    sigma = fwd.sigma_at_truth()
    # box wide compared with sigma, as in the real problem, so bounds play no
    # part in what the optimiser does
    prior = BoxPrior(truth - 1e4 * sigma, truth + 1e4 * sigma)
    mu_true = fwd.counts_torch(
        torch.as_tensor(truth[None, :]), grad=False).cpu().numpy()[0]
    rng = np.random.default_rng(7)
    data = rng.poisson(mu_true[None, :].repeat(4, 0)).astype(np.float64)
    return fwd, prior, truth, sigma, data


def test_toy_is_actually_ill_conditioned(toy):
    """Guard the guard: if the toy stops spanning decades it tests nothing."""
    _, _, _, sigma, _ = toy
    assert sigma.max() / sigma.min() > 1e5


def test_preconditioning_does_not_change_the_optimum(toy):
    """Both paths parameterise the same problem, so given a budget large
    enough for each they must agree. If preconditioning moved the answer it
    would corrupt ``k`` rather than fix its convergence."""
    fwd, prior, truth, sigma, data = toy
    raw, _ = lbfgs_batch(fwd, prior, data, truth, 300, sigma_ref=sigma,
                         n_restarts=4, max_eval=6000)
    pre, _ = lbfgs_batch(fwd, prior, data, truth, 300, sigma_ref=sigma,
                         n_restarts=4, max_eval=6000, precondition=True)
    # per parameter, in units of THAT parameter's sigma
    assert (np.abs(raw - pre) / sigma[None, :]).max() < 1e-2


def test_raw_path_line_search_returns_zero_steps(toy):
    """The observed P6 failure: on badly-scaled coordinates strong-Wolfe
    returns ``t = 0``, torch's ``d.mul(t).abs().max() <= tolerance_change``
    is then ``0 <= 0``, and the pass breaks while descent is still available.
    Only a handful of the requested iterations run."""
    fwd, prior, truth, sigma, data = toy
    ls = []
    with _capture(ls):
        lbfgs_batch(fwd, prior, data, truth, 60, sigma_ref=sigma, n_restarts=3)
    ts = np.array([t for t, _ in ls])
    assert (ts == 0).sum() > 0
    assert len(ls) < 30, "expected the passes to stop far short of 3 x 60"


def test_torch_max_eval_default_caps_function_evals_not_iterations():
    """Pins the code fact behind ``--max_eval``: torch's default budget is
    ``max_iter * 5 // 4`` FUNCTION EVALS, and ``step`` derives the line
    search's own budget from what is left of it, so the late searches are
    starved. Asserted against the source rather than demonstrated, because the
    toy converges too fast to exhaust it -- on the real problem, at
    ``--max_iter 400``, it is a live constraint."""
    import inspect
    import torch.optim.lbfgs as lb
    assert lb.LBFGS([torch.zeros(1, requires_grad=True)], max_iter=400
                    ).param_groups[0]["max_eval"] == 400 * 5 // 4
    src = "".join(inspect.getsource(lb.LBFGS.step).split())
    assert "max_ls=max_eval-current_evals" in src


def test_preconditioned_line_search_stops_thrashing(toy):
    """The diagnostic that matters: after rescaling, strong-Wolfe accepts its
    first trial step (t ~ 1) almost every time instead of collapsing the
    bracket toward zero."""
    fwd, prior, truth, sigma, data = toy
    ls = []
    with _capture(ls):
        lbfgs_batch(fwd, prior, data, truth, 60, sigma_ref=sigma, n_restarts=1,
                    precondition=True)
    ts = np.array([t for t, _ in ls])
    evals = np.array([e for _, e in ls])
    assert evals.mean() < 3.0
    assert (ts == 0).sum() == 0, "a zero step means the line search gave up"
    assert np.median(ts) > 0.1


def test_zero_step_breaks_the_pass_even_with_tol_change_zero():
    """Documents why ``--tol_change 0`` was not enough. torch breaks on
    ``d.mul(t).abs().max() <= tolerance_change``; when the line search returns
    ``t = 0`` exactly that test is ``0 <= 0``, which is True for any
    tolerance, including zero."""
    assert (torch.zeros(3).abs().max() <= 0.0).item() is True


def test_zero_line_search_tolerance_makes_cubic_interpolation_nan():
    """Why ``ls_tol_change`` must stay None.

    ``_strong_wolfe``'s ``tolerance_change`` guards the ZOOM BRACKET
    (lbfgs.py:110), and ``_cubic_interpolate`` divides by ``x1 - x2``
    (lbfgs.py:27). Zero the guard and a collapsed bracket interpolates 0/0.
    The resulting NaN ``t`` poisons every parameter and surfaces far away --
    it was first seen as ``cannot convert float NaN to integer`` inside
    ``fft_broaden``, a traceback naming nothing to do with the optimiser.
    """
    from torch.optim.lbfgs import _cubic_interpolate

    def t64(v):
        return torch.tensor(v, dtype=torch.float64)

    # A bracket ground down to ~1e-200 wide while the two loss values still
    # differ: (f1 - f2) / (x1 - x2) overflows, d1**2 becomes inf, and the
    # interpolation reduces to inf/inf. Reachable in a few hundred zoom
    # bisections, which is exactly what a large max_ls with no bracket guard
    # allows.
    out = _cubic_interpolate(t64(0.25), t64(-1.0), t64(-1e-3),
                             t64(0.25 + 1e-200), t64(-1.0 + 1e-9), t64(-1e-3))
    assert not torch.isfinite(torch.as_tensor(out)), (
        "a collapsed bracket must be prevented, not survived")


def test_hook_raises_on_a_non_finite_step(toy):
    """The guard that turns that NaN into an error naming the optimiser."""
    from mle_reseed import _line_search_hook
    import torch.optim.lbfgs as lb

    fwd, prior, truth, sigma, data = toy
    real = lb._strong_wolfe

    def poisoned(*a, **kw):
        f_new, g_new, _, evals = real(*a, **kw)
        return f_new, g_new, float("nan"), evals

    lb._strong_wolfe = poisoned
    try:
        with pytest.raises(FloatingPointError, match="zoom bracket"):
            with _line_search_hook([]):
                lbfgs_batch(fwd, prior, data, truth, 10, sigma_ref=sigma,
                            n_restarts=1, precondition=True)
    finally:
        lb._strong_wolfe = real


def test_strong_wolfe_tolerance_is_not_forwarded_by_torch():
    """``LBFGS.step`` calls ``_strong_wolfe`` without passing the optimizer's
    ``tolerance_change``, so ``--tol_change 0`` does NOT reach the line search
    -- which is correct and must stay that way, since the two mean different
    things. Pinned because the fix for one was briefly assumed to be the fix
    for the other."""
    import inspect
    import torch.optim.lbfgs as lb
    src = inspect.getsource(lb.LBFGS.step)
    call = src[src.index("_strong_wolfe("):]
    call = call[:call.index(")")]
    assert "tolerance_change" not in call


def test_bsys_starts_layout_and_units(toy):
    """Row 0 is exactly truth + b_sys; the rest scatter by |b_sys| as a
    STANDARD DEVIATION (component-wise), and everything stays in the box."""
    from mle_reseed import bsys_starts

    class _P:
        def __init__(self, lo, hi):
            self.low, self.high = lo, hi

    truth = np.zeros(NDIM)
    b_sys = np.linspace(0.1, 2.0, NDIM)
    pars = [_P(-1e3, 1e3) for _ in range(NDIM)]
    s = bsys_starts(truth, b_sys, pars, 4000, seed=0)

    assert s.shape == (4000, NDIM)
    np.testing.assert_allclose(s[0], truth + b_sys, rtol=0, atol=0)
    # scatter rows: per-component sd should be |b_sys|, not b_sys**2
    sd = s[1:].std(axis=0, ddof=1)
    np.testing.assert_allclose(sd, np.abs(b_sys), rtol=0.1)
    assert np.all(s > -1e3) and np.all(s < 1e3)


def test_bsys_starts_respects_the_box():
    """A b_sys that would push a start outside its bound gets clipped strictly
    inside -- to_unconstrained saturates at the bound, and a start there has
    no gradient to move on."""
    from mle_reseed import bsys_starts

    class _P:
        def __init__(self, lo, hi):
            self.low, self.high = lo, hi

    truth = np.zeros(3)
    pars = [_P(-1.0, 1.0)] * 3
    s = bsys_starts(truth, np.array([5.0, -5.0, 0.0]), pars, 50, seed=1)
    assert np.all(s > -1.0) and np.all(s < 1.0)


def test_noiseless_fit_recovers_a_planted_bias(toy):
    """End-to-end check of the noiseless design on a toy with a KNOWN answer.

    Data is the 'truth' model evaluated at a displaced theta_star and used
    without a Poisson draw, so the fit must return theta_star itself. This is
    the noiseless construction P6 uses: fit the emulator to the reference
    spectrum and read off the pseudo-true parameters.
    """
    fwd, prior, truth, sigma, _ = toy
    theta_star = truth + 3.0 * sigma                    # the planted answer
    data = fwd.counts_torch(
        torch.as_tensor(theta_star[None, :]), grad=False).cpu().numpy()
    data = np.repeat(data, 4, axis=0)                   # 4 identical rows

    from mle_reseed import bsys_starts

    class _P:
        def __init__(self, lo, hi):
            self.low, self.high = lo, hi

    pars = [_P(float(lo), float(hi))
            for lo, hi in zip(prior.lo.cpu().numpy(), prior.hi.cpu().numpy())]
    start = bsys_starts(truth, 3.0 * sigma, pars, 4, seed=3)

    mle, _ = lbfgs_batch(fwd, prior, data, truth, 200, sigma_ref=sigma,
                         n_restarts=3, start=start, precondition=True,
                         max_eval=4000)
    err = np.abs(mle - theta_star[None, :]) / sigma[None, :]
    assert err.max() < 0.05, f"worst {err.max():.3f} sigma from theta_star"
    # every start must land in the same place, or the spread is not a
    # convergence certificate
    spread = (mle.max(0) - mle.min(0)) / sigma
    assert spread.max() < 0.05


def _capture(records):
    """Collect ``(t, ls_func_evals)`` from every line search in the block.

    ``tol_change`` deliberately left at None -- overriding it to 0 is the bug
    ``test_zero_line_search_tolerance_makes_cubic_interpolation_nan`` pins."""
    from mle_reseed import _line_search_hook
    return _line_search_hook(records)
