"""The shared posterior must agree with the scalar reference likelihood.

Every sampler in the bake-off consumes ``PoissonPosterior``, so a bug here is a
bug in all of them at once, and it would land as miscalibration in exactly the
SBC campaign meant to detect miscalibration. These tests pin the pieces that
can go wrong silently: the unconstrained transform's log-Jacobian (omitting it
biases posteriors toward the middle of every box, invisibly), the agreement
between the vectorised and scalar likelihoods, and the gradients.
"""
import numpy as np
import pytest
import torch

from spexai.inference.posterior import BoxPrior, PoissonPosterior


@pytest.fixture
def prior():
    return BoxPrior([0.0, -2.0, 1.0], [1.0, 5.0, 3.0], ["a", "b", "c"])


def test_ptform_maps_cube_corners_to_box(prior):
    got = prior.ptform(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                 [0.5, 0.5, 0.5]]))
    assert np.allclose(got[0], [0.0, -2.0, 1.0])
    assert np.allclose(got[1], [1.0, 5.0, 3.0])
    assert np.allclose(got[2], [0.5, 1.5, 2.0])


def test_inside_flags_out_of_box_rows(prior):
    theta = np.array([[0.5, 0.0, 2.0],      # inside
                      [1.5, 0.0, 2.0],      # a above hi
                      [0.5, -3.0, 2.0]])    # b below lo
    assert list(prior.inside(theta)) == [True, False, False]


def test_unconstrained_roundtrip(prior):
    theta = torch.tensor([[0.25, 1.0, 2.5], [0.9, -1.5, 1.2]],
                         dtype=torch.float64)
    back, _ = prior.to_constrained(prior.to_unconstrained(theta))
    assert torch.allclose(back, theta, atol=1e-9)


def test_log_jacobian_matches_autodiff(prior):
    # the log|det J| must equal log|d theta / dz|; a wrong (or missing) term is
    # invisible in the samples but biases every gradient-based posterior
    z = torch.tensor([[0.3, -1.2, 0.8]], dtype=torch.float64,
                     requires_grad=True)
    theta, logdet = prior.to_constrained(z)
    jac = torch.autograd.functional.jacobian(
        lambda zz: prior.to_constrained(zz)[0].squeeze(0),
        z.detach()).squeeze(1)                          # (ndim, ndim)
    expected = torch.log(torch.abs(torch.det(jac)))
    assert torch.allclose(logdet.squeeze(0), expected, atol=1e-8)


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="hi > lo"):
        BoxPrior([0.0, 5.0], [1.0, 2.0])


class _LinearForward:
    """Tiny stand-in for VectorForward: counts linear in theta, so the Poisson
    log-likelihood is analytic and the posterior's arithmetic can be checked
    without loading the emulator."""

    device = "cpu"

    def __init__(self, weights):
        self.w = torch.as_tensor(weights, dtype=torch.float32)   # (ndim, n_ch)

    def counts_torch(self, th, grad=False):
        return torch.exp(th @ self.w.to(th.dtype))               # (B, n_ch)

    def __call__(self, theta):
        th = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float64)
        return self.counts_torch(th).detach().cpu().numpy()


@pytest.fixture
def toy():
    g = torch.Generator().manual_seed(0)
    fwd = _LinearForward(torch.rand(3, 5, generator=g))
    prior = BoxPrior([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    data = np.array([3.0, 0.0, 7.0, 2.0, 11.0])      # a zero channel included
    return PoissonPosterior(fwd, data, prior)


def test_logp_matches_hand_computed_poisson(toy):
    theta = np.array([[0.2, 0.4, 0.6]])
    mu = toy.forward(theta)[0]
    expect = float((toy.data_np * np.log(mu) - mu).sum())
    assert np.isclose(toy.logp(theta)[0], expect)


def test_logp_is_minus_inf_outside_the_box(toy):
    got = toy.logp(np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]))
    assert np.isfinite(got[0]) and got[1] == -np.inf


def test_out_of_bounds_rows_skip_the_forward(toy):
    # the forward is the entire cost, so rejected rows must not reach it
    toy.n_eval = 0
    toy.logp(np.array([[0.5, 0.5, 0.5], [9.0, 9.0, 9.0], [0.1, 0.2, 0.3]]))
    assert toy.n_eval == 2


def test_torch_and_numpy_likelihoods_agree(toy):
    theta = np.array([[0.2, 0.4, 0.6], [0.7, 0.1, 0.9]])
    ref = toy.loglike(theta)
    got = toy.loglike_torch(torch.as_tensor(theta, dtype=torch.float64))
    assert np.allclose(got.detach().numpy(), ref, rtol=1e-5)


def test_potential_includes_the_log_jacobian(toy):
    # potential = -(loglike + logdet); dropping logdet is the classic bug
    z = torch.tensor([[0.3, -0.7, 1.1]], dtype=torch.float64)
    theta, logdet = toy.prior.to_constrained(z)
    ll = toy.loglike(theta.detach().numpy())[0]
    assert np.isclose(float(toy.potential(z)[0]), -(ll + float(logdet[0])),
                      rtol=1e-5)


def test_potential_grad_matches_finite_differences(toy):
    z = torch.tensor([[0.3, -0.7, 1.1]], dtype=torch.float64)
    _, grad = toy.potential_and_grad(z)
    for i in range(3):
        h = 1e-4
        zp, zm = z.clone(), z.clone()
        zp[0, i] += h
        zm[0, i] -= h
        fd = float((toy.potential(zp) - toy.potential(zm))[0]) / (2 * h)
        assert np.isclose(float(grad[0, i]), fd, rtol=1e-3, atol=1e-6), i
