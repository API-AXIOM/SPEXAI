"""One vectorised posterior, shared by every sampler.

The point of this module is that the sampler bake-off compares *samplers*, not
likelihood implementations. emcee, zeus, UltraNest, NUTS and VI all consume the
same :class:`PoissonPosterior` over the same :class:`VectorForward`, so a
difference between their posteriors is a difference between the algorithms.

Each sampler family needs a different view of the same object, and all of them
are here:

===================  ==========================================================
gradient-free MCMC   :meth:`PoissonPosterior.logp` -- ``(B, ndim)`` -> ``(B,)``,
(emcee, zeus)        numpy in and out, walkers batched into one forward
nested sampling      the same ``logp``, plus :meth:`PriorSet.ptform` mapping the
(UltraNest)          unit cube to parameters
HMC/NUTS, VI         :meth:`PoissonPosterior.potential` -- the negative log
                     posterior in *unconstrained* space with its log-Jacobian,
                     differentiable, so leapfrog and ELBO gradients both work
===================  ==========================================================

The prior itself lives in :mod:`spexai.inference.priors`
(:class:`~spexai.inference.priors.PriorSet`). This module used to carry its
own uniform-only ``BoxPrior``; that was removed once ``PriorSet`` -- which
serves the same three interfaces and adds a real ``logpdf`` -- became the only
prior in the package. The unconstrained transform is still a plain scaled
logit, so nothing here needs a per-parameter dispatch.
"""

import numpy as np
import torch


class PoissonPosterior:
    """Poisson log posterior for one observation, vectorised over walkers.

    ``forward`` maps ``(B, ndim)`` to predicted counts ``(B, n_keep)``; ``data``
    is the observed counts on the same channels. The likelihood drops the
    ``log(d!)`` constant, which is fine for every sampler here (it shifts log Z
    by a constant too, so nested-sampling evidence *ratios* are unaffected).
    """

    def __init__(self, forward, data, prior, mu_floor: float = 1e-30):
        self.forward = forward
        self.prior = prior
        self.device = forward.device
        self.mu_floor = float(mu_floor)
        self.data_np = np.asarray(data, dtype=np.float64)
        self.data = torch.as_tensor(self.data_np, dtype=torch.float32,
                                    device=self.device)
        if self.data.ndim != 1:
            raise ValueError(f"data must be 1-D, got {tuple(self.data.shape)}")
        self.n_eval = 0            # forwards consumed -- the bake-off's currency

    # --- gradient-free -------------------------------------------------------

    def loglike(self, theta) -> np.ndarray:
        """(B, ndim) -> (B,) log-likelihood. No prior, no bounds check."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        self.n_eval += th.shape[0]
        mu = np.clip(self.forward(th), self.mu_floor, None)
        return (self.data_np[None, :] * np.log(mu) - mu).sum(1)

    def logp(self, theta) -> np.ndarray:
        """(B, ndim) -> (B,) log posterior, ``-inf`` outside the prior support.

        Out-of-bounds rows are dropped before the forward rather than computed
        and discarded: with a wide box that is most of the early ensemble, and
        the forward is the entire cost.

        The prior *density* is added here and **only** here. emcee and zeus
        reach the prior through this method, so a non-uniform prior has no
        effect unless it is summed in. UltraNest instead calls ``loglike``
        with points already drawn through ``prior.ptform``, where the prior is
        encoded in the sampling rather than the density -- adding ``logpdf``
        there too would apply the prior twice. That asymmetry is why
        ``loglike`` stays a pure likelihood."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        ok = self.prior.inside(th)
        out = np.full(th.shape[0], -np.inf)
        if ok.any():
            out[ok] = self.loglike(th[ok]) + self.prior.logpdf(th[ok])
        return out

    # --- gradient-based ------------------------------------------------------

    def loglike_torch(self, th: torch.Tensor) -> torch.Tensor:
        """Differentiable ``(B, ndim)`` -> ``(B,)`` log-likelihood.

        Dtype follows the forward's output rather than being imposed here, so a
        float64 forward stays float64 all the way to the gradient."""
        self.n_eval += int(th.shape[0])
        mu = self.forward.counts_torch(th, grad=True).clamp_min(self.mu_floor)
        return (self.data.to(mu.dtype) * torch.log(mu) - mu).sum(-1)

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """Negative log posterior in unconstrained space -> ``(B,)``.

        What HMC/NUTS integrates and VI minimises in expectation. The box prior
        is flat, so it contributes only through the transform's log-Jacobian --
        which is exactly the term that is easy to forget and impossible to see
        in the output."""
        theta, logdet = self.prior.to_constrained(z.double())
        ll = self.loglike_torch(theta)
        return -(ll.double() + logdet)

    def potential_and_grad(self, z: torch.Tensor):
        """``(potential, d potential/dz)``, both ``(B, ...)``. One backward."""
        z = z.detach().requires_grad_(True)
        u = self.potential(z)
        grad, = torch.autograd.grad(u.sum(), z)
        return u.detach(), grad
