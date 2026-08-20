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
nested sampling      the same ``logp``, plus :meth:`BoxPrior.ptform` mapping the
(UltraNest)          unit cube to parameters
HMC/NUTS, VI         :meth:`PoissonPosterior.potential` -- the negative log
                     posterior in *unconstrained* space with its log-Jacobian,
                     differentiable, so leapfrog and ELBO gradients both work
===================  ==========================================================

Priors are boxes, matching how the fit parameters have always been specified
(``Param(low, high)``, ``fisher_bias.Par``). That keeps the unconstrained
transform a plain scaled logit rather than a per-parameter dispatch.
"""
from typing import Optional, Sequence

import numpy as np
import torch


class BoxPrior:
    """Independent uniform priors on a box, with the transforms samplers need.

    Nested samplers want the unit cube mapped to the box (``ptform``);
    gradient-based samplers want an unbounded space to move in, which here is a
    scaled logit -- ``theta = lo + (hi - lo) * sigmoid(z)`` -- whose log-Jacobian
    must be added to the log posterior or the bounds silently bias the result.
    """

    def __init__(self, lo, hi, names: Optional[Sequence[str]] = None,
                 device: str = "cpu"):
        self.lo = torch.as_tensor(np.asarray(lo, dtype=np.float64),
                                  dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(np.asarray(hi, dtype=np.float64),
                                  dtype=torch.float64, device=device)
        if (self.hi <= self.lo).any():
            bad = [i for i in range(self.lo.numel()) if self.hi[i] <= self.lo[i]]
            raise ValueError(f"prior bounds must satisfy hi > lo; bad at {bad}")
        self.span = self.hi - self.lo
        self.ndim = int(self.lo.numel())
        self.names = list(names) if names is not None else [
            f"p{i}" for i in range(self.ndim)]

    # --- gradient-free view --------------------------------------------------

    def inside(self, theta: np.ndarray) -> np.ndarray:
        """(B, ndim) -> (B,) bool: is each row within the box?"""
        lo = self.lo.cpu().numpy()
        hi = self.hi.cpu().numpy()
        return np.all((theta >= lo) & (theta <= hi), axis=1)

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """(B, ndim) -> (B,) zeros: a flat prior is a constant.

        Deliberately *unnormalised*, so ``logp`` keeps the exact values it had
        before priors became pluggable and every existing study stays
        reproducible. The general
        :class:`~spexai.inference.priors.PriorSet` normalises instead, which
        matters only for absolute log Z."""
        return np.zeros(np.atleast_2d(theta).shape[0])

    def ptform(self, cube) -> np.ndarray:
        """Unit cube (B, ndim) -> parameters (B, ndim), for nested sampling."""
        u = np.atleast_2d(np.asarray(cube, dtype=np.float64))
        return self.lo.cpu().numpy() + u * self.span.cpu().numpy()

    # --- unconstrained view --------------------------------------------------

    def to_constrained(self, z: torch.Tensor):
        """Unconstrained ``z`` -> ``(theta, log|det J|)``, both differentiable.

        The Jacobian term is not optional bookkeeping: without it the sampler
        targets the density times an implicit sigmoid-shaped weight, which pulls
        posteriors toward the middle of every box."""
        s = torch.sigmoid(z)
        theta = self.lo + self.span * s
        # d(theta)/dz = span * s * (1 - s); log|det J| sums over parameters
        logdet = (torch.log(self.span) + torch.log(s.clamp_min(1e-300))
                  + torch.log((1.0 - s).clamp_min(1e-300))).sum(-1)
        return theta, logdet

    def to_unconstrained(self, theta: torch.Tensor) -> torch.Tensor:
        """Parameters -> unconstrained ``z`` (the inverse of the above)."""
        s = ((theta - self.lo) / self.span).clamp(1e-12, 1 - 1e-12)
        return torch.log(s) - torch.log1p(-s)

    @classmethod
    def from_params(cls, params, device: str = "cpu") -> "BoxPrior":
        """Build from anything with ``.name``/``.low``/``.high`` -- both
        ``fitting.Param`` and ``fisher_bias.Par`` qualify."""
        return cls([p.low for p in params], [p.high for p in params],
                   [p.name for p in params], device=device)


class PoissonPosterior:
    """Poisson log posterior for one observation, vectorised over walkers.

    ``forward`` maps ``(B, ndim)`` to predicted counts ``(B, n_keep)``; ``data``
    is the observed counts on the same channels. The likelihood drops the
    ``log(d!)`` constant, which is fine for every sampler here (it shifts log Z
    by a constant too, so nested-sampling evidence *ratios* are unaffected).
    """

    def __init__(self, forward, data, prior: BoxPrior, mu_floor: float = 1e-30):
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
