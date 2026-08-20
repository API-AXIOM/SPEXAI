"""User-facing prior specification, shared by every sampler.

Until now the package offered exactly one prior: a uniform box
(:class:`~spexai.inference.posterior.BoxPrior`). That is fine for the internal
studies, where flat priors are the deliberate choice, but it makes the package
unusable by anyone who wants to bring real external information -- an
abundance measured elsewhere, a literature velocity dispersion, an ``n_H`` from
an HI survey.

The obstacle was never Pyro: :class:`~spexai.inference.ppl.SpectrumModel`
already accepts any mapping of name to Pyro distribution, so NUTS and VI have
been general all along. It was the *gradient-free* samplers, which reach the
prior through three different interfaces:

===============  ==========================================================
emcee / zeus     ``logpdf(theta)`` added to the log-likelihood
UltraNest        ``ptform(cube)``, the inverse CDF -- **not** a density
NUTS / VI        a Pyro distribution, plus a transform to unconstrained space
===============  ==========================================================

A :class:`Prior` therefore has to supply all three from one declaration, and
:class:`PriorSet` is the drop-in replacement for ``BoxPrior`` that does so. The
uniform case stays bit-compatible with the old behaviour, so existing studies
are unaffected.

    from spexai.inference.priors import PriorSet, Uniform, Normal, LogUniform

    prior = PriorSet({
        "kT":       Uniform(1.5, 7.5),
        "Fe":       Normal(0.55, 0.05, low=0.0),   # a measurement, truncated
        "sigma_v":  Uniform(30.0, 600.0),
        "n_h":      Normal(1.36, 0.10, low=0.0),   # 1e21 cm^-2, from an HI map
        "log_norm": Uniform(10.0, 12.0),
    }, names=forward.names)

Every prior is independent (a product prior). Correlated priors need a joint
distribution, which is a Pyro-only feature and would not survive the
``ptform`` interface -- see :meth:`PriorSet.ptform`.
"""
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


class Prior:
    """One parameter's prior. Subclasses supply the three views."""

    #: finite support, used to seed walkers and to validate
    low: float
    high: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """(B,) -> (B,) log density, ``-inf`` outside the support."""
        raise NotImplementedError

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF on ``u in [0, 1]`` -- what nested sampling consumes."""
        raise NotImplementedError

    def to_pyro(self, device=None):
        """The equivalent Pyro distribution, for NUTS and VI.

        ``device`` places the distribution's parameters, which is what decides
        where Pyro's sample sites -- and hence the sampler's own state -- live.
        """
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.ppf(rng.random(size))

    @property
    def center(self) -> float:
        """A sensible starting point -- the median."""
        return float(self.ppf(np.array([0.5]))[0])


class Uniform(Prior):
    """Flat on ``[low, high]``. The package default, and the old behaviour."""

    def __init__(self, low: float, high: float):
        if not high > low:
            raise ValueError(f"Uniform needs high > low, got {low}, {high}")
        self.low, self.high = float(low), float(high)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, -np.log(self.high - self.low), -np.inf)

    def ppf(self, u):
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)

    def to_pyro(self, device=None):
        import pyro.distributions as dist
        return dist.Uniform(
            torch.tensor(self.low, dtype=torch.float64, device=device),
            torch.tensor(self.high, dtype=torch.float64, device=device))

    def __repr__(self):
        return f"Uniform({self.low:g}, {self.high:g})"


class LogUniform(Prior):
    """Flat in ``log10(x)`` -- the right default for a scale like a norm."""

    def __init__(self, low: float, high: float):
        if not (0 < low < high):
            raise ValueError(f"LogUniform needs 0 < low < high, got {low}, "
                             f"{high}")
        self.low, self.high = float(low), float(high)
        self._ll, self._lh = np.log(self.low), np.log(self.high)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.low) & (x <= self.high)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = -np.log(x) - np.log(self._lh - self._ll)
        return np.where(inside, lp, -np.inf)

    def ppf(self, u):
        return np.exp(self._ll + np.asarray(u, dtype=float)
                      * (self._lh - self._ll))

    def to_pyro(self, device=None):
        import pyro.distributions as dist
        base = dist.Uniform(
            torch.tensor(self._ll, dtype=torch.float64, device=device),
            torch.tensor(self._lh, dtype=torch.float64, device=device))
        return dist.TransformedDistribution(
            base, torch.distributions.ExpTransform())

    def __repr__(self):
        return f"LogUniform({self.low:g}, {self.high:g})"


class Normal(Prior):
    """Gaussian, optionally truncated -- an external measurement.

    ``low``/``high`` default to +-6 sigma rather than infinity. The samplers
    need a finite box to initialise walkers and to build a unit cube, and 6
    sigma removes none of the probability that matters (2e-9 per tail) while
    keeping every interface well defined.
    """

    def __init__(self, mu: float, sigma: float, low: Optional[float] = None,
                 high: Optional[float] = None):
        if sigma <= 0:
            raise ValueError(f"Normal needs sigma > 0, got {sigma}")
        self.mu, self.sigma = float(mu), float(sigma)
        self.low = float(mu - 6 * sigma) if low is None else float(low)
        self.high = float(mu + 6 * sigma) if high is None else float(high)
        if not self.high > self.low:
            raise ValueError(f"Normal needs high > low, got {self.low}, "
                             f"{self.high}")
        from scipy import stats
        self._d = stats.truncnorm((self.low - self.mu) / self.sigma,
                                  (self.high - self.mu) / self.sigma,
                                  loc=self.mu, scale=self.sigma)

    def logpdf(self, x):
        return self._d.logpdf(np.asarray(x, dtype=float))

    def ppf(self, u):
        return self._d.ppf(np.asarray(u, dtype=float))

    def to_pyro(self, device=None):
        import pyro.distributions as dist
        # Pyro has no truncated normal; the box is imposed by the sampler's
        # own support constraint, and at +-6 sigma the difference is 4e-9
        return dist.Normal(
            torch.tensor(self.mu, dtype=torch.float64, device=device),
            torch.tensor(self.sigma, dtype=torch.float64, device=device))

    def __repr__(self):
        return (f"Normal({self.mu:g}, {self.sigma:g}, low={self.low:g}, "
                f"high={self.high:g})")


class PriorSet:
    """A product prior over named parameters -- the ``BoxPrior`` replacement.

    Exposes the same attributes ``BoxPrior`` did (``lo``, ``hi``, ``ndim``,
    ``names``, ``inside``, ``ptform``, ``to_constrained``,
    ``to_unconstrained``) so :class:`~spexai.inference.posterior.PoissonPosterior`
    and every sampler keep working, and adds :meth:`logpdf`, which is what
    makes non-uniform priors actually influence the fit.
    """

    def __init__(self, priors: Mapping[str, Prior],
                 names: Optional[Sequence[str]] = None, device: str = "cpu"):
        self.names: List[str] = list(names) if names is not None \
            else list(priors)
        missing = [n for n in self.names if n not in priors]
        if missing:
            raise ValueError(f"no prior supplied for {missing}")
        extra = [n for n in priors if n not in self.names]
        if extra:
            raise ValueError(f"priors given for unknown parameters {extra}; "
                             f"expected only {self.names}")
        self.priors = [priors[n] for n in self.names]
        self.ndim = len(self.names)
        lo = np.array([p.low for p in self.priors], dtype=float)
        hi = np.array([p.high for p in self.priors], dtype=float)
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.span = self.hi - self.lo
        self._lo_np, self._hi_np = lo, hi
        self.device = device

    # --- gradient-free view --------------------------------------------------

    def inside(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.all((theta >= self._lo_np) & (theta <= self._hi_np), axis=1)

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """(B, ndim) -> (B,) log prior density.

        This is the method ``BoxPrior`` never needed: with a flat prior the
        density is a constant that samplers ignore, so the old code could get
        away with a bounds check alone. Any other prior must be *added* to the
        log-likelihood or it has no effect at all -- a silent failure that
        would look like the user's prior being ignored, because it would be.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        out = np.zeros(theta.shape[0])
        for i, p in enumerate(self.priors):
            out += p.logpdf(theta[:, i])
        return out

    def ptform(self, cube) -> np.ndarray:
        """Unit cube -> parameters, per-parameter inverse CDF.

        Independence is what makes this possible: a joint prior has no
        coordinate-wise inverse CDF, which is why correlated priors are not
        supported for nested sampling.
        """
        u = np.atleast_2d(np.asarray(cube, dtype=float))
        out = np.empty_like(u)
        for i, p in enumerate(self.priors):
            out[:, i] = p.ppf(u[:, i])
        return out

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """(size, ndim) draws from the prior -- what SBC needs."""
        return np.stack([p.sample(rng, size) for p in self.priors], axis=-1)

    @property
    def center(self) -> np.ndarray:
        return np.array([p.center for p in self.priors])

    # --- unconstrained view (identical to BoxPrior) --------------------------

    def to_constrained(self, z: torch.Tensor):
        s = torch.sigmoid(z)
        theta = self.lo + self.span * s
        logdet = (torch.log(self.span) + torch.log(s.clamp_min(1e-300))
                  + torch.log((1.0 - s).clamp_min(1e-300))).sum(-1)
        return theta, logdet

    def to_unconstrained(self, theta: torch.Tensor) -> torch.Tensor:
        s = ((theta - self.lo) / self.span).clamp(1e-12, 1 - 1e-12)
        return torch.log(s) - torch.log1p(-s)

    # --- Pyro view -----------------------------------------------------------

    def to_pyro(self, device=None) -> Dict[str, "object"]:
        """``{name: pyro distribution}`` for :class:`ppl.SpectrumModel`.

        Defaults to the set's own device so Pyro's sample sites land where the
        forward is. Building these on CPU against a CUDA forward is the exact
        mismatch that made NUTS die inside ``VectorForward.fold``.
        """
        device = self.device if device is None else device
        return {n: p.to_pyro(device=device)
                for n, p in zip(self.names, self.priors)}

    # --- constructors --------------------------------------------------------

    @classmethod
    def uniform(cls, names, lo, hi, device: str = "cpu") -> "PriorSet":
        """The old box prior, expressed in the new API."""
        lo = np.asarray(lo, dtype=float).reshape(-1)
        hi = np.asarray(hi, dtype=float).reshape(-1)
        return cls({n: Uniform(lo[i], hi[i]) for i, n in enumerate(names)},
                   names=names, device=device)

    @classmethod
    def from_params(cls, params, device: str = "cpu") -> "PriorSet":
        """From ``fitting.Param`` / ``fisher_bias.Par`` -- uniform on bounds."""
        return cls.uniform([p.name for p in params], [p.low for p in params],
                           [p.high for p in params], device=device)

    def __repr__(self):
        body = ", ".join(f"{n}={p!r}" for n, p in zip(self.names, self.priors))
        return f"PriorSet({body})"
