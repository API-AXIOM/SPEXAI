"""Temperature-distribution (DEM) models for multi-temperature fitting.

The emission-measure-weighted spectrum is ``integral DEM(T) f(T) dT``, which on
a discrete temperature grid is the weighted sum ``predict_counts_dem`` takes.
Each model here exposes the contract that ``fitting.make_loglike`` expects:

* ``temp_grid`` -- a 1-D tensor of temperatures (keV) to evaluate the emulator at
* ``weights(params)`` -- a 1-D tensor of non-negative weights (the total
  emission measure is carried separately by ``norm``). Free-weight shapes
  (``BinnedDEM``) renormalise these to sum to 1, because their weights are
  otherwise exactly degenerate with ``norm``. Parametric shapes (including the
  two-Gaussian mixture) do **not**: their weights are ``pdf * quadrature``,
  whose sum is the fraction of the distribution the grid contains and is
  meant to be read as such.
* ``param_names`` -- the fit parameters the DEM consumes
* ``suggested_bounds()`` -- default ``{name: (low, high)}`` for building Params

Parametric shapes wrap ``scipy.stats`` distributions (per the design decision to
reuse existing libraries); ``BinnedDEM`` is the non-parametric free-weight option
(no smoothness regulariser).

**Batched weights.** ``weights`` above is scalar: it calls ``float()`` on each
parameter and evaluates a scipy pdf, so it can neither accept one value per
walker nor carry a gradient. A vectorised fit needs both -- the whole ensemble
in one forward, and (for NUTS/VI) ``d weights / d theta``. Shapes that can be
written in closed form therefore also provide:

* ``weights_batch(params)`` -- ``{name: (B,) tensor}`` -> ``(B, G)`` tensor,
  pure torch, differentiable, with the same normalisation as ``weights``.

It is deliberately optional. ``ParametricDEM`` accepts any frozen scipy
distribution, and most have no torch equivalent; those keep only the scalar
path, and ``fitting`` falls back to the scalar likelihood for them rather than
pretending. The presets built here (Gaussian in log T or T), the two-Gaussian
mixture and the binned DEM are all closed-form, so all of them batch.
``tests/test_dem.py`` asserts the two agree row by row.
"""
from typing import Callable, Dict, List, Sequence, Tuple

import math

import numpy as np
import torch


# The per-element PCHIP SPEX truth model (``spex_truth.SpexTruthModel``) is
# fit down to ~0.501 keV; a temperature grid point below that extrapolates
# and can blow up (e.g. Ar -> inf at 0.5 keV). A grid meant to feed a truth
# model, not just the emulator, should stay above this with a safety margin.
PCHIP_TRUTH_SAFE_LO_KEV = 0.7

# Top of the emulator's trained TEMPERATURE range -- the intersection over the
# 30 element checkpoints, which are not bit-identical (min t_hi = 19.9415007
# keV). Truncated downward so a grid endpoint sits strictly inside it. This is
# the plasma kT axis, NOT the photon-energy axis (which runs 0.1-12 keV); the
# two share a unit and nothing else. A stale value fails loudly rather than
# silently: JointOperatorModel.check_temperature raises on any grid point
# outside the trained range.
EMULATOR_T_HI_KEV = 19.9415


def _normalise(raw: torch.Tensor) -> torch.Tensor:
    """Rows to sum to 1, leaving an all-zero row alone rather than dividing by
    zero (which would turn an out-of-range DEM into NaNs instead of a flat
    -inf likelihood the sampler can simply reject)."""
    s = raw.sum(dim=-1, keepdim=True)
    return torch.where(s > 0, raw / s.clamp(min=1e-300), raw)


def _gauss_pdf(x: torch.Tensor, loc: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
    """Normal pdf, broadcasting ``x (G,)`` against ``loc/scale (B, 1)``."""
    z = (x - loc) / scale
    return torch.exp(-0.5 * z * z) / (scale * math.sqrt(2.0 * math.pi))


class TempGrid:
    """A fixed temperature grid + log-T quadrature weights for parametric DEMs.

    ``log=True`` spaces the grid uniformly in log10 T (the natural variable for
    an X-ray emission-measure distribution). ``dlogt`` is the per-node quadrature
    weight for integrating a density defined in log10 T.
    """

    def __init__(self, t_lo: float = 0.2, t_hi: float = 10.0, n: int = 60,
                 log: bool = True):
        if log:
            temps = np.logspace(np.log10(t_lo), np.log10(t_hi), n)
        else:
            temps = np.linspace(t_lo, t_hi, n)
        self.temps_np = temps.astype(np.float64)
        self.logtemps = np.log10(self.temps_np)
        self.dlogt = np.gradient(self.logtemps)
        self.dt = np.gradient(self.temps_np)
        self.temp_grid = torch.as_tensor(self.temps_np, dtype=torch.float32)


class ParametricDEM:
    """DEM whose shape is a frozen ``scipy.stats`` distribution.

    ``dist_factory(values)`` returns a frozen distribution given the sampled
    parameter values (in order of ``param_names``); ``variable`` selects whether
    the distribution lives in ``"logT"`` (log10 keV) or ``"T"`` (keV). Weights
    are ``pdf(x_g) * quadrature_g``, **not** renormalised.

    Renormalising would be wrong here. Writing the emission-measure
    distribution as ``Y(T) = Y_tot p(T)`` with ``p`` a normalised pdf, the
    observable spectrum is ``Y_tot * sum_g p(T_g) dx_g f(T_g)``, and
    ``sum_g p(T_g) dx_g`` is the fraction of the distribution lying inside the
    grid. For a contained, resolved DEM that quadrature is 1 to four decimals
    and the division was a no-op; the only case where it did anything was a
    distribution running off the grid, and there it asserted that all of
    ``Y_tot`` sits on the grid when it does not -- silently redistributing the
    missing emission measure across the temperatures that remain. The sum is
    now left alone, so it doubles as a diagnostic: a value below 1 says this
    much of the DEM lies outside the representable temperature range.

    ``BinnedDEM`` still renormalises, and must: its free per-bin weights are
    otherwise exactly degenerate with ``norm``.
    """

    def __init__(self, grid: TempGrid, dist_factory: Callable[[Sequence[float]], object],
                 param_names: Sequence[str], variable: str = "logT",
                 bounds: Dict[str, Tuple[float, float]] = None,
                 torch_pdf: Callable = None):
        self.grid = grid
        self.temp_grid = grid.temp_grid
        self.param_names = list(param_names)
        self._factory = dist_factory
        self._x = grid.logtemps if variable == "logT" else grid.temps_np
        self._quad = grid.dlogt if variable == "logT" else grid.dt
        self._bounds = dict(bounds or {})
        # a torch equivalent of dist_factory's pdf, when one exists; its absence
        # is what makes this DEM scalar-only (see the module docstring)
        self._torch_pdf = torch_pdf
        if torch_pdf is not None:
            self._xt = torch.as_tensor(self._x, dtype=torch.float32)
            self._quadt = torch.as_tensor(self._quad, dtype=torch.float32)

    def weights(self, params: Dict[str, float]) -> torch.Tensor:
        """``pdf(x_g) * quadrature_g``, NOT renormalised -- see the class
        docstring. The sum is the fraction of the distribution the grid
        contains, and callers may read it as exactly that."""
        dist = self._factory([float(params[n]) for n in self.param_names])
        raw = np.asarray(dist.pdf(self._x), dtype=np.float64) * self._quad
        return torch.as_tensor(raw, dtype=torch.float32)

    def weights_batch(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{name: (B,)}`` -> ``(B, G)``. Only for closed-form shapes."""
        if self._torch_pdf is None:
            raise NotImplementedError(
                f"{type(self).__name__} wraps a scipy distribution with no "
                "torch equivalent, so it has no batched form; fits using it "
                "run on the scalar likelihood")
        vals = [torch.as_tensor(params[n]).reshape(-1, 1)
                for n in self.param_names]                    # each (B, 1)
        x = self._xt.to(vals[0].device)
        raw = self._torch_pdf(x, *vals) * self._quadt.to(vals[0].device)
        return raw                                            # (B, G), NOT normalised

    def suggested_bounds(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._bounds)


class TwoGaussianDEM:
    """Two log-T Gaussians with a mixing fraction (a bimodal DEM).

    Params: ``logT1, sig1, logT2, sig2, frac`` (frac in [0,1] weights the first
    component). Weights are ``pdf * dlogT``, **not** renormalised, for the same
    reason as ``ParametricDEM``: the mixture is a normalised pdf, so the sum is
    the fraction of it the grid contains, and dividing by it would silently
    redistribute an off-grid component's emission measure onto the grid.
    """

    def __init__(self, grid: TempGrid, names=("logT1", "sig1", "logT2", "sig2",
                                              "frac")):
        self.grid = grid
        self.temp_grid = grid.temp_grid
        self.param_names = list(names)
        self._x = grid.logtemps
        self._quad = grid.dlogt

    def weights(self, params: Dict[str, float]) -> torch.Tensor:
        from scipy.stats import norm
        m1, s1, m2, s2, f = (float(params[n]) for n in self.param_names)
        f = min(max(f, 0.0), 1.0)
        pdf = f * norm(m1, s1).pdf(self._x) + (1 - f) * norm(m2, s2).pdf(self._x)
        raw = pdf * self._quad                  # NOT renormalised, see class doc
        return torch.as_tensor(raw, dtype=torch.float32)

    def weights_batch(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{name: (B,)}`` -> ``(B, G)``, differentiable, NOT renormalised."""
        m1, s1, m2, s2, f = (torch.as_tensor(params[n]).reshape(-1, 1)
                             for n in self.param_names)
        f = f.clamp(0.0, 1.0)
        x = torch.as_tensor(self._x, dtype=torch.float32, device=f.device)
        quad = torch.as_tensor(self._quad, dtype=torch.float32, device=f.device)
        pdf = f * _gauss_pdf(x, m1, s1) + (1.0 - f) * _gauss_pdf(x, m2, s2)
        return pdf * quad                                     # (B, G)

    def suggested_bounds(self):
        return {"logT1": (np.log10(0.3), np.log10(10.0)), "sig1": (0.02, 0.6),
                "logT2": (np.log10(0.3), np.log10(10.0)), "sig2": (0.02, 0.6),
                "frac": (0.0, 1.0)}


class BinnedDEM:
    """Non-parametric DEM: one free weight per (coarse) temperature bin.

    The grid *is* the set of coarse bins; the sampled weights (clamped to be
    non-negative) are renormalised to sum to 1. No smoothness regulariser, by
    design. Params are ``dem0..dem{n-1}``.
    """

    def __init__(self, t_lo: float = 0.2, t_hi: float = 10.0, n_bins: int = 8,
                 log: bool = True):
        if log:
            temps = np.logspace(np.log10(t_lo), np.log10(t_hi), n_bins)
        else:
            temps = np.linspace(t_lo, t_hi, n_bins)
        self.temp_grid = torch.as_tensor(temps, dtype=torch.float32)
        self.param_names = [f"dem{i}" for i in range(n_bins)]

    def weights(self, params: Dict[str, float]) -> torch.Tensor:
        w = np.array([max(float(params[n]), 0.0) for n in self.param_names])
        s = w.sum()
        return torch.as_tensor(w / s if s > 0 else w, dtype=torch.float32)

    def weights_batch(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{name: (B,)}`` -> ``(B, G)``, differentiable.

        The clamp at zero is a hard non-negativity constraint, so a walker
        sitting at a negative weight gets exactly zero gradient on that bin --
        correct, and worth knowing when a binned DEM fit stalls."""
        w = torch.stack([torch.as_tensor(params[n]).reshape(-1)
                         for n in self.param_names], dim=-1)   # (B, n_bins)
        return _normalise(w.clamp(min=0.0))

    def suggested_bounds(self):
        return {n: (0.0, 1.0) for n in self.param_names}


# --- named parametric presets ----------------------------------------------

def gaussian_logT(grid: TempGrid, mean: str = "logT_mean",
                  sigma: str = "logT_sigma") -> ParametricDEM:
    """Gaussian in log10 T (params in log10 keV) -- SPEX's own ``gdem``.

    Default bounds are the bias campaign's design range: centre 0.7-15 keV
    (the PCHIP-safe truth floor up to the hottest clusters), width 0.056-0.4
    dex. The width floor is a RESOLUTION limit, not physics: on the 70-node
    campaign grid (0.0211 dex per cell) it is ~2.7 cells, and below it the
    width's Fisher information comes from grid interpolation."""
    from scipy.stats import norm
    return ParametricDEM(
        grid, lambda v: norm(loc=v[0], scale=v[1]), [mean, sigma],
        variable="logT",
        bounds={mean: (np.log10(PCHIP_TRUTH_SAFE_LO_KEV), np.log10(15.0)),
                sigma: (0.056, 0.4)},
        torch_pdf=_gauss_pdf)


def gaussian_T(grid: TempGrid, mean: str = "T_mean",
               sigma: str = "T_sigma") -> ParametricDEM:
    """Gaussian in linear T (keV) -- matches the thesis DEM parametrisation."""
    from scipy.stats import norm
    return ParametricDEM(
        grid, lambda v: norm(loc=v[0], scale=v[1]), [mean, sigma],
        variable="T", bounds={mean: (0.3, 10.0), sigma: (0.05, 4.0)},
        torch_pdf=_gauss_pdf)


def lognormal_T(grid: TempGrid, mu: str = "logT_mean",
                sigma: str = "logT_sigma") -> ParametricDEM:
    """Log-normal in T == Gaussian in log10 T (alias of gaussian_logT)."""
    return gaussian_logT(grid, mean=mu, sigma=sigma)
