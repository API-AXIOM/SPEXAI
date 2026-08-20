"""Simulation-based calibration (SBC) primitives.

SBC is the one test that can falsify the whole inference chain at once: draw
``theta ~ prior``, simulate data from it, fit, and record the *rank* of the
truth inside the posterior draws. Averaged over simulations the ranks are
uniform **if and only if** the posterior sampler is calibrated for the model
being fitted (Talts et al. 2018).

Two details decide whether a campaign measures anything:

**Ranks need independent draws.** The rank of the truth among ``L`` posterior
samples is uniform on ``{0..L}`` only when those samples are independent. Feed
it a raw MCMC chain and neighbouring draws repeat, which compresses the rank
distribution toward the centre -- a *perfectly* calibrated sampler then fails
the uniformity test. :func:`thin_to_independent` therefore thins by the
integrated autocorrelation time implied by the ESS before
:func:`sbc_rank` counts anything.

**SBC tests the model you fit, not the universe.** Because the truth must be
drawn from the same prior the fit uses, the simulator has to be the fitted
model itself -- here the emulator. That is not a shortcut around the
SPEX-vs-emulator comparison, it is a different question: SBC asks "is the
posterior self-consistent", the point/pull study asks "is the emulator right".
Injecting SPEX truth into an SBC run conflates the two and makes a rank
non-uniformity unattributable.
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def autocorr_thin(n_steps: int, n_walkers: int, ess: float) -> int:
    """Step-axis thinning stride implied by ``ess``.

    arviz counts walkers as separate chains, so ``ess = n_walkers * n_steps /
    tau`` and the integrated autocorrelation time is ``tau`` *in steps*.
    Thinning the step axis by ``ceil(tau)`` leaves draws that are independent
    along the chain while keeping every walker.
    """
    if not np.isfinite(ess) or ess <= 0:
        return int(n_steps)          # no usable diagnostic: keep one row/walker
    tau = n_walkers * n_steps / float(ess)
    return int(max(1, min(n_steps, np.ceil(tau))))


def thin_to_independent(chain: np.ndarray, ess: Optional[np.ndarray],
                        max_draws: Optional[int] = None,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, int]:
    """``(n_steps, n_walkers, ndim)`` chain -> ``(L, ndim)`` near-independent.

    The stride comes from the *worst-mixing* parameter, because a rank is only
    as trustworthy as the slowest direction. Returns the draws and the stride
    used, so a caller can record how much of the chain was actually
    informative.
    """
    if chain.ndim != 3:
        raise ValueError(f"chain must be (steps, walkers, ndim), got "
                         f"{chain.shape}")
    n_steps, n_walkers, _ = chain.shape
    worst = np.nan if ess is None or not len(ess) else float(np.nanmin(ess))
    thin = autocorr_thin(n_steps, n_walkers, worst)
    # take from the END: the tail of a chain is the best-converged part
    kept = chain[n_steps - 1::-thin][::-1]                 # (n_kept, W, ndim)
    draws = kept.reshape(-1, chain.shape[-1])              # (n_kept*W, ndim)
    if max_draws is not None and len(draws) > max_draws:
        rng = np.random.default_rng() if rng is None else rng
        draws = draws[rng.choice(len(draws), max_draws, replace=False)]
    return draws, thin


def sbc_rank(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-parameter SBC rank: how many of ``L`` draws fall below the truth.

    Returns integers in ``{0..L}`` -- *not* the ``mean(s < t)`` fraction. The
    uniformity test is over the discrete rank, and the number of draws must be
    reported with it, since a rank of 37 means nothing without ``L``.
    """
    if draws.ndim != 2:
        raise ValueError(f"draws must be (L, ndim), got {draws.shape}")
    return (draws < np.asarray(truth)[None, :]).sum(axis=0).astype(int)


def rank_uniformity(ranks: np.ndarray, n_draws: int, n_bins: int = 20) -> dict:
    """Chi-square test that a set of SBC ranks is uniform.

    ``ranks`` is ``(n_sims,)`` for one parameter. Bins the ranks and compares
    to the flat expectation. A small p-value means the posterior is
    miscalibrated for that parameter; the sign of the deviation says how:
    ``mean_rank`` above/below 0.5 is bias, and a U or inverted-U shaped
    histogram is an over- or under-dispersed posterior respectively.
    """
    from scipy import stats

    ranks = np.asarray(ranks, dtype=float)
    n_sims = len(ranks)
    if n_sims == 0:
        return {"n_sims": 0, "p_value": float("nan"),
                "mean_rank": float("nan"), "shape": "no data"}
    n_bins = int(min(n_bins, n_draws + 1))
    counts, _ = np.histogram(ranks, bins=n_bins, range=(0, n_draws + 1))
    expected = n_sims / n_bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, n_bins - 1))
    u = ranks / n_draws                       # normalised to [0, 1]
    mean_rank = float(u.mean())
    # variance of U(0,1) is 1/12; more spread => posterior too narrow
    var_ratio = float(u.var() / (1.0 / 12.0))
    if p >= 0.05:
        shape = "uniform"
    elif var_ratio > 1.15:
        shape = "U-shaped (posterior too narrow)"
    elif var_ratio < 0.85:
        shape = "peaked (posterior too wide)"
    elif mean_rank > 0.5:
        shape = "shifted high (posterior biased low)"
    else:
        shape = "shifted low (posterior biased high)"
    return {"n_sims": n_sims, "chi2": chi2, "p_value": p,
            "mean_rank": mean_rank, "var_ratio": var_ratio, "shape": shape,
            "counts": counts.tolist()}


def summarise(ranks_by_param: dict, n_draws: int,
              names: Optional[Sequence[str]] = None) -> str:
    """Human-readable calibration table over every parameter."""
    names = list(ranks_by_param) if names is None else list(names)
    lines = [f"SBC over {n_draws} independent draws/sim",
             f"{'param':>12} {'n_sims':>7} {'mean rank':>10} {'p':>8}  verdict"]
    for n in names:
        r = rank_uniformity(np.asarray(ranks_by_param[n]), n_draws)
        lines.append(f"{n:>12} {r['n_sims']:>7d} {r['mean_rank']:>10.3f} "
                     f"{r['p_value']:>8.3f}  {r['shape']}")
    return "\n".join(lines)
