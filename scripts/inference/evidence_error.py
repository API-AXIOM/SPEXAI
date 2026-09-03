"""Recover the marginal-likelihood uncertainty that nautilus and i-nessai
did not report for the bake-off runs.

Both returned ``logzerr = nan``, for two unrelated reasons:

* **nautilus** (v1.0.6) exposes no evidence-error API at all -- ``Sampler.log_z``
  is a bare point estimate and there is no companion uncertainty. The ``nan`` we
  stored is therefore a faithful record, not a bug on our side. We estimate one
  here by bootstrapping the per-shell mean likelihood.

* **i-nessai** (nessai 0.15.2) *does* implement one, and it overflows.
  ``nessai.evidence._INSIntegralState.compute_uncertainty`` evaluates
  ``np.exp(logZ, dtype=np.longdouble)`` in *linear* evidence space. Our
  ``logZ`` is ${\\sim}3.4\\times10^{6}$ -- an unremarkable value for a Poisson
  likelihood over ${\\sim}10^{6}$ counts -- and ``exp`` of that overflows to
  ``inf`` on any float type (the limit is ~709 for float64, ~11356 for x86
  float80), so the standard error comes out ``nan``. The fix is to do the same
  algebra in log space, which is what ``inessai_logz_error`` below does; it is
  the identical estimator, not a different one.

Run: ``conda run -n spexai python scripts/inference/evidence_error.py``
"""
from typing import Tuple
import argparse
import os

import h5py
import numpy as np
from scipy.special import logsumexp

BAKEOFF = os.path.expanduser("~/work/data/spexai/bakeoff")
SEED = 0


def nautilus_logz_error(path: str, n_boot: int = 2000,
                        seed: int = SEED) -> Tuple[float, float, float]:
    """Bootstrap log Z for a nautilus run from its HDF5 checkpoint.

    nautilus builds the evidence as a sum over shells,
    ``log Z = logsumexp_i(log <L>_i + log V_i)``, where ``log <L>_i`` is the log
    mean likelihood of shell ``i`` estimated from that shell's sampled points
    and ``log V_i`` is its log volume. The dominant sampling error is in the
    per-shell mean, so we resample each shell's likelihoods with replacement
    and recompute.

    This deliberately holds ``log V_i`` fixed, so the result is a *lower bound*
    on the true uncertainty: it propagates the Monte Carlo error in the
    likelihood means but not the error in nautilus's volume estimates.

    Returns ``(log_z, sigma_boot, log_z_recomputed)``.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        g = f["sampler"]
        log_v = np.asarray(g.attrs["shell_log_v"], dtype=float)   # (n_shell,)
        log_l_ref = np.asarray(g.attrs["shell_log_l"], dtype=float)
        # the mean is over the points KEPT in the shell (shell_n), not over
        # the points drawn to fill it (shell_n_sample) -- verified below, the
        # two differ by ~0.65 nats and only shell_n reproduces the stored value
        n_keep = np.asarray(g.attrs["shell_n"], dtype=float)
        shells = [np.asarray(g[f"log_l_{i}"][:], dtype=float)
                  for i in range(len(log_v))]

    # Reproduce the stored per-shell log mean likelihood as a correctness
    # check on the reconstruction before bootstrapping it.
    recomputed = np.array([logsumexp(s[np.isfinite(s)]) - np.log(n)
                           for s, n in zip(shells, n_keep)])     # (n_shell,)
    max_dev = float(np.max(np.abs(recomputed - log_l_ref)))
    print(f"  shell log<L> reproduced to {max_dev:.3e} nats "
          f"({'OK' if max_dev < 1e-6 else 'MISMATCH -- do not trust the bootstrap'})")

    log_z = float(logsumexp(log_l_ref + log_v))

    boot = np.empty(n_boot)                                       # (n_boot,)
    finite = [s[np.isfinite(s)] for s in shells]
    for b in range(n_boot):
        ll = np.array([logsumexp(rng.choice(s, size=s.size, replace=True))
                       - np.log(n) for s, n in zip(finite, n_keep)])
        boot[b] = logsumexp(ll + log_v)
    return log_z, float(np.std(boot, ddof=1)), float(logsumexp(recomputed + log_v))


def inessai_logz_error(path: str) -> Tuple[float, float, float]:
    """nessai's own evidence standard error, evaluated in log space.

    nessai computes, in linear space,
    ``sigma[Z] = sqrt(sum_i (Z_i - Zhat)^2 / (n(n-1)))`` and
    ``sigma[log Z] = sigma[Z] / Zhat``, with ``Zhat = exp(logZ)`` and
    ``Z_i = exp(w_i)`` for per-sample log contributions ``w_i = logL_i + logW_i``
    and ``logZ = logsumexp(w) - log n``.

    Expanding the square, ``sum_i (Z_i - Zhat)^2 = sum_i Z_i^2 - n Zhat^2``
    (using ``sum_i Z_i = n Zhat``), so

        sigma[log Z] = sqrt((exp(logsumexp(2w) - 2 logZ) - n) / (n(n-1))).

    Every exponential here is of an O(log n) quantity, so nothing overflows.

    Returns ``(log_z, sigma_log_z, kish_ess)``.
    """
    with h5py.File(path, "r") as f:
        s = f["samples"]
        w = np.asarray(s["logL"][:], dtype=float) + \
            np.asarray(s["logW"][:], dtype=float)                 # (n,)

    w = w[np.isfinite(w)]
    n = w.size
    log_z = float(logsumexp(w) - np.log(n))
    # exp of an O(log n) argument -- the whole point of the rearrangement
    ratio = float(np.exp(logsumexp(2.0 * w) - 2.0 * log_z))
    sigma = float(np.sqrt(max(ratio - n, 0.0) / (n * (n - 1))))
    kish = float(np.exp(2.0 * logsumexp(w) - logsumexp(2.0 * w)))
    return log_z, sigma, kish


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=BAKEOFF)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    print("nautilus (no evidence-error API in v1.0.6; bootstrapped here)")
    lz, sig, lz_re = nautilus_logz_error(
        os.path.join(args.dir, "nautilus.hdf5"), n_boot=args.n_boot)
    print(f"  log Z = {lz:.3f} +- {sig:.3f}  (recomputed {lz_re:.3f})")

    print("\ni-nessai (nessai's own estimator, redone in log space)")
    lz, sig, kish = inessai_logz_error(
        os.path.join(args.dir, "inessai", "result.hdf5"))
    print(f"  log Z = {lz:.3f} +- {sig:.3f}   (Kish ESS of the evidence "
          f"weights: {kish:.0f})")

    print("\nreference: pocoMC 3381277.97 +- 0.035, "
          "UltraNest 3381277.86 +- 0.445")


if __name__ == "__main__":
    main()
