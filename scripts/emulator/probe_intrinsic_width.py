"""P4 design probe: what is the smallest line width the emulator can carry?

Two independent floors on sigma_v, measured rather than assumed:

  1. **Grid floor.** The native training grid is logarithmic, so one bin
     subtends a fixed velocity c * dlnE. A Gaussian narrower than that is not
     representable on the grid at all, whatever the broadening kernel does.
  2. **Physical floor.** SPEX broadens every line by the ion's thermal Doppler
     width (T_ion = T_e by default, rt = 1), so the cached per-element spectra
     are ALREADY thermally broadened. sigma_v in the forward model is the
     *extra* turbulent term on top. If that is true, a strong isolated line in
     the cache has a measurable width consistent with
     sigma_th = c * sqrt(kT / (A m_p)) and sigma_v = 0 is not "no broadening".

Usage:
    python scripts/emulator/probe_intrinsic_width.py \
        --datadir ~/work/data/spexai/processed --z 26 --line 6.700
"""

import argparse
import os
from typing import Tuple

import numpy as np

C_KMS = 299792.458
# atomic masses (u) for the elements we probe; m_p c^2 in eV via u c^2
U_MEV = 931.49410242e6  # u c^2 in eV
A_MASS = {1: 1.008, 8: 15.999, 14: 28.085, 16: 32.06, 26: 55.845, 28: 58.693}


def thermal_sigma_v(kt_kev: float, z: int) -> float:
    """1D thermal velocity dispersion (km/s) of element ``z`` at kT (keV)."""
    return C_KMS * np.sqrt(kt_kev * 1e3 / (A_MASS[z] * U_MEV))


def grid_floor(energy: np.ndarray) -> Tuple[float, float]:
    """(median, min) velocity subtended by one bin, km/s."""
    dlnE = np.diff(np.log(energy))  # (K-1,) log-spacing of bin centres
    return C_KMS * float(np.median(dlnE)), C_KMS * float(dlnE.min())


def measure_line_sigma(energy: np.ndarray, flux: np.ndarray,
                       e0: float, halfwidth_kms: float = 3000.) -> float:
    """Flux-weighted RMS width (km/s) of the feature nearest ``e0``.

    Continuum is removed as the minimum flux inside the window, which is crude
    but adequate for a strong resonance line sitting on bremsstrahlung.
    """
    # window +/- halfwidth in velocity around e0
    lo, hi = e0 * (1 - halfwidth_kms / C_KMS), e0 * (1 + halfwidth_kms / C_KMS)
    m = (energy >= lo) & (energy <= hi)                 # (K,) bool
    e, f = energy[m], flux[m]
    f = np.clip(f - f.min(), 0., None)
    if f.sum() <= 0:
        return float("nan")
    ec = float((e * f).sum() / f.sum())
    var = float((f * (e - ec) ** 2).sum() / f.sum())
    return C_KMS * np.sqrt(var) / ec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.expanduser(
        "~/work/data/spexai/processed"))
    ap.add_argument("--z", type=int, default=26)
    ap.add_argument("--line", type=float, default=6.700,
                    help="line energy in keV (Fe XXV He-alpha w by default)")
    ap.add_argument("--temps", type=float, nargs="*", default=[2., 5., 10.])
    args = ap.parse_args()

    d = os.path.join(args.datadir, f"element{args.z}")
    energy = np.load(os.path.join(d, "energy.npy")).astype(np.float64)  # (K,)
    temps = np.load(os.path.join(d, "temps.npy")).astype(np.float64)    # (N,)
    logflux = np.load(os.path.join(d, "logflux.npy"))                   # (N, K)

    med, mn = grid_floor(energy)
    print(f"grid: {len(energy)} bins, {energy[0]:.4f}-{energy[-1]:.4f} keV")
    print(f"grid floor (one bin): median {med:.1f} km/s, min {mn:.1f} km/s")
    print(f"element Z={args.z}, A={A_MASS[args.z]}, probe line {args.line} keV")
    print(f"{'kT':>7} {'sigma_th':>9} {'sigma_meas':>11} {'ratio':>7}")
    for kt in args.temps:
        i = int(np.argmin(np.abs(temps - kt)))
        flux = np.exp(logflux[i].astype(np.float64))
        s_th = thermal_sigma_v(float(temps[i]), args.z)
        s_me = measure_line_sigma(energy, flux, args.line)
        print(f"{temps[i]:7.3f} {s_th:9.1f} {s_me:11.1f} {s_me / s_th:7.2f}")


if __name__ == "__main__":
    main()
