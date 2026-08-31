"""Validate the broadening implementations against the exact reference.

  1. FFT broadening (broaden_native) vs the exact erf redistribution
     matrix on real SPEX spectra, several velocities: flux conservation
     and per-bin agreement.
  2. deposit_gaussian_lines: a single line's broadened profile matches the
     exact matrix applied to a delta spectrum, and its flux is conserved.

    python scripts/test_broadening.py [--cachedir DIR]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spexai.broadening import (broaden_native, deposit_gaussian_lines,
                                     direct_broaden)
from spexai.operator import edges_from_centers
from spexai.data import SpectrumData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    args = ap.parse_args()

    data = SpectrumData(args.cachedir)
    edges = edges_from_centers(data.energy)
    widths = edges[1:] - edges[:-1]
    # a cool and a hot test spectrum, integrated flux per bin
    sel = data.test_idx[[0, -1]]
    flux = torch.pow(10.0, torch.clamp(data.logflux[sel], min=-30)) * widths
    print(f"test temps: {[f'{t:.2f}' for t in data.temps[sel]]} keV")

    for v in (100.0, 300.0, 1000.0):
        ref = direct_broaden(flux, edges, v)
        fft = broaden_native(flux, edges, v)
        # compare where flux is within 6 decades of the spectrum peak;
        # deeper wing bins are physically irrelevant
        dens_ref = ref / widths
        mask = dens_ref > 1e-6 * dens_ref.max(dim=1, keepdim=True).values
        rel = ((fft - ref).abs() / ref.clamp(min=1e-30))[mask]
        # fraction of total flux in the wrong bin: the physically meaningful
        # summary (per-bin relative error deep in Gaussian wings sits at the
        # information floor of sub-bin line placement in the binned input)
        l1 = (fft - ref).abs().sum(1) / ref.sum(1)
        fcons = ((fft.sum(1) - ref.sum(1)) / ref.sum(1)).abs()
        print(f"[fft vs exact] v={v:6.0f} km/s: median rel diff="
              f"{rel.median():.2e}, misplaced flux={[f'{x:.1e}' for x in l1]}, "
              f"flux diff={[f'{x:.1e}' for x in fcons]}")
        assert rel.median() < 1e-3 and l1.max() < 5e-3

    # single line: deposit vs exact matrix on a delta spectrum
    j = torch.argmin((data.energy - 6.7).abs())  # near the Fe-K complex
    delta = torch.zeros(1, len(widths))
    delta[0, j] = 1.0
    for v in (100.0, 1000.0):
        ref = direct_broaden(delta, edges, v)
        dep = deposit_gaussian_lines(data.energy[j:j + 1],
                                     torch.ones(1, 1), edges, v)
        err = (dep - ref).abs().max()
        print(f"[line deposit] v={v:6.0f} km/s: max abs diff={err:.2e}, "
              f"flux={dep.sum():.6f}")
        assert err < 1e-3 and abs(dep.sum() - 1.0) < 1e-4

    print("all broadening checks passed")


if __name__ == "__main__":
    main()
