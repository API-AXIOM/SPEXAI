"""Validate SpectralOperator.forward_on_grid (multi-instrument evaluation).

forward_on_grid takes only bin EDGES and returns the integrated flux per
bin (the quantity response folding consumes). Checks, using a trained
line-head checkpoint:

  1. native grid: integrated fluxes agree with the standard density
     forward pass times bin width (median per-bin and band-integrated);
  2. flux conservation: the 0.5-10 keV band flux agrees between the
     native grid and an 8x coarser grid (auto quadrature + line deposit);
  3. a completely different grid (linear ~5 eV bins, XRISM/Resolve-like)
     evaluates finite and conserves band flux.

    python scripts/test_forward_on_grid.py [--ckpt PATH] [--cachedir DIR]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_operator import load_model
from spexai.train.operator import edges_from_centers
from spexai.train.train_operator import SpectrumData


def band_flux(int_flux, edges, lo=0.5, hi=10.0):
    """Sum of integrated bin fluxes for bins fully inside [lo, hi] keV."""
    sel = (edges[:-1] >= lo) & (edges[1:] <= hi)
    return int_flux[:, sel].sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/line_head.pt")
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    args = ap.parse_args()

    data = SpectrumData(args.cachedir)
    model, variant = load_model(args.ckpt, data)
    temps = torch.tensor([1.0, 3.0, 8.0])
    energy = data.energy
    edges = edges_from_centers(energy)
    widths = edges[1:] - edges[:-1]

    with torch.no_grad():
        # reference: standard density forward pass -> integrated per bin
        ref_int = torch.pow(10.0, model(temps, energy)) * widths

        # 1. native grid
        on_native = model.forward_on_grid(temps, edges)
        rel = ((on_native - ref_int) / ref_int).abs()
        f_ref, f_nat = band_flux(ref_int, edges), band_flux(on_native, edges)
        band_rel = ((f_nat - f_ref) / f_ref).abs()
        print(f"[1] native grid: median per-bin rel diff = {rel.median():.2e}, "
              f"band flux rel diff = {[f'{r:.2e}' for r in band_rel]}")
        assert rel.median() < 1e-3 and band_rel.max() < 1e-3

        # 2. 8x coarser grid (n_sub chosen automatically)
        coarse_edges = edges[::8].contiguous()
        on_coarse = model.forward_on_grid(temps, coarse_edges)
        rel2 = ((band_flux(on_coarse, coarse_edges) - f_ref) / f_ref).abs()
        print(f"[2] 8x coarser grid: band flux rel diff = "
              f"{[f'{r:.2e}' for r in rel2]}")
        assert rel2.max() < 5e-3

        # 3. linear ~5 eV grid (XRISM/Resolve-like)
        xrism_edges = torch.linspace(0.3, 11.0, 2141)
        on_xrism = model.forward_on_grid(temps, xrism_edges)
        assert torch.isfinite(on_xrism).all() and (on_xrism >= 0).all()
        rel3 = ((band_flux(on_xrism, xrism_edges) - f_ref) / f_ref).abs()
        print(f"[3] linear 5 eV grid: shape {tuple(on_xrism.shape)}, "
              f"band flux rel diff = {[f'{r:.2e}' for r in rel3]}")
        assert rel3.max() < 2e-2

    print(f"all checks passed ({variant}, {os.path.basename(args.ckpt)})")


if __name__ == "__main__":
    main()
