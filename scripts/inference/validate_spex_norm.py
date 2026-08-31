"""Validate the emulator's ABSOLUTE flux normalisation against SPEX itself.

Two steps, in two conda envs (SPEX and spexai cannot share one interpreter):

1. In a SPEX env, dump a CIE model spectrum at (T, Y=1, D=1e22 m):

     source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate spex
     export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib"   # find conda's libXext
     python scripts/validate_spex_norm.py --mode spex --out /tmp/spex_cie.npz --temp 4.0

2. In the spexai env, compare the emulator flux density to that dump:

     conda run -n spexai python scripts/validate_spex_norm.py \
         --mode compare --in /tmp/spex_cie.npz

Result (2026-07-28, T=4 keV, 16/30 elements): the emulator matches SPEX to a
median ~2.5% across the 3-8 keV continuum, confirming that the training flux is
SPEX's native SI photon flux (ph s^-1 m^-2 bin^-1), that norm=1 corresponds to
Y=1e64 m^-3, and D_ref=1e22 m -- i.e. the physical parametrisation in
spexai.inference.units (Y * (D_ref/D)^2 * 1e-4[m^2->cm^2]) is correct. The
small high-energy deficit is the expected effect of the missing elements 16-30.
"""
import argparse

import numpy as np


def dump_spex(out, temp, elow=0.1, ehigh=12.0, nbins=4000):
    from pyspex.spex import Session
    s = Session()
    s.egrid(elow, ehigh, nbins, "kev", True)
    s.com("cie")
    s.dist(1, 1e22, "m")                 # SPEX reference distance
    s.par(1, 1, "t", float(temp))
    s.par(1, 1, "norm", 1.0)             # Y = 1 (1e64 m^-3)
    s.calc()
    spec = s.mod_spectrum
    spec.get(1)
    E = np.asarray(spec.energy.to_value("keV"))
    W = np.asarray(spec.energy_width.to_value("keV"))
    F = np.asarray(spec.spectrum.value)  # ph / (bin s m^2)
    np.savez(out, energy=E, width=W, flux=F, unit=str(spec.spectrum.unit),
             temp=float(temp))
    print(f"wrote {out}: {len(E)} bins, unit={spec.spectrum.unit}, "
          f"sum={F.sum():.4e} ph/s/m2")


def compare(inp):
    import torch
    from spexai.inference.operator_model import JointOperatorModel
    d = np.load(inp)
    Espx, Wspx, Fspx, T = d["energy"], d["width"], d["flux"], float(d["temp"])
    dens_spx = Fspx / Wspx                                   # ph/s/m2/keV

    model = JointOperatorModel(device="cpu")                # all present elements
    edges = torch.logspace(np.log10(0.15), np.log10(11.5), 2001)
    cen = np.sqrt(edges[:-1].numpy() * edges[1:].numpy())
    wid = (edges[1:] - edges[:-1]).numpy()
    dens_emu = model.flux(torch.tensor([T]), {}, 0.0, edges).squeeze(0).numpy() / wid
    dens_spx_i = np.exp(np.interp(np.log(cen), np.log(Espx),
                                  np.log(np.clip(dens_spx, 1e-300, None))))
    print(f"emulator elements: {model.elements}")
    print(f"{'E[keV]':>7} {'emu':>12} {'SPEX':>12} {'ratio':>8}")
    for Ec in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        j = int(np.argmin(np.abs(cen - Ec)))
        print(f"{cen[j]:7.2f} {dens_emu[j]:12.4e} {dens_spx_i[j]:12.4e} "
              f"{dens_emu[j] / max(dens_spx_i[j], 1e-300):8.3f}")
    band = (cen > 3) & (cen < 8)
    print("median continuum ratio (3-8 keV): %.3f"
          % np.median(dens_emu[band] / np.clip(dens_spx_i[band], 1e-300, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spex", "compare"], required=True)
    ap.add_argument("--out", default="spex_cie.npz")
    ap.add_argument("--in", dest="inp", default="spex_cie.npz")
    ap.add_argument("--temp", type=float, default=4.0)
    args = ap.parse_args()
    if args.mode == "spex":
        dump_spex(args.out, args.temp)
    else:
        compare(args.inp)


if __name__ == "__main__":
    main()
