"""Tabulate tbabs sigma(E) from XSPEC into a cached .npz for spexai.

conda-forge ships no XSPEC/HEASoft for macOS, so run this in a HEASoft conda
environment (PyXspec ``import xspec``; shersa's XSPEC models also work if present)
-- e.g. the ``heasoft-test`` env or the analysis cluster:

    conda run -n heasoft-test python scripts/build_tbabs_table.py

It writes spexai/inference/data/tbabs_sigma.npz with {energy, sigma}, where
sigma is the per-H cross-section (cm^2) obtained from the tbabs transmission at a
reference column. Transmission is measured cleanly as the ratio of an absorbed to
an unabsorbed flat (PhoIndex=0) power law, which avoids ambiguity in how XSPEC
evaluates a lone multiplicative model. Once the file exists,
``spexai.inference.absorption.Absorption.tbabs()`` loads it and no XSPEC is
needed at fit time; until then the code uses the wabs fallback.
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "spexai", "inference", "data", "tbabs_sigma.npz")
N_REF = 1e21          # reference column (cm^-2) used to invert transmission -> sigma
E_MIN, E_MAX, N_BINS = 0.05, 15.0, 8191   # log-spaced XSPEC evaluation grid


def tbabs_transmission_pyxspec(nh_1e22):
    """(edges, transmission) via PyXspec, as absorbed/unabsorbed flat powerlaw.

    The edges are reconstructed from the grid we set (this PyXspec build has no
    ``AllModels.energies`` getter). ``abund wilm`` is the abundance table tbabs
    is defined with (Wilms, Allen & McCray 2000)."""
    import xspec
    xspec.Xset.chatter = 0
    xspec.Xset.allowPrompting = False
    xspec.Xset.abund = "wilm"
    xspec.AllModels.setEnergies(f"{E_MIN} {E_MAX} {N_BINS} log")
    m = xspec.Model("tbabs*powerlaw")
    m.powerlaw.PhoIndex = 0.0
    m.powerlaw.norm = 1.0
    m.TBabs.nH = 0.0
    unabs = np.asarray(m.values(0), dtype=np.float64)
    m.TBabs.nH = nh_1e22
    absb = np.asarray(m.values(0), dtype=np.float64)
    edges = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_BINS + 1)
    trans = absb / np.clip(unabs, 1e-300, None)
    return edges, trans


def tbabs_transmission_sherpa(nh_1e22):
    """(edges, transmission) via sherpa's XSPEC models."""
    from sherpa.astro import xspec
    e = np.logspace(np.log10(0.05), np.log10(15.0), 8192)
    elo, ehi = e[:-1], e[1:]
    m = xspec.XStbabs()
    m.nH = nh_1e22
    trans = np.asarray(m.calc([m.nH.val], elo, ehi), dtype=np.float64)
    return e, trans


def main():
    try:
        edges, trans = tbabs_transmission_pyxspec(N_REF / 1e22)
        src = "PyXspec"
    except Exception:
        edges, trans = tbabs_transmission_sherpa(N_REF / 1e22)
        src = "sherpa"
    ecen = np.sqrt(edges[:-1] * edges[1:])
    sigma = -np.log(np.clip(trans, 1e-300, 1.0)) / N_REF    # cm^2 per H
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, energy=ecen.astype(np.float64), sigma=sigma.astype(np.float64))
    print(f"[{src}] wrote {OUT} with {len(ecen)} points; "
          f"sigma(1keV) = {np.interp(1.0, ecen, sigma):.3e} cm^2/H")


if __name__ == "__main__":
    main()
