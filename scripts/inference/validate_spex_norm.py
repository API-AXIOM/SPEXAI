"""Validate the emulator's ABSOLUTE flux normalisation against SPEX itself.

Three steps, in two conda envs (SPEX and spexai cannot share one interpreter).
``conda run`` strips ``DYLD_*``, so the SPEX step must activate the env and run
its python directly:

1. In the spexai env, write the emulator's native energy grid to a file:

     KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python \
         scripts/inference/validate_spex_norm.py --mode edges --out /tmp/egrid.npz

2. In the SPEX env, dump CIE model spectra on that grid at (T, Y=1, D=1e22 m):

     source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate spex
     export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib"   # find conda's libXext
     python scripts/inference/validate_spex_norm.py --mode spex \
         --edges /tmp/egrid.npz --temps 1,2,4,8,16 --out /tmp/spex_cie.npz

3. In the spexai env, compare the emulator flux to that dump:

     KMP_DUPLICATE_LIB_OK=TRUE conda run -n spexai python \
         scripts/inference/validate_spex_norm.py --mode compare --in /tmp/spex_cie.npz

What is being tested: that ``sum_Z JointOperatorModel`` at solar abundance
reproduces SPEX's own CIE spectrum bin for bin, in absolute units -- i.e. that
the training flux is SPEX's native SI photon flux (ph s^-1 m^-2 bin^-1), that
norm=1 corresponds to Y=1e64 m^-3 and D_ref=1e22 m, and hence that the physical
parametrisation in ``spexai.inference.units`` is correct.

Two SPEX settings are NOT recorded anywhere for the training spectra (the
generator script predates this repo): the SPEXACT version and the abundance
table. Both are recoverable empirically -- ``--spexact`` and ``--abun`` sweep
them, and the setting that matches is the one the training data was made with.

History: measured 2026-07-28 at T=4 keV with only 16/30 elements trained, on a
coarse 2001-bin log grid, giving a median ~2.5% over 3-8 keV rising to ~5% at
8 keV -- consistent with the missing elements' continuum. Redone under P2 with
all 30 elements on the native grid.
"""
import argparse
import os

import numpy as np

DEFAULT_TEMPS = (1.0, 2.0, 4.0, 8.0, 16.0)
BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0),
         (5.0, 8.0), (8.0, 10.0), (10.0, 12.0))


# ----------------------------------------------------------------- mode: edges
def dump_edges(out):
    """Write the emulator's native training grid (spexai env, needs torch)."""
    import torch
    from spexai.config import STORE
    from spexai.inference.operator_model import load_operator
    import json
    with open(os.path.join(STORE, "manifest.json")) as f:
        first = sorted(json.load(f)["elements"].items(), key=lambda kv: int(kv[0]))
    model = load_operator(os.path.join(STORE, first[0][1]["file"]))
    edges = model.train_edges.detach().cpu().numpy().astype(np.float64)
    cen = model.train_energy.detach().cpu().numpy().astype(np.float64)
    # every element shares one native grid; assert it rather than assume it
    for _, entry in first[1:]:
        e = load_operator(os.path.join(STORE, entry["file"]))
        assert np.allclose(e.train_edges.detach().cpu().numpy(), edges), \
            f"{entry['file']} has a different native grid"
    np.savez(out, edges=edges, centers=cen, store=STORE)
    print(f"wrote {out}: {len(edges) - 1} bins, "
          f"{edges[0]:.6f}-{edges[-1]:.6f} keV, {len(first)} elements checked")


# ------------------------------------------------------------------ mode: spex
def dump_spex(out, edges_file, temps, spexact, abun, timing_only=False,
              ibal=None):
    """Dump SPEX CIE spectra on the emulator's native grid (SPEX env)."""
    import time
    from pyspex.spex import Session

    edges = np.load(edges_file)["edges"].astype(np.float64)
    s = Session()
    if spexact is not None:
        s.var_calc(int(spexact))             # 0=SPEXACT2, 1=quick 3, 2=SPEXACT3
    if abun:
        s.abun(abun)
    if ibal:
        s.ibal(ibal)
    s.egrid_set(edges)                       # (nbins+1,) bin boundaries in keV
    s.com("cie")
    s.dist(1, 1e22, "m")                     # SPEX reference distance
    s.par(1, 1, "norm", 1.0)                 # Y = 1 (1e64 m^-3)

    flux = np.empty((len(temps), len(edges) - 1), dtype=np.float64)
    for i, t in enumerate(temps):
        s.par(1, 1, "t", float(t))
        t0 = time.time()
        s.calc()
        dt = time.time() - t0
        spec = s.mod_spectrum
        spec.get(1)
        F = np.asarray(spec.spectrum.value)  # ph / (bin s m^2)
        E = np.asarray(spec.energy.to_value("keV"))
        assert len(F) == len(edges) - 1, \
            f"SPEX returned {len(F)} bins, grid has {len(edges) - 1}"
        flux[i] = F
        print(f"  T={t:6.3f} keV  {dt:6.1f}s  sum={F.sum():.6e} ph/s/m2",
              flush=True)
        if timing_only:
            return

    ref = s.abun_show()
    np.savez(out, edges=edges, centers=E, temps=np.asarray(temps, float),
             flux=flux, unit=str(spec.spectrum.unit),
             spexact=("default" if spexact is None else str(spexact)),
             abun=str(abun or "default"), abun_ref=str(ref),
             ibal=str(ibal or "default"), ibal_ref=str(s.ibal_show()))
    print(f"wrote {out}: {len(temps)} spectra x {flux.shape[1]} bins, "
          f"unit={spec.spectrum.unit}, spexact={spexact}, abun={abun}")


def nearest_cache_temp(datadir, z, temp):
    """The training-grid temperature of element ``z`` nearest ``temp``."""
    t = np.load(os.path.join(datadir, f"element{z}", "temps.npy"))
    return float(t[int(np.argmin(np.abs(t - temp)))])


def dump_spex_elements(out, edges_file, temps, elements, spexact, abun,
                       keep_h=False, ibal=None, match_cache=None, gacc=None):
    """Dump SINGLE-ELEMENT SPEX CIE spectra: one element at solar, rest zero.

    This is the shape the per-element training spectra must have had, since the
    emulator sums elements linearly. Dumping them lets the comparison separate
    two error sources that the full-CIE check confounds: **per-element emulator
    error** (element Z's operator vs SPEX's element Z) and **composition
    error** (sum_Z of SPEX's own single-element runs vs SPEX's full CIE, which
    is not exactly 1 because the electron density depends on composition).

    ``keep_h`` leaves hydrogen at solar in every run instead of zeroing it.
    **Zeroing hydrogen is a fatal SPEX error** (no electrons), so ``keep_h`` is
    in practice mandatory, and element Z's own emission has to be recovered by
    subtracting an H-only run -- which is what ``compare_elements`` does, and
    what reproduces the training spectra to 1e-4 (verified 2026-09-01 for H, O).
    """
    import time
    from pyspex.spex import Session

    edges = np.load(edges_file)["edges"].astype(np.float64)
    s = Session()
    if spexact is not None:
        s.var_calc(int(spexact))
    if abun:
        s.abun(abun)
    if ibal:
        s.ibal(ibal)
    if gacc is not None:
        s.var_gacc(float(gacc))   # free-bound accuracy; see check_training_provenance
    s.egrid_set(edges)
    s.com("cie")
    s.dist(1, 1e22, "m")
    s.par(1, 1, "norm", 1.0)

    flux = np.zeros((len(temps), len(elements), len(edges) - 1), dtype=np.float64)
    # actual temperature used per (temp, element): with --match-cache each
    # element is run at ITS OWN nearest training-grid temperature, because the
    # three generation batches do not share one temperature grid. Skipping this
    # contaminates a low-T comparison: at kT=0.7 keV a 2.6e-4 offset in T
    # already moves the 1.9-12 keV bremsstrahlung by 0.2%.
    t_used = np.zeros((len(temps), len(elements)), dtype=np.float64)
    # H's continuum must be subtracted at the SAME temperature as the element
    # it is subtracted from, so the H-only run is cached per temperature rather
    # than done once (it is 10-100x the element's own in-band flux, so even a
    # 1e-4 temperature mismatch would swamp the difference).
    flux_h = np.zeros_like(flux)
    t0 = time.time()

    def run(tz, on_z):
        for zz in range(1, 31):              # one element on, the rest off
            on = (zz == on_z) or (keep_h and zz == 1)
            s.par(1, 1, f"{zz:02d}", 1.0 if on else 0.0)
        s.par(1, 1, "t", float(tz))
        s.calc()
        s.mod_spectrum.get(1)
        return np.asarray(s.mod_spectrum.spectrum.value)

    for i, t in enumerate(temps):
        hcache = {}
        for j, z in enumerate(elements):
            tz = (nearest_cache_temp(match_cache, z, t) if match_cache
                  else float(t))
            t_used[i, j] = tz
            if tz not in hcache:
                hcache[tz] = run(tz, 1)
            flux_h[i, j] = hcache[tz]
            flux[i, j] = flux_h[i, j] if z == 1 else run(tz, z)
        print(f"  T={t:6.3f} keV  {len(elements)} elements  "
              f"{time.time() - t0:6.1f}s cumulative", flush=True)
    spec = s.mod_spectrum

    E = np.asarray(spec.energy.to_value("keV"))
    ref = s.abun_show()
    np.savez(out, edges=edges, centers=E, temps=np.asarray(temps, float),
             elements=np.asarray(elements, int), flux=flux, flux_h=flux_h,
             temps_used=t_used, keep_h=bool(keep_h),
             match_cache=str(match_cache or ""), unit=str(spec.spectrum.unit),
             gacc=("default" if gacc is None else float(gacc)),
             spexact=("default" if spexact is None else str(spexact)),
             abun=str(abun or "default"), abun_ref=str(ref),
             ibal=str(ibal or "default"), ibal_ref=str(s.ibal_show()))
    print(f"wrote {out}: {flux.shape} (temps, elements, bins), keep_h={keep_h}")


# --------------------------------------------------------------- mode: compare
def band_stats(femu, fspx, cen):
    """Per-band ratio summaries.

    Two different questions, so two different statistics. The **integrated**
    ratio (sum/sum) is the science-relevant one: it weights each bin by its
    flux, which is what a fit sees. The **median bin** ratio and its 16/84
    spread describe the per-bin shape error, in which a bin holding 1e-30 of
    the flux counts as much as a line peak -- informative about where the
    emulator struggles, misleading as an error budget.
    """
    ok = fspx > 0
    ratio = np.where(ok, femu / np.where(ok, fspx, 1.0), np.nan)
    rows = []
    for lo, hi in BANDS:
        m = (cen >= lo) & (cen < hi) & ok
        if m.sum():
            r = ratio[m]
            rows.append((lo, hi, int(m.sum()), femu[m].sum() / fspx[m].sum(),
                         np.median(r), np.percentile(r, 16),
                         np.percentile(r, 84)))
    return rows, ratio


def compare(inp, device="cpu", elements=None, save=None):
    import torch
    from spexai.inference.operator_model import JointOperatorModel
    d = np.load(inp)
    edges, cen = d["edges"], d["centers"]
    temps, Fspx = d["temps"], d["flux"]
    print(f"SPEX dump: spexact={d['spexact']} abun={d['abun']} "
          f"({d['abun_ref']}) unit={d['unit']}")

    model = JointOperatorModel(device=device, elements=elements)
    print(f"emulator: {len(model.elements)} elements {model.elements}")
    et = torch.as_tensor(edges, dtype=torch.float32)
    # (T, nbins) integrated flux per native bin, solar abundance, no broadening
    Femu = model.flux(torch.as_tensor(temps, dtype=torch.float32), {}, 0.0,
                      et).cpu().numpy().astype(np.float64)

    for i, T in enumerate(temps):
        rows, ratio = band_stats(Femu[i], Fspx[i], cen)
        ok = Fspx[i] > 0
        band = (cen > 3) & (cen < 8) & ok
        tot = (cen >= 0.5) & (cen <= 12) & ok
        print(f"\nT = {T:g} keV   bins with SPEX flux: {ok.sum()}/{len(ok)}")
        print(f"  integrated flux ratio (0.5-12 keV): "
              f"{Femu[i][tot].sum() / Fspx[i][tot].sum():.5f}")
        print(f"  integrated flux ratio (3-8 keV):    "
              f"{Femu[i][band].sum() / Fspx[i][band].sum():.5f}")
        print(f"  median bin ratio      (3-8 keV):    "
              f"{np.median(ratio[band]):.5f}")
        print(f"  {'band[keV]':>12} {'nbins':>7} {'integ':>8} {'median':>8} "
              f"{'p16':>8} {'p84':>8}")
        for lo, hi, n, integ, med, p16, p84 in rows:
            print(f"  {lo:5.1f}-{hi:5.1f} {n:7d} {integ:8.4f} {med:8.4f} "
                  f"{p16:8.4f} {p84:8.4f}")

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez(save, edges=edges, centers=cen, temps=temps, flux_spex=Fspx,
                 flux_emu=Femu, elements=np.asarray(model.elements),
                 spexact=d["spexact"], abun=d["abun"], abun_ref=d["abun_ref"],
                 unit=d["unit"])
        print(f"\nwrote {save}")


def compare_elements(inp, full=None, device="cpu", save=None, band=(3.0, 8.0)):
    """Per-element emulator-vs-SPEX, plus the composition (linearity) check.

    ``inp`` is a ``--mode spex --elements ...`` dump. If ``full`` (a full-CIE
    dump on the same grid and settings) is given, also reports
    ``sum_Z SPEX_Z / SPEX_full``, which involves no emulator at all and
    isolates how much of the full-CIE mismatch is the linear-sum assumption.
    """
    import torch
    from spexai.inference.abundances import SYMBOL
    from spexai.inference.operator_model import JointOperatorModel
    d = np.load(inp)
    edges, cen, temps = d["edges"], d["centers"], d["temps"]
    elements, Fspx = [int(z) for z in d["elements"]], d["flux"]   # (T, Z, bins)
    print(f"SPEX dump: spexact={d['spexact']} abun={d['abun']} "
          f"keep_h={d['keep_h']} elements={elements}")
    if bool(d["keep_h"]):
        # every run carries H's continuum (SPEX cannot run without electrons),
        # so element Z alone = CIE(H+Z) - CIE(H), using the H run at that
        # element's own temperature. H's own row is left untouched.
        Fspx = np.where(np.asarray(elements)[None, :, None] == 1,
                        Fspx, Fspx - d["flux_h"])

    model = JointOperatorModel(device=device, elements=elements)
    et = torch.as_tensor(edges, dtype=torch.float32)
    # (T, Z, nbins): each element alone at solar abundance, no broadening, at
    # the temperature its SPEX counterpart was actually run at
    tu = d["temps_used"]
    Femu = np.stack([_element_flux(model, z,
                                   torch.as_tensor(tu[:, j], dtype=torch.float32),
                                   et)
                     for j, z in enumerate(elements)], axis=1)

    m = (cen >= band[0]) & (cen < band[1])
    hdr = "  ".join(f"{t:g}keV" for t in temps)
    print(f"\nper-element integrated flux ratio emu/SPEX, "
          f"{band[0]}-{band[1]} keV\n{'Z':>3} {'el':>3}  {hdr}")
    for j, z in enumerate(elements):
        cells = []
        for i in range(len(temps)):
            den = Fspx[i, j][m].sum()
            cells.append("      -" if den <= 0
                         else f"{Femu[i, j][m].sum() / den:7.4f}")
        print(f"{z:3d} {SYMBOL.get(z, '?'):>3}  " + " ".join(cells))

    if full is not None:
        f = np.load(full)
        assert np.allclose(f["temps"], temps), "full-CIE dump has other temps"
        print(f"\ncomposition check (SPEX only): sum_Z SPEX_Z / SPEX_full, "
              f"{band[0]}-{band[1]} keV")
        for i, t in enumerate(temps):
            print(f"  T={t:6.3f} keV  "
                  f"{Fspx[i].sum(axis=0)[m].sum() / f['flux'][i][m].sum():.5f}")

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez(save, edges=edges, centers=cen, temps=temps,
                 elements=np.asarray(elements), flux_spex=Fspx, flux_emu=Femu,
                 keep_h=d["keep_h"], spexact=d["spexact"], abun=d["abun"])
        print(f"\nwrote {save}")


def _element_flux(model, z, temps, edges):
    """(T, nbins) integrated flux for one element alone, via its operator."""
    from spexai.inference.operator_model import element_broadened_flux
    return element_broadened_flux(model.models[z], temps, 0.0, edges) \
        .cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["edges", "spex", "compare", "compare-elements"])
    ap.add_argument("--out", default="spex_cie.npz")
    ap.add_argument("--in", dest="inp", default="spex_cie.npz")
    ap.add_argument("--edges", default="egrid.npz")
    ap.add_argument("--temps", default=",".join(str(t) for t in DEFAULT_TEMPS))
    ap.add_argument("--spexact", type=int, choices=[0, 1, 2], default=None,
                    help="SPEX var calc: 0=SPEXACT2, 1=quick SPEXACT3, 2=SPEXACT3")
    ap.add_argument("--abun", default=None,
                    help="SPEX abundance table (reset/ag/allen/asplund/ra/"
                         "grevesse/gs/lodders/solar); default = SPEX's own")
    ap.add_argument("--timing", action="store_true",
                    help="spex mode: calc the first temperature only, no output")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None,
                    help="compare mode: npz to write both spectra to")
    ap.add_argument("--elements", default=None,
                    help="spex mode: comma-separated Z for SINGLE-element "
                         "dumps (one element on, rest off), or 'all'")
    ap.add_argument("--ibal", default=None,
                    help="SPEX ionisation balance (ar85/ar92/bryans/"
                         "oldbryans/urdampilleta); default = SPEX's own")
    ap.add_argument("--gacc", type=float, default=None,
                    help="SPEX free-bound accuracy (var gacc). The training "
                         "spectra were made with ~0.01; SPEX defaults near 1e-3")
    ap.add_argument("--match-cache", default=None,
                    help="single-element dumps: processed-cache dir; run each "
                         "element at ITS OWN nearest training temperature")
    ap.add_argument("--keep-h", action="store_true",
                    help="single-element dumps: leave H at solar in every run")
    ap.add_argument("--full", default=None,
                    help="compare-elements mode: full-CIE dump for the "
                         "composition check")
    args = ap.parse_args()

    temps = [float(t) for t in args.temps.split(",")]
    if args.mode == "edges":
        dump_edges(args.out)
    elif args.mode == "spex" and args.elements:
        zs = (list(range(1, 31)) if args.elements == "all"
              else [int(z) for z in args.elements.split(",")])
        dump_spex_elements(args.out, args.edges, temps, zs, args.spexact,
                           args.abun, keep_h=args.keep_h, ibal=args.ibal,
                           match_cache=args.match_cache, gacc=args.gacc)
    elif args.mode == "spex":
        dump_spex(args.out, args.edges, temps, args.spexact, args.abun,
                  timing_only=args.timing, ibal=args.ibal)
    elif args.mode == "compare":
        compare(args.inp, device=args.device, save=args.save)
    else:
        compare_elements(args.inp, full=args.full, device=args.device,
                         save=args.save)


if __name__ == "__main__":
    main()
