"""Emulator accuracy on instrument energy grids.

Two evaluations per instrument, per emulator. Truth for the unbroadened
emulator is the raw SPEX spectrum; truth for the (T, v) emulators is the
exact erf-integral broadening of the SPEX spectrum at each test velocity.

(1) Resolution-matched binning: predictions and truth are rebinned onto
    grids with bin width = FWHM / --oversample (no redistribution), and
    scored per bin.

(2) RMF folding: predictions and truth are rebinned onto the RMF's
    energy grid, folded through the actual redistribution matrix
    (MATRIX extension), and scored per detector channel (EBOUNDS grid)
    within the instrument band. Effective area (ARF) is deliberately
    excluded: it weights channels but does not mix them, and the scores
    here are per-channel relative errors. Response files expected in
    --responses_dir:

      resolve - rsl_Hp_L_2025.rmf      (XRISM Cycle 3, HEASARC)
      acis    - aciss_aimpt_cy28.rmf   (Chandra CALDB proposal planning)
      heg     - aciss_heg1_cy28.grmf
      meg     - aciss_meg1_cy28.grmf

Instrument grids for (1):

  resolve - XRISM Resolve microcalorimeter: 4.5 eV FWHM (composite,
            in-flight), 1.7-12 keV (Be window / closed gate valve band;
            the design band extends to 0.3 keV)
  acis    - Chandra ACIS CCD: FWHM interpolated linearly between the
            canonical 96 eV @ 1.49 keV and 150 eV @ 5.9 keV, 0.3-10 keV
  heg     - Chandra HETG/HEG: 0.012 A FWHM constant in wavelength,
            0.8-10 keV
  meg     - Chandra HETG/MEG: 0.023 A FWHM constant in wavelength,
            0.4-5 keV

Emulators are picked up if their checkpoints exist: the unbroadened
line-head model (--linehead_ckpt), the plain (T, v) model
(--broadened_ckpt, train_broadened) and the Gaussian-line (T, v) model
(--broadened2_ckpt, train_broadened2).

    python scripts/benchmark_instruments.py --nspec 16 \\
        --linehead_ckpt <path>/t04_long.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_operator import load_model
from spexai.train.broadening import (direct_broaden, rebin_flux,
                                     uniform_log_edges)
from spexai.train.operator import OperatorConfig, SpectralOperator, \
    edges_from_centers
from spexai.train.train_broadened import norm_velocity
from spexai.train.train_operator import SpectrumData

HC_KEV_A = 12.398425  # keV * Angstrom


def resolve_edges(oversample, lo=1.7, hi=12.0, fwhm_kev=0.0045):
    return torch.arange(lo, hi, fwhm_kev / oversample, dtype=torch.float64)


def acis_edges(oversample, lo=0.3, hi=10.0):
    def fwhm(e):  # keV; linear through (1.49, 0.096) and (5.9, 0.150)
        return max(0.060, 0.096 + (0.150 - 0.096) * (e - 1.49) / (5.9 - 1.49))
    e, out = lo, [lo]
    while e < hi:
        e += fwhm(e) / oversample
        out.append(e)
    return torch.tensor(out, dtype=torch.float64)


def grating_edges(oversample, dlam, lam_lo, lam_hi):
    lam = np.arange(lam_lo, lam_hi, dlam / oversample)
    return torch.from_numpy(HC_KEV_A / lam[::-1].copy())


def instrument_grids(oversample):
    return {
        "resolve": resolve_edges(oversample),
        "acis": acis_edges(oversample),
        "heg": grating_edges(oversample, 0.012, HC_KEV_A / 10.0,
                             HC_KEV_A / 0.8),
        "meg": grating_edges(oversample, 0.023, HC_KEV_A / 5.0,
                             HC_KEV_A / 0.4),
    }


BANDS = {"resolve": (1.7, 12.0), "acis": (0.3, 10.0),
         "heg": (0.8, 10.0), "meg": (0.4, 5.0)}
RMF_FILES = {"resolve": "rsl_Hp_L_2025.rmf", "acis": "aciss_aimpt_cy28.rmf",
             "heg": "aciss_heg1_cy28.grmf", "meg": "aciss_meg1_cy28.grmf"}


def load_rmf(path):
    """OGIP RMF -> (energy edges (N+1,), sparse redistribution (N, C),
    channel E_MIN/E_MAX). Parsed matrix is cached next to the file."""
    import scipy.sparse as sp
    cache = path + ".npz"
    if os.path.exists(cache):
        z = np.load(cache)
        R = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"]))
        return z["edges"], R, z["e_min"], z["e_max"]
    from astropy.io import fits
    with fits.open(path) as h:
        m = h["MATRIX"]
        det = m.header["DETCHANS"]
        names = [c.name for c in m.columns]
        tlmin = m.header.get(f"TLMIN{names.index('F_CHAN') + 1}", 1)
        d = m.data
        e_lo = np.asarray(d["ENERG_LO"], dtype=np.float64)
        e_hi = np.asarray(d["ENERG_HI"], dtype=np.float64)
        assert np.allclose(e_lo[1:], e_hi[:-1]), "non-contiguous RMF grid"
        rows, cols, vals = [], [], []
        for i in range(len(d)):
            fch = np.atleast_1d(d["F_CHAN"][i]).astype(np.int64)
            nch = np.atleast_1d(d["N_CHAN"][i]).astype(np.int64)
            mat = np.atleast_1d(d["MATRIX"][i])
            pos = 0
            for g in range(int(d["N_GRP"][i])):
                n = int(nch[g])
                if n <= 0:
                    continue
                f = int(fch[g]) - tlmin
                cols.append(np.arange(f, f + n))
                rows.append(np.full(n, i))
                vals.append(mat[pos:pos + n])
                pos += n
        R = sp.csr_matrix(
            (np.concatenate(vals).astype(np.float64),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(d), det))
        eb = h["EBOUNDS"].data
        e_min = np.asarray(eb["E_MIN"], dtype=np.float64)
        e_max = np.asarray(eb["E_MAX"], dtype=np.float64)
    edges = np.append(e_lo, e_hi[-1])
    np.savez(cache, data=R.data, indices=R.indices, indptr=R.indptr,
             shape=np.array(R.shape), edges=edges, e_min=e_min, e_max=e_max)
    return edges, R, e_min, e_max


def fold(flux, edges_in, rmf_edges, R):
    """Rebin integrated flux to the RMF energy grid and fold through the
    redistribution matrix; returns counts-space flux per channel."""
    f = rebin_flux(flux, edges_in, torch.from_numpy(rmf_edges)).numpy()
    return torch.from_numpy(f @ R)


def metrics(pred, truth):
    """Per-spectrum mean relative error over occupied instrument bins."""
    valid = truth > 0
    eps = torch.where(valid,
                      torch.abs(pred / truth.clamp(min=1e-300) - 1.0),
                      torch.zeros_like(truth))
    mre = (eps.sum(1) / valid.sum(1).clamp(min=1)).numpy()
    return {"mre_mean": float(mre.mean()),
            "mre_median": float(np.median(mre)),
            "yield_1pct": float((mre <= 0.01).mean() * 100),
            "yield_10pct": float((mre <= 0.10).mean() * 100)}


def load_broadened(path, edges):
    b = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OperatorConfig(**b["config"])
    uni = uniform_log_edges(float(edges[0]), float(edges[-1]),
                            b["args"]["dlx"])
    uni_cen = torch.sqrt(uni[:-1] * uni[1:])
    stats = ((b["state_dict"]["bn_mu"], b["state_dict"]["bn_sigma"])
             if cfg.use_binnorm else None)
    model = SpectralOperator(cfg, energy_grid=uni_cen, bin_stats=stats)
    model.load_state_dict(b["state_dict"])
    return model.eval(), uni, uni_cen, b["args"]


@torch.no_grad()
def predict_broadened_uniflux(model, uni, uni_cen, margs, temps, v,
                              device, chunk=8192):
    """(T,v) emulator (train_broadened): integrated flux on its uniform grid."""
    tn = model.norm_temp(temps.to(device))
    vv = torch.full_like(tn, float(v))
    theta = torch.stack([tn, norm_velocity(vv, margs["vmin"],
                                           margs["vmax"])], dim=1)
    K = len(uni_cen)
    x = model.norm_energy(uni_cen.to(device)).view(1, -1, 1)
    dens = torch.cat(
        [torch.pow(10.0, model.forward_norm(
            theta, x[:, lo:lo + chunk].expand(len(temps), -1, -1),
            bins=torch.arange(lo, min(lo + chunk, K), device=device)))
         for lo in range(0, K, chunk)], dim=1).cpu()
    return dens * (uni[1:] - uni[:-1])


@torch.no_grad()
def predict_broadened2_uniflux(model, margs, temps, v, device, chunk=8192):
    """Gaussian-line (T,v) emulator: integrated flux on its uniform grid."""
    from spexai.train.operator import edges_from_centers as e_f_c
    tn = model.trunk.norm_temp(temps.to(device))
    vv_full = torch.full_like(tn, float(v))
    theta = torch.stack([tn, norm_velocity(vv_full, margs["vmin"],
                                           margs["vmax"])], dim=1)
    K = len(model.centers)
    dens = torch.cat(
        [torch.pow(10.0, model(theta, vv_full,
                               torch.arange(lo, min(lo + chunk, K),
                                            device=device)))
         for lo in range(0, K, chunk)], dim=1).cpu()
    uni = e_f_c(model.centers.cpu())
    return dens * (uni[1:] - uni[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--linehead_ckpt", default=None,
                    help="unbroadened line-head checkpoint (default "
                         "<rundir>/line_head.pt)")
    ap.add_argument("--broadened_ckpt", default=None,
                    help="default <rundir>/broadened/broadened.pt")
    ap.add_argument("--broadened2_ckpt", default=None,
                    help="default <rundir>/broadened2/broadened2.pt")
    ap.add_argument("--nspec", type=int, default=16)
    ap.add_argument("--velocities", nargs="+", type=float,
                    default=[100.0, 300.0, 1000.0])
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="instrument bins per resolution FWHM")
    ap.add_argument("--responses_dir",
                    default="/Users/danielahuppenkothen/work/data/spexai/responses",
                    help="directory with instrument RMF files (see "
                         "docstring); missing files skip the RMF pass")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    edges = edges_from_centers(data.energy).double()
    widths = edges[1:] - edges[:-1]

    idx = data.test_idx if args.split == "test" else data.val_idx
    sel = idx[torch.linspace(0, len(idx) - 1, args.nspec).long()]
    temps = data.temps[sel]
    flux = (torch.pow(10.0, torch.clamp(data.logflux[sel].double(), min=-30))
            * widths)

    grids = instrument_grids(args.oversample)
    for name, ge in grids.items():
        print(f"{name}: {len(ge) - 1} bins, "
              f"{float(ge[0]):.2f}-{float(ge[-1]):.2f} keV", flush=True)

    rmfs = {}
    for gname, fn in RMF_FILES.items():
        p = os.path.join(args.responses_dir, fn)
        if os.path.exists(p):
            t0 = time.time()
            try:
                redges, R, e_min, e_max = load_rmf(p)
            except Exception as e:
                print(f"{gname} RMF unreadable ({e}), skipping", flush=True)
                continue
            lo, hi = BANDS[gname]
            chan = (e_min >= lo) & (e_max <= hi)
            rmfs[gname] = (redges, R, chan)
            print(f"{gname} RMF: {R.shape[0]} energies x {R.shape[1]} "
                  f"channels ({chan.sum()} in band, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"{gname} RMF not found ({p}), skipping", flush=True)

    results = {}

    # ---- unbroadened line-head emulator vs raw SPEX truth ----------------
    ckpt = args.linehead_ckpt or os.path.join(args.rundir, "line_head.pt")
    if os.path.exists(ckpt):
        # forward_on_grid rebins in float64, which MPS does not support
        lh_device = "cpu" if device == "mps" else device
        model, _ = load_model(ckpt, data)
        model = model.to(lh_device)
        name = os.path.basename(ckpt).replace(".pt", "")
        print(f"unbroadened emulator: {name}", flush=True)
        for gname, ge in grids.items():
            truth = rebin_flux(flux, edges, ge)
            with torch.no_grad():
                pred = model.forward_on_grid(temps.to(lh_device),
                                             ge.float().to(lh_device))
            r = metrics(pred.double().cpu(), truth)
            results.setdefault(gname, {})[name] = r
            print(f"  {gname:8s} MRE={r['mre_mean']:.5f} "
                  f"median={r['mre_median']:.5f} "
                  f"yield1%={r['yield_1pct']:.1f}", flush=True)
        for gname, (redges, R, chan) in rmfs.items():
            truth = fold(flux, edges, redges, R)[:, chan]
            with torch.no_grad():
                pred_int = model.forward_on_grid(
                    temps.to(lh_device),
                    torch.from_numpy(redges).float().to(lh_device))
            pred = torch.from_numpy(
                pred_int.double().cpu().numpy() @ R)[:, chan]
            r = metrics(pred, truth)
            results.setdefault(f"{gname}_rmf", {})[name] = r
            print(f"  {gname:8s} (RMF) MRE={r['mre_mean']:.5f} "
                  f"median={r['mre_median']:.5f} "
                  f"yield1%={r['yield_1pct']:.1f}", flush=True)

    # ---- (T, v) emulators vs erf-broadened SPEX truth --------------------
    broadened = []
    bckpt = args.broadened_ckpt or os.path.join(args.rundir, "broadened",
                                                "broadened.pt")
    if os.path.exists(bckpt):
        m, uni, uni_cen, margs = load_broadened(bckpt, edges)
        broadened.append(("broadened", lambda v, m=m.to(device), u=uni,
                          c=uni_cen, a=margs: (
                              predict_broadened_uniflux(m, u, c, a, temps,
                                                        v, device), u)))
    b2ckpt = args.broadened2_ckpt or os.path.join(args.rundir, "broadened2",
                                                  "broadened2.pt")
    if os.path.exists(b2ckpt):
        from spexai.train.train_broadened2 import load_broadened2
        m2, margs2 = load_broadened2(b2ckpt, data)
        uni2 = edges_from_centers(m2.centers)
        broadened.append(("broadened2", lambda v, m=m2.to(device),
                          a=margs2, u=uni2: (
                              predict_broadened2_uniflux(m, a, temps, v,
                                                         device), u)))

    for v in (args.velocities if broadened else []):
        print(f"--- v = {v:.0f} km/s", flush=True)
        truth_native = direct_broaden(flux.float(), edges.float(), v).double()
        for mname, predict in broadened:
            pred_uni, uni = predict(v)
            for gname, ge in grids.items():
                truth = rebin_flux(truth_native, edges, ge)
                pred = rebin_flux(pred_uni.double(), uni.double(), ge)
                r = metrics(pred, truth)
                results.setdefault(gname, {})[f"{mname}_v{v:.0f}"] = r
                print(f"  {mname:11s} {gname:8s} MRE={r['mre_mean']:.5f} "
                      f"median={r['mre_median']:.5f} "
                      f"yield1%={r['yield_1pct']:.1f}", flush=True)
            for gname, (redges, R, chan) in rmfs.items():
                truth = fold(truth_native, edges, redges, R)[:, chan]
                pred = fold(pred_uni.double(), uni.double(),
                            redges, R)[:, chan]
                r = metrics(pred, truth)
                results.setdefault(f"{gname}_rmf",
                                   {})[f"{mname}_v{v:.0f}"] = r
                print(f"  {mname:11s} {gname:8s} (RMF) "
                      f"MRE={r['mre_mean']:.5f} "
                      f"median={r['mre_median']:.5f} "
                      f"yield1%={r['yield_1pct']:.1f}", flush=True)

    out = os.path.join(args.rundir, f"benchmark_instruments_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)

    md = ["| instrument | model | MRE | median | yield1% | yield10% |",
          "|---|---|---|---|---|---|"]
    for gname, models in results.items():
        for mname, r in models.items():
            md.append(f"| {gname} | {mname} | {r['mre_mean']:.5f} | "
                      f"{r['mre_median']:.5f} | {r['yield_1pct']:.1f} | "
                      f"{r['yield_10pct']:.1f} |")
    with open(out.replace(".json", ".md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
