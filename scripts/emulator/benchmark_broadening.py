"""Compare velocity-broadening implementations: accuracy and speed.

Ground truth: the exact bin-integrated Gaussian redistribution
(direct_broaden, float64, CPU) applied to true SPEX test spectra.

Two groups of methods:

  convolution methods (input = the TRUE unbroadened spectrum;
  measures the broadening algorithm alone):
    * sparse   - faithful standalone replica of the current
                 CombinedModel.broadening (banded sparse matrix rebuilt
                 per call, row-normalised)
    * fft      - broaden_native (scatter to uniform log grid, FFT, rebin)

  emulator methods (input = temperature/velocity only; measures the
  full emulate-and-broaden pipeline against broadened SPEX truth):
    * emulator - (T, v) model trained on broadened targets (option 2),
                 checkpoint from spexai.train.train_broadened, if present
    * hybrid   - unbroadened line-head emulator + FFT-broadened trunk +
                 analytic Gaussian lines (option 3), if a line_head/combo
                 checkpoint is present

    python scripts/benchmark_broadening.py [--nspec 16]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.emulator.benchmark_operator import load_model
from spexai.broadening import (broaden_native, direct_broaden,
                                     hybrid_broadened_on_grid, rebin_flux,
                                     uniform_log_edges)
from spexai.operator import OperatorConfig, SpectralOperator, \
    edges_from_centers
from spexai.data import SpectrumData


# --- faithful replica of the current implementation -----------------------
# make_sparsex reproduced from spexai/inference/write_tensors.py (avoiding
# its astropy import); apply_sparse follows CombinedModel.broadening.

def make_sparsex(x, n=300):
    collumn_index = torch.tensor([])
    row_index = torch.tensor([])
    x_values = torch.tensor([])
    for i, x_i in enumerate(x):
        full_col = torch.arange(-int((n / 25) * x_i), int((n / 25) * x_i),
                                dtype=torch.long) + i
        full_col = full_col[full_col >= 0]
        col = full_col[full_col <= len(x) - 1]
        row = torch.ones_like(col) * i
        x_row = (x[col] - x_i) / x_i
        collumn_index = torch.cat((collumn_index, col)).type(torch.long)
        row_index = torch.cat((row_index, row))
        x_values = torch.cat((x_values, x_row))
    index = torch.cat((row_index.view(1, -1), collumn_index.view(1, -1)))
    return torch.sparse_coo_tensor(
        index, x_values, (len(x), len(x))).type(torch.float32)


def apply_sparse(sm_x_csr, e_diff, density, velocity):
    """CombinedModel.broadening, standalone (density in, density out)."""
    shape = (len(e_diff), len(e_diff))
    prev_values = sm_x_csr.values().type(torch.float32)
    stdev = torch.tensor(velocity * 1e3 / 299792458.0, dtype=torch.float32)
    gaussian_values = torch.exp(-0.5 * (prev_values / stdev) ** 2)
    normal_matrix = torch.sparse_csr_tensor(
        sm_x_csr.crow_indices(), sm_x_csr.col_indices(), gaussian_values,
        shape, dtype=torch.float32)
    norm_values = 1.0 / torch.mv(normal_matrix, e_diff).flatten()
    normal_matrix = (normal_matrix.to_sparse_coo()
                     * norm_values[:, None]).to_sparse_csr()
    spectrum_dx = density.view(-1, 1) * e_diff.view(-1, 1)
    return torch.sparse.mm(normal_matrix, spectrum_dx).flatten()


def accuracy(pred_int, ref_int, widths):
    dens_ref = ref_int / widths
    mask = dens_ref > 1e-6 * dens_ref.max(dim=1, keepdim=True).values
    rel = ((pred_int - ref_int).abs() / ref_int.clamp(min=1e-30))[mask]
    l1 = ((pred_int - ref_int).abs().sum(1) / ref_int.sum(1)).mean()
    return {"mre": float(rel.mean()), "rel_median": float(rel.median()),
            "misplaced_flux": float(l1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--rundir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--linehead_ckpt", default=None,
                    help="line-head/combo checkpoint for the hybrid method "
                         "(default: <rundir>/line_head.pt)")
    ap.add_argument("--broadened_ckpt", default=None,
                    help="(T,v) emulator checkpoint (default: "
                         "<rundir>/broadened/broadened.pt)")
    ap.add_argument("--broadened2_ckpt", default=None,
                    help="revised Gaussian-line (T,v) emulator (default: "
                         "<rundir>/broadened2/broadened2.pt)")
    ap.add_argument("--nspec", type=int, default=16)
    ap.add_argument("--velocities", nargs="+", type=float,
                    default=[100.0, 300.0, 1000.0])
    ap.add_argument("--nrep", type=int, default=10)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    edges = edges_from_centers(data.energy)
    widths = edges[1:] - edges[:-1]
    sel = data.test_idx[np.linspace(0, len(data.test_idx) - 1,
                                    args.nspec).astype(int)]
    temps = data.temps[sel]
    flux = torch.pow(10.0, torch.clamp(data.logflux[sel], min=-30)) * widths

    print("building sparse matrix (one-off setup of the current method) ...",
          flush=True)
    t0 = time.time()
    sm_x_csr = make_sparsex(data.energy, n=300).to_sparse_csr()
    print(f"  {time.time()-t0:.1f}s, nnz={sm_x_csr.values().shape[0]:,}")

    # optional emulator checkpoints
    hybrid_model = None
    ckpt = args.linehead_ckpt or os.path.join(args.rundir, "line_head.pt")
    if os.path.exists(ckpt):
        hybrid_model, _ = load_model(ckpt, data)
        hybrid_model = hybrid_model.to(device)
        print(f"hybrid method uses {os.path.basename(ckpt)}")
    broadened_model = None
    bckpt = args.broadened_ckpt or os.path.join(args.rundir, "broadened",
                                                "broadened.pt")
    if os.path.exists(bckpt):
        b = torch.load(bckpt, map_location="cpu", weights_only=False)
        cfg = OperatorConfig(**b["config"])
        uni = uniform_log_edges(float(edges[0]), float(edges[-1]),
                                b["args"]["dlx"])
        uni_cen = torch.sqrt(uni[:-1] * uni[1:])
        stats = ((b["state_dict"]["bn_mu"], b["state_dict"]["bn_sigma"])
                 if cfg.use_binnorm else None)
        broadened_model = SpectralOperator(cfg, energy_grid=uni_cen,
                                           bin_stats=stats)
        broadened_model.load_state_dict(b["state_dict"])
        broadened_model = broadened_model.eval().to(device)
        bvmin, bvmax = b["args"]["vmin"], b["args"]["vmax"]
        print(f"(T,v) emulator uses {os.path.basename(bckpt)}")

    # revised Gaussian-line (T, v) emulator (train_broadened2)
    b2_model = b2_uni = None
    b2ckpt = args.broadened2_ckpt or os.path.join(args.rundir, "broadened2",
                                                  "broadened2.pt")
    if os.path.exists(b2ckpt):
        from spexai.train.train_broadened2 import load_broadened2
        b2_model, b2_margs = load_broadened2(b2ckpt, data)
        b2_model = b2_model.to(device)
        b2_uni = edges_from_centers(b2_model.centers)
        print(f"(T,v) emulator v2 uses {os.path.basename(b2ckpt)}")

    results = {}
    for v in args.velocities:
        print(f"--- v = {v:.0f} km/s", flush=True)
        ref = direct_broaden(flux, edges, v)
        res = {}

        # sparse replica (CPU, like-for-like with the original)
        t0 = time.time()
        for _ in range(args.nrep):
            out = apply_sparse(sm_x_csr, widths, flux[0] / widths, v)
        t_sparse = (time.time() - t0) / args.nrep
        pred = torch.stack([apply_sparse(sm_x_csr, widths, flux[i] / widths, v)
                            for i in range(len(sel))]) * widths
        res["sparse"] = {**accuracy(pred, ref, widths),
                         "t_ms": t_sparse * 1e3, "device": "cpu"}

        # fft convolution (CPU for like-for-like timing)
        t0 = time.time()
        for _ in range(args.nrep):
            out = broaden_native(flux[:1], edges, v)
        t_fft = (time.time() - t0) / args.nrep
        pred = broaden_native(flux, edges, v)
        res["fft"] = {**accuracy(pred, ref, widths),
                      "t_ms": t_fft * 1e3, "device": "cpu"}

        # (T, v) emulator, end-to-end vs broadened truth
        if broadened_model is not None:
            with torch.no_grad():
                tn = broadened_model.norm_temp(temps.to(device))
                lv = torch.log10(torch.tensor(v, device=device))
                vn = 2.0 * (lv - np.log10(bvmin)) / np.log10(bvmax / bvmin) - 1.0
                theta = torch.stack([tn, vn.expand(len(sel))], dim=1)
                K = len(uni_cen)
                x = broadened_model.norm_energy(uni_cen.to(device)).view(1, -1, 1)
                t0 = time.time()
                for _ in range(args.nrep):
                    dens_u = torch.cat(
                        [torch.pow(10.0, broadened_model.forward_norm(
                            theta[:1], x[:, lo:lo + 8192],
                            bins=torch.arange(lo, min(lo + 8192, K),
                                              device=device)))
                         for lo in range(0, K, 8192)], dim=1)
                if device == "mps":
                    torch.mps.synchronize()
                t_emu = (time.time() - t0) / args.nrep
                dens_u = torch.cat(
                    [torch.pow(10.0, broadened_model.forward_norm(
                        theta, x[:, lo:lo + 8192].expand(len(sel), -1, -1),
                        bins=torch.arange(lo, min(lo + 8192, K),
                                          device=device)))
                     for lo in range(0, K, 8192)], dim=1)
                uni_w = (uni[1:] - uni[:-1]).to(device)
                pred = rebin_flux((dens_u * uni_w), uni.to(device),
                                  edges.to(device)).cpu()
            res["emulator_Tv"] = {**accuracy(pred, ref, widths),
                                  "t_ms": t_emu * 1e3, "device": device}

        # hybrid, end-to-end vs broadened truth
        if hybrid_model is not None:
            t0 = time.time()
            for _ in range(args.nrep):
                out = hybrid_broadened_on_grid(
                    hybrid_model, temps[:1].to(device), v, edges.to(device))
            if device == "mps":
                torch.mps.synchronize()
            t_hyb = (time.time() - t0) / args.nrep
            pred = hybrid_broadened_on_grid(
                hybrid_model, temps.to(device), v, edges.to(device)).cpu()
            res["hybrid"] = {**accuracy(pred, ref, widths),
                             "t_ms": t_hyb * 1e3, "device": device}

        # revised (T, v) emulator v2, end-to-end vs broadened truth
        if b2_model is not None:
            from scripts.benchmark_instruments import predict_broadened2_uniflux
            t0 = time.time()
            for _ in range(args.nrep):
                pu = predict_broadened2_uniflux(b2_model, b2_margs,
                                                temps[:1], v, device)
            if device == "mps":
                torch.mps.synchronize()
            t_b2 = (time.time() - t0) / args.nrep
            pu = predict_broadened2_uniflux(b2_model, b2_margs, temps, v,
                                            device)
            pred = rebin_flux(pu.to(device), b2_uni.to(device),
                              edges.to(device)).cpu()
            res["emulator_Tv2"] = {**accuracy(pred, ref, widths),
                                   "t_ms": t_b2 * 1e3, "device": device}

        for m, r in res.items():
            print(f"  {m:12s} MRE={r['mre']:.4f} median={r['rel_median']:.5f} "
                  f"misplaced={r['misplaced_flux']:.2e} "
                  f"t={r['t_ms']:.1f} ms [{r['device']}]", flush=True)
        results[f"v{v:.0f}"] = res

    out_path = os.path.join(args.rundir, "benchmark_broadening.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
