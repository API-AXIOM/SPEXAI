"""Whole-manifest and joint-model evaluation for the operator emulator.

Two capabilities missing from the per-element benchmark scripts:

* :func:`evaluate_manifest` iterates every element in ``spexai/models/
  manifest.json``, scores each self-contained checkpoint against its held-out
  SPEX cache with the shared :func:`spexai.metrics.spectrum_metrics`, and
  produces one aggregate table (dict + JSON + markdown). It scales from the
  current 16 elements to the full 30 automatically as the model store grows.

* :func:`evaluate_joint` scores the *assembled* :class:`JointOperatorModel`
  (abundance-weighted sum of elements) against an independent truth generator
  (e.g. ``SpexTruthModel``), both in flux space on an instrument grid and in
  detector-channel space after folding through a :class:`Response` -- the space
  fitting actually sees.

Runs on CPU by default: predicting the full native grid for a whole split is
memory-heavy, and MPS full-grid evaluation has frozen the laptop before.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from spexai.inference.operator_model import MODELS_DIR, load_operator
from spexai.data import SpectrumData
from spexai.metrics import spectrum_metrics


def _default_datadir(models_dir: str, manifest: dict) -> str:
    """Locate the preprocessed per-element caches (``processed/elementZ``).

    The manifest records the training ``runroot`` (``.../<data>/runs``); the
    caches live in the sibling ``.../<data>/processed``.
    """
    runroot = manifest.get("runroot", "")
    if runroot:
        return os.path.join(os.path.dirname(runroot.rstrip("/")), "processed")
    return os.path.join(os.path.dirname(models_dir), "processed")


@torch.no_grad()
def _predict_grid(model, data: SpectrumData, idx, device, echunk: int = 4096,
                  batch: int = 64):
    """Predicted log10 flux on the full native grid for spectra ``idx``.

    Batched over spectra AND chunked over energy so the (batch x points x
    embed) coordinate tensor stays bounded, mirroring ``train_operator.
    evaluate`` (predicting the whole split at once OOMs). Returns numpy
    (len(idx), n_bins)."""
    energy = data.energy.to(device)
    out = np.empty((len(idx), len(energy)), dtype=np.float32)
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        temps = data.temps[sel].to(device)                 # (b,)
        parts = [model(temps, energy[lo:lo + echunk],
                       bins=torch.arange(lo, min(lo + echunk, len(energy)),
                                         device=device))
                 for lo in range(0, len(energy), echunk)]
        out[i:i + batch] = torch.cat(parts, dim=1).cpu().numpy()
    return out                                             # (len(idx), n_bins)


def evaluate_manifest(models_dir: str = MODELS_DIR, datadir: Optional[str] = None,
                      split: str = "test", device: str = "cpu",
                      echunk: int = 4096, write: bool = True) -> Dict[int, dict]:
    """Score every element in the manifest against its held-out SPEX cache.

    Returns ``{Z: {"symbol", "overall", "lines", "continuum", "floor",
    "status"?}}``. When ``write`` is set, also drops ``manifest_eval_<split>.
    {json,md}`` next to the model store. Elements whose cache is missing are
    skipped with a recorded ``"error"`` entry rather than aborting the sweep.
    """
    with open(os.path.join(models_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if datadir is None:
        datadir = _default_datadir(models_dir, manifest)

    results: Dict[int, dict] = {}
    for zstr, entry in sorted(manifest["elements"].items(), key=lambda kv: int(kv[0])):
        z = int(zstr)
        cache = os.path.join(datadir, f"element{z}")
        if not os.path.isdir(cache):
            results[z] = {"symbol": entry.get("symbol", ""),
                          "error": f"cache not found: {cache}"}
            continue
        data = SpectrumData(cache)
        idx = data.test_idx if split == "test" else data.val_idx
        model = load_operator(os.path.join(models_dir, entry["file"]),
                              map_location=device).to(device)
        pred = _predict_grid(model, data, idx, device, echunk)
        target = data.logflux[idx].numpy()
        m = spectrum_metrics(pred, target)
        m["symbol"] = entry.get("symbol", "")
        if "status" in entry:
            m["status"] = entry["status"]
        results[z] = m

    if write:
        with open(os.path.join(models_dir, f"manifest_eval_{split}.json"), "w") as f:
            json.dump({str(z): r for z, r in results.items()}, f, indent=2)
        with open(os.path.join(models_dir, f"manifest_eval_{split}.md"), "w") as f:
            f.write(manifest_table(results) + "\n")
    return results


def manifest_table(results: Dict[int, dict]) -> str:
    """Markdown summary table from :func:`evaluate_manifest` output."""
    rows = ["| Z | el | overall MRE | line MRE | cont MRE | yield1% | "
            "yield0.1% | floor viol % | status |",
            "|---|---|---|---|---|---|---|---|---|"]
    for z in sorted(results):
        r = results[z]
        if "error" in r:
            rows.append(f"| {z} | {r.get('symbol','')} | - | - | - | - | - | - "
                        f"| {r['error']} |")
            continue
        o, ln, c, fl = r["overall"], r["lines"], r["continuum"], r["floor"]
        rows.append(
            f"| {z} | {r['symbol']} | {o['mre_mean']:.5f} | {ln['mre_mean']:.5f} "
            f"| {c['mre_mean']:.5f} | {o['yield_1pct']:.2f} | {o['yield_01pct']:.2f} "
            f"| {fl['violation_pct']:.2f} | {r.get('status', '')} |")
    return "\n".join(rows)


@torch.no_grad()
def evaluate_joint(joint, truth, response, params_list: List[dict],
                   velocity: float = 0.0) -> dict:
    """Compare the assembled emulator against an independent truth generator.

    ``joint`` and ``truth`` both expose ``flux(temp_kev, abundances, velocity,
    bin_edges) -> (B, M)`` on the response's incident-energy grid; ``truth`` is
    typically a ``SpexTruthModel`` (PCHIP over SPEX + exact broadening). For
    each ``{temp, abundances}`` in ``params_list`` we measure the median
    per-bin relative error in flux space, and the median per-channel relative
    error after folding through ``response`` (the space the likelihood uses).

    Returns ``{"flux_mre_median", "channel_mre_median", "per_case": [...]}``.
    """
    edges = response.energy_edges
    per_case, flux_errs, chan_errs = [], [], []
    for p in params_list:
        t = torch.as_tensor([float(p["temp"])], dtype=torch.float32)
        ab = p.get("abundances", {})
        fe = joint.flux(t, ab, velocity, edges).squeeze(0).cpu().numpy()
        ft = truth.flux(t, ab, velocity, edges).squeeze(0).cpu().numpy()
        ce = response.fold(torch.as_tensor(fe)).cpu().numpy()
        ct = response.fold(torch.as_tensor(ft)).cpu().numpy()
        f_rel = _median_rel(fe, ft)
        c_rel = _median_rel(ce, ct)
        flux_errs.append(f_rel)
        chan_errs.append(c_rel)
        per_case.append({"temp": float(p["temp"]),
                         "flux_mre_median": f_rel, "channel_mre_median": c_rel})
    return {
        "flux_mre_median": float(np.median(flux_errs)) if flux_errs else float("nan"),
        "channel_mre_median": float(np.median(chan_errs)) if chan_errs else float("nan"),
        "per_case": per_case,
    }


def _median_rel(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-30) -> float:
    """Median relative error over bins where truth carries real flux."""
    m = truth > truth.max() * 1e-6 if truth.size and truth.max() > 0 else truth > eps
    if not np.any(m):
        return float("nan")
    return float(np.median(np.abs(pred[m] - truth[m]) / np.maximum(truth[m], eps)))


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Evaluate every element in the model-store manifest against "
                    "its held-out SPEX cache.")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--datadir", default=None,
                    help="preprocessed caches dir (default: derived from manifest)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    res = evaluate_manifest(models_dir=args.models_dir, datadir=args.datadir,
                            split=args.split, device=args.device)
    print(manifest_table(res))


if __name__ == "__main__":
    _main()
