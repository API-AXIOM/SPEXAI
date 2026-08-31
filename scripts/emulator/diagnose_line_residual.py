"""Per-line residual diagnostic for a trained operator checkpoint.

On the NATIVE grid a line bin's only degree of freedom is its amplitude (the
Gaussian width/position is a downstream broadening step), so the native-grid
line-bin error IS the line-amplitude head's error. A plain interpolation of the
SPEX grid reproduces these same bins to ~1e-6 (baselines_test.json), so any
large error here is the head failing to fit a trivially-fittable target.

This script asks WHERE that error lives, to tell a capacity problem (uniform)
apart from a structural one (concentrated in dense line forests / weak lines /
a T band):

  * per-bin relative error vs energy (line vs continuum bins)
  * per-line error vs line STRENGTH (are the strong "line heads" or the weak
    lines the problem?)
  * per-line error vs LOCAL LINE DENSITY (isolated lines vs blended forests)
  * per-spectrum line-MRE vs TEMPERATURE

Usage (Ti, the v0_base sweep checkpoint):
  python scripts/diagnose_line_residual.py \
      --checkpoint ~/work/data/spexai/runs_linehead/v0_base/element22/tier1/reweight_full.pt \
      --cachedir   ~/work/data/spexai/processed/element22 \
      --outdir     ~/work/data/spexai/runs_linehead/diag_Z22
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from spexai.eval import _predict_grid
from spexai.inference.operator_model import load_operator
from spexai.metrics import FLOOR, abs_rel_error, line_continuum_masks
from spexai.data import SpectrumData

torch.manual_seed(0)
np.random.seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cachedir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--n_spectra", type=int, default=500,
                    help="subsample this many spectra (per-line stats are "
                         "robust well below the full split)")
    ap.add_argument("--density_window", type=int, default=25,
                    help="+/- native bins counted for local line density")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = "cpu"

    model = load_operator(args.checkpoint, map_location=dev).to(dev)
    if model.line_head is None:
        raise SystemExit("checkpoint has no line head -- nothing to diagnose")
    data = SpectrumData(args.cachedir)
    idx = (data.test_idx if args.split == "test" else data.val_idx).numpy()
    rng = np.random.default_rng(0)
    if len(idx) > args.n_spectra:
        idx = np.sort(rng.choice(idx, args.n_spectra, replace=False))
    temps = data.temps[idx].numpy()                       # (N,)

    # predicted vs true log10 flux on the full native grid. Use the benchmark's
    # guarded relative error (clamped to the empty-bin FLOOR, +/-4 dex) so these
    # numbers match benchmark_test.json exactly.
    pred_log = _predict_grid(model, data, idx, dev)        # (N, n_bins)
    true_log = data.logflux[idx].numpy()                   # (N, n_bins)
    err = abs_rel_error(pred_log, true_log)                # (N, n_bins), guarded
    strength_bin = 10.0 ** np.clip(true_log, FLOOR, None)  # (N, n_bins) linear

    energy = data.energy.numpy()                           # (n_bins,)
    line_ids = model.line_head.line_ids.cpu().numpy()      # (n_bins,), -1 = cont
    is_line = line_ids >= 0                                # model's line slots
    # per-bin error over the spectra where the bin is VALID (target > FLOOR);
    # bins rarely valid are dropped so floor-only bins don't create fake spikes.
    valid0 = data.logflux[idx].numpy() > FLOOR
    frac_valid = valid0.mean(axis=0)                       # (n_bins,)
    with np.errstate(invalid="ignore"):
        rel_bin = np.where(valid0, err, np.nan)
        rel_bin = np.nanmean(rel_bin, axis=0)              # per-bin, valid only
    show = frac_valid > 0.2                                # bins valid often enough

    # headline line/continuum split uses the DATA-DRIVEN benchmark masks
    valid, lmask, cmask = line_continuum_masks(true_log)
    line_mre = err[lmask].mean()
    cont_mre = err[cmask].mean()
    print(f"checkpoint: {args.checkpoint}")
    print(f"{args.split}: {len(idx)} spectra, {int(is_line.sum())} model line "
          f"bins ({100*is_line.mean():.1f}% of {len(energy)})")
    print(f"line  MRE = {line_mre:.5f}   (benchmark line/cont masks)")
    print(f"cont  MRE = {cont_mre:.5f}")

    # is the error a hard-band problem hitting BOTH lines and continuum?
    ecol = np.broadcast_to(energy, err.shape)
    for name, cut in (("soft E<1.2keV", ecol < 1.2), ("hard E>=1.2keV", ecol >= 1.2)):
        lm, cm = lmask & cut, cmask & cut
        print(f"  {name:16}  line MRE={err[lm].mean():.5f} "
              f"({int(lm.sum())} bins) | cont MRE={err[cm].mean():.5f} "
              f"({int(cm.sum())} bins)")

    # --- aggregate per line (slot) -----------------------------------------
    slots = line_ids[is_line]
    uniq = np.unique(slots)
    line_bin_pos = np.nonzero(is_line)[0]                  # native-bin index of each line bin
    per_line_err, per_line_strength, per_line_energy, per_line_density = \
        [], [], [], []
    # local density: line bins within +/- window of each line bin
    win = args.density_window
    line_mask_int = is_line.astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(line_mask_int)])
    for s in uniq:
        bins_s = np.nonzero(line_ids == s)[0]
        vb = valid[:, bins_s]                              # (N, nb_s) present?
        if not vb.any():
            continue
        per_line_err.append(err[:, bins_s][vb].mean())     # over valid (spectrum,bin)
        per_line_strength.append(strength_bin[:, bins_s].mean(axis=0).max())
        per_line_energy.append(energy[bins_s].mean())
        b = bins_s[len(bins_s) // 2]                       # central bin
        lo, hi = max(0, b - win), min(len(energy), b + win + 1)
        per_line_density.append(int(csum[hi] - csum[lo]))  # line bins in window
    per_line_err = np.array(per_line_err)
    per_line_strength = np.array(per_line_strength)
    per_line_energy = np.array(per_line_energy)
    per_line_density = np.array(per_line_density)

    # concentration: share of total line abs-error from the strongest lines
    order = np.argsort(per_line_strength)[::-1]
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    print(f"\nlines: {len(uniq)}")
    print(f"  rho(per-line err, strength) = {spearman(per_line_err, per_line_strength):+.2f}")
    print(f"  rho(per-line err, density)  = {spearman(per_line_err, per_line_density):+.2f}")
    # weak vs strong halves
    med = np.median(per_line_strength)
    print(f"  weak-half  line MRE = {per_line_err[per_line_strength <= med].mean():.5f}")
    print(f"  strong-half line MRE = {per_line_err[per_line_strength > med].mean():.5f}")
    # isolated vs blended (low vs high density) halves
    dmed = np.median(per_line_density)
    print(f"  isolated (density<=med) MRE = {per_line_err[per_line_density <= dmed].mean():.5f}")
    print(f"  blended  (density> med) MRE = {per_line_err[per_line_density > dmed].mean():.5f}")

    # per-spectrum line MRE vs temperature (each spectrum's own line bins)
    spec_line_mre = np.array([err[i, lmask[i]].mean() if lmask[i].any() else np.nan
                              for i in range(len(idx))])   # (N,)
    good = ~np.isnan(spec_line_mre)
    print(f"  rho(per-spectrum line MRE, T) = "
          f"{spearman(spec_line_mre[good], temps[good]):+.2f}")

    # --- plots -------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    csel, lsel = show & ~is_line, show & is_line
    a.scatter(energy[csel], rel_bin[csel], s=2, alpha=0.3,
              label="continuum", color="#888")
    a.scatter(energy[lsel], rel_bin[lsel], s=6, alpha=0.6,
              label="line", color="crimson")
    a.axhline(1e-3, color="k", ls=":", lw=1, label="0.1%")
    a.set(xscale="log", yscale="log", xlabel="energy (keV)",
          ylabel="mean rel. error (valid bins)",
          title="per-bin error vs energy")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.scatter(per_line_strength, per_line_err, s=8, alpha=0.5, color="crimson")
    a.axhline(1e-3, color="k", ls=":", lw=1)
    a.set(xscale="log", yscale="log", xlabel="line peak strength (flux)",
          ylabel="per-line MRE", title="per-line error vs strength")

    a = ax[1, 0]
    a.scatter(per_line_density, per_line_err, s=8, alpha=0.5, color="crimson")
    a.axhline(1e-3, color="k", ls=":", lw=1)
    a.set(yscale="log", xlabel=f"local line density (bins in +/-{win})",
          ylabel="per-line MRE", title="per-line error vs local density")

    a = ax[1, 1]
    a.scatter(temps, spec_line_mre, s=8, alpha=0.5, color="crimson")
    a.axhline(1e-3, color="k", ls=":", lw=1)
    a.set(yscale="log", xlabel="temperature (keV)",
          ylabel="per-spectrum line MRE", title="line error vs temperature")

    fig.suptitle(f"Line-amplitude residual diagnostic — "
                 f"{os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))}"
                 f"  (line MRE {line_mre:.4f}, cont {cont_mre:.4f})")
    fig.tight_layout()
    out = os.path.join(args.outdir, "line_residual_diagnostic.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
