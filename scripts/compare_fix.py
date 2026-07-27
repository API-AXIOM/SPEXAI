"""Close the diagnostic loop: show the T-embedding removes the exact residual
signature the diagnosis blamed.

residual_fft.py diagnosed the failing-band floor as a smooth LOW-frequency
continuum-residual misfit, concentrated in each element's hard temperature
band. This script overlays that same diagnostic for a BASELINE checkpoint and
its +T-embedding counterpart on the same element, in that element's hard band,
so the causal claim is testable directly: if the low-frequency continuum
power collapses, the fix removed the diagnosed cause (not just raised yield).

Reuses residual_fft's per-spectrum machinery (element_residual_spectrum +
aggregate) and adds only the base-vs-fixed overlay.

    python scripts/compare_fix.py --cachedir ~/work/data/spexai/processed/element11 \
        --base ~/work/data/spexai/runs_sweep/na_base/sweep.pt \
        --fixed ~/work/data/spexai/runs_sweep/na_tembed/sweep.pt \
        --band cold --label Na --out docs/development_plots/fix_confirmation
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from residual_fft import (aggregate, element_residual_spectrum,  # noqa: E402
                          _band_lines)

# hard-T band -> (lo, hi) keV selection for the stratified overlay
BANDS = {"cold": (0.0, 1.0), "mid": (1.0, 5.0), "hot": (5.0, np.inf),
         "all": (0.0, np.inf)}


def band_subset(temps: np.ndarray, band: str) -> np.ndarray:
    lo, hi = BANDS[band]
    return np.nonzero((temps >= lo) & (temps < hi))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir", required=True)
    ap.add_argument("--base", required=True, help="baseline checkpoint (.pt)")
    ap.add_argument("--fixed", required=True, help="+T-embedding checkpoint (.pt)")
    ap.add_argument("--band", default="all", choices=list(BANDS),
                    help="temperature band to overlay (the element's hard band)")
    ap.add_argument("--label", default="", help="element label for titles")
    ap.add_argument("--n_spec", type=int, default=150)
    ap.add_argument("--n_fft", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/development_plots/fix_confirmation")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    # z only labels the plot; element_residual_spectrum takes explicit paths
    z = 0
    print(f"{args.label}: baseline ...", flush=True)
    rb = element_residual_spectrum(z, args.cachedir, args.base, args.device,
                                   args.n_spec, args.n_fft)
    print(f"{args.label}: +T-embedding ...", flush=True)
    rf = element_residual_spectrum(z, args.cachedir, args.fixed, args.device,
                                   args.n_spec, args.n_fft)
    if rb is None or rf is None:
        raise SystemExit("a checkpoint or cache was missing")

    sb = band_subset(rb["temps"], args.band)
    sf = band_subset(rf["temps"], args.band)
    ab, af = aggregate(rb, sb), aggregate(rf, sf)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
    a0.plot(rb["energy"], ab["resid_rms"], lw=0.8, color="#D55E00",
            label=f"baseline (med {np.median(ab['resid_rms']):.1e})")
    a0.plot(rf["energy"], af["resid_rms"], lw=0.8, color="#0072B2",
            label=f"+T-embedding (med {np.median(af['resid_rms']):.1e})")
    a0.axhline(4e-4, ls=":", color="k", lw=1)
    a0.text(rb["energy"][0], 4.2e-4, "0.1% target", fontsize=8, va="bottom")
    a0.set(xscale="log", yscale="log", xlabel="Energy (keV)",
           ylabel="RMS log10 residual",
           title=f"{args.label}: residual vs energy  [{args.band} band, "
                 f"n={ab['n']}/{af['n']}]")
    a0.grid(True, which="major", color="#eee", lw=0.6)
    a0.legend(frameon=False)

    a1.loglog(rb["freqs"][1:], ab["pw_cont"][1:], lw=1.0, color="#D55E00",
              label="baseline")
    a1.loglog(rf["freqs"][1:], af["pw_cont"][1:], lw=1.0, color="#0072B2",
              label="+T-embedding")
    _band_lines(a1, rb)
    a1.set(xlabel="frequency (cycles per unit x)",
           ylabel="continuum residual power",
           title=f"{args.label}: continuum residual power  "
                 f"[{args.band} band] -- the diagnosed low-f misfit")
    a1.grid(True, which="major", color="#eee", lw=0.6)
    a1.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(args.out, f"{args.label or 'element'}_{args.band}_fix.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # low-frequency continuum power (f < 10 cycles/x): the diagnosed quantity
    lo = rb["freqs"] < 10.0
    pb = float(ab["pw_cont"][lo].sum())
    pf = float(af["pw_cont"][lo].sum())
    print(f"wrote {out}")
    print(f"  low-f (<10 cyc/x) continuum power: baseline {pb:.3e} -> "
          f"fixed {pf:.3e}  ({pb / max(pf, 1e-30):.1f}x reduction)")
    print(f"  median RMS residual: baseline {np.median(ab['resid_rms']):.2e} -> "
          f"fixed {np.median(af['resid_rms']):.2e}")


if __name__ == "__main__":
    main()
