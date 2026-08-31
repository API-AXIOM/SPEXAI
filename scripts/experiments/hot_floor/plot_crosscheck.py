"""Closing cross-check: overlay the MCMC posterior offsets (median-truth, in
units of the posterior sigma) on the analytic linearised-bias prediction
(|b_sys|/sigma_stat(N) = |b_sys|/sigma_ref * sqrt(N/N_ref)) from bias_single.npz,
across the simulated count levels. If the MCMC points track the predicted lines,
the Fisher-based hot-floor result is confirmed end-to-end.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCIENCE = {"Cr", "Mn", "Ni"}
COLOR = {"Fe": "#111111", "Cr": "#d62728", "Mn": "#ff7f0e", "Ni": "#9467bd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.environ.get(
        "SPEXAI_RESULTS", os.path.expanduser("~/data/spexai_data/results")),
        "hot_floor"))
    args = ap.parse_args()
    R = args.results

    b = np.load(os.path.join(R, "bias_single.npz"), allow_pickle=True)
    bnames = [str(s) for s in b["names"]]
    ratio0 = np.abs(b["b_sys"]) / b["sigma_ref"]        # |b_sys|/sigma at N_ref
    nref = float(b["n_ref"])
    pred = dict(zip(bnames, ratio0))

    levels = []
    for f in sorted(glob.glob(os.path.join(R, "mcmc_single_vec_c*.npz"))):
        z = np.load(f, allow_pickle=True)
        names = [str(s) for s in z["names"]]
        chain, truth = z["chain"], np.asarray(z["truth"], float)
        med = np.median(chain, 0)
        sig = 0.5 * (np.percentile(chain, 84, 0) - np.percentile(chain, 16, 0))
        is_ref = "_f64" in os.path.basename(f)          # float64 reference run
        levels.append((float(z["counts"]), names, (med - truth) / sig, is_ref))
    levels.sort(key=lambda t: (t[0], t[3]))             # by counts, then ref flag

    # table
    pnames = levels[0][1]
    print(f"observed |median-truth|/sigma  vs  predicted |b_sys|/sigma(N)")
    for counts, names, off, is_ref in levels:
        tag = " [float64 ref]" if is_ref else ""
        print(f"\n--- N = {counts:.0e} in-band counts{tag} ---")
        print(f"{'param':>9} {'obs(sig)':>9} {'pred(sig)':>10}")
        for n, o in zip(names, off):
            p = pred.get(n, np.nan) * np.sqrt(counts / nref)
            print(f"{n:>9} {o:>+9.2f} {p:>10.2f}")

    # figure: |offset|/sigma vs counts, predicted lines + observed markers
    cgrid = np.logspace(4, 8.3, 100)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in pnames:
        c = COLOR.get(n, "#9aa0a6")
        emph = n in SCIENCE or n == "Fe"
        ax.plot(cgrid, pred.get(n, np.nan) * np.sqrt(cgrid / nref), "-",
                color=c, lw=2.0 if emph else 0.8, alpha=0.9 if emph else 0.5,
                zorder=3 if emph else 1, label=n if emph else None)
    for counts, names, off, is_ref in levels:
        x = counts * (1.25 if is_ref else 1.0)          # nudge float64 ref aside
        for n, o in zip(names, off):
            c = COLOR.get(n, "#9aa0a6")
            ax.plot(x, abs(o), "s" if is_ref else "o", color=c,
                    mfc="none" if is_ref else c, ms=7 if n in COLOR else 4,
                    mec=c if is_ref else "k", mew=1.0 if is_ref else 0.5, zorder=5)
    ax.plot([], [], "ko", label="MCMC (accelerated)")
    ax.plot([], [], "ks", mfc="none", label="MCMC (float64 ref)")
    ax.axhline(1.0, color="0.4", ls=":", lw=1)
    ax.text(1.1e4, 1.15, "bias = noise", color="0.4", fontsize=8)
    for x, lab in ((4e4, "XRISM/region"), (1e6, "deep core")):
        ax.axvline(x, color="0.8", lw=1)
        ax.text(x, 1.3e-2, lab, rotation=90, va="bottom", ha="right",
                color="0.55", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("in-band counts"); ax.set_ylabel(r"|median $-$ truth| / $\sigma$")
    ax.set_title("MCMC posterior offsets (points) vs linearised-bias prediction (lines)")
    ax.legend(title="lines: Fisher pred.", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.15)
    out = os.path.join(R, "crosscheck_overlay.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
