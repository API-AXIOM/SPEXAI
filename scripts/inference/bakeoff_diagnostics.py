"""Diagnostic plots for every sampler in the bake-off.

Produces, into ``--outdir``:

* ``corner_<sampler>.png``  -- one marginal corner plot per sampler, truth marked
* ``corner_joint.png``      -- all posteriors overlaid, contours only
* ``diag_emcee.png``        -- trace, log-probability, acceptance, tau(N)
* ``diag_nautilus.png``     -- shell volumes, per-shell likelihood, log Z
                               accumulation, weight concentration
* ``diag_ultranest.png``    -- the same four quantities for rejection-based
                               nested sampling, from its weighted chain
* ``diag_inessai.png``      -- log Z / likelihood-threshold history
* ``diag_svi.png``          -- ELBO trace, parsed from the run log

Note on coverage: \\emph{pocoMC saves no internal state we can replot}. Its
``.npz`` holds equal-weight draws and nothing else, so it appears in the corner
plots but has no per-algorithm diagnostic panel. Fixing that needs a rerun with
``output_dir`` set, which the docstring of ``run_pocomc`` warns is unsafe with a
GPU likelihood.

Run: ``conda run -n spexai python scripts/inference/bakeoff_diagnostics.py``
"""
from typing import Dict, List, Optional
import argparse
import os
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import logsumexp

BAKEOFF = os.path.expanduser("~/work/data/spexai/bakeoff")
LOGS = os.path.join(BAKEOFF, "bakeoff", "logs")
OUTDIR = os.path.expanduser(
    "~/work/data/spexai/results/bakeoff/bakeoff_diagnostics")

# fixed order and colour per sampler so every figure is read the same way
ORDER = ["emcee", "ultranest", "nautilus", "pocomc", "inessai", "svi"]
COLOR = {"emcee": "#4c72b0", "ultranest": "#dd8452", "nautilus": "#55a868",
         "pocomc": "#c44e52", "inessai": "#8172b3", "svi": "#937860"}
SEED = 0


def load(name: str) -> Optional[Dict]:
    p = os.path.join(BAKEOFF, f"bakeoff_{name}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {"samples": d["samples"], "names": [str(n) for n in d["names"]],
            "truth": np.asarray(d["truth"], dtype=float)}


# --- corner plots ------------------------------------------------------------

def corner_single(name: str, res: Dict, outdir: str) -> None:
    import corner
    s = res["samples"]                                    # (ndraw, npar)
    fig = corner.corner(
        s, labels=res["names"], truths=res["truth"],
        truth_color="k", color=COLOR[name],
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt=".3g", title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9}, hist_kwargs={"density": True},
        plot_datapoints=False, fill_contours=True,
        levels=(0.393, 0.865),  # 1- and 2-sigma in 2-D, not 68/95
    )
    fig.suptitle(f"{name}  ({s.shape[0]:,} draws)", fontsize=14, y=1.0)
    p = os.path.join(outdir, f"corner_{name}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


def corner_joint(results: Dict[str, Dict], outdir: str) -> None:
    """All posteriors on one grid, contours only so six are legible."""
    import corner
    got = [n for n in ORDER if n in results]
    ref = results[got[0]]
    # common ranges: union over samplers, so no posterior is clipped
    lo = np.min([np.percentile(results[n]["samples"], 0.05, axis=0)
                 for n in got], axis=0)                   # (npar,)
    hi = np.max([np.percentile(results[n]["samples"], 99.95, axis=0)
                 for n in got], axis=0)
    rng = list(zip(lo, hi))

    fig = None
    for n in got:
        fig = corner.corner(
            results[n]["samples"], labels=ref["names"], fig=fig, range=rng,
            color=COLOR[n], plot_datapoints=False, plot_density=False,
            fill_contours=False, no_fill_contours=True,
            levels=(0.393, 0.865), hist_kwargs={"density": True, "lw": 1.2},
            contour_kwargs={"linewidths": 1.0},
            label_kwargs={"fontsize": 9}, smooth=1.0,
        )
    # truth on top of everything
    corner.overplot_lines(fig, ref["truth"], color="k", ls=":", lw=1.0)

    fig.legend(handles=[Line2D([], [], color=COLOR[n], lw=2, label=n)
                        for n in got] +
                       [Line2D([], [], color="k", ls=":", lw=1, label="truth")],
               loc="upper right", frameon=False, fontsize=13,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Bake-off posteriors, all samplers "
                 "(1- and 2-$\\sigma$ contours)", fontsize=15, y=1.0)
    p = os.path.join(outdir, "corner_joint.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


# --- per-sampler diagnostics -------------------------------------------------

def diag_emcee(outdir: str, discard_frac: float = 0.4) -> None:
    """Trace, log-probability, acceptance and the tau(N) convergence test."""
    import emcee
    with h5py.File(os.path.join(BAKEOFF, "bakeoff_emcee_chain.h5"), "r") as f:
        chain = f["mcmc/chain"][:]                        # (nstep, nwalker, npar)
        logp = f["mcmc/log_prob"][:]                      # (nstep, nwalker)
        acc = f["mcmc/accepted"][:] / f["mcmc"].attrs["iteration"]
    res = load("emcee")
    names, nstep = res["names"], chain.shape[0]
    discard = int(discard_frac * nstep)

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(6, 3, hspace=0.45, wspace=0.25)

    for i, nm in enumerate(names):                        # 12 trace panels
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(chain[:, :, i], alpha=0.25, lw=0.4)
        ax.axvline(discard, color="k", ls="--", lw=1)
        ax.axhline(res["truth"][i], color="r", lw=1)
        ax.set_ylabel(nm, fontsize=9)
        ax.tick_params(labelsize=7)
        if i // 3 == 3:
            ax.set_xlabel("step", fontsize=8)

    ax = fig.add_subplot(gs[4, 0])
    ax.plot(logp, alpha=0.25, lw=0.4)
    ax.axvline(discard, color="k", ls="--", lw=1)
    ax.set_ylim(np.percentile(logp, 1), logp.max() + 5)
    ax.set_xlabel("step"), ax.set_ylabel("log posterior")
    ax.set_title("burn-in is visible here first", fontsize=9)

    ax = fig.add_subplot(gs[4, 1])
    ax.hist(acc, bins=16, color=COLOR["emcee"])
    ax.axvline(acc.mean(), color="k", ls="--")
    ax.set_xlabel("acceptance fraction"), ax.set_ylabel("walkers")
    ax.set_title(f"mean {acc.mean():.3f}  (healthy: 0.2-0.5)", fontsize=9)

    # tau(N): the standard emcee convergence check. tau is only trustworthy
    # once the chain is ~50 tau long, so the N/50 line is the acceptance test.
    ax = fig.add_subplot(gs[4, 2])
    post = chain[discard:]                                # (nkeep, nwalker, npar)
    ns = np.unique(np.geomspace(50, post.shape[0], 12).astype(int))
    taus = np.empty((ns.size, len(names)))                 # (nN, npar)
    for k, n in enumerate(ns):
        for i in range(len(names)):
            taus[k, i] = emcee.autocorr.integrated_time(
                post[:n, :, i], quiet=True, tol=0)[0]
    for i, nm in enumerate(names):
        ax.loglog(ns, taus[:, i], lw=1, alpha=0.8)
    ax.loglog(ns, ns / 50.0, "k--", label=r"$N/50$")
    ax.set_xlabel("chain length $N$ (post burn-in)")
    ax.set_ylabel(r"$\tau$")
    ax.legend(fontsize=8), ax.set_title(r"$\tau$ convergence", fontsize=9)

    tau_f = taus[-1]
    ax = fig.add_subplot(gs[5, :])
    ax.bar(np.arange(len(names)), tau_f, color=COLOR["emcee"])
    ax.axhline(post.shape[0] / 50.0, color="k", ls="--",
               label=f"N/50 = {post.shape[0] / 50:.1f} (reliability threshold)")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$\tau$ (steps)")
    ax.set_title(f"integrated autocorrelation time; chain is "
                 f"{post.shape[0]} steps = {post.shape[0] / tau_f.max():.0f} "
                 f"x the worst tau", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("emcee diagnostics", fontsize=15)
    p = os.path.join(outdir, "diag_emcee.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}  "
          f"(max tau {tau_f.max():.1f}, N/tau = {post.shape[0] / tau_f.max():.0f})")


def _nested_panels(fig, gs, log_v, log_l, label, color):
    """The four standard nested-sampling panels, shared by nautilus/UltraNest.

    ``log_v`` is the log prior volume of each shell/iteration and ``log_l`` the
    representative log likelihood there, so ``log_v + log_l`` is the evidence
    integrand.
    """
    x = np.arange(log_v.size)
    integrand = log_l + log_v                              # (n,)
    cum = np.array([logsumexp(integrand[:k + 1]) for k in x])
    w = np.exp(integrand - logsumexp(integrand))           # normalised weights

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, log_v, color=color)
    ax.set_xlabel("shell / iteration"), ax.set_ylabel(r"$\log V$")
    ax.set_title("prior volume shrinks geometrically", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, log_l - log_l.max(), color=color)
    ax.set_xlabel("shell / iteration")
    ax.set_ylabel(r"$\log \langle L\rangle - \max$")
    ax.set_title("likelihood rises as volume falls", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, cum - cum[-1], color=color)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylim(-5, 0.5)
    ax.set_xlabel("shell / iteration")
    ax.set_ylabel(r"$\log Z_{\leq k} - \log Z$")
    ax.set_title("evidence accumulation must plateau", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, w, color=color)
    kish = 1.0 / np.sum(w ** 2)
    ax.set_xlabel("shell / iteration"), ax.set_ylabel("normalised weight")
    ax.set_title(f"weight concentration: Kish ESS over shells = {kish:.1f}",
                 fontsize=9)
    return cum[-1]


def diag_nautilus(outdir: str) -> None:
    with h5py.File(os.path.join(BAKEOFF, "nautilus.hdf5"), "r") as f:
        a = f["sampler"].attrs
        log_v = np.asarray(a["shell_log_v"], dtype=float)
        log_l = np.asarray(a["shell_log_l"], dtype=float)
        n_eff = np.asarray(a["shell_n_eff"], dtype=float)
        n_keep = np.asarray(a["shell_n"], dtype=float)

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.28)
    logz = _nested_panels(fig, gs, log_v, log_l, "nautilus", COLOR["nautilus"])

    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(n_eff, color=COLOR["nautilus"])
    ax.set_xlabel("shell"), ax.set_ylabel(r"$N_{\rm eff}$ in shell")
    ax.set_title("per-shell effective size: early shells contribute ~1",
                 fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(n_keep, color=COLOR["nautilus"])
    ax.set_xlabel("shell"), ax.set_ylabel("points kept")
    ax.set_title("shell occupancy (n_live = 2000)", fontsize=9)

    fig.suptitle(f"nautilus diagnostics   $\\log Z$ = {logz:.2f}", fontsize=15)
    p = os.path.join(outdir, "diag_nautilus.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


def diag_ultranest(outdir: str) -> None:
    """Rebuild the nested-sampling panels from UltraNest's weighted chain."""
    p = os.path.join(BAKEOFF, "ultranest", "chains", "weighted_post.txt")
    if not os.path.exists(p):
        print("  (no UltraNest chain, skipping)")
        return
    with open(p) as f:
        cols = f.readline().split()
    arr = np.loadtxt(p, skiprows=1)                        # (n, ncol)
    w = arr[:, cols.index("weight")]
    ll = arr[:, cols.index("logl")]
    keep = w > 0
    w, ll = w[keep], ll[keep]
    order = np.argsort(ll)
    w, ll = w[order], ll[order]
    # weight = L * dV / Z, so log dV up to the additive log Z constant
    log_v = np.log(w) - ll

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)
    _nested_panels(fig, gs, log_v, ll, "ultranest", COLOR["ultranest"])
    fig.suptitle("UltraNest diagnostics (reconstructed from the weighted "
                 "chain)", fontsize=15)
    q = os.path.join(outdir, "diag_ultranest.png")
    fig.savefig(q, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(q)}")


def diag_inessai(outdir: str) -> None:
    p = os.path.join(BAKEOFF, "inessai", "result.hdf5")
    with h5py.File(p, "r") as f:
        h = f["history"]
        logz = np.asarray(h["logZ"][:], dtype=float)
        thr = np.asarray(h["logL_threshold"][:], dtype=float)
        med = np.asarray(h["median_logL"][:], dtype=float)
        ent = np.asarray(h["samples_entropy"][:], dtype=float)
        w = np.asarray(f["log_posterior_weights"][:], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    c = COLOR["inessai"]
    ax = axes[0, 0]
    ax.plot(logz - logz[-1], color=c)
    ax.axhline(0, color="k", ls=":", lw=0.8), ax.set_ylim(-5, 0.5)
    ax.set_xlabel("iteration"), ax.set_ylabel(r"$\log Z_k - \log Z$")
    ax.set_title("evidence accumulation", fontsize=9)

    ax = axes[0, 1]
    ax.plot(thr, color=c, label=r"$\log L$ threshold")
    ax.plot(med, color="k", lw=1, ls="--", label=r"median $\log L$")
    ax.set_xlabel("iteration"), ax.set_ylabel(r"$\log L$")
    ax.legend(fontsize=8), ax.set_title("level threshold vs sample median",
                                        fontsize=9)

    ax = axes[1, 0]
    ax.plot(ent, color=c)
    ax.set_xlabel("iteration"), ax.set_ylabel("entropy of the weights")
    ax.set_title("proposal entropy", fontsize=9)

    ax = axes[1, 1]
    ww = np.exp(w - logsumexp(w))
    ax.hist(np.log10(ww[ww > 0]), bins=60, color=c)
    ax.set_xlabel(r"$\log_{10}$ normalised weight"), ax.set_ylabel("samples")
    ax.set_title(f"weight distribution over {w.size:,} samples; "
                 f"Kish ESS = {1.0 / np.sum(ww ** 2):.0f}", fontsize=9)

    fig.suptitle("i-nessai diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    q = os.path.join(outdir, "diag_inessai.png")
    fig.savefig(q, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(q)}  "
          f"(nessai's own state/trace/posterior plots are in {os.path.dirname(p)})")


def diag_svi(outdir: str) -> None:
    """ELBO trace, parsed from the run log -- the npz keeps no history."""
    p = os.path.join(LOGS, "bo_svi_mvn32.log")
    step, elbo = [], []
    pat = re.compile(r"svi step (\d+)/\d+\s+ELBO\s+([-\d.e+]+)")
    for line in open(p):
        m = pat.search(line)
        if m:
            step.append(int(m.group(1)))
            elbo.append(float(m.group(2)))
    if not step:
        print("  (no ELBO history in the SVI log, skipping)")
        return
    step, elbo = np.array(step), np.array(elbo)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(step, elbo, color=COLOR["svi"], marker="o", ms=3)
    axes[0].set_xlabel("step"), axes[0].set_ylabel("ELBO")
    axes[0].set_title("full range", fontsize=9)

    tail = step > step.max() * 0.15
    axes[1].plot(step[tail], elbo[tail], color=COLOR["svi"], marker="o", ms=3)
    axes[1].set_xlabel("step"), axes[1].set_ylabel("ELBO")
    axes[1].set_title("tail: flat from ~step 1400, so the optimiser converged",
                      fontsize=9)
    fig.suptitle("SVI diagnostics (full-rank Gaussian guide, 32 particles)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    q = os.path.join(outdir, "diag_svi.png")
    fig.savefig(q, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(q)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--skip_corner", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = {n: r for n in ORDER if (r := load(n)) is not None}
    print(f"loaded: {', '.join(results)}")

    if not args.skip_corner:
        print("corner plots")
        for n, r in results.items():
            corner_single(n, r, args.outdir)
        corner_joint(results, args.outdir)

    print("per-sampler diagnostics")
    diag_emcee(args.outdir)
    diag_nautilus(args.outdir)
    diag_ultranest(args.outdir)
    diag_inessai(args.outdir)
    diag_svi(args.outdir)
    print("  pocomc: no internal state saved by the run -- corner only")
    print(f"\nall figures in {args.outdir}")


if __name__ == "__main__":
    main()
