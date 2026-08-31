"""Convergence diagnostics for a hot-floor MCMC run: trace plots, integrated
autocorrelation time, a corner plot, and (when the run saved them) a
posterior-predictive check.

The vectorized driver saves the FLAT post-burn chain; the per-walker structure
is recovered by reshaping with the walker count (--nwalkers), which is enough
for trace plots and the autocorrelation time on the post-burn samples. Runs made
after the mcmc_check save-fix also store ``obs_data``/``chan_e``/``ppc`` and this
script draws the posterior-predictive panel from them.

  python scripts/experiments/hot_floor/plot_mcmc.py <mcmc_*_c*.npz> --nwalkers 64
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, nwalkers):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    names = [str(s) for s in d["names"]]
    truth = np.asarray(d["truth"], dtype=float)
    ndim = len(names)
    if "full_chain" in d:                                  # self-contained save
        full = np.asarray(d["full_chain"])                 # (nsteps, nwalkers, ndim)
        flat = np.asarray(d.get("chain", full.reshape(-1, ndim)))
    else:                                                  # old: flat post-burn only
        flat = np.asarray(d["chain"])                      # (N, ndim)
        nw = int(d["nwalkers"]) if "nwalkers" in d else nwalkers
        if flat.shape[0] % nw:
            raise SystemExit(f"flat chain {flat.shape[0]} not divisible by "
                             f"nwalkers={nw}; pass the right --nwalkers")
        full = flat.reshape(-1, nw, ndim)                  # (nsteps_post, nw, ndim)
    return d, names, truth, full, flat


def trace_plot(names, truth, full, out):
    nsteps, nw, ndim = full.shape
    fig, ax = plt.subplots(ndim, 1, figsize=(9, 1.4 * ndim), sharex=True)
    for i in range(ndim):
        ax[i].plot(full[:, :, i], color="k", alpha=0.12, lw=0.4)
        ax[i].axhline(truth[i], color="C3", lw=1.3)
        ax[i].set_ylabel(names[i], fontsize=9)
    ax[-1].set_xlabel("step (post-burn-in)")
    ax[0].set_title(f"traces: {nsteps} steps x {nw} walkers (truth in red)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def autocorr_report(names, full):
    import emcee
    nsteps, nw, ndim = full.shape
    try:
        tau = emcee.autocorr.integrated_time(full, quiet=True)
    except Exception as e:
        print("autocorr failed:", e)
        tau = np.full(ndim, np.nan)
    print(f"\nautocorrelation time (post-burn: {nsteps} steps x {nw} walkers)")
    print(f"{'param':>9} {'tau':>8} {'steps/tau':>10} {'ESS':>9}  flag")
    for i, n in enumerate(names):
        r = nsteps / tau[i] if tau[i] > 0 else np.nan
        ess = nsteps * nw / tau[i] if tau[i] > 0 else np.nan
        flag = "" if r >= 50 else "  <- steps<50*tau (chain short for this tau)"
        print(f"{n:>9} {tau[i]:>8.1f} {r:>10.1f} {ess:>9.0f}{flag}")
    return tau


def corner_plot(names, truth, flat, out):
    try:
        import corner
        fig = corner.corner(flat, labels=names, truths=truth,
                            truth_color="C3", show_titles=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return
    except Exception:
        pass                                               # dependency-free fallback
    ndim = flat.shape[1]
    q = np.percentile(flat, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(ndim, ndim, figsize=(1.5 * ndim, 1.5 * ndim))
    for i in range(ndim):
        for j in range(ndim):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            if i == j:
                a.hist(flat[:, i], bins=40, color="0.5")
                a.axvline(truth[i], color="C3", lw=1.0)
                a.set_title(f"{names[i]}={q[1,i]:.3g}\n"
                            f"(-{q[1,i]-q[0,i]:.2g}/+{q[2,i]-q[1,i]:.2g})",
                            fontsize=6)
            else:
                a.hist2d(flat[:, j], flat[:, i], bins=40, cmap="Greys")
                a.plot(truth[j], truth[i], "s", color="C3", ms=3)
            if i == ndim - 1:
                a.set_xlabel(names[j], fontsize=7)
            if j == 0 and i > 0:
                a.set_ylabel(names[i], fontsize=7)
            a.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def ppc_plot(d, out):
    if not all(k in d for k in ("obs_data", "ppc", "chan_e")):
        return False
    e = np.asarray(d["chan_e"])
    obs = np.asarray(d["obs_data"])
    ppc = np.asarray(d["ppc"])                              # (Nsamp, n_chan)
    lo, mid, hi = np.percentile(ppc, [16, 50, 84], axis=0)
    fig, (a, r) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    a.plot(e, obs, drawstyle="steps-mid", color="k", lw=0.6, label="data")
    a.fill_between(e, lo, hi, step="mid", color="C0", alpha=0.4,
                   label="posterior 16-84%")
    a.plot(e, mid, drawstyle="steps-mid", color="C0", lw=0.8)
    a.set_yscale("log"); a.set_ylabel("counts / bin"); a.legend(fontsize=8)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (obs - mid) / np.sqrt(np.clip(mid, 1e-9, None))
    r.plot(e, pull, drawstyle="steps-mid", color="0.3", lw=0.5)
    r.axhline(0, color="C3", lw=0.8)
    r.set_ylabel("(d-m)/sqrt(m)"); r.set_xlabel("energy (keV)"); r.set_ylim(-5, 5)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--nwalkers", type=int, default=64,
                    help="walker count used for the run (to un-flatten the chain)")
    ap.add_argument("--out", default=None, help="output prefix (default: next to npz)")
    args = ap.parse_args()
    prefix = args.out or os.path.splitext(args.npz)[0]

    d, names, truth, full, flat = load(args.npz, args.nwalkers)
    counts = float(d["counts"]) if "counts" in d else float("nan")
    print(f"{os.path.basename(args.npz)}: {counts:.1e} counts, "
          f"post-burn chain {full.shape} (steps x walkers x dim), "
          f"{flat.shape[0]} flat samples")

    trace_plot(names, truth, full, prefix + "_trace.png")
    corner_plot(names, truth, flat, prefix + "_corner.png")
    tau = autocorr_report(names, full)
    made_ppc = ppc_plot(d, prefix + "_ppc.png")

    # posterior summary vs truth
    q = np.percentile(flat, [16, 50, 84], axis=0)
    print(f"\n{'param':>9} {'truth':>10} {'median':>10} {'-16%':>8} {'+84%':>8}"
          f"  truth in 68%?")
    for i, n in enumerate(names):
        inside = q[0, i] <= truth[i] <= q[2, i]
        print(f"{n:>9} {truth[i]:>10.4g} {q[1,i]:>10.4g} {q[1,i]-q[0,i]:>8.3g} "
              f"{q[2,i]-q[1,i]:>8.3g}  {'yes' if inside else 'NO'}")

    outs = [prefix + s for s in ("_trace.png", "_corner.png")]
    if made_ppc:
        outs.append(prefix + "_ppc.png")
    print("\nwrote:", *[os.path.basename(o) for o in outs])
    if not made_ppc:
        print("(no posterior-predictive: this run didn't save obs_data/ppc; "
              "re-run with the updated mcmc_check to get it)")


if __name__ == "__main__":
    main()
