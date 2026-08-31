"""Diagnostic, corner, and posterior-predictive plots for the emcee and
UltraNest fits produced by `spexai.inference.fitting`."""
import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

C_MCMC, C_NS, INK = "#0072B2", "#D55E00", "#222222"   # blue / vermillion (CVD-safe)


def _predict(model, obs, names, theta, fixed, abundance_model=None):
    p = dict(zip(names, theta))
    vel = p.get("velocity", fixed.get("velocity", 0.0))
    abund = ({**fixed.get("abundances", {}), **abundance_model.to_abundances(p)}
             if abundance_model is not None else fixed.get("abundances", {}))
    return model.predict_counts(
        torch.tensor([float(p["temp"])]), abund,
        float(fixed.get("logz", -10.0)), 10.0 ** float(p["log_norm"]),
        float(vel), obs.response, obs.exposure).squeeze(0).cpu().numpy()


def plot_emcee_trace(er, outpath):
    """MCMC trace per parameter + autocorrelation time."""
    ndim = len(er.names)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 2.2 * ndim), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(er.chain[:, :, i], color="k", alpha=0.25, lw=0.5)
        ax.axvline(er.discard, color="#888", ls=":", lw=1)          # burn-in
        if np.isfinite(er.truths[i]):
            ax.axhline(er.truths[i], color=C_NS, lw=1.5)            # truth
        ax.set_ylabel(er.labels[i], color=INK)
        tau = er.tau[i]
        ax.text(0.99, 0.06, (rf"$\tau\approx{tau:.0f}$ steps"
                             if np.isfinite(tau) else r"$\tau$: n/a"),
                transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    np.atleast_1d(axes)[-1].set_xlabel("step")
    np.atleast_1d(axes)[0].set_title(
        f"emcee traces — burn-in={er.discard}, "
        f"{er.samples.shape[0]} post-burn-in samples "
        f"(dotted = burn-in, orange = truth)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ultranest_diagnostics(ur, outpath):
    """UltraNest's trace (the NS analogue of the MCMC trace) + evidence/ESS."""
    import ultranest.plot as up
    try:
        up.traceplot(ur.result, labels=ur.labels)
        fig = plt.gcf()
        fig.suptitle(f"UltraNest trace — ln Z = {ur.logz:.2f} ± {ur.logzerr:.2f}, "
                     f"ESS = {ur.ess:.0f}, {ur.n_eval:,} likelihood calls",
                     fontsize=11, y=1.02)
    except Exception as exc:                                  # pragma: no cover
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"traceplot failed:\n{exc}", ha="center", va="center")
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_corner_overlay(er, ur, outpath):
    """Corner plot with emcee and UltraNest posteriors overlaid."""
    truths = [t if np.isfinite(t) else None for t in er.truths]
    rng = [(min(er.samples[:, i].min(), ur.samples[:, i].min()),
            max(er.samples[:, i].max(), ur.samples[:, i].max()))
           for i in range(len(er.names))]
    ckw = dict(labels=er.labels, range=rng, plot_datapoints=False,
               plot_density=False, fill_contours=False, levels=(0.393, 0.865),
               hist_kwargs=dict(density=True, alpha=0.85))
    fig = corner.corner(er.samples, color=C_MCMC,
                        contour_kwargs=dict(alpha=0.9), **ckw)
    corner.corner(ur.samples, fig=fig, color=C_NS, truths=truths,
                  truth_color="k", contour_kwargs=dict(alpha=0.9), **ckw)
    fig.legend([Line2D([0], [0], color=C_MCMC), Line2D([0], [0], color=C_NS),
                Line2D([0], [0], color="k", ls="--")],
               ["emcee (MCMC)", "UltraNest (NS)", "truth"],
               loc="upper right", frameon=False, fontsize=12)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_predictive(obs, model, er, ur, fixed, outpath,
                              ndraw=20, seed=0, abundance_model=None):
    """Two panels (MCMC | UltraNest); each has a spectrum subplot (data,
    posterior median, `ndraw` posterior draws) over a residual subplot."""
    e = obs.response.chan_e_cent.numpy()
    width = (obs.response.chan_e_max - obs.response.chan_e_min).numpy()
    band = (obs.expected > 0.02 * obs.expected.max() if obs.expected is not None
            else obs.counts > 0)
    rng = np.random.default_rng(seed)

    panels = [(res, label, color) for res, label, color in
              [(er, "emcee (MCMC)", C_MCMC), (ur, "UltraNest (NS)", C_NS)]
              if res is not None]
    fig = plt.figure(figsize=(7 * len(panels), 6))
    gs = GridSpec(2, len(panels), height_ratios=[3, 1], hspace=0.05, wspace=0.16)
    for col, (res, label, color) in enumerate(panels):
        med = _predict(model, obs, res.names, res.median, fixed, abundance_model)
        idx = rng.choice(len(res.samples),
                         size=min(ndraw, len(res.samples)), replace=False)
        draws = [_predict(model, obs, res.names, res.samples[k], fixed,
                          abundance_model)
                 for k in idx]

        ax0 = fig.add_subplot(gs[0, col])
        ax1 = fig.add_subplot(gs[1, col], sharex=ax0)
        ax0.step(e, np.where(obs.counts > 0, obs.counts / width, np.nan),
                 where="mid", color="#444", lw=1.0, label="data", zorder=3)
        for d in draws:
            ax0.plot(e, d / width, color=color, alpha=0.15, lw=0.7)
        ax0.plot(e, med / width, color=color, lw=1.6,
                 label="posterior median", zorder=4)
        ax0.plot([], [], color=color, alpha=0.4, lw=1.0,
                 label=f"{len(idx)} posterior draws")
        ax0.set(xscale="log", yscale="log", title=label)
        ax0.set_ylabel("counts keV$^{-1}$", color=INK)
        ax0.legend(loc="upper right", frameon=False, fontsize=9)
        peak = np.nanmax(med / width)
        ax0.set_ylim(peak / 3e3, peak * 4)

        resid = (obs.counts - med) / np.sqrt(np.clip(med, 1e-30, None))
        ax1.axhline(0, color="#888", lw=0.8)
        ax1.step(e, np.where(band, resid, np.nan), where="mid",
                 color=color, lw=0.8)
        ax1.set(xscale="log", ylim=(-5, 5))
        ax1.set_ylabel(r"$(d-m)/\sqrt{m}$", color=INK)
        ax1.set_xlabel("Energy (keV)")
        if band.any():
            ax0.set_xlim(max(0.2, e[band].min() * 0.9), e[band].max() * 1.1)
        plt.setp(ax0.get_xticklabels(), visible=False)

    fig.suptitle(f"{obs.instrument} — posterior predictive "
                 f"(true $T$ = {obs.true_params['temp']} keV, "
                 f"$v$ = {obs.true_params.get('velocity', 0)} km/s)",
                 fontsize=12)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
