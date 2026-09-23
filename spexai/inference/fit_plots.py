"""Diagnostic, corner, and posterior-predictive plots for any sampler result.

These used to take ``fitting.EmceeResult`` and ``fitting.UltranestResult``,
which is part of why those two classes could not be collapsed into the one
:class:`~spexai.inference.samplers.SamplerResult` the bake-off already used.
They now take ``SamplerResult``, so they work for all nine samplers rather than
two.

One deliberate signature change: **truths are passed in, not read off the
result**. A posterior does not know the true answer -- only a simulation study
does -- so making the result carry ``truths`` forced every real fit to invent a
value. The studies that have truths now hand them over explicitly, and a fit to
real data simply omits them.
"""
import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

C_MCMC, C_NS, INK = "#0072B2", "#D55E00", "#222222"   # blue / vermillion (CVD-safe)


def _truth_array(truths, n):
    """``truths`` as a float array of length ``n``, NaN where unknown."""
    if truths is None:
        return np.full(n, np.nan)
    return np.array([np.nan if t is None else float(t) for t in truths])


def _predict(model, obs, names, theta, fixed, abundance_model=None):
    """Counts for one parameter vector, under either naming convention.

    ``kT``/``sigma_v`` are canonical; ``temp``/``velocity`` are the retired
    spelling and still appear in saved results and older scripts.
    """
    p = dict(zip(names, theta))
    temp = p.get("kT", p.get("temp"))
    if temp is None:
        raise KeyError(f"no temperature parameter in {list(p)}; expected "
                       f"'kT' (or the legacy 'temp')")
    vel = p.get("sigma_v", p.get("velocity", fixed.get("velocity", 0.0)))
    abund = ({**fixed.get("abundances", {}), **abundance_model.to_abundances(p)}
             if abundance_model is not None else fixed.get("abundances", {}))
    return model.predict_counts(
        torch.tensor([float(temp)]), abund,
        float(fixed.get("logz", -10.0)), 10.0 ** float(p["log_norm"]),
        float(vel), obs.response, obs.exposure).squeeze(0).cpu().numpy()


def plot_emcee_trace(res, outpath, truths=None):
    """Per-parameter trace + autocorrelation time, for any ensemble sampler."""
    if res.chain is None:
        raise ValueError(f"{res.name} produced no chain to trace; "
                         f"traces are for ensemble samplers")
    ndim = len(res.names)
    tr = _truth_array(truths, ndim)
    tau = res.tau if res.tau is not None else np.full(ndim, np.nan)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 2.2 * ndim), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(res.chain[:, :, i], color="k", alpha=0.25, lw=0.5)
        ax.axvline(res.discard, color="#888", ls=":", lw=1)          # burn-in
        if np.isfinite(tr[i]):
            ax.axhline(tr[i], color=C_NS, lw=1.5)                    # truth
        ax.set_ylabel(res.labels[i], color=INK)
        ax.text(0.99, 0.06, (rf"$\tau\approx{tau[i]:.0f}$ steps"
                             if np.isfinite(tau[i]) else r"$\tau$: n/a"),
                transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    np.atleast_1d(axes)[-1].set_xlabel("step")
    np.atleast_1d(axes)[0].set_title(
        f"{res.name} traces — burn-in={res.discard}, "
        f"{res.samples.shape[0]} post-burn-in samples "
        f"(dotted = burn-in, orange = truth)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ultranest_diagnostics(res, outpath):
    """UltraNest's own trace plot + evidence/ESS summary."""
    import ultranest.plot as up
    raw = res.extra.get("result")
    if raw is None:
        raise ValueError("no UltraNest result payload on this SamplerResult; "
                         "this plot is specific to run_ultranest")
    try:
        up.traceplot(raw, labels=list(res.labels))
        fig = plt.gcf()
        fig.suptitle(f"UltraNest trace — ln Z = {res.logz:.2f} ± "
                     f"{res.logzerr:.2f}, ESS = {res.min_ess:.0f}, "
                     f"{res.n_eval:,} likelihood calls", fontsize=11, y=1.02)
    except Exception as exc:                                  # pragma: no cover
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"traceplot failed:\n{exc}", ha="center", va="center")
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_corner_overlay(a, b, outpath, truths=None,
                        labels=("emcee (MCMC)", "UltraNest (NS)")):
    """Corner plot with two posteriors overlaid.

    Any two ``SamplerResult``s over the same parameters -- the pair is no
    longer hardwired to emcee and UltraNest, only its default legend is.
    """
    tr = _truth_array(truths, len(a.names))
    truth_list = [t if np.isfinite(t) else None for t in tr]
    rng = [(min(a.samples[:, i].min(), b.samples[:, i].min()),
            max(a.samples[:, i].max(), b.samples[:, i].max()))
           for i in range(len(a.names))]
    ckw = dict(labels=list(a.labels), range=rng, plot_datapoints=False,
               plot_density=False, fill_contours=False, levels=(0.393, 0.865),
               hist_kwargs=dict(density=True, alpha=0.85))
    fig = corner.corner(a.samples, color=C_MCMC,
                        contour_kwargs=dict(alpha=0.9), **ckw)
    corner.corner(b.samples, fig=fig, color=C_NS, truths=truth_list,
                  truth_color="k", contour_kwargs=dict(alpha=0.9), **ckw)
    fig.legend([Line2D([0], [0], color=C_MCMC), Line2D([0], [0], color=C_NS),
                Line2D([0], [0], color="k", ls="--")],
               [*labels, "truth"],
               loc="upper right", frameon=False, fontsize=12)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_predictive(obs, model, a, b, fixed, outpath,
                              ndraw=20, seed=0, abundance_model=None,
                              labels=("emcee (MCMC)", "UltraNest (NS)")):
    """Two panels; each has a spectrum subplot (data, posterior median,
    ``ndraw`` posterior draws) over a residual subplot."""
    e = obs.response.chan_e_cent.numpy()
    width = (obs.response.chan_e_max - obs.response.chan_e_min).numpy()
    band = (obs.expected > 0.02 * obs.expected.max() if obs.expected is not None
            else obs.counts > 0)
    rng = np.random.default_rng(seed)

    panels = [(res, label, color) for res, label, color in
              [(a, labels[0], C_MCMC), (b, labels[1], C_NS)] if res is not None]
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

    # a real observation has no true_params; only the simulation studies do
    tp = getattr(obs, "true_params", None) or {}
    truth_bit = ""
    if tp:
        t = tp.get("kT", tp.get("temp"))
        v = tp.get("sigma_v", tp.get("velocity", 0))
        if t is not None:
            truth_bit = f" (true $T$ = {t} keV, $v$ = {v} km/s)"
    fig.suptitle(f"{obs.instrument} — posterior predictive{truth_bit}",
                 fontsize=12)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
