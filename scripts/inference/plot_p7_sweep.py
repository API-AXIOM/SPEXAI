"""Figures for the P7 bias sweep (1000 points, single-T and DEM log-T).

Four figures, all reading the two ``bias_sweep --stage bias`` jsonls and
recomputing nothing:

A ``p7_safety_map``            worst bias/sigma per point against temperature
B ``p7_bias_by_param``         the full per-parameter distributions
C ``p7_nstar_vs_temperature``  the same result as an exposure limit
D ``p7_mn_outliers``           the single-T Mn pair, in abundance space

The story these have to carry is two-sided: the bulk of the design is safe by a
wide margin, AND there are two distinct failure modes (a cold ``sigma_v`` tail
in both flavours, an abundance-driven Mn pair in single-T only). A figure that
shows only the reassuring half is the wrong figure.

**Colour is assigned by role, capped at three hues.** Scatter forms put every
pair of series on screen at once, and the documented palette validates
its first three slots under that condition; past three, series fold
into a muted "Other".
The three are the three that actually bind (sigma_v 276/308 points, Fe 271/204,
thermal 272/139 -- nothing else clears 100). Colour follows the parameter and
never its rank, so it is identical across all four figures and both flavours.

These are print figures for the paper: light surface only, no hover layer, and
identity is never carried by colour alone (every series is also direct-labelled
or annotated).

**Bias is shown RAW, not k-corrected.** P6 measured k = 0.99 +- 0.11 (DEM) /
0.20 (single-T) with a worst case near 1.5x, so the correction is a
small multiplicative adjustment that belongs in the text and, for N*, in the
annotated arrow on figure C -- not silently folded into the points.

    python scripts/inference/plot_p7_sweep.py --fig all
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LogNorm, LinearSegmentedColormap   # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

# --- palette: reference instance, first three categorical slots + chrome -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"          # status, reserved: "exceeds 1 sigma"
ROLE_COLOR = {"sigma_v": "#2a78d6", "Fe": "#eb6834", "thermal": "#1baf7a",
              "other": MUTED}
# sequential blue ramp, steps 100->700 (magnitude encoding, figure D)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
THERMAL = {"kT", "logT_mean", "logT_sigma"}
BAND_LO_KEV = 1.8712          # rest-frame band edge; see band_restriction
K_WORST = 1.54                # P6's worst case (single-T point 11)
X_LO = 1e-3                   # figure B x floor; off-scale points are counted
N_REF_LABEL = 1e6

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def role_of(name):
    """Parameter -> colour role. Identity, not rank: fixed across figures."""
    if name == "sigma_v":
        return "sigma_v"
    if name == "Fe":
        return "Fe"
    if name in THERMAL:
        return "thermal"
    return "other"


class Sweep:
    """One flavour's jsonl, with the derived quantities the figures share."""

    def __init__(self, path, target_counts=N_REF_LABEL):
        recs = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[int(r["point"])] = r   # last wins: append dupes
        self.pts = sorted(recs)
        self.recs = [recs[i] for i in self.pts]
        self.names = self.recs[0]["names"]
        self.mode = "dem" if "logT_mean" in self.names else "single"
        b = np.array([r["b_sys"] for r in self.recs])           # (P, n)
        sig = np.array([r["sigma_ref"] for r in self.recs])     # (P, n)
        n_ref = self.recs[0]["n_ref"]
        # sigma ~ 1/sqrt(N); b_sys is count-independent, so the ratio scales
        self.ratio = np.abs(b) / (sig * np.sqrt(n_ref / target_counts))
        self.nstar = np.where(b != 0, n_ref * (sig / np.abs(b)) ** 2, np.inf)
        self.kT = (10.0 ** np.array([r["params"]["logT_mean"]
                                     for r in self.recs])
                   if self.mode == "dem"
                   else np.array([r["params"]["kT"] for r in self.recs]))
        self.worst = self.ratio.max(axis=1)                     # (P,)
        self.binding = self.ratio.argmax(axis=1)                # (P,)
        self.min_nstar = self.nstar.min(axis=1)
        self.binding_nstar = self.nstar.argmin(axis=1)

    @property
    def title(self):
        return ("DEM (Gaussian in $\\log T$)" if self.mode == "dem"
                else "single temperature")

    @property
    def tlabel(self):
        return (r"$\langle kT \rangle$ [keV]" if self.mode == "dem"
                else r"$kT$ [keV]")


def _tidy(ax, logx=True, logy=True):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def _band_edge(ax, y=0.985, va="top"):
    """The 1.87 keV band edge, labelled ALONG the line so it never lands on a
    point: below it a plasma is in band only through its exponential tail."""
    ax.axvline(BAND_LO_KEV, color=AXIS, lw=1.0)
    ax.text(BAND_LO_KEV * 1.05, y, "band edge 1.87 keV",
            transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7,
            rotation=90, va=va, ha="left")


def _role_legend(ax, sw, loc="upper right"):
    """Legend by ROLE + binding count: identity is never colour-alone."""
    counts = {r: 0 for r in ROLE_COLOR}
    for j in sw.binding:
        counts[role_of(sw.names[j])] += 1
    thermal_name = "kT" if sw.mode == "single" else r"$\log T$ shape"
    labels = {"sigma_v": r"$\sigma_v$", "Fe": "Fe", "thermal": thermal_name,
              "other": "other"}
    handles = [Line2D([], [], marker="o", ls="", markersize=5,
                      markerfacecolor=ROLE_COLOR[r], markeredgecolor=SURFACE,
                      markeredgewidth=0.5,
                      label=f"{labels[r]}  ({counts[r]})")
               for r in ("sigma_v", "Fe", "thermal", "other") if counts[r]]
    leg = ax.legend(handles=handles, loc=loc, title="binding parameter",
                    handletextpad=0.4, borderaxespad=0.6, frameon=True,
                    facecolor=SURFACE, edgecolor="none", framealpha=0.88)
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_2)


# --- A: the safety map ------------------------------------------------------

def fig_safety_map(sweeps, out):
    """Worst bias/sigma per point against temperature, coloured by who binds.

    One dot per sweep point, not per point-parameter: the question a reader has
    is "is THIS cluster safe", and that is set by whichever parameter is worst
    there. The 1 sigma line is the whole decision boundary.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, sw in zip(axes, sweeps):
        colors = [ROLE_COLOR[role_of(sw.names[j])] for j in sw.binding]
        ax.scatter(sw.kT, sw.worst, s=17, c=colors, alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.4, zorder=3)
        ax.axhline(1.0, color=INK_2, lw=1.0, ls=":", zorder=2)
        _tidy(ax)
        ax.set_xlabel(sw.tlabel)
        ax.set_title(sw.title, color=INK, pad=8)
        n_over = int((sw.worst > 1).sum())
        ax.text(0.98, 0.02, f"{n_over} of {len(sw.worst)} points above "
                            f"1$\\sigma$", transform=ax.transAxes,
                ha="right", va="bottom", color=INK_2, fontsize=8)
        _band_edge(ax)
        _role_legend(ax, sw, loc="lower left")
    axes[0].set_ylabel(f"worst $|b_{{\\rm sys}}|/\\sigma$ "
                       f"at $10^{{{int(np.log10(N_REF_LABEL))}}}$ counts")
    axes[0].text(0.985, 1.12, "bias = noise", transform=
                 axes[0].get_yaxis_transform(), color=INK_2, fontsize=8,
                 ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- B: per-parameter distributions -----------------------------------------

def fig_by_param(sweeps, out, seed=7):
    """Every point of every parameter, with the >1 sigma population called out.

    A strip rather than a box: the two Mn points are the finding, and a box
    would render them as anonymous flier dots or hide them entirely. Jitter is
    seeded so the figure is reproducible.

    The x limit is cut at ``X_LO`` because the distributions run five decades
    and the bottom two are empty space; the number of points that fall off the
    scale is PRINTED per panel rather than quietly dropped.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    x_hi = max(sw.ratio.max() for sw in sweeps) * 3.0
    for ax, sw in zip(axes, sweeps):
        n = len(sw.names)
        ax.set_xscale("log")
        ax.set_xlim(X_LO, x_hi)
        off = 0
        for i, nm in enumerate(sw.names):
            y = n - 1 - i
            v = sw.ratio[:, i]
            off += int((v < X_LO).sum())
            jit = rng.uniform(-0.17, 0.17, size=v.size)
            ok = v <= 1.0
            ax.scatter(v[ok], y + jit[ok], s=3.5,
                       c=ROLE_COLOR[role_of(nm)], alpha=0.22,
                       linewidths=0, zorder=2)
            ax.scatter(v[~ok], y + jit[~ok], s=13, c=CRITICAL, alpha=0.9,
                       edgecolors=SURFACE, linewidths=0.4, zorder=4)
            ax.plot([np.median(v)], [y], marker="|", markersize=13,
                    color=INK, markeredgewidth=1.6, zorder=5)
            if (~ok).sum():
                # anchored to the row's own worst point, so counts cannot
                # stack on top of each other at a shared right margin
                ax.text(v.max() * 1.35, y, f"{(~ok).sum()}", color=CRITICAL,
                        fontsize=7.5, ha="left", va="center")
        ax.axvline(1.0, color=INK_2, lw=1.0, ls=":", zorder=3)
        ax.set_yticks(range(n))
        ax.set_yticklabels([_pretty(x) for x in sw.names][::-1])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel(f"$|b_{{\\rm sys}}|/\\sigma$ at "
                      f"$10^{{{int(np.log10(N_REF_LABEL))}}}$ counts")
        ax.set_title(sw.title, color=INK, pad=8)
        _tidy(ax, logy=False)
        ax.grid(False, axis="y")
        if off:
            ax.text(0.01, 0.01, f"{off} of {sw.ratio.size} point-parameters "
                                f"below the axis", transform=ax.transAxes,
                    color=MUTED, fontsize=7, ha="left", va="bottom")
    # annotate the two Mn points, which is the single-T-only failure mode
    sw = sweeps[0]
    j = sw.names.index("Mn")
    y = len(sw.names) - 1 - j
    for k, dy in zip(np.argsort(-sw.ratio[:, j])[:2], (34, 16)):
        axes[0].annotate(f"point {sw.pts[k]}: {sw.ratio[k, j]:.1f}$\\sigma$",
                         xy=(sw.ratio[k, j], y), xytext=(-6, dy),
                         textcoords="offset points", color=CRITICAL,
                         fontsize=7, ha="right",
                         arrowprops=dict(arrowstyle="-", color=CRITICAL,
                                         lw=0.8))
    handles = [Line2D([], [], marker="o", ls="", markersize=5,
                      markerfacecolor=CRITICAL, markeredgecolor=SURFACE,
                      label="above 1$\\sigma$ (count at right)"),
               Line2D([], [], marker="|", ls="", markersize=9, color=INK,
                      markeredgewidth=1.6, label="median")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _pretty(name):
    return {"sigma_v": r"$\sigma_v$", "n_h": r"$n_{\rm H}$",
            "log_norm": r"$\log\,$norm", "kT": r"$kT$",
            "logT_mean": r"$\log T_{\rm mean}$",
            "logT_sigma": r"$\log T_{\sigma}$"}.get(name, name)


# --- C: the same result as an exposure limit --------------------------------

def fig_nstar(sweeps, out):
    """N* -- the counts at which the first parameter goes bias-dominated.

    The science-facing form of figure A: sigma shrinks as 1/sqrt(N) while
    b_sys does not, so every point has an exposure beyond which the emulator
    floor dominates. Lower is worse.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, sw in zip(axes, sweeps):
        colors = [ROLE_COLOR[role_of(sw.names[j])] for j in sw.binding_nstar]
        ax.scatter(sw.kT, sw.min_nstar, s=17, c=colors, alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.4, zorder=3)
        _tidy(ax)
        ax.set_xlabel(sw.tlabel)
        ax.set_title(sw.title, color=INK, pad=8)
        _band_edge(ax, y=0.02, va="bottom")
        _role_legend(ax, sw, loc="lower right")
    axes[0].set_ylabel(r"$N_\star$ [in-band counts]")
    # the k correction, shown rather than silently applied
    ax = axes[0]
    x0, y1 = ax.get_xlim()[0] * 1.25, ax.get_ylim()[1]
    ax.annotate("", xy=(x0, y1 / (2.6 * K_WORST ** 2)), xytext=(x0, y1 / 2.6),
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2))
    ax.text(x0 * 1.16, y1 / (2.6 * K_WORST),
            f"worst-case $k={K_WORST}$\n$N_\\star \\div {K_WORST ** 2:.1f}$",
            color=INK_2, fontsize=7, va="center", ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- D: the Mn mode ---------------------------------------------------------

def fig_mn(sweep, out, nbin=5):
    """Mn's bias in abundance space: binned medians + the two outliers.

    A per-point scatter was tried first and is the wrong form here -- 1000 dots
    of nearly the same blue read as noise, and the ordering (Spearman +0.20 on
    a_Mn, -0.12 on a_Fe) is too gentle to see one point at a time. Binning to
    medians shows the gradient the correlation measures; the two outliers are
    then drawn ON TOP as themselves, because a median map would otherwise hide
    exactly the points the figure exists to show.
    """
    from scipy.stats import spearmanr
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    j = sweep.names.index("Mn")
    v = sweep.ratio[:, j]
    a_mn = np.array([r["params"]["a_Mn"] for r in sweep.recs])
    a_fe = np.array([r["params"]["a_Fe"] for r in sweep.recs])

    edges = np.linspace(0.2, 2.0, nbin + 1)
    grid = np.full((nbin, nbin), np.nan)          # (a_Fe row, a_Mn col)
    for r in range(nbin):
        for c in range(nbin):
            m = ((a_fe >= edges[r]) & (a_fe < edges[r + 1])
                 & (a_mn >= edges[c]) & (a_mn < edges[c + 1]))
            if m.sum():
                grid[r, c] = np.median(v[m])

    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    mesh = ax.pcolormesh(edges, edges, grid, cmap=cmap, shading="flat",
                         edgecolors=SURFACE, linewidth=1.5)
    for r in range(nbin):                          # direct labels: the relief
        for c in range(nbin):                      # rule, since the pale steps
            if np.isfinite(grid[r, c]):            # sit under 3:1 on white
                mid = grid[r, c] > np.nanmedian(grid)
                ax.text((edges[c] + edges[c + 1]) / 2,
                        (edges[r] + edges[r + 1]) / 2,
                        f"{grid[r, c]:.3f}", ha="center", va="center",
                        fontsize=7.5, color=SURFACE if mid else INK_2)
    for k in np.argsort(-v)[:2]:
        ax.scatter([a_mn[k]], [a_fe[k]], s=90, facecolors="none",
                   edgecolors=CRITICAL, linewidths=1.6, zorder=5)
        ax.annotate(f"point {sweep.pts[k]}: {v[k]:.2f}$\\sigma$",
                    xy=(a_mn[k], a_fe[k]), xytext=(-16, 20),
                    textcoords="offset points", color=CRITICAL, fontsize=8,
                    ha="right",
                    arrowprops=dict(arrowstyle="-", color=CRITICAL, lw=0.9),
                    zorder=6,
                    # the cells carry direct labels, so the callout needs a
                    # surface backing to stay legible wherever it lands
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6,
                              alpha=0.92))
    rho_mn = spearmanr(a_mn, v)[0]
    rho_fe = spearmanr(a_fe, v)[0]
    ax.set_xlabel(r"$a_{\rm Mn}$ [solar]")
    ax.set_ylabel(r"$a_{\rm Fe}$ [solar]")
    ax.set_title("Mn bias, single temperature", color=INK, pad=8)
    ax.text(0.5, -0.17, f"cell = median $|b_{{\\rm sys}}|/\\sigma$ of "
            f"{len(v) // (nbin * nbin)} points;  Spearman "
            f"$\\rho$ = {rho_mn:+.2f} on $a_{{\\rm Mn}}$, {rho_fe:+.2f} on "
            f"$a_{{\\rm Fe}}$", transform=ax.transAxes, color=INK_2,
            fontsize=8, ha="center", va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(f"median $|b_{{\\rm sys}}|/\\sigma$ at "
                 f"$10^{{{int(np.log10(N_REF_LABEL))}}}$ counts", fontsize=9)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


AX_LABEL = {"kT": r"$\log kT$", "logT_mean": r"$\log T_{\rm mean}$",
            "logT_sigma": r"$\log T_\sigma$", "sigma_v": r"$\sigma_v$",
            "n_h": r"$n_{\rm H}$"}


def _design_axes(sw):
    """The design coordinates, in the order the corner plot shows them.

    kT is drawn as log10 because the design SAMPLES it log-uniform: on a linear
    axis the marginal would slope and a flat tail would look like a cold excess
    that is really just the sampling density.
    """
    keys = (["logT_mean", "logT_sigma"] if sw.mode == "dem" else ["kT"])
    keys += [k for k in sw.recs[0]["params"] if k.startswith("a_")]
    keys += ["sigma_v", "n_h"]
    out = []
    for k in keys:
        v = np.array([r["params"][k] for r in sw.recs])
        if k == "kT":
            v = np.log10(v)
        lab = AX_LABEL.get(k, k.replace("a_", r"$a_{\rm ") + "}$"
                           if k.startswith("a_") else k)
        out.append((k, lab, v))
    return out


def fig_corner(sw, out, nbin=18):
    """Design space with the >1 sigma points picked out -- are they confined?

    Every point grey, the points whose WORST parameter exceeds 1 sigma in red.
    The diagonal carries the marginal of each axis (grey outline = the whole
    design, red fill = the tail, both normalised to their own area), because
    that is the panel that actually answers the question: an axis where the red
    marginal sits on top of the grey one does not confine the tail, and an axis
    where it collapses into a corner does.
    """
    axes_data = _design_axes(sw)
    n = len(axes_data)
    bad = sw.worst > 1.0
    fig, axarr = plt.subplots(n, n, figsize=(1.05 * n, 1.05 * n),
                              sharex="col")
    for i in range(n):
        for j in range(n):
            ax = axarr[i, j]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j > i:
                ax.set_visible(False)
                continue
            ax.tick_params(labelsize=6, length=2)
            if i == j:
                v = axes_data[i][2]
                bins = np.linspace(v.min(), v.max(), nbin)
                ax.hist(v, bins=bins, color=GRID, edgecolor=MUTED, lw=0.6,
                        density=True)
                ax.hist(v[bad], bins=bins, color=CRITICAL, alpha=0.55,
                        density=True)
                ax.set_yticks([])
            else:
                x, y = axes_data[j][2], axes_data[i][2]
                ax.scatter(x[~bad], y[~bad], s=2.0, c=MUTED, alpha=0.30,
                           linewidths=0, zorder=2)
                ax.scatter(x[bad], y[bad], s=7.0, c=CRITICAL, alpha=0.95,
                           linewidths=0, zorder=3)
            if j == 0 and i > 0:
                ax.set_ylabel(axes_data[i][1], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(axes_data[j][1], fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", markersize=4, color=MUTED,
                      label=f"all {len(bad)} sweep points"),
               Line2D([], [], marker="o", ls="", markersize=5, color=CRITICAL,
                      label=f"worst parameter above 1$\\sigma$ ({bad.sum()})")]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle(f"P7 design space, {sw.title}", x=0.12, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    R = os.path.expanduser("~/work/data/spexai/results/bias_sweep")
    ap.add_argument("--single", default=os.path.join(
        R, "bias_single_n1000_s39235.jsonl"))
    ap.add_argument("--dem", default=os.path.join(
        R, "bias_dem_n1000_s39235.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "docs", "figures"))
    ap.add_argument("--fig", default="all",
                    choices=["a", "b", "c", "d", "e", "all"])
    ap.add_argument("--target_counts", type=float, default=N_REF_LABEL)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sweeps = [Sweep(args.single, args.target_counts),
              Sweep(args.dem, args.target_counts)]
    o = lambda n: os.path.join(args.outdir, n)          # noqa: E731
    if args.fig in ("a", "all"):
        fig_safety_map(sweeps, o("p7_safety_map.png"))
    if args.fig in ("b", "all"):
        fig_by_param(sweeps, o("p7_bias_by_param.png"))
    if args.fig in ("c", "all"):
        fig_nstar(sweeps, o("p7_nstar_vs_temperature.png"))
    if args.fig in ("d", "all"):
        fig_mn(sweeps[0], o("p7_mn_outliers.png"))
    if args.fig in ("e", "all"):
        fig_corner(sweeps[0], o("p7_corner_single.png"))
        fig_corner(sweeps[1], o("p7_corner_dem.png"))


if __name__ == "__main__":
    main()
