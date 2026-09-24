"""Diagnostic plots for the emulator-bias posterior check.

Reads the summary jsonl written by ``emulator_bias_posterior_check.py`` and, when
present, the per-point sample npz the driver writes by default (disable
with its ``--no_save_samples``). The jsonl
alone carries only summary statistics, so it supports the diagnostics but NOT a
corner plot; corner plots need the npz.

Panels produced per point:

``pull``
    measured pull per parameter, with the screen's rescaled prediction beside
    it. This is the comparison P8 exists to make.
``k``
    pull / screen per parameter, on a symmetric log axis because k spans
    decades once the screen's prediction approaches zero. Parameters whose
    screen prediction is small carry a meaningless k and are greyed out.
``intervals``
    posterior median with its 16-84 interval against the true injected value,
    each parameter scaled to its own sigma so twelve different units share an
    axis. Prior bounds are drawn where they are known, because a parameter
    pressed against a bound is not a measurement.
``corner``
    only with a sample npz.

A note on honesty of representation: with summary statistics alone this script
draws INTERVALS, never a density. Turning (median, sigma) into a Gaussian curve
would show an assumed shape as if it had been measured.

    conda run -n spexai python -u \\
        scripts/inference/plot_bias_posterior_check.py \\
        --jsonl ~/work/data/spexai/results/bias_posterior_check/probe_single.jsonl \\
        --outdir docs/figures
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))

# log_norm's box is [truth-1, truth+1] (bias_sweep.build_pars); the abundance
# box is [0.02, 3.0]. Only the ones we can reconstruct from the record are drawn.
ABUND_BOX = (0.02, 3.0)
SIGMA_V_BOX = (10.0, 700.0)
N_H_BOX = (0.0, 6.0)
LOG_NORM_HALF_WIDTH = 1.0


def load(jsonl):
    return [json.loads(l) for l in open(jsonl) if l.strip()]


def param_box(name, truth_lognorm):
    """The prior box for one parameter, or None when it is not reconstructible."""
    if name == "log_norm":
        return (
            truth_lognorm - LOG_NORM_HALF_WIDTH,
            truth_lognorm + LOG_NORM_HALF_WIDTH,
        )
    if name == "sigma_v":
        return SIGMA_V_BOX
    if name == "n_h":
        return N_H_BOX
    if name in ("kT", "logT_mean", "logT_sigma"):
        return None  # depends on RANGES/LOGT_PAD_DEX
    return ABUND_BOX  # the free abundances


def saturation_flags(rec):
    """Which parameters sit within 3 sigma of a known prior bound.

    A posterior pressed against a bound has not measured anything, and its pull
    and k are artifacts. This is what caught the 2026-09-23 log_norm bug.
    """
    names, med = rec["names"], np.array(rec["median"])
    sig = np.array(rec["sigma"])
    j = names.index("log_norm") if "log_norm" in names else None
    tl = rec["truth"][j] if j is not None else 0.0
    out = {}
    for i, n in enumerate(names):
        box = param_box(n, tl)
        if box is None:
            out[n] = None
            continue
        d = min(med[i] - box[0], box[1] - med[i]) / max(sig[i], 1e-30)
        out[n] = d
    return out


def plot_pull(rec, sat, path):
    names = rec["names"]
    pull = np.array(rec["pull"])
    ps = np.array(rec["pull_screen"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.2))

    bad = [n for n in names if sat[n] is not None and sat[n] < 3.0]
    colors = ["#c1121f" if n in bad else "#1d3557" for n in names]
    ax.bar(x - 0.2, pull, width=0.4, color=colors, label="measured pull")
    ax.bar(
        x + 0.2, ps, width=0.4, color="#a8a8a8", label="screen prediction (rescaled)"
    )

    for b in (-1, 1):
        ax.axhline(b, color="k", lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"(median $-$ truth) / $\sigma$")
    lim = np.percentile(np.abs(np.concatenate([pull, ps])), 90) * 2.5
    ax.set_ylim(-max(lim, 4), max(lim, 4))
    ttl = (
        f"point {rec['point']} ({rec['tag']}), {rec['sampler']}, "
        f"{rec['target_counts']:.0e} counts"
    )
    if bad:
        ttl += f"\nRED = within 3$\\sigma$ of a prior bound: {', '.join(bad)}"
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    note = [n for n in names if abs(pull[names.index(n)]) > max(lim, 4)]
    if note:
        ax.text(
            0.99,
            0.03,
            f"off scale: "
            + ", ".join(f"{n}={pull[names.index(n)]:+.0f}" for n in note),
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            color="#c1121f",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_k(rec, sat, path):
    names = rec["names"]
    k = np.array(rec["k"], dtype=float)
    ps = np.array(rec["pull_screen"])
    x = np.arange(len(names))
    # k is only interpretable where the screen actually predicts something
    weak = np.abs(ps) < 0.5
    fig, ax = plt.subplots(figsize=(10, 4.2))
    col = ["#a8a8a8" if w else "#1d3557" for w in weak]
    for i, n in enumerate(names):
        if sat[n] is not None and sat[n] < 3.0:
            col[i] = "#c1121f"
    ax.bar(x, np.where(np.isfinite(k), k, 0.0), color=col)
    ax.axhline(1.0, color="#2a9d8f", lw=1.2, label="k = 1 (screen exactly right)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("k = measured pull / screen pull")
    ax.set_title(
        f"point {rec['point']}: nonlinearity factor k "
        f"(grey = |screen| < 0.5, k meaningless; red = at a bound)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_intervals(rec, sat, path):
    """Median + 16-84 interval against the injected truth, in units of sigma."""
    names = rec["names"]
    t = np.array(rec["truth"])
    m = np.array(rec["median"])
    s = np.array(rec["sigma"])
    j = names.index("log_norm") if "log_norm" in names else None
    tl = rec["truth"][j] if j is not None else 0.0

    q16 = np.array(rec.get("q16", m - s))
    q84 = np.array(rec.get("q84", m + s))
    lim = 8.0

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, n in enumerate(names):
        yy = y[i]
        c = "#c1121f" if (sat[n] is not None and sat[n] < 3.0) else "#1d3557"
        # everything expressed as (value - truth)/sigma, so the interval is
        # centred on the MEDIAN and truth sits at 0
        lo, hi = (q16[i] - t[i]) / s[i], (q84[i] - t[i]) / s[i]
        mid = (m[i] - t[i]) / s[i]
        if lo > lim or hi < -lim:
            # entirely off scale: say so rather than drawing it at the edge
            xs = lim - 0.4 if lo > lim else -lim + 0.4
            ax.annotate(
                f"{mid:+.0f}$\\sigma$ $\\rightarrow$",
                (xs, yy),
                color=c,
                fontsize=8,
                ha="right" if lo > lim else "left",
                va="center",
                fontweight="bold",
            )
            continue
        ax.plot(
            [max(lo, -lim), min(hi, lim)],
            [yy, yy],
            color=c,
            lw=3,
            solid_capstyle="butt",
        )
        ax.plot(mid, yy, "o", color=c, ms=5)
        box = param_box(n, tl)
        if box is not None:
            for b in box:
                bs = (b - t[i]) / s[i]
                if abs(bs) < lim:
                    ax.plot([bs], [yy], "|", color="#e76f51", ms=14, mew=2)
    ax.axvline(0, color="k", lw=1.0, label="injected truth")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"(value $-$ injected truth) / posterior $\sigma$")
    ax.set_title(
        f"point {rec['point']}: posterior 16-84 interval vs truth\n"
        f"orange ticks = prior bounds; red = interval within "
        f"3$\\sigma$ of one",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_corner(npz_path, path):
    """Corner plot from a sample npz, truths marked."""
    try:
        import corner
    except ImportError:
        print("  corner not installed; skipping corner plot")
        return None
    z = np.load(npz_path, allow_pickle=False)
    samples = z["samples"]
    names = [str(x) for x in z["names"]]
    truth = z["truth"]
    fig = corner.corner(
        samples,
        labels=names,
        truths=list(truth),
        truth_color="#c1121f",
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 7},
        label_kwargs={"fontsize": 8},
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
    )
    fig.suptitle(
        f"point {int(z['point'])}: posterior, red = injected truth", fontsize=11
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", default=os.path.join(REPO, "docs", "figures"))
    ap.add_argument(
        "--samples_dir",
        default=None,
        help="where the sample npz files are (default: beside --jsonl)",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sdir = args.samples_dir or os.path.dirname(os.path.abspath(args.jsonl))

    recs = load(args.jsonl)
    print(f"{len(recs)} point(s) in {args.jsonl}")
    written = []
    for rec in recs:
        sat = saturation_flags(rec)
        tag = f"pt{rec['point']}_{rec['sampler']}"
        written.append(
            plot_pull(rec, sat, os.path.join(args.outdir, f"biascheck_pull_{tag}.png"))
        )
        written.append(
            plot_k(rec, sat, os.path.join(args.outdir, f"biascheck_k_{tag}.png"))
        )
        written.append(
            plot_intervals(
                rec, sat, os.path.join(args.outdir, f"biascheck_intervals_{tag}.png")
            )
        )

        hits = {n: d for n, d in sat.items() if d is not None and d < 3.0}
        if hits:
            print(
                f"  point {rec['point']}: PRIOR-SATURATED parameters "
                f"(distance to bound, in sigma): "
                + ", ".join(f"{n}={d:.2f}" for n, d in hits.items())
            )

        cands = [
            os.path.join(sdir, f)
            for f in sorted(os.listdir(sdir))
            if f.startswith("samples_")
            and f"_pt{rec['point']}_" in f
            and f.endswith(".npz")
        ]
        if cands:
            p = plot_corner(
                cands[0], os.path.join(args.outdir, f"biascheck_corner_{tag}.png")
            )
            if p:
                written.append(p)
        else:
            print(
                f"  point {rec['point']}: no sample npz in {sdir} -- no "
                f"corner plot -- the run predates sample saving, or used "
                f"--no_save_samples."
            )

    for p in [w for w in written if w]:
        print("wrote", p)


if __name__ == "__main__":
    main()
