"""Shared per-run diagnostics for the emulator trainers.

Every training run produces, in <outdir>/figures/:
  * <tag>_history.png - training loss, train-vs-validation MRE, and
    validation yield@1% over the run
  * <tag>_spectra_T<T>.png (three temperatures spanning the test range) -
    SPEX truth vs emulator over the full band with a residual strip, plus
    three randomly picked bright lines zoomed in with their residuals

Also provides EMAWeights (Polyak averaging of the model parameters);
checkpoints are saved with the EMA weights when enabled.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
EMU = "#2a78d6"     # emulator prediction
RESID = "#c23b2a"   # residuals
TRAIN = "#eda100"   # train-split curve (val uses EMU)


class EMAWeights:
    """Exponential moving average of the float entries of a state dict."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def swap_in(self, model):
        """Load EMA weights into the model (remember the live ones)."""
        self.backup = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)

    def swap_out(self, model):
        model.load_state_dict(self.backup, strict=False)
        self.backup = None


def _style(fig):
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        for s in ax.spines.values():
            s.set_visible(False)


def plot_history(history, out, tag=""):
    """Loss and train-vs-val metric curves; `history` is the list of eval
    records written by the trainers."""
    steps = [r["step"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=SURFACE)
    axes[0].plot(steps, [r["loss"] for r in history], color=INK, lw=1.4)
    axes[0].set(yscale="log", xlabel="step", ylabel="training loss")
    if any("train_mre_mean" in r for r in history):
        axes[1].plot(steps, [r.get("train_mre_mean", np.nan) for r in history],
                     color=TRAIN, lw=1.4, label="train")
    axes[1].plot(steps, [r["val_mre_mean"] for r in history],
                 color=EMU, lw=1.4, label="validation")
    axes[1].set(yscale="log", xlabel="step", ylabel="MRE")
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].plot(steps, [r["val_yield_1pct"] for r in history],
                 color=EMU, lw=1.4)
    axes[2].set(xlabel="step", ylabel="val yield@1% (%)")
    fig.suptitle(tag, x=0.01, ha="left", fontsize=11, color=INK)
    _style(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _predict_full(model, temp, energy, device, echunk=4096):
    t = temp.view(1).to(device)
    parts = [model(t, energy[lo:lo + echunk].to(device),
                   bins=torch.arange(lo, min(lo + echunk, len(energy)),
                                     device=device))
             for lo in range(0, len(energy), echunk)]
    return torch.cat(parts, dim=1).squeeze(0).cpu().numpy()


def plot_spectra(model, data, outdir, tag, device="cpu", n_lines=3,
                 zoom_bins=25, seed=0, temp_fracs=(0.1, 0.5, 0.9)):
    """Truth-vs-emulator figures at three test temperatures: full band +
    residual strip + `n_lines` random bright lines zoomed in."""
    from spexai.train.train_operator import FLOOR, find_line_bins
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    energy = data.energy
    e_np = energy.numpy()
    lb_all = torch.nonzero(find_line_bins(data) >= 0).squeeze(-1).numpy()

    idx = data.test_idx.numpy()
    order = idx[np.argsort(data.temps.numpy()[idx])]
    rows = [order[int(f * (len(order) - 1))] for f in temp_fracs]

    for row in rows:
        T = float(data.temps[row])
        truth = data.logflux[row].numpy()
        pred = _predict_full(model, data.temps[row], energy, device)
        valid = truth > FLOOR
        resid = 100.0 * (10.0 ** np.clip(pred - np.clip(truth, FLOOR, None),
                                         -4, 4) - 1.0)

        # three bright lines for this spectrum, seeded
        bright = lb_all[np.argsort(truth[lb_all])[-300:]]
        lines = np.sort(rng.choice(bright, size=min(n_lines, len(bright)),
                                   replace=False))

        fig = plt.figure(figsize=(11, 9.5), facecolor=SURFACE)
        gs = fig.add_gridspec(4, n_lines, height_ratios=[1.2, 0.45, 0.8, 0.5],
                              hspace=0.6, wspace=0.3)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(e_np, np.clip(truth, FLOOR, None), color=INK, lw=0.6,
                 label="SPEX")
        ax0.plot(e_np, np.clip(pred, FLOOR, None), color=EMU, lw=0.6,
                 ls="--", alpha=0.8, label="emulator")
        ax0.set(xscale="log", ylabel="log10 flux density")
        ax0.tick_params(labelbottom=False)
        ax0.set_title(f"{tag}: T = {T:.3f} keV", loc="left", fontsize=11,
                      color=INK)
        ax0.legend(loc="upper right", fontsize=9, ncols=2, frameon=False)

        axr = fig.add_subplot(gs[1, :], sharex=ax0)
        axr.plot(e_np[valid], resid[valid], color=RESID, lw=0.4)
        axr.axhline(0.0, color=INK2, lw=0.6, ls=":")
        span = np.percentile(np.abs(resid[valid]), 99)
        axr.set_ylim(-1.3 * span, 1.3 * span)
        axr.set(xscale="log", xlabel="Energy (keV)", ylabel="resid (%)")

        for k, j in enumerate(lines):
            lo, hi = max(0, j - zoom_bins), min(len(e_np), j + zoom_bins)
            axp = fig.add_subplot(gs[2, k])
            axz = fig.add_subplot(gs[3, k], sharex=axp)
            axp.plot(e_np[lo:hi], np.clip(truth[lo:hi], FLOOR, None),
                     color=INK, lw=1.6)
            axp.plot(e_np[lo:hi], np.clip(pred[lo:hi], FLOOR, None),
                     color=EMU, lw=1.2, ls="--")
            axp.set_title(f"line at {e_np[j]:.4f} keV", fontsize=9,
                          color=INK2)
            axp.tick_params(labelsize=8, labelbottom=False)
            v = valid[lo:hi]
            axz.plot(e_np[lo:hi][v], resid[lo:hi][v], color=RESID, lw=1.0)
            axz.axhline(0.0, color=INK2, lw=0.6, ls=":")
            axz.set(xlabel="Energy (keV)")
            axz.tick_params(labelsize=8)
            if k == 0:
                axp.set_ylabel("log10 flux density", fontsize=9)
                axz.set_ylabel("resid (%)", fontsize=9)

        _style(fig)
        out = os.path.join(outdir, f"{tag}_spectra_T{T:.2f}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}", flush=True)


def run_diagnostics(model, data, history, outdir, tag, device="cpu",
                    spectra=True):
    """Standard end-of-run bundle: history curves + spectrum figures."""
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    if history:
        plot_history(history, os.path.join(figdir, f"{tag}_history.png"), tag)
        print(f"saved {figdir}/{tag}_history.png", flush=True)
    if spectra:
        plot_spectra(model, data, figdir, tag, device=device)
