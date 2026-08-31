"""Train a (temperature, velocity) emulator of BROADENED spectra (option 2).

The model is the same SpectralOperator trunk (Fourier features + FiLM,
optionally per-bin normalisation) with two conditioning parameters; the
line head is not used because broadening removes the delta-spike
structure that motivated it.

Targets are generated on the fly from the ORIGINAL SPEX spectra: each
training example pairs a real spectrum (its true temperature, possibly
drawn repeatedly) with a velocity sampled log-uniformly in
[--vmin, --vmax], broadened by FFT on a fine uniform log-energy grid.
That uniform grid is also the model's training grid (stored in the
checkpoint, so forward_on_grid works as usual).

    python -m spexai.train.train_broadened --steps 20000 --lr 1e-3

The scattered uniform-grid flux cache (a few GB) is built automatically
on first use and stored next to the preprocessed data.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from spexai.broadening import (fft_broaden, scatter_to_grid,
                                     uniform_log_edges)
from spexai.operator import OperatorConfig, SpectralOperator, \
    edges_from_centers
from spexai.train.train_operator import FLOOR, relative_error_loss
from spexai.data import SpectrumData

# broadened wings below this fraction of the spectrum peak are zeroed:
# they are physically invisible, and float32 FFT noise lives down there
REL_FLOOR = 1e-9


def build_uniform_cache(data, dlx, path):
    """Scatter every native spectrum's integrated flux onto the uniform
    log grid once (cloud-in-cell); stored as a float32 memmap."""
    edges = edges_from_centers(data.energy)
    uni = uniform_log_edges(float(edges[0]), float(edges[-1]), dlx)
    widths = edges[1:] - edges[:-1]
    K = len(uni) - 1
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(data.logflux.shape[0], K))
    t0 = time.time()
    for lo in range(0, data.logflux.shape[0], 256):
        hi = min(lo + 256, data.logflux.shape[0])
        flux = torch.pow(10.0, torch.clamp(data.logflux[lo:hi], min=-30)) * widths
        out[lo:hi] = scatter_to_grid(flux, edges, uni).numpy()
        if lo % 2560 == 0:
            print(f"  cache {lo}/{data.logflux.shape[0]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    return uni


class BroadenedTargets:
    """Generates broadened log-density targets on the uniform grid."""

    def __init__(self, data, dlx, cachedir, device):
        self.dlx = dlx
        self.device = device
        path = os.path.join(cachedir, f"uniform_flux_dlx{dlx:g}.npy")
        edges = edges_from_centers(data.energy)
        self.uni = uniform_log_edges(float(edges[0]), float(edges[-1]), dlx)
        if not os.path.exists(path):
            print("building uniform-grid flux cache ...", flush=True)
            build_uniform_cache(data, dlx, path)
        self.flux_uni = torch.from_numpy(np.load(path, mmap_mode="r"))
        self.widths = (self.uni[1:] - self.uni[:-1]).to(device)
        self.centers = torch.sqrt(self.uni[:-1] * self.uni[1:]).to(device)
        self.n_bins = len(self.centers)

    def __call__(self, rows, velocity):
        """rows: (B,) spectrum indices; velocity: (B,) km/s.
        Returns log10 flux density on the uniform grid, (B, K)."""
        flux = self.flux_uni[rows].to(self.device)
        broad = fft_broaden(flux, self.dlx, velocity)
        peak = broad.max(dim=1, keepdim=True).values
        broad = torch.where(broad > REL_FLOOR * peak, broad,
                            torch.zeros_like(broad))
        dens = broad / self.widths
        return torch.log10(torch.clamp(dens, min=1e-30))


def sample_velocity(n, vmin, vmax, device):
    u = torch.rand(n, device=device)
    return vmin * (vmax / vmin) ** u


def norm_velocity(v, vmin, vmax):
    lv = torch.log10(v)
    return 2.0 * (lv - math.log10(vmin)) / math.log10(vmax / vmin) - 1.0


@torch.no_grad()
def evaluate(model, targets, data, idx, vlist, vmin, vmax, device,
             batch=64, echunk=8192):
    """Mean relative error / yields over val spectra at fixed velocities."""
    model.eval()
    x_all = model.norm_energy(targets.centers).view(1, -1, 1)
    mre = []
    for v in vlist:
        for i in range(0, len(idx), batch):
            sel = idx[i:i + batch]
            vv = torch.full((len(sel),), v, device=device)
            target = targets(sel, vv)
            tn = model.norm_temp(data.temps[sel].to(device))
            vn = norm_velocity(vv, vmin, vmax)
            theta = torch.stack([tn, vn], dim=1)
            pred = torch.cat(
                [model.forward_norm(
                    theta, x_all[:, lo:lo + echunk].expand(len(sel), -1, -1),
                    bins=torch.arange(lo, min(lo + echunk, targets.n_bins),
                                      device=device))
                 for lo in range(0, targets.n_bins, echunk)], dim=1)
            valid = target > FLOOR
            d = torch.clamp(pred - torch.clamp(target, min=FLOOR), -4., 4.)
            eps = torch.where(valid, torch.abs(torch.pow(10.0, d) - 1.0),
                              torch.zeros_like(d))
            mre.append((eps.sum(1) / valid.sum(1).clamp(min=1)).cpu())
    mre = torch.cat(mre).numpy()
    return {"mre_mean": float(mre.mean()),
            "mre_median": float(np.median(mre)),
            "yield_01pct": float((mre <= 0.001).mean() * 100),
            "yield_1pct": float((mre <= 0.01).mean() * 100),
            "yield_10pct": float((mre <= 0.10).mean() * 100)}


def train(args):
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = BroadenedTargets(data, args.dlx, args.cachedir, device)

    cfg = OperatorConfig(
        n_params=2, use_linehead=False,
        use_trend=bool(args.use_trend), use_binnorm=bool(args.use_binnorm),
        hidden_size=args.hidden, n_hidden=args.layers,
        n_freqs=args.n_freqs, f_max=args.f_max, activation=args.activation,
        x_lo=math.log10(targets.centers[0].item()),
        x_hi=math.log10(targets.centers[-1].item()),
        t_lo=math.log10(data.temps.min().item()),
        t_hi=math.log10(data.temps.max().item()))

    bin_stats = None
    if cfg.use_binnorm:
        # stats over a broadened sample spanning (T, v)
        rows = data.train_idx[torch.linspace(0, len(data.train_idx) - 1,
                                             384).long()]
        vv = sample_velocity(len(rows), args.vmin, args.vmax, device)
        lf = torch.clamp(targets(rows, vv), min=FLOOR).cpu()
        bin_stats = (lf.mean(dim=0), lf.std(dim=0))

    model = SpectralOperator(cfg, energy_grid=targets.centers.cpu(),
                             bin_stats=bin_stats).to(device)
    print(f"broadened emulator: params={model.count_parameters():,} "
          f"uniform grid K={targets.n_bins} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(1, int(0.02 * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    x_grid = model.norm_energy(targets.centers).unsqueeze(-1)  # (K, 1)

    os.makedirs(args.outdir, exist_ok=True)
    vlist = [float(v) for v in args.eval_velocities]
    val_idx = data.val_idx[:args.val_spectra]
    best = {"val_yield_1pct": -1.0}
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        sel = data.train_idx[torch.randint(len(data.train_idx), (args.batch,))]
        vv = sample_velocity(args.batch, args.vmin, args.vmax, device)
        target_full = targets(sel, vv)                      # (B, K)
        pts, _ = torch.sort(torch.randperm(targets.n_bins,
                                           device=device)[:args.points])
        target = target_full[:, pts]
        tn = model.norm_temp(data.temps[sel].to(device))
        theta = torch.stack([tn, norm_velocity(vv, args.vmin, args.vmax)], dim=1)
        alpha = min(1.0, step / (args.curriculum_frac * args.steps))
        pred = model.forward_norm(
            theta, x_grid[pts].unsqueeze(0).expand(args.batch, -1, -1),
            alpha=alpha, bins=pts)

        valid = target > FLOOR
        target_c = torch.clamp(target, min=FLOOR)
        loss = relative_error_loss(pred, target_c, valid)
        w_log = args.w_log * max(0.0, 1.0 - step / (0.2 * args.steps))
        if w_log > 0:
            loss = loss + w_log * F.mse_loss(pred, target_c)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, targets, data, val_idx, vlist,
                           args.vmin, args.vmax, device)
            rec = {"step": step, "loss": float(loss.item()),
                   "elapsed_s": time.time() - t0,
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            print(f"[broadened] step {step}/{args.steps} loss={rec['loss']:.4f} "
                  f"val MRE={val['mre_mean']:.4f} yield1%={val['yield_1pct']:.2f} "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
            if val["yield_1pct"] >= best["val_yield_1pct"]:
                best = {"step": step,
                        **{f"val_{k}": v for k, v in val.items()}}
                torch.save({"state_dict": model.state_dict(),
                            "config": cfg.__dict__, "args": vars(args)},
                           os.path.join(args.outdir, f"{args.tag}.pt"))

    with open(os.path.join(args.outdir, f"{args.tag}_history.json"), "w") as f:
        json.dump({"history": history, "best": best,
                   "params": model.count_parameters(),
                   "args": vars(args)}, f, indent=2)
    print(f"[broadened] done in {(time.time()-t0)/60:.1f} min; best "
          f"val yield1%={best['val_yield_1pct']:.2f}", flush=True)
    return model, best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/broadened")
    ap.add_argument("--tag", default="broadened")
    # defaults are the t04 HPO winner (0.31% test MRE, 94.7% yield@1%);
    # line_dim is the only t04 setting that doesn't carry over (no line head)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--n_freqs", type=int, default=512)
    ap.add_argument("--f_max", type=float, default=4000.0)
    ap.add_argument("--activation", default="gelu",
                    choices=["gelu", "silu", "tanh", "sine"])
    ap.add_argument("--use_trend", type=int, default=1)
    ap.add_argument("--use_binnorm", type=int, default=1)
    ap.add_argument("--vmin", type=float, default=50.0)
    ap.add_argument("--vmax", type=float, default=1000.0)
    ap.add_argument("--dlx", type=float, default=2.5e-5)
    ap.add_argument("--eval_velocities", nargs="+",
                    default=[100.0, 250.0, 600.0])
    ap.add_argument("--val_spectra", type=int, default=192)
    ap.add_argument("--w_log", type=float, default=0.1)
    ap.add_argument("--curriculum_frac", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
