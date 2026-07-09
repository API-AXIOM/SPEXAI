"""Training for the operator-style emulator (Ricketts et al. 2026 recipe).

Loss = Huber(relative error in linear flux)
     + one-sided floor penalty (bins with no real flux must stay below floor)
     + weak Sobolev derivative-matching term in (log10 E, log10 flux) space
     + small log-space MSE stabiliser that is ramped to zero early in training

Training samples random subsets of energy points per step, so interpolation
along the energy axis is part of the task rather than an artefact of the grid.

Run as a script for a single training run, e.g.

    python -m spexai.train.train_operator --variant base --steps 20000
    python -m spexai.train.train_operator --variant no_film ...

Variants: base, no_sobolev, no_trend, no_film, no_fourier, fixed_grid,
plus two additions beyond the paper: hash_grid (multi-resolution learned
grid encoding instead of Fourier features) and line_head (dedicated
amplitude head for line bins at fixed energies).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from spexai.train.operator import OperatorConfig, SpectralOperator, FixedGridMLP

FLOOR = -10.0  # log10 flux below which the spectrum is treated as empty
LINE_THRESHOLD_DEX = math.log10(2.0)  # flux > 2x continuum -> line bin


def continuum_estimate(logflux, window=501, stride=50):
    """Running median of clamped log flux, computed at coarse centres and
    linearly interpolated back to the full grid."""
    nspec, nbins = logflux.shape
    lf = np.clip(logflux, FLOOR, None)
    centers = np.arange(0, nbins, stride)
    half = window // 2
    cont_coarse = np.empty((nspec, len(centers)), dtype=np.float32)
    for j, c in enumerate(centers):
        lo, hi = max(0, c - half), min(nbins, c + half + 1)
        cont_coarse[:, j] = np.median(lf[:, lo:hi], axis=1)
    x = np.arange(nbins)
    cont = np.empty_like(lf)
    for i in range(nspec):
        cont[i] = np.interp(x, centers, cont_coarse[i])
    return cont


def find_line_bins(data, n_sample=256):
    """Union of line bins over a temperature-ordered sample of training
    spectra. Deterministic, so a checkpoint can be rebuilt from the cache.

    Returns a LongTensor (n_bins,) mapping bin index -> line slot (-1 if
    the bin never hosts a line).
    """
    idx = data.train_idx.numpy()
    idx = idx[np.argsort(data.temps.numpy()[idx])]
    sel = idx[np.linspace(0, len(idx) - 1, n_sample).astype(int)]
    lf = data.logflux[torch.from_numpy(sel)].numpy()
    cont = continuum_estimate(lf)
    is_line = ((lf > FLOOR) &
               (np.clip(lf, FLOOR, None) - cont > LINE_THRESHOLD_DEX)).any(axis=0)
    line_ids = torch.full((data.n_bins,), -1, dtype=torch.long)
    line_ids[torch.from_numpy(is_line.copy())] = torch.arange(int(is_line.sum()))
    return line_ids


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class SpectrumData:
    """Holds the preprocessed cache in memory (float32)."""

    def __init__(self, cachedir):
        self.energy = torch.from_numpy(np.load(os.path.join(cachedir, "energy.npy")))
        self.temps = torch.from_numpy(np.load(os.path.join(cachedir, "temps.npy")))
        self.logflux = torch.from_numpy(
            np.load(os.path.join(cachedir, "logflux.npy"), mmap_mode=None))
        splits = np.load(os.path.join(cachedir, "splits.npz"))
        self.train_idx = torch.from_numpy(splits["train"]).long()
        self.val_idx = torch.from_numpy(splits["val"]).long()
        self.test_idx = torch.from_numpy(splits["test"]).long()
        self.n_bins = self.logflux.shape[1]


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def relative_error_loss(pred, target, valid, huber_delta=1.0):
    """Huber loss on |10^(pred - target) - 1| for valid bins,
    one-sided floor penalty for empty bins (target at FLOOR)."""
    d = torch.clamp(pred - target, -4.0, 4.0)
    # valid bins: symmetric relative error in linear flux
    eps_valid = torch.abs(torch.pow(10.0, torch.where(valid, d, torch.zeros_like(d))) - 1.0)
    # floor bins: only penalise predictions above the floor
    d_floor = torch.clamp(torch.relu(pred - FLOOR), max=4.0)
    eps_floor = torch.pow(10.0, d_floor) - 1.0
    eps = torch.where(valid, eps_valid, eps_floor)
    return F.huber_loss(eps, torch.zeros_like(eps), delta=huber_delta)


def sobolev_loss(pred, target, x, valid):
    """Derivative matching between consecutive sampled points (Huber on the
    slope mismatch), computed only where both endpoints hold real flux."""
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    dx = (x[:, 1:] - x[:, :-1]).clamp(min=1e-6)
    both = valid[:, 1:] & valid[:, :-1]
    slope_err = (dp - dt) / dx
    slope_err = torch.where(both, slope_err, torch.zeros_like(slope_err))
    loss = F.huber_loss(slope_err, torch.zeros_like(slope_err), reduction="none", delta=1.0)
    return loss.sum() / both.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, idx, device, batch=64, fixed_grid=False, return_bins=False):
    """Per-spectrum mean relative error over non-empty bins, plus yields."""
    model.eval()
    energy = data.energy.to(device)
    mre = []
    bin_err_sum = None
    bin_err_cnt = None
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        target = data.logflux[sel].to(device)
        temps = data.temps[sel].to(device)
        if fixed_grid:
            pred = model(temps)
        else:
            pred = model(temps, energy)
        valid = target > FLOOR
        d = torch.clamp(pred - torch.clamp(target, min=FLOOR), -4.0, 4.0)
        eps = torch.abs(torch.pow(10.0, d) - 1.0)
        eps = torch.where(valid, eps, torch.zeros_like(eps))
        n_valid = valid.sum(dim=1).clamp(min=1)
        mre.append((eps.sum(dim=1) / n_valid).cpu())
        if return_bins:
            if bin_err_sum is None:
                bin_err_sum = torch.zeros(data.n_bins)
                bin_err_cnt = torch.zeros(data.n_bins)
            bin_err_sum += torch.where(valid, eps, torch.zeros_like(eps)).sum(0).cpu()
            bin_err_cnt += valid.sum(0).cpu()
    mre = torch.cat(mre).numpy()
    out = {
        "mre_mean": float(mre.mean()),
        "mre_median": float(np.median(mre)),
        "yield_1pct": float((mre <= 0.01).mean() * 100),
        "yield_10pct": float((mre <= 0.10).mean() * 100),
    }
    if return_bins:
        out["bin_mre"] = (bin_err_sum / bin_err_cnt.clamp(min=1)).numpy()
        out["per_spectrum_mre"] = mre
    return out


# ---------------------------------------------------------------------------
# training loops
# ---------------------------------------------------------------------------

def make_variant(variant, data, args):
    """Instantiate the model for a named ablation variant."""
    kw = dict(
        hidden_size=args.hidden, n_hidden=args.layers,
        n_freqs=args.n_freqs, f_max=args.f_max,
        activation=getattr(args, "activation", "gelu"),
        line_dim=getattr(args, "line_dim", 16),
        x_lo=math.log10(data.energy[0].item()),
        x_hi=math.log10(data.energy[-1].item()),
        t_lo=math.log10(data.temps.min().item()),
        t_hi=math.log10(data.temps.max().item()),
    )
    if variant == "fixed_grid":
        return FixedGridMLP(data.n_bins, hidden_size=args.fg_hidden,
                            n_hidden=args.fg_layers,
                            t_lo=kw["t_lo"], t_hi=kw["t_hi"])
    flags = dict(
        base=dict(),
        no_sobolev=dict(),
        no_trend=dict(use_trend=False),
        no_film=dict(use_film=False),
        no_fourier=dict(use_fourier=False),
        hash_grid=dict(use_grid=True),
        line_head=dict(use_linehead=True),
        # winning combination from the element-26 ablation:
        # line head on, Sobolev off (handled by use_sobolev in train());
        # trend head switchable for hyperparameter search
        combo=dict(use_linehead=True,
                   use_trend=bool(getattr(args, "use_trend", 1))),
    )[variant]
    line_ids = None
    if flags.get("use_linehead"):
        line_ids = find_line_bins(data)
        n_lines = int((line_ids >= 0).sum())
        print(f"line head: {n_lines} line bins "
              f"({100.0 * n_lines / data.n_bins:.1f}% of grid)", flush=True)
    return SpectralOperator(OperatorConfig(**{**kw, **flags}),
                            line_ids=line_ids, energy_grid=data.energy)


def train(args):
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = make_variant(args.variant, data, args).to(device)
    fixed_grid = args.variant == "fixed_grid"
    use_sobolev = (args.variant not in ("no_sobolev", "fixed_grid", "combo")
                   and args.w_sobolev > 0)
    print(f"variant={args.variant} model={model} params={model.count_parameters():,} "
          f"device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(1, int(0.02 * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    x_grid = None
    if not fixed_grid:
        x_grid = model.norm_energy(data.energy.to(device)).unsqueeze(-1)  # (M, 1)

    train_idx = data.train_idx
    target_all = data.logflux
    temps_all = data.temps

    os.makedirs(args.outdir, exist_ok=True)
    runname = getattr(args, "tag", None) or args.variant
    best = {"val_yield_1pct": -1.0}
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        sel = train_idx[torch.randint(len(train_idx), (args.batch,))]
        target = target_all[sel]
        temps = temps_all[sel].to(device)

        if fixed_grid:
            target = target.to(device)
            pred = model(temps)
            x_pts = None
        else:
            pts, _ = torch.sort(torch.randperm(data.n_bins)[:args.points])
            target = target[:, pts].to(device)
            x_pts = x_grid[pts]  # (P, 1)
            alpha = min(1.0, step / (args.curriculum_frac * args.steps)) \
                if (model.config.use_fourier or model.config.use_grid) else 1.0
            tnorm = model.norm_temp(temps).view(-1, 1)
            pred = model.forward_norm(tnorm, x_pts.unsqueeze(0).expand(args.batch, -1, -1),
                                      alpha=alpha, bins=pts.to(device))

        valid = target > FLOOR
        target_c = torch.clamp(target, min=FLOOR)
        loss = relative_error_loss(pred, target_c, valid)

        # early log-space stabiliser, ramped to zero
        w_log = args.w_log * max(0.0, 1.0 - step / (0.2 * args.steps))
        if w_log > 0:
            loss = loss + w_log * F.mse_loss(pred, target_c)

        if use_sobolev:
            xb = x_pts.squeeze(-1).unsqueeze(0).expand(args.batch, -1)
            loss = loss + args.w_sobolev * sobolev_loss(pred, target_c, xb, valid)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, data, data.val_idx, device, fixed_grid=fixed_grid)
            rec = {"step": step, "loss": float(loss.item()),
                   "elapsed_s": time.time() - t0,
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            print(f"[{args.variant}] step {step}/{args.steps} loss={rec['loss']:.4f} "
                  f"val MRE={val['mre_mean']:.4f} yield1%={val['yield_1pct']:.2f} "
                  f"yield10%={val['yield_10pct']:.2f} ({rec['elapsed_s']:.0f}s)",
                  flush=True)
            if val["yield_1pct"] >= best["val_yield_1pct"]:
                best = {"step": step, **{f"val_{k}": v for k, v in val.items()},
                        "val_yield_1pct": val["yield_1pct"]}
                torch.save({"state_dict": model.state_dict(),
                            "variant": args.variant,
                            "args": vars(args)},
                           os.path.join(args.outdir, f"{runname}.pt"))

    with open(os.path.join(args.outdir, f"{runname}_history.json"), "w") as f:
        json.dump({"history": history, "best": best,
                   "params": model.count_parameters(),
                   "args": vars(args)}, f, indent=2)
    print(f"[{args.variant}] done in {(time.time()-t0)/60:.1f} min; "
          f"best val yield1%={best['val_yield_1pct']:.2f} at step {best.get('step')}",
          flush=True)
    return model, best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26")
    ap.add_argument("--variant", default="base",
                    choices=["base", "no_sobolev", "no_trend", "no_film",
                             "no_fourier", "fixed_grid", "hash_grid",
                             "line_head", "combo"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--n_freqs", type=int, default=256)
    ap.add_argument("--f_max", type=float, default=8000.0)
    ap.add_argument("--activation", default="gelu",
                    choices=["gelu", "silu", "tanh", "sine"])
    ap.add_argument("--line_dim", type=int, default=16)
    ap.add_argument("--use_trend", type=int, default=1,
                    help="combo variant only: include the trend head (1/0)")
    ap.add_argument("--fg_hidden", type=int, default=512)
    ap.add_argument("--fg_layers", type=int, default=4)
    ap.add_argument("--w_sobolev", type=float, default=1e-3)
    ap.add_argument("--w_log", type=float, default=0.1)
    ap.add_argument("--curriculum_frac", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None,
                    help="checkpoint/history name (defaults to variant)")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
