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


# hydrogenic ionisation potential: the H-like recombination edge sits here
EDGE_RYD_KEV = 0.0136057  # 1 Rydberg in keV; I(H-like) = EDGE_RYD_KEV * Z^2


def edge_element_guess(energy_kev):
    """Nearest element whose H-like recombination edge (13.6 Z^2 eV) lies at
    this energy; a label for diagnostics, not a physical identification."""
    return int(round((float(energy_kev) / EDGE_RYD_KEV) ** 0.5))


def find_edge_bins(data, n_sample=256, window=40, jump_dex=0.25,
                   min_count=8, min_gap=30):
    """Empirically locate radiative-recombination edges: sharp, persistent
    upward steps in the line-removed continuum (flux jumps up at an ion's
    ionisation threshold and decays above it). Detected from a
    temperature-ordered sample -- an edge appears only where its recombining
    ion is abundant, so we union over temperature. Deterministic given the
    cache. Returns a sorted LongTensor of edge bin indices (empty for
    elements whose edges lie outside the band, e.g. H, He)."""
    idx = data.train_idx.numpy()
    idx = idx[np.argsort(data.temps.numpy()[idx])]
    sel = idx[np.linspace(0, len(idx) - 1, min(n_sample, len(idx))).astype(int)]
    lf = data.logflux[torch.from_numpy(sel)].numpy()
    cont = continuum_estimate(lf)
    # de-line: replace line / empty bins with the smooth continuum so only
    # genuine continuum steps (edges) survive
    line = np.clip(lf, FLOOR, None) - cont > LINE_THRESHOLD_DEX
    d = np.where(line | (lf <= FLOOR), cont, lf).astype(np.float64)
    S, N = d.shape
    if N < 2 * window + 2:
        return torch.empty(0, dtype=torch.long)
    # windowed-mean step at each boundary: mean of the `window` bins above
    # minus the `window` bins below. Smooth bremsstrahlung falls with energy
    # (step < 0); an edge is a strongly positive step.
    csum = np.zeros((S, N + 1))
    np.cumsum(d, axis=1, out=csum[:, 1:])
    i = np.arange(window, N - window)
    below = (csum[:, i] - csum[:, i - window]) / window
    above = (csum[:, i + window] - csum[:, i]) / window
    step = above - below
    count = (step > jump_dex).sum(axis=0)
    strength = step.max(axis=0)
    cand = np.where(count >= min_count)[0]
    if len(cand) == 0:
        return torch.empty(0, dtype=torch.long)
    # cluster adjacent candidates; keep the strongest boundary per cluster
    edges, start = [], 0
    for j in range(1, len(cand) + 1):
        if j == len(cand) or i[cand[j]] - i[cand[j - 1]] > min_gap:
            grp = cand[start:j]
            edges.append(int(i[grp[np.argmax(strength[grp])]]))
            start = j
    return torch.tensor(sorted(set(edges)), dtype=torch.long)


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

def relative_error_loss(pred, target, valid, huber_delta=1.0,
                        sample_weight=None, point_weight=None):
    """Huber loss on |10^(pred - target) - 1| for valid bins,
    one-sided floor penalty for empty bins (target at FLOOR).

    sample_weight: optional (B,) per-spectrum weights (e.g. InfoBatch-style
    importance-sampling corrections). When given, the loss is the mean over
    points then the weighted mean over spectra, instead of a flat mean.

    point_weight: optional (P,) per-point (per-bin) weights. Used to up-weight
    sparse signal bins so they are not drowned by the floor bins in the
    per-point mean (essential for floor-dominated low-Z spectra); the
    per-spectrum loss becomes a weighted mean over points."""
    d = torch.clamp(pred - target, -4.0, 4.0)
    # valid bins: symmetric relative error in linear flux
    eps_valid = torch.abs(torch.pow(10.0, torch.where(valid, d, torch.zeros_like(d))) - 1.0)
    # floor bins: only penalise predictions above the floor
    d_floor = torch.clamp(torch.relu(pred - FLOOR), max=4.0)
    eps_floor = torch.pow(10.0, d_floor) - 1.0
    eps = torch.where(valid, eps_valid, eps_floor)
    if sample_weight is None and point_weight is None:
        return F.huber_loss(eps, torch.zeros_like(eps), delta=huber_delta)
    h = F.huber_loss(eps, torch.zeros_like(eps), delta=huber_delta,
                     reduction="none")                        # (B, P)
    if point_weight is not None:
        w = point_weight.view(1, -1)
        per = (h * w).sum(dim=1) / w.sum()                    # (B,)
    else:
        per = h.mean(dim=1)                                   # (B,)
    if sample_weight is not None:
        return (per * sample_weight).mean()
    return per.mean()


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
def evaluate(model, data, idx, device, batch=64, fixed_grid=False,
             return_bins=False, echunk=4096):
    """Per-spectrum mean relative error over non-empty bins, plus yields.

    The energy grid is evaluated in chunks of `echunk` points: the
    coordinate embedding is batch x points x embed_dim, which for wide
    configurations does not fit in GPU memory all at once."""
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
            pred = torch.cat(
                [model(temps, energy[lo:lo + echunk],
                       bins=torch.arange(lo, min(lo + echunk, len(energy)),
                                         device=device))
                 for lo in range(0, len(energy), echunk)], dim=1)
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
        "yield_01pct": float((mre <= 0.001).mean() * 100),
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
        finer_k=getattr(args, "finer_k", 10.0),
        use_fourier=bool(getattr(args, "use_fourier", 1)),
        line_dim=getattr(args, "line_dim", 16),
        line_hidden=getattr(args, "line_hidden", 128),
        line_t_freqs=getattr(args, "line_t_freqs", 0),
        line_t_fmax=getattr(args, "line_t_fmax", 64.0),
        film_t_freqs=getattr(args, "film_t_freqs", 0),
        film_t_fmax=getattr(args, "film_t_fmax", 64.0),
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
        # trend head and per-bin normalisation switchable for the
        # hyperparameter search
        combo=dict(use_linehead=str(getattr(args, "use_linehead", "auto"))
                   != "off",
                   use_trend=bool(getattr(args, "use_trend", 1)),
                   use_binnorm=bool(getattr(args, "use_binnorm", 0))),
    )[variant]
    line_ids = None
    if flags.get("use_linehead"):
        line_ids = find_line_bins(data)
        n_lines = int((line_ids >= 0).sum())
        # a fully ionised element (e.g. H over 0.5-20 keV) has no lines;
        # the line head would be useless and cannot build a 0-line
        # embedding, so auto-disable it and train a trunk-only continuum
        # model. Same model class, different config.
        lh_mode = str(getattr(args, "use_linehead", "auto"))
        min_lb = getattr(args, "min_line_bins", 32)
        if lh_mode == "auto" and n_lines < min_lb:
            print(f"line head auto-disabled: {n_lines} line bins < {min_lb} "
                  f"-> trunk-only continuum model", flush=True)
            flags = {**flags, "use_linehead": False}
            line_ids = None
        else:
            print(f"line head: {n_lines} line bins "
                  f"({100.0 * n_lines / data.n_bins:.1f}% of grid)", flush=True)
    bin_stats = None
    if flags.get("use_binnorm"):
        lf = torch.clamp(data.logflux[data.train_idx], min=FLOOR)
        bin_stats = (lf.mean(dim=0), lf.std(dim=0))
    return SpectralOperator(OperatorConfig(**{**kw, **flags}),
                            line_ids=line_ids, energy_grid=data.energy,
                            bin_stats=bin_stats)


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
    lr_min = getattr(args, "lr_min_frac", 0.0)
    schedule = getattr(args, "schedule", "cosine")
    decay_start = int((1.0 - getattr(args, "wsd_decay_frac", 0.15))
                      * args.steps)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        if schedule == "wsd":
            # warmup-stable-decay: flat until decay_start, then linear;
            # extending a run only re-pays the decay leg, not the schedule
            if step < decay_start:
                return 1.0
            p = (step - decay_start) / max(1, args.steps - decay_start)
            return max(lr_min, 1.0 - p * (1.0 - lr_min))
        p = (step - warmup) / max(1, args.steps - warmup)
        return max(lr_min, 0.5 * (1 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # Polyak-averaged weights: evaluated and saved instead of the live ones
    ema = None
    if getattr(args, "ema_decay", 0.0) > 0:
        from spexai.train.diagnostics import EMAWeights
        ema = EMAWeights(model, args.ema_decay)

    x_grid = None
    if not fixed_grid:
        x_grid = model.norm_energy(data.energy.to(device)).unsqueeze(-1)  # (M, 1)

    train_idx = data.train_idx
    target_all = data.logflux
    temps_all = data.temps
    # fixed train subset for the train-vs-val curve
    train_sub = train_idx[torch.linspace(0, len(train_idx) - 1,
                                         min(384, len(train_idx))).long()]

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
        if ema is not None:
            ema.update(model)

        if step % args.eval_every == 0 or step == args.steps:
            if ema is not None:
                ema.swap_in(model)
            val = evaluate(model, data, data.val_idx, device, fixed_grid=fixed_grid)
            trn = evaluate(model, data, train_sub, device, fixed_grid=fixed_grid)
            rec = {"step": step, "loss": float(loss.item()),
                   "elapsed_s": time.time() - t0,
                   "train_mre_mean": trn["mre_mean"],
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            print(f"[{args.variant}] step {step}/{args.steps} loss={rec['loss']:.4f} "
                  f"val MRE={val['mre_mean']:.4f} yield1%={val['yield_1pct']:.1f} "
                  f"yield0.1%={val['yield_01pct']:.1f} ({rec['elapsed_s']:.0f}s)",
                  flush=True)
            if val["yield_1pct"] >= best["val_yield_1pct"]:
                best = {"step": step, **{f"val_{k}": v for k, v in val.items()},
                        "val_yield_1pct": val["yield_1pct"]}
                torch.save({"state_dict": model.state_dict(),
                            "variant": args.variant,
                            "args": vars(args)},
                           os.path.join(args.outdir, f"{runname}.pt"))
            if ema is not None:
                ema.swap_out(model)

    with open(os.path.join(args.outdir, f"{runname}_history.json"), "w") as f:
        json.dump({"history": history, "best": best,
                   "params": model.count_parameters(),
                   "args": vars(args)}, f, indent=2)
    print(f"[{args.variant}] done in {(time.time()-t0)/60:.1f} min; "
          f"best val yield1%={best['val_yield_1pct']:.2f} at step {best.get('step')}",
          flush=True)
    if getattr(args, "diag_plots", 1):
        from spexai.train.diagnostics import run_diagnostics
        ckpt = torch.load(os.path.join(args.outdir, f"{runname}.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        run_diagnostics(model, data, history, args.outdir, runname,
                        device=device, spectra=not fixed_grid)
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
    ap.add_argument("--line_hidden", type=int, default=128)
    ap.add_argument("--line_t_freqs", type=int, default=0,
                    help="Fourier embedding of T in the line head "
                         "(0 = plain MLP conditioning)")
    ap.add_argument("--line_t_fmax", type=float, default=64.0)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                    help="wsd = warmup-stable-decay: flat LR, linear decay "
                         "over the last --wsd_decay_frac of the run")
    ap.add_argument("--wsd_decay_frac", type=float, default=0.15)
    ap.add_argument("--use_trend", type=int, default=1,
                    help="combo variant only: include the trend head (1/0)")
    ap.add_argument("--use_binnorm", type=int, default=0,
                    help="combo variant only: per-bin target normalisation (1/0)")
    ap.add_argument("--fg_hidden", type=int, default=512)
    ap.add_argument("--fg_layers", type=int, default=4)
    ap.add_argument("--w_sobolev", type=float, default=1e-3)
    ap.add_argument("--w_log", type=float, default=0.1)
    ap.add_argument("--curriculum_frac", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None,
                    help="checkpoint/history name (defaults to variant)")
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="Polyak weight averaging (0 disables); checkpoints "
                         "hold the averaged weights")
    ap.add_argument("--lr_min_frac", type=float, default=0.02,
                    help="cosine schedule floor as a fraction of --lr")
    ap.add_argument("--diag_plots", type=int, default=1,
                    help="write history + spectrum diagnostics at the end")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
