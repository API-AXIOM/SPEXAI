"""Broadened (T, v) emulator, second attempt (option 2, revised).

Changes relative to train_broadened, following the design review:

  1. Gaussian line head: the known line list (find_line_bins) is kept,
     with learned per-line integrated fluxes conditioned on T ONLY
     (broadening conserves line flux), deposited as analytic Gaussian
     profiles of width sigma_u = v/c in ln E. The network is handed the
     convolution structure instead of having to learn it; the trunk only
     models continuum + pseudo-continuum.
  2. Line-aware point sampling: half of each batch's grid points are
     drawn around line centers with offsets ~ N(0, sigma(v')) at random
     velocities, so line cores are supervised at every scale.
  3. Validation covers the full velocity range (50 and 1000 km/s
     included), so checkpoint selection sees the hardest regime.
  4. Wing masking: faint-wing bins (below REL_FLOOR of the spectrum
     peak, where float FFT noise corrupts the labels) are excluded from
     the relative-error term; predictions there are only penalised if
     they exceed the cutoff by more than --wing_margin dex.

Trunk hyperparameters remain the t04 HPO winner.

    python -m spexai.train.train_broadened2 --steps 20000
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spexai.broadening import fft_broaden
from spexai.operator import CondNet, OperatorConfig, SpectralOperator
from spexai.train.train_broadened import (REL_FLOOR, BroadenedTargets,
                                          build_parser as broadened_parser,
                                          norm_velocity, sample_velocity)
from spexai.train.train_operator import FLOOR, find_line_bins
from spexai.data import SpectrumData

C_KMS = 299792.458
LN10 = math.log(10.0)


class GaussianLineOperator(nn.Module):
    """SpectralOperator trunk (theta = [T, v]) plus an analytic Gaussian
    line head: per-line integrated flux from (embedding, T), profile width
    sigma = v/c exactly - the same kernel that generates the targets."""

    def __init__(self, cfg, centers, line_energies, bin_stats=None,
                 n_neighbors=32, line_dim=16, line_hidden=128, floor=-12.0):
        super().__init__()
        self.trunk = SpectralOperator(cfg, energy_grid=centers,
                                      bin_stats=bin_stats)
        self.n_neighbors = n_neighbors
        self.floor = floor
        self.register_buffer("centers", centers.clone())
        self.register_buffer("line_energies", line_energies.clone())
        n_lines = len(line_energies)
        self.embed = nn.Embedding(n_lines, line_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.cond = CondNet(1, cfg.cond_hidden, cfg.cond_layers,
                            cfg.cond_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(line_dim + cfg.cond_hidden, line_hidden), nn.GELU(),
            nn.Linear(line_hidden, line_hidden), nn.GELU(),
            nn.Linear(line_hidden, 1))
        # K nearest lines per grid bin, precomputed (u = log10 E)
        u_bins = torch.log10(centers).double()
        u_lines, order = torch.sort(torch.log10(line_energies).double())
        K = min(n_neighbors, n_lines)
        pos = torch.searchsorted(u_lines, u_bins)
        cand = pos.unsqueeze(1) + torch.arange(-K, K)
        cand = cand.clamp(0, n_lines - 1)
        du = u_bins.unsqueeze(1) - u_lines[cand]
        near = du.abs().topk(K, largest=False).indices
        cand = cand.gather(1, near)
        self.register_buffer("nbr_idx", order[cand].int())
        self.register_buffer("nbr_du",
                             du.gather(1, near).float() * LN10)  # in ln E

    def line_flux_log10(self, tnorm_T, lines):
        """log10 integrated flux of the given line slots, shape (B, U)."""
        emb = self.embed(lines)
        c = self.cond(tnorm_T)
        h = torch.cat([emb.unsqueeze(0).expand(c.shape[0], -1, -1),
                       c.unsqueeze(1).expand(-1, len(lines), -1)], dim=-1)
        return self.mlp(h).squeeze(-1) + self.floor

    def forward(self, theta, velocity, pts, alpha=1.0):
        """theta: (B, 2) normalised [T, v]; velocity: (B,) km/s (physical,
        sets the profile width); pts: (P,) uniform-grid bin indices.
        Returns log10 flux density at the sampled bins, (B, P)."""
        x = self.trunk.norm_energy(self.centers[pts]).view(1, -1, 1)
        trunk_log = self.trunk.forward_norm(
            theta, x.expand(theta.shape[0], -1, -1), alpha=alpha, bins=pts)

        lidx = self.nbr_idx[pts].long()                     # (P, K)
        uniq, inv = torch.unique(lidx, return_inverse=True)
        amp = self.line_flux_log10(theta[:, :1], uniq)      # (B, U)
        A = amp[:, inv.view(*lidx.shape)]                   # (B, P, K)
        sigma = (velocity / C_KMS).view(-1, 1, 1)           # in ln E
        prof = torch.exp(-0.5 * (self.nbr_du[pts] / sigma) ** 2) \
            / (sigma * math.sqrt(2.0 * math.pi))            # per unit ln E
        dens_line = (torch.pow(10.0, A) * prof).sum(-1) \
            / self.centers[pts]                             # per keV
        return torch.logaddexp(trunk_log * LN10,
                               torch.log(dens_line + 1e-37)) / LN10

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WingTargets(BroadenedTargets):
    """Broadened targets WITHOUT the wing zeroing; returns the log
    density, the wing mask, and each sample's log10 density cutoff."""

    def __call__(self, rows, velocity):
        flux = self.flux_uni[rows].to(self.device)
        broad = fft_broaden(flux, self.dlx, velocity)
        peak = broad.max(dim=1, keepdim=True).values
        wing = broad <= REL_FLOOR * peak
        dens = broad / self.widths
        cut_log = (torch.log10(REL_FLOOR * peak)
                   - torch.log10(self.widths).unsqueeze(0))
        return torch.log10(torch.clamp(dens, min=1e-30)), wing, cut_log


def masked_loss(pred, target, wing, cut_log, margin, huber_delta=1.0):
    """Relative-error Huber on trusted bins; wing bins only penalise
    predictions more than `margin` dex above the label cutoff."""
    valid = (target > FLOOR) & ~wing
    d = torch.clamp(pred - torch.clamp(target, min=FLOOR), -4.0, 4.0)
    eps_valid = torch.abs(torch.pow(
        10.0, torch.where(valid, d, torch.zeros_like(d))) - 1.0)
    d_wing = torch.clamp(torch.relu(pred - (cut_log + margin)), max=4.0)
    eps = torch.where(valid, eps_valid, torch.pow(10.0, d_wing) - 1.0)
    return F.huber_loss(eps, torch.zeros_like(eps), delta=huber_delta)


def sample_points(model, n_bins, n_pts, line_frac, dlx, vmin, vmax, device):
    """Half uniform, half around line centers with offsets ~ N(0, sigma(v'))
    at log-uniform velocities, so cores are supervised at every width."""
    n_line = int(line_frac * n_pts)
    pts = torch.randperm(n_bins, device=device)[:n_pts - n_line]
    if n_line:
        li = torch.randint(len(model.line_energies), (n_line,), device=device)
        vs = sample_velocity(n_line, vmin, vmax, device)
        u = (torch.log10(model.line_energies[li])
             + torch.randn(n_line, device=device) * (vs / C_KMS) / LN10)
        u0 = torch.log10(model.centers[0])
        j = torch.clamp(((u - u0) / dlx).round().long(), 0, n_bins - 1)
        pts = torch.cat([pts, j])
    return torch.sort(torch.unique(pts)).values


@torch.no_grad()
def evaluate(model, targets, data, idx, vlist, vmin, vmax, device,
             batch=64, echunk=8192):
    model.eval()
    mre = []
    for v in vlist:
        for i in range(0, len(idx), batch):
            sel = idx[i:i + batch]
            vv = torch.full((len(sel),), v, device=device)
            target, wing, _ = targets(sel, vv)
            tn = model.trunk.norm_temp(data.temps[sel].to(device))
            theta = torch.stack([tn, norm_velocity(vv, vmin, vmax)], dim=1)
            pred = torch.cat(
                [model(theta, vv,
                       torch.arange(lo, min(lo + echunk, targets.n_bins),
                                    device=device))
                 for lo in range(0, targets.n_bins, echunk)], dim=1)
            valid = (target > FLOOR) & ~wing
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

    targets = WingTargets(data, args.dlx, args.cachedir, device)
    line_ids = find_line_bins(data)
    line_energies = data.energy[line_ids >= 0]
    print(f"gaussian line head: {len(line_energies)} lines", flush=True)

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
        rows = data.train_idx[torch.linspace(0, len(data.train_idx) - 1,
                                             args.binnorm_samples).long()]
        vv = sample_velocity(len(rows), args.vmin, args.vmax, device)
        lf, wing, _ = targets(rows, vv)
        lf = torch.where(wing, torch.full_like(lf, FLOOR), lf)
        lf = torch.clamp(lf, min=FLOOR).cpu()
        bin_stats = (lf.mean(dim=0), lf.std(dim=0))

    model = GaussianLineOperator(
        cfg, targets.centers.cpu(), line_energies, bin_stats=bin_stats,
        n_neighbors=args.n_neighbors, line_dim=args.line_dim).to(device)
    print(f"broadened2: params={model.count_parameters():,} "
          f"K={targets.n_bins} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(1, int(0.02 * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

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
        target_full, wing_full, cut_log = targets(sel, vv)
        pts = sample_points(model, targets.n_bins, args.points,
                            args.line_frac, args.dlx, args.vmin, args.vmax,
                            device)
        target, wing = target_full[:, pts], wing_full[:, pts]
        tn = model.trunk.norm_temp(data.temps[sel].to(device))
        theta = torch.stack([tn, norm_velocity(vv, args.vmin, args.vmax)],
                            dim=1)
        alpha = min(1.0, step / (args.curriculum_frac * args.steps))
        pred = model(theta, vv, pts, alpha=alpha)

        loss = masked_loss(pred, target, wing, cut_log[:, pts],
                           args.wing_margin)
        w_log = args.w_log * max(0.0, 1.0 - step / (0.2 * args.steps))
        if w_log > 0:
            trusted = torch.where(wing, pred.detach(),
                                  torch.clamp(target, min=FLOOR))
            loss = loss + w_log * F.mse_loss(pred, trusted)

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
            print(f"[broadened2] step {step}/{args.steps} "
                  f"loss={rec['loss']:.4f} val MRE={val['mre_mean']:.4f} "
                  f"yield1%={val['yield_1pct']:.2f} "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
            if val["yield_1pct"] >= best["val_yield_1pct"]:
                best = {"step": step,
                        **{f"val_{k}": v for k, v in val.items()}}
                torch.save({"state_dict": model.state_dict(),
                            "config": cfg.__dict__, "args": vars(args),
                            "n_lines": len(line_energies)},
                           os.path.join(args.outdir, f"{args.tag}.pt"))

    with open(os.path.join(args.outdir, f"{args.tag}_history.json"), "w") as f:
        json.dump({"history": history, "best": best,
                   "params": model.count_parameters(),
                   "args": vars(args)}, f, indent=2)
    print(f"[broadened2] done in {(time.time()-t0)/60:.1f} min; best "
          f"val yield1%={best['val_yield_1pct']:.2f}", flush=True)
    return model, best


def load_broadened2(path, data):
    """Rebuild a GaussianLineOperator from a train_broadened2 checkpoint."""
    from spexai.broadening import uniform_log_edges
    from spexai.operator import edges_from_centers
    b = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OperatorConfig(**b["config"])
    edges = edges_from_centers(data.energy)
    uni = uniform_log_edges(float(edges[0]), float(edges[-1]),
                            b["args"]["dlx"])
    centers = torch.sqrt(uni[:-1] * uni[1:])
    line_ids = find_line_bins(data)
    stats = ((b["state_dict"]["trunk.bn_mu"], b["state_dict"]["trunk.bn_sigma"])
             if cfg.use_binnorm else None)
    model = GaussianLineOperator(
        cfg, centers, data.energy[line_ids >= 0], bin_stats=stats,
        n_neighbors=b["args"]["n_neighbors"], line_dim=b["args"]["line_dim"])
    model.load_state_dict(b["state_dict"])
    return model.eval(), b["args"]


def build_parser():
    ap = broadened_parser()
    ap.set_defaults(
        outdir="/Users/danielahuppenkothen/work/data/spexai/runs/element26/broadened2",
        tag="broadened2",
        eval_velocities=[50.0, 100.0, 250.0, 600.0, 1000.0])
    ap.add_argument("--line_dim", type=int, default=16)
    ap.add_argument("--n_neighbors", type=int, default=32,
                    help="lines contributing to each grid bin")
    ap.add_argument("--line_frac", type=float, default=0.5,
                    help="fraction of loss points drawn around line centers")
    ap.add_argument("--wing_margin", type=float, default=1.0,
                    help="dex above the wing cutoff before predictions "
                         "are penalised")
    ap.add_argument("--binnorm_samples", type=int, default=1024)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
