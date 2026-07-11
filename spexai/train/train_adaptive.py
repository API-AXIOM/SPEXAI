"""Adaptive-data training of the (unbroadened) temperature emulator.

Test of dynamically adding training data where the emulator needs it,
generated with the best classical interpolator (per-bin PCHIP in log T,
see scripts/baseline_interpolation.py). The model is the t04 HPO winner
(combo variant: line head on, trend head + per-bin normalisation on).
Three arms, selected via --mode:

  baseline  - plain training on the (optionally subsampled) SPEX grid.
  reweight  - no new data; existing grid spectra are sampled with
              probability proportional to their running training error
              (prioritized sampling). Controls for "attention" effects.
  adaptive  - reweight + gated synthetic generation: every
              --acquire_every steps, new temperatures are drawn RAD-style
              (density ~ error^k) from grid intervals whose leave-one-out
              interpolation error is below --gate_thresh, PCHIP spectra
              are generated there and mixed into batches with weight
              --synth_weight.

Validation and the saved-checkpoint criterion use ONLY original SPEX
spectra (the frozen val split); synthetic spectra never enter any split.
Intervals rejected by the trust gate are recorded in the history file as
candidates for real SPEX runs.

Use --n_train to subsample the training grid (evenly in log T): on the
full ~11.5k grid PCHIP is near-exact but has little to add; the
interesting regime for this test is a sparse grid (e.g. --n_train 300).

    python -m spexai.train.train_adaptive --mode adaptive --n_train 300
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from spexai.train.train_operator import (FLOOR, SpectrumData, evaluate,
                                         make_variant, relative_error_loss)


def loo_interpolation_error(lt, Y, chunk=512):
    """Leave-one-out linear interpolation error per grid point: drop each
    interior point, predict it from its neighbours, mean relative flux
    error over valid bins. Proxy for where the T-dependence is too sharp
    for interpolation (and hence where PCHIP generation is untrusted)."""
    n = len(lt)
    loo = np.zeros(n, dtype=np.float64)
    for lo in range(1, n - 1, chunk):
        hi = min(lo + chunk, n - 1)
        j = np.arange(lo, hi)
        w = ((lt[j] - lt[j - 1]) / (lt[j + 1] - lt[j - 1]))[:, None]
        pred = Y[j - 1] * (1 - w) + Y[j + 1] * w
        target = Y[j]
        valid = target > FLOOR
        d = np.clip(pred - np.clip(target, FLOOR, None), -4, 4)
        eps = np.where(valid, np.abs(10.0 ** d - 1.0), 0.0)
        loo[j] = eps.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    loo[0], loo[-1] = loo[1], loo[-2]
    return loo


def pchip_generate(lt_grid, Y, lt_new, half=4):
    """Per-bin PCHIP log-flux spectra at new log-temperatures, fit on a
    local stencil of the training grid (train rows only, never val/test)."""
    from scipy.interpolate import PchipInterpolator
    out = np.empty((len(lt_new), Y.shape[1]), dtype=np.float32)
    for i, lt in enumerate(lt_new):
        j = int(np.searchsorted(lt_grid, lt))
        lo, hi = max(0, j - half), min(len(lt_grid), j + half)
        out[i] = PchipInterpolator(lt_grid[lo:hi], Y[lo:hi],
                                   axis=0)(lt).astype(np.float32)
    return out


def per_sample_mre(pred, target):
    valid = target > FLOOR
    d = torch.clamp(pred - torch.clamp(target, min=FLOOR), -4.0, 4.0)
    eps = torch.where(valid, torch.abs(torch.pow(10.0, d) - 1.0),
                      torch.zeros_like(d))
    return eps.sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def train(args):
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data = SpectrumData(args.cachedir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # training grid: train split sorted in T, optionally subsampled
    temps_np = data.temps.numpy()
    tr = data.train_idx.numpy()
    assert (len(np.intersect1d(tr, data.val_idx.numpy())) == 0
            and len(np.intersect1d(tr, data.test_idx.numpy())) == 0)
    tr = tr[np.argsort(temps_np[tr])]
    if args.n_train and args.n_train < len(tr):
        tr = tr[np.linspace(0, len(tr) - 1, args.n_train).astype(int)]
    grid = torch.from_numpy(tr).long()
    lt_grid = np.log10(temps_np[tr].astype(np.float64))
    n_grid = len(grid)
    # line bins and per-bin normalisation must only see the subsampled grid
    data.train_idx = grid

    # trust gate: LOO interpolation error on the actual training grid
    Y_grid = np.clip(data.logflux[grid].numpy(), FLOOR, None)
    print(f"computing LOO interpolation error on {n_grid} grid points ...",
          flush=True)
    loo = loo_interpolation_error(lt_grid, Y_grid)
    eligible = np.maximum(loo[:-1], loo[1:]) < args.gate_thresh
    print(f"trust gate: {eligible.mean() * 100:.1f}% of {n_grid - 1} "
          f"intervals eligible (LOO median {np.median(loo):.2e}, "
          f"max {loo.max():.2e})", flush=True)

    # synthetic pool: PCHIP-generated log-flux rows + their temperatures
    pool_flux = torch.empty(0, data.n_bins)
    pool_temps = torch.empty(0)

    model = make_variant("combo", data, args).to(device)
    print(f"adaptive[{args.mode}] params={model.count_parameters():,} "
          f"grid={n_grid} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(1, int(0.02 * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    x_grid = model.norm_energy(data.energy.to(device)).unsqueeze(-1)  # (M, 1)

    # running per-grid-point error estimate (optimistic init -> coverage)
    ema = torch.full((n_grid,), args.ema_init)
    prioritize = args.mode in ("reweight", "adaptive")

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or args.mode
    best = {"val_yield_1pct": -1.0}
    history, acquired = [], []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()

        # --- acquisition round ------------------------------------------
        if (args.mode == "adaptive" and step > args.acquire_warmup
                and step % args.acquire_every == 0
                and len(pool_flux) < args.max_synth):
            interval_err = 0.5 * (ema[:-1] + ema[1:]).numpy()
            dens = np.where(eligible, interval_err ** args.rad_k, 0.0)
            if dens.sum() > 0:
                pick = np.random.choice(n_grid - 1, size=args.acquire_n,
                                        p=dens / dens.sum())
                u = np.random.rand(args.acquire_n)
                lt_new = lt_grid[pick] + u * (lt_grid[pick + 1]
                                              - lt_grid[pick])
                new = torch.from_numpy(pchip_generate(lt_grid, Y_grid, lt_new))
                pool_flux = torch.cat([pool_flux, new])[-args.max_synth:]
                pool_temps = torch.cat(
                    [pool_temps,
                     torch.from_numpy(10.0 ** lt_new).float()])[-args.max_synth:]
                acquired += [{"step": step, "logT": float(x)}
                             for x in lt_new]

        # --- batch assembly ---------------------------------------------
        ns = 0
        if args.mode == "adaptive" and len(pool_flux) > 0:
            ns = min(int(round(args.synth_frac * args.batch)), len(pool_flux))
        nr = args.batch - ns

        if prioritize:
            p = (ema + 1e-4) ** args.pr_alpha
            p = (1 - args.pr_mix) / n_grid + args.pr_mix * p / p.sum()
            pos = torch.multinomial(p, nr, replacement=True)
        else:
            pos = torch.randint(n_grid, (nr,))
        target = data.logflux[grid[pos]]
        temps = data.temps[grid[pos]]
        if ns > 0:
            sidx = torch.randint(len(pool_flux), (ns,))
            target = torch.cat([target, pool_flux[sidx]])
            temps = torch.cat([temps, pool_temps[sidx]])

        pts, _ = torch.sort(torch.randperm(data.n_bins)[:args.points])
        target = target[:, pts].to(device)
        alpha = min(1.0, step / (args.curriculum_frac * args.steps))
        tnorm = model.norm_temp(temps.to(device)).view(-1, 1)
        pred = model.forward_norm(
            tnorm, x_grid[pts].unsqueeze(0).expand(args.batch, -1, -1),
            alpha=alpha, bins=pts.to(device))

        target_c = torch.clamp(target, min=FLOOR)
        loss = relative_error_loss(pred[:nr], target_c[:nr],
                                   target[:nr] > FLOOR)
        if ns > 0:
            loss = loss + args.synth_weight * relative_error_loss(
                pred[nr:], target_c[nr:], target[nr:] > FLOOR)
        w_log = args.w_log * max(0.0, 1.0 - step / (0.2 * args.steps))
        if w_log > 0:
            loss = loss + w_log * F.mse_loss(pred, target_c)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            mre = per_sample_mre(pred[:nr], target[:nr]).cpu()
            ema[pos] = (1 - args.ema_beta) * ema[pos] + args.ema_beta * mre

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, data, data.val_idx, device)
            rec = {"step": step, "loss": float(loss.item()),
                   "pool_n": len(pool_flux), "ema_mean": float(ema.mean()),
                   "elapsed_s": time.time() - t0,
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            print(f"[{args.mode}] step {step}/{args.steps} "
                  f"loss={rec['loss']:.4f} val MRE={val['mre_mean']:.4f} "
                  f"yield1%={val['yield_1pct']:.2f} pool={len(pool_flux)} "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
            if val["yield_1pct"] >= best["val_yield_1pct"]:
                best = {"step": step,
                        **{f"val_{k}": v for k, v in val.items()}}
                torch.save({"state_dict": model.state_dict(),
                            "variant": "combo", "args": vars(args)},
                           os.path.join(args.outdir, f"{tag}.pt"))

    rejected = [{"logT_lo": float(lt_grid[j]), "logT_hi": float(lt_grid[j + 1]),
                 "loo": float(max(loo[j], loo[j + 1]))}
                for j in np.where(~eligible)[0]]
    with open(os.path.join(args.outdir, f"{tag}_history.json"), "w") as f:
        json.dump({"history": history, "best": best,
                   "params": model.count_parameters(),
                   "n_grid": n_grid, "acquired": acquired,
                   "gate_rejected_intervals": rejected,
                   "loo_median": float(np.median(loo)),
                   "args": vars(args)}, f, indent=2)
    print(f"[{args.mode}] done in {(time.time()-t0)/60:.1f} min; best "
          f"val yield1%={best['val_yield_1pct']:.2f} "
          f"(synthetic spectra generated: {len(acquired)})", flush=True)
    return model, best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/adaptive")
    ap.add_argument("--tag", default=None,
                    help="checkpoint/history name (defaults to mode)")
    ap.add_argument("--mode", default="adaptive",
                    choices=["baseline", "reweight", "adaptive"])
    ap.add_argument("--n_train", type=int, default=300,
                    help="subsample training grid (0 = full grid)")
    # model/optimiser defaults are the t04 HPO winner
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
    ap.add_argument("--line_dim", type=int, default=16)
    ap.add_argument("--use_trend", type=int, default=1)
    ap.add_argument("--use_binnorm", type=int, default=1)
    ap.add_argument("--w_log", type=float, default=0.1)
    ap.add_argument("--curriculum_frac", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    # acquisition
    ap.add_argument("--gate_thresh", type=float, default=0.01,
                    help="max LOO interpolation error for synthetic "
                         "generation in an interval")
    ap.add_argument("--acquire_every", type=int, default=500)
    ap.add_argument("--acquire_warmup", type=int, default=2000)
    ap.add_argument("--acquire_n", type=int, default=32,
                    help="synthetic spectra per acquisition round")
    ap.add_argument("--max_synth", type=int, default=2048)
    ap.add_argument("--rad_k", type=float, default=2.0,
                    help="exponent on the error density (RAD)")
    ap.add_argument("--synth_frac", type=float, default=0.25,
                    help="fraction of each batch drawn from the pool")
    ap.add_argument("--synth_weight", type=float, default=0.5,
                    help="loss weight of synthetic samples")
    ap.add_argument("--pr_alpha", type=float, default=1.0)
    ap.add_argument("--pr_mix", type=float, default=0.5,
                    help="prioritized fraction of real-sample draws")
    ap.add_argument("--ema_beta", type=float, default=0.1)
    ap.add_argument("--ema_init", type=float, default=1.0)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
