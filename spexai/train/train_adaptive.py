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
import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from spexai.train.train_operator import (FLOOR, SpectrumData, evaluate,
                                         find_edge_bins, find_line_bins,
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
    # --- speed knobs (effective on CUDA; no-ops elsewhere) ---
    if device == "cuda":
        torch.set_float32_matmul_precision("high")   # TF32 matmuls (Ampere+)
    use_amp = bool(args.amp) and device == "cuda"    # bf16 autocast on the fwd
    use_data_gpu = bool(args.data_on_gpu) and device == "cuda"  # data resident
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
          f"grid={n_grid} device={device} (amp={use_amp} "
          f"data_on_gpu={use_data_gpu} compile={bool(args.compile)})", flush=True)
    if args.init_from:
        # warm start: load weights from a previous checkpoint (same
        # architecture / cache) so training continues from those instead of
        # from scratch. Optimizer/schedule/step still start fresh -- this is a
        # warm start, not a seamless resume.
        # expanduser so a `~` that slipped through quoting (e.g. inside a
        # double-quoted --train_flags) still resolves.
        ck = torch.load(os.path.expanduser(args.init_from),
                        map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"],
                                                    strict=False)
        print(f"warm-started from {args.init_from} "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)",
              flush=True)
    if args.compile:
        # compile the hot path only (training calls forward_norm, not
        # forward); compiling the module would also prefix state_dict keys
        # with _orig_mod. and break checkpoint loading.
        model.forward_norm = torch.compile(model.forward_norm)

    # optimizer: adamw (default) | schedulefree | muon. schedule-free and
    # its own averaging replace the external LR schedule and the EMA;
    # muon drives 2-D matrices with an AdamW companion for the rest.
    warmup = max(1, int(0.02 * args.steps))
    decay_start = int((1.0 - args.wsd_decay_frac) * args.steps)
    decay_len = max(1, args.steps - decay_start)
    # early stopping may move the WSD decay earlier; `sched_state` lets the
    # LR schedule read a decay start chosen at run time
    sched_state = {"decay_from": None}
    is_sf = args.optimizer == "schedulefree"
    aux_opt = None
    if args.optimizer == "adamw":
        if args.mup:
            from spexai.train.optim import mup_param_groups
            groups = mup_param_groups(model, args.lr, args.mup_base_width)
            opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    elif is_sf:
        try:
            from schedulefree import AdamWScheduleFree
        except ImportError:
            raise SystemExit("optimizer=schedulefree needs `pip install "
                             "schedulefree`; arm skipped by the driver if absent")
        opt = AdamWScheduleFree(model.parameters(), lr=args.lr,
                                weight_decay=1e-4, warmup_steps=warmup)
    elif args.optimizer == "muon":
        from spexai.train.optim import Muon, split_matrix_params
        mats, rest = split_matrix_params(model)
        opt = Muon(mats, lr=args.lr_muon)
        aux_opt = torch.optim.AdamW(rest, lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError(args.optimizer)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        if args.schedule == "wsd":
            ds = (sched_state["decay_from"] if sched_state["decay_from"]
                  is not None else decay_start)
            if step < ds:
                return 1.0
            p = min(1.0, (step - ds) / decay_len)
            return max(args.lr_min_frac, 1.0 - p * (1.0 - args.lr_min_frac))
        p = (step - warmup) / max(1, args.steps - warmup)
        return max(args.lr_min_frac, 0.5 * (1 + math.cos(math.pi * p)))

    # schedule-free carries its own warmup/averaging and takes no scheduler
    scheds = ([] if is_sf else
              [torch.optim.lr_scheduler.LambdaLR(o, lr_at)
               for o in (opt, aux_opt) if o is not None])
    x_grid = model.norm_energy(data.energy.to(device)).unsqueeze(-1)  # (M, 1)

    wema = None
    if args.ema_decay > 0 and not is_sf:
        from spexai.train.diagnostics import EMAWeights
        wema = EMAWeights(model, args.ema_decay)
    train_sub = grid[torch.linspace(0, n_grid - 1, min(384, n_grid)).long()]

    # signal-aware point sampling: guarantee supervision on the sparse
    # non-floor bins (spectral lines + recombination edges). Uniform sampling
    # almost never hits a line on floor-dominated low-Z spectra (Li/Be/B have
    # < 1.2% of bins above floor), so those bins are both forced into every
    # batch and, via `point_weight`, given `signal_frac` of the loss so they
    # are not drowned by the floor bins in the per-point mean.
    signal_pool = torch.empty(0, dtype=torch.long)
    if args.signal_frac > 0:
        pools = []
        line_bins = torch.nonzero(find_line_bins(data) >= 0).flatten()
        if len(line_bins):
            pools.append(line_bins)
        if args.use_edges != "off":
            core = find_edge_bins(data)
            if len(core):
                r = args.edge_radius
                pools.append(torch.unique(torch.clamp(
                    core[:, None] + torch.arange(-r, r + 1),
                    0, data.n_bins - 1)))
        if pools:
            signal_pool = torch.unique(torch.cat(pools))
            print(f"signal-aware sampling: {len(signal_pool)} signal bins "
                  f"({100 * len(signal_pool) / data.n_bins:.2f}% of grid), "
                  f"target loss share {args.signal_frac:.0%}", flush=True)
    is_signal = torch.zeros(data.n_bins, dtype=torch.bool)
    is_signal[signal_pool] = True

    # running per-grid-point error estimate (optimistic init -> coverage)
    ema = torch.full((n_grid,), args.ema_init)
    prioritize = args.mode in ("reweight", "adaptive")

    # Keep sampling/gather tensors resident on `ddev` so the (tiny) model is
    # not stalled by per-step host<->device transfers, CPU-side gather, OR
    # per-step GPU->CPU syncs. The prioritized-sampling error estimate (ema),
    # the multinomial draw, and the ema update all live on `ddev`, so nothing
    # in the step reads a GPU result back to Python (which would serialize the
    # otherwise-async CPU/GPU pipeline). Forward inputs are moved to the
    # model's `device` (a no-op when they already match, i.e. data_on_gpu on).
    ddev = torch.device(device if use_data_gpu else "cpu")
    # Separate GPU-resident copies for the training loop; `data.*` is left on
    # the CPU so numpy-based eval/diagnostics (find_line_bins, plot_spectra,
    # ...) keep working. When data_on_gpu is off these alias `data.*` (no copy).
    train_flux = data.logflux.to(ddev) if use_data_gpu else data.logflux
    train_temps = data.temps.to(ddev) if use_data_gpu else data.temps
    train_grid = grid.to(ddev)
    signal_pool = signal_pool.to(ddev)
    is_signal = is_signal.to(ddev)
    ema = ema.to(ddev)

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or args.mode
    best = {"val_mre_mean": float("inf")}   # checkpoint on lowest val MRE
    history, acquired = [], []
    # early stopping on a smoothed validation-MRE plateau
    stop_at = args.steps
    mre_hist, best_smoothed, no_improve = [], float("inf"), 0
    early_stop_off = False   # set if a plateau is reached while still poor
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()

        # --- acquisition round ------------------------------------------
        if (args.mode == "adaptive" and step > args.acquire_warmup
                and step % args.acquire_every == 0
                and len(pool_flux) < args.max_synth):
            interval_err = 0.5 * (ema[:-1] + ema[1:]).cpu().numpy()
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

        w_real = None
        if prioritize:
            p = (ema + 1e-4) ** args.pr_alpha
            p = (1 - args.pr_mix) / n_grid + args.pr_mix * p / p.sum()
            pos = torch.multinomial(p, nr, replacement=True)
            if args.pr_correct > 0:
                # InfoBatch-style importance-sampling correction: weight
                # each draw by (uniform / sampling prob)^beta so the
                # expected gradient stays unbiased despite oversampling
                w = ((1.0 / n_grid) / p[pos]) ** args.pr_correct
                w_real = (w / w.mean()).to(device)
        else:
            pos = torch.randint(n_grid, (nr,), device=ddev)
        gidx = train_grid[pos]    # GPU-resident copies; pos already on ddev
        target = train_flux[gidx]
        temps = train_temps[gidx]
        if ns > 0:
            sidx = torch.randint(len(pool_flux), (ns,))
            target = torch.cat([target, pool_flux[sidx].to(ddev)])
            temps = torch.cat([temps, pool_temps[sidx].to(ddev)])

        # draw a FIXED number of points with randint (on `ddev`) -- avoids a
        # per-step randperm(n_bins) on the CPU and keeps shapes static for
        # torch.compile -- guaranteeing the signal bins are represented. Rare
        # duplicate bins are harmless (the loss is a mean over points).
        if len(signal_pool):
            n_sig = min(len(signal_pool), int(args.signal_frac * args.points))
            sig_pts = signal_pool[torch.randint(len(signal_pool), (n_sig,),
                                                device=ddev)]
            rand_pts = torch.randint(data.n_bins, (args.points - n_sig,),
                                     device=ddev)
            pts = torch.sort(torch.cat([rand_pts, sig_pts])).values
        else:
            pts = torch.sort(torch.randint(data.n_bins, (args.points,),
                                           device=ddev)).values

        # per-point loss weight: hold `signal_frac` of the loss on the signal
        # bins regardless of how few of them there are (decouples influence
        # from count, so it self-tunes across Be's ~6 and B's ~292 bins).
        point_weight = None
        if len(signal_pool) and 0 < args.signal_frac < 1:
            sig_mask = is_signal[pts]
            # keep the counts as on-device tensors (no int()/.item()) so this
            # doesn't force a per-step GPU->CPU sync; clamps guard the ratio.
            n_s = sig_mask.sum().clamp(min=1)
            n_r = (sig_mask.numel() - n_s).clamp(min=1)
            w_sig = (args.signal_frac / (1 - args.signal_frac)) * (n_r / n_s)
            point_weight = torch.where(
                sig_mask, w_sig, torch.ones((), device=ddev)).to(device)
        pts_d = pts.to(device)
        target = target[:, pts].to(device)
        alpha = min(1.0, step / (args.curriculum_frac * args.steps))
        tnorm = model.norm_temp(temps.to(device)).view(-1, 1)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
              else contextlib.nullcontext()):
            pred = model.forward_norm(
                tnorm, x_grid[pts_d].unsqueeze(0).expand(args.batch, -1, -1),
                alpha=alpha, bins=pts_d)
        pred = pred.float()   # keep the loss / 10**() in fp32 for stability

        target_c = torch.clamp(target, min=FLOOR)
        loss = relative_error_loss(pred[:nr], target_c[:nr],
                                   target[:nr] > FLOOR, sample_weight=w_real,
                                   point_weight=point_weight)
        if ns > 0:
            loss = loss + args.synth_weight * relative_error_loss(
                pred[nr:], target_c[nr:], target[nr:] > FLOOR,
                point_weight=point_weight)
        w_log = args.w_log * max(0.0, 1.0 - step / (0.2 * args.steps))
        if w_log > 0:
            loss = loss + w_log * F.mse_loss(pred, target_c)

        if is_sf:
            opt.train()
        opt.zero_grad()
        if aux_opt is not None:
            aux_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if aux_opt is not None:
            aux_opt.step()
        for s in scheds:
            s.step()
        if wema is not None:
            wema.update(model)

        with torch.no_grad():
            # keep the ema update on `ddev` (no .cpu()) so it doesn't stall the
            # async CPU/GPU pipeline every step; ema, pos and mre are all there.
            mre = per_sample_mre(pred[:nr], target[:nr]).to(ddev)
            ema[pos] = (1 - args.ema_beta) * ema[pos] + args.ema_beta * mre

        if step % args.eval_every == 0 or step == args.steps:
            if wema is not None:
                wema.swap_in(model)
            if is_sf:
                opt.eval()  # swap params to the schedule-free average
            val = evaluate(model, data, data.val_idx, device)
            trn = evaluate(model, data, train_sub, device)
            rec = {"step": step, "loss": float(loss.item()),
                   "pool_n": len(pool_flux), "ema_mean": float(ema.mean()),
                   "elapsed_s": time.time() - t0,
                   "train_mre_mean": trn["mre_mean"],
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            print(f"[{args.mode}] step {step}/{stop_at} "
                  f"loss={rec['loss']:.4f} val MRE={val['mre_mean']:.4f} "
                  f"yield1%={val['yield_1pct']:.1f} "
                  f"yield0.1%={val['yield_01pct']:.1f} "
                  f"pool={len(pool_flux)} ({rec['elapsed_s']:.0f}s)", flush=True)
            # checkpoint the lowest val MRE (post-decay evals win naturally)
            if val["mre_mean"] <= best["val_mre_mean"]:
                best = {"step": step,
                        **{f"val_{k}": v for k, v in val.items()}}
                torch.save({"state_dict": model.state_dict(),
                            "variant": "combo", "args": vars(args)},
                           os.path.join(args.outdir, f"{tag}.pt"))
            # early stopping: trigger the WSD decay (or hard-stop) once the
            # smoothed val MRE has plateaued in the stable phase
            mre_hist.append(val["mre_mean"])
            smoothed = float(np.mean(mre_hist[-args.mre_smooth:]))
            if (args.early_stop_patience and not early_stop_off
                    and sched_state["decay_from"] is None
                    and step < decay_start):
                if smoothed < best_smoothed * (1 - args.early_stop_rel):
                    best_smoothed, no_improve = smoothed, 0
                else:
                    no_improve += 1
                if no_improve >= args.early_stop_patience:
                    # A plateau reached while yield@0.1% is still poor is a
                    # noise-triggered false alarm (cf. Na/Al/S): don't stop --
                    # disable early stopping and run the full budget instead.
                    if val["yield_01pct"] < args.early_stop_min_yield:
                        early_stop_off = True
                        print(f"[{args.mode}] plateau at step {step} but "
                              f"yield0.1%={val['yield_01pct']:.1f} < "
                              f"{args.early_stop_min_yield:.0f}: disabling early "
                              f"stop, running to {args.steps}", flush=True)
                    elif args.schedule == "wsd":
                        sched_state["decay_from"] = step
                        stop_at = min(args.steps, step + decay_len)
                        print(f"[{args.mode}] early stop: smoothed MRE plateaued "
                              f"at step {step} (yield0.1%="
                              f"{val['yield_01pct']:.1f}); decaying, finish at "
                              f"{stop_at}", flush=True)
                    else:
                        stop_at = step
                        print(f"[{args.mode}] early stop at step {step} "
                              f"(smoothed MRE plateaued)", flush=True)
            if wema is not None:
                wema.swap_out(model)
            if is_sf:
                opt.train()  # restore the training iterate

        if step >= stop_at:
            break

    if is_sf:
        opt.eval()  # leave the model holding the averaged weights
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
    print(f"[{args.mode}] done in {(time.time()-t0)/60:.1f} min at step "
          f"{history[-1]['step'] if history else 0}; best val MRE="
          f"{best['val_mre_mean']:.4f} yield1%={best.get('val_yield_1pct', 0):.1f} "
          f"yield0.1%={best.get('val_yield_01pct', 0):.1f} "
          f"(synthetic spectra: {len(acquired)})", flush=True)
    if args.diag_plots:
        from spexai.train.diagnostics import run_diagnostics
        ckpt = torch.load(os.path.join(args.outdir, f"{tag}.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        run_diagnostics(model, data, history, args.outdir, tag, device=device)
    return model, best


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cachedir",
                    default="/Users/danielahuppenkothen/work/data/spexai/processed/element26")
    ap.add_argument("--outdir",
                    default="/Users/danielahuppenkothen/work/data/spexai/runs/element26/adaptive")
    ap.add_argument("--tag", default=None,
                    help="checkpoint/history name (defaults to mode)")
    ap.add_argument("--init_from", default=None,
                    help="warm start: load model weights from this checkpoint "
                         "(.pt) instead of random init, e.g. to continue a "
                         "prematurely-stopped run. Same architecture/cache "
                         "required. Optimizer/schedule/step start fresh.")
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
                    choices=["gelu", "silu", "tanh", "sine", "finer"])
    ap.add_argument("--finer_k", type=float, default=10.0,
                    help="finer activation: trunk bias init U(-k, k)")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "schedulefree", "muon"])
    ap.add_argument("--lr_muon", type=float, default=0.02,
                    help="Muon learning rate for matrix params")
    ap.add_argument("--mup", type=int, default=0,
                    help="approximate muP LR scaling (adamw only)")
    ap.add_argument("--mup_base_width", type=int, default=128)
    ap.add_argument("--pr_correct", type=float, default=0.0,
                    help="InfoBatch-style importance-sampling correction "
                         "exponent (0 = biased oversampling, 1 = unbiased)")
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
    ap.add_argument("--early_stop_patience", type=int, default=8,
                    help="evals with no smoothed-MRE improvement before "
                         "triggering the decay (wsd) or a hard stop; 0 = off. "
                         "--steps is then just an upper bound. Raised from 5 "
                         "to 8 after noisy high-LR runs (Na/Al) false-triggered "
                         "on a single eval spike and stopped while still "
                         "improving 30-40%/eval.")
    ap.add_argument("--early_stop_rel", type=float, default=0.02,
                    help="relative val-MRE improvement needed to reset "
                         "early-stop patience")
    ap.add_argument("--mre_smooth", type=int, default=5,
                    help="evals averaged for the early-stop MRE signal "
                         "(raised from 3 to 5 so a single noisy eval spike "
                         "cannot dominate the plateau detector)")
    ap.add_argument("--early_stop_min_yield", type=float, default=25.0,
                    help="if the plateau detector fires while yield@0.1%% is "
                         "below this, treat it as a noise-triggered false "
                         "alarm: disable early stopping and run the full "
                         "--steps budget instead (protects hard/noisy elements "
                         "like Na/Al/S from premature stops)")
    ap.add_argument("--use_linehead", default="auto",
                    choices=["auto", "on", "off"],
                    help="line head: 'off' for pure-continuum elements "
                         "(e.g. H); 'auto' disables it below --min_line_bins")
    ap.add_argument("--min_line_bins", type=int, default=32,
                    help="auto mode: fewer line bins than this -> no line head")
    ap.add_argument("--use_edges", default="auto", choices=["auto", "off"],
                    help="include detected recombination edges in the signal "
                         "pool (auto = on when any edge is found)")
    ap.add_argument("--signal_frac", type=float, default=0.15,
                    help="target share of the point loss placed on signal "
                         "bins (spectral lines + edges); 0 = plain uniform "
                         "sampling. Self-tunes the per-point weight so the "
                         "share holds whether an element has ~6 or ~300 "
                         "non-floor bins. Raise for floor-dominated low-Z "
                         "elements (Li/Be/B).")
    ap.add_argument("--edge_radius", type=int, default=4,
                    help="bins each side of an edge to add to the signal pool")
    ap.add_argument("--use_trend", type=int, default=1)
    ap.add_argument("--use_binnorm", type=int, default=1)
    ap.add_argument("--w_log", type=float, default=0.1)
    ap.add_argument("--curriculum_frac", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    # --- speed knobs (effective on CUDA) ---
    ap.add_argument("--amp", type=int, default=1,
                    help="bf16 autocast on the forward pass (CUDA only); the "
                         "loss stays fp32. 1=on")
    ap.add_argument("--data_on_gpu", type=int, default=1,
                    help="keep logflux/temps/indices resident on the GPU to "
                         "avoid per-step host<->device transfers and CPU "
                         "gather (CUDA only). 1=on; set 0 if GPU memory is "
                         "tight (cache is ~1.4 GB)")
    ap.add_argument("--compile", type=int, default=0,
                    help="torch.compile the forward hot path (experimental; "
                         "fixed-shape point sampling makes it viable). 1=on")
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
