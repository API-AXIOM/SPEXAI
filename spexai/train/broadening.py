"""Gaussian velocity broadening of binned spectra, three ways.

All functions work on INTEGRATED flux per bin (photons per bin), the same
convention as SpectralOperator.forward_on_grid, and treat broadening as a
Doppler redistribution: a photon emitted in bin j (energy E_j) lands at
E_j * (1 + v_los / c) with v_los ~ N(0, v^2), i.e. a Gaussian of width
sigma_j = E_j * v / c in linear energy.

  * direct_broadening_matrix: exact bin-integrated Gaussian redistribution
    (erf differences), assembled as a sparse matrix. The ground-truth
    reference; also fast to APPLY once built for a given velocity.
  * fft_broaden / broaden_native: convolution in log-energy space via FFT,
    where velocity broadening is (to first order in v/c) shift-invariant.
    Fast per call for arbitrary velocities, batched, differentiable.
  * deposit_gaussian_lines: exact bin-integrated Gaussians for a list of
    discrete lines (energy + integrated flux), used by the hybrid scheme
    where the line head's amplitudes are broadened analytically.

Velocities are in km/s throughout.
"""

import math

import torch

C_KMS = 299792.458


# ---------------------------------------------------------------------------
# flux-conserving rebinning (shared with SpectralOperator.forward_on_grid)
# ---------------------------------------------------------------------------

def rebin_flux(flux_int, edges_in, edges_out):
    """Rebin integrated fluxes (B, K) from edges_in (K+1,) to edges_out
    (M+1,) via the piecewise-linear cumulative flux. Flux-conserving for
    finer and coarser target grids. float64 accumulation internally."""
    if flux_int.device.type == "mps":  # no float64 on MPS: hop through CPU
        return rebin_flux(flux_int.cpu(), edges_in.cpu(),
                          edges_out.cpu()).to(flux_int.device)
    B = flux_int.shape[0]
    f64 = flux_int.double()
    cum = torch.cat([torch.zeros(B, 1, dtype=torch.float64,
                                 device=flux_int.device),
                     torch.cumsum(f64, dim=1)], dim=1)
    e_in = edges_in.double()
    k = torch.searchsorted(e_in, edges_out.double()).clamp(1, len(e_in) - 1)
    e_lo = e_in[k - 1]
    frac = ((edges_out.double() - e_lo) / (e_in[k] - e_lo)).clamp(0., 1.)
    c_at = cum[:, k - 1] + f64[:, (k - 1).clamp(max=f64.shape[1] - 1)] * frac
    return (c_at[:, 1:] - c_at[:, :-1]).float()


# ---------------------------------------------------------------------------
# exact reference: sparse redistribution matrix
# ---------------------------------------------------------------------------

def direct_broadening_matrix(edges, velocity, nsigma=6.0):
    """Exact Gaussian redistribution matrix R for one velocity.

    R[m, j] = probability that a photon from source bin j (center E_j,
    width sigma_j = E_j v/c) lands in target bin m, computed as the erf
    difference over the target bin edges. Returns a sparse CSR (M, K)
    matrix; broadened = flux_int @ R.T (or R @ flux for one spectrum).
    Flux leaving the grid at the boundaries is lost (physical).
    """
    centers = torch.sqrt(edges[:-1] * edges[1:]).double()
    e = edges.double()
    sigma = centers * (velocity / C_KMS)
    K = len(centers)

    lo = torch.searchsorted(e, centers - nsigma * sigma).clamp(min=1) - 1
    hi = torch.searchsorted(e, centers + nsigma * sigma).clamp(max=K)
    width = int((hi - lo).max())
    # windowed edge positions per source bin: (K, width + 1)
    idx = (lo.unsqueeze(1) + torch.arange(width + 1)).clamp(max=K)
    z = (e[idx] - centers.unsqueeze(1)) / (math.sqrt(2.0) * sigma.unsqueeze(1))
    w = 0.5 * (torch.erf(z[:, 1:]) - torch.erf(z[:, :-1]))  # (K, width)
    # target-bin index of each window column; mask out-of-range columns
    tgt = idx[:, :-1]
    valid = (tgt < K) & ((lo.unsqueeze(1) + torch.arange(width)) < hi.unsqueeze(1))
    src = torch.arange(K).unsqueeze(1).expand(-1, width)
    indices = torch.stack([tgt[valid], src[valid]])
    values = w[valid].float()
    return torch.sparse_coo_tensor(indices, values, (K, K)).to_sparse_csr()


def direct_broaden(flux_int, edges, velocity):
    """Exact broadening of integrated fluxes (B, K): the reference."""
    R = direct_broadening_matrix(edges, velocity)
    return torch.sparse.mm(R, flux_int.T).T


# ---------------------------------------------------------------------------
# FFT broadening on a uniform log-energy grid
# ---------------------------------------------------------------------------

def uniform_log_edges(emin, emax, dlx):
    """Uniform grid in log10 E with spacing dlx covering [emin, emax]."""
    n = int(math.ceil((math.log10(emax) - math.log10(emin)) / dlx))
    return torch.pow(10.0, math.log10(emin)
                     + dlx * torch.arange(n + 1, dtype=torch.float64)).float()


# Opt-in: use float32 for the FFT continuum broadening on CUDA as well as MPS.
# ~4x faster on most GPUs, and safe in the inference forward because only the
# SMOOTH continuum is FFT-broadened (lines are deposited analytically), so
# float32 spectral leakage is negligible. Off by default -> training/eval path
# unchanged.
USE_FLOAT32_FFT = False

# Zero-padding is rounded up to a multiple of this. The padding only has to be
# *at least* the kernel's reach, so rounding up is always numerically safe -- it
# adds zeros and nothing else. The point is plan stability: cuFFT builds (and
# caches, with a GPU workspace) one plan per distinct transform length, and the
# raw padding is a continuous function of the batch's maximum velocity. With a
# per-walker sigma_v that changes every MCMC step, so an un-quantised length
# allocates a fresh plan per step until the GPU runs out of memory and cuFFT
# fails with CUFFT_INTERNAL_ERROR. Quantising collapses a run to a couple of
# lengths. 256 costs at most ~0.3% extra transform on a ~160k-bin grid.
FFT_PAD_QUANTUM = 256


def limit_cufft_plan_cache(max_size=16):
    """Bound cuFFT's per-device plan cache (no-op off CUDA).

    A backstop for the same failure: even with quantised lengths, distinct batch
    row counts produce distinct plans, and torch's default cache holds many.
    Each entry pins a workspace, so an unbounded cache is a slow GPU memory
    leak. Evicting old plans costs a re-plan, which is negligible next to the
    transforms themselves."""
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.cufft_plan_cache.max_size = int(max_size)
    except (AttributeError, RuntimeError):          # older/newer torch layouts
        pass


def fft_broaden(flux_uni, dlx, velocity):
    """Broaden integrated fluxes on a uniform log10-E grid by FFT.

    flux_uni: (B, K); dlx: grid spacing in log10 E; velocity: scalar or
    (B,) tensor in km/s. In u = ln E the Gaussian kernel has width
    sigma_u = v/c, so the transfer function is exp(-(2 pi f sigma_u)^2 / 2)
    with f in cycles per unit u. Zero-padded to avoid wraparound.
    """
    B, K = flux_uni.shape
    # float64 unless on MPS (no float64 support) or the float32-FFT opt-in is set
    dtype = (torch.float32
             if (flux_uni.device.type == "mps" or USE_FLOAT32_FFT)
             else torch.float64)
    v = torch.as_tensor(velocity, dtype=dtype,
                        device=flux_uni.device).reshape(-1)
    sigma_u = (v / C_KMS).clamp(min=1e-12)  # width in ln E
    du = dlx * math.log(10.0)
    # round the kernel reach up to FFT_PAD_QUANTUM so the transform length is
    # stable across calls (see the constant's note); more padding is always safe
    reach = int(torch.ceil(8.0 * sigma_u.max() / du)) + 1
    q = max(1, int(FFT_PAD_QUANTUM))
    pad = -(-reach // q) * q
    n = K + 2 * pad
    # in float32 the FFT's leakage noise (~1e-7 of the peak, spread over
    # every bin) dominates the wings of spiky spectra, so float64 is used
    # where available; callers on MPS should apply a relative noise floor.
    x = torch.nn.functional.pad(flux_uni.to(dtype), (pad, pad))
    f = torch.fft.rfftfreq(n, d=du, device=flux_uni.device, dtype=dtype)
    transfer = torch.exp(-0.5 * (2.0 * math.pi * f.unsqueeze(0)
                                 * sigma_u.to(dtype).unsqueeze(1)) ** 2)
    out = torch.fft.irfft(torch.fft.rfft(x, dim=1) * transfer, n=n, dim=1)
    return out[:, pad:pad + K].clamp(min=0.0).float()


def scatter_to_grid(flux_int, edges_in, edges_out):
    """Place each input bin's integrated flux into the output bin holding
    its (geometric) center. Unlike rebin_flux this preserves the
    delta-at-bin-center interpretation of binned line flux, matching the
    exact reference and the original spexai broadening; only appropriate
    when the output grid is finer than the input grid."""
    if flux_int.device.type == "mps":  # no float64 on MPS: hop through CPU
        return scatter_to_grid(flux_int.cpu(), edges_in.cpu(),
                               edges_out.cpu()).to(flux_int.device)
    B = flux_int.shape[0]
    M = edges_out.shape[0] - 1
    centers = torch.sqrt(edges_in[:-1] * edges_in[1:]).double()
    out_cen = torch.sqrt(edges_out[:-1] * edges_out[1:]).double()
    # cloud-in-cell: split each bin's flux linearly between the two output
    # bins whose centers bracket it, so placement error is second order
    j = (torch.searchsorted(out_cen, centers) - 1).clamp(0, M - 2)
    frac = ((centers - out_cen[j]) / (out_cen[j + 1] - out_cen[j])).clamp(0., 1.)
    frac = frac.float().unsqueeze(0)
    out = torch.zeros(B, M, device=flux_int.device)
    out.index_add_(1, j, flux_int * (1.0 - frac))
    out.index_add_(1, j + 1, flux_int * frac)
    return out


def broaden_native(flux_int, edges, velocity, dlx=1e-5):
    """FFT broadening for spectra on an arbitrary (e.g. adaptive) grid:
    scatter to a fine uniform log grid (delta-at-center assumption),
    convolve, rebin back. velocity may be a scalar or a per-spectrum
    tensor (B,)."""
    uni = uniform_log_edges(float(edges[0]), float(edges[-1]), dlx)
    uni = uni.to(flux_int.device)
    f_uni = scatter_to_grid(flux_int, edges, uni)
    f_uni = fft_broaden(f_uni, dlx, velocity)
    return rebin_flux(f_uni, uni, edges)


# ---------------------------------------------------------------------------
# analytic Gaussian deposit of discrete lines (hybrid scheme)
# ---------------------------------------------------------------------------

def deposit_gaussian_lines(line_energies, line_flux, bin_edges, velocity,
                           nsigma=6.0, chunk=512):
    """Deposit lines as exact bin-integrated Gaussians onto any grid.

    line_energies: (N,) keV; line_flux: (B, N) integrated fluxes;
    bin_edges: (M+1,); velocity: scalar km/s **or a (B,) tensor** (one
    sigma_v per walker, so the MCMC can sample it). Returns (B, M) integrated
    flux. Each line spreads over the target bins within +-nsigma of its
    Gaussian of width sigma_j = E_j v/c; erf differences make each line's
    total flux exact regardless of the target binning.

    The scalar and per-walker branches differ only in bookkeeping: with one
    sigma_v the +-nsigma window of a line is the same for every walker, so the
    scatter indices are shared across the batch and ``index_add_`` runs on the
    energy axis. Per-walker widths make those windows ragged across the batch,
    so the deposit is done into a flattened (B*M,) buffer with walker-offset
    indices instead. Same erf weights, same exactness.
    """
    if line_flux.device.type == "mps":  # no float64 on MPS: hop through CPU
        v_cpu = (velocity.cpu() if torch.is_tensor(velocity) else velocity)
        return deposit_gaussian_lines(
            line_energies.cpu(), line_flux.cpu(), bin_edges.cpu(), v_cpu,
            nsigma=nsigma, chunk=chunk).to(line_flux.device)
    B, N = line_flux.shape
    M = bin_edges.shape[0] - 1
    e = bin_edges.double()
    dev = line_flux.device
    v = torch.as_tensor(velocity, dtype=torch.float64, device=dev).reshape(-1)
    if v.numel() not in (1, B):
        raise ValueError(f"velocity must be scalar or ({B},), got {tuple(v.shape)}")

    if v.numel() == 1:                       # ---- shared window (scalar) ----
        out = torch.zeros(B, M, dtype=torch.float64, device=dev)
        # v is (1,) here, so this broadcasts to (N,) exactly as v.item() did --
        # but without detaching, which would silently zero d(flux)/d(sigma_v)
        # for every gradient-based sampler.
        sigma = line_energies.double() * (v / C_KMS)                 # (N,)
        for s in range(0, N, chunk):
            en = line_energies[s:s + chunk].double()
            sg = sigma[s:s + chunk].clamp(min=1e-12)
            lo = torch.searchsorted(e, en - nsigma * sg).clamp(min=1) - 1
            hi = torch.searchsorted(e, en + nsigma * sg).clamp(max=M)
            width = int((hi - lo).max().clamp(min=1))
            idx = (lo.unsqueeze(1) + torch.arange(width + 1,
                                                  device=e.device)).clamp(max=M)
            z = (e[idx] - en.unsqueeze(1)) / (math.sqrt(2.0) * sg.unsqueeze(1))
            w = 0.5 * (torch.erf(z[:, 1:]) - torch.erf(z[:, :-1]))  # (n, width)
            tgt = idx[:, :-1].clamp(max=M - 1)                       # (n, width)
            contrib = line_flux[:, s:s + chunk].double().unsqueeze(-1) * w
            out.index_add_(1, tgt.reshape(-1), contrib.reshape(B, -1))
        return out.float()

    # ---- per-walker windows: scatter into a flat (B*M,) buffer ----
    flat = torch.zeros(B * M, dtype=torch.float64, device=dev)
    offs = (torch.arange(B, device=dev) * M).view(-1, 1, 1)          # (B,1,1)
    sigma = line_energies.double().unsqueeze(0) * (v.view(-1, 1) / C_KMS)  # (B,N)
    for s in range(0, N, chunk):
        en = line_energies[s:s + chunk].double()                      # (n,)
        sg = sigma[:, s:s + chunk].clamp(min=1e-12)                   # (B, n)
        lo = torch.searchsorted(e, en.unsqueeze(0) - nsigma * sg).clamp(min=1) - 1
        hi = torch.searchsorted(e, en.unsqueeze(0) + nsigma * sg).clamp(max=M)
        # one width for the chunk, set by the widest (walker, line) pair; each
        # line's own erf weights still cut off at its own sigma, so the extra
        # columns a narrow walker picks up contribute ~0, not spurious flux.
        width = int((hi - lo).max().clamp(min=1))
        idx = (lo.unsqueeze(-1) + torch.arange(width + 1,
                                               device=e.device)).clamp(max=M)
        z = (e[idx] - en.view(1, -1, 1)) / (math.sqrt(2.0) * sg.unsqueeze(-1))
        w = 0.5 * (torch.erf(z[..., 1:]) - torch.erf(z[..., :-1]))   # (B,n,width)
        tgt = idx[..., :-1].clamp(max=M - 1)                          # (B,n,width)
        contrib = line_flux[:, s:s + chunk].double().unsqueeze(-1) * w
        flat.index_add_(0, (tgt + offs).reshape(-1), contrib.reshape(-1))
    return flat.view(B, M).float()


# ---------------------------------------------------------------------------
# option 3: hybrid broadening of a trained line-head emulator
# ---------------------------------------------------------------------------

@torch.no_grad()
def hybrid_broadened_on_grid(model, temp_kev, velocity, bin_edges,
                             dlx=1e-5, echunk=8192):
    """Broadened spectrum from an UNBROADENED line-head emulator.

    The trunk (smooth continuum + weak lines) is FFT-broadened on a fine
    uniform grid and rebinned to the target edges; the line head's lines
    are rendered as exact bin-integrated Gaussians at their stored
    energies. No convolution matrix and no retraining: works with any
    trained SpectralOperator that has a line head.

    temp_kev: (B,); velocity: scalar km/s; bin_edges: (M+1,).
    Returns integrated flux per bin, (B, M).
    """
    tnorm = model.norm_temp(temp_kev).view(-1, model.config.n_params)
    B = tnorm.shape[0]
    device = model.train_energy.device

    # trunk integrated flux on the training grid (lines excluded)
    x = model.norm_energy(model.train_energy).view(1, -1, 1)
    dens = torch.cat(
        [torch.pow(10.0, model.forward_norm(
            tnorm, x[:, lo:lo + echunk].expand(B, -1, -1),
            bins=torch.arange(lo, min(lo + echunk, x.shape[1]),
                              device=device),
            add_lines=False))
         for lo in range(0, x.shape[1], echunk)], dim=1)
    widths = model.train_edges[1:] - model.train_edges[:-1]
    f_train = dens * widths

    # broaden the trunk: scatter to uniform grid, convolve, rebin to target
    uni = uniform_log_edges(float(model.train_edges[0]),
                            float(model.train_edges[-1]), dlx).to(device)
    f_uni = fft_broaden(scatter_to_grid(f_train, model.train_edges, uni),
                        dlx, velocity)
    out = rebin_flux(f_uni, uni, bin_edges)

    # lines: integrated fluxes from the head, analytic Gaussian deposit
    lh = model.line_head
    amp = lh.all_line_amplitudes(tnorm)
    line_flux = torch.pow(10.0, amp) * lh.line_widths
    out = out + deposit_gaussian_lines(lh.line_energies, line_flux,
                                       bin_edges, velocity)
    return out
