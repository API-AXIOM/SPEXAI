"""Operator-based joint emulator for inference.

Loads the trained per-element SpectralOperator checkpoints from the package
model store (`spexai/models/`) and combines them into one model that
produces an integrated flux spectrum on an arbitrary target energy grid
(e.g. an instrument's RMF incident-energy grid). Replaces the CNN-based
`CombinedModel` in `spexai.inference.model`.

Design notes:
- Checkpoints are self-contained: every data-derived quantity (energy grid,
  line ids/energies/widths, per-bin norm, T/x ranges) is a registered
  buffer, so `load_operator` rebuilds the model with NO training cache.
- Each element is evaluated directly on the target grid via the operator's
  grid-portable line head (continuum on the target grid + analytic line
  deposit), so we never pay for the native 24212-bin grid. Velocity
  broadening is folded in during that evaluation.
"""
import glob
import json
import math
import os

import torch

from spexai.train.operator import OperatorConfig, SpectralOperator
from spexai.train.broadening import (deposit_gaussian_lines, fft_broaden,
                                     rebin_flux, scatter_to_grid,
                                     uniform_log_edges)
from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2, distance_factor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models")


def enable_inference_acceleration(models, device):
    """Turn on the validated production inference accelerations, in place.

    Three knobs, all measured on an A10 (``docs/inference_methodology.tex``,
    the hot-floor MCMC cross-check): TF32 tensor-core matmul (~1.4x), then
    ``torch.compile`` of each element's coordinate MLP (the ~75% bottleneck;
    ~2.6x cumulative), then a float32 continuum-FFT broadening (~3.0x). The
    accuracy cost (~1e-3, only the *smooth* continuum is float32-FFT'd) is far
    below the emulator's own error and validated science-safe (<=0.3 sigma at
    1e8 counts).

    CUDA-only by design: on CPU/MPS TF32 does not exist and ``torch.compile``
    only adds a stall, so this is a **no-op** off CUDA and the numerics stay
    byte-identical to the reference (float64) path. ``models`` is an iterable of
    ``SpectralOperator``; each has its ``forward_norm`` wrapped in place. Returns
    the list of knobs actually enabled."""
    dev = device.type if isinstance(device, torch.device) else \
        torch.device(device).type
    if dev != "cuda":
        return []
    import spexai.train.broadening as _broadening
    torch.backends.cuda.matmul.allow_tf32 = True       # TF32 tensor cores (Ampere+)
    torch.backends.cudnn.allow_tf32 = True
    _broadening.USE_FLOAT32_FFT = True                  # float32 continuum FFT
    for m in models:                                   # compile the net hot path
        m.forward_norm = torch.compile(m.forward_norm, dynamic=True)
    return ["tf32", "compile", "fft32"]


def load_operator(path, map_location="cpu"):
    """Rebuild a trained SpectralOperator from a checkpoint alone (no cache).

    Component flags are detected from the presence of their buffers/params in
    the state dict (robust to older checkpoints whose `args` left them None);
    scalar hyperparameters come from `args`; grids and normalisation come from
    the buffers via a strict `load_state_dict` (which also asserts the rebuild
    is exact)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    a, sd = ck["args"], ck["state_dict"]
    has = lambda p: any(k.startswith(p) for k in sd)

    n_bins = sd["train_energy"].shape[0]
    use_binnorm = "bn_mu" in sd
    use_linehead = has("line_head.")
    cfg = OperatorConfig(
        use_fourier="embed.freqs" in sd,
        use_grid="embed.levels" in sd or has("embed.grid"),
        use_film=has("film."),
        use_trend=has("trend."),
        use_binnorm=use_binnorm,
        use_linehead=use_linehead,
        n_freqs=int(a.get("n_freqs") or sd.get("embed.freqs",
                    torch.zeros(0)).shape[0]),
        f_max=float(a.get("f_max", 8000.0)),
        line_dim=int(a.get("line_dim", 16)),
        line_hidden=int(a.get("line_hidden") or 128),
        line_t_freqs=int(a.get("line_t_freqs") or 0),
        line_t_fmax=float(a.get("line_t_fmax", 64.0)),
        film_t_freqs=int(a.get("film_t_freqs")
                         or sd.get("film_t_embed.freqs",
                                   torch.zeros(0)).shape[0]),
        film_t_fmax=float(a.get("film_t_fmax", 64.0)),
        hidden_size=int(a["hidden"]),
        n_hidden=int(a["layers"]),
        activation=a.get("activation", "gelu"),
        x_lo=float(sd["x_lo"]), x_hi=float(sd["x_hi"]),
        t_lo=float(sd["t_lo"]), t_hi=float(sd["t_hi"]),
    )
    line_ids = sd["line_head.line_ids"] if use_linehead else None
    bin_stats = ((torch.zeros(n_bins), torch.ones(n_bins))
                 if use_binnorm else None)
    model = SpectralOperator(cfg, line_ids=line_ids,
                             energy_grid=sd["train_energy"], bin_stats=bin_stats)
    model.load_state_dict(sd, strict=True)   # exactness check
    model.eval()
    return model


@torch.no_grad()
def element_broadened_flux(model, temp_kev, velocity, bin_edges,
                           dlx=1e-5, echunk=8192, absorption=None, n_h=0.0,
                           redshift=0.0, use_torch_absorption=False):
    """Integrated flux per target bin for one element operator.

    Continuum trunk is evaluated on the training grid, FFT-broadened and
    rebinned to `bin_edges`; if the model has a line head its lines are added
    as analytic bin-integrated Gaussians. Handles trunk-only (line-head-off)
    elements by simply skipping the line deposit.

    Galactic absorption (a foreground screen ``exp(-n_h*sigma(E))``) is applied
    on the fine `uni` grid **before** rebinning and on the line amplitudes at
    their exact energies, in the observed frame (energies ``E_rest/(1+redshift)``).
    This keeps absorption instrument-resolution independent. ``n_h`` in cm^-2.

    temp_kev: (B,) tensor; velocity: scalar km/s; bin_edges: (M+1,) tensor.
    Returns (B, M) integrated flux.
    """
    device = model.train_energy.device
    tnorm = model.norm_temp(temp_kev.to(device)).view(-1, model.config.n_params)
    B = tnorm.shape[0]
    absorb = (absorption is not None
              and float(torch.as_tensor(n_h, dtype=torch.float64).max()) > 0.0)
    tfun = (absorption.transmission_torch if use_torch_absorption
            else absorption.transmission) if absorb else None

    x = model.norm_energy(model.train_energy).view(1, -1, 1)
    dens = torch.cat(
        [torch.pow(10.0, model.forward_norm(
            tnorm, x[:, lo:lo + echunk].expand(B, -1, -1),
            bins=torch.arange(lo, min(lo + echunk, x.shape[1]), device=device),
            add_lines=False))
         for lo in range(0, x.shape[1], echunk)], dim=1)
    widths = model.train_edges[1:] - model.train_edges[:-1]
    f_train = dens * widths

    uni = uniform_log_edges(float(model.train_edges[0]),
                            float(model.train_edges[-1]), dlx).to(device)
    f_uni = fft_broaden(scatter_to_grid(f_train, model.train_edges, uni),
                        dlx, velocity)
    if absorb:  # apply on the fine grid, observed frame, before rebinning
        uni_cent = torch.sqrt(uni[:-1] * uni[1:])
        f_uni = f_uni * tfun(uni_cent / (1.0 + redshift), n_h, device=device)
    out = rebin_flux(f_uni, uni, bin_edges.to(device))

    if model.line_head is not None:
        lh = model.line_head
        line_flux = torch.pow(10.0, lh.all_line_amplitudes(tnorm)) * lh.line_widths
        if absorb:  # each line multiplied by transmission at its exact energy
            line_flux = line_flux * tfun(
                lh.line_energies / (1.0 + redshift), n_h, device=device)
        out = out + deposit_gaussian_lines(lh.line_energies, line_flux,
                                           bin_edges.to(device), velocity)
    return out


class JointOperatorModel:
    """All available per-element operators combined into one emitter.

    ``flux(temp, abundances, velocity, bin_edges)`` returns the abundance-
    weighted, velocity-broadened integrated flux on ``bin_edges`` (the
    incident-energy grid of whatever instrument you fold through next).
    """

    def __init__(self, models_dir=MODELS_DIR, device="cpu", elements=None,
                 accelerate=True):
        self.device = device
        self.models_dir = models_dir
        with open(os.path.join(models_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.models = {}
        for zstr, entry in self.manifest["elements"].items():
            z = int(zstr)
            if elements is not None and z not in elements:
                continue
            m = load_operator(os.path.join(models_dir, entry["file"]), device)
            self.models[z] = m.to(device)
        self.elements = sorted(self.models)
        # low-Z primordial elements held at solar (H, He) unless overridden
        self.fixed_solar = {1, 2}
        # production inference accelerations (TF32 + compile + float32 FFT); a
        # CUDA-only no-op, so CPU/MPS numerics are unchanged. accelerate=False
        # gives the float64 reference path (e.g. for accuracy cross-checks).
        self.accel = (enable_inference_acceleration(self.models.values(), device)
                      if accelerate else [])
        self._batched = None

    @property
    def batched(self):
        """Element-batched forward (same numerics, trunk nets vmapped over
        elements); built lazily and cached. See spexai.inference.batched."""
        if self._batched is None:
            from spexai.inference.batched import BatchedJointForward
            self._batched = BatchedJointForward(self)
        return self._batched

    def __repr__(self):
        miss = self.manifest.get("missing_elements", [])
        return (f"JointOperatorModel({len(self.models)} elements "
                f"{self.elements}, missing={miss}, device={self.device})")

    @torch.no_grad()
    def flux(self, temp_kev, abundances, velocity, bin_edges,
             absorption=None, n_h=0.0, redshift=0.0):
        """Abundance-weighted summed flux on ``bin_edges``.

        temp_kev: (B,) tensor. abundances: {Z: solar-relative value}; any
        loaded element not in the dict defaults to 1.0 for the primordial
        set and to 1.0 (solar) otherwise. velocity: km/s. bin_edges: (M+1,).
        Optional Galactic absorption (``absorption``, ``n_h`` cm^-2) is applied
        as a foreground screen in the observed frame (``redshift`` = source z).
        """
        temp_kev = torch.as_tensor(temp_kev, dtype=torch.float32,
                                   device=self.device).view(-1)
        bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32,
                                    device=self.device)
        total = torch.zeros((temp_kev.numel(), bin_edges.numel() - 1),
                            device=self.device)
        for z, model in self.models.items():
            a = float(abundances.get(z, 1.0)) if abundances else 1.0
            if a == 0.0:
                continue
            total = total + a * element_broadened_flux(
                model, temp_kev, velocity, bin_edges,
                absorption=absorption, n_h=n_h, redshift=redshift)
        return total   # (B, M)

    @torch.no_grad()
    def predict_counts(self, temp_kev, abundances, logz, norm, velocity,
                       response, exposure=1.0, luminosity_distance=D_REF_M,
                       absorption=None, n_h=0.0):
        """Predicted detector-channel counts for one instrument.

        ``norm`` is the SPEX emission measure ``Y = n_H n_e V`` in units of
        ``1e64 m^-3``; ``luminosity_distance`` is the source distance in metres
        (default = the SPEX reference ``1e22 m``). Evaluates the joint flux on
        the response's incident-energy grid, redshifted (a source at z emits, in
        the rest frame, at (1+z) x the observed energies), applies optional
        Galactic absorption (``absorption``, ``n_h`` cm^-2), folds through
        ARF + RMF, and scales to physical counts (Sec. :mod:`spexai.inference.
        units`): ``Y * (D_ref/D)^2 * (m^2->cm^2) * exposure``. Returns
        (B, N_channels).

        Note: the model is only defined on its training band; response
        energies outside it fold to zero flux (a real band limit, not a bug).
        """
        z = 10.0 ** float(logz)
        edges_rest = response.energy_edges * (1.0 + z)
        flux = self.flux(temp_kev, abundances, velocity, edges_rest,
                         absorption=absorption, n_h=n_h, redshift=z)
        rate = response.fold(flux).to(self.device)       # (B, N_channels)
        scale = (float(norm) * float(exposure)
                 * distance_factor(luminosity_distance) * FLUX_M2_TO_CM2)
        return scale * rate

    @torch.no_grad()
    def predict_counts_dem(self, temp_grid, weights, abundances, logz, norm,
                           velocity, response, exposure=1.0,
                           luminosity_distance=D_REF_M, absorption=None,
                           n_h=0.0):
        """Counts-rate for a temperature *distribution* (DEM).

        Evaluates the abundance-weighted flux at every temperature in
        ``temp_grid`` (G,), takes the emission-measure-weighted sum over that
        grid with ``weights`` (G,), then folds **once** (folding is linear, so
        summing before folding is exact and G-times cheaper). ``weights`` carry
        the DEM shape x grid spacing (see :mod:`spexai.inference.tempdist`);
        ``norm`` carries the total emission measure. Returns (1, N_channels).
        """
        temp_grid = torch.as_tensor(temp_grid, dtype=torch.float32,
                                    device=self.device).view(-1)
        weights = torch.as_tensor(weights, dtype=torch.float32,
                                  device=self.device).view(-1, 1)   # (G, 1)
        z = 10.0 ** float(logz)
        edges_rest = response.energy_edges * (1.0 + z)
        flux = self.flux(temp_grid, abundances, velocity, edges_rest,
                         absorption=absorption, n_h=n_h, redshift=z)  # (G, M)
        dem_flux = (weights * flux).sum(dim=0, keepdim=True)           # (1, M)
        rate = response.fold(dem_flux).to(self.device)                # (1, N)
        scale = (float(norm) * float(exposure)
                 * distance_factor(luminosity_distance) * FLUX_M2_TO_CM2)
        return scale * rate
