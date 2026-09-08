"""Walker-batched, all-device forward: ``theta (B, ndim)`` -> counts ``(B, n_keep)``.

A vectorised sampler hands the log-probability its whole ensemble at once, and
this turns that block into predicted in-band counts in a single pass -- one
process, no CPU pool. The batch dimension is the walkers throughout: per-walker
temperature (native to the operator), per-walker abundances (weights applied
inside the batched flux), per-walker ``sigma_v`` and ``N_H``, and the response
fold as a ``torch.sparse`` CSR mat-mul on the device.

This is the production forward, promoted out of ``inference_demo/hot_floor``
(where it was ``gpu_forward.EnsembleForward``, wired to that demo's Perseus
constants and 28-element store). Everything experiment-specific -- redshift,
distance, the abundance parametrisation, the fitted parameter list -- is now
injected, so the same class serves the hot-floor cross-check, the sampler
bake-off and the SBC campaign.

Two independent axes, both defaulting to the fast path:

* ``batched=True`` evaluates every element's trunk as a few grouped GEMMs
  (:class:`~spexai.inference.batched.BatchedJointForward`) instead of looping
  elements; with ``compile_trunk`` it is 2.19x the old serial-accel path at no
  accuracy cost. ``batched=False`` keeps the per-element loop as a reference.
* ``grad=True`` (via :meth:`counts_torch`) keeps the autograd graph so
  gradient-based samplers can get d(counts)/d(theta). Off by default: the graph
  over the trunk pins every activation.

Correctness is checked against the serial reference forward in
``tests/test_vector_forward.py`` and, for the hot-floor configuration, by
``inference_demo/hot_floor/gpu_forward.py``.
"""
import contextlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from spexai.inference.operator_model import (element_broadened_flux,
                                             ensure_recompile_limit)
from spexai.inference.units import D_REF_M, FLUX_M2_TO_CM2, distance_factor

# Emulator rows per DEM block UNDER GRADIENTS (see VectorForward.dem_gchunk).
# Anchored on measurement, not taste: the single-T P6 backward at 8 rows cost
# 11-15 GB, because a row retains its (n_elements, K~2e5) FFT broadening
# intermediates -- `batched._density` checkpoints the trunk, `_continuum` does
# not. The value-only budget is `chunk` (32) and unrelated.
DEM_GRAD_ROWS = 8


class VectorForward:
    """theta ``(B, ndim)`` -> in-band counts ``(B, n_keep)`` on ``device``.

    Parameters
    ----------
    emu : JointOperatorModel
        Loaded emulator; its element set drives the abundance resolution.
    response : Response
        Instrument response; the fold uses its ARF and RMF.
    keep : np.ndarray of bool
        Channel mask selecting the fit band.
    param_names : sequence of str
        The sampled parameters, in the order the sampler supplies them. Must
        contain the temperature and log-normalisation names; the velocity and
        column density are read when present and otherwise taken from
        ``fixed``. Every remaining name is passed to ``abundance_model``.
    temp_name, norm_name, velocity_name, nh_name : str
        Which sampled parameter plays each physical role. Defaults match the
        hot-floor convention (``kT``/``sigma_v``); ``spexai.inference.fitting``
        passes its own spelling (``temp``/``velocity``) instead of forcing
        every caller onto one vocabulary.
    exposure : float
        Seconds. Multiplies the predicted counts, matching
        ``JointOperatorModel.predict_counts``.
    abundance_model : AbundanceModel
        Maps the sampled abundance parameters to ``{Z: value}``. Its
        ``param_names`` must all appear in ``param_names``.
    absorption : Absorption or None
        Galactic screen, applied in the observed frame before rebinning.
    redshift : float
        Source redshift; the response grid is shifted to the rest frame.
    luminosity_distance : float
        Metres. Fixed, not fitted -- it is degenerate with the normalisation.
    velocity : float or None
        ``None`` samples ``sigma_v`` per walker (it must then be in
        ``param_names``); a float pins it and drops it from the vector.
    n_h_scale : float
        Multiplier taking the sampled ``n_h`` to cm^-2 (the parameter is
        conventionally in units of 1e21).
    chunk : int
        Walkers are processed in sub-batches of this size so peak device memory
        is bounded and the ensemble size stays independent of it.
    dem_grid_chunk : int or None
        DEM only: temperature-grid points per emulator call. ``None`` sizes it
        so one call is ~``chunk`` rows (see ``dem_gchunk``). Under gradients
        each block is checkpointed, so this -- not ``chunk`` -- is what bounds
        a DEM backward.
    """

    def __init__(self, emu, response, keep, param_names: Sequence[str],
                 abundance_model, absorption=None, redshift: float = 0.0,
                 luminosity_distance: float = D_REF_M,
                 velocity: Optional[float] = None, fixed: Optional[Dict] = None,
                 n_h_scale: float = 1e21, device: str = "cpu", chunk: int = 32,
                 batched: bool = True, compile_trunk: bool = False,
                 mem_gb: float = 2.0, echunk: Optional[int] = None,
                 exposure: float = 1.0,
                 temp_name: str = "kT", norm_name: str = "log_norm",
                 velocity_name: str = "sigma_v", nh_name: str = "n_h",
                 dem=None, dem_grid_chunk: Optional[int] = None):
        self.emu, self.absn, self.device = emu, absorption, device
        # DEM: the emulator is evaluated on a whole temperature grid per walker
        # and the fluxes summed with the walker's own weights. Requires the
        # batched `weights_batch` contract (see spexai.inference.tempdist).
        self.dem = dem
        self.dem_grid_chunk = dem_grid_chunk
        if dem is not None and not hasattr(dem, "weights_batch"):
            raise ValueError(
                f"{type(dem).__name__} has no weights_batch(), so it cannot be "
                "evaluated per walker; use the scalar likelihood for it")
        self.names: List[str] = list(param_names)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.abmodel = abundance_model
        self.fixed = dict(fixed or {})
        self.z = float(redshift)
        self.n_h_scale = float(n_h_scale)
        self.chunk = int(chunk)
        self.mem_gb = float(mem_gb)
        # Energy-axis chunk. Auto-sized from mem_gb when None, which is right
        # for the value-only path. Under gradients it is also the checkpoint
        # segment size, and therefore the main memory lever: the retained graph
        # scales with one chunk, so halving it roughly halves peak memory at
        # the cost of a little more recompute.
        self.echunk = echunk
        self.temp_name = temp_name
        self.norm_name = norm_name
        self.velocity_name = velocity_name
        self.nh_name = nh_name
        self.velocity = None if velocity is None else float(velocity)
        self.fit_sigma_v = self.velocity is None
        if self.fit_sigma_v and velocity_name not in self.col:
            raise ValueError(f"velocity=None samples the velocity, so "
                             f"'{velocity_name}' must be in param_names "
                             f"(pass velocity=<float> to pin it)")
        # with a DEM the temperature is not sampled at all -- the DEM's own
        # shape parameters are, and the grid is fixed
        required = [norm_name] if dem is not None else [temp_name, norm_name]
        missing = [n for n in required if n not in self.col]
        if missing:
            raise ValueError(f"param_names must contain {missing}")
        if dem is not None:
            unknown_dem = [n for n in dem.param_names if n not in self.col]
            if unknown_dem:
                raise ValueError(f"DEM parameters absent from param_names: "
                                 f"{unknown_dem}")
        unknown = [n for n in abundance_model.param_names if n not in self.col]
        if unknown:
            raise ValueError(f"abundance parameters absent from param_names: "
                             f"{unknown}")

        self.use_batched = bool(batched)
        self.compile_trunk = bool(compile_trunk) and self.use_batched
        if compile_trunk and not self.use_batched:
            # Serial path: compile the per-element coordinate-MLP. Dynamo caches
            # per *code object* and every element compiles the same
            # forward_norm, so they share one budget -- without this the default
            # limit of 8 is exhausted and the remaining elements silently run
            # eager, which is exactly how the old serial timings lost their
            # speedup without any error.
            ensure_recompile_limit(8 * max(1, len(emu.models)))
            for m in emu.models.values():
                m.forward_norm = torch.compile(m.forward_norm, dynamic=True)

        # Device sparse fold: R^T (C, N) so counts = (R^T @ eff^T)^T = eff @ R.
        rt = response.R.T.tocsr()
        self.Rt = torch.sparse_csr_tensor(
            torch.from_numpy(rt.indptr.astype(np.int64)),
            torch.from_numpy(rt.indices.astype(np.int64)),
            torch.from_numpy(rt.data.astype(np.float32)),
            size=rt.shape, device=device)
        self.arf = torch.as_tensor(response.arf, dtype=torch.float32,
                                   device=device)
        self.edges_rest = (response.energy_edges * (1.0 + self.z)).to(device)
        self.keep_idx = torch.as_tensor(np.where(keep)[0], device=device)
        # matches predict_counts: Y * exposure * (D_ref/D)^2 * (m^2 -> cm^2)
        self.scale_const = (float(exposure)
                            * distance_factor(float(luminosity_distance))
                            * FLUX_M2_TO_CM2)

    # --- parameter unpacking -------------------------------------------------

    def _column(self, th: torch.Tensor, name: str, default=None):
        """One parameter as a (B,) tensor: sampled if present, else fixed."""
        if name in self.col:
            return th[:, self.col[name]]
        if name in self.fixed:
            return torch.full((th.shape[0],), float(self.fixed[name]),
                              dtype=th.dtype, device=self.device)
        if default is None:
            raise KeyError(f"'{name}' is neither sampled nor fixed")
        return torch.full((th.shape[0],), float(default), dtype=th.dtype,
                          device=self.device)

    def unpack(self, th: torch.Tensor):
        """theta ``(B, ndim)`` tensor -> ``(temps, vel, n_h, abund, norm)``.

        Every value is a ``(B,)`` tensor on ``self.device`` (``vel`` is a scalar
        when pinned). Abundances come from ``abundance_model``, so the tying
        scheme is defined in exactly one place and applies identically to a
        scalar fit and to a walker ensemble; any constant abundances carried in
        ``fixed["abundances"]`` sit underneath and are overridden by it."""
        temps = (th[:, self.col[self.temp_name]] if self.dem is None
                 else None)                        # DEM supplies its own grid
        n_h = self._column(th, self.nh_name, default=0.0) * self.n_h_scale
        vel = (th[:, self.col[self.velocity_name]] if self.fit_sigma_v
               else self.velocity)
        norm = torch.pow(10.0, th[:, self.col[self.norm_name]])
        p = {n: th[:, i] for n, i in self.col.items()}
        abund = {**self.fixed.get("abundances", {}),
                 **self.abmodel.to_abundances(p)}
        return temps, vel, n_h, abund, norm

    # --- stages --------------------------------------------------------------

    def flux(self, th: torch.Tensor) -> torch.Tensor:
        """Abundance-weighted, broadened flux on the response grid -> (B, M).

        Everything except the fold: operator trunks, FFT continuum broadening,
        on-device absorption, rebin, analytic line deposit. Split out so it can
        be timed on its own.

        The batched path evaluates all trunks as grouped GEMMs and takes the
        per-walker abundances straight through; the serial path loops elements
        and weights each outside the flux. Same numbers either way."""
        temps, vel, n_h, abund, _ = self.unpack(th)
        if self.dem is not None:
            return self._flux_dem(th, vel, n_h, abund)
        if self.use_batched:
            return self.emu.batched.flux(
                temps, abund, vel, self.edges_rest, absorption=self.absn,
                n_h=n_h, redshift=self.z, mem_gb=self.mem_gb,
                echunk=self.echunk,
                compile_trunk=self.compile_trunk)                  # (B, M)
        total = None
        for z, model in self.emu.models.items():
            ef = element_broadened_flux(                           # (B, M)
                model, temps, vel, self.edges_rest, absorption=self.absn,
                n_h=n_h, redshift=self.z, use_torch_absorption=True)
            a = abund.get(z, 1.0)
            a = a[:, None] if torch.is_tensor(a) else a
            total = a * ef if total is None else total + a * ef
        return total

    def dem_gchunk(self, B: int, grad: bool = False) -> int:
        """Grid points per emulator call, for ``B`` walkers at a time.

        The budget is in EMULATOR ROWS (``B * gchunk``), because that is what
        both the trunk and the broadening scale with; the DEM's G is a
        multiplier on the row count, not a separate budget.

        The two budgets differ by a lot, and deliberately. Value-only rows are
        freed as they are computed, so ``chunk`` (32) is fine. A row under
        gradients also retains its FFT broadening intermediates -- ``(N*B, K)``
        with K ~ 2e5 -- and ``DEM_GRAD_ROWS`` is anchored on the measurement
        that the single-T P6 run's backward at 8 rows cost 11-15 GB. Four times
        that is a 22 GB card, which is exactly what the first DEM attempt hit.

        ``dem_grid_chunk`` overrides both.
        """
        if self.dem_grid_chunk is not None:
            return max(1, int(self.dem_grid_chunk))
        rows = DEM_GRAD_ROWS if grad else self.chunk
        return max(1, rows // max(1, int(B)))

    def _flux_dem(self, th, vel, n_h, abund) -> torch.Tensor:
        """Emission-measure-weighted flux over a temperature grid -> (B, M).

        Every walker needs the emulator at all ``G`` grid temperatures, so the
        two axes are flattened into one batch of ``B*G`` rows ordered
        ``row = b*G + g``. The per-walker quantities (velocity, column density,
        abundances) are therefore ``repeat_interleave``d by ``G`` -- a plain
        ``repeat`` would tile them in the wrong order and silently pair each
        walker's abundances with another walker's temperatures.

        Summing over the grid *before* folding is exact and G times cheaper,
        since the fold is linear -- the same trick ``predict_counts_dem`` uses.

        GRID CHUNKING (memory). That sum is also what lets the grid be walked in
        blocks: each block contributes its own weighted partial sum, and the
        blocks are added. Under gradients each block is CHECKPOINTED, so only
        one block's activations are alive at a time instead of all G.

        This is not a micro-optimisation. ``batched._density`` already
        checkpoints over the energy axis, but the FFT broadening does not, and
        its graph scales with ``n_elements * B * G`` rows -- so a 48-point grid
        made even a SINGLE walker's backward larger than an 8-walker single-T
        one, and no amount of walker chunking could bring it down. Peak memory
        is now set by ``dem_gchunk`` rows -- ``DEM_GRAD_ROWS`` under gradients
        -- at the cost of one recomputed forward per block.
        """
        B = th.shape[0]
        grid = torch.as_tensor(self.dem.temp_grid, dtype=torch.float32,
                               device=self.device).reshape(-1)
        G = grid.numel()
        p = {n: th[:, i] for n, i in self.col.items()}
        w = self.dem.weights_batch(p).to(self.device)          # (B, G)
        if w.shape != (B, G):
            raise ValueError(f"weights_batch returned {tuple(w.shape)}, "
                             f"expected {(B, G)}")

        # Captured OUTSIDE the block, restored INSIDE it. Checkpoint recomputes
        # during backward, by which time the caller's `grad_enabled()` context
        # has long exited and `batched.track_grad` is False again -- so
        # `_grad_aware` would run the recomputation under no_grad and save
        # nothing, which torch reports as "158 tensors saved during forward, 1
        # during recomputation". Re-entering the flag makes the recompute match
        # the forward. On the value path `track` is False, so this is a no-op
        # and cannot switch grad back on inside a no_grad region.
        bt = getattr(self.emu, "batched", None)
        track = bool(getattr(bt, "track_grad", False))

        def block(lo: int, hi: int) -> torch.Tensor:
            """Grid points [lo, hi) -> that block's (B, M) weighted flux."""
            with (bt.grad_enabled(track) if bt is not None
                  else contextlib.nullcontext()):
                return self._dem_block(lo, hi, grid, w, vel, n_h, abund, B)

        # use_reentrant=False so the closure's tensors (w, vel, n_h, abund --
        # all downstream of th) get their gradients, not just explicit args.
        ckpt = bool(torch.is_grad_enabled() and th.requires_grad)
        gc = self.dem_gchunk(B, grad=ckpt)
        if gc >= G:
            return block(0, G)
        total = None
        for lo in range(0, G, gc):
            hi = min(lo + gc, G)
            part = (torch.utils.checkpoint.checkpoint(
                        block, lo, hi, use_reentrant=False)
                    if ckpt else block(lo, hi))
            total = part if total is None else total + part
        return total

    def _dem_block(self, lo, hi, grid, w, vel, n_h, abund, B) -> torch.Tensor:
        """Grid points [lo, hi) -> that block's (B, M) weighted flux.

        The per-walker quantities are ``repeat_interleave``d by the block's
        width, matching the ``row = b*g + gi`` flattening; a plain ``repeat``
        would tile them in the wrong order and silently pair each walker's
        abundances with another walker's temperatures.
        """
        g = hi - lo
        temps = grid[lo:hi].unsqueeze(0).expand(B, g).reshape(-1)  # (B*g,)
        rep = (lambda t: t.repeat_interleave(g, dim=0)
               if torch.is_tensor(t) and t.numel() > 1 else t)
        vel_f, nh_f = rep(vel), rep(n_h)
        abund_f = {z: rep(a) for z, a in abund.items()}
        if self.use_batched:
            flux = self.emu.batched.flux(
                temps, abund_f, vel_f, self.edges_rest,
                absorption=self.absn, n_h=nh_f, redshift=self.z,
                mem_gb=self.mem_gb, echunk=self.echunk,
                compile_trunk=self.compile_trunk)              # (B*g, M)
        else:
            flux = None
            for z, model in self.emu.models.items():
                ef = element_broadened_flux(
                    model, temps, vel_f, self.edges_rest,
                    absorption=self.absn, n_h=nh_f, redshift=self.z,
                    use_torch_absorption=True)
                a = abund_f.get(z, 1.0)
                a = a[:, None] if torch.is_tensor(a) else a
                flux = a * ef if flux is None else flux + a * ef
        return (w[:, lo:hi].unsqueeze(-1)
                * flux.reshape(B, g, -1)).sum(dim=1)           # (B, M)

    def fold(self, flux: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Fold flux (B, M) through ARF+RMF on-device, scale by the norm."""
        norm = torch.pow(10.0, th[:, self.col[self.norm_name]])
        eff = (flux * self.arf).transpose(0, 1).contiguous()       # (N, B)
        counts = torch.sparse.mm(self.Rt, eff).transpose(0, 1)     # (B, C)
        return counts[:, self.keep_idx] * norm[:, None] * self.scale_const

    # --- entry points --------------------------------------------------------

    def counts_torch(self, th: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """theta ``(B, ndim)`` tensor -> counts ``(B, n_keep)`` tensor.

        The differentiable entry point: with ``grad=True`` (and a ``th`` that
        requires grad) the autograd graph survives, which is what HMC/NUTS and
        reparameterised VI need. Unchunked -- a gradient step needs the whole
        graph anyway, so chunking would only fragment it.

        The operator is float32, so ``th`` is cast here rather than by the
        caller; the cast is differentiable, so a float64 sampler still gets its
        gradient back in float64.

        **Device is normalised here too**, not just dtype. A PPL hands over
        whatever device its sample sites live on, and Pyro's sites default to
        CPU unless the prior distributions were built on the GPU. ``flux``
        happens to survive that -- ``batched.flux`` rebuilds ``temp_kev`` with
        an explicit device -- but ``fold`` reads the normalisation straight off
        ``th``, so a CPU ``th`` met a CUDA ``counts`` and only blew up at the
        very last multiply. Moving 12 floats is free; the cast is
        differentiable, so gradients still flow back to the sampler's
        device."""
        th = th.to(device=self.device, dtype=torch.float32)
        if not grad:
            return self._counts_chunked(th)
        with self.emu.batched.grad_enabled():
            return self.fold(self.flux(th), th)

    @property
    def walker_chunk(self) -> int:
        """Walkers per sub-batch, allowing for the DEM's grid.

        A DEM turns each walker into ``G`` emulator rows, so the walker chunk
        has to shrink by ``G`` to keep peak memory where ``chunk`` intended it
        -- otherwise a 60-point grid quietly makes the batch 60x larger and
        OOMs the GPU."""
        if self.dem is None:
            return self.chunk
        g = int(torch.as_tensor(self.dem.temp_grid).reshape(-1).numel())
        return max(1, self.chunk // max(1, g))

    def _counts_chunked(self, th: torch.Tensor) -> torch.Tensor:
        chunk = self.walker_chunk
        if th.shape[0] <= chunk:
            return self.fold(self.flux(th), th)
        parts = [self.fold(self.flux(th[i:i + chunk]), th[i:i + chunk])
                 for i in range(0, th.shape[0], chunk)]
        return torch.cat(parts, dim=0)

    def __call__(self, theta) -> np.ndarray:
        """theta ``(B, ndim)`` array-like -> counts ``(B, n_keep)`` ndarray.

        The value-only path used by the gradient-free samplers; walkers are
        processed in sub-batches of ``chunk`` to bound device memory."""
        # ascontiguousarray, not asarray: samplers that hand us a view into a
        # structured array (nessai live points) produce strides that are not a
        # multiple of the itemsize, which torch.as_tensor refuses. asarray is
        # a no-op there because the dtype already matches.
        th = torch.as_tensor(
            np.ascontiguousarray(np.atleast_2d(theta), dtype=np.float64),
            dtype=torch.float32, device=self.device)
        return self._counts_chunked(th).detach().cpu().numpy()
