"""Element-batched inference forward.

The per-element operator *trunks* (the smooth continuum nets) are the ~75%
forward bottleneck and are run once per element in the serial
``JointOperatorModel.flux`` loop. Here the identical-shape elements are grouped
and their trunks evaluated in a single pass with ``torch.func.stack_module_state``
+ ``torch.vmap``: vmap rewrites each element's ``nn.Linear`` into a batched
matmul, so the 25 big same-architecture nets become a few grouped GEMMs.

Because the grouped call goes through the **unmodified** ``forward_norm``
(via ``functional_call``), the result is numerically equivalent to the serial
path (asserted <1e-5 in ``tests/test_batched.py``) and automatically follows any
future change to the net. Only the trunk evaluation is vmapped; the downstream
stages (velocity broadening, absorption, rebin, analytic line deposit) are the
same helpers as the serial path, additionally *stacked across elements* for the
continuum since velocity and N_H are shared across elements. The ragged per-
element line heads (different line counts) stay a small Python loop.

Requires all elements to share one training energy grid (true for the store;
checked in ``__init__``). Per-element temperature normalisation (``t_lo/t_hi``)
may still differ and is handled element-by-element when building ``tnorm``.
"""
import contextlib
import copy
import functools

import torch
from torch.func import functional_call, stack_module_state

from spexai.train.operator import SpectralOperator
from spexai.inference.operator_model import (abundance_weight,
                                             ensure_recompile_limit)
from spexai.train.broadening import (deposit_gaussian_lines, fft_broaden,
                                      limit_cufft_plan_cache, rebin_flux,
                                      scatter_to_grid, uniform_log_edges)

_DLX = 1e-5  # fine-grid log spacing for the FFT broadening (matches the serial)


def _grad_aware(fn):
    """``@torch.no_grad()`` unless this forward has gradients switched on.

    Sampling only ever needs the value, and building an autograd graph over the
    trunk pins every activation -- that is the bug that presented as a 21 GiB
    CUDA OOM when the staged methods were called directly, so no_grad stays the
    default on every stage, not just on ``flux``. Gradient-based samplers
    (HMC/NUTS, reparameterised VI) do need d(flux)/d(theta), so
    :meth:`BatchedJointForward.grad_enabled` flips these to grad-tracking for
    the duration of a block. Every stage is decorated, so entering the context
    is enough however the forward is called."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with torch.set_grad_enabled(self.track_grad):
            return fn(self, *args, **kwargs)
    return wrapper


def _trunk_view(m):
    """A shallow view of a SpectralOperator with its line head removed.

    ``forward_norm(add_lines=False)`` never touches ``line_head``, but its
    parameters are ragged across elements (different line counts) and would
    break ``stack_module_state``. The view shares all trunk tensors (no copy)
    but owns its submodule mapping, so nulling ``line_head`` here does not touch
    the original model."""
    v = copy.copy(m)                       # shares _parameters/_buffers/tensors
    v._modules = dict(m._modules)          # own submodule map (module objs shared)
    v._modules["line_head"] = None
    return v


class _TrunkWrap(torch.nn.Module):
    """Expose the eager ``forward_norm(add_lines=False)`` as ``forward`` so
    ``functional_call`` (which invokes ``forward``) evaluates the continuum
    trunk. Calls the class method explicitly to bypass any compiled instance
    attribute (torch.compile-inside-vmap is avoided). ``op`` is a line-head-
    stripped view, so stacked parameter names align across a group."""

    def __init__(self, op_view):
        super().__init__()
        self.op = op_view

    def forward(self, tnorm, x, bins):
        return SpectralOperator.forward_norm(self.op, tnorm, x, bins=bins,
                                             add_lines=False)


class _TrunkGroup:
    """A set of same-shape element operators, trunk-evaluated together via vmap."""

    def __init__(self, models, zs):
        self.models = models
        self.zs = zs
        self.n_params = models[0].config.n_params
        wraps = [_TrunkWrap(_trunk_view(m)) for m in models]
        self.params, self.buffers = stack_module_state(wraps)
        base = copy.deepcopy(wraps[0]).to("meta")

        def _one(p, b, tnorm, x, bins):
            return functional_call(base, (p, b), (tnorm, x, bins))

        # map over the element axis of params/buffers/tnorm; x and bins shared
        self._vfwd = torch.vmap(_one, in_dims=(0, 0, 0, None, None))
        self._vfwd_compiled = None            # torch.compile(vmap) built on demand

    def _fwd(self, compile_trunk):
        """The vmapped trunk, optionally torch.compiled (compile ∘ vmap, so the
        grouped GEMMs *also* get kernel fusion).

        ``dynamic=False`` is required, not a tuning choice: under dynamic shapes
        the energy-chunk dimension becomes symbolic, and functorch's
        ``BatchedTensor`` cannot answer ``sizes()`` on a symbolic shape ("Cannot
        call sizes() on tensor with symbolic sizes/strides" inside ``linear``).
        It must also be explicit — the default ``dynamic=None`` would silently
        promote to symbolic shapes on the second distinct chunk size and hit the
        same failure. Static shapes mean one compiled graph per (B, chunk) shape;
        ``_echunk`` balances the chunks so that is at most two per group."""
        if not compile_trunk:
            return self._vfwd
        if self._vfwd_compiled is None:
            self._vfwd_compiled = torch.compile(self._vfwd, dynamic=False)
        return self._vfwd_compiled

    def log_density(self, temp_kev, x, bins, compile_trunk=False):
        """log10 continuum density for every element in the group.

        temp_kev: (B,) physical keV; x: (B, P, 1) shared normalised coords;
        bins: (P,) long grid indices. Returns (E, B, P)."""
        # per-element temperature normalisation (t_lo/t_hi may differ)
        tnorm = torch.stack([m.norm_temp(temp_kev).view(-1, self.n_params)
                             for m in self.models], dim=0)      # (E, B, n_params)
        return self._fwd(compile_trunk)(self.params, self.buffers, tnorm, x, bins)


class BatchedJointForward:
    """Element-batched analogue of ``JointOperatorModel.flux``.

    Construct once from a loaded ``JointOperatorModel`` (reuses its models and
    device); call :meth:`flux` with the same signature as
    ``JointOperatorModel.flux``. Numerically equivalent, but the trunk nets are
    evaluated grouped instead of looped."""

    def __init__(self, joint):
        self.joint = joint
        self.device = joint.device
        self.track_grad = False        # see grad_enabled / _grad_aware
        # this path drives long runs of large FFTs whose shapes must not
        # accumulate plans; see limit_cufft_plan_cache
        limit_cufft_plan_cache()
        m0 = joint.models[joint.elements[0]]
        # batched continuum requires a shared training energy grid + energy
        # normalisation across elements (temperature norm may still differ).
        ref_e = m0.train_energy
        for z in joint.elements:
            m = joint.models[z]
            if (m.train_energy.shape != ref_e.shape
                    or not torch.equal(m.train_energy, ref_e)
                    or float(m.x_lo) != float(m0.x_lo)
                    or float(m.x_hi) != float(m0.x_hi)):
                raise ValueError(
                    "BatchedJointForward requires a shared training energy grid "
                    f"across elements; element Z={z} differs.")
        self._m0 = m0
        # group by the config signature that fixes parameter/buffer shapes and
        # the (static) forward_norm control-flow branches
        groups = {}
        for z in joint.elements:
            groups.setdefault(self._key(joint.models[z]), []).append(z)
        self.groups = [_TrunkGroup([joint.models[z] for z in zs], zs)
                       for zs in groups.values()]

        # fixed grids, shared across elements + calls -> precompute once (the
        # fine grid uni is ~200k bins; rebuilding it per call was wasteful)
        dev = self.device
        te = m0.train_edges.to(dev)
        self._train_edges = te
        self._P = m0.train_energy.shape[0]
        self._x = m0.norm_energy(m0.train_energy).view(1, -1, 1).to(dev)
        self._widths = (te[1:] - te[:-1])
        self._uni = uniform_log_edges(float(te[0]), float(te[-1]), _DLX).to(dev)
        self._K = self._uni.numel() - 1
        # largest E*(embed feats): sets how small echunk must be to bound memory
        self._max_ef = max(len(g.zs) * (1 + 2 * g.models[0].config.n_freqs)
                           for g in self.groups)

    @contextlib.contextmanager
    def grad_enabled(self, on=True):
        """Track gradients through the forward for the duration of the block.

        Off by default: sampling needs only the value, and the autograd graph
        over the trunk is what turned a staged call into a 21 GiB CUDA OOM.
        Gradient-based samplers (NUTS, reparameterised VI) need
        d(counts)/d(theta), which is available through temperature, sigma_v,
        n_h and the abundances alike -- backward costs ~0.8x the forward.

        Note the *inputs* still have to carry ``requires_grad``; this only stops
        the forward from throwing the graph away. Nesting restores the previous
        setting, so a grad block inside a no-grad benchmark is safe."""
        prev = self.track_grad
        self.track_grad = bool(on)
        try:
            yield self
        finally:
            self.track_grad = prev

    @staticmethod
    def _key(m):
        c = m.config
        return (c.hidden_size, c.n_hidden, c.n_freqs, bool(c.use_film),
                int(c.film_t_freqs), bool(c.use_binnorm), bool(c.use_trend),
                bool(c.use_grid), bool(c.use_fourier), c.cond_hidden,
                c.cond_layers, c.n_params, c.activation)

    def _echunk(self, B, mem_gb):
        """Energy-axis chunk sized so the E-fold trunk embedding stays under the
        budget (peak ~ max_E * B * echunk * feats * 4 bytes).

        The budget gives an upper bound; we then spread the P bins evenly over
        the resulting number of chunks. That only ever shrinks the chunk (so the
        memory bound still holds) and makes every chunk the same size bar a
        remainder, which keeps the static-shape ``torch.compile`` path down to
        one or two compiled graphs instead of one per ragged tail."""
        cap = int(0.5 * float(mem_gb) * 1e9 / max(1, self._max_ef * B * 4))
        cap = max(256, min(self._P, cap))
        n = -(-self._P // cap)                  # ceil: number of chunks needed
        return -(-self._P // n)                 # spread evenly over those n

    @_grad_aware
    def _density(self, temp_kev, echunk, compile_trunk):
        """Batched trunk: log10 continuum density for every element, stacked as
        (N, B, P). Returns (dens, zs) with zs the element order of the N axis.

        The trunk is evaluated on every native energy point. Subsampling it and
        interpolating back was measured and rejected -- the trunk is not a smooth
        continuum, it carries the metals' line structure (see
        docs/session_summary.md), so even stride 2 costs ~44% in flux.

        **Under ``grad_enabled`` each energy chunk is gradient-checkpointed.**
        Chunking alone bounds *forward* memory but not the autograd graph: every
        chunk's activations are kept alive for backward, so the graph grows as
        ``N_elements x B x P x hidden x n_layers`` regardless of ``echunk`` --
        ~40 GB for 28 elements at 8 particles, which no device will hold.
        Checkpointing stores only each chunk's inputs and recomputes its
        activations during backward, trading roughly one extra forward for a
        graph that scales with one chunk instead of all of them. Without it,
        gradient-based sampling is simply not runnable at production scale."""
        B, P, x = temp_kev.numel(), self._P, self._x
        use_ckpt = self.track_grad and torch.is_grad_enabled()
        dens, zs = [], []
        for g in self.groups:
            gd = []
            for lo in range(0, P, echunk):
                hi = min(lo + echunk, P)
                xb = x[:, lo:hi].expand(B, -1, -1)              # (B, chunk, 1)
                bins = torch.arange(lo, hi, device=self.device)
                if use_ckpt:
                    # use_reentrant=False: the reentrant variant needs an input
                    # that requires grad, and temp_kev may not when only the
                    # abundances are being differentiated
                    gd.append(torch.utils.checkpoint.checkpoint(
                        g.log_density, temp_kev, xb, bins, compile_trunk,
                        use_reentrant=False))
                else:
                    gd.append(g.log_density(temp_kev, xb, bins, compile_trunk))
            dens.append(torch.pow(10.0, torch.cat(gd, dim=2)))  # (E, B, P)
            zs += g.zs
        return torch.cat(dens, dim=0), zs                        # (N, B, P)

    @_grad_aware
    def _continuum(self, dens, bin_edges, velocity, absorb, tfun, n_h,
                   redshift, mem_gb):
        """Broaden + absorb + rebin the stacked densities to (N, B, M). The fine
        grid K is large, so the (N*B, K) work is done in row-chunks (per-row
        independent, so numerically identical).

        ``velocity`` and ``n_h`` may be (B,) per-walker tensors. Rows here are
        element-major (row = n*B + b), so neither lines up with the flattened
        stack on its own: the velocity is tiled N times, and the (B, K)
        transmission is gathered by each row's walker index. Both are then
        sliced per row-chunk. A scalar velocity or n_h broadcasts as before."""
        N, B, P = dens.shape
        uni, te, K = self._uni, self._train_edges, self._K
        f_train = (dens * self._widths).reshape(N * B, P)        # (N*B, P)
        vel = velocity
        per_walker_v = torch.is_tensor(vel) and vel.numel() > 1
        if per_walker_v:
            vel = vel.reshape(-1).repeat(N)                      # (N*B,) row-aligned
        trans = None
        if absorb:                                               # observed frame
            uni_cent = torch.sqrt(uni[:-1] * uni[1:])
            trans = tfun(uni_cent / (1.0 + redshift), n_h, device=self.device)
        # a per-walker n_h gives (B, K); gather rather than tile to (N*B, K),
        # which at K ~ 2e5 would be gigabytes for no reason.
        per_walker_nh = absorb and trans.dim() == 2 and trans.shape[0] > 1
        # A FIXED row-chunk, with the final partial block zero-padded up to it.
        # Sizing it as min(N*B, cap) instead made the transform's batch size
        # follow N*B -- which changes whenever the sampler rejects a different
        # number of walkers -- and every distinct batch size is another cached
        # cuFFT plan holding a GPU workspace, until allocation fails with
        # CUFFT_INTERNAL_ERROR partway through a long run. Padded rows are
        # zeros, so they broaden to zeros and are sliced off: no numerical
        # effect, only a stable shape.
        n_rows = N * B
        rchunk = max(1, int(0.25 * float(mem_gb) * 1e9 / max(1, K * 8 * 4)))
        cont = []
        for lo in range(0, n_rows, rchunk):
            hi = min(lo + rchunk, n_rows)
            m = hi - lo
            block = f_train[lo:hi]
            v = vel[lo:hi] if per_walker_v else vel
            rows = torch.arange(lo, hi, device=self.device)
            if m < rchunk:                       # pad up to the fixed shape
                block = torch.cat(
                    [block, block.new_zeros(rchunk - m, block.shape[1])], dim=0)
                if per_walker_v:
                    # repeat the last velocity: it must stay inside the real
                    # range so the padding does not inflate the kernel reach
                    v = torch.cat([v, v[-1:].expand(rchunk - m)])
                rows = torch.cat([rows, rows[-1:].expand(rchunk - m)])
            fu = fft_broaden(scatter_to_grid(block, te, uni), _DLX, v)
            if absorb:
                fu = fu * (trans[rows % B] if per_walker_nh else trans)
            cont.append(rebin_flux(fu, uni, bin_edges)[:m])      # drop padding
        return torch.cat(cont, dim=0).reshape(N, B, -1)          # (N, B, M)

    @_grad_aware
    def _combine(self, cont, zs, abundances, temp_kev, bin_edges, velocity,
                 absorb, tfun, n_h, redshift):
        """Abundance-weight the continua and add the ragged per-element line
        heads -> (B, M).

        Abundances may be scalars or ``(B,)`` per-walker tensors (see
        :func:`abundance_weight`); mixing the two across elements is fine."""
        B = cont.shape[1]
        total = torch.zeros((B, bin_edges.numel() - 1), device=self.device)
        for i, z in enumerate(zs):
            raw = abundances.get(z, 1.0) if abundances else 1.0
            a, skip = abundance_weight(raw, self.device)
            if skip:
                continue
            out_e = cont[i]
            model = self.joint.models[z]
            if model.line_head is not None:
                out_e = out_e + self._lines(model, temp_kev, bin_edges,
                                            velocity, absorb, tfun, n_h, redshift)
            total = total + a * out_e
        return total

    @_grad_aware
    def flux(self, temp_kev, abundances, velocity, bin_edges, absorption=None,
             n_h=0.0, redshift=0.0, echunk=None, mem_gb=2.0, compile_trunk=False):
        """Abundance-weighted summed flux on ``bin_edges`` — see
        ``JointOperatorModel.flux`` for the parameter semantics.

        ``velocity`` is km/s, scalar or a (B,) tensor for a per-walker sigma_v.

        Memory: the vmapped trunk is chunked over energy (``echunk``, auto-sized
        from ``mem_gb`` when None) and the broadening over rows; both are
        numerically transparent. ``compile_trunk`` additionally torch.compiles
        the vmapped group forward (compile ∘ vmap), stacking kernel fusion on top
        of the batching (one-off compile stall on the first call)."""
        device = self.device
        temp_kev = torch.as_tensor(temp_kev, dtype=torch.float32,
                                   device=device).view(-1)
        bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32, device=device)
        absorb = (absorption is not None
                  and float(torch.as_tensor(n_h, dtype=torch.float64).max()) > 0.0)
        tfun = absorption.transmission_torch if absorb else None
        if echunk is None:
            echunk = self._echunk(temp_kev.numel(), mem_gb)
        if compile_trunk:                    # a few chunk shapes per group, +margin
            ensure_recompile_limit(8 * len(self.groups))

        dens, zs = self._density(temp_kev, echunk, compile_trunk)
        cont = self._continuum(dens, bin_edges, velocity, absorb, tfun, n_h,
                               redshift, mem_gb)
        return self._combine(cont, zs, abundances, temp_kev, bin_edges,
                             velocity, absorb, tfun, n_h, redshift)

    def _lines(self, model, temp_kev, bin_edges, velocity, absorb, tfun,
               n_h, redshift):
        """Analytic line deposit for one element (mirrors
        ``element_broadened_flux``'s line block exactly)."""
        tnorm = model.norm_temp(temp_kev).view(-1, model.config.n_params)
        lh = model.line_head
        line_flux = torch.pow(10.0, lh.all_line_amplitudes(tnorm)) * lh.line_widths
        if absorb:
            line_flux = line_flux * tfun(lh.line_energies / (1.0 + redshift),
                                         n_h, device=self.device)
        return deposit_gaussian_lines(lh.line_energies, line_flux, bin_edges,
                                      velocity)
