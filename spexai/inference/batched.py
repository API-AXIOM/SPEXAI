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
import copy

import torch
from torch.func import functional_call, stack_module_state

from spexai.train.operator import SpectralOperator
from spexai.train.broadening import (deposit_gaussian_lines, fft_broaden,
                                      rebin_flux, scatter_to_grid,
                                      uniform_log_edges)

_DLX = 1e-5  # fine-grid log spacing for the FFT broadening (matches the serial)


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

    def log_density(self, temp_kev, x, bins):
        """log10 continuum density for every element in the group.

        temp_kev: (B,) physical keV; x: (B, P, 1) shared normalised coords;
        bins: (P,) long grid indices. Returns (E, B, P)."""
        # per-element temperature normalisation (t_lo/t_hi may differ)
        tnorm = torch.stack([m.norm_temp(temp_kev).view(-1, self.n_params)
                             for m in self.models], dim=0)      # (E, B, n_params)
        return self._vfwd(self.params, self.buffers, tnorm, x, bins)


class BatchedJointForward:
    """Element-batched analogue of ``JointOperatorModel.flux``.

    Construct once from a loaded ``JointOperatorModel`` (reuses its models and
    device); call :meth:`flux` with the same signature as
    ``JointOperatorModel.flux``. Numerically equivalent, but the trunk nets are
    evaluated grouped instead of looped."""

    def __init__(self, joint):
        self.joint = joint
        self.device = joint.device
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

    @staticmethod
    def _key(m):
        c = m.config
        return (c.hidden_size, c.n_hidden, c.n_freqs, bool(c.use_film),
                int(c.film_t_freqs), bool(c.use_binnorm), bool(c.use_trend),
                bool(c.use_grid), bool(c.use_fourier), c.cond_hidden,
                c.cond_layers, c.n_params, c.activation)

    @torch.no_grad()
    def flux(self, temp_kev, abundances, velocity, bin_edges,
             absorption=None, n_h=0.0, redshift=0.0, echunk=None, mem_gb=2.0):
        """Abundance-weighted summed flux on ``bin_edges`` — see
        ``JointOperatorModel.flux`` for the parameter semantics.

        Two memory knobs (both numerically transparent, energy-/row-independent):
        the P (energy) axis of the vmapped trunk is processed in ``echunk`` bins
        and the fine-grid broadening in row-chunks. With ``echunk=None`` both are
        sized from ``mem_gb`` (a soft per-intermediate GPU budget) using the
        largest element group and the walker batch, so peak memory does not blow
        up with the group size the way a fixed chunk would."""
        device = self.device
        temp_kev = torch.as_tensor(temp_kev, dtype=torch.float32,
                                   device=device).view(-1)
        bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32, device=device)
        B = temp_kev.numel()

        m0 = self._m0
        train_edges = m0.train_edges
        P = m0.train_energy.shape[0]
        x = m0.norm_energy(m0.train_energy).view(1, -1, 1)      # (1, P, 1) shared
        widths = train_edges[1:] - train_edges[:-1]             # (P,) shared
        uni = uniform_log_edges(float(train_edges[0]),
                                float(train_edges[-1]), _DLX).to(device)
        K = uni.numel() - 1

        absorb = (absorption is not None
                  and float(torch.as_tensor(n_h, dtype=torch.float64).max()) > 0.0)
        tfun = absorption.transmission_torch if absorb else None

        budget = float(mem_gb) * 1e9
        # trunk embedding peak ~ E * B * echunk * (1 + 2*n_freqs) floats; size
        # echunk from the largest group so the E-fold stack stays bounded.
        if echunk is None:
            max_ef = max(len(g.zs) * (1 + 2 * g.models[0].config.n_freqs)
                         for g in self.groups)
            echunk = int(0.5 * budget / max(1, max_ef * B * 4))
            echunk = max(256, min(P, echunk))

        # --- batched trunk: continuum density for every element -------------
        dens, zs_all = [], []
        for g in self.groups:
            gd = []
            for lo in range(0, P, echunk):
                hi = min(lo + echunk, P)
                xb = x[:, lo:hi].expand(B, -1, -1)              # (B, chunk, 1)
                bins = torch.arange(lo, hi, device=device)
                gd.append(g.log_density(temp_kev, xb, bins))    # (E, B, chunk)
            dens.append(torch.pow(10.0, torch.cat(gd, dim=2)))  # (E, B, P)
            zs_all += g.zs
        dens = torch.cat(dens, dim=0)                            # (N, B, P)
        N = dens.shape[0]

        # --- broaden all elements together (velocity + N_H shared), but in
        #     row-chunks: scatter/FFT/rebin are per-row, and the fine grid K is
        #     large, so the full (N*B, K) stack is the second memory hotspot ----
        f_train = (dens * widths).reshape(N * B, P)              # (N*B, P)
        trans = None
        if absorb:                                               # observed frame
            uni_cent = torch.sqrt(uni[:-1] * uni[1:])
            trans = tfun(uni_cent / (1.0 + redshift), n_h, device=device)
        rchunk = max(1, min(N * B, int(0.25 * budget / max(1, K * 8 * 4))))
        cont = []
        for lo in range(0, N * B, rchunk):
            fr = f_train[lo:lo + rchunk]                         # (rc, P)
            fu = fft_broaden(scatter_to_grid(fr, train_edges, uni), _DLX, velocity)
            if absorb:
                fu = fu * trans
            cont.append(rebin_flux(fu, uni, bin_edges))          # (rc, M)
        cont = torch.cat(cont, dim=0).reshape(N, B, -1)          # (N, B, M)

        # --- abundance weight + ragged per-element line heads ---------------
        total = torch.zeros((B, bin_edges.numel() - 1), device=device)
        for i, z in enumerate(zs_all):
            a = float(abundances.get(z, 1.0)) if abundances else 1.0
            if a == 0.0:
                continue
            out_e = cont[i]
            model = self.joint.models[z]
            if model.line_head is not None:
                out_e = out_e + self._lines(model, temp_kev, bin_edges,
                                            velocity, absorb, tfun, n_h, redshift)
            total = total + a * out_e
        return total                                             # (B, M)

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
