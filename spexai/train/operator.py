"""Operator-style spectral emulator, following Ricketts, Hadzi Veljkovic et al. (2026).

The spectrum is modelled as a continuous function of energy conditioned on
physical parameters: log10-flux = f(x; theta), where x is the normalised
log10-energy coordinate and theta the (normalised) model parameters
(for the CIE single-element case: temperature).

Components (each can be disabled for ablations):
  * Fourier feature embedding of x, with a coarse-to-fine curriculum
  * FiLM conditioning of the trunk on theta
  * trend head predicting a(theta) * x + b(theta), factoring out the
    smooth global spectral slope
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


@dataclass
class OperatorConfig:
    # ablation switches
    use_fourier: bool = True
    use_film: bool = True
    use_trend: bool = True
    use_grid: bool = False      # multi-resolution grid encoding instead of Fourier
    use_linehead: bool = False  # dedicated per-line amplitude head
    use_binnorm: bool = False   # per-training-bin normalisation of the target
    # fourier embedding
    n_freqs: int = 256
    f_min: float = 0.25
    f_max: float = 8000.0
    # multi-resolution grid encoding (Instant-NGP style; dense in 1D)
    grid_levels: int = 12
    grid_base_res: int = 32
    grid_max_res: int = 32768
    grid_dim: int = 4
    # line head
    line_dim: int = 16
    line_hidden: int = 128
    # trunk
    hidden_size: int = 256
    n_hidden: int = 6
    activation: str = "gelu"  # gelu | silu | tanh | sine
    # conditioning network (FiLM generator / theta embedding)
    cond_hidden: int = 128
    cond_layers: int = 3
    n_params: int = 1
    # coordinate/parameter normalisation (set from data before training)
    x_lo: float = math.log10(0.1)
    x_hi: float = math.log10(12.0)
    t_lo: float = math.log10(0.5)
    t_hi: float = math.log10(20.0)


class FourierFeatures(nn.Module):
    """Map scalar coordinate x in [-1, 1] to [x, cos(2 pi f_k x), sin(2 pi f_k x)].

    A coarse-to-fine weight per frequency channel implements the curriculum:
    with progress alpha in [0, 1], channel k gets weight
    clip(alpha * K - k, 0, 1), so low frequencies are active from the start
    and high frequencies fade in as training progresses.
    """

    def __init__(self, n_freqs, f_min, f_max):
        super().__init__()
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs)
        self.register_buffer("freqs", freqs)
        self.n_freqs = n_freqs

    @property
    def out_dim(self):
        return 1 + 2 * self.n_freqs

    def forward(self, x, alpha=1.0):
        # x: (..., 1)
        phase = 2.0 * math.pi * x * self.freqs
        feats = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        if alpha < 1.0:
            k = torch.arange(self.n_freqs, device=x.device)
            w = torch.clamp(alpha * self.n_freqs - k, 0.0, 1.0)
            feats = feats * torch.cat([w, w])
        return torch.cat([x, feats], dim=-1)


def edges_from_centers(centers):
    """Bin edges for a (log-spaced) grid of bin centers: geometric means of
    neighbouring centers, with the end edges extrapolated at the same ratio."""
    inner = torch.sqrt(centers[:-1] * centers[1:])
    first = centers[0] ** 2 / inner[0]
    last = centers[-1] ** 2 / inner[-1]
    return torch.cat([first.view(1), inner, last.view(1)])


class Sine(nn.Module):
    """SIREN-style sinusoidal activation. With a Fourier/grid embedding in
    front, w0 = 1 is appropriate (frequencies come from the embedding)."""

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def make_activation(name):
    return {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
            "sine": Sine}[name]()


class GridFeatures(nn.Module):
    """Multi-resolution learned grid encoding of a 1D coordinate (dense
    Instant-NGP): each level holds a feature table over a uniform grid in
    [-1, 1]; features are linearly interpolated at x and concatenated.

    Because the tables are dense in 1D there are no hash collisions. The
    coarse-to-fine curriculum weight per level mirrors FourierFeatures.
    """

    def __init__(self, n_levels, base_res, max_res, level_dim):
        super().__init__()
        growth = (max_res / base_res) ** (1.0 / max(1, n_levels - 1))
        self.resolutions = [int(round(base_res * growth ** l)) for l in range(n_levels)]
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.empty(r + 1, level_dim).uniform_(-1e-3, 1e-3))
             for r in self.resolutions])
        self.n_levels = n_levels
        self.level_dim = level_dim

    @property
    def out_dim(self):
        return 1 + self.n_levels * self.level_dim

    def forward(self, x, alpha=1.0):
        # x: (..., 1) in [-1, 1]
        feats = [x]
        u01 = (x.squeeze(-1) + 1.0) * 0.5
        for l, (r, tab) in enumerate(zip(self.resolutions, self.tables)):
            u = torch.clamp(u01, 0.0, 1.0) * r
            i0 = torch.clamp(u.long(), max=r - 1)
            frac = (u - i0.float()).unsqueeze(-1)
            f = tab[i0] * (1.0 - frac) + tab[i0 + 1] * frac
            if alpha < 1.0:
                w = min(max(alpha * self.n_levels - l, 0.0), 1.0)
                if w == 0.0:
                    f = f * 0.0
                elif w < 1.0:
                    f = f * w
            feats.append(f)
        return torch.cat(feats, dim=-1)


class LineHead(nn.Module):
    """Amplitude head for spectral lines at fixed, known energy bins.

    Each line bin gets a learned embedding; a shared MLP maps
    [line embedding, conditioning vector c(theta)] to a log10 line flux.
    The head output is flux-added (logaddexp) to the smooth trunk output,
    so the trunk only has to model the continuum.
    """

    def __init__(self, line_ids, n_params, cond_hidden, cond_layers,
                 line_dim=16, hidden=128, floor=-12.0, energy_grid=None):
        super().__init__()
        # line_ids: (n_bins,) long, slot index for line bins, -1 elsewhere
        self.register_buffer("line_ids", line_ids)
        n_lines = int(line_ids.max().item()) + 1
        self.n_lines = n_lines
        # energy of each line slot and the width of its training-grid bin;
        # these make the head portable to arbitrary instrument grids: the
        # head predicts log10 flux DENSITY on the training grid, so the
        # integrated line flux is 10^amp * line_widths, which can then be
        # re-deposited into any target bin (see deposit()).
        if energy_grid is not None:
            edges = edges_from_centers(energy_grid)
            sel = torch.nonzero(line_ids >= 0).squeeze(-1)
            self.register_buffer("line_energies", energy_grid[sel].clone())
            self.register_buffer("line_widths",
                                 (edges[1:] - edges[:-1])[sel].clone())
        else:  # legacy checkpoints; deposit() unavailable
            self.register_buffer("line_energies", torch.zeros(n_lines))
            self.register_buffer("line_widths", torch.zeros(n_lines))
        self.embed = nn.Embedding(n_lines, line_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.cond = CondNet(n_params, cond_hidden, cond_layers, cond_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(line_dim + cond_hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.floor = floor

    def forward(self, tnorm, bins, trunk_out):
        """tnorm: (B, n_params); bins: (P,) long bin indices;
        trunk_out: (B, P) log10 flux. Returns combined (B, P)."""
        lid = self.line_ids[bins]           # (P,)
        sel = lid >= 0
        if not torch.any(sel):
            return trunk_out
        B = tnorm.shape[0]
        emb = self.embed(lid[sel])          # (Nl, D)
        c = self.cond(tnorm)                # (B, C)
        nl = emb.shape[0]
        h = torch.cat([emb.unsqueeze(0).expand(B, -1, -1),
                       c.unsqueeze(1).expand(-1, nl, -1)], dim=-1)
        line_log10 = self.mlp(h).squeeze(-1) + self.floor  # (B, Nl), starts tiny
        # add fluxes in linear space: log10(10^a + 10^b)
        ln10 = math.log(10.0)
        combined = torch.logaddexp(trunk_out[:, sel] * ln10,
                                   line_log10 * ln10) / ln10
        out = trunk_out.clone()
        out[:, sel] = combined
        return out

    def all_line_amplitudes(self, tnorm):
        """log10 flux density (training grid) of every line, shape (B, n_lines)."""
        B = tnorm.shape[0]
        emb = self.embed.weight
        c = self.cond(tnorm)
        h = torch.cat([emb.unsqueeze(0).expand(B, -1, -1),
                       c.unsqueeze(1).expand(-1, self.n_lines, -1)], dim=-1)
        return self.mlp(h).squeeze(-1) + self.floor

    def deposit(self, tnorm, bin_edges, trunk_int):
        """Add the lines' integrated fluxes onto an arbitrary instrument grid.

        tnorm: (B, n_params); bin_edges: (M+1,) target bin edges in keV,
        ascending; trunk_int: (B, M) LINEAR integrated trunk flux per bin.
        Each line contributes its full integrated flux
        (10^amp * its training bin width) to whichever target bin contains
        its energy, so line flux is conserved exactly. Returns linear
        integrated flux per bin, shape (B, M).
        """
        amp = self.all_line_amplitudes(tnorm)                      # (B, n_lines)
        flux = torch.pow(10.0, amp) * self.line_widths             # integrated
        j = torch.searchsorted(bin_edges, self.line_energies) - 1  # target bin
        inside = (j >= 0) & (j < bin_edges.shape[0] - 1)
        j_in = j[inside].clamp(min=0)
        out = trunk_int.clone()
        out.index_add_(1, j_in, flux[:, inside])
        return out


class CondNet(nn.Module):
    """Small MLP embedding of the (normalised) physical parameters."""

    def __init__(self, n_params, hidden, n_layers, out_dim):
        super().__init__()
        layers = []
        d = n_params
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


class SpectralOperator(nn.Module):
    """log10-flux(x; theta) with optional Fourier features, FiLM and trend head."""

    def __init__(self, config=None, line_ids=None, energy_grid=None,
                 bin_stats=None, **kwargs):
        super().__init__()
        if config is None:
            config = OperatorConfig(**kwargs)
        self.config = c = config

        if c.use_binnorm:
            # per-training-bin mean/std of the (floor-clamped) log flux: the
            # network predicts the z-scored residual, so the static line
            # forest lives in bn_mu and the trunk only models O(1)
            # temperature-dependent deviations around it
            if bin_stats is None:
                raise ValueError("use_binnorm requires bin_stats=(mu, sigma)")
            mu, sigma = bin_stats
            self.register_buffer("bn_mu", mu.clone())
            self.register_buffer("bn_sigma", sigma.clamp(min=0.05).clone())

        if c.use_grid:
            self.embed = GridFeatures(c.grid_levels, c.grid_base_res,
                                      c.grid_max_res, c.grid_dim)
            in_dim = self.embed.out_dim
        elif c.use_fourier:
            self.embed = FourierFeatures(c.n_freqs, c.f_min, c.f_max)
            in_dim = self.embed.out_dim
        else:
            self.embed = None
            in_dim = 1

        if c.use_linehead:
            if line_ids is None:
                raise ValueError("use_linehead requires line_ids")
            self.line_head = LineHead(line_ids, c.n_params, c.cond_hidden,
                                      c.cond_layers, c.line_dim, c.line_hidden,
                                      energy_grid=energy_grid)
        else:
            self.line_head = None

        cond_dim = c.cond_hidden
        if c.use_film:
            # FiLM generator: gamma and beta for every hidden layer
            self.film = CondNet(c.n_params, c.cond_hidden, c.cond_layers,
                                2 * c.hidden_size * c.n_hidden)
        else:
            # fall back to concatenating a theta embedding to the coordinate
            self.film = None
            self.cond_embed = CondNet(c.n_params, c.cond_hidden, c.cond_layers, cond_dim)
            in_dim += cond_dim

        self.trunk = nn.ModuleList()
        d = in_dim
        for _ in range(c.n_hidden):
            self.trunk.append(nn.Linear(d, c.hidden_size))
            d = c.hidden_size
        self.head = nn.Linear(d, 1)
        self.act = make_activation(c.activation)

        if c.use_trend:
            self.trend = CondNet(c.n_params, c.cond_hidden, c.cond_layers, 2)
        else:
            self.trend = None

        # training grid (centers and derived edges): the trunk is only
        # constrained AT the training centers (the SPEX grid is adaptive,
        # not log-uniform), so forward_on_grid evaluates there and rebins
        if energy_grid is not None:
            self.register_buffer("train_energy", energy_grid.clone())
            self.register_buffer("train_edges", edges_from_centers(energy_grid))

        # normalisation constants, stored so the model is self-contained
        self.register_buffer("x_lo", torch.tensor(float(c.x_lo)))
        self.register_buffer("x_hi", torch.tensor(float(c.x_hi)))
        self.register_buffer("t_lo", torch.tensor(float(c.t_lo)))
        self.register_buffer("t_hi", torch.tensor(float(c.t_hi)))

    # -- normalisation helpers ------------------------------------------------
    def norm_energy(self, energy_kev):
        x = torch.log10(energy_kev)
        return 2.0 * (x - self.x_lo) / (self.x_hi - self.x_lo) - 1.0

    def norm_temp(self, temp_kev):
        t = torch.log10(temp_kev)
        return 2.0 * (t - self.t_lo) / (self.t_hi - self.t_lo) - 1.0

    # -- forward passes -------------------------------------------------------
    def forward_norm(self, tnorm, x, alpha=1.0, bins=None, add_lines=True):
        """tnorm: (B, n_params); x: (B, P, 1) normalised coordinates.
        bins: (P,) long bin indices on the training grid (only needed by
        the line head). add_lines=False returns the trunk-only output
        (used by forward_on_grid, which deposits the lines itself)."""
        B, P, _ = x.shape
        h = self.embed(x, alpha) if self.embed is not None else x

        if self.film is not None:
            gb = self.film(tnorm)  # (B, 2*H*L)
            c = self.config
            gb = gb.view(B, c.n_hidden, 2, c.hidden_size)
            for i, layer in enumerate(self.trunk):
                h = layer(h)
                gamma = 1.0 + gb[:, i, 0].unsqueeze(1)   # (B, 1, H)
                beta = gb[:, i, 1].unsqueeze(1)
                h = self.act(gamma * h + beta)
        else:
            ce = self.cond_embed(tnorm).unsqueeze(1).expand(B, P, -1)
            h = torch.cat([h, ce], dim=-1)
            for layer in self.trunk:
                h = self.act(layer(h))

        out = self.head(h).squeeze(-1)  # (B, P)

        if self.trend is not None:
            ab = self.trend(tnorm)  # (B, 2)
            out = out + ab[:, 0:1] * x.squeeze(-1) + ab[:, 1:2]

        if self.config.use_binnorm:
            # map the z-scored prediction back to log10 flux; bins index the
            # training grid (assumed to be the full grid when omitted)
            if bins is None:
                bins = torch.arange(x.shape[1], device=x.device)
            out = out * self.bn_sigma[bins] + self.bn_mu[bins]

        if self.line_head is not None and add_lines:
            if bins is None:
                bins = torch.arange(x.shape[1], device=x.device)
            out = self.line_head(tnorm, bins, out)
        return out

    def forward(self, temp_kev, energy_kev, alpha=1.0, bins=None):
        """Physical units: temp_kev (B,), energy_kev (P,) or (B, P).

        Returns predicted log10 flux, shape (B, P). If the line head is
        active and `bins` is omitted, energy_kev is assumed to be the full
        training grid.
        """
        tnorm = self.norm_temp(temp_kev).view(-1, self.config.n_params)
        if energy_kev.dim() == 1:
            energy_kev = energy_kev.unsqueeze(0).expand(tnorm.shape[0], -1)
        x = self.norm_energy(energy_kev).unsqueeze(-1)
        return self.forward_norm(tnorm, x, alpha=alpha, bins=bins)

    def forward_on_grid(self, temp_kev, bin_edges):
        """Evaluate the emulator as an X-ray model on an instrument grid.

        temp_kev: (B,) temperatures in keV; bin_edges: (M+1,) ascending
        bin edges in keV. Returns the INTEGRATED flux per bin (linear, not
        log), shape (B, M) - i.e. the quantity proportional to the
        expected photon counts per bin, which is what response folding
        consumes.

        Method: the trunk (a flux density) is only constrained at the
        training-grid energies - the SPEX grid is adaptive, not uniform -
        so it is evaluated exactly there, converted to integrated fluxes
        on the training bins, and rebinned onto the target edges via the
        cumulative flux (flux-conserving for both finer and coarser
        target grids). If the model has a line head, each line's
        integrated flux is then added to the target bin containing its
        energy, so line fluxes land in single bins rather than being
        smeared over their training bin.
        """
        if not hasattr(self, "train_energy"):
            raise RuntimeError("forward_on_grid needs the training grid; "
                               "construct the model with energy_grid=...")
        tnorm = self.norm_temp(temp_kev).view(-1, self.config.n_params)
        B = tnorm.shape[0]

        # integrated flux on the training bins
        x = self.norm_energy(self.train_energy).view(1, -1, 1).expand(B, -1, -1)
        dens = torch.pow(10.0, self.forward_norm(tnorm, x, add_lines=False))
        widths = self.train_edges[1:] - self.train_edges[:-1]
        f_train = dens * widths                                     # (B, K)

        # flux-conserving rebin: piecewise-linear cumulative flux evaluated
        # at the target edges. float64 throughout - recovering faint bins
        # by differencing a cumulative sum cancels catastrophically in
        # float32 (the total flux is ~10 decades above the faintest bins)
        f64 = f_train.double()
        cum = torch.cat([torch.zeros(B, 1, dtype=torch.float64,
                                     device=f_train.device),
                         torch.cumsum(f64, dim=1)], dim=1)
        edges_tr = self.train_edges.double()
        k = torch.searchsorted(edges_tr, bin_edges.double()).clamp(
            1, len(edges_tr) - 1)
        e_lo = edges_tr[k - 1]
        frac = ((bin_edges.double() - e_lo) / (edges_tr[k] - e_lo)).clamp(0., 1.)
        c_at = cum[:, k - 1] + f64[:, (k - 1).clamp(max=f64.shape[1] - 1)] * frac
        trunk_int = (c_at[:, 1:] - c_at[:, :-1]).float()            # (B, M)

        if self.line_head is not None:
            return self.line_head.deposit(tnorm, bin_edges, trunk_int)
        return trunk_int

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        c = self.config
        tags = [t for t, on in [("grid", c.use_grid),
                                ("FF", c.use_fourier and not c.use_grid),
                                ("FiLM", c.use_film), ("trend", c.use_trend),
                                ("lines", c.use_linehead)] if on]
        return (f"SpectralOperator({'+'.join(tags) or 'plain'}_"
                f"H{c.hidden_size}x{c.n_hidden})")


class FixedGridMLP(nn.Module):
    """Baseline: theta -> full flux vector on a fixed energy grid.

    Matches the original SpexAI formulation (an MLP from temperature to all
    bins), used to quantify the effect of the operator formulation.
    """

    def __init__(self, n_bins, n_params=1, hidden_size=512, n_hidden=4,
                 t_lo=math.log10(0.5), t_hi=math.log10(20.0)):
        super().__init__()
        layers = []
        d = n_params
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_size), nn.GELU()]
            d = hidden_size
        layers += [nn.Linear(d, n_bins)]
        self.net = nn.Sequential(*layers)
        self.n_bins = n_bins
        self.register_buffer("t_lo", torch.tensor(float(t_lo)))
        self.register_buffer("t_hi", torch.tensor(float(t_hi)))

    def norm_temp(self, temp_kev):
        t = torch.log10(temp_kev)
        return 2.0 * (t - self.t_lo) / (self.t_hi - self.t_lo) - 1.0

    def forward(self, temp_kev):
        tnorm = self.norm_temp(temp_kev).view(-1, 1)
        return self.net(tnorm)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        return f"FixedGridMLP({self.n_bins})"
