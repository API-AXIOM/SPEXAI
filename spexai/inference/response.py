"""Instrument response handling for operator-based inference.

Parses OGIP RMF + ARF into a single `Response` object that folds a
model flux (integrated per incident-energy bin, on the RMF's own energy
grid) through effective area and redistribution to detector-channel counts.

The RMF parser is the corrected OGIP decompression (reads the F_CHAN
channel-offset TLMIN, honours N_GRP groups, asserts a contiguous grid,
builds R as (n_energy, n_channels)); the legacy
`spexai.deprecated.write_tensors_cnn.rmf_to_torchmatrix` hardcoded a
zero channel offset and a transposed index and should not be reused.
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
from astropy.io import fits


def load_rmf(path):
    """OGIP RMF -> (energy edges (N+1,), CSR redistribution (N, C),
    channel E_MIN, E_MAX). Parsed matrix is cached next to the file."""
    cache = path + ".npz"
    if os.path.exists(cache):
        z = np.load(cache)
        R = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"]))
        return z["edges"], R, z["e_min"], z["e_max"]
    with fits.open(path) as h:
        m = h["MATRIX"]
        det = m.header["DETCHANS"]
        names = [c.name for c in m.columns]
        tlmin = m.header.get(f"TLMIN{names.index('F_CHAN') + 1}", 1)
        d = m.data
        e_lo = np.asarray(d["ENERG_LO"], dtype=np.float64)
        e_hi = np.asarray(d["ENERG_HI"], dtype=np.float64)
        assert np.allclose(e_lo[1:], e_hi[:-1]), "non-contiguous RMF grid"
        rows, cols, vals = [], [], []
        for i in range(len(d)):
            fch = np.atleast_1d(d["F_CHAN"][i]).astype(np.int64)
            nch = np.atleast_1d(d["N_CHAN"][i]).astype(np.int64)
            mat = np.atleast_1d(d["MATRIX"][i])
            pos = 0
            for g in range(int(d["N_GRP"][i])):
                n = int(nch[g])
                if n <= 0:
                    continue
                f = int(fch[g]) - tlmin
                cols.append(np.arange(f, f + n))
                rows.append(np.full(n, i))
                vals.append(mat[pos:pos + n])
                pos += n
        R = sp.csr_matrix(
            (np.concatenate(vals).astype(np.float64),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(d), det))
        eb = h["EBOUNDS"].data
        e_min = np.asarray(eb["E_MIN"], dtype=np.float64)
        e_max = np.asarray(eb["E_MAX"], dtype=np.float64)
    edges = np.append(e_lo, e_hi[-1])
    np.savez(cache, data=R.data, indices=R.indices, indptr=R.indptr,
             shape=np.array(R.shape), edges=edges, e_min=e_min, e_max=e_max)
    return edges, R, e_min, e_max


def load_arf(path):
    """OGIP ARF -> (SPECRESP (N,), energy edges (N+1,)). Effective area in
    cm^2 per incident-energy bin."""
    with fits.open(path) as h:
        h.verify("fix")
        d = h["SPECRESP"].data
        resp = np.asarray(d["SPECRESP"], dtype=np.float64)
        e_lo = np.asarray(d["ENERG_LO"], dtype=np.float64)
        e_hi = np.asarray(d["ENERG_HI"], dtype=np.float64)
    edges = np.append(e_lo, e_hi[-1])
    return resp, edges


class Response:
    """RMF (+ optional ARF) for one instrument. `energy_edges` is the grid
    the model flux must be evaluated on; `fold` maps that flux to channel
    counts-rate space."""

    def __init__(self, rmf_path, arf_path=None):
        edges, R, e_min, e_max = load_rmf(rmf_path)
        self.energy_edges = torch.tensor(edges, dtype=torch.float32)
        self.energy_lo = self.energy_edges[:-1]
        self.energy_hi = self.energy_edges[1:]
        self.R = R                                   # scipy CSR (N_energy, N_chan)
        self.n_energy, self.n_channels = R.shape
        self.chan_e_min = torch.tensor(e_min, dtype=torch.float32)
        self.chan_e_max = torch.tensor(e_max, dtype=torch.float32)
        self.chan_e_cent = 0.5 * (self.chan_e_min + self.chan_e_max)

        if arf_path is not None:
            resp, arf_edges = load_arf(arf_path)
            if len(resp) == self.n_energy and np.allclose(
                    arf_edges, edges, rtol=1e-4):
                arf = resp
            else:  # ARF on a different grid -> interpolate onto RMF centres
                arf_cent = 0.5 * (arf_edges[:-1] + arf_edges[1:])
                rmf_cent = 0.5 * (edges[:-1] + edges[1:])
                arf = np.interp(rmf_cent, arf_cent, resp, left=0.0, right=0.0)
            self.arf = torch.tensor(arf, dtype=torch.float32)
            self.has_arf = True
        else:
            self.arf = torch.ones(self.n_energy, dtype=torch.float32)
            self.has_arf = False

    def __repr__(self):
        return (f"Response(N_energy={self.n_energy}, N_chan={self.n_channels}, "
                f"band=[{float(self.energy_edges[0]):.3g}, "
                f"{float(self.energy_edges[-1]):.3g}] keV, arf={self.has_arf})")

    def fold(self, flux_per_bin):
        """flux_per_bin: (..., N_energy) integrated flux on ``energy_edges``.
        Returns (..., N_channels) counts-rate: sum_e flux[e]*ARF[e]*R[e, c]."""
        f = torch.as_tensor(flux_per_bin, dtype=torch.float32)
        eff = (f * self.arf).detach().cpu().numpy()     # (..., N_energy)
        counts = np.asarray(eff @ self.R)               # (..., N_channels)
        return torch.tensor(counts, dtype=torch.float32)
