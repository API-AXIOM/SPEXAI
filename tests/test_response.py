"""Unit tests for OGIP response parsing and folding."""
import shutil

import numpy as np
import scipy.sparse as sp
import torch

from spexai.inference.response import Response, load_arf, load_rmf


# --- cold RMF parse (the OGIP decompression, bypassing the .npz cache) ------

def test_cold_rmf_parse_invariants_and_matches_cache(acis_paths, tmp_path):
    """Force the FITS-parse path (copy the RMF without its .npz sidecar) and
    check OGIP invariants + that it reproduces the pre-built cache."""
    rmf, _ = acis_paths
    dst = tmp_path / "acis.rmf"
    shutil.copy(rmf, dst)                       # no .npz alongside -> cold parse
    edges, R, e_min, e_max = load_rmf(str(dst))

    n_e, n_c = len(edges) - 1, R.shape[1]
    assert R.shape == (n_e, n_c)
    assert len(e_min) == n_c and len(e_max) == n_c
    assert R.min() >= 0.0                        # response is non-negative
    row_sums = np.asarray(R.sum(axis=1)).ravel()
    assert row_sums.max() <= 1.05                # rows ~ probabilities (small
    #                                              calibration overshoot is normal)
    assert R.indices.min() >= 0 and R.indices.max() < n_c  # channels in range
    assert np.all(edges[1:] > edges[:-1])        # contiguous/monotonic

    z = np.load(rmf + ".npz")                    # cache built by the same logic
    R_cached = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                             shape=tuple(z["shape"]))
    assert R.shape == R_cached.shape
    assert np.allclose(R.toarray(), R_cached.toarray(), atol=1e-8)


def test_cold_parse_writes_cache(acis_paths, tmp_path):
    rmf, _ = acis_paths
    dst = tmp_path / "acis.rmf"
    shutil.copy(rmf, dst)
    assert not (tmp_path / "acis.rmf.npz").exists()
    load_rmf(str(dst))
    assert (tmp_path / "acis.rmf.npz").exists()   # cache is created on cold parse


# --- ARF branches ----------------------------------------------------------

def test_response_without_arf(acis_paths):
    rmf, _ = acis_paths
    r = Response(rmf, arf_path=None)
    assert not r.has_arf
    assert torch.all(r.arf == 1.0)
    assert r.arf.shape[0] == r.n_energy


def test_arf_regridded_when_grid_mismatches(acis_paths, monkeypatch):
    """A matched-grid ARF is used as-is; a mismatched one is interpolated onto
    the RMF grid. Force the mismatch branch and check it still lands on the
    RMF energy grid."""
    import spexai.inference.response as resp_mod
    rmf, arf = acis_paths
    resp_vals, real_edges = load_arf(arf)
    # same values but shifted edges -> triggers the interpolation fallback
    monkeypatch.setattr(resp_mod, "load_arf",
                        lambda p: (resp_vals, real_edges * 1.001))
    r = resp_mod.Response(rmf, arf)
    assert r.has_arf
    assert r.arf.shape[0] == r.n_energy          # regridded onto the RMF grid


def test_load_arf_shapes(acis_paths):
    _, arf = acis_paths
    resp, edges = load_arf(arf)
    assert edges.shape[0] == resp.shape[0] + 1
    assert (resp >= 0).all()



def test_response_shapes(acis_response):
    r = acis_response
    assert r.energy_edges.shape[0] == r.n_energy + 1
    assert r.chan_e_min.shape[0] == r.n_channels
    assert r.chan_e_max.shape[0] == r.n_channels
    assert r.arf.shape[0] == r.n_energy
    assert r.R.shape == (r.n_energy, r.n_channels)


def test_energy_grid_monotonic(acis_response):
    e = acis_response.energy_edges
    assert torch.all(e[1:] > e[:-1])


def test_fold_orientation(acis_response):
    """A unit flux in a single incident-energy bin must produce exactly
    arf[e] * R[e, :] in channel space. Directly checks fold orientation
    (energy->rows, channels->cols) and ARF application."""
    r = acis_response
    e = int(np.asarray(r.R.sum(axis=1)).ravel().argmax())  # bin with response
    flux = torch.zeros(r.n_energy)
    flux[e] = 1.0
    counts = r.fold(flux).numpy()
    expected = float(r.arf[e]) * np.asarray(r.R[e].todense()).ravel()
    assert counts.sum() > 0                      # this bin redistributes somewhere
    assert np.allclose(counts, expected, atol=1e-6)


def test_fold_nonnegative(acis_response):
    counts = acis_response.fold(torch.rand(acis_response.n_energy))
    assert (counts >= 0).all()


def test_fold_is_linear(acis_response):
    flux = torch.rand(acis_response.n_energy)
    c1 = acis_response.fold(flux)
    c2 = acis_response.fold(2.0 * flux)
    assert torch.allclose(c2, 2.0 * c1, rtol=1e-5)


def test_fold_batch(acis_response):
    counts = acis_response.fold(torch.rand(5, acis_response.n_energy))
    assert counts.shape == (5, acis_response.n_channels)


def test_arf_applied(acis_response):
    """fold with ARF differs from fold without it (unless ARF is all ones)."""
    r = acis_response
    flux = torch.rand(r.n_energy)
    with_arf = r.fold(flux).numpy()
    without = np.asarray(flux.numpy() @ r.R)
    assert not np.allclose(with_arf, without)
