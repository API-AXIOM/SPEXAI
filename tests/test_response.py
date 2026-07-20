"""Unit tests for OGIP response parsing and folding."""
import numpy as np
import torch


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
