"""Tests for the independent PCHIP SPEX-truth model."""
import json
import os

import numpy as np
import pytest
import torch

from spexai.inference.spex_truth import ElementTruth, SpexTruthModel
from spexai.inference.operator_model import MODELS_DIR
from spexai.train.operator import edges_from_centers


def _synthetic_element(n_bins=50, n_t=5):
    lt = np.log10(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))       # keV nodes
    # distinct log10 flux-density per temperature node so node recovery bites
    base = np.linspace(-4.0, -2.0, n_bins)
    Y = np.stack([base + 0.3 * k for k in range(n_t)]).astype(np.float32)
    centers = np.logspace(np.log10(0.5), np.log10(10.0), n_bins).astype(np.float32)
    return lt, Y, centers


def test_element_truth_reproduces_node():
    lt, Y, centers = _synthetic_element()
    el = ElementTruth(lt, Y, centers)
    widths = (edges_from_centers(torch.as_tensor(centers)).diff()).numpy()
    # temp = 3.0 keV is the 3rd node -> PCHIP passes through Y[2] exactly
    f = el.native_flux(3.0).numpy()
    assert f.shape == (1, 50)
    np.testing.assert_allclose(f[0], 10.0 ** Y[2] * widths, rtol=1e-4)


def test_element_truth_offgrid_is_finite_positive():
    lt, Y, centers = _synthetic_element()
    el = ElementTruth(lt, Y, centers)
    f = el.native_flux([2.5, 3.5]).numpy()                   # between nodes
    assert f.shape == (2, 50)
    assert np.all(np.isfinite(f)) and np.all(f > 0)


def _truth_available():
    if not os.path.exists(os.path.join(MODELS_DIR, "manifest.json")):
        return False
    from spexai.eval import _default_datadir
    man = json.load(open(os.path.join(MODELS_DIR, "manifest.json")))
    dd = _default_datadir(MODELS_DIR, man)
    return os.path.isdir(os.path.join(dd, "element26"))


@pytest.mark.skipif(not _truth_available(), reason="SPEX caches not present")
def test_spex_truth_model_flux_shape_and_positive():
    truth = SpexTruthModel(elements=[26], device="cpu")
    edges = torch.logspace(np.log10(0.5), np.log10(9.0), 101)
    f = truth.flux(torch.tensor([4.0]), {}, velocity=150.0, bin_edges=edges)
    assert f.shape == (1, 100)
    assert torch.all(torch.isfinite(f)) and float(f.sum()) > 0
