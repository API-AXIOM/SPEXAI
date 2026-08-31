"""Unit tests for the operator loader and JointOperatorModel."""
import os

import numpy as np
import pytest
import torch

import spexai.broadening as _broadening
from spexai.inference.operator_model import (MODELS_DIR, element_broadened_flux,
                                             enable_inference_acceleration,
                                             load_operator)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
    reason="model store not present")


# --- cache-free loader -----------------------------------------------------

def test_load_trunk_only():
    m = load_operator(os.path.join(MODELS_DIR, "Z02_He.pt"))  # He: no lines
    assert m.line_head is None


def test_load_line_head():
    m = load_operator(os.path.join(MODELS_DIR, "Z26_Fe.pt"))   # Fe: line head
    assert m.line_head is not None


def test_load_is_deterministic(edges):
    m = load_operator(os.path.join(MODELS_DIR, "Z26_Fe.pt"))
    t = torch.tensor([2.0])
    a = element_broadened_flux(m, t, 150.0, edges)
    b = element_broadened_flux(m, t, 150.0, edges)
    assert torch.equal(a, b)


def test_element_flux_shape_finite(edges):
    m = load_operator(os.path.join(MODELS_DIR, "Z26_Fe.pt"))
    f = element_broadened_flux(m, torch.tensor([1.0, 4.0]), 200.0, edges)
    assert f.shape == (2, edges.numel() - 1)
    assert torch.isfinite(f).all()
    assert (f >= 0).all()


# --- joint model -----------------------------------------------------------

def test_flux_shape_finite_positive(small_joint, edges):
    T = torch.tensor([0.7, 2.0, 8.0])
    f = small_joint.flux(T, {}, 150.0, edges)
    assert f.shape == (3, edges.numel() - 1)
    assert torch.isfinite(f).all()
    assert (f >= 0).all()


def test_abundance_is_linear(fe_model, edges):
    T = torch.tensor([6.0])
    f1 = fe_model.flux(T, {26: 1.0}, 150.0, edges)
    f2 = fe_model.flux(T, {26: 2.0}, 150.0, edges)
    assert torch.allclose(f2, 2.0 * f1, rtol=1e-5, atol=1e-30)


def test_zero_abundance_returns_zeros(fe_model, edges):
    # regression: flux() used to return None if every element was zeroed
    f = fe_model.flux(torch.tensor([6.0]), {26: 0.0}, 150.0, edges)
    assert f.shape == (1, edges.numel() - 1)
    assert torch.all(f == 0)


def test_missing_elements_reported(small_joint):
    assert set(small_joint.models) == {2, 26}
    # manifest tracks the full periodic range; the store is now complete (Z1-30)
    assert isinstance(small_joint.manifest["missing_elements"], list)
    assert set(map(int, small_joint.manifest["elements"])) >= {2, 26}


# --- inference acceleration (CUDA-only; must be a no-op off CUDA) -----------

@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_acceleration_is_noop_off_cuda(device):
    fe = load_operator(os.path.join(MODELS_DIR, "Z26_Fe.pt"))
    before = _broadening.USE_FLOAT32_FFT
    enabled = enable_inference_acceleration([fe], device)
    assert enabled == []                               # nothing turned on
    assert _broadening.USE_FLOAT32_FFT == before       # fft32 flag untouched


def test_joint_accel_default_noop_on_cpu(small_joint):
    # default accelerate=True on a CPU model must leave numerics byte-identical
    assert small_joint.accel == []
    assert _broadening.USE_FLOAT32_FFT is False


# --- counts / response folding via the model -------------------------------

def test_predict_counts_shape_nonneg(fe_model, acis_response):
    c = fe_model.predict_counts(torch.tensor([2.0]), {}, -10.0, 1e10, 150.0,
                                acis_response, exposure=1e4)
    assert c.shape == (1, acis_response.n_channels)
    assert torch.isfinite(c).all()
    assert (c >= 0).all()


def test_counts_linear_in_norm_and_exposure(fe_model, acis_response):
    args = dict(temp_kev=torch.tensor([2.0]), abundances={}, logz=-10.0,
                velocity=150.0, response=acis_response)
    base = fe_model.predict_counts(norm=1e10, exposure=1e4, **args)
    assert torch.allclose(fe_model.predict_counts(norm=2e10, exposure=1e4,
                                                  **args), 2 * base, rtol=1e-5)
    assert torch.allclose(fe_model.predict_counts(norm=1e10, exposure=2e4,
                                                  **args), 2 * base, rtol=1e-5)


def test_redshift_changes_spectrum(fe_model, acis_response):
    args = dict(temp_kev=torch.tensor([6.0]), abundances={}, norm=1e10,
                velocity=150.0, response=acis_response, exposure=1e4)
    c0 = fe_model.predict_counts(logz=-10.0, **args)          # z ~ 0
    cz = fe_model.predict_counts(logz=float(np.log10(0.3)), **args)  # z = 0.3
    assert not torch.allclose(c0, cz)
