"""Shared fixtures for the inference unit tests.

Tests run off the trained checkpoints in `spexai/models/` and a small
cached instrument response (Chandra ACIS); they skip cleanly if those
artifacts are not present (e.g. a fresh checkout without the model store).
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR

MODELS_OK = (os.path.exists(os.path.join(MODELS_DIR, "manifest.json"))
             and os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")))

RESP_DIR = os.path.expanduser("~/work/data/spexai/responses")
ACIS_RMF = os.path.join(RESP_DIR, "aciss_aimpt_cy28.rmf")
ACIS_ARF = os.path.join(RESP_DIR, "aciss_aimpt_cy28.arf")
ACIS_OK = os.path.exists(ACIS_RMF) and os.path.exists(ACIS_ARF)


@pytest.fixture(scope="session")
def fe_model():
    if not MODELS_OK:
        pytest.skip("model store not present")
    return JointOperatorModel(device="cpu", elements=[26])


@pytest.fixture(scope="session")
def small_joint():
    if not MODELS_OK:
        pytest.skip("model store not present")
    return JointOperatorModel(device="cpu", elements=[2, 26])


@pytest.fixture(scope="session")
def acis_response():
    if not ACIS_OK:
        pytest.skip("Chandra ACIS response not present")
    from spexai.inference.response import Response
    return Response(ACIS_RMF, ACIS_ARF)


@pytest.fixture
def edges():
    return torch.logspace(np.log10(0.3), np.log10(10.0), 201)
