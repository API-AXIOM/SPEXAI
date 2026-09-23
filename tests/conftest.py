"""Shared fixtures for the inference unit tests.

Tests run off the trained checkpoints in the model store (`spexai/models/`
by default, moved by `SPEXAI_STORE`) and a small cached instrument response
(Chandra ACIS); they skip cleanly if those artifacts are not present (e.g. a
fresh checkout without the model store).

A missing store silently disables ~25 tests, so the run ends with a banner
naming what was skipped and where it looked. Set `SPEXAI_REQUIRE_MODELS=1`
(CI, campaign runs) to turn that into a hard failure instead.
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

REQUIRE_MODELS = os.environ.get("SPEXAI_REQUIRE_MODELS", "") not in ("", "0")

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
def acis_paths():
    if not ACIS_OK:
        pytest.skip("Chandra ACIS response not present")
    return ACIS_RMF, ACIS_ARF


@pytest.fixture
def edges():
    return torch.logspace(np.log10(0.3), np.log10(10.0), 201)


def pytest_configure(config):
    """Refuse to run a store-less suite when the caller demanded the store."""
    if REQUIRE_MODELS and not MODELS_OK:
        raise pytest.UsageError(
            f"SPEXAI_REQUIRE_MODELS is set but no model store was found at "
            f"{MODELS_DIR} (need manifest.json and Z26_Fe.pt). Point "
            f"SPEXAI_STORE at the store, or unset SPEXAI_REQUIRE_MODELS.")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say out loud that a green run tested none of the model-backed code.

    Without this the only trace is the skip count, which is easy to read past
    -- and the suite reports success having exercised no inference model.
    """
    if MODELS_OK and ACIS_OK:
        return
    missing = []
    if not MODELS_OK:
        missing.append(f"model store (manifest.json + Z26_Fe.pt) at {MODELS_DIR}"
                       f"  [set SPEXAI_STORE to move it]")
    if not ACIS_OK:
        missing.append(f"Chandra ACIS response at {RESP_DIR}")
    terminalreporter.write_sep("=", "INCOMPLETE RUN: artifacts missing", red=True)
    for item in missing:
        terminalreporter.write_line(f"  missing: {item}")
    terminalreporter.write_line(
        "  Tests depending on these were SKIPPED -- a pass here does not mean "
        "that code works.")
    terminalreporter.write_line(
        "  Set SPEXAI_REQUIRE_MODELS=1 to make a missing model store fail.")
