"""The FFT transform shape must come from a small, FIXED set.

cuFFT caches one plan per distinct transform shape, each pinning a GPU
workspace, and a long run that keeps producing new shapes eventually dies with
CUFFT_INTERNAL_ERROR partway through. The shape is a ``(rows, length)`` PAIR,
so the two axes multiply and both have to be pinned:

* ``length`` = K + 2*pad, with the pad tracking the batch's maximum sigma_v.
  ``FFT_PAD_QUANTUM`` is sized so the whole fitted velocity range lands in one
  quantum -- this axis should be a constant, not merely quantised.
* ``rows``, quantised to powers of two in ``batched._continuum``. A constant
  there would be better still for the plan cache, but it made every small
  batch pay for the memory cap's worth of rows, which is most of the
  contracted forward.

These run on CPU: the shape count is device-independent, so the guard does not
need a GPU even though the failure it prevents is GPU-only.
"""
import contextlib
import os

import numpy as np
import pytest
import torch

from spexai.broadening import fft_broaden
from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR

_DLX = 1e-5
# the campaign's sigma_v prior; the pad quantum is sized against this range
SIGMA_V = [30.0, 75.0, 150.0, 300.0, 450.0, 600.0]


@contextlib.contextmanager
def fft_shapes():
    """Collect the shape of every real FFT taken inside the block."""
    seen = set()
    real = torch.fft.rfft

    def spy(x, *args, **kwargs):
        seen.add(tuple(x.shape))
        return real(x, *args, **kwargs)

    torch.fft.rfft = spy
    try:
        yield seen
    finally:
        torch.fft.rfft = real


def test_one_transform_length_across_the_velocity_prior():
    flux = torch.zeros(2, 20000)
    flux[:, 10000] = 1.0
    with fft_shapes() as seen:
        for v in SIGMA_V:
            fft_broaden(flux, _DLX, v)
    lengths = {s[-1] for s in seen}
    assert len(lengths) == 1, f"{len(lengths)} transform lengths: {lengths}"


def test_one_transform_length_for_per_walker_velocities():
    """A per-walker sigma_v takes the batch maximum, so a batch that mixes the
    ends of the prior must still land on the same length as either end alone."""
    flux = torch.zeros(3, 20000)
    flux[:, 10000] = 1.0
    with fft_shapes() as seen:
        fft_broaden(flux, _DLX, torch.tensor([30.0, 300.0, 600.0]))
        fft_broaden(flux, _DLX, torch.tensor([600.0, 600.0, 600.0]))
        fft_broaden(flux, _DLX, 30.0)
    assert len({s[-1] for s in seen}) == 1


def test_padding_still_covers_the_kernel():
    """Quantising up is only safe because it adds zeros. Guard the property
    that actually matters: a line broadened near the edge must not wrap."""
    K = 20000
    flux = torch.zeros(1, K)
    flux[0, 40] = 1.0                       # deliberately close to the edge
    out = fft_broaden(flux, _DLX, 600.0)
    assert torch.isfinite(out).all()
    assert float(out[0, -200:].sum()) < 1e-6 * float(out.sum())


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
                    reason="model store not present")
def test_forward_shape_set_is_bounded_and_history_free():
    """The whole forward, over the walker counts a sampler actually produces.

    The property to assert is NOT "the shape count stopped growing" -- with a
    quantised row axis, a walker count seen for the first time late in a run
    legitimately contributes a shape, and a growth criterion cannot tell that
    apart from something unbounded. What must hold is stronger and exact: the
    shape is a pure function of the walker count, drawn from a set that is
    bounded before the run starts. So sweep every walker count twice and
    require the second sweep to introduce nothing.
    """
    joint = JointOperatorModel(device="cpu", elements=[2, 8, 26])
    edges = torch.linspace(0.5, 9.0, 2001)
    rng = np.random.default_rng(0)
    counts = list(range(1, 9))
    first, second = set(), set()
    for half in (first, second):
        with fft_shapes() as seen:
            for b in counts:
                T = torch.as_tensor(rng.uniform(1.0, 8.0, b),
                                    dtype=torch.float32)
                v = torch.as_tensor(rng.uniform(30.0, 600.0, b),
                                    dtype=torch.float32)
                joint.batched.flux(T, {}, v, edges, contract=True)
        half |= seen
        rng = np.random.default_rng(1)       # different temperatures/velocities
    assert len({s[-1] for s in first | second}) == 1, "transform length varies"
    rows = {s[0] for s in first | second}
    assert all(r & (r - 1) == 0 for r in rows), f"not powers of 2: {rows}"
    # 8 walker counts collapse onto 4 row shapes: 1, 2, 4, 8
    assert len(first | second) <= len(counts), f"{len(first | second)} shapes"
    assert not (second - first), (
        f"shape depends on call history, not walker count: {second - first}")
