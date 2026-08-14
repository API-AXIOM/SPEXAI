"""Element-batched forward must match the serial JointOperatorModel.flux.

The whole point of BatchedJointForward is that it changes *how* the per-element
trunks are evaluated (grouped vmap) without changing the numbers, so these tests
assert equivalence to the reference serial path across the abundance and
absorption code paths. Element set is chosen to exercise a real grouped pass
(Fe+Si share the big 384/5/512 architecture -> E=2 in one vmap group) alongside
a trunk-only element (O has no line head) and a small singleton group (He)."""
import os

import pytest
import torch

from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
    reason="model store not present")

ELEMENTS = [2, 8, 14, 26]  # He (G1), O (G3, trunk-only), Si + Fe (G4, E=2)


@pytest.fixture(scope="module")
def joint():
    return JointOperatorModel(device="cpu", elements=ELEMENTS)


@pytest.fixture
def edges():
    return torch.logspace(torch.log10(torch.tensor(0.5)),
                          torch.log10(torch.tensor(10.0)), 201)


def _max_rel(a, b):
    denom = b.abs().clamp(min=b.abs().max() * 1e-6)
    return (a - b).abs().div(denom).max().item()


def test_groups_actually_batch(joint):
    # Si+Fe must land in one group (E=2) or the vmap path isn't being exercised
    sizes = sorted(len(g.zs) for g in joint.batched.groups)
    assert 2 in sizes


def test_batched_matches_serial_no_absorption(joint, edges):
    T = torch.tensor([0.7, 2.0, 8.0])
    ab = {8: 0.6, 14: 1.4, 26: 0.9}          # non-unity abundances
    ref = joint.flux(T, ab, 150.0, edges)
    bat = joint.batched.flux(T, ab, 150.0, edges)
    assert bat.shape == ref.shape
    assert _max_rel(bat, ref) < 1e-4


def test_batched_matches_serial_with_absorption(joint, edges):
    from spexai.inference.absorption import Absorption
    absn = Absorption.default()
    T = torch.tensor([1.5, 6.0])
    ab = {8: 0.7, 26: 1.1}
    ref = joint.flux(T, ab, 200.0, edges, absorption=absn, n_h=3e21,
                     redshift=0.02)
    bat = joint.batched.flux(T, ab, 200.0, edges, absorption=absn, n_h=3e21,
                             redshift=0.02)
    assert _max_rel(bat, ref) < 1e-4


def test_batched_respects_zero_abundance(joint, edges):
    T = torch.tensor([3.0])
    ref = joint.flux(T, {26: 0.0}, 150.0, edges)
    bat = joint.batched.flux(T, {26: 0.0}, 150.0, edges)
    assert _max_rel(bat, ref) < 1e-4


def test_batched_echunk_invariant(joint, edges):
    # chunking the energy axis must not change the result
    T = torch.tensor([2.0, 5.0])
    full = joint.batched.flux(T, {}, 150.0, edges)
    chunked = joint.batched.flux(T, {}, 150.0, edges, echunk=4096)
    assert _max_rel(chunked, full) < 1e-5


def test_batched_stage_split_matches_flux(joint, edges):
    # the staged path (_density -> _continuum -> _combine) must equal flux()
    b = joint.batched
    T = torch.tensor([1.5, 4.0])
    ab = {8: 0.9, 26: 1.1}
    ref = b.flux(T, ab, 150.0, edges)
    e = torch.as_tensor(edges, dtype=torch.float32)
    dens, zs = b._density(T.float(), b._echunk(T.numel(), 2.0), False)
    cont = b._continuum(dens, e, 150.0, False, None, 0.0, 0.0, 2.0)
    staged = b._combine(cont, zs, ab, T.float(), e, 150.0, False, None, 0.0, 0.0)
    assert _max_rel(staged, ref) < 1e-6
    # each stage must be no_grad on its own: called directly (as the --stages
    # benchmark does) an autograd graph would pin every trunk activation and
    # OOM the GPU -- flux()'s own no_grad does not cover direct stage calls.
    assert not dens.requires_grad and not cont.requires_grad
    assert not staged.requires_grad


def test_batched_tiny_mem_budget_invariant(joint, edges):
    # a tiny mem_gb forces minimal trunk/broadening chunks (the OOM-mitigation
    # path); result must be unchanged vs the serial reference
    T = torch.tensor([1.0, 3.0, 7.0])
    ab = {8: 0.8, 26: 1.2}
    ref = joint.flux(T, ab, 150.0, edges)
    bat = joint.batched.flux(T, ab, 150.0, edges, mem_gb=1e-4)
    assert _max_rel(bat, ref) < 1e-4
