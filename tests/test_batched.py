"""Element-batched forward must match the serial JointOperatorModel.flux.

The whole point of BatchedJointForward is that it changes *how* the per-element
trunks are evaluated (grouped vmap) without changing the numbers, so these tests
assert equivalence to the reference serial path across the abundance and
absorption code paths. Element set is chosen to exercise a real grouped pass
(Fe+Si share the big 384/5/512 architecture -> E=2 in one vmap group) alongside
a trunk-only element (O has no line head) and a small singleton group (He)."""
import os

import numpy as np
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


def test_recompile_limit_raised_not_lowered(joint):
    # dynamo's cache is per code object, so all trunk groups (and, in the serial
    # path, all 30 elements compiling the same forward_norm) share one budget;
    # past it dynamo silently falls back to eager and the speedup vanishes
    import torch._dynamo as dynamo
    from spexai.inference.operator_model import ensure_recompile_limit
    ensure_recompile_limit(64)
    assert dynamo.config.recompile_limit >= 64
    ensure_recompile_limit(4)                  # must not lower an existing value
    assert dynamo.config.recompile_limit >= 64


def test_per_walker_velocity_matches_walker_loop(joint, edges):
    # THE test for per-walker sigma_v: a (B,) velocity must reproduce, walker by
    # walker, what a scalar velocity gives for that walker alone. Distinct
    # velocities per walker, so a silently-shared sigma_v cannot pass.
    vels = torch.tensor([80.0, 400.0, 1200.0])
    T = torch.tensor([1.2, 3.0, 6.0])
    ab = {8: 0.7, 14: 1.3, 26: 1.1}
    for name, fn in (("serial", joint.flux), ("batched", joint.batched.flux)):
        got = fn(T, ab, vels, edges)                       # (3, M), one v each
        for i, v in enumerate(vels):
            ref = fn(T[i:i + 1], ab, float(v), edges)      # scalar, that walker
            assert _max_rel(got[i:i + 1], ref) < 1e-5, f"{name} walker {i}"


def test_per_walker_velocity_constant_equals_scalar(joint, edges):
    # a constant (B,) vector must take the per-walker branch yet land on the
    # scalar answer -- guards the flat-buffer scatter's index arithmetic
    T = torch.tensor([1.5, 4.0])
    ab = {26: 1.0}
    ref = joint.batched.flux(T, ab, 300.0, edges)
    vec = joint.batched.flux(T, ab, torch.full((2,), 300.0), edges)
    assert _max_rel(vec, ref) < 1e-5


def test_per_walker_velocity_rejects_wrong_length(joint, edges):
    from spexai.train.broadening import deposit_gaussian_lines
    with pytest.raises(ValueError, match="velocity must be scalar"):
        deposit_gaussian_lines(torch.tensor([1.0, 2.0]), torch.ones(3, 2),
                               torch.as_tensor(edges), torch.tensor([1.0, 2.0]))


def test_per_walker_abundance_matches_walker_loop(joint, edges):
    # THE test for per-walker abundances: a {Z: (B,)} dict must reproduce,
    # walker by walker, what scalar abundances give for that walker alone.
    # Distinct values per walker and per element, so a weight broadcast against
    # the wrong axis (or shared across the batch) cannot pass.
    T = torch.tensor([1.2, 3.0, 6.0])
    per = {8: torch.tensor([0.4, 0.9, 1.6]), 26: torch.tensor([1.3, 0.7, 1.0])}
    for name, fn in (("serial", joint.flux), ("batched", joint.batched.flux)):
        got = fn(T, per, 150.0, edges)                     # (3, M)
        for i in range(3):
            scalar = {z: float(v[i]) for z, v in per.items()}
            ref = fn(T[i:i + 1], scalar, 150.0, edges)
            assert _max_rel(got[i:i + 1], ref) < 1e-5, f"{name} walker {i}"


def test_per_walker_abundance_constant_equals_scalar(joint, edges):
    # a constant (B,) abundance must take the tensor branch yet land on the
    # scalar answer -- guards abundance_weight's (B, 1) reshape
    T = torch.tensor([1.5, 4.0])
    ref = joint.batched.flux(T, {26: 0.8}, 150.0, edges)
    vec = joint.batched.flux(T, {26: torch.full((2,), 0.8)}, 150.0, edges)
    assert _max_rel(vec, ref) < 1e-5


def test_per_walker_n_h_matches_walker_loop(joint, edges):
    # per-walker N_H: _continuum's rows are element-major (row = n*B + b), so a
    # (B, K) transmission has to be gathered by each row's walker index. A
    # scalar n_h broadcasts either way, so only distinct per-walker values
    # catch a mis-aligned screen.
    from spexai.inference.absorption import Absorption
    absn = Absorption.default()
    T = torch.tensor([2.0, 5.0, 3.0])
    nh = torch.tensor([1e20, 4e21, 2e22])
    ab = {8: 0.9, 26: 1.1}
    got = joint.batched.flux(T, ab, 150.0, edges, absorption=absn, n_h=nh,
                             redshift=0.02)
    for i in range(3):
        ref = joint.flux(T[i:i + 1], ab, 150.0, edges, absorption=absn,
                         n_h=float(nh[i]), redshift=0.02)
        assert _max_rel(got[i:i + 1], ref) < 1e-4, f"walker {i}"


def test_flux_drops_the_graph_by_default(joint, edges):
    # the default must stay no_grad even when the inputs require grad: an
    # autograd graph over the trunk pins every activation (the 21 GiB OOM)
    T = torch.tensor([3.0], requires_grad=True)
    v = torch.tensor([150.0], requires_grad=True)
    out = joint.batched.flux(T, {26: 1.0}, v, edges)
    assert not out.requires_grad


def test_grad_enabled_matches_finite_differences(joint, edges):
    # NUTS and reparameterised VI need d(flux)/d(theta). Check the graph
    # survives grad_enabled() AND that the gradients are right, against a
    # central difference -- a detached intermediate (e.g. the .item() that used
    # to sit in the line deposit) yields a clean zero, which "is not None"
    # would happily pass.
    b = joint.batched
    ab = {8: 0.9, 26: 1.1}
    w = torch.rand(edges.numel() - 1, generator=torch.Generator().manual_seed(0))

    def scalarised(t, v, requires_grad=False):
        T = torch.tensor([t], requires_grad=requires_grad)
        V = torch.tensor([v], requires_grad=requires_grad)
        with b.grad_enabled():
            out = b.flux(T, ab, V, edges)
        return (out.squeeze(0) * w).sum(), T, V

    loss, T, V = scalarised(3.0, 150.0, requires_grad=True)
    assert loss.requires_grad
    loss.backward()

    for name, p, base, h, tol in (("temp", T, 3.0, 1e-3, 0.05),
                                  ("velocity", V, 150.0, 1.0, 0.15)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        args = {"temp": lambda d: (base + d, 150.0),
                "velocity": lambda d: (3.0, base + d)}[name]
        # detach: the operator's own parameters require grad, so under
        # grad_enabled the perturbed losses carry a graph we have no use for
        fd = float((scalarised(*args(h))[0] - scalarised(*args(-h))[0]).detach()
                   / (2 * h))
        ad = float(p.grad.reshape(-1)[0])
        # float32 forward + truncation: agreement to a few percent is the most
        # a central difference can show here
        assert abs(ad - fd) <= tol * max(abs(fd), 1e-30), (
            f"{name}: autodiff {ad:.4e} vs finite-diff {fd:.4e}")


def test_gradient_checkpointing_does_not_change_gradients(joint, edges):
    # checkpointing recomputes activations in backward; the gradients must be
    # identical, only the memory differs. Two echunk sizes => different numbers
    # of checkpointed segments, which must not matter.
    b = joint.batched
    ab = {26: 1.1}
    w = torch.rand(edges.numel() - 1, generator=torch.Generator().manual_seed(1))

    def grad_at(echunk):
        T = torch.tensor([3.0], requires_grad=True)
        V = torch.tensor([180.0], requires_grad=True)
        with b.grad_enabled():
            out = b.flux(T, ab, V, edges, echunk=echunk)
        (out.squeeze(0) * w).sum().backward()
        return float(T.grad[0]), float(V.grad[0])

    coarse = grad_at(None)
    fine = grad_at(2048)
    assert np.isclose(coarse[0], fine[0], rtol=1e-4), f"{coarse} vs {fine}"
    assert np.isclose(coarse[1], fine[1], rtol=1e-3), f"{coarse} vs {fine}"


def test_checkpointing_only_engages_under_grad(joint, edges):
    # the no-grad path must not pay the recompute cost
    b = joint.batched
    assert not b.track_grad
    out = b.flux(torch.tensor([3.0]), {26: 1.0}, 150.0, edges)
    assert not out.requires_grad


def test_grad_enabled_restores_previous_setting(joint, edges):
    b = joint.batched
    assert not b.track_grad
    with b.grad_enabled():
        assert b.track_grad
        with b.grad_enabled(False):
            assert not b.track_grad
        assert b.track_grad
    assert not b.track_grad


def test_batched_tiny_mem_budget_invariant(joint, edges):
    # a tiny mem_gb forces minimal trunk/broadening chunks (the OOM-mitigation
    # path); result must be unchanged vs the serial reference
    T = torch.tensor([1.0, 3.0, 7.0])
    ab = {8: 0.8, 26: 1.2}
    ref = joint.flux(T, ab, 150.0, edges)
    bat = joint.batched.flux(T, ab, 150.0, edges, mem_gb=1e-4)
    assert _max_rel(bat, ref) < 1e-4
