"""Unit tests for the SpectralOperator architecture, focused on the trunk
Fourier temperature embedding (film_t_freqs).

Self-contained: models are built from a synthetic OperatorConfig + energy
grid, so these run without the model store or a data cache (unlike the
inference tests, which skip when those artefacts are absent).
"""
import torch

from spexai.operator import OperatorConfig, SpectralOperator

ENERGY = torch.linspace(0.1, 12.0, 200)          # synthetic monotonic keV grid
TEMPS = torch.tensor([0.7, 3.0, 9.0])            # cold / mid / hot


def build(film_t_freqs=0, use_film=True, seed=0, **cfg):
    torch.manual_seed(seed)
    c = OperatorConfig(hidden_size=64, n_hidden=3, n_freqs=32,
                       use_linehead=False, use_binnorm=False,
                       use_film=use_film, film_t_freqs=film_t_freqs, **cfg)
    return SpectralOperator(c, energy_grid=ENERGY)


# --- off by default: backward compatibility --------------------------------

def test_off_by_default():
    m = build()
    assert m.config.film_t_freqs == 0
    assert m.film_t_embed is None


def test_off_forward_finite():
    m = build()
    out = m(TEMPS, ENERGY)
    assert out.shape == (len(TEMPS), len(ENERGY))
    assert torch.isfinite(out).all()


# --- on: builds, adds params, stays finite ---------------------------------

def test_on_builds_embedding():
    m = build(film_t_freqs=8)
    assert m.film_t_embed is not None


def test_on_adds_exactly_the_embedding_params():
    off, on = build(film_t_freqs=0), build(film_t_freqs=8)
    # only the first CondNet layer widens: n_params(1+2F) - n_params inputs,
    # times cond_hidden, plus nothing else -- so the delta is positive and
    # equals the extra input weights of the FiLM + trend conditioning nets.
    extra = on.count_parameters() - off.count_parameters()
    assert extra > 0
    # FiLM generator and trend head each gain (2F * n_params * cond_hidden)
    f, npar, ch = 8, 1, on.config.cond_hidden
    assert extra == 2 * (2 * f * npar) * ch


def test_on_forward_finite():
    m = build(film_t_freqs=16)
    out = m(TEMPS, ENERGY)
    assert out.shape == (len(TEMPS), len(ENERGY))
    assert torch.isfinite(out).all()


def test_embedding_changes_output():
    """With matched seeds the embedded model must differ from the plain one
    (otherwise the embedding is not wired into the forward pass)."""
    a = build(film_t_freqs=8, seed=1)(TEMPS, ENERGY)
    b = build(film_t_freqs=0, seed=1)(TEMPS, ENERGY)
    assert not torch.allclose(a, b)


# --- both conditioning paths honour the embedding --------------------------

def test_no_film_path_honours_embedding():
    m = build(film_t_freqs=8, use_film=False)
    assert m.film_t_embed is not None
    out = m(TEMPS, ENERGY)
    assert torch.isfinite(out).all()


# --- exactness: determinism and save/load round-trip -----------------------

def test_deterministic_forward():
    m = build(film_t_freqs=16)
    m.eval()
    with torch.no_grad():
        a, b = m(TEMPS, ENERGY), m(TEMPS, ENERGY)
    assert torch.equal(a, b)


def test_state_dict_roundtrip():
    m1 = build(film_t_freqs=16, seed=3)
    m2 = build(film_t_freqs=16, seed=99)          # different init
    with torch.no_grad():
        before = m2(TEMPS, ENERGY)
    m2.load_state_dict(m1.state_dict())            # strict load
    with torch.no_grad():
        after, ref = m2(TEMPS, ENERGY), m1(TEMPS, ENERGY)
    assert not torch.allclose(before, ref)         # sanity: seeds differed
    assert torch.equal(after, ref)                 # exact after load


def test_forward_on_grid_with_embedding():
    """Inference uses forward_on_grid (continuum rebin + line deposit); the
    embedding must not break it."""
    m = build(film_t_freqs=8)
    edges = torch.linspace(0.5, 10.0, 60)
    out = m.forward_on_grid(TEMPS, edges)
    assert out.shape == (len(TEMPS), len(edges) - 1)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()                        # integrated flux is linear >= 0


# --- the cond-input helper matches the Fourier feature width ---------------

def test_cond_input_width():
    m = build(film_t_freqs=8)
    tnorm = m.norm_temp(TEMPS).view(-1, 1)         # (B, n_params=1)
    tc = m._cond_input(tnorm)
    assert tc.shape == (len(TEMPS), 1 * (1 + 2 * 8))
    # off -> identity
    m0 = build(film_t_freqs=0)
    assert torch.equal(m0._cond_input(tnorm), tnorm)
