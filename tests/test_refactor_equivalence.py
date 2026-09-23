"""The new SpectralFit path must reproduce the old fitting.py path, exactly.

These assertions are checked against ``tests/data/refactor_golden.npz``, frozen
from the pre-refactor code by ``scripts/inference/make_refactor_golden.py``.

The golden file, rather than an old-vs-new comparison in the test itself, is
the point. A live A/B dies the moment the old path is deleted -- which is
precisely when the guarantee stops being checkable, and precisely what this
refactor does in its last phase. Frozen values outlive the deletion.

Two cases. ``single`` is the plain path. ``dem_abs`` carries a Gaussian DEM,
Galactic absorption and a per-walker ``n_h``: the DEM wiring is what
``tier_c_mcmc.py`` had wrong (a ``VectorForward`` built with no ``dem=``), and
per-walker ``n_h`` is the axis that carried a real broadcast bug before. A
single-temperature golden case on its own would pass while proving nothing
about either.
"""
import os

import numpy as np
import pytest

from spexai.inference.abundances import AbundanceModel
from spexai.inference.absorption import Absorption
from spexai.inference.operator_model import JointOperatorModel, MODELS_DIR
from spexai.inference.priors import PriorSet
from spexai.inference.response import Response
from spexai.inference.spectral_fit import SpectralFit
from spexai.inference import tempdist as td

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data", "refactor_golden.npz")
RMF = os.path.expanduser("~/work/data/spexai/responses/aciss_aimpt_cy28.rmf")
ELEMENTS = [2, 8, 26]
FIXED = {"abundances": {}, "logz": -10.0}

pytestmark = [
    pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, "Z26_Fe.pt")),
                       reason="model store not present"),
    pytest.mark.skipif(not os.path.exists(RMF),
                       reason="ACIS response not present"),
    pytest.mark.skipif(not os.path.exists(GOLDEN),
                       reason="golden file not generated"),
]


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def model():
    return JointOperatorModel(device="cpu", elements=ELEMENTS)


def _fit(case, golden, model):
    """Rebuild one golden case through the NEW path.

    The redshift/distance/exposure must match what ``build_posterior`` used, or
    this compares two different physical models and the tolerances below are
    meaningless. ``logz=-10`` in the old ``fixed`` dict means z = 10**-10, i.e.
    "not redshifted" -- not z = -10.
    """
    counts = golden[f"{case}__counts"]
    names = [str(n) for n in golden[f"{case}__names"]]
    priors = PriorSet.box(golden[f"{case}__lo"], golden[f"{case}__hi"], names)

    kw = {}
    if case == "dem_abs":
        kw = dict(dem=td.gaussian_T(td.TempGrid(1.0, 8.0, n=10)),
                  absorption=Absorption.default(),
                  abundances=AbundanceModel(model.elements),
                  # the golden came from build_posterior, which sampled n_h in
                  # absolute cm^-2; the campaign's 1e21 convention would be
                  # wrong here by that factor
                  n_h_scale=1.0)

    return SpectralFit(
        emulator=model, response=Response(RMF, None), counts=counts,
        exposure=float(golden["meta__exposure"]), priors=priors,
        redshift=10.0 ** FIXED["logz"], fixed=FIXED, device="cpu", **kw)


@pytest.mark.parametrize("case", ["single", "dem_abs"])
def test_loglike_matches_the_frozen_reference(case, golden, model):
    """The likelihood itself is unchanged -- the tight check.

    This is the assertion that protects the physics: 64 points spread across
    the prior box, so a wiring change (a dropped DEM, a mis-set redshift, an
    abundance that stopped being applied, the wrong n_h scale) cannot hide.

    It compares ``loglike``, not ``logp``, and the golden ``logp`` is the right
    reference for it: the old ``BoxPrior.logpdf`` returned 0 inside the box, so
    the frozen ``logp`` *is* the bare log-likelihood. The prior term that is no
    longer zero is pinned separately below.
    """
    got = _fit(case, golden, model).posterior.loglike(golden[f"{case}__theta"])
    ref = golden[f"{case}__logp"]
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-12, atol=0.0), (
        f"{case}: worst relative deviation "
        f"{np.max(np.abs(got - ref) / np.abs(ref)):.3e}")


@pytest.mark.parametrize("case", ["single", "dem_abs"])
def test_prior_is_now_normalised(case, golden, model):
    """``logp`` gains a constant: the uniform prior's normalisation.

    ``BoxPrior`` only bounds-checked, contributing 0 inside the box -- an
    unnormalised prior. ``PriorSet`` returns a real density, so ``logp`` is
    now offset by exactly ``-sum(log(hi - lo))``. Being constant it cancels in
    every acceptance ratio and moves no posterior sample, and UltraNest is
    untouched because it scores through ``loglike``. This test states the size
    of the change so it stays a deliberate one rather than a surprise in
    someone's absolute log-evidence.
    """
    lo, hi = golden[f"{case}__lo"], golden[f"{case}__hi"]
    fit = _fit(case, golden, model)
    theta = golden[f"{case}__theta"]
    offset = fit.posterior.logp(theta) - golden[f"{case}__logp"]
    expected = -np.sum(np.log(hi - lo))
    # The offset is a difference of two numbers of order 1e8, so its error
    # floor is the float32 forward's ABSOLUTE noise (~4e-8), not a relative
    # tolerance on the offset's own magnitude. atol=1e-5 still pins the
    # constant to five decimals against a value of order 10.
    assert np.allclose(offset, expected, rtol=0.0, atol=1e-5), (
        f"{case}: offset {offset.mean():.6f} +- {offset.std():.2e}, "
        f"expected {expected:.6f}")


def test_n_h_scale_must_be_stated_when_n_h_is_fitted(golden, model):
    """No silent default for the one parameter whose units the two stacks
    disagreed on.

    ``fitting`` sampled ``n_h`` in absolute cm^-2 (n_h_scale=1.0); the campaign
    sampled it in units of 1e21 (VectorForward's default). Either default would
    silently mis-scale the other caller's absorption by 10^21 -- which produces
    no error, just a wrong spectrum. So it has to be said out loud.
    """
    case = "dem_abs"
    names = [str(n) for n in golden[f"{case}__names"]]
    assert "n_h" in names
    with pytest.raises(ValueError, match="n_h_scale"):
        SpectralFit(emulator=model, response=Response(RMF, None),
                    counts=golden[f"{case}__counts"],
                    exposure=float(golden["meta__exposure"]),
                    priors=PriorSet.box(golden[f"{case}__lo"],
                                        golden[f"{case}__hi"], names),
                    dem=td.gaussian_T(td.TempGrid(1.0, 8.0, n=10)),
                    absorption=Absorption.default(),
                    abundances=AbundanceModel(model.elements),
                    redshift=10.0 ** FIXED["logz"], fixed=FIXED, device="cpu")


@pytest.mark.parametrize("case", ["single", "dem_abs"])
def test_walker_init_matches(case, golden, model):
    """Walker initialisation is bit-identical.

    ``fitting.run_emcee`` clipped the cloud to ``lo + 1e-6`` and
    ``samplers.run_emcee`` to ``lo + 1e-9``. Neither bound is reached for these
    cases, so the two agree -- this test is what would notice if that stopped
    being true and the chains diverged from the first step for a reason that
    had nothing to do with the likelihood.
    """
    from spexai.inference.samplers import _init_walkers
    fit = _fit(case, golden, model)
    rng = np.random.default_rng(int(golden["meta__emcee_seed"]))
    got = _init_walkers(fit.posterior.prior, int(golden["meta__nwalkers"]),
                        rng, center=golden[f"{case}__truth"])
    assert np.array_equal(got, golden[f"{case}__p0"])


@pytest.mark.parametrize("case", ["single", "dem_abs"])
def test_emcee_chain_matches_the_frozen_reference(case, golden, model):
    """End-to-end: same seed, same chain, through the whole new stack.

    Reproducibility here is itself a 2026-09-22 fix. emcee snapshots NumPy's
    global legacy RNG when the sampler is constructed, so the old ``seed``
    argument covered only the walker cloud and two identical runs diverged.
    Both ``run_emcee``s now pin that state; if either stops, this test fails.
    """
    fit = _fit(case, golden, model)
    res = fit.sample("emcee", nwalkers=int(golden["meta__nwalkers"]),
                     nsteps=int(golden["meta__nsteps"]),
                     seed=int(golden["meta__emcee_seed"]),
                     center=golden[f"{case}__truth"])
    assert res.chain.shape == golden[f"{case}__chain"].shape
    assert np.allclose(res.chain, golden[f"{case}__chain"], rtol=1e-10)


def test_legacy_parameter_names_still_work(golden, model):
    """``temp``/``velocity`` are accepted, warn, and mean ``kT``/``sigma_v``.

    The golden file was generated under the old spelling, so the cases above
    already exercise the alias. This pins the warning, which is the part a user
    relies on to know their script needs updating.
    """
    case = "single"
    names = [str(n) for n in golden[f"{case}__names"]]
    assert "temp" in names and "velocity" in names, names
    with pytest.warns(DeprecationWarning, match="temp"):
        fit = _fit(case, golden, model)
    assert fit.temp_name == "temp"
    assert fit.velocity_name == "velocity"


def test_unknown_sampler_names_the_alternatives(golden, model):
    fit = _fit("single", golden, model)
    with pytest.raises(KeyError, match="nautilus"):
        fit.sample("definitely-not-a-sampler")
