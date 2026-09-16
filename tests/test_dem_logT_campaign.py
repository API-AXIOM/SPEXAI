"""Log-T DEM wiring of the bias campaign (campaign / bias_sweep / mle_reseed).

The campaign moved from a Gaussian in linear T to SPEX's own ``gdem``, a
Gaussian in log10 T. The switch itself is one factory; what can go wrong is the
ripple -- parameter names and their order, the design draw, files written
under the old parametrisation being read as if they were new, and the frozen
hot_floor path (linear-T ``gaussian_dem`` through ``campaign.Forward``)
silently changing underneath results that were already produced with it.

No emulator and no response: ``campaign.Forward`` is driven with a stub whose
"counts" are the DEM weights it was handed, which is exactly the quantity the
parametrisation changes.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "inference"))
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments", "hot_floor"))

import bias_sweep as bs                                      # noqa: E402
import campaign as cp                                        # noqa: E402
from mle_reseed import tierb_mode                            # noqa: E402

LOG_T_LO, LOG_T_HI = np.log10(0.7), np.log10(15.0)


class _StubEmu:
    """Returns the DEM weights it receives as the spectrum, (1, G)."""
    elements = [1, 2, 8] + cp.FREE_Z

    def predict_counts_dem(self, grid, w, abund, logz, norm, sigma_v, resp,
                           exposure, **kw):
        return torch.as_tensor(w, dtype=torch.float64).reshape(1, -1)

    def predict_counts(self, temp, abund, logz, norm, sigma_v, resp,
                       exposure, **kw):
        return torch.full((1, 70), float(temp.reshape(-1)[0]),
                          dtype=torch.float64)


def _forward(mode, dem=None):
    keep = np.ones(70, dtype=bool)
    return cp.Forward(_StubEmu(), None, None, keep, mode, dem=dem)


def _theta(fwd, thermal):
    """Full theta in ``fwd.names`` order with the given thermal values."""
    vals = {n: 1.0 for n in fwd.abnames}
    vals.update(thermal)
    vals.update({"sigma_v": 200.0, "n_h": 1.0, "log_norm": 10.0})
    return np.array([vals[n] for n in fwd.names])


# --- the DEM model ----------------------------------------------------------

def test_logT_dem_fiducial_is_perseus_and_contained():
    dem, p = cp.gaussian_logT_dem()
    assert dem.param_names == ["logT_mean", "logT_sigma"]
    assert p["logT_mean"] == pytest.approx(np.log10(3.9))
    # 1 keV at 3.9 keV, converted at the centre: sigma_T / (T ln 10)
    assert p["logT_sigma"] == pytest.approx(1.0 / (3.9 * np.log(10.0)))
    grid = dem.temp_grid.numpy()
    assert grid.size == 70
    assert grid[0] == pytest.approx(0.7, rel=1e-6)
    assert grid[-1] == pytest.approx(cp.td.EMULATOR_T_HI_KEV, rel=1e-6)
    assert float(dem.weights(p).sum()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("mean,sigma", [(3.9, 0.1),     # keV passed as log10
                                        (0.5, 1.0),     # keV width as dex
                                        (0.5, 0.0)])    # degenerate width
def test_logT_dem_refuses_linear_values(mean, sigma):
    with pytest.raises(ValueError):
        cp.gaussian_logT_dem(mean=mean, sigma=sigma)


# --- campaign.Forward -------------------------------------------------------

def test_forward_takes_thermal_names_from_the_dem():
    dem, p = cp.gaussian_logT_dem()
    fwd = _forward("dem", dem)
    assert fwd.thermal == ["logT_mean", "logT_sigma"]
    got = fwd(_theta(fwd, p))
    np.testing.assert_array_equal(got, dem.weights(p).double().numpy())


def test_forward_linear_T_hot_floor_path_unchanged():
    """The hot_floor scripts are frozen: they build ``campaign.Forward`` with
    the linear-T ``gaussian_dem`` and ``campaign.build_params``. Their names,
    order and weights must be exactly what they were."""
    dem, p = cp.gaussian_dem()
    fwd = _forward("dem", dem)
    assert fwd.thermal == ["T_mean", "T_sigma"]
    assert p == {"T_mean": cp.PERSEUS["dem_mean"],
                 "T_sigma": cp.PERSEUS["dem_sigma"]}
    got = fwd(_theta(fwd, p))
    np.testing.assert_array_equal(got, dem.weights(p).double().numpy())
    names = [q.name for q in cp.build_params(fwd, 10.0)]
    assert names == fwd.names


def test_campaign_build_params_refuses_a_logT_forward():
    """``campaign.build_params`` is the hot_floor parametrisation and knows
    only the linear-T names; handed a log-T forward it must say so rather than
    return a vector in the wrong order."""
    fwd = _forward("dem", cp.gaussian_logT_dem()[0])
    with pytest.raises(ValueError, match="linear-T"):
        cp.build_params(fwd, 10.0)


# --- bias_sweep design ------------------------------------------------------

def _stratified(x, lo, hi):
    """LHS: exactly one sample in each of n equal-width strata."""
    n = len(x)
    idx = np.floor((np.asarray(x) - lo) / (hi - lo) * n).astype(int)
    return sorted(idx.tolist()) == list(range(n))


def test_dem_design_ranges_and_stratification():
    pts = bs.sample_points(64, "dem", seed=1)
    assert not any("T_mean" in p or "T_sigma" in p for p in pts)
    m = np.array([p["logT_mean"] for p in pts])
    s = np.array([p["logT_sigma"] for p in pts])
    assert m.min() >= LOG_T_LO and m.max() <= LOG_T_HI
    assert s.min() >= 0.056 and s.max() <= 0.4
    assert _stratified(m, LOG_T_LO, LOG_T_HI)
    assert _stratified(s, 0.056, 0.4)


def test_single_T_design_is_log_uniform_over_same_range():
    pts = bs.sample_points(64, "single", seed=1)
    kt = np.array([p["kT"] for p in pts])
    assert kt.min() >= 0.7 and kt.max() <= 15.0
    assert _stratified(np.log10(kt), LOG_T_LO, LOG_T_HI)   # uniform in log
    assert not _stratified(kt, 0.7, 15.0)                  # not in linear T


@pytest.mark.parametrize("mode", ["single", "dem"])
def test_design_is_reproducible(mode):
    assert bs.sample_points(16, mode, seed=3) == bs.sample_points(16, mode, 3)


@pytest.mark.parametrize("mode", ["single", "dem"])
def test_build_pars_order_matches_forward(mode):
    dem = cp.gaussian_logT_dem()[0] if mode == "dem" else None
    fwd = _forward(mode, dem)
    pt = bs.sample_points(1, mode, seed=0)[0]
    names = [q.name for q in bs.build_pars(fwd, pt, 10.0, mode)]
    assert names == fwd.names


def test_build_pars_bounds_steps_and_containment_of_truths():
    for mode, keys in (("dem", ("logT_mean", "logT_sigma")), ("single", ("kT",))):
        for pt in bs.sample_points(64, mode, seed=2):
            pars = {q.name: q for q in bs.build_pars(None, pt, 10.0, mode)}
            for k in keys:
                assert pars[k].low < pars[k].truth < pars[k].high, (mode, k)
    pars = {q.name: q for q in bs.build_pars(
        None, bs.sample_points(1, "dem", 0)[0], 10.0, "dem")}
    assert (pars["logT_mean"].low, pars["logT_mean"].high) == pytest.approx(
        (LOG_T_LO - 0.1, LOG_T_HI + 0.1))
    assert (pars["logT_sigma"].low, pars["logT_sigma"].high) == pytest.approx(
        (0.045, 0.5))
    assert pars["logT_mean"].step == pars["logT_sigma"].step == 5e-4
    single = {q.name: q for q in bs.build_pars(
        None, bs.sample_points(1, "single", 0)[0], 10.0, "single")}
    # the fit box stays inside the emulator's trained temperature range
    assert single["kT"].low >= 0.5013 and single["kT"].high <= cp.td.EMULATOR_T_HI_KEV


def test_single_T_step_is_relative_not_absolute():
    """kT is in keV but its step is the DEM's dex step converted at the point.
    An absolute step cannot serve 0.7 and 15 keV at once: at the cold end the
    in-band counts fall away exponentially with kT, and the inherited 5e-3 keV
    biased the Jacobian there by ~8% (h -> h/2 moved the Fisher-weighted column
    by 6.0e-2, against 3.1e-3 at 14 keV)."""
    steps = {}
    for kt in (0.75, 3.9, 14.0):
        pars = {q.name: q for q in bs.build_pars(
            None, dict(bs.sample_points(1, "single", 0)[0], kT=kt),
            10.0, "single")}
        steps[kt] = pars["kT"].step
        assert pars["kT"].step == pytest.approx(kt * np.log(10.0) * bs.STEP_DEX)
    # the same fraction of kT everywhere, and no longer the old absolute value
    fracs = [s / kt for kt, s in steps.items()]
    assert fracs == pytest.approx([fracs[0]] * len(fracs))
    assert all(s != 5e-3 for s in steps.values())
    # and the DEM axes, already in dex, take STEP_DEX directly
    dem = {q.name: q for q in bs.build_pars(
        None, bs.sample_points(1, "dem", 0)[0], 10.0, "dem")}
    assert dem["logT_mean"].step == dem["logT_sigma"].step == bs.STEP_DEX


def test_contained_fraction_recorded_per_point():
    assert bs.contained_fraction({"kT": 4.0}, "single") == 1.0
    fid = cp.gaussian_logT_dem()[1]
    assert bs.contained_fraction(fid, "dem") == pytest.approx(1.0, abs=1e-3)
    hot = {"logT_mean": LOG_T_HI, "logT_sigma": 0.4}
    assert 0.0 < bs.contained_fraction(hot, "dem") < 0.9


# --- legacy guards ----------------------------------------------------------

def test_tierb_mode_reads_names_and_refuses_linear_T_files():
    assert tierb_mode(["Fe", "logT_mean", "logT_sigma", "sigma_v"]) == "dem"
    assert tierb_mode(["Fe", "kT", "sigma_v"]) == "single"
    with pytest.raises(SystemExit, match="linear-T"):
        tierb_mode(["Fe", "T_mean", "T_sigma", "sigma_v"])


@pytest.mark.parametrize("fields,ok", [
    ({"mode": "dem"}, False),                          # predates the stamp
    ({"mode": "dem", "dem_param": "T"}, False),        # linear-T truth
    ({"mode": "dem", "dem_param": "logT"}, True),
    ({"mode": "single"}, True),                        # no DEM, nothing to check
])
def test_truth_npz_dem_parametrisation_guard(tmp_path, fields, ok):
    path = tmp_path / "truth.npz"
    np.savez(path, counts=np.zeros((1, 3)), **fields)
    tz = np.load(path, allow_pickle=True)
    if ok:
        cp.check_truth_dem_param(tz)
    else:
        with pytest.raises(SystemExit, match="log"):
            cp.check_truth_dem_param(tz)
