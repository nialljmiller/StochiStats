"""
Tests for stochistats.cleaning module.

Covers every individual stage and the full pipeline orchestrator.
"""

import numpy as np
import pytest

from stochistats.cleaning import (
    CleaningConfig,
    clean_lightcurve,
    clip_magerr,
    detrend_linear,
    quality_cut,
    remove_ambiguous_points,
    sigma_clip,
    temporal_bin,
    validate_pawprint_pairs,
    variability_check,
)


# ---- fixtures -------------------------------------------------------------

@pytest.fixture
def simple_lc():
    """100-point clean sinusoidal lightcurve."""
    rng = np.random.default_rng(42)
    n = 100
    time = np.sort(rng.uniform(55000, 58500, n))
    mag = 14.0 + 0.3 * np.sin(2 * np.pi * time / 10.0) + rng.normal(0, 0.02, n)
    magerr = np.full(n, 0.02)
    return mag, magerr, time


@pytest.fixture
def lc_with_metadata(simple_lc):
    """Lightcurve with chi, ast_res_chisq, and ambiguous arrays."""
    mag, magerr, time = simple_lc
    rng = np.random.default_rng(99)
    n = len(mag)
    chi = rng.exponential(2.0, n)            # mostly < 10
    chi[0] = 50.0                            # one bad point
    ast = rng.exponential(3.0, n)            # mostly < 20
    ast[1] = 100.0                           # one bad point
    ambiguous = np.zeros(n, dtype=bool)
    ambiguous[2] = True                      # one blended point
    return mag, magerr, time, chi, ast, ambiguous


# ---- Stage 1: quality cuts -----------------------------------------------

class TestQualityCut:
    def test_removes_bad_chi(self, lc_with_metadata):
        mag, magerr, time, chi, ast, _ = lc_with_metadata
        m, me, t, mask = quality_cut(mag, magerr, time, chi, None, max_chi=10.0)
        assert len(m) < len(mag)
        assert not mask[0]  # chi[0] = 50

    def test_removes_bad_ast(self, lc_with_metadata):
        mag, magerr, time, chi, ast, _ = lc_with_metadata
        m, me, t, mask = quality_cut(mag, magerr, time, None, ast, max_ast_res_chisq=20.0)
        assert not mask[1]  # ast[1] = 100

    def test_no_arrays_no_op(self, simple_lc):
        mag, magerr, time = simple_lc
        m, me, t, mask = quality_cut(mag, magerr, time, None, None)
        assert len(m) == len(mag)


# ---- Stage 2: ambiguous removal ------------------------------------------

class TestAmbiguous:
    def test_removes_flagged(self, lc_with_metadata):
        mag, magerr, time, _, _, ambiguous = lc_with_metadata
        m, me, t, mask = remove_ambiguous_points(mag, magerr, time, ambiguous)
        assert len(m) == len(mag) - 1
        assert not mask[2]


# ---- Stage 3: hard error cap ---------------------------------------------

class TestClipMagerr:
    def test_removes_high_err(self):
        mag = np.array([14.0, 14.1, 14.2, 14.3])
        magerr = np.array([0.01, 0.15, 0.25, 0.05])
        time = np.array([1.0, 2.0, 3.0, 4.0])
        m, me, t, mask = clip_magerr(mag, magerr, time, max_magerr=0.2)
        assert len(m) == 3
        assert not mask[2]  # magerr = 0.25

    def test_configurable_threshold(self):
        magerr = np.array([0.01, 0.3, 0.5])
        mag = np.ones(3) * 14.0
        time = np.arange(3, dtype=float)
        m, me, t, _ = clip_magerr(mag, magerr, time, max_magerr=0.4)
        assert len(m) == 2


# ---- Stage 4: sigma clipping ---------------------------------------------

class TestSigmaClip:
    def test_mag_outlier(self):
        mag = np.array([14.0] * 50 + [20.0])  # huge outlier
        magerr = np.full(51, 0.02)
        time = np.arange(51, dtype=float)
        m, me, t, mask = sigma_clip(mag, magerr, time, sigma_mag=3.0)
        assert len(m) == 50
        assert not mask[-1]

    def test_magerr_outlier(self):
        magerr = np.array([0.02] * 50 + [1.0])  # huge error
        mag = np.full(51, 14.0)
        time = np.arange(51, dtype=float)
        m, me, t, mask = sigma_clip(mag, magerr, time, sigma_magerr=3.0)
        assert not mask[-1]


# ---- Stage 5: paw-print pair validation -----------------------------------

class TestPawprintPairs:
    def test_bad_pair_rejected(self):
        """Two close points with wildly different magnitudes → both dropped."""
        time = np.array([100.0, 100.01, 200.0, 200.01])
        mag = np.array([14.0, 14.0, 14.0, 16.0])   # pair at t=200 disagrees
        magerr = np.array([0.02, 0.02, 0.02, 0.02])
        m, me, t, mask = validate_pawprint_pairs(
            mag, magerr, time, max_dt=0.02, merr_tol=2.0,
        )
        # points at t=200, 200.01 should be gone
        assert len(m) == 2
        assert mask[0] and mask[1]
        assert not mask[2] and not mask[3]

    def test_good_pair_kept(self):
        time = np.array([100.0, 100.01])
        mag = np.array([14.0, 14.01])
        magerr = np.array([0.02, 0.02])
        m, me, t, mask = validate_pawprint_pairs(
            mag, magerr, time, max_dt=0.02, merr_tol=2.0,
        )
        assert len(m) == 2


# ---- Stage 6: temporal binning -------------------------------------------

class TestTemporalBin:
    def test_bins_close_points(self):
        # 3 points within 30 min, all with magerr > 0.1
        time = np.array([100.0, 100.01, 100.02, 200.0])
        mag = np.array([14.0, 14.1, 14.2, 15.0])
        magerr = np.array([0.15, 0.15, 0.15, 0.15])
        m, me, t = temporal_bin(mag, magerr, time, window_hours=1.0, min_magerr=0.1)
        # first 3 should become 1 bin, last stays
        assert len(m) == 2

    def test_low_err_not_binned(self):
        time = np.array([100.0, 100.01, 100.02])
        mag = np.array([14.0, 14.1, 14.2])
        magerr = np.array([0.01, 0.01, 0.01])  # below threshold
        m, me, t = temporal_bin(mag, magerr, time, window_hours=1.0, min_magerr=0.1)
        assert len(m) == 3

    def test_empty_input(self):
        m, me, t = temporal_bin(
            np.array([]), np.array([]), np.array([]),
        )
        assert len(m) == 0


# ---- Stage 7: linear detrending ------------------------------------------

class TestDetrendLinear:
    def test_subtracts_trend(self):
        time = np.linspace(55000, 58500, 200)
        # Strong linear trend + sine
        trend_mag = 14.0 + 0.001 * (time - 55000)
        mag = trend_mag + 0.3 * np.sin(2 * np.pi * time / 10.0)
        magerr = np.full(200, 0.02)
        mag_out, trend, r2 = detrend_linear(
            mag, magerr, time, min_slope=2e-4, min_r2=0.1,
        )
        # After detrending, the overall slope should be much smaller
        p_before = np.polyfit(time, mag, 1)
        p_after = np.polyfit(time, mag_out, 1)
        assert abs(p_after[0]) < abs(p_before[0])

    def test_no_subtract_if_flat(self, simple_lc):
        mag, magerr, time = simple_lc
        mag_out, trend, r2 = detrend_linear(
            mag, magerr, time, min_slope=0.01, min_r2=0.5,
        )
        # Should NOT detrend a nearly-flat lightcurve with strict thresholds
        np.testing.assert_array_almost_equal(mag_out, mag)


# ---- Stage 8: variability check ------------------------------------------

class TestVariabilityCheck:
    def test_passes_good_lc(self, simple_lc):
        mag, magerr, _ = simple_lc
        passes, metrics = variability_check(mag, magerr, min_obs=10)
        assert passes
        assert metrics["n_obs"] == 100

    def test_fails_too_few(self, simple_lc):
        mag, magerr, _ = simple_lc
        passes, _ = variability_check(mag[:5], magerr[:5], min_obs=40)
        assert not passes

    def test_fails_low_amplitude(self):
        mag = np.full(100, 14.0)
        magerr = np.full(100, 0.02)
        passes, _ = variability_check(mag, magerr, min_amplitude=0.1)
        assert not passes

    def test_configurable_iqr(self, simple_lc):
        mag, magerr, _ = simple_lc
        # Very strict IQR requirement
        passes, metrics = variability_check(
            mag, magerr, iqr_factor=1000.0,
        )
        assert not passes

    def test_disabled_checks(self, simple_lc):
        mag, magerr, _ = simple_lc
        passes, _ = variability_check(
            mag, magerr,
            min_obs=1,
            max_magerr=None,
            min_amplitude=None,
            iqr_factor=None,
        )
        assert passes


# ---- Full pipeline --------------------------------------------------------

class TestCleanLightcurve:
    def test_runs_with_defaults(self, simple_lc):
        mag, magerr, time = simple_lc
        result = clean_lightcurve(mag, magerr, time)
        assert "mag" in result
        assert "log" in result
        assert result["n_final"] <= result["n_initial"]

    def test_with_metadata(self, lc_with_metadata):
        mag, magerr, time, chi, ast, ambig = lc_with_metadata
        result = clean_lightcurve(
            mag, magerr, time,
            chi=chi, ast_res_chisq=ast, ambiguous=ambig,
        )
        # Should have removed at least the 3 flagged points
        assert result["n_final"] < result["n_initial"]
        assert len(result["log"]) > 0

    def test_override_via_kwargs(self, simple_lc):
        mag, magerr, time = simple_lc
        result = clean_lightcurve(mag, magerr, time, max_magerr=0.01)
        # All points have magerr=0.02, so everything should be clipped
        assert result["n_final"] == 0

    def test_override_on_config(self, simple_lc):
        mag, magerr, time = simple_lc
        cfg = CleaningConfig(max_magerr=0.5)
        result = clean_lightcurve(mag, magerr, time, config=cfg, max_magerr=0.01)
        assert result["n_final"] == 0  # override wins

    def test_detrend_flag_off(self, simple_lc):
        mag, magerr, time = simple_lc
        result = clean_lightcurve(mag, magerr, time, detrend=False)
        assert result["detrend_r2"] == 0.0
