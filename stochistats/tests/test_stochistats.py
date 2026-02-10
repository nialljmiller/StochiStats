"""
Tests for stochistats package.

Verifies that:
1. All public functions are importable from the top-level
2. Backwards-compatible ``import StochiStats`` works
3. Each statistic returns a finite scalar on synthetic data
4. calculate_all_features returns the expected keys
"""

import numpy as np
import pytest

# ---- Synthetic lightcurve fixture ----------------------------------------

@pytest.fixture
def lightcurve():
    """Generate a noisy sinusoidal lightcurve."""
    rng = np.random.default_rng(42)
    n = 200
    time = np.sort(rng.uniform(0, 365, n))
    period = 10.0
    mag = 14.0 + 0.3 * np.sin(2 * np.pi * time / period) + rng.normal(0, 0.02, n)
    magerr = np.full(n, 0.02)
    return mag, magerr, time, period


# ---- Top-level imports ---------------------------------------------------

class TestImports:
    def test_import_stochistats(self):
        import stochistats  # noqa: F401

    def test_import_backwards_compat(self):
        import StochiStats  # noqa: F401

    def test_from_import(self):
        from stochistats import (
            cody_M, Stetson_K, Eta, Eta_e, medianBRP,
            RangeCumSum, MaxSlope, MedianAbsDev, Meanvariance,
            PercentAmplitude, RoMS, ptop_var, lagauto, AndersonDarling,
            stdnxs, weighted_mean, weighted_variance, weighted_skew,
            weighted_kurtosis, mu, sigma, skewness, kurtosis,
            calculate_all_features,
        )

    def test_backwards_compat_from_import(self):
        from StochiStats import Stetson_K, Eta, weighted_mean  # noqa: F401


# ---- Variability indices -------------------------------------------------

class TestVariability:
    def test_cody_M(self, lightcurve):
        from stochistats import cody_M
        mag, _, time, _ = lightcurve
        assert np.isfinite(cody_M(mag, time))

    def test_Stetson_K(self, lightcurve):
        from stochistats import Stetson_K
        mag, magerr, _, _ = lightcurve
        k = Stetson_K(mag, magerr)
        assert np.isfinite(k)
        assert k > 0

    def test_Eta(self, lightcurve):
        from stochistats import Eta
        mag, _, time, _ = lightcurve
        assert np.isfinite(Eta(mag, time))

    def test_Eta_e(self, lightcurve):
        from stochistats import Eta_e
        mag, _, time, _ = lightcurve
        assert np.isfinite(Eta_e(mag, time))

    def test_medianBRP(self, lightcurve):
        from stochistats import medianBRP
        mag, magerr, _, _ = lightcurve
        v = medianBRP(mag, magerr)
        assert 0 <= v <= 1

    def test_RangeCumSum(self, lightcurve):
        from stochistats import RangeCumSum
        mag, _, _, _ = lightcurve
        assert np.isfinite(RangeCumSum(mag))

    def test_MaxSlope(self, lightcurve):
        from stochistats import MaxSlope
        mag, _, time, _ = lightcurve
        assert np.isfinite(MaxSlope(mag, time))
        assert MaxSlope(mag, time) >= 0

    def test_MedianAbsDev(self, lightcurve):
        from stochistats import MedianAbsDev
        mag, magerr, _, _ = lightcurve
        assert np.isfinite(MedianAbsDev(mag, magerr))

    def test_Meanvariance(self, lightcurve):
        from stochistats import Meanvariance
        mag, _, _, _ = lightcurve
        assert np.isfinite(Meanvariance(mag))

    def test_PercentAmplitude(self, lightcurve):
        from stochistats import PercentAmplitude
        mag, _, _, _ = lightcurve
        assert np.isfinite(PercentAmplitude(mag))

    def test_RoMS(self, lightcurve):
        from stochistats import RoMS
        mag, magerr, _, _ = lightcurve
        assert np.isfinite(RoMS(mag, magerr))

    def test_stdnxs(self, lightcurve):
        from stochistats import stdnxs
        mag, magerr, _, _ = lightcurve
        assert np.isfinite(stdnxs(mag, magerr))

    def test_ptop_var(self, lightcurve):
        from stochistats import ptop_var
        mag, magerr, _, _ = lightcurve
        assert np.isfinite(ptop_var(mag, magerr))

    def test_lagauto(self, lightcurve):
        from stochistats import lagauto
        mag, _, _, _ = lightcurve
        v = lagauto(mag)
        assert np.isfinite(v)
        assert -1 <= v <= 1

    def test_AndersonDarling(self, lightcurve):
        from stochistats import AndersonDarling
        mag, _, _, _ = lightcurve
        v = AndersonDarling(mag)
        assert np.isfinite(v)


# ---- Moments -------------------------------------------------------------

class TestMoments:
    def test_unweighted(self, lightcurve):
        from stochistats import mu, sigma, skewness, kurtosis
        mag, _, _, _ = lightcurve
        assert np.isfinite(mu(mag))
        assert np.isfinite(sigma(mag))
        assert np.isfinite(skewness(mag))
        assert np.isfinite(kurtosis(mag))

    def test_weighted(self, lightcurve):
        from stochistats import (
            weighted_mean, weighted_variance, weighted_skew, weighted_kurtosis,
        )
        mag, magerr, _, _ = lightcurve
        assert np.isfinite(weighted_mean(mag, magerr))
        assert np.isfinite(weighted_variance(mag, magerr))
        assert np.isfinite(weighted_skew(mag, magerr))
        assert np.isfinite(weighted_kurtosis(mag, magerr))


# ---- Fitting -------------------------------------------------------------

class TestFitting:
    def test_sine_fit(self, lightcurve):
        from stochistats import sine_fit
        mag, _, time, period = lightcurve
        y_fit, params, r2 = sine_fit(mag, time, period)
        assert len(y_fit) == len(mag)
        assert np.isfinite(r2)
        assert r2 > 0.5  # should fit well for a sinusoidal signal

    def test_bin_lc(self, lightcurve):
        from stochistats import bin_lc
        _, _, time, _ = lightcurve
        mag = lightcurve[0]
        rx, rq50 = bin_lc(time, mag, res=5)
        assert len(rx) == 5
        assert len(rq50) == 5


# ---- High-level features -------------------------------------------------

class TestFeatures:
    def test_calculate_all_features(self, lightcurve):
        from stochistats import calculate_all_features
        mag, magerr, time, _ = lightcurve
        feats = calculate_all_features(mag, magerr, time)

        expected_keys = {
            "mag_n", "mag_avg", "magerr_avg", "time_range", "true_amplitude",
            "Cody_M", "stet_k", "eta", "eta_e", "med_BRP",
            "range_cum_sum", "max_slope", "MAD", "mean_var",
            "percent_amp", "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
            "weight_mean", "weight_std", "weight_skew", "weight_kurt",
            "mean", "std", "skew", "kurt",
        }
        assert expected_keys == set(feats.keys())

        # All values should be finite for a well-behaved lightcurve
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


# ---- Utilities -----------------------------------------------------------

class TestUtils:
    def test_phaser(self):
        from stochistats import phaser
        import stochistats.utils as u
        phase = u.phaser(np.array([0, 5, 10, 15]), 10.0)
        np.testing.assert_allclose(phase, [0, 0.5, 0, 0.5])

    def test_normalise(self):
        from stochistats import normalise
        out = normalise([1, 2, 3, 4, 5])
        assert out[0] == 0.0
        assert out[-1] == 1.0

    def test_dtw(self):
        from stochistats import dtw
        s = [1, 2, 3]
        t = [1, 2, 3]
        matrix = dtw(s, t)
        assert matrix[-1, -1] == 0.0  # identical sequences
