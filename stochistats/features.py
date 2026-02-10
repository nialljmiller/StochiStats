"""
High-level feature calculation — all PRIMVS statistics in one call.
"""

import numpy as np
from stochistats.variability import (
    cody_M, Stetson_K, Eta, Eta_e, medianBRP,
    RangeCumSum, MaxSlope, MedianAbsDev, Meanvariance,
    PercentAmplitude, RoMS, ptop_var, lagauto, AndersonDarling,
    stdnxs,
)
from stochistats.moments import (
    weighted_mean, weighted_variance, weighted_skew, weighted_kurtosis,
    mu, sigma, skewness, kurtosis,
)


def calculate_all_features(mag, magerr, time):
    """
    Calculate all statistical features used in the PRIMVS catalogue.

    This is the main entry point for the PRIMVS pipeline integration.
    Every statistic is wrapped in a try/except so a single failure
    does not prevent the rest from being computed.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.
    time : array_like
        Observation times (MJD or similar).

    Returns
    -------
    dict
        Dictionary of feature_name → float.  NaN for any failed computation.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    time = np.asarray(time, dtype=float)

    q1, q50, q99 = np.percentile(mag, [1, 50, 99])

    features = {}

    # ---- basic stats ----
    features["mag_n"] = len(mag)
    features["mag_avg"] = q50
    features["magerr_avg"] = np.median(magerr)
    features["time_range"] = np.ptp(time)
    features["true_amplitude"] = abs(q99 - q1)

    # ---- variability indices ----
    _safe = _safe_call  # alias for readability
    features["Cody_M"]        = _safe(cody_M, mag, time)
    features["stet_k"]        = _safe(Stetson_K, mag, magerr)
    features["eta"]           = _safe(Eta, mag, time)
    features["eta_e"]         = _safe(Eta_e, mag, time)
    features["med_BRP"]       = _safe(medianBRP, mag, magerr)
    features["range_cum_sum"] = _safe(RangeCumSum, mag)
    features["max_slope"]     = _safe(MaxSlope, mag, time)
    features["MAD"]           = _safe(MedianAbsDev, mag, magerr)
    features["mean_var"]      = _safe(Meanvariance, mag)
    features["percent_amp"]   = _safe(PercentAmplitude, mag)
    features["roms"]          = _safe(RoMS, mag, magerr)
    features["p_to_p_var"]    = _safe(ptop_var, mag, magerr)
    features["lag_auto"]      = _safe(lagauto, mag)
    features["AD"]            = _safe(AndersonDarling, mag)
    features["std_nxs"]       = _safe(stdnxs, mag, magerr)

    # ---- weighted moments ----
    features["weight_mean"]   = _safe(weighted_mean, mag, magerr)
    features["weight_std"]    = _safe(weighted_variance, mag, magerr)
    features["weight_skew"]   = _safe(weighted_skew, mag, magerr)
    features["weight_kurt"]   = _safe(weighted_kurtosis, mag, magerr)

    # ---- unweighted moments ----
    features["mean"]          = _safe(mu, mag)
    features["std"]           = _safe(sigma, mag)
    features["skew"]          = _safe(skewness, mag)
    features["kurt"]          = _safe(kurtosis, mag)

    return features


def _safe_call(func, *args):
    """Call *func* and return NaN on any exception."""
    try:
        val = func(*args)
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan
