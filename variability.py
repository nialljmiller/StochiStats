"""
Variability indices for stochastically sampled time series.

These statistics quantify different aspects of variability in lightcurves,
designed to be robust to uneven sampling and photometric noise typical of
ground-based surveys.

References:
    - Cody et al. (2014): https://arxiv.org/abs/1401.6582  (M and Q values)
    - Stetson (1996): Stetson K variability index
    - Von Neumann (1941): Eta ratio
    - Enoch et al. (2003): Eta_e weighted variant
"""

import numpy as np
from stochistats.utils import phaser
from stochistats.fitting import sine_fit


def cody_M(mag, time):
    """
    Cody M statistic — measures flux asymmetry in a lightcurve.

    Positive M indicates dipping behaviour (e.g. disk occultation),
    negative M indicates bursting behaviour.

    Reference: Cody et al. (2014), §5.2, p.26

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times (unused but kept for API consistency).

    Returns
    -------
    float
        M statistic value.
    """
    mag = np.asarray(mag, dtype=float)
    q90, q10 = np.percentile(mag, [90, 10])
    mask = (mag > q90) | (mag < q10)
    percentile_mean = np.mean(mag[mask])
    rms = np.sqrt(np.mean(mag ** 2))
    return (percentile_mean - np.median(mag)) / rms


def cody_Q(mag, time, period):
    """
    Cody Q statistic — ratio of residual to total variance after sine subtraction.

    Measures how well a sinusoidal model explains the variability.
    Q ≈ 0 for perfectly periodic signals, Q ≈ 1 for stochastic.

    Reference: Cody et al. (2014), §5.1, p.25

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.
    period : float
        Trial period in same units as *time*.

    Returns
    -------
    float
        Q statistic value.
    """
    mag = np.asarray(mag, dtype=float)
    _, _, r2 = sine_fit(mag, time, period)
    phase = phaser(time, period)
    # Build residuals from sine fit
    from scipy.optimize import curve_fit

    def sinus(x, A, B, C):
        return A * np.sin(2.0 * np.pi * x + C) + B

    params, _ = curve_fit(sinus, phase, mag)
    mag_resid = mag - sinus(phase, *params)

    var_total = np.mean(mag ** 2) - np.std(mag) ** 2
    if var_total == 0:
        return np.nan
    return (np.mean(mag_resid ** 2) - np.std(mag) ** 2) / var_total


def Stetson_K(mag, magerr):
    """
    Stetson K variability index.

    Robust measure of the kurtosis of the magnitude distribution
    weighted by photometric uncertainties.  K = 1/√N · Σ|δᵢ| / √(Σδᵢ²)
    where δᵢ = √(N/(N-1)) · (mag - <mag>) / σᵢ.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Stetson K value.  K ≈ 0.798 for Gaussian noise.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    N = len(mag)
    wmean = np.sum(mag / magerr ** 2) / np.sum(1.0 / magerr ** 2)
    delta = np.sqrt(N / (N - 1.0)) * (mag - wmean) / magerr
    return (1.0 / np.sqrt(N)) * np.sum(np.abs(delta)) / np.sqrt(np.sum(delta ** 2))


def Eta(mag, time):
    """
    Von Neumann Eta ratio — ratio of mean-square successive difference to variance.

    Eta ≈ 2 for uncorrelated data; Eta < 2 suggests positive autocorrelation
    (i.e. smooth variability); Eta > 2 suggests anti-correlated noise.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times (used only for ordering; data should be time-sorted).

    Returns
    -------
    float
        Eta value.
    """
    mag = np.asarray(mag, dtype=float)
    N = len(mag)
    mean_mag = np.mean(mag)
    numerator = np.sum((mag[1:] - mag[:-1]) ** 2) / (N - 1)
    denominator = np.sum((mag - mean_mag) ** 2) / (N - 1)
    if denominator == 0:
        return np.nan
    return numerator / denominator


def Eta_e(mag, time):
    """
    Weighted Eta_e — time-gap-weighted version of the Von Neumann ratio.

    Accounts for uneven time sampling by weighting successive differences
    inversely by the square of the time gap.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.

    Returns
    -------
    float
        Eta_e value.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)
    N = len(time)
    dt = time[1:] - time[:-1]
    w = 1.0 / dt ** 2
    w_mean = np.mean(w)
    sigma2 = np.var(mag)
    if sigma2 == 0:
        return np.nan
    S1 = np.sum(w * (mag[1:] - mag[:-1]) ** 2)
    S2 = np.sum(w)
    return w_mean * (time[-1] - time[0]) ** 2 * S1 / (sigma2 * S2 * N ** 2)


def medianBRP(mag, magerr):
    """
    Median Buffer Range Percentage.

    Fraction of points within amplitude/10 of the median magnitude.
    High values indicate concentrated (low-amplitude or symmetric) distributions.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties (unused, kept for API consistency).

    Returns
    -------
    float
        Fraction of points near the median.
    """
    mag = np.asarray(mag, dtype=float)
    median = np.median(mag)
    amplitude = (np.max(mag) - np.min(mag)) / 10.0
    return float(np.sum((mag > median - amplitude) & (mag < median + amplitude))) / len(mag)


def RangeCumSum(mag):
    """
    Range of the cumulative sum.

    Rcs → 0 for a symmetric distribution about the mean.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    float
        Rcs value.
    """
    mag = np.asarray(mag, dtype=float)
    N = len(mag)
    sigma = np.std(mag)
    if sigma == 0:
        return np.nan
    s = np.cumsum(mag - np.mean(mag)) / (N * sigma)
    return np.max(s) - np.min(s)


def MaxSlope(mag, time):
    """
    Maximum slope between consecutive observations.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.

    Returns
    -------
    float
        Maximum |Δmag/Δt|.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)
    dt = time[1:] - time[:-1]
    # Guard against zero time differences
    mask = dt != 0
    slopes = np.abs(mag[1:] - mag[:-1])
    slopes[mask] /= dt[mask]
    slopes[~mask] = 0.0
    return np.max(slopes)


def MedianAbsDev(mag, magerr=None):
    """
    Median Absolute Deviation from the median.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like, optional
        Unused, kept for API consistency.

    Returns
    -------
    float
        MAD value.
    """
    mag = np.asarray(mag, dtype=float)
    return np.median(np.abs(mag - np.median(mag)))


def Meanvariance(mag):
    """
    Coefficient of variation: σ / μ.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    float
        Mean variance ratio.
    """
    mag = np.asarray(mag, dtype=float)
    mean = np.mean(mag)
    if mean == 0:
        return np.nan
    return np.std(mag) / mean


def PercentAmplitude(mag):
    """
    Percent amplitude — maximum distance from the median, normalised by the median.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    float
        Percent amplitude.
    """
    mag = np.asarray(mag, dtype=float)
    median = np.median(mag)
    if median == 0:
        return np.nan
    return np.max(np.abs(mag - median)) / median


def RoMS(mag, magerr):
    """
    Robust Median Statistic.

    1/(N-1) · Σ |mag - median(mag)| / σᵢ

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        RoMS value.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    N = len(mag)
    return np.sum(np.abs(mag - np.median(mag)) / magerr) / (N - 1)


def stdnxs(mag, magerr):
    """
    Normalised excess variance.

    σ²_NXS = 1/(N·<mag>²) · Σ[(mag - <mag>)² - σᵢ²]

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Normalised excess variance.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    mean = np.mean(mag)
    if mean == 0:
        return np.nan
    return np.sum((mag - mean) ** 2 - magerr ** 2) / (len(mag) * mean ** 2)


def ptop_var(mag, magerr):
    """
    Peak-to-peak variability.

    v = (max(mag - σ) - min(mag - σ)) / (max(mag - σ) + min(mag - σ))

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Peak-to-peak variability.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    diff = mag - magerr
    denom = np.max(diff) + np.min(diff)
    if denom == 0:
        return np.nan
    return (np.max(diff) - np.min(diff)) / denom


def lagauto(mag):
    """
    Lag-1 autocorrelation.

    Pearson correlation between the series and its one-step shifted version.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    float
        Lag-1 autocorrelation coefficient.
    """
    mag = np.asarray(mag, dtype=float)
    mean = np.mean(mag)
    denom = np.sum((mag - mean) ** 2)
    if denom == 0:
        return np.nan
    numer = np.sum((mag[:-1] - mean) * (mag[1:] - mean))
    return numer / denom


def AndersonDarling(mag):
    """
    Anderson-Darling statistic for normality (logistic-transformed).

    The raw magnitudes are standardised and mapped through the normal CDF
    before computing A².  The output is squeezed through a logistic so it
    lives in (0, 1): ``1 / (1 + exp(-10·(A² - 0.3)))``.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    float
        Transformed A² statistic.
    """
    from scipy.stats import norm

    mag = np.asarray(mag, dtype=float)
    n = len(mag)
    if n < 3:
        return np.nan

    # Standardise and map through normal CDF
    mu_val = np.mean(mag)
    sigma_val = np.std(mag, ddof=1)
    if sigma_val == 0:
        return np.nan
    z = np.sort((mag - mu_val) / sigma_val)
    cdf_vals = norm.cdf(z)
    # Clamp to avoid log(0)
    cdf_vals = np.clip(cdf_vals, 1e-15, 1.0 - 1e-15)

    k = np.arange(1, n + 1)
    A2 = -n - np.sum((2 * k - 1) * (np.log(cdf_vals) + np.log(1.0 - cdf_vals[::-1]))) / n
    return 1.0 / (1.0 + np.exp(-10.0 * (A2 - 0.3)))


def IQR(mag):
    """
    Extended inter-quartile range percentiles.

    Parameters
    ----------
    mag : array_like
        Magnitude values.

    Returns
    -------
    tuple
        (q0.01, q0.1, q1, q25, q75, q99, q99.9, q99.99)
    """
    mag = np.asarray(mag, dtype=float)
    return tuple(np.percentile(mag, [0.01, 0.1, 1, 25, 75, 99, 99.9, 99.99]))
