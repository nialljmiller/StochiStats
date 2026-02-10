"""
Statistical moments — weighted and unweighted.

All weighted variants use inverse-variance weighting (wᵢ = 1/σᵢ²)
which is the natural choice for photometric uncertainties.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Unweighted moments
# ---------------------------------------------------------------------------

def mu(mag):
    """Arithmetic mean."""
    return np.mean(np.asarray(mag, dtype=float))


def sigma(mag):
    """Standard deviation (population, ddof=0)."""
    return np.std(np.asarray(mag, dtype=float))


def skewness(mag):
    """
    Sample skewness (Fisher definition, bias-corrected).

    Uses the standard N·/((N-1)(N-2)) adjustment.
    """
    mag = np.asarray(mag, dtype=float)
    n = len(mag)
    if n < 3:
        return np.nan
    m = np.mean(mag)
    s = np.std(mag, ddof=1)
    if s == 0:
        return np.nan
    return (n / ((n - 1) * (n - 2))) * np.sum(((mag - m) / s) ** 3)


def kurtosis(mag):
    """
    Sample excess kurtosis (Fisher definition, bias-corrected).

    Returns 0 for a perfect Gaussian.
    """
    mag = np.asarray(mag, dtype=float)
    n = len(mag)
    if n < 4:
        return np.nan
    m = np.mean(mag)
    s = np.std(mag, ddof=1)
    if s == 0:
        return np.nan
    m4 = np.sum(((mag - m) / s) ** 4)
    return (
        n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * m4
        - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    )


# ---------------------------------------------------------------------------
# Weighted moments (inverse-variance weighting)
# ---------------------------------------------------------------------------

def _weights(magerr):
    """Return inverse-variance weights from magnitude errors."""
    magerr = np.asarray(magerr, dtype=float)
    return 1.0 / magerr ** 2


def weighted_mean(mag, magerr):
    """
    Inverse-variance weighted mean.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Weighted mean.
    """
    mag = np.asarray(mag, dtype=float)
    w = _weights(magerr)
    return np.sum(w * mag) / np.sum(w)


def weighted_variance(mag, magerr):
    """
    Weighted standard deviation.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Weighted standard deviation.
    """
    mag = np.asarray(mag, dtype=float)
    w = _weights(magerr)
    wm = np.sum(w * mag) / np.sum(w)
    return np.sqrt(np.sum(w * (mag - wm) ** 2) / np.sum(w))


def weighted_skew(mag, magerr):
    """
    Weighted skewness.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Weighted skewness.
    """
    mag = np.asarray(mag, dtype=float)
    w = _weights(magerr)
    wm = np.sum(w * mag) / np.sum(w)
    ws = np.sqrt(np.sum(w * (mag - wm) ** 2) / np.sum(w))
    if ws == 0:
        return np.nan
    return np.sum(w * ((mag - wm) / ws) ** 3) / np.sum(w)


def weighted_kurtosis(mag, magerr):
    """
    Weighted excess kurtosis.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.

    Returns
    -------
    float
        Weighted excess kurtosis (0 for Gaussian).
    """
    mag = np.asarray(mag, dtype=float)
    w = _weights(magerr)
    wm = np.sum(w * mag) / np.sum(w)
    ws = np.sqrt(np.sum(w * (mag - wm) ** 2) / np.sum(w))
    if ws == 0:
        return np.nan
    return np.sum(w * ((mag - wm) / ws) ** 4) / np.sum(w) - 3.0
