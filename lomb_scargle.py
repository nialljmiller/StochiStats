"""
Lomb-Scargle periodogram.

Primary implementation uses astropy.timeseries.LombScargle (with FIT_MEAN=True
as per PRIMVS).  Falls back to a chi-squared design-matrix implementation
(from the original LS2.py) if astropy is not available.

Reference: Zechmeister & Kürster (2009), A&A 496, 577
"""

import numpy as np
from typing import Optional, Tuple
from stochistats.periodograms.frequency_grid import make_frequency_grid


def LS(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    F_start: Optional[float] = None,
    F_stop: float = 10.0,
    n_freqs: int = 100_000,
    nterms: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lomb-Scargle periodogram.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.
    time : array_like
        Observation times.
    F_start : float, optional
        Minimum frequency (default: 2/baseline).
    F_stop : float, optional
        Maximum frequency (default: 10).
    n_freqs : int, optional
        Number of trial frequencies (default: 100,000).
    nterms : int, optional
        Number of Fourier terms (default: 1).

    Returns
    -------
    freqs : ndarray
        Trial frequencies.
    power : ndarray
        LS power at each frequency.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    baseline = time.max() - time.min()
    freqs = make_frequency_grid(baseline, F_stop=F_stop, F_start=F_start, n_freqs=n_freqs)
    power = lomb_scargle(time, mag, magerr, freqs, nterms=nterms)
    return freqs, power


def lomb_scargle(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    frequency: np.ndarray,
    nterms: int = 1,
    normalization: str = "standard",
    fit_mean: bool = True,
) -> np.ndarray:
    """
    Compute Lomb-Scargle power at given frequencies.

    Tries astropy first, falls back to chi-squared implementation.

    Parameters
    ----------
    time, mag, magerr : ndarray
        Lightcurve data.
    frequency : ndarray
        Trial frequencies (cycles/day).
    nterms : int
        Fourier terms.
    normalization : str
        'standard', 'model', 'log', or 'psd'.
    fit_mean : bool
        Fit a constant offset.

    Returns
    -------
    ndarray
        Power spectrum.
    """
    try:
        return _ls_astropy(time, mag, magerr, frequency, nterms=nterms,
                           normalization=normalization, fit_mean=fit_mean)
    except ImportError:
        return _ls_chi2(time, mag, magerr, frequency, nterms=nterms,
                        normalization=normalization, fit_mean=fit_mean)


def _ls_astropy(time, mag, magerr, frequency, nterms=1,
                normalization="standard", fit_mean=True):
    """Astropy LombScargle wrapper (preferred)."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(time, mag, magerr, nterms=nterms, fit_mean=fit_mean)
    return ls.power(frequency, normalization=normalization)


def _ls_chi2(time, mag, magerr, frequency, nterms=1,
             normalization="standard", fit_mean=True, center_data=True):
    """
    Chi-squared design-matrix Lomb-Scargle (from original LS2.py).

    References: Zechmeister & Kürster (2009); Press et al. (2002)
    """
    t, y, dy = np.broadcast_arrays(time, mag, magerr)
    frequency = np.asarray(frequency)

    w = dy ** -2.0
    w /= w.sum()

    if center_data or fit_mean:
        yw = (y - np.dot(w, y)) / dy
    else:
        yw = y / dy

    chi2_ref = np.dot(yw, yw)

    def _design_matrix(t, freq, dy, bias, nterms):
        cols = []
        if bias:
            cols.append(1.0 / dy)
        for n in range(1, nterms + 1):
            cols.append(np.sin(2 * np.pi * n * freq * t) / dy)
            cols.append(np.cos(2 * np.pi * n * freq * t) / dy)
        return np.column_stack(cols)

    def _power_at_freq(f):
        X = _design_matrix(t, f, dy=dy, bias=fit_mean, nterms=nterms)
        XTX = X.T @ X
        XTy = X.T @ yw
        try:
            return XTy @ np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            return 0.0

    p = np.array([_power_at_freq(f) for f in frequency])

    if normalization == "standard":
        p /= chi2_ref
    elif normalization == "model":
        p /= (chi2_ref - p)
    elif normalization == "log":
        p = -np.log(1 - p / chi2_ref)
    elif normalization == "psd":
        p *= 0.5
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return p


def ls_false_alarm_probability(power_max, time, mag, magerr=None):
    """
    Baluev (2008) false alarm probability for LS peak power.

    Parameters
    ----------
    power_max : float
        Maximum LS power.
    time, mag : ndarray
        Lightcurve data.
    magerr : ndarray, optional
        Errors.

    Returns
    -------
    float
        Baluev FAP.
    """
    try:
        from astropy.timeseries import LombScargle
        if magerr is None:
            ls = LombScargle(time, mag)
        else:
            ls = LombScargle(time, mag, magerr)
        return float(ls.false_alarm_probability(power_max, method="baluev"))
    except ImportError:
        return np.nan
