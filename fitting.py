"""
Curve fitting utilities for lightcurve modelling.

Provides sine, polynomial, and straight-line fitting, along with
lightcurve binning and template model generation.
"""

import numpy as np
from scipy.optimize import curve_fit
from stochistats.utils import phaser


def sine_fit(mag, time, period):
    """
    Fit a sinusoid to phase-folded data and return the fit, parameters, and R².

    Model: A · sin(2π·φ + C) + B

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.
    period : float
        Trial period for phase folding.

    Returns
    -------
    y_fit : ndarray
        Fitted magnitude values.
    params : ndarray
        [A, B, C] — amplitude, offset, phase shift.
    r2 : float
        Coefficient of determination.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)
    phase = phaser(time, period)

    def sinus(x, A, B, C):
        return A * np.sin(2.0 * np.pi * x + C) + B

    try:
        params, _ = curve_fit(sinus, phase, mag)
    except RuntimeError:
        return np.full_like(mag, np.nan), np.array([np.nan] * 3), np.nan

    y_fit = sinus(phase, *params)
    ss_res = np.sum((mag - y_fit) ** 2)
    ss_tot = np.sum((mag - np.mean(mag)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return y_fit, params, r2


def straight_line_fit(mag, time, magerr, res=10):
    """
    Fit a straight line to binned lightcurve data.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.
    magerr : array_like
        Magnitude uncertainties.
    res : int, optional
        Number of bins (default 10).

    Returns
    -------
    y_fit : ndarray
        Fitted values at original time stamps.
    params : ndarray
        [gradient, intercept].
    r2 : float
        Coefficient of determination.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)

    rx, rq50 = bin_lc(time, mag, res=res)

    def line(x, A, B):
        return A * x + B

    try:
        params, _ = curve_fit(line, rx, rq50)
    except RuntimeError:
        return np.full_like(mag, np.nan), np.array([np.nan, np.nan]), np.nan

    y_fit = line(time, *params)
    ss_res = np.sum((mag - y_fit) ** 2)
    ss_tot = np.sum((mag - np.mean(mag)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return y_fit, params, r2


def polyn_fit(mag, time, magerr, terms=3):
    """
    Polynomial fit to a lightcurve.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    time : array_like
        Observation times.
    magerr : array_like
        Magnitude uncertainties (used as sigma in the fit).
    terms : int, optional
        Polynomial degree (default 3).

    Returns
    -------
    y_fit : ndarray
        Fitted values.
    params : ndarray
        Polynomial coefficients (lowest order first).
    r2 : float
        Coefficient of determination.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    def polynomial(x, *coeffs):
        return sum(c * x ** i for i, c in enumerate(coeffs))

    try:
        params, _ = curve_fit(
            polynomial, time, mag,
            p0=[0.0] * (terms + 1),
            sigma=magerr,
            absolute_sigma=True,
        )
    except RuntimeError:
        return np.full_like(mag, np.nan), np.array([np.nan] * (terms + 1)), np.nan

    y_fit = polynomial(time, *params)
    ss_res = np.sum((mag - y_fit) ** 2)
    ss_tot = np.sum((mag - np.mean(mag)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return y_fit, params, r2


def bin_lc(time, mag, res=10):
    """
    Bin a lightcurve into *res* equal-width time bins, returning median magnitudes.

    Parameters
    ----------
    time : array_like
        Observation times.
    mag : array_like
        Magnitude values.
    res : int, optional
        Number of bins (default 10).

    Returns
    -------
    bin_centres : ndarray
        Bin centre times.
    bin_medians : ndarray
        Median magnitude in each bin.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    rx = np.linspace(time.min(), time.max(), res)
    delta = (time.max() - time.min()) / (2.0 * res)

    rq50 = np.empty(res)
    for i in range(res):
        mask = (time > rx[i] - delta) & (time < rx[i] + delta)
        rq50[i] = np.median(mag[mask]) if np.any(mask) else np.nan
    return rx, rq50


def lc_model(cat_type, phase):
    """
    Generate a template lightcurve model for a given variable-star class.

    Parameters
    ----------
    cat_type : str
        Class label, one of ``'EB0'``, ``'EB1'``, ``'EB2'``, ``'EB3'``,
        ``'pulsator'``, ``'sinusoidal'``.
    phase : array_like
        Phase values in [0, 1].

    Returns
    -------
    ndarray
        Model magnitudes at the requested phases.
    """
    x = np.asarray(phase, dtype=float)

    if "EB0" in cat_type:
        A1, A2 = 0.7, 0.3
    elif "EB1" in cat_type:
        A1, A2 = 0.3, 0.7
    elif "EB2" in cat_type:
        A1, A2 = 0.01, 0.99
    elif "EB3" in cat_type:
        A1, A2 = 0.5, 0.5
    elif "pulsator" in cat_type:
        return np.sin(2.0 * np.pi * x)
    elif "sinusoidal" in cat_type:
        return np.sin(2.0 * np.pi * x)
    else:
        raise ValueError(f"Unknown cat_type: {cat_type}")

    return A1 * np.sin(2.0 * np.pi * x) ** 2 - A2 * np.sin(np.pi * x) ** 2
