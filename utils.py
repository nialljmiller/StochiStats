"""
Utility helpers for time-series analysis.
"""

import numpy as np
from math import floor, log10


def phaser(time, period):
    """
    Phase-fold a time array on a given period.

    Parameters
    ----------
    time : array_like
        Observation times.
    period : float
        Period to fold on.

    Returns
    -------
    ndarray
        Phase values in [0, 1).
    """
    time = np.asarray(time, dtype=float)
    return (time % period) / period


def normalise(x):
    """
    Min-max normalise an array to [0, 1].

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    ndarray
        Normalised values.
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def round_sig(x, sig=3):
    """
    Round *x* to *sig* significant figures.

    Parameters
    ----------
    x : float
        Value to round.
    sig : int, optional
        Number of significant figures (default 3).

    Returns
    -------
    float
        Rounded value.
    """
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def average_separation(x):
    """
    Compute the differences between consecutive sorted values.

    Parameters
    ----------
    x : array_like
        Input values (will be sorted).

    Returns
    -------
    ndarray
        Array of consecutive differences.
    """
    x = np.sort(np.asarray(x, dtype=float))
    return np.diff(x)


def dtw(s, t):
    """
    Dynamic Time Warping distance matrix.

    Parameters
    ----------
    s, t : array_like
        Two 1-D sequences.

    Returns
    -------
    ndarray
        (len(s)+1, len(t)+1) cost matrix.  The DTW distance is at [-1, -1].
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    n, m = len(s), len(t)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )
    return dtw_matrix
