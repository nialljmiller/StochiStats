"""
Two-sample comparison statistics.

Useful for comparing lightcurve segments or distributions
(e.g. pre/post-outburst, different epochs).
"""

import numpy as np
from math import erf


def mann_whitney_u_test(data1, data2):
    """
    Mann-Whitney U test (non-parametric rank-sum test).

    Parameters
    ----------
    data1, data2 : array_like
        Two samples to compare.

    Returns
    -------
    min_u : float
        Smaller U statistic.
    max_u : float
        Larger U statistic.
    p_value : float
        Two-sided p-value (normal approximation).
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    n1, n2 = len(data1), len(data2)

    combined = np.concatenate((data1, data2))
    ranks = np.argsort(combined)
    rank_sum1 = np.sum(ranks[:n1])
    rank_sum2 = np.sum(ranks[n1:])

    u1 = rank_sum1 - n1 * (n1 + 1) / 2.0
    u2 = rank_sum2 - n2 * (n2 + 1) / 2.0
    min_u, max_u = min(u1, u2), max(u1, u2)

    expected_u = n1 * n2 / 2.0
    std_error = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_error == 0:
        return min_u, max_u, np.nan
    z = (min_u - expected_u) / std_error
    p_value = 2.0 * (1.0 - abs(0.5 - 0.5 * erf(-z / np.sqrt(2))))
    return min_u, max_u, p_value


def anderson_darling_test(data1, data2):
    """
    Two-sample Anderson-Darling test.

    Parameters
    ----------
    data1, data2 : array_like
        Two samples to compare.

    Returns
    -------
    A2 : float
        Anderson-Darling statistic.
    p_value : float
        Interpolated p-value.
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    combined = np.sort(np.concatenate([data1, data2]))
    n = len(data1)
    ecdf = np.arange(1, len(combined) + 1) / len(combined)
    # Clamp to avoid log(0)
    ecdf_clamped = np.clip(ecdf, 1e-15, 1.0 - 1e-15)
    k = np.arange(1, n + 1)
    A2 = -n - np.sum(
        (2 * k - 1) * (np.log(ecdf_clamped[:n]) + np.log(1.0 - ecdf_clamped[-n:][::-1]))
    )
    critical_values = np.array([0.576, 0.656, 0.787, 0.918, 1.092])
    p_value = np.interp(A2, [0.2, 0.6, 1.0, 1.5, 2.0], critical_values)
    return A2, p_value


def cohens_d(data1, data2):
    """
    Cohen's d effect size.

    Parameters
    ----------
    data1, data2 : array_like
        Two samples to compare.

    Returns
    -------
    float
        Cohen's d (absolute value).
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(data1, ddof=1) ** 2 + (n2 - 1) * np.std(data2, ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return np.nan
    return np.abs(np.mean(data1) - np.mean(data2)) / pooled_std


def emp_cramer_von_mises(data1, data2):
    """
    Empirical Cramér-von Mises statistic.

    Parameters
    ----------
    data1, data2 : array_like
        Two samples to compare.

    Returns
    -------
    float
        CvM statistic.
    """
    data1 = np.sort(np.asarray(data1, dtype=float))
    data2 = np.sort(np.asarray(data2, dtype=float))
    combined = np.sort(np.concatenate([data1, data2]))
    ecdf1 = np.searchsorted(data1, combined, side="right") / len(data1)
    ecdf2 = np.searchsorted(data2, combined, side="right") / len(data2)
    return np.sum((ecdf1 - ecdf2) ** 2)
