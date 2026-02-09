"""
Periodogram peak extraction and alias analysis.

Extracts the top N peaks from a periodogram (requiring troughs on both sides),
characterises peak widths, and checks for common aliases (diurnal, lunar, yearly,
harmonics).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


# Known alias periods to exclude (days)
DIURNAL_ALIASES = [1.0, 0.5, 1.0 / 3]  # 1d, 0.5d, 8h
LUNAR_ALIAS = 29.53  # synodic month
YEARLY_ALIAS = 365.25


def extract_peaks(
    freqs: np.ndarray,
    power: np.ndarray,
    n_peaks: int = 3,
    minimize: bool = False,
) -> List[Dict[str, float]]:
    """
    Extract the top N peaks from a periodogram.

    A peak is defined as a local extremum with a trough on each side,
    preventing double-extraction of wide features or edge artefacts
    (per PRIMVS §A.6).

    Parameters
    ----------
    freqs : ndarray
        Frequency array.
    power : ndarray
        Power / statistic array.
    n_peaks : int, optional
        Number of peaks to extract (default 3).
    minimize : bool, optional
        If True, search for minima (for PDM/CE where lower = better).
        If False (default), search for maxima (LS).

    Returns
    -------
    list of dict
        Each dict has keys: 'freq', 'period', 'power', 'width', 'width_freq'.
        Sorted by significance (best first).
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)

    if len(freqs) < 3:
        return []

    # For minimization methods, invert so we can always search for maxima
    search_power = -power if minimize else power.copy()

    # Find all local maxima (with troughs on both sides)
    peaks = []
    for i in range(1, len(search_power) - 1):
        if search_power[i] > search_power[i - 1] and search_power[i] > search_power[i + 1]:
            # Find trough boundaries
            left = i - 1
            while left > 0 and search_power[left] > search_power[left - 1]:
                left -= 1
            right = i + 1
            while right < len(search_power) - 1 and search_power[right] > search_power[right + 1]:
                right += 1

            width_freq = freqs[right] - freqs[left]
            width_period = abs(1.0 / freqs[left] - 1.0 / freqs[right]) if freqs[left] > 0 else np.inf

            peaks.append({
                "freq": float(freqs[i]),
                "period": float(1.0 / freqs[i]) if freqs[i] > 0 else np.inf,
                "power": float(power[i]),
                "width_freq": float(width_freq),
                "width": float(width_period),
                "_score": float(search_power[i]),
            })

    # Sort by score (highest first) and take top N
    peaks.sort(key=lambda p: p["_score"], reverse=True)
    for p in peaks:
        del p["_score"]

    return peaks[:n_peaks]


def check_alias(
    period: float,
    tolerance: float = 0.01,
) -> Optional[str]:
    """
    Check if a period is near a known alias.

    Parameters
    ----------
    period : float
        Candidate period in days.
    tolerance : float, optional
        Fractional tolerance for matching (default 1%).

    Returns
    -------
    str or None
        Description of the alias if matched, else None.
    """
    checks = [
        (1.0, "diurnal (1 day)"),
        (0.5, "diurnal half (0.5 day)"),
        (1.0 / 3, "diurnal third (8 hours)"),
        (2.0, "2-day alias"),
        (LUNAR_ALIAS, "lunar (29.53 days)"),
        (YEARLY_ALIAS, "yearly (365.25 days)"),
    ]

    for alias_period, label in checks:
        if abs(period - alias_period) / alias_period < tolerance:
            return label

    # Check harmonics: P/2, 2P
    # (these aren't necessarily aliases but common confusions)
    return None


def check_harmonic(period1: float, period2: float, tolerance: float = 0.01) -> Optional[str]:
    """
    Check if two periods are harmonically related.

    Parameters
    ----------
    period1, period2 : float
        Two candidate periods.
    tolerance : float
        Fractional tolerance.

    Returns
    -------
    str or None
        Relationship description if harmonic, else None.
    """
    if period1 <= 0 or period2 <= 0:
        return None
    ratio = period1 / period2
    for n in [0.5, 1.0 / 3, 2.0, 3.0]:
        if abs(ratio - n) / n < tolerance:
            return f"P1/P2 ≈ {n:.1f}"
    return None


def exclude_alias_regions(
    freqs: np.ndarray,
    power: np.ndarray,
    alias_periods: Optional[List[float]] = None,
    tolerance: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask out frequency regions near known aliases.

    Sets the power to NaN in regions near diurnal, lunar, and yearly aliases
    (per PRIMVS §A.6 which excludes these problematic regions).

    Parameters
    ----------
    freqs : ndarray
        Frequency array.
    power : ndarray
        Power array (will be copied, not modified in-place).
    alias_periods : list of float, optional
        Alias periods to exclude. Default: diurnal (1d, 2d, 29d, 60d, 365d).
    tolerance : float, optional
        Fractional width of exclusion zone (default 2%).

    Returns
    -------
    freqs : ndarray
        Same frequency array.
    power_masked : ndarray
        Power with alias regions set to NaN.
    """
    if alias_periods is None:
        alias_periods = [1.0, 2.0, LUNAR_ALIAS, 60.0, YEARLY_ALIAS]

    freqs = np.asarray(freqs, dtype=float)
    power_masked = np.array(power, dtype=float)

    for ap in alias_periods:
        alias_freq = 1.0 / ap
        mask = np.abs(freqs - alias_freq) / alias_freq < tolerance
        power_masked[mask] = np.nan

    return freqs, power_masked
