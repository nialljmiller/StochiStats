"""
Frequency grid construction for periodogram computation.

PRIMVS uses log-spaced frequency grids split into three successive
test ranges (Table A.1 in the paper) to maximise completeness.
"""

import numpy as np
from typing import Optional, Tuple, List


def make_frequency_grid(
    time_baseline: float,
    F_stop: float = 10.0,
    F_start: Optional[float] = None,
    n_freqs: int = 100_000,
    log_spacing: bool = True,
) -> np.ndarray:
    """
    Build a frequency grid for periodogram computation.

    Parameters
    ----------
    time_baseline : float
        Total time span of the lightcurve (max(time) - min(time)).
    F_stop : float, optional
        Maximum frequency (cycles/day).  Default 10 (P_min = 0.1 d).
    F_start : float, optional
        Minimum frequency.  Default is 2/time_baseline (≥ 2 cycles).
    n_freqs : int, optional
        Number of trial frequencies (default 100,000).
    log_spacing : bool, optional
        If True (default), frequencies are log-spaced; otherwise linear.

    Returns
    -------
    ndarray
        1-D array of trial frequencies in cycles/day.
    """
    if F_start is None:
        F_start = 2.0 / time_baseline
    if F_start <= 0:
        F_start = 1e-6
    if F_start >= F_stop:
        F_stop = F_start * 10

    if log_spacing:
        return np.logspace(np.log10(F_start), np.log10(F_stop), n_freqs)
    else:
        return np.linspace(F_start, F_stop, n_freqs)


def make_primvs_grids(
    time_baseline: float,
    n_freqs: int = 100_000,
) -> List[np.ndarray]:
    """
    Build the three successive frequency grids used in the PRIMVS pipeline.

    Test 1:  1 d < P < 500 d
    Test 2:  0.01 d < P < 1 d
    Test 3:  500 d < P < T_lc / 2

    Parameters
    ----------
    time_baseline : float
        Total time span of the lightcurve.
    n_freqs : int, optional
        Frequencies per grid (default 100,000).

    Returns
    -------
    list of ndarray
        Three frequency grids [test1, test2, test3].
    """
    grids = []
    # Test 1: P in [1, 500] => F in [1/500, 1]
    grids.append(make_frequency_grid(time_baseline, F_start=1.0/500, F_stop=1.0, n_freqs=n_freqs*0.5))
    # Test 2: P in [0.01, 1] => F in [1, 100]
    grids.append(make_frequency_grid(time_baseline, F_start=0.9, F_stop=100.0, n_freqs=n_freqs*2))
    # Test 3: P in [500, T_lc/2] => F in [2/T_lc, 1/500]
    F_start_3 = 2.0 / time_baseline
    F_stop_3 = 1.0 / 500
    if F_start_3 < F_stop_3:
        grids.append(make_frequency_grid(time_baseline, F_start=F_start_3, F_stop=F_stop_3, n_freqs=n_freqs))
    else:
        # time baseline too short for test 3
        grids.append(np.array([]))
    return grids
