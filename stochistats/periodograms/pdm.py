"""
Phase Dispersion Minimization (PDM) periodogram.

Pure-Python implementation of the PDM2 algorithm with subharmonic sampling,
replacing the original C binary wrapper for portability.

The PDM statistic Θ is the ratio of phase-binned variance to total variance.
Θ ≈ 1 for random noise; Θ → 0 for a perfectly periodic signal.

References:
    - Stellingwerf (1978), ApJ 224, 953
    - PDM2: Stellingwerf (2011) with subharmonic sampling
"""

import numpy as np
from typing import Optional, Tuple
from stochistats.periodograms.frequency_grid import make_frequency_grid


def PDM(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    F_start: Optional[float] = None,
    F_stop: float = 10.0,
    n_freqs: int = 100_000,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PDM periodogram.

    Parameters
    ----------
    mag : array_like
        Magnitude values.
    magerr : array_like
        Magnitude uncertainties.
    time : array_like
        Observation times.
    F_start : float, optional
        Minimum frequency.
    F_stop : float, optional
        Maximum frequency.
    n_freqs : int, optional
        Number of trial frequencies.
    n_bins : int, optional
        Number of phase bins (default 10, using 5+5 interleaved as in PDM2).

    Returns
    -------
    freqs : ndarray
        Trial frequencies.
    theta : ndarray
        PDM Θ statistic at each frequency (lower = more periodic).
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    baseline = time.max() - time.min()
    freqs = make_frequency_grid(baseline, F_stop=F_stop, F_start=F_start, n_freqs=n_freqs)
    theta = pdm(time, mag, magerr, freqs, n_bins=n_bins)
    return freqs, theta


def pdm(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    frequency: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Compute the PDM Θ statistic at each trial frequency.

    Uses the PDM2 interleaved-bin scheme (5 odd + 5 even bins)
    for subharmonic sampling.

    Parameters
    ----------
    time, mag, magerr : ndarray
        Lightcurve data.
    frequency : ndarray
        Trial frequencies.
    n_bins : int
        Total number of interleaved bins (default 10).

    Returns
    -------
    ndarray
        Θ values (lower = better period).
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    frequency = np.asarray(frequency, dtype=float)

    total_var = np.var(mag)
    if total_var == 0:
        return np.ones_like(frequency)

    n_half = n_bins // 2
    theta = np.empty(len(frequency))

    for i, freq in enumerate(frequency):
        phase = (time * freq) % 1.0

        # Interleaved bins: odd set and even set (offset by half a bin)
        bin_var_sum = 0.0
        bin_count = 0

        for offset in [0.0, 0.5 / n_half]:
            shifted_phase = (phase + offset) % 1.0
            bin_indices = (shifted_phase * n_half).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_half - 1)

            for b in range(n_half):
                mask = bin_indices == b
                n_in_bin = np.sum(mask)
                if n_in_bin > 1:
                    bin_var_sum += np.var(mag[mask]) * (n_in_bin - 1)
                    bin_count += n_in_bin - 1

        if bin_count > 0:
            theta[i] = (bin_var_sum / bin_count) / total_var
        else:
            theta[i] = 1.0

    return theta
