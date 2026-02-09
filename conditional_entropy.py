"""
Conditional Entropy (CE) periodogram.

Pure-Python CPU implementation of the conditional entropy method
(Graham et al. 2013).  The original PRIMVS pipeline uses cuvarbase
for GPU-accelerated CE; this provides a portable CPU fallback.

H(m|φ) = Σ_{i,j} p(m_i, φ_j) · ln[ p(φ_j) / p(m_i, φ_j) ]

Lower conditional entropy means a better phase-fold (less scatter).
"""

import numpy as np
from typing import Optional, Tuple
from stochistats.periodograms.frequency_grid import make_frequency_grid


def CE(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    F_start: Optional[float] = None,
    F_stop: float = 10.0,
    n_freqs: int = 100_000,
    phase_bins: int = 10,
    mag_bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Conditional Entropy periodogram.

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
    phase_bins : int, optional
        Number of phase bins (default 10, per Graham et al. 2013).
    mag_bins : int, optional
        Number of magnitude bins.  If None, determined by Hogg (2008)
        jackknife method (default: Sturges' rule as proxy).

    Returns
    -------
    freqs : ndarray
        Trial frequencies.
    entropy : ndarray
        Conditional entropy at each frequency (lower = more periodic).
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    baseline = time.max() - time.min()
    freqs = make_frequency_grid(baseline, F_stop=F_stop, F_start=F_start, n_freqs=n_freqs)
    entropy = conditional_entropy(time, mag, freqs, phase_bins=phase_bins, mag_bins=mag_bins)
    return freqs, entropy


def conditional_entropy(
    time: np.ndarray,
    mag: np.ndarray,
    frequency: np.ndarray,
    phase_bins: int = 10,
    mag_bins: Optional[int] = None,
) -> np.ndarray:
    """
    Compute H(m|φ) at each trial frequency.

    Parameters
    ----------
    time, mag : ndarray
        Lightcurve data.
    frequency : ndarray
        Trial frequencies.
    phase_bins : int
        Number of phase bins.
    mag_bins : int or None
        Number of magnitude bins. If None, uses Sturges' rule.

    Returns
    -------
    ndarray
        Conditional entropy values.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    N = len(mag)

    if mag_bins is None:
        # Sturges' rule as a simple proxy for Hogg (2008) jackknife
        mag_bins = max(3, int(np.ceil(np.log2(N) + 1)))

    # Pre-compute magnitude bin edges
    mag_edges = np.linspace(mag.min() - 1e-10, mag.max() + 1e-10, mag_bins + 1)

    # Digitise magnitudes once (bins are 0-indexed)
    mag_bin_idx = np.digitize(mag, mag_edges) - 1
    mag_bin_idx = np.clip(mag_bin_idx, 0, mag_bins - 1)

    entropy = np.empty(len(frequency))

    for i, freq in enumerate(frequency):
        phase = (time * freq) % 1.0
        phase_bin_idx = (phase * phase_bins).astype(int)
        phase_bin_idx = np.clip(phase_bin_idx, 0, phase_bins - 1)

        # Build 2D histogram
        hist2d = np.zeros((mag_bins, phase_bins), dtype=float)
        for k in range(N):
            hist2d[mag_bin_idx[k], phase_bin_idx[k]] += 1.0

        # Normalise to joint probability
        p_joint = hist2d / N
        # Marginal probability for phase columns
        p_phase = p_joint.sum(axis=0)  # shape (phase_bins,)

        # Compute H(m|φ) = Σ p(m,φ) · ln[ p(φ) / p(m,φ) ]
        H = 0.0
        for mi in range(mag_bins):
            for pj in range(phase_bins):
                if p_joint[mi, pj] > 0 and p_phase[pj] > 0:
                    H += p_joint[mi, pj] * np.log(p_phase[pj] / p_joint[mi, pj])

        entropy[i] = H

    return entropy
