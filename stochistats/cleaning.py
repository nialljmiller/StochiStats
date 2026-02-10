"""
Lightcurve cleaning and preprocessing for stochastically sampled time series.

Implements the full cleaning pipeline described in Miller et al. (2026),
PRIMVS Section 2.1. Every threshold is configurable via keyword arguments
or via the :class:`CleaningConfig` dataclass.

Stages (in order):
    1. Quality cuts       — chi, ast_res_chisq thresholds
    2. Ambiguous removal  — blended / neighbour-flagged points
    3. Hard error cap     — reject points with magerr above a ceiling
    4. Sigma clipping     — iterative outlier rejection in mag or magerr
    5. Paw-print pairs    — validate close-in-time observation pairs
    6. Temporal binning   — combine points within a time window
    7. Linear detrending  — fit & subtract straight-line trends
    8. Variability check  — post-cleaning selection cuts

Each stage can be called individually or via :func:`clean_lightcurve`
which runs the full pipeline and returns a cleaned dict.

Example
-------
>>> from stochistats.cleaning import clean_lightcurve, CleaningConfig
>>> cfg = CleaningConfig(max_magerr=0.3, min_obs_after=30)
>>> cleaned = clean_lightcurve(mag, magerr, time, config=cfg)
>>> mag_clean = cleaned['mag']
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from stochistats.fitting import straight_line_fit, bin_lc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CleaningConfig:
    """All configurable thresholds for the cleaning pipeline.

    Every field has a sensible default matching the PRIMVS paper.
    Override any subset when constructing.

    Parameters
    ----------
    max_chi : float
        Maximum DoPhot chi for quality cut.
    max_ast_res_chisq : float
        Maximum astrometric residual chi-squared.
    remove_ambiguous : bool
        Whether to drop points flagged as ambiguous / blended.
    max_magerr : float
        Hard ceiling on magnitude error.
    sigma_clip_mag : float or None
        If set, reject mag outliers beyond this many sigma from the median.
    sigma_clip_magerr : float or None
        If set, reject magerr outliers beyond this many sigma.
    pair_max_dt : float
        Maximum time separation (days) for paw-print pair validation.
    pair_merr_tol : float
        Multiplicative tolerance for magnitude agreement in pairs
        (points must be within ``pair_merr_tol * magerr`` of each other).
    pair_err_ratio : float
        Maximum allowed ratio between errors in a pair.
    bin_window_hours : float
        Time window (hours) within which to bin close points.
    bin_min_magerr : float
        Only bin groups whose mean magerr exceeds this value.
    detrend : bool
        Whether to attempt linear detrending.
    detrend_bins : int
        Number of bins for the weighted-median straight-line fit.
    detrend_min_slope : float
        Minimum |dm/dt| (mag/day) to bother subtracting.
    detrend_min_r2 : float
        Minimum R² of the linear fit to bother subtracting.
    min_obs_after : int
        Reject the lightcurve entirely if fewer points remain.
    var_max_magerr : float or None
        Post-cleaning cut: median(magerr) must be below this.
    var_min_amplitude : float or None
        Post-cleaning cut: Q99 − Q01 must exceed this (mag).
    var_iqr_factor : float or None
        Post-cleaning cut: IQR must exceed
        ``var_iqr_factor * median(magerr)``.
    """

    # --- Stage 1: quality cuts ---
    max_chi: float = 10.0
    max_ast_res_chisq: float = 20.0

    # --- Stage 2: ambiguous removal ---
    remove_ambiguous: bool = True

    # --- Stage 3: hard error cap ---
    max_magerr: float = 0.2

    # --- Stage 4: sigma clipping ---
    sigma_clip_mag: Optional[float] = None
    sigma_clip_magerr: Optional[float] = 4.0

    # --- Stage 5: paw-print pair validation ---
    pair_max_dt: float = 0.02  # ~29 min in days
    pair_merr_tol: float = 2.0
    pair_err_ratio: float = 2.0

    # --- Stage 6: temporal binning ---
    bin_window_hours: float = 1.0
    bin_min_magerr: float = 0.1

    # --- Stage 7: linear detrending ---
    detrend: bool = True
    detrend_bins: int = 10
    detrend_min_slope: float = 2e-4  # mag / day
    detrend_min_r2: float = 0.2

    # --- Stage 8: post-cleaning variability check ---
    min_obs_after: int = 40
    var_max_magerr: Optional[float] = 0.5
    var_min_amplitude: Optional[float] = 0.1
    var_iqr_factor: Optional[float] = 2.0


# ---------------------------------------------------------------------------
# Individual cleaning stages
# ---------------------------------------------------------------------------


def quality_cut(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    chi: Optional[np.ndarray] = None,
    ast_res_chisq: Optional[np.ndarray] = None,
    *,
    max_chi: float = 10.0,
    max_ast_res_chisq: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove points that fail chi / ast_res_chisq quality cuts.

    Parameters
    ----------
    mag, magerr, time : array_like
        Lightcurve arrays.
    chi : array_like or None
        DoPhot chi values.  Skipped if ``None``.
    ast_res_chisq : array_like or None
        Astrometric residual chi² values.  Skipped if ``None``.
    max_chi, max_ast_res_chisq : float
        Thresholds.

    Returns
    -------
    mag, magerr, time, mask : ndarrays
        Filtered arrays and the boolean mask applied.
    """
    n = len(mag)
    mask = np.ones(n, dtype=bool)
    if chi is not None:
        mask &= np.asarray(chi) < max_chi
    if ast_res_chisq is not None:
        mask &= np.asarray(ast_res_chisq) < max_ast_res_chisq
    return mag[mask], magerr[mask], time[mask], mask


def remove_ambiguous_points(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    ambiguous: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove points flagged as ambiguous / blended neighbours.

    Parameters
    ----------
    ambiguous : array_like of bool or int
        True / 1 for ambiguous points.

    Returns
    -------
    mag, magerr, time, mask
    """
    mask = ~np.asarray(ambiguous, dtype=bool)
    return mag[mask], magerr[mask], time[mask], mask


def clip_magerr(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    *,
    max_magerr: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hard ceiling on magnitude error.

    Returns
    -------
    mag, magerr, time, mask
    """
    mask = magerr <= max_magerr
    return mag[mask], magerr[mask], time[mask], mask


def sigma_clip(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    *,
    sigma_mag: Optional[float] = None,
    sigma_magerr: Optional[float] = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sigma-clip outliers in magnitude and/or magnitude error.

    Parameters
    ----------
    sigma_mag : float or None
        Clip mag values beyond ``sigma_mag`` × MAD from the median.
    sigma_magerr : float or None
        Clip magerr values beyond ``sigma_magerr`` × std from the mean.

    Returns
    -------
    mag, magerr, time, mask
    """
    mask = np.ones(len(mag), dtype=bool)
    if sigma_mag is not None:
        med = np.median(mag)
        mad = np.median(np.abs(mag - med))
        # 1.4826 converts MAD to Gaussian σ equivalent
        scale = 1.4826 * mad if mad > 0 else np.std(mag)
        mask &= np.abs(mag - med) < sigma_mag * scale
    if sigma_magerr is not None:
        mu_err = np.mean(magerr)
        std_err = np.std(magerr)
        if std_err > 0:
            mask &= magerr < mu_err + sigma_magerr * std_err
    return mag[mask], magerr[mask], time[mask], mask


def validate_pawprint_pairs(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    *,
    max_dt: float = 0.02,
    merr_tol: float = 2.0,
    err_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate paw-print pair observations (Section 2.1, paragraph 4).

    For every pair of points within ``max_dt`` days of each other:
    - Their magnitudes must agree within ``merr_tol × max(magerr_i, magerr_j)``
    - Their errors must be within a factor of ``err_ratio``
    If either condition fails, **both** points are rejected.

    Parameters
    ----------
    max_dt : float
        Maximum time separation in days.
    merr_tol : float
        Tolerance multiplier for magnitude agreement.
    err_ratio : float
        Maximum allowed ratio between the two errors.

    Returns
    -------
    mag, magerr, time, mask
    """
    n = len(mag)
    keep = np.ones(n, dtype=bool)

    # Sort by time
    order = np.argsort(time)
    t_sorted = time[order]
    m_sorted = mag[order]
    e_sorted = magerr[order]

    for i in range(n - 1):
        j = i + 1
        while j < n and (t_sorted[j] - t_sorted[i]) < max_dt:
            dt = t_sorted[j] - t_sorted[i]
            if dt > 0:  # genuine pair
                max_err = max(e_sorted[i], e_sorted[j])
                mag_ok = abs(m_sorted[i] - m_sorted[j]) < merr_tol * max_err
                min_err = min(e_sorted[i], e_sorted[j])
                err_ok = (max_err / min_err) < err_ratio if min_err > 0 else False
                if not (mag_ok and err_ok):
                    keep[order[i]] = False
                    keep[order[j]] = False
            j += 1

    return mag[keep], magerr[keep], time[keep], keep


def temporal_bin(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    *,
    window_hours: float = 1.0,
    min_magerr: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine points within a time window (Section 2.1, paragraph 4).

    Groups of points within ``window_hours`` of each other where the
    mean magerr > ``min_magerr`` are binned into a single point:
    - mag_new   = weighted mean (weights = 1/magerr²)
    - magerr_new = (1/√N) × Σ magerr_i
    - time_new  = mean(time)

    Groups where mean magerr ≤ ``min_magerr`` are left untouched.

    Returns
    -------
    mag, magerr, time  (new arrays, possibly shorter)
    """
    if len(mag) == 0:
        return mag.copy(), magerr.copy(), time.copy()

    window_days = window_hours / 24.0
    order = np.argsort(time)
    t = time[order]
    m = mag[order]
    e = magerr[order]

    new_m, new_e, new_t = [], [], []
    i = 0
    n = len(t)

    while i < n:
        # Collect group within window
        j = i + 1
        while j < n and (t[j] - t[i]) < window_days:
            j += 1

        group_m = m[i:j]
        group_e = e[i:j]
        group_t = t[i:j]
        N = len(group_m)

        if N > 1 and np.mean(group_e) > min_magerr:
            # Bin: weighted mean mag, combined error, mean time
            w = 1.0 / group_e ** 2
            wm = np.average(group_m, weights=w)
            combined_err = (1.0 / np.sqrt(N)) * np.sum(group_e)
            new_m.append(wm)
            new_e.append(combined_err)
            new_t.append(np.mean(group_t))
        else:
            # Keep individual points
            for k in range(N):
                new_m.append(group_m[k])
                new_e.append(group_e[k])
                new_t.append(group_t[k])
        i = j

    return np.array(new_m), np.array(new_e), np.array(new_t)


def detrend_linear(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    *,
    n_bins: int = 10,
    min_slope: float = 2e-4,
    min_r2: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit and optionally subtract a straight-line trend (Section 2.1, paragraph 5).

    A straight line is fitted to ``n_bins`` weighted-median bins of the
    lightcurve.  It is subtracted only if |dm/dt| > ``min_slope`` mag/day
    **and** R² > ``min_r2``.

    Parameters
    ----------
    n_bins : int
        Number of bins for the weighted-median fit.
    min_slope : float
        Minimum |gradient| in mag/day to trigger subtraction.
    min_r2 : float
        Minimum R² of the linear fit to trigger subtraction.

    Returns
    -------
    mag_out : ndarray
        Detrended (or original) magnitudes.
    trend : ndarray
        The linear trend that was subtracted (zeros if not subtracted).
    r2 : float
        R² of the linear fit.
    """
    mag = np.asarray(mag, dtype=float)
    time = np.asarray(time, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    if len(mag) < n_bins:
        return mag.copy(), np.zeros_like(mag), 0.0

    y_fit, params, r2 = straight_line_fit(mag, time, magerr, res=n_bins)

    if np.isnan(r2):
        return mag.copy(), np.zeros_like(mag), 0.0

    gradient = params[0]  # mag / day
    if abs(gradient) > min_slope and r2 > min_r2:
        return mag - y_fit + np.median(mag), y_fit, r2
    else:
        return mag.copy(), np.zeros_like(mag), r2


def variability_check(
    mag: np.ndarray,
    magerr: np.ndarray,
    *,
    min_obs: int = 40,
    max_magerr: Optional[float] = 0.5,
    min_amplitude: Optional[float] = 0.1,
    iqr_factor: Optional[float] = 2.0,
) -> Tuple[bool, Dict[str, float]]:
    """Post-cleaning variability selection (Table 2 in PRIMVS paper).

    Returns whether the lightcurve passes *all* enabled checks, plus
    a dict of the computed metrics for diagnostics.

    Parameters
    ----------
    min_obs : int
        Minimum number of observations.
    max_magerr : float or None
        Maximum median(magerr).  ``None`` disables.
    min_amplitude : float or None
        Minimum (Q99 − Q01).  ``None`` disables.
    iqr_factor : float or None
        IQR must exceed ``iqr_factor × median(magerr)``.  ``None`` disables.

    Returns
    -------
    passes : bool
    metrics : dict
    """
    n = len(mag)
    med_err = float(np.median(magerr)) if n > 0 else np.nan
    q01, q25, q75, q99 = (
        np.percentile(mag, [1, 25, 75, 99]) if n > 0
        else (np.nan, np.nan, np.nan, np.nan)
    )
    amplitude = float(q99 - q01)
    iqr = float(q75 - q25)

    metrics = {
        "n_obs": n,
        "median_magerr": med_err,
        "amplitude_q99_q01": amplitude,
        "iqr": iqr,
    }

    passes = True
    if n < min_obs:
        passes = False
    if max_magerr is not None and med_err > max_magerr:
        passes = False
    if min_amplitude is not None and amplitude < min_amplitude:
        passes = False
    if iqr_factor is not None and iqr < iqr_factor * med_err:
        passes = False

    return passes, metrics


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def clean_lightcurve(
    mag,
    magerr,
    time,
    chi=None,
    ast_res_chisq=None,
    ambiguous=None,
    config: Optional[CleaningConfig] = None,
    **overrides,
) -> Dict:
    """Run the complete lightcurve cleaning pipeline.

    Applies every stage in order.  Any parameter from
    :class:`CleaningConfig` can also be passed as a keyword argument
    to override individual thresholds without constructing a config object.

    Parameters
    ----------
    mag, magerr, time : array_like
        Core lightcurve arrays.
    chi : array_like or None
        DoPhot chi values.
    ast_res_chisq : array_like or None
        Astrometric residual chi².
    ambiguous : array_like or None
        Ambiguous-match flags.
    config : CleaningConfig or None
        Configuration object.  If ``None``, defaults are used.
    **overrides
        Any :class:`CleaningConfig` field name → value.

    Returns
    -------
    dict
        ``mag``, ``magerr``, ``time`` : cleaned arrays.
        ``n_initial``, ``n_final`` : point counts.
        ``detrend_r2`` : R² of the linear fit (0 if not detrended).
        ``trend`` : the subtracted trend array.
        ``passes_variability`` : bool from post-cleaning check.
        ``variability_metrics`` : dict of computed metrics.
        ``log`` : list of (stage_name, n_before, n_after) tuples.
    """
    mag = np.asarray(mag, dtype=float).copy()
    magerr = np.asarray(magerr, dtype=float).copy()
    time = np.asarray(time, dtype=float).copy()

    # Build config
    if config is None:
        config = CleaningConfig(**overrides)
    elif overrides:
        # Apply overrides on top of provided config
        from dataclasses import asdict
        d = asdict(config)
        d.update(overrides)
        config = CleaningConfig(**d)

    n_initial = len(mag)
    log = []

    def _log(name, n_before, n_after):
        log.append((name, n_before, n_after))

    # ---- Stage 1: quality cuts ----
    if chi is not None or ast_res_chisq is not None:
        n_before = len(mag)
        mag, magerr, time, mask = quality_cut(
            mag, magerr, time, chi, ast_res_chisq,
            max_chi=config.max_chi,
            max_ast_res_chisq=config.max_ast_res_chisq,
        )
        # Apply mask to auxiliary arrays
        if chi is not None:
            chi = np.asarray(chi)[mask] if mask is not None else chi
        if ast_res_chisq is not None:
            ast_res_chisq = np.asarray(ast_res_chisq)[mask] if mask is not None else ast_res_chisq
        if ambiguous is not None:
            ambiguous = np.asarray(ambiguous)[mask] if mask is not None else ambiguous
        _log("quality_cut", n_before, len(mag))

    # ---- Stage 2: ambiguous removal ----
    if config.remove_ambiguous and ambiguous is not None:
        n_before = len(mag)
        mag, magerr, time, _ = remove_ambiguous_points(mag, magerr, time, ambiguous)
        _log("remove_ambiguous", n_before, len(mag))

    # ---- Stage 3: hard error cap ----
    n_before = len(mag)
    mag, magerr, time, _ = clip_magerr(
        mag, magerr, time, max_magerr=config.max_magerr,
    )
    _log("clip_magerr", n_before, len(mag))

    # ---- Stage 4: sigma clipping ----
    if config.sigma_clip_mag is not None or config.sigma_clip_magerr is not None:
        n_before = len(mag)
        mag, magerr, time, _ = sigma_clip(
            mag, magerr, time,
            sigma_mag=config.sigma_clip_mag,
            sigma_magerr=config.sigma_clip_magerr,
        )
        _log("sigma_clip", n_before, len(mag))

    # ---- Stage 5: paw-print pair validation ----
    n_before = len(mag)
    mag, magerr, time, _ = validate_pawprint_pairs(
        mag, magerr, time,
        max_dt=config.pair_max_dt,
        merr_tol=config.pair_merr_tol,
        err_ratio=config.pair_err_ratio,
    )
    _log("pawprint_pairs", n_before, len(mag))

    # ---- Stage 6: temporal binning ----
    n_before = len(mag)
    mag, magerr, time = temporal_bin(
        mag, magerr, time,
        window_hours=config.bin_window_hours,
        min_magerr=config.bin_min_magerr,
    )
    _log("temporal_bin", n_before, len(mag))

    # ---- Stage 7: linear detrending ----
    detrend_r2 = 0.0
    trend = np.zeros_like(mag)
    if config.detrend and len(mag) >= config.detrend_bins:
        mag, trend, detrend_r2 = detrend_linear(
            mag, magerr, time,
            n_bins=config.detrend_bins,
            min_slope=config.detrend_min_slope,
            min_r2=config.detrend_min_r2,
        )
        _log("detrend_linear", len(mag), len(mag))  # no points removed

    # ---- Stage 8: variability check ----
    passes, var_metrics = variability_check(
        mag, magerr,
        min_obs=config.min_obs_after,
        max_magerr=config.var_max_magerr,
        min_amplitude=config.var_min_amplitude,
        iqr_factor=config.var_iqr_factor,
    )

    return {
        "mag": mag,
        "magerr": magerr,
        "time": time,
        "n_initial": n_initial,
        "n_final": len(mag),
        "detrend_r2": detrend_r2,
        "trend": trend,
        "passes_variability": passes,
        "variability_metrics": var_metrics,
        "log": log,
    }
