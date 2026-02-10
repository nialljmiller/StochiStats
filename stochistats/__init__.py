"""
StochiStats - Time series analysis tools for stochastically sampled periodic signals.

Designed for identifying periodicities and computing useful statistics in
lightcurves from ground-based astronomical surveys (e.g. VVV/PRIMVS).

All public functions are available at the top level::

    import stochistats as ss
    features = ss.calculate_all_features(mag, magerr, time)
    freqs, power = ss.LS(mag, magerr, time)
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Variability indices
# ---------------------------------------------------------------------------
from stochistats.variability import (
    cody_M,
    cody_Q,
    Stetson_K,
    Eta,
    Eta_e,
    medianBRP,
    RangeCumSum,
    MaxSlope,
    MedianAbsDev,
    Meanvariance,
    PercentAmplitude,
    RoMS,
    ptop_var,
    lagauto,
    AndersonDarling,
    stdnxs,
    IQR,
)

# ---------------------------------------------------------------------------
# Statistical moments
# ---------------------------------------------------------------------------
from stochistats.moments import (
    mu,
    sigma,
    skewness,
    kurtosis,
    weighted_mean,
    weighted_variance,
    weighted_skew,
    weighted_kurtosis,
)

# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------
from stochistats.fitting import (
    sine_fit,
    straight_line_fit,
    polyn_fit,
    bin_lc,
    lc_model,
)

# ---------------------------------------------------------------------------
# Two-sample comparison tests
# ---------------------------------------------------------------------------
from stochistats.comparison import (
    mann_whitney_u_test,
    anderson_darling_test,
    cohens_d,
    emp_cramer_von_mises,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
from stochistats.utils import (
    phaser,
    normalise,
    round_sig,
    average_separation,
    dtw,
)

# ---------------------------------------------------------------------------
# High-level feature extraction
# ---------------------------------------------------------------------------
from stochistats.features import calculate_all_features

# ---------------------------------------------------------------------------
# Periodograms (period finding)
# ---------------------------------------------------------------------------
from stochistats.periodograms import (
    LS,
    PDM,
    CE,
    GP,
    make_frequency_grid,
    extract_peaks,
    check_alias,
    exclude_alias_regions,
)

# ---------------------------------------------------------------------------
# Lightcurve cleaning & preprocessing
# ---------------------------------------------------------------------------
from stochistats.cleaning import (
    CleaningConfig,
    clean_lightcurve,
    quality_cut,
    remove_ambiguous_points,
    clip_magerr,
    sigma_clip,
    validate_pawprint_pairs,
    temporal_bin,
    detrend_linear,
    variability_check,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # variability
    "cody_M", "cody_Q", "Stetson_K", "Eta", "Eta_e", "medianBRP",
    "RangeCumSum", "MaxSlope", "MedianAbsDev", "Meanvariance",
    "PercentAmplitude", "RoMS", "ptop_var", "lagauto", "AndersonDarling",
    "stdnxs", "IQR",
    # moments
    "mu", "sigma", "skewness", "kurtosis",
    "weighted_mean", "weighted_variance", "weighted_skew", "weighted_kurtosis",
    # fitting
    "sine_fit", "straight_line_fit", "polyn_fit", "bin_lc", "lc_model",
    # comparison
    "mann_whitney_u_test", "anderson_darling_test", "cohens_d",
    "emp_cramer_von_mises",
    # utils
    "phaser", "normalise", "round_sig", "average_separation", "dtw",
    # features
    "calculate_all_features",
    # periodograms
    "LS", "PDM", "CE", "GP", "make_frequency_grid",
    "extract_peaks", "check_alias", "exclude_alias_regions",
    # cleaning
    "CleaningConfig", "clean_lightcurve",
    "quality_cut", "remove_ambiguous_points", "clip_magerr",
    "sigma_clip", "validate_pawprint_pairs", "temporal_bin",
    "detrend_linear", "variability_check",
]
