# StochiStats

Time series analysis tools for stochastically sampled periodic signals — designed for identifying periodicities and computing variability statistics in lightcurves from ground-based astronomical surveys (e.g. VVV/PRIMVS).

## Installation

```bash
git clone https://github.com/nialljmiller/StochiStats.git
cd StochiStats
pip install -e .
```

With optional dependencies (emcee, celerite2, scikit-learn):
```bash
pip install -e ".[full]"
```

## Quick start

```python
import stochistats as ss

# Calculate all PRIMVS features in one call
features = ss.calculate_all_features(mag, magerr, time)

# Or use individual functions
k = ss.Stetson_K(mag, magerr)
eta = ss.Eta(mag, time)
wm = ss.weighted_mean(mag, magerr)
```

Backwards-compatible: `import StochiStats as ss` still works.

## Package structure

```
stochistats/
├── __init__.py        # Public API (all symbols importable from top-level)
├── variability.py     # Cody M/Q, Stetson K, Eta, RoMS, MAD, etc.
├── moments.py         # Weighted & unweighted mean, std, skew, kurtosis
├── fitting.py         # Sine, polynomial, straight-line fits; binning
├── comparison.py      # Mann-Whitney, Anderson-Darling, Cohen's d, CvM
├── utils.py           # phaser, normalise, DTW, round_sig
├── features.py        # calculate_all_features() convenience function
└── tests/
    └── test_stochistats.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Modules

| Module | Contents |
|---|---|
| `variability` | Cody M/Q, Stetson K, Eta, Eta_e, medianBRP, RangeCumSum, MaxSlope, MAD, Meanvariance, PercentAmplitude, RoMS, stdnxs, ptop_var, lagauto, AndersonDarling, IQR |
| `moments` | mu, sigma, skewness, kurtosis, weighted_mean, weighted_variance, weighted_skew, weighted_kurtosis |
| `fitting` | sine_fit, straight_line_fit, polyn_fit, bin_lc, lc_model |
| `comparison` | mann_whitney_u_test, anderson_darling_test, cohens_d, emp_cramer_von_mises |
| `utils` | phaser, normalise, round_sig, average_separation, dtw |
| `features` | calculate_all_features (one-call interface for PRIMVS pipeline) |
