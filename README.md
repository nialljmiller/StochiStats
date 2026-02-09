# StochiStats

Time series analysis tools for stochastically sampled periodic signals — designed for identifying periodicities and computing variability statistics in lightcurves from ground-based astronomical surveys (e.g. VVV/PRIMVS).

## Installation

```bash
git clone https://github.com/nialljmiller/StochiStats.git
cd StochiStats
pip install -e .
```

With optional dependencies (astropy, emcee, celerite2, scikit-learn):
```bash
pip install -e ".[full]"
```

## Quick start

```python
import stochistats as ss

# Calculate all PRIMVS variability features in one call
features = ss.calculate_all_features(mag, magerr, time)

# Or use individual functions
k = ss.Stetson_K(mag, magerr)
eta = ss.Eta(mag, time)
wm = ss.weighted_mean(mag, magerr)
```

Backwards-compatible: `from StochiStats import Stetson_K, Eta, LS, PDM` still works.

## Period finding

```python
import stochistats as ss

# Lomb-Scargle periodogram
freqs, power = ss.LS(mag, magerr, time, n_freqs=100000)

# Phase Dispersion Minimization
freqs, theta = ss.PDM(mag, magerr, time, n_freqs=100000)

# Conditional Entropy
freqs, entropy = ss.CE(mag, magerr, time, n_freqs=100000)

# Gaussian Process (requires celerite2 or celerite)
result = ss.GP(mag, magerr, time)  # returns dict with 'period', 'log_likelihood', etc.

# Extract top 3 peaks from any periodogram
peaks = ss.extract_peaks(freqs, power, n_peaks=3)            # for LS (maximize)
peaks = ss.extract_peaks(freqs, theta, n_peaks=3, minimize=True)  # for PDM/CE

# Check for known aliases (diurnal, lunar, yearly)
alias = ss.check_alias(peaks[0]['period'])

# Mask alias regions before peak extraction
freqs, power_clean = ss.exclude_alias_regions(freqs, power)
```

## Package structure

```
StochiStats/
├── stochistats/                        # Main package
│   ├── __init__.py                     # Public API (all symbols at top level)
│   ├── variability.py                  # 17 variability indices
│   ├── moments.py                      # Weighted & unweighted statistics
│   ├── fitting.py                      # Curve fitting utilities
│   ├── comparison.py                   # Two-sample comparison tests
│   ├── utils.py                        # Helpers (phaser, normalise, DTW)
│   ├── features.py                     # calculate_all_features() for PRIMVS
│   ├── periodograms/                   # Period-finding methods
│   │   ├── __init__.py
│   │   ├── frequency_grid.py           # Frequency grid construction
│   │   ├── lomb_scargle.py             # LS (astropy + chi² fallback)
│   │   ├── pdm.py                      # Phase Dispersion Minimization
│   │   ├── conditional_entropy.py      # Conditional Entropy
│   │   ├── gp.py                       # Gaussian Process (celerite2/celerite)
│   │   └── peak_analysis.py            # Peak extraction & alias checking
│   └── tests/
│       ├── test_stochistats.py
│       └── test_periodograms.py
├── StochiStats.py                      # Backwards-compatibility shim
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```

## Modules

| Module | Contents |
|---|---|
| `variability` | Cody M/Q, Stetson K, Eta, Eta_e, medianBRP, RangeCumSum, MaxSlope, MAD, Meanvariance, PercentAmplitude, RoMS, stdnxs, ptop_var, lagauto, AndersonDarling, IQR |
| `moments` | mu, sigma, skewness, kurtosis, weighted_mean, weighted_variance, weighted_skew, weighted_kurtosis |
| `fitting` | sine_fit, straight_line_fit, polyn_fit, bin_lc, lc_model |
| `comparison` | mann_whitney_u_test, anderson_darling_test, cohens_d, emp_cramer_von_mises |
| `utils` | phaser, normalise, round_sig, average_separation, dtw |
| `features` | calculate_all_features (one-call interface returning all 28 PRIMVS features) |
| `periodograms` | LS, PDM, CE, GP, make_frequency_grid, extract_peaks, check_alias, exclude_alias_regions |

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Dependencies

**Required:** numpy ≥ 1.20, scipy ≥ 1.7

**Optional (`[full]`):** astropy, scikit-learn, emcee, celerite2

## Citation

If you use StochiStats in your research, please cite the PRIMVS paper:

> Miller, N. et al. (2026), "PRIMVS: PeRiodic Infrared Milky-way VVV Star-catalog", A&A.
