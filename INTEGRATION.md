# Adding the cleaning module to StochiStats

## Files to add

1. `stochistats/cleaning.py` → drop into your existing `stochistats/` directory
2. `stochistats/tests/test_cleaning.py` → drop into `stochistats/tests/`

## Update `stochistats/__init__.py`

Add this block after the existing imports:

```python
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
```

And add these to `__all__`:

```python
"CleaningConfig", "clean_lightcurve",
"quality_cut", "remove_ambiguous_points", "clip_magerr",
"sigma_clip", "validate_pawprint_pairs", "temporal_bin",
"detrend_linear", "variability_check",
```

## Usage

```python
import stochistats as ss

# Use defaults (paper values)
result = ss.clean_lightcurve(mag, magerr, time, chi=chi, ambiguous=ambig)

# Override specific thresholds
cfg = ss.CleaningConfig(
    var_max_magerr=0.3,      # relax Ks error cut
    var_iqr_factor=3.0,      # stricter IQR requirement
    max_magerr=0.15,         # tighter hard error cap
    min_obs_after=30,         # fewer minimum points
    var_min_amplitude=None,   # disable amplitude check entirely
)
result = ss.clean_lightcurve(mag, magerr, time, config=cfg)

# Check what happened
for stage, n_before, n_after in result['log']:
    print(f"{stage:20s}: {n_before} → {n_after}")
print(f"Passes variability: {result['passes_variability']}")
print(f"Metrics: {result['variability_metrics']}")
```

## All configurable thresholds (with paper defaults)

| Field | Default | Description |
|---|---|---|
| `max_chi` | 10.0 | DoPhot chi quality cut |
| `max_ast_res_chisq` | 20.0 | Astrometric residual chi² cut |
| `remove_ambiguous` | True | Drop blended/neighbour-flagged points |
| `max_magerr` | 0.2 | Hard magnitude error ceiling |
| `sigma_clip_mag` | None | σ-clip mag outliers (None = off) |
| `sigma_clip_magerr` | 4.0 | σ-clip magerr outliers |
| `pair_max_dt` | 0.02 | Paw-print pair max Δt (days) |
| `pair_merr_tol` | 2.0 | Pair mag agreement tolerance |
| `pair_err_ratio` | 2.0 | Max error ratio in a pair |
| `bin_window_hours` | 1.0 | Temporal binning window |
| `bin_min_magerr` | 0.1 | Only bin if mean(magerr) > this |
| `detrend` | True | Enable linear detrending |
| `detrend_bins` | 10 | Bins for weighted-median fit |
| `detrend_min_slope` | 2e-4 | Min slope to subtract (mag/day) |
| `detrend_min_r2` | 0.2 | Min R² to subtract |
| `min_obs_after` | 40 | Reject LC if fewer points remain |
| `var_max_magerr` | 0.5 | Post-clean: max median(magerr) |
| `var_min_amplitude` | 0.1 | Post-clean: min Q99−Q01 (mag) |
| `var_iqr_factor` | 2.0 | Post-clean: IQR > factor × median(magerr) |

Set any `Optional` field to `None` to disable that check.
