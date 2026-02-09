"""
Period-finding methods for stochastically sampled time series.

Implements the four period-finding techniques used in the PRIMVS pipeline:
    - Lomb-Scargle (LS)
    - Phase Dispersion Minimization (PDM)
    - Conditional Entropy (CE)
    - Gaussian Process (GP)

Plus utilities for frequency grid construction, peak extraction,
and alias checking.
"""

from stochistats.periodograms.frequency_grid import make_frequency_grid
from stochistats.periodograms.lomb_scargle import lomb_scargle, LS
from stochistats.periodograms.pdm import pdm, PDM
from stochistats.periodograms.conditional_entropy import conditional_entropy, CE
from stochistats.periodograms.gp import gp_period, GP
from stochistats.periodograms.peak_analysis import (
    extract_peaks,
    check_alias,
    exclude_alias_regions,
)

__all__ = [
    "make_frequency_grid",
    "lomb_scargle", "LS",
    "pdm", "PDM",
    "conditional_entropy", "CE",
    "gp_period", "GP",
    "extract_peaks",
    "check_alias",
    "exclude_alias_regions",
]
