"""
Periodogram methods for period finding in stochastically sampled time series.

Methods
-------
LS : Lomb-Scargle periodogram (astropy + chi² fallback)
PDM : Phase Dispersion Minimization
CE : Conditional Entropy
GP : Gaussian Process quasi-periodic fitting

Utilities
---------
make_frequency_grid : Construct frequency grids for the PRIMVS 3-test scheme
extract_peaks : Extract top-N peaks from a periodogram
check_alias : Check if a period is a known alias
exclude_alias_regions : Mask alias regions in a periodogram
"""

from stochistats.periodograms.frequency_grid import make_frequency_grid
from stochistats.periodograms.lomb_scargle import LS
from stochistats.periodograms.pdm import PDM
from stochistats.periodograms.conditional_entropy import CE
from stochistats.periodograms.gp import GP
from stochistats.periodograms.peak_analysis import (
    extract_peaks,
    check_alias,
    exclude_alias_regions,
)

__all__ = [
    "LS", "PDM", "CE", "GP",
    "make_frequency_grid",
    "extract_peaks", "check_alias", "exclude_alias_regions",
]
