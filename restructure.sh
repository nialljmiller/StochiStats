#!/bin/bash
# =============================================================================
# StochiStats repo restructure: flat → proper Python package layout
#
# Run this from the root of your local StochiStats clone:
#   cd /path/to/StochiStats
#   bash restructure.sh
#
# What it does:
#   1. Creates stochistats/, stochistats/periodograms/, stochistats/tests/
#   2. git mv's every module into the correct subdirectory
#   3. Creates new __init__.py files for periodograms/ and tests/
#   4. Leaves root-level files (StochiStats.py, pyproject.toml, etc.) in place
# =============================================================================

set -e  # exit on any error

echo "=== Creating package directories ==="
mkdir -p stochistats/periodograms
mkdir -p stochistats/tests

# ── 1. Move main package modules into stochistats/ ──
echo "=== Moving core modules ==="
git mv __init__.py       stochistats/__init__.py
git mv variability.py    stochistats/variability.py
git mv moments.py        stochistats/moments.py
git mv fitting.py        stochistats/fitting.py
git mv comparison.py     stochistats/comparison.py
git mv utils.py          stochistats/utils.py
git mv features.py       stochistats/features.py
git mv cleaning.py       stochistats/cleaning.py

# ── 2. Move periodogram modules into stochistats/periodograms/ ──
echo "=== Moving periodogram modules ==="
git mv lomb_scargle.py        stochistats/periodograms/lomb_scargle.py
git mv pdm.py                 stochistats/periodograms/pdm.py
git mv conditional_entropy.py stochistats/periodograms/conditional_entropy.py
git mv gp.py                  stochistats/periodograms/gp.py
git mv frequency_grid.py      stochistats/periodograms/frequency_grid.py
git mv peak_analysis.py       stochistats/periodograms/peak_analysis.py

# ── 3. Move test files into stochistats/tests/ ──
echo "=== Moving test files ==="
git mv test_stochistats.py  stochistats/tests/test_stochistats.py
git mv test_cleaning.py     stochistats/tests/test_cleaning.py

# ── 4. Move test data if present ──
if [ -f reclass_LC.fits ]; then
    echo "=== Moving test data ==="
    git mv reclass_LC.fits stochistats/tests/reclass_LC.fits
fi

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Copy the new __init__.py files from the output:"
echo "     - stochistats/periodograms/__init__.py  (provided separately)"
echo "     - stochistats/tests/__init__.py          (provided separately)"
echo "  2. Review stochistats/__init__.py imports (may need updating)"
echo "  3. Reinstall:  pip install -e . --break-system-packages"
echo "  4. Run tests:  pytest"
echo "  5. Commit:     git add -A && git commit -m 'Restructure into proper package layout'"
