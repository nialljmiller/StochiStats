# How to restructure your StochiStats repo

Run these commands from your local StochiStats clone directory.

## Step 1: Run the restructure script

```bash
cd /path/to/StochiStats
bash restructure.sh
```

This uses `git mv` to move all files into the correct package structure.

## Step 2: Copy in the new files

The restructure script creates the directories but you need to add these new files:

```bash
# Replace the moved __init__.py with the complete version
cp stochistats___init__.py          stochistats/__init__.py

# Add the new periodograms and tests __init__.py files
cp stochistats_periodograms___init__.py  stochistats/periodograms/__init__.py
cp stochistats_tests___init__.py         stochistats/tests/__init__.py

# Replace pyproject.toml with the fixed version
cp pyproject.toml  pyproject.toml
```

## Step 3: Verify internal imports in modules

The files I can't see (variability.py, moments.py, fitting.py, comparison.py,
features.py, and all periodogram files) may use flat imports like:

```python
from variability import Stetson_K    # BROKEN after restructure
```

These need to become:

```python
from stochistats.variability import Stetson_K    # correct
```

Quick check:
```bash
grep -rn "^from [a-z_]* import" stochistats/ --include="*.py" | grep -v "from stochistats" | grep -v "from dataclasses" | grep -v "from typing" | grep -v "from math"
```

If that returns anything, those imports need updating to use `stochistats.` prefix.

## Step 4: Reinstall and test

```bash
pip install -e ".[dev]" --break-system-packages
pytest -v
```

## Step 5: Commit

```bash
git add -A
git commit -m "Restructure into proper Python package layout"
git push
```

## Final structure

```
StochiStats/
├── StochiStats.py                      # Backward-compat shim (stays at root)
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
└── stochistats/
    ├── __init__.py                     # All exports at top level
    ├── variability.py
    ├── moments.py
    ├── fitting.py
    ├── comparison.py
    ├── utils.py
    ├── features.py
    ├── cleaning.py
    ├── periodograms/
    │   ├── __init__.py
    │   ├── frequency_grid.py
    │   ├── lomb_scargle.py
    │   ├── pdm.py
    │   ├── conditional_entropy.py
    │   ├── gp.py
    │   └── peak_analysis.py
    └── tests/
        ├── __init__.py
        ├── test_stochistats.py
        ├── test_cleaning.py
        └── reclass_LC.fits
```
