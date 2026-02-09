"""
Backwards-compatibility shim.

Allows existing code that does ``import StochiStats`` or
``from StochiStats import Eta, ...`` to keep working without changes.

All symbols are re-exported from the canonical ``stochistats`` package.
"""

from stochistats import *  # noqa: F401,F403
from stochistats import __version__, __all__  # noqa: F401
