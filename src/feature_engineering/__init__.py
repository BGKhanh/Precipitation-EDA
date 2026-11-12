"""
PEP 8 Standard Alias for src.featurengineering
=============================================
Directs submodule lookups to src.featurengineering.
"""
from pathlib import Path
import src.featurengineering as _orig

# Point __path__ to original featurengineering package directory
__path__ = [str(Path(_orig.__file__).parent)]

# Re-export all symbols
from src.featurengineering import *
