"""
Compatibility shim for RecursiveForecaster.

.. deprecated:: 2.1
   Use `src.models.legacy.recursive.RecursiveForecaster` instead.
"""

import warnings

warnings.warn(
    "src.forecasting.recursive is deprecated. Use src.models.legacy.recursive instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..models.legacy.recursive import RecursiveForecaster

__all__ = ['RecursiveForecaster']
