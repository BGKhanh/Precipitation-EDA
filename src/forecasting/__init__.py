"""
Forecasting package for DS108 Weather Prediction Project.

.. deprecated:: 2.1
   `src.forecasting` is deprecated as part of the architecture consolidation.
   - For Nixtlaverse models (StatsForecast, MLForecast, NeuralForecast), use `src.models.nixtla`.
   - For recursive rollout of legacy two-stage models, use `src.models.legacy.recursive`.
"""

import warnings

warnings.warn(
    "src.forecasting is deprecated. Use src.models.nixtla for Nixtlaverse models "
    "or src.models.legacy.recursive for RecursiveForecaster.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-exports for backward compatibility
from ..models.legacy.recursive import RecursiveForecaster
from ..models.nixtla import (
    BaseNixtlaAdapter,
    StatsForecastAdapter,
    MLForecastAdapter,
    NeuralForecastAdapter,
    to_nixtla_format,
    get_default_stats_models,
    get_default_ml_models,
    get_default_neural_models,
)
from ..models import nixtla

__all__ = [
    'RecursiveForecaster',
    'BaseNixtlaAdapter',
    'StatsForecastAdapter',
    'MLForecastAdapter',
    'NeuralForecastAdapter',
    'to_nixtla_format',
    'get_default_stats_models',
    'get_default_ml_models',
    'get_default_neural_models',
    'nixtla',
]
