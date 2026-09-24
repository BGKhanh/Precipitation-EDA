"""Nixtla ecosystem adapters for forecasting."""

from .base_adapter import BaseNixtlaAdapter, to_nixtla_format
from .stats_adapter import StatsForecastAdapter
from .ml_adapter import MLForecastAdapter
from .neural_adapter import NeuralForecastAdapter
from .defaults import (
    get_default_stats_models,
    get_default_ml_models,
    get_default_neural_models,
)

__all__ = [
    "BaseNixtlaAdapter",
    "StatsForecastAdapter",
    "MLForecastAdapter",
    "NeuralForecastAdapter",
    "to_nixtla_format",
    "get_default_stats_models",
    "get_default_ml_models",
    "get_default_neural_models",
]
