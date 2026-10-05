"""Nixtla ecosystem adapters for forecasting."""

from .base_adapter import BaseNixtlaAdapter, to_nixtla_format
from .stats_adapter import StatsForecastAdapter
from .ml_adapter import MLForecastAdapter
from .neural_adapter import NeuralForecastAdapter
from .defaults import (
    get_statistical_models,
    get_eda_arima_model,
    get_eda_sarima_model,
    get_ml_direct_models,
    get_ml_tweedie_models,
    get_ml_hurdle_models,
    get_neural_direct_models,
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
    "get_statistical_models",
    "get_eda_arima_model",
    "get_eda_sarima_model",
    "get_ml_direct_models",
    "get_ml_tweedie_models",
    "get_ml_hurdle_models",
    "get_neural_direct_models",
    "get_default_stats_models",
    "get_default_ml_models",
    "get_default_neural_models",
]
