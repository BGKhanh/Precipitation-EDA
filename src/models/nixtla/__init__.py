"""
Nixtla Ecosystem Wrappers & Canonical Defaults
==============================================
Provides production wrappers for StatsForecast and NeuralForecast:
- StatsForecastWrapper: Intermittent & statistical models
- FrozenIntermittentWrapper: Croston/ADIDA/IMAPA/TSB wrappers
- defaults: Canonical model registries and hyperparameters
"""

from .wrappers import StatsForecastWrapper, FrozenIntermittentWrapper
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
    "StatsForecastWrapper",
    "FrozenIntermittentWrapper",
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
