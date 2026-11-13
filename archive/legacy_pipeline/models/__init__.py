"""Legacy two-stage and classical models for DS108 Weather Prediction Project."""

from typing import Dict, List, Any
import pandas as pd

from .base import (
    BaseRainfallModel,
    BaseTimeSeriesModel,
    calculate_metrics,
    evaluate_rainfall_model,
)
from .tree_models import (
    RandomForestRainfallModel,
    LightGBMRainfallModel,
    XGBoostRainfallModel,
)

try:
    from .neural_models import (
        RNNRainfallModel,
        LSTMRainfallModel,
        GRURainfallModel,
        BiLSTMRainfallModel,
    )
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False

try:
    from .linear_models import LinearRainfallModel
    LINEAR_MODELS_AVAILABLE = True
except ImportError:
    LINEAR_MODELS_AVAILABLE = False

from .time_series import (
    ARIMAModel,
    SARIMAModel,
    ARIMAXModel,
    SARIMAXModel,
    StationarityTester,
    evaluate_time_series_model,
    prepare_time_series_data,
)
from .recursive import RecursiveForecaster

__all__ = [
    "BaseRainfallModel",
    "BaseTimeSeriesModel",
    "RandomForestRainfallModel",
    "LightGBMRainfallModel",
    "XGBoostRainfallModel",
    "ARIMAModel",
    "SARIMAModel",
    "ARIMAXModel",
    "SARIMAXModel",
    "StationarityTester",
    "calculate_metrics",
    "evaluate_rainfall_model",
    "evaluate_time_series_model",
    "prepare_time_series_data",
    "RecursiveForecaster",
]

if NEURAL_MODELS_AVAILABLE:
    __all__.extend([
        "RNNRainfallModel",
        "LSTMRainfallModel",
        "GRURainfallModel",
        "BiLSTMRainfallModel",
    ])

if LINEAR_MODELS_AVAILABLE:
    __all__.extend(["LinearRainfallModel"])
