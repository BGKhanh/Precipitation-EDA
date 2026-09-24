"""
Models package for DS108 Weather Prediction Project.

Central organization:
- `src.models.nixtla`: Modern Nixtlaverse models (StatsForecast, MLForecast Direct Tweedie, NeuralForecast).
- `src.models.legacy`: Canonical Two-Stage Hurdle baseline models, classical time series, and RecursiveForecaster.
"""

from typing import Dict, List, Any
import pandas as pd

# Subpackages
from . import legacy
from . import nixtla

# Modern Nixtlaverse exports
from .nixtla import (
    BaseNixtlaAdapter,
    StatsForecastAdapter,
    MLForecastAdapter,
    NeuralForecastAdapter,
    to_nixtla_format,
    get_default_stats_models,
    get_default_ml_models,
    get_default_neural_models,
)

# Legacy exports
from .legacy import (
    BaseRainfallModel,
    BaseTimeSeriesModel,
    calculate_metrics,
    evaluate_rainfall_model,
    RandomForestRainfallModel,
    LightGBMRainfallModel,
    XGBoostRainfallModel,
    ARIMAModel,
    SARIMAModel,
    ARIMAXModel,
    SARIMAXModel,
    StationarityTester,
    evaluate_time_series_model,
    prepare_time_series_data,
    RecursiveForecaster,
)

try:
    from .legacy import (
        RNNRainfallModel,
        LSTMRainfallModel,
        GRURainfallModel,
        BiLSTMRainfallModel,
    )
    NEURAL_MODELS_AVAILABLE = True
except (ImportError, AttributeError):
    NEURAL_MODELS_AVAILABLE = False

try:
    from .legacy import LinearRainfallModel
    LINEAR_MODELS_AVAILABLE = True
except (ImportError, AttributeError):
    LINEAR_MODELS_AVAILABLE = False


__all__ = [
    # Subpackages
    'legacy',
    'nixtla',
    # Nixtlaverse
    'BaseNixtlaAdapter',
    'StatsForecastAdapter',
    'MLForecastAdapter',
    'NeuralForecastAdapter',
    'to_nixtla_format',
    'get_default_stats_models',
    'get_default_ml_models',
    'get_default_neural_models',
    # Legacy models
    'BaseRainfallModel',
    'BaseTimeSeriesModel',
    'RandomForestRainfallModel',
    'LightGBMRainfallModel',
    'XGBoostRainfallModel',
    'ARIMAModel',
    'SARIMAModel',
    'ARIMAXModel',
    'SARIMAXModel',
    'StationarityTester',
    'calculate_metrics',
    'evaluate_rainfall_model',
    'evaluate_time_series_model',
    'prepare_time_series_data',
    'RecursiveForecaster',
]

if NEURAL_MODELS_AVAILABLE:
    __all__.extend([
        'RNNRainfallModel',
        'LSTMRainfallModel',
        'GRURainfallModel',
        'BiLSTMRainfallModel',
    ])

if LINEAR_MODELS_AVAILABLE:
    __all__.extend(['LinearRainfallModel'])


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available model families and their models (legacy)."""
    available = {
        'tree_based': [
            'RandomForestRainfallModel',
            'LightGBMRainfallModel',
            'XGBoostRainfallModel',
        ],
        'time_series': [
            'ARIMAModel',
            'SARIMAModel',
            'ARIMAXModel',
            'SARIMAXModel',
        ],
    }

    if NEURAL_MODELS_AVAILABLE:
        available['neural_network'] = [
            'RNNRainfallModel',
            'LSTMRainfallModel',
            'GRURainfallModel',
            'BiLSTMRainfallModel',
        ]

    if LINEAR_MODELS_AVAILABLE:
        available['linear'] = ['LinearRainfallModel']

    return available


def create_model(model_type: str, model_name: str, **kwargs) -> BaseRainfallModel:
    """Factory function to create legacy model instances."""
    available = get_available_models()

    if model_type not in available:
        raise ValueError(f"Model type '{model_type}' not available. Available: {list(available.keys())}")

    if model_name not in available[model_type]:
        raise ValueError(f"Model '{model_name}' not available in '{model_type}'. Available: {available[model_type]}")

    from . import legacy
    model_class = getattr(legacy, model_name)
    return model_class(**kwargs)


def compare_approaches(model: BaseRainfallModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compare two-stage vs single-stage approach for legacy model."""
    if not hasattr(model, 'use_two_stage'):
        raise ValueError("Model must support two-stage configuration")

    results = {
        'model_type': model.__class__.__name__,
        'test_samples': len(y_test),
    }

    if model.use_two_stage:
        two_stage_results = evaluate_rainfall_model(model, X_test, y_test)
        results['two_stage'] = two_stage_results

    return results


def print_model_summary():
    """Print summary of available legacy models and their capabilities."""
    print("DS108 RAINFALL PREDICTION MODELS SUMMARY (LEGACY)")
    print("=" * 60)
    available = get_available_models()
    for family, models in available.items():
        print(f"\n{family.upper().replace('_', ' ')} MODELS:")
        for model in models:
            print(f"   * {model}")