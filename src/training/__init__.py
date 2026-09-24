"""Training and validation utilities for rainfall prediction."""

from .validator import RainfallCrossValidator, UnifiedBenchmark, UnifiedBenchmarkRunner
from .optimizer import (
    ModelOptimizer,
    get_lightgbm_tweedie_config,
    get_xgboost_tweedie_config,
    get_neural_tweedie_config,
)
from .trainer import (
    RainfallTrainer,
    build_two_stage_from_eda,
    build_sarima_from_eda,
    build_stats_adapter_from_eda,
    build_ml_adapter_from_eda,
    build_neural_adapter_from_eda,
)

__all__ = [
    'RainfallCrossValidator',
    'UnifiedBenchmark',
    'UnifiedBenchmarkRunner',
    'ModelOptimizer',
    'get_lightgbm_tweedie_config',
    'get_xgboost_tweedie_config',
    'get_neural_tweedie_config',
    'RainfallTrainer',
    'build_two_stage_from_eda',
    'build_sarima_from_eda',
    'build_stats_adapter_from_eda',
    'build_ml_adapter_from_eda',
    'build_neural_adapter_from_eda',
]