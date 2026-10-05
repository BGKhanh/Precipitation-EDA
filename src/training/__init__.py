"""Training and validation utilities for rainfall prediction."""

from .validator import RainfallCrossValidator, UnifiedBenchmark, UnifiedBenchmarkRunner
from .optimizer import (
    ModelOptimizer,
    get_lightgbm_tweedie_config,
    get_xgboost_tweedie_config,
    get_neural_tweedie_config,
    tune_hurdle_model,
)
from .trainer import (
    RainfallTrainer,
    build_two_stage_from_eda,
    build_sarima_from_eda,
    build_stats_adapter_from_eda,
    build_ml_adapter_from_eda,
    build_neural_adapter_from_eda,
    build_ml_direct_adapter_from_eda,
    build_ml_tweedie_adapter_from_eda,
    build_neural_direct_adapter_from_eda,
)
from .metrics import (
    cumulative_rainfall_mae,
    cumulative_rainfall_bias,
    evaluate_cumulative_rainfall,
)
from .benchmark_utils import (
    UNIFIED_SCHEMA,
    get_default_contract,
    save_modeling_contract,
    load_modeling_contract,
    load_canonical_data,
    build_purged_direct_samples,
    compute_benchmark_metrics,
    save_benchmark_artifacts,
)

__all__ = [
    'RainfallCrossValidator',
    'UnifiedBenchmark',
    'UnifiedBenchmarkRunner',
    'ModelOptimizer',
    'get_lightgbm_tweedie_config',
    'get_xgboost_tweedie_config',
    'get_neural_tweedie_config',
    'tune_hurdle_model',
    'RainfallTrainer',
    'build_two_stage_from_eda',
    'build_sarima_from_eda',
    'build_stats_adapter_from_eda',
    'build_ml_adapter_from_eda',
    'build_neural_adapter_from_eda',
    'build_ml_direct_adapter_from_eda',
    'build_ml_tweedie_adapter_from_eda',
    'build_neural_direct_adapter_from_eda',
    'cumulative_rainfall_mae',
    'cumulative_rainfall_bias',
    'evaluate_cumulative_rainfall',
    'UNIFIED_SCHEMA',
    'get_default_contract',
    'save_modeling_contract',
    'load_modeling_contract',
    'load_canonical_data',
    'build_purged_direct_samples',
    'compute_benchmark_metrics',
    'save_benchmark_artifacts',
]