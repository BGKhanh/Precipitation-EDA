"""Evaluation and benchmarking module for rainfall prediction.

Submodules:
- metrics: Cumulative rainfall metrics (CR-MAE@H, CR-Bias@H) and rain-day metrics.
- protocols: Canonical split, Horizon Purge, hardware detection, resource tracking, failure protocol, contracts.
- benchmark: UnifiedBenchmark engine for multi-family model comparisons.
"""

from .metrics import (
    cumulative_rainfall_mae,
    cumulative_rainfall_bias,
    evaluate_cumulative_rainfall,
    rain_day_mae,
    rain_day_rmse,
)

from .protocols import (
    UNIFIED_SCHEMA,
    ExecutionStatus,
    FutureCovariatePolicy,
    GENUINELY_KNOWN_FUTURE_COLUMNS,
    get_hardware_metadata,
    measure_execution_resource,
    load_canonical_data,
    load_full_refit_data,
    build_purged_direct_samples,
    compute_benchmark_metrics,
    save_benchmark_artifacts,
    get_default_contract,
    save_modeling_contract,
    load_modeling_contract,
    FORECAST_HORIZON,
    CONTEXT_LENGTH,
    SEASON_LENGTH,
    RAIN_THRESHOLD,
    TARGET_COL,
    DATE_COL,
)

from .benchmark import (
    UnifiedBenchmark,
    UnifiedBenchmarkRunner,
)

__all__ = [
    # Metrics
    'cumulative_rainfall_mae',
    'cumulative_rainfall_bias',
    'evaluate_cumulative_rainfall',
    'rain_day_mae',
    'rain_day_rmse',
    # Protocols & Contracts
    'UNIFIED_SCHEMA',
    'ExecutionStatus',
    'FutureCovariatePolicy',
    'GENUINELY_KNOWN_FUTURE_COLUMNS',
    'get_hardware_metadata',
    'measure_execution_resource',
    'load_canonical_data',
    'load_full_refit_data',
    'build_purged_direct_samples',
    'compute_benchmark_metrics',
    'save_benchmark_artifacts',
    'get_default_contract',
    'save_modeling_contract',
    'load_modeling_contract',
    'FORECAST_HORIZON',
    'CONTEXT_LENGTH',
    'SEASON_LENGTH',
    'RAIN_THRESHOLD',
    'TARGET_COL',
    'DATE_COL',
    # Benchmarks
    'UnifiedBenchmark',
    'UnifiedBenchmarkRunner',
]
