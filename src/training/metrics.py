"""Point forecast evaluation metrics shim.

Canonical location: ``src.evaluation.metrics``
"""

from src.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_scaled_error,
    compute_dev_train_mase_scale,
    evaluate_point_forecast_metrics,
    CANONICAL_DEV_TRAIN_MASE_SCALE,
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mase,
)

__all__ = [
    'mean_absolute_error',
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_scaled_error',
    'compute_dev_train_mase_scale',
    'evaluate_point_forecast_metrics',
    'CANONICAL_DEV_TRAIN_MASE_SCALE',
    'calc_mae',
    'calc_mse',
    'calc_rmse',
    'calc_mase',
]
