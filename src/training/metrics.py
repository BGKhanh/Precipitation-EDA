"""Cumulative rainfall evaluation metrics (Backward Compatibility Shim).

Canonical location: ``src.evaluation.metrics``
"""

from src.evaluation.metrics import (
    cumulative_rainfall_mae,
    cumulative_rainfall_bias,
    evaluate_cumulative_rainfall,
    rain_day_mae,
    rain_day_rmse,
)

__all__ = [
    'cumulative_rainfall_mae',
    'cumulative_rainfall_bias',
    'evaluate_cumulative_rainfall',
    'rain_day_mae',
    'rain_day_rmse',
]
