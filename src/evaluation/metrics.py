"""Cumulative and point-wise rainfall evaluation metrics.

This module provides specialized metrics for rainfall forecasting:
- CR-MAE@H: Cumulative Rainfall Mean Absolute Error at Horizon H
- CR-Bias@H: Cumulative Rainfall Bias (Signed Cumulative Error) at Horizon H
- rain_day_mae / rain_day_rmse: Regression metrics evaluated on rain days only (threshold >= 0.1 mm)
"""

from typing import Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd


def cumulative_rainfall_mae(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    horizon: Optional[int] = None,
) -> float:
    """Calculate Cumulative Rainfall MAE at Horizon H (CR-MAE@H).

    Measures the mean absolute error between actual accumulated rainfall and
    predicted accumulated rainfall over an H-step forecast horizon across N valid
    forecast origins:

        CR-MAE@H = (1/N) * sum_{i=1}^N | sum_{h=1}^H y_hat_{i,h} - sum_{h=1}^H y_{i,h} |

    IMPORTANT:
    This measures the error of accumulated rainfall within the entire window.
    It does NOT measure the sum of daily absolute errors.

    Args:
        y_true: Actual rainfall. Shape (N, H) or (H,).
        y_pred: Predicted rainfall. Shape (N, H) or (H,).
        horizon: Optional expected horizon H to validate input shape.

    Returns:
        CR-MAE@H in the same unit as rainfall target (mm).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)

    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shapes, got {yt.shape} vs {yp.shape}"
        )

    H = yt.shape[1]
    if horizon is not None and horizon != H:
        raise ValueError(f"Configured horizon {horizon} does not match input shape {H}")

    # Keep only complete valid forecast origins
    valid_mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]

    if len(yt_valid) == 0:
        return np.nan

    # Accumulated rainfall for each forecast origin
    actual_total = yt_valid.sum(axis=1)
    forecast_total = yp_valid.sum(axis=1)

    # Absolute error of accumulated rainfall
    cumulative_abs_error = np.abs(forecast_total - actual_total)

    return float(cumulative_abs_error.mean())


def cumulative_rainfall_bias(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    horizon: Optional[int] = None,
) -> float:
    """Calculate Cumulative Rainfall Bias at Horizon H (CR-Bias@H).

    Measures the signed error of total accumulated rainfall:
        CR-Bias > 0: over-forecasting cumulative rain
        CR-Bias < 0: under-forecasting cumulative rain

        CR-Bias@H = (1/N) * sum_{i=1}^N ( sum_{h=1}^H y_hat_{i,h} - sum_{h=1}^H y_{i,h} )

    Args:
        y_true: Actual rainfall. Shape (N, H) or (H,).
        y_pred: Predicted rainfall. Shape (N, H) or (H,).
        horizon: Optional expected horizon H to validate input shape.

    Returns:
        CR-Bias@H in mm.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)

    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shapes, got {yt.shape} vs {yp.shape}"
        )

    H = yt.shape[1]
    if horizon is not None and horizon != H:
        raise ValueError(f"Configured horizon {horizon} does not match input shape {H}")

    valid_mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]

    if len(yt_valid) == 0:
        return np.nan

    actual_total = yt_valid.sum(axis=1)
    forecast_total = yp_valid.sum(axis=1)
    cumulative_error = forecast_total - actual_total

    return float(cumulative_error.mean())


def evaluate_cumulative_rainfall(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate cumulative metrics across multiple horizons (e.g. H=1, 3, 5, 7)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)

    max_H = yt.shape[1]
    eval_horizons = horizons or [h for h in [1, 3, 5, 7] if h <= max_H]
    results = {}

    for h in eval_horizons:
        results[f"CR_MAE_{h}"] = round(cumulative_rainfall_mae(yt[:, :h], yp[:, :h], horizon=h), 4)
        results[f"CR_Bias_{h}"] = round(cumulative_rainfall_bias(yt[:, :h], yp[:, :h], horizon=h), 4)

    return results


def rain_day_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rain_threshold: float = 0.1,
) -> float:
    """Mean absolute error computed strictly on rain days (actual >= rain_threshold)."""
    mask = y_true >= rain_threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def rain_day_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rain_threshold: float = 0.1,
) -> float:
    """Root mean squared error computed strictly on rain days (actual >= rain_threshold)."""
    mask = y_true >= rain_threshold
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
