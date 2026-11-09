"""Official Point Forecast Evaluation Metrics Suite.

Strictly adheres to Protocol v1.0.20 and Model Evaluation Standards:
- Evaluates continuous rainfall amount Y_t in [0, +inf) mm.
- Core 4 Canonical Metrics:
    1. MAE: Mean Absolute Error (optimal for conditional median)
    2. MSE: Mean Squared Error (optimal for conditional mean)
    3. RMSE: Root Mean Squared Error (in mm)
    4. MASE: Mean Absolute Scaled Error (relative to in-sample Naive Lag-1 on Dev Train)
- Levels of Aggregation (§9.3):
    * Pooled (N x H): Primary overall benchmark metric
    * Macro-Average: Unweighted mean across H horizons
    * Horizon Breakdown: h in 1..H
- Inherited Decision — Purged Metrics:
    CR-MAE@H and CR-Bias@H are strictly permanently purged from the project.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error as _sk_mae,
    mean_squared_error as _sk_mse,
    root_mean_squared_error as _sk_rmse,
)

try:
    import utilsforecast.losses as ufl
    NIXTLA_LOSSES_AVAILABLE: bool = True
except ImportError:
    ufl = None
    NIXTLA_LOSSES_AVAILABLE = False

# Canonical in-sample Naive Lag-1 MAE on Dev Train (2001-01-01 to 2018-04-30, T=6,329 days, 6,328 diffs)
CANONICAL_DEV_TRAIN_MASE_SCALE: float = 3.7015424361567634


def compute_dev_train_mase_scale(
    y_dev_train: Union[np.ndarray, pd.Series, Sequence[float]],
    sp: int = 1,
) -> float:
    """Compute scale_DevTrain = (1 / (T - sp)) * sum_{t=sp+1}^T |Y_t - Y_{t-sp}|.

    Strictly computed on the full continuous 1D target series of Dev Train (6,329 days).
    Mirrors sktime.performance_metrics.forecasting.mean_absolute_scaled_error scale computation.
    """
    y_arr = np.asarray(y_dev_train, dtype=np.float64).reshape(-1)
    y_clean = y_arr[np.isfinite(y_arr)]
    if len(y_clean) <= sp:
        raise ValueError(f"Need at least {sp + 1} valid observations to compute MASE scale, got {len(y_clean)}")
    scale = float(np.mean(np.abs(y_clean[sp:] - y_clean[:-sp])))
    if scale <= 1e-12:
        raise ValueError(f"Degenerate MASE scale (near zero): {scale}")
    return scale


def mean_absolute_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """Mean Absolute Error (MAE) via scikit-learn implementation.

    MAE = (1 / M) * sum_{i=1}^M |y_i - y_hat_i|
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return float(np.nan)
    return float(_sk_mae(yt[valid], yp[valid]))


def mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """Mean Squared Error (MSE) via scikit-learn implementation.

    MSE = (1 / M) * sum_{i=1}^M (y_i - y_hat_i)^2
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return float(np.nan)
    return float(_sk_mse(yt[valid], yp[valid]))


def root_mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """Root Mean Squared Error (RMSE) in mm via scikit-learn implementation.

    RMSE = sqrt(MSE)
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return float(np.nan)
    return float(_sk_rmse(yt[valid], yp[valid]))


def mean_absolute_scaled_error(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    y_train: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    sp: int = 1,
    scale: Optional[float] = None,
) -> float:
    """Mean Absolute Scaled Error (MASE) following sktime / Hyndman-Koehler (2006) formulation.

    Formula:
        MASE = MAE(y_true, y_pred) / scale

    where scale is the in-sample seasonal naive MAE:
        scale = (1 / (T - sp)) * sum_{t=sp+1}^T |y_train[t] - y_train[t-sp]|

    Per Protocol v1.0.20 §9.3.A and data-leakage-prevention.md:
    - Default sp = 1 (Lag 1 persistence).
    - If scale is not explicitly given:
        * If y_train is provided: computed via compute_dev_train_mase_scale(y_train, sp=sp).
        * If y_train is None: fixed canonical scale_DevTrain = 3.7015424 mm (precomputed
          once strictly on Dev Train, preventing any test set contamination).
    """
    if scale is None:
        if y_train is not None:
            scale = compute_dev_train_mase_scale(y_train, sp=sp)
        else:
            scale = CANONICAL_DEV_TRAIN_MASE_SCALE

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"scale must be a finite positive number, got {scale}")
    mae = mean_absolute_error(y_true, y_pred)
    if np.isnan(mae):
        return float(np.nan)
    return float(mae / scale)


# Auxiliary diagnostics (secondary diagnostics on wet days, never used for primary ranking)
def rain_day_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rain_threshold: float = 1.0,
) -> float:
    """Mean absolute error computed strictly on rain days (actual >= rain_threshold).

    Returns np.nan if no observations satisfy y_true >= rain_threshold.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt >= rain_threshold
    if not np.any(mask):
        return float(np.nan)
    return float(np.mean(np.abs(yp[mask] - yt[mask])))


def rain_day_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rain_threshold: float = 1.0,
) -> float:
    """Root mean squared error computed strictly on rain days (actual >= rain_threshold).

    Returns np.nan if no observations satisfy y_true >= rain_threshold.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt >= rain_threshold
    if not np.any(mask):
        return float(np.nan)
    return float(np.sqrt(np.mean((yp[mask] - yt[mask]) ** 2)))


# Shorthand aliases
calc_mae = mean_absolute_error
calc_mse = mean_squared_error
calc_rmse = root_mean_squared_error
calc_mase = mean_absolute_scaled_error


def evaluate_point_forecast_metrics(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    scale: float = CANONICAL_DEV_TRAIN_MASE_SCALE,
    horizon: int = 7,
) -> Dict[str, float]:
    """Evaluate full suite of point forecast metrics across Pooled, Macro, and Horizons.

    Args:
        y_true: Array or DataFrame of shape (N, H) or 1D (M,).
        y_pred: Array or DataFrame of shape (N, H) or 1D (M,).
        scale: Fixed Dev Train scale denominator (default: canonical 3.7015424 mm).
        horizon: Forecast horizon H (default 7).

    Returns:
        Dict containing:
        - Pooled metrics: mae_pooled, mse_pooled, rmse_pooled, mase_lag1_pooled
        - Aliases: MAE, MSE, RMSE, MASE
        - Macro metrics: mae_macro, mse_macro, rmse_macro, mase_lag1_macro
        - Per-horizon breakdowns: mae_h{h}, mse_h{h}, rmse_h{h}, mase_lag1_h{h}
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    # Ensure 2D (N, H)
    if yt.ndim == 1:
        if len(yt) % horizon == 0:
            yt = yt.reshape(-1, horizon)
            yp = yp.reshape(-1, horizon)
        else:
            yt = yt.reshape(-1, 1)
            yp = yp.reshape(-1, 1)

    n_samples, H = yt.shape

    # 1. Pooled Overall Metrics (N x H)
    mae_pooled = mean_absolute_error(yt, yp)
    mse_pooled = mean_squared_error(yt, yp)
    rmse_pooled = root_mean_squared_error(yt, yp)
    mase_pooled = mean_absolute_scaled_error(yt, yp, scale=scale)

    results: Dict[str, float] = {
        "mae_pooled": mae_pooled,
        "mse_pooled": mse_pooled,
        "rmse_pooled": rmse_pooled,
        "mase_lag1_pooled": mase_pooled,
        # Standard table aliases
        "MAE": mae_pooled,
        "MSE": mse_pooled,
        "RMSE": rmse_pooled,
        "MASE": mase_pooled,
    }

    # 2. Per-Horizon Breakdown
    h_maes: List[float] = []
    h_mses: List[float] = []
    h_rmses: List[float] = []
    h_mases: List[float] = []

    for h_idx in range(H):
        h_num = h_idx + 1
        yt_h = yt[:, h_idx]
        yp_h = yp[:, h_idx]

        mae_h = mean_absolute_error(yt_h, yp_h)
        mse_h = mean_squared_error(yt_h, yp_h)
        rmse_h = root_mean_squared_error(yt_h, yp_h)
        mase_h = mean_absolute_scaled_error(yt_h, yp_h, scale=scale)

        results[f"mae_h{h_num}"] = mae_h
        results[f"mse_h{h_num}"] = mse_h
        results[f"rmse_h{h_num}"] = rmse_h
        results[f"mase_lag1_h{h_num}"] = mase_h

        h_maes.append(mae_h)
        h_mses.append(mse_h)
        h_rmses.append(rmse_h)
        h_mases.append(mase_h)

    # 3. Macro-Average Metrics across H
    results["mae_macro"] = float(np.nanmean(h_maes)) if h_maes else float(np.nan)
    results["mse_macro"] = float(np.nanmean(h_mses)) if h_mses else float(np.nan)
    results["rmse_macro"] = float(np.nanmean(h_rmses)) if h_rmses else float(np.nan)
    results["mase_lag1_macro"] = float(np.nanmean(h_mases)) if h_mases else float(np.nan)

    return results


def evaluate_nixtla_dataframe_losses(
    df: pd.DataFrame,
    models: Sequence[str],
    train_df: Optional[pd.DataFrame] = None,
    seasonality: int = 1,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "ds",
) -> Dict[str, pd.DataFrame]:
    """Evaluate point forecast metrics directly using Nixtla's native utilsforecast.losses.

    Directly leverages Nixtla's vectorized DataFrame operations across multiple model columns.
    Conforms to Nixtla ecosystem standard (StatsForecast, MLForecast, NeuralForecast).

    Args:
        df: Input DataFrame containing actuals and predictions.
        models: List of model prediction column names.
        train_df: Optional training DataFrame for in-sample MASE scaling.
        seasonality: Seasonal period for MASE (default 1 for daily lag-1).
        id_col: Series identifier column.
        target_col: Target column.
        cutoff_col: Forecast origin/cutoff column.
        time_col: Timestamp column.

    Returns:
        Dict mapping metric names ('mae', 'mse', 'rmse', 'mase') to their respective DataFrames.
    """
    if not NIXTLA_LOSSES_AVAILABLE or ufl is None:
        raise ImportError("utilsforecast.losses is not available in the current environment.")

    model_list = list(models)
    results: Dict[str, pd.DataFrame] = {
        "mae": ufl.mae(df, models=model_list, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col),
        "mse": ufl.mse(df, models=model_list, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col),
        "rmse": ufl.rmse(df, models=model_list, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col),
    }

    if train_df is not None:
        results["mase"] = ufl.mase(
            df,
            models=model_list,
            seasonality=seasonality,
            train_df=train_df,
            id_col=id_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
            time_col=time_col,
        )

    return results

