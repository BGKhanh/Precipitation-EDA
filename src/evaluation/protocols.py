"""Evaluation protocols, split contracts, and artifact schemas.

Ensures strict compliance with:
- data-leakage-prevention.md (fit/transform, shift(1), calendar diffs)
- evaluation-standards.md (naive baselines, horizon breakdowns, rain-day metrics)
- P0.1: Horizon Purge Protocol (t + H <= T_split_end; exactly H=7 origins purged per split)
- P0.2: Zero Test Contamination (Dev Train for fit/selection/HPO, Full Train for final refit)
- P0.3: Future Covariate Contract (historical meteo + genuinely known calendar features only)
- P0.4: Seasonal period clarification (s=7 is weekly; annual monsoon is regime)
- P0.5: Failure Protocol (SUCCESS, FAILED_CONVERGENCE, FAILED_NUMERICAL, FAILED_RESOURCE)
- Metric Contract: CR-MAE@H and CR-Bias@H
- Artifact Contract: Unified schema across all modeling notebooks
"""

import os
import json
import time
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable
import numpy as np
import pandas as pd
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.loader import DataLoader, time_series_split
from .metrics import (
    cumulative_rainfall_mae,
    cumulative_rainfall_bias,
    evaluate_cumulative_rainfall,
)

REPORTS_DIR = Path("reports")
RESULTS_DIR = REPORTS_DIR / "modeling" / "benchmarks"
CONTRACT_PATH = REPORTS_DIR / "modeling" / "modeling_contract.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda" / "eda_report.json"

UNIFIED_SCHEMA = [
    "model",
    "group",
    "strategy",
    "status",
    "MAE",
    "RMSE",
    "rain_day_MAE",
    "rain_day_RMSE",
    "CR-MAE@1",
    "CR-MAE@3",
    "CR-MAE@5",
    "CR-MAE@7",
    "CR-Bias@7",
    "training_time",
    "training_time_std",
    "inference_time",
    "inference_time_std",
    "cpu_peak_ram_mb",
    "gpu_peak_vram_mb",
    "parameter_count",
    "failure_reason",
    "exception_type",
]


# ======================================================================
# Enums for Contracts
# ======================================================================

class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_CONVERGENCE = "FAILED_CONVERGENCE"
    FAILED_NUMERICAL = "FAILED_NUMERICAL"
    FAILED_RESOURCE = "FAILED_RESOURCE"


class FutureCovariatePolicy(str, Enum):
    HISTORICAL = "HISTORICAL"
    KNOWN_FUTURE = "KNOWN_FUTURE"
    REQUIRES_EXTERNAL_FORECAST = "REQUIRES_EXTERNAL_FORECAST"


# Genuinely known calendar and cyclical features
GENUINELY_KNOWN_FUTURE_COLUMNS = [
    "Month_sin",
    "Month_cos",
    "DayOfYear_sin",
    "DayOfYear_cos",
    "Is_Wet_Season",
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "DayOfYear",
]


# ======================================================================
# Dynamic Hardware Detection
# ======================================================================

def get_hardware_metadata() -> Dict[str, Any]:
    """Dynamically detect runtime hardware specifications without hardcoding."""
    return {
        "cpu_processor": platform.processor() or "Unknown",
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "system_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": HAS_TORCH and torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if HAS_TORCH and torch.cuda.is_available() else "None",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if HAS_TORCH and torch.cuda.is_available() else 0.0,
        "cuda_version": torch.version.cuda if HAS_TORCH and torch.cuda.is_available() else "None",
        "platform": platform.platform(),
    }


# ======================================================================
# Resource Measurement Protocol
# ======================================================================

def measure_execution_resource(
    fn: Callable[[], Any],
    warm_up_fn: Optional[Callable[[], Any]] = None,
    repetitions: int = 3,
    is_gpu: bool = False,
) -> Dict[str, Any]:
    """Measure wall-clock time and peak memory with family-specific warm-up.

    - Repetitions: N >= 3, reports mean ± std
    - GPU synchronization before and after timing if CUDA available
    - Tracks CPU peak RAM and GPU peak VRAM
    """
    import tracemalloc

    # 1. Family-specific warm-up
    if warm_up_fn is not None:
        try:
            warm_up_fn()
            if is_gpu and HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    times: List[float] = []
    last_result: Any = None
    peak_cpu_ram: float = 0.0

    for _ in range(max(1, repetitions)):
        if is_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        tracemalloc.start()
        t0 = time.perf_counter()
        
        res = fn()
        
        if is_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        last_result = res
        if peak_traced > peak_cpu_ram:
            peak_cpu_ram = peak_traced

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    cpu_peak_mb = round(peak_cpu_ram / (1024 ** 2), 2)
    gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if is_gpu and HAS_TORCH and torch.cuda.is_available() else 0.0

    return {
        "result": last_result,
        "time_mean_sec": round(mean_time, 4),
        "time_std_sec": round(std_time, 4),
        "time_raw_runs": [round(t, 4) for t in times],
        "cpu_peak_ram_mb": cpu_peak_mb,
        "gpu_peak_vram_mb": gpu_peak_mb,
    }


# ======================================================================
# Modeling Contract
# ======================================================================

def get_default_contract() -> Dict[str, Any]:
    """Return standard project-wide modeling contract specifications."""
    return {
        "canonical_split": {
            "dev_train": {
                "start": "2001-01-01",
                "end": "2018-04-30",
                "days": 6329,
                "valid_origins": 6322,
                "purged_origins": 7,
            },
            "validation": {
                "start": "2018-05-01",
                "end": "2020-04-30",
                "days": 731,
                "valid_origins": 724,
                "purged_origins": 7,
            },
            "final_test": {
                "start": "2020-05-01",
                "end": "2026-04-30",
                "days": 2191,
                "valid_origins": 2184,
                "purged_origins": 7,
            },
            "full_train_refit": {
                "start": "2001-01-01",
                "end": "2020-04-30",
                "days": 7060,
            }
        },
        "forecast_horizon": 7,  # days
        "context_length": 30,  # days
        "season_length": 7,  # weekly seasonality (lag 7)
        "target_col": "Lượng mưa",
        "date_col": "Ngày",
        "rain_threshold": 0.1,  # mm (WMO trace standard)
        "horizon_purge_policy": "Strict Origin Purge: Origin t must satisfy t + H <= T_split_end. Exactly H=7 origins purged at end of each split.",
        "fit_boundary_policy": "Strict Zero Test Contamination: Dev Train ONLY for feature selection, scaling, and HPO. Full Train for final model refit.",
        "benchmark_a_policy": {
            "name": "Scientific Feature Construction Ablation",
            "construction": "F0 -> F1 -> F2 -> F3",
            "selection": "None (all generated features fed to model)",
            "transformation": {
                "rule": "Fixed per model family across all F0->F3 levels",
                "LightGBM": "None",
                "SARIMAX": "StandardScaler(fit on Dev Train only)",
                "PatchTST": "StandardScaler(fit on Dev Train only)"
            },
            "failure_protocol": "Records status (SUCCESS, FAILED_CONVERGENCE, FAILED_NUMERICAL, FAILED_RESOURCE), failure_reason, exception_type without pre-emptive causal claims"
        },
        "benchmark_b_policy": {
            "name": "Model-Optimized Pipeline",
            "selection": {
                "rule": "Fit strictly on Dev Train only",
                "SARIMAX": "VIF < 10 (fit on Dev Train only)",
                "LightGBM": "Tree Feature Importance threshold (fit on Dev Train only)"
            },
            "transformation": "Model-specific transformations (fit on Dev Train only)",
            "protocol_sequence": "Dev Train only -> fit Selection/Scaler/HPO -> Validation -> Freeze policy -> Refit Dev+Val -> Final Test"
        },
        "future_covariate_policy": "No future ground-truth meteorology permitted. Only historical lags/predictions and genuinely known calendar cyclical features allowed.",
        "hardware_metadata": get_hardware_metadata(),
        "schema": UNIFIED_SCHEMA,
    }


FORECAST_HORIZON = 7
CONTEXT_LENGTH = 30
SEASON_LENGTH = 7
RAIN_THRESHOLD = 0.1
TARGET_COL = "Lượng mưa"
DATE_COL = "Ngày"


def save_modeling_contract(contract: Optional[Dict[str, Any]] = None, filepath: Path = CONTRACT_PATH) -> None:
    """Save the formal modeling contract JSON (optional utility)."""
    c = contract or get_default_contract()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2, ensure_ascii=False)


def load_modeling_contract(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Load the modeling contract specification in-memory (or from file if explicitly provided and exists)."""
    if filepath is not None and filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return get_default_contract()


# ======================================================================
# Data Loading & Partitioning
# ======================================================================

def load_canonical_data(
    return_dev_train: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load clean weather data (9,251 rows x 36 cols, 2001-2026) and partition into
    non-overlapping temporal splits:

    If return_dev_train is True:
        Returns (dev_train, val_set, test_set):
        - dev_train:  2001-01-01 to 2018-04-30 (6,329 days, 6,322 valid origins)
        - val_set:    2018-05-01 to 2020-04-30 (731 days, 724 valid origins)
        - test_set:   2020-05-01 to 2026-04-30 (2,191 days, 2,184 valid origins)
    If return_dev_train is False:
        Returns (train_full, val_set, test_set):
        - train_full: 2001-01-01 to 2020-04-30 (7,060 days, Dev Train + Validation container for final refit)
    """
    df = DataLoader().load_data()
    train_full, test_set = time_series_split(df, date_col="Ngày")
    
    val_mask = train_full["Ngày"] >= "2018-05-01"
    dev_train = train_full[~val_mask].copy()
    val_set = train_full[val_mask].copy()
    
    if return_dev_train:
        return dev_train, val_set, test_set
    return train_full, val_set, test_set


def load_full_refit_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_full, test_set) for the final refit stage (7,060 train / 2,191 test)."""
    df = DataLoader().load_data()
    train_full, test_set = time_series_split(df, date_col="Ngày")
    return train_full, test_set


# ======================================================================
# Direct Samples with Horizon Purge
# ======================================================================

def build_purged_direct_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Lượng mưa",
    horizon: int = 7,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Build multi-horizon direct regression samples (X_t -> [y_{t+1}, ..., y_{t+H}]).

    Enforces Horizon Purge: Samples where t + H > len(df) are strictly purged
    (exactly H origins discarded at the end of df), guaranteeing zero boundary
    bleed across splits.
    """
    n = len(df)
    max_origin = n - horizon
    if max_origin <= 0:
        return np.empty((0, len(feature_cols))), np.empty((0, horizon)), pd.Series(dtype="object")
    
    X_mat = df[feature_cols].values
    y_vec = df[target_col].values
    dates = df["Ngày"] if "Ngày" in df.columns else pd.Series(range(n))
    
    X_samples = []
    Y_samples = []
    origin_dates = []
    
    for i in range(max_origin):
        X_samples.append(X_mat[i])
        Y_samples.append(y_vec[i + 1 : i + 1 + horizon])
        origin_dates.append(dates.iloc[i])
        
    return np.array(X_samples), np.array(Y_samples), pd.Series(origin_dates)


# ======================================================================
# Metrics & Schema Evaluation
# ======================================================================

def compute_benchmark_metrics(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "Unknown",
    group: str = "Unknown",
    strategy: str = "direct",
    training_time: float = 0.0,
    training_time_std: float = 0.0,
    inference_time: float = 0.0,
    inference_time_std: float = 0.0,
    cpu_peak_ram_mb: float = 0.0,
    gpu_peak_vram_mb: float = 0.0,
    parameter_count: int = 0,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    failure_reason: str = "",
    exception_type: str = "",
    rain_threshold: float = 0.1,
) -> pd.DataFrame:
    """Compute benchmark metrics conforming to the unified schema.
    Handles both successful and failed execution records.
    """
    if status != ExecutionStatus.SUCCESS or y_true is None or y_pred is None:
        record = {
            "model": model_name,
            "group": group,
            "strategy": strategy,
            "status": str(status.value if hasattr(status, 'value') else status),
            "MAE": np.nan,
            "RMSE": np.nan,
            "rain_day_MAE": np.nan,
            "rain_day_RMSE": np.nan,
            "CR-MAE@1": np.nan,
            "CR-MAE@3": np.nan,
            "CR-MAE@5": np.nan,
            "CR-MAE@7": np.nan,
            "CR-Bias@7": np.nan,
            "training_time": round(float(training_time), 4),
            "training_time_std": round(float(training_time_std), 4),
            "inference_time": round(float(inference_time), 4),
            "inference_time_std": round(float(inference_time_std), 4),
            "cpu_peak_ram_mb": round(float(cpu_peak_ram_mb), 2),
            "gpu_peak_vram_mb": round(float(gpu_peak_vram_mb), 2),
            "parameter_count": int(parameter_count),
            "failure_reason": str(failure_reason),
            "exception_type": str(exception_type),
        }
        return pd.DataFrame([record])[UNIFIED_SCHEMA]

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    
    # Clip negative forecasts
    yp = np.clip(yp, 0, None)
    
    # Flatten for point-wise metrics
    yt_flat = yt.flatten()
    yp_flat = yp.flatten()
    
    mae = float(np.mean(np.abs(yp_flat - yt_flat)))
    rmse = float(np.sqrt(np.mean((yp_flat - yt_flat) ** 2)))
    
    # Rain-day metrics (actual rain > threshold)
    rain_mask = yt_flat > rain_threshold
    if np.any(rain_mask):
        rain_day_mae = float(np.mean(np.abs(yp_flat[rain_mask] - yt_flat[rain_mask])))
        rain_day_rmse = float(np.sqrt(np.mean((yp_flat[rain_mask] - yt_flat[rain_mask]) ** 2)))
    else:
        rain_day_mae = np.nan
        rain_day_rmse = np.nan
        
    # Cumulative metrics
    cr_metrics = evaluate_cumulative_rainfall(yt, yp, horizons=[1, 3, 5, 7])
    
    record = {
        "model": model_name,
        "group": group,
        "strategy": strategy,
        "status": ExecutionStatus.SUCCESS.value,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "rain_day_MAE": round(rain_day_mae, 4),
        "rain_day_RMSE": round(rain_day_rmse, 4),
        "CR-MAE@1": cr_metrics.get("CR_MAE_1", np.nan),
        "CR-MAE@3": cr_metrics.get("CR_MAE_3", np.nan),
        "CR-MAE@5": cr_metrics.get("CR_MAE_5", np.nan),
        "CR-MAE@7": cr_metrics.get("CR_MAE_7", np.nan),
        "CR-Bias@7": cr_metrics.get("CR_Bias_7", np.nan),
        "training_time": round(float(training_time), 4),
        "training_time_std": round(float(training_time_std), 4),
        "inference_time": round(float(inference_time), 4),
        "inference_time_std": round(float(inference_time_std), 4),
        "cpu_peak_ram_mb": round(float(cpu_peak_ram_mb), 2),
        "gpu_peak_vram_mb": round(float(gpu_peak_vram_mb), 2),
        "parameter_count": int(parameter_count),
        "failure_reason": "",
        "exception_type": "",
    }
    
    return pd.DataFrame([record])[UNIFIED_SCHEMA]


def save_benchmark_artifacts(
    results_df: pd.DataFrame,
    forecasts_dict: Dict[str, np.ndarray],
    y_test_true: np.ndarray,
    group_name: str,
    results_dir: Path = RESULTS_DIR,
) -> Tuple[Path, Path]:
    """Save results CSV and predictions NPZ to the standardized results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / f"{group_name}_results.csv"
    npz_path = results_dir / f"{group_name}_forecasts.npz"
    
    results_df.to_csv(csv_path, index=False)
    
    npz_payload = {"y_true": y_test_true}
    for model_name, preds in forecasts_dict.items():
        npz_payload[f"pred_{model_name}"] = np.asarray(preds, dtype=np.float32)
        
    np.savez_compressed(npz_path, **npz_payload)
    
    return csv_path, npz_path
