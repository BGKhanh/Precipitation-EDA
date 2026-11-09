"""Evaluation protocols, split contracts, typed gatechecks, and artifact schemas.

Ensures strict compliance with Modeling Plan v1.4.18:
- data-leakage-prevention.md (fit/transform, shift(1) / inclusive causal origin window, calendar diffs)
- evaluation-standards.md (naive baselines, horizon breakdowns, rain-day metrics at tau_rain = 1.0 mm)
- P0.1: Horizon Purge Protocol (t + H <= T_split_end; exactly H=7 origins purged per split)
- P0.2: Zero Test Contamination (Dev Train for fit/selection/HPO, Full Train for final refit after freeze)
- P0.3: Future Covariate & Issue-Time Contract (D1b: Lượng mưa_lag_0 at origin t)
- P0.4: Seasonal period clarification (s=7 is weekly; annual monsoon is regime)
- P0.5 & P6-1: Model-Scoped Failure (FAILED_TEST) vs Process-Scoped Interruption (TEST_EVALUATION_INTERRUPTED)
- Metric Contract: CR-MAE@H, CR-Bias@H, and RainDay-MAE_1.0 (inclusive >= 1.0 mm)
"""

import os
import inspect
import json
import time
import math
import contextlib
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable, Union
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
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_scaled_error,
    evaluate_point_forecast_metrics,
    CANONICAL_DEV_TRAIN_MASE_SCALE,
    rain_day_mae as calc_rain_day_mae,
    rain_day_rmse as calc_rain_day_rmse,
)

REPORTS_DIR = Path("reports")
RESULTS_DIR = REPORTS_DIR / "modeling" / "benchmarks"
CONTRACT_PATH = REPORTS_DIR / "modeling" / "modeling_contract.json"
PHASE0_AUDIT_PATH = REPORTS_DIR / "modeling" / "phase0_audit_report.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda" / "eda_report.json"
PROCESSED_EDA_REPORT_PATH = Path("data") / "processed" / "eda_report.json"
NOTEBOOK_EDA_REPORT_PATH = Path("notebooks") / "eda" / "outputs" / "eda_report.json"

FORECAST_HORIZON = 7
CONTEXT_LENGTH = 30
SEASON_LENGTH = 7
RAIN_THRESHOLD = 1.0
RAIN_COMPARISON_OPERATOR = ">="
TARGET_COL = "Lượng mưa"
DATE_COL = "Ngày"
WARMUP_ROWS_F1 = 29

# ======================================================================
# Canonical Long-Format Forecast Schema (13 Columns, Protocol v1.0.20 §13.1)
# ======================================================================
CANONICAL_PREDICTIONS_SCHEMA = [
    "run_id",
    "scenario_id",
    "scenario_key",
    "unique_id",
    "ds",
    "cutoff",
    "horizon",
    "y_true",
    "y_pred",
    "model_id",
    "partition",
    "output_type",
    "point_forecast_semantics",
]

# ======================================================================
# Unified Benchmark Schema (20 Columns, Protocol v1.0.20 §9.3.B / §13.2.D)
# ======================================================================
UNIFIED_SCHEMA = [
    "model",
    "group",
    "strategy",
    "status",
    "MAE",
    "MSE",
    "RMSE",
    "MASE",
    "mae_macro",
    "mse_macro",
    "rmse_macro",
    "mase_lag1_macro",
    "training_time",
    "training_time_std",
    "inference_time",
    "inference_time_std",
    "cpu_peak_ram_mb",
    "gpu_peak_vram_mb",
    "failure_reason",
    "exception_type",
]


# ======================================================================
# Enums for Contracts & State Machines (Plan v1.4.18)
# ======================================================================

class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_CONVERGENCE = "FAILED_CONVERGENCE"
    FAILED_NUMERICAL = "FAILED_NUMERICAL"
    FAILED_RESOURCE = "FAILED_RESOURCE"
    VALIDATION_SUCCESS = "VALIDATION_SUCCESS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    TEST_SUCCESS = "TEST_SUCCESS"
    FAILED_TEST = "FAILED_TEST"


class ModelTerminalStatus(str, Enum):
    VALIDATION_SUCCESS = "VALIDATION_SUCCESS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    TEST_SUCCESS = "TEST_SUCCESS"
    FAILED_TEST = "FAILED_TEST"
    INCOMPLETE_UNRECORDED = "INCOMPLETE_UNRECORDED"


class SystemEvaluationStatus(str, Enum):
    TEST_EVALUATION_COMPLETE = "TEST_EVALUATION_COMPLETE"
    TEST_EVALUATION_FAILED = "TEST_EVALUATION_FAILED"
    TEST_EVALUATION_INTERRUPTED = "TEST_EVALUATION_INTERRUPTED"


class InferenceModuleStatus(str, Enum):
    CONFIRMATORY_INFERENCE_COMPLETE = "CONFIRMATORY_INFERENCE_COMPLETE"
    CONFIRMATORY_INFERENCE_PARTIALLY_EVALUABLE = "CONFIRMATORY_INFERENCE_PARTIALLY_EVALUABLE"
    NO_EVALUABLE_CONFIRMATORY_PAIRS = "NO_EVALUABLE_CONFIRMATORY_PAIRS"
    NO_CONFIRMATORY_PAIRS_AVAILABLE = "NO_CONFIRMATORY_PAIRS_AVAILABLE"
    CONFIRMATORY_INFERENCE_UNAVAILABLE = "CONFIRMATORY_INFERENCE_UNAVAILABLE"


class GatecheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


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
# P0 Canonical Threshold Normalization & Decoupling
# ======================================================================

CANONICAL_RAIN_THRESHOLD = 1.0
CANONICAL_RAIN_THRESHOLD_OPERATOR = ">="
LEGACY_EDA_RAIN_THRESHOLD = 0.1
LEGACY_EDA_RAIN_THRESHOLD_OPERATOR = ">="


def validate_and_normalize_canonical_threshold(
    val: Any, source_name: str = "canonical_rain_threshold"
) -> float:
    """Validate and normalize canonical rainfall classification threshold to exact float 1.0 (P0).

    Fail-closed validation:
    - Rejects None (missing)
    - Rejects boolean types (isinstance(val, bool))
    - Rejects non-numeric types (strings, lists, dicts, etc.)
    - Rejects non-finite values (math.isnan, math.isinf)
    - Rejects values outside documented numerical tolerance |v - 1.0| > 1e-6
    - Normalizes accepted values within tolerance to exact float 1.0.
    """
    if val is None:
        raise ValueError(f"[{source_name}] Canonical rainfall threshold is missing (None).")
    if isinstance(val, bool):
        raise ValueError(f"[{source_name}] Boolean value {val} is strictly forbidden as rainfall threshold.")
    if not isinstance(val, (int, float, np.floating, np.integer)):
        raise TypeError(f"[{source_name}] Threshold must be numeric float/int, got {type(val).__name__}: {val}")

    f_val = float(val)
    if math.isnan(f_val):
        raise ValueError(f"[{source_name}] Threshold value is NaN, which is strictly prohibited.")
    if math.isinf(f_val):
        raise ValueError(f"[{source_name}] Threshold value is infinite, which is strictly prohibited.")

    if abs(f_val - 1.0) > 1e-6:
        raise ValueError(
            f"[{source_name}] Threshold value {f_val} violates canonical 1.0 mm/day contract "
            f"(tolerance |v - 1.0| <= 1e-6)."
        )
    return 1.0


# ======================================================================
# P2 Pre-Read I/O Barrier & Data Leakage Guard
# ======================================================================

class DataLeakageSecurityError(PermissionError):
    """Raised when an unauthorized read of raw or processed time-series data is attempted."""
    pass


class PreReadFileReadGuard:
    """Pre-Read I/O Barrier protecting against target data leakage in Phase 0.

    Denies read access to:
      - data/raw/*
      - data/processed/*.csv, *.parquet, *.feather, *.pkl, etc.
    Allows:
      - data/processed/eda_report.json (approved synthesis artifact)
      - reports/eda/*.json
      - notebooks/eda/outputs/*.json
      - reports/modeling/*.json
      - src/*, tests/*, python runtime, and temporary files.
    """

    def __init__(self) -> None:
        self.access_attempts: int = 0
        self.blocked_attempts: int = 0
        self.actual_target_reads: int = 0
        self.is_active: bool = False
        self._orig_open = None
        self._orig_path_open = None
        self._orig_read_bytes = None
        self._orig_read_text = None
        self._active_target = self

    def is_forbidden_path(self, path: Union[str, Path, int]) -> bool:
        if isinstance(path, int):
            return False
        try:
            p = Path(path).resolve()
        except Exception:
            return False

        p_str = str(p).replace("\\", "/").lower()

        # Explicit allowlist check first
        if p_str.endswith("data/processed/eda_report.json"):
            return False
        if "/reports/" in p_str:
            return False
        if "/notebooks/eda/outputs/" in p_str:
            return False
        if "/data/processed/results/" in p_str:
            return False
        if "/src/" in p_str or "/tests/" in p_str:
            return False

        # Deny checks:
        # 1. Any file inside data/raw
        if "/data/raw" in p_str or p_str.endswith("/data/raw"):
            return True

        # 2. Time-series data files inside data/processed
        if "/data/processed" in p_str:
            for ext in (".csv", ".parquet", ".feather", ".pkl", ".h5", ".npy", ".npz", ".tsv"):
                if p_str.endswith(ext):
                    return True
            if any(name in p.name.lower() for name in ("train", "test", "val", "dev")):
                return True

        return False

    def check_and_intercept(self, path: Union[str, Path, int]) -> None:
        target = self._active_target
        if self.is_forbidden_path(path):
            target.access_attempts += 1
            target.blocked_attempts += 1
            raise DataLeakageSecurityError(
                f"[PreReadFileReadGuard] Blocked unauthorized read access to '{path}'. "
                f"Inspection of raw or processed time-series data is strictly forbidden in Phase 0."
            )

    @contextlib.contextmanager
    def isolated_scope(self):
        """Dedicated execution context where counters do not contaminate global session stats."""
        class IsolatedMonitor:
            def __init__(self):
                self.access_attempts: int = 0
                self.blocked_attempts: int = 0
                self.actual_target_reads: int = 0

        iso = IsolatedMonitor()
        prev = self._active_target
        self._active_target = iso
        try:
            yield iso
        finally:
            self._active_target = prev

    def activate(self) -> None:
        if self.is_active:
            return

        import builtins
        import pathlib

        self._orig_open = builtins.open
        self._orig_path_open = pathlib.Path.open
        self._orig_read_bytes = pathlib.Path.read_bytes
        self._orig_read_text = pathlib.Path.read_text

        guard_self = self

        def guarded_open(file, *args, **kwargs):
            mode = args[0] if len(args) > 0 else kwargs.get("mode", "r")
            if any(m in mode for m in ("r", "+")) or ("w" not in mode and "a" not in mode and "x" not in mode):
                guard_self.check_and_intercept(file)
            return guard_self._orig_open(file, *args, **kwargs)

        def guarded_path_open(self_path, *args, **kwargs):
            mode = args[0] if len(args) > 0 else kwargs.get("mode", "r")
            if any(m in mode for m in ("r", "+")) or ("w" not in mode and "a" not in mode and "x" not in mode):
                guard_self.check_and_intercept(self_path)
            return guard_self._orig_path_open(self_path, *args, **kwargs)

        def guarded_read_bytes(self_path):
            guard_self.check_and_intercept(self_path)
            return guard_self._orig_read_bytes(self_path)

        def guarded_read_text(self_path, *args, **kwargs):
            guard_self.check_and_intercept(self_path)
            return guard_self._orig_read_text(self_path, *args, **kwargs)

        builtins.open = guarded_open
        pathlib.Path.open = guarded_path_open
        pathlib.Path.read_bytes = guarded_read_bytes
        pathlib.Path.read_text = guarded_read_text
        self.is_active = True

    def deactivate(self) -> None:
        if not self.is_active:
            return
        import builtins
        import pathlib

        if self._orig_open:
            builtins.open = self._orig_open
        if self._orig_path_open:
            pathlib.Path.open = self._orig_path_open
        if self._orig_read_bytes:
            pathlib.Path.read_bytes = self._orig_read_bytes
        if self._orig_read_text:
            pathlib.Path.read_text = self._orig_read_text
        self.is_active = False

    def reset_stats(self) -> None:
        self.access_attempts = 0
        self.blocked_attempts = 0
        self.actual_target_reads = 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "access_attempts": self.access_attempts,
            "blocked_attempts": self.blocked_attempts,
            "actual_target_reads": self.actual_target_reads,
            "is_active": self.is_active,
        }


_GLOBAL_READ_GUARD: Optional[PreReadFileReadGuard] = None


def get_global_read_guard() -> PreReadFileReadGuard:
    global _GLOBAL_READ_GUARD
    if _GLOBAL_READ_GUARD is None:
        _GLOBAL_READ_GUARD = PreReadFileReadGuard()
    return _GLOBAL_READ_GUARD


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
    """Measure wall-clock time and peak memory with family-specific warm-up."""
    import tracemalloc

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
    gpu_peak_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        if is_gpu and HAS_TORCH and torch.cuda.is_available()
        else 0.0
    )

    return {
        "result": last_result,
        "time_mean_sec": round(mean_time, 4),
        "time_std_sec": round(std_time, 4),
        "time_raw_runs": [round(t, 4) for t in times],
        "cpu_peak_ram_mb": cpu_peak_mb,
        "gpu_peak_vram_mb": gpu_peak_mb,
    }


# ======================================================================
# Modeling Contract (Plan v1.4.18)
# ======================================================================

def get_default_contract() -> Dict[str, Any]:
    """Return standard project-wide modeling contract specifications (Plan v1.4.19)."""
    return {
        "plan_version": "v1.4.19",
        "canonical_split": {
            "dev_train": {
                "start": "2001-01-01",
                "end": "2018-04-30",
                "days": 6329,
                "raw_valid_origins": 6322,
                "valid_origins": 6322,
                "common_origins": 6293,
                "warmup_rows_f1": 29,
                "purged_origins": 7,
                "max_origin_date": "2018-04-23",
                "max_target_date": "2018-04-30",
            },
            "dev_train_step_a": {
                "dev_train_fit": {
                    "start": "2001-01-01",
                    "end": "2016-07-31",
                    "days": 5691,
                    "valid_origins": 5684,
                    "purged_origins": 7,
                    "max_origin_date": "2016-07-24",
                    "max_target_date": "2016-07-31",
                },
                "dev_train_es": {
                    "start": "2016-08-01",
                    "end": "2018-04-30",
                    "days": 638,
                    "valid_origins": 631,
                    "purged_origins": 7,
                    "max_origin_date": "2018-04-23",
                    "max_target_date": "2018-04-30",
                },
            },
            "validation": {
                "start": "2018-05-01",
                "end": "2020-04-30",
                "days": 731,
                "valid_origins": 724,
                "purged_origins": 7,
                "max_origin_date": "2020-04-23",
                "max_target_date": "2020-04-30",
            },
            "final_test": {
                "start": "2020-05-01",
                "end": "2026-04-30",
                "days": 2191,
                "valid_origins": 2184,
                "purged_origins": 7,
                "max_origin_date": "2026-04-23",
                "max_target_date": "2026-04-30",
            },
            "full_train_refit": {
                "start": "2001-01-01",
                "end": "2020-04-30",
                "days": 7060,
                "raw_valid_origins": 7053,
                "common_origins": 7024,
                "purged_origins": 7,
                "max_origin_date": "2020-04-23",
                "max_target_date": "2020-04-30",
            },
        },
        "forecast_horizon": 7,
        "context_length": 30,
        "neural_context_candidates": [30, 60],
        "season_length": 7,
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
        "rain_threshold": RAIN_THRESHOLD,
        "rain_comparison_operator": RAIN_COMPARISON_OPERATOR,
        "canonical_rain_threshold": CANONICAL_RAIN_THRESHOLD,
        "canonical_rain_threshold_operator": CANONICAL_RAIN_THRESHOLD_OPERATOR,
        "canonical_intermittency_adi": 2.30,
        "canonical_intermittency_cv2": 1.25,
        "legacy_eda_rain_threshold": LEGACY_EDA_RAIN_THRESHOLD,
        "legacy_eda_rain_threshold_operator": LEGACY_EDA_RAIN_THRESHOLD_OPERATOR,
        "legacy_eda_intermittency_adi": 1.64,
        "legacy_eda_intermittency_cv2": 2.02,
        "decisions": {
            "G1": "APPROVED_OPTION_1",
            "M1": "APPROVED",
            "M2": "APPROVED",
            "M3": "SUPERSEDED_BY_USER_DECISION",
            "M4": "APPROVED",
            "M5": "APPROVED",
            "M6": "APPROVED",
            "M6a": "APPROVED",
            "M7": "APPROVED_OPTION_B",
        },
        "excluded_models": ["TiDE"],
        "total_benchmark_models": 24,
        "issue_time_contract": {
            "target_lag_0_col": "Lượng mưa_lag_0",
            "research_status": "RESEARCH_OFFLINE_BENCHMARK_PERMITTED",
            "operational_status": "OPERATIONAL_AVAILABILITY_UNVERIFIED",
            "description": (
                "Forecast at origin t is issued only after the 24-hour accumulated rainfall "
                "observation ending at origin t (Lượng mưa_lag_0 = Y_t) is available."
            ),
        },
        "horizon_purge_policy": (
            "Strict Origin Purge: Origin t must satisfy t + H <= T_split_end. "
            "Exactly H=7 origins purged at end of each split."
        ),
        "fit_boundary_policy": (
            "Strict Zero Test Contamination: Dev Train ONLY for feature selection, scaling, "
            "and HPO. Full Train (7,060 days) for final model refit ONLY after Validation Freeze Manifest."
        ),
        "benchmark_a_policy": {
            "name": "Scientific Feature Construction Ablation",
            "construction": "F0 -> F1 -> F2 -> F3",
            "selection": "None (all generated features fed to model)",
            "transformation": {
                "rule": "Fixed per model family across all F0->F3 levels",
                "LightGBM": "None",
                "SARIMAX": "StandardScaler(fit on Dev Train only)",
                "PatchTST": "StandardScaler(fit on Dev Train only)",
            },
            "failure_protocol": (
                "Records status (VALIDATION_SUCCESS, FAILED_VALIDATION, TEST_SUCCESS, FAILED_TEST), "
                "failure_reason, exception_type without pre-emptive causal claims"
            ),
        },
        "benchmark_b_policy": {
            "name": "Model-Optimized Pipeline",
            "selection": {
                "rule": "Fit strictly on Dev Train only",
                "SARIMAX": "VIF < 10 (fit on Dev Train only)",
                "LightGBM": "Tree Feature Importance threshold (fit on Dev Train only)",
            },
            "transformation": "Model-specific transformations (fit on Dev Train only)",
            "protocol_sequence": (
                "Dev Train only -> fit Selection/Scaler/HPO -> Validation -> "
                "Freeze Manifest -> Refit Dev+Val -> Final Test"
            ),
        },
        "future_covariate_policy": (
            "No future ground-truth meteorology permitted after origin t. "
            "Only causal observations <= t and genuinely known calendar cyclical features allowed."
        ),
        "hardware_metadata": get_hardware_metadata(),
        "schema": UNIFIED_SCHEMA,
    }


def save_modeling_contract(
    contract: Optional[Dict[str, Any]] = None,
    filepath: Path = CONTRACT_PATH,
) -> None:
    """Save the formal modeling contract JSON."""
    c = contract or get_default_contract()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
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
    non-overlapping temporal splits.
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
    origin_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Build multi-horizon direct regression samples (X_t -> [y_{t+1}, ..., y_{t+H}]).

    Enforces Horizon Purge: Samples where t + H > len(df) are strictly purged
    (exactly H origins discarded at the end of df), guaranteeing zero boundary
    bleed across splits.
    """
    n = len(df)
    max_origin = max(0, n - horizon)

    if max_origin == 0:
        if origin_mask is not None and len(origin_mask) != 0:
            raise ValueError("origin_mask length must be 0 when max_origin == 0")
        return (
            np.empty((0, len(feature_cols)), dtype=np.float64),
            np.empty((0, horizon), dtype=np.float64),
            pd.Series(dtype="datetime64[ns]"),
        )

    if origin_mask is not None:
        if len(origin_mask) != max_origin:
            raise ValueError(
                f"origin_mask length ({len(origin_mask)}) does not match candidate origin count ({max_origin})"
            )

    X_mat = df[feature_cols].to_numpy(dtype=np.float64)
    y_vec = df[target_col].to_numpy(dtype=np.float64)
    dates = df["Ngày"] if "Ngày" in df.columns else pd.Series(range(n))

    X_samples = []
    Y_samples = []
    origin_dates = []

    for i in range(max_origin):
        if origin_mask is not None and not origin_mask[i]:
            continue
        X_samples.append(X_mat[i])
        Y_samples.append(y_vec[i + 1 : i + 1 + horizon])
        origin_dates.append(dates.iloc[i])

    if len(X_samples) == 0:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float64),
            np.empty((0, horizon), dtype=np.float64),
            pd.Series(dtype="datetime64[ns]"),
        )

    X_arr = np.array(X_samples, dtype=np.float64)
    Y_arr = np.array(Y_samples, dtype=np.float64)
    dates_series = pd.Series(origin_dates)
    if pd.api.types.is_datetime64_any_dtype(dates):
        dates_series = pd.to_datetime(dates_series)

    return X_arr, Y_arr, dates_series


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
    rain_threshold: float = 1.0,
    scale: float = CANONICAL_DEV_TRAIN_MASE_SCALE,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Compute benchmark metrics conforming to the unified schema (Protocol v1.0.20 §9.1, §9.3).

    Calculates 4 canonical point metrics: MAE, MSE, RMSE, MASE (Pooled and Macro averages).
    Uses canonical scale_DevTrain = 3.7015424 mm and rain_threshold = 1.0 mm/day.
    """
    valid_success_statuses = {
        ExecutionStatus.SUCCESS,
        ExecutionStatus.VALIDATION_SUCCESS,
        ExecutionStatus.TEST_SUCCESS,
        "SUCCESS",
        "VALIDATION_SUCCESS",
        "TEST_SUCCESS",
    }
    status_str = str(status.value if hasattr(status, "value") else status)
    if (status not in valid_success_statuses and status_str not in valid_success_statuses) or y_true is None or y_pred is None:
        record = {
            "model": model_name,
            "group": group,
            "strategy": strategy,
            "status": status_str,
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MASE": np.nan,
            "mae_macro": np.nan,
            "mse_macro": np.nan,
            "rmse_macro": np.nan,
            "mase_lag1_macro": np.nan,
            "training_time": round(float(training_time), 4),
            "training_time_std": round(float(training_time_std), 4),
            "inference_time": round(float(inference_time), 4),
            "inference_time_std": round(float(inference_time_std), 4),
            "cpu_peak_ram_mb": round(float(cpu_peak_ram_mb), 2),
            "gpu_peak_vram_mb": round(float(gpu_peak_vram_mb), 2),
            "failure_reason": str(failure_reason),
            "exception_type": str(exception_type),
        }
        return pd.DataFrame([record])[UNIFIED_SCHEMA]

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    yp = np.clip(yp, 0.0, None)

    point_metrics = evaluate_point_forecast_metrics(yt, yp, scale=scale, horizon=horizon)

    record = {
        "model": model_name,
        "group": group,
        "strategy": strategy,
        "status": status_str,
        "MAE": round(point_metrics["MAE"], 4),
        "MSE": round(point_metrics["MSE"], 4),
        "RMSE": round(point_metrics["RMSE"], 4),
        "MASE": round(point_metrics["MASE"], 4),
        "mae_macro": round(point_metrics["mae_macro"], 4),
        "mse_macro": round(point_metrics["mse_macro"], 4),
        "rmse_macro": round(point_metrics["rmse_macro"], 4),
        "mase_lag1_macro": round(point_metrics["mase_lag1_macro"], 4),
        "training_time": round(float(training_time), 4),
        "training_time_std": round(float(training_time_std), 4),
        "inference_time": round(float(inference_time), 4),
        "inference_time_std": round(float(inference_time_std), 4),
        "cpu_peak_ram_mb": round(float(cpu_peak_ram_mb), 2),
        "gpu_peak_vram_mb": round(float(gpu_peak_vram_mb), 2),
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


# ======================================================================
# Six Typed Gatechecks G1-G6 (§2.6 of Modeling Plan v1.4.18)
# ======================================================================

def evaluate_six_gatechecks(
    y_pred: Optional[np.ndarray] = None,
    expected_shape: Optional[Tuple[int, int]] = None,
    origin_dates_aligned: bool = True,
    max_origin_le_purge_bound: bool = True,
    max_target_le_split_end: bool = True,
    future_perturbation_passed: bool = True,
    preprocessing_fit_on_train_only: bool = True,
    threshold_aligned: bool = True,
    single_fit_verified: bool = True,
    hurdle_applicable: bool = False,
    hurdle_support_valid: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the 6 typed gatechecks G1-G6 returning GatecheckStatus (PASS/FAIL/NOT_APPLICABLE/ERROR)."""
    results: Dict[str, Dict[str, Any]] = {}

    # G1: Temporal Purge & Split Boundary Gate
    g1_ok = bool(origin_dates_aligned and max_origin_le_purge_bound and max_target_le_split_end)
    results["G1"] = {
        "name": "Temporal Purge & Split Boundary Gate",
        "status": GatecheckStatus.PASS.value if g1_ok else GatecheckStatus.FAIL.value,
        "details": {
            "origin_dates_aligned": origin_dates_aligned,
            "max_origin_le_purge_bound": max_origin_le_purge_bound,
            "max_target_le_split_end": max_target_le_split_end,
        },
    }

    # G2: Feature Causality & Preprocessing Split-Isolation Gate
    g2_ok = bool(future_perturbation_passed and preprocessing_fit_on_train_only)
    results["G2"] = {
        "name": "Feature Causality & Preprocessing Split-Isolation Gate",
        "status": GatecheckStatus.PASS.value if g2_ok else GatecheckStatus.FAIL.value,
        "details": {
            "future_perturbation_passed": future_perturbation_passed,
            "preprocessing_fit_on_train_only": preprocessing_fit_on_train_only,
        },
    }

    # G3: Threshold Alignment Gate
    results["G3"] = {
        "name": "Threshold Alignment Gate (tau_rain = 1.0 mm, >=)",
        "status": GatecheckStatus.PASS.value if threshold_aligned else GatecheckStatus.FAIL.value,
        "details": {"rain_threshold": RAIN_THRESHOLD, "operator": RAIN_COMPARISON_OPERATOR},
    }

    # G4: Numerical & Physical Plausibility Gate
    if y_pred is None:
        results["G4"] = {
            "name": "Numerical & Physical Plausibility Gate",
            "status": GatecheckStatus.NOT_APPLICABLE.value,
            "details": {"reason": "No forecast array provided"},
        }
    else:
        try:
            arr = np.asarray(y_pred, dtype=np.float64)
            shape_ok = (expected_shape is None) or (arr.shape == tuple(expected_shape))
            finite_ok = bool(np.all(np.isfinite(arr)))
            non_neg_ok = bool(np.all(arr >= 0.0)) if finite_ok else False
            g4_ok = bool(shape_ok and finite_ok and non_neg_ok)
            results["G4"] = {
                "name": "Numerical & Physical Plausibility Gate",
                "status": GatecheckStatus.PASS.value if g4_ok else GatecheckStatus.FAIL.value,
                "details": {
                    "shape": list(arr.shape),
                    "expected_shape": list(expected_shape) if expected_shape else None,
                    "shape_ok": shape_ok,
                    "finite_ok": finite_ok,
                    "non_neg_ok": non_neg_ok,
                },
            }
        except Exception as e:
            results["G4"] = {
                "name": "Numerical & Physical Plausibility Gate",
                "status": GatecheckStatus.ERROR.value,
                "details": {"error": str(e)},
            }

    # G5: Single-Fit Parameter Invariance Gate
    results["G5"] = {
        "name": "Single-Fit Parameter Invariance Gate (M2)",
        "status": GatecheckStatus.PASS.value if single_fit_verified else GatecheckStatus.FAIL.value,
        "details": {"single_fit_verified": single_fit_verified},
    }

    # G6: Hurdle Conditional Support Gate (M1)
    if not hurdle_applicable:
        results["G6"] = {
            "name": "Hurdle Conditional Support Gate (M1)",
            "status": GatecheckStatus.NOT_APPLICABLE.value,
            "details": {"reason": "Single-stage model"},
        }
    else:
        results["G6"] = {
            "name": "Hurdle Conditional Support Gate (M1)",
            "status": GatecheckStatus.PASS.value if hurdle_support_valid else GatecheckStatus.FAIL.value,
            "details": {"hurdle_support_valid": hurdle_support_valid},
        }

    return results


# ======================================================================
# Phase 0 Automated Contract Gatecheck Functions (§4 Phase 0)
# ======================================================================

def gatecheck_threshold_source_alignment(
    contract_path: Path = CONTRACT_PATH,
    eda_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Phase 0 Gate 1: Verify all threshold sources align at 1.0 mm/day with inclusive >= and legacy EDA decoupling (P0)."""
    from . import metrics as metrics_mod
    from . import benchmark as benchmark_mod

    if not contract_path.exists():
        save_modeling_contract(filepath=contract_path)

    paths_to_check = eda_paths or [
        EDA_REPORT_PATH,
        PROCESSED_EDA_REPORT_PATH,
        NOTEBOOK_EDA_REPORT_PATH,
    ]

    canonical_eda_values: Dict[str, float] = {}
    legacy_eda_values: Dict[str, float] = {}
    canonical_operators: Dict[str, str] = {}
    legacy_operators: Dict[str, str] = {}
    canonical_adis: Dict[str, float] = {}
    canonical_cv2s: Dict[str, float] = {}
    legacy_adis: Dict[str, float] = {}
    legacy_cv2s: Dict[str, float] = {}

    for p in paths_to_check:
        if not p.exists():
            return {
                "status": GatecheckStatus.FAIL.value,
                "reason": f"Missing EDA artifact: {p}",
            }
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        canonical_eda_values[str(p)] = float(data.get("canonical_rain_threshold", -1.0))
        canonical_operators[str(p)] = str(data.get("canonical_rain_threshold_operator", ""))
        canonical_adis[str(p)] = float(data.get("canonical_intermittency_adi", -1.0))
        canonical_cv2s[str(p)] = float(data.get("canonical_intermittency_cv2", -1.0))
        legacy_eda_values[str(p)] = float(data.get("validated_rain_threshold", -1.0))
        legacy_operators[str(p)] = str(data.get("validated_rain_threshold_operator", ""))
        legacy_adis[str(p)] = float(data.get("intermittency_adi", -1.0))
        legacy_cv2s[str(p)] = float(data.get("intermittency_cv2", -1.0))

    with open(contract_path, "r", encoding="utf-8") as f:
        contract_data = json.load(f)
    contract_thresh = float(contract_data.get("canonical_rain_threshold", contract_data.get("rain_threshold", -1.0)))
    contract_op = str(contract_data.get("canonical_rain_threshold_operator", contract_data.get("rain_comparison_operator", "")))

    # Inspect function defaults in metrics.py, protocols.py, benchmark.py
    mae_sig = inspect.signature(metrics_mod.rain_day_mae)
    rmse_sig = inspect.signature(metrics_mod.rain_day_rmse)
    comp_sig = inspect.signature(compute_benchmark_metrics)
    bench_sig = inspect.signature(benchmark_mod.UnifiedBenchmark.__init__)

    metrics_mae_def = float(mae_sig.parameters["rain_threshold"].default)
    metrics_rmse_def = float(rmse_sig.parameters["rain_threshold"].default)
    protocols_comp_def = float(comp_sig.parameters["rain_threshold"].default)
    bench_def = float(bench_sig.parameters["rain_threshold"].default)

    # Validator verification: normalize acceptable tolerance value to exact 1.0, reject non-finite
    norm_val = validate_and_normalize_canonical_threshold(1.0000001)
    validator_normalized_ok = (norm_val == 1.0)

    validator_rejected_nan = False
    try:
        validate_and_normalize_canonical_threshold(float("nan"))
    except ValueError:
        validator_rejected_nan = True

    # Boundary test at exact 1.0 mm (must be included as wet day) and empty mask (must return np.nan)
    y_true_boundary_1 = np.array([0.0, 0.5, 0.999, 1.0, 2.0])
    y_pred_boundary_1 = np.array([0.0, 0.0, 0.0, 1.5, 3.0])
    boundary_mae_1 = metrics_mod.rain_day_mae(y_true_boundary_1, y_pred_boundary_1, rain_threshold=1.0)
    empty_mae_1 = metrics_mod.rain_day_mae(np.array([0.0, 0.2, 0.99]), np.array([0.1, 0.2, 0.3]), rain_threshold=1.0)

    # Legacy boundary test at exact 0.1 mm
    y_true_boundary_01 = np.array([0.0, 0.05, 0.099, 0.1, 1.0])
    y_pred_boundary_01 = np.array([0.0, 0.0, 0.0, 0.2, 1.5])
    boundary_mae_01 = metrics_mod.rain_day_mae(y_true_boundary_01, y_pred_boundary_01, rain_threshold=0.1)

    all_passed = (
        all(v == 1.0 for v in canonical_eda_values.values())
        and all(op == ">=" for op in canonical_operators.values())
        and all(abs(adi - 2.30) < 0.05 for adi in canonical_adis.values())
        and all(abs(cv2 - 1.25) < 0.05 for cv2 in canonical_cv2s.values())
        and all(v == 0.1 for v in legacy_eda_values.values())
        and all(op == ">=" for op in legacy_operators.values())
        and all(abs(adi - 1.64) < 0.05 for adi in legacy_adis.values())
        and all(abs(cv2 - 2.02) < 0.05 for cv2 in legacy_cv2s.values())
        and contract_thresh == 1.0
        and contract_op == ">="
        and RAIN_THRESHOLD == 1.0
        and metrics_mae_def == 1.0
        and metrics_rmse_def == 1.0
        and protocols_comp_def == 1.0
        and bench_def == 1.0
        and validator_normalized_ok
        and validator_rejected_nan
        and np.isclose(boundary_mae_1, 0.75)
        and np.isnan(empty_mae_1)
        and np.isclose(boundary_mae_01, 0.30)
    )

    return {
        "status": GatecheckStatus.PASS.value if all_passed else GatecheckStatus.FAIL.value,
        "canonical_eda_thresholds": canonical_eda_values,
        "canonical_eda_operators": canonical_operators,
        "legacy_eda_thresholds": legacy_eda_values,
        "legacy_eda_operators": legacy_operators,
        "modeling_contract_threshold": contract_thresh,
        "modeling_contract_operator": contract_op,
        "protocols_RAIN_THRESHOLD": RAIN_THRESHOLD,
        "metrics_rain_day_mae_default": metrics_mae_def,
        "metrics_rain_day_rmse_default": metrics_rmse_def,
        "protocols_compute_benchmark_metrics_default": protocols_comp_def,
        "benchmark_UnifiedBenchmark_default": bench_def,
        "validator_normalized_ok": validator_normalized_ok,
        "validator_rejected_nan": validator_rejected_nan,
        "boundary_1_0_included_mae": float(boundary_mae_1),
        "boundary_0_1_included_mae": float(boundary_mae_01),
        "empty_mask_returns_nan": bool(np.isnan(empty_mae_1)),
    }


def gatecheck_target_boundary_purge_h7() -> Dict[str, Any]:
    """Phase 0 Gate 2: Verify H=7 target boundary purge across all splits and Step A sub-splits.

    CRITICAL ISOLATION RULE: Verifies split/origin/target date boundaries strictly via
    synthetic date ranges matching the canonical calendar boundaries—NEVER loading or inspecting
    Final Test target rainfall labels Y.

    ORIGIN VS TARGET DISTINCTION (P2):
    - Final Test split spans 2,191 calendar days (2020-05-01 to 2026-04-30).
    - Valid forecast origins: 2,184 (ending at 2026-04-23).
    - Calendar dates 2026-04-24 through 2026-04-30 cannot serve as forecast origins t
      because t + 7 would exceed the split boundary (strict H=7 origin purge).
    - Crucially, dates 2026-04-24 through 2026-04-30 are NOT deleted or discarded:
      they remain valid target dates for forecast origins 2026-04-17 through 2026-04-23.
    """
    splits_spec = [
        ("dev_train", "2001-01-01", "2018-04-30", 6329, 6322, "2018-04-23", "2018-04-30"),
        ("dev_train_fit", "2001-01-01", "2016-07-31", 5691, 5684, "2016-07-24", "2016-07-31"),
        ("dev_train_es", "2016-08-01", "2018-04-30", 638, 631, "2018-04-23", "2018-04-30"),
        ("validation", "2018-05-01", "2020-04-30", 731, 724, "2020-04-23", "2020-04-30"),
        ("final_test_date_fixture", "2020-05-01", "2026-04-30", 2191, 2184, "2026-04-23", "2026-04-30"),
        ("full_train_refit", "2001-01-01", "2020-04-30", 7060, 7053, "2020-04-23", "2020-04-30"),
    ]

    split_reports: Dict[str, Any] = {}
    all_passed = True

    for name, start_str, end_str, exp_days, exp_origins, exp_max_origin, exp_max_target in splits_spec:
        dates = pd.date_range(start_str, end_str, freq="D")
        syn_df = pd.DataFrame({
            "Ngày": dates,
            "feat_dummy": np.ones(len(dates), dtype=np.float64),
            "Lượng mưa": np.zeros(len(dates), dtype=np.float64),
        })
        X_arr, Y_arr, origin_dates = build_purged_direct_samples(
            syn_df, feature_cols=["feat_dummy"], target_col="Lượng mưa", horizon=FORECAST_HORIZON
        )
        max_origin_str = pd.Timestamp(origin_dates.iloc[-1]).strftime("%Y-%m-%d")
        max_target_ts = pd.Timestamp(origin_dates.iloc[-1]) + pd.Timedelta(days=FORECAST_HORIZON)
        max_target_str = max_target_ts.strftime("%Y-%m-%d")

        # Common origin mask check for dev_train and full_train_refit (29 warmup rows)
        common_origins_count = None
        if name in ("dev_train", "full_train_refit"):
            mask = np.ones(len(origin_dates), dtype=bool)
            mask[:WARMUP_ROWS_F1] = False
            X_c, Y_c, orig_c = build_purged_direct_samples(
                syn_df,
                feature_cols=["feat_dummy"],
                target_col="Lượng mưa",
                horizon=FORECAST_HORIZON,
                origin_mask=mask,
            )
            common_origins_count = len(orig_c)

        ok = (
            len(dates) == exp_days
            and len(origin_dates) == exp_origins
            and X_arr.shape == (exp_origins, 1)
            and Y_arr.shape == (exp_origins, FORECAST_HORIZON)
            and max_origin_str == exp_max_origin
            and max_target_str == exp_max_target
        )
        if name == "dev_train":
            ok = ok and (common_origins_count == 6293)
        elif name == "full_train_refit":
            ok = ok and (common_origins_count == 7024)

        if not ok:
            all_passed = False

        split_reports[name] = {
            "days": len(dates),
            "valid_origins": len(origin_dates),
            "common_origins": common_origins_count,
            "max_origin_date": max_origin_str,
            "max_target_date": max_target_str,
            "passed": ok,
        }

    return {
        "status": GatecheckStatus.PASS.value if all_passed else GatecheckStatus.FAIL.value,
        "final_test_labels_accessed": False,
        "splits": split_reports,
    }


def build_causal_direct_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Build strictly causal direct multi-horizon features X_t at origin t (§1.2 & D1b).

    Includes:
    - Target lags ending at origin t: Lượng mưa_lag_0 (Y_t), lag_1..7, lag_14, lag_21, lag_30
    - Inclusive rolling windows ending at origin t ([t-w+1 .. t]): rolling_mean_3, 7, 14, 30; rolling_std_7, 30
    - Calendar cyclical features for origin t: Month_sin, Month_cos, DayOfYear_sin, DayOfYear_cos
    """
    out = pd.DataFrame(index=df.index)
    y = df[target_col].astype(float)

    # 1. Target lags (lag_0 = Y_t per D1b issue-time contract)
    out[f"{target_col}_lag_0"] = y
    for lag in (1, 2, 3, 4, 5, 6, 7, 14, 21, 30):
        out[f"{target_col}_lag_{lag}"] = y.shift(lag)

    # 2. Inclusive rolling windows ending at origin t ([t-w+1 .. t])
    for w in (3, 7, 14, 30):
        out[f"{target_col}_rolling_mean_{w}"] = y.rolling(window=w, min_periods=w).mean()
        out[f"{target_col}_rolling_max_{w}"] = y.rolling(window=w, min_periods=w).max()
    for w in (7, 30):
        out[f"{target_col}_rolling_std_{w}"] = y.rolling(window=w, min_periods=w).std(ddof=0)

    # 3. Exogenous meteorology lags (if present)
    for col in ("Nhiệt độ", "Độ ẩm", "Áp suất", "Tốc độ gió"):
        if col in df.columns:
            s = df[col].astype(float)
            out[f"{col}_lag_0"] = s
            out[f"{col}_lag_1"] = s.shift(1)
            out[f"{col}_rolling_mean_7"] = s.rolling(window=7, min_periods=7).mean()

    # 4. Calendar cyclical features
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col])
        month = dt.dt.month.astype(float)
        doy = dt.dt.dayofyear.astype(float)
        out["Month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        out["Month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        out["DayOfYear_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        out["DayOfYear_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    return out


def gatecheck_feature_causality_and_split_isolation() -> Dict[str, Any]:
    """Phase 0 Gate 6: Verify Family A (Future-Perturbation Invariance) & Family B (Preprocessing Split-Isolation)."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n_days = 200
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    base_df = pd.DataFrame({
        "Ngày": dates,
        "Lượng mưa": rng.exponential(scale=4.0, size=n_days),
        "Nhiệt độ": 25.0 + rng.normal(0.0, 2.0, size=n_days),
        "Độ ẩm": 75.0 + rng.normal(0.0, 5.0, size=n_days),
    })

    # Family A: Raw Feature Future-Perturbation Invariance across 5 origins
    test_origins = [40, 75, 110, 145, 180]
    raw_base = build_causal_direct_features(base_df)
    max_raw_diff = 0.0

    for t_idx in test_origins:
        pert_df = base_df.copy()
        # Perturb all future rows s > t_idx
        pert_df.loc[t_idx + 1 :, "Lượng mưa"] += 10000.0
        pert_df.loc[t_idx + 1 :, "Nhiệt độ"] += 50.0
        pert_df.loc[t_idx + 1 :, "Độ ẩm"] -= 40.0

        raw_pert = build_causal_direct_features(pert_df)
        vec_base = raw_base.iloc[t_idx].to_numpy(dtype=np.float64)
        vec_pert = raw_pert.iloc[t_idx].to_numpy(dtype=np.float64)
        diff = float(np.max(np.abs(vec_base - vec_pert)))
        if diff > max_raw_diff:
            max_raw_diff = diff

    # Verify inclusive rolling window [t-w+1 .. t] formula at t=50, w=7
    t_check = 50
    w_check = 7
    manual_mean_7 = float(np.mean(base_df["Lượng mưa"].iloc[t_check - w_check + 1 : t_check + 1]))
    feature_mean_7 = float(raw_base["Lượng mưa_rolling_mean_7"].iloc[t_check])
    inclusive_window_match = bool(np.isclose(manual_mean_7, feature_mean_7, atol=1e-12))

    # Verify D1b lag_0 identity
    lag0_match = bool(np.allclose(raw_base["Lượng mưa_lag_0"].to_numpy(), base_df["Lượng mưa"].to_numpy(), atol=1e-12))

    # Family B: Preprocessing Split-Isolation
    # Fit StandardScaler strictly on train_fit slice (rows 30..120), then verify frozen transform
    # on origin t=150 is invariant to perturbing rows > 150
    valid_raw_base = raw_base.iloc[30:].copy()
    train_slice = valid_raw_base.iloc[:90]  # indices 30..119
    scaler = StandardScaler().fit(train_slice.to_numpy(dtype=np.float64))

    pert_after_150 = base_df.copy()
    pert_after_150.loc[151:, "Lượng mưa"] += 50000.0
    raw_pert_150 = build_causal_direct_features(pert_after_150)

    x_tilde_base = scaler.transform(raw_base.iloc[[150]].to_numpy(dtype=np.float64))
    x_tilde_pert = scaler.transform(raw_pert_150.iloc[[150]].to_numpy(dtype=np.float64))
    max_scaled_diff = float(np.max(np.abs(x_tilde_base - x_tilde_pert)))

    family_a_pass = bool((max_raw_diff <= 1e-7) and inclusive_window_match and lag0_match)
    family_b_pass = bool((max_scaled_diff <= 1e-7) and (int(scaler.n_samples_seen_) == 90))

    return {
        "status": GatecheckStatus.PASS.value if (family_a_pass and family_b_pass) else GatecheckStatus.FAIL.value,
        "family_a_raw_feature_causality": {
            "passed": family_a_pass,
            "origins_tested": test_origins,
            "max_abs_diff_under_future_perturbation": max_raw_diff,
            "inclusive_window_formula_verified": inclusive_window_match,
            "d1b_lag_0_identity_verified": lag0_match,
            "research_status": "RESEARCH_OFFLINE_BENCHMARK_PERMITTED",
            "operational_status": "OPERATIONAL_AVAILABILITY_UNVERIFIED",
        },
        "family_b_preprocessing_split_isolation": {
            "passed": family_b_pass,
            "scaler_fit_rows": int(scaler.n_samples_seen_),
            "max_abs_diff_frozen_transform": max_scaled_diff,
        },
    }
