"""Resource Metrics Normalization and Profiling.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and CR-06.
Implements the resource data dictionary rules:
1. Latency = (inference_time / 724) * 1000 (ms/origin across 724 validation origins).
2. Throughput = 724 / inference_time (origins/sec).
3. Heuristic models (SC-01, SC-02): training time marked N/A (not 0.0 on log scale).
4. Clear separation between CPU models and GPU CUDA models (GTX 1650).
5. All resource metrics are descriptive profiling only; NEVER used for model ranking.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def normalize_resource_metrics(df_results: pd.DataFrame) -> pd.DataFrame:
    """Normalize and profile runtime resources across all 72 benchmark scenarios.
    
    Args:
        df_results: DataFrame loaded from load_all_validation_results().
        
    Returns:
        DataFrame with standardized resource metrics:
            scenario_id, model, family, device, training_time_sec,
            latency_ms_per_origin, throughput_origins_per_sec, parameter_count.
    """
    df = df_results.copy()

    # 1. Device assignment
    def assign_device(sc_id: str) -> str:
        try:
            sc_num = int(sc_id.replace("SC-", ""))
            if 36 <= sc_num <= 40:
                return "GPU (NVIDIA GeForce GTX 1650)"
            return "CPU (Host System)"
        except Exception:
            return "Unknown"

    df["device"] = df["scenario_id"].apply(assign_device)

    # 2. Heuristic models: training time is N/A
    is_heuristic = df["scenario_id"].isin(["SC-01", "SC-02"])
    df.loc[is_heuristic, "training_time"] = np.nan

    # 3. Latency & Throughput (standardized over 724 origins)
    safe_inf = df["inference_time"].replace(0.0, np.nan)
    df["latency_ms_per_origin"] = (safe_inf / 724.0) * 1000.0
    df["throughput_origins_per_sec"] = 724.0 / safe_inf

    # 4. Standardize column names
    df["training_time_sec"] = df["training_time"]

    res_cols = [
        "scenario_id",
        "model",
        "canonical_model_id",
        "family",
        "strategy",
        "feature_set",
        "device",
        "training_time_sec",
        "inference_time",
        "latency_ms_per_origin",
        "throughput_origins_per_sec",
        "parameter_count",
        "MAE",
        "RMSE",
    ]
    present_cols = [c for c in res_cols if c in df.columns]
    return df[present_cols].reset_index(drop=True)

