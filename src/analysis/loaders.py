"""Data Loading and Provenance Verification for Post-Modelling Result Analysis.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and Project Rules:
- data-leakage-prevention.md
- evaluation-standards.md
"""

from pathlib import Path
import json
import hashlib
import logging
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical mapping for Phase 1 Statistical models to Scenario IDs
STATISTICAL_SCENARIO_MAPPING = {
    "Naive": "SC-01",
    "SeasonalNaive": "SC-02",
    "AutoARIMA": "SC-03",
    "AutoETS": "SC-04",
    "SARIMA_EDA": "SC-05",
    "ARIMA": "SC-06",
    "CrostonOptimized": "SC-07",
    "CrostonSBA": "SC-08",
    "TSB": "SC-09",
    "ADIDA": "SC-10",
    "IMAPA": "SC-11",
}

# Canonical 7 Families mapping
def assign_canonical_family(scenario_id: str) -> str:
    """Assign one of the 7 canonical families strictly defined by Protocol v1.0.21."""
    try:
        sc_num = int(scenario_id.replace("SC-", ""))
    except Exception:
        return "Unknown"

    if sc_num in [1, 2]:
        return "Heuristic Baseline"
    elif 3 <= sc_num <= 6:
        return "Classical Statistical"
    elif 7 <= sc_num <= 11:
        return "Intermittent Demand"
    elif 12 <= sc_num <= 27:
        return "Tabular ML Direct"
    elif 28 <= sc_num <= 35:
        return "Tabular ML Tweedie"
    elif 36 <= sc_num <= 40:
        return "Deep Neural"
    elif 41 <= sc_num <= 72:
        return "Two-Stage Hurdle"
    return "Unknown"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_provenance_hashes(
    provenance_path: Path | str = "reports/validation_benchmark/provenance.json",
    results_dir: Path | str = "data/processed/results",
    raise_on_mismatch: bool = False,
) -> bool:
    """Verify that current result files match the SHA-256 hashes recorded in provenance.json.
    
    Adheres strictly to Protocol Option A / PROP-04 provenance requirements.
    """
    prov_file = Path(provenance_path)
    res_dir = Path(results_dir)

    if not prov_file.exists():
        msg = f"Provenance file not found at {prov_file}. Run Step 0 audit first."
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return False

    with open(prov_file, "r", encoding="utf-8") as f:
        prov_data = json.load(f)

    # Extract hash mapping from provenance.json (supports both schema layouts)
    raw_hashes = prov_data.get("input_artifacts_hashes", prov_data.get("input_files_sha256", {}))
    all_matched = True

    for file_key, info in raw_hashes.items():
        if isinstance(info, dict):
            expected_hash = info.get("sha256")
        else:
            expected_hash = str(info)

        # Resolve path: either relative to res_dir or as full path
        target_path = res_dir / file_key
        if not target_path.exists():
            target_path = Path(file_key)
        
        if not target_path.exists():
            msg = f"Tracked input file missing: {target_path}"
            logger.warning(msg)
            if raise_on_mismatch:
                raise FileNotFoundError(msg)
            all_matched = False
            continue

        actual_hash = compute_sha256(target_path)
        if actual_hash != expected_hash:
            msg = (
                f"SHA-256 mismatch for {target_path.name}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
            logger.warning(msg)
            if raise_on_mismatch:
                raise ValueError(msg)
            all_matched = False

    return all_matched


def load_all_benchmark_results(
    results_dir: Path | str = "data/processed/results",
    partition: str = "validation",
) -> pd.DataFrame:
    """Load and unify benchmark results across all 5 phases into a standardized 72-scenario DataFrame.
    
    Args:
        results_dir: Directory where result CSVs are stored.
        partition: 'validation' or 'test'.
        
    Schema:
        scenario_id, model, canonical_model_id, family, strategy, feature_set,
        MAE, MSE, RMSE, MASE, training_time, inference_time, parameter_count,
        selected_hurdle_threshold, status
    """
    res_dir = Path(results_dir)
    tag = "val" if partition == "validation" else "test"

    # 1. Statistical & Intermittent (SC-01 to SC-11)
    df_stat = pd.read_csv(res_dir / f"statistical_{tag}_results.csv")
    if "scenario_id" not in df_stat.columns:
        df_stat["scenario_id"] = df_stat["model"].map(STATISTICAL_SCENARIO_MAPPING)
    if "feature_set" not in df_stat.columns:
        df_stat["feature_set"] = "None"
    if "parameter_count" not in df_stat.columns:
        df_stat["parameter_count"] = np.nan
    if "selected_hurdle_threshold" not in df_stat.columns:
        df_stat["selected_hurdle_threshold"] = np.nan

    # 2. Tabular ML Direct (SC-12 to SC-27)
    df_direct = pd.read_csv(res_dir / f"ml_direct_{tag}_results.csv")
    if "parameter_count" not in df_direct.columns:
        df_direct["parameter_count"] = np.nan
    if "selected_hurdle_threshold" not in df_direct.columns:
        df_direct["selected_hurdle_threshold"] = np.nan

    # 3. Tabular ML Tweedie (SC-28 to SC-35)
    df_tweedie = pd.read_csv(res_dir / f"ml_tweedie_{tag}_results.csv")
    if "parameter_count" not in df_tweedie.columns:
        df_tweedie["parameter_count"] = np.nan
    if "selected_hurdle_threshold" not in df_tweedie.columns:
        df_tweedie["selected_hurdle_threshold"] = np.nan

    # 4. Deep Neural (SC-36 to SC-40)
    df_neural = pd.read_csv(res_dir / f"neural_{tag}_results.csv")
    if "feature_set" not in df_neural.columns:
        df_neural["feature_set"] = "None"
    if "selected_hurdle_threshold" not in df_neural.columns:
        df_neural["selected_hurdle_threshold"] = np.nan

    # 5. Two-Stage Hurdle (SC-41 to SC-72)
    df_hurdle = pd.read_csv(res_dir / f"ml_hurdle_{tag}_results.csv")

    # Combine all 5 DataFrames
    dfs = [df_stat, df_direct, df_tweedie, df_neural, df_hurdle]
    df_all = pd.concat(dfs, ignore_index=True)

    # Normalize canonical columns
    df_all["family"] = df_all["scenario_id"].apply(assign_canonical_family)
    df_all["canonical_model_id"] = df_all["model"].str.lower()
    df_all["status"] = "COMPLETED"

    # Standardize column selection
    canonical_cols = [
        "scenario_id",
        "model",
        "canonical_model_id",
        "family",
        "strategy",
        "feature_set",
        "MAE",
        "MSE",
        "RMSE",
        "MASE",
        "training_time",
        "inference_time",
        "parameter_count",
        "selected_hurdle_threshold",
        "status",
    ]
    df_unified = df_all[canonical_cols].copy()

    # Sort strictly by Scenario ID numeric order
    df_unified["sc_num"] = df_unified["scenario_id"].str.replace("SC-", "").astype(int)
    df_unified = df_unified.sort_values("sc_num").drop(columns=["sc_num"]).reset_index(drop=True)

    # Assert exactly 72 scenarios
    assert len(df_unified) == 72, f"Expected exactly 72 scenarios, got {len(df_unified)}"
    assert df_unified["scenario_id"].nunique() == 72, "Duplicate scenario_id detected!"

    return df_unified


def load_all_validation_results(
    results_dir: Path | str = "data/processed/results",
    partition: str = "validation",
) -> pd.DataFrame:
    """Load and unify validation results across all 5 phases into a standardized 72-scenario DataFrame."""
    return load_all_benchmark_results(results_dir=results_dir, partition=partition)


def load_all_horizon_breakdowns(
    results_dir: Path | str = "data/processed/results",
    partition: str = "validation",
) -> pd.DataFrame:
    """Load and unify horizon breakdown results across all 5 phases for h=1..7.
    
    Provides 4 canonical metrics per horizon: MAE@h, MSE@h, RMSE@h, MASE@h (CR-07).
    """
    res_dir = Path(results_dir)
    tag = "val" if partition == "validation" else "test"

    files = [
        res_dir / f"statistical_{tag}_horizon_breakdown.csv",
        res_dir / f"ml_direct_{tag}_horizon_breakdown.csv",
        res_dir / f"ml_tweedie_{tag}_horizon_breakdown.csv",
        res_dir / f"neural_{tag}_horizon_breakdown.csv",
        res_dir / f"ml_hurdle_{tag}_horizon_breakdown.csv",
    ]

    dfs = []
    for f in files:
        df_part = pd.read_csv(f)
        if "scenario_id" not in df_part.columns and "model" in df_part.columns:
            df_part["scenario_id"] = df_part["model"].map(STATISTICAL_SCENARIO_MAPPING)
        dfs.append(df_part)

    df_h = pd.concat(dfs, ignore_index=True)
    df_h["family"] = df_h["scenario_id"].apply(assign_canonical_family)
    df_h["canonical_model_id"] = df_h["model"].str.lower()

    if "feature_set" not in df_h.columns:
        df_h["feature_set"] = "None"
    else:
        df_h["feature_set"] = df_h["feature_set"].fillna("None")

    # Ensure MSE@h1..h7 exist. If not present in source CSV, compute as (RMSE@h)**2 per CR-07
    for h in range(1, 8):
        mse_col = f"MSE@h{h}"
        rmse_col = f"RMSE@h{h}"
        if mse_col not in df_h.columns and rmse_col in df_h.columns:
            # Reconstruct MSE from RMSE per CR-07
            df_h[mse_col] = df_h[rmse_col] ** 2

    # Sort strictly by Scenario ID numeric order
    df_h["sc_num"] = df_h["scenario_id"].str.replace("SC-", "").astype(int)
    df_h = df_h.sort_values("sc_num").drop(columns=["sc_num"]).reset_index(drop=True)

    assert len(df_h) == 72, f"Expected exactly 72 scenarios in horizon breakdown, got {len(df_h)}"
    return df_h


def load_all_forecast_matrices(
    results_dir: Path | str = "data/processed/results",
    partition: str = "validation",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load all 72 forecast matrices and ground truth targets from the 5 NPZ files.
    
    Returns:
        forecasts_dict: Dict mapping scenario_id ('SC-01'..'SC-72') to array of shape (N_origins, 7).
        y_true: Ground truth target array of shape (N_origins, 7).
        origin_dates: Array of origin dates.
    """
    res_dir = Path(results_dir)
    tag = "val" if partition == "validation" else "test"
    expected_origins = 724 if partition == "validation" else 2184

    # 1. Statistical (SC-01 to SC-11)
    stat_npz = np.load(res_dir / f"statistical_{tag}_forecasts.npz")
    y_true = stat_npz["y_true"]
    origin_dates = stat_npz["origin_dates"]

    forecasts: Dict[str, np.ndarray] = {}
    for model_name, sc_id in STATISTICAL_SCENARIO_MAPPING.items():
        if model_name in stat_npz:
            forecasts[sc_id] = stat_npz[model_name]

    # 2. Direct ML (SC-12 to SC-27)
    direct_npz = np.load(res_dir / f"ml_direct_{tag}_forecasts.npz")
    for k in direct_npz.files:
        if k.startswith("SC-"):
            sc_id = k.split("_")[0]
            forecasts[sc_id] = direct_npz[k]

    # 3. Tweedie ML (SC-28 to SC-35)
    tweedie_npz = np.load(res_dir / f"ml_tweedie_{tag}_forecasts.npz")
    for k in tweedie_npz.files:
        if k.startswith("SC-"):
            sc_id = k.split("_")[0]
            forecasts[sc_id] = tweedie_npz[k]

    # 4. Neural (SC-36 to SC-40)
    neural_npz = np.load(res_dir / f"neural_{tag}_forecasts.npz")
    neural_sc_map = {
        "NHITS_preds": "SC-36",
        "PatchTST_preds": "SC-37",
        "TFT_preds": "SC-38",
        "TCN_preds": "SC-39",
        "LSTM_preds": "SC-40",
    }
    for k in neural_npz.files:
        if k in neural_sc_map:
            forecasts[neural_sc_map[k]] = neural_npz[k]
        elif k.startswith("SC-"):
            forecasts[k.split("_")[0]] = neural_npz[k]

    # 5. Hurdle (SC-41 to SC-72)
    hurdle_npz = np.load(res_dir / f"ml_hurdle_{tag}_forecasts.npz")
    for k in hurdle_npz.files:
        if k.startswith("SC-"):
            sc_id = k.split("_")[0]
            forecasts[sc_id] = hurdle_npz[k]

    # Assert exactly 72 scenarios present and verified
    assert len(forecasts) == 72, f"Expected 72 scenario forecast matrices, found {len(forecasts)}"
    for sc_id, mat in forecasts.items():
        assert mat.shape == (expected_origins, 7), f"{sc_id} has unexpected shape {mat.shape}, expected ({expected_origins}, 7)"
        assert not np.isnan(mat).any(), f"{sc_id} contains NaN values"
        assert not np.isinf(mat).any(), f"{sc_id} contains Inf values"

    return forecasts, y_true, origin_dates


def load_hurdle_probabilities(
    results_dir: Path | str = "data/processed/results",
    partition: str = "validation",
) -> Dict[str, Any]:
    """Load Stage 1 Hurdle probabilities and wet expectations from NPZ (CR-05).
    
    Contains 16 combos of (p_hat, mu_wet) + y_true + origin_dates.
    """
    tag = "val" if partition == "validation" else "test"
    prob_path = Path(results_dir) / f"ml_hurdle_{tag}_probabilities.npz"
    if not prob_path.exists():
        raise FileNotFoundError(f"Hurdle probabilities NPZ not found at {prob_path}")

    npz_data = np.load(prob_path)
    return {k: npz_data[k] for k in npz_data.files}
    if not prob_path.exists():
        raise FileNotFoundError(f"Hurdle probabilities NPZ not found at {prob_path}")

    npz_data = np.load(prob_path)
    return {k: npz_data[k] for k in npz_data.files}


def load_hurdle_threshold_metadata(
    results_dir: Path | str = "data/processed/results",
) -> Dict[str, Any]:
    """Load Hurdle threshold tuning metadata and optimal thresholds from JSON."""
    json_path = Path(results_dir) / "ml_hurdle_threshold_selection.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Hurdle threshold JSON not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
