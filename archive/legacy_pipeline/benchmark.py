"""Unified Benchmark Engine & Rolling-Origin Coordinator for Multi-Family Model Comparison.

Implements Modeling Plan v1.4.18:
- Inclusive rain-day threshold tau_rain = 1.0 mm/day (Y >= 1.0 mm)
- Isolation of legacy RecursiveForecaster from official direct benchmarks
- RollingBenchmarkCoordinator with:
  * Exhaustive 6-trigger Model-Scoped Failure handling (FAILED_TEST, metrics NaN, empty (0, 7) array,
    partial predictions quarantined to test_diagnostic_partial_forecasts.npz)
  * Process-Scoped Interruption disambiguation (TEST_EVALUATION_INTERRUPTED, in-flight model remains
    INCOMPLETE_UNRECORDED, atomic per-model checkpointing & recovery)
- M4 Epsilon-Equivalence Selection (epsilon = 1e-4 mm, MAE -> RMSE -> Model ID)
- M6a Exact-Equality Neural Context-Window Selection (W in {30, 60}, preferring W*=30 when M6a approved)
- M5 + P5-1 Confirmatory Diebold-Mariano Test (Newey-West HAC L=7), Conservative Holm Step-Down
  using K_planned (never replacing FAILED_TEST representatives), and Moving Block Bootstrap (b=14).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .metrics import (
    mean_absolute_error as calc_mae,
    mean_squared_error as calc_mse,
    root_mean_squared_error as calc_rmse,
    mean_absolute_scaled_error,
    evaluate_point_forecast_metrics,
    CANONICAL_DEV_TRAIN_MASE_SCALE,
    rain_day_mae as calc_rain_day_mae,
    rain_day_rmse as calc_rain_day_rmse,
)
from .protocols import (
    ExecutionStatus,
    ModelTerminalStatus,
    SystemEvaluationStatus,
    InferenceModuleStatus,
    UNIFIED_SCHEMA,
    compute_benchmark_metrics,
)


class ProcessScopedInterruption(BaseException):
    """Exception representing a process-scoped interruption (crash, OS signal, storage failure).

    Inherits from BaseException so model-scoped `except Exception:` blocks never accidentally
    classify a process-level abort as a model-level `FAILED_TEST`.
    """


# ======================================================================
# Candidate-Tier Ranking & Statistical Selection Helpers (Protocol v1.0.20 §9.3.D)
# ======================================================================

def rank_benchmark_results(
    results_df: pd.DataFrame,
    epsilon: float = 1e-4,
    valid_statuses: Optional[Sequence[str]] = None,
    primary_metric: str = "MAE",
    tiebreak_metric: str = "RMSE",
    model_col: str = "model",
) -> pd.DataFrame:
    """Implement Candidate-Tier Iterative Ranking Procedure per Protocol v1.0.20 §9.3.D.

    Full Leaderboard Iterative Procedure:
    1. Filter eligible set S_valid,p: execution_status == COMPLETED (or in valid_statuses)
       with both MAE and RMSE finite and non-negative.
    2. Primary metric: Pooled MAE (mu_k = min MAE in remaining set R_k).
       Candidate tier C_k = {s in R_k | MAE(s) <= mu_k + epsilon}.
    3. Sort scenarios in C_k by:
       (RMSE ascending IEEE 754 unrounded, model_id ascending lexicographically).
    4. Group strictly tied scenarios G_j within C_k iff simultaneously:
       RMSE_a == RMSE_b AND model_id_a == model_id_b.
    5. Position-based Rank Assignment:
       Rank(s) = r_k + sum_{l < j} |G_l| for all s in G_j.
    6. Next tier rank: r_{k+1} = r_k + |C_k| (Standard Competition 1224 Ranking).
    7. Ineligible scenarios: Rank = 'N/A [<status>]'.
    """
    if results_df.empty:
        return results_df.copy()

    df = results_df.copy()
    if valid_statuses is None:
        valid_statuses = [
            "COMPLETED",
            ExecutionStatus.SUCCESS.value,
            ExecutionStatus.VALIDATION_SUCCESS.value,
            ExecutionStatus.TEST_SUCCESS.value,
            ModelTerminalStatus.VALIDATION_SUCCESS.value,
            ModelTerminalStatus.TEST_SUCCESS.value,
        ]

    df["rank"] = None
    status_col = "status" if "status" in df.columns else ("execution_status" if "execution_status" in df.columns else None)
    if model_col not in df.columns and "model_id" in df.columns:
        model_col = "model_id"

    # Separate eligible vs ineligible
    eligible_mask = pd.Series(True, index=df.index)
    if status_col is not None:
        eligible_mask &= df[status_col].isin(valid_statuses)

    if primary_metric in df.columns:
        p_vals = pd.to_numeric(df[primary_metric], errors="coerce")
        eligible_mask &= p_vals.notna() & (p_vals >= 0.0)
    else:
        eligible_mask = pd.Series(False, index=df.index)

    if tiebreak_metric in df.columns:
        t_vals = pd.to_numeric(df[tiebreak_metric], errors="coerce")
        eligible_mask &= t_vals.notna() & (t_vals >= 0.0)

    # Ineligible scenarios receive N/A [<status>]
    for idx in df[~eligible_mask].index:
        st = str(df.loc[idx, status_col]) if status_col else "INELIGIBLE"
        df.loc[idx, "rank"] = f"N/A [{st}]"

    eligible_indices = list(df[eligible_mask].index)
    if not eligible_indices:
        return df

    unranked = set(eligible_indices)
    r_k = 1

    while unranked:
        sub_df = df.loc[list(unranked)]
        mu_k = float(sub_df[primary_metric].min())

        tier_indices = [
            idx for idx in unranked
            if float(df.loc[idx, primary_metric]) <= (mu_k + float(epsilon))
        ]

        tier_df = df.loc[tier_indices].copy()
        tier_df = tier_df.sort_values(
            by=[tiebreak_metric, model_col],
            ascending=[True, True],
            kind="mergesort",
        )
        sorted_tier_indices = list(tier_df.index)

        groups: List[List[Any]] = []
        for idx in sorted_tier_indices:
            rmse_val = float(df.loc[idx, tiebreak_metric])
            model_val = str(df.loc[idx, model_col])
            if not groups:
                groups.append([idx])
            else:
                last_group = groups[-1]
                last_idx = last_group[0]
                last_rmse = float(df.loc[last_idx, tiebreak_metric])
                last_model = str(df.loc[last_idx, model_col])
                if rmse_val == last_rmse and model_val == last_model:
                    last_group.append(idx)
                else:
                    groups.append([idx])

        offset = 0
        for group in groups:
            assigned_rank = r_k + offset
            for idx in group:
                df.loc[idx, "rank"] = assigned_rank
            offset += len(group)

        r_k += len(tier_indices)
        unranked -= set(tier_indices)

    return df


def select_with_epsilon_tiebreak(
    results_df: pd.DataFrame,
    epsilon: float = 1e-4,
    valid_status: str = ModelTerminalStatus.VALIDATION_SUCCESS.value,
) -> Dict[str, Any]:
    """Decision M4: Select winner using candidate-tier procedure on Pooled MAE (Protocol v1.0.20 §9.3.D).

    Primary metric: Pooled MAE (with epsilon = 1e-4 mm equivalence band).
    Tie-break hierarchy within Band C = {m : MAE(m) <= min MAE + epsilon}:
    1. Lowest unrounded IEEE 754 RMSE
    2. Lexicographically smallest model ID
    """
    if results_df.empty:
        return {"selected_model": None, "status": "NO_CANDIDATES", "equivalence_band": []}

    df = results_df.copy()
    if "status" in df.columns:
        df = df[df["status"].isin([valid_status, ExecutionStatus.SUCCESS.value, "COMPLETED"])]

    mae_col = "MAE" if "MAE" in df.columns else "CR-MAE@7"
    rmse_col = "RMSE" if "RMSE" in df.columns else None

    cols_to_check = [mae_col]
    if rmse_col:
        cols_to_check.append(rmse_col)
    df = df.dropna(subset=cols_to_check)
    if df.empty:
        return {"selected_model": None, "status": "NO_ELIGIBLE_MODELS", "equivalence_band": []}

    model_col = "model" if "model" in df.columns else "model_id"
    min_mae = float(df[mae_col].min())
    band_df = df[df[mae_col] <= (min_mae + float(epsilon))].copy()

    sort_cols = []
    if rmse_col:
        sort_cols.append(rmse_col)
    sort_cols.append(model_col)

    band_df = band_df.sort_values(
        by=sort_cols,
        ascending=[True] * len(sort_cols),
        kind="mergesort",
    )
    winner_row = band_df.iloc[0]
    return {
        "selected_model": str(winner_row[model_col]),
        "selected_group": str(winner_row.get("group", "")),
        "min_mae": min_mae,
        "selected_mae": float(winner_row[mae_col]),
        "selected_rmse": float(winner_row[rmse_col]) if rmse_col else 0.0,
        "epsilon": float(epsilon),
        "equivalence_band": [str(m) for m in band_df[model_col].tolist()],
        "status": "SELECTED",
    }


def select_neural_context_window(
    metrics_w30: Dict[str, float],
    metrics_w60: Dict[str, float],
    epsilon: float = 1e-4,
    m6a_approved: bool = True,
) -> Dict[str, Any]:
    """Decision M6 & M6a: Select neural context window W* in {30, 60} on Validation.

    Per Decision M6a and Protocol v1.0.20, W=30 is permanently adopted.
    Compares on MAE (within epsilon=1e-4 mm) -> exact RMSE -> W=30 parsimony.
    """
    mae30 = float(metrics_w30.get("MAE", metrics_w30.get("CR-MAE@7", 0.0)))
    mae60 = float(metrics_w60.get("MAE", metrics_w60.get("CR-MAE@7", 0.0)))
    rmse30 = float(metrics_w30.get("RMSE", 0.0))
    rmse60 = float(metrics_w60.get("RMSE", 0.0))

    if abs(mae30 - mae60) > float(epsilon):
        w_star = 30 if mae30 < mae60 else 60
        return {
            "selected_window": w_star,
            "status": "SELECTED",
            "tie_break_reason": "PRIMARY_MAE",
        }

    # Within epsilon band on MAE: check exact numerical inequality on RMSE
    if rmse30 != rmse60:
        w_star = 30 if rmse30 < rmse60 else 60
        return {
            "selected_window": w_star,
            "status": "SELECTED",
            "tie_break_reason": "SECONDARY_EXACT_RMSE",
        }

    # Exact numerical tie across all criteria
    if m6a_approved:
        return {
            "selected_window": 30,
            "status": "SELECTED",
            "tie_break_reason": "M6a_PARSIMONY_W30",
        }
    return {
        "selected_window": None,
        "status": "TIE_BREAK_PENDING_USER_DECISION",
        "tie_break_reason": "EXACT_TIE_ALL_CRITERIA",
    }


def diebold_mariano_hac_test(
    loss_champion: Sequence[float],
    loss_competitor: Sequence[float],
    max_lag: int = 7,
) -> Dict[str, float]:
    """Decision M5: Diebold-Mariano test with Newey-West HAC Bartlett kernel variance (lag L=7).

    Defines loss differential d_t = L(champion)_t - L(competitor)_t (negative means champion has lower loss).
    """
    l_champ = np.asarray(loss_champion, dtype=np.float64).ravel()
    l_comp = np.asarray(loss_competitor, dtype=np.float64).ravel()
    if len(l_champ) != len(l_comp) or len(l_champ) < 2:
        raise ValueError("Loss series must have equal length >= 2.")

    d = l_champ - l_comp
    n = len(d)
    d_bar = float(np.mean(d))
    d_centered = d - d_bar

    gamma_0 = float(np.sum(d_centered * d_centered) / n)
    hac_var = gamma_0
    L = min(int(max_lag), n - 1)
    for k in range(1, L + 1):
        weight = 1.0 - (k / float(L + 1))
        gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / n)
        hac_var += 2.0 * weight * gamma_k

    # Guard against numerical non-positivity in degenerate constant series
    if hac_var <= 1e-18:
        if abs(d_bar) <= 1e-15:
            return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "hac_se": 0.0}
        dm_stat = float(np.sign(d_bar) * 1e6)
        return {"dm_stat": dm_stat, "p_value": 0.0, "mean_loss_diff": d_bar, "hac_se": 0.0}

    se = float(np.sqrt(hac_var / n))
    dm_stat = float(d_bar / se)
    p_val = float(2.0 * stats.norm.sf(abs(dm_stat)))
    return {
        "dm_stat": dm_stat,
        "p_value": p_val,
        "mean_loss_diff": d_bar,
        "hac_se": se,
    }


def moving_block_bootstrap_ci(
    diff_series: Sequence[float],
    block_length: int = 14,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Decision M5: Moving Block Bootstrap (b=14) 95% confidence interval for mean difference."""
    d = np.asarray(diff_series, dtype=np.float64).ravel()
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    b = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / float(b)))
    max_start = n - b
    rng = np.random.default_rng(random_state)

    if max_start <= 0:
        return (float(np.mean(d)), float(np.mean(d)))

    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + b) for s in starts[i]])[:n]
        boot_means[i] = np.mean(d[idx])

    lower = float(np.quantile(boot_means, alpha / 2.0))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return (lower, upper)


def run_confirmatory_dm_inference(
    champion_id: str,
    champion_status: str,
    planned_representatives: List[Dict[str, str]],
    model_test_statuses: Dict[str, str],
    origin_losses: Dict[str, np.ndarray],
    alpha: float = 0.05,
    hac_lag: int = 7,
    bootstrap_block: int = 14,
    n_boot: int = 500,
    system_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Decision M5 & P5-1: Execute Confirmatory DM + Conservative Holm Step-Down Inference.

    Enforces:
    1. Blocks execution if system_status == TEST_EVALUATION_INTERRUPTED.
    2. If Core Champion failed on Test (FAILED_TEST), returns CONFIRMATORY_INFERENCE_UNAVAILABLE.
    3. Preserves frozen planned_representatives (never replaces a FAILED_TEST representative).
    4. Applies conservative Holm step-down denominator using K_planned:
           alpha_j = alpha / (K_planned - j + 1)  for j = 1 .. K_eval.
    """
    if system_status == SystemEvaluationStatus.TEST_EVALUATION_INTERRUPTED.value:
        raise RuntimeError(
            "Confirmatory statistical inference is strictly prohibited while system status is "
            "TEST_EVALUATION_INTERRUPTED."
        )

    k_planned = len(planned_representatives)
    if k_planned > 5:
        raise ValueError(f"P3-1 Violation: K_planned ({k_planned}) exceeds maximum bound of 5.")

    if champion_status != ModelTerminalStatus.TEST_SUCCESS.value:
        return {
            "inference_module_status": InferenceModuleStatus.CONFIRMATORY_INFERENCE_UNAVAILABLE.value,
            "champion_id": champion_id,
            "k_planned": k_planned,
            "k_eval": 0,
            "k_failed": k_planned,
            "comparisons": [],
        }

    if k_planned == 0:
        return {
            "inference_module_status": InferenceModuleStatus.NO_CONFIRMATORY_PAIRS_AVAILABLE.value,
            "champion_id": champion_id,
            "k_planned": 0,
            "k_eval": 0,
            "k_failed": 0,
            "comparisons": [],
        }

    champ_loss = np.asarray(origin_losses[champion_id], dtype=np.float64).ravel()
    evaluable_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []

    for rep in planned_representatives:
        rep_id = str(rep["model"])
        rep_group = str(rep.get("group", ""))
        rep_status = model_test_statuses.get(rep_id, ModelTerminalStatus.FAILED_TEST.value)

        if rep_status == ModelTerminalStatus.TEST_SUCCESS.value and rep_id in origin_losses:
            comp_loss = np.asarray(origin_losses[rep_id], dtype=np.float64).ravel()
            dm_res = diebold_mariano_hac_test(champ_loss, comp_loss, max_lag=hac_lag)
            ci_low, ci_high = moving_block_bootstrap_ci(
                champ_loss - comp_loss,
                block_length=bootstrap_block,
                n_boot=n_boot,
                alpha=alpha,
            )
            evaluable_records.append({
                "competitor_model": rep_id,
                "competitor_group": rep_group,
                "competitor_status": rep_status,
                "evaluable": True,
                "dm_stat": dm_res["dm_stat"],
                "raw_p_value": dm_res["p_value"],
                "mean_loss_diff": dm_res["mean_loss_diff"],
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
            })
        else:
            failed_records.append({
                "competitor_model": rep_id,
                "competitor_group": rep_group,
                "competitor_status": rep_status,
                "evaluable": False,
                "dm_stat": np.nan,
                "raw_p_value": np.nan,
                "holm_threshold": np.nan,
                "holm_adjusted_p_value": np.nan,
                "reject_null": False,
                "mean_loss_diff": np.nan,
                "ci_95_lower": np.nan,
                "ci_95_upper": np.nan,
                "note": "COMPETITOR_FAILED_TEST_NOT_REPLACED",
            })

    k_eval = len(evaluable_records)
    k_failed = len(failed_records)

    # Sort evaluable comparisons by raw_p_value ascending and apply conservative Holm with K_planned
    evaluable_records.sort(key=lambda r: (float(r["raw_p_value"]), str(r["competitor_model"])))
    still_rejecting = True
    running_max_adj_p = 0.0
    for idx, rec in enumerate(evaluable_records):
        j = idx + 1  # 1-indexed rank among evaluable comparisons
        denom = float(k_planned - j + 1)
        holm_thresh = float(alpha / denom)
        adj_p = min(1.0, max(running_max_adj_p, float(rec["raw_p_value"]) * denom))
        running_max_adj_p = adj_p
        if still_rejecting and float(rec["raw_p_value"]) <= holm_thresh:
            reject = True
        else:
            still_rejecting = False
            reject = False
        rec["holm_rank_j"] = j
        rec["holm_denominator"] = int(k_planned - j + 1)
        rec["holm_threshold"] = holm_thresh
        rec["holm_adjusted_p_value"] = adj_p
        rec["reject_null"] = reject

    if k_eval == 0:
        inf_status = InferenceModuleStatus.NO_EVALUABLE_CONFIRMATORY_PAIRS.value
    elif k_eval < k_planned:
        inf_status = InferenceModuleStatus.CONFIRMATORY_INFERENCE_PARTIALLY_EVALUABLE.value
    else:
        inf_status = InferenceModuleStatus.CONFIRMATORY_INFERENCE_COMPLETE.value

    return {
        "inference_module_status": inf_status,
        "champion_id": champion_id,
        "k_planned": k_planned,
        "k_eval": k_eval,
        "k_failed": k_failed,
        "comparisons": evaluable_records + failed_records,
    }


# ======================================================================
# RollingBenchmarkCoordinator (§2.2, §2.3.C, Phase 6.3, P5-2, P6-1)
# ======================================================================

class RollingBenchmarkCoordinator:
    """Coordinator for Validation and Final Test Rolling-Origin Evaluation (Plan v1.4.18).

    Enforces:
    1. Exhaustive 6-trigger Model-Scoped Failure handling:
       - Caught runtime exception / CUDA OOM / per-model timeout
       - Shape mismatch != (N_origins, H)
       - Non-finite values (NaN / Inf)
       - Negative predictions (< 0.0 mm)
       - Origin date / horizon misalignment
       - Incomplete origin coverage (< N_origins)
       -> Terminal status FAILED_TEST (or FAILED_VALIDATION), metrics NaN, official forecast array
          np.empty((0, H), dtype=np.float32), partial predictions quarantined to
          test_diagnostic_partial_forecasts.npz, coordinator continues to next model.
    2. Process-Scoped Interruption disambiguation (P6-1):
       - Unhandled coordinator/process interruption (ProcessScopedInterruption / KeyboardInterrupt)
         leaves in-flight model as INCOMPLETE_UNRECORDED (NEVER marked FAILED_TEST), sets
         system_status = TEST_EVALUATION_INTERRUPTED, persists atomic per-model checkpoint,
         and blocks confirmatory inference until resumed and completed.
    """

    def __init__(
        self,
        expected_origin_dates: Sequence[Any],
        horizon: int = 7,
        rain_threshold: float = 1.0,
        stage: str = "test",
        checkpoint_path: Optional[Path] = None,
        diagnostic_partial_npz_path: Optional[Path] = None,
        manifest_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.expected_origin_dates = pd.to_datetime(pd.Series(expected_origin_dates)).reset_index(drop=True)
        self.n_origins = len(self.expected_origin_dates)
        self.horizon = int(horizon)
        self.rain_threshold = float(rain_threshold)
        self.stage = stage.lower()
        self.checkpoint_path = checkpoint_path
        self.diagnostic_partial_npz_path = diagnostic_partial_npz_path
        self.manifest_payload = manifest_payload or {}
        self.manifest_sha256 = self._compute_manifest_hash(self.manifest_payload)

        self.model_statuses: Dict[str, str] = {}
        self.results_records: Dict[str, pd.DataFrame] = {}
        self.official_forecasts: Dict[str, np.ndarray] = {}
        self.diagnostic_partial_forecasts: Dict[str, np.ndarray] = {}
        self.system_status: Optional[str] = None

    @staticmethod
    def _compute_manifest_hash(payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _success_status(self) -> ExecutionStatus:
        return (
            ExecutionStatus.TEST_SUCCESS
            if self.stage == "test"
            else ExecutionStatus.VALIDATION_SUCCESS
        )

    def _failure_status(self) -> ExecutionStatus:
        return (
            ExecutionStatus.FAILED_TEST
            if self.stage == "test"
            else ExecutionStatus.FAILED_VALIDATION
        )

    def _validate_prediction_contract(
        self,
        raw_output: Any,
    ) -> Tuple[bool, Optional[np.ndarray], str, str]:
        """Validate the 5 numerical/structural output conditions (Triggers 2-6 of §2.2 / P5-2)."""
        pred_dates = None
        pred_horizons = None
        if isinstance(raw_output, dict):
            preds = raw_output.get("predictions", None)
            pred_dates = raw_output.get("origin_dates", None)
            pred_horizons = raw_output.get("horizons", None)
        elif isinstance(raw_output, tuple) and len(raw_output) >= 2:
            preds = raw_output[0]
            pred_dates = raw_output[1]
            if len(raw_output) >= 3:
                pred_horizons = raw_output[2]
        else:
            preds = raw_output

        if preds is None:
            return False, None, "Trigger 2: Forecast output is None", "OutputShapeError"

        arr = np.asarray(preds, dtype=np.float64)

        # Trigger 6: Incomplete origin coverage (< N_origins)
        if arr.ndim >= 1 and arr.shape[0] < self.n_origins:
            return (
                False,
                arr,
                f"Trigger 6: Incomplete origin coverage ({arr.shape[0]} < {self.n_origins})",
                "IncompleteOriginCoverageError",
            )

        # Trigger 2: Shape mismatch != (N_origins, H)
        if arr.shape != (self.n_origins, self.horizon):
            return (
                False,
                arr,
                f"Trigger 2: Forecast shape {arr.shape} != expected ({self.n_origins}, {self.horizon})",
                "OutputShapeError",
            )

        # Trigger 5: Date or horizon index misalignment
        if pred_dates is not None:
            try:
                p_dates = pd.to_datetime(pd.Series(pred_dates)).reset_index(drop=True)
                if len(p_dates) != self.n_origins or not (p_dates == self.expected_origin_dates).all():
                    return (
                        False,
                        arr,
                        "Trigger 5: Origin date index misalignment against canonical split origins",
                        "OriginDateAlignmentError",
                    )
            except Exception as e:
                return False, arr, f"Trigger 5: Invalid origin_dates metadata ({e})", "OriginDateAlignmentError"

        if pred_horizons is not None:
            expected_h = list(range(1, self.horizon + 1))
            if list(pred_horizons) != expected_h:
                return (
                    False,
                    arr,
                    f"Trigger 5: Horizon index misalignment {list(pred_horizons)} != {expected_h}",
                    "HorizonAlignmentError",
                )

        # Trigger 3: NaN or Inf
        if not np.all(np.isfinite(arr)):
            return (
                False,
                arr,
                "Trigger 3: Forecast array contains NaN or Inf values",
                "NumericalNonFiniteError",
            )

        # Trigger 4: Negative rainfall prediction (< 0.0 mm)
        if np.any(arr < 0.0):
            min_val = float(np.min(arr))
            return (
                False,
                arr,
                f"Trigger 4: Forecast array contains negative value ({min_val:.6f} < 0.0 mm)",
                "NegativeForecastError",
            )

        return True, arr, "", ""

    def _persist_checkpoint(self) -> None:
        """Atomically persist per-model terminal statuses and system status."""
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        rows_serialized = {}
        for m_name, df_rec in self.results_records.items():
            rows_serialized[m_name] = df_rec.to_dict(orient="records")[0]

        payload = {
            "manifest_sha256": self.manifest_sha256,
            "stage": self.stage,
            "n_origins": self.n_origins,
            "horizon": self.horizon,
            "system_status": self.system_status,
            "model_statuses": self.model_statuses,
            "results_records": rows_serialized,
        }
        tmp_path = self.checkpoint_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.checkpoint_path)

    def _persist_diagnostic_partial_npz(self) -> None:
        """Save quarantined partial/invalid predictions to test_diagnostic_partial_forecasts.npz."""
        if self.diagnostic_partial_npz_path is None or not self.diagnostic_partial_forecasts:
            return
        self.diagnostic_partial_npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"partial_{k}": np.asarray(v, dtype=np.float32)
            for k, v in self.diagnostic_partial_forecasts.items()
        }
        np.savez_compressed(self.diagnostic_partial_npz_path, **payload)

    def evaluate_models(
        self,
        model_specs: List[Dict[str, Any]],
        y_true: np.ndarray,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate all eligible models with strict model-scoped vs process-scoped separation.

        Each item in `model_specs` is a dict:
            {"model": str, "group": str, "strategy": str, "predict_fn": Callable[[], Any]}
        """
        yt = np.asarray(y_true, dtype=np.float64)
        if yt.shape != (self.n_origins, self.horizon):
            raise ValueError(
                f"y_true shape {yt.shape} does not match expected ({self.n_origins}, {self.horizon})"
            )

        if resume and self.checkpoint_path is not None and self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            if ckpt.get("manifest_sha256") != self.manifest_sha256:
                raise ValueError(
                    "Checkpoint manifest_sha256 mismatch! Cannot resume with a modified Validation Freeze Manifest."
                )
            for m_name, st in ckpt.get("model_statuses", {}).items():
                if st in (
                    ModelTerminalStatus.TEST_SUCCESS.value,
                    ModelTerminalStatus.FAILED_TEST.value,
                    ModelTerminalStatus.VALIDATION_SUCCESS.value,
                    ModelTerminalStatus.FAILED_VALIDATION.value,
                ):
                    self.model_statuses[m_name] = st
                    rec_dict = ckpt["results_records"][m_name]
                    self.results_records[m_name] = pd.DataFrame([rec_dict])[UNIFIED_SCHEMA]

        # Initialize any unrecorded models as INCOMPLETE_UNRECORDED
        for spec in model_specs:
            m_name = str(spec["model"])
            if m_name not in self.model_statuses:
                self.model_statuses[m_name] = ModelTerminalStatus.INCOMPLETE_UNRECORDED.value

        try:
            for spec in model_specs:
                m_name = str(spec["model"])
                grp = str(spec.get("group", "Unknown"))
                strat = str(spec.get("strategy", "direct"))
                predict_fn: Callable[[], Any] = spec["predict_fn"]

                # Skip models that already have a recorded terminal status during checkpoint resume
                if self.model_statuses.get(m_name) in (
                    ModelTerminalStatus.TEST_SUCCESS.value,
                    ModelTerminalStatus.FAILED_TEST.value,
                    ModelTerminalStatus.VALIDATION_SUCCESS.value,
                    ModelTerminalStatus.FAILED_VALIDATION.value,
                ):
                    continue

                try:
                    raw_out = predict_fn()
                    is_valid, arr_or_partial, fail_reason, exc_type = self._validate_prediction_contract(raw_out)

                    if is_valid and arr_or_partial is not None:
                        succ_status = self._success_status()
                        df_rec = compute_benchmark_metrics(
                            y_true=yt,
                            y_pred=arr_or_partial,
                            model_name=m_name,
                            group=grp,
                            strategy=strat,
                            status=succ_status,
                            rain_threshold=self.rain_threshold,
                        )
                        self.model_statuses[m_name] = succ_status.value
                        self.results_records[m_name] = df_rec
                        self.official_forecasts[m_name] = np.asarray(arr_or_partial, dtype=np.float32)
                    else:
                        fail_status = self._failure_status()
                        df_rec = compute_benchmark_metrics(
                            y_true=None,
                            y_pred=None,
                            model_name=m_name,
                            group=grp,
                            strategy=strat,
                            status=fail_status,
                            failure_reason=fail_reason,
                            exception_type=exc_type,
                            rain_threshold=self.rain_threshold,
                        )
                        self.model_statuses[m_name] = fail_status.value
                        self.results_records[m_name] = df_rec
                        self.official_forecasts[m_name] = np.empty((0, self.horizon), dtype=np.float32)
                        if arr_or_partial is not None:
                            self.diagnostic_partial_forecasts[m_name] = np.asarray(
                                arr_or_partial, dtype=np.float32
                            )
                            self._persist_diagnostic_partial_npz()

                except Exception as model_exc:
                    # Trigger 1: Caught model-scoped exception (OOM, timeout, numerical runtime error)
                    fail_status = self._failure_status()
                    df_rec = compute_benchmark_metrics(
                        y_true=None,
                        y_pred=None,
                        model_name=m_name,
                        group=grp,
                        strategy=strat,
                        status=fail_status,
                        failure_reason=f"Trigger 1: Caught model-scoped error: {model_exc}",
                        exception_type=type(model_exc).__name__,
                        rain_threshold=self.rain_threshold,
                    )
                    self.model_statuses[m_name] = fail_status.value
                    self.results_records[m_name] = df_rec
                    self.official_forecasts[m_name] = np.empty((0, self.horizon), dtype=np.float32)

                # Atomic checkpoint after each model reaches its terminal status
                self._persist_checkpoint()

        except (ProcessScopedInterruption, KeyboardInterrupt, SystemExit) as proc_exc:
            # Process-scoped interruption: in-flight model remains INCOMPLETE_UNRECORDED
            self.system_status = SystemEvaluationStatus.TEST_EVALUATION_INTERRUPTED.value
            self._persist_checkpoint()
            return {
                "system_status": self.system_status,
                "interruption_type": type(proc_exc).__name__,
                "interruption_reason": str(proc_exc),
                "model_statuses": dict(self.model_statuses),
                "results_df": (
                    pd.concat(list(self.results_records.values()), ignore_index=True)
                    if self.results_records
                    else pd.DataFrame(columns=UNIFIED_SCHEMA)
                ),
                "official_forecasts": self.official_forecasts,
                "diagnostic_partial_forecasts": self.diagnostic_partial_forecasts,
            }

        # All models have recorded terminal statuses -> determine mutually exclusive system status
        succ_val = self._success_status().value
        any_success = any(st == succ_val for st in self.model_statuses.values())
        if any_success:
            self.system_status = SystemEvaluationStatus.TEST_EVALUATION_COMPLETE.value
        else:
            self.system_status = SystemEvaluationStatus.TEST_EVALUATION_FAILED.value

        self._persist_checkpoint()
        ordered_dfs = [self.results_records[ str(s["model"]) ] for s in model_specs if str(s["model"]) in self.results_records]
        results_df = pd.concat(ordered_dfs, ignore_index=True) if ordered_dfs else pd.DataFrame(columns=UNIFIED_SCHEMA)

        return {
            "system_status": self.system_status,
            "model_statuses": dict(self.model_statuses),
            "results_df": results_df,
            "official_forecasts": self.official_forecasts,
            "diagnostic_partial_forecasts": self.diagnostic_partial_forecasts,
        }


# ======================================================================
# UnifiedBenchmark (Backward-Compatible Multi-Group Runner)
# ======================================================================

class UnifiedBenchmark:
    """Compare model groups on a temporal split conforming to evaluation-standards.md."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rain_threshold: float = 1.0,
        horizon: int = 7,
        allow_legacy_recursive: bool = False,
    ):
        from src.models.nixtla import to_nixtla_format

        self.raw_train_df = train_df.copy()
        self.raw_test_df = test_df.copy()
        train_nixtla = (
            to_nixtla_format(train_df)
            if not {"unique_id", "ds", "y"}.issubset(train_df.columns)
            else train_df.copy()
        )
        test_nixtla = (
            to_nixtla_format(test_df)
            if not {"unique_id", "ds", "y"}.issubset(test_df.columns)
            else test_df.copy()
        )
        self.train_df = train_nixtla[["unique_id", "ds", "y"]].copy()
        self.test_df = test_nixtla[["unique_id", "ds", "y"]].copy()
        self.rain_threshold = float(rain_threshold)
        self.horizon = int(horizon)
        self.allow_legacy_recursive = bool(allow_legacy_recursive)

    def run_group_statistical(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
        legacy_groups: bool = False,
    ) -> pd.DataFrame:
        """Benchmark Group 1: Statistical, Intermittent & Naive baselines."""
        from src.training.trainer import build_stats_adapter_from_eda
        from src.models.nixtla.defaults import get_statistical_models

        chosen_models = models if models is not None else get_statistical_models(eda_report)
        sf_adapter = build_stats_adapter_from_eda(eda_report=eda_report, models=chosen_models)
        sf_adapter.fit(self.train_df)
        preds_df = sf_adapter.predict(h=self.horizon)

        y_test = self.test_df["y"].values[: self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ["unique_id", "ds"]:
                continue
            if legacy_groups:
                if col in ["Naive", "SeasonalNaive"]:
                    grp = "1. Naive Baselines"
                elif col in ["CrostonOptimized", "CrostonSBA", "TSB", "ADIDA", "IMAPA"]:
                    grp = "3a. Intermittent Baselines"
                else:
                    grp = "3b. Classical TS Baselines"
            else:
                grp = "1. Statistical"

            preds = preds_df[col].values[: len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group=grp))

        return pd.DataFrame(results)

    def run_group_ml_direct(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 2: ML Direct Multi-Horizon (Linear, RF, XGB, LGBM - Standard loss)."""
        from src.training.trainer import build_ml_direct_adapter_from_eda
        from src.models.nixtla.defaults import get_ml_direct_models

        chosen_models = models if models is not None else get_ml_direct_models(eda_report)
        ml_adapter = build_ml_direct_adapter_from_eda(
            eda_report=eda_report, models=chosen_models, max_horizon=self.horizon
        )
        ml_adapter.fit(self.train_df)
        preds_df = ml_adapter.predict(h=self.horizon)

        y_test = self.test_df["y"].values[: self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ["unique_id", "ds"]:
                continue
            preds = preds_df[col].values[: len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group="2. ML Direct"))

        return pd.DataFrame(results)

    def run_group_ml_hurdle(
        self,
        hurdle_models: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 3: ML Hurdle / Two-Stage."""
        from src.models.nixtla.defaults import get_ml_hurdle_models

        models = hurdle_models or get_ml_hurdle_models(classification_threshold=self.rain_threshold)
        y_test = self.test_df["y"].values[: self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for name, model in models.items():
            try:
                preds = self._predict_legacy(model, feature_builder, self.horizon)
                if preds is not None and len(preds) > 0:
                    preds = preds[: len(y_test)]
                    metrics = self._compute_metrics(
                        y_test, preds, horizons[: len(preds)], name, group="3. ML Hurdle"
                    )
                    results.extend(metrics)
            except RuntimeError:
                raise
            except Exception:
                pass

        return pd.DataFrame(results)

    def run_group_ml_tweedie(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 4: ML Tweedie Single-Stage (LGBM-Tweedie, XGBoost-Tweedie)."""
        from src.training.trainer import build_ml_tweedie_adapter_from_eda
        from src.models.nixtla.defaults import get_ml_tweedie_models

        chosen_models = models if models is not None else get_ml_tweedie_models(eda_report)
        ml_adapter = build_ml_tweedie_adapter_from_eda(
            eda_report=eda_report, models=chosen_models, max_horizon=self.horizon
        )
        ml_adapter.fit(self.train_df)
        preds_df = ml_adapter.predict(h=self.horizon)

        y_test = self.test_df["y"].values[: self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ["unique_id", "ds"]:
                continue
            preds = preds_df[col].values[: len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group="4. ML Tweedie"))

        return pd.DataFrame(results)

    def run_group_neural_direct(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
        max_steps: int = 100,
    ) -> pd.DataFrame:
        """Benchmark Group 5: Neural Direct (LSTM baseline, TCN, NHITS, PatchTST, TFT)."""
        from src.training.trainer import build_neural_direct_adapter_from_eda
        from src.models.nixtla.defaults import get_neural_direct_models

        chosen_models = (
            models
            if models is not None
            else get_neural_direct_models(
                horizon=self.horizon,
                max_steps=max_steps,
                eda_report=eda_report,
            )
        )
        neural_adapter = build_neural_direct_adapter_from_eda(
            eda_report=eda_report,
            models=chosen_models,
            horizon=self.horizon,
            max_steps=max_steps,
        )

        results = []
        if neural_adapter.is_available:
            try:
                neural_adapter.fit(self.train_df)
                preds_df = neural_adapter.predict(h=self.horizon)
                y_test = self.test_df["y"].values[: self.horizon]
                horizons = list(range(1, len(y_test) + 1))

                for col in preds_df.columns:
                    if col in ["unique_id", "ds"]:
                        continue
                    preds = preds_df[col].values[: len(y_test)]
                    metrics = self._compute_metrics(
                        y_test, preds, horizons, col, group="5. Neural Direct"
                    )
                    results.extend(metrics)
            except Exception:
                pass

        return pd.DataFrame(results)

    @staticmethod
    def get_best_model_per_group(
        results_df: pd.DataFrame,
        metric: str = "MAE",
    ) -> pd.DataFrame:
        """Select best model (lowest metric) in each model_group."""
        if results_df.empty:
            return pd.DataFrame()
        agg_dict = {
            "MAE": "mean",
            "RMSE": "mean",
        }
        if "MSE" in results_df.columns:
            agg_dict["MSE"] = "mean"
        if "MASE" in results_df.columns:
            agg_dict["MASE"] = "mean"
        if "rain_day_MAE" in results_df.columns:
            agg_dict["rain_day_MAE"] = "mean"
        if "rain_day_RMSE" in results_df.columns:
            agg_dict["rain_day_RMSE"] = "mean"

        summary = results_df.groupby(["model_group", "model_name"]).agg(agg_dict).reset_index()
        summary = summary.sort_values(["model_group", metric], ascending=[True, True])
        best_df = summary.groupby("model_group").first().reset_index()
        return best_df

    def run(
        self,
        eda_report: Optional[Any] = None,
        legacy_models: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
        include_statistical: bool = True,
        include_ml_direct: bool = True,
        include_ml_hurdle: bool = True,
        include_ml_tweedie: bool = True,
        include_neural_direct: bool = True,
        neural_max_steps: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute independent multi-group benchmark."""
        legacy_mode = any(
            k in kwargs for k in ["include_statsforecast", "include_mlforecast", "include_neuralforecast"]
        )
        if "include_statsforecast" in kwargs:
            include_statistical = kwargs.pop("include_statsforecast")
            include_ml_direct = False
            include_ml_hurdle = False
        if "include_mlforecast" in kwargs:
            include_ml_tweedie = kwargs.pop("include_mlforecast")
        if "include_neuralforecast" in kwargs:
            include_neural_direct = kwargs.pop("include_neuralforecast")

        all_dfs = []
        if include_statistical:
            try:
                all_dfs.append(self.run_group_statistical(eda_report, legacy_groups=legacy_mode))
            except Exception:
                pass
        if include_ml_direct:
            try:
                all_dfs.append(self.run_group_ml_direct(eda_report))
            except Exception:
                pass
        if include_ml_hurdle:
            try:
                all_dfs.append(self.run_group_ml_hurdle(legacy_models, feature_builder))
            except Exception:
                pass
        if include_ml_tweedie:
            try:
                all_dfs.append(self.run_group_ml_tweedie(eda_report))
            except Exception:
                pass
        if include_neural_direct:
            try:
                all_dfs.append(self.run_group_neural_direct(eda_report, max_steps=neural_max_steps))
            except Exception:
                pass

        return (
            pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
            if all_dfs
            else pd.DataFrame()
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizons: List[int],
        model_name: str,
        group: str = "Model",
    ) -> List[Dict[str, Any]]:
        """Compute MAE/MSE/RMSE/MASE per horizon and auxiliary rain-day metrics (>= 1.0 mm)."""
        rows = []
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)

        for h in sorted(set(horizons)):
            mask = [i for i, hh in enumerate(horizons) if hh == h]
            if not mask:
                continue

            yt = y_true_arr[mask]
            yp = y_pred_arr[mask]

            mae = float(mean_absolute_error(yt, yp))
            mse = float(mean_squared_error(yt, yp))
            rmse = float(np.sqrt(mse))
            mase = float(mean_absolute_scaled_error(yt, yp, scale=CANONICAL_DEV_TRAIN_MASE_SCALE))

            rain_mask = yt >= self.rain_threshold
            if np.any(rain_mask):
                rain_mae = float(mean_absolute_error(yt[rain_mask], yp[rain_mask]))
                rain_rmse = float(np.sqrt(mean_squared_error(yt[rain_mask], yp[rain_mask])))
            else:
                rain_mae = np.nan
                rain_rmse = np.nan

            rows.append({
                "model_group": group,
                "model_name": model_name,
                "horizon": h,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MASE": round(mase, 4),
                "rain_day_MAE": round(rain_mae, 4) if not np.isnan(rain_mae) else np.nan,
                "rain_day_RMSE": round(rain_rmse, 4) if not np.isnan(rain_rmse) else np.nan,
                "n_samples": len(mask),
                "y_true": round(float(yt[0]), 4) if len(yt) > 0 else np.nan,
                "y_pred": round(float(yp[0]), 4) if len(yp) > 0 else np.nan,
            })
        return rows

    def _predict_legacy(
        self,
        model: Any,
        feature_builder: Optional[Any],
        steps: int,
    ) -> Optional[np.ndarray]:
        """Generate forecasts from DirectHurdleForecaster or legacy model.

        Per Phase 0 Gate 6, RecursiveForecaster is strictly isolated from official benchmark execution
        unless `allow_legacy_recursive=True` is explicitly set for legacy unit tests.
        """
        if hasattr(model, "forecast"):
            return np.asarray(model.forecast(steps=steps), dtype=np.float64)

        if feature_builder is not None:
            if not self.allow_legacy_recursive:
                raise RuntimeError(
                    "Phase 0 Gate 6 Isolation: RecursiveForecaster is prohibited in official direct "
                    "multi-horizon benchmarks. Use DirectHurdleForecaster instead."
                )
            from src.models.legacy.recursive import RecursiveForecaster

            forecaster = RecursiveForecaster(model, feature_builder)
            raw_train = getattr(self, "raw_train_df", self.train_df)
            history_df = raw_train.iloc[-60:] if len(raw_train) >= 60 else raw_train
            forecast_df = forecaster.forecast(history_df, steps=steps)
            if "prediction" in forecast_df.columns:
                return forecast_df["prediction"].values

        return None


UnifiedBenchmarkRunner = UnifiedBenchmark
