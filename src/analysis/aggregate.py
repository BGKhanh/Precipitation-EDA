"""Multi-Dimensional Aggregations, Horizon Decompositions, and Seasonal Breakdown.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and Project Rules:
- evaluation-standards.md
- data-leakage-prevention.md
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.analysis.integrity import assert_no_purged_metrics


def compute_family_aggregates(df_results: pd.DataFrame) -> pd.DataFrame:
    """Compute Macro-average and Best-in-Family aggregations across the 7 canonical families.
    
    Provides two complementary perspectives per CR-10:
    1. Macro-average: Arithmetic mean of MAE, RMSE, MASE, training_time, inference_time.
    2. Best-in-family: Performance of the single best scenario (minimum Pooled MAE) in that family.
    """
    assert_no_purged_metrics(df_results)
    df = df_results.copy()

    records = []
    for fam, grp in df.groupby("family", sort=False):
        n_scenarios = len(grp)
        macro_mae = grp["MAE"].mean()
        macro_rmse = grp["RMSE"].mean()
        macro_mase = grp["MASE"].mean()

        # Best in family
        best_row = grp.sort_values(by=["MAE", "RMSE"]).iloc[0]
        best_scenario = best_row["scenario_id"]
        best_model = best_row["model"]
        best_mae = best_row["MAE"]
        best_rmse = best_row["RMSE"]
        best_mase = best_row["MASE"]

        records.append({
            "family": fam,
            "n_scenarios": n_scenarios,
            "macro_mae": macro_mae,
            "macro_rmse": macro_rmse,
            "macro_mase": macro_mase,
            "best_scenario_id": best_scenario,
            "best_model": best_model,
            "best_mae": best_mae,
            "best_rmse": best_rmse,
            "best_mase": best_mase,
        })

    df_family = pd.DataFrame(records)
    # Order by best_mae ascending
    return df_family.sort_values("best_mae").reset_index(drop=True)


def compute_feature_set_ablation(df_results: pd.DataFrame) -> pd.DataFrame:
    """Analyze the marginal contribution of feature sets F0 -> F1 -> F2 -> F3.
    
    Covers Tabular ML Direct (4 algorithms), Tabular ML Tweedie (2 algorithms),
    and Hurdle (4 base families x 2 strategies).
    Computes delta MAE across successive feature sets.
    """
    assert_no_purged_metrics(df_results)
    df = df_results[df_results["feature_set"].isin(["F0", "F1", "F2", "F3"])].copy()

    records = []
    # Group by algorithm/architecture family
    for (fam, strat), grp in df.groupby(["family", "strategy"]):
        # Extract base algorithm name
        for model_prefix in grp["model"].apply(lambda m: m.split("_F")[0]).unique():
            sub = grp[grp["model"].str.startswith(model_prefix + "_F")].set_index("feature_set")
            if not all(f in sub.index for f in ["F0", "F1", "F2", "F3"]):
                continue

            mae_f0 = sub.loc["F0", "MAE"]
            mae_f1 = sub.loc["F1", "MAE"]
            mae_f2 = sub.loc["F2", "MAE"]
            mae_f3 = sub.loc["F3", "MAE"]

            d_f1_f0 = mae_f1 - mae_f0
            d_f2_f1 = mae_f2 - mae_f1
            d_f3_f2 = mae_f3 - mae_f2
            d_total = mae_f3 - mae_f0

            # Determine best feature set F*
            f_star = sub["MAE"].idxmin()
            best_mae = sub["MAE"].min()

            records.append({
                "family": fam,
                "strategy": strat,
                "algorithm": model_prefix,
                "MAE_F0": mae_f0,
                "MAE_F1": mae_f1,
                "MAE_F2": mae_f2,
                "MAE_F3": mae_f3,
                "delta_F1_minus_F0": d_f1_f0,
                "delta_F2_minus_F1": d_f2_f1,
                "delta_F3_minus_F2": d_f3_f2,
                "delta_F3_minus_F0": d_total,
                "best_feature_set": f_star,
                "best_MAE": best_mae,
            })

    return pd.DataFrame(records).sort_values("best_MAE").reset_index(drop=True)


def compute_hurdle_ablation(df_results: pd.DataFrame) -> pd.DataFrame:
    """Analyze the 16 matched pairs of Hurdle Strategy A (Mixture) vs Strategy B (Hard-Gated).
    
    Computes delta MAE and delta RMSE (Strat B - Strat A) across all 4 Hurdle base families
    and 4 feature sets F0..F3.
    """
    assert_no_purged_metrics(df_results)
    df_h = df_results[df_results["family"] == "Two-Stage Hurdle"].copy()

    # Map Strategy A vs Strategy B
    strat_a = df_h[df_h["strategy"] == "Mixture Expectation"].copy()
    strat_b = df_h[df_h["strategy"] == "Hard-Gated Regime"].copy()

    records = []
    # Identify pairs by matching base algorithm and feature set
    for _, row_a in strat_a.iterrows():
        base_name = row_a["model"].replace("_stratA", "")
        f_set = row_a["feature_set"]

        # Find matching row in Strat B
        match_b = strat_b[
            (strat_b["feature_set"] == f_set)
            & (strat_b["model"].str.replace("_stratB", "") == base_name)
        ]
        if len(match_b) == 0:
            continue
        row_b = match_b.iloc[0]

        delta_mae = row_b["MAE"] - row_a["MAE"]
        delta_rmse = row_b["RMSE"] - row_a["RMSE"]
        winner = "Strategy A" if delta_mae > 0 else "Strategy B"

        records.append({
            "base_family": row_a["model"].split("_stratA")[0],
            "feature_set": f_set,
            "scenario_A": row_a["scenario_id"],
            "MAE_StratA": row_a["MAE"],
            "RMSE_StratA": row_a["RMSE"],
            "scenario_B": row_b["scenario_id"],
            "MAE_StratB": row_b["MAE"],
            "RMSE_StratB": row_b["RMSE"],
            "selected_tau_B": row_b["selected_hurdle_threshold"],
            "delta_MAE_B_minus_A": delta_mae,
            "delta_RMSE_B_minus_A": delta_rmse,
            "advantageous_strategy": winner,
        })

    return pd.DataFrame(records).sort_values("delta_MAE_B_minus_A").reset_index(drop=True)


def compute_seasonal_breakdown(
    forecasts_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    origin_dates: np.ndarray,
) -> pd.DataFrame:
    """Compute error stratification by Climate Season based strictly on Target Date t+h (CR-09 Option 9a).
    
    Definitions:
    - Target Date = origin_date + h days.
    - Rainy Season (Mùa Mưa): Target month in [5, 6, 7, 8, 9, 10] (May to October).
    - Dry Season (Mùa Khô): Target month in [11, 12, 1, 2, 3, 4] (November to April).
    
    Guarantees: Total observations across Rainy and Dry equal exactly 724 at every horizon h=1..7.
    """
    origin_dt = pd.to_datetime(origin_dates)
    num_origins, num_horizons = y_true.shape
    assert num_origins == 724 and num_horizons == 7

    # Build target date matrix (724, 7)
    target_dates = np.zeros((num_origins, num_horizons), dtype="datetime64[s]")
    for h in range(1, 8):
        target_dates[:, h - 1] = (origin_dt + pd.to_timedelta(h, unit="D")).values

    # Target month matrix (724, 7)
    target_months = pd.DatetimeIndex(target_dates.flatten()).month.values.reshape(num_origins, num_horizons)

    # Boolean masks
    rainy_mask = np.isin(target_months, [5, 6, 7, 8, 9, 10])
    dry_mask = ~rainy_mask

    # Verify that at each horizon, rainy + dry == 724
    for h_idx in range(num_horizons):
        n_rainy = rainy_mask[:, h_idx].sum()
        n_dry = dry_mask[:, h_idx].sum()
        assert n_rainy + n_dry == 724, f"Mismatch at horizon h={h_idx+1}: {n_rainy} + {n_dry} != 724"

    records = []
    for sc_id, y_pred in forecasts_dict.items():
        # Overall pooled seasonal metrics
        err_all = np.abs(y_true - y_pred)
        sq_err_all = (y_true - y_pred) ** 2

        rainy_mae = err_all[rainy_mask].mean()
        rainy_rmse = np.sqrt(sq_err_all[rainy_mask].mean())
        dry_mae = err_all[dry_mask].mean()
        dry_rmse = np.sqrt(sq_err_all[dry_mask].mean())

        records.append({
            "scenario_id": sc_id,
            "overall_MAE": err_all.mean(),
            "overall_RMSE": np.sqrt(sq_err_all.mean()),
            "rainy_MAE": rainy_mae,
            "rainy_RMSE": rainy_rmse,
            "dry_MAE": dry_mae,
            "dry_RMSE": dry_rmse,
            "delta_seasonal_MAE": rainy_mae - dry_mae,
            "n_rainy_points": int(rainy_mask.sum()),
            "n_dry_points": int(dry_mask.sum()),
        })

    df_seasonal = pd.DataFrame(records)
    assert_no_purged_metrics(df_seasonal)
    return df_seasonal.sort_values("overall_MAE").reset_index(drop=True)
