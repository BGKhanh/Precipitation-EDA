"""
Empirical Benchmark: Canonical Hurdle Two-Stage vs Direct Tweedie.

Addresses all feedback review requirements:
1. Occurrence metrics semantics:
   - Hurdle Stage 1 (shared by Hard-Gate and Expected) outputs true calibrated probability p_cal.
     Brier score is computed on continuous p_cal in [0, 1].
     POD, FAR, CSI are computed using decision threshold tau: (p_cal >= tau).
     Hard-Gate and Expected share identical occurrence metrics because both use the same Stage 1.
   - Direct Tweedie is a point regression model (no probability distribution output).
     Its occurrence is derived strictly by thresholding point forecast (y_hat > c).
     Brier score is marked N/A (np.nan).
2. Conditional Amount metrics semantics:
   - Stage-2 Pure Conditional (on actual wet days Y > c): evaluates pure regression output mu vs Y.
   - End-to-End Wet-Day (on actual wet days Y > c): evaluates end-to-end output Y_hat vs Y
     (incorporates Stage 1 misses where p < tau -> Y_hat = 0).
3. Overall Hurdle Target Metrics:
   - Primary: evaluated against canonical hurdle target Y* = Y * 1(Y > c).
   - Supplementary: evaluated against raw rainfall Y.
4. Reporting: Full tables across all 7 horizons (h = 1..7).
5. Temporal Protocol:
   - Canonical split at 2020-05-01. All fitting, calibration, and tau-tuning occur strictly on Train (< 2020-05-01).
   - At each test origin t >= 2020-05-01, models only receive history <= t. Zero future leakage.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.loader import DataLoader, time_series_split
from src.featurengineering.builder import FeatureBuilder, FeatureConfig
from src.models.legacy.tree_models import RandomForestRainfallModel
from src.models.legacy.recursive import RecursiveForecaster


def run_benchmark(n_test_windows: int = 50, step_size: int = 7) -> pd.DataFrame:
    print("=" * 95)
    print("EMPIRICAL BENCHMARK: CANONICAL HURDLE RECURSIVE vs DIRECT TWEEDIE")
    print("=" * 95)

    # 1. Load canonical dataset
    loader = DataLoader()
    df = loader.load_data(start_date="20000101", end_date="20260430")
    if df is None:
        raise RuntimeError("Failed to load dataset.")

    # Canonical split (single source of truth: 2020-05-01)
    train_df, test_df = time_series_split(df)
    print(f"Loaded dataset:")
    print(f"   Train: {train_df.shape} (2000-01-01 to 2020-04-30)")
    print(f"   Test:  {test_df.shape} (2020-05-01 to 2026-04-30)")

    # 2. Hurdle threshold c
    c_threshold = 0.1
    eda_path = Path("data/processed/eda_report.json")
    if eda_path.exists():
        with open(eda_path, "r", encoding="utf-8") as f:
            eda_data = json.load(f)
            c_threshold = eda_data.get("validated_rain_threshold", 0.1)
    print(f"Hurdle rainfall threshold c: {c_threshold} mm")

    # 3. Fit FeatureBuilder on train_df ONLY (Zero Leakage)
    cfg = FeatureConfig(
        lag_columns=["Lượng mưa"],
        lag_periods=[1, 2, 3, 7],
        rolling_columns=["Lượng mưa"],
        rolling_windows=[7, 14, 30],
        temporal_features=["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "Is_Wet_Season"],
        apply_quality_filter=False,
    )
    fb = FeatureBuilder(cfg)
    fb.fit(train_df)

    # 4. Train Hurdle Two-Stage Baseline Model
    print("\n--- [1 & 2] Training Hurdle Two-Stage Model ---")
    df_train_feat = fb.transform(train_df)
    feature_cols = [c for c in df_train_feat.columns if c not in [fb.target_col, fb.date_col]]
    X_train = df_train_feat[feature_cols]
    y_train = df_train_feat[fb.target_col]

    # Temporal split within train for calibration and threshold tuning (strictly past data)
    n_total = len(X_train)
    n_train_sub = int(n_total * 0.85)
    n_cal_sub = int(n_total * 0.95)

    X_tr = X_train.iloc[:n_train_sub]
    y_tr = y_train.iloc[:n_train_sub]
    X_cal = X_train.iloc[n_train_sub:n_cal_sub]
    y_cal = y_train.iloc[n_train_sub:n_cal_sub]
    X_val = X_train.iloc[n_cal_sub:]
    y_val = y_train.iloc[n_cal_sub:]

    print(f"   Train sub (base models):  {len(X_tr)} samples")
    print(f"   Cal sub (Platt scaling):  {len(X_cal)} samples (FrozenEstimator)")
    print(f"   Val sub (tau tuning):     {len(X_val)} samples")

    hurdle_model = RandomForestRainfallModel(
        use_two_stage=True,
        classification_threshold=c_threshold,
        n_estimators=50,
        random_state=42,
    )
    # Fit with temporal probability calibration (FrozenEstimator)
    hurdle_model.fit_calibrated(X_tr, y_tr, X_cal, y_cal, method="sigmoid")
    # Tune tau on validation data
    tau_opt = hurdle_model.find_optimal_threshold(X_val, y_val, metric="f1", return_calibrated=True)
    print(f"   Optimal decision threshold tau: {tau_opt}")

    # Set up forecasters
    forecaster_hard = RecursiveForecaster(
        model=hurdle_model,
        feature_builder=fb,
        clf_threshold=tau_opt,
        strategy="hard_gate",
    )
    forecaster_exp = RecursiveForecaster(
        model=hurdle_model,
        feature_builder=fb,
        clf_threshold=tau_opt,
        strategy="expected",
    )

    # 5. Train Direct Tweedie Forecaster
    # Trains 7 separate single-stage direct LightGBM models with Tweedie loss
    print("\n--- [3] Training Direct Single-Stage Tweedie Forecasters (h=1..7) ---")
    tweedie_models = {}
    for h in range(1, 8):
        y_target_h = y_train.shift(-h).dropna()
        X_target_h = X_train.iloc[:len(y_target_h)]
        
        reg_tweedie = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=60,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        reg_tweedie.fit(X_target_h, y_target_h)
        tweedie_models[h] = reg_tweedie
    print("   Trained 7 direct Tweedie models successfully")

    # 6. Rolling Evaluation on Canonical Test Set
    print(f"\n--- Running Rolling-Origin Evaluation on Test Set ({n_test_windows} windows, step={step_size}) ---")
    print("   Protocol: For each origin t >= 2020-05-01, model only sees history <= t.")
    
    history_len = 45
    full_eval_df = pd.concat([train_df.tail(history_len), test_df]).reset_index(drop=True)
    test_start_idx = history_len

    records = []
    max_origins = min(n_test_windows, (len(test_df) - 7) // step_size)

    for w in range(max_origins):
        origin_idx = test_start_idx + w * step_size
        if origin_idx + 7 > len(full_eval_df):
            break

        history_window = full_eval_df.iloc[origin_idx - history_len:origin_idx].copy()
        ground_truth_window = full_eval_df.iloc[origin_idx:origin_idx + 7].copy()
        y_true_7 = ground_truth_window[fb.target_col].values

        # 1. Hurdle Hard-Gate Rollout
        fc_hg = forecaster_hard.forecast(history=history_window, steps=7)
        preds_hg = fc_hg["prediction"].values
        probs_hg = fc_hg["prob"].values if "prob" in fc_hg.columns else np.zeros(7)
        mus_hg = fc_hg["mu"].values if "mu" in fc_hg.columns else preds_hg

        # 2. Hurdle Expected Rollout
        fc_exp = forecaster_exp.forecast(history=history_window, steps=7)
        preds_exp = fc_exp["prediction"].values
        probs_exp = fc_exp["prob"].values if "prob" in fc_exp.columns else np.zeros(7)
        mus_exp = fc_exp["mu"].values if "mu" in fc_exp.columns else preds_exp

        # 3. Direct Tweedie Forecasting
        df_origin_feat = fb.transform(history_window)
        if len(df_origin_feat) == 0:
            continue
        X_origin_latest = df_origin_feat[feature_cols].iloc[[-1]]
        
        preds_tweedie = np.zeros(7)
        for h in range(1, 8):
            preds_tweedie[h - 1] = max(0.0, float(tweedie_models[h].predict(X_origin_latest)[0]))

        origin_date = str(full_eval_df.iloc[origin_idx - 1][fb.date_col].date())
        # Record predictions per horizon
        for h in range(1, 8):
            yt = y_true_7[h - 1]
            yt_star = yt if yt > c_threshold else 0.0
            is_rain = int(yt > c_threshold)
            fc_date = str(ground_truth_window.iloc[h - 1][fb.date_col].date())

            records.append({
                "window": w,
                "origin_date": origin_date,
                "forecast_date": fc_date,
                "horizon": h,
                "y_true": yt,
                "y_hurdle_target": yt_star,
                "is_rain": is_rain,
                # Stage 1 Probabilities
                "prob_hurdle_hg": probs_hg[h - 1],
                "prob_hurdle_exp": probs_exp[h - 1],
                # Stage 2 Pure Regression (mu)
                "mu_hurdle_hg": mus_hg[h - 1],
                "mu_hurdle_exp": mus_exp[h - 1],
                # End-to-End Predictions (Y_hat)
                "pred_hard_gate": preds_hg[h - 1],
                "pred_expected": preds_exp[h - 1],
                "pred_tweedie": preds_tweedie[h - 1],
                # Binary occurrence decisions
                "bin_hurdle_hg": int(probs_hg[h - 1] >= tau_opt),
                "bin_hurdle_exp": int(probs_exp[h - 1] >= tau_opt),  # Stage 1 decision
                "bin_tweedie": int(preds_tweedie[h - 1] > c_threshold),  # Point-forecast thresholded
            })

    results_df = pd.DataFrame(records)
    print(f"Total evaluated horizon forecasts: {len(results_df)} points across {max_origins} origins.")

    # 7. Aggregate Metrics per Horizon
    summary_rows = []
    
    # Models to benchmark:
    # (Model Name, End-to-End Pred Col, Pure Mu Col, Prob Col, Binary Decision Col)
    models_config = [
        ("Hurdle + Recursive + Hard-Gate", "pred_hard_gate", "mu_hurdle_hg", "prob_hurdle_hg", "bin_hurdle_hg"),
        ("Hurdle + Recursive + Expected", "pred_expected", "mu_hurdle_exp", "prob_hurdle_exp", "bin_hurdle_exp"),
        ("Direct Tweedie (Single-Stage)", "pred_tweedie", "pred_tweedie", None, "bin_tweedie"),
    ]

    for model_name, pred_col, mu_col, prob_col, bin_col in models_config:
        for h in range(1, 8):
            sub = results_df[results_df["horizon"] == h]
            yt = sub["y_true"].values
            yt_star = sub["y_hurdle_target"].values
            yp = sub[pred_col].values
            mu = sub[mu_col].values
            y_bin = sub["is_rain"].values
            yp_bin = sub[bin_col].values

            # 1. Occurrence Metrics
            hits = int(((yp_bin == 1) & (y_bin == 1)).sum())
            misses = int(((yp_bin == 0) & (y_bin == 1)).sum())
            false_alarms = int(((yp_bin == 1) & (y_bin == 0)).sum())
            pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
            csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

            # Brier Score: only valid when predicted probabilities in [0, 1] are available
            if prob_col is not None:
                p_vals = sub[prob_col].values
                brier = float(np.mean((p_vals - y_bin) ** 2))
            else:
                brier = np.nan  # Direct Tweedie does not output occurrence probabilities

            # 2. Conditional Amount Metrics on Actual Wet Days (Y > c)
            wet_mask = y_bin == 1
            if wet_mask.sum() > 0:
                # Tier A: Pure Stage-2 Regression Capacity (mu vs Y)
                pure_cond_mae = float(np.mean(np.abs(yt[wet_mask] - mu[wet_mask])))
                pure_cond_rmse = float(np.sqrt(np.mean((yt[wet_mask] - mu[wet_mask]) ** 2)))
                pure_cond_bias = float(np.mean(mu[wet_mask] - yt[wet_mask]))

                # Tier B: End-to-End Wet-Day Forecast (Y_hat vs Y, penalizing Stage 1 misses)
                e2e_wet_mae = float(np.mean(np.abs(yt[wet_mask] - yp[wet_mask])))
                e2e_wet_rmse = float(np.sqrt(np.mean((yt[wet_mask] - yp[wet_mask]) ** 2)))
                e2e_wet_bias = float(np.mean(yp[wet_mask] - yt[wet_mask]))
            else:
                pure_cond_mae, pure_cond_rmse, pure_cond_bias = np.nan, np.nan, np.nan
                e2e_wet_mae, e2e_wet_rmse, e2e_wet_bias = np.nan, np.nan, np.nan

            # 3. Overall Metrics (evaluated on canonical hurdle target Y*)
            overall_mae = float(np.mean(np.abs(yt_star - yp)))
            overall_rmse = float(np.sqrt(np.mean((yt_star - yp) ** 2)))
            overall_bias = float(np.mean(yp - yt_star))

            # Supplementary: overall on raw Y
            raw_mae = float(np.mean(np.abs(yt - yp)))

            summary_rows.append({
                "Model": model_name,
                "Horizon": h,
                "POD": pod,
                "FAR": far,
                "CSI": csi,
                "Brier": brier,
                # Tier A: Pure Stage 2 Regression
                "Pure_Cond_MAE": pure_cond_mae,
                "Pure_Cond_RMSE": pure_cond_rmse,
                "Pure_Cond_Bias": pure_cond_bias,
                # Tier B: End-to-End Wet Day
                "E2E_Wet_MAE": e2e_wet_mae,
                "E2E_Wet_RMSE": e2e_wet_rmse,
                "E2E_Wet_Bias": e2e_wet_bias,
                # Overall
                "Overall_MAE": overall_mae,
                "Overall_RMSE": overall_rmse,
                "Overall_Bias": overall_bias,
                "Raw_MAE": raw_mae,
                "n_samples": len(sub),
                "n_wet": int(wet_mask.sum()),
            })

    summary_df = pd.DataFrame(summary_rows)

    # 8. Print Complete Full-Horizon Reports (h=1..7)
    print("\n" + "=" * 95)
    print("TABLE 1: OCCURRENCE METRICS ACROSS HORIZONS h=1..7")
    print("   Note: Hurdle Hard-Gate & Expected share identical Stage 1 occurrence model.")
    print("   Direct Tweedie occurrence is thresholded from point forecast (y_hat > c); Brier is N/A.")
    print("=" * 95)
    occ_view = summary_df.pivot(index="Horizon", columns="Model", values=["CSI", "POD", "FAR", "Brier"])
    print(occ_view.round(4).to_string())

    print("\n" + "=" * 95)
    print("TABLE 2A: PURE STAGE-2 REGRESSION CAPACITY ON WET DAYS (mu vs Y in mm, h=1..7)")
    print("   Evaluates the regression model directly without gating penalties.")
    print("=" * 95)
    pure_cond_view = summary_df.pivot(index="Horizon", columns="Model", values=["Pure_Cond_MAE", "Pure_Cond_RMSE", "Pure_Cond_Bias"])
    print(pure_cond_view.round(2).to_string())

    print("\n" + "=" * 95)
    print("TABLE 2B: END-TO-END WET-DAY PERFORMANCE (Y_hat vs Y in mm, h=1..7)")
    print("   Evaluates end-to-end output on rainy days, penalizing false negative misses (where Y_hat=0).")
    print("=" * 95)
    e2e_wet_view = summary_df.pivot(index="Horizon", columns="Model", values=["E2E_Wet_MAE", "E2E_Wet_RMSE", "E2E_Wet_Bias"])
    print(e2e_wet_view.round(2).to_string())

    print("\n" + "=" * 95)
    print("TABLE 3: OVERALL CANONICAL HURDLE TARGET Y* (Y_hat vs Y* in mm, h=1..7)")
    print("   Primary evaluation across all days (dry and wet).")
    print("=" * 95)
    overall_view = summary_df.pivot(index="Horizon", columns="Model", values=["Overall_MAE", "Overall_RMSE", "Overall_Bias"])
    print(overall_view.round(2).to_string())

    # Save results
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "hurdle_vs_tweedie_benchmark_audited.csv", index=False)
    results_df.to_csv(out_dir / "hurdle_vs_tweedie_predictions_per_window.csv", index=False)
    print(f"\nSaved audited benchmark summary to {out_dir / 'hurdle_vs_tweedie_benchmark_audited.csv'}")
    print(f"Saved detailed predictions ({len(results_df)} records) to {out_dir / 'hurdle_vs_tweedie_predictions_per_window.csv'}")

    return summary_df


if __name__ == "__main__":
    run_benchmark(n_test_windows=50, step_size=7)
