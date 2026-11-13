"""
Standalone Serving Inference Class (DS108 Inference Module)
============================================================
Self-contained predictor for operational serving:
- Loads fitted models and feature builders from artifact directory.
- Requires only raw historical DataFrame (minimum 30 calendar days).
- Implements dynamic effective forecast calculation based on data_as_of vs run_date.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Union, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class StandaloneRainfallPredictor:
    def __init__(self, artifact_dir: Optional[Union[str, Path]] = None):
        if artifact_dir is None:
            artifact_dir = PROJECT_ROOT / "artifacts" / "models"
        self.artifact_dir = Path(artifact_dir)

        # Load metadata
        meta_path = self.artifact_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load models & feature builders
        self.model_serving = joblib.load(self.artifact_dir / "sc35_xgboost_tweedie_f3.joblib")
        self.fb_f3 = joblib.load(self.artifact_dir / "feature_builder_f3.joblib")

        self.model_shadow_val = joblib.load(self.artifact_dir / "sc30_lightgbm_tweedie_f2.joblib")
        self.fb_f2 = joblib.load(self.artifact_dir / "feature_builder_f2.joblib")

        self.model_shadow_contender = joblib.load(self.artifact_dir / "sc31_lightgbm_tweedie_f3.joblib")

    def predict(
        self,
        raw_history_df: pd.DataFrame,
        run_date: Optional[Union[str, date, datetime]] = None,
        date_col: str = "Ngày",
        target_col: str = "Lượng mưa",
    ) -> Dict[str, Any]:
        """
        Execute prediction from raw historical observations.
        
        Args:
            raw_history_df: DataFrame containing at least 30 continuous daily records.
            run_date: Date when prediction is issued (defaults to current system date).
            date_col: Name of the timestamp column.
            target_col: Name of rainfall target column.
            
        Returns:
            Structured JSON-ready dictionary conforming to Phase C API contract.
        """
        if len(raw_history_df) < 30:
            raise ValueError(
                f"History DataFrame must contain at least 30 continuous days (got {len(raw_history_df)} rows). "
                f"Warmup lookback requires 29 days + 1 origin day."
            )

        df = raw_history_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)

        # Identify data_as_of (origin date t)
        origin_row = df.iloc[-1]
        data_as_of = origin_row[date_col].date() if hasattr(origin_row[date_col], "date") else origin_row[date_col]

        # Determine run_date
        if run_date is None:
            run_d = datetime.now().date()
        elif isinstance(run_date, str):
            run_d = datetime.strptime(run_date, "%Y-%m-%d").date()
        elif isinstance(run_date, datetime):
            run_d = run_date.date()
        else:
            run_d = run_date

        # Compute dynamic lag
        lag_days = (run_d - data_as_of).days
        effective_future_days = max(0, 7 - max(0, lag_days))

        # 1. Feature Extraction via build_single_step
        feat_df3 = self.fb_f3.build_single_step(df)
        cols3 = [c for c in feat_df3.columns if c not in [self.fb_f3.date_col, self.fb_f3.target_col]]
        X3 = feat_df3[cols3].values.astype(np.float64)

        feat_df2 = self.fb_f2.build_single_step(df)
        cols2 = [c for c in feat_df2.columns if c not in [self.fb_f2.date_col, self.fb_f2.target_col]]
        X2 = feat_df2[cols2].values.astype(np.float64)

        # 2. Model Predictions (7 horizons)
        preds_serving = np.clip(self.model_serving.predict(X3)[0], 0.0, None).tolist()
        preds_shadow_val = np.clip(self.model_shadow_val.predict(X2)[0], 0.0, None).tolist()
        preds_shadow_contender = np.clip(self.model_shadow_contender.predict(X3)[0], 0.0, None).tolist()

        # 3. Dynamic Horizon Labeling
        hindcast_and_current = []
        forecast_future = []

        for h in range(1, 8):
            target_date = data_as_of + timedelta(days=h)
            pred_val = round(float(preds_serving[h - 1]), 2)

            if target_date < run_d:
                status = "past_unverified"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status,
                    "observed_mm": None,
                    "note": "Day has passed but observational data has not been released yet."
                }
                hindcast_and_current.append(item)
            elif target_date == run_d:
                status = "today"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status,
                    "observed_mm": None,
                    "note": "Current operational day (observation in progress)."
                }
                hindcast_and_current.append(item)
            else:
                status = "future_forecast"
                item = {
                    "horizon": h,
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pred_mm": pred_val,
                    "status": status
                }
                forecast_future.append(item)

        return {
            "run_timestamp": datetime.now().isoformat() + "Z",
            "run_date": run_d.strftime("%Y-%m-%d"),
            "data_as_of": data_as_of.strftime("%Y-%m-%d"),
            "lag_days": lag_days,
            "effective_future_days": effective_future_days,
            "serving_model": {
                "scenario_id": "SC-35",
                "model_name": "xgboost_tweedie_F3",
                "governance_role": "test_ranking_winner_as_originally_benchmarked",
                "registry_alias": "serving",
                "feature_level": "F3",
                "predictions_7d_mm": [round(float(p), 2) for p in preds_serving],
                "hindcast_and_current": hindcast_and_current,
                "forecast_future": forecast_future
            },
            "shadow_models": {
                "shadow_val_champ": {
                    "scenario_id": "SC-30",
                    "model_name": "lightgbm_tweedie_F2",
                    "governance_role": "validation_selected_champion",
                    "registry_alias": "shadow_val_champ",
                    "feature_level": "F2",
                    "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_val]
                },
                "shadow_test_contender": {
                    "scenario_id": "SC-31",
                    "model_name": "lightgbm_tweedie_F3",
                    "governance_role": "post_audit_test_contender",
                    "registry_alias": "shadow_test_contender",
                    "feature_level": "F3",
                    "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_contender]
                }
            },
            # Backwards compatibility key
            "shadow_model": {
                "scenario_id": "SC-30",
                "model_name": "lightgbm_tweedie_F2",
                "governance_role": "validation_selected_champion",
                "predictions_7d_mm": [round(float(p), 2) for p in preds_shadow_val]
            }
        }

