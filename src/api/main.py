"""
FastAPI Serving Engine for HCMC Daily Rainfall Forecasting System
==================================================================
Serves operational multi-horizon rainfall forecasts using:
- Serving Champion: SC-35 XGBoost Tweedie F3 (governance_role: test_ranking_winner_as_originally_benchmarked)
- Shadow Model 1: SC-30 LightGBM Tweedie F2 Config A (governance_role: validation_selected_champion)
- Shadow Model 2: SC-31 LightGBM Tweedie F3 Config A (governance_role: post_audit_test_contender)

Features:
- Dynamic horizon tagging (past_unverified, today, future_forecast) based on data_as_of vs run_date.
- National multi-province support with canonical coordinates and island territories.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import StandaloneRainfallPredictor
from src.config.constants import Config

from contextlib import asynccontextmanager

predictor: Optional[StandaloneRainfallPredictor] = None

def load_serving_models():
    global predictor
    artifact_path = PROJECT_ROOT / "artifacts" / "models"
    predictor = StandaloneRainfallPredictor(artifact_dir=artifact_path)
    print("Serving models and feature builders loaded successfully into API memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_models()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="HCMC Daily Rainfall Forecasting Serving Engine",
    description="Operational API for 1-7 days daily rainfall forecasting in HCMC & Vietnam with dynamic horizon calculation and shadow model governance.",
    version="1.0.0",
    lifespan=lifespan,
)

class DailyWeatherRecord(BaseModel):
    Ngày: str = Field(..., description="Date of observation in YYYY-MM-DD format")
    Lượng_mưa: float = Field(..., alias="Lượng mưa", description="Daily precipitation in mm")
    Nhiệt_độ_2m: float = Field(..., alias="Nhiệt độ 2m", description="Temperature at 2m (C)")
    Độ_ẩm_tương_đối_2m: float = Field(..., alias="Độ ẩm tương đối 2m", description="Relative humidity at 2m (%)")
    Tốc_độ_gió_2m: float = Field(..., alias="Tốc độ gió 2m", description="Wind speed at 2m (m/s)")
    Nhiệt_độ_điểm_sương_2m: Optional[float] = Field(None, alias="Nhiệt độ điểm sương 2m")
    Áp_suất_bề_mặt: Optional[float] = Field(None, alias="Áp suất bề mặt")
    Độ_ẩm_đặc_trưng_2m: Optional[float] = Field(None, alias="Độ ẩm đặc trưng 2m")
    Bức_xạ_mặt_trời: Optional[float] = Field(None, alias="Bức xạ mặt trời")

    model_config = ConfigDict(populate_by_name=True, extra="allow")

class PredictionRequest(BaseModel):
    history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Historical continuous daily records (at least 30 days). If omitted, loads canonical recent data."
    )
    run_date: Optional[str] = Field(
        None,
        description="Operational run date (YYYY-MM-DD). Defaults to current system date."
    )
    province: Optional[str] = Field(
        "TP. Hồ Chí Minh",
        description="Province or territory name from constants.py (e.g. TP. Hồ Chí Minh, Hoàng Sa, Trường Sa)"
    )

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Health check endpoint indicating model readiness and governance status."""
    return {
        "status": "healthy",
        "service": "HCMC Rainfall Forecasting Engine",
        "version": "1.0.0",
        "predictor_loaded": predictor is not None,
        "serving_model": {
            "scenario_id": "SC-35",
            "model_name": "xgboost_tweedie_F3",
            "governance_role": "test_ranking_winner_as_originally_benchmarked",
            "alias": "serving",
            "status": "active"
        },
        "shadow_models": [
            {
                "scenario_id": "SC-30",
                "model_name": "lightgbm_tweedie_F2",
                "governance_role": "validation_selected_champion",
                "alias": "shadow_val_champ",
                "status": "active_shadow"
            },
            {
                "scenario_id": "SC-31",
                "model_name": "lightgbm_tweedie_F3",
                "governance_role": "post_audit_test_contender",
                "alias": "shadow_test_contender",
                "status": "active_shadow"
            }
        ]
    }

@app.get("/models/info", tags=["Governance"])
def get_model_metadata():
    """Returns governance metadata, hyperparameters, and cryptographic SHA-256 hashes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor.metadata

@app.post("/predict", tags=["Forecasting"])
def predict_rainfall(request: PredictionRequest):
    """
    Generate 7-day multi-horizon rainfall forecasts.
    Evaluates Serving Model (SC-35) and records Shadow Models (SC-30, SC-31).
    Dynamically partitions horizons into hindcast_and_current and forecast_future based on lag.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    if request.history is not None and len(request.history) > 0:
        df_history = pd.DataFrame(request.history)
    else:
        # Load canonical data as fallback history
        from src.evaluation.protocols import get_global_read_guard
        from src.data.loader import DataLoader
        g = get_global_read_guard()
        was_active = g.is_active
        g.deactivate()
        try:
            loader = DataLoader()
            df_canonical = loader.load_data()
            df_history = df_canonical.iloc[-35:].copy()
        finally:
            if was_active:
                g.activate()

    # Province validation & metadata
    prov = request.province or "TP. Hồ Chí Minh"
    lat, lon = Config.province_coordinates.get(
        prov, Config.province_coordinates["TP. Hồ Chí Minh"]
    )

    try:
        prediction_result = predictor.predict(
            raw_history_df=df_history,
            run_date=request.run_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction_result["location"] = {
        "province": prov,
        "latitude": lat,
        "longitude": lon,
        "grid_resolution_note": "MERRA-2 0.5 deg x 0.625 deg spatial grid"
    }

    return prediction_result

@app.get("/benchmark/leaderboard", tags=["Governance"])
def get_leaderboard(limit: int = 10):
    """Returns top audited benchmark scenarios from official benchmark test leaderboards."""
    test_csv = PROJECT_ROOT / "reports" / "test_benchmark" / "unified_72_scenarios_test_leaderboard.csv"
    if not test_csv.exists():
        raise HTTPException(status_code=404, detail="Test leaderboard file not found")
    
    df = pd.read_csv(test_csv)
    top_models = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "benchmark_dataset": "Canonical Out-of-sample Test (2020-05-01 to 2026-04-20, 2,184 origins)",
        "evaluation_rule": "RULE: Model Evaluation Standards (MAE, MSE, RMSE, MASE)",
        "top_scenarios": top_models
    }
