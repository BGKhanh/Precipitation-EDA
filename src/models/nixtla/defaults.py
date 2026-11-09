"""Default model lists, configurations, and registries for 7-group benchmarking.

Strictly aligned with Protocol v1.0.20 and Model Specifications Report v1.1.3:
- Decision M6a: Fixed context window W = 30 days (input_size = 30) for deep neural models.
- Decision PROP-03: Fixed training budget max_steps = 10,000 steps (~50 epochs), batch_size = 32, lr = 1e-3.
- Decision M7: Intermittent models with domain bucket size N=2 for ADIDA and K=2 for IMAPA.
- Decision M1: Only 2 canonical Hurdle models: Logistic+Linear and LightGBM_Hurdle.
"""

from typing import List, Any, Optional, Dict
import logging

from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    CrostonOptimized,
    CrostonSBA,
    TSB,
    ADIDA,
    IMAPA,
    Naive,
    SeasonalNaive,
    ARIMA,
)

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Statistical & Intermittent Models
# ======================================================================

def get_statistical_models(eda_report: Optional[Any] = None) -> List[Any]:
    """Return statistical, intermittent, and naive baseline models.

    Models:
    - Baselines: Naive (persistence), SeasonalNaive
    - Intermittent: CrostonOptimized, CrostonSBA, TSB, ADIDA, IMAPA
    - Classical TS: ARIMA (non-seasonal), SARIMA_EDA (seasonal from EDA), AutoARIMA, AutoETS
    """
    season_length = 7
    order = (3, 0, 3)
    seasonal_order = (1, 1, 1)

    if eda_report is not None:
        periods = getattr(eda_report, 'representative_periods', [])
        valid_periods = [p for p in periods if isinstance(p, (int, float)) and 1 < p <= 52]
        if valid_periods:
            season_length = int(valid_periods[0])

        sarima_order = getattr(eda_report, 'suggested_sarima_order', None)
        if sarima_order and len(sarima_order) >= 3:
            order = tuple(int(x) for x in sarima_order[:3])

        seas_order = getattr(eda_report, 'suggested_seasonal_order', None)
        if seas_order and len(seas_order) >= 4:
            seasonal_order = tuple(int(x) for x in seas_order[:3])
            season_length = int(seas_order[3])

    return [
        Naive(),
        SeasonalNaive(season_length=season_length),
        ARIMA(order=order, alias='ARIMA'),
        ARIMA(order=order, seasonal_order=seasonal_order, season_length=season_length, alias='SARIMA_EDA'),
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        CrostonOptimized(),
        CrostonSBA(),
        TSB(alpha_d=0.2, alpha_p=0.2),
        ADIDA(),
        IMAPA(),
    ]


def get_eda_arima_model(eda_report: Optional[Any] = None) -> Any:
    """Return non-seasonal ARIMA model (p, d, q) based on EDA order."""
    order = (3, 0, 3)
    if eda_report is not None:
        sarima_order = getattr(eda_report, 'suggested_sarima_order', None)
        if sarima_order and len(sarima_order) >= 3:
            order = tuple(int(x) for x in sarima_order[:3])

    return ARIMA(
        order=order,
        alias='ARIMA_EDA',
    )


def get_eda_sarima_model(eda_report: Optional[Any] = None) -> Any:
    """Return seasonal ARIMA (SARIMA) model with seasonal order and length from EDA findings."""
    order = (3, 0, 3)
    seasonal_order = (1, 1, 1)
    season_length = 7

    if eda_report is not None:
        sarima_order = getattr(eda_report, 'suggested_sarima_order', None)
        if sarima_order and len(sarima_order) >= 3:
            order = tuple(int(x) for x in sarima_order[:3])

        seas_order = getattr(eda_report, 'suggested_seasonal_order', None)
        if seas_order and len(seas_order) >= 4:
            seasonal_order = tuple(int(x) for x in seas_order[:3])
            season_length = int(seas_order[3])

    return ARIMA(
        order=order,
        season_length=season_length,
        seasonal_order=seasonal_order,
        alias='SARIMA_EDA',
    )


# ======================================================================
# 2. ML Direct (Standard Loss, Single-Stage)
# ======================================================================

def get_ml_direct_models(eda_report: Optional[Any] = None) -> List[Any]:
    """Return ML models for direct multi-horizon forecasting with standard loss.

    Models:
    - ML Baseline: LinearRegression
    - Tree Ensembles: RandomForestRegressor, XGBRegressor, LGBMRegressor (scaled)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    return [
        LinearRegression(),
        RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42),
        XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
        LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=42,
        ),
    ]


# ======================================================================
# 3. ML Tweedie (Single-Stage Compound Poisson-Gamma)
# ======================================================================

def get_ml_tweedie_models(eda_report: Optional[Any] = None) -> List[Any]:
    """Return ML single-stage models using Tweedie objective.

    Only applied to ML single-stage/direct forecasting (NOT Hurdle, NOT Neural).
    """
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return [
        LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=1.5,
            learning_rate=0.03,
            num_leaves=63,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=42,
        ),
        XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=1.5,
            learning_rate=0.03,
            max_depth=6,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
    ]


# ======================================================================
# 4. ML Hurdle / Two-Stage Models
# ======================================================================

def get_ml_hurdle_models(
    classification_threshold: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return canonical ML two-stage Hurdle models (Classifier + Regressor).

    Restored 4 canonical models across Strategies A & B (72 Scenarios):
    - Logistic+Linear
    - RandomForest_Hurdle
    - XGBoost_Hurdle
    - LightGBM_Hurdle
    """
    from ..legacy.linear_models import LinearRainfallModel
    from ..legacy.tree_models import (
        RandomForestRainfallModel,
        XGBoostRainfallModel,
        LightGBMRainfallModel,
    )

    return {
        'Logistic+Linear': LinearRainfallModel(
            use_two_stage=True,
            regression_type='linear',
            classification_threshold=classification_threshold,
            random_state=random_state,
        ),
        'RandomForest_Hurdle': RandomForestRainfallModel(
            use_two_stage=True,
            classification_threshold=classification_threshold,
            random_state=random_state,
        ),
        'XGBoost_Hurdle': XGBoostRainfallModel(
            use_two_stage=True,
            classification_threshold=classification_threshold,
            random_state=random_state,
        ),
        'LightGBM_Hurdle': LightGBMRainfallModel(
            use_two_stage=True,
            classification_threshold=classification_threshold,
            random_state=random_state,
        ),
    }


# ======================================================================
# 5. Neural Direct Models (Standard Loss, Scaled for GPU Acceleration)
# ======================================================================

def get_neural_direct_models(
    horizon: int = 7,
    input_size: int = 30,  # Decision M6a: Locked W = 30 days
    max_steps: int = 10000,  # Decision PROP-03: Fixed budget 50 epochs / 10,000 steps
    batch_size: int = 32,  # Decision PROP-03: batch_size = 32
    learning_rate: float = 1e-3,  # Decision PROP-03: learning_rate = 1e-3
    eda_report: Optional[Any] = None,
) -> List[Any]:
    """Return Neural deep learning models with standard loss (MAE).

    Strictly aligned with Protocol v1.0.20:
    - 5 canonical models: LSTM, TCN, NHITS, PatchTST, TFT
    - Context window W = 30 days (Decision M6a)
    - Controlled Experimental Design: max_steps = 10,000, batch_size = 32, lr = 1e-3 (PROP-03)
    """
    try:
        from neuralforecast.models import LSTM, TCN, NHITS, PatchTST, TFT
        from neuralforecast.losses.pytorch import MAE

        return [
            LSTM(
                h=horizon,
                input_size=input_size,
                encoder_hidden_size=128,
                encoder_n_layers=2,
                loss=MAE(),
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=42,
            ),
            TCN(
                h=horizon,
                input_size=input_size,
                encoder_hidden_size=64,
                kernel_size=3,
                dilations=[1, 2, 4, 8],
                loss=MAE(),
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=42,
            ),
            NHITS(
                h=horizon,
                input_size=input_size,
                n_blocks=[1, 1, 1],
                mlp_units=[[256, 256], [256, 256], [256, 256]],
                n_pool_kernel_size=[2, 2, 1],
                loss=MAE(),
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=42,
            ),
            PatchTST(
                h=horizon,
                input_size=input_size,
                patch_len=7,
                stride=3,
                hidden_size=128,
                n_heads=8,
                linear_hidden_size=256,
                loss=MAE(),
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=42,
            ),
            TFT(
                h=horizon,
                input_size=input_size,
                hidden_size=64,
                n_head=4,
                dropout=0.1,
                loss=MAE(),
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=42,
            ),
        ]
    except Exception as e:
        logger.warning("NeuralForecast models not available in current environment: %s", e)
        return []


# ======================================================================
# Backward-compatibility aliases
# ======================================================================
get_default_stats_models = get_statistical_models
get_default_ml_models = get_ml_tweedie_models
get_default_neural_models = get_neural_direct_models
