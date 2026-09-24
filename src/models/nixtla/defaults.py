"""Default model lists and configurations based on EDA findings."""

from typing import List, Any, Optional
import logging

from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    CrostonOptimized,
    TSB,
    IMAPA,
    Naive,
    SeasonalNaive,
)
from mlforecast.auto import AutoLightGBM, AutoXGBoost

from ...training.optimizer import (
    get_lightgbm_tweedie_config,
    get_xgboost_tweedie_config,
    get_neural_tweedie_config,
)

logger = logging.getLogger(__name__)


def get_default_stats_models(eda_report: Optional[Any] = None) -> List[Any]:
    """Return default models for StatsForecastAdapter.

    Combines:
    - Time-series baselines: AutoARIMA, AutoETS
    - Intermittent baselines (for LUMPY data from EDA): CrostonOptimized, TSB, IMAPA
    - Standard Naive baselines: Naive (persistence), SeasonalNaive
    """
    season_length = 7
    if eda_report is not None:
        periods = getattr(eda_report, 'representative_periods', [])
        valid_periods = [p for p in periods if isinstance(p, (int, float)) and 1 < p <= 52]
        if valid_periods:
            season_length = int(valid_periods[0])

    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        CrostonOptimized(),
        TSB(alpha_d=0.2, alpha_p=0.2),
        IMAPA(),
        Naive(),
        SeasonalNaive(season_length=season_length),
    ]
    return models


def get_default_ml_models(eda_report: Optional[Any] = None) -> List[Any]:
    """Return default models for MLForecastAdapter with Tweedie loss."""
    lgb_config = get_lightgbm_tweedie_config(eda_report)
    xgb_config = get_xgboost_tweedie_config(eda_report)

    return [
        AutoLightGBM(config=lgb_config),
        AutoXGBoost(config=xgb_config),
    ]


def get_default_neural_models(eda_report: Optional[Any] = None, horizon: int = 7) -> List[Any]:
    """Return default models for NeuralForecastAdapter with Tweedie DistributionLoss.

    Gracefully falls back to an empty list if neural environment dependencies fail.
    """
    try:
        from neuralforecast.auto import AutoNHITS
        from neuralforecast.losses.pytorch import DistributionLoss

        neural_config = get_neural_tweedie_config(eda_report)
        return [
            AutoNHITS(
                h=horizon,
                loss=DistributionLoss(distribution='Tweedie', rho=1.5),
                config=neural_config,
            )
        ]
    except Exception as e:
        logger.warning("NeuralForecast models not available in current environment: %s", e)
        return []
