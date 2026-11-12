"""
DS108 Models Library
====================
Provides model architectures and factories:
- Tweedie: build_xgboost_tweedie, build_lightgbm_tweedie
- Direct: build_random_forest_direct, build_xgboost_direct, build_lightgbm_direct
- Hurdle: build_logistic_linear_hurdle, build_lightgbm_hurdle, build_xgboost_hurdle, build_random_forest_hurdle, HurdleModel
"""

from src.models.tweedie import build_xgboost_tweedie, build_lightgbm_tweedie
from src.models.direct import (
    build_random_forest_direct,
    build_xgboost_direct,
    build_lightgbm_direct,
)
from src.models.hurdle import (
    DirectHurdleForecaster,
    HurdleModel,
    build_logistic_linear_hurdle,
    build_lightgbm_hurdle,
    build_xgboost_hurdle,
    build_random_forest_hurdle,
)

__all__ = [
    "build_xgboost_tweedie",
    "build_lightgbm_tweedie",
    "build_random_forest_direct",
    "build_xgboost_direct",
    "build_lightgbm_direct",
    "DirectHurdleForecaster",
    "HurdleModel",
    "build_logistic_linear_hurdle",
    "build_lightgbm_hurdle",
    "build_xgboost_hurdle",
    "build_random_forest_hurdle",
]
