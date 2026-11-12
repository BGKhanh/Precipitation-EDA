"""
Tweedie Multi-Horizon Direct Forecasters for Rainfall Amount
============================================================
Defines factory functions for XGBoost and LightGBM Tweedie regressors
using canonical hyperparameters audited under Benchmark v1.0.
"""

from typing import Optional
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def build_xgboost_tweedie(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    tweedie_variance_power: float = 1.5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiOutputRegressor:
    """
    Factory for XGBoost Tweedie Regressor (Serving Champion SC-35).
    """
    base_model = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=tweedie_variance_power,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base_model, n_jobs=n_jobs)

def build_lightgbm_tweedie(
    n_estimators: int = 100,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.03,
    tweedie_variance_power: float = 1.5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiOutputRegressor:
    """
    Factory for LightGBM Tweedie Regressor (Shadow Champion SC-30 Config A / SC-31 Config A).
    Default max_depth=-1 matches canonical Validation Champion configuration.
    """
    base_model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=tweedie_variance_power,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    return MultiOutputRegressor(base_model, n_jobs=n_jobs)

