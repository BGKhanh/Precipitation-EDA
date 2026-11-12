"""
Direct Multi-Horizon Forecasters (Standard ML Regressors)
========================================================
Defines factory functions for Random Forest, XGBoost, and LightGBM Direct
using canonical hyperparameters audited under Benchmark v1.0.
"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def build_random_forest_direct(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiOutputRegressor:
    """Factory for Random Forest Direct Regressor."""
    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base_model, n_jobs=n_jobs)

def build_xgboost_direct(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiOutputRegressor:
    """Factory for XGBoost Direct Regressor."""
    base_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base_model, n_jobs=n_jobs)

def build_lightgbm_direct(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiOutputRegressor:
    """Factory for LightGBM Direct Regressor."""
    base_model = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    return MultiOutputRegressor(base_model, n_jobs=n_jobs)

