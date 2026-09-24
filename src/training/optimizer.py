"""Hyperparameter optimization configurations for Nixtlaverse models.

This module acts as a THIN configuration layer. It defines search-space
callables for Auto-models (AutoLightGBM, AutoXGBoost, AutoNHITS) using Optuna.
It contains NO handwritten optimization loops (no study.optimize() or tune.Tuner()).
All search loops are managed internally by the Auto-adapters.

Legacy:
    The `ModelOptimizer` class is deprecated and preserved only for
    backward compatibility with legacy two-stage models.
"""

from typing import Dict, Any, Optional, Callable, Tuple
import warnings
import optuna


# ======================================================================
# Nixtlaverse Search-Space Configurations (Thin Layer)
# ======================================================================

def get_lightgbm_tweedie_config(eda_report: Optional[Any] = None) -> Callable[[optuna.Trial], Dict[str, Any]]:
    """Callable search-space config for AutoLightGBM with Tweedie loss.

    Zero-inflation is handled directly via Compound Poisson-Gamma (Tweedie).
    Variance power 1 < p < 2 spans between Poisson (p=1) and Gamma (p=2).
    """
    def _config(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'objective': 'tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'verbosity': -1,
        }
    return _config


def get_xgboost_tweedie_config(eda_report: Optional[Any] = None) -> Callable[[optuna.Trial], Dict[str, Any]]:
    """Callable search-space config for AutoXGBoost with Tweedie loss."""
    def _config(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        }
    return _config


def get_neural_tweedie_config(eda_report: Optional[Any] = None) -> Callable[[optuna.Trial], Dict[str, Any]]:
    """Callable search-space config for AutoNHITS with Tweedie loss."""
    def _config(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'max_steps': trial.suggest_int('max_steps', 200, 1000, step=100),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        }
    return _config


# ======================================================================
# Legacy Two-Stage Optimizer (Deprecated)
# ======================================================================

class ModelOptimizer:
    """Legacy hyperparameter optimizer for two-stage models.

    .. deprecated:: 2.0
       Direct Auto-models (AutoLightGBM, AutoXGBoost) in Nixtlaverse manage
       hyperparameter tuning internally via their own objective functions.
    """

    def __init__(
        self,
        n_splits: int = 3,
        n_trials: int = 50,
        random_state: int = 42,
        classification_threshold: float = 0.1,
    ):
        warnings.warn(
            "ModelOptimizer is deprecated. Nixtla Auto-models tune internally.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.classification_threshold = classification_threshold

    def optimize_two_stage_model(
        self,
        model_class,
        X,
        y,
        param_space_classification: Dict[str, Any],
        param_space_regression: Dict[str, Any],
        optimization_metric: str = 'combined',
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Legacy stub returning default params."""
        return ({}, {})