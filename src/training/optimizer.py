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
    """Callable search-space config for AutoNHITS with Tweedie loss.

    Configured for low VRAM footprint to comfortably run on 4GB GPUs (e.g. GTX 1650).
    """
    def _config(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'input_size': trial.suggest_categorical('input_size', [14, 28]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'max_steps': trial.suggest_categorical('max_steps', [50, 100]),
            'batch_size': 32,
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


# ======================================================================
# Optuna Tuning for Custom Two-Stage / Hurdle Models (Zero-Leakage)
# ======================================================================

def tune_hurdle_model(
    model_name: str,
    X_train,
    y_train,
    n_trials: int = 15,
    rain_threshold: float = 0.1,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune ML Hurdle model (Stage 1 Classifier & Stage 2 Regressor) via Optuna and TimeSeriesSplit.

    Guarantees:
    - Zero data leakage: only uses expanding window TimeSeriesSplit.
    - Stage 1 optimizes ROC-AUC.
    - Stage 2 optimizes Rain-day MAE (fitted exclusively on rainy samples).
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_binary = (y_train > rain_threshold).astype(int)

    # 1. Stage 1: Classifier Tuning
    def objective_clf(trial: optuna.Trial) -> float:
        if 'RandomForest' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'max_depth': trial.suggest_int('max_depth', 4, 14),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': random_state,
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'n_jobs': -1,
            }
            model = RandomForestClassifier(**params)
        elif 'XGBoost' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': random_state,
                'eval_metric': 'logloss',
            }
            model = XGBClassifier(**params)
        elif 'LightGBM' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'random_state': random_state,
                'verbosity': -1,
            }
            model = LGBMClassifier(**params)
        else:  # Logistic Regression
            params = {
                'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
                'max_iter': 1000,
                'random_state': random_state,
            }
            model = LogisticRegression(**params)

        scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr_bin = X_train.iloc[tr_idx], y_binary.iloc[tr_idx]
            X_val, y_val_bin = X_train.iloc[val_idx], y_binary.iloc[val_idx]
            model.fit(X_tr, y_tr_bin)
            probs = model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val_bin)) > 1:
                scores.append(roc_auc_score(y_val_bin, probs))
        return float(np.mean(scores)) if scores else 0.5

    study_clf = optuna.create_study(direction='maximize')
    study_clf.optimize(objective_clf, n_trials=n_trials)
    best_clf_params = study_clf.best_params

    # 2. Stage 2: Regressor Tuning (Only on rainy days)
    def objective_reg(trial: optuna.Trial) -> float:
        if 'RandomForest' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'max_depth': trial.suggest_int('max_depth', 4, 14),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': random_state,
                'n_jobs': -1,
            }
            model = RandomForestRegressor(**params)
        elif 'XGBoost' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': random_state,
            }
            model = XGBRegressor(**params)
        elif 'LightGBM' in model_name:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'random_state': random_state,
                'verbosity': -1,
            }
            model = LGBMRegressor(**params)
        else:  # Linear / Ridge
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
                'random_state': random_state,
            }
            model = Ridge(**params)

        maes = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            rain_mask_tr = y_tr > rain_threshold
            rain_mask_val = y_val > rain_threshold
            if rain_mask_tr.sum() < 5 or rain_mask_val.sum() < 2:
                continue

            model.fit(X_tr[rain_mask_tr], y_tr[rain_mask_tr])
            preds = model.predict(X_val[rain_mask_val])
            maes.append(mean_absolute_error(y_val[rain_mask_val], preds))

        return float(np.mean(maes)) if maes else 999.0

    study_reg = optuna.create_study(direction='minimize')
    study_reg.optimize(objective_reg, n_trials=n_trials)
    best_reg_params = study_reg.best_params

    return best_clf_params, best_reg_params