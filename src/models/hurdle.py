"""Direct Multi-Horizon Two-Stage Hurdle Forecaster (Decision M1 & PROP-02 — LOCKED).

Implements the canonical per-horizon Direct Hurdle architecture preserving
continuous rainfall targets Y_t in [0, +inf) mm:
- Threshold: tau_rain = 1.0 mm/day (ETCCDI RR1 / WMO standard, inclusive >= 1.0)
- Per-horizon independent classifier C_h and regressor R_h for h in 1..H
- Per-horizon Stage 2 sample filtering: S_{2, h} = {(X_t, y_{t+h}) | y_{t+h} >= 1.0 mm}
- Constant dry-regime baseline fitted once on Dev Train dry observations (y < 1.0 mm):
  hat{mu}_{dry} approx 0.1419849... mm (approx 0.142 mm)
- Stage 2 support domain clipping: hat{mu}_{wet, t+h} = max(hat{y}_{reg, t+h}, 1.0 mm)
- Combination Strategy A (expected_value):
    hat{y}_{t+h} = hat{p}_{t+h} * hat{mu}_{wet, t+h} + (1 - hat{p}_{t+h}) * hat{mu}_{dry}
- Combination Strategy B (hard_gate):
    hat{y}_{t+h} = hat{mu}_{wet, t+h} if hat{p}_{t+h} >= tau_clf else hat{mu}_{dry}
  with global threshold grid 3a: tau_clf in [0.10, 0.90] with step 0.05 (17 points),
  optimized on Validation Pooled MAE (tie-breaker 4a: closest to Dev Train wet rate ~0.4340).
- Boundary hit detection: sets warning flag HURDLE_THRESHOLD_BOUNDARY_HIT if tau* in {0.10, 0.90}.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import mean_absolute_error

CANONICAL_RAIN_THRESHOLD_MM: float = 1.0
DEV_TRAIN_BASE_RR1_WET_RATE: float = 0.4340

# Grid 3a: 17 candidate thresholds from 0.10 to 0.90 with step 0.05
DEFAULT_TAU_CLF_GRID: Tuple[float, ...] = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))
HARD_GATE_THRESHOLD_GRID: Tuple[float, ...] = DEFAULT_TAU_CLF_GRID


class DirectHurdleForecaster:
    """Direct Multi-Horizon Two-Stage Hurdle Forecaster per Decision M1 and PROP-02."""

    def __init__(
        self,
        classifier_factory: Optional[Callable[[], Any]] = None,
        regressor_factory: Optional[Callable[[], Any]] = None,
        horizon: int = 7,
        rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
        strategy: str = "expected_value",
        tau_clf: float = 0.50,
        random_state: int = 42,
        model_alias: str = "LightGBM_Hurdle",
    ) -> None:
        if abs(float(rain_threshold) - CANONICAL_RAIN_THRESHOLD_MM) > 1e-6:
            raise ValueError(
                f"DirectHurdleForecaster requires canonical rain_threshold=1.0 mm, got {rain_threshold}"
            )
        if strategy not in ("expected_value", "hard_gate"):
            raise ValueError(
                f"strategy must be 'expected_value' or 'hard_gate', got '{strategy}'"
            )

        self.horizon = int(horizon)
        self.rain_threshold = float(rain_threshold)
        self.strategy = strategy
        self.tau_clf = float(tau_clf)
        self.random_state = int(random_state)
        self.model_alias = str(model_alias)

        self.classifier_factory = classifier_factory or self._default_classifier_factory
        self.regressor_factory = regressor_factory or self._default_regressor_factory

        self.classifiers_: List[Any] = []
        self.regressors_: List[Any] = []
        self.fitted_dry_mean_: Optional[float] = None
        self.horizon_wet_counts_: List[int] = []
        self.is_fitted_: bool = False
        self.selected_strategy_: str = strategy
        self.selected_tau_clf_: float = float(tau_clf)
        self.validation_selection_summary_: Optional[Dict[str, Any]] = None
        self.warning_flags_: List[str] = []

    def _default_classifier_factory(self) -> Any:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=self.random_state,
        )

    def _default_regressor_factory(self) -> Any:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=self.random_state,
        )

    @staticmethod
    def compute_dev_train_dry_mean(
        y_dev: Union[np.ndarray, pd.Series, Sequence[float]],
        rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
    ) -> float:
        """Compute constant dry mean hat{mu}_{dry} strictly on Dev Train observations y < 1.0 mm."""
        y_arr = np.asarray(y_dev, dtype=np.float64).reshape(-1)
        y_arr = y_arr[np.isfinite(y_arr)]
        dry_mask = y_arr < rain_threshold
        if not np.any(dry_mask):
            return 0.0
        return float(np.mean(y_arr[dry_mask]))

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        Y_train: np.ndarray,
        y_dev_daily: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "DirectHurdleForecaster":
        """Fit per-horizon Stage 1 classifiers and Stage 2 conditional regressors on Dev Train.

        Args:
            X_train: Feature matrix of shape (N_origins, n_features) from Dev Train.
            Y_train: Target matrix of shape (N_origins, H) with continuous rainfall (mm).
            y_dev_daily: Optional full 1D daily target series on Dev Train (6,329 days)
                used to compute hat{mu}_{dry}. If omitted, computed from Y_train[:, 0].
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        Y_arr = np.asarray(Y_train, dtype=np.float64)

        if Y_arr.ndim != 2 or Y_arr.shape[1] != self.horizon:
            raise ValueError(
                f"Y_train must have shape (N, {self.horizon}), got {Y_arr.shape}"
            )
        if X_arr.shape[0] != Y_arr.shape[0]:
            raise ValueError(
                f"X_train rows ({X_arr.shape[0]}) != Y_train rows ({Y_arr.shape[0]})"
            )

        dry_source = y_dev_daily if y_dev_daily is not None else Y_arr[:, 0]
        self.fitted_dry_mean_ = self.compute_dev_train_dry_mean(
            dry_source, rain_threshold=self.rain_threshold
        )

        self.classifiers_ = []
        self.regressors_ = []
        self.horizon_wet_counts_ = []

        for h_idx in range(self.horizon):
            y_h = Y_arr[:, h_idx]
            wet_mask_h = y_h >= self.rain_threshold
            y_clf_h = wet_mask_h.astype(int)

            clf_h = self.classifier_factory()
            clf_h.fit(X_arr, y_clf_h)
            self.classifiers_.append(clf_h)

            wet_count = int(np.sum(wet_mask_h))
            self.horizon_wet_counts_.append(wet_count)
            if wet_count == 0:
                raise ValueError(
                    f"Horizon h={h_idx + 1} has 0 wet samples (y >= {self.rain_threshold} mm) in Dev Train."
                )

            X_wet_h = X_arr[wet_mask_h]
            y_wet_h = y_h[wet_mask_h]

            reg_h = self.regressor_factory()
            reg_h.fit(X_wet_h, y_wet_h)
            self.regressors_.append(reg_h)

        self.is_fitted_ = True
        return self

    def predict_components(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P_hat, Mu_wet_hat) each of shape (N, H).

        Applies mandatory Stage 2 support domain clipping:
            hat{mu}_{wet, t+h} = max(hat{y}_{reg, t+h}, 1.0 mm).
        """
        if not self.is_fitted_ or self.fitted_dry_mean_ is None:
            raise RuntimeError("DirectHurdleForecaster must be fitted before prediction.")

        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        P_hat = np.zeros((n_samples, self.horizon), dtype=np.float64)
        Mu_wet_hat = np.zeros((n_samples, self.horizon), dtype=np.float64)

        for h_idx in range(self.horizon):
            clf_h = self.classifiers_[h_idx]
            reg_h = self.regressors_[h_idx]

            if hasattr(clf_h, "predict_proba"):
                probs = clf_h.predict_proba(X_arr)
                p_h = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                p_h = np.clip(clf_h.predict(X_arr), 0.0, 1.0)

            raw_reg_h = np.asarray(reg_h.predict(X_arr), dtype=np.float64).reshape(-1)
            # Support domain clipping >= 1.0 mm per Decision M1 (§1.2.B.2)
            mu_wet_h = np.maximum(raw_reg_h, self.rain_threshold)

            P_hat[:, h_idx] = np.clip(p_h, 0.0, 1.0)
            Mu_wet_hat[:, h_idx] = mu_wet_h

        return P_hat, Mu_wet_hat

    def combine_forecasts(
        self,
        P_hat: np.ndarray,
        Mu_wet_hat: np.ndarray,
        strategy: Optional[str] = None,
        tau_clf: Optional[float] = None,
    ) -> np.ndarray:
        """Combine Stage 1 probabilities and Stage 2 clipped amounts into continuous forecasts.

        - Strategy A (expected_value):
            y_hat = p * mu_wet + (1 - p) * mu_dry
        - Strategy B (hard_gate):
            y_hat = mu_wet if p >= tau else mu_dry (Decision M1: dry constant mu_dry, NOT zero!)
        """
        if self.fitted_dry_mean_ is None:
            raise RuntimeError("fitted_dry_mean_ is not initialized.")

        strat = strategy or self.selected_strategy_
        tau = float(self.selected_tau_clf_ if tau_clf is None else tau_clf)
        mu_dry = float(self.fitted_dry_mean_)
        mu_wet = np.maximum(Mu_wet_hat, self.rain_threshold)

        if strat == "expected_value":
            y_hat = P_hat * mu_wet + (1.0 - P_hat) * mu_dry
        elif strat == "hard_gate":
            y_hat = np.where(P_hat >= tau, mu_wet, mu_dry)
        else:
            raise ValueError(f"Unknown combination strategy: '{strat}'")

        return np.maximum(0.0, y_hat)

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        strategy: Optional[str] = None,
        tau_clf: Optional[float] = None,
    ) -> np.ndarray:
        """Predict continuous multi-horizon rainfall matrix of shape (N, H)."""
        P_hat, Mu_wet_hat = self.predict_components(X)
        return self.combine_forecasts(P_hat, Mu_wet_hat, strategy=strategy, tau_clf=tau_clf)

    def tune_threshold_on_validation(
        self,
        X_val: Union[np.ndarray, pd.DataFrame],
        Y_val: np.ndarray,
        candidate_thresholds: Sequence[float] = DEFAULT_TAU_CLF_GRID,
    ) -> float:
        """Tune global hard_gate threshold tau* on Validation Pooled MAE per PROP-02.

        - Search Grid 3a: 17 candidate points in [0.10, 0.90] with step 0.05.
        - Tie-break Rule 4a: Closest to Dev Train base wet rate ~0.4340.
        - Boundary hit: Sets HURDLE_THRESHOLD_BOUNDARY_HIT if tau* in {0.10, 0.90}.
        """
        Y_val_arr = np.asarray(Y_val, dtype=np.float64)
        P_val, Mu_wet_val = self.predict_components(X_val)

        best_tau = 0.50
        best_mae = float("inf")
        grid_records: List[Dict[str, float]] = []

        for tau in candidate_thresholds:
            tau_f = float(tau)
            preds_hg = self.combine_forecasts(
                P_val, Mu_wet_val, strategy="hard_gate", tau_clf=tau_f
            )
            mae_tau = mean_absolute_error(Y_val_arr, preds_hg)
            grid_records.append({"tau_clf": tau_f, "val_mae_pooled": mae_tau})

            if mae_tau < best_mae - 1e-12:
                best_mae = mae_tau
                best_tau = tau_f
            elif abs(mae_tau - best_mae) <= 1e-12:
                # Tie-breaker Rule 4a: Closest to Dev Train wet rate ~0.4340
                if abs(tau_f - DEV_TRAIN_BASE_RR1_WET_RATE) < abs(best_tau - DEV_TRAIN_BASE_RR1_WET_RATE):
                    best_tau = tau_f

        self.selected_tau_clf_ = best_tau
        self.tau_clf = best_tau
        self.strategy = "hard_gate"
        self.selected_strategy_ = "hard_gate"

        self.warning_flags_ = []
        if best_tau in (0.10, 0.90):
            self.warning_flags_.append("HURDLE_THRESHOLD_BOUNDARY_HIT")

        self.validation_selection_summary_ = {
            "selected_strategy": "hard_gate",
            "selected_tau_clf": best_tau,
            "winning_val_mae_pooled": best_mae,
            "hard_gate_grid": grid_records,
            "warning_flags": list(self.warning_flags_),
            "fitted_dry_mean_mm": float(self.fitted_dry_mean_ or 0.0),
        }
        return best_tau

    def select_optimal_strategy_on_validation(
        self,
        X_val: Union[np.ndarray, pd.DataFrame],
        Y_val: np.ndarray,
        candidate_thresholds: Sequence[float] = DEFAULT_TAU_CLF_GRID,
    ) -> Dict[str, Any]:
        """Compare Strategy A (expected_value) vs Strategy B (hard_gate) on Validation Pooled MAE.

        Optimization criterion: Minimum Validation Pooled MAE.
        Tie-breaker for tau_clf in Strategy B: Rule 4a (closest to base Dev Train wet rate ~0.4340).
        Freezes selected_strategy_ and selected_tau_clf_ on the instance.
        """
        Y_val_arr = np.asarray(Y_val, dtype=np.float64)
        P_val, Mu_wet_val = self.predict_components(X_val)

        # 1. Strategy A: Expected Value
        preds_exp = self.combine_forecasts(P_val, Mu_wet_val, strategy="expected_value")
        mae_exp = mean_absolute_error(Y_val_arr, preds_exp)

        # 2. Strategy B: Hard-Gate global threshold grid search
        grid_records: List[Dict[str, float]] = []
        best_tau = 0.50
        best_hg_mae = float("inf")

        for tau in candidate_thresholds:
            tau_f = float(tau)
            preds_hg = self.combine_forecasts(
                P_val, Mu_wet_val, strategy="hard_gate", tau_clf=tau_f
            )
            mae_tau = mean_absolute_error(Y_val_arr, preds_hg)
            grid_records.append({"tau_clf": tau_f, "val_mae_pooled": mae_tau})

            if mae_tau < best_hg_mae - 1e-12:
                best_hg_mae = mae_tau
                best_tau = tau_f
            elif abs(mae_tau - best_hg_mae) <= 1e-12:
                # Rule 4a tie-break: Closest to ~0.4340
                if abs(tau_f - DEV_TRAIN_BASE_RR1_WET_RATE) < abs(best_tau - DEV_TRAIN_BASE_RR1_WET_RATE):
                    best_tau = tau_f

        if mae_exp <= best_hg_mae:
            self.selected_strategy_ = "expected_value"
            self.selected_tau_clf_ = best_tau
            winning_mae = mae_exp
        else:
            self.selected_strategy_ = "hard_gate"
            self.selected_tau_clf_ = best_tau
            winning_mae = best_hg_mae

        self.strategy = self.selected_strategy_
        self.tau_clf = self.selected_tau_clf_

        self.warning_flags_ = []
        if best_tau in (0.10, 0.90) and self.selected_strategy_ == "hard_gate":
            self.warning_flags_.append("HURDLE_THRESHOLD_BOUNDARY_HIT")

        summary = {
            "selected_strategy": self.selected_strategy_,
            "selected_tau_clf": self.selected_tau_clf_,
            "winning_val_mae_pooled": winning_mae,
            "expected_value_val_mae_pooled": mae_exp,
            "hard_gate_best_val_mae_pooled": best_hg_mae,
            "hard_gate_best_tau_clf": best_tau,
            "hard_gate_grid": grid_records,
            "warning_flags": list(self.warning_flags_),
            "fitted_dry_mean_mm": float(self.fitted_dry_mean_ or 0.0),
        }
        self.validation_selection_summary_ = summary
        return summary

    select_strategy_on_validation = select_optimal_strategy_on_validation


def build_logistic_linear_hurdle(
    horizon: int = 7,
    rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
    random_state: int = 42,
) -> DirectHurdleForecaster:
    """Factory for [23] Logistic+Linear_Hurdle baseline per Decision M1."""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def clf_factory() -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        )

    def reg_factory() -> Any:
        return make_pipeline(
            StandardScaler(),
            LinearRegression(),
        )

    return DirectHurdleForecaster(
        classifier_factory=clf_factory,
        regressor_factory=reg_factory,
        horizon=horizon,
        rain_threshold=rain_threshold,
        random_state=random_state,
        model_alias="Logistic+Linear_Hurdle",
    )


def build_lightgbm_hurdle(
    horizon: int = 7,
    rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
    random_state: int = 42,
) -> DirectHurdleForecaster:
    """Factory for [24] LightGBM_Hurdle per Decision M1."""
    return DirectHurdleForecaster(
        horizon=horizon,
        rain_threshold=rain_threshold,
        random_state=random_state,
        model_alias="LightGBM_Hurdle",
    )


def build_random_forest_hurdle(
    horizon: int = 7,
    rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
    random_state: int = 42,
) -> DirectHurdleForecaster:
    """Factory for RandomForest_Hurdle baseline."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def clf_factory() -> Any:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    def reg_factory() -> Any:
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    return DirectHurdleForecaster(
        classifier_factory=clf_factory,
        regressor_factory=reg_factory,
        horizon=horizon,
        rain_threshold=rain_threshold,
        random_state=random_state,
        model_alias="RandomForest_Hurdle",
    )


def build_xgboost_hurdle(
    horizon: int = 7,
    rain_threshold: float = CANONICAL_RAIN_THRESHOLD_MM,
    random_state: int = 42,
) -> DirectHurdleForecaster:
    """Factory for XGBoost_Hurdle baseline."""
    from xgboost import XGBClassifier, XGBRegressor

    def clf_factory() -> Any:
        return XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )

    def reg_factory() -> Any:
        return XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )

    return DirectHurdleForecaster(
        classifier_factory=clf_factory,
        regressor_factory=reg_factory,
        horizon=horizon,
        rain_threshold=rain_threshold,
        random_state=random_state,
        model_alias="XGBoost_Hurdle",
    )


# Alias for backward compatibility
HurdleModel = DirectHurdleForecaster


