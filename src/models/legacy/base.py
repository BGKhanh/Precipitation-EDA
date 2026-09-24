# =============================================================================
# BASE MODEL CLASSES WITH CONFIGURABLE TWO-STAGE APPROACH
# =============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_recall_curve, brier_score_loss, log_loss,
    precision_score, recall_score,
)
import warnings

warnings.filterwarnings('ignore')


class BaseRainfallModel(ABC):
    """
    Abstract base class for all rainfall prediction models.
    Supports both single-stage and two-stage approaches.
    """
    
    def __init__(self, 
                 use_two_stage: bool = True,
                 classification_threshold: float = 0.1,
                 random_state: int = 42):
        """
        Initialize base rainfall model.
        
        Args:
            use_two_stage: Whether to use two-stage approach (classification + regression)
            classification_threshold: Threshold for rain/no-rain classification (mm/day)
            random_state: Random seed for reproducibility
        """
        self.use_two_stage = use_two_stage
        self.classification_threshold = classification_threshold
        self.random_state = random_state
        self.is_fitted = False
        
        # Model storage
        self.classification_model = None
        self.regression_model = None
        self.single_stage_model = None
        self.calibrated_classifier_ = None
        
    @abstractmethod
    def _create_classification_model(self, **kwargs):
        """Create classification model for stage 1."""
        pass
    
    @abstractmethod
    def _create_regression_model(self, **kwargs):
        """Create regression model for stage 2."""
        pass
    
    @abstractmethod
    def _create_single_stage_model(self, **kwargs):
        """Create single-stage regression model."""
        pass
    
    def _prepare_targets(self, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare classification and regression targets.
        
        Hurdle Semantics Invariant:
            `classification_threshold` (c) defines the hurdle boundary:
            - y <= c: dry component (Z = 0)
            - y > c: wet component (Z = 1)
            The `expected` strategy (p * mu) assumes the dry component (Y <= c)
            contributes zero to the hurdle target: Y* = Y * 1(Y > c).
        
        Args:
            y: Original target variable (rainfall in mm/day)
            
        Returns:
            Tuple of (classification_target, regression_target)
        """
        # Classification: rain (1) vs no-rain (0)
        y_clf = (y > self.classification_threshold).astype(int)
        
        # Regression: original rainfall values
        y_reg = y.copy()
        
        return y_clf, y_reg
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit the model using either single-stage or two-stage approach.
        
        Args:
            X: Feature matrix
            y: Target variable (rainfall in mm/day)
            **kwargs: Additional parameters
        """
        print(f"Training {self.__class__.__name__}...")
        print(f"   Approach: {'Two-stage' if self.use_two_stage else 'Single-stage'}")
        
        if self.use_two_stage:
            self._fit_two_stage(X, y, **kwargs)
        else:
            self._fit_single_stage(X, y, **kwargs)
            
        self.is_fitted = True
        print("   Training completed")
    
    def _fit_two_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit two-stage model."""
        y_clf, y_reg = self._prepare_targets(y)
        
        print(f"   Stage 1: Classification ({y_clf.sum()} rain days / {len(y_clf)} total)")
        
        # Stage 1: Classification
        self.classification_model = self._create_classification_model(**kwargs)
        self.classification_model.fit(X, y_clf)
        
        # Stage 2: Regression (only on rainy days)
        rain_mask = y_clf == 1
        if rain_mask.sum() > 0:
            print(f"   Stage 2: Regression on {rain_mask.sum()} rainy days")
            X_rain = X[rain_mask]
            y_rain = y_reg[rain_mask]
            
            self.regression_model = self._create_regression_model(**kwargs)
            self.regression_model.fit(X_rain, y_rain)
        else:
            print("   Warning: No rainy days found for regression training")
            self.regression_model = None
    
    def _fit_single_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit single-stage model."""
        print(f"   Single-stage regression on all {len(y)} samples")
        self.single_stage_model = self._create_single_stage_model(**kwargs)
        self.single_stage_model.fit(X, y)

    def fit_calibrated(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "sigmoid",
        **kwargs,
    ) -> "BaseRainfallModel":
        """Fit base model on train data, then fit temporal-safe probability calibrator on calibration data.
        
        Temporal-Safe Lifecycle:
            TRAIN (past): fit base classifier & regressor
            CALIBRATION (subsequent in timeline): fit calibration mapping on frozen classifier
            VALIDATION (optional): tune decision threshold tau
            TEST: final evaluation
            
        Args:
            X_train: Training features (past)
            y_train: Training target
            X_cal: Calibration features (strictly following train in time, out-of-sample)
            y_cal: Calibration target
            method: 'sigmoid' (Platt scaling) or 'isotonic'
            **kwargs: Extra parameters passed to self.fit()
        """
        self.fit(X_train, y_train, **kwargs)
        
        if not self.use_two_stage:
            return self

        from sklearn.calibration import CalibratedClassifierCV
        try:
            from sklearn.frozen import FrozenEstimator
            frozen_clf = FrozenEstimator(self.classification_model)
        except ImportError:
            frozen_clf = self.classification_model

        y_cal_binary = (y_cal > self.classification_threshold).astype(int)
        self.calibrated_classifier_ = CalibratedClassifierCV(
            estimator=frozen_clf,
            method=method,
        )
        self.calibrated_classifier_.fit(X_cal, y_cal_binary)
        print(f"   Temporal probability calibration completed ({method})")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        return_calibrated: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            return_calibrated: If True and calibrated_classifier_ exists,
                use calibrated probabilities for stage 1.
            
        Returns:
            If two-stage: Tuple of (classification_probs, regression_values)
            If single-stage: Array of rainfall predictions
        """
        self._validate_fitted()
        
        if self.use_two_stage:
            return self._predict_two_stage(X, return_calibrated=return_calibrated)
        else:
            return self._predict_single_stage(X)
    
    def _predict_two_stage(
        self,
        X: pd.DataFrame,
        return_calibrated: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make two-stage predictions."""
        # Classification predictions (use calibrated classifier if available)
        clf_estimator = (
            self.calibrated_classifier_
            if (self.calibrated_classifier_ is not None and return_calibrated)
            else self.classification_model
        )

        if hasattr(clf_estimator, 'predict_proba'):
            clf_probs = clf_estimator.predict_proba(X)[:, 1]
        else:
            clf_probs = clf_estimator.predict(X)
        
        # Regression predictions
        if self.regression_model is not None:
            reg_preds = self.regression_model.predict(X)
        else:
            reg_preds = np.zeros(len(X))
            
        return clf_probs, reg_preds
    
    def _predict_single_stage(self, X: pd.DataFrame) -> np.ndarray:
        """Make single-stage predictions."""
        return self.single_stage_model.predict(X)
    
    def predict_combined(
        self,
        X: pd.DataFrame,
        threshold: float = None,
        strategy: str = "hard_gate",
        return_calibrated: bool = True,
    ) -> np.ndarray:
        """
        Make combined predictions for two-stage model.
        
        Args:
            X: Feature matrix
            threshold: Probability decision threshold tau for hard_gate. If None,
                uses self.optimal_threshold_ (set by find_optimal_threshold)
                or falls back to 0.5.
            strategy: Inference policy:
                - 'hard_gate' (default): if p >= tau -> reg_pred, else 0.0.
                - 'expected': plug-in estimate of unconditional mean: p * mu.
                  Assumes dry component (Y <= c) contributes zero to hurdle target.
            return_calibrated: Whether to use calibrated probabilities if fitted.
            
        Returns:
            Array of rainfall predictions (all non-negative).
        """
        if not self.use_two_stage:
            raise ValueError("Combined prediction only available for two-stage models")
        
        clf_probs, reg_preds = self.predict(X, return_calibrated=return_calibrated)
        
        if strategy == "hard_gate":
            if threshold is None:
                threshold = getattr(self, 'optimal_threshold_', 0.5)
            combined = np.zeros(len(X))
            rain_mask = clf_probs >= threshold
            combined[rain_mask] = np.maximum(0.0, reg_preds[rain_mask])
            return combined
        elif strategy == "expected":
            # Plug-in estimate: p_t * mu_t (clamped non-negative)
            mu_pos = np.maximum(0.0, reg_preds)
            return np.maximum(0.0, clf_probs * mu_pos)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported strategies: 'hard_gate', 'expected'."
            )
    
    def hurdleize_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Transform raw rainfall amounts Y into canonical hurdle target Y*.
        
        Semantics:
            Y* = Y  if Y > classification_threshold (c)
            Y* = 0  if Y <= classification_threshold (c)
            
        The two-stage expected plug-in estimate (p * mu) estimates E[Y*|X],
        assuming the dry component contributes zero rainfall to the hurdle target.
        """
        vals = np.asarray(y, dtype=float)
        return np.where(vals > self.classification_threshold, vals, 0.0)

    def find_optimal_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f1',
        return_calibrated: bool = True,
    ) -> float:
        """Find the optimal probability decision threshold tau on a validation set.

        Important Temporal Protocol:
            `tau` is a decision threshold selected on a dedicated validation partition.
            Validation data used for threshold tuning must NOT be reused as final
            unbiased test evaluation.
            
            Note: `tau` is the probability decision threshold (p >= tau -> rain),
            which is distinct from `classification_threshold` c (the rainfall amount
            defining physical wet/dry events).

        Args:
            X_val: Validation features.
            y_val: Validation target (rainfall mm/day).
            metric: Decision policy objective: 'f1' (default), 'csi', 'precision', or 'recall'.
            return_calibrated: Whether to use calibrated probabilities if calibrated_classifier_
                is fitted. Default True, ensuring tau is tuned on the exact probability
                scale used in downstream inference.

        Returns:
            Optimal probability threshold tau (float in (0, 1)).
        """
        self._validate_fitted()
        if not self.use_two_stage:
            raise ValueError("Threshold selection only for two-stage models")

        y_binary = (y_val > self.classification_threshold).astype(int)
        clf_probs, _ = self.predict(X_val, return_calibrated=return_calibrated)

        # Sweep thresholds
        best_score = -1.0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.91, 0.01):
            y_pred_binary = (clf_probs >= t).astype(int)
            if metric == 'f1':
                score = f1_score(y_binary, y_pred_binary, zero_division=0)
            elif metric == 'csi':
                hits = ((y_pred_binary == 1) & (y_binary == 1)).sum()
                misses = ((y_pred_binary == 0) & (y_binary == 1)).sum()
                false_alarms = ((y_pred_binary == 1) & (y_binary == 0)).sum()
                denom = hits + misses + false_alarms
                score = float(hits / denom) if denom > 0 else 0.0
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_binary, y_pred_binary, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_binary, y_pred_binary, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_thresh = t

        self.optimal_threshold_ = round(best_thresh, 2)
        print(f"   Optimal threshold: {self.optimal_threshold_} "
              f"({metric}={best_score:.4f})")
        return self.optimal_threshold_
    
    def _validate_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    Note: Time series models use single-stage approach only.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (3, 0, 3),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize time series model.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """Fit the time series model."""
        pass
    
    @abstractmethod
    def forecast(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate forecasts."""
        pass
    
    def _validate_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      task: str = 'regression',
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values (probabilities for classification/occurrence; continuous for regression)
        task: 'classification', 'occurrence', or 'regression'
        threshold: Decision threshold tau for binary occurrence metrics
        
    Returns:
        Dictionary of metrics
    """
    if task in ('classification', 'occurrence'):
        metrics = {}
        # 1. Probability metrics
        try:
            metrics['ROC_AUC'] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            metrics['ROC_AUC'] = np.nan
            
        try:
            metrics['Brier'] = float(brier_score_loss(y_true, y_pred))
        except Exception:
            metrics['Brier'] = np.nan
            
        try:
            clipped_preds = np.clip(y_pred, 1e-15, 1 - 1e-15)
            metrics['LogLoss'] = float(log_loss(y_true, clipped_preds))
        except Exception:
            metrics['LogLoss'] = np.nan
            
        # 2. Binary decision metrics at decision threshold tau
        y_binary_pred = (y_pred >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_binary_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_binary_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_binary_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_binary_pred == 0)))
        
        # POD (Probability of Detection / Recall / Hit Rate) = TP / (TP + FN)
        metrics['POD'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # FAR (False Alarm Ratio) = FP / (TP + FP)
        metrics['FAR'] = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # CSI (Critical Success Index / Threat Score) = TP / (TP + FN + FP)
        metrics['CSI'] = float(tp / (tp + fn + fp)) if (tp + fn + fp) > 0 else 0.0
        
        metrics['Precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics['Recall'] = metrics['POD']
        metrics['F1'] = float(f1_score(y_true, y_binary_pred, zero_division=0))
        return metrics
        
    elif task == 'regression':
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        bias = float(np.mean(y_pred - y_true))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = np.nan
        return {
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias,
            'R2': r2,
        }
    else:
        raise ValueError("Task must be 'classification', 'occurrence', or 'regression'")


def evaluate_rainfall_model(model: BaseRainfallModel,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            threshold: float = None) -> Dict[str, Any]:
    """
    Evaluate rainfall model performance across 3 canonical tiers:
    1. Occurrence (probability & binary metrics at decision threshold tau)
    2. Conditional Amount (only on rainy days y > c)
    3. Overall (evaluated against canonical hurdle target Y* = Y * 1(Y > c),
       comparing hard_gate and expected policies).

    Target Semantics:
        - Primary 'overall' tier evaluates predictions against the canonical hurdle target:
          Y* = Y if Y > c else 0.0. This aligns mathematically with the expected
          strategy which estimates E[Y*|X].
        - Supplementary 'overall_raw_supplementary' tier evaluates predictions against
          the un-truncated raw rainfall amounts Y.
    
    Args:
        model: Fitted rainfall model
        X_test: Test features
        y_test: Test targets
        threshold: Optional probability decision threshold tau
        
    Returns:
        Dictionary with structured 3-tier evaluation metrics
    """
    results = {
        'approach': 'two_stage' if model.use_two_stage else 'single_stage',
        'classification_threshold': model.classification_threshold,
    }
    
    if model.use_two_stage:
        tau = threshold if threshold is not None else getattr(model, 'optimal_threshold_', 0.5)
        results['probability_threshold_tau'] = tau
        
        y_clf_true, y_reg_true = model._prepare_targets(y_test)
        clf_probs, reg_preds = model.predict(X_test)
        
        # 1. Occurrence tier
        occ_metrics = calculate_metrics(y_clf_true.values, clf_probs, task='occurrence', threshold=tau)
        results['occurrence'] = occ_metrics
        results['classification'] = occ_metrics  # backward compatibility alias
        
        # 2. Conditional amount tier (only on rainy days y > c)
        rain_mask = (y_clf_true == 1).values
        if rain_mask.sum() > 0:
            reg_metrics = calculate_metrics(
                y_reg_true[rain_mask].values, 
                reg_preds[rain_mask], 
                task='regression'
            )
            results['conditional_amount'] = reg_metrics
            results['regression'] = reg_metrics  # backward compatibility alias
            results['rain_days_count'] = int(rain_mask.sum())
        else:
            empty_reg = {'MAE': np.nan, 'RMSE': np.nan, 'Bias': np.nan, 'R2': np.nan}
            results['conditional_amount'] = empty_reg
            results['regression'] = empty_reg
            results['rain_days_count'] = 0
            
        # 3. Overall tier (evaluated against canonical hurdle target Y* = Y * 1(Y > c))
        overall_hard = model.predict_combined(X_test, threshold=tau, strategy='hard_gate')
        overall_expected = model.predict_combined(X_test, strategy='expected')
        y_test_hurdle = model.hurdleize_target(y_test.values)
        
        results['overall'] = {
            'hard_gate': calculate_metrics(y_test_hurdle, overall_hard, task='regression'),
            'expected': calculate_metrics(y_test_hurdle, overall_expected, task='regression'),
        }
        # Backward compatibility alias: combined maps to hard_gate on hurdle target
        results['combined'] = results['overall']['hard_gate']

        # Supplementary tier: evaluated against un-truncated raw rainfall amounts Y
        results['overall_raw_supplementary'] = {
            'hard_gate': calculate_metrics(y_test.values, overall_hard, task='regression'),
            'expected': calculate_metrics(y_test.values, overall_expected, task='regression'),
        }
        
    else:
        # Single-stage evaluation
        predictions = model.predict(X_test)
        single_metrics = calculate_metrics(y_test.values, predictions, task='regression')
        results['single_stage'] = single_metrics
        results['overall'] = single_metrics
        if hasattr(model, 'hurdleize_target'):
            y_test_hurdle = model.hurdleize_target(y_test.values)
            results['overall_hurdle'] = calculate_metrics(y_test_hurdle, predictions, task='regression')
    
    results['total_samples'] = len(y_test)
    return results


def evaluate_per_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Break down MAE/RMSE by forecast horizon.

    This makes error compounding visible — day-1 forecasts are typically
    better than day-7 forecasts, and this function quantifies exactly how
    much accuracy degrades with horizon.

    Args:
        y_true: Ground-truth values (ordered by horizon 1, 2, …, N).
        y_pred: Predicted values (same order).
        horizons: If ``None``, each element is treated as horizon 1..N.

    Returns:
        DataFrame with columns ``['horizon', 'MAE', 'RMSE', 'n_samples']``.
    """
    n = len(y_true)
    if horizons is None:
        horizons = list(range(1, n + 1))

    rows = []
    for h in sorted(set(horizons)):
        mask = [i for i, hh in enumerate(horizons) if hh == h]
        if not mask:
            continue
        yt = np.array(y_true)[mask]
        yp = np.array(y_pred)[mask]
        rows.append({
            'horizon': h,
            'MAE': mean_absolute_error(yt, yp),
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'n_samples': len(mask),
        })

    return pd.DataFrame(rows) 