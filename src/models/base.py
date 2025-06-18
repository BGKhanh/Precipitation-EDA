# =============================================================================
# BASE MODEL CLASSES WITH CONFIGURABLE TWO-STAGE APPROACH
# =============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
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
        print(f"🚀 Training {self.__class__.__name__}...")
        print(f"   Approach: {'Two-stage' if self.use_two_stage else 'Single-stage'}")
        
        if self.use_two_stage:
            self._fit_two_stage(X, y, **kwargs)
        else:
            self._fit_single_stage(X, y, **kwargs)
            
        self.is_fitted = True
        print("   ✅ Training completed")
    
    def _fit_two_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit two-stage model."""
        y_clf, y_reg = self._prepare_targets(y)
        
        print(f"   📊 Stage 1: Classification ({y_clf.sum()} rain days / {len(y_clf)} total)")
        
        # Stage 1: Classification
        self.classification_model = self._create_classification_model(**kwargs)
        self.classification_model.fit(X, y_clf)
        
        # Stage 2: Regression (only on rainy days)
        rain_mask = y_clf == 1
        if rain_mask.sum() > 0:
            print(f"   📈 Stage 2: Regression on {rain_mask.sum()} rainy days")
            X_rain = X[rain_mask]
            y_rain = y_reg[rain_mask]
            
            self.regression_model = self._create_regression_model(**kwargs)
            self.regression_model.fit(X_rain, y_rain)
        else:
            print("   ⚠️ No rainy days found for regression training")
            self.regression_model = None
    
    def _fit_single_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit single-stage model."""
        print(f"   📈 Single-stage regression on all {len(y)} samples")
        self.single_stage_model = self._create_single_stage_model(**kwargs)
        self.single_stage_model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            If two-stage: Tuple of (classification_probs, regression_values)
            If single-stage: Array of rainfall predictions
        """
        self._validate_fitted()
        
        if self.use_two_stage:
            return self._predict_two_stage(X)
        else:
            return self._predict_single_stage(X)
    
    def _predict_two_stage(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make two-stage predictions."""
        # Classification predictions
        if hasattr(self.classification_model, 'predict_proba'):
            clf_probs = self.classification_model.predict_proba(X)[:, 1]
        else:
            clf_probs = self.classification_model.predict(X)
        
        # Regression predictions
        if self.regression_model is not None:
            reg_preds = self.regression_model.predict(X)
        else:
            reg_preds = np.zeros(len(X))
            
        return clf_probs, reg_preds
    
    def _predict_single_stage(self, X: pd.DataFrame) -> np.ndarray:
        """Make single-stage predictions."""
        return self.single_stage_model.predict(X)
    
    def predict_combined(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make combined predictions for two-stage model.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Combined predictions (0 for no rain, regression value for rain)
        """
        if not self.use_two_stage:
            raise ValueError("Combined prediction only available for two-stage models")
            
        clf_probs, reg_preds = self.predict(X)
        
        combined = np.zeros(len(X))
        rain_mask = clf_probs >= threshold
        combined[rain_mask] = reg_preds[rain_mask]
        
        return combined
    
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
                     task: str = 'regression') -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    if task == 'classification':
        return {
            'ROC_AUC': roc_auc_score(y_true, y_pred)
        }
    elif task == 'regression':
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
    else:
        raise ValueError("Task must be 'classification' or 'regression'")


def evaluate_rainfall_model(model: BaseRainfallModel,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate rainfall model performance.
    
    Args:
        model: Fitted rainfall model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'approach': 'two_stage' if model.use_two_stage else 'single_stage',
        'classification_threshold': model.classification_threshold
    }
    
    if model.use_two_stage:
        # Two-stage evaluation
        y_clf_true, y_reg_true = model._prepare_targets(y_test)
        clf_probs, reg_preds = model.predict(X_test)
        
        # Classification metrics
        clf_metrics = calculate_metrics(y_clf_true, clf_probs, task='classification')
        results['classification'] = clf_metrics
        
        # Regression metrics (only on rainy days)
        rain_mask = y_clf_true == 1
        if rain_mask.sum() > 0:
            reg_metrics = calculate_metrics(
                y_reg_true[rain_mask], 
                reg_preds[rain_mask], 
                task='regression'
            )
            results['regression'] = reg_metrics
            results['rain_days_count'] = rain_mask.sum()
        else:
            results['regression'] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
            results['rain_days_count'] = 0
            
        # Combined prediction metrics
        combined_preds = model.predict_combined(X_test)
        combined_metrics = calculate_metrics(y_test, combined_preds, task='regression')
        results['combined'] = combined_metrics
        
    else:
        # Single-stage evaluation
        predictions = model.predict(X_test)
        single_metrics = calculate_metrics(y_test, predictions, task='regression')
        results['single_stage'] = single_metrics
    
    results['total_samples'] = len(y_test)
    return results 