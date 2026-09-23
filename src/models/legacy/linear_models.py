# =============================================================================
# LINEAR MODELS: LOGISTIC REGRESSION + RIDGE/LASSO
# =============================================================================

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

from .base import BaseRainfallModel

warnings.filterwarnings('ignore')


class LinearRainfallModel(BaseRainfallModel):
    """
    Linear model for rainfall prediction.
    Two-stage: Logistic Regression + Ridge/Lasso
    Single-stage: Ridge/Lasso only
    """
    
    def __init__(self,
                 use_two_stage: bool = True,
                 classification_threshold: float = 0.1,
                 regression_type: str = 'ridge',
                 alpha: float = 1.0,
                 normalize_features: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Linear model.
        
        Args:
            use_two_stage: Whether to use two-stage approach
            classification_threshold: Threshold for rain/no-rain (mm/day)
            regression_type: 'ridge', 'lasso', or 'linear'
            alpha: Regularization strength
            normalize_features: Whether to normalize features
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(use_two_stage, classification_threshold, random_state)
        
        self.regression_type = regression_type
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.additional_params = kwargs
        
        # Scalers for feature normalization
        self.clf_scaler = None
        self.reg_scaler = None
        self.single_scaler = None
        
    def _create_classification_model(self, **kwargs):
        """Create Logistic Regression classifier."""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            **self.additional_params
        )
    
    def _create_regression_model(self, **kwargs):
        """Create regularized regression model."""
        if self.regression_type == 'ridge':
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.regression_type == 'lasso':
            return Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=1000)
        elif self.regression_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError("regression_type must be 'ridge', 'lasso', or 'linear'")
    
    def _create_single_stage_model(self, **kwargs):
        """Create single-stage regression model."""
        return self._create_regression_model(**kwargs)
    
    def _fit_two_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit two-stage linear model with normalization."""
        y_clf, y_reg = self._prepare_targets(y)
        
        print(f"   📊 Stage 1: Logistic Regression ({y_clf.sum()} rain days / {len(y_clf)} total)")
        
        # Stage 1: Classification with feature scaling
        if self.normalize_features:
            self.clf_scaler = StandardScaler()
            X_scaled = self.clf_scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        self.classification_model = self._create_classification_model(**kwargs)
        self.classification_model.fit(X_scaled, y_clf)
        
        # Stage 2: Regression (only on rainy days)
        rain_mask = y_clf == 1
        if rain_mask.sum() > 0:
            print(f"   📈 Stage 2: {self.regression_type.title()} Regression on {rain_mask.sum()} rainy days")
            X_rain = X[rain_mask]
            y_rain = y_reg[rain_mask]
            
            if self.normalize_features:
                self.reg_scaler = StandardScaler()
                X_rain_scaled = self.reg_scaler.fit_transform(X_rain)
                X_rain_scaled = pd.DataFrame(X_rain_scaled, columns=X_rain.columns, index=X_rain.index)
            else:
                X_rain_scaled = X_rain
            
            self.regression_model = self._create_regression_model(**kwargs)
            self.regression_model.fit(X_rain_scaled, y_rain)
        else:
            print("   ⚠️ No rainy days found for regression training")
            self.regression_model = None
            self.reg_scaler = None
    
    def _fit_single_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit single-stage linear model with normalization."""
        print(f"   📈 Single-stage {self.regression_type.title()} Regression on all {len(y)} samples")
        
        if self.normalize_features:
            self.single_scaler = StandardScaler()
            X_scaled = self.single_scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
            
        self.single_stage_model = self._create_single_stage_model(**kwargs)
        self.single_stage_model.fit(X_scaled, y)
    
    def _predict_two_stage(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Make two-stage predictions with normalization."""
        # Classification predictions
        if self.normalize_features and self.clf_scaler is not None:
            X_clf_scaled = self.clf_scaler.transform(X)
            X_clf_scaled = pd.DataFrame(X_clf_scaled, columns=X.columns, index=X.index)
        else:
            X_clf_scaled = X
        
        clf_probs = self.classification_model.predict_proba(X_clf_scaled)[:, 1]
        
        # Regression predictions
        if self.regression_model is not None:
            if self.normalize_features and self.reg_scaler is not None:
                X_reg_scaled = self.reg_scaler.transform(X)
                X_reg_scaled = pd.DataFrame(X_reg_scaled, columns=X.columns, index=X.index)
            else:
                X_reg_scaled = X
            
            reg_preds = self.regression_model.predict(X_reg_scaled)
        else:
            reg_preds = np.zeros(len(X))
        
        return clf_probs, reg_preds
    
    def _predict_single_stage(self, X: pd.DataFrame) -> np.ndarray:
        """Make single-stage predictions with normalization."""
        if self.normalize_features and self.single_scaler is not None:
            X_scaled = self.single_scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
            
        return self.single_stage_model.predict(X_scaled)
    
    def get_coefficients(self) -> Dict[str, Any]:
        """Get model coefficients."""
        coefs = {}
        
        if self.use_two_stage:
            if self.classification_model is not None:
                coefs['classification'] = {
                    'coefficients': self.classification_model.coef_[0],
                    'intercept': self.classification_model.intercept_[0]
                }
            if self.regression_model is not None:
                coefs['regression'] = {
                    'coefficients': self.regression_model.coef_,
                    'intercept': self.regression_model.intercept_
                }
        else:
            if self.single_stage_model is not None:
                coefs['single_stage'] = {
                    'coefficients': self.single_stage_model.coef_,
                    'intercept': self.single_stage_model.intercept_
                }
                
        return coefs 