# =============================================================================
# TREE-BASED MODELS: RANDOM FOREST, XGBOOST, LIGHTGBM
# =============================================================================

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings

from .base import BaseRainfallModel

warnings.filterwarnings('ignore')


class RandomForestRainfallModel(BaseRainfallModel):
    """
    Random Forest model for rainfall prediction.
    Supports both single-stage and two-stage approaches.
    """
    
    def __init__(self,
                 use_two_stage: bool = True,
                 classification_threshold: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            use_two_stage: Whether to use two-stage approach
            classification_threshold: Threshold for rain/no-rain (mm/day)
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            random_state: Random seed
            **kwargs: Additional RandomForest parameters
        """
        super().__init__(use_two_stage, classification_threshold, random_state)
        
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            **kwargs
        }
    
    def _create_classification_model(self, **kwargs):
        """Create Random Forest classifier."""
        return RandomForestClassifier(**self.rf_params)
    
    def _create_regression_model(self, **kwargs):
        """Create Random Forest regressor."""
        return RandomForestRegressor(**self.rf_params)
    
    def _create_single_stage_model(self, **kwargs):
        """Create single-stage Random Forest regressor."""
        return RandomForestRegressor(**self.rf_params)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance."""
        importance = {}
        
        if self.use_two_stage:
            if self.classification_model is not None:
                importance['classification'] = self.classification_model.feature_importances_
            if self.regression_model is not None:
                importance['regression'] = self.regression_model.feature_importances_
        else:
            if self.single_stage_model is not None:
                importance['single_stage'] = self.single_stage_model.feature_importances_
                
        return importance


class LightGBMRainfallModel(BaseRainfallModel):
    """
    LightGBM model for rainfall prediction.
    Supports both single-stage and two-stage approaches.
    """
    
    def __init__(self,
                 use_two_stage: bool = True,
                 classification_threshold: float = 0.1,
                 num_boost_round: int = 100,
                 clf_params: Optional[Dict] = None,
                 reg_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize LightGBM model.
        
        Args:
            use_two_stage: Whether to use two-stage approach
            classification_threshold: Threshold for rain/no-rain (mm/day)
            num_boost_round: Number of boosting rounds
            clf_params: Classification parameters
            reg_params: Regression parameters
            random_state: Random seed
        """
        super().__init__(use_two_stage, classification_threshold, random_state)
        
        self.num_boost_round = num_boost_round
        
        # Default classification parameters
        self.clf_params = clf_params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_child_samples': 20,
            'verbosity': -1,
            'random_state': random_state
        }
        
        # Default regression parameters
        self.reg_params = reg_params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_child_samples': 20,
            'verbosity': -1,
            'random_state': random_state
        }
    
    def _create_classification_model(self, **kwargs):
        """Create LightGBM classifier wrapper."""
        return LightGBMWrapper(self.clf_params, self.num_boost_round, task='classification')
    
    def _create_regression_model(self, **kwargs):
        """Create LightGBM regressor wrapper."""
        return LightGBMWrapper(self.reg_params, self.num_boost_round, task='regression')
    
    def _create_single_stage_model(self, **kwargs):
        """Create single-stage LightGBM regressor."""
        return LightGBMWrapper(self.reg_params, self.num_boost_round, task='regression')


class XGBoostRainfallModel(BaseRainfallModel):
    """
    XGBoost model for rainfall prediction.
    Supports both single-stage and two-stage approaches.
    """
    
    def __init__(self,
                 use_two_stage: bool = True,
                 classification_threshold: float = 0.1,
                 clf_params: Optional[Dict] = None,
                 reg_params: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            use_two_stage: Whether to use two-stage approach
            classification_threshold: Threshold for rain/no-rain (mm/day)
            clf_params: Classification parameters
            reg_params: Regression parameters
            random_state: Random seed
        """
        super().__init__(use_two_stage, classification_threshold, random_state)
        
        # Default classification parameters
        self.clf_params = clf_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'verbosity': 0,
            'random_state': random_state
        }
        
        # Default regression parameters
        self.reg_params = reg_params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'verbosity': 0,
            'random_state': random_state
        }
    
    def _create_classification_model(self, **kwargs):
        """Create XGBoost classifier."""
        return xgb.XGBClassifier(**self.clf_params)
    
    def _create_regression_model(self, **kwargs):
        """Create XGBoost regressor."""
        return xgb.XGBRegressor(**self.reg_params)
    
    def _create_single_stage_model(self, **kwargs):
        """Create single-stage XGBoost regressor."""
        return xgb.XGBRegressor(**self.reg_params)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance."""
        importance = {}
        
        if self.use_two_stage:
            if self.classification_model is not None:
                importance['classification'] = self.classification_model.feature_importances_
            if self.regression_model is not None:
                importance['regression'] = self.regression_model.feature_importances_
        else:
            if self.single_stage_model is not None:
                importance['single_stage'] = self.single_stage_model.feature_importances_
                
        return importance


class LightGBMWrapper:
    """
    Wrapper class for LightGBM to provide sklearn-like interface.
    """
    
    def __init__(self, params: Dict, num_boost_round: int, task: str):
        """
        Initialize LightGBM wrapper.
        
        Args:
            params: LightGBM parameters
            num_boost_round: Number of boosting rounds
            task: 'classification' or 'regression'
        """
        self.params = params
        self.num_boost_round = num_boost_round
        self.task = task
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit LightGBM model."""
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (for classification)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        probs = self.model.predict(X)
        # Return probabilities in sklearn format
        return np.column_stack([1 - probs, probs]) 