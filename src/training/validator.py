# =============================================================================
# CROSS-VALIDATION UTILITIES FOR RAINFALL PREDICTION
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class RainfallCrossValidator:
    """
    Cross-validation utilities for rainfall prediction models
    ✅ Supports both two-stage and single-stage approaches
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV splits
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def validate_model(
        self, 
        model, 
        X: pd.DataFrame, 
        y: pd.Series,
        classification_threshold: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation for any model.
        
        Args:
            model: Model instance with fit() and predict() methods
            X: Feature matrix
            y: Target variable
            classification_threshold: Threshold for binary classification
            verbose: Print progress
            
        Returns:
            Dict with validation results
        """
        if verbose:
            print(f"🔄 Cross-validation with {self.n_splits} splits")
            print(f"   Total samples: {len(X)}")
        
        # Storage for results
        results = {
            'classification': {'ROC_AUC': []},
            'regression': {'MAE': [], 'RMSE': [], 'R2': []},
            'combined': {'Classification_AUC': [], 'Regression_MAE': [], 'Regression_RMSE': [], 'Regression_R2': []}
        }
        
        # Prepare binary target for classification
        y_binary = (y > classification_threshold).astype(int)
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            if verbose:
                print(f"   Fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_train_binary, y_test_binary = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
            
            try:
                # Fit model
                model.fit(X_train, y_train, y_train_binary)
                
                # Check if model has two-stage capability
                if hasattr(model, 'predict_combined') and model.use_two_stage:
                    # Two-stage prediction
                    pred_binary, pred_continuous = model.predict_combined(X_test)
                    
                    # Classification metrics
                    if len(np.unique(y_test_binary)) > 1:  # Check for both classes
                        auc = roc_auc_score(y_test_binary, pred_binary)
                        results['combined']['Classification_AUC'].append(auc)
                    
                    # Regression metrics (only for actual rain days)
                    rain_mask = y_test > classification_threshold
                    if rain_mask.sum() > 0:
                        mae = mean_absolute_error(y_test[rain_mask], pred_continuous[rain_mask])
                        rmse = np.sqrt(mean_squared_error(y_test[rain_mask], pred_continuous[rain_mask]))
                        r2 = r2_score(y_test[rain_mask], pred_continuous[rain_mask])
                        
                        results['combined']['Regression_MAE'].append(mae)
                        results['combined']['Regression_RMSE'].append(rmse)
                        results['combined']['Regression_R2'].append(r2)
                
                else:
                    # Single-stage prediction
                    pred = model.predict(X_test)
                    
                    # Convert to binary for classification evaluation
                    pred_binary = (pred > classification_threshold).astype(int)
                    
                    # Classification metrics
                    if len(np.unique(y_test_binary)) > 1:
                        auc = roc_auc_score(y_test_binary, pred_binary)
                        results['classification']['ROC_AUC'].append(auc)
                    
                    # Regression metrics
                    mae = mean_absolute_error(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    r2 = r2_score(y_test, pred)
                    
                    results['regression']['MAE'].append(mae)
                    results['regression']['RMSE'].append(rmse)
                    results['regression']['R2'].append(r2)
                
            except Exception as e:
                if verbose:
                    print(f"     ❌ Fold {fold + 1} failed: {str(e)}")
                continue
        
        # Calculate averages
        final_results = {}
        for stage, metrics in results.items():
            final_results[stage] = {}
            for metric, values in metrics.items():
                if values:  # Only if we have values
                    final_results[stage][f'{metric}_mean'] = np.mean(values)
                    final_results[stage][f'{metric}_std'] = np.std(values)
                    final_results[stage][f'{metric}_scores'] = values
        
        if verbose:
            self._print_results(final_results)
        
        return final_results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        print("\n📊 Cross-Validation Results:")
        print("="*50)
        
        for stage, metrics in results.items():
            if metrics:  # Only print if we have results
                print(f"\n{stage.upper()} STAGE:")
                for metric, value in metrics.items():
                    if metric.endswith('_mean'):
                        metric_name = metric.replace('_mean', '')
                        std_key = f'{metric_name}_std'
                        if std_key in metrics:
                            print(f"   {metric_name}: {value:.4f} ± {metrics[std_key]:.4f}") 