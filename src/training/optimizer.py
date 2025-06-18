# =============================================================================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ModelOptimizer:
    """
    Hyperparameter optimization for rainfall prediction models using Optuna
    ✅ Supports both two-stage and single-stage optimization
    """
    
    def __init__(
        self, 
        n_splits: int = 3,
        n_trials: int = 100,
        random_state: int = 42,
        classification_threshold: float = 0.1
    ):
        """
        Initialize optimizer.
        
        Args:
            n_splits: CV splits for evaluation
            n_trials: Number of optimization trials
            random_state: Random state
            classification_threshold: Threshold for binary classification
        """
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.classification_threshold = classification_threshold
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def optimize_two_stage_model(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        param_space_classification: Dict[str, Any],
        param_space_regression: Dict[str, Any],
        optimization_metric: str = 'combined',  # 'classification', 'regression', 'combined'
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize two-stage model hyperparameters.
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target variable  
            param_space_classification: Classification parameter space
            param_space_regression: Regression parameter space
            optimization_metric: Metric to optimize
            verbose: Print progress
            
        Returns:
            Tuple of (best_classification_params, best_regression_params)
        """
        if verbose:
            print(f"🎯 Optimizing Two-Stage {model_class.__name__}")
            print(f"   Trials: {self.n_trials}, CV: {self.n_splits}")
            print(f"   Metric: {optimization_metric}")
        
        # Prepare data
        y_binary = (y > self.classification_threshold).astype(int)
        
        # Step 1: Optimize Classification Stage
        if verbose:
            print("\n🔄 Step 1: Optimizing Classification Stage...")
        
        def objective_classification(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space_classification.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Cross-validation
            scores = []
            for train_idx, test_idx in self.tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train_binary, y_test_binary = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
                
                try:
                    # Create and train classification model
                    model = model_class(use_two_stage=True, classification_params=params)
                    model.fit_classification(X_train, y_train_binary)
                    
                    # Predict and evaluate
                    pred_binary = model.predict_classification(X_test)
                    if len(np.unique(y_test_binary)) > 1:
                        score = roc_auc_score(y_test_binary, pred_binary)
                        scores.append(score)
                except Exception:
                    return 0.0
            
            return np.mean(scores) if scores else 0.0
        
        # Optimize classification
        study_classification = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study_classification.optimize(objective_classification, n_trials=self.n_trials//2, show_progress_bar=verbose)
        best_classification_params = study_classification.best_params
        
        if verbose:
            print(f"   ✅ Best Classification Score: {study_classification.best_value:.4f}")
        
        # Step 2: Optimize Regression Stage
        if verbose:
            print("\n🔄 Step 2: Optimizing Regression Stage...")
        
        def objective_regression(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space_regression.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Cross-validation with fixed classification params
            scores = []
            for train_idx, test_idx in self.tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                y_train_binary = y_binary.iloc[train_idx]
                
                try:
                    # Create model with best classification params and trial regression params
                    model = model_class(
                        use_two_stage=True, 
                        classification_params=best_classification_params,
                        regression_params=params
                    )
                    model.fit(X_train, y_train, y_train_binary)
                    
                    # Predict regression only for rain days
                    pred_continuous = model.predict_regression(X_test)
                    rain_mask = y_test > self.classification_threshold
                    
                    if rain_mask.sum() > 0:
                        score = mean_absolute_error(y_test[rain_mask], pred_continuous[rain_mask])
                        scores.append(score)
                except Exception:
                    return float('inf')
            
            return np.mean(scores) if scores else float('inf')
        
        # Optimize regression
        study_regression = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study_regression.optimize(objective_regression, n_trials=self.n_trials//2, show_progress_bar=verbose)
        best_regression_params = study_regression.best_params
        
        if verbose:
            print(f"   ✅ Best Regression Score: {study_regression.best_value:.4f}")
            print(f"\n🎯 Optimization Complete!")
        
        return best_classification_params, best_regression_params
    
    def optimize_single_stage_model(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        optimization_metric: str = 'mae',  # 'mae', 'rmse', 'r2'
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize single-stage model hyperparameters.
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target variable
            param_space: Parameter space definition
            optimization_metric: Metric to optimize ('mae', 'rmse', 'r2')
            verbose: Print progress
            
        Returns:
            Best hyperparameters
        """
        if verbose:
            print(f"🎯 Optimizing Single-Stage {model_class.__name__}")
            print(f"   Trials: {self.n_trials}, CV: {self.n_splits}")
            print(f"   Metric: {optimization_metric}")
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Cross-validation
            scores = []
            for train_idx, test_idx in self.tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                try:
                    # Create and train model
                    model = model_class(use_two_stage=False, **params)
                    model.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    pred = model.predict(X_test)
                    
                    if optimization_metric == 'mae':
                        score = mean_absolute_error(y_test, pred)
                    elif optimization_metric == 'rmse':
                        score = np.sqrt(mean_squared_error(y_test, pred))
                    elif optimization_metric == 'r2':
                        score = r2_score(y_test, pred)
                    
                    scores.append(score)
                except Exception:
                    if optimization_metric == 'r2':
                        return -float('inf')
                    else:
                        return float('inf')
            
            return np.mean(scores) if scores else (float('inf') if optimization_metric != 'r2' else -float('inf'))
        
        # Optimize
        direction = 'maximize' if optimization_metric == 'r2' else 'minimize'
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)
        
        if verbose:
            print(f"   ✅ Best Score: {study.best_value:.4f}")
            print(f"   ✅ Best Params: {study.best_params}")
        
        return study.best_params 