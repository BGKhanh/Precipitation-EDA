# =============================================================================
# TRAINING MANAGER FOR RAINFALL PREDICTION MODELS
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import warnings

from .validator import RainfallCrossValidator
from .optimizer import ModelOptimizer

warnings.filterwarnings('ignore')


class RainfallTrainer:
    """
    Comprehensive training manager for rainfall prediction models
    ✅ Handles both two-stage and single-stage training workflows
    """
    
    def __init__(
        self,
        classification_threshold: float = 0.1,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_splits: int = 5
    ):
        """
        Initialize trainer.
        
        Args:
            classification_threshold: Threshold for binary classification (mm)
            test_size: Test set proportion
            random_state: Random state for reproducibility
            cv_splits: Number of CV splits
        """
        self.classification_threshold = classification_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.cv_splits = cv_splits
        
        # Initialize utilities
        self.validator = RainfallCrossValidator(n_splits=cv_splits, random_state=random_state)
        self.optimizer = ModelOptimizer(
            n_splits=max(3, cv_splits-2),  # Use fewer splits for optimization
            random_state=random_state,
            classification_threshold=classification_threshold
        )
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            feature_columns: Feature column names (if None, use all except target)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("🔧 Preparing data for training...")
        
        # Handle missing feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.dropna()
        y = y.loc[X.index]
        
        print(f"   Features: {len(feature_columns)}")
        print(f"   Samples: {len(X)}")
        print(f"   Rain events (>{self.classification_threshold}mm): {(y > self.classification_threshold).sum()} ({(y > self.classification_threshold).mean()*100:.1f}%)")
        
        # Time series split (preserve temporal order)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def train_with_optimization(
        self,
        model_class,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_spaces: Dict[str, Any],
        use_two_stage: bool = True,
        n_trials: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train model with hyperparameter optimization.
        
        Args:
            model_class: Model class to train
            X_train: Training features
            y_train: Training target
            param_spaces: Parameter spaces for optimization
            use_two_stage: Whether to use two-stage approach
            n_trials: Number of optimization trials
            verbose: Print progress
            
        Returns:
            Dict with best model and optimization results
        """
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Training {model_class.__name__} with optimization")
            print(f"   Two-stage: {use_two_stage}")
            print(f"   Trials: {n_trials}")
        
        # Set up optimizer
        self.optimizer.n_trials = n_trials
        
        if use_two_stage:
            # Two-stage optimization
            best_clf_params, best_reg_params = self.optimizer.optimize_two_stage_model(
                model_class=model_class,
                X=X_train,
                y=y_train,
                param_space_classification=param_spaces['classification'],
                param_space_regression=param_spaces['regression'],
                verbose=verbose
            )
            
            # Train final model with best parameters
            final_model = model_class(
                use_two_stage=True,
                classification_params=best_clf_params,
                regression_params=best_reg_params
            )
            
            y_train_binary = (y_train > self.classification_threshold).astype(int)
            final_model.fit(X_train, y_train, y_train_binary)
            
            optimization_results = {
                'best_classification_params': best_clf_params,
                'best_regression_params': best_reg_params,
                'optimization_time': time.time() - start_time
            }
        
        else:
            # Single-stage optimization
            best_params = self.optimizer.optimize_single_stage_model(
                model_class=model_class,
                X=X_train,
                y=y_train,
                param_space=param_spaces['single_stage'],
                verbose=verbose
            )
            
            # Train final model with best parameters
            final_model = model_class(use_two_stage=False, **best_params)
            final_model.fit(X_train, y_train)
            
            optimization_results = {
                'best_params': best_params,
                'optimization_time': time.time() - start_time
            }
        
        if verbose:
            print(f"   ✅ Optimization completed in {optimization_results['optimization_time']:.1f}s")
        
        return {
            'model': final_model,
            'optimization_results': optimization_results
        }
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            verbose: Print results
            
        Returns:
            Evaluation results
        """
        if verbose:
            print(f"\n📊 Evaluating {model.__class__.__name__} on test set...")
        
        y_test_binary = (y_test > self.classification_threshold).astype(int)
        
        # Make predictions
        if hasattr(model, 'predict_combined') and model.use_two_stage:
            # Two-stage prediction
            pred_binary, pred_continuous = model.predict_combined(X_test)
            
            # Evaluate classification
            from sklearn.metrics import roc_auc_score, classification_report
            clf_auc = roc_auc_score(y_test_binary, pred_binary) if len(np.unique(y_test_binary)) > 1 else 0.0
            
            # Evaluate regression (only on rain days)
            rain_mask = y_test > self.classification_threshold
            if rain_mask.sum() > 0:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                reg_mae = mean_absolute_error(y_test[rain_mask], pred_continuous[rain_mask])
                reg_rmse = np.sqrt(mean_squared_error(y_test[rain_mask], pred_continuous[rain_mask]))
                reg_r2 = r2_score(y_test[rain_mask], pred_continuous[rain_mask])
            else:
                reg_mae = reg_rmse = reg_r2 = np.nan
            
            results = {
                'model_type': 'two_stage',
                'classification': {
                    'ROC_AUC': clf_auc,
                    'predictions': pred_binary
                },
                'regression': {
                    'MAE': reg_mae,
                    'RMSE': reg_rmse,
                    'R2': reg_r2,
                    'predictions': pred_continuous,
                    'rain_days_evaluated': rain_mask.sum()
                }
            }
            
            if verbose:
                print(f"   Classification AUC: {clf_auc:.4f}")
                print(f"   Regression MAE: {reg_mae:.4f}")
                print(f"   Regression RMSE: {reg_rmse:.4f}")
                print(f"   Regression R²: {reg_r2:.4f}")
                print(f"   Rain days evaluated: {rain_mask.sum()}")
        
        else:
            # Single-stage prediction
            pred = model.predict(X_test)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            
            results = {
                'model_type': 'single_stage',
                'regression': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'predictions': pred
                }
            }
            
            if verbose:
                print(f"   MAE: {mae:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   R²: {r2:.4f}")
        
        return results
    
    def full_training_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        model_configs: Dict[str, Dict[str, Any]],
        feature_columns: Optional[List[str]] = None,
        optimize_hyperparameters: bool = True,
        n_trials: int = 100,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Complete training pipeline for multiple models.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            model_configs: Dict of {model_name: {'class': ModelClass, 'param_spaces': ..., 'use_two_stage': bool}}
            feature_columns: Feature columns
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            verbose: Print progress
            
        Returns:
            Dict of training results for each model
        """
        print("🚀 FULL TRAINING PIPELINE")
        print("="*60)
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(df, target_column, feature_columns)
        
        results = {}
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*60}")
            print(f"🎯 Training {model_name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                if optimize_hyperparameters:
                    # Train with optimization
                    training_result = self.train_with_optimization(
                        model_class=config['class'],
                        X_train=X_train,
                        y_train=y_train,
                        param_spaces=config['param_spaces'],
                        use_two_stage=config.get('use_two_stage', True),
                        n_trials=n_trials,
                        verbose=verbose
                    )
                    model = training_result['model']
                    optimization_results = training_result['optimization_results']
                else:
                    # Train with default parameters
                    model = config['class'](use_two_stage=config.get('use_two_stage', True))
                    y_train_binary = (y_train > self.classification_threshold).astype(int)
                    if hasattr(model, 'fit') and config.get('use_two_stage', True):
                        model.fit(X_train, y_train, y_train_binary)
                    else:
                        model.fit(X_train, y_train)
                    optimization_results = None
                
                # Evaluate model
                evaluation_results = self.evaluate_model(model, X_test, y_test, verbose)
                
                # Cross-validation
                if verbose:
                    print(f"\n🔄 Cross-validation...")
                cv_results = self.validator.validate_model(model, X_train, y_train, self.classification_threshold, verbose=False)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'evaluation': evaluation_results,
                    'cross_validation': cv_results,
                    'optimization': optimization_results,
                    'training_time': time.time() - start_time
                }
                
                if verbose:
                    print(f"   ✅ {model_name} completed in {results[model_name]['training_time']:.1f}s")
            
            except Exception as e:
                if verbose:
                    print(f"   ❌ {model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        print(f"\n{'='*60}")
        print("🎉 TRAINING PIPELINE COMPLETED")
        print(f"{'='*60}")
        
        return results 