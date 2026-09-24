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
                            print(f"   {metric_name}: {value:.4f} +/- {metrics[std_key]:.4f}")


# ======================================================================
# ======================================================================
# Unified Benchmark — Step 5 (Nixtlaverse 5-Group Integration)
# ======================================================================

class UnifiedBenchmark:
    """Compare ALL 5 model groups on the canonical test split.

    Groups evaluated:
    1. Naive Baselines (Persistence, Seasonal Naive) — obtained via StatsForecastAdapter.
    2. Legacy Two-Stage Models (via RecursiveForecaster or precomputed predictions).
    3. Statistical & Intermittent Baselines (AutoARIMA, AutoETS, Croston, TSB, IMAPA).
    4. Direct ML Forecast (AutoLightGBM, AutoXGBoost with Tweedie & max_horizon).
    5. Neural Forecast (AutoNHITS with Tweedie, if available in environment).

    Produces results conforming to ``evaluation-standards.md``:
    - Mandatory naive baselines (deduplicated).
    - Multi-step breakdown per-horizon (h=1, 3, 7).
    - Rain-day regression error reported separately from overall error.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rain_threshold: float = 0.1,
        horizon: int = 7,
    ):
        from ..models.nixtla import to_nixtla_format

        self.train_df = to_nixtla_format(train_df) if not {'unique_id', 'ds', 'y'}.issubset(train_df.columns) else train_df.copy()
        self.test_df = to_nixtla_format(test_df) if not {'unique_id', 'ds', 'y'}.issubset(test_df.columns) else test_df.copy()
        self.rain_threshold = rain_threshold
        self.horizon = horizon

    def run(
        self,
        eda_report: Optional[Any] = None,
        legacy_models: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
        include_statsforecast: bool = True,
        include_mlforecast: bool = True,
        include_neuralforecast: bool = True,
    ) -> pd.DataFrame:
        """Execute unified 5-group benchmark.

        Args:
            eda_report: Optional EDAReport instance.
            legacy_models: Dict of legacy models e.g. {'RandomForest_TwoStage': rf_model}.
            feature_builder: FeatureBuilder instance for recursive two-stage forecasting.
            include_statsforecast: Run statistical + intermittent + naive baselines.
            include_mlforecast: Run direct ML forecasting with Tweedie loss.
            include_neuralforecast: Run neural models if available.

        Returns:
            DataFrame with per-model, per-horizon metrics.
        """
        print("=" * 70)
        print("UNIFIED BENCHMARK (5-GROUP NIXTLAVERSE EVALUATION)")
        print("=" * 70)

        all_results = []
        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))

        # -------------------------------------------------------------
        # [1 & 3] StatsForecastAdapter (Naive + Intermittent + TS)
        # -------------------------------------------------------------
        if include_statsforecast:
            print("\n[Group 1 & 3] StatsForecastAdapter (Naive, Intermittent & Classical TS)")
            try:
                from .trainer import build_stats_adapter_from_eda
                sf_adapter = build_stats_adapter_from_eda(eda_report=eda_report)
                sf_adapter.fit(self.train_df)
                sf_preds = sf_adapter.predict(h=self.horizon)

                for col in sf_preds.columns:
                    if col in ['unique_id', 'ds']:
                        continue
                    # Group classification
                    if col in ['Naive', 'SeasonalNaive']:
                        group_name = "1. Naive Baselines"
                    elif col in ['CrostonOptimized', 'TSB', 'IMAPA']:
                        group_name = "3a. Intermittent Baselines"
                    else:
                        group_name = "3b. Classical TS Baselines"

                    preds = sf_preds[col].values[:len(y_test)]
                    metrics = self._compute_metrics(y_test, preds, horizons, col, group=group_name)
                    all_results.extend(metrics)
                    print(f"   * Evaluated: {col} ({group_name})")
            except Exception as e:
                print(f"   [WARN] StatsForecastAdapter run error: {e}")

        # -------------------------------------------------------------
        # [2] Legacy Two-Stage Models
        # -------------------------------------------------------------
        if legacy_models:
            print(f"\n[Group 2] Legacy Two-Stage Models ({len(legacy_models)} models)")
            for name, model in legacy_models.items():
                try:
                    preds = self._predict_legacy(model, feature_builder, self.horizon)
                    if preds is not None and len(preds) > 0:
                        preds = preds[:len(y_test)]
                        metrics = self._compute_metrics(
                            y_test, preds, horizons[:len(preds)], name, group="2. Legacy Two-Stage"
                        )
                        all_results.extend(metrics)
                        print(f"   * Evaluated: {name} (Group 2. Legacy Two-Stage)")
                except Exception as e:
                    print(f"   [WARN] Legacy {name} failed: {e}")

        # -------------------------------------------------------------
        # [4] Direct MLForecast (Tweedie Loss & max_horizon=H)
        # -------------------------------------------------------------
        if include_mlforecast:
            print("\n[Group 4] Direct MLForecast (AutoLightGBM/AutoXGBoost + Tweedie)")
            try:
                from .trainer import build_ml_adapter_from_eda
                ml_adapter = build_ml_adapter_from_eda(eda_report=eda_report, max_horizon=self.horizon)
                ml_adapter.fit(self.train_df)
                ml_preds = ml_adapter.predict(h=self.horizon)

                for col in ml_preds.columns:
                    if col in ['unique_id', 'ds']:
                        continue
                    preds = ml_preds[col].values[:len(y_test)]
                    metrics = self._compute_metrics(
                        y_test, preds, horizons, col, group="4. Direct MLForecast"
                    )
                    all_results.extend(metrics)
                    print(f"   * Evaluated: {col} (Group 4. Direct MLForecast)")
            except Exception as e:
                print(f"   [WARN] MLForecastAdapter run error: {e}")

        # -------------------------------------------------------------
        # [5] NeuralForecast (Tweedie DistributionLoss)
        # -------------------------------------------------------------
        if include_neuralforecast:
            print("\n[Group 5] NeuralForecast (AutoNHITS + Tweedie)")
            try:
                from .trainer import build_neural_adapter_from_eda
                neural_adapter = build_neural_adapter_from_eda(eda_report=eda_report, horizon=self.horizon)
                if neural_adapter.is_available:
                    neural_adapter.fit(self.train_df)
                    neural_preds = neural_adapter.predict(h=self.horizon)
                    for col in neural_preds.columns:
                        if col in ['unique_id', 'ds']:
                            continue
                        preds = neural_preds[col].values[:len(y_test)]
                        metrics = self._compute_metrics(
                            y_test, preds, horizons, col, group="5. NeuralForecast"
                        )
                        all_results.extend(metrics)
                        print(f"   * Evaluated: {col} (Group 5. NeuralForecast)")
                else:
                    print("   [INFO] NeuralForecast unavailable in this environment; skipping.")
            except Exception as e:
                print(f"   [WARN] NeuralForecastAdapter run error: {e}")

        # -------------------------------------------------------------
        # Build Results Table
        # -------------------------------------------------------------
        results_df = pd.DataFrame(all_results)
        if len(results_df) > 0:
            print("\n" + "=" * 80)
            print("BENCHMARK SUMMARY (Aggregate across horizons)")
            print("=" * 80)
            summary = results_df.groupby(['model_group', 'model_name']).agg({
                'MAE': 'mean',
                'RMSE': 'mean',
                'rain_day_MAE': 'mean',
                'rain_day_RMSE': 'mean',
                'n_samples': 'sum',
            }).round(4)
            print(summary.to_string())
            print()

        return results_df

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizons: List[int],
        model_name: str,
        group: str = "Model",
    ) -> List[Dict[str, Any]]:
        """Compute MAE/RMSE per horizon and rain-day-only metrics."""
        rows = []
        for h in sorted(set(horizons)):
            mask = [i for i, hh in enumerate(horizons) if hh == h]
            if not mask:
                continue

            yt = np.array(y_true)[mask]
            yp = np.array(y_pred)[mask]

            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))

            rain_mask = yt > self.rain_threshold
            if rain_mask.sum() > 0:
                rain_mae = mean_absolute_error(yt[rain_mask], yp[rain_mask])
                rain_rmse = np.sqrt(mean_squared_error(yt[rain_mask], yp[rain_mask]))
            else:
                rain_mae = np.nan
                rain_rmse = np.nan

            rows.append({
                'model_group': group,
                'model_name': model_name,
                'horizon': h,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'rain_day_MAE': round(rain_mae, 4) if not np.isnan(rain_mae) else np.nan,
                'rain_day_RMSE': round(rain_rmse, 4) if not np.isnan(rain_rmse) else np.nan,
                'n_samples': len(mask),
            })
        return rows

    def _predict_legacy(
        self,
        model: Any,
        feature_builder: Optional[Any],
        steps: int,
    ) -> Optional[np.ndarray]:
        """Generate forecasts from legacy two-stage model."""
        if hasattr(model, 'forecast'):
            return np.array(model.forecast(steps=steps))

        if feature_builder is not None:
            from ..models.legacy.recursive import RecursiveForecaster
            forecaster = RecursiveForecaster(model, feature_builder)
            forecast_df = forecaster.forecast(self.train_df, steps=steps)
            if 'prediction' in forecast_df.columns:
                return forecast_df['prediction'].values

        return None


UnifiedBenchmarkRunner = UnifiedBenchmark
 