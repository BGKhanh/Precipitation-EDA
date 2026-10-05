"""Unified Benchmark Engine for Multi-Family Model Comparison.

Compares:
1. Naive Baselines (Persistence, Seasonal Naive)
2. Statistical & Intermittent Baselines (AutoARIMA, SARIMA from EDA, Croston, TSB)
3. ML Direct Multi-Horizon (Linear, RF, XGB, LGBM - Standard loss)
4. ML Hurdle / Two-Stage (Logistic + Linear, RF, XGB, LGBM)
5. ML Tweedie Single-Stage (Compound Poisson-Gamma)
6. Neural Direct (NHITS, PatchTST)
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .metrics import cumulative_rainfall_mae, cumulative_rainfall_bias


class UnifiedBenchmark:
    """Compare ALL model groups on the canonical test split.

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
        from src.models.nixtla import to_nixtla_format

        self.raw_train_df = train_df.copy()
        self.raw_test_df = test_df.copy()
        train_nixtla = to_nixtla_format(train_df) if not {'unique_id', 'ds', 'y'}.issubset(train_df.columns) else train_df.copy()
        test_nixtla = to_nixtla_format(test_df) if not {'unique_id', 'ds', 'y'}.issubset(test_df.columns) else test_df.copy()
        self.train_df = train_nixtla[['unique_id', 'ds', 'y']].copy()
        self.test_df = test_nixtla[['unique_id', 'ds', 'y']].copy()
        self.rain_threshold = rain_threshold
        self.horizon = horizon

    def run_group_statistical(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
        legacy_groups: bool = False,
    ) -> pd.DataFrame:
        """Benchmark Group 1: Statistical, Intermittent & Naive baselines."""
        from src.training.trainer import build_stats_adapter_from_eda
        from src.models.nixtla.defaults import get_statistical_models

        print("\n[Group 1] Statistical, Intermittent & Naive Baselines")
        chosen_models = models if models is not None else get_statistical_models(eda_report)
        sf_adapter = build_stats_adapter_from_eda(eda_report=eda_report, models=chosen_models)
        sf_adapter.fit(self.train_df)
        preds_df = sf_adapter.predict(h=self.horizon)

        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ['unique_id', 'ds']:
                continue
            if legacy_groups:
                if col in ['Naive', 'SeasonalNaive']:
                    grp = "1. Naive Baselines"
                elif col in ['CrostonOptimized', 'CrostonSBA', 'TSB', 'ADIDA', 'IMAPA']:
                    grp = "3a. Intermittent Baselines"
                else:
                    grp = "3b. Classical TS Baselines"
            else:
                grp = "1. Statistical"

            preds = preds_df[col].values[:len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group=grp))
            print(f"   * Evaluated: {col} ({grp})")

        return pd.DataFrame(results)

    def run_group_ml_direct(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 2: ML Direct Multi-Horizon (Linear, RF, XGB, LGBM - Standard loss)."""
        from src.training.trainer import build_ml_direct_adapter_from_eda
        from src.models.nixtla.defaults import get_ml_direct_models

        print("\n[Group 2] ML Direct Multi-Horizon (Standard Loss)")
        chosen_models = models if models is not None else get_ml_direct_models(eda_report)
        ml_adapter = build_ml_direct_adapter_from_eda(eda_report=eda_report, models=chosen_models, max_horizon=self.horizon)
        ml_adapter.fit(self.train_df)
        preds_df = ml_adapter.predict(h=self.horizon)

        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ['unique_id', 'ds']:
                continue
            preds = preds_df[col].values[:len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group="2. ML Direct"))
            print(f"   * Evaluated: {col} (Group 2. ML Direct)")

        return pd.DataFrame(results)

    def run_group_ml_hurdle(
        self,
        hurdle_models: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 3: ML Hurdle / Two-Stage (Logistic+Linear, RF, XGB, LGBM)."""
        from src.models.nixtla.defaults import get_ml_hurdle_models

        print("\n[Group 3] ML Hurdle / Two-Stage Models")
        models = hurdle_models or get_ml_hurdle_models(classification_threshold=self.rain_threshold)
        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for name, model in models.items():
            try:
                preds = self._predict_legacy(model, feature_builder, self.horizon)
                if preds is not None and len(preds) > 0:
                    preds = preds[:len(y_test)]
                    metrics = self._compute_metrics(
                        y_test, preds, horizons[:len(preds)], name, group="3. ML Hurdle"
                    )
                    results.extend(metrics)
                    print(f"   * Evaluated: {name} (Group 3. ML Hurdle)")
            except Exception as e:
                print(f"   [WARN] Hurdle {name} failed: {e}")

        return pd.DataFrame(results)

    def run_group_ml_tweedie(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Benchmark Group 4: ML Tweedie Single-Stage (LGBM-Tweedie, XGBoost-Tweedie)."""
        from src.training.trainer import build_ml_tweedie_adapter_from_eda
        from src.models.nixtla.defaults import get_ml_tweedie_models

        print("\n[Group 4] ML Tweedie Single-Stage (Compound Poisson-Gamma)")
        chosen_models = models if models is not None else get_ml_tweedie_models(eda_report)
        ml_adapter = build_ml_tweedie_adapter_from_eda(eda_report=eda_report, models=chosen_models, max_horizon=self.horizon)
        ml_adapter.fit(self.train_df)
        preds_df = ml_adapter.predict(h=self.horizon)

        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in preds_df.columns:
            if col in ['unique_id', 'ds']:
                continue
            preds = preds_df[col].values[:len(y_test)]
            results.extend(self._compute_metrics(y_test, preds, horizons, col, group="4. ML Tweedie"))
            print(f"   * Evaluated: {col} (Group 4. ML Tweedie)")

        return pd.DataFrame(results)

    def run_group_neural_direct(
        self,
        eda_report: Optional[Any] = None,
        models: Optional[List[Any]] = None,
        max_steps: int = 100,
    ) -> pd.DataFrame:
        """Benchmark Group 5: Neural Direct (LSTM baseline, TCN, NHITS, PatchTST, TFT)."""
        from src.training.trainer import build_neural_direct_adapter_from_eda
        from src.models.nixtla.defaults import get_neural_direct_models

        print("\n[Group 5] Neural Direct Models (MAE Loss, No Tweedie)")
        chosen_models = models if models is not None else get_neural_direct_models(
            horizon=self.horizon,
            max_steps=max_steps,
            eda_report=eda_report,
        )
        neural_adapter = build_neural_direct_adapter_from_eda(
            eda_report=eda_report,
            models=chosen_models,
            horizon=self.horizon,
            max_steps=max_steps,
        )

        results = []
        if neural_adapter.is_available:
            try:
                neural_adapter.fit(self.train_df)
                preds_df = neural_adapter.predict(h=self.horizon)
                y_test = self.test_df['y'].values[:self.horizon]
                horizons = list(range(1, len(y_test) + 1))

                for col in preds_df.columns:
                    if col in ['unique_id', 'ds']:
                        continue
                    preds = preds_df[col].values[:len(y_test)]
                    metrics = self._compute_metrics(
                        y_test, preds, horizons, col, group="5. Neural Direct"
                    )
                    results.extend(metrics)
                    print(f"   * Evaluated: {col} (Group 5. Neural Direct)")
            except Exception as e:
                print(f"   [WARN] Neural direct run error: {e}")
        else:
            print("   [INFO] NeuralForecast unavailable in this environment; skipping.")

        return pd.DataFrame(results)

    def compare_arima_eda_vs_auto(
        self,
        eda_report: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Head-to-head comparison: ARIMA (Non-seasonal), SARIMA (Seasonal), and AutoARIMA."""
        from statsforecast import StatsForecast
        from src.models.nixtla.defaults import get_eda_arima_model, get_eda_sarima_model
        from statsforecast.models import AutoARIMA

        print("\n[Comparison] ARIMA (Non-seasonal) vs SARIMA (Seasonal from EDA) vs AutoARIMA")
        season_length = 7
        if eda_report is not None:
            periods = getattr(eda_report, 'representative_periods', [])
            valid_periods = [p for p in periods if isinstance(p, (int, float)) and 1 < p <= 52]
            if valid_periods:
                season_length = int(valid_periods[0])

        eda_arima = get_eda_arima_model(eda_report)
        eda_sarima = get_eda_sarima_model(eda_report)
        auto_arima = AutoARIMA(season_length=season_length, alias='AutoARIMA')

        sf = StatsForecast(models=[eda_arima, eda_sarima, auto_arima], freq='D')
        sf.fit(self.train_df[['unique_id', 'ds', 'y']])
        preds_df = sf.predict(h=self.horizon)

        y_test = self.test_df['y'].values[:self.horizon]
        horizons = list(range(1, len(y_test) + 1))
        results = []

        for col in ['ARIMA_EDA', 'SARIMA_EDA', 'AutoARIMA']:
            if col in preds_df.columns:
                preds = preds_df[col].values[:len(y_test)]
                results.extend(self._compute_metrics(y_test, preds, horizons, col, group="ARIMA/SARIMA Comparison"))
                print(f"   * Evaluated: {col}")

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            summary = df_res.groupby('model_name').agg({
                'MAE': 'mean',
                'RMSE': 'mean',
                'rain_day_MAE': 'mean',
                'rain_day_RMSE': 'mean',
                'CR_MAE': 'last',
                'CR_Bias': 'last',
            }).round(4)
            print("\nARIMA vs SARIMA vs AutoARIMA Summary:")
            print(summary.to_string())
        return df_res

    @staticmethod
    def get_best_model_per_group(
        results_df: pd.DataFrame,
        metric: str = 'rain_day_MAE',
    ) -> pd.DataFrame:
        """Select best model (lowest metric) in each model_group."""
        if results_df.empty:
            return pd.DataFrame()
        agg_dict = {
            'MAE': 'mean',
            'RMSE': 'mean',
            'rain_day_MAE': 'mean',
            'rain_day_RMSE': 'mean',
        }
        if 'CR_MAE' in results_df.columns:
            agg_dict['CR_MAE'] = 'last'
        if 'CR_Bias' in results_df.columns:
            agg_dict['CR_Bias'] = 'last'

        summary = results_df.groupby(['model_group', 'model_name']).agg(agg_dict).reset_index()

        summary = summary.sort_values(['model_group', metric], ascending=[True, True])
        best_df = summary.groupby('model_group').first().reset_index()
        return best_df

    def run(
        self,
        eda_report: Optional[Any] = None,
        legacy_models: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
        include_statistical: bool = True,
        include_ml_direct: bool = True,
        include_ml_hurdle: bool = True,
        include_ml_tweedie: bool = True,
        include_neural_direct: bool = True,
        neural_max_steps: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute independent 5-group benchmark and final champion comparison."""
        legacy_mode = any(k in kwargs for k in ['include_statsforecast', 'include_mlforecast', 'include_neuralforecast'])
        if 'include_statsforecast' in kwargs:
            include_statistical = kwargs.pop('include_statsforecast')
            include_ml_direct = False
            include_ml_hurdle = False
        if 'include_mlforecast' in kwargs:
            include_ml_tweedie = kwargs.pop('include_mlforecast')
        if 'include_neuralforecast' in kwargs:
            include_neural_direct = kwargs.pop('include_neuralforecast')

        print("=" * 70)
        print("UNIFIED BENCHMARK: 5-GROUP INDEPENDENT EVALUATION")
        print("=" * 70)

        all_dfs = []

        # 1. Statistical
        if include_statistical:
            try:
                df1 = self.run_group_statistical(eda_report, legacy_groups=legacy_mode)
                all_dfs.append(df1)
            except Exception as e:
                print(f"   [WARN] Group 1 error: {e}")

        # 2. ML Direct
        if include_ml_direct:
            try:
                df2 = self.run_group_ml_direct(eda_report)
                all_dfs.append(df2)
            except Exception as e:
                print(f"   [WARN] Group 2 error: {e}")

        # 3. ML Hurdle
        if include_ml_hurdle:
            try:
                df3 = self.run_group_ml_hurdle(legacy_models, feature_builder)
                all_dfs.append(df3)
            except Exception as e:
                print(f"   [WARN] Group 3 error: {e}")

        # 4. ML Tweedie
        if include_ml_tweedie:
            try:
                df4 = self.run_group_ml_tweedie(eda_report)
                all_dfs.append(df4)
            except Exception as e:
                print(f"   [WARN] Group 4 error: {e}")

        # 5. Neural Direct
        if include_neural_direct:
            try:
                df5 = self.run_group_neural_direct(eda_report, max_steps=neural_max_steps)
                all_dfs.append(df5)
            except Exception as e:
                print(f"   [WARN] Group 5 error: {e}")

        results_df = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True) if all_dfs else pd.DataFrame()

        if not results_df.empty:
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

            print("\n" + "=" * 80)
            print("BEST MODEL PER GROUP (Selected by Rain-day MAE)")
            print("=" * 80)
            best_df = self.get_best_model_per_group(results_df, metric='rain_day_MAE')
            print(best_df.to_string(index=False))
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
        """Compute MAE/RMSE per horizon, rain-day metrics, and cumulative CR-MAE/CR-Bias."""
        rows = []
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        for h in sorted(set(horizons)):
            mask = [i for i, hh in enumerate(horizons) if hh == h]
            if not mask:
                continue

            yt = y_true_arr[mask]
            yp = y_pred_arr[mask]

            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))

            rain_mask = yt > self.rain_threshold
            if rain_mask.sum() > 0:
                rain_mae = mean_absolute_error(yt[rain_mask], yp[rain_mask])
                rain_rmse = np.sqrt(mean_squared_error(yt[rain_mask], yp[rain_mask]))
            else:
                rain_mae = np.nan
                rain_rmse = np.nan

            yt_cum = y_true_arr[:h]
            yp_cum = y_pred_arr[:h]
            cr_mae = cumulative_rainfall_mae(yt_cum, yp_cum, horizon=h)
            cr_bias = cumulative_rainfall_bias(yt_cum, yp_cum, horizon=h)

            rows.append({
                'model_group': group,
                'model_name': model_name,
                'horizon': h,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'rain_day_MAE': round(rain_mae, 4) if not np.isnan(rain_mae) else np.nan,
                'rain_day_RMSE': round(rain_rmse, 4) if not np.isnan(rain_rmse) else np.nan,
                'CR_MAE': round(cr_mae, 4) if not np.isnan(cr_mae) else np.nan,
                'CR_Bias': round(cr_bias, 4) if not np.isnan(cr_bias) else np.nan,
                'n_samples': len(mask),
                'y_true': round(float(yt[0]), 4) if len(yt) > 0 else np.nan,
                'y_pred': round(float(yp[0]), 4) if len(yp) > 0 else np.nan,
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
            from src.models.legacy.recursive import RecursiveForecaster
            forecaster = RecursiveForecaster(model, feature_builder)
            raw_train = getattr(self, 'raw_train_df', self.train_df)
            history_df = raw_train.iloc[-60:] if len(raw_train) >= 60 else raw_train
            forecast_df = forecaster.forecast(history_df, steps=steps)
            if 'prediction' in forecast_df.columns:
                return forecast_df['prediction'].values

        return None


UnifiedBenchmarkRunner = UnifiedBenchmark
