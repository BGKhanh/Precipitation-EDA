# =============================================================================
# TIME SERIES MODELS: ARIMA, SARIMA (SINGLE-STAGE ONLY)
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

from .base import BaseTimeSeriesModel, calculate_metrics
# StationarityTester lives in featurengineering (single source of truth)
from ...featurengineering.stationarity_test import StationarityTester

warnings.filterwarnings('ignore')


# StationarityTester class removed — now imported from
# src.featurengineering.stationarity_test (single source of truth).
# The import is at the top of this file.


class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA model for time series forecasting.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (3, 0, 3)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        super().__init__(order=order)
        
    def fit(self, y: pd.Series, verbose: bool = False, **kwargs) -> None:
        """
        Fit ARIMA model.
        
        Args:
            y: Target time series
            verbose: Whether to print fitting progress
            **kwargs: Additional parameters
        """
        try:
            print(f"🚀 Training ARIMA{self.order} model...")
            self.model = SARIMAX(y, order=self.order)
            self.fitted_model = self.model.fit(disp=verbose)
            self.is_fitted = True
            
            if verbose:
                print(f"✅ ARIMA{self.order} fitted successfully")
                
        except Exception as e:
            raise RuntimeError(f"ARIMA fitting failed: {e}")
    
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            Forecast values
        """
        self._validate_fitted()
        return self.fitted_model.forecast(steps=steps)

    def forecast_with_interval(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Forecast with confidence intervals.

        Args:
            steps: Number of steps to forecast.
            alpha: Significance level (0.05 = 95% CI).

        Returns:
            ``(point_forecast, conf_int_df)`` where *conf_int_df* has
            columns ``['lower', 'upper']``.
        """
        self._validate_fitted()
        result = self.fitted_model.get_forecast(steps=steps)
        point = result.predicted_mean.values
        ci = result.conf_int(alpha=alpha)
        ci.columns = ['lower', 'upper']
        return point, ci


class SARIMAModel(BaseTimeSeriesModel):
    """
    SARIMA model for seasonal time series forecasting.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (3, 0, 3),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)):
        """
        Initialize SARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
        """
        super().__init__(order=order, seasonal_order=seasonal_order)
        
    def fit(self, y: pd.Series, verbose: bool = False, **kwargs) -> None:
        """
        Fit SARIMA model.
        
        Args:
            y: Target time series
            verbose: Whether to print fitting progress
            **kwargs: Additional parameters
        """
        try:
            print(f"🚀 Training SARIMA{self.order}x{self.seasonal_order} model...")
            self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=verbose)
            self.is_fitted = True
            
            if verbose:
                print(f"✅ SARIMA{self.order}x{self.seasonal_order} fitted successfully")
                
        except Exception as e:
            raise RuntimeError(f"SARIMA fitting failed: {e}")
    
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            Forecast values
        """
        self._validate_fitted()
        return self.fitted_model.forecast(steps=steps)

    def forecast_with_interval(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Forecast with confidence intervals.

        Args:
            steps: Number of steps to forecast.
            alpha: Significance level (0.05 = 95% CI).

        Returns:
            ``(point_forecast, conf_int_df)`` where *conf_int_df* has
            columns ``['lower', 'upper']``.
        """
        self._validate_fitted()
        result = self.fitted_model.get_forecast(steps=steps)
        point = result.predicted_mean.values
        ci = result.conf_int(alpha=alpha)
        ci.columns = ['lower', 'upper']
        return point, ci


class ARIMAXModel(BaseTimeSeriesModel):
    """
    ARIMAX model with exogenous variables.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (3, 0, 3)):
        """
        Initialize ARIMAX model.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        super().__init__(order=order)
        self.exog_columns = None
        
    def fit(self, y: pd.Series, exog: pd.DataFrame, verbose: bool = False, **kwargs) -> None:
        """
        Fit ARIMAX model.
        
        Args:
            y: Target time series
            exog: Exogenous variables
            verbose: Whether to print fitting progress
            **kwargs: Additional parameters
        """
        try:
            print(f"🚀 Training ARIMAX{self.order} with {len(exog.columns)} exogenous variables...")
            self.exog_columns = exog.columns.tolist()
            self.model = SARIMAX(y, exog=exog, order=self.order)
            self.fitted_model = self.model.fit(disp=verbose)
            self.is_fitted = True
            
            if verbose:
                print(f"✅ ARIMAX{self.order} with {len(self.exog_columns)} exog vars fitted successfully")
                
        except Exception as e:
            raise RuntimeError(f"ARIMAX fitting failed: {e}")
    
    def forecast(self, steps: int, exog: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            exog: Future exogenous variables
            **kwargs: Additional parameters
            
        Returns:
            Forecast values
        """
        self._validate_fitted()
        
        if exog.columns.tolist() != self.exog_columns:
            raise ValueError("Exogenous variables columns don't match training data")
            
        return self.fitted_model.forecast(steps=steps, exog=exog)


class SARIMAXModel(BaseTimeSeriesModel):
    """
    SARIMAX model with exogenous variables and seasonality.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (3, 0, 3),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)):
        """
        Initialize SARIMAX model.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
        """
        super().__init__(order=order, seasonal_order=seasonal_order)
        self.exog_columns = None
        
    def fit(self, y: pd.Series, exog: pd.DataFrame, verbose: bool = False, **kwargs) -> None:
        """
        Fit SARIMAX model.
        
        Args:
            y: Target time series
            exog: Exogenous variables
            verbose: Whether to print fitting progress
            **kwargs: Additional parameters
        """
        try:
            print(f"🚀 Training SARIMAX{self.order}x{self.seasonal_order} with {len(exog.columns)} exogenous variables...")
            self.exog_columns = exog.columns.tolist()
            self.model = SARIMAX(y, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=verbose)
            self.is_fitted = True
            
            if verbose:
                print(f"✅ SARIMAX{self.order}x{self.seasonal_order} with {len(self.exog_columns)} exog vars fitted successfully")
                
        except Exception as e:
            raise RuntimeError(f"SARIMAX fitting failed: {e}")
    
    def forecast(self, steps: int, exog: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            exog: Future exogenous variables
            **kwargs: Additional parameters
            
        Returns:
            Forecast values
        """
        self._validate_fitted()
        
        if exog.columns.tolist() != self.exog_columns:
            raise ValueError("Exogenous variables columns don't match training data")
            
        return self.fitted_model.forecast(steps=steps, exog=exog)


def evaluate_time_series_model(model: BaseTimeSeriesModel,
                              y_test: pd.Series,
                              exog_test: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Evaluate time series model performance.
    
    Args:
        model: Fitted time series model
        y_test: Test target values
        exog_test: Test exogenous variables (if applicable)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Generate forecasts
    if exog_test is not None:
        forecasts = model.forecast(steps=len(y_test), exog=exog_test)
    else:
        forecasts = model.forecast(steps=len(y_test))
    
    # Calculate metrics
    return calculate_metrics(y_test, forecasts, task='regression')


def prepare_time_series_data(df: pd.DataFrame, 
                           target_col: str,
                           exog_cols: List[str],
                           make_stationary: bool = True,
                           verbose: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare data for time series modeling.
    
    Args:
        df: Input DataFrame
        target_col: Target variable column name
        exog_cols: Exogenous variable column names
        make_stationary: Whether to make variables stationary
        verbose: Whether to print preparation details
        
    Returns:
        Tuple of (target_series, exog_dataframe)
    """
    if verbose:
        print("🚀 Preparing Time Series Data...")
    
    # Extract target and exogenous variables
    target = df[target_col].copy()
    exog_data = df[exog_cols].copy()
    
    if make_stationary:
        # Test and make exogenous variables stationary
        exog_stationary = StationarityTester.make_stationary(exog_data, verbose=verbose)
        
        # Align target with stationary exogenous data
        target = target.loc[exog_stationary.index]
        
        if verbose:
            print(f"   📊 Data aligned: {len(target)} samples")
            
        return target, exog_stationary
    else:
        return target, exog_data 