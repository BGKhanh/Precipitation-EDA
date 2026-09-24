"""MLForecast adapter for direct multi-horizon machine learning models."""

from __future__ import annotations

import logging
from typing import List, Any, Optional, Dict
import pandas as pd
from mlforecast import MLForecast

from .base_adapter import BaseNixtlaAdapter, to_nixtla_format

logger = logging.getLogger(__name__)


class MLForecastAdapter(BaseNixtlaAdapter):
    """Wrapper around Nixtla MLForecast.

    Key Architectural Guarantee:
        Uses Direct Multi-Horizon forecasting (`max_horizon=H`) rather than
        recursive forecasting. Trains H independent models predicting each horizon
        step directly from time t, eliminating recursive error compounding.
    """

    def __init__(
        self,
        models: List[Any],
        max_horizon: int = 7,
        freq: str = 'D',
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict[int, List[Any]]] = None,
        date_features: Optional[List[str]] = None,
        **mlforecast_kwargs,
    ):
        super().__init__()
        self.models = models
        self.max_horizon = max_horizon
        self.freq = freq
        self.lags = lags or [1, 2, 3, 7]
        self.lag_transforms = lag_transforms
        self.date_features = date_features or ['month', 'dayofweek']

        self._mlf = MLForecast(
            models=self.models,
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            **mlforecast_kwargs,
        )

    def fit(self, df: pd.DataFrame, **kwargs) -> "MLForecastAdapter":
        """Fit MLForecast models using direct multi-horizon strategy."""
        self._assert_no_test_leakage(df)

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        # Enforce direct multi-horizon training
        self._mlf.fit(df, max_horizon=self.max_horizon, **kwargs)
        self.is_fitted = True
        return self

    def predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Forecast horizon h directly."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        return self._mlf.predict(h=h, X_df=X_df, **kwargs)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Run rolling-origin cross-validation with leakage check and direct horizon."""
        self._assert_no_test_leakage(df)

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        return self._mlf.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            max_horizon=self.max_horizon,
            **kwargs,
        )
