"""StatsForecast adapter for statistical, intermittent, and naive baselines."""

from __future__ import annotations

import logging
from typing import List, Any, Optional
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from .base_adapter import BaseNixtlaAdapter, to_nixtla_format

logger = logging.getLogger(__name__)


class StatsForecastAdapter(BaseNixtlaAdapter):
    """Wrapper around Nixtla StatsForecast.

    Generic wrapper accepting any list of models (ARIMA, ETS, Croston, TSB, Naive, etc.).
    Enforces runtime data leakage assertions.
    """

    def __init__(
        self,
        models: List[Any],
        freq: str = 'D',
        n_jobs: int = 1,
        fallback_to_seasonal_naive: bool = True,
        **statsforecast_kwargs,
    ):
        super().__init__()
        self.models = models
        self.freq = freq
        self.n_jobs = n_jobs

        fallback_model = SeasonalNaive(season_length=7) if fallback_to_seasonal_naive else None

        self._sf = StatsForecast(
            models=self.models,
            freq=self.freq,
            n_jobs=self.n_jobs,
            fallback_model=fallback_model,
            **statsforecast_kwargs,
        )

    def fit(self, df: pd.DataFrame, **kwargs) -> "StatsForecastAdapter":
        """Fit StatsForecast models on training data with leakage check."""
        self._assert_no_test_leakage(df)

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        self._sf.fit(df)
        self.is_fitted = True
        return self

    def predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Forecast horizon h."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        return self._sf.predict(h=h, X_df=X_df, level=level, **kwargs)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Run rolling-origin cross-validation with leakage check."""
        self._assert_no_test_leakage(df)

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        return self._sf.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            **kwargs,
        )
