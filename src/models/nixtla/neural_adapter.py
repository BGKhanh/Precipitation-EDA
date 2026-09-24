"""NeuralForecast adapter for deep learning time-series models."""

from __future__ import annotations

import logging
from typing import List, Any, Optional
import pandas as pd

from .base_adapter import BaseNixtlaAdapter, to_nixtla_format

logger = logging.getLogger(__name__)


class NeuralForecastAdapter(BaseNixtlaAdapter):
    """Wrapper around Nixtla NeuralForecast.

    Accepts models list and horizon. Drops static exogenous variables (stat_exog_list)
    since station coordinates have zero variance across single-region data.
    """

    def __init__(
        self,
        models: List[Any],
        horizon: int = 7,
        freq: str = 'D',
        **neuralforecast_kwargs,
    ):
        super().__init__()
        self.models = models
        self.horizon = horizon
        self.freq = freq
        self._nf = None

        self._check_and_init(neuralforecast_kwargs)

    def _check_and_init(self, kwargs: dict) -> None:
        """Defensive environment and initialization check."""
        try:
            from neuralforecast import NeuralForecast
            if self.models:
                self._nf = NeuralForecast(
                    models=self.models,
                    freq=self.freq,
                    **kwargs,
                )
        except Exception as e:
            logger.warning(
                "NeuralForecast could not be initialized in this environment: %s. "
                "Calls to fit/predict will be safely skipped or fall back.",
                e,
            )
            self._nf = None

    @property
    def is_available(self) -> bool:
        """Check whether NeuralForecast is functional in current environment."""
        return self._nf is not None

    def fit(self, df: pd.DataFrame, **kwargs) -> "NeuralForecastAdapter":
        """Fit NeuralForecast models with leakage check."""
        self._assert_no_test_leakage(df)

        if not self.is_available:
            logger.warning("NeuralForecast is not available. Skipping neural fit.")
            return self

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        self._nf.fit(df, **kwargs)
        self.is_fitted = True
        return self

    def predict(
        self,
        h: Optional[int] = None,
        futr_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Forecast horizon h."""
        if not self.is_available or not self.is_fitted:
            raise RuntimeError("NeuralForecast model is not fitted or unavailable.")

        h = h or self.horizon
        return self._nf.predict(futr_df=futr_df, **kwargs)

    def cross_validation(
        self,
        df: pd.DataFrame,
        n_windows: int = 1,
        step_size: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Run rolling-origin cross-validation."""
        self._assert_no_test_leakage(df)

        if not self.is_available:
            raise RuntimeError("NeuralForecast model is not available in this environment.")

        if not {'unique_id', 'ds', 'y'}.issubset(df.columns):
            df = to_nixtla_format(df)

        return self._nf.cross_validation(
            df=df,
            n_windows=n_windows,
            step_size=step_size,
            **kwargs,
        )
