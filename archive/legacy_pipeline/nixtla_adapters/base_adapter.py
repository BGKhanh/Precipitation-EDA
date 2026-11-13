"""Base adapter and data leakage safeguards for Nixtlaverse models."""

from __future__ import annotations

import abc
import logging
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np

from ...data.loader import DEFAULT_TEST_START_DATE
from ...data.dataquality import check_continuity
from ...config.resolve import resolve_date_col, resolve_target_col

logger = logging.getLogger(__name__)


def to_nixtla_format(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
    unique_id: str = "station_1",
) -> pd.DataFrame:
    """Convert project DataFrame to Nixtla long format (unique_id, ds, y)."""
    date_col = resolve_date_col(date_col)
    target_col = resolve_target_col(target_col)

    nixtla_df = df.copy()
    nixtla_df = nixtla_df.rename(columns={date_col: 'ds', target_col: 'y'})
    nixtla_df['unique_id'] = unique_id

    if not pd.api.types.is_datetime64_any_dtype(nixtla_df['ds']):
        nixtla_df['ds'] = pd.to_datetime(nixtla_df['ds'])

    nixtla_df = nixtla_df.sort_values('ds').reset_index(drop=True)

    is_ok, missing = check_continuity(
        nixtla_df.rename(columns={'ds': date_col}),
        date_col=date_col,
    )
    if not is_ok:
        logger.warning(
            "to_nixtla_format: %d missing days detected. Reindexing to daily frequency.",
            len(missing),
        )
        full_idx = pd.date_range(nixtla_df['ds'].min(), nixtla_df['ds'].max(), freq='D')
        nixtla_df = nixtla_df.set_index('ds').reindex(full_idx)
        nixtla_df.index.name = 'ds'
        nixtla_df = nixtla_df.reset_index()
        nixtla_df['unique_id'] = nixtla_df['unique_id'].fillna(unique_id)
        numeric_cols = nixtla_df.select_dtypes(include=[np.number]).columns
        nixtla_df[numeric_cols] = nixtla_df[numeric_cols].interpolate(
            method='linear', limit_direction='both'
        )

    return nixtla_df


class BaseNixtlaAdapter(abc.ABC):
    """Abstract base adapter for all Nixtla ecosystem wrappers."""

    def __init__(
        self,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.is_fitted = False

    def _assert_no_test_leakage(
        self,
        df: pd.DataFrame,
        test_cutoff: Optional[str] = None,
    ) -> None:
        """Enforce strict code-level runtime assertion against test data leakage.

        Uses `DEFAULT_TEST_START_DATE` from `src.data.loader` as the single source
        of truth. Raises ValueError immediately if input data reaches or exceeds the test cutoff.
        """
        if test_cutoff is None:
            test_cutoff = DEFAULT_TEST_START_DATE

        if self.time_col not in df.columns:
            # Check if there is an alternative date column name
            date_col = resolve_date_col()
            if date_col in df.columns:
                series = df[date_col]
            else:
                return
        else:
            series = df[self.time_col]

        max_dt = pd.to_datetime(series).max()
        cutoff_dt = pd.Timestamp(test_cutoff)

        if max_dt >= cutoff_dt:
            raise ValueError(
                f"CRITICAL DATA LEAKAGE DETECTED: Input DataFrame contains timestamps up to "
                f"{max_dt.date()}, which reaches or exceeds canonical test cutoff {cutoff_dt.date()}. "
                f"Internal Auto-tuning, feature transformations, and fitting must only receive train_df."
            )

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> "BaseNixtlaAdapter":
        """Fit adapter on training data."""
        pass

    @abc.abstractmethod
    def predict(self, h: int, **kwargs) -> pd.DataFrame:
        """Generate forecasts for horizon h."""
        pass

    @abc.abstractmethod
    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Run rolling-origin cross validation."""
        pass
