"""
Unified feature engineering pipeline.

``FeatureBuilder`` is the **single source of truth** for feature generation.
It wraps the low-level primitives in ``utils.py`` with an explicit
``fit`` / ``transform`` / ``build_single_step`` interface that enforces
leakage-safe behaviour and guarantees train–serve consistency.

Rule: ``feature-engineering-consistency.md``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..config.resolve import resolve_target_col, resolve_date_col
from .utils import (
    create_lag_features,
    create_rolling_features,
    create_temporal_features,
    create_interaction_features,
    create_mstl_features,
    fit_missing_value_stats,
    apply_missing_value_fill,
    fit_feature_quality,
    filter_features,
)


# ======================================================================
# Configuration dataclass
# ======================================================================

@dataclass
class FeatureConfig:
    """All knobs for feature generation, in one place.

    Populate from ``EDAReport`` (when available) or hardcode for quick
    experiments — either way, the full config is explicit and traceable.
    """

    # Lag features
    lag_columns: List[str] = field(default_factory=list)
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 7])

    # Rolling features
    rolling_columns: List[str] = field(default_factory=list)
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std'])

    # Temporal features
    temporal_features: List[str] = field(
        default_factory=lambda: [
            'Month_sin', 'Month_cos',
            'DayOfYear_sin', 'DayOfYear_cos',
            'Is_Wet_Season',
        ]
    )
    wet_season_months: List[int] = field(
        default_factory=lambda: [5, 6, 7, 8, 9, 10, 11]
    )

    # Interaction features
    interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    interaction_operations: List[str] = field(
        default_factory=lambda: ['multiply']
    )

    # MSTL features (fit from TemporalAnalysis results)
    use_mstl: bool = False
    mstl_lag_periods: Optional[List[int]] = None
    mstl_rolling_windows: Optional[List[int]] = None
    mstl_rolling_stats: Optional[List[str]] = None

    # Missing-value imputation strategy
    imputation_strategy: str = 'fill_mean'

    # Feature quality filtering
    correlation_threshold: float = 0.05
    variance_threshold: float = 0.01
    apply_quality_filter: bool = True

    # Column names
    target_col: Optional[str] = None
    date_col: Optional[str] = None


# ======================================================================
# FeatureBuilder
# ======================================================================

class FeatureBuilder:
    """Single feature-generation pipeline for training and forecasting.

    Usage::

        fb = FeatureBuilder(config)
        fb.fit(train_df)                    # learn stats on train only
        X_train = fb.transform(train_df)    # batch features
        X_test  = fb.transform(test_df)     # uses train-derived stats

        # recursive forecast — one row at a time:
        X_next = fb.build_single_step(history)
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.target_col = resolve_target_col(config.target_col)
        self.date_col = resolve_date_col(config.date_col)

        # Fitted state (populated by .fit())
        self._imputation_stats: Optional[Dict[str, Any]] = None
        self._quality_stats: Optional[Dict[str, Any]] = None
        self._mstl_results: Optional[Dict[str, Any]] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # fit — train-only
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn data-dependent state from the **training split only**.

        What gets fitted:
        - Imputation fill values (mean / median of each column)
        - Feature-quality stats (correlation / variance thresholds)
        - MSTL decomposition results (if ``config.use_mstl`` is True)

        This method must NEVER be called on test or forecast-time data.
        """
        print("FeatureBuilder.fit() - learning from train data only")

        # 1. Imputation stats
        self._imputation_stats = fit_missing_value_stats(
            train_df, strategy=self.config.imputation_strategy
        )

        # 2. Build features on train to compute quality stats
        df_featured = self._apply_feature_transforms(train_df)

        # 3. Feature quality (train-only)
        if self.config.apply_quality_filter:
            self._quality_stats = fit_feature_quality(
                df_featured,
                target_col=self.target_col,
                correlation_threshold=self.config.correlation_threshold,
                variance_threshold=self.config.variance_threshold,
            )

        self._is_fitted = True
        print("   FeatureBuilder fitted")
        return self

    # ------------------------------------------------------------------
    # transform — batch (train or test)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch feature generation using fitted state.

        Safe to call on train, test, or any temporal slice — no new
        statistics are computed from *df*.
        """
        self._check_fitted()

        # 1. Impute missing values using train-derived stats
        df_out = apply_missing_value_fill(df, self._imputation_stats)

        # 2. Apply all feature transforms
        df_out = self._apply_feature_transforms(df_out)

        # 3. Apply quality filter (train-derived column list)
        if self.config.apply_quality_filter and self._quality_stats is not None:
            df_out = filter_features(
                df_out,
                self._quality_stats,
                target_col=self.target_col,
                date_col=self.date_col,
            )

        # 4. Drop rows with NaN created by lag/rolling
        df_out = df_out.dropna().reset_index(drop=True)

        return df_out

    # ------------------------------------------------------------------
    # build_single_step — recursive forecast
    # ------------------------------------------------------------------

    def build_single_step(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute features for the NEXT timestep from recent history.

        **Implementation**: calls ``self.transform()`` on the tail of
        *history* and returns just the last row.  This guarantees the
        exact same code path as batch training — no parallel
        implementation, no train/serve skew.

        Args:
            history: Recent rows of (possibly partially predicted)
                history.  Must contain at least
                ``max(lag_periods + rolling_windows)`` rows.

        Returns:
            Single-row DataFrame with feature columns.
        """
        self._check_fitted()

        # How many rows do we need to compute the longest lag/rolling?
        max_lookback = max(
            max(self.config.lag_periods, default=0),
            max(self.config.rolling_windows, default=0),
        )
        # Add padding for shift(1) + rolling window
        needed = max_lookback + 10

        tail = history.tail(needed).copy()
        transformed = self.transform(tail)

        if len(transformed) == 0:
            raise ValueError(
                f"build_single_step: transform() returned 0 rows from "
                f"{len(tail)}-row tail.  Need more history "
                f"(at least {needed} rows)."
            )

        return transformed.tail(1).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_feature_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms (lag, rolling, temporal, etc.).

        This is the shared code path that both ``transform()`` and the
        quality-stats computation in ``fit()`` use.  It does NOT apply
        imputation or quality filtering — those are handled by the
        caller.
        """
        df_out = df.copy()

        # Default: use target column for lag/rolling if not specified
        lag_cols = self.config.lag_columns or [self.target_col]
        roll_cols = self.config.rolling_columns or [self.target_col]

        # Lag features
        if self.config.lag_periods:
            df_out = create_lag_features(
                df_out,
                columns_to_lag=lag_cols,
                lags=self.config.lag_periods,
            )

        # Rolling features (shift(1) applied by default — leakage safe)
        if self.config.rolling_windows:
            df_out = create_rolling_features(
                df_out,
                columns_to_roll=roll_cols,
                windows=self.config.rolling_windows,
                stats=self.config.rolling_stats,
                include_current=False,
            )

        # Temporal features
        if self.config.temporal_features:
            df_out = create_temporal_features(
                df_out,
                date_col=self.date_col,
                features_to_create=self.config.temporal_features,
                wet_season_months=self.config.wet_season_months,
            )

        # Interaction features
        if self.config.interaction_pairs:
            df_out = create_interaction_features(
                df_out,
                feature_pairs=self.config.interaction_pairs,
                operations=self.config.interaction_operations,
            )

        # MSTL features (if fitted)
        if self.config.use_mstl and self._mstl_results is not None:
            df_out = create_mstl_features(
                df_out,
                mstl_results=self._mstl_results,
                date_col=self.date_col,
                lag_periods=self.config.mstl_lag_periods,
                rolling_windows=self.config.mstl_rolling_windows,
                rolling_stats=self.config.mstl_rolling_stats,
            )

        return df_out

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureBuilder must be fitted before transform/build_single_step. "
                "Call .fit(train_df) first."
            )

    def set_mstl_results(self, mstl_results: Dict[str, Any]) -> None:
        """Inject MSTL decomposition results (from TemporalAnalysis).

        Should be called after fit() but before transform() if
        ``config.use_mstl`` is True.
        """
        self._mstl_results = mstl_results
        print("   MSTL results injected into FeatureBuilder")
