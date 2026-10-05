"""
Unified feature engineering pipeline.

``FeatureBuilder`` is the **single source of truth** for feature generation.
It wraps the low-level primitives in ``utils.py`` with an explicit
``fit`` / ``transform`` / ``build_single_step`` interface that enforces
leakage-safe behaviour and guarantees train–serve consistency.

Rule: ``feature-engineering-consistency.md``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
# Feature Level Enum & Configuration dataclass
# ======================================================================

class FeatureLevel(int, Enum):
    LEVEL_0_MINIMAL = 0
    LEVEL_1_TEMPORAL = 1
    LEVEL_2_STATISTICS = 2
    LEVEL_3_FULL_SEASONAL = 3


@dataclass
class FeatureConfig:
    """All knobs for feature generation, in one place.

    Populate via ``FeatureConfig.from_level(k)`` or customize explicitly.
    Default: ``apply_quality_filter = False`` to strictly separate
    Feature Construction from Feature Selection Policy.
    """

    # Lag features
    lag_columns: List[str] = field(default_factory=list)
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 7])

    # Rolling features
    rolling_columns: List[str] = field(default_factory=list)
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std'])

    # Temporal features
    temporal_features: List[str] = field(default_factory=list)
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

    # Feature quality filtering (Default False: Construction != Selection)
    correlation_threshold: float = 0.05
    variance_threshold: float = 0.01
    apply_quality_filter: bool = False

    # Column names
    target_col: Optional[str] = None
    date_col: Optional[str] = None

    @classmethod
    def from_level(
        cls,
        level: Union[int, FeatureLevel] = 3,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> "FeatureConfig":
        """Factory method defining exact, reproducible Feature Construction levels (F0 -> F3).

        - Level 0 (Minimal): Raw usable predictors (34 features) + basic calendar if any.
        - Level 1 (Temporal): Level 0 + 16 lags ([1, 2, 3, 7] on 4 key meteo columns).
        - Level 2 (Temporal Statistics): Level 1 + 12 rolling ([7, 14, 30] mean/std with shift(1)).
        - Level 3 (Full / Seasonal): Level 2 + 5 seasonal features (Month_sin/cos, DayOfYear_sin/cos, Is_Wet_Season).
        """
        lvl = int(level)
        t_col = target_col or "Lượng mưa"
        d_col = date_col or "Ngày"

        # 4 Core columns for lags
        key_lag_cols = [t_col, "Nhiệt độ 2m", "Độ ẩm tương đối 2m", "Tốc độ gió 2m"]
        # 2 Key columns for rolling
        key_roll_cols = [t_col, "Nhiệt độ 2m"]
        # 5 Seasonal features
        seasonal_cols = [
            "Month_sin", "Month_cos",
            "DayOfYear_sin", "DayOfYear_cos",
            "Is_Wet_Season"
        ]

        if lvl == 0:
            return cls(
                lag_columns=[],
                lag_periods=[],
                rolling_columns=[],
                rolling_windows=[],
                rolling_stats=[],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
            )
        elif lvl == 1:
            return cls(
                lag_columns=key_lag_cols,
                lag_periods=[1, 2, 3, 7],
                rolling_columns=[],
                rolling_windows=[],
                rolling_stats=[],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
            )
        elif lvl == 2:
            return cls(
                lag_columns=key_lag_cols,
                lag_periods=[1, 2, 3, 7],
                rolling_columns=key_roll_cols,
                rolling_windows=[7, 14, 30],
                rolling_stats=["mean", "std"],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
            )
        elif lvl == 3:
            return cls(
                lag_columns=key_lag_cols,
                lag_periods=[1, 2, 3, 7],
                rolling_columns=key_roll_cols,
                rolling_windows=[7, 14, 30],
                rolling_stats=["mean", "std"],
                temporal_features=seasonal_cols,
                wet_season_months=[5, 6, 7, 8, 9, 10, 11],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
            )
        else:
            raise ValueError(f"Unknown feature level: {level}. Expected 0, 1, 2, or 3.")


def get_feature_breakdown(
    feature_cols: List[str],
    target_col: str = "Lượng mưa",
    date_col: str = "Ngày",
) -> Dict[str, int]:
    """Dynamically categorize feature columns into raw, lag, rolling, seasonal counts.
    Never uses hardcoded counts.
    """
    clean_cols = [c for c in feature_cols if c not in [target_col, date_col]]
    
    lag_count = sum(1 for c in clean_cols if "lag" in c)
    rolling_count = sum(1 for c in clean_cols if any(k in c for k in ["_mean_", "_std_", "_min_", "_max_", "_sum_"]))
    seasonal_count = sum(1 for c in clean_cols if any(k in c for k in ["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "Is_Wet_Season"]))
    
    raw_count = len(clean_cols) - (lag_count + rolling_count + seasonal_count)
    
    return {
        "feature_count_total": len(clean_cols),
        "feature_count_raw": raw_count,
        "feature_count_lag": lag_count,
        "feature_count_rolling": rolling_count,
        "feature_count_seasonal": seasonal_count,
    }


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
        - Feature-quality stats (if apply_quality_filter is explicitly True)
        - MSTL decomposition results (if config.use_mstl is True)

        This method must NEVER be called on test or forecast-time data.
        """
        print("FeatureBuilder.fit() - learning from train data only")

        # 1. Imputation stats
        self._imputation_stats = fit_missing_value_stats(
            train_df, strategy=self.config.imputation_strategy
        )

        # 2. Build features on train to compute quality stats if requested
        df_featured = self._apply_feature_transforms(train_df)

        # 3. Feature quality (optional, default False)
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
        statistics are computed from df.
        """
        self._check_fitted()

        # 1. Impute missing values using train-derived stats
        df_out = apply_missing_value_fill(df, self._imputation_stats)

        # 2. Apply all feature transforms
        df_out = self._apply_feature_transforms(df_out)

        # 3. Apply quality filter if configured
        if self.config.apply_quality_filter and self._quality_stats is not None:
            df_out = filter_features(
                df_out,
                self._quality_stats,
                target_col=self.target_col,
                date_col=self.date_col,
            )

        # 4. Drop rows with NaN created by lag/rolling (if any were created)
        if self.config.lag_periods or self.config.rolling_windows:
            df_out = df_out.dropna().reset_index(drop=True)

        return df_out

    # ------------------------------------------------------------------
    # build_single_step — recursive forecast
    # ------------------------------------------------------------------

    def build_single_step(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute features for the NEXT timestep from recent history.

        Guarantees exact same code path as batch training — zero train/serve skew.
        """
        self._check_fitted()

        max_lookback = max(
            max(self.config.lag_periods, default=0) if self.config.lag_columns else 0,
            max(self.config.rolling_windows, default=0) if self.config.rolling_columns else 0,
        )
        needed = max_lookback + 10 if max_lookback > 0 else 5

        tail = history.tail(needed).copy()
        transformed = self.transform(tail)

        if len(transformed) == 0:
            raise ValueError(
                f"build_single_step: transform() returned 0 rows from "
                f"{len(tail)}-row tail. Need more history "
                f"(at least {needed} rows)."
            )

        return transformed.tail(1).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_feature_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms (lag, rolling, temporal, etc.)."""
        df_out = df.copy()

        # Lag features
        if self.config.lag_columns and self.config.lag_periods:
            df_out = create_lag_features(
                df_out,
                columns_to_lag=self.config.lag_columns,
                lags=self.config.lag_periods,
            )

        # Rolling features (shift(1) applied by default — leakage safe)
        if self.config.rolling_columns and self.config.rolling_windows:
            df_out = create_rolling_features(
                df_out,
                columns_to_roll=self.config.rolling_columns,
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
                lag_periods=self.config.mstl_lag_periods,
                rolling_windows=self.config.mstl_rolling_windows,
                rolling_stats=self.config.mstl_rolling_stats,
                target_col=self.target_col,
            )

        return df_out

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureBuilder is not fitted yet. Call .fit(train_df) first."
            )
