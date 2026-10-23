"""
Unified Feature Engineering Pipeline (Phase 3 Core Builder)
============================================================

``FeatureBuilder`` is the single source of truth for feature generation.
It implements:
- Dual-mode warm-up handling (`drop_warmup=True` vs `drop_warmup=False`).
- 4-layer continuity verification via `check_continuity(..., raise_on_gap=True)`.
- 5-step imputation & feature construction order:
    1. Validation of continuity, schema, and target completeness (fail-fast on target NaN).
    2. Source predictor imputation via frozen Dev Train `_imputation_stats` before transforms.
    3. Causal feature construction (lags, inclusive rolling, circular wind, calendar).
    4. Segregation and distinct tracking of warm-up lookback NaNs (never routed to imputation).
    5. Zero blanket `dropna()`, deterministic slicing, and fail-fast NaN guards.
- Mode A (no context), Mode B (with valid context), and Mode C (zero-warmup F0 path).
- FeatureSet generation binding maxtrix X, dates, y_source, contract, and provenance.
- MSTL guard (`use_mstl=True` raises `NotImplementedError`).
- Config flags with Phase 3 transitional defaults (flipped to confirmed defaults in Phase 5).
"""

from __future__ import annotations

import sys
import hashlib
import json
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
from ..data.dataquality import check_continuity
from .contract import (
    FeatureAvailability,
    FeatureSpec,
    SplitProvenance,
    FeatureLineage,
    FeatureContract,
    FeatureSet,
    compute_dataframe_fingerprint,
)
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

    level: int = 3

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

    # Methodology flags (Phase 5 confirmed defaults = True)
    rolling_include_current: bool = True
    include_target_lag0: bool = True
    encode_circular: bool = True
    circular_wind_columns: List[str] = field(
        default_factory=lambda: ["Hướng gió 2m", "Hướng gió 10m"]
    )

    @property
    def warmup_rows(self) -> int:
        """
        Approved warmup_rows calculation:
        warmup_rows = max(lag_lookback, rolling_lookback)
        where rolling_lookback = max(W) - 1 if rolling_include_current else max(W).
        """
        lag_lb = max(self.lag_periods, default=0) if self.lag_columns and self.lag_periods else 0
        if self.rolling_columns and self.rolling_windows:
            max_w = max(self.rolling_windows, default=0)
            roll_lb = (max_w - 1) if self.rolling_include_current else max_w
        else:
            roll_lb = 0
        return max(lag_lb, roll_lb)

    def compute_config_hash(self) -> str:
        """Deterministic hash of feature configuration."""
        payload = {
            "level": int(self.level),
            "lag_columns": list(self.lag_columns),
            "lag_periods": list(self.lag_periods),
            "rolling_columns": list(self.rolling_columns),
            "rolling_windows": list(self.rolling_windows),
            "rolling_stats": list(self.rolling_stats),
            "temporal_features": list(self.temporal_features),
            "wet_season_months": list(self.wet_season_months),
            "use_mstl": self.use_mstl,
            "imputation_strategy": self.imputation_strategy,
            "apply_quality_filter": self.apply_quality_filter,
            "rolling_include_current": self.rolling_include_current,
            "include_target_lag0": self.include_target_lag0,
            "encode_circular": self.encode_circular,
            "circular_wind_columns": list(self.circular_wind_columns),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_level(
        cls,
        level: Union[int, FeatureLevel] = 3,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        rolling_include_current: bool = True,
        include_target_lag0: bool = True,
        encode_circular: bool = True,
    ) -> "FeatureConfig":
        """Factory method defining Feature Construction levels (F0 -> F3).

        Phase 5 confirmed approved methodology:
        - rolling_include_current = True (inclusive rolling [t-29 .. t])
        - include_target_lag0 = True (Lượng mưa_lag_0 for F1-F3)
        - encode_circular = True (circular wind encoding)
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
                level=0,
                lag_columns=[],
                lag_periods=[],
                rolling_columns=[],
                rolling_windows=[],
                rolling_stats=[],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
                rolling_include_current=rolling_include_current,
                include_target_lag0=False,
                encode_circular=encode_circular,
            )
        elif lvl == 1:
            return cls(
                level=1,
                lag_columns=key_lag_cols,
                lag_periods=[1, 2, 3, 7],
                rolling_columns=[],
                rolling_windows=[],
                rolling_stats=[],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
                rolling_include_current=rolling_include_current,
                include_target_lag0=include_target_lag0,
                encode_circular=encode_circular,
            )
        elif lvl == 2:
            return cls(
                level=2,
                lag_columns=key_lag_cols,
                lag_periods=[1, 2, 3, 7],
                rolling_columns=key_roll_cols,
                rolling_windows=[7, 14, 30],
                rolling_stats=["mean", "std"],
                temporal_features=[],
                apply_quality_filter=False,
                target_col=t_col,
                date_col=d_col,
                rolling_include_current=rolling_include_current,
                include_target_lag0=include_target_lag0,
                encode_circular=encode_circular,
            )
        elif lvl == 3:
            return cls(
                level=3,
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
                rolling_include_current=rolling_include_current,
                include_target_lag0=include_target_lag0,
                encode_circular=encode_circular,
            )
        else:
            raise ValueError(f"Unknown feature level: {level}. Expected 0, 1, 2, or 3.")

    @classmethod
    def legacy_v1(
        cls,
        level: Union[int, FeatureLevel] = 3,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> "FeatureConfig":
        """Explicit legacy v1 configuration (shifted rolling, no lag0, raw wind)."""
        return cls.from_level(
            level=level,
            target_col=target_col,
            date_col=date_col,
            rolling_include_current=False,
            include_target_lag0=False,
            encode_circular=False,
        )


def get_feature_breakdown(
    feature_cols: List[str],
    target_col: str = "Lượng mưa",
    date_col: str = "Ngày",
) -> Dict[str, int]:
    """Dynamically categorize feature columns into raw, lag, rolling, seasonal counts."""
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
        fb.fit(train_df)                                       # learn stats on Dev Train only
        fset_train = fb.build_feature_set(train_df)            # Mode A
        fset_val   = fb.build_feature_set(val_df, context_df=train_df)  # Mode B
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.target_col = resolve_target_col(config.target_col)
        self.date_col = resolve_date_col(config.date_col)

        # Fitted state (populated strictly by .fit())
        self._imputation_stats: Optional[Dict[str, Any]] = None
        self._quality_stats: Optional[Dict[str, Any]] = None
        self._mstl_results: Optional[Dict[str, Any]] = None
        self._is_fitted = False

        # Fit metadata for provenance and fingerprinting (T26, T30)
        self.fit_date_min: Optional[pd.Timestamp] = None
        self.fit_date_max: Optional[pd.Timestamp] = None
        self.fit_n_rows: Optional[int] = None
        self.fit_data_hash: Optional[str] = None

    def warmup_rows(self) -> int:
        """Calculate warmup_rows based on current configuration."""
        return self.config.warmup_rows

    # ------------------------------------------------------------------
    # fit — Dev Train only
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn data-dependent state from the **training split only**.

        This method must NEVER be called on validation, test, or forecast-time data.
        """
        # Step 1: Target completeness fail-fast check (T5, T20)
        if self.target_col in train_df.columns and train_df[self.target_col].isna().any():
            raise ValueError(
                f"Target column '{self.target_col}' contains NaN; target observations are never imputed."
            )

        # Continuity check
        if self.date_col in train_df.columns and len(train_df) > 1:
            check_continuity(train_df, date_col=self.date_col, raise_on_gap=True)

        # MSTL guard (T18)
        if self.config.use_mstl:
            raise NotImplementedError("MSTL feature decomposition is not implemented in primary pipeline.")

        # Record Dev Train fit metadata
        if self.date_col in train_df.columns:
            self.fit_date_min = pd.Timestamp(train_df[self.date_col].min())
            self.fit_date_max = pd.Timestamp(train_df[self.date_col].max())
        self.fit_n_rows = len(train_df)
        self.fit_data_hash = compute_dataframe_fingerprint(train_df)

        # 1. Imputation stats (predictor columns only)
        self._imputation_stats = fit_missing_value_stats(
            train_df, strategy=self.config.imputation_strategy
        )

        # 2. Optional feature quality filtering stats
        if self.config.apply_quality_filter:
            # Build features with drop_warmup=True to fit quality stats
            df_featured = self.transform(train_df, drop_warmup=True)
            self._quality_stats = fit_feature_quality(
                df_featured,
                target_col=self.target_col,
                correlation_threshold=self.config.correlation_threshold,
                variance_threshold=self.config.variance_threshold,
            )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform — 5-Step Execution Order
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        context_df: Optional[pd.DataFrame] = None,
        drop_warmup: bool = True,
    ) -> pd.DataFrame:
        """Batch feature generation using frozen fitted state.

        Executes the mandatory 5-step sequence:
        1. Validation of continuity, schema, and target completeness.
        2. Source predictor imputation via frozen Dev Train _imputation_stats.
        3. Causal feature construction (lags, inclusive rolling, circular wind, calendar).
        4. Distinct tracking of warm-up lookback NaNs (never routed to imputation).
        5. Mode-specific warm-up handling & fail-fast NaN guard (zero blanket dropna()).
        """
        self._check_fitted()

        # Step 1: Validation of Continuity, Schema, and Target Completeness
        if self.target_col in df.columns and df[self.target_col].isna().any():
            raise ValueError(
                f"Target column '{self.target_col}' contains NaN; target observations are never imputed."
            )

        if self.date_col in df.columns and len(df) > 1:
            check_continuity(df, date_col=self.date_col, raise_on_gap=True)

        warmup = self.warmup_rows()

        # Context evaluation (Mode A, Mode B, Mode C)
        context_tail: Optional[pd.DataFrame] = None
        if warmup > 0:
            if context_df is not None:
                # Mode B or fail-fast on insufficient context
                if len(context_df) < warmup:
                    raise ValueError(
                        f"Insufficient context_df length: received {len(context_df)}, but {warmup} rows "
                        f"are required to guarantee complete feature generation without losing split origins."
                    )
                if self.date_col in context_df.columns and len(context_df) > 1:
                    check_continuity(context_df, date_col=self.date_col, raise_on_gap=True)

                context_tail = context_df.tail(warmup).copy()

                # Boundary continuity check
                if self.date_col in df.columns and self.date_col in context_tail.columns:
                    gap_days = (pd.Timestamp(df[self.date_col].iloc[0]) - pd.Timestamp(context_tail[self.date_col].iloc[-1])).days
                    if gap_days != 1:
                        raise ValueError(
                            f"Split boundary continuity violation: {gap_days} days gap between "
                            f"context tail ({context_tail[self.date_col].iloc[-1]}) and split start ({df[self.date_col].iloc[0]})."
                        )
            else:
                # Mode A: no context
                context_tail = None
        else:
            # Mode C: warmup == 0 (Level F0)
            context_tail = None

        # Assemble combined series for transform calculation
        if context_tail is not None and len(context_tail) > 0:
            combined_df = pd.concat([context_tail, df], ignore_index=True)
            if self.date_col in combined_df.columns and len(combined_df) > 1:
                check_continuity(combined_df, date_col=self.date_col, raise_on_gap=True)
        else:
            combined_df = df.copy().reset_index(drop=True)

        # Step 2: Source Predictor Imputation (Dev Train Frozen Parameters)
        # Target column is preserved without imputation
        target_series = combined_df[self.target_col].copy() if self.target_col in combined_df.columns else None
        imputed_df = apply_missing_value_fill(combined_df, self._imputation_stats)
        if target_series is not None:
            imputed_df[self.target_col] = target_series

        # Step 3: Causal Feature Construction
        df_out = self._apply_feature_transforms(imputed_df)

        # Quality filter if configured
        if self.config.apply_quality_filter and self._quality_stats is not None:
            df_out = filter_features(
                df_out,
                self._quality_stats,
                target_col=self.target_col,
                date_col=self.date_col,
            )

        # Step 4 & 5: Mode-Specific Warm-Up Handling & Fail-Fast NaN Guard
        if drop_warmup:
            if context_tail is not None and len(context_tail) > 0:
                # Mode B: slice off prepended context
                df_final = df_out.iloc[len(context_tail):].reset_index(drop=True)
            elif warmup > 0:
                # Mode A: drop exactly leading warmup_rows
                df_final = df_out.iloc[warmup:].reset_index(drop=True)
            else:
                # Mode C: zero warmup
                df_final = df_out.reset_index(drop=True)

            # Post-Slicing NaN Guard: 100% feature columns must be non-null
            feat_cols = [c for c in df_final.columns if c not in [self.target_col, self.date_col]]
            if df_final[feat_cols].isna().any().any():
                nan_cols = df_final[feat_cols].isna().sum()[df_final[feat_cols].isna().sum() > 0].to_dict()
                raise ValueError(
                    f"Unexpected NaN in feature columns after drop_warmup=True: {nan_cols}"
                )
        else:
            # Full diagnostic mode (drop_warmup=False)
            if context_tail is not None and len(context_tail) > 0:
                df_final = df_out.iloc[len(context_tail):].reset_index(drop=True)
            else:
                df_final = df_out.reset_index(drop=True)

            feat_cols = [c for c in df_final.columns if c not in [self.target_col, self.date_col]]

            # Lookback-Boundary NaN Guard:
            # Rows >= warmup must contain 0 NaNs
            post_warmup = df_final.iloc[warmup:]
            if post_warmup[feat_cols].isna().any().any():
                nan_cols = post_warmup[feat_cols].isna().sum()[post_warmup[feat_cols].isna().sum() > 0].to_dict()
                raise ValueError(
                    f"Unexpected NaN in post-warmup rows (rows >= {warmup}): {nan_cols}"
                )

            # Non-lookback columns must contain 0 NaNs across all rows
            lookback_keywords = ["_lag_", "_mean_", "_std_", "_min_", "_max_", "_sum_"]
            lookback_cols = [c for c in feat_cols if any(k in c for k in lookback_keywords)]
            non_lookback = [c for c in feat_cols if c not in lookback_cols]
            if len(non_lookback) > 0 and df_final[non_lookback].isna().any().any():
                nan_cols = df_final[non_lookback].isna().sum()[df_final[non_lookback].isna().sum() > 0].to_dict()
                raise ValueError(
                    f"Unexpected NaN in non-lookback columns during warm-up period: {nan_cols}"
                )

        return df_final

    # ------------------------------------------------------------------
    # build_feature_set — FeatureSet Production
    # ------------------------------------------------------------------

    def build_feature_set(
        self,
        df: pd.DataFrame,
        context_df: Optional[pd.DataFrame] = None,
        split_name: str = "dev_train",
        clean_data_hash: Optional[str] = None,
        eda_report_hash: Optional[str] = None,
        methodology_decisions: Optional[Dict[str, Any]] = None,
        drop_warmup: bool = True,
    ) -> FeatureSet:
        """
        Build standardized FeatureSet binding feature matrix X, dates, target y, and lineage.

        Separates X (strictly feature columns) from dates and y_source.
        Produces full FeatureContract and SplitProvenance.
        """
        df_transformed = self.transform(df, context_df=context_df, drop_warmup=drop_warmup)

        dates = df_transformed[self.date_col].copy()
        y_source = df_transformed[self.target_col].copy()
        feature_cols = [c for c in df_transformed.columns if c not in [self.target_col, self.date_col]]
        X = df_transformed[feature_cols].copy()

        # Build formal FeatureSpecs for contract
        specs: List[FeatureSpec] = []
        for col in feature_cols:
            if col == f"{self.target_col}_lag_0":
                spec = FeatureSpec(
                    name=col,
                    source_columns=(self.target_col,),
                    transform_type="lag",
                    lag_or_window=0,
                    include_current=True,
                    availability_class=FeatureAvailability.HISTORICAL_AT_ORIGIN,
                    availability_assumption="Observed rainfall on origin day t",
                    metadata={"role": "target_persistence"},
                )
            elif "_lag_" in col:
                parts = col.rsplit("_lag_", 1)
                src = parts[0]
                lag_k = int(parts[1]) if parts[1].isdigit() else None
                spec = FeatureSpec(
                    name=col,
                    source_columns=(src,),
                    transform_type="lag",
                    lag_or_window=lag_k,
                    include_current=False,
                    availability_class=FeatureAvailability.HISTORICAL_AT_ORIGIN,
                    availability_assumption=f"Historical observation at t - {lag_k}",
                )
            elif any(k in col for k in ["_mean_", "_std_", "_min_", "_max_", "_sum_"]):
                src = col.split("_")[0]
                win = None
                for w in self.config.rolling_windows:
                    if f"_{w}d" in col or f"_{w}_" in col or f"_{w}" in col:
                        win = w
                        break
                spec = FeatureSpec(
                    name=col,
                    source_columns=(src,),
                    transform_type="rolling",
                    lag_or_window=win,
                    include_current=self.config.rolling_include_current,
                    availability_class=FeatureAvailability.HISTORICAL_AT_ORIGIN,
                    availability_assumption="Causal rolling window",
                )
            elif col.endswith("_sin") or col.endswith("_cos"):
                is_wind = any(w in col for w in self.config.circular_wind_columns)
                if is_wind:
                    base_wind = col[:-4]
                    spec = FeatureSpec(
                        name=col,
                        source_columns=(base_wind,),
                        transform_type="temporal",
                        lag_or_window=None,
                        include_current=True,
                        availability_class=FeatureAvailability.HISTORICAL_AT_ORIGIN,
                        expanded_from=base_wind,
                        availability_assumption="Trigonometric wind direction encoding",
                    )
                else:
                    spec = FeatureSpec(
                        name=col,
                        source_columns=(self.date_col,),
                        transform_type="temporal",
                        lag_or_window=None,
                        include_current=True,
                        availability_class=FeatureAvailability.KNOWN_FUTURE_CALENDAR,
                        availability_assumption="Deterministic calendar attribute",
                    )
            elif col == "Is_Wet_Season":
                spec = FeatureSpec(
                    name=col,
                    source_columns=(self.date_col,),
                    transform_type="temporal",
                    lag_or_window=None,
                    include_current=True,
                    availability_class=FeatureAvailability.KNOWN_FUTURE_CALENDAR,
                    availability_assumption="Calendar month classification",
                    metadata={"wet_season_months": tuple(self.config.wet_season_months)},
                )
            else:
                # Raw meteorological predictor
                spec = FeatureSpec(
                    name=col,
                    source_columns=(col,),
                    transform_type="raw",
                    lag_or_window=None,
                    include_current=True,
                    availability_class=FeatureAvailability.HISTORICAL_AT_ORIGIN,
                    availability_assumption="Observed meteorology on origin day t",
                )
            specs.append(spec)

        contract = FeatureContract(
            features=tuple(specs),
            feature_level=f"F{int(self.config.level)}",
            config_hash=self.config.compute_config_hash(),
            feature_names=tuple(feature_cols),
            wet_season_months=tuple(self.config.wet_season_months),
        )

        warmup = self.warmup_rows()
        warmup_dropped = warmup if (context_df is None and drop_warmup and warmup > 0) else 0
        context_avail = len(context_df) if context_df is not None else 0
        context_cons = min(warmup, context_avail) if (context_df is not None and warmup > 0) else 0

        prov = SplitProvenance(
            split_name=split_name,
            date_min=pd.Timestamp(df[self.date_col].min()),
            date_max=pd.Timestamp(df[self.date_col].max()),
            n_days=len(df),
            fit_date_min=self.fit_date_min,
            fit_date_max=self.fit_date_max,
            fit_n_rows=self.fit_n_rows,
            fit_data_hash=self.fit_data_hash,
            context_rows_available=context_avail,
            context_rows_consumed=context_cons,
            warmup_rows_dropped=warmup_dropped,
            feature_valid_row_count=len(X),
            feature_valid_purged_origin_count=max(0, len(X) - 7),
            raw_purged_origin_count=max(0, len(df) - 7),
            common_origin_count=None,
        )

        lineage = FeatureLineage(
            contract_hash=contract.compute_contract_hash(),
            clean_data_hash=clean_data_hash or "",
            eda_report_hash=eda_report_hash or "",
            wet_season_months=contract.wet_season_months,
            methodology_decisions=methodology_decisions or {},
            provenance=prov,
        )

        return FeatureSet(
            X=X,
            dates=dates,
            y_source=y_source,
            warmup_dropped=warmup_dropped,
            contract=contract,
            lineage=lineage,
        )

    # ------------------------------------------------------------------
    # build_single_step — recursive forecast parity
    # ------------------------------------------------------------------

    def build_single_step(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute features for the NEXT timestep from recent history.

        Guarantees exact same code path as batch training — zero train/serve skew.
        Note: Single-step train-serve parity verifies implementation consistency,
        but does NOT validate multi-step recursive forecast validity (which compounds error
        and requires legitimate future covariates).
        """
        self._check_fitted()

        needed = self.warmup_rows() + 1
        if len(history) < needed:
            raise ValueError(
                f"build_single_step: history length ({len(history)}) must be >= "
                f"warmup_rows + 1 ({needed})."
            )

        tail = history.tail(needed + 10).copy()
        transformed = self.transform(tail, drop_warmup=True)

        if len(transformed) == 0:
            raise ValueError(
                f"build_single_step: transform() returned 0 rows from {len(tail)}-row tail."
            )

        return transformed.tail(1).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_feature_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms (lag, rolling, temporal, etc.)."""
        df_out = df.copy()

        # Circular wind direction encoding
        if self.config.encode_circular:
            for w_col in self.config.circular_wind_columns:
                if w_col in df_out.columns:
                    rad = np.radians(df_out[w_col].astype(float))
                    df_out[f"{w_col}_sin"] = np.sin(rad)
                    df_out[f"{w_col}_cos"] = np.cos(rad)
                    df_out = df_out.drop(columns=[w_col])

        # Approved target lag-0 (D1b)
        if self.config.include_target_lag0 and self.target_col in df_out.columns:
            df_out[f"{self.target_col}_lag_0"] = df_out[self.target_col].copy()

        # Lag features
        if self.config.lag_columns and self.config.lag_periods:
            df_out = create_lag_features(
                df_out,
                columns_to_lag=self.config.lag_columns,
                lags=self.config.lag_periods,
            )

        # Rolling features (with inclusive / shifted support)
        if self.config.rolling_columns and self.config.rolling_windows:
            df_out = create_rolling_features(
                df_out,
                columns_to_roll=self.config.rolling_columns,
                windows=self.config.rolling_windows,
                stats=self.config.rolling_stats,
                include_current=self.config.rolling_include_current,
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
