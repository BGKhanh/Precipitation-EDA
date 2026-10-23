"""
Feature Contract Dataclasses & Data Structures (Phase 1)
=========================================================

Defines immutable, leakage-safe data structures governing the feature engineering pipeline:
- `FeatureAvailability`: Classification enum for feature availability at origin.
- `FeatureSpec`: Formal specification of an individual feature column.
- `SplitProvenance`: Strict provenance tracking row counts, warm-up drops, and purged origins.
- `FeatureLineage`: Hash-bound reproducibility metadata across data, EDA, and decisions.
- `FeatureContract`: Contract definition binding specs, levels, hashes, and feature names.
- `FeatureSet`: Standardized split output binding feature matrix X, timestamps, target y, and lineage.
- `compute_dataframe_fingerprint`: Deterministic hash of DataFrame content, dtypes, and ordering.
"""

from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np


class FeatureAvailability(str, Enum):
    """
    Availability classification of a feature at forecast origin t.

    - HISTORICAL_AT_ORIGIN: Observed on or before origin day t (e.g., lags, historical rolling).
    - KNOWN_FUTURE_CALENDAR: Deterministically known for future dates (e.g., day-of-week, month).
    - REQUIRES_EXTERNAL_FORECAST: Unobserved future values requiring external forecasts (FORBIDDEN in F0-F3).
    """
    HISTORICAL_AT_ORIGIN = "HISTORICAL_AT_ORIGIN"
    KNOWN_FUTURE_CALENDAR = "KNOWN_FUTURE_CALENDAR"
    REQUIRES_EXTERNAL_FORECAST = "REQUIRES_EXTERNAL_FORECAST"


@dataclass(frozen=True)
class FeatureSpec:
    """
    Formal specification of an individual feature column.

    Attributes:
        name: Output feature column name (e.g. 'Lượng mưa_lag_0', 'Nhiệt độ 2m_roll_mean_7').
        source_columns: Raw predictor columns used to build this feature.
        transform_type: Transform classification ('raw', 'lag', 'rolling', 'temporal', 'interaction').
        lag_or_window: Lag offset k or rolling window size w in days.
        include_current: True if origin day t is included (e.g. inclusive rolling or lag 0).
        availability_class: FeatureAvailability enum value.
        expanded_from: Source variable name if expanded (e.g. 'Hướng gió 2m' for sin/cos pairs).
        availability_assumption: Textual description of availability rationale.
        metadata: Domain metadata (e.g. wet_season_months, min_periods, ddof).
    """
    name: str
    source_columns: Tuple[str, ...]
    transform_type: str
    lag_or_window: Optional[int]
    include_current: bool
    availability_class: FeatureAvailability
    expanded_from: Optional[str] = None
    availability_assumption: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("FeatureSpec.name must be a non-empty string.")
        if not isinstance(self.source_columns, tuple):
            object.__setattr__(self, "source_columns", tuple(self.source_columns))
        if not isinstance(self.availability_class, FeatureAvailability):
            try:
                object.__setattr__(self, "availability_class", FeatureAvailability(self.availability_class))
            except Exception as e:
                raise ValueError(f"Invalid availability_class: {self.availability_class}") from e


@dataclass(frozen=True)
class SplitProvenance:
    """
    Strict split provenance tracking data boundaries, fit isolation, and origin counts.

    Attributes:
        split_name: Split identifier ('dev_train', 'val', 'test').
        date_min: Minimum timestamp in the split.
        date_max: Maximum timestamp in the split.
        n_days: Total calendar days in the split.
        fit_date_min: Minimum timestamp of the data split used for fitting (Dev Train only).
        fit_date_max: Maximum timestamp of the data split used for fitting (Dev Train only).
        fit_n_rows: Number of rows in the fitted data split (Dev Train only).
        fit_data_hash: Deterministic DataFrame content fingerprint of Dev Train.
        context_rows_available: Total rows in context_df supplied (e.g. 6329 for Val).
        context_rows_consumed: Rows actually consumed (min(warmup_rows, available)).
        warmup_rows_dropped: warmup_rows dropped for dev_train; 0 for val/test.
        feature_valid_row_count: Valid rows after warm-up, before purge (6329/6322/6300).
        feature_valid_purged_origin_count: Forecast origins after H=7 purge (6322/6315/6293).
        raw_purged_origin_count: Raw split origins after purge (6322, 724, 2184).
        common_origin_count: Intersection across F0-F3. None in Stage A; finalized in Stage B (6293).
    """
    split_name: str
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_days: int
    fit_date_min: Optional[pd.Timestamp]
    fit_date_max: Optional[pd.Timestamp]
    fit_n_rows: Optional[int]
    fit_data_hash: Optional[str]
    context_rows_available: int
    context_rows_consumed: int
    warmup_rows_dropped: int
    feature_valid_row_count: int
    feature_valid_purged_origin_count: int
    raw_purged_origin_count: int
    common_origin_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.split_name not in {"dev_train", "val", "test"}:
            raise ValueError(f"Unknown split_name '{self.split_name}', must be 'dev_train', 'val', or 'test'.")
        if self.date_min > self.date_max:
            raise ValueError(f"date_min ({self.date_min}) cannot be after date_max ({self.date_max}).")
        if self.feature_valid_row_count < 0:
            raise ValueError("feature_valid_row_count cannot be negative.")
        if self.feature_valid_purged_origin_count < 0:
            raise ValueError("feature_valid_purged_origin_count cannot be negative.")
        if self.raw_purged_origin_count < 0:
            raise ValueError("raw_purged_origin_count cannot be negative.")
        if self.common_origin_count is not None and self.common_origin_count < 0:
            raise ValueError("common_origin_count cannot be negative.")


@dataclass(frozen=True)
class FeatureLineage:
    """
    Hash-bound reproducibility metadata across data, EDA, and methodology decisions.

    Attributes:
        contract_hash: Deterministic SHA-256 hash of the FeatureContract.
        clean_data_hash: SHA-256 hash of raw input data.
        eda_report_hash: SHA-256 hash of canonical EDA report (01.7 / eda_report.json).
        wet_season_months: Tuple of wet season months (e.g. (5, 6, 7, 8, 9, 10, 11)).
        methodology_decisions: Structured dictionary of confirmed architecture decisions.
        provenance: SplitProvenance record for the current split.
    """
    contract_hash: str
    clean_data_hash: str
    eda_report_hash: str
    wet_season_months: Tuple[int, ...]
    methodology_decisions: Dict[str, Any]
    provenance: SplitProvenance

    def with_common_origin_count(self, common_origin_count: int) -> "FeatureLineage":
        """Return a copy with finalized common_origin_count in provenance."""
        new_prov = replace(self.provenance, common_origin_count=common_origin_count)
        return replace(self, provenance=new_prov)


@dataclass(frozen=True)
class FeatureContract:
    """
    Formal contract specifying all feature columns, levels, and deterministic hashes.

    Attributes:
        features: Tuple of FeatureSpec definitions.
        feature_level: Feature complexity level ('F0', 'F1', 'F2', 'F3').
        config_hash: SHA-256 hash of the builder configuration.
        feature_names: Tuple of exact output column names in canonical order.
        wet_season_months: Tuple of months designated as wet season.
    """
    features: Tuple[FeatureSpec, ...]
    feature_level: str
    config_hash: str
    feature_names: Tuple[str, ...]
    wet_season_months: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.feature_names, tuple):
            object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not isinstance(self.wet_season_months, tuple):
            object.__setattr__(self, "wet_season_months", tuple(self.wet_season_months))

        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"Mismatch: len(features)={len(self.features)} != len(feature_names)={len(self.feature_names)}"
            )

    def compute_contract_hash(self) -> str:
        """
        Compute deterministic SHA-256 hash over contract components.

        Covers: feature_level, config_hash, wet_season_months, feature_names,
        and every individual FeatureSpec attribute.
        """
        payload = {
            "feature_level": self.feature_level,
            "config_hash": self.config_hash,
            "wet_season_months": list(self.wet_season_months),
            "feature_names": list(self.feature_names),
            "specs": [
                {
                    "name": s.name,
                    "source_columns": list(s.source_columns),
                    "transform_type": s.transform_type,
                    "lag_or_window": s.lag_or_window,
                    "include_current": s.include_current,
                    "availability_class": s.availability_class.value,
                    "expanded_from": s.expanded_from,
                    "availability_assumption": s.availability_assumption,
                    "metadata": s.metadata,
                }
                for s in self.features
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to JSON-serializable dictionary."""
        return {
            "feature_level": self.feature_level,
            "config_hash": self.config_hash,
            "contract_hash": self.compute_contract_hash(),
            "wet_season_months": list(self.wet_season_months),
            "feature_names": list(self.feature_names),
            "features": [
                {
                    "name": s.name,
                    "source_columns": list(s.source_columns),
                    "transform_type": s.transform_type,
                    "lag_or_window": s.lag_or_window,
                    "include_current": s.include_current,
                    "availability_class": s.availability_class.value,
                    "expanded_from": s.expanded_from,
                    "availability_assumption": s.availability_assumption,
                    "metadata": dict(s.metadata),
                }
                for s in self.features
            ],
        }


@dataclass(frozen=True)
class FeatureSet:
    """
    Standardized split feature set binding feature matrix X, timestamps, target y, and lineage.

    Invariants:
        1. X contains ONLY contract.feature_names (no raw target 'Lượng mưa', no 'Ngày').
        2. dates contains origin timestamps, strictly aligned with X rows.
        3. y_source contains raw observed rainfall, strictly aligned with X rows.
        4. warmup_dropped records leading rows sliced to satisfy warm-up lookback.
        5. contract and lineage provide complete provenance and verification hashes.
    """
    X: pd.DataFrame
    dates: pd.Series
    y_source: pd.Series
    warmup_dropped: int
    contract: FeatureContract
    lineage: FeatureLineage

    def __post_init__(self) -> None:
        if len(self.X) != len(self.dates):
            raise ValueError(f"Row count mismatch: len(X)={len(self.X)} != len(dates)={len(self.dates)}")
        if len(self.X) != len(self.y_source):
            raise ValueError(f"Row count mismatch: len(X)={len(self.X)} != len(y_source)={len(self.y_source)}")

        # Target and date structural exclusion invariants
        for forbidden in ("Lượng mưa", "Ngày"):
            if forbidden in self.X.columns:
                raise ValueError(
                    f"Structural violation: raw column '{forbidden}' must not be present in FeatureSet.X"
                )

        # Feature schema alignment invariant
        if list(self.X.columns) != list(self.contract.feature_names):
            raise ValueError("FeatureSet.X columns do not match contract.feature_names.")

    def to_sample_frame(self, target_col: str = "Lượng mưa", date_col: str = "Ngày") -> pd.DataFrame:
        """
        Assemble sample frame S containing feature columns, date column, and target column.

        Used by `build_purged_direct_samples(S, feature_cols=contract.feature_names, ...)`
        to generate multi-step direct forecast targets Y and origin-aligned arrays.
        """
        if target_col in self.X.columns:
            raise ValueError(f"Target column '{target_col}' already exists in feature matrix X.")
        if date_col in self.X.columns:
            raise ValueError(f"Date column '{date_col}' already exists in feature matrix X.")

        sample_frame = self.X.copy()
        sample_frame[date_col] = self.dates.to_numpy()
        sample_frame[target_col] = self.y_source.to_numpy()
        return sample_frame.reset_index(drop=True)

    def with_finalized_origins(self, common_origin_count: int) -> "FeatureSet":
        """Return a copy of FeatureSet with finalized common_origin_count in lineage."""
        new_lineage = self.lineage.with_common_origin_count(common_origin_count)
        return replace(self, lineage=new_lineage)


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Deterministic DataFrame content fingerprint.

    Detects value changes, row permutations, column additions/deletions, and dtype casts.
    """
    hasher = hashlib.sha256()

    # 1. Schema metadata: column names and dtypes in order
    col_meta = [(str(col), str(df[col].dtype)) for col in df.columns]
    hasher.update(json.dumps(col_meta).encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))

    # 2. Exact values and row ordering
    # pd.util.hash_pandas_object computes 64-bit uint hash per row preserving values and dtypes
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    hasher.update(row_hashes.tobytes())

    return hasher.hexdigest()


def get_canonical_repo_root() -> Path:
    """Resolve repository root directory robustly from module location."""
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def get_verified_eda_report_hash(
    artifact_path: Optional[Union[str, Path]] = None,
    expected_baseline_hash: Optional[str] = "F1C2EA61FD9B32A7171D465B76EE7DEFE59C0BF63958F66BCD59DAC2D0F5ED38",
) -> str:
    """Read actual reports/eda/eda_report.json, compute SHA-256, verify against baseline, and return actual hash."""
    from pathlib import Path
    if artifact_path is None:
        report_path = get_canonical_repo_root() / "reports" / "eda" / "eda_report.json"
    else:
        report_path = Path(artifact_path)

    if not report_path.exists():
        raise FileNotFoundError(f"EDA report artifact missing: {report_path}")

    actual_hash = hashlib.sha256(report_path.read_bytes()).hexdigest().upper()
    if expected_baseline_hash and actual_hash != expected_baseline_hash:
        raise ValueError(
            f"EDA report artifact hash drift detected!\n"
            f"  Path checked:      {report_path}\n"
            f"  Expected baseline: {expected_baseline_hash}\n"
            f"  Actual on disk:    {actual_hash}\n"
            f"  Remediation: If this artifact change is intentional, update the versioned baseline."
        )
    return actual_hash
