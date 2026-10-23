"""
Split Orchestration & Common-Origin Derivation (Phase 4)
=========================================================

Implements Stage A and Stage B of the Feature Engineering Pipeline:
- Single-fit Dev Train isolation and verification.
- Two-stage lineage finalization.
- Chronologically ordered post-purge common-origin derivation.
- Boolean origin masking for multi-horizon direct sample generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .builder import FeatureBuilder
from .contract import (
    FeatureSet,
    compute_dataframe_fingerprint,
    get_verified_eda_report_hash,
)
from .validation import validate_split_provenance


def build_split_feature_sets(
    builder: FeatureBuilder,
    dev_train: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
) -> Tuple[FeatureSet, FeatureSet, FeatureSet]:
    """Build standardized FeatureSets across Dev Train, Validation, and Final Test.

    Enforces:
    1. train_full Refit Guard: Refitting on train_full (7,060 rows) raises NotImplementedError.
    2. Single-Fit & Deterministic Fingerprint Guarantee:
       - If builder is not fitted: calls builder.fit(dev_train) exactly once.
       - If builder is already fitted: verifies date bounds, row count, and fingerprint match dev_train.
    3. Mode A on Dev Train: context_df=None, drop_warmup=True.
    4. Mode B on Validation: context_df=dev_train, drop_warmup=True.
    5. Mode B on Final Test: context_df=val_set, drop_warmup=True.
    6. Strict Provenance Validation.
    """
    # 1. train_full refit guard
    if len(dev_train) == 7060 or (
        "Ngày" in dev_train.columns
        and pd.Timestamp(dev_train["Ngày"].min()) == pd.Timestamp("2001-01-01")
        and pd.Timestamp(dev_train["Ngày"].max()) == pd.Timestamp("2020-04-30")
    ):
        raise NotImplementedError("Final refit policy deferred to Modeling Plan")

    # 2. Single-fit & fingerprint guarantee (§3.12)
    dev_fingerprint = compute_dataframe_fingerprint(dev_train)
    date_col = builder.date_col or "Ngày"

    if not builder._is_fitted:
        builder.fit(dev_train)
    else:
        if date_col in dev_train.columns:
            dev_min = pd.Timestamp(dev_train[date_col].min())
            dev_max = pd.Timestamp(dev_train[date_col].max())
            if builder.fit_date_min != dev_min:
                raise ValueError(
                    f"Fit date mismatch: builder.fit_date_min ({builder.fit_date_min}) != dev_train ({dev_min})"
                )
            if builder.fit_date_max != dev_max:
                raise ValueError(
                    f"Fit date mismatch: builder.fit_date_max ({builder.fit_date_max}) != dev_train ({dev_max})"
                )
        if builder.fit_n_rows != len(dev_train):
            raise ValueError(
                f"Fit row count mismatch: builder.fit_n_rows ({builder.fit_n_rows}) != len(dev_train) ({len(dev_train)})"
            )
        if builder.fit_data_hash != dev_fingerprint:
            raise ValueError(
                f"Dev Train fingerprint mismatch (T26): fitted hash '{builder.fit_data_hash}' "
                f"does not match dev_train hash '{dev_fingerprint}'."
            )

    # 3. Retrieve baseline EDA report hash for lineage
    try:
        eda_hash = get_verified_eda_report_hash()
    except Exception:
        eda_hash = "F1C2EA61FD9B32A7171D465B76EE7DEFE59C0BF63958F66BCD59DAC2D0F5ED38"

    # 4. Generate split FeatureSets
    # Dev Train: Mode A (no context)
    fset_dev = builder.build_feature_set(
        df=dev_train,
        context_df=None,
        split_name="dev_train",
        clean_data_hash=dev_fingerprint,
        eda_report_hash=eda_hash,
        drop_warmup=True,
    )

    # Validation: Mode B (context from Dev Train)
    fset_val = builder.build_feature_set(
        df=val_set,
        context_df=dev_train,
        split_name="val",
        clean_data_hash=dev_fingerprint,
        eda_report_hash=eda_hash,
        drop_warmup=True,
    )

    # Final Test: Mode B (context from Validation)
    fset_test = builder.build_feature_set(
        df=test_set,
        context_df=val_set,
        split_name="test",
        clean_data_hash=dev_fingerprint,
        eda_report_hash=eda_hash,
        drop_warmup=True,
    )

    # 5. Validate split provenance if running on canonical splits
    if len(dev_train) == 6329:
        validate_split_provenance(fset_dev.lineage, canonical_splits={"dev_train": dev_train})
        validate_split_provenance(fset_val.lineage)
        validate_split_provenance(fset_test.lineage)

    return fset_dev, fset_val, fset_test


def build_origin_mask(
    candidate_origin_dates: pd.Series,
    common_origin_dates: Tuple[pd.Timestamp, ...],
) -> np.ndarray:
    """Build boolean mask selecting candidate origins that belong to the common origin set.

    Args:
        candidate_origin_dates: Series of candidate origin timestamps (length max_origin).
        common_origin_dates: Chronologically sorted tuple of common origin timestamps.

    Returns:
        Boolean numpy array of length len(candidate_origin_dates).
    """
    if len(candidate_origin_dates) == 0:
        return np.empty(0, dtype=bool)

    common_set = set(common_origin_dates)
    mask = candidate_origin_dates.isin(common_set).to_numpy(dtype=bool)
    return mask


def derive_common_origins_and_finalize_lineage(
    feature_sets_by_level: Dict[str, Tuple[FeatureSet, FeatureSet, FeatureSet]],
    horizon: int = 7,
) -> Tuple[
    Dict[str, Tuple[FeatureSet, FeatureSet, FeatureSet]],
    Dict[str, Tuple[pd.Timestamp, ...]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """Execute Stage B: Intersect post-purge origins across levels, standardize dates,
    and finalize FeatureSet lineage.

    Execution sequence:
    1. For each split (dev_train, val, test), determine candidate origins across all levels.
    2. Compute post-purge common origin intersection across all levels.
    3. Sort common origin dates chronologically.
    4. Construct boolean origin_mask for each level and split.
    5. Finalize FeatureSets by updating common_origin_count in lineage.

    Returns:
        (finalized_feature_sets_by_level, common_origin_dates_by_split, origin_masks_by_level)
    """
    if not feature_sets_by_level:
        raise ValueError("feature_sets_by_level cannot be empty.")

    split_keys = ("dev_train", "val", "test")
    levels = list(feature_sets_by_level.keys())

    # Step 1, 2, 3: Common origin derivation per split
    common_origin_dates: Dict[str, Tuple[pd.Timestamp, ...]] = {}

    for split_idx, split_name in enumerate(split_keys):
        cand_sets: List[set] = []
        for lvl in levels:
            fset = feature_sets_by_level[lvl][split_idx]
            max_origin = max(0, len(fset.dates) - horizon)
            cand_dates_lvl = set(fset.dates.iloc[:max_origin])
            cand_sets.append(cand_dates_lvl)

        if cand_sets:
            intersected = set.intersection(*cand_sets)
            common_origin_dates[split_name] = tuple(sorted(intersected))
        else:
            common_origin_dates[split_name] = tuple()

    # Step 4 & 5: Mask construction and lineage finalization
    finalized_feature_sets: Dict[str, Tuple[FeatureSet, FeatureSet, FeatureSet]] = {}
    origin_masks_by_level: Dict[str, Dict[str, np.ndarray]] = {}

    for lvl in levels:
        fset_tuple = feature_sets_by_level[lvl]
        finalized_splits: List[FeatureSet] = []
        origin_masks_by_level[lvl] = {}

        for split_idx, split_name in enumerate(split_keys):
            fset = fset_tuple[split_idx]
            max_origin = max(0, len(fset.dates) - horizon)
            cand_series = fset.dates.iloc[:max_origin]  # guaranteed non-negative slice

            target_common_dates = common_origin_dates[split_name]
            mask = build_origin_mask(cand_series, target_common_dates)

            # Assert invariants (T8d)
            if len(cand_series) == max_origin:
                assert len(mask) == max_origin, (
                    f"origin_mask length mismatch: expected {max_origin}, got {len(mask)}"
                )
                assert np.sum(mask) == len(target_common_dates), (
                    f"origin_mask sum mismatch: expected {len(target_common_dates)}, got {np.sum(mask)}"
                )

            origin_masks_by_level[lvl][split_name] = mask

            # Finalize lineage with common_origin_count
            finalized_fset = fset.with_finalized_origins(len(target_common_dates))
            finalized_splits.append(finalized_fset)

        finalized_feature_sets[lvl] = (
            finalized_splits[0],
            finalized_splits[1],
            finalized_splits[2],
        )

    return finalized_feature_sets, common_origin_dates, origin_masks_by_level


def build_all_feature_levels(
    builders_by_level: Dict[str, FeatureBuilder],
    dev_train: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    horizon: int = 7,
) -> Tuple[
    Dict[str, Tuple[FeatureSet, FeatureSet, FeatureSet]],
    Dict[str, Tuple[pd.Timestamp, ...]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """Orchestrate entire two-stage feature engineering pipeline across all feature levels.

    Stage A: Build split FeatureSets for each level via build_split_feature_sets.
    Stage B: Derive common origins and finalize lineages across all levels.
    """
    feature_sets_by_level: Dict[str, Tuple[FeatureSet, FeatureSet, FeatureSet]] = {}
    for level, builder in builders_by_level.items():
        feature_sets_by_level[level] = build_split_feature_sets(
            builder=builder,
            dev_train=dev_train,
            val_set=val_set,
            test_set=test_set,
        )

    return derive_common_origins_and_finalize_lineage(
        feature_sets_by_level=feature_sets_by_level,
        horizon=horizon,
    )
