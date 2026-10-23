"""
Structural Validators for Feature Engineering Pipeline (Phase 2)
================================================================

Implements formal validation functions enforcing structural invariants,
leakage prevention, schema compliance, and provenance integrity:
- `validate_target_excluded`: Checks that raw target 'Lượng mưa' and date 'Ngày' are excluded from X.
- `validate_feature_matrix_schema`: Checks column names, sequence, counts, NaN/Inf absence, and tau absence.
- `validate_availability`: Checks that zero features require external future forecasts.
- `validate_causality`: Checks non-anticipative causality under data perturbation at t' > t.
- `validate_label_alignment`: Checks that Y[i, h-1] strictly matches calendar target on origin + h days.
- `validate_warmup_vs_missing`: Checks dual-mode warm-up handling (drop_warmup=True vs False).
- `validate_split_provenance`: Checks split bounds, fit isolation, and origin counts.
- `diagnose_target_correlation`: Auxiliary diagnostic reporting correlations with future targets.
- `assert_no_target_leakage`: Backward-compatible legacy check for suspicious current-target correlations.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import warnings
import pandas as pd
import numpy as np

from .contract import (
    FeatureAvailability,
    FeatureContract,
    FeatureLineage,
    FeatureSet,
    compute_dataframe_fingerprint,
)


def validate_target_excluded(
    X: pd.DataFrame,
    target_col: str = "Lượng mưa",
    date_col: str = "Ngày",
) -> None:
    """
    T1: Validate target and date exclusion from the feature matrix X.

    Verifies that neither raw target_col nor date_col exists in X.columns.
    Declared lag features (e.g. 'Lượng mưa_lag_0', 'Lượng mưa_lag_1') are permitted.

    Raises:
        ValueError: If raw target_col or date_col is present in X.columns.
    """
    for forbidden in (target_col, date_col):
        if forbidden in X.columns:
            raise ValueError(
                f"Target/date exclusion violation: raw column '{forbidden}' must not exist in feature matrix X. "
                f"(Declared target lags such as '{forbidden}_lag_0' are permitted)."
            )


def validate_feature_matrix_schema(
    X: pd.DataFrame,
    contract: FeatureContract,
    expected_level_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    T11, T11b, T16: Validate feature matrix schema, column ordering, completeness, and threshold absence.

    Invariants checked:
        1. Exact column names and ordering match contract.feature_names.
        2. Column counts match expected level counts (F0: 36, F1: 53, F2: 65, F3: 70).
        3. Zero NaN values across all cells.
        4. Zero Infinite values across all cells.
        5. Zero rainfall threshold parameters (tau, rain_threshold, classification_threshold)
           or threshold-dependent binary features in column names (T16).

    Raises:
        ValueError: If any schema invariant is violated.
    """
    if expected_level_counts is None:
        expected_level_counts = {"F0": 36, "F1": 53, "F2": 65, "F3": 70}

    # 1. Column name and ordering match
    if list(X.columns) != list(contract.feature_names):
        missing = set(contract.feature_names) - set(X.columns)
        extra = set(X.columns) - set(contract.feature_names)
        order_diff = [
            (i, act, exp)
            for i, (act, exp) in enumerate(zip(X.columns, contract.feature_names))
            if act != exp
        ] if len(X.columns) == len(contract.feature_names) else []
        raise ValueError(
            f"Feature matrix schema mismatch with contract.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns: {extra}\n"
            f"Order mismatches (first 5): {order_diff[:5]}"
        )

    # 2. Count verification
    level = contract.feature_level
    if level in expected_level_counts:
        exp_count = expected_level_counts[level]
        if len(X.columns) != exp_count:
            raise ValueError(
                f"Feature count mismatch for level '{level}': expected {exp_count}, got {len(X.columns)}"
            )

    # 3. Completeness (no NaN)
    if X.isna().any().any():
        nan_cols = X.isna().sum()[X.isna().sum() > 0].to_dict()
        raise ValueError(f"Feature matrix contains unexpected NaN values: {nan_cols}")

    # 4. Numerical validity (no Inf)
    numeric_data = X.select_dtypes(include=[np.number]).to_numpy()
    if np.isinf(numeric_data).any():
        raise ValueError("Feature matrix contains Infinite values.")

    # 5. Absence of rainfall threshold (T16)
    threshold_keywords = ("tau", "rain_threshold", "classification_threshold", "is_wet", "dry_day", "wet_day")
    for col in X.columns:
        if col == "Is_Wet_Season":
            continue
        col_lower = col.lower()
        for kw in threshold_keywords:
            if kw in col_lower:
                raise ValueError(
                    f"Threshold absence violation (T16): column '{col}' contains threshold keyword '{kw}'. "
                    f"Baseline feature levels F0-F3 must be completely threshold-independent."
                )


def validate_availability(contract: FeatureContract) -> None:
    """
    T10: Validate feature availability classifications in FeatureContract.

    Invariants checked:
        Zero features with availability_class == FeatureAvailability.REQUIRES_EXTERNAL_FORECAST.
        Only HISTORICAL_AT_ORIGIN and KNOWN_FUTURE_CALENDAR are permitted for F0-F3.

    Raises:
        ValueError: If any feature requires unavailable external future forecasts.
    """
    for spec in contract.features:
        if spec.availability_class == FeatureAvailability.REQUIRES_EXTERNAL_FORECAST:
            raise ValueError(
                f"Availability violation (T10): feature '{spec.name}' has forbidden availability class "
                f"'REQUIRES_EXTERNAL_FORECAST'. F0-F3 features must be observable at origin t."
            )
        if spec.availability_class not in {
            FeatureAvailability.HISTORICAL_AT_ORIGIN,
            FeatureAvailability.KNOWN_FUTURE_CALENDAR,
        }:
            raise ValueError(
                f"Unknown availability class '{spec.availability_class}' on feature '{spec.name}'."
            )


def validate_causality(
    builder_or_clean_X: Any,
    df_or_perturbed_X: Any = None,
    origin_date: Optional[Union[pd.Timestamp, int]] = None,
    perturb_date: Optional[pd.Timestamp] = None,
    target_col: str = "Lượng mưa",
    date_col: str = "Ngày",
    context_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    T2, T2b: Validate non-anticipative causality.

    Assures that perturbing observation values at timestamps t' > t
    produces strictly zero changes in the feature vector at origin t.

    Supported calling patterns:
    1. Builder-based:
       validate_causality(builder, df, origin_date=Timestamp('2010-01-15'), perturb_date=Timestamp('2010-01-16'))
    2. Direct matrix-based:
       validate_causality(X_clean, X_perturbed, origin_date=row_index_int)

    Raises:
        ValueError: If perturbation at t' > t altered features at origin t.
    """
    # Pattern 2: Direct matrix comparison
    if isinstance(builder_or_clean_X, pd.DataFrame) and isinstance(df_or_perturbed_X, pd.DataFrame):
        X_clean = builder_or_clean_X
        X_perturbed = df_or_perturbed_X
        idx = origin_date if isinstance(origin_date, int) else 0

        clean_row = X_clean.iloc[idx].to_numpy(dtype=float)
        perturbed_row = X_perturbed.iloc[idx].to_numpy(dtype=float)

        if not np.allclose(clean_row, perturbed_row, atol=1e-12, rtol=0, equal_nan=True):
            diffs = X_clean.columns[~np.isclose(clean_row, perturbed_row, atol=1e-12, rtol=0, equal_nan=True)].tolist()
            raise ValueError(
                f"Causality violation: perturbation at t' > t altered feature values at origin row {idx}: {diffs}"
            )
        return

    # Pattern 1: Builder transformation
    builder = builder_or_clean_X
    df = df_or_perturbed_X

    if origin_date is None or perturb_date is None:
        raise ValueError("origin_date and perturb_date must be provided for builder-based causality validation.")

    o_ts = pd.Timestamp(origin_date)
    p_ts = pd.Timestamp(perturb_date)

    if p_ts <= o_ts:
        raise ValueError(
            f"Invalid causality test configuration: perturb_date ({p_ts}) must be strictly after origin_date ({o_ts})."
        )

    # Clean transform
    fset_clean = builder.transform(df, context_df=context_df)
    X_clean = fset_clean.X if hasattr(fset_clean, "X") else fset_clean

    # Perturbed transform
    df_perturbed = df.copy()
    mask = df_perturbed[date_col] == p_ts
    if not mask.any():
        raise ValueError(f"perturb_date '{p_ts}' not found in dataframe.")
    df_perturbed.loc[mask, target_col] = df_perturbed.loc[mask, target_col] + 99.0

    fset_perturbed = builder.transform(df_perturbed, context_df=context_df)
    X_perturbed = fset_perturbed.X if hasattr(fset_perturbed, "X") else fset_perturbed

    # Locate origin date row
    origin_mask = df[date_col] == o_ts
    if not origin_mask.any():
        raise ValueError(f"origin_date '{o_ts}' not found in dataframe.")
    origin_idx = df.index[origin_mask][0]

    # Adjust for warmup dropped if transform returned sliced matrix
    if len(X_clean) < len(df):
        warmup_dropped = len(df) - len(X_clean)
        origin_idx = origin_idx - warmup_dropped
        if origin_idx < 0:
            raise ValueError(f"origin_date '{o_ts}' lies in dropped warm-up region.")

    clean_row = X_clean.iloc[origin_idx].to_numpy(dtype=float)
    perturbed_row = X_perturbed.iloc[origin_idx].to_numpy(dtype=float)

    if not np.allclose(clean_row, perturbed_row, atol=1e-12, rtol=0, equal_nan=True):
        diffs = X_clean.columns[~np.isclose(clean_row, perturbed_row, atol=1e-12, rtol=0, equal_nan=True)].tolist()
        raise ValueError(
            f"Causality violation: perturbation at date '{p_ts}' altered features at origin '{o_ts}': {diffs}"
        )


def validate_label_alignment(
    S: pd.DataFrame,
    Y_samples: np.ndarray,
    origin_dates: Union[pd.Series, np.ndarray, List[pd.Timestamp]],
    target_col: str = "Lượng mưa",
    date_col: str = "Ngày",
    horizon: int = 7,
) -> None:
    """
    T3, T8d, T9: Validate multi-horizon label alignment and chronological origin integrity.

    Invariants checked:
        1. origin_dates is strictly monotonic increasing and unique.
        2. len(origin_dates) == len(Y_samples).
        3. For every sample i and horizon h in {1 .. horizon}:
           Y_samples[i, h - 1] strictly equals S.loc[origin_dates[i] + h days, target_col].

    Raises:
        ValueError: If any alignment discrepancy is detected.
    """
    dates_series = pd.Series(pd.to_datetime(origin_dates)).reset_index(drop=True)

    if len(dates_series) != len(Y_samples):
        raise ValueError(
            f"Sample count mismatch: len(origin_dates)={len(dates_series)} != len(Y_samples)={len(Y_samples)}"
        )

    if not dates_series.is_monotonic_increasing:
        raise ValueError("Origin dates sequence violation: origin_dates is not strictly monotonic increasing.")

    if not dates_series.is_unique:
        duplicates = dates_series[dates_series.duplicated()].tolist()
        raise ValueError(f"Origin dates uniqueness violation: duplicate dates found: {duplicates[:5]}")

    # Build calendar target lookup from sample frame S
    s_dates = pd.to_datetime(S[date_col])
    target_lookup = dict(zip(s_dates, S[target_col].to_numpy()))

    for i, origin_ts in enumerate(dates_series):
        o_pd_ts = pd.Timestamp(origin_ts)
        for h in range(1, horizon + 1):
            target_ts = o_pd_ts + pd.DateOffset(days=int(h))
            if target_ts not in target_lookup:
                raise ValueError(
                    f"Label alignment violation at sample {i}: target date '{target_ts}' "
                    f"(origin '{origin_ts}' + {h} days) not found in sample frame."
                )
            expected_val = target_lookup[target_ts]
            actual_val = Y_samples[i, h - 1]

            if not np.isclose(actual_val, expected_val, atol=1e-8, rtol=1e-5, equal_nan=True):
                raise ValueError(
                    f"Label alignment mismatch at sample {i}, origin '{origin_ts}', horizon {h} "
                    f"(calendar date '{target_ts}'): expected {expected_val}, got {actual_val}."
                )


def validate_warmup_vs_missing(
    T: pd.DataFrame,
    warmup_rows: int,
    drop_warmup: bool = True,
    lookback_cols: Optional[List[str]] = None,
) -> None:
    """
    T5, Step 5: Validate missing-value vs. warm-up lookback NaN invariants.

    Invariants checked:
        - When drop_warmup=True: 100% of feature columns must be non-null. Any NaN raises ValueError.
        - When drop_warmup=False:
            • Rows i >= warmup_rows must contain 0 NaNs across all columns.
            • Non-lookback columns (raw meteo, calendar) must contain 0 NaNs across all rows.
            • Only lookback columns (lags, rolling) are permitted to have NaNs in rows < warmup_rows.

    Raises:
        ValueError: If unexpected NaNs are detected.
    """
    if drop_warmup:
        if T.isna().any().any():
            nan_summary = T.isna().sum()[T.isna().sum() > 0].to_dict()
            raise ValueError(
                f"Missing value violation (drop_warmup=True): feature matrix contains unexpected NaNs: {nan_summary}"
            )
    else:
        # 1. Post-warmup region must be 100% complete
        post_warmup = T.iloc[warmup_rows:]
        if post_warmup.isna().any().any():
            nan_summary = post_warmup.isna().sum()[post_warmup.isna().sum() > 0].to_dict()
            raise ValueError(
                f"Missing value violation (drop_warmup=False): unexpected NaNs in post-warmup rows "
                f"(rows >= {warmup_rows}): {nan_summary}"
            )

        # 2. Non-lookback columns must have 0 NaNs across entire series
        if lookback_cols is not None:
            non_lookback = [c for c in T.columns if c not in lookback_cols]
            if len(non_lookback) > 0 and T[non_lookback].isna().any().any():
                bad_cols = T[non_lookback].isna().sum()[T[non_lookback].isna().sum() > 0].to_dict()
                raise ValueError(
                    f"Missing value violation (drop_warmup=False): non-lookback columns contain NaNs: {bad_cols}"
                )


def validate_split_provenance(
    lineage: FeatureLineage,
    canonical_splits: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """
    T26, T30: Validate split provenance, Dev Train fit scope isolation, and origin counts.

    Invariants checked:
        1. split_name in {'dev_train', 'val', 'test'}.
        2. context_rows_consumed <= context_rows_available.
        3. Dev Train fit boundaries strictly: 2001-01-01 to 2018-04-30, N=6329.
        4. Split dates, raw purged origin counts, and valid origin relationships match specification.
        5. If canonical_splits is provided, fit_data_hash strictly matches Dev Train fingerprint.

    Raises:
        ValueError: If any provenance invariant is violated.
    """
    prov = lineage.provenance

    if prov.split_name not in {"dev_train", "val", "test"}:
        raise ValueError(f"Unknown split_name '{prov.split_name}'.")

    if prov.context_rows_consumed > prov.context_rows_available:
        raise ValueError(
            f"Context overflow: consumed {prov.context_rows_consumed} > available {prov.context_rows_available}"
        )

    # Dev Train fit scope isolation invariants
    if prov.split_name == "dev_train":
        if prov.fit_date_min != pd.Timestamp("2001-01-01"):
            raise ValueError(f"Dev Train fit_date_min violation: expected 2001-01-01, got {prov.fit_date_min}")
        if prov.fit_date_max != pd.Timestamp("2018-04-30"):
            raise ValueError(f"Dev Train fit_date_max violation: expected 2018-04-30, got {prov.fit_date_max}")
        if prov.fit_n_rows != 6329:
            raise ValueError(f"Dev Train fit_n_rows violation: expected 6329, got {prov.fit_n_rows}")
        if prov.raw_purged_origin_count != 6322:
            raise ValueError(f"Dev Train raw_purged_origin_count violation: expected 6322, got {prov.raw_purged_origin_count}")
        if prov.feature_valid_row_count != (6329 - prov.warmup_rows_dropped):
            raise ValueError(
                f"Dev Train feature_valid_row_count mismatch: expected {6329 - prov.warmup_rows_dropped}, "
                f"got {prov.feature_valid_row_count}"
            )
        if prov.feature_valid_purged_origin_count != (prov.feature_valid_row_count - 7):
            raise ValueError(
                f"Dev Train feature_valid_purged_origin_count mismatch: "
                f"expected {prov.feature_valid_row_count - 7}, got {prov.feature_valid_purged_origin_count}"
            )

    elif prov.split_name == "val":
        if prov.date_min != pd.Timestamp("2018-05-01") or prov.date_max != pd.Timestamp("2020-04-30"):
            raise ValueError(f"Validation date bounds violation: got {prov.date_min} to {prov.date_max}")
        if prov.n_days != 731:
            raise ValueError(f"Validation n_days violation: expected 731, got {prov.n_days}")
        if prov.raw_purged_origin_count != 724:
            raise ValueError(f"Validation raw_purged_origin_count violation: expected 724, got {prov.raw_purged_origin_count}")
        if prov.warmup_rows_dropped != 0:
            raise ValueError(f"Validation warmup_rows_dropped violation: expected 0, got {prov.warmup_rows_dropped}")

    elif prov.split_name == "test":
        if prov.date_min != pd.Timestamp("2020-05-01") or prov.date_max != pd.Timestamp("2026-04-30"):
            raise ValueError(f"Test date bounds violation: got {prov.date_min} to {prov.date_max}")
        if prov.n_days != 2191:
            raise ValueError(f"Test n_days violation: expected 2191, got {prov.n_days}")
        if prov.raw_purged_origin_count != 2184:
            raise ValueError(f"Test raw_purged_origin_count violation: expected 2184, got {prov.raw_purged_origin_count}")
        if prov.warmup_rows_dropped != 0:
            raise ValueError(f"Test warmup_rows_dropped violation: expected 0, got {prov.warmup_rows_dropped}")

    # Canonical fingerprint match
    if canonical_splits is not None and "dev_train" in canonical_splits:
        dev_train = canonical_splits["dev_train"]
        expected_hash = compute_dataframe_fingerprint(dev_train)
        if prov.fit_data_hash != expected_hash:
            raise ValueError(
                f"Dev Train fingerprint mismatch (T26): fitted hash '{prov.fit_data_hash}' "
                f"does not match canonical Dev Train hash '{expected_hash}'."
            )


def diagnose_target_correlation(
    S: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Lượng mưa",
    horizon: int = 7,
    high_correlation_threshold: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    T15: Auxiliary diagnostic reporting correlations between features and future targets.

    Computes Pearson correlation between feature[t] and future target[t + h] for h in 1..horizon.
    Reports findings as diagnostic metadata.

    CRITICAL LEAKAGE PRINCIPLE (D1b, T15):
        This function is strictly an auxiliary diagnostic. It never independently certifies
        safety or rejects a legitimate feature solely because of correlation.
    """
    results: Dict[str, Dict[str, float]] = {}
    high_corr_warnings: List[str] = []

    for col in feature_cols:
        if col not in S.columns or col == target_col:
            continue
        col_res: Dict[str, float] = {}
        for h in range(1, horizon + 1):
            future_target = S[target_col].shift(-h)
            valid_mask = ~(S[col].isna() | future_target.isna())
            if valid_mask.sum() > 10:
                corr = float(S.loc[valid_mask, col].corr(future_target.loc[valid_mask]))
                col_res[f"h{h}"] = round(corr, 4) if not np.isnan(corr) else 0.0
                if abs(corr) >= high_correlation_threshold:
                    high_corr_warnings.append(f"{col} with target[t+{h}]: r={corr:.3f}")
            else:
                col_res[f"h{h}"] = 0.0
        results[col] = col_res

    if high_corr_warnings:
        warnings.warn(
            f"diagnose_target_correlation: {len(high_corr_warnings)} feature-horizon pairs have "
            f"|r| >= {high_correlation_threshold} (auxiliary diagnostic only):\n"
            + "\n".join(high_corr_warnings[:5]),
            UserWarning,
            stacklevel=2,
        )

    return results


def assert_no_target_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: Optional[str] = None,
    threshold: float = 0.15,
    raise_on_leak: bool = False,
) -> List[str]:
    """Flag features that correlate suspiciously with the current target (legacy helper)."""
    flagged: List[str] = []
    target_current = df[target_col]
    target_lagged = df[target_col].shift(1)

    for col in feature_cols:
        if col == target_col or col not in df.columns:
            continue

        corr_t0 = df[col].corr(target_current)
        corr_t1 = df[col].corr(target_lagged)

        if pd.isna(corr_t0) or pd.isna(corr_t1):
            continue

        diff = abs(corr_t0) - abs(corr_t1)
        if diff > threshold:
            flagged.append(col)

    if flagged:
        msg = (
            f"assert_no_target_leakage: {len(flagged)} feature(s) correlate "
            f"suspiciously more with target[t] than target[t-1] "
            f"(threshold={threshold}): {flagged[:10]}"
        )
        if raise_on_leak:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return flagged
