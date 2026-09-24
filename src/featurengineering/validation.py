"""
Lightweight leakage guards for feature engineering.

These are heuristic sanity checks, not exhaustive proofs — they catch
the most common failure mode (a feature that suspiciously correlates
more with the *current* target than any lagged version) cheaply enough
to run routinely.
"""

from typing import List, Optional

import pandas as pd
import numpy as np
import warnings


def assert_no_target_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: Optional[str] = None,
    threshold: float = 0.15,
    raise_on_leak: bool = False,
) -> List[str]:
    """Flag features that correlate suspiciously with the current target.

    For each feature, compare ``corr(feature[t], target[t])`` vs
    ``corr(feature[t], target[t-1])``.  If the former exceeds the latter
    by more than *threshold*, the feature is flagged — it may be
    "peeking" at the current target value (e.g., an un-shifted rolling
    mean that includes today's rainfall in its window).

    Args:
        df: DataFrame with features and target, sorted by time.
        feature_cols: Columns to check.
        target_col: Name of the target column.
        date_col: Unused, kept for API consistency.
        threshold: Min difference ``|corr_t0 - corr_t1|`` to flag.
        raise_on_leak: If ``True``, raise ``ValueError`` instead of
            just returning the list + warning.

    Returns:
        List of flagged feature names (empty if clean).
    """
    flagged: List[str] = []
    target_current = df[target_col]
    target_lagged = df[target_col].shift(1)

    for col in feature_cols:
        if col == target_col or col not in df.columns:
            continue

        corr_t0 = df[col].corr(target_current)
        corr_t1 = df[col].corr(target_lagged)

        # Skip if either correlation is NaN (constant column, etc.)
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
