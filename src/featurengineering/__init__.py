"""Feature Engineering Utilities for DS108 Weather Prediction Project"""

from .utils import (
    select_features,
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    create_interaction_features,
    create_mstl_features,
    # Leakage-safe fit/apply pattern (preferred)
    fit_missing_value_stats,
    apply_missing_value_fill,
    fit_feature_quality,
    filter_features,
    # Legacy (deprecated — emit FutureWarning)
    handle_missing_values,
    validate_feature_quality,
)

from .builder import FeatureBuilder, FeatureConfig
from .stationarity_test import StationarityTester
from .validation import assert_no_target_leakage

__all__ = [
    # Builder (primary API going forward)
    'FeatureBuilder',
    'FeatureConfig',
    # Low-level primitives
    'select_features',
    'create_temporal_features',
    'create_lag_features',
    'create_rolling_features',
    'create_interaction_features',
    'create_mstl_features',
    # Leakage-safe helpers
    'fit_missing_value_stats',
    'apply_missing_value_fill',
    'fit_feature_quality',
    'filter_features',
    # Legacy (deprecated)
    'handle_missing_values',
    'validate_feature_quality',
    # Stationarity
    'StationarityTester',
    # Validation
    'assert_no_target_leakage',
]