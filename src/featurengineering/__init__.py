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
from .contract import (
    FeatureAvailability,
    FeatureSpec,
    SplitProvenance,
    FeatureLineage,
    FeatureContract,
    FeatureSet,
    compute_dataframe_fingerprint,
    get_canonical_repo_root,
    get_verified_eda_report_hash,
)
from .splits import (
    build_split_feature_sets,
    build_origin_mask,
    derive_common_origins_and_finalize_lineage,
    build_all_feature_levels,
)
from .stationarity_test import StationarityTester
from .validation import (
    assert_no_target_leakage,
    validate_target_excluded,
    validate_feature_matrix_schema,
    validate_availability,
    validate_causality,
    validate_label_alignment,
    validate_warmup_vs_missing,
    validate_split_provenance,
    diagnose_target_correlation,
)

__all__ = [
    # Builder & Splits Orchestration (primary API)
    'FeatureBuilder',
    'FeatureConfig',
    'build_split_feature_sets',
    'build_origin_mask',
    'derive_common_origins_and_finalize_lineage',
    'build_all_feature_levels',
    # Contract dataclasses
    'FeatureAvailability',
    'FeatureSpec',
    'SplitProvenance',
    'FeatureLineage',
    'FeatureContract',
    'FeatureSet',
    'compute_dataframe_fingerprint',
    'get_canonical_repo_root',
    'get_verified_eda_report_hash',
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
    'validate_target_excluded',
    'validate_feature_matrix_schema',
    'validate_availability',
    'validate_causality',
    'validate_label_alignment',
    'validate_warmup_vs_missing',
    'validate_split_provenance',
    'diagnose_target_correlation',
]