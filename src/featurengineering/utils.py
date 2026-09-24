# =============================================================================
# FEATURE ENGINEERING UTILITY FUNCTIONS
# =============================================================================

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import warnings

from ..config.constants import Config

warnings.filterwarnings('ignore')


def select_features(df: pd.DataFrame, 
                   features_to_keep: List[str],
                   target_col: str = None,
                   date_col: str = None,
                   keep_essential: bool = True) -> pd.DataFrame:
    """
    Select specific features from DataFrame
    
    Args:
        df: Input DataFrame
        features_to_keep: List of feature names to keep
        target_col: Target variable column name
        date_col: Date column name
        keep_essential: Whether to automatically include target and date columns
        
    Returns:
        DataFrame with selected features
        
    Example:
        # User decides based on correlation analysis
        strong_features = ['Nhiệt độ tối đa 2m', 'Độ ẩm tương đối 2m', 'Áp suất bề mặt']
        df_selected = select_features(df, strong_features)
    """
    print(f"[FEATURES] SELECTING FEATURES")
    print(f"   Features to select: {len(features_to_keep)}")
    
    # Get default columns if not provided
    target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    
    # Build final column list
    final_cols = []
    
    if keep_essential:
        if date_col in df.columns:
            final_cols.append(date_col)
        if target_col in df.columns:
            final_cols.append(target_col)
    
    # Add requested features that exist in DataFrame
    for feature in features_to_keep:
        if feature in df.columns and feature not in final_cols:
            final_cols.append(feature)
        elif feature not in df.columns:
            print(f"   [WARN] Feature '{feature}' not found in DataFrame")
    
    df_selected = df[final_cols].copy()
    
    print(f"   Original shape: {df.shape}")
    print(f"   Selected shape: {df_selected.shape}")
    print(f"   [OK] Feature selection completed")
    
    return df_selected


def create_temporal_features(df: pd.DataFrame,
                           date_col: str = None,
                           features_to_create: List[str] = None,
                           wet_season_months: List[int] = None) -> pd.DataFrame:
    """
    Create temporal features from date column
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        features_to_create: List of temporal features to create
        wet_season_months: Months considered as wet season
        
    Returns:
        DataFrame with temporal features added
        
    Example:
        # User decides based on seasonal analysis
        temporal_features = ['Month_sin', 'Month_cos', 'Is_Wet_Season']
        df_temporal = create_temporal_features(df, features_to_create=temporal_features)
    """
    print(f"[TEMPORAL] CREATING TEMPORAL FEATURES")
    
    # Default parameters
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    wet_season_months = wet_season_months or [5, 6, 7, 8, 9, 10, 11]  # May-November for HCMC
    
    if features_to_create is None:
        features_to_create = ['Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Is_Wet_Season']
    
    df_temporal = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_temporal[date_col]):
        df_temporal[date_col] = pd.to_datetime(df_temporal[date_col])
        print(f"   [OK] Converted {date_col} to datetime")
    
    feature_count = 0
    
    # Extract base time components if needed
    if any('Month' in f or 'DayOfYear' in f or 'Wet_Season' in f for f in features_to_create):
        df_temporal['_temp_month'] = df_temporal[date_col].dt.month
        df_temporal['_temp_dayofyear'] = df_temporal[date_col].dt.dayofyear
    
    # Create requested features
    for feature in features_to_create:
        if feature == 'Year':
            df_temporal['Year'] = df_temporal[date_col].dt.year
            feature_count += 1
        elif feature == 'Month':
            df_temporal['Month'] = df_temporal['_temp_month']
            feature_count += 1
        elif feature == 'DayofMonth':
            df_temporal['DayofMonth'] = df_temporal[date_col].dt.day
            feature_count += 1
        elif feature == 'DayofYear':
            df_temporal['DayofYear'] = df_temporal['_temp_dayofyear']
            feature_count += 1
        elif feature == 'Month_sin':
            df_temporal['Month_sin'] = np.sin(2 * np.pi * df_temporal['_temp_month'] / 12)
            feature_count += 1
        elif feature == 'Month_cos':
            df_temporal['Month_cos'] = np.cos(2 * np.pi * df_temporal['_temp_month'] / 12)
            feature_count += 1
        elif feature == 'DayOfYear_sin':
            df_temporal['DayOfYear_sin'] = np.sin(2 * np.pi * df_temporal['_temp_dayofyear'] / 365.25)
            feature_count += 1
        elif feature == 'DayOfYear_cos':
            df_temporal['DayOfYear_cos'] = np.cos(2 * np.pi * df_temporal['_temp_dayofyear'] / 365.25)
            feature_count += 1
        elif feature == 'Is_Wet_Season':
            df_temporal['Is_Wet_Season'] = df_temporal['_temp_month'].isin(wet_season_months).astype(int)
            feature_count += 1
        else:
            print(f"   [WARN] Unknown temporal feature: {feature}")
    
    # Clean up temporary columns
    temp_cols = [col for col in df_temporal.columns if col.startswith('_temp_')]
    df_temporal = df_temporal.drop(columns=temp_cols)
    
    print(f"   [OK] Created {feature_count} temporal features")
    print(f"   Dataset shape: {df_temporal.shape}")
    
    return df_temporal


def create_lag_features(df: pd.DataFrame,
                       columns_to_lag: Union[str, List[str]],
                       lags: List[int],
                       suffix: str = 'lag') -> pd.DataFrame:
    """
    Create lag features for specified columns
    
    Args:
        df: Input DataFrame
        columns_to_lag: Column name(s) to create lag features for
        lags: List of lag periods
        suffix: Suffix for lag feature names
        
    Returns:
        DataFrame with lag features added
        
    Example:
        # User decides based on autocorrelation analysis
        df_lag = create_lag_features(df, 'Lượng mưa', [1, 2, 3, 7])
        df_lag = create_lag_features(df_lag, ['Nhiệt độ tối đa 2m'], [1])
    """
    print(f"[LAG] CREATING LAG FEATURES")
    
    if isinstance(columns_to_lag, str):
        columns_to_lag = [columns_to_lag]
    
    df_lag = df.copy()
    feature_count = 0
    
    for column in columns_to_lag:
        if column not in df_lag.columns:
            print(f"   [WARN] Column '{column}' not found")
            continue
        
        for lag in lags:
            feature_name = f'{column}_{suffix}_{lag}'
            df_lag[feature_name] = df_lag[column].shift(lag)
            feature_count += 1
    
    print(f"   [OK] Created {feature_count} lag features")
    print(f"   Columns lagged: {columns_to_lag}")
    print(f"   Lag periods: {lags}")
    print(f"   Dataset shape: {df_lag.shape}")
    
    return df_lag


def create_rolling_features(df: pd.DataFrame,
                           columns_to_roll: Union[str, List[str]],
                           windows: List[int],
                           stats: List[str] = ['mean', 'std', 'min', 'max', 'sum'],
                           include_current: bool = False) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.

    By default, the series is **shifted by 1 period before rolling** so
    that the current row's value is excluded from the window.  This
    prevents target-in-feature leakage when rolling the target (or a
    target-derived) column.

    Args:
        df: Input DataFrame
        columns_to_roll: Column name(s) to create rolling features for
        windows: List of window sizes
        stats: List of statistics to calculate
        include_current: If ``True``, do NOT shift before rolling —
            only set this for columns that are NOT the target and
            where there is a deliberate reason (rare).

    Returns:
        DataFrame with rolling features added

    Example:
        # User decides based on temporal analysis
        df_rolling = create_rolling_features(df, 'Lượng mưa', [7, 14, 30], ['sum', 'mean'])
        df_rolling = create_rolling_features(df_rolling, ['Nhiệt độ tối đa 2m'], [7], ['mean', 'std'])
    """
    print(f"[ROLLING] CREATING ROLLING FEATURES")
    if not include_current:
        print(f"   [LEAK-SAFE] Leakage-safe mode: shift(1) applied before rolling")
    
    if isinstance(columns_to_roll, str):
        columns_to_roll = [columns_to_roll]
    
    df_rolling = df.copy()
    feature_count = 0
    
    stat_funcs = {
        'mean': lambda s, w: s.rolling(window=w).mean(),
        'std':  lambda s, w: s.rolling(window=w).std(),
        'min':  lambda s, w: s.rolling(window=w).min(),
        'max':  lambda s, w: s.rolling(window=w).max(),
        'sum':  lambda s, w: s.rolling(window=w).sum(),
    }
    
    for column in columns_to_roll:
        if column not in df_rolling.columns:
            print(f"   [WARN] Column '{column}' not found")
            continue
        
        # Shift to exclude current row (leakage prevention)
        series = df_rolling[column] if include_current else df_rolling[column].shift(1)
        
        for window in windows:
            for stat in stats:
                if stat not in stat_funcs:
                    print(f"   [WARN] Unknown statistic: {stat}")
                    continue
                
                feature_name = f'{column}_{stat}_{window}d'
                df_rolling[feature_name] = stat_funcs[stat](series, window)
                feature_count += 1
    
    print(f"   [OK] Created {feature_count} rolling features")
    print(f"   Columns: {columns_to_roll}")
    print(f"   Windows: {windows}")
    print(f"   Statistics: {stats}")
    print(f"   Dataset shape: {df_rolling.shape}")
    
    return df_rolling


def create_interaction_features(df: pd.DataFrame,
                              feature_pairs: List[tuple],
                              operations: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> pd.DataFrame:
    """
    Create interaction features between feature pairs
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples with feature pairs
        operations: List of operations to perform
        
    Returns:
        DataFrame with interaction features added
        
    Example:
        # User decides based on domain knowledge
        pairs = [('Nhiệt độ tối đa 2m', 'Độ ẩm tương đối 2m')]
        df_interact = create_interaction_features(df, pairs, ['multiply'])
    """
    print(f"[INTERACTION] CREATING INTERACTION FEATURES")
    
    df_interact = df.copy()
    feature_count = 0
    
    for feature1, feature2 in feature_pairs:
        if feature1 not in df_interact.columns or feature2 not in df_interact.columns:
            print(f"   [WARN] Feature pair ({feature1}, {feature2}) not found")
            continue
        
        for operation in operations:
            if operation == 'multiply':
                feature_name = f'{feature1}_x_{feature2}'
                df_interact[feature_name] = df_interact[feature1] * df_interact[feature2]
            elif operation == 'divide':
                feature_name = f'{feature1}_div_{feature2}'
                df_interact[feature_name] = df_interact[feature1] / (df_interact[feature2] + 1e-8)  # Avoid division by zero
            elif operation == 'add':
                feature_name = f'{feature1}_plus_{feature2}'
                df_interact[feature_name] = df_interact[feature1] + df_interact[feature2]
            elif operation == 'subtract':
                feature_name = f'{feature1}_minus_{feature2}'
                df_interact[feature_name] = df_interact[feature1] - df_interact[feature2]
            else:
                print(f"   [WARN] Unknown operation: {operation}")
                continue
            
            feature_count += 1
    
    print(f"   [OK] Created {feature_count} interaction features")
    print(f"   Feature pairs: {len(feature_pairs)}")
    print(f"   Operations: {operations}")
    print(f"   Dataset shape: {df_interact.shape}")
    
    return df_interact


# -------------------------------------------------------------------------
# Missing-value imputation: fit / apply separation (leakage-safe)
# -------------------------------------------------------------------------

def fit_missing_value_stats(
    train_df: pd.DataFrame,
    strategy: str = 'fill_mean',
) -> Dict[str, Any]:
    """Compute imputation statistics on **training data only**.

    The returned dict is then passed to :func:`apply_missing_value_fill`
    which can be called on *any* split (train, test, forecast-time).

    Args:
        train_df: Training split — the *only* data this function may see.
        strategy: ``'fill_mean'`` or ``'fill_median'``.

    Returns:
        Dict with ``{'strategy': ..., 'fill_values': {col: value}}``.
    """
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    if strategy == 'fill_mean':
        fill_values = {col: train_df[col].mean() for col in numeric_cols}
    elif strategy == 'fill_median':
        fill_values = {col: train_df[col].median() for col in numeric_cols}
    else:
        raise ValueError(f"strategy must be 'fill_mean' or 'fill_median', got '{strategy}'")

    print(f"[STATS] fit_missing_value_stats: computed {strategy} for {len(fill_values)} columns (train only)")
    return {'strategy': strategy, 'fill_values': fill_values}


def apply_missing_value_fill(
    df: pd.DataFrame,
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Apply imputation using pre-computed stats from :func:`fit_missing_value_stats`.

    Safe to call on train, test, or forecast-time data — no new statistics
    are computed from *df*.
    """
    df_filled = df.copy()
    fill_values = stats['fill_values']
    filled_count = 0
    for col, value in fill_values.items():
        if col in df_filled.columns:
            n_missing = df_filled[col].isnull().sum()
            if n_missing > 0:
                df_filled[col] = df_filled[col].fillna(value)
                filled_count += n_missing
    print(f"[FEATURE-APPLY] apply_missing_value_fill: filled {filled_count} values using train-derived stats")
    return df_filled


def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'drop',
                         threshold: float = 0.8,
                         date_col: str = None) -> pd.DataFrame:
    """Handle missing values in DataFrame.

    .. deprecated::
        For ``strategy='fill_mean'``, use :func:`fit_missing_value_stats`
        + :func:`apply_missing_value_fill` instead to avoid global-fit
        leakage.  The ``'drop'`` and ``'fill_forward'`` strategies remain
        safe because they don't compute dataset-wide statistics.
    """
    print(f"[DROP] HANDLING MISSING VALUES")
    print(f"   Strategy: {strategy}")
    
    if strategy == 'fill_mean':
        warnings.warn(
            "handle_missing_values(strategy='fill_mean') computes means on "
            "whatever DataFrame is passed in.  If this is the full dataset "
            "(pre-split), this leaks test information into training.  "
            "Use fit_missing_value_stats() + apply_missing_value_fill() instead.",
            FutureWarning,
            stacklevel=2,
        )
    
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    
    original_shape = df.shape
    missing_count = df.isnull().sum().sum()
    
    print(f"   Original shape: {original_shape}")
    print(f"   Missing values: {missing_count}")
    
    if missing_count == 0:
        print(f"   [OK] No missing values found")
        return df.copy()
    
    df_handled = df.copy()
    
    if strategy == 'drop':
        df_handled = df_handled.dropna().reset_index(drop=True)
    elif strategy == 'fill_forward':
        df_handled = df_handled.fillna(method='ffill')
    elif strategy == 'fill_mean':
        numeric_cols = df_handled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
    elif strategy == 'drop_columns':
        keep_cols = []
        for col in df_handled.columns:
            non_null_ratio = df_handled[col].count() / len(df_handled)
            if non_null_ratio >= threshold:
                keep_cols.append(col)
            else:
                print(f"   [DROP] Dropping column '{col}' (only {non_null_ratio:.1%} non-null)")
        df_handled = df_handled[keep_cols]
    
    final_shape = df_handled.shape
    final_missing = df_handled.isnull().sum().sum()
    
    print(f"   Final shape: {final_shape}")
    print(f"   Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"   Missing values remaining: {final_missing}")
    
    if date_col in df_handled.columns:
        print(f"   Date range: {df_handled[date_col].min()} to {df_handled[date_col].max()}")
    
    print(f"   [OK] Missing value handling completed")
    
    return df_handled


# -------------------------------------------------------------------------
# Feature quality validation: fit / filter separation (leakage-safe)
# -------------------------------------------------------------------------

def fit_feature_quality(
    train_df: pd.DataFrame,
    target_col: str = None,
    correlation_threshold: float = 0.05,
    variance_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Compute feature-quality stats on **training data only**.

    Returns a stats dict that can be passed to :func:`filter_features`
    to apply the same column filter to any split (train, test, forecast).

    Args:
        train_df: Training split — the *only* data this function may see.
        target_col: Target variable column name.
        correlation_threshold: Min |correlation| with target to keep.
        variance_threshold: Min variance to keep.

    Returns:
        Dict with ``{'feature_stats': ..., 'recommended_to_keep': [...],
        'recommended_to_drop': [...], 'thresholds': {...}}``.
    """
    target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]

    feature_stats = {}
    low_correlation = []
    low_variance = []
    high_missing = []

    for feature in feature_cols:
        correlation = train_df[feature].corr(train_df[target_col])
        variance = train_df[feature].var()
        missing_pct = train_df[feature].isnull().mean()

        feature_stats[feature] = {
            'correlation': correlation,
            'variance': variance,
            'missing_pct': missing_pct,
        }

        if abs(correlation) < correlation_threshold:
            low_correlation.append(feature)
        if variance < variance_threshold:
            low_variance.append(feature)
        if missing_pct > 0.1:
            high_missing.append(feature)

    recommended_to_drop = list(set(low_correlation + low_variance + high_missing))
    recommended_to_keep = [f for f in feature_cols if f not in recommended_to_drop]

    print(f"[STATS] fit_feature_quality: {len(feature_cols)} features analyzed (train only)")
    print(f"   Keep: {len(recommended_to_keep)}, Drop: {len(recommended_to_drop)}")

    return {
        'feature_stats': feature_stats,
        'recommended_to_keep': recommended_to_keep,
        'recommended_to_drop': recommended_to_drop,
        'thresholds': {
            'correlation': correlation_threshold,
            'variance': variance_threshold,
        },
    }


def filter_features(
    df: pd.DataFrame,
    quality_stats: Dict[str, Any],
    target_col: str = None,
    date_col: str = None,
) -> pd.DataFrame:
    """Drop low-quality features using stats from :func:`fit_feature_quality`.

    Safe to call on train, test, or forecast-time data — no new statistics
    are computed from *df*.
    """
    target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')

    keep = quality_stats['recommended_to_keep']
    # Always keep target and date columns
    essential = [c for c in [target_col, date_col] if c in df.columns]
    final_cols = essential + [c for c in keep if c in df.columns and c not in essential]

    df_filtered = df[final_cols].copy()
    print(f"[FEATURE-APPLY] filter_features: {df.shape[1]} → {df_filtered.shape[1]} columns")
    return df_filtered


def validate_feature_quality(df: pd.DataFrame,
                           target_col: str = None,
                           correlation_threshold: float = 0.05,
                           variance_threshold: float = 0.01) -> Dict[str, Any]:
    """Validate quality of engineered features.

    .. deprecated::
        This function computes correlation/variance on whatever DataFrame
        is passed in.  If called on the full dataset (pre-split), test
        information leaks into the feature-selection decision.  Use
        :func:`fit_feature_quality` + :func:`filter_features` instead.
    """
    warnings.warn(
        "validate_feature_quality() computes stats on whatever df is passed in. "
        "If this is the full dataset (pre-split), this leaks test information "
        "into feature selection. Use fit_feature_quality() + filter_features() "
        "instead.",
        FutureWarning,
        stacklevel=2,
    )

    # Delegate to the new fit function (caller's responsibility for split)
    stats = fit_feature_quality(df, target_col, correlation_threshold, variance_threshold)

    # Build legacy return format
    validation_results = {
        'total_features': len(stats['feature_stats']),
        'low_correlation': [f for f, s in stats['feature_stats'].items()
                           if abs(s['correlation']) < correlation_threshold],
        'low_variance': [f for f, s in stats['feature_stats'].items()
                        if s['variance'] < variance_threshold],
        'high_missing': [f for f, s in stats['feature_stats'].items()
                        if s['missing_pct'] > 0.1],
        'recommended_to_drop': stats['recommended_to_drop'],
        'feature_stats': stats['feature_stats'],
    }

    summary = f"""
[FEATURES] FEATURE QUALITY VALIDATION:
   * Total features: {validation_results['total_features']}
   * Low correlation (|r| < {correlation_threshold}): {len(validation_results['low_correlation'])}
   * Low variance (< {variance_threshold}): {len(validation_results['low_variance'])}
   * High missing (> 10%): {len(validation_results['high_missing'])}
   * Recommended to drop: {len(validation_results['recommended_to_drop'])}
   * Good quality features: {validation_results['total_features'] - len(validation_results['recommended_to_drop'])}
"""
    validation_results['summary'] = summary
    print(summary)
    print(f"   [OK] Feature validation completed")

    return validation_results


def create_mstl_features(df: pd.DataFrame,
                        mstl_results: Dict[str, Any],
                        date_col: str = None,
                        lag_periods: Optional[List[int]] = None,
                        rolling_windows: Optional[List[int]] = None,
                        rolling_stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add MSTL decomposition features and optionally create lag/rolling features
    
    Args:
        df: Original DataFrame
        mstl_results: Results from MSTL decomposition
        date_col: Date column name for merging
        lag_periods: List of lag periods to create (e.g., [1, 2, 3, 7])
        rolling_windows: List of rolling windows (e.g., [7, 14, 30])
        rolling_stats: List of rolling statistics (e.g., ['mean', 'std', 'sum'])
        
    Returns:
        DataFrame with MSTL features (and optional lag/rolling features)
        
    Example:
        # Basic MSTL features only
        df_mstl = create_mstl_features(df, mstl_results)
        
        # MSTL + lag features
        df_mstl = create_mstl_features(df, mstl_results, lag_periods=[1, 2, 3, 7])
        
        # MSTL + rolling features  
        df_mstl = create_mstl_features(df, mstl_results, 
                                      rolling_windows=[7, 14], 
                                      rolling_stats=['mean', 'sum'])
        
        # MSTL + both lag and rolling
        df_mstl = create_mstl_features(df, mstl_results, 
                                      lag_periods=[1, 3, 7],
                                      rolling_windows=[7, 30],
                                      rolling_stats=['mean', 'std'])
    """
    print(f"[FEATURES] ADDING MSTL FEATURES")
    
    # Check if MSTL was successful
    if not mstl_results.get('success', False):
        print(f"   [ERROR] MSTL decomposition failed, cannot add features")
        return df.copy()
    
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    df_result = df.copy()
    
    # Get MSTL components
    trend = mstl_results['trend']
    seasonal = mstl_results['seasonal']
    residual = mstl_results['resid']
    
    print(f"   Components shape: {len(trend)} observations")
    print(f"   Seasonal shape: {seasonal.shape}")
    
    # Create MSTL features dictionary
    mstl_features = {
        'MSTL_Trend': trend,
        'MSTL_Residual': residual
    }
    
    # Add seasonal components dynamically
    n_seasonal = seasonal.shape[1] if len(seasonal.shape) > 1 else 1
    
    if n_seasonal == 1:
        mstl_features['MSTL_Seasonal'] = seasonal
    else:
        for i in range(n_seasonal):
            feature_name = f'MSTL_Seasonal_{i+1}'
            mstl_features[feature_name] = seasonal.iloc[:, i]
    
    # Create DataFrame with MSTL components
    mstl_df = pd.DataFrame(mstl_features, index=trend.index)
    
    # Merge with original DataFrame
    if date_col in df_result.columns:
        df_result = df_result.set_index(date_col)
        df_result = df_result.join(mstl_df, how='left')
        df_result = df_result.reset_index()
    else:
        df_result = df_result.join(mstl_df, how='left')
    
    # Report MSTL features added
    mstl_feature_names = list(mstl_features.keys())
    print(f"   [OK] Added {len(mstl_feature_names)} MSTL features:")
    for feat in mstl_feature_names:
        if feat in df_result.columns:
            missing_count = df_result[feat].isnull().sum()
            print(f"      * {feat}: {missing_count} missing values")
    
    # Create lag features if requested
    if lag_periods is not None:
        print(f"\n   [TEMPORAL] Creating lag features for MSTL components...")
        df_result = create_lag_features(df_result, mstl_feature_names, lag_periods)
    
    # Create rolling features if requested
    if rolling_windows is not None:
        rolling_stats = rolling_stats or ['mean']  # Default to mean if not specified
        print(f"\n   [ROLLING] Creating rolling features for MSTL components...")
        df_result = create_rolling_features(df_result, mstl_feature_names, rolling_windows, rolling_stats)
    
    print(f"\n   [FEATURES] Final shape: {df_result.shape}")
    return df_result


# =============================================================================
# CONVENIENCE FUNCTION FOR COMPLETE WORKFLOW
# =============================================================================

def apply_feature_engineering_steps(df: pd.DataFrame,
                                   steps: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply multiple feature engineering steps in sequence
    
    Args:
        df: Input DataFrame
        steps: List of step dictionaries with 'function' and 'params'
        
    Returns:
        DataFrame after applying all steps
        
    Example:
        steps = [
            {'function': 'select_features', 'params': {'features_to_keep': strong_features}},
            {'function': 'create_temporal_features', 'params': {'features_to_create': ['Month_sin', 'Month_cos']}},
            {'function': 'create_lag_features', 'params': {'columns_to_lag': 'Lượng mưa', 'lags': [1, 2, 3]}},
            {'function': 'handle_missing_values', 'params': {'strategy': 'drop'}}
        ]
        df_final = apply_feature_engineering_steps(df, steps)
    """
    print(f"[TRANSFORM] APPLYING FEATURE ENGINEERING STEPS")
    print(f"   Number of steps: {len(steps)}")
    
    df_result = df.copy()
    
    # Available functions
    available_functions = {
        'select_features': select_features,
        'create_temporal_features': create_temporal_features,
        'create_lag_features': create_lag_features,
        'create_rolling_features': create_rolling_features,
        'create_interaction_features': create_interaction_features,
        'handle_missing_values': handle_missing_values,
        'create_mstl_features': create_mstl_features
    }
    
    for i, step in enumerate(steps):
        function_name = step['function']
        params = step.get('params', {})
        
        print(f"\n   Step {i+1}: {function_name}")
        
        if function_name in available_functions:
            function = available_functions[function_name]
            df_result = function(df_result, **params)
        else:
            print(f"   [WARN] Unknown function: {function_name}")
    
    print(f"\n   [OK] All steps completed")
    print(f"   Final shape: {df_result.shape}")
    
    return df_result 