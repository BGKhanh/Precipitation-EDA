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
    print(f"📊 SELECTING FEATURES")
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
            print(f"   ⚠️ Feature '{feature}' not found in DataFrame")
    
    df_selected = df[final_cols].copy()
    
    print(f"   Original shape: {df.shape}")
    print(f"   Selected shape: {df_selected.shape}")
    print(f"   ✅ Feature selection completed")
    
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
    print(f"🕒 CREATING TEMPORAL FEATURES")
    
    # Default parameters
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    wet_season_months = wet_season_months or [5, 6, 7, 8, 9, 10, 11]  # May-November for HCMC
    
    if features_to_create is None:
        features_to_create = ['Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Is_Wet_Season']
    
    df_temporal = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_temporal[date_col]):
        df_temporal[date_col] = pd.to_datetime(df_temporal[date_col])
        print(f"   ✅ Converted {date_col} to datetime")
    
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
            print(f"   ⚠️ Unknown temporal feature: {feature}")
    
    # Clean up temporary columns
    temp_cols = [col for col in df_temporal.columns if col.startswith('_temp_')]
    df_temporal = df_temporal.drop(columns=temp_cols)
    
    print(f"   ✅ Created {feature_count} temporal features")
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
    print(f"⏰ CREATING LAG FEATURES")
    
    if isinstance(columns_to_lag, str):
        columns_to_lag = [columns_to_lag]
    
    df_lag = df.copy()
    feature_count = 0
    
    for column in columns_to_lag:
        if column not in df_lag.columns:
            print(f"   ⚠️ Column '{column}' not found")
            continue
        
        for lag in lags:
            feature_name = f'{column}_{suffix}_{lag}'
            df_lag[feature_name] = df_lag[column].shift(lag)
            feature_count += 1
    
    print(f"   ✅ Created {feature_count} lag features")
    print(f"   Columns lagged: {columns_to_lag}")
    print(f"   Lag periods: {lags}")
    print(f"   Dataset shape: {df_lag.shape}")
    
    return df_lag


def create_rolling_features(df: pd.DataFrame,
                          columns_to_roll: Union[str, List[str]],
                          windows: List[int],
                          stats: List[str] = ['mean', 'std', 'min', 'max', 'sum']) -> pd.DataFrame:
    """
    Create rolling window features for specified columns
    
    Args:
        df: Input DataFrame
        columns_to_roll: Column name(s) to create rolling features for
        windows: List of window sizes
        stats: List of statistics to calculate
        
    Returns:
        DataFrame with rolling features added
        
    Example:
        # User decides based on temporal analysis
        df_rolling = create_rolling_features(df, 'Lượng mưa', [7, 14, 30], ['sum', 'mean'])
        df_rolling = create_rolling_features(df_rolling, ['Nhiệt độ tối đa 2m'], [7], ['mean', 'std'])
    """
    print(f"🪟 CREATING ROLLING FEATURES")
    
    if isinstance(columns_to_roll, str):
        columns_to_roll = [columns_to_roll]
    
    df_rolling = df.copy()
    feature_count = 0
    
    for column in columns_to_roll:
        if column not in df_rolling.columns:
            print(f"   ⚠️ Column '{column}' not found")
            continue
        
        for window in windows:
            for stat in stats:
                feature_name = f'{column}_{stat}_{window}d'
                
                if stat == 'mean':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).mean()
                elif stat == 'std':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).std()
                elif stat == 'min':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).min()
                elif stat == 'max':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).max()
                elif stat == 'sum':
                    df_rolling[feature_name] = df_rolling[column].rolling(window=window).sum()
                else:
                    print(f"   ⚠️ Unknown statistic: {stat}")
                    continue
                
                feature_count += 1
    
    print(f"   ✅ Created {feature_count} rolling features")
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
    print(f"🔗 CREATING INTERACTION FEATURES")
    
    df_interact = df.copy()
    feature_count = 0
    
    for feature1, feature2 in feature_pairs:
        if feature1 not in df_interact.columns or feature2 not in df_interact.columns:
            print(f"   ⚠️ Feature pair ({feature1}, {feature2}) not found")
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
                print(f"   ⚠️ Unknown operation: {operation}")
                continue
            
            feature_count += 1
    
    print(f"   ✅ Created {feature_count} interaction features")
    print(f"   Feature pairs: {len(feature_pairs)}")
    print(f"   Operations: {operations}")
    print(f"   Dataset shape: {df_interact.shape}")
    
    return df_interact


def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'drop',
                         threshold: float = 0.8,
                         date_col: str = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'fill_forward', 'fill_mean')
        threshold: Threshold for dropping columns (fraction of non-null values required)
        date_col: Date column name for reporting
        
    Returns:
        DataFrame with missing values handled
        
    Example:
        # User decides how to handle missing values
        df_clean = handle_missing_values(df, strategy='drop')
    """
    print(f"🗑️ HANDLING MISSING VALUES")
    print(f"   Strategy: {strategy}")
    
    date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
    
    original_shape = df.shape
    missing_count = df.isnull().sum().sum()
    
    print(f"   Original shape: {original_shape}")
    print(f"   Missing values: {missing_count}")
    
    if missing_count == 0:
        print(f"   ✅ No missing values found")
        return df.copy()
    
    df_handled = df.copy()
    
    if strategy == 'drop':
        # Drop rows with any missing values
        df_handled = df_handled.dropna().reset_index(drop=True)
        
    elif strategy == 'fill_forward':
        # Forward fill missing values
        df_handled = df_handled.fillna(method='ffill')
        
    elif strategy == 'fill_mean':
        # Fill with column means (for numeric columns only)
        numeric_cols = df_handled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
    
    elif strategy == 'drop_columns':
        # Drop columns with too many missing values
        keep_cols = []
        for col in df_handled.columns:
            non_null_ratio = df_handled[col].count() / len(df_handled)
            if non_null_ratio >= threshold:
                keep_cols.append(col)
            else:
                print(f"   🗑️ Dropping column '{col}' (only {non_null_ratio:.1%} non-null)")
        
        df_handled = df_handled[keep_cols]
    
    final_shape = df_handled.shape
    final_missing = df_handled.isnull().sum().sum()
    
    print(f"   Final shape: {final_shape}")
    print(f"   Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"   Missing values remaining: {final_missing}")
    
    # Report date range if date column exists
    if date_col in df_handled.columns:
        print(f"   Date range: {df_handled[date_col].min()} to {df_handled[date_col].max()}")
    
    print(f"   ✅ Missing value handling completed")
    
    return df_handled


def validate_feature_quality(df: pd.DataFrame,
                           target_col: str = None,
                           correlation_threshold: float = 0.05,
                           variance_threshold: float = 0.01) -> Dict[str, Any]:
    """
    Validate quality of engineered features
    
    Args:
        df: DataFrame with features
        target_col: Target variable column name
        correlation_threshold: Minimum correlation with target to keep feature
        variance_threshold: Minimum variance to keep feature
        
    Returns:
        Dictionary with validation results and recommendations
        
    Example:
        # User validates engineered features
        validation = validate_feature_quality(df_engineered)
        print(validation['summary'])
    """
    print(f"✅ VALIDATING FEATURE QUALITY")
    
    target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Get numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    validation_results = {
        'total_features': len(feature_cols),
        'low_correlation': [],
        'low_variance': [],
        'high_missing': [],
        'recommended_to_drop': [],
        'feature_stats': {}
    }
    
    print(f"   Analyzing {len(feature_cols)} features...")
    
    for feature in feature_cols:
        stats = {}
        
        # Calculate correlation with target
        correlation = df[feature].corr(df[target_col])
        stats['correlation'] = correlation
        
        # Calculate variance
        variance = df[feature].var()
        stats['variance'] = variance
        
        # Calculate missing percentage
        missing_pct = df[feature].isnull().mean()
        stats['missing_pct'] = missing_pct
        
        validation_results['feature_stats'][feature] = stats
        
        # Flag low correlation features
        if abs(correlation) < correlation_threshold:
            validation_results['low_correlation'].append(feature)
        
        # Flag low variance features
        if variance < variance_threshold:
            validation_results['low_variance'].append(feature)
        
        # Flag high missing features
        if missing_pct > 0.1:  # More than 10% missing
            validation_results['high_missing'].append(feature)
    
    # Combine recommendations
    all_flagged = set(validation_results['low_correlation'] + 
                     validation_results['low_variance'] + 
                     validation_results['high_missing'])
    validation_results['recommended_to_drop'] = list(all_flagged)
    
    # Create summary
    summary = f"""
📊 FEATURE QUALITY VALIDATION:
   • Total features: {validation_results['total_features']}
   • Low correlation (|r| < {correlation_threshold}): {len(validation_results['low_correlation'])}
   • Low variance (< {variance_threshold}): {len(validation_results['low_variance'])}
   • High missing (> 10%): {len(validation_results['high_missing'])}
   • Recommended to drop: {len(validation_results['recommended_to_drop'])}
   • Good quality features: {validation_results['total_features'] - len(validation_results['recommended_to_drop'])}
"""
    
    validation_results['summary'] = summary
    
    print(summary)
    print(f"   ✅ Feature validation completed")
    
    return validation_results


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
    print(f"🔄 APPLYING FEATURE ENGINEERING STEPS")
    print(f"   Number of steps: {len(steps)}")
    
    df_result = df.copy()
    
    # Available functions
    available_functions = {
        'select_features': select_features,
        'create_temporal_features': create_temporal_features,
        'create_lag_features': create_lag_features,
        'create_rolling_features': create_rolling_features,
        'create_interaction_features': create_interaction_features,
        'handle_missing_values': handle_missing_values
    }
    
    for i, step in enumerate(steps):
        function_name = step['function']
        params = step.get('params', {})
        
        print(f"\n   Step {i+1}: {function_name}")
        
        if function_name in available_functions:
            function = available_functions[function_name]
            df_result = function(df_result, **params)
        else:
            print(f"   ⚠️ Unknown function: {function_name}")
    
    print(f"\n   ✅ All steps completed")
    print(f"   Final shape: {df_result.shape}")
    
    return df_result 