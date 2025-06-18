"""# Feature Engineering

## Feature Selection
"""

selected_features = [
    'Nhiệt độ tối đa 2m',      # T2M_MAX
    'Nhiệt độ tối thiểu 2m',    # T2M_MIN
    'Độ ẩm tương đối 2m',       # RH2M
    'Độ ẩm đất bề mặt',         # GWETTOP
    'Hướng gió 10m',            # WD10M
    'Áp suất bề mặt',           # PS
    'Bức xạ sóng dài xuống'     # ALLSKY_SFC_LW_DWN
]

# Keep essential columns (date, target) + selected features
essential_columns = ['Ngày', 'Lượng mưa']  # Date and target variable
final_columns = essential_columns + selected_features

# Filter dataset
df_selected = df_all[final_columns].copy()

print(f"📊 Feature Selection completed:")
print(f"   Original shape: {df_all.shape}")
print(f"   Selected shape: {df_selected.shape}")
print(f"   Features selected: {len(selected_features)}")
print(f"   Selected features: {selected_features}")

"""## Temporal feature"""

import numpy as np
import pandas as pd

# Convert date column to datetime if not already
df_selected['Ngày'] = pd.to_datetime(df_selected['Ngày'])

# Extract basic time features
df_selected['Year'] = df_selected['Ngày'].dt.year
df_selected['Month'] = df_selected['Ngày'].dt.month
df_selected['DayofMonth'] = df_selected['Ngày'].dt.day
df_selected['DayofYear'] = df_selected['Ngày'].dt.dayofyear

# Create cyclical features for Month (1-12)
df_selected['Month_sin'] = np.sin(2 * np.pi * df_selected['Month'] / 12)
df_selected['Month_cos'] = np.cos(2 * np.pi * df_selected['Month'] / 12)

# Create cyclical features for Day of Year (1-365/366)
df_selected['DayOfYear_sin'] = np.sin(2 * np.pi * df_selected['DayofYear'] / 365.25)
df_selected['DayOfYear_cos'] = np.cos(2 * np.pi * df_selected['DayofYear'] / 365.25)

# Create Wet Season feature (May-November = months 5,6,7,8,9,10,11)
df_selected['Is_Wet_Season'] = df_selected['Month'].isin([5, 6, 7, 8, 9, 10, 11]).astype(int)

print("🕒 Time-based Feature Engineering completed:")
print(f"   Dataset shape: {df_selected.shape}")
print(f"   New time features added: 8")
print(f"   Wet season months (value=1): May-November")
print(f"   Dry season months (value=0): December-April")

# Display sample of new features
print("\n📊 Sample of new time features:")
print(df_selected[['Ngày', 'Year', 'Month', 'DayofMonth', 'DayofYear',
                   'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                   'Is_Wet_Season']].head(10))

# Check wet season distribution
print(f"\n🌧️ Wet Season Distribution:")
print(df_selected['Is_Wet_Season'].value_counts())

"""## Lag and Rolling window Engineering

"""

# ==============================================================================
# LAG FEATURES & ROLLING WINDOW FEATURES
# ==============================================================================

# A. TARGET VARIABLE LAG & ROLLING FEATURES (HIGHEST PRIORITY)
print("🎯 Creating Target Variable Features (Lượng mưa)...")

# Lag features for rainfall (1, 2, 3, 4, 5, 7 days)
for lag in [1, 2, 3, 4, 5, 7]:
    df_selected[f'Rainfall_lag_{lag}'] = df_selected['Lượng mưa'].shift(lag)

# Rolling sum features (cumulative rainfall)
for window in [3, 7, 14, 30]:
    df_selected[f'Rainfall_sum_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).sum()

# Rolling statistical features
for window in [7, 14]:
    df_selected[f'Rainfall_mean_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).mean()
    df_selected[f'Rainfall_max_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).max()
    df_selected[f'Rainfall_std_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).std()

# B. IMPORTANT PREDICTOR VARIABLES LAG & ROLLING FEATURES
print("🌡️ Creating Temperature Features...")

# Temperature features (T2M_MAX, T2M_MIN)
temp_vars = ['Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m']
for var in temp_vars:
    # Lag 1
    df_selected[f'{var}_lag_1'] = df_selected[var].shift(1)

    # Rolling statistics
    for window in [3, 7, 14]:
        df_selected[f'{var}_mean_{window}d'] = df_selected[var].rolling(window=window).mean()
        df_selected[f'{var}_max_{window}d'] = df_selected[var].rolling(window=window).max()
        df_selected[f'{var}_min_{window}d'] = df_selected[var].rolling(window=window).min()
        df_selected[f'{var}_std_{window}d'] = df_selected[var].rolling(window=window).std()

print("💧 Creating Humidity Features...")

# Humidity features (RH2M, GWETTOP)
humidity_vars = ['Độ ẩm tương đối 2m', 'Độ ẩm đất bề mặt']
for var in humidity_vars:
    # Lag 1
    df_selected[f'{var}_lag_1'] = df_selected[var].shift(1)

    # Rolling statistics
    for window in [3, 7, 14]:
        df_selected[f'{var}_mean_{window}d'] = df_selected[var].rolling(window=window).mean()
        df_selected[f'{var}_std_{window}d'] = df_selected[var].rolling(window=window).std()

print("✅ Lag & Rolling Window Features completed:")
print(f"   Final dataset shape: {df_selected.shape}")
print(f"   Target lag features: 6")
print(f"   Target rolling features: 10")
print(f"   Temperature features: 24")
print(f"   Humidity features: 12")
print(f"   Total new features: 52")

# Check for missing values (expected due to lag/rolling)
missing_count = df_selected.isnull().sum().sum()
print(f"   Missing values created: {missing_count} (due to lag/rolling operations)")

# ==============================================================================
# CHECK MISSING VALUES BY DATE
# ==============================================================================

# Find rows with any missing values
missing_rows = df_selected.isnull().any(axis=1)
missing_dates = df_selected[missing_rows]['Ngày']

print("📅 Dates with Missing Values:")
print(f"   Total dates with missing values: {len(missing_dates)}")
print(f"   Date range: {missing_dates.min()} to {missing_dates.max()}")

print("\n🗓️ First 20 dates with missing values:")
print(missing_dates.head(20).tolist())

if len(missing_dates) > 20:
    print(f"\n... and {len(missing_dates) - 20} more dates")

# Check which features have the most missing values
print(f"\n🔍 Missing values by feature (top 10):")
missing_by_feature = df_selected.isnull().sum().sort_values(ascending=False)
print(missing_by_feature[missing_by_feature > 0].head(10))

# ==============================================================================
# DROP MISSING VALUES
# ==============================================================================

# Store original shape for comparison
original_shape = df_selected.shape

# Drop rows with any missing values
df_selected = df_selected.dropna()

# Reset index after dropping rows
df_selected = df_selected.reset_index(drop=True)

print("🗑️ Missing Values Removal completed:")
print(f"   Original shape: {original_shape}")
print(f"   Final shape: {df_selected.shape}")
print(f"   Rows dropped: {original_shape[0] - df_selected.shape[0]}")
print(f"   Remaining data: {df_selected.shape[0]} days")
print(f"   Missing values remaining: {df_selected.isnull().sum().sum()}")

# Check date range of final dataset
print(f"\n📅 Final dataset date range:")
print(f"   Start date: {df_selected['Ngày'].min()}")
print(f"   End date: {df_selected['Ngày'].max()}")

"""## Multicolinear and cross-correlation checking for new feature"""

# ==============================================================================
# MULTICOLLINEARITY & CROSS-CORRELATION CHECKING (4 PARTS)
# ==============================================================================

# Define feature groups
print("🔍 Defining Feature Groups for Correlation Analysis...")

# 1. Original selected features (after feature selection)
original_features = ['Ngày', 'Lượng mưa'] + [
    'Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m', 'Độ ẩm tương đối 2m',
    'Độ ẩm đất bề mặt', 'Hướng gió 10m', 'Áp suất bề mặt', 'Bức xạ sóng dài xuống'
]

# 2. Temporal features
temporal_features = ['Ngày', 'Lượng mưa'] + [
    'Year', 'Month', 'DayofMonth', 'DayofYear',
    'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Is_Wet_Season'
]

# 3. Lag features
lag_columns = ['Ngày', 'Lượng mưa'] + [col for col in df_selected.columns if '_lag_' in col]

# 4. Rolling window features
rolling_columns = ['Ngày', 'Lượng mưa'] + [col for col in df_selected.columns
                   if any(pattern in col for pattern in ['_sum_', '_mean_', '_max_', '_std_'])
                   and col.endswith('d')]

print(f"📊 Feature Groups Summary:")
print(f"   Original features: {len(original_features)-2}")
print(f"   Temporal features: {len(temporal_features)-2}")
print(f"   Lag features: {len(lag_columns)-2}")
print(f"   Rolling features: {len(rolling_columns)-2}")

# Part 1: Original Features Analysis
print("\n" + "="*50)
print("PART 1: ORIGINAL FEATURES CORRELATION ANALYSIS")
print("="*50)
df_original = df_selected[original_features].copy()
original_results = run_city_level_correlation_analysis(
    df_original,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 2: Temporal Features Analysis
print("\n" + "="*50)
print("PART 2: TEMPORAL FEATURES CORRELATION ANALYSIS")
print("="*50)
df_temporal = df_selected[temporal_features].copy()
temporal_results = run_city_level_correlation_analysis(
    df_temporal,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 3: Lag Features Analysis
print("\n" + "="*50)
print("PART 3: LAG FEATURES CORRELATION ANALYSIS")
print("="*50)
df_lag = df_selected[lag_columns].copy()
lag_results = run_city_level_correlation_analysis(
    df_lag,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 4: Rolling Window Features Analysis
print("\n" + "="*50)
print("PART 4: ROLLING WINDOW FEATURES CORRELATION ANALYSIS")
print("="*50)
df_rolling = df_selected[rolling_columns].copy()
rolling_results = run_city_level_correlation_analysis(
    df_rolling,
    target_col='Lượng mưa',
    date_col='Ngày'
)

print("\n✅ All correlation analyses completed!")

"""# Modeling

## Tree-based model
"""

!pip install -q optuna

# ==============================================================================
# LIGHTGBM TWO-STAGE MODEL WITH OPTUNA OPTIMIZATION (FIXED)
# ==============================================================================

import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

print("🚀 LIGHTGBM TWO-STAGE MODEL EXPERIMENTATION (FIXED)")
print("="*60)

print(f"📊 Data Setup:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Total samples: {len(X)}")
print(f"   Rain days: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
print(f"   No rain days: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")

# Fixed Optuna optimization functions
def objective_classification_fixed(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1
    }

    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=100)  # Removed verbose_eval

        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def objective_regression_fixed(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1
    }

    mae_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        rain_mask_train = y_clf.iloc[train_idx] == 1
        rain_mask_test = y_clf.iloc[test_idx] == 1

        if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
            X_train_rain = X_train[rain_mask_train]
            y_train_rain = y_train[rain_mask_train]
            X_test_rain = X_test[rain_mask_test]
            y_test_rain = y_test[rain_mask_test]

            train_data = lgb.Dataset(X_train_rain, label=y_train_rain)
            model = lgb.train(params, train_data, num_boost_round=100)  # Removed verbose_eval

            y_pred = model.predict(X_test_rain)
            mae = mean_absolute_error(y_test_rain, y_pred)
            mae_scores.append(mae)

    return np.mean(mae_scores) if mae_scores else float('inf')

# Optimize Classification Model
print("\n🔍 Optimizing Classification Model...")
study_clf_fixed = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_clf_fixed.optimize(objective_classification_fixed, n_trials=20)
best_params_clf_fixed = study_clf_fixed.best_params
print(f"   Best AUC: {study_clf_fixed.best_value:.4f}")

# Optimize Regression Model
print("\n🔍 Optimizing Regression Model...")
study_reg_fixed = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_reg_fixed.optimize(objective_regression_fixed, n_trials=20)
best_params_reg_fixed = study_reg_fixed.best_params
print(f"   Best MAE: {study_reg_fixed.best_value:.4f}")

# Final evaluation with best parameters
print("\n📊 Final Evaluation with Best Parameters:")
clf_results_fixed = {'AUC': []}
reg_results_fixed = {'MAE': [], 'RMSE': [], 'R2': []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n🔄 Fold {fold + 1}/3")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # Classification Model
    best_params_clf_fixed.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})
    train_data_clf = lgb.Dataset(X_train, label=y_train_clf)
    clf_model = lgb.train(best_params_clf_fixed, train_data_clf, num_boost_round=100)

    clf_pred = clf_model.predict(X_test)
    auc = roc_auc_score(y_test_clf, clf_pred)
    clf_results_fixed['AUC'].append(auc)

    # Regression Model
    rain_mask_train = y_train_clf == 1
    rain_mask_test = y_test_clf == 1

    if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
        X_train_rain = X_train[rain_mask_train]
        y_train_rain = y_train_reg[rain_mask_train]
        X_test_rain = X_test[rain_mask_test]
        y_test_rain = y_test_reg[rain_mask_test]

        best_params_reg_fixed.update({'objective': 'regression', 'metric': 'mae', 'verbosity': -1})
        train_data_reg = lgb.Dataset(X_train_rain, label=y_train_rain)
        reg_model = lgb.train(best_params_reg_fixed, train_data_reg, num_boost_round=100)

        reg_pred = reg_model.predict(X_test_rain)
        mae = mean_absolute_error(y_test_rain, reg_pred)
        rmse = np.sqrt(mean_squared_error(y_test_rain, reg_pred))
        r2 = r2_score(y_test_rain, reg_pred)

        reg_results_fixed['MAE'].append(mae)
        reg_results_fixed['RMSE'].append(rmse)
        reg_results_fixed['R2'].append(r2)

    print(f"   Classification AUC: {auc:.4f}")
    if rain_mask_test.sum() > 0:
        print(f"   Regression MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Summary Results
print("\n" + "="*60)
print("📋 LIGHTGBM FINAL RESULTS SUMMARY (FIXED)")
print("="*60)
print(f"Classification Model:")
print(f"   AUC: {np.mean(clf_results_fixed['AUC']):.4f} ± {np.std(clf_results_fixed['AUC']):.4f}")

if reg_results_fixed['MAE']:
    print(f"\nRegression Model (on rainy days only):")
    print(f"   MAE:  {np.mean(reg_results_fixed['MAE']):.4f} ± {np.std(reg_results_fixed['MAE']):.4f}")
    print(f"   RMSE: {np.mean(reg_results_fixed['RMSE']):.4f} ± {np.std(reg_results_fixed['RMSE']):.4f}")
    print(f"   R2:   {np.mean(reg_results_fixed['R2']):.4f} ± {np.std(reg_results_fixed['R2']):.4f}")

# ==============================================================================
# XGBOOST TWO-STAGE MODEL WITH OPTUNA OPTIMIZATION
# ==============================================================================

import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

print("🚀 XGBOOST TWO-STAGE MODEL EXPERIMENTATION")
print("="*60)

# Use same data setup as LightGBM
print(f"📊 Data Setup:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Total samples: {len(X)}")
print(f"   Rain days: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
print(f"   No rain days: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")

# Optuna optimization functions for XGBoost
def objective_classification_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'verbosity': 0,
        'random_state': 42
    }

    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def objective_regression_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'verbosity': 0,
        'random_state': 42
    }

    mae_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        # Only use rainy days for regression training
        rain_mask_train = y_clf.iloc[train_idx] == 1
        rain_mask_test = y_clf.iloc[test_idx] == 1

        if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
            X_train_rain = X_train[rain_mask_train]
            y_train_rain = y_train[rain_mask_train]
            X_test_rain = X_test[rain_mask_test]
            y_test_rain = y_test[rain_mask_test]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_rain, y_train_rain, verbose=False)

            y_pred = model.predict(X_test_rain)
            mae = mean_absolute_error(y_test_rain, y_pred)
            mae_scores.append(mae)

    return np.mean(mae_scores) if mae_scores else float('inf')

# Optimize Classification Model
print("\n🔍 Optimizing XGBoost Classification Model...")
study_clf_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_clf_xgb.optimize(objective_classification_xgb, n_trials=20)
best_params_clf_xgb = study_clf_xgb.best_params
print(f"   Best AUC: {study_clf_xgb.best_value:.4f}")

# Optimize Regression Model
print("\n🔍 Optimizing XGBoost Regression Model...")
study_reg_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_reg_xgb.optimize(objective_regression_xgb, n_trials=20)
best_params_reg_xgb = study_reg_xgb.best_params
print(f"   Best MAE: {study_reg_xgb.best_value:.4f}")

# Final evaluation with best parameters
print("\n📊 Final Evaluation with Best Parameters:")
clf_results_xgb = {'AUC': []}
reg_results_xgb = {'MAE': [], 'RMSE': [], 'R2': []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n🔄 Fold {fold + 1}/3")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # Classification Model
    best_params_clf_xgb.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'random_state': 42
    })
    clf_model_xgb = xgb.XGBClassifier(**best_params_clf_xgb)
    clf_model_xgb.fit(X_train, y_train_clf, verbose=False)

    # Classification predictions
    clf_pred_xgb = clf_model_xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_clf, clf_pred_xgb)
    clf_results_xgb['AUC'].append(auc)

    # Regression Model (only on rainy days)
    rain_mask_train = y_train_clf == 1
    rain_mask_test = y_test_clf == 1

    if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
        X_train_rain = X_train[rain_mask_train]
        y_train_rain = y_train_reg[rain_mask_train]
        X_test_rain = X_test[rain_mask_test]
        y_test_rain = y_test_reg[rain_mask_test]

        best_params_reg_xgb.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
            'random_state': 42
        })
        reg_model_xgb = xgb.XGBRegressor(**best_params_reg_xgb)
        reg_model_xgb.fit(X_train_rain, y_train_rain, verbose=False)

        # Regression predictions
        reg_pred_xgb = reg_model_xgb.predict(X_test_rain)
        mae = mean_absolute_error(y_test_rain, reg_pred_xgb)
        rmse = np.sqrt(mean_squared_error(y_test_rain, reg_pred_xgb))
        r2 = r2_score(y_test_rain, reg_pred_xgb)

        reg_results_xgb['MAE'].append(mae)
        reg_results_xgb['RMSE'].append(rmse)
        reg_results_xgb['R2'].append(r2)

    print(f"   Classification AUC: {auc:.4f}")
    if rain_mask_test.sum() > 0:
        print(f"   Regression MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Summary Results
print("\n" + "="*60)
print("📋 XGBOOST FINAL RESULTS SUMMARY")
print("="*60)
print(f"Classification Model:")
print(f"   AUC: {np.mean(clf_results_xgb['AUC']):.4f} ± {np.std(clf_results_xgb['AUC']):.4f}")

if reg_results_xgb['MAE']:
    print(f"\nRegression Model (on rainy days only):")
    print(f"   MAE:  {np.mean(reg_results_xgb['MAE']):.4f} ± {np.std(reg_results_xgb['MAE']):.4f}")
    print(f"   RMSE: {np.mean(reg_results_xgb['RMSE']):.4f} ± {np.std(reg_results_xgb['RMSE']):.4f}")
    print(f"   R2:   {np.mean(reg_results_xgb['R2']):.4f} ± {np.std(reg_results_xgb['R2']):.4f}")

"""## Times series model"""

# ==============================================================================
# STATIONARITY TESTING FOR EXOGENOUS FEATURES (ADF & KPSS)
# ==============================================================================

from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Define exogenous features (original selected features)
exogenous_features = [
    'Nhiệt độ tối đa 2m',      # T2M_MAX
    'Nhiệt độ tối thiểu 2m',    # T2M_MIN
    'Độ ẩm tương đối 2m',       # RH2M
    'Độ ẩm đất bề mặt',         # GWETTOP
    'Hướng gió 10m',            # WD10M
    'Áp suất bề mặt',           # PS
    'Bức xạ sóng dài xuống'     # ALLSKY_SFC_LW_DWN
]

print("📊 STATIONARITY TESTING FOR EXOGENOUS FEATURES")
print("="*60)

stationarity_results = {}

for feature in exogenous_features:
    print(f"\n🔍 Testing: {feature}")

    # ADF Test
    adf_result = adfuller(df_selected[feature].dropna())
    adf_stationary = adf_result[1] < 0.05

    # KPSS Test
    kpss_result = kpss(df_selected[feature].dropna())
    kpss_stationary = kpss_result[1] > 0.05

    # Overall conclusion
    if adf_stationary and kpss_stationary:
        conclusion = "STATIONARY"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "NON-STATIONARY"
    else:
        conclusion = "INCONCLUSIVE"

    # Store results
    stationarity_results[feature] = {
        'ADF_pvalue': adf_result[1],
        'ADF_stationary': adf_stationary,
        'KPSS_pvalue': kpss_result[1],
        'KPSS_stationary': kpss_stationary,
        'Conclusion': conclusion
    }

    print(f"   ADF p-value: {adf_result[1]:.6f} ({'Stationary' if adf_stationary else 'Non-stationary'})")
    print(f"   KPSS p-value: {kpss_result[1]:.6f} ({'Stationary' if kpss_stationary else 'Non-stationary'})")
    print(f"   ➤ Conclusion: {conclusion}")

# Summary
print("\n" + "="*60)
print("📋 STATIONARITY SUMMARY:")
stationary_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'STATIONARY')
non_stationary_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'NON-STATIONARY')
inconclusive_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'INCONCLUSIVE')

print(f"   Stationary features: {stationary_count}")
print(f"   Non-stationary features: {non_stationary_count}")
print(f"   Inconclusive features: {inconclusive_count}")

# Chỉ cần import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Prepare data
print("🚀 TIME SERIES MODELS EXPERIMENTATION")
print("="*60)

# Use original features for exogenous variables (7 features)
exog_features = [
    'Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m', 'Độ ẩm tương đối 2m',
    'Độ ẩm đất bề mặt', 'Hướng gió 10m', 'Áp suất bề mặt', 'Bức xạ sóng dài xuống'
]

# Target variable
target = df_selected['Lượng mưa']
exog_data_original = df_selected[exog_features]

# ==============================================================================
# >>>>>>>>>>>> START OF FIX 1: MAKE EXOGENOUS VARIABLES STATIONARY <<<<<<<<<<<<
# ==============================================================================
exog_data_stationary = pd.DataFrame(index=exog_data_original.index)

print("🩺 Checking and transforming exogenous variables for stationarity...")
for col in exog_data_original.columns:
    # ADF test to check for unit root
    adf_pvalue = adfuller(exog_data_original[col].dropna())[1]
    if adf_pvalue >= 0.05:
        # If not stationary, apply first difference
        print(f"   - Column '{col}' is non-stationary (p={adf_pvalue:.3f}). Applying differencing.")
        exog_data_stationary[col] = exog_data_original[col].diff()
    else:
        # If stationary, use as is
        print(f"   - Column '{col}' is stationary (p={adf_pvalue:.3f}).")
        exog_data_stationary[col] = exog_data_original[col]

# Drop NaNs created by differencing
exog_data_stationary = exog_data_stationary.dropna()
# Align target variable with the new stationary exogenous data
target = target.loc[exog_data_stationary.index]
print("   ✅ All exogenous variables are now stationary.")
# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> END OF FIX 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================


# Time Series Cross-Validation setup
tscv = TimeSeriesSplit(n_splits=5)
print(f"\n📊 Data Setup:")
print(f"   Total samples after alignment: {len(target)}")
print(f"   CV splits: 5")

# Results storage
results = {
    'ARIMA': {'MAE': [], 'RMSE': [], 'R2': []},
    'SARIMA': {'MAE': [], 'RMSE': [], 'R2': []},
    'ARIMAX': {'MAE': [], 'RMSE': [], 'R2': []},
    'SARIMAX': {'MAE': [], 'RMSE': [], 'R2': []}
}

# Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(tscv.split(target)):
    print(f"\n🔄 Fold {fold + 1}/5")

    # Split data
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    # Use the stationary exogenous data
    X_train, X_test = exog_data_stationary.iloc[train_idx], exog_data_stationary.iloc[test_idx]

    # Define model orders
    order = (3, 0, 3)
    seasonal_order = (1, 1, 1, 7)

    # ==============================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>> START OF FIX 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    # Use SARIMAX for all models for consistency

    # 1. ARIMA
    try:
        model = SARIMAX(y_train, order=order).fit(disp=False)
        pred = model.forecast(steps=len(y_test))
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['ARIMA']['MAE'].append(mae); results['ARIMA']['RMSE'].append(rmse); results['ARIMA']['R2'].append(r2)
        print(f"   ARIMA: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   ARIMA: Failed - {e}")

    # 2. SARIMA
    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
        pred = model.forecast(steps=len(y_test))
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['SARIMA']['MAE'].append(mae); results['SARIMA']['RMSE'].append(rmse); results['SARIMA']['R2'].append(r2)
        print(f"   SARIMA: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   SARIMA: Failed - {e}")

    # 3. ARIMAX
    try:
        model = SARIMAX(y_train, exog=X_train, order=order).fit(disp=False)
        pred = model.forecast(steps=len(y_test), exog=X_test)
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['ARIMAX']['MAE'].append(mae); results['ARIMAX']['RMSE'].append(rmse); results['ARIMAX']['R2'].append(r2)
        print(f"   ARIMAX: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   ARIMAX: Failed - {e}")

    # 4. SARIMAX
    try:
        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
        pred = model.forecast(steps=len(y_test), exog=X_test)
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['SARIMAX']['MAE'].append(mae); results['SARIMAX']['RMSE'].append(rmse); results['SARIMAX']['R2'].append(r2)
        print(f"   SARIMAX: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   SARIMAX: Failed - {e}")

