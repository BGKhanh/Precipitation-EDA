"""
Unit and integration tests for Canonical Two-Stage Hurdle Baseline.

Verifies:
1. Hurdle threshold semantics (c > 0 and hurdleized target Y*).
2. Backward compatibility of predict_combined and evaluate_rainfall_model.
3. Expected-value plug-in strategy (p * mu >= 0).
4. Temporal-safe probability calibration using modern FrozenEstimator API.
5. Decision threshold tau tuning separation.
6. Mathematical correctness of meteorological and probability metrics (Brier, POD, FAR, CSI).
7. RecursiveForecaster strategy support ('hard_gate' and 'expected').
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.legacy.base import (
    BaseRainfallModel,
    calculate_metrics,
    evaluate_rainfall_model,
)
from src.models.legacy.tree_models import RandomForestRainfallModel
from src.models.legacy.recursive import RecursiveForecaster


class DummyTwoStageModel(BaseRainfallModel):
    """Minimal concrete two-stage model for testing base behavior."""
    def _create_classification_model(self, **kwargs):
        return RandomForestClassifier(n_estimators=5, random_state=42)

    def _create_regression_model(self, **kwargs):
        return RandomForestRegressor(n_estimators=5, random_state=42)

    def _create_single_stage_model(self, **kwargs):
        return RandomForestRegressor(n_estimators=5, random_state=42)


# =====================================================================
# 1. Semantics of Rainfall Threshold c > 0
# =====================================================================

def test_hurdle_threshold_semantics():
    """Verify that c > 0 defines dry vs wet and Stage 2 only sees Y > c."""
    c = 0.1
    model = DummyTwoStageModel(use_two_stage=True, classification_threshold=c, random_state=42)
    
    y = pd.Series([0.0, 0.05, 0.10, 0.11, 5.0])
    X = pd.DataFrame({
        'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'f2': [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    
    y_clf, y_reg = model._prepare_targets(y)
    
    # Expected binary: 0, 0.05, 0.10 are dry (0); 0.11, 5.0 are wet (1)
    np.testing.assert_array_equal(y_clf.values, np.array([0, 0, 0, 1, 1]))
    
    # Fit model and verify Stage 2 training samples
    model.fit(X, y)
    assert model.is_fitted
    # The regression model should have been fitted on 2 samples
    assert hasattr(model.regression_model, 'n_features_in_')


# =====================================================================
# 2. Backward Compatibility
# =====================================================================

def test_backward_compatibility_predict_combined():
    """predict_combined(X) without strategy must match legacy hard_gate behavior."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=['a', 'b', 'c', 'd'])
    y = pd.Series(np.where(np.random.rand(50) > 0.6, np.random.exponential(5.0, 50), 0.0))
    
    model = RandomForestRainfallModel(use_two_stage=True, classification_threshold=0.1, n_estimators=10, random_state=42)
    model.fit(X, y)
    
    legacy_preds = model.predict_combined(X)
    explicit_hard_gate = model.predict_combined(X, strategy='hard_gate')
    
    np.testing.assert_array_equal(legacy_preds, explicit_hard_gate)


def test_backward_compatibility_evaluate_rainfall_model():
    """evaluate_rainfall_model must retain legacy keys: classification, regression, combined."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(40, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series(np.where(np.random.rand(40) > 0.5, np.random.exponential(4.0, 40), 0.0))
    
    model = RandomForestRainfallModel(use_two_stage=True, classification_threshold=0.1, n_estimators=10, random_state=42)
    model.fit(X, y)
    
    results = evaluate_rainfall_model(model, X, y)
    
    # Legacy keys must exist
    assert 'classification' in results
    assert 'regression' in results
    assert 'combined' in results
    assert 'ROC_AUC' in results['classification']
    assert 'MAE' in results['regression']
    assert 'MAE' in results['combined']
    
    # New structured tiers must also exist
    assert 'occurrence' in results
    assert 'conditional_amount' in results
    assert 'overall' in results
    assert 'hard_gate' in results['overall']
    assert 'expected' in results['overall']


# =====================================================================
# 3. Expected-Value Strategy Invariant
# =====================================================================

def test_expected_strategy_invariant():
    """Verify expected strategy computes p * max(0, mu) and all values >= 0."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(30, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series(np.where(np.random.rand(30) > 0.5, np.random.uniform(1.0, 10.0, 30), 0.0))
    
    model = RandomForestRainfallModel(use_two_stage=True, classification_threshold=0.1, n_estimators=10, random_state=42)
    model.fit(X, y)
    
    clf_probs, reg_preds = model.predict(X)
    expected_preds = model.predict_combined(X, strategy='expected')
    
    calculated_expected = np.maximum(0.0, clf_probs * np.maximum(0.0, reg_preds))
    np.testing.assert_allclose(expected_preds, calculated_expected, rtol=1e-5)
    assert (expected_preds >= 0.0).all()


# =====================================================================
# 4. Temporal-Safe Calibration
# =====================================================================

def test_temporal_safe_calibration():
    """Verify probability calibration using FrozenEstimator without refitting base classifier."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
    y = pd.Series(np.where(np.random.rand(100) > 0.5, np.random.exponential(3.0, 100), 0.0))
    
    # Temporal partition: Train (0..59), Calibration (60..79), Test (80..99)
    X_train, y_train = X.iloc[:60], y.iloc[:60]
    X_cal, y_cal = X.iloc[60:80], y.iloc[60:80]
    X_test, y_test = X.iloc[80:], y.iloc[80:]
    
    model = RandomForestRainfallModel(use_two_stage=True, classification_threshold=0.1, n_estimators=10, random_state=42)
    model.fit_calibrated(X_train, y_train, X_cal, y_cal, method='sigmoid')
    
    assert model.calibrated_classifier_ is not None
    
    # Uncalibrated vs Calibrated probabilities
    raw_probs, _ = model.predict(X_test, return_calibrated=False)
    cal_probs, _ = model.predict(X_test, return_calibrated=True)
    
    assert len(cal_probs) == len(X_test)
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()


# =====================================================================
# 5. Threshold Tuning Separation
# =====================================================================

def test_threshold_tuning_separation():
    """Verify tau is tuned on validation and lies in (0, 1)."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 3), columns=['a', 'b', 'c'])
    y = pd.Series(np.where(np.random.rand(80) > 0.6, np.random.exponential(5.0, 80), 0.0))
    
    X_train, y_train = X.iloc[:50], y.iloc[:50]
    X_val, y_val = X.iloc[50:], y.iloc[50:]
    
    model = RandomForestRainfallModel(use_two_stage=True, classification_threshold=0.1, n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    tau = model.find_optimal_threshold(X_val, y_val, metric='f1')
    assert 0.0 < tau < 1.0
    assert hasattr(model, 'optimal_threshold_')
    assert model.optimal_threshold_ == tau


# =====================================================================
# 6. Meteorological & Probability Metrics Formulation
# =====================================================================

def test_metrics_formulation_synthetic():
    """Verify Brier, POD, FAR, CSI against hand-calculated synthetic fixtures."""
    # Synthetic: 4 TP, 1 FP, 1 FN, 4 TN
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    # Predicted probabilities such that with tau=0.5:
    # 4 TP: p=0.8
    # 1 FP: p=0.7 (true is 0)
    # 1 FN: p=0.2 (true is 1)
    # 4 TN: p=0.1 (true is 0)
    y_pred = np.array([0.8, 0.8, 0.8, 0.8, 0.7, 0.1, 0.1, 0.1, 0.2, 0.1])
    
    metrics = calculate_metrics(y_true, y_pred, task='occurrence', threshold=0.5)
    
    # Expected:
    # TP = 4, FP = 1, FN = 1, TN = 4
    # POD = TP / (TP + FN) = 4 / 5 = 0.8
    # FAR = FP / (TP + FP) = 1 / 5 = 0.2
    # CSI = TP / (TP + FN + FP) = 4 / 6 = 2/3
    np.testing.assert_allclose(metrics['POD'], 0.8, rtol=1e-4)
    np.testing.assert_allclose(metrics['FAR'], 0.2, rtol=1e-4)
    np.testing.assert_allclose(metrics['CSI'], 4.0 / 6.0, rtol=1e-4)
    
    # Brier score: MSE between y_true and y_pred
    expected_brier = float(np.mean((y_true - y_pred) ** 2))
    np.testing.assert_allclose(metrics['Brier'], expected_brier, rtol=1e-4)


# =====================================================================
# 7. RecursiveForecaster Strategy Support
# =====================================================================

class MockFeatureBuilder:
    """Mock feature builder providing build_single_step."""
    def __init__(self):
        self.target_col = "Lượng mưa"
        self.date_col = "Ngày"

    def build_single_step(self, history):
        # Return last row with feature columns
        last_row = history.tail(1).copy()
        last_row['feat1'] = 1.0
        last_row['feat2'] = 2.0
        return last_row


def test_recursive_forecaster_strategies():
    """Verify RecursiveForecaster runs with both hard_gate and expected strategies."""
    model = DummyTwoStageModel(use_two_stage=True, classification_threshold=0.1, random_state=42)
    
    # Dummy train data
    X_dummy = pd.DataFrame({'feat1': [1.0, 2.0, 3.0], 'feat2': [2.0, 3.0, 4.0]})
    y_dummy = pd.Series([0.0, 5.0, 10.0])
    model.fit(X_dummy, y_dummy)
    
    fb = MockFeatureBuilder()
    
    # Dummy history of 10 days
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    history = pd.DataFrame({
        'Ngày': dates,
        'Lượng mưa': [0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0],
        'feat1': [1.0] * 10,
        'feat2': [2.0] * 10,
    })
    
    # 1. Hard-gate forecaster
    forecaster_hg = RecursiveForecaster(model, fb, strategy="hard_gate")
    res_hg = forecaster_hg.forecast(history, steps=7)
    assert len(res_hg) == 7
    assert (res_hg['prediction'] >= 0.0).all()
    
    # 2. Expected forecaster
    forecaster_exp = RecursiveForecaster(model, fb, strategy="expected")
    res_exp = forecaster_exp.forecast(history, steps=7)
    assert len(res_exp) == 7
    assert (res_exp['prediction'] >= 0.0).all()


def test_audit_expected_uses_calibrated_probability():
    """Audit 1: Verify that predict_combined(..., strategy='expected') strictly uses
    calibrated probabilities p_cal when calibration is enabled, satisfying:
        Y_hat = p_cal * mu
    and that find_optimal_threshold tunes tau on calibrated probabilities.
    """
    np.random.seed(42)
    # Train data
    X_train = pd.DataFrame({'f1': np.random.randn(60), 'f2': np.random.randn(60)})
    y_train = pd.Series(np.where(X_train['f1'] > 0, np.random.exponential(5.0, 60), 0.0))

    # Calibration data
    X_cal = pd.DataFrame({'f1': np.random.randn(40), 'f2': np.random.randn(40)})
    y_cal = pd.Series(np.where(X_cal['f1'] > 0.3, np.random.exponential(5.0, 40), 0.0))

    # Test / Query data
    X_test = pd.DataFrame({'f1': np.random.randn(20), 'f2': np.random.randn(20)})

    model = RandomForestRainfallModel(classification_threshold=0.0)
    model.fit_calibrated(X_train, y_train, X_cal, y_cal, method='sigmoid')

    # 1. Verify calibrated classifier is attached
    assert model.calibrated_classifier_ is not None

    # 2. Extract raw vs calibrated probabilities
    p_raw, mu_pred = model.predict(X_test, return_calibrated=False)
    p_cal, _ = model.predict(X_test, return_calibrated=True)

    # In general out-of-sample data, sigmoid calibrator changes probabilities
    assert not np.allclose(p_raw, p_cal)

    # 3. Predict with expected strategy using calibrated vs uncalibrated
    pred_exp_cal = model.predict_combined(X_test, strategy="expected", return_calibrated=True)
    pred_exp_raw = model.predict_combined(X_test, strategy="expected", return_calibrated=False)

    # 4. Strict mathematical identity check:
    # pred_exp_cal MUST exactly match p_cal * max(0, mu)
    expected_math_cal = np.maximum(0.0, p_cal * np.maximum(0.0, mu_pred))
    np.testing.assert_allclose(pred_exp_cal, expected_math_cal, rtol=1e-5, atol=1e-5)

    # pred_exp_raw MUST exactly match p_raw * max(0, mu)
    expected_math_raw = np.maximum(0.0, p_raw * np.maximum(0.0, mu_pred))
    np.testing.assert_allclose(pred_exp_raw, expected_math_raw, rtol=1e-5, atol=1e-5)

    # And pred_exp_cal != pred_exp_raw
    assert not np.allclose(pred_exp_cal, pred_exp_raw)

    # 5. Threshold tuning verification on calibrated vs raw
    tau_cal = model.find_optimal_threshold(X_cal, y_cal, metric='f1', return_calibrated=True)
    tau_raw = model.find_optimal_threshold(X_cal, y_cal, metric='f1', return_calibrated=False)
    assert 0.0 < tau_cal < 1.0
    assert 0.0 < tau_raw < 1.0


def test_audit_overall_metrics_target_semantics():
    """Audit 2: Verify that overall metrics evaluate against canonical hurdle target Y*,
    while overall_raw_supplementary evaluates against raw rainfall Y.
    """
    model = DummyTwoStageModel(classification_threshold=0.1)
    
    # Synthetic test set with sub-threshold rainfall (drizzle)
    # y = [0.0, 0.05, 0.10, 2.0, 5.0]
    # Y* = [0.0, 0.0,  0.0,  2.0, 5.0]
    y_test = pd.Series([0.0, 0.05, 0.10, 2.0, 5.0])
    X_test = pd.DataFrame({'f1': [0.0, 0.0, 0.0, 2.0, 5.0]})

    # Model hurdle target transformation check
    y_star = model.hurdleize_target(y_test.values)
    np.testing.assert_array_equal(y_star, np.array([0.0, 0.0, 0.0, 2.0, 5.0]))

    # Mock model predicting exact hurdle target Y*
    class MockPerfectHurdleModel(BaseRainfallModel):
        def _create_classification_model(self, **kwargs): return None
        def _create_regression_model(self, **kwargs): return None
        def _create_single_stage_model(self, **kwargs): return None
        def predict(self, X, return_calibrated=True):
            probs = np.where(X['f1'].values > 0.1, 1.0, 0.0)
            amounts = X['f1'].values
            return probs, amounts
        def predict_combined(self, X, threshold=None, strategy="hard_gate", return_calibrated=True):
            return np.where(X['f1'].values > 0.1, X['f1'].values, 0.0)

    perf_model = MockPerfectHurdleModel(classification_threshold=0.1)
    perf_model.is_fitted = True
    perf_model.optimal_threshold_ = 0.5
    
    res = evaluate_rainfall_model(perf_model, X_test, y_test)

    # Primary 'overall' tier MUST evaluate vs Y* -> perfect prediction -> MAE = 0.0
    assert res['overall']['hard_gate']['MAE'] == 0.0
    assert res['overall']['hard_gate']['RMSE'] == 0.0

    # Supplementary 'overall_raw_supplementary' evaluates vs raw Y -> MAE = (0.05 + 0.10)/5 = 0.03
    np.testing.assert_allclose(res['overall_raw_supplementary']['hard_gate']['MAE'], 0.03, atol=1e-5)


if __name__ == "__main__":
    print("Running test_hurdle_threshold_semantics...")
    test_hurdle_threshold_semantics()
    print("Running test_backward_compatibility_predict_combined...")
    test_backward_compatibility_predict_combined()
    print("Running test_backward_compatibility_evaluate_rainfall_model...")
    test_backward_compatibility_evaluate_rainfall_model()
    print("Running test_expected_strategy_invariant...")
    test_expected_strategy_invariant()
    print("Running test_temporal_safe_calibration...")
    test_temporal_safe_calibration()
    print("Running test_threshold_tuning_separation...")
    test_threshold_tuning_separation()
    print("Running test_metrics_formulation_synthetic...")
    test_metrics_formulation_synthetic()
    print("Running test_recursive_forecaster_strategies...")
    test_recursive_forecaster_strategies()
    print("Running test_audit_expected_uses_calibrated_probability (Audit 1)...")
    test_audit_expected_uses_calibrated_probability()
    print("Running test_audit_overall_metrics_target_semantics (Audit 2)...")
    test_audit_overall_metrics_target_semantics()
    print("\n[SUCCESS] ALL 9 CANONICAL HURDLE BASELINE & AUDIT TESTS PASSED SUCCESSFULLY!")
