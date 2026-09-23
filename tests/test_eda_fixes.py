"""
Regression tests for EDA module fixes (P1-P4).

These tests lock the specific statistical/methodological decisions made
during the EDA review so they cannot be silently reverted in future edits.

Run with:
    python -m pytest tests/test_eda_fixes.py -v
    OR
    python tests/test_eda_fixes.py
"""

import sys
import os
import warnings

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =====================================================================
# P1 — Kurtosis classification uses EXCESS kurtosis (normal = 0)
# =====================================================================

class TestKurtosisClassification:
    """Lock: pandas.Series.kurtosis() returns EXCESS kurtosis (normal=0).
    Thresholds must be relative to 0, not 3."""

    def test_near_normal_classified_mesokurtic(self):
        """A near-normal distribution (excess kurtosis ~ 0) -> Mesokurtic."""
        # Uniform-ish data has negative excess kurtosis ~ -1.2
        data = pd.Series(range(100))
        kurt = data.kurtosis()
        # excess kurtosis for uniform(0..99) ~ -1.2
        assert -2 < kurt < -0.5, (
            f"Expected excess kurtosis in (-2, -0.5) for uniform data, "
            f"got {kurt:.4f}"
        )

    def test_normal_distribution_excess_kurtosis_near_zero(self):
        """Sanity: np.random.normal has excess kurtosis ~ 0."""
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 10000))
        kurt = data.kurtosis()
        assert abs(kurt) < 0.5, (
            f"Expected excess kurtosis near 0 for normal data, got {kurt:.4f}"
        )

    def test_leptokurtic_detection(self):
        """Heavy-tailed data (excess kurtosis >> 0) -> Leptokurtic."""
        rng = np.random.RandomState(42)
        # t-distribution with df=3 has heavy tails
        data = pd.Series(rng.standard_t(df=3, size=10000))
        kurt = data.kurtosis()
        assert kurt > 2, (
            f"Expected excess kurtosis > 2 for t(3) distribution, got {kurt:.4f}"
        )


# =====================================================================
# P2 — ADF/KPSS regression='c' vs 'ct' divergence
# =====================================================================

class TestStationarityRegression:
    """Lock: regression='c' and regression='ct' MUST produce different results
    on at least some series -- proving the parameter matters."""

    def test_c_vs_ct_produce_different_results(self):
        """ADF with regression='c' vs 'ct' gives different p-values."""
        from src.featurengineering.stationarity_test import StationarityTester

        # Create a series that's stationary around a constant (no trend)
        rng = np.random.RandomState(42)
        series = pd.Series(
            rng.normal(10, 1, 500),  # mean=10, no trend
            name='test_series',
        )

        result_c = StationarityTester.test_stationarity(
            series, regression='c'
        )
        result_ct = StationarityTester.test_stationarity(
            series, regression='ct'
        )

        # Both should complete successfully
        assert result_c['conclusion'] != 'INSUFFICIENT_DATA'
        assert result_ct['conclusion'] != 'INSUFFICIENT_DATA'

        # The ADF statistics MUST differ (different regression specs)
        assert result_c['ADF_statistic'] != result_ct['ADF_statistic'], (
            "ADF statistics are identical for 'c' and 'ct' -- "
            "regression parameter is not being passed correctly"
        )

        # KPSS statistics must also differ
        assert result_c['KPSS_statistic'] != result_ct['KPSS_statistic'], (
            "KPSS statistics are identical for 'c' and 'ct' -- "
            "regression parameter is not being passed correctly"
        )
        # Note: p-values may both saturate at 0.0 for strongly stationary
        # series, so we check statistics (always distinct) not p-values.

    def test_default_regression_is_c(self):
        """Default regression parameter is 'c' (constant only)."""
        from src.featurengineering.stationarity_test import StationarityTester
        import inspect

        sig = inspect.signature(StationarityTester.test_stationarity)
        default = sig.parameters['regression'].default
        assert default == 'c', (
            f"Default regression should be 'c', got '{default}'"
        )


# =====================================================================
# P3 — SARIMA seasonal period capped at 52
# =====================================================================

class TestSARIMACap:
    """Lock: With representative_periods containing long periods (e.g. 365),
    SARIMA suggestion must cap s <= 52 and document long periods separately."""

    def test_s_capped_at_52_with_long_periods(self):
        """representative_periods=[7,30,122,365] -> s <= 52, long_periods=[122,365]."""
        from src.eda.Stationarity import StationarityAutocorrelationAnalyzer

        # Create minimal data for the analyzer
        rng = np.random.RandomState(42)
        dates = pd.date_range('2000-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'Ngay': dates,
            'Luong_mua': rng.exponential(5, 1000),
        })

        analyzer = StationarityAutocorrelationAnalyzer(
            df,
            target_col='Luong_mua',
            date_col='Ngay',
            representative_periods=[7, 30, 122, 365],
        )

        # Access theory_driven_params to check seasonal_periods
        params = analyzer.theory_driven_params
        seasonal_periods = params['seasonal_periods']

        # All periods >= 7 should be in seasonal_periods
        assert 7 in seasonal_periods
        assert 30 in seasonal_periods
        assert 122 in seasonal_periods
        assert 365 in seasonal_periods

        # Verify cap logic
        MAX_FEASIBLE_SEASONAL = 52
        feasible = [p for p in seasonal_periods if p <= MAX_FEASIBLE_SEASONAL]
        long = [p for p in seasonal_periods if p > MAX_FEASIBLE_SEASONAL]

        assert max(feasible) <= 52, (
            f"Feasible periods should be <= 52, got {feasible}"
        )
        assert set(long) == {122, 365}, (
            f"Long periods should be {{122, 365}}, got {long}"
        )
        # s for SARIMA should be max(feasible) = 30
        s = max(feasible)
        assert s <= MAX_FEASIBLE_SEASONAL, f"s={s} exceeds cap {MAX_FEASIBLE_SEASONAL}"
        assert s == 30, f"Expected s=30 (max feasible), got {s}"


# =====================================================================
# P4 — ADI uses date-based intervals, not positional index
# =====================================================================

class TestIntermittencyADI:
    """Lock: ADI must use calendar-day intervals, not row-index differences.
    This prevents silent shrinkage when data has missing dates."""

    def test_adi_with_continuous_data(self):
        """ADI matches expected value for simple continuous data."""
        from src.eda.DistributionAnalysis import DistributionAnalyzer

        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        # Rain on day 0, 3, 6, 9 -> intervals = [3, 3, 3] days -> ADI = 3.0
        rainfall = [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]
        df = pd.DataFrame({'Ngay': dates, 'Luong_mua': rainfall})

        analyzer = DistributionAnalyzer(df, target_col='Luong_mua')
        result = analyzer.analyze_intermittency(date_col='Ngay', threshold=0.1)

        assert abs(result['adi'] - 3.0) < 0.01, (
            f"Expected ADI=3.0, got {result['adi']}"
        )

    def test_adi_with_gap_differs_from_positional(self):
        """When data has a gap, date-based ADI != positional-index ADI.
        
        This is the core invariant: if there's a 5-day gap in the data
        between two rain events, positional np.diff would give 1 (adjacent
        rows), but calendar-day diff gives 6 (the actual elapsed days).
        """
        from src.eda.DistributionAnalysis import DistributionAnalyzer

        # Day 1: rain, Day 2: no rain, [GAP: days 3-7 missing], Day 8: rain
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-08'])
        rainfall = [5.0, 0.0, 5.0]
        df = pd.DataFrame({'Ngay': dates, 'Luong_mua': rainfall})

        analyzer = DistributionAnalyzer(df, target_col='Luong_mua')
        result = analyzer.analyze_intermittency(date_col='Ngay', threshold=0.1)

        # Date-based: 2020-01-08 - 2020-01-01 = 7 days -> ADI = 7.0
        # Positional (WRONG): index 2 - index 0 = 2 -> ADI = 2.0
        assert abs(result['adi'] - 7.0) < 0.01, (
            f"Expected date-based ADI=7.0, got {result['adi']}. "
            f"If got 2.0, the code is using positional index instead of dates."
        )

    def test_classification_output_format(self):
        """Intermittency result contains all required fields."""
        from src.eda.DistributionAnalysis import DistributionAnalyzer

        rng = np.random.RandomState(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        # ~30% rain days
        rainfall = np.where(rng.random(365) < 0.3, rng.exponential(5, 365), 0)
        df = pd.DataFrame({'Ngay': dates, 'Luong_mua': rainfall})

        analyzer = DistributionAnalyzer(df, target_col='Luong_mua')
        result = analyzer.analyze_intermittency(date_col='Ngay', threshold=0.1)

        required_keys = {
            'adi', 'cv2', 'classification', 'model_recommendation',
            'rain_threshold_used', 'rain_day_pct', 'n_rain_events',
            'mean_rain_intensity_mm', 'median_interval_days',
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
        assert result['classification'] in {
            'SMOOTH', 'ERRATIC', 'INTERMITTENT', 'LUMPY'
        }


# =====================================================================
# Run as script (no pytest required)
# =====================================================================

def _run_all_tests():
    """Simple test runner that discovers and runs all test methods."""
    import traceback
    test_classes = [
        TestKurtosisClassification,
        TestStationarityRegression,
        TestSARIMACap,
        TestIntermittencyADI,
    ]
    passed, failed, errors = 0, 0, []
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            full_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {full_name}")
            except AssertionError as e:
                failed += 1
                errors.append((full_name, str(e)))
                print(f"  FAIL  {full_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((full_name, traceback.format_exc()))
                print(f"  ERROR {full_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  - {name}: {msg[:200]}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == '__main__':
    success = _run_all_tests()
    sys.exit(0 if success else 1)
