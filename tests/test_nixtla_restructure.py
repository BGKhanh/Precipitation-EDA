"""Unit tests for Nixtlaverse restructuring, direct multi-horizon forecasting,

leakage guard code assertions, and flexible adapter wrappers.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import contextmanager
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.loader import DEFAULT_TEST_START_DATE
from src.models.nixtla import (
    StatsForecastAdapter,
    MLForecastAdapter,
    to_nixtla_format,
    get_default_stats_models,
    get_default_ml_models,
)
from src.training.trainer import (
    build_stats_adapter_from_eda,
    build_ml_adapter_from_eda,
)
from src.training.validator import UnifiedBenchmark


@contextmanager
def assert_raises(exc_type, match=None):
    """Assertion helper to verify exception type and message match."""
    try:
        yield
    except exc_type as e:
        if match:
            assert match in str(e), f"Expected '{match}' in '{str(e)}'"
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")



def test_models_central_package_and_forecasting_deprecation():
    """Verify src.models is central hub and src.forecasting emits DeprecationWarning."""
    # 1. src.forecasting must emit DeprecationWarning
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        import src.forecasting as forecasting

        assert any(
            issubclass(w.category, DeprecationWarning)
            and "src.forecasting is deprecated" in str(w.message)
            for w in recorded
        ), "DeprecationWarning was not emitted when importing src.forecasting"

    # Backward compatibility of forecasting shim
    assert hasattr(forecasting, "RecursiveForecaster")
    assert hasattr(forecasting, "MLForecastAdapter")

    # 2. src.models is central package (clean imports, exports both legacy and nixtla)
    import src.models as models
    assert hasattr(models, "RandomForestRainfallModel")
    assert hasattr(models, "RecursiveForecaster")
    assert hasattr(models, "MLForecastAdapter")
    assert hasattr(models, "StatsForecastAdapter")

    from src.models.legacy import BaseRainfallModel, ARIMAModel, RecursiveForecaster
    assert issubclass(models.RandomForestRainfallModel, BaseRainfallModel)
    assert ARIMAModel is not None
    assert RecursiveForecaster is not None


def test_leakage_guard_code_assertion():
    """Verify runtime assertion raises ValueError if timestamps reach canonical test cutoff."""
    adapter = StatsForecastAdapter(models=[])

    # Valid train DataFrame
    valid_train_dates = pd.date_range("2020-01-01", "2020-04-30", freq="D")
    df_valid = pd.DataFrame({"unique_id": "station_1", "ds": valid_train_dates, "y": 1.0})
    # Should not raise
    adapter._assert_no_test_leakage(df_valid)

    # Invalid DataFrame with test data (starting on DEFAULT_TEST_START_DATE)
    leaked_dates = pd.date_range("2020-04-20", DEFAULT_TEST_START_DATE, freq="D")
    df_leaked = pd.DataFrame({"unique_id": "station_1", "ds": leaked_dates, "y": 1.0})

    with assert_raises(ValueError, match="CRITICAL DATA LEAKAGE DETECTED"):
        adapter._assert_no_test_leakage(df_leaked)


def test_mlforecast_adapter_direct_max_horizon():
    """Verify MLForecastAdapter enforces direct multi-horizon (max_horizon != None)."""
    custom_model = LGBMRegressor(verbosity=-1, n_estimators=10)
    adapter = MLForecastAdapter(models=[custom_model], max_horizon=7)

    assert adapter.max_horizon == 7
    assert adapter._mlf is not None

    # Dummy train data
    train_dates = pd.date_range("2019-01-01", periods=60, freq="D")
    train_df = pd.DataFrame({
        "unique_id": "station_1",
        "ds": train_dates,
        "y": np.random.exponential(scale=3.0, size=60),
    })

    adapter.fit(train_df)
    assert adapter.is_fitted

    preds = adapter.predict(h=7)
    assert len(preds) == 7
    assert "LGBMRegressor" in preds.columns


def test_flexible_wrapper_model_override():
    """Verify B.3.4: Adapters accept custom models list overriding defaults."""
    from statsforecast.models import Naive

    custom_stats_models = [Naive()]
    adapter_custom = build_stats_adapter_from_eda(None, models=custom_stats_models)
    assert len(adapter_custom.models) == 1
    assert isinstance(adapter_custom.models[0], Naive)

    # Calling without models returns default models
    adapter_default = build_stats_adapter_from_eda(None)
    assert len(adapter_default.models) > 1

    custom_ml_models = [LGBMRegressor(n_estimators=5, verbosity=-1)]
    ml_adapter_custom = build_ml_adapter_from_eda(None, models=custom_ml_models, max_horizon=3)
    assert len(ml_adapter_custom.models) == 1
    assert ml_adapter_custom.max_horizon == 3


def test_statsforecast_adapter_intermittent_and_naive():
    """Verify StatsForecast default models include Intermittent and Naive models."""
    default_models = get_default_stats_models(None)
    model_names = [type(m).__name__ for m in default_models]

    assert "CrostonOptimized" in model_names
    assert "TSB" in model_names
    assert "IMAPA" in model_names
    assert "Naive" in model_names
    assert "SeasonalNaive" in model_names
    assert "AutoARIMA" in model_names
    assert "AutoETS" in model_names


def test_empirical_intermittent_models_ignore_exog():
    """Empirically confirm CrostonOptimized, TSB, and IMAPA ignore exogenous variables."""
    from statsforecast.models import CrostonOptimized, TSB, IMAPA

    y = np.array([0.0, 2.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0])
    X1 = np.ones((len(y), 2))
    X2 = np.ones((len(y), 2)) * 100.0

    for ModelCls, kwargs in [
        (CrostonOptimized, {}),
        (TSB, {"alpha_d": 0.2, "alpha_p": 0.2}),
        (IMAPA, {}),
    ]:
        assert ModelCls.uses_exog is False

        m1 = ModelCls(**kwargs).fit(y, X=X1)
        p1 = m1.predict(h=3, X=np.ones((3, 2)))["mean"]

        m2 = ModelCls(**kwargs).fit(y, X=X2)
        p2 = m2.predict(h=3, X=np.ones((3, 2)) * 100.0)["mean"]

        # Predictions must be identical regardless of exogenous variable values
        np.testing.assert_allclose(p1, p2, err_msg=f"{ModelCls.__name__} did not ignore exogenous X!")


def test_unified_benchmark_execution():
    """Verify UnifiedBenchmark runs across groups on test split without leakage."""
    from statsforecast.models import Naive, SeasonalNaive

    train_dates = pd.date_range("2019-01-01", "2020-04-30", freq="D")
    test_dates = pd.date_range("2020-05-01", periods=7, freq="D")

    train_df = pd.DataFrame({
        "unique_id": "station_1",
        "ds": train_dates,
        "y": np.random.exponential(scale=2.0, size=len(train_dates)),
    })
    test_df = pd.DataFrame({
        "unique_id": "station_1",
        "ds": test_dates,
        "y": np.random.exponential(scale=2.0, size=len(test_dates)),
    })

    bench = UnifiedBenchmark(train_df=train_df, test_df=test_df, horizon=7)
    results = bench.run(
        include_statsforecast=True,
        include_mlforecast=False,
        include_neuralforecast=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model_group" in results.columns
    assert "model_name" in results.columns
    assert "horizon" in results.columns
    assert "MAE" in results.columns
    assert "RMSE" in results.columns

    # Naive must be present in group 1
    assert "1. Naive Baselines" in results["model_group"].values


if __name__ == "__main__":
    tests = [
        test_models_central_package_and_forecasting_deprecation,
        test_leakage_guard_code_assertion,
        test_mlforecast_adapter_direct_max_horizon,
        test_flexible_wrapper_model_override,
        test_statsforecast_adapter_intermittent_and_naive,
        test_empirical_intermittent_models_ignore_exog,
        test_unified_benchmark_execution,
    ]
    print(f"Running {len(tests)} test functions...")
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print("\nAll tests passed successfully!")

