"""
Single source of truth for stationarity testing (ADF + KPSS).

Both ``src/eda/stationarity.py`` (EDA module) and ``src/models/time_series.py``
(model preparation) import from here instead of maintaining their own
implementations — this eliminates the risk of silent divergence in alpha
thresholds or NaN-handling.
"""

from typing import Any, Dict

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')


class StationarityTester:
    """Utility class for testing and ensuring stationarity of time series."""

    @staticmethod
    def test_stationarity(
        series: pd.Series,
        alpha: float = 0.05,
        regression: str = 'c',
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Test stationarity using ADF and KPSS tests.

        Args:
            series: Time series to test.
            alpha: Significance level for both tests (default 0.05).
            regression: Regression specification for both tests.
                ``'c'`` (default) = constant only — appropriate when the
                series has no deterministic trend (e.g. rainfall).
                ``'ct'`` = constant + trend — use only when a linear trend
                is hypothesised (e.g. temperature with climate change).
                Using ``'ct'`` reduces test power and may falsely retain
                a unit-root null when the series is trend-stationary.
            verbose: Whether to print results.

        Returns:
            Dictionary with test results.
        """
        clean = series.dropna()
        if len(clean) < 10:
            return {
                'series_name': series.name or 'unnamed',
                'conclusion': 'INSUFFICIENT_DATA',
            }

        # ADF Test (null hypothesis: unit root exists → non-stationary)
        adf_result = adfuller(clean, regression=regression, autolag='AIC')
        adf_stationary = adf_result[1] < alpha

        # KPSS Test (null hypothesis: series is stationary)
        kpss_result = kpss(clean, regression=regression)
        kpss_stationary = kpss_result[1] > alpha

        # Overall conclusion
        if adf_stationary and kpss_stationary:
            conclusion = "STATIONARY"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "NON-STATIONARY"
        else:
            conclusion = "INCONCLUSIVE"

        results = {
            'series_name': series.name or 'unnamed',
            'ADF_statistic': adf_result[0],
            'ADF_pvalue': adf_result[1],
            'ADF_stationary': adf_stationary,
            'KPSS_statistic': kpss_result[0],
            'KPSS_pvalue': kpss_result[1],
            'KPSS_stationary': kpss_stationary,
            'alpha': alpha,
            'conclusion': conclusion,
        }

        if verbose:
            print(f"📊 Stationarity Test: {results['series_name']}")
            print(f"   ADF p-value: {results['ADF_pvalue']:.6f} "
                  f"({'Stationary' if adf_stationary else 'Non-stationary'})")
            print(f"   KPSS p-value: {results['KPSS_pvalue']:.6f} "
                  f"({'Stationary' if kpss_stationary else 'Non-stationary'})")
            print(f"   ➤ Conclusion: {conclusion}")

        return results

    @staticmethod
    def make_stationary(
        data: pd.DataFrame,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Make all columns in DataFrame stationary by differencing if needed.

        Args:
            data: DataFrame with time series columns.
            alpha: Significance level for the stationarity tests.
            verbose: Whether to print transformation details.

        Returns:
            DataFrame with stationary series (NaN rows from differencing
            are dropped).
        """
        if verbose:
            print("🩺 Checking and transforming variables for stationarity...")

        stationary_data = pd.DataFrame(index=data.index)

        for col in data.columns:
            test_result = StationarityTester.test_stationarity(
                data[col], alpha=alpha
            )

            if test_result['conclusion'] == 'NON-STATIONARY':
                stationary_data[col] = data[col].diff()
                if verbose:
                    print(f"   - Column '{col}' is non-stationary. Applying differencing.")
            else:
                stationary_data[col] = data[col]
                if verbose:
                    print(f"   - Column '{col}' is stationary.")

        stationary_data = stationary_data.dropna()

        if verbose:
            print("   ✅ All variables are now stationary.")

        return stationary_data
