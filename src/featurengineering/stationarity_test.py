"""
Single source of truth for stationarity testing (ADF + KPSS).

Both ``src/eda/stationarity.py`` (EDA module) and ``src/models/time_series.py``
(model preparation) import from here instead of maintaining their own
implementations — this eliminates the risk of silent divergence in alpha
thresholds or NaN-handling.
"""

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CANONICAL STATIONARITY CONFIGURATION DEFAULTS (FROZEN FOR 01.5)
# =============================================================================
DEFAULT_ADF_REGRESSION: str = 'c'
DEFAULT_ADF_AUTOLAG: str = 'AIC'
DEFAULT_KPSS_REGRESSION: str = 'c'
DEFAULT_KPSS_NLAGS: str = 'legacy'
DEFAULT_STATIONARITY_ALPHA: float = 0.05


# =============================================================================
# CANONICAL CONSENSUS & D-RECOMMENDATION LOGIC (SINGLE SOURCE OF TRUTH)
# =============================================================================

def derive_d_recommendation(conclusion: str) -> Optional[int]:
    """Derive EDA heuristic d_recommendation from combined conclusion.

    STATIONARY     -> 0
    NON-STATIONARY -> 1
    INCONCLUSIVE   -> None (requires downstream domain/modeling judgment)
    """
    if conclusion == "STATIONARY":
        return 0
    elif conclusion == "NON-STATIONARY":
        return 1
    elif conclusion == "INCONCLUSIVE":
        return None
    raise ValueError(f"Unknown stationarity conclusion: {conclusion}")


def build_consensus_result(adf_stationary: bool, kpss_stationary: bool) -> Dict[str, Any]:
    """Pure function mapping ADF and KPSS test verdicts to consensus and d_recommendation.

    Single source of truth for stationarity consensus across the entire repository.
    """
    if adf_stationary and kpss_stationary:
        conclusion = "STATIONARY"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "NON-STATIONARY"
    else:
        conclusion = "INCONCLUSIVE"

    d_rec = derive_d_recommendation(conclusion)
    return {
        "conclusion": conclusion,                   # canonical upstream key
        "combined_conclusion": conclusion,          # EDA serialized field name alias
        "d_recommendation": d_rec,
        "d_recommendation_note": (
            "EDA heuristic recommendation only. NOT a mandatory modeling requirement. "
            "Final differencing order decided in 02_feature_engineering and 03_modeling."
        ),
    }


class StationarityTester:
    """Utility class for testing and ensuring stationarity of time series."""

    @staticmethod
    def test_stationarity(
        series: pd.Series,
        alpha: float = DEFAULT_STATIONARITY_ALPHA,
        regression: str = DEFAULT_ADF_REGRESSION,
        autolag: str = DEFAULT_ADF_AUTOLAG,
        kpss_nlags: str = DEFAULT_KPSS_NLAGS,
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
            autolag: ADF lag selection criterion (default 'AIC').
            kpss_nlags: KPSS lag specification (default 'legacy').
            verbose: Whether to print results.

        Returns:
            Dictionary with test results.
        """
        clean = series.dropna()
        if len(clean) < 10:
            return {
                'series_name': series.name or 'unnamed',
                'conclusion': 'INSUFFICIENT_DATA',
                'combined_conclusion': 'INSUFFICIENT_DATA',
                'd_recommendation': None,
            }

        # ADF Test (null hypothesis: unit root exists -> non-stationary)
        adf_result = adfuller(
            clean,
            regression=regression,
            autolag=autolag,
            result_object=False,
        )
        adf_stationary = bool(adf_result[1] < alpha)

        # KPSS Test (null hypothesis: series is stationary)
        kpss_result = kpss(
            clean,
            regression=regression,
            nlags=kpss_nlags,
            result_object=False,
        )
        kpss_stationary = bool(kpss_result[1] > alpha)

        # Consensus via upstream pure function
        consensus = build_consensus_result(adf_stationary, kpss_stationary)

        results = {
            'series_name': series.name or 'unnamed',
            'ADF_statistic': float(adf_result[0]),
            'ADF_pvalue': float(adf_result[1]),
            'ADF_selected_lag': int(adf_result[2]),   # AIC-selected lag count
            'ADF_nobs': int(adf_result[3]),           # Number of observations used
            'ADF_critical_values': {k: float(v) for k, v in adf_result[4].items()},
            'ADF_stationary': adf_stationary,
            'KPSS_statistic': float(kpss_result[0]),
            'KPSS_pvalue': float(kpss_result[1]),
            'KPSS_used_lags': int(kpss_result[2]),    # Lags used in KPSS statistic
            'KPSS_critical_values': {k: float(v) for k, v in kpss_result[3].items()},
            'KPSS_stationary': kpss_stationary,
            'alpha': float(alpha),
            'conclusion': consensus['conclusion'],
            'combined_conclusion': consensus['combined_conclusion'],
            'd_recommendation': consensus['d_recommendation'],
            'd_recommendation_note': consensus['d_recommendation_note'],
        }

        if verbose:
            print(f"📊 Stationarity Test: {results['series_name']}")
            print(f"   ADF p-value: {results['ADF_pvalue']:.6f} "
                  f"({'Stationary' if adf_stationary else 'Non-stationary'})")
            print(f"   KPSS p-value: {results['KPSS_pvalue']:.6f} "
                  f"({'Stationary' if kpss_stationary else 'Non-stationary'})")
            print(f"   ➤ Conclusion: {results['combined_conclusion']}")

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
