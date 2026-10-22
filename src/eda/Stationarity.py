# =============================================================================
# COMPONENT 3: STATIONARITY & AUTOCORRELATION DIAGNOSTICS
# =============================================================================

from typing import Dict, List, Tuple, Sequence, Any, Optional, Union
from pathlib import Path
import json
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import probplot, jarque_bera

# Single Source of Truth for Stationarity & Consensus
from src.featurengineering.stationarity_test import (
    StationarityTester,
    build_consensus_result,
    derive_d_recommendation,
    DEFAULT_ADF_REGRESSION as ADF_REGRESSION,
    DEFAULT_ADF_AUTOLAG as ADF_AUTOLAG,
    DEFAULT_KPSS_REGRESSION as KPSS_REGRESSION,
    DEFAULT_KPSS_NLAGS as KPSS_NLAGS,
    DEFAULT_STATIONARITY_ALPHA,
)

# Re-export pure helpers for EDA module compatibility
__all__ = [
    "StationarityAutocorrelationAnalyzer",
    "build_consensus_result",
    "derive_d_recommendation",
    "classify_seasonal_periods",
    "extract_significant_lags",
    "build_exploratory_sarima_heuristic",
    "run_residual_diagnostics",
    "export_stationarity_artifacts",
    "analyze_stationarity_autocorrelation",
    "get_model_recommendations",
    # Constants
    "INFERENCE_ALPHA",
    "STATIONARITY_ALPHA",
    "ADF_REGRESSION",
    "ADF_AUTOLAG",
    "KPSS_REGRESSION",
    "KPSS_NLAGS",
    "ACF_NLAGS",
    "ACF_ALPHA",
    "ACF_FFT",
    "ACF_ADJUSTED",
    "ACF_BARTLETT_CONFINT",
    "PACF_NLAGS",
    "PACF_ALPHA",
    "PACF_METHOD",
    "SIGNIFICANCE_ALPHA",
    "SIGNIFICANT_LAG_TOP_N",
    "EXCLUDE_LAG_ZERO",
    "EDA_SARIMA_SEASONAL_CAP",
    "REPRESENTATIVE_SEASONAL_CANDIDATES",
    "SCREENED_OUT_LONG_PERIODS",
    "RESIDUAL_ACF_NLAGS",
    "RESIDUAL_ACF_ALPHA",
    "RESIDUAL_ACF_FFT",
    "RESIDUAL_ACF_ADJUSTED",
    "RESIDUAL_ACF_BARTLETT_CONFINT",
    "LJUNG_BOX_LAGS",
    "LJUNG_BOX_ALPHA",
    "LJUNG_BOX_MODEL_DF",
    "LJUNG_BOX_AUTO_LAG",
    "LJUNG_BOX_RETURN_DF",
    "ROLLING_STATS_WINDOW",
]

warnings.filterwarnings('ignore')

# --- Master Inference Significance Level ---
INFERENCE_ALPHA: float = DEFAULT_STATIONARITY_ALPHA

# Derived semantic levels (single source: INFERENCE_ALPHA)
STATIONARITY_ALPHA: float = INFERENCE_ALPHA
ACF_ALPHA: float = INFERENCE_ALPHA
PACF_ALPHA: float = INFERENCE_ALPHA
SIGNIFICANCE_ALPHA: float = INFERENCE_ALPHA
RESIDUAL_ACF_ALPHA: float = INFERENCE_ALPHA
LJUNG_BOX_ALPHA: float = INFERENCE_ALPHA

# --- Primary Autocorrelation Configuration (Raw Rainfall Y_t) ---
ACF_NLAGS: int = 200
ACF_FFT: bool = True
ACF_ADJUSTED: bool = False
ACF_BARTLETT_CONFINT: bool = True  # Frozen for reproducibility

PACF_NLAGS: int = 200
PACF_METHOD: str = "ywm"  # Yule-Walker without sample-size adjustment (biased denominator n)

# --- Autocorrelation Significance Rule ---
SIGNIFICANT_LAG_TOP_N: int = 10  # Reporting only; does NOT cap p/q heuristic
EXCLUDE_LAG_ZERO: bool = True

# --- Seasonal Screening Rule & Immutable Tuples ---
EDA_SARIMA_SEASONAL_CAP: int = 52
REPRESENTATIVE_SEASONAL_CANDIDATES: tuple[int, ...] = (7, 30)
SCREENED_OUT_LONG_PERIODS: tuple[int, ...] = (122, 365)

# --- Residual Diagnostics Configuration (MSTL Residual R_t) ---
RESIDUAL_ACF_NLAGS: int = 40
RESIDUAL_ACF_FFT: bool = True
RESIDUAL_ACF_ADJUSTED: bool = False
RESIDUAL_ACF_BARTLETT_CONFINT: bool = True

# --- Ljung-Box Operational Configuration ---
LJUNG_BOX_LAGS: tuple[int, ...] = (10, 30, 52)
LJUNG_BOX_MODEL_DF: int = 0      # Raw residual, no parameter estimation in 01.5
LJUNG_BOX_AUTO_LAG: bool = False  # Evaluate exact specified lags
LJUNG_BOX_RETURN_DF: bool = True  # Return structured DataFrame

# --- Visual Diagnostics ---
ROLLING_STATS_WINDOW: int = 365   # 1-year rolling window for visual stationarity inspection


# =============================================================================
# PURE FUNCTIONS & CORE LOGIC
# =============================================================================

def classify_seasonal_periods(
    candidate_periods: Sequence[int],
    cap: int = EDA_SARIMA_SEASONAL_CAP,
) -> tuple[list[int], list[int]]:
    """Partition candidate seasonal periods into feasible and screened-out subsets.

    Validates that cap and all candidate periods are strictly positive integers.

    Returns:
        (feasible_periods, screened_out_periods)

    Raises:
        ValueError: If cap <= 0, not integer, or any period <= 0, or not integer.
    """
    if not isinstance(cap, (int, np.integer)) or isinstance(cap, bool) or cap <= 0:
        raise ValueError(f"Invalid cap: {cap}. Cap must be a positive integer.")

    for p in candidate_periods:
        if not isinstance(p, (int, np.integer)) or isinstance(p, bool) or p <= 0:
            raise ValueError(f"Invalid candidate period: {p}. Periods must be positive integers.")

    feasible = [int(p) for p in candidate_periods if p <= cap]
    screened_out = [int(p) for p in candidate_periods if p > cap]
    return feasible, screened_out


def extract_significant_lags(
    values: np.ndarray,
    confint: np.ndarray,
    exclude_lag_zero: bool = EXCLUDE_LAG_ZERO,
) -> tuple[int, list[int]]:
    """Determine significant lags where confidence interval excludes zero.

    A lag k is significant iff:
        confint[k, 0] > 0 or confint[k, 1] < 0

    Returns:
        (n_significant_all, top_10_lags)
    """
    start_idx = 1 if exclude_lag_zero else 0
    significant_lags = []

    for lag in range(start_idx, len(values)):
        if confint[lag, 0] > 0 or confint[lag, 1] < 0:
            significant_lags.append(lag)

    n_significant_all = len(significant_lags)

    ranked_by_magnitude = sorted(
        significant_lags,
        key=lambda k: abs(values[k]),
        reverse=True,
    )
    top_lags = ranked_by_magnitude[:SIGNIFICANT_LAG_TOP_N]

    return n_significant_all, top_lags


def build_exploratory_sarima_heuristic(
    n_sig_pacf_all: int,
    n_sig_acf_all: int,
    d_rec: Optional[int],
    feasible_periods: Sequence[int] = REPRESENTATIVE_SEASONAL_CANDIDATES,
) -> dict[str, Any]:
    """Build exploratory SARIMA heuristic with explicit seasonal template semantics.

    Field Semantic Note (R39): 'candidate_seasonal_orders' contains seasonal templates
    (P=1, D=1, Q=1, s). When d_recommendation is None (INCONCLUSIVE), they are NOT
    executable full SARIMA orders because non-seasonal d remains undetermined.

    Methodology Note (R33): This p/q rule is a project-specific EDA heuristic based
    on the count of significant ACF/PACF lags. It is NOT a formal Box-Jenkins order
    identification procedure and must NOT be interpreted as an optimal order.
    """
    p_rec = min(3, n_sig_pacf_all)
    q_rec = min(3, n_sig_acf_all)

    candidate_templates = []
    for s in feasible_periods:
        order_dict = {
            "P": 1,
            "D": 1,
            "Q": 1,
            "s": int(s),
            "candidate_label": "weekly" if s == 7 else "sub-monthly" if s == 30 else f"period_{s}",
            "is_executable_full_order": d_rec is not None,
        }
        if d_rec is not None:
            order_dict["full_order_label"] = f"SARIMA({p_rec},{d_rec},{q_rec})x(1,1,1,{s})"
        else:
            order_dict["full_order_label"] = None
        candidate_templates.append(order_dict)

    return {
        "status": "EDA-suggested baseline heuristic (NOT optimal model search)",
        "methodology_note": (
            "This p/q rule is a project-specific EDA heuristic based on the count "
            "of significant ACF/PACF lags. It is not a formal Box-Jenkins order "
            "identification procedure and must not be interpreted as an optimal order."
        ),
        "p_recommendation": p_rec,
        "q_recommendation": q_rec,
        "d_recommendation": d_rec,
        "derivation_rule": {
            "p": "min(3, n_significant_pacf_lags_all)",
            "q": "min(3, n_significant_acf_lags_all)",
            "d": "0 if STATIONARY; 1 if NON-STATIONARY; null if INCONCLUSIVE",
            "P": "1 (heuristic for feasible seasonal candidate)",
            "Q": "1 (heuristic for feasible seasonal candidate)",
            "D": "1 (seasonal differencing heuristic for feasible seasonal candidate)",
        },
        "candidate_seasonal_orders": candidate_templates,
        "seasonal_template_note": (
            "candidate_seasonal_orders are seasonal templates (P=1, D=1, Q=1, s). "
            "When d_recommendation is null, they are NOT executable full SARIMA orders "
            "because the non-seasonal differencing order d remains undetermined. "
            "Downstream modeling (02_feature_engineering and 03_modeling) must select d."
        ),
        "feasible_seasonal_periods": list(feasible_periods),
        "eda_sarima_seasonal_cap": EDA_SARIMA_SEASONAL_CAP,
        "primary_seasonal_period_selection": "deferred to 03_modeling",
    }


def run_residual_diagnostics(residual_series: pd.Series) -> dict[str, Any]:
    """Execute portmanteau and normality diagnostics on MSTL residual series R_t."""
    r_clean = residual_series.dropna()
    n_obs = len(r_clean)

    # 1. Residual ACF (40 lags)
    resid_acf_vals, resid_acf_conf = acf(
        r_clean,
        nlags=RESIDUAL_ACF_NLAGS,
        alpha=RESIDUAL_ACF_ALPHA,
        fft=RESIDUAL_ACF_FFT,
        adjusted=RESIDUAL_ACF_ADJUSTED,
        bartlett_confint=RESIDUAL_ACF_BARTLETT_CONFINT,
    )

    # 2. Ljung-Box Portmanteau Test (evaluated with frozen defaults)
    lb_df = acorr_ljungbox(
        r_clean,
        lags=list(LJUNG_BOX_LAGS),
        model_df=LJUNG_BOX_MODEL_DF,
        auto_lag=LJUNG_BOX_AUTO_LAG,
        return_df=LJUNG_BOX_RETURN_DF,
    )

    lb_stats = {}
    lb_pvals = {}
    for lag in LJUNG_BOX_LAGS:
        if lag in lb_df.index:
            lb_stats[str(lag)] = float(lb_df.loc[lag, "lb_stat"])
            lb_pvals[str(lag)] = float(lb_df.loc[lag, "lb_pvalue"])

    # 3. Jarque-Bera Test
    jb_stat, jb_pval = jarque_bera(r_clean)

    # 4. Summary Statistics
    mean_val = float(np.mean(r_clean))
    std_val = float(np.std(r_clean, ddof=1))
    skew_val = float(pd.Series(r_clean).skew())
    kurt_val = float(pd.Series(r_clean).kurtosis())  # excess kurtosis

    return {
        "series": "mstl_residual_R_t (log1p scale)",
        "residual_acf_configuration": {
            "nlags": RESIDUAL_ACF_NLAGS,
            "alpha": RESIDUAL_ACF_ALPHA,
            "fft": RESIDUAL_ACF_FFT,
            "adjusted": RESIDUAL_ACF_ADJUSTED,
            "bartlett_confint": RESIDUAL_ACF_BARTLETT_CONFINT,
            "estimator_convention": "same as primary ACF",
        },
        "ljung_box": {
            "series": "mstl_residual_R_t",
            "lags_evaluated": list(LJUNG_BOX_LAGS),
            "model_df": LJUNG_BOX_MODEL_DF,
            "auto_lag": LJUNG_BOX_AUTO_LAG,
            "return_df": LJUNG_BOX_RETURN_DF,
            "lag_rationale": {
                "10": "short-term dependence (~10 days)",
                "30": "sub-monthly / intra-month dependence scale",
                "52": "medium-short-range dependence beyond one month",
            },
            "portmanteau_note": (
                "Ljung-Box at lag L is a portmanteau test for cumulative autocorrelation "
                "up to L; not a test for seasonality at period L."
            ),
            "alpha": LJUNG_BOX_ALPHA,
            "lb_statistics": lb_stats,
            "lb_pvalues": lb_pvals,
        },
        "jarque_bera": {
            "statistic": float(jb_stat),
            "p_value": float(jb_pval),
        },
        "residual_stats": {
            "count": int(n_obs),
            "mean": mean_val,
            "std": std_val,
            "skewness": skew_val,
            "excess_kurtosis": kurt_val,
        },
        "eda_implication": [
            "MSTL with periods=(7,30,365) does not fully capture all short-term structure in residuals.",
            "Residual autocorrelation (if present) suggests AR/lag representations merit downstream evaluation.",
            "Non-Gaussian residuals (if present) suggest conditional distribution modeling merits downstream evaluation.",
            "Neither result establishes superiority of any specific model family.",
            "Model selection deferred to 02_feature_engineering and 03_modeling.",
        ],
    }


def export_stationarity_artifacts(
    report_dict: dict[str, Any],
    canonical_path: Union[str, Path],
    bridge_path: Union[str, Path],
) -> tuple[Path, Path]:
    """Export canonical and backward-compatible bridge stationarity artifacts."""
    canonical_p = Path(canonical_path)
    bridge_p = Path(bridge_path)

    canonical_p.parent.mkdir(parents=True, exist_ok=True)
    bridge_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export canonical artifact (pure schema)
    with open(canonical_p, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # 2. Build bridge artifact with dynamically derived runtime aliases
    heuristic = report_dict["exploratory_sarima_heuristic"]
    p_rec = heuristic["p_recommendation"]
    d_rec = heuristic["d_recommendation"]
    q_rec = heuristic["q_recommendation"]
    feasible = heuristic["feasible_seasonal_periods"]
    first_s = feasible[0] if feasible else 7

    bridge_dict = copy.deepcopy(report_dict)
    bridge_dict["__meta_bridge__"] = {
        "is_bridge": True,
        "canonical_path": str(canonical_p).replace("\\", "/"),
        "compatibility_target": "01.7_eda_synthesis_report.ipynb / EDAReport",
    }
    bridge_dict["suggested_sarima_order"] = [p_rec, d_rec if d_rec is not None else 0, q_rec]
    bridge_dict["suggested_seasonal_order"] = [1, 1, 1, int(first_s)]
    bridge_dict["backward_compatibility_disclaimer"] = (
        "The top-level fields 'suggested_sarima_order' and 'suggested_seasonal_order' "
        "are backward-compatibility aliases provided solely for legacy consumers (such as "
        "EDAReport and 01.7 synthesis). When d_recommendation is null (INCONCLUSIVE), "
        "'suggested_sarima_order' falls back to d=0 solely to maintain Tuple[int, int, int] "
        "schema validity for SARIMA_EDA initialization. This does NOT represent a finding "
        "of stationarity or a recommendation of d=0. Downstream modeling MUST consume "
        "'stationarity_tests.combined_conclusion' and 'stationarity_tests.d_recommendation' "
        "instead of this legacy alias."
    )

    with open(bridge_p, "w", encoding="utf-8") as f:
        json.dump(bridge_dict, f, indent=2, ensure_ascii=False)

    return canonical_p, bridge_p


TARGET_COLUMN_ALIASES: tuple[str, ...] = ("Lượng mưa", "rain", "rainfall", "precipitation")
DATE_COLUMN_ALIASES: tuple[str, ...] = ("Ngày", "date", "datetime", "timestamp")


def resolve_column(
    df: pd.DataFrame,
    alias_list: Sequence[str] = TARGET_COLUMN_ALIASES,
    role: str = "target",
    explicit_col: Optional[str] = None,
) -> str:
    """Fail-fast column resolver.
    0 matches -> ValueError (informative)
    1 match   -> return column name
    >1 matches -> ValueError (ambiguity)
    """
    if explicit_col is not None:
        if explicit_col in df.columns:
            return explicit_col
        raise ValueError(f"Explicit {role} column '{explicit_col}' not found in DataFrame.")

    matches = [col for col in alias_list if col in df.columns]
    if len(matches) == 0:
        raise ValueError(
            f"Column resolver: No {role} column found. "
            f"Searched: {list(alias_list)}. Available: {list(df.columns)[:10]}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Column resolver: Ambiguous {role} columns — multiple aliases match: {matches}. "
            f"Rename DataFrame columns to remove ambiguity."
        )
    return matches[0]


_resolve_column = resolve_column


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class StationarityAutocorrelationAnalyzer:
    """Stationarity & Autocorrelation diagnostics using representative periods.
    
    Accepts ``representative_periods`` (e.g. ``[7, 30, 122, 365]``) to guide
    ACF/PACF lag range and SARIMA seasonal-order suggestion.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Lượng mưa', 
                 date_col: str = 'Ngày', 
                 mstl_results: Optional[Dict] = None,
                 representative_periods: Optional[List[int]] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.mstl_results = mstl_results
        self.representative_periods = representative_periods or []
        self._prepare_data()
        self.theory_driven_params = self._extract_theory_driven_parameters()

    def _prepare_data(self) -> None:
        """Prepare data for analysis"""
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    def _extract_theory_driven_parameters(self) -> Dict[str, Any]:
        if not self.representative_periods:
            return {
                'has_periods': False,
                'max_lags_recommended': ACF_NLAGS,
                'seasonal_periods': [],
                'representative_periods': [],
                'parameter_source': 'default'
            }

        max_period = max(self.representative_periods)
        buffer = max(20, max_period // 4)
        max_lags_recommended = min(max_period + buffer, ACF_NLAGS)
        seasonal_periods = [p for p in self.representative_periods if p >= 7]

        return {
            'has_periods': True,
            'max_lags_recommended': max_lags_recommended,
            'seasonal_periods': seasonal_periods,
            'representative_periods': self.representative_periods,
            'parameter_source': 'theory_driven',
            'max_period': max_period,
            'buffer_used': buffer
        }

    def compute_autocorrelation(self) -> Dict[str, Any]:
        """Compute ACF and PACF autocorrelation diagnostics."""
        target_ts = self.df.set_index(self.date_col)[self.target_col].dropna()
        return self._autocorrelation_analysis(target_ts)

    def _autocorrelation_analysis(self, target_ts: pd.Series) -> Dict[str, Any]:
        """Bước 2: Phân tích Tự tương quan với Representative Periods"""
        max_lags = ACF_NLAGS
        acf_results = self._perform_acf_analysis(target_ts, max_lags, "Original Series")
        pacf_results = self._perform_pacf_analysis(target_ts, max_lags, "Original Series")
        sarima_suggestions = self._generate_theory_driven_sarima_suggestions(acf_results, pacf_results)

        return {
            'acf_results': acf_results,
            'pacf_results': pacf_results,
            'sarima_suggestions': sarima_suggestions,
            'theory_driven_params': self.theory_driven_params,
            'max_lags_used': max_lags,
            'success': True
        }

    def _perform_acf_analysis(self, ts: pd.Series, max_lags: int, label: str) -> Dict[str, Any]:
        """ACF Analysis using frozen module constants"""
        try:
            acf_values, acf_confint = acf(
                ts,
                nlags=ACF_NLAGS,
                alpha=ACF_ALPHA,
                fft=ACF_FFT,
                adjusted=ACF_ADJUSTED,
                bartlett_confint=ACF_BARTLETT_CONFINT,
            )
            n_sig_all, top_lags = extract_significant_lags(acf_values, acf_confint)
            significant_lags = [(lag, float(acf_values[lag])) for lag in top_lags]

            periods_validated_lags = self._find_lags_near_periods(significant_lags) if self.theory_driven_params['has_periods'] else []

            return {
                'acf_values': acf_values,
                'significant_lags': significant_lags,
                'n_significant_all': n_sig_all,
                'periods_validated_lags': periods_validated_lags,
                'max_lags': ACF_NLAGS,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _perform_pacf_analysis(self, ts: pd.Series, max_lags: int, label: str) -> Dict[str, Any]:
        """PACF Analysis using frozen module constants"""
        try:
            pacf_values, pacf_confint = pacf(
                ts,
                nlags=PACF_NLAGS,
                alpha=PACF_ALPHA,
                method=PACF_METHOD,
            )
            n_sig_all, top_lags = extract_significant_lags(pacf_values, pacf_confint)
            significant_lags = [(lag, float(pacf_values[lag])) for lag in top_lags]

            periods_validated_lags = self._find_lags_near_periods(significant_lags) if self.theory_driven_params['has_periods'] else []

            return {
                'pacf_values': pacf_values,
                'significant_lags': significant_lags,
                'n_significant_all': n_sig_all,
                'periods_validated_lags': periods_validated_lags,
                'max_lags': PACF_NLAGS,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _find_lags_near_periods(self, significant_lags: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not self.representative_periods:
            return significant_lags

        validated_lags = []
        tolerance = 5
        for lag, value in significant_lags:
            for period in self.representative_periods:
                if abs(lag - period) <= tolerance:
                    validated_lags.append((lag, value))
                    break
        return validated_lags

    def _generate_theory_driven_sarima_suggestions(self, acf_results: Dict, pacf_results: Dict) -> Dict[str, Any]:
        if not (acf_results.get('success', False) and pacf_results.get('success', False)):
            return {'success': False}

        n_sig_pacf = pacf_results.get('n_significant_all', len(pacf_results.get('significant_lags', [])))
        n_sig_acf = acf_results.get('n_significant_all', len(acf_results.get('significant_lags', [])))

        # Seasonal candidate classification
        seasonal_periods = self.theory_driven_params.get('seasonal_periods', list(REPRESENTATIVE_SEASONAL_CANDIDATES))
        feasible_periods, long_periods = classify_seasonal_periods(seasonal_periods, cap=EDA_SARIMA_SEASONAL_CAP)

        heuristic = build_exploratory_sarima_heuristic(
            n_sig_pacf_all=n_sig_pacf,
            n_sig_acf_all=n_sig_acf,
            d_rec=0,  # placeholder before stationarity check or default
            feasible_periods=feasible_periods,
        )

        p = heuristic['p_recommendation']
        q = heuristic['q_recommendation']
        first_s = feasible_periods[0] if feasible_periods else 7

        model_suggestions = [
            f'SARIMA({p},1,{q})x(1,1,1,{s})' for s in feasible_periods
        ]

        return {
            'nonseasonal_ar': p,
            'nonseasonal_ma': q,
            'seasonal_ar': 1 if feasible_periods else 0,
            'seasonal_ma': 1 if feasible_periods else 0,
            'seasonal_period': first_s,
            'all_seasonal_periods_detected': seasonal_periods,
            'feasible_periods': feasible_periods,
            'long_periods_fourier_recommended': long_periods,
            'suggested_models': model_suggestions,
            'heuristic': heuristic,
            'methodology': 'theory_driven' if self.theory_driven_params['has_periods'] else 'traditional',
            'success': True,
            'orders': [{'order': (p, 0, q)}],
            'seasonal_orders': [{'order': (1, 1, 1, first_s)}],
        }

    def _theory_driven_autocorrelation_visualization(self, ts: pd.Series, max_lags: int,
                                                     label: str, return_fig: bool = True) -> Optional[plt.Figure]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Autocorrelation Diagnostics (ACF & PACF) - {label}', fontsize=14, fontweight='bold')

        periods_info_text = f"Representative Periods: {self.representative_periods}" if self.representative_periods else ""

        try:
            plot_acf(ts, lags=max_lags, ax=axes[0, 0], alpha=ACF_ALPHA)
            axes[0, 0].set_title(f'ACF - {label}\n{periods_info_text}')
            if self.representative_periods:
                for period in self.representative_periods[:5]:
                    if period <= max_lags:
                        axes[0, 0].axvline(x=period, color='red', linestyle='--', alpha=0.6)
            axes[0, 0].grid(True, alpha=0.3)
        except Exception:
            axes[0, 0].text(0.5, 0.5, 'ACF Plot Error', ha='center', va='center', transform=axes[0, 0].transAxes)

        try:
            plot_pacf(ts, lags=max_lags, ax=axes[0, 1], alpha=PACF_ALPHA)
            axes[0, 1].set_title(f'PACF - {label}\nMax lags: {max_lags} (theory-driven)')
            if self.representative_periods:
                for period in self.representative_periods[:5]:
                    if period <= max_lags:
                        axes[0, 1].axvline(x=period, color='red', linestyle='--', alpha=0.6)
            axes[0, 1].grid(True, alpha=0.3)
        except Exception:
            axes[0, 1].text(0.5, 0.5, 'PACF Plot Error', ha='center', va='center', transform=axes[0, 1].transAxes)

        first_diff = ts.diff().dropna()
        if len(first_diff) > max_lags:
            try:
                plot_acf(first_diff, lags=max_lags, ax=axes[1, 0], alpha=ACF_ALPHA)
                axes[1, 0].set_title('ACF - First Difference')
                axes[1, 0].grid(True, alpha=0.3)
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'ACF Diff Error', ha='center', va='center', transform=axes[1, 0].transAxes)

            try:
                plot_pacf(first_diff, lags=max_lags, ax=axes[1, 1], alpha=PACF_ALPHA)
                axes[1, 1].set_title('PACF - First Difference')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'PACF Diff Error', ha='center', va='center', transform=axes[1, 0].transAxes)

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    def _comprehensive_synthesis(self, stationarity_results: Dict, 
                                autocorr_results: Dict, 
                                residual_results: Optional[Dict]) -> Dict[str, Any]:
        assessment = stationarity_results.get('assessment', {})
        is_stationary = assessment.get('tests_agree', False)
        stationarity_type = assessment.get('stationarity_type', 'Unknown')

        all_significant_lags = autocorr_results.get('acf_results', {}).get('significant_lags', [])
        sarima_suggestions = autocorr_results.get('sarima_suggestions', {})

        synthesis_report = {
            'stationarity_conclusion': stationarity_type,
            'differencing_needed': not is_stationary,
            'significant_lags': [lag for lag, _ in all_significant_lags[:5]],
            'all_significant_lags_count': len(all_significant_lags),
            'theory_driven_approach': {
                'periods_available': self.theory_driven_params.get('has_periods', False),
                'methodology': sarima_suggestions.get('methodology', 'traditional'),
                'representative_periods': self.representative_periods,
            },
            'sarima_suggestions': sarima_suggestions,
            'seasonal_lags_detected': sarima_suggestions.get('all_seasonal_periods_detected', []),
            'model_recommendations': sarima_suggestions.get('suggested_models', [])
        }
        return synthesis_report

    @staticmethod
    def get_model_recommendations(synthesis_report: Dict[str, Any]) -> List[str]:
        if not synthesis_report:
            return []
        return synthesis_report.get('model_recommendations', [])

    def analyze(self) -> Dict[str, Any]:
        """Main workflow: Stationarity -> Autocorrelation -> Residual -> Synthesis"""
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        if len(target_ts) < 50:
            return {'success': False, 'error': 'Insufficient data'}

        stationarity_results = self._stationarity_diagnostics(target_ts)
        autocorr_results = self._autocorrelation_analysis(target_ts)
        residual_results = self._residual_diagnostics() if self.mstl_results else None
        synthesis = self._comprehensive_synthesis(stationarity_results, autocorr_results, residual_results)

        return {
            'stationarity': stationarity_results,
            'autocorrelation': autocorr_results,
            'residual_diagnostics': residual_results,
            'synthesis': synthesis,
            'theory_driven_approach': autocorr_results.get('theory_driven_params', {}), 
            'component_name': 'StationarityAutocorrelationAnalyzer_Theory_Driven',
            'success': True
        }

    def _stationarity_diagnostics(self, target_ts: pd.Series) -> Dict[str, Any]:
        """Bước 1: Chẩn đoán Tính dừng"""
        adf_results = self._perform_adf_test(target_ts)
        kpss_results = self._perform_kpss_test(target_ts)
        assessment = self._stationarity_assessment(adf_results, kpss_results)

        return {
            'adf_test': adf_results,
            'kpss_test': kpss_results,
            'assessment': assessment,
            'success': True
        }

    def _perform_adf_test(self, ts: pd.Series) -> Dict[str, Any]:
        try:
            result = StationarityTester.test_stationarity(ts)
            return {
                'statistic': result['ADF_statistic'],
                'pvalue': result['ADF_pvalue'],
                'is_stationary': result['ADF_stationary'],
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _perform_kpss_test(self, ts: pd.Series) -> Dict[str, Any]:
        try:
            result = StationarityTester.test_stationarity(ts)
            return {
                'statistic': result['KPSS_statistic'],
                'pvalue': result['KPSS_pvalue'],
                'is_stationary': result['KPSS_stationary'],
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _stationarity_assessment(self, adf: Dict, kpss: Dict) -> Dict[str, Any]:
        adf_stat = adf.get('is_stationary', False)
        kpss_stat = kpss.get('is_stationary', False)

        consensus = build_consensus_result(adf_stat, kpss_stat)
        conclusion = consensus['combined_conclusion']

        return {
            'conclusion': conclusion,
            'stationarity_type': conclusion,
            'tests_agree': adf_stat == kpss_stat,
            'adf_agrees': adf_stat,
            'kpss_agrees': kpss_stat,
            'd_recommendation': consensus['d_recommendation']
        }

    def _visual_stationarity_analysis(self, target_ts: pd.Series, return_fig: bool = True) -> Optional[plt.Figure]:
        window = min(ROLLING_STATS_WINDOW, len(target_ts) // 4)
        rolling_mean = target_ts.rolling(window=window).mean()
        rolling_std = target_ts.rolling(window=window).std()

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stationarity Visual Diagnostics', fontsize=14, fontweight='bold')

        axes[0, 0].plot(target_ts.index, target_ts.values, 'b-', alpha=0.6, label='Original')
        axes[0, 0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, 
                       label=f'Rolling Mean ({window}d)')
        axes[0, 0].set_title('Time Series with Rolling Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(rolling_std.index, rolling_std.values, 'g-', linewidth=2)
        axes[0, 1].set_title(f'Rolling Standard Deviation ({window}d)')
        axes[0, 1].grid(True, alpha=0.3)

        first_diff = target_ts.diff().dropna()
        axes[1, 0].plot(first_diff.index, first_diff.values, 'purple', alpha=0.7)
        axes[1, 0].set_title('First Difference')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(target_ts.values, bins=30, alpha=0.7, label='Original', density=True)
        axes[1, 1].hist(first_diff.values, bins=30, alpha=0.7, label='First Difference', density=True)
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    def _residual_diagnostics(self) -> Dict[str, Any]:
        if not self.mstl_results or 'resid' not in self.mstl_results:
            return {'success': False, 'error': 'No MSTL results'}

        residual = self.mstl_results['resid'].dropna()
        if len(residual) < 50:
            return {'success': False, 'error': 'Insufficient residual data'}

        return self.diagnose_residuals(residual)

    def diagnose_residuals(self, residual_series: pd.Series, **kwargs) -> Dict[str, Any]:
        """Wrapper calling run_residual_diagnostics() pure function."""
        try:
            diag_results = run_residual_diagnostics(residual_series)
            diag_results['success'] = True
            diag_results['residuals'] = residual_series.dropna()
            return diag_results
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def plot_residual_diagnostics(self,
                                 residual_results: Dict[str, Any],
                                 figsize: Tuple[int, int] = (15, 10),
                                 hist_bins: int = 50,
                                 layout: str = '2x2',
                                 return_fig: bool = True) -> Optional[plt.Figure]:
        if not residual_results.get('success', False):
            print("❌ Cannot plot diagnostics: Residual analysis failed")
            return None

        residuals = residual_results['residuals']
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Residual Diagnostic Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        axes[0].plot(residuals.index, residuals.values, 'purple', alpha=0.7, linewidth=1)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Residuals Over Time')
        axes[0].set_ylabel('Residual')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals.values, bins=hist_bins, alpha=0.7, color='purple', density=True, edgecolor='black')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)

        try:
            plot_acf(residuals, ax=axes[2], lags=RESIDUAL_ACF_NLAGS, alpha=RESIDUAL_ACF_ALPHA)
            axes[2].set_title(f'ACF of Residuals (lags={RESIDUAL_ACF_NLAGS})')
        except Exception as e:
            axes[2].set_title('ACF Plot Error')

        try:
            probplot(residuals.values, dist="norm", plot=axes[3])
            axes[3].set_title('Q-Q Plot (Normal)')
        except Exception as e:
            axes[3].set_title('Q-Q Plot Error')

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    def plot_stationarity_diagnostics(self, target_ts: Optional[pd.Series] = None,
                                      return_fig: bool = True) -> Optional[plt.Figure]:
        if target_ts is None:
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()
        return self._visual_stationarity_analysis(target_ts, return_fig=return_fig)

    def plot_autocorrelation(self, target_ts: Optional[pd.Series] = None,
                             max_lags: Optional[int] = None,
                             return_fig: bool = True) -> Optional[plt.Figure]:
        if target_ts is None:
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()
        if max_lags is None:
            max_lags = ACF_NLAGS
        return self._theory_driven_autocorrelation_visualization(target_ts, max_lags, "Original Series", return_fig=return_fig)

    def plot_residual(self, residual_results: Optional[Dict[str, Any]] = None,
                      return_fig: bool = True) -> Optional[plt.Figure]:
        if residual_results is None:
            if not self.mstl_results or 'resid' not in self.mstl_results:
                return None
            residual = self.mstl_results['resid'].dropna()
            residual_results = self.diagnose_residuals(residual)
        return self.plot_residual_diagnostics(residual_results, return_fig=return_fig)


def analyze_stationarity_autocorrelation(df: pd.DataFrame,
                                        target_col: str = 'Lượng mưa',
                                        date_col: str = 'Ngày',
                                        mstl_results: Optional[Dict] = None,
                                        representative_periods: Optional[List[int]] = None) -> Dict[str, Any]:
    analyzer = StationarityAutocorrelationAnalyzer(df, target_col, date_col, mstl_results, representative_periods)
    return analyzer.analyze()


get_model_recommendations = StationarityAutocorrelationAnalyzer.get_model_recommendations
