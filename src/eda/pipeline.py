"""
EDA orchestrator and structured report.

``EDAPipeline`` runs the five analysis modules in the correct dependency
order and produces an ``EDAReport`` that model constructors consume
programmatically — no more eyeballing printed output.

Dependency order
----------------
1. DistributionAnalyzer  — no deps, validates rain threshold
2. CorrelationAnalyzer   — no deps, produces VIF-filtered features
3. TemporalStructureAnalyzer — no deps, produces mstl_results + seasonal periods
4. StationarityAutocorrelationAnalyzer — takes mstl_results from (3)
5. ExtremeEventsAnalyzer — no deps
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from ..config.resolve import resolve_target_col, resolve_date_col

logger = logging.getLogger(__name__)


# ======================================================================
# EDAReport dataclass
# ======================================================================

@dataclass
class EDAReport:
    """Structured output from EDAPipeline.run().

    Contains pure numerical values that downstream steps (FeatureBuilder,
    model constructors, RecursiveForecaster) consume. Raw analyzer outputs
    are stored in ``raw_results`` for notebooks/audit.
    """

    suggested_sarima_order: Tuple[int, int, int] = (2, 1, 2)
    suggested_seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 1, 1, 7)
    vif_filtered_features: List[str] = field(default_factory=list)
    validated_rain_threshold: float = 0.0
    wet_season_months: List[int] = field(default_factory=list)
    representative_periods: List[int] = field(default_factory=list)
    high_skew_features: List[str] = field(default_factory=list)
    extreme_event_thresholds: Dict[str, float] = field(default_factory=dict)
    # Intermittency (Syntetos-Boylan) pure numerical statistics
    intermittency_adi: Optional[float] = None
    intermittency_cv2: Optional[float] = None
    # Optional diagnostic labels strictly isolated from core numerical fields
    diagnostic_labels_optional: Dict[str, str] = field(default_factory=dict)
    raw_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def intermittency_classification(self) -> str:
        """Optional label for backward compatibility; read from diagnostic_labels_optional."""
        return self.diagnostic_labels_optional.get("intermittency_classification", "")

    def to_dict(self, include_raw: bool = False, include_diagnostics: bool = False) -> Dict[str, Any]:
        """Convert EDAReport to dictionary containing pure numerical configurations.

        Args:
            include_raw: If True, include raw_results dictionary.
            include_diagnostics: If True, include diagnostic_labels_optional dictionary.
        """
        d: Dict[str, Any] = {
            "suggested_sarima_order": list(self.suggested_sarima_order),
            "suggested_seasonal_order": list(self.suggested_seasonal_order) if self.suggested_seasonal_order else None,
            "vif_filtered_features": list(self.vif_filtered_features),
            "validated_rain_threshold": float(self.validated_rain_threshold),
            "wet_season_months": [int(m) for m in self.wet_season_months],
            "representative_periods": [int(p) for p in self.representative_periods],
            "high_skew_features": list(self.high_skew_features),
            "extreme_event_thresholds": {k: float(v) for k, v in self.extreme_event_thresholds.items()},
            "intermittency_adi": float(self.intermittency_adi) if self.intermittency_adi is not None else None,
            "intermittency_cv2": float(self.intermittency_cv2) if self.intermittency_cv2 is not None else None,
        }
        if include_diagnostics and self.diagnostic_labels_optional:
            d["diagnostic_labels_optional"] = dict(self.diagnostic_labels_optional)
        if include_raw:
            d["raw_results"] = self.raw_results
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EDAReport":
        """Reconstruct EDAReport from dictionary."""
        sarima_order = tuple(data.get("suggested_sarima_order", (2, 1, 2)))
        seasonal_order = data.get("suggested_seasonal_order")
        if seasonal_order is not None:
            seasonal_order = tuple(seasonal_order)

        diag_labels = dict(data.get("diagnostic_labels_optional", {}))
        # Support reading legacy JSON files that might have flat fields
        if "intermittency_classification" in data and "intermittency_classification" not in diag_labels:
            diag_labels["intermittency_classification"] = data["intermittency_classification"]

        return cls(
            suggested_sarima_order=sarima_order,
            suggested_seasonal_order=seasonal_order,
            vif_filtered_features=data.get("vif_filtered_features", []),
            validated_rain_threshold=float(data.get("validated_rain_threshold", 0.0)),
            wet_season_months=[int(m) for m in data.get("wet_season_months", [])],
            representative_periods=[int(p) for p in data.get("representative_periods", [])],
            high_skew_features=data.get("high_skew_features", []),
            extreme_event_thresholds={k: float(v) for k, v in data.get("extreme_event_thresholds", {}).items()},
            intermittency_adi=data.get("intermittency_adi"),
            intermittency_cv2=data.get("intermittency_cv2"),
            diagnostic_labels_optional=diag_labels,
            raw_results=data.get("raw_results", {}),
        )

    def to_json(self, filepath: Union[str, Path], include_diagnostics: bool = False) -> None:
        """Save JSON-serializable report configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_raw=False, include_diagnostics=include_diagnostics), f, indent=2, ensure_ascii=False)
        logger.info("Saved EDAReport to %s", path)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "EDAReport":
        """Load EDAReport from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_raw(
        cls,
        distribution: Dict[str, Any],
        correlation: Dict[str, Any],
        temporal: Dict[str, Any],
        stationarity: Dict[str, Any],
        extreme: Dict[str, Any],
        validated_threshold: float = 0.0,
        representative_periods: Optional[List[int]] = None,
        diagnostic_labels: Optional[Dict[str, str]] = None,
    ) -> "EDAReport":
        """Build an EDAReport from the raw outputs of the five analyzers."""

        # --- SARIMA order ---
        sarima_order = (3, 0, 3)  # default fallback
        seasonal_order = (1, 1, 1, 7)
        if stationarity:
            synthesis = stationarity.get('synthesis', {})
            sarima_suggestions = synthesis.get('sarima_suggestions', {})
            if sarima_suggestions:
                order_list = sarima_suggestions.get('orders', [])
                if order_list:
                    sarima_order = tuple(order_list[0].get('order', (3, 0, 3)))
                seasonal_list = sarima_suggestions.get('seasonal_orders', [])
                if seasonal_list:
                    seasonal_order = tuple(seasonal_list[0].get('order', (1, 1, 1, 7)))

        # --- VIF filtered features ---
        vif_features: List[str] = []
        if correlation:
            vif_df = correlation.get('multicollinearity_results', correlation.get('multicollinearity', None))
            if isinstance(vif_df, pd.DataFrame):
                vif_col = 'VIF_Score' if 'VIF_Score' in vif_df.columns else ('VIF' if 'VIF' in vif_df.columns else None)
                if vif_col and 'Feature' in vif_df.columns:
                    vif_features = vif_df[vif_df[vif_col] < 10]['Feature'].tolist()

        # --- Wet season months ---
        wet_months: List[int] = [5, 6, 7, 8, 9, 10, 11]
        if temporal:
            seasonal = temporal.get('seasonal', {})
            wet = seasonal.get('wet_months', None)
            if wet:
                wet_months = list(wet)

        # --- Representative periods ---
        periods: List[int] = representative_periods or []
        if not periods and temporal:
            fft_res = temporal.get('fft', temporal.get('frequency_analysis', {}))
            detected = fft_res.get('detected_periods', fft_res.get('dominant_periods', []))
            if detected:
                periods = [int(p) for p in detected[:5]]

        # --- High-skew features ---
        high_skew: List[str] = []
        if distribution and isinstance(distribution, dict):
            stats_df = distribution.get('descriptive_stats', None)
            if isinstance(stats_df, pd.DataFrame) and 'Skewness' in stats_df.columns:
                high_skew = stats_df[abs(stats_df['Skewness']) > 2].index.tolist()

        # --- Extreme event thresholds ---
        extreme_thresholds: Dict[str, float] = {}
        if extreme:
            thresholds = extreme.get('thresholds', extreme.get('extreme_definition', {}).get('thresholds', {}))
            if thresholds:
                extreme_thresholds = {
                    k: float(v) for k, v in thresholds.items()
                    if isinstance(v, (int, float))
                }

        # --- Intermittency (Syntetos-Boylan) ---
        intermittency = {}
        if distribution:
            intermittency = distribution.get('intermittency', {})

        diag_labels = dict(diagnostic_labels or {})
        if 'classification' in intermittency and 'intermittency_classification' not in diag_labels:
            diag_labels['intermittency_classification'] = intermittency['classification']

        return cls(
            suggested_sarima_order=sarima_order,
            suggested_seasonal_order=seasonal_order,
            vif_filtered_features=vif_features,
            validated_rain_threshold=float(validated_threshold),
            wet_season_months=wet_months,
            representative_periods=periods,
            high_skew_features=high_skew,
            extreme_event_thresholds=extreme_thresholds,
            intermittency_adi=intermittency.get('adi'),
            intermittency_cv2=intermittency.get('cv2'),
            diagnostic_labels_optional=diag_labels,
            raw_results={
                'distribution': distribution,
                'correlation': correlation,
                'temporal': temporal,
                'stationarity': stationarity,
                'extreme': extreme,
            },
        )


# ======================================================================
# Rain Threshold Comparison Helper
# ======================================================================

def compare_rain_thresholds(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    candidate_thresholds: Optional[List[float]] = None,
    inclusive_wet: bool = True,
) -> pd.DataFrame:
    """Compute empirical distribution statistics across candidate rain thresholds.

    Empirically describes rainfall occurrence, class balance, and conditional intensity
    across candidate thresholds without selecting or freezing any threshold in EDA.

    Architectural boundaries:
        - Primary forecasting target: continuous daily rainfall amount Y_t (mm).
          Statistical, ML, Tweedie, and Neural models forecast continuous Y_t directly,
          fully independent of any rainfall threshold.
        - Threshold tau is auxiliary with only two downstream uses:
          1. Descriptive sensitivity analysis in EDA (characterizing occurrence structure).
          2. Two-stage Hurdle decomposition (Stage 1 occurrence & Stage 2 conditional amount).
        - Canonical project wet-day convention:
          Wet day is defined as Y >= 1.0 mm/day (WMO RR1 standard), and dry day as Y < 1.0 mm/day.
          This definition is unified across 01.1 threshold analysis, 01.2 intermittency,
          Hurdle Stage 1 occurrence, and secondary rain-day diagnostics.
        - Feature Engineering: No mandatory threshold-derived features (is_wet, wet_lag, etc.)
          in baseline levels F0-F3.
        - Metrics: Primary metrics (MAE, RMSE, CR-MAE@7) remain threshold-independent.
          Rain-day metrics (RainDay-MAE_tau) are secondary diagnostics.
        - Nomenclature: Observations in 0 < Y <= 0.1 mm represent a 'near-zero / very-light
          precipitation regime' rather than an unverified label like sensor noise or dew.
        - Notation distinction:
          * Rainfall threshold (tau): physical rainfall event definition (e.g. tau = 1.0 mm/day).
          * Classification threshold (c): Stage 1 probability cutoff (P(Wet | X) > c).

    Args:
        df: DataFrame containing the target variable.
        target_col: Target column name.
        candidate_thresholds: Configurable list of thresholds in mm
            (default [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]).
        inclusive_wet: If True (default), defines Wet as Y >= tau (and Dry as Y < tau for tau > 0,
            or Y == 0 for tau = 0.0), aligning with project canonical wet-day convention.
            If False, defines No-Rain as Y <= tau and Rain as Y > tau.

    Returns:
        pd.DataFrame with columns:
            ['threshold_mm', 'no_rain_condition', 'no_rain_days', 'rain_days',
             'no_rain_pct', 'rain_pct', 'exact_threshold_days', 'incremental_gap_pct',
             'rain_mean_conditional', 'rain_median_conditional']
    """
    col = resolve_target_col(target_col)
    target = df[col].dropna()
    total_days = len(target)

    if candidate_thresholds is None:
        candidate_thresholds = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    candidate_thresholds = sorted(list(set(candidate_thresholds)))
    records = []
    prev_no_rain_frac = None

    for th in candidate_thresholds:
        exact_mask = (target == th)
        exact_count = int(exact_mask.sum())

        if inclusive_wet:
            if th == 0.0:
                no_rain_mask = (target == 0.0)
                rain_mask = (target > 0.0)
                cond_desc = "== 0.0 mm"
            else:
                no_rain_mask = (target < th)
                rain_mask = (target >= th)
                cond_desc = f"< {th:.1f} mm"
        else:
            no_rain_mask = (target <= th)
            rain_mask = (target > th)
            cond_desc = f"<= {th:.1f} mm"

        no_rain_count = int(no_rain_mask.sum())
        rain_count = int(rain_mask.sum())

        no_rain_frac = float(no_rain_count / total_days) if total_days > 0 else 0.0
        rain_frac = float(rain_count / total_days) if total_days > 0 else 0.0

        if prev_no_rain_frac is not None:
            gap_pct = (no_rain_frac - prev_no_rain_frac) * 100.0
        else:
            gap_pct = 0.0

        # Conditional rainfall intensity: statistics computed on wet population
        rain_amounts = target[rain_mask]
        if len(rain_amounts) > 0:
            rain_mean_cond = float(rain_amounts.mean())
            rain_median_cond = float(rain_amounts.median())
        else:
            rain_mean_cond = 0.0
            rain_median_cond = 0.0

        records.append({
            "threshold_mm": float(th),
            "no_rain_condition": cond_desc,
            "no_rain_days": no_rain_count,
            "rain_days": rain_count,
            "no_rain_pct": round(no_rain_frac * 100.0, 2),
            "rain_pct": round(rain_frac * 100.0, 2),
            "exact_threshold_days": exact_count,
            "incremental_gap_pct": round(gap_pct, 2),
            "rain_mean_conditional": round(rain_mean_cond, 2),
            "rain_median_conditional": round(rain_median_cond, 2),
        })
        prev_no_rain_frac = no_rain_frac

    return pd.DataFrame(records)


# ======================================================================
# EDAPipeline
# ======================================================================

class EDAPipeline:
    """Run all 5 EDA analyzers in correct dependency order.

    Usage::

        pipeline = EDAPipeline(df, target_col='Lượng mưa', date_col='Ngày')
        report = pipeline.run()
        # report.suggested_sarima_order, report.vif_filtered_features, etc.

    Can also receive precomputed results from individual analyzers:
        report = pipeline.run(distribution=dist_res, correlation=corr_res, ...)

    Period selection
    ~~~~~~~~~~~~~~~~
    ``representative_periods`` control MSTL decomposition and SARIMA
    seasonal-order suggestion.  Two strategies:

    - ``'domain'`` (default): Use fixed periods.
      Default ``[7, 30, 122, 365]`` covers candidate weekly (7d), monthly (30d),
      seasonal (~122d), and annual (365d) cycles.
    - ``'fft'``: Use FFT-detected top-N periods from
      ``TemporalStructureAnalyzer.analyze_all()`` — data-driven, but
      rainfall FFT spectra are noisy and may capture spurious peaks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        date_col: str = None,
        period_selection: str = 'domain',
        domain_periods: List[int] = None,
    ):
        """
        Args:
            df: Weather data DataFrame.
            target_col: Target column name (default from Config).
            date_col: Date column name (default from Config).
            period_selection: ``'domain'`` (fixed periods) or ``'fft'``
                (FFT-detected).  See class docstring.
            domain_periods: Custom period list when ``period_selection='domain'``.
                Defaults to ``[7, 30, 122, 365]``.
        """
        self.df = df.copy()
        self.target_col = resolve_target_col(target_col)
        self.date_col = resolve_date_col(date_col)
        self.period_selection = period_selection
        self.domain_periods = domain_periods or [7, 30, 122, 365]

    def run(
        self,
        distribution: Optional[Dict[str, Any]] = None,
        correlation: Optional[Dict[str, Any]] = None,
        temporal: Optional[Dict[str, Any]] = None,
        stationarity: Optional[Dict[str, Any]] = None,
        extreme: Optional[Dict[str, Any]] = None,
        validated_threshold: Optional[float] = None,
    ) -> EDAReport:
        """Execute all analyzers (or reuse precomputed) and produce a structured EDAReport.

        Args:
            distribution: Precomputed DistributionAnalyzer results (optional).
            correlation: Precomputed CorrelationAnalyzer results (optional).
            temporal: Precomputed TemporalStructureAnalyzer results (optional).
            stationarity: Precomputed StationarityAutocorrelationAnalyzer results (optional).
            extreme: Precomputed ExtremeEventsAnalyzer results (optional).
            validated_threshold: Operational rain threshold (default 0.0). Set explicitly
                by the user based on empirical comparison table (compare_rain_thresholds).
        """
        print("=" * 70)
        print("🔬 EDAPipeline: aggregating EDA results into structured EDAReport")
        print("=" * 70)

        # ---- 1. Distribution ----
        if distribution is not None:
            print("\n📊 [1/5] DistributionAnalyzer (reusing precomputed results)")
            distribution_results = distribution
        else:
            print("\n📊 [1/5] DistributionAnalyzer")
            distribution_results = self._run_distribution()

        # ---- 2. Correlation ----
        if correlation is not None:
            print("\n📊 [2/5] CorrelationAnalyzer (reusing precomputed results)")
            correlation_results = correlation
        else:
            print("\n📊 [2/5] CorrelationAnalyzer")
            correlation_results = self._run_correlation()

        # ---- 3. Temporal (produces mstl_results for step 4) ----
        if temporal is not None:
            print("\n📊 [3/5] TemporalStructureAnalyzer (reusing precomputed results)")
            temporal_results = temporal
        else:
            print("\n📊 [3/5] TemporalStructureAnalyzer")
            temporal_results = self._run_temporal()

        # ---- 4. Stationarity (depends on temporal.mstl_results) ----
        if stationarity is not None:
            print("\n📊 [4/5] StationarityAutocorrelationAnalyzer (reusing precomputed results)")
            stationarity_results = stationarity
        else:
            print("\n📊 [4/5] StationarityAutocorrelationAnalyzer")
            stationarity_results = self._run_stationarity(temporal_results)

        # ---- 5. Extreme Events ----
        if extreme is not None:
            print("\n📊 [5/5] ExtremeEventsAnalyzer (reusing precomputed results)")
            extreme_results = extreme
        else:
            print("\n📊 [5/5] ExtremeEventsAnalyzer")
            extreme_results = self._run_extreme()

        # ---- Operational rain threshold ----
        # Do NOT auto-compute or auto-assign a threshold.
        # User explicitly configures validated_threshold based on empirical comparison table.
        threshold = float(validated_threshold) if validated_threshold is not None else 0.0
        if validated_threshold is not None:
            print(f"\n🌧️ Operational rain threshold: {threshold} mm (user-specified)")
        else:
            print("\n🌧️ Operational rain threshold: not specified (0.0 mm default; user should set explicitly)")

        # Determine representative periods for report
        if self.period_selection == 'domain':
            periods = self.domain_periods
        elif self.period_selection == 'fft':
            fft_res = temporal_results.get('fft', {}) if temporal_results else {}
            detected = fft_res.get('detected_periods', [])
            periods = [int(p) for p in detected[:5]] if detected else self.domain_periods
        else:
            periods = self.domain_periods

        # ---- Build report ----
        report = EDAReport.from_raw(
            distribution=distribution_results,
            correlation=correlation_results,
            temporal=temporal_results,
            stationarity=stationarity_results,
            extreme=extreme_results,
            validated_threshold=threshold,
            representative_periods=periods,
        )

        print("\n" + "=" * 70)
        print("✅ EDAPipeline complete — EDAReport ready")
        print(f"   SARIMA order: {report.suggested_sarima_order}")
        print(f"   Seasonal order: {report.suggested_seasonal_order}")
        print(f"   Rain threshold: {report.validated_rain_threshold} mm")
        print(f"   VIF-filtered features: {len(report.vif_filtered_features)}")
        print(f"   Wet season months: {report.wet_season_months}")
        if report.intermittency_adi is not None and report.intermittency_cv2 is not None:
            print(f"   Intermittency stats: ADI={report.intermittency_adi:.2f}, "
                  f"CV²={report.intermittency_cv2:.2f}")
        print("=" * 70)

        return report

    # ------------------------------------------------------------------
    # Individual analyzer runners (wrap try/except for robustness)
    # ------------------------------------------------------------------

    def _run_distribution(self) -> Dict[str, Any]:
        try:
            from .DistributionAnalysis import DistributionAnalyzer
            analyzer = DistributionAnalyzer(self.df, target_col=self.target_col)
            results = {}
            results['descriptive_stats'] = analyzer.analyze_descriptive_stats()
            results['target'] = analyzer.analyze_target_variable()
            # Intermittency analysis — threshold will be re-wired after
            # rain threshold validation; for now use default 0.1
            results['intermittency'] = analyzer.analyze_intermittency(
                date_col=self.date_col, threshold=0.1
            )
            return results
        except Exception as e:
            print(f"   ⚠️ DistributionAnalyzer failed: {e}")
            return {}

    def _run_correlation(self) -> Dict[str, Any]:
        try:
            from .CorrelationAnalysis import CorrelationAnalyzer
            analyzer = CorrelationAnalyzer(
                self.df,
                target_col=self.target_col,
                date_col=self.date_col,
            )
            return analyzer.generate_insights_report()
        except Exception as e:
            print(f"   ⚠️ CorrelationAnalyzer failed: {e}")
            return {}

    def _run_temporal(self) -> Dict[str, Any]:
        try:
            from .TemporalAnalysis import TemporalStructureAnalyzer
            analyzer = TemporalStructureAnalyzer(
                self.df,
                target_col=self.target_col,
                date_col=self.date_col,
            )
            return analyzer.analyze_all()
        except Exception as e:
            print(f"   ⚠️ TemporalStructureAnalyzer failed: {e}")
            return {}

    def _run_stationarity(self, temporal_results: Dict) -> Dict[str, Any]:
        try:
            from .Stationarity import StationarityAutocorrelationAnalyzer

            # Extract MSTL results from temporal step
            mstl_results = None
            if temporal_results:
                mstl_results = temporal_results.get('mstl', None)

            # Determine representative periods based on period_selection strategy
            if self.period_selection == 'domain':
                rep_periods = self.domain_periods
            elif self.period_selection == 'fft':
                rep_periods = None
                if temporal_results:
                    fft_res = temporal_results.get('fft', {})
                    detected = fft_res.get('detected_periods', [])
                    if detected:
                        rep_periods = [int(p) for p in detected[:5]]
                if not rep_periods:
                    rep_periods = self.domain_periods
            else:
                rep_periods = self.domain_periods

            analyzer = StationarityAutocorrelationAnalyzer(
                self.df,
                target_col=self.target_col,
                date_col=self.date_col,
                mstl_results=mstl_results,
                representative_periods=rep_periods,
            )
            return analyzer.analyze()
        except Exception as e:
            print(f"   ⚠️ StationarityAutocorrelationAnalyzer failed: {e}")
            return {}

    def _run_extreme(self) -> Dict[str, Any]:
        try:
            from .ExtremeEventAnalysis import ExtremeEventsAnalyzer
            analyzer = ExtremeEventsAnalyzer(
                self.df,
                target_col=self.target_col,
                date_col=self.date_col,
            )
            return analyzer.analyze()
        except Exception as e:
            print(f"   ⚠️ ExtremeEventsAnalyzer failed: {e}")
            return {}

