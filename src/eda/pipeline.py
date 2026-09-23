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

    Contains the specific values that downstream steps (FeatureBuilder,
    model constructors, RecursiveForecaster) actually consume.  Raw
    analyzer outputs are stored in ``raw_results`` for the report/notebook.
    """

    suggested_sarima_order: Tuple[int, int, int] = (2, 1, 2)
    suggested_seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 1, 1, 7)
    vif_filtered_features: List[str] = field(default_factory=list)
    validated_rain_threshold: float = 0.1
    wet_season_months: List[int] = field(default_factory=list)
    representative_periods: List[int] = field(default_factory=list)
    high_skew_features: List[str] = field(default_factory=list)
    extreme_event_thresholds: Dict[str, float] = field(default_factory=dict)
    rain_threshold_justification: str = ""
    # Intermittency (Syntetos-Boylan) — P4
    intermittency_classification: str = ""   # SMOOTH/ERRATIC/INTERMITTENT/LUMPY
    intermittency_model_rec: str = ""
    intermittency_adi: Optional[float] = None
    intermittency_cv2: Optional[float] = None
    raw_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """Convert EDAReport to dictionary.

        Args:
            include_raw: If True, include raw_results dictionary. If False,
                only return JSON-serializable core configuration fields.
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
            "rain_threshold_justification": str(self.rain_threshold_justification),
            "intermittency_classification": str(self.intermittency_classification),
            "intermittency_model_rec": str(self.intermittency_model_rec),
            "intermittency_adi": float(self.intermittency_adi) if self.intermittency_adi is not None else None,
            "intermittency_cv2": float(self.intermittency_cv2) if self.intermittency_cv2 is not None else None,
        }
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

        return cls(
            suggested_sarima_order=sarima_order,
            suggested_seasonal_order=seasonal_order,
            vif_filtered_features=data.get("vif_filtered_features", []),
            validated_rain_threshold=float(data.get("validated_rain_threshold", 0.1)),
            wet_season_months=[int(m) for m in data.get("wet_season_months", [])],
            representative_periods=[int(p) for p in data.get("representative_periods", [])],
            high_skew_features=data.get("high_skew_features", []),
            extreme_event_thresholds={k: float(v) for k, v in data.get("extreme_event_thresholds", {}).items()},
            rain_threshold_justification=data.get("rain_threshold_justification", ""),
            intermittency_classification=data.get("intermittency_classification", ""),
            intermittency_model_rec=data.get("intermittency_model_rec", ""),
            intermittency_adi=data.get("intermittency_adi"),
            intermittency_cv2=data.get("intermittency_cv2"),
            raw_results=data.get("raw_results", {}),
        )

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save JSON-serializable report configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_raw=False), f, indent=2, ensure_ascii=False)
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
        validated_threshold: float = 0.1,
        threshold_justification: str = "",
        representative_periods: Optional[List[int]] = None,
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
            fft_res = temporal.get('fft', {})
            detected = fft_res.get('detected_periods', [])
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
            thresholds = extreme.get('thresholds', {})
            if thresholds:
                extreme_thresholds = {
                    k: float(v) for k, v in thresholds.items()
                    if isinstance(v, (int, float))
                }

        # --- Intermittency (Syntetos-Boylan) ---
        intermittency = {}
        if distribution:
            intermittency = distribution.get('intermittency', {})

        return cls(
            suggested_sarima_order=sarima_order,
            suggested_seasonal_order=seasonal_order,
            vif_filtered_features=vif_features,
            validated_rain_threshold=validated_threshold,
            wet_season_months=wet_months,
            representative_periods=periods,
            high_skew_features=high_skew,
            extreme_event_thresholds=extreme_thresholds,
            rain_threshold_justification=threshold_justification,
            intermittency_classification=intermittency.get('classification', ''),
            intermittency_model_rec=intermittency.get('model_recommendation', ''),
            intermittency_adi=intermittency.get('adi'),
            intermittency_cv2=intermittency.get('cv2'),
            raw_results={
                'distribution': distribution,
                'correlation': correlation,
                'temporal': temporal,
                'stationarity': stationarity,
                'extreme': extreme,
            },
        )


# ======================================================================
# EDAPipeline
# ======================================================================

class EDAPipeline:
    """Run all 5 EDA analyzers in correct dependency order.

    Usage::

        pipeline = EDAPipeline(df, target_col='Lượng mưa', date_col='Ngày')
        report = pipeline.run()
        # report.suggested_sarima_order, report.vif_filtered_features, etc.

    Period selection
    ~~~~~~~~~~~~~~~~
    ``representative_periods`` control MSTL decomposition and SARIMA
    seasonal-order suggestion.  Two strategies:

    - ``'domain'`` (default): Use fixed, climatologically-motivated periods.
      Default ``[7, 30, 122, 365]`` covers weekly cycle, MJO (~30d),
      monsoon onset (~122d), and annual cycle.  This is what the notebook
      currently uses (DS.ipynb Cell[26]: ``analysis_periods = [7, 20, 122, 365]``).
    - ``'fft'``: Use FFT-detected top-N periods from
      ``TemporalStructureAnalyzer.analyze_all()`` — data-driven, but
      rainfall FFT spectra are noisy and may capture spurious peaks.

    When ``period_selection='fft'``, the ``representative_periods`` fed to
    ``StationarityAutocorrelationAnalyzer`` come from FFT, not domain
    knowledge — despite the field being labelled 'theory-driven' in
    ``Stationarity.py``.  This is a known naming inconsistency documented
    here for transparency.
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

    def run(self) -> EDAReport:
        """Execute all analyzers and produce a structured EDAReport."""
        print("=" * 70)
        print("🔬 EDAPipeline: running all 5 analyzers in dependency order")
        print("=" * 70)

        # ---- 1. Distribution ----
        print("\n📊 [1/5] DistributionAnalyzer")
        distribution_results = self._run_distribution()

        # ---- 2. Correlation ----
        print("\n📊 [2/5] CorrelationAnalyzer")
        correlation_results = self._run_correlation()

        # ---- 3. Temporal (produces mstl_results for step 4) ----
        print("\n📊 [3/5] TemporalStructureAnalyzer")
        temporal_results = self._run_temporal()

        # ---- 4. Stationarity (depends on temporal.mstl_results) ----
        print("\n📊 [4/5] StationarityAutocorrelationAnalyzer")
        stationarity_results = self._run_stationarity(temporal_results)

        # ---- 5. Extreme Events ----
        print("\n📊 [5/5] ExtremeEventsAnalyzer")
        extreme_results = self._run_extreme()

        # ---- Validate rain threshold ----
        threshold, justification = self._validate_rain_threshold(
            distribution_results
        )

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
            threshold_justification=justification,
            representative_periods=periods,
        )

        print("\n" + "=" * 70)
        print("✅ EDAPipeline complete — EDAReport ready")
        print(f"   SARIMA order: {report.suggested_sarima_order}")
        print(f"   Seasonal order: {report.suggested_seasonal_order}")
        print(f"   Rain threshold: {report.validated_rain_threshold} mm "
              f"({report.rain_threshold_justification})")
        print(f"   VIF-filtered features: {len(report.vif_filtered_features)}")
        print(f"   Wet season months: {report.wet_season_months}")
        if report.intermittency_classification:
            print(f"   Intermittency: {report.intermittency_classification} "
                  f"(ADI={report.intermittency_adi:.2f}, "
                  f"CV²={report.intermittency_cv2:.2f})")
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

    # ------------------------------------------------------------------
    # Rain threshold validation
    # ------------------------------------------------------------------

    def _validate_rain_threshold(
        self,
        distribution_results: Dict[str, Any],
    ) -> Tuple[float, str]:
        """Validate the rain/no-rain threshold against the real distribution.

        Checks where common meteorological thresholds (0.1mm, 0.6mm, 1mm)
        fall relative to the empirical distribution and picks the one that
        best separates rain from no-rain.
        """
        target = self.df[self.target_col].dropna()

        # Fraction of days at various thresholds
        frac_zero = (target == 0).mean()
        frac_below_01 = (target < 0.1).mean()
        frac_below_06 = (target < 0.6).mean()
        frac_below_1 = (target < 1.0).mean()

        print(f"   🌧️ Rain threshold validation:")
        print(f"      Exactly 0 mm: {frac_zero:.1%}")
        print(f"      < 0.1 mm:     {frac_below_01:.1%}")
        print(f"      < 0.6 mm:     {frac_below_06:.1%}")
        print(f"      < 1.0 mm:     {frac_below_1:.1%}")

        # Heuristic: pick the threshold that creates the clearest gap
        # between "no rain" and "rain" categories
        gap_01 = frac_below_01 - frac_zero  # days in (0, 0.1)
        gap_06 = frac_below_06 - frac_below_01  # days in [0.1, 0.6)
        gap_1 = frac_below_1 - frac_below_06  # days in [0.6, 1.0)

        # If very few days fall in (0, 0.1), then 0.1mm is a good threshold
        # (tight gap means most zero-rain days are exactly 0)
        if gap_01 < 0.02:
            threshold = 0.1
            justification = (
                f"0.1mm chosen — only {gap_01:.1%} of days fall in (0, 0.1mm), "
                f"indicating a clean separation at this threshold"
            )
        elif gap_06 < 0.03:
            threshold = 0.6
            justification = (
                f"0.6mm chosen (Vietnamese meteorological standard for 'trace rain') "
                f"— {frac_below_06:.1%} of days are below this threshold"
            )
        else:
            threshold = 0.1
            justification = (
                f"0.1mm used as default — no strong bimodal gap detected. "
                f"Consider domain expert review."
            )

        print(f"      ➤ Validated threshold: {threshold} mm ({justification})")

        return threshold, justification
