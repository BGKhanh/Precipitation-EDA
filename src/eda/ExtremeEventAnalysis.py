"""Component 01.6: Extreme Precipitation Events & Tail Risk Analysis.

Strict adherence to Plan v4:
- Dev Train population only (Strict Analysis & Diagnostic Isolation).
- Rule 7 compliance: Elapsed calendar time recurrence intervals (calendar-day gaps).
- Calendar-day uniqueness: Daily series normalized to midnight timestamps.
- Weibull empirical return periods: Daily observation exceedance frequency diagnostics.
- Strict validation: Shared validators rejecting NaN, inf, negatives, and invalid types.
- Pure-function separation with zero empirical assumptions in test/artifact contracts.
- Dual-artifact generation: Canonical Schema v4 (pure domain, no 'success') and Bridge Schema.
"""

from typing import Dict, List, Tuple, Any, Optional, Sequence, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# FROZEN CONFIGURATION CONSTANTS (IMMUTABLE TUPLES & LITERALS)
# =============================================================================
EXTREME_PERCENTILES: tuple[float, ...] = (0.95, 0.99, 0.999)
RECURRENCE_TARGET_KEYS: tuple[str, ...] = ("p95", "p99")
WET_SEASON_MONTHS: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)
DRY_SEASON_MONTHS: tuple[int, ...] = (12, 1, 2, 3, 4)
RETURN_PERIOD_TOP_N: int = 100
DAYS_PER_YEAR: float = 365.25  # Locked constant (R17)
PLOTTING_POSITION_FORMULA: str = "weibull"

TARGET_COLUMN_ALIASES: tuple[str, ...] = (
    "Lượng mưa", "Luong_mua", "rain", "rainfall", "precip", "precipitation",
)
DATE_COLUMN_ALIASES: tuple[str, ...] = (
    "Ngày", "Ngay", "date", "datetime", "timestamp", "ds",
)


# =============================================================================
# SAFE NUMERIC HELPER
# =============================================================================
def round_or_none(value: Optional[Union[float, int]], ndigits: int = 2) -> Optional[float]:
    """Safely round a numeric value or return None if value is None.
    
    Prevents TypeError: type NoneType doesn't define __round__ method.
    """
    if value is None:
        return None
    return round(float(value), ndigits)


# =============================================================================
# FAIL-FAST COLUMN RESOLVER (R15)
# =============================================================================
def resolve_column(
    df: pd.DataFrame,
    alias_list: Sequence[str] = TARGET_COLUMN_ALIASES,
    role: str = "target",
    explicit_col: Optional[str] = None,
) -> str:
    """Fail-fast column resolver with duplicate column detection, case collision check, and no overwriting."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

    # R15: Check duplicate column names in DataFrame
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"DataFrame contains duplicate column labels: {dups}. "
            f"Cannot safely resolve {role} column."
        )

    if explicit_col is not None:
        if explicit_col in df.columns:
            return explicit_col
        raise ValueError(f"Explicit {role} column '{explicit_col}' not found in DataFrame.")

    # 1. Exact case-sensitive match
    exact_matches = [col for col in df.columns if col in alias_list]
    if len(exact_matches) == 1:
        matching_col = exact_matches[0]
        # Check case variant collisions in df (e.g. both 'rain' and 'RAIN' present)
        case_collisions = [c for c in df.columns if c.lower() == matching_col.lower() and c != matching_col]
        if case_collisions:
            raise ValueError(
                f"Column resolver: Ambiguous {role} columns — exact match '{matching_col}' collided with "
                f"case variant(s) {case_collisions}. Rename DataFrame columns to disambiguate."
            )
        return matching_col
    elif len(exact_matches) > 1:
        raise ValueError(
            f"Column resolver: Ambiguous {role} columns — multiple exact matches found: {exact_matches}. "
            f"Specify explicit column or rename DataFrame."
        )

    # 2. Case-insensitive fallback
    alias_lower_set = {a.lower() for a in alias_list}
    ci_matches = [col for col in df.columns if col.lower() in alias_lower_set]
    ci_matches = list(dict.fromkeys(ci_matches))
    if len(ci_matches) == 1:
        return ci_matches[0]
    elif len(ci_matches) > 1:
        raise ValueError(
            f"Column resolver: Ambiguous {role} columns — multiple case-insensitive matches found: {ci_matches}. "
            f"Specify explicit column or rename DataFrame."
        )

    raise ValueError(
        f"Column resolver: No {role} column found. "
        f"Searched aliases: {list(alias_list)}. Available DataFrame columns: {list(df.columns)[:10]}"
    )


_resolve_column = resolve_column


# =============================================================================
# SHARED PRECONDITION VALIDATORS (R11, R12)
# =============================================================================
def _validate_rainfall_series(values: pd.Series) -> None:
    """Validate numeric, finite, non-negative rainfall Series."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"Values must be a pandas Series, got {type(values).__name__}.")
    if len(values) == 0:
        raise ValueError("Cannot process empty rainfall series.")
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise TypeError(f"Values must have numeric, non-boolean dtype, got {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must contain only finite numbers (no NaN, inf, -inf).")
    if not np.all(values >= 0.0):
        raise ValueError("Rainfall values must be non-negative (physical constraint violation).")


def _validate_paired_dates_values(dates: pd.Series, values: pd.Series) -> None:
    """Validate strict temporal alignment, calendar-day uniqueness, and physical contracts."""
    _validate_rainfall_series(values)

    if not isinstance(dates, pd.Series):
        raise TypeError(f"Dates must be a pandas Series, got {type(dates).__name__}.")
    if len(dates) != len(values):
        raise ValueError(f"Length mismatch: dates ({len(dates)}) vs values ({len(values)}).")
    if not dates.index.equals(values.index):
        raise ValueError("Dates and values must have identical index alignment (Option A contract).")
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise TypeError(f"Dates must have datetime64 dtype, got {dates.dtype}.")

    # R11: Enforce calendar-day normalization (midnight timestamps)
    if not dates.equals(dates.dt.normalize()):
        raise ValueError("Dates must be normalized to calendar dates (midnight timestamps, 00:00:00).")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in strictly ascending chronological order.")
    if not dates.is_unique:
        duplicate_dates = dates[dates.duplicated()].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"Dates must contain unique calendar dates. Duplicate dates found: {duplicate_dates[:5]}")


def _validate_threshold_scalar(threshold: Any, name: str = "threshold") -> float:
    """Validate that threshold is a numeric scalar, finite, non-negative, and not boolean (R12)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a numeric scalar (int or float), got {type(threshold).__name__}: {threshold}")
    val = float(threshold)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite (got {val}).")
    if val < 0.0:
        raise ValueError(f"{name} must be non-negative (got {val}).")
    return val


def _validate_thresholds_dict(
    thresholds: dict[str, Any],
    required_keys: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Validate dictionary of thresholds: non-empty dict of validated numeric scalars (R12)."""
    if not isinstance(thresholds, dict) or len(thresholds) == 0:
        raise ValueError("Thresholds must be a non-empty dictionary.")
    validated = {k: _validate_threshold_scalar(v, name=f"Threshold '{k}'") for k, v in thresholds.items()}
    if required_keys:
        for rk in required_keys:
            if rk not in validated:
                raise KeyError(f"Required threshold key '{rk}' not found in thresholds: {list(thresholds.keys())}")
    return validated


def _validate_percentiles_sequence(percentiles: Sequence[Any]) -> tuple[float, ...]:
    """Validate sequence of percentiles: non-empty, numeric scalars, finite, strictly in (0, 1) (R12)."""
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
        raise ValueError("Percentiles must be a non-empty sequence.")
    validated = []
    for p in percentiles:
        if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
            raise TypeError(f"Percentile must be numeric, got {type(p).__name__}: {p}")
        p_val = float(p)
        if not np.isfinite(p_val) or not (0.0 < p_val < 1.0):
            raise ValueError(f"Percentile must be finite float strictly in (0.0, 1.0), got {p_val}")
        validated.append(p_val)
    return tuple(validated)


# =============================================================================
# PURE ANALYTICAL FUNCTIONS
# =============================================================================
def compute_extreme_thresholds(
    series: pd.Series,
    percentiles: Sequence[float] = EXTREME_PERCENTILES,
) -> dict[str, float]:
    """Compute empirical percentile thresholds and maximum record."""
    _validate_rainfall_series(series)
    valid_percentiles = _validate_percentiles_sequence(percentiles)

    thresholds = {}
    for p in valid_percentiles:
        key = f"p{str(round(p * 100, 1)).replace('.', '_')}"
        if key == "p95_0":
            key = "p95"
        elif key == "p99_0":
            key = "p99"
        thresholds[key] = float(series.quantile(p))

    thresholds["max_daily_mm"] = float(series.max())
    return thresholds


def compute_exceedance_counts(
    series: pd.Series,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Compute exceedance counts and percentages for each extreme threshold."""
    _validate_rainfall_series(series)
    valid_thresholds = _validate_thresholds_dict(thresholds)

    total_records = len(series)
    counts = {}
    rates = {}
    for key, th in valid_thresholds.items():
        if key == "max_daily_mm":
            continue
        cnt = int((series > th).sum())
        counts[key] = cnt
        rates[key] = float(cnt / total_records * 100)

    return {
        "counts": counts,
        "rates_pct": rates,
        "total_records": int(total_records),
    }


def compute_calendar_recurrence_intervals(
    dates: pd.Series,
    values: pd.Series,
    thresholds: dict[str, float],
    target_keys: Sequence[str] = RECURRENCE_TARGET_KEYS,
) -> dict[str, Any]:
    """Compute elapsed calendar time recurrence intervals (Rule 7 compliant).
    
    Terminology: Recurrence intervals are calendar-day gaps between exceedance dates.
    Uses explicit date-based differencing: `date_series.diff().dt.days`.
    """
    _validate_paired_dates_values(dates, values)
    valid_thresholds = _validate_thresholds_dict(thresholds, required_keys=target_keys)

    result = {
        "calculation_method": "date_series.diff().dt.days (Rule 7 compliant)",
        "semantic_note": "Calendar-day gaps between exceedance dates; not independent storm recurrence.",
        "intervals": {},
    }

    for key in target_keys:
        th = valid_thresholds[key]
        mask = values > th
        exceed_dates = dates[mask]
        count_events = int(len(exceed_dates))

        # Edge cases: 0 or 1 event yields 0 intervals and None summary statistics (R03)
        if count_events <= 1:
            result["intervals"][key] = {
                "count_events": count_events,
                "count_intervals": 0,
                "mean_days": None,
                "median_days": None,
                "std_days": None,
                "min_days": None,
                "max_days": None,
            }
            continue

        # Calendar elapsed time differencing under Rule 7
        gaps = exceed_dates.diff().dt.days.dropna()
        n_intervals = int(len(gaps))

        # R20: When len(gaps) < 2 (only 1 interval from 2 events), sample std (ddof=1) is undefined -> None
        std_val = float(gaps.std(ddof=1)) if n_intervals >= 2 else None

        result["intervals"][key] = {
            "count_events": count_events,
            "count_intervals": n_intervals,
            "mean_days": float(gaps.mean()),
            "median_days": float(gaps.median()),
            "std_days": std_val,
            "min_days": int(gaps.min()),
            "max_days": int(gaps.max()),
        }

    return result


def compute_seasonal_extreme_distribution(
    dates: pd.Series,
    values: pd.Series,
    p95_threshold: float,
    wet_months: Sequence[int] = WET_SEASON_MONTHS,
    dry_months: Sequence[int] = DRY_SEASON_MONTHS,
) -> dict[str, Any]:
    """Compute monthly breakdown and wet/dry season concentration of extreme daily events."""
    _validate_paired_dates_values(dates, values)
    th_p95 = _validate_threshold_scalar(p95_threshold, name="p95_threshold")

    set_wet = set(wet_months)
    set_dry = set(dry_months)
    if set_wet.intersection(set_dry):
        raise ValueError(f"Wet and dry season months cannot overlap: {set_wet.intersection(set_dry)}")
    if set_wet.union(set_dry) != set(range(1, 13)):
        raise ValueError("Wet and dry season months must cover exactly all 12 calendar months.")

    mask = values > th_p95
    extreme_dates = dates[mask]

    monthly_counts_series = (
        extreme_dates.dt.month.value_counts()
        .reindex(range(1, 13), fill_value=0)
        .sort_index()
    )
    monthly_counts = {int(m): int(cnt) for m, cnt in monthly_counts_series.items()}

    total_extreme = int(mask.sum())
    wet_count = int(sum(monthly_counts[m] for m in wet_months))
    dry_count = int(sum(monthly_counts[m] for m in dry_months))

    # T13: When total_extreme == 0 -> peak_month is None and peak_month_count == 0
    peak_month = int(monthly_counts_series.idxmax()) if total_extreme > 0 else None
    peak_month_count = int(monthly_counts_series.max()) if total_extreme > 0 else 0

    return {
        "monthly_counts": monthly_counts,
        "total_extreme_events_p95": total_extreme,
        "wet_season": {
            "months": list(wet_months),
            "count": wet_count,
            "proportion_pct": float(wet_count / total_extreme * 100) if total_extreme > 0 else 0.0,
        },
        "dry_season": {
            "months": list(dry_months),
            "count": dry_count,
            "proportion_pct": float(dry_count / total_extreme * 100) if total_extreme > 0 else 0.0,
        },
        "peak_month": peak_month,
        "peak_month_count": peak_month_count,
    }


def compute_empirical_return_periods(
    series: pd.Series,
    n_top: int = RETURN_PERIOD_TOP_N,
    formula: str = PLOTTING_POSITION_FORMULA,
) -> dict[str, Any]:
    """Compute empirical return periods using Weibull plotting position T = (N + 1) / m.
    
    Formula is strictly locked to 'weibull' and annual scaling is locked to DAYS_PER_YEAR (365.25) (R17).
    """
    _validate_rainfall_series(series)

    if formula != PLOTTING_POSITION_FORMULA:
        raise ValueError(
            f"Unsupported formula '{formula}'. The 01.6 canonical contract strictly requires '{PLOTTING_POSITION_FORMULA}'."
        )

    # R17: Strict check on n_top: reject bool, reject float, accept int scalar >= 1
    if isinstance(n_top, bool) or not isinstance(n_top, (int, np.integer)):
        raise TypeError(f"n_top must be an integer, got {type(n_top).__name__}: {n_top}")
    n_top_val = int(n_top)
    if n_top_val < 1:
        raise ValueError(f"n_top must be a positive integer >= 1, got {n_top_val}")

    clean = series.sort_values(ascending=False).reset_index(drop=True)
    n_obs = len(clean)

    m_ranks = np.arange(1, n_obs + 1)
    t_days = (n_obs + 1.0) / m_ranks
    t_years = t_days / DAYS_PER_YEAR  # Locked constant

    limit = min(n_top_val, n_obs)
    top_records = []
    for i in range(limit):
        top_records.append({
            "rank": int(m_ranks[i]),
            "rainfall_mm": float(clean.iloc[i]),
            "return_period_days": float(t_days[i]),
            "return_period_years": float(t_years[i]),
        })

    return {
        "formula": formula,
        "formula_description": "Weibull plotting position T = (N + 1) / m",
        "methodology_note": (
            "Return periods are empirical daily-observation-scale exceedance diagnostics "
            "computed using the Weibull plotting position. They represent empirical average recurrence "
            "frequencies of daily rainfall amounts exceeding a given level under daily sampling, "
            "NOT return periods of declustered independent storm events or annual-maximum flood events."
        ),
        "total_observations": int(n_obs),
        "days_per_year": float(DAYS_PER_YEAR),
        "top_daily_observations": top_records,
    }


def export_extreme_events_artifacts(
    canonical_report: dict[str, Any],
    canonical_path: Union[str, Path],
    bridge_paths: Sequence[Union[str, Path]],
) -> tuple[Path, list[Path]]:
    """Export canonical report and backward-compatible bridge artifacts.
    
    Guarantees canonical artifact is pure domain schema and contains NO 'success' wrapper (R19).
    """
    can_p = Path(canonical_path)
    can_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Clean canonical dict: explicitly strip 'success' wrapper if present (R19)
    clean_canonical = {k: v for k, v in canonical_report.items() if k != "success"}

    with open(can_p, "w", encoding="utf-8") as f:
        json.dump(clean_canonical, f, indent=2, ensure_ascii=False)

    # 2. Build Bridge dictionary matching exact legacy contract with round_or_none (R03)
    th = clean_canonical["thresholds"]
    cnt = clean_canonical["counts"]
    rec = clean_canonical["recurrence_intervals"]["intervals"]

    p95_stats = rec.get("p95", {})
    p99_stats = rec.get("p99", {})

    bridge_dict = {
        "__meta_bridge__": {
            "is_bridge": True,
            "canonical_path": str(can_p).replace("\\", "/"),
            "compatibility_target": "01.7_eda_synthesis_report.ipynb / EDAReport",
        },
        "thresholds": {
            "p95": round_or_none(th.get("p95"), 2),
            "p99": round_or_none(th.get("p99"), 2),
            "p99_9": round_or_none(th.get("p99_9"), 2),
            "max_daily_mm": round_or_none(th.get("max_daily_mm"), 2),
        },
        "counts": {
            "p95": int(cnt["p95"]) if "p95" in cnt else None,
            "p99": int(cnt["p99"]) if "p99" in cnt else None,
            "p99_9": int(cnt["p99_9"]) if "p99_9" in cnt else None,
        },
        "recurrence_intervals_calendar_days": {
            "mean_p95_days": round_or_none(p95_stats.get("mean_days"), 2),
            "median_p95_days": round_or_none(p95_stats.get("median_days"), 2),
            "mean_p99_days": round_or_none(p99_stats.get("mean_days"), 2),
            "median_p99_days": round_or_none(p99_stats.get("median_days"), 2),
            "calculation_method": "date_series.diff().dt.days (Rule 7 compliant)",
        },
        "backward_compatibility_disclaimer": (
            "This bridge artifact is maintained solely for legacy compatibility with "
            "01.7 synthesis and EDAReport. Comprehensive diagnostics and metadata are "
            "located in the canonical artifact at reports/eda/01.6_extreme_events_report.json."
        ),
    }

    saved_bridge_paths = []
    for bp in bridge_paths:
        p = Path(bp)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(bridge_dict, f, indent=2, ensure_ascii=False)
        saved_bridge_paths.append(p)

    return can_p, saved_bridge_paths


# =============================================================================
# ANALYZER CLASS (R11, R13, R18, R19)
# =============================================================================
class ExtremeEventsAnalyzer:
    """Extreme Precipitation Events & Tail Risk Diagnostics Analyzer.
    
    Consumes canonical Dev Train data and orchestrates pure analytical functions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ):
        self.df = df.copy()
        self.target_col = resolve_column(self.df, TARGET_COLUMN_ALIASES, role="target", explicit_col=target_col)
        self.date_col = resolve_column(self.df, DATE_COLUMN_ALIASES, role="date", explicit_col=date_col)
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Create paired clean DataFrame sorted chronologically with midnight calendar-day uniqueness (R11)."""
        pair_df = self.df[[self.date_col, self.target_col]].dropna().copy()

        # Datetime conversion with strict error checking
        if not pd.api.types.is_datetime64_any_dtype(pair_df[self.date_col]):
            pair_df[self.date_col] = pd.to_datetime(pair_df[self.date_col], errors="raise")

        # R11: Normalize to midnight calendar dates
        pair_df[self.date_col] = pair_df[self.date_col].dt.normalize()

        # Physical constraints: fail-fast on non-finite or negative values
        target_vals = pair_df[self.target_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(target_vals)):
            raise ValueError(f"Target column '{self.target_col}' contains non-finite values (NaN, inf, -inf).")
        if not np.all(target_vals >= 0.0):
            raise ValueError(f"Target column '{self.target_col}' contains negative rainfall values.")

        # R11: Calendar-day uniqueness check
        if pair_df[self.date_col].duplicated().any():
            dup_dates = pair_df.loc[pair_df[self.date_col].duplicated(), self.date_col].dt.strftime("%Y-%m-%d").unique().tolist()
            raise ValueError(f"Duplicate calendar dates detected in '{self.date_col}': {dup_dates[:5]}")

        # Chronological sort and clean index reset
        pair_df = pair_df.sort_values(self.date_col).reset_index(drop=True)

        self.dates = pair_df[self.date_col]
        self.values = pair_df[self.target_col].astype(float)

    def analyze(self) -> dict[str, Any]:
        """Execute end-to-end extreme precipitation event diagnostics.
        
        Returns pure domain report without 'success' wrapper (R19).
        """
        thresholds = compute_extreme_thresholds(self.values, EXTREME_PERCENTILES)
        exceedance = compute_exceedance_counts(self.values, thresholds)
        recurrence = compute_calendar_recurrence_intervals(
            self.dates, self.values, thresholds, target_keys=RECURRENCE_TARGET_KEYS
        )
        seasonal = compute_seasonal_extreme_distribution(
            self.dates, self.values, thresholds["p95"], WET_SEASON_MONTHS, DRY_SEASON_MONTHS
        )
        return_periods = compute_empirical_return_periods(
            self.values, RETURN_PERIOD_TOP_N, PLOTTING_POSITION_FORMULA
        )

        # R13: Safe skewness & excess kurtosis handling (separate undefined / NaN cases)
        skew_val = float(self.values.skew())
        kurt_val = float(self.values.kurtosis())

        if not np.isfinite(skew_val):
            skew_desc = "undefined/indeterminate skewness"
            skew_str = "NaN"
        elif abs(skew_val) < 1e-4:
            skew_desc = "near-zero skewness"
            skew_str = f"{skew_val:.2f}"
        elif skew_val > 0.0:
            skew_desc = "positive skewness (right-skewed tail)"
            skew_str = f"{skew_val:.2f}"
        else:
            skew_desc = "negative skewness (left-skewed tail)"
            skew_str = f"{skew_val:.2f}"

        if not np.isfinite(kurt_val):
            kurt_desc = "undefined/indeterminate kurtosis"
            kurt_str = "NaN"
        elif abs(kurt_val) < 1e-4:
            kurt_desc = "near-zero excess kurtosis"
            kurt_str = f"{kurt_val:.2f}"
        elif kurt_val > 0.0:
            kurt_desc = "positive excess kurtosis (leptokurtic tail)"
            kurt_str = f"{kurt_val:.2f}"
        else:
            kurt_desc = "negative excess kurtosis (platykurtic tail)"
            kurt_str = f"{kurt_val:.2f}"

        # R13: Safe seasonal peak handling (avoid 'Month None')
        if seasonal["total_extreme_events_p95"] > 0 and seasonal["peak_month"] is not None:
            peak_str = f"Calendar month with highest P95 exceedances: Month {seasonal['peak_month']} ({seasonal['peak_month_count']} occurrences)."
            seasonal_wet_str = (
                f"Observed P95 extreme exceedances in designated wet season (May–Nov): "
                f"{seasonal['wet_season']['proportion_pct']:.1f}% "
                f"({seasonal['wet_season']['count']}/{seasonal['total_extreme_events_p95']} occurrences)."
            )
        else:
            peak_str = "No extreme exceedance events observed above P95 threshold."
            seasonal_wet_str = "No extreme exceedance events observed to evaluate seasonal distribution."

        # R13 & R20: Recurrence intervals reporting
        p95_rec = recurrence["intervals"].get("p95", {})
        mean_p95 = p95_rec.get("mean_days")
        med_p95 = p95_rec.get("median_days")
        if mean_p95 is not None and med_p95 is not None and np.isfinite(mean_p95) and np.isfinite(med_p95):
            rec_str = f"Empirical recurrence intervals for P95 exceedances: mean={mean_p95:.1f}d, median={med_p95:.1f}d (calendar-day gaps)."
        else:
            rec_str = "Insufficient exceedances (< 2 events) to evaluate empirical recurrence statistics."

        tail_synthesis = [
            f"Precipitation distribution on Dev Train has observed skewness = {skew_str} ({skew_desc}) and excess kurtosis = {kurt_str} ({kurt_desc}).",
            seasonal_wet_str,
            peak_str,
            rec_str,
            "Extreme thresholds serve as diagnostic evaluation slices, NOT as predictive regression features.",
        ]

        # Domain report dict (clean domain schema with machine-readable metadata, NO 'success') (R18, R19)
        start_date_str = self.dates.min().strftime("%Y-%m-%d")
        end_date_str = self.dates.max().strftime("%Y-%m-%d")
        n_obs = int(len(self.values))

        domain_report = {
            "metadata": {
                "component": "01.6_extreme_events",
                "split": "dev_train",
                "start_date": start_date_str,
                "end_date": end_date_str,
                "n_observations": n_obs,
                "target_col": self.target_col,
                "date_col": self.date_col,
                "percentiles_evaluated": list(EXTREME_PERCENTILES),
            },
            "thresholds": thresholds,
            "counts": exceedance["counts"],
            "rates_pct": exceedance["rates_pct"],
            "total_records": exceedance["total_records"],
            "recurrence_intervals": recurrence,
            "seasonal_patterns": seasonal,
            "empirical_return_periods": return_periods,
            "tail_risk_synthesis": tail_synthesis,
        }

        return domain_report

    def plot_extreme_events(self, return_fig: bool = True) -> Optional[plt.Figure]:
        """Generate standardized 4-panel diagnostic visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("01.6 Extreme Precipitation Diagnostics & Tail Risk", fontsize=15, fontweight="bold")

        thresholds = compute_extreme_thresholds(self.values, EXTREME_PERCENTILES)
        p95 = thresholds["p95"]
        seasonal = compute_seasonal_extreme_distribution(self.dates, self.values, p95)
        rp_data = compute_empirical_return_periods(self.values, RETURN_PERIOD_TOP_N)

        # Panel 1: Monthly count bar chart with Wet/Dry differentiation
        months = list(range(1, 13))
        counts = [seasonal["monthly_counts"][m] for m in months]
        colors = ["teal" if m in WET_SEASON_MONTHS else "orange" for m in months]
        axes[0, 0].bar(months, counts, color=colors, alpha=0.75, edgecolor="black")
        axes[0, 0].set_title(f"Panel 1: Extreme Exceedance Dates by Month (> P95: {p95:.1f} mm)", fontweight="bold")
        axes[0, 0].set_xlabel("Calendar Month")
        axes[0, 0].set_ylabel("Number of Exceedances")
        axes[0, 0].set_xticks(months)
        axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: Exceedance timeline scatter plot
        mask = self.values > p95
        axes[0, 1].plot(self.dates, self.values, color="lightblue", alpha=0.4, lw=0.6, label="Daily rainfall")
        axes[0, 1].scatter(self.dates[mask], self.values[mask], color="red", s=18, label=f"Exceedance > P95 ({p95:.1f} mm)")
        axes[0, 1].axhline(p95, color="darkred", linestyle="--", lw=1.2)
        axes[0, 1].set_title("Panel 2: Extreme Exceedance Timeline (Dev Train)", fontweight="bold")
        axes[0, 1].set_ylabel("mm / day")
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Distribution of tail values (> P95)
        tail_vals = self.values[mask]
        axes[1, 0].hist(tail_vals, bins=25, color="coral", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title(f"Panel 3: Histogram of Extreme Tail Values (> {p95:.1f} mm)", fontweight="bold")
        axes[1, 0].set_xlabel("Rainfall (mm)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: Empirical return period curve (Log-Log)
        rp_df = pd.DataFrame(rp_data["top_daily_observations"])
        axes[1, 1].loglog(rp_df["return_period_days"], rp_df["rainfall_mm"], "bo-", markersize=3.5, lw=1)
        axes[1, 1].set_title("Panel 4: Empirical Daily-Observation Return Period (Weibull)", fontweight="bold")
        axes[1, 1].set_xlabel("Return Period (days, log scale)")
        axes[1, 1].set_ylabel("Daily Precipitation (mm, log scale)")
        axes[1, 1].grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig


# =============================================================================
# CONVENIENCE BACKWARD-COMPATIBLE WRAPPERS
# =============================================================================
def analyze_extreme_events(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience function for complete extreme events analysis."""
    analyzer = ExtremeEventsAnalyzer(df, target_col=target_col, date_col=date_col)
    return analyzer.analyze()


def get_extreme_thresholds(
    df: pd.DataFrame,
    target_col: str = "Lượng mưa",
    percentiles: Sequence[float] = EXTREME_PERCENTILES,
) -> dict[str, float]:
    """Get extreme event thresholds for specified percentiles."""
    clean = df[target_col].dropna()
    return compute_extreme_thresholds(clean, percentiles=percentiles)


def count_extreme_events(
    df: pd.DataFrame,
    target_col: str = "Lượng mưa",
    threshold: Optional[float] = None,
    percentile: float = 0.95,
) -> int:
    """Count extreme events above a threshold or percentile."""
    clean = df[target_col].dropna()
    if len(clean) == 0:
        return 0
    if threshold is None:
        th = float(clean.quantile(percentile))
    else:
        th = float(threshold)
    return int((clean > th).sum())
