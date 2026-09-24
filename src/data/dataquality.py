"""
Data Quality Assessment cho weather data.

Cải tiến so với bản gốc:
- Chọn cách hiển thị kết quả bằng 1 flag đơn giản (REPORT_MODE / tham số `mode`),
  không tự dò môi trường:
  -> "notebook": in ra console cho dễ đọc (mặc định)
  -> "production": ghi qua `logging` (đẩy được vào CloudWatch/ELK...) + 1 dòng JSON tổng hợp
  Khi deploy lên Airflow/production, chỉ cần đổi flag này (qua tham số hoặc
  biến môi trường REPORT_MODE=production).
- Thêm các check: outlier (IQR), giá trị âm bất hợp lệ, cột hằng số, mixed-type,
  high-cardinality, tính liên tục của cột thời gian (nếu có)
- Format số liệu rõ ràng: %, số dòng, index cụ thể (giới hạn số lượng hiển thị)
- Có thể raise exception khi score dưới ngưỡng -> để Airflow fail task thay vì
  âm thầm đi tiếp với dữ liệu tệ
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..config.constants import Config


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

# Mặc định lấy từ biến môi trường REPORT_MODE nếu có, nếu không thì "notebook".
# Đổi giá trị này (hoặc set env REPORT_MODE=production) khi deploy lên Airflow/production.
DEFAULT_REPORT_MODE = os.environ.get("REPORT_MODE", "notebook")


class QualityReporter:
    """
    Điều hướng output theo 1 flag đơn giản `mode`:
    - "notebook"   -> in ra console cho dễ đọc
    - "production" -> ghi log có cấu trúc (logging) + 1 dòng JSON tổng hợp
    """

    def __init__(self, name: str = "DataQualityAssessment", mode: Optional[str] = None):
        self.mode = mode or DEFAULT_REPORT_MODE
        if self.mode not in ("notebook", "production"):
            raise ValueError(f"mode phải là 'notebook' hoặc 'production', nhận được: {self.mode!r}")

        self.logger = logging.getLogger(name)

        if self.mode == "production" and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def section(self, title: str) -> None:
        if self.mode == "notebook":
            print(f"\n{'='*62}\n{title}\n{'='*62}")
        else:
            self.logger.info(f"--- {title} ---")

    def info(self, msg: str) -> None:
        print(msg) if self.mode == "notebook" else self.logger.info(msg)

    def warning(self, msg: str) -> None:
        print(f"⚠️  {msg}") if self.mode == "notebook" else self.logger.warning(msg)

    def error(self, msg: str) -> None:
        print(f"❌ {msg}") if self.mode == "notebook" else self.logger.error(msg)

    def result_json(self, results: Dict[str, Any]) -> None:
        """1 dòng JSON tổng hợp toàn bộ kết quả — chỉ ghi ở production để log
        aggregator (CloudWatch, ELK, Datadog...) có thể parse/query được."""
        if self.mode != "notebook":
            self.logger.info("QUALITY_RESULT_JSON " + json.dumps(results, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _sample_indices(idx: pd.Index, limit: int = 10) -> List[Any]:
    """Trả về tối đa `limit` index đầu tiên, kèm ghi chú nếu bị cắt bớt."""
    idx_list = list(idx[:limit])
    return idx_list


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class DataQualityAssessment:
    """Essential + extended data quality checks cho weather data."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = Config.TARGET_COL_VI,
        mode: Optional[str] = None,
        outlier_iqr_factor: float = 1.5,
        high_cardinality_threshold: int = 50,
        max_reported_indices: int = 10,
        missing_data_threshold: float = Config.MISSING_DATA_THRESHOLD,
        duplicate_threshold: float = Config.DUPLICATE_THRESHOLD,
    ):
        self.df = df.copy()
        # Resolve target column: nếu tên chỉ định không có trong df,
        # tự fallback giữa tên EN ↔ VI (theo COLUMN_MAPPING).
        if target_col not in df.columns:
            fallback = self._resolve_target_col(target_col, df.columns)
            if fallback is not None:
                target_col = fallback
        self.target_col = target_col
        self.n_rows = len(df)
        self.n_cols = len(df.columns)
        self.reporter = QualityReporter(mode=mode)
        self.outlier_iqr_factor = outlier_iqr_factor
        self.high_cardinality_threshold = high_cardinality_threshold
        self.max_reported_indices = max_reported_indices
        self.missing_data_threshold = missing_data_threshold  # fraction, e.g. 0.1
        self.duplicate_threshold = duplicate_threshold          # fraction, e.g. 0.05

    @staticmethod
    def _resolve_target_col(target_col: str, columns: pd.Index) -> Optional[str]:
        """Fallback: nếu target_col không có, thử tên EN↔VI từ Config."""
        if target_col == Config.TARGET_COL_VI and Config.TARGET_COL_EN in columns:
            return Config.TARGET_COL_EN
        if target_col == Config.TARGET_COL_EN and Config.TARGET_COL_VI in columns:
            return Config.TARGET_COL_VI
        return None

    # ------------------------------------------------------------------- #
    # Orchestration
    # ------------------------------------------------------------------- #

    def assess_quality(
        self,
        raise_on_low_score: bool = False,
        min_score: int = 60,
        datetime_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ các check.

        Args:
            raise_on_low_score: nếu True, raise ValueError khi score < min_score
                (hữu ích để Airflow fail task khi dữ liệu quá tệ).
            min_score: ngưỡng tối thiểu chấp nhận được.
            datetime_col: tên cột thời gian nếu muốn check tính liên tục.
        """
        r = self.reporter
        r.section("DATA QUALITY ASSESSMENT")
        r.info(f"Dataset shape: {self.n_rows:,} rows x {self.n_cols} cols")

        results: Dict[str, Any] = {
            "basic_info": self._basic_info(),
            "missing_data": self._check_missing_data(),
            "duplicates": self._check_duplicates(),
            "data_types": self._check_data_types(),
            "mixed_type_columns": self._check_mixed_types(),
            "constant_columns": self._check_constant_columns(),
            "high_cardinality_columns": self._check_high_cardinality(),
            "target_variable": self._check_target_variable(),
        }

        if datetime_col:
            results["datetime_continuity"] = self._check_datetime_continuity(datetime_col)

        r.result_json(results)
        return results

    def _basic_info(self) -> Dict[str, Any]:
        return {
            "shape": self.df.shape,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "columns": list(self.df.columns),
        }

    def _check_missing_data(self) -> Dict[str, Any]:
        missing_counts = self.df.isnull().sum()
        total_missing = int(missing_counts.sum())
        missing_pct = _pct(total_missing, self.n_rows * self.n_cols)

        cols_with_missing = {
            col: {
                "count": int(cnt),
                "percentage": _pct(cnt, self.n_rows),
            }
            for col, cnt in missing_counts[missing_counts > 0].items()
        }

        self.reporter.info(
            f"Missing data: {total_missing:,} / {self.n_rows * self.n_cols:,} cells "
            f"({missing_pct}%) across {len(cols_with_missing)} column(s)"
        )
        for col, info in sorted(cols_with_missing.items(), key=lambda x: -x[1]["count"]):
            self.reporter.info(f"    - {col}: {info['count']:,} missing ({info['percentage']}%)")

        threshold_pct = self.missing_data_threshold * 100
        if missing_pct > threshold_pct:
            self.reporter.warning(
                f"Tỷ lệ missing ({missing_pct}%) vượt ngưỡng chấp nhận ({threshold_pct}%)"
            )

        return {
            "total_missing": total_missing,
            "missing_percentage": missing_pct,
            "columns_with_missing": cols_with_missing,
            "threshold_exceeded": missing_pct > threshold_pct,
        }

    def _check_duplicates(self) -> Dict[str, Any]:
        dup_mask = self.df.duplicated()
        duplicates = int(dup_mask.sum())
        duplicate_pct = _pct(duplicates, self.n_rows)
        sample_idx = _sample_indices(self.df.index[dup_mask], self.max_reported_indices)

        self.reporter.info(
            f"Duplicate rows: {duplicates:,} ({duplicate_pct}%)"
            + (f" | sample index: {sample_idx}" if duplicates else "")
        )

        threshold_pct = self.duplicate_threshold * 100
        if duplicate_pct > threshold_pct:
            self.reporter.warning(
                f"Tỷ lệ trùng lặp ({duplicate_pct}%) vượt ngưỡng chấp nhận ({threshold_pct}%)"
            )

        return {
            "duplicate_rows": duplicates,
            "duplicate_percentage": duplicate_pct,
            "sample_indices": sample_idx,
            "threshold_exceeded": duplicate_pct > threshold_pct,
        }

    def _check_data_types(self) -> Dict[str, Any]:
        dtypes = self.df.dtypes.value_counts().to_dict()
        result = {str(k): int(v) for k, v in dtypes.items()}
        self.reporter.info(f"Data types: {result}")
        return result

    def _check_target_variable(self) -> Dict[str, Any]:
        if self.target_col not in self.df.columns:
            self.reporter.error(f"Target column '{self.target_col}' not found in dataframe.")
            return {"exists": False}

        target_data = self.df[self.target_col]
        missing_count = int(target_data.isnull().sum())
        missing_pct = _pct(missing_count, len(target_data))
        zero_days = int((target_data == 0).sum())
        zero_pct = _pct(zero_days, len(target_data))
        negative_count = int((target_data < 0).sum())
        negative_pct = _pct(negative_count, len(target_data))
        negative_idx = _sample_indices(target_data[target_data < 0].index, self.max_reported_indices)

        self.reporter.info(
            f"Target '{self.target_col}': missing={missing_count:,} ({missing_pct}%), "
            f"zero={zero_days:,} ({zero_pct}%), "
            f"negative(invalid)={negative_count:,} ({negative_pct}%), "
            f"range=[{target_data.min():.2f}, {target_data.max():.2f}], "
            f"mean={target_data.mean():.2f}"
        )
        if negative_count:
            self.reporter.warning(
                f"Target has {negative_count} negative value(s) — invalid for rainfall. "
                f"Sample index: {negative_idx}"
            )

        return {
            "exists": True,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "min_value": float(target_data.min()),
            "max_value": float(target_data.max()),
            "mean_value": round(float(target_data.mean()), 2),
            "zero_days": zero_days,
            "zero_percentage": zero_pct,
            "negative_count": negative_count,
            "negative_percentage": negative_pct,
            "negative_sample_indices": negative_idx,
        }

    # ------------------------------------------------------------------- #
    # Extended checks (mới)
    # ------------------------------------------------------------------- #

    
    def _check_constant_columns(self) -> Dict[str, Any]:
        """Cột chỉ có 1 giá trị duy nhất (bỏ NaN) — vô nghĩa cho model."""
        constant_cols = [
            col for col in self.df.columns
            if self.df[col].nunique(dropna=True) <= 1
        ]
        if constant_cols:
            self.reporter.warning(f"Constant column(s) (no predictive value): {constant_cols}")
        return {"count": len(constant_cols), "columns": constant_cols}

    def _check_mixed_types(self) -> Dict[str, Any]:
        """Object columns mà thực chất có thể ép được sang số — dấu hiệu lỗi parse."""
        suspicious: Dict[str, Dict[str, Any]] = {}
        for col in self.df.select_dtypes(include="object").columns:
            coerced = pd.to_numeric(self.df[col], errors="coerce")
            non_null = self.df[col].notna().sum()
            coercible = coerced.notna().sum()
            if non_null > 0 and 0 < coercible < non_null:
                pct_numeric_like = _pct(coercible, non_null)
                suspicious[col] = {"numeric_like_percentage": pct_numeric_like}

        if suspicious:
            self.reporter.warning(f"Object column(s) with mixed numeric/text values: {list(suspicious.keys())}")

        return {"count": len(suspicious), "columns": suspicious}

    def _check_high_cardinality(self) -> Dict[str, Any]:
        """Categorical/object columns có quá nhiều giá trị unique."""
        flagged = {}
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            n_unique = self.df[col].nunique(dropna=True)
            if n_unique > self.high_cardinality_threshold:
                flagged[col] = n_unique

        if flagged:
            self.reporter.warning(f"High-cardinality column(s) (> {self.high_cardinality_threshold} unique): {flagged}")

        return {"threshold": self.high_cardinality_threshold, "columns": flagged}

    def _check_datetime_continuity(self, datetime_col: str) -> Dict[str, Any]:
        """Kiểm tra cột thời gian: parse được không, trùng lặp, có gap không."""
        if datetime_col not in self.df.columns:
            self.reporter.error(f"Datetime column '{datetime_col}' not found.")
            return {"checked": False, "reason": "column_not_found"}

        dt = pd.to_datetime(self.df[datetime_col], errors="coerce")
        unparsed = int(dt.isna().sum()) - int(self.df[datetime_col].isna().sum())
        dt_valid = dt.dropna().sort_values()

        duplicate_dates = int(dt_valid.duplicated().sum())

        gaps = []
        if len(dt_valid) > 1:
            inferred_freq = pd.infer_freq(dt_valid)
            diffs = dt_valid.diff().dropna()
            if inferred_freq:
                expected_step = pd.tseries.frequencies.to_offset(inferred_freq)
                gap_mask = diffs > pd.Timedelta(expected_step)
                gaps = diffs[gap_mask].index.tolist()[: self.max_reported_indices]

        self.reporter.info(
            f"Datetime '{datetime_col}': unparsable={unparsed}, duplicate_dates={duplicate_dates}, "
            f"gaps_detected={len(gaps)}"
        )

        return {
            "checked": True,
            "unparsable_count": max(unparsed, 0),
            "duplicate_dates": duplicate_dates,
            "gap_count": len(gaps),
            "sample_gap_indices": gaps,
        }


def assess_data_quality(
    df: pd.DataFrame,
    target_col: str = Config.TARGET_COL_VI,
    mode: Optional[str] = None,
    raise_on_low_score: bool = False,
    min_score: int = 60,
    datetime_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function.

    mode: ép "notebook" hoặc "production" thay vì auto-detect (hữu ích khi test).
    target_col: mặc định dùng tên tiếng Việt từ Config; tự động fallback
        sang tên tiếng Anh nếu DataFrame chưa được rename.
    """
    assessor = DataQualityAssessment(df, target_col, mode=mode)
    return assessor.assess_quality(
        raise_on_low_score=raise_on_low_score,
        min_score=min_score,
        datetime_col=datetime_col,
    )


# --------------------------------------------------------------------------- #
# Date continuity check (Single source of truth)
# --------------------------------------------------------------------------- #

def check_continuity(
    df: pd.DataFrame,
    date_col: str = "Ngày",
    freq: str = "D",
    raise_on_gap: bool = False,
) -> Tuple[bool, List[pd.Timestamp]]:
    """Check that *date_col* has no missing days.

    Detects missing days in the date index — a gap means lag features
    silently shift (e.g., a "yesterday" lag becomes "3-days-ago" across
    a 2-day gap).

    Args:
        df: DataFrame with a datetime column.
        date_col: Name of the date column.
        freq: Expected frequency (``'D'`` for daily).
        raise_on_gap: If ``True``, raise ``ValueError`` when gaps are
            found instead of just logging a warning.

    Returns:
        ``(is_continuous, missing_dates)`` where *is_continuous* is
        ``True`` if no gaps exist, and *missing_dates* is the list of
        dates that should be present but are not.
    """
    logger = logging.getLogger(__name__)

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        dates = pd.to_datetime(df[date_col])
    else:
        dates = df[date_col]

    dates = dates.dropna().sort_values()

    if len(dates) < 2:
        logger.warning("check_continuity: fewer than 2 dates, skipping check")
        return True, []

    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = full_range.difference(dates)
    missing_list: List[pd.Timestamp] = missing.tolist()

    if len(missing_list) == 0:
        logger.info(
            "check_continuity: no gaps found (%d days, %s → %s)",
            len(dates),
            dates.min().date(),
            dates.max().date(),
        )
        return True, []

    # Log up to 20 missing dates for readability
    sample = missing_list[:20]
    sample_str = ", ".join(str(d.date()) for d in sample)
    suffix = f" ... and {len(missing_list) - 20} more" if len(missing_list) > 20 else ""

    msg = (
        f"check_continuity: {len(missing_list)} missing day(s) detected "
        f"in [{dates.min().date()} → {dates.max().date()}]: {sample_str}{suffix}"
    )

    if raise_on_gap:
        raise ValueError(msg)

    logger.warning(msg)
    return False, missing_list
