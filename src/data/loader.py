import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..config.constants import Config

logger = logging.getLogger(__name__)


DROPPED_RADIATION_COLUMNS = [
    # Vietnamese column names
    "Bức xạ trực tiếp pháp tuyến",
    "Bức xạ quang hợp trời quang",
    "Bức xạ UVA",
    "Bức xạ UVB",
    "Chỉ số UV",
    # English column names (nếu rename_columns=False)
    "ALLSKY_SFC_SW_DNI",
    "CLRSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB",
    "ALLSKY_SFC_UV_INDEX",
]


class DataLoader:
    """
    Simple data loader with basic preprocessing:
    load CSV -> rename columns (optional) -> parse date column -> drop redundant radiation cols.
    """

    def __init__(self):
        # Config chỉ chứa static attributes, không cần khởi tạo instance.
        # Giữ tham chiếu tới class (không phải instance) để code cũ dùng
        # `loader.config.XYZ` vẫn hoạt động như trước.
        self.config = Config
        self.data_dir = Path(Config.DATA_DIR)

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        temporal_resolution: str = "daily",
        rename_columns: bool = True,
        drop_redundant_radiation: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load weather data from CSV file (do NASAPowerCrawler tạo ra).

        Args:
            start_date: YYYYMMDD (mặc định Config.DEFAULT_START_DATE)
            end_date: YYYYMMDD (mặc định Config.DEFAULT_END_DATE)
            temporal_resolution: 'daily' hoặc 'hourly' — PHẢI khớp với file
                mà crawler đã lưu (tên file có chứa phần này).
            rename_columns: có đổi tên cột sang tiếng Việt theo Config.COLUMN_MAPPING không
            drop_redundant_radiation: có drop 5 cột bức xạ/UV thiếu 100% trong năm 2000
                (đã duyệt theo Option A dựa trên Feature Redundancy & Signal Capture)

        Returns:
            DataFrame hoặc None nếu load thất bại.
        """
        start_date = start_date or Config.DEFAULT_START_DATE
        end_date = end_date or Config.DEFAULT_END_DATE
        temporal_resolution = (temporal_resolution or "daily").lower()

        if temporal_resolution not in {"daily", "hourly"}:
            logger.error("Unsupported temporal resolution: %s", temporal_resolution)
            return None

        try:
            self._validate_date_range(start_date, end_date)
        except ValueError as e:
            logger.error("Date validation failed: %s", e)
            return None

        # Tên file phải khớp CHÍNH XÁC với format mà NASAPowerCrawler.save_data() dùng,
        # bao gồm cả temporal_resolution — thiếu phần này là bug khiến loader không
        # bao giờ tìm thấy file crawler vừa lưu.
        filename = f"hcmc_weather_data_{temporal_resolution}_{start_date}_{end_date}.csv"
        filepath = self.data_dir / filename

        logger.info("Loading weather data from %s", filepath)

        if not filepath.exists():
            logger.error("File not found: %s", filepath)
            return None

        df = self._read_csv(filepath)
        if df is None:
            return None

        if df.empty:
            logger.error("Loaded dataframe has 0 rows: %s", filepath)
            return None

        logger.info("Data loaded: %s", df.shape)

        if rename_columns:
            df = self._rename_columns(df)

        df = self._convert_date_column(df)

        if drop_redundant_radiation:
            df = self._drop_redundant_radiation_columns(df)

        return df

    @staticmethod
    def _drop_redundant_radiation_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop 5 radiation/UV columns that have 100% missing data in the year 2000.

        Rationale (Approved Option A):
        The missingness (366 days in year 2000) is caused by sensor collection onset,
        not random sensor failure. The meteorological and predictive signals are already
        fully captured with 0 missing data by retained variables:
          - 'Bức xạ sóng ngắn bề mặt' (ALLSKY_SFC_SW_DWN, r=-0.3873 vs r=-0.3876)
          - 'Bức xạ quang hợp tổng' (ALLSKY_SFC_PAR_TOT)
        Dropping avoids imputing an entire year of synthetic data while preserving 100%
        of the temporal continuity and training sample size (2000-2025).
        """
        cols_to_drop = [c for c in DROPPED_RADIATION_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(
                "Đã drop %d cột bức xạ/UV dư thừa (100%% missing năm 2000): %s",
                len(cols_to_drop),
                cols_to_drop,
            )
        return df

    # ------------------------------------------------------------------- #
    # Internals
    # ------------------------------------------------------------------- #

    @staticmethod
    def _validate_date_range(start_date: str, end_date: str) -> None:
        try:
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
        except ValueError as e:
            raise ValueError(
                f"start_date/end_date phải đúng định dạng YYYYMMDD "
                f"(nhận được start={start_date!r}, end={end_date!r}): {e}"
            ) from e

        if start_dt > end_dt:
            raise ValueError(f"start_date ({start_date}) phải <= end_date ({end_date})")

    @staticmethod
    def _read_csv(filepath: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(filepath, encoding="utf-8")
        except pd.errors.EmptyDataError:
            logger.error("File rỗng hoặc không có dữ liệu: %s", filepath)
        except pd.errors.ParserError as e:
            logger.error("Lỗi parse CSV %s: %s", filepath, e)
        except UnicodeDecodeError as e:
            logger.error("Lỗi encoding khi đọc %s: %s", filepath, e)
        except OSError as e:
            logger.error("Lỗi I/O khi đọc %s: %s", filepath, e)
        return None

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = Config.COLUMN_MAPPING
        present_cols = set(df.columns)
        known_cols = set(mapping.keys())

        unmapped = present_cols - known_cols
        if unmapped:
            logger.warning(
                "%d cột không có trong COLUMN_MAPPING, giữ nguyên tên gốc: %s",
                len(unmapped), sorted(unmapped),
            )

        matched = len(present_cols & known_cols)
        df = df.rename(columns=mapping)
        logger.info("Columns renamed sang tiếng Việt (%d/%d cột khớp mapping)", matched, len(present_cols))
        return df

    @staticmethod
    def _convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
        """Parse cột ngày sang datetime, sort tăng dần, cảnh báo giá trị lỗi/trùng."""
        if "Ngày" in df.columns:
            date_col = "Ngày"
        elif "DATE" in df.columns:
            date_col = "DATE"
        else:
            logger.warning("Không tìm thấy cột ngày ('Ngày' hoặc 'DATE'), bỏ qua bước convert datetime")
            return df

        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            return df

        try:
            converted = pd.to_datetime(df[date_col], errors="coerce")
        except (ValueError, TypeError) as e:
            logger.warning("Convert datetime cho cột '%s' thất bại, giữ nguyên dữ liệu gốc: %s", date_col, e)
            return df

        n_failed = int(converted.isna().sum()) - int(df[date_col].isna().sum())
        if n_failed > 0:
            logger.warning(
                "%d giá trị trong cột '%s' không parse được sang datetime -> chuyển thành NaT",
                n_failed, date_col,
            )

        df[date_col] = converted
        df = df.sort_values(date_col).reset_index(drop=True)

        dup_count = int(df[date_col].duplicated().sum())
        if dup_count:
            logger.warning("Cột '%s' có %d giá trị ngày trùng lặp", date_col, dup_count)

        logger.info("Cột '%s' đã convert sang datetime và sort tăng dần", date_col)
        return df


# ------------------------------------------------------------------- #
# Canonical train / test split — single source of truth
# ------------------------------------------------------------------- #

DEFAULT_TEST_START_DATE: str = "2020-05-01"


def time_series_split(
    df: pd.DataFrame,
    date_col: str = "Ngày",
    test_start_date: str = DEFAULT_TEST_START_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train and test sets.

    This is the **only** place in the project where the train/test boundary
    is defined.  Every module that needs a split must call this function
    rather than inventing ad-hoc logic.

    Args:
        df: DataFrame with a datetime (or parseable) column *date_col*.
        date_col: Name of the date column.
        test_start_date: First date that belongs to the test set
            (ISO-8601 string, e.g. ``'2020-05-01'``).

    Returns:
        ``(train_df, test_df)`` — row-disjoint DataFrames preserving the
        original index and sorted by *date_col*.

    Default rationale
    -----------------
    With data spanning 2000-01-01 → 2025-04-30 (~25.3 years), setting
    ``test_start_date='2020-05-01'`` yields a ~80 / 20 split (train ≈ 20.3 y,
    test ≈ 5 y).  The 5-year test window covers at least one full ENSO cycle
    (El Niño / La Niña, typical period 3–7 years), reducing the risk that
    evaluation falls entirely within an anomalous or benign climate phase.

    Note: the test period 2020-05 → 2025-04 overlaps with COVID-era reduced
    air-traffic and industrial activity, which *may* have perturbed some
    satellite-derived radiation / aerosol parameters.  This is worth
    mentioning in the report but does not invalidate the split.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)

    cutoff = pd.Timestamp(test_start_date)
    train_df = df[df[date_col] < cutoff].copy()
    test_df = df[df[date_col] >= cutoff].copy()

    logger.info(
        "time_series_split: train %d rows [%s → %s], test %d rows [%s → %s]",
        len(train_df),
        train_df[date_col].min().date() if len(train_df) else "N/A",
        train_df[date_col].max().date() if len(train_df) else "N/A",
        len(test_df),
        test_df[date_col].min().date() if len(test_df) else "N/A",
        test_df[date_col].max().date() if len(test_df) else "N/A",
    )

    return train_df, test_df