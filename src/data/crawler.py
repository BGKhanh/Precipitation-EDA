import logging
import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.constants import Config

logger = logging.getLogger(__name__)


class NASAPowerCrawler:
    """
    NASA POWER API Crawler - Focused on data collection only.
    Handles parameter batching to stay within API limits.
    """

    # NASA POWER dùng -999 làm fill value mặc định cho ô thiếu dữ liệu
    DEFAULT_FILL_VALUE = -999

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        # Các giá trị mặc định (daily) chỉ mang tính tham khảo/thông tin,
        # KHÔNG được các method khác đọc/ghi lại trong lúc crawl để tránh side-effect.
        self.default_base_url = Config.NASA_API_BASE_URL
        self.default_parameters = Config.NASA_PARAMETERS_DAILY

        hcmc_coords = Config.province_coordinates["TP. Hồ Chí Minh"]
        self.coordinates = {"lat": hcmc_coords[0], "lon": hcmc_coords[1]}
        self.output_dir = Config.DATA_DIR
        self.max_params_per_batch = Config.NASA_API_MAX_PARAMS

        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        os.makedirs(self.output_dir, exist_ok=True)

        self.session = self._build_session()

    # ------------------------------------------------------------------- #
    # Session / networking
    # ------------------------------------------------------------------- #

    def _build_session(self) -> requests.Session:
        """
        Session dùng chung cho toàn bộ request (connection pooling) + retry
        tự động ở tầng transport (connection reset, 5xx, 429) với backoff nhẹ.
        Retry ở tầng application (lỗi JSON, lỗi nghiệp vụ từ API) được xử lý
        riêng trong `_crawl_batch` để có thể log chi tiết hơn.
        """
        session = requests.Session()
        retry = Retry(
            total=1,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def close(self) -> None:
        self.session.close()

    # ------------------------------------------------------------------- #
    # Resolution-aware config (local, không mutate self.*)
    # ------------------------------------------------------------------- #

    def _resolve_endpoint(self, temporal_resolution: str) -> str:
        if temporal_resolution == "hourly":
            return "https://power.larc.nasa.gov/api/temporal/hourly/point"
        return Config.NASA_API_BASE_URL

    def _resolve_parameters(self, temporal_resolution: str) -> List[str]:
        if temporal_resolution == "hourly":
            return Config.NASA_PARAMETERS_HOURLY
        return Config.NASA_PARAMETERS_DAILY

    def _batch_parameters(self, parameters: List[str]) -> List[List[str]]:
        return [
            parameters[i:i + self.max_params_per_batch]
            for i in range(0, len(parameters), self.max_params_per_batch)
        ]

    @staticmethod
    def _validate_date_range(start_date: str, end_date: str) -> None:
        """Raise ValueError nếu format sai hoặc start_date > end_date."""
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

    # ------------------------------------------------------------------- #
    # Core crawl
    # ------------------------------------------------------------------- #

    def crawl_data(
        self,
        start_date: str = None,
        end_date: str = None,
        temporal_resolution: str = "daily",
    ) -> Optional[pd.DataFrame]:
        """
        Crawl weather data từ NASA POWER API với batching + retry tự động.

        Args:
            start_date: YYYYMMDD (mặc định Config.DEFAULT_START_DATE)
            end_date: YYYYMMDD (mặc định Config.DEFAULT_END_DATE)
            temporal_resolution: 'daily' hoặc 'hourly'

        Returns:
            DataFrame đã lưu, hoặc None nếu thất bại.
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

        # Biến cục bộ - không ghi đè lên self.base_url / self.parameters
        base_url = self._resolve_endpoint(temporal_resolution)
        parameters = self._resolve_parameters(temporal_resolution)

        logger.info("=== CRAWLING NASA POWER DATA ===")
        logger.info("Period: %s -> %s", start_date, end_date)
        logger.info("Location: %s", self.coordinates)
        logger.info("Resolution: %s", temporal_resolution)
        logger.info("Total parameters: %d", len(parameters))

        batches = self._batch_parameters(parameters)
        logger.info("Split into %d batch(es), max %d param(s)/batch", len(batches), self.max_params_per_batch)

        all_dfs = []
        for i, batch in enumerate(batches, 1):
            df = self._crawl_batch(base_url, batch, i, len(batches), start_date, end_date)
            if df is None:
                logger.error("Batch %d failed permanently, aborting crawl", i)
                return None
            all_dfs.append(df)

        logger.info("Merging %d batch(es)...", len(all_dfs))
        merged_df = self._merge_batches(all_dfs)
        if merged_df is None:
            return None

        logger.info("Data merged: %s", merged_df.shape)

        return self.save_data(merged_df, start_date, end_date, temporal_resolution)

    # ------------------------------------------------------------------- #
    # Batch crawling with retry + exponential backoff
    # ------------------------------------------------------------------- #

    def _crawl_batch(
        self,
        base_url: str,
        batch: List[str],
        batch_num: int,
        total_batches: int,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Crawl 1 batch tham số, retry với exponential backoff khi lỗi."""
        params = {
            "parameters": ",".join(batch),
            "community": "RE",
            "longitude": self.coordinates["lon"],
            "latitude": self.coordinates["lat"],
            "start": start_date,
            "end": end_date,
            "format": "JSON",
        }

        last_error: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Batch %d/%d (%d params) - attempt %d/%d",
                batch_num, total_batches, len(batch), attempt, self.max_retries,
            )
            try:
                response = self.session.get(base_url, params=params, timeout=Config.NASA_API_TIMEOUT)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning("Batch %d attempt %d timed out (%ss)", batch_num, attempt, Config.NASA_API_TIMEOUT)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning("Batch %d attempt %d connection error: %s", batch_num, attempt, e)
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.warning("Batch %d attempt %d HTTP error: %s", batch_num, attempt, e)
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning("Batch %d attempt %d request failed: %s", batch_num, attempt, e)
            except ValueError as e:
                # response.json() raises ValueError/JSONDecodeError on malformed body
                last_error = e
                logger.warning("Batch %d attempt %d returned invalid JSON: %s", batch_num, attempt, e)
            else:
                api_messages = data.get("messages")
                if api_messages:
                    last_error = RuntimeError(f"NASA POWER API error: {api_messages}")
                    logger.warning("Batch %d attempt %d API error: %s", batch_num, attempt, api_messages)
                else:
                    df = self._parse_response(data, batch_num)
                    if df is not None:
                        logger.info("Batch %d retrieved: %d rows, %d columns", batch_num, df.shape[0], df.shape[1])
                        return df
                    last_error = RuntimeError("_parse_response trả về None")

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor ** attempt
                logger.info("Batch %d: chờ %.1fs rồi thử lại...", batch_num, sleep_time)
                time.sleep(sleep_time)

        logger.error("Batch %d thất bại sau %d lần thử. Lỗi cuối: %s", batch_num, self.max_retries, last_error)
        return None

    # ------------------------------------------------------------------- #
    # Parsing
    # ------------------------------------------------------------------- #

    def _parse_response(self, data: Dict[str, Any], batch_num: int) -> Optional[pd.DataFrame]:
        """
        Parse response JSON của NASA POWER thành DataFrame index=datetime (tên 'DATE').

        Response có dạng: properties.parameter = {param_name: {date_str: value, ...}, ...}
        -> pd.DataFrame(...) tự động dùng param_name làm cột, date_str làm index.
        Cần: (1) thay fill_value (-999 mặc định) bằng NaN, (2) parse index sang
        datetime thật, (3) loại bỏ index không parse được / trùng lặp.
        """
        properties = data.get("properties")
        if not isinstance(properties, dict):
            logger.error("Batch %d: response thiếu hoặc sai kiểu 'properties'", batch_num)
            return None

        parameter_data = properties.get("parameter")
        if not parameter_data or not isinstance(parameter_data, dict):
            logger.error("Batch %d: không có dữ liệu trong 'properties.parameter'", batch_num)
            return None

        try:
            df = pd.DataFrame(parameter_data)
        except ValueError as e:
            logger.error("Batch %d: không dựng được DataFrame từ response: %s", batch_num, e)
            return None

        if df.empty:
            logger.error("Batch %d: DataFrame parse ra rỗng", batch_num)
            return None

        # --- Xử lý fill value (missing data sentinel của NASA POWER) ---
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        fill_value = header.get("fill_value", self.DEFAULT_FILL_VALUE)
        if fill_value is not None:
            n_filled = int((df == fill_value).sum().sum())
            if n_filled:
                logger.warning(
                    "Batch %d: %d ô có giá trị fill_value=%s -> chuyển thành NaN",
                    batch_num, n_filled, fill_value,
                )
                df = df.replace(fill_value, np.nan)

        # --- Parse index (chuỗi ngày) sang datetime thật ---
        raw_index = df.index.astype(str)
        sample_len = len(raw_index[0]) if len(raw_index) else 0
        if sample_len == 8:
            date_format = "%Y%m%d"          # daily: YYYYMMDD
        elif sample_len == 10:
            date_format = "%Y%m%d%H"        # hourly: YYYYMMDDHH
        else:
            date_format = None              # để pandas tự suy luận

        parsed_index = pd.to_datetime(raw_index, format=date_format, errors="coerce")
        n_unparsed = int(parsed_index.isna().sum())
        if n_unparsed:
            logger.warning(
                "Batch %d: %d timestamp không parse được, sẽ loại các dòng này",
                batch_num, n_unparsed,
            )

        df.index = parsed_index
        df = df[~df.index.isna()]
        if df.empty:
            logger.error("Batch %d: sau khi loại timestamp lỗi, DataFrame rỗng", batch_num)
            return None

        df.index.name = "DATE"
        df = df.sort_index()

        dup_count = int(df.index.duplicated().sum())
        if dup_count:
            logger.warning(
                "Batch %d: %d timestamp trùng lặp, giữ lại lần xuất hiện đầu tiên",
                batch_num, dup_count,
            )
            df = df[~df.index.duplicated(keep="first")]

        return df

    # ------------------------------------------------------------------- #
    # Merge (đảm bảo toàn vẹn time series)
    # ------------------------------------------------------------------- #

    def _merge_batches(self, dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Ghép các batch (mỗi batch là 1 tập tham số, cùng khoảng thời gian)
        theo trục cột, dùng chung index datetime.

        Kiểm tra toàn vẹn:
        - So sánh index giữa các batch, cảnh báo nếu lệch (nghĩa là 1 batch bị
          thiếu/thừa timestamp so với batch còn lại -> inner join sẽ làm mất dữ liệu).
        - Loại cột trùng tên nếu tham số bị lặp giữa các batch.
        - Sau khi merge, kiểm tra gap trong time series theo tần suất suy luận được.
        """
        if not dfs:
            logger.error("Không có DataFrame nào để merge")
            return None

        if len(dfs) == 1:
            return dfs[0].sort_index()

        reference_index = dfs[0].index
        for i, df in enumerate(dfs[1:], start=2):
            if not df.index.equals(reference_index):
                extra = df.index.difference(reference_index)
                missing = reference_index.difference(df.index)
                logger.warning(
                    "Batch %d có index lệch so với batch 1: %d timestamp thừa, %d timestamp thiếu "
                    "(sẽ dùng inner join -> phần lệch sẽ bị loại khỏi kết quả cuối)",
                    i, len(extra), len(missing),
                )

        try:
            merged = pd.concat(dfs, axis=1, join="inner")
        except (ValueError, TypeError) as e:
            logger.error("Merge thất bại: %s", e)
            return None

        if merged.empty:
            logger.error("Merge ra DataFrame rỗng — có thể các batch không còn timestamp chung nào")
            return None

        if merged.columns.duplicated().any():
            dup_cols = merged.columns[merged.columns.duplicated()].tolist()
            logger.warning("Phát hiện cột trùng lặp sau merge, giữ lần xuất hiện đầu: %s", dup_cols)
            merged = merged.loc[:, ~merged.columns.duplicated()]

        merged = merged.sort_index()
        self._check_time_series_gaps(merged)

        return merged

    @staticmethod
    def _check_time_series_gaps(df: pd.DataFrame) -> None:
        """Cảnh báo (không chặn) nếu time series sau merge có khoảng trống."""
        if len(df.index) < 3:
            # pd.infer_freq cần tối thiểu 3 mốc thời gian để suy luận đáng tin cậy
            return

        try:
            inferred_freq = pd.infer_freq(df.index)
        except ValueError as e:
            logger.warning("Không suy luận được tần suất time series: %s", e)
            return

        if inferred_freq is None:
            logger.warning(
                "Không thể tự suy luận tần suất (freq) của time series sau merge "
                "-> không kiểm tra được gap tự động, cần soát thủ công."
            )
            return

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=inferred_freq)
        missing = full_range.difference(df.index)
        if len(missing):
            logger.warning(
                "Time series có %d timestamp bị thiếu (freq=%s), ví dụ: %s",
                len(missing), inferred_freq, list(missing[:5]),
            )

    # ------------------------------------------------------------------- #
    # Save
    # ------------------------------------------------------------------- #

    def save_data(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        temporal_resolution: str = "daily",
    ) -> Optional[pd.DataFrame]:
        """
        Lưu DataFrame ra CSV. DataFrame có index là datetime tên 'DATE' —
        cần reset_index() trước khi to_csv(index=False), nếu không cột DATE
        sẽ biến mất hoàn toàn khỏi file (bug ở bản gốc).
        """
        filename = f"hcmc_weather_data_{temporal_resolution}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.output_dir, filename)

        try:
            df_to_save = df.reset_index()  # DATE: index -> cột tường minh
            df_to_save.to_csv(filepath, index=False, encoding="utf-8")
        except OSError as e:
            logger.error("Lưu file thất bại (lỗi I/O) tại %s: %s", filepath, e)
            return None
        except (ValueError, UnicodeEncodeError) as e:
            logger.error("Lưu file thất bại (lỗi encode/format dữ liệu): %s", e)
            return None

        logger.info("Data saved: %s", filename)
        logger.info("Shape: %s", df.shape)

        return df