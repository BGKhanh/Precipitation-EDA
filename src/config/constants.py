"""Configuration constants for the DS108 Weather Prediction Project"""

import os
from pathlib import Path

class Config:
    """Central configuration for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "nasa_power_hcmc_data"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # NASA POWER API configuration
    NASA_API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_API_TIMEOUT = 120
    NASA_API_MAX_PARAMS = 20
    
    # Ho Chi Minh City coordinates
    HCMC_COORDINATES = {
        "lat": 10.78,
        "lon": 106.7
    }
    
    # Data parameters
    NASA_PARAMETERS = [
        "PRECTOTCORR", "QV2M", "RH2M", "T2M", "T2MDEW", "T2MWET", 
        "T2M_MAX", "TS", "T2M_MIN", "ALLSKY_SFC_LW_DWN", "PS",
        "WS10M", "WD10M", "WS10M_MAX", "WS2M_MAX", "WS2M", "WD2M",
        "GWETPROF", "GWETTOP", "GWETROOT"
    ]
    
    # Column mapping (English -> Vietnamese)
    COLUMN_MAPPING = {
        "PRECTOTCORR": "Lượng mưa",
        "QV2M": "Độ ẩm tuyệt đối 2m",
        "RH2M": "Độ ẩm tương đối 2m",
        "T2M": "Nhiệt độ 2m",
        "T2MDEW": "Điểm sương 2m",
        "T2MWET": "Nhiệt độ bầu ướt 2m",
        "T2M_MAX": "Nhiệt độ tối đa 2m",
        "TS": "Nhiệt độ bề mặt đất",
        "T2M_MIN": "Nhiệt độ tối thiểu 2m",
        "ALLSKY_SFC_LW_DWN": "Bức xạ sóng dài xuống",
        "PS": "Áp suất bề mặt",
        "WS10M": "Tốc độ gió 10m",
        "WD10M": "Hướng gió 10m",
        "WS10M_MAX": "Tốc độ gió tối đa 10m",
        "WS2M_MAX": "Tốc độ gió tối đa 2m",
        "WS2M": "Tốc độ gió 2m",
        "WD2M": "Hướng gió 2m",
        "GWETPROF": "Độ ẩm đất mặt cắt",
        "GWETTOP": "Độ ẩm đất bề mặt",
        "GWETROOT": "Độ ẩm đất vùng rễ",
        "DATE": "Ngày",
        "LATITUDE": "Vĩ độ",
        "LONGITUDE": "Kinh độ"
    }
    
    # Default date range
    DEFAULT_START_DATE = "20000101"
    DEFAULT_END_DATE = "20250430"
    
    # Data quality parameters
    MISSING_DATA_THRESHOLD = 0.1  # 10%
    DUPLICATE_THRESHOLD = 0.05    # 5%