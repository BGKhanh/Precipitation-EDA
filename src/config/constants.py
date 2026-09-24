"""Configuration constants for the DS108 Weather Prediction Project"""

import os
from pathlib import Path

class Config:
    """Central configuration for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    # NOTE: Thư mục dữ liệu hiện tại chỉ phục vụ pipeline cho TP.HCM.
    # Nếu mở rộng crawl cho các tỉnh khác (xem province_coordinates),
    # cần refactor DATA_DIR và filename convention trong crawler.py / loader.py.
    DATA_DIR = PROJECT_ROOT / "nasa_power_hcmc_data"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # NASA POWER API configuration
    NASA_API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_API_TIMEOUT = 120
    NASA_API_MAX_PARAMS = 20
    
    # Vietnam province coordinates
    # NOTE: Hiện tại pipeline (crawler, loader) chỉ sử dụng tọa độ TP.HCM.
    # Các tỉnh còn lại là dữ liệu tham khảo cho mở rộng tương lai.
    province_coordinates = {
        'An Giang': (10.178240725000078, 105.09770970000004),
        'Bắc Ninh': (21.349187305000157, 106.74542450000001),
        'Cà Mau': (9.13654448699998, 105.18621643600011),
        'Cần Thơ': (9.741703733000064, 105.75918675100017),
        'Cao Bằng': (22.74423518800006, 106.08566689400014),
        'Đà Nẵng': (15.630195780000177, 107.96775361599998),
        'Đắk Lắk': (12.703161849000082, 108.47152799999999),
        'Điện Biên': (21.472984652999966, 103.23483625000006),
        'Đồng Nai': (11.435487532000025, 107.0357544190001),
        'Đồng Tháp': (10.429642495, 105.99752250000006),
        'Gia Lai': (13.889126754000186, 108.4407434330002),
        'Hà Nội': (20.999185512, 105.69985103800013),
        'Hà Tĩnh': (18.29024111300015, 105.73701072300013),
        'Hải Phòng': (20.874078408000116, 106.47570187500008),
        'Huế': (16.330725226000084, 107.51944430200007),
        'Hưng Yên': (20.450079853000027, 106.46798250000006),
        'Khánh Hòa': (12.068182784000044, 108.96232374900006),
        'Lai Châu': (22.316816083000102, 103.18704443600006),
        'Lâm Đồng': (11.673126970000055, 107.97005788600005),
        'Lạng Sơn': (21.772559049000165, 106.72962900000014),
        'Lào Cai': (22.05954434800003, 104.34920012200006),
        'Nghệ An': (19.236395044000066, 104.94602442800003),
        'Ninh Bình': (20.308246808000153, 106.05150048400009),
        'Phú Thọ': (20.705206085000153, 105.33597150000007),
        'Quảng Ngãi': (14.767478504000108, 108.14492415100005),
        'Quảng Ninh': (21.258509962000172, 107.24335904700017),
        'Quảng Trị': (17.239325264000055, 106.52924471000004),
        'Sơn La': (21.192675261000087, 104.07150831800004),
        'Tây Ninh': (10.693153264999976, 106.1242355000001),
        'Thái Nguyên': (22.022424194000052, 105.82529783900013),
        'Thanh Hóa': (20.0455284250001, 105.31950083899999),
        'TP. Hồ Chí Minh': (10.993062602000066, 106.63774975000011),
        'Tuyên Quang': (22.488710736000087, 105.10102955900003),
        'Vĩnh Long': (9.996340202000003, 106.28911404300004)
    }
    
    # Data parameters
    NASA_PARAMETERS_HOURLY = [
        # Solar Fluxes and Related
        "ALLSKY_SFC_SW_DWN",   # All Sky Surface Shortwave Downward Irradiance
        "CLRSKY_SFC_SW_DWN",   # Clear Sky Surface Shortwave Downward Irradiance
        "ALLSKY_SFC_SW_DNI",   # All Sky Surface Shortwave Downward Direct Normal Irradiance
        "ALLSKY_SFC_SW_DIFF",  # All Sky Surface Shortwave Diffuse Irradiance
        "TOA_SW_DWN",          # Top-Of-Atmosphere Shortwave Downward Irradiance
        "ALLSKY_SFC_PAR_TOT",  # All Sky Surface PAR Total
        "CLRSKY_SFC_PAR_TOT",  # Clear Sky Surface PAR Total
        "ALLSKY_SFC_UVA",      # All Sky Surface UVA Irradiance
        "ALLSKY_SFC_UVB",      # All Sky Surface UVB Irradiance
        "ALLSKY_SFC_UV_INDEX",  # All Sky Surface UV Index
        
        # Temperatures
        "T2M",                 # Temperature at 2 Meters
        "T2MDEW",              # Dew/Frost Point at 2 Meters
        "T2MWET",              # Wet Bulb Temperature at 2 Meters
        
        # Humidity/Precipitation
        "QV2M",                # Specific Humidity at 2 Meters
        "RH2M",                # Relative Humidity at 2 Meters
        "PRECTOTCORR",         # Precipitation (Corrected)
        
        # Wind/Pressure
        "PS",                  # Surface Pressure
        "WS2M",                # Wind Speed at 2 Meters
        "WD2M",                # Wind Direction at 2 Meters
        "WS10M",               # Wind Speed at 10 Meters
        "WD10M"                # Wind Direction at 10 Meters
    ]

    NASA_PARAMETERS_DAILY = [
        # Solar Fluxes and Related
        "ALLSKY_SFC_SW_DWN",   # All Sky Surface Shortwave Downward Irradiance
        "CLRSKY_SFC_SW_DWN",   # Clear Sky Surface Shortwave Downward Irradiance
        "ALLSKY_SFC_SW_DNI",   # All Sky Surface Shortwave Downward Direct Normal Irradiance
        "ALLSKY_SFC_SW_DIFF",  # All Sky Surface Shortwave Diffuse Irradiance
        "TOA_SW_DWN",          # Top-Of-Atmosphere Shortwave Downward Irradiance
        "ALLSKY_SFC_PAR_TOT",  # All Sky Surface PAR Total
        "CLRSKY_SFC_PAR_TOT",  # Clear Sky Surface PAR Total
        "ALLSKY_SFC_UVA",      # All Sky Surface UVA Irradiance
        "ALLSKY_SFC_UVB",      # All Sky Surface UVB Irradiance
        "ALLSKY_SFC_UV_INDEX",  # All Sky Surface UV Index
        
        # Temperature/Thermal IR Flux
        "T2M",                 # Temperature at 2 Meters
        "T2MDEW",              # Dew/Frost Point at 2 Meters
        "T2MWET",              # Wet Bulb Temperature at 2 Meters
        "TS",                  # Earth Skin Temperature
        "T2M_RANGE",           # Temperature at 2 Meters Range
        "T2M_MAX",             # Temperature at 2 Meters Maximum
        "T2M_MIN",             # Temperature at 2 Meters Minimum
        "ALLSKY_SFC_LW_DWN",   # All Sky Surface Longwave Downward Irradiance
        
        # Humidity/Precipitation
        "QV2M",                # Specific Humidity at 2 Meters
        "RH2M",                # Relative Humidity at 2 Meters
        "PRECTOTCORR",         # Precipitation (Corrected from model)
        # "IMERG_PRECTOT",       # Precipitation (IMERG Multi-satellite) Need setting "time-standard": "UTC" to crawl
        
        # Wind/Pressure
        "PS",                  # Surface Pressure
        "WS2M",                # Wind Speed at 2 Meters
        "WS2M_MAX",            # Wind Speed at 2 Meters Maximum
        "WS2M_MIN",            # Wind Speed at 2 Meters Minimum
        "WS2M_RANGE",          # Wind Speed at 2 Meters Range
        "WD2M",                # Wind Direction at 2 Meters
        "WS10M",               # Wind Speed at 10 Meters
        "WS10M_MAX",           # Wind Speed at 10 Meters Maximum
        "WS10M_MIN",           # Wind Speed at 10 Meters Minimum
        "WS10M_RANGE",         # Wind Speed at 10 Meters Range
        "WD10M",               # Wind Direction at 10 Meters
        
        # Soil Properties
        "GWETTOP",             # Surface Soil Wetness
        "GWETROOT",            # Root Zone Soil Wetness
        "GWETPROF"             # Profile Soil Moisture
    ]
    
    # Column mapping (English -> Vietnamese)
    # NOTE: Mapping PHẢI cover TẤT CẢ parameters trong NASA_PARAMETERS_DAILY
    # và NASA_PARAMETERS_HOURLY để tránh cột lẫn lộn Anh/Việt sau rename.
    COLUMN_MAPPING = {
        # Solar Fluxes and Related
        "ALLSKY_SFC_SW_DWN": "Bức xạ sóng ngắn bề mặt",
        "CLRSKY_SFC_SW_DWN": "Bức xạ sóng ngắn trời quang",
        "ALLSKY_SFC_SW_DNI": "Bức xạ trực tiếp pháp tuyến",
        "ALLSKY_SFC_SW_DIFF": "Bức xạ khuếch tán",
        "TOA_SW_DWN": "Bức xạ đỉnh khí quyển",
        "ALLSKY_SFC_PAR_TOT": "Bức xạ quang hợp tổng",
        "CLRSKY_SFC_PAR_TOT": "Bức xạ quang hợp trời quang",
        "ALLSKY_SFC_UVA": "Bức xạ UVA",
        "ALLSKY_SFC_UVB": "Bức xạ UVB",
        "ALLSKY_SFC_UV_INDEX": "Chỉ số UV",
        
        # Temperature / Thermal IR Flux
        "T2M": "Nhiệt độ 2m",
        "T2MDEW": "Điểm sương 2m",
        "T2MWET": "Nhiệt độ bầu ướt 2m",
        "TS": "Nhiệt độ bề mặt đất",
        "T2M_RANGE": "Biên độ nhiệt 2m",
        "T2M_MAX": "Nhiệt độ tối đa 2m",
        "T2M_MIN": "Nhiệt độ tối thiểu 2m",
        "ALLSKY_SFC_LW_DWN": "Bức xạ sóng dài xuống",
        
        # Humidity / Precipitation
        "QV2M": "Độ ẩm tuyệt đối 2m",
        "RH2M": "Độ ẩm tương đối 2m",
        "PRECTOTCORR": "Lượng mưa",
        "IMERG_PRECTOT": "Lượng mưa vệ tinh IMERG",
        
        # Wind / Pressure
        "PS": "Áp suất bề mặt",
        "WS2M": "Tốc độ gió 2m",
        "WS2M_MAX": "Tốc độ gió tối đa 2m",
        "WS2M_MIN": "Tốc độ gió tối thiểu 2m",
        "WS2M_RANGE": "Biên độ gió 2m",
        "WD2M": "Hướng gió 2m",
        "WS10M": "Tốc độ gió 10m",
        "WS10M_MAX": "Tốc độ gió tối đa 10m",
        "WS10M_MIN": "Tốc độ gió tối thiểu 10m",
        "WS10M_RANGE": "Biên độ gió 10m",
        "WD10M": "Hướng gió 10m",
        
        # Soil Properties
        "GWETTOP": "Độ ẩm đất bề mặt",
        "GWETROOT": "Độ ẩm đất vùng rễ",
        "GWETPROF": "Độ ẩm đất mặt cắt",
        
        # Metadata columns
        "DATE": "Ngày",
        "LATITUDE": "Vĩ độ",
        "LONGITUDE": "Kinh độ",
    }
    
    # Tên cột target variable (lượng mưa) — EN & VI phải khớp với COLUMN_MAPPING.
    # dataquality.py dùng TARGET_COL_VI làm mặc định, tự fallback sang
    # TARGET_COL_EN nếu DataFrame chưa được rename (rename_columns=False).
    TARGET_COL_EN = "PRECTOTCORR"
    TARGET_COL_VI = "Lượng mưa"
    
    # Vietnamese Meteorological Standards Classification (24h precipitation)
    # Based on Vietnamese National Weather Service Standards
    PRECIPITATION_CLASSIFICATION = {
        "categories": {
            "no_rain": {
                "range": (0, 0),
                "label_vi": "Không mưa",
                "label_en": "No Rain",
                "description": "No precipitation"
            },
            "trace_rain": {
                "range": (0, 0.6),
                "label_vi": "Mưa lượng không đáng kể",
                "label_en": "Trace Rain",
                "description": "Negligible precipitation"
            },
            "light_rain": {
                "range": (0.6, 6.0),
                "label_vi": "Mưa nhỏ",
                "label_en": "Light Rain",
                "description": "Light precipitation"
            },
            "moderate_rain": {
                "range": (6.0, 16.0),
                "label_vi": "Mưa",
                "label_en": "Moderate Rain",
                "description": "Moderate precipitation"
            },
            "heavy_rain": {
                "range": (16.0, 50.0),
                "label_vi": "Mưa vừa",
                "label_en": "Heavy Rain",
                "description": "Heavy precipitation"
            },
            "very_heavy_rain": {
                "range": (50.0, 100.0),
                "label_vi": "Mưa to",
                "label_en": "Very Heavy Rain",
                "description": "Very heavy precipitation"
            },
            "extremely_heavy_rain": {
                "range": (100.0, float('inf')),
                "label_vi": "Mưa rất to",
                "label_en": "Extremely Heavy Rain",
                "description": "Extremely heavy precipitation"
            }
        },
        "thresholds": [0, 0.6, 6.0, 16.0, 50.0, 100.0]  # For easy access
    }
    
    # Default date range
    DEFAULT_START_DATE = "20000101"
    DEFAULT_END_DATE = "20260430"
    
    # Data quality thresholds — sử dụng trong dataquality.py để cảnh báo
    # khi tỷ lệ missing/duplicate vượt ngưỡng chấp nhận được.
    MISSING_DATA_THRESHOLD = 0.1  # 10% — tỷ lệ ô missing tối đa chấp nhận
    DUPLICATE_THRESHOLD = 0.05    # 5%  — tỷ lệ dòng trùng lặp tối đa chấp nhận