"""Configuration constants for the DS108 Weather Prediction Project"""

import os
from pathlib import Path

class Config:
    """Central configuration for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_ROOT / "raw"
    PROCESSED_DATA_DIR = DATA_ROOT / "processed"
    # Canonical DATA_DIR points to data/raw for raw data ingestion & crawling
    DATA_DIR = RAW_DATA_DIR
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # NASA POWER API configuration
    NASA_API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_API_TIMEOUT = 120
    NASA_API_MAX_PARAMS = 20
    
    # Vietnam province coordinates
    # Tọa độ chuẩn hóa cho 63 tỉnh/thành phố và 2 huyện đảo đặc biệt (Hoàng Sa, Trường Sa).
    # Xác định theo nguyên tắc: Trung tâm hành chính đô thị / Trạm Khí tượng Thủy văn tỉnh lỵ,
    # đảm bảo nằm trong nội địa đất liền để tránh rơi vào ô lưới biển (Marine Grid Cell) của NASA POWER.
    province_coordinates = {
        # 5 Thành phố trực thuộc Trung ương
        'Hà Nội': (21.0285, 105.8542),          # Trạm KT Láng / UBND TP (Hoàn Kiếm)
        'TP. Hồ Chí Minh': (10.7769, 106.7009),  # Trạm KT Tân Sơn Hòa / UBND TP (Quận 1) - Cùng ô lưới MERRA-2 với Củ Chi (10.993, 106.638)
        'Hải Phòng': (20.8651, 106.6838),       # Trạm KT Phù Liễn / Hồng Bàng
        'Đà Nẵng': (16.0544, 108.2022),         # Trạm KT Đà Nẵng / Hải Châu
        'Cần Thơ': (10.0452, 105.7469),         # Trạm KT Cần Thơ / Ninh Kiều

        # Vùng Đồng bằng sông Hồng (8 tỉnh)
        'Bắc Ninh': (21.1861, 106.0763),        # TP. Bắc Ninh
        'Hà Nam': (20.5452, 105.9122),          # TP. Phủ Lý
        'Hải Dương': (20.9386, 106.3155),       # TP. Hải Dương
        'Hưng Yên': (20.6464, 106.0511),        # TP. Hưng Yên
        'Nam Định': (20.4344, 106.1775),        # TP. Nam Định
        'Ninh Bình': (20.2539, 105.9750),       # TP. Ninh Bình
        'Thái Bình': (20.4463, 106.3366),       # TP. Thái Bình
        'Vĩnh Phúc': (21.3089, 105.6049),       # TP. Vĩnh Yên

        # Vùng Đông Bắc Bộ (9 tỉnh)
        'Hà Giang': (22.8233, 104.9839),        # TP. Hà Giang
        'Cao Bằng': (22.6657, 106.2570),        # TP. Cao Bằng
        'Bắc Kạn': (22.1470, 105.8348),         # TP. Bắc Kạn
        'Tuyên Quang': (21.8236, 105.2144),     # TP. Tuyên Quang
        'Lào Cai': (22.4856, 103.9707),         # TP. Lào Cai
        'Yên Bái': (21.7168, 104.9113),         # TP. Yên Bái
        'Thái Nguyên': (21.5942, 105.8482),     # TP. Thái Nguyên
        'Lạng Sơn': (21.8537, 106.7615),        # TP. Lạng Sơn
        'Quảng Ninh': (20.9505, 107.0734),      # TP. Hạ Long

        # Vùng Tây Bắc Bộ (6 tỉnh)
        'Điện Biên': (21.3860, 103.0232),       # TP. Điện Biên Phủ
        'Lai Châu': (22.3963, 103.4624),        # TP. Lai Châu
        'Sơn La': (21.3283, 103.9188),          # TP. Sơn La
        'Hòa Bình': (20.8172, 105.3376),        # TP. Hòa Bình
        'Phú Thọ': (21.3228, 105.4019),         # TP. Việt Trì
        'Bắc Giang': (21.2731, 106.1946),       # TP. Bắc Giang

        # Vùng Bắc Trung Bộ (6 tỉnh)
        'Thanh Hóa': (19.8067, 105.7852),       # TP. Thanh Hóa
        'Nghệ An': (18.6734, 105.6813),         # TP. Vinh
        'Hà Tĩnh': (18.3435, 105.9058),         # TP. Hà Tĩnh
        'Quảng Bình': (17.4689, 106.6225),      # TP. Đồng Hới
        'Quảng Trị': (16.8163, 107.1004),       # TP. Đông Hà
        'Thừa Thiên Huế': (16.4637, 107.5909),  # TP. Huế
        'Huế': (16.4637, 107.5909),             # Alias cho Thừa Thiên Huế

        # Vùng Duyên hải Nam Trung Bộ (8 tỉnh)
        'Quảng Nam': (15.5736, 108.4840),       # TP. Tam Kỳ
        'Quảng Ngãi': (15.1213, 108.7923),      # TP. Quảng Ngãi
        'Bình Định': (13.7830, 109.2197),       # TP. Quy Nhơn
        'Phú Yên': (13.0882, 109.3175),         # TP. Tuy Hòa
        'Khánh Hòa': (12.2388, 109.1967),       # TP. Nha Trang
        'Ninh Thuận': (11.5681, 108.9864),      # TP. Phan Rang - Tháp Chàm
        'Bình Thuận': (10.9273, 108.1019),      # TP. Phan Thiết

        # Vùng Tây Nguyên (5 tỉnh)
        'Kon Tum': (14.3497, 108.0005),         # TP. Kon Tum
        'Gia Lai': (13.9831, 108.0004),         # TP. Pleiku
        'Đắk Lắk': (12.6662, 108.0383),        # TP. Buôn Ma Thuột
        'Đắk Nông': (12.0033, 107.6871),        # TP. Gia Nghĩa
        'Lâm Đồng': (11.9404, 108.4583),        # TP. Đà Lạt

        # Vùng Đông Nam Bộ (5 tỉnh)
        'Bà Rịa - Vũng Tàu': (10.4966, 107.1685), # TP. Bà Rịa (trung tâm hành chính)
        'Bình Dương': (10.9805, 106.6519),        # TP. Thủ Dầu Một
        'Bình Phước': (11.5333, 106.8833),        # TP. Đồng Xoài
        'Đồng Nai': (10.9574, 106.8427),          # TP. Biên Hòa
        'Tây Ninh': (11.3131, 106.0988),          # TP. Tây Ninh

        # Vùng Đồng bằng sông Cửu Long (13 tỉnh)
        'An Giang': (10.3759, 105.4358),        # TP. Long Xuyên
        'Bạc Liêu': (9.2941, 105.7278),         # TP. Bạc Liêu
        'Bến Tre': (10.2415, 106.3759),         # TP. Bến Tre
        'Cà Mau': (9.1769, 105.1524),           # TP. Cà Mau
        'Đồng Tháp': (10.4538, 105.6322),       # TP. Cao Lãnh
        'Hậu Giang': (9.7845, 105.4701),        # TP. Vị Thanh
        'Kiên Giang': (10.0125, 105.0809),      # TP. Rạch Giá
        'Long An': (10.5364, 106.4104),         # TP. Tân An
        'Sóc Trăng': (9.6033, 105.9722),        # TP. Sóc Trăng
        'Tiền Giang': (10.3600, 106.3600),      # TP. Mỹ Tho
        'Trà Vinh': (9.9347, 106.3455),         # TP. Trà Vinh
        'Vĩnh Long': (10.2537, 105.9722),       # TP. Vĩnh Long

        # 2 Quần đảo / Huyện đảo đặc biệt của Việt Nam
        'Hoàng Sa': (16.5333, 111.6000),             # Đảo Hoàng Sa - Huyện đảo Hoàng Sa, TP. Đà Nẵng
        'Quần đảo Hoàng Sa': (16.5333, 111.6000),    # Alias
        'Trường Sa': (8.6439, 111.9197),             # Đảo Trường Sa Lớn - Huyện đảo Trường Sa, Tỉnh Khánh Hòa
        'Quần đảo Trường Sa': (8.6439, 111.9197)     # Alias
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
    # Reference: QCVN 47:2018/BTNMT (Quy chuẩn kỹ thuật quốc gia về dự báo, cảnh báo khí tượng thủy văn)
    # and Quyết định 18/2021/QĐ-TTg của Thủ tướng Chính phủ (Quy định về dự báo, cảnh báo thiên tai).
    PRECIPITATION_CLASSIFICATION = {
        "authority": "Tổng cục Khí tượng Thủy văn / Bộ Tài nguyên và Môi trường Việt Nam",
        "standard": "QCVN 47:2018/BTNMT & Quyết định 18/2021/QĐ-TTg",
        "time_window": "24h daily accumulated precipitation (mm)",
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