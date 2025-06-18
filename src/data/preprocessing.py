data_dir = "nasa_power_hcmc_data"
start_date_str = "20000101"  # 1/1/2000
end_date_str = "20250430"    # 30/04/2025

import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- 1. Rename Columns (Vietnamese Standard) ---
column_mapping = {
    "PRECTOTCORR": "Lượng mưa",           # Precipitation
    "QV2M": "Độ ẩm tuyệt đối 2m",                  # Specific Humidity at 2 Meters
    "RH2M": "Độ ẩm tương đối 2m",                  # Relative Humidity at 2 Meters
    "T2M": "Nhiệt độ 2m",                   # Temperature at 2 Meters
    "T2MDEW": "Điểm sương 2m",                # Dew/Frost Point at 2 Meters
    "T2MWET": "Nhiệt độ bầu ướt 2m",                # Wet Bulb Temperature at 2 Meters
    "T2M_MAX": "Nhiệt độ tối đa 2m",               # Temperature at 2 Meters Maximum
    "TS": "Nhiệt độ bề mặt đất",                    # Earth Skin Temperature
    "T2M_MIN": "Nhiệt độ tối thiểu 2m",               # Temperature at 2 Meters Minimum
    "ALLSKY_SFC_LW_DWN": "Bức xạ sóng dài xuống",     # All Sky Surface Longwave Downward Irradiance
    "PS": "Áp suất bề mặt",                    # Surface Pressure
    "WS10M": "Tốc độ gió 10m",                 # Wind Speed at 10 Meters
    "WD10M": "Hướng gió 10m",                 # Wind Direction at 10 Meters
    "WS10M_MAX": "Tốc độ gió tối đa 10m",             # Wind Speed at 10 Meters Maximum
    'WS2M_MAX': "Tốc độ gió tối đa 2m",              # MERRA-2 Wind Speed at 2 Meters Maximum (m/s)
    'WS2M': "Tốc độ gió 2m",                  # MERRA-2 Wind Speed at 2 Meters (m/s)
    'WD2M': "Hướng gió 2m",                  # MERRA-2 Wind Direction at 2 Meters (Degrees)
    "GWETPROF": "Độ ẩm đất mặt cắt",              # Profile Soil Moisture (surface to bedrock)
    "GWETTOP": "Độ ẩm đất bề mặt",               # Surface Soil Wetness (surface to 5 cm below)
    "GWETROOT": "Độ ẩm đất vùng rễ",               # Root Zone Soil Wetness (surface to 100 cm below)
    # Cột thông tin
    "DATE": "Ngày",
    "LATITUDE": "Vĩ độ",
    "LONGITUDE": "Kinh độ",
}


# --- 2. Load Data ---
hcmc_complete_file_path = os.path.join(data_dir, f"hcmc_weather_data_{start_date_str}_{end_date_str}.csv")

print("🌟 LOADING WEATHER DATA FOR HO CHI MINH CITY")
print("="*60)

# Load dữ liệu hoàn chỉnh cho TP.HCM
if final_dataset is None:
  try:
      df_all = pd.read_csv(hcmc_complete_file)
      print(f"✅ Đã tải thành công file: {os.path.basename(hcmc_complete_file)}")
      print(f"📊 Kích thước ban đầu: {df_all.shape[0]:,} rows × {df_all.shape[1]} columns")

      # Rename columns
      original_columns = len(df_all.columns)
      df_all.rename(columns=column_mapping, inplace=True)
      print(f"📝 Đã đổi tên {original_columns} cột sang tiếng Việt")

  except FileNotFoundError:
      print(f"❌ Lỗi: Không tìm thấy file '{hcmc_complete_file}'")
      print(f"💡 Vui lòng chạy crawler trước để tạo dữ liệu!")
      df_all = None
  except Exception as e:
      print(f"❌ Lỗi khi tải file '{hcmc_complete_file}': {e}")
      df_all = None
else:
  df_all = final_dataset
  original_columns = len(df_all.columns)
  df_all.rename(columns=column_mapping, inplace=True)

if df_all is not None:
    # --- 3. Check Basic Info ---
    print("\n📋 THÔNG TIN CƠ BẢN CỦA DỮ LIỆU TP.HCM")
    print("="*60)

    print("\n1. 📏 Kích thước (Shape):")
    print(f"   {df_all.shape[0]:,} rows × {df_all.shape[1]} columns")

    print("\n2. 📊 Thông tin biến (Info):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df_all.info(verbose=True, show_counts=True)

    print("\n3. 📈 Thống kê mô tả (Describe):")
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    print(f"   Hiển thị thống kê cho {len(numeric_cols)} cột số:")
    print(df_all[numeric_cols].describe().T)

    # --- 4. Correct Date Format ---
    print("\n📅 CHỈNH SỬA ĐỊNH DẠNG NGÀY")
    print("="*40)

    def correct_date_format(df, date_column='Ngày'):
        """Converts a date column to datetime objects."""
        if date_column in df.columns:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[date_column]):
                print(f"✅ Cột '{date_column}' đã ở định dạng datetime")
                return df
            try:
                # Try different date formats
                if df[date_column].dtype == 'object':
                    # If string, try to parse
                    df[date_column] = pd.to_datetime(df[date_column])
                else:
                    # If numeric (YYYYMMDD), convert to string first
                    df[date_column] = df[date_column].astype(str)
                    df[date_column] = pd.to_datetime(df[date_column], format='%Y%m%d')

                print(f"✅ Đã chuyển đổi cột '{date_column}' sang datetime")
                print(f"   📅 Khoảng thời gian: {df[date_column].min()} → {df[date_column].max()}")
                print(f"   📊 Số ngày dữ liệu: {df[date_column].nunique():,} days")

            except Exception as e:
                print(f"❌ Lỗi khi chuyển đổi cột '{date_column}': {e}")
                print("💡 Kiểm tra lại định dạng dữ liệu trong file CSV")
        else:
            print(f"⚠️ Không tìm thấy cột '{date_column}' để chuyển đổi")
        return df

    df_all = correct_date_format(df_all)

    print(f"📊 Tổng số weather parameters: {len([col for col in df_all.columns if col not in ['Ngày', 'Vĩ độ', 'Kinh độ', 'Nhóm']])}")

else:
    print("\n❌ Không thể load dữ liệu. Vui lòng kiểm tra lại!")