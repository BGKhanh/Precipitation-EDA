import requests
import pandas as pd
import time
import os
from datetime import datetime
import json

class NASAPowerCrawlerCitySimplified:
    """
    Class để crawl dữ liệu từ NASA POWER API cho toàn thành phố TP.HCM
    Version đơn giản với 20 parameters - crawl 1 lần duy nhất
    """

    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        self.parameters = [
            "PRECTOTCORR",           # Precipitation
            "QV2M",                  # Specific Humidity at 2 Meters
            "RH2M",                  # Relative Humidity at 2 Meters
            "T2M",                   # Temperature at 2 Meters
            "T2MDEW",                # Dew/Frost Point at 2 Meters
            "T2MWET",                # Wet Bulb Temperature at 2 Meters
            "T2M_MAX",               # Temperature at 2 Meters Maximum
            "TS",                    # Earth Skin Temperature
            "T2M_MIN",               # Temperature at 2 Meters Minimum
            "ALLSKY_SFC_LW_DWN",     # All Sky Surface Longwave Downward Irradiance
            "PS",                    # Surface Pressure
            "WS10M",                 # Wind Speed at 10 Meters
            "WD10M",                 # Wind Direction at 10 Meters
            "WS10M_MAX",             # Wind Speed at 10 Meters Maximum
            'WS2M_MAX',              # Wind Speed at 2 Meters Maximum
            'WS2M',                  # Wind Speed at 2 Meters
            'WD2M',                  # Wind Direction at 2 Meters
            "GWETPROF",              # Profile Soil Moisture
            "GWETTOP",               # Surface Soil Wetness
            "GWETROOT"               # Root Zone Soil Wetness
        ]

        # Tọa độ trung tâm TP.HCM
        self.hcmc_center = {
            "lat": 10.78,  # Trung tâm TP.HCM (chính xác hơn)
            "lon": 106.7
        }

        # Tạo thư mục lưu dữ liệu
        self.output_dir = "nasa_power_hcmc_data"
        os.makedirs(self.output_dir, exist_ok=True)

    def crawl_data(self, start_date, end_date):
        """
        Crawl tất cả dữ liệu trong 1 lần duy nhất

        Args:
            start_date: Ngày bắt đầu YYYYMMDD
            end_date: Ngày kết thúc YYYYMMDD
        """
        print("🚀 BẮT ĐẦU CRAWL DỮ LIỆU CHO TP.HỒ CHÍ MINH")
        print("="*80)
        print(f"📅 Thời gian: {start_date} → {end_date}")
        print(f"📊 Tổng số parameters: {len(self.parameters)}")
        print(f"📍 Vị trí: lat={self.hcmc_center['lat']}, lon={self.hcmc_center['lon']}")

        # Validate số lượng parameters
        if len(self.parameters) > 20:
            print(f"⚠️ CẢNH BÁO: {len(self.parameters)} parameters vượt quá giới hạn API (20)")
            return None

        # Setup request parameters
        params = {
            "parameters": ",".join(self.parameters),
            "community": "RE",
            "longitude": self.hcmc_center["lon"],
            "latitude": self.hcmc_center["lat"],
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }

        print(f"\n🌍 Đang crawl dữ liệu từ NASA POWER API...")
        print(f"   📊 Parameters: {len(self.parameters)} indicators")

        try:
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=120)
            response.raise_for_status()

            data = response.json()

            # Kiểm tra lỗi API
            if "messages" in data and data["messages"]:
                print(f"   ❌ Lỗi API: {data['messages']}")
                return None

            # Parse JSON data
            properties = data.get("properties", {})
            parameters_data = properties.get("parameter", {})

            if not parameters_data:
                print(f"   ❌ Không có dữ liệu được trả về")
                return None

            # Tạo DataFrame từ JSON data
            df_dict = {}
            for param, values in parameters_data.items():
                df_dict[param] = values

            df = pd.DataFrame(df_dict)

            # Thêm metadata columns
            df["DATE"] = pd.to_datetime(df.index)
            df["LATITUDE"] = self.hcmc_center["lat"]
            df["LONGITUDE"] = self.hcmc_center["lon"]

            print(f"   ✅ Crawl thành công: {df.shape[0]:,} days × {df.shape[1]} columns")

            # Lưu dataset
            saved_df = self.save_dataset(df, start_date, end_date)

            return saved_df

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Lỗi network request: {str(e)}")
            return None
        except Exception as e:
            print(f"   ❌ Lỗi xử lý dữ liệu: {str(e)}")
            return None

    def save_dataset(self, df, start_date, end_date):
        """
        Lưu dataset và tạo báo cáo tóm tắt
        """
        try:
            # Tạo tên file
            output_filename = f"hcmc_weather_data_{start_date}_{end_date}.csv"
            output_filepath = os.path.join(self.output_dir, output_filename)

            # Lưu CSV file
            df.to_csv(output_filepath, index=False, encoding='utf-8')

            print(f"\n💾 ĐÃ LƯU DATASET THÀNH CÔNG!")
            print(f"   📁 File: {output_filename}")
            print(f"   📂 Path: {output_filepath}")
            print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"   📅 Date range: {df['DATE'].min()} → {df['DATE'].max()}")

            # Tạo báo cáo chi tiết
            self.generate_data_summary(df)

            return df

        except Exception as e:
            print(f"   ❌ Lỗi khi lưu dataset: {str(e)}")
            return None

    def generate_data_summary(self, df):
        """
        Tạo báo cáo tóm tắt chi tiết về dữ liệu
        """
        print("\n📋 BÁO CÁO TÓM TẮT DỮ LIỆU WEATHER TP.HCM")
        print("="*60)

        # Basic info
        print(f"📊 Kích thước dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"📅 Khoảng thời gian: {df['DATE'].min().strftime('%Y-%m-%d')} → {df['DATE'].max().strftime('%Y-%m-%d')}")
        print(f"📈 Số ngày dữ liệu: {df['DATE'].nunique():,} days")

        # Tính số năm
        years = (df['DATE'].max() - df['DATE'].min()).days / 365.25
        print(f"⏱️ Tổng thời gian: {years:.1f} năm")

        # Weather parameters info
        weather_params = [col for col in df.columns if col not in ['DATE', 'LATITUDE', 'LONGITUDE']]
        print(f"🌤️ Số weather parameters: {len(weather_params)}")

        print(f"\n📋 Danh sách parameters:")
        for i, param in enumerate(weather_params, 1):
            print(f"   {i:2d}. {param}")

        # Data quality checks
        print(f"\n🔍 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU:")

        # Missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]

        if len(missing_cols) > 0:
            print(f"⚠️ Missing values tìm thấy:")
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                print(f"   - {col}: {count:,} ({pct:.1f}%)")
        else:
            print(f"✅ Không có missing values")

        # Check for -999 (NASA missing indicator)
        print(f"\n🔍 Kiểm tra giá trị -999 (NASA missing indicator):")
        has_999 = False
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_cols:
            count_999 = (df[col] == -999).sum()
            if count_999 > 0:
                pct = (count_999 / len(df)) * 100
                print(f"   ⚠️ {col}: {count_999:,} values = -999 ({pct:.1f}%)")
                has_999 = True

        if not has_999:
            print(f"✅ Không có giá trị -999")

    def validate_parameters(self):
        """
        Validate danh sách parameters
        """
        print("🔍 VALIDATION PARAMETERS:")
        print(f"   📊 Tổng số: {len(self.parameters)}")

        print(f"\n📋 Danh sách parameters sẽ crawl:")
        for i, param in enumerate(self.parameters, 1):
            print(f"   {i:2d}. {param}")

        return len(self.parameters) <= 20

def main():
    """
    Hàm chính để thực hiện crawl dữ liệu
    """
    print("🌟 NASA POWER WEATHER DATA CRAWLER FOR HO CHI MINH CITY")
    print("🔄 SIMPLIFIED VERSION - SINGLE REQUEST")
    print("="*80)

    # Khởi tạo crawler
    crawler = NASAPowerCrawlerCitySimplified()

    # Validate parameters trước khi crawl
    if not crawler.validate_parameters():
        print("❌ Parameters validation failed!")
        return None

    # Định nghĩa khoảng thời gian
    start_date_str = "20000101"  # 1/1/2000
    end_date_str = "20250430"    # 30/04/2025

    print(f"\n🎯 Mục tiêu: Dự đoán thời tiết toàn TP.HCM")
    print(f"📍 Vị trí: Trung tâm TP.HCM")
    print(f"⚠️ Đã loại bỏ 2 chỉ số bức xạ: ALLSKY_SFC_SW_DWN, TOA_SW_DWN")

    # Thực hiện crawl
    final_dataset = crawler.crawl_data(start_date_str, end_date_str)

    if final_dataset is not None:
        print("\n🎉 HOÀN TẤT CRAWL DỮ LIỆU THÀNH CÔNG!")
        print("="*80)
        print(f"✨ Dataset sẵn sàng cho EDA và modeling")
        print(f"📄 File đã lưu trong thư mục: {crawler.output_dir}")
        return final_dataset
    else:
        print("\n❌ CRAWL DỮ LIỆU THẤT BẠI!")
        return None

# Chạy crawler
if __name__ == "__main__":
    final_dataset = main()
    if final_dataset is not None:
        print(f"\n🎯 Sử dụng biến 'final_dataset' để truy cập dữ liệu")
        print(f"📊 Shape: {final_dataset.shape}")
    print("Hoàn tất quá trình crawl dữ liệu")