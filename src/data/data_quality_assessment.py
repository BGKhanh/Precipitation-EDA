# -*- coding: utf-8 -*-
"""
Data Quality Assessment Module for Weather Time Series Data
Purpose: Comprehensive data quality evaluation for precipitation prediction project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataQualityAssessment:
    """
    Comprehensive Data Quality Assessment for Weather Data
    """

    def __init__(self, df, target_col='Lượng mưa', date_col='Ngày'):
        """
        Initialize Data Quality Assessment

        Args:
            df: DataFrame to assess
            target_col: Target variable column name
            date_col: Date column name
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.n_rows = len(self.df)
        self.n_cols = len(self.df.columns)

        print("🔍 DATA QUALITY ASSESSMENT INITIALIZED")
        print("="*60)
        print(f"📊 Dataset Overview:")
        print(f"   - Shape: {self.df.shape}")
        print(f"   - Target Variable: {self.target_col}")
        print(f"   - Date Column: {self.date_col}")
        print(f"   - Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def basic_info_analysis(self):
        """
        1. Phân tích thông tin cơ bản về dataset
        """
        print("\n" + "="*60)
        print("📋 1. BASIC DATASET INFORMATION")
        print("="*60)

        # Dataset shape and size
        print(f"📏 Kích thước Dataset:")
        print(f"   - Số dòng (observations): {self.n_rows:,}")
        print(f"   - Số cột (variables): {self.n_cols}")
        print(f"   - Tổng ô dữ liệu: {self.n_rows * self.n_cols:,}")

        # Data types analysis
        print(f"\n📊 Phân tích kiểu dữ liệu:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count} columns ({count/self.n_cols*100:.1f}%)")

        # Column categories
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()

        print(f"\n📂 Phân loại cột:")
        print(f"   - Numerical columns: {len(self.numerical_cols)}")
        print(f"   - Categorical columns: {len(self.categorical_cols)}")
        print(f"   - Datetime columns: {len(self.datetime_cols)}")

        return {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'dtype_counts': dtype_counts,
            'column_categories': {
                'numerical': self.numerical_cols,
                'categorical': self.categorical_cols,
                'datetime': self.datetime_cols
            }
        }

    def missing_values_analysis(self):
        """
        2. Phân tích chi tiết về missing values
        """
        print("\n" + "="*60)
        print("❓ 2. MISSING VALUES ANALYSIS")
        print("="*60)

        # Calculate missing values for each column
        missing_data = []
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percentage = (missing_count / self.n_rows) * 100
            non_null_count = self.df[col].notna().sum()
            unique_count = self.df[col].nunique()

            missing_data.append({
                'Column': col,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage,
                'Non_Null_Count': non_null_count,
                'Unique_Values': unique_count,
                'Data_Type': str(self.df[col].dtype)
            })

        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)

        # Overall missing data statistics
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.n_rows * self.n_cols
        overall_missing_pct = (total_missing / total_cells) * 100

        print(f"📊 Tổng quan Missing Data:")
        print(f"   - Tổng số ô thiếu dữ liệu: {total_missing:,}")
        print(f"   - Tỷ lệ thiếu dữ liệu tổng thể: {overall_missing_pct:.2f}%")

        # Columns with missing data
        cols_with_missing = missing_df[missing_df['Missing_Count'] > 0]

        # Initialize empty DataFrames to avoid UnboundLocalError
        high_missing = pd.DataFrame()
        medium_missing = pd.DataFrame()
        low_missing = pd.DataFrame()

        if len(cols_with_missing) > 0:
            print(f"\n🚨 Các cột có dữ liệu thiếu ({len(cols_with_missing)} cột):")
            print(cols_with_missing[['Column', 'Missing_Count', 'Missing_Percentage', 'Data_Type']].to_string(index=False))

            # Categorize missing data severity
            high_missing = cols_with_missing[cols_with_missing['Missing_Percentage'] > 50]
            medium_missing = cols_with_missing[(cols_with_missing['Missing_Percentage'] > 10) &
                                             (cols_with_missing['Missing_Percentage'] <= 50)]
            low_missing = cols_with_missing[cols_with_missing['Missing_Percentage'] <= 10]

            print(f"\n📋 Phân loại mức độ thiếu dữ liệu:")
            print(f"   - Thiếu nghiêm trọng (>50%): {len(high_missing)} cột")
            print(f"   - Thiếu trung bình (10-50%): {len(medium_missing)} cột")
            print(f"   - Thiếu ít (<10%): {len(low_missing)} cột")

            if len(high_missing) > 0:
                print(f"   ⚠️ Cột thiếu nghiêm trọng: {high_missing['Column'].tolist()}")
        else:
            print("✅ Không có cột nào thiếu dữ liệu!")

        # Check target variable missing data
        if self.target_col in missing_df['Column'].values:
            target_missing = missing_df[missing_df['Column'] == self.target_col]['Missing_Percentage'].iloc[0]
            if target_missing > 0:
                print(f"\n🎯 Biến mục tiêu '{self.target_col}' thiếu {target_missing:.2f}% dữ liệu")
            else:
                print(f"\n✅ Biến mục tiêu '{self.target_col}' không thiếu dữ liệu")

        return missing_df, {
            'total_missing': total_missing,
            'overall_missing_pct': overall_missing_pct,
            'columns_with_missing': len(cols_with_missing),
            'high_missing_cols': high_missing['Column'].tolist() if len(high_missing) > 0 else [],
            'medium_missing_cols': medium_missing['Column'].tolist() if len(medium_missing) > 0 else [],
            'low_missing_cols': low_missing['Column'].tolist() if len(low_missing) > 0 else []
        }

    def duplicate_analysis(self):
        """
        3. Phân tích dữ liệu trùng lặp
        """
        print("\n" + "="*60)
        print("🔄 3. DUPLICATE DATA ANALYSIS")
        print("="*60)

        # Full row duplicates
        full_duplicates = self.df.duplicated().sum()
        full_duplicate_pct = (full_duplicates / self.n_rows) * 100

        print(f"📊 Dòng trùng lặp hoàn toàn:")
        print(f"   - Số dòng trùng lặp: {full_duplicates:,}")
        print(f"   - Tỷ lệ trùng lặp: {full_duplicate_pct:.2f}%")

        # Check for duplicates excluding certain columns (like ID columns)
        key_columns = [col for col in self.df.columns
                      if col not in ['Vĩ độ', 'Kinh độ']]  # Exclude coordinate columns

        key_duplicates = 0
        if len(key_columns) < len(self.df.columns):
            key_duplicates = self.df.duplicated(subset=key_columns).sum()
            key_duplicate_pct = (key_duplicates / self.n_rows) * 100

            print(f"\n📊 Dòng trùng lặp (loại trừ tọa độ):")
            print(f"   - Số dòng trùng lặp: {key_duplicates:,}")
            print(f"   - Tỷ lệ trùng lặp: {key_duplicate_pct:.2f}%")

        # Check for date + location duplicates (if applicable)
        date_location_duplicates = 0
        if self.date_col in self.df.columns and 'Quận Huyện' in self.df.columns:
            date_location_duplicates = self.df.duplicated(subset=[self.date_col, 'Quận Huyện']).sum()
            date_location_dup_pct = (date_location_duplicates / self.n_rows) * 100

            print(f"\n📊 Trùng lặp ngày + địa điểm:")
            print(f"   - Số dòng trùng lặp: {date_location_duplicates:,}")
            print(f"   - Tỷ lệ trùng lặp: {date_location_dup_pct:.2f}%")

            if date_location_duplicates > 0:
                print("   ⚠️ Có thể có dữ liệu đo đạc trùng lặp cho cùng ngày và địa điểm!")

        # Check for potential duplicate patterns in target variable
        if self.target_col in self.df.columns:
            target_value_counts = self.df[self.target_col].value_counts()
            most_common_value = target_value_counts.iloc[0]
            most_common_pct = (most_common_value / self.n_rows) * 100

            print(f"\n🎯 Phân tích giá trị trùng lặp của biến mục tiêu:")
            print(f"   - Giá trị phổ biến nhất: {target_value_counts.index[0]}")
            print(f"   - Số lần xuất hiện: {most_common_value:,} ({most_common_pct:.2f}%)")

            # Check for excessive zeros in precipitation data
            if target_value_counts.index[0] == 0:
                zero_pct = (target_value_counts.iloc[0] / self.n_rows) * 100
                print(f"   📊 Tỷ lệ ngày không mưa: {zero_pct:.2f}%")
                if zero_pct > 70:
                    print("   ⚠️ Tỷ lệ ngày không mưa rất cao - cần xem xét!")

        return {
            'full_duplicates': full_duplicates,
            'full_duplicate_pct': full_duplicate_pct,
            'key_duplicates': key_duplicates,
            'date_location_duplicates': date_location_duplicates
        }


    def data_consistency_check(self):
        """
        5. Kiểm tra tính nhất quán của dữ liệu
        """
        print("\n" + "="*60)
        print("🔍 4. DATA CONSISTENCY CHECK")
        print("="*60)

        consistency_issues = []

        # Check for negative values where they shouldn't exist
        weather_vars_positive = [
            'Lượng mưa', 'Độ ẩm tương đối 2m', 'Tốc độ gió 10m', 'Tốc độ gió 2m',
            'Tốc độ gió tối đa 10m', 'Tốc độ gió tối đa 2m'
        ]

        print("🔍 Kiểm tra giá trị âm cho các biến phải dương:")
        for col in weather_vars_positive:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    negative_pct = (negative_count / len(self.df)) * 100
                    print(f"   ⚠️ {col}: {negative_count} giá trị âm ({negative_pct:.2f}%)")
                    consistency_issues.append(f"{col} has {negative_count} negative values")
                else:
                    print(f"   ✅ {col}: Không có giá trị âm")

        # Check humidity ranges (should be 0-100%)
        humidity_cols = ['Độ ẩm tương đối 2m']
        print(f"\n🔍 Kiểm tra phạm vi độ ẩm (0-100%):")
        for col in humidity_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                out_of_range = data[(data < 0) | (data > 100)]
                if len(out_of_range) > 0:
                    print(f"   ⚠️ {col}: {len(out_of_range)} giá trị ngoài phạm vi [0,100]")
                    print(f"      - Min: {data.min():.2f}, Max: {data.max():.2f}")
                    consistency_issues.append(f"{col} has values outside [0,100] range")
                else:
                    print(f"   ✅ {col}: Tất cả giá trị trong phạm vi [0,100]")

        # Check wind direction ranges (should be 0-360 degrees)
        wind_direction_cols = ['Hướng gió 10m', 'Hướng gió 2m']
        print(f"\n🔍 Kiểm tra phạm vi hướng gió (0-360 độ):")
        for col in wind_direction_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                out_of_range = data[(data < 0) | (data > 360)]
                if len(out_of_range) > 0:
                    print(f"   ⚠️ {col}: {len(out_of_range)} giá trị ngoài phạm vi [0,360]")
                    print(f"      - Min: {data.min():.2f}, Max: {data.max():.2f}")
                    consistency_issues.append(f"{col} has values outside [0,360] range")
                else:
                    print(f"   ✅ {col}: Tất cả giá trị trong phạm vi [0,360]")

        # Check temperature consistency (min <= avg <= max)
        temp_cols = {
            'min': 'Nhiệt độ tối thiểu 2m',
            'avg': 'Nhiệt độ 2m',
            'max': 'Nhiệt độ tối đa 2m'
        }

        if all(col in self.df.columns for col in temp_cols.values()):
            print(f"\n🔍 Kiểm tra tính nhất quán nhiệt độ (min ≤ avg ≤ max):")
            temp_min = self.df[temp_cols['min']]
            temp_avg = self.df[temp_cols['avg']]
            temp_max = self.df[temp_cols['max']]

            # Check min <= avg
            min_avg_violations = (temp_min > temp_avg).sum()
            # Check avg <= max
            avg_max_violations = (temp_avg > temp_max).sum()
            # Check min <= max
            min_max_violations = (temp_min > temp_max).sum()

            total_violations = min_avg_violations + avg_max_violations + min_max_violations

            if total_violations > 0:
                print(f"   ⚠️ Phát hiện {total_violations} vi phạm tính nhất quán nhiệt độ:")
                if min_avg_violations > 0:
                    print(f"      - T_min > T_avg: {min_avg_violations} trường hợp")
                if avg_max_violations > 0:
                    print(f"      - T_avg > T_max: {avg_max_violations} trường hợp")
                if min_max_violations > 0:
                    print(f"      - T_min > T_max: {min_max_violations} trường hợp")
                consistency_issues.append(f"Temperature consistency violations: {total_violations}")
            else:
                print(f"   ✅ Tất cả dữ liệu nhiệt độ nhất quán")

        # Check for extremely unrealistic values
        print(f"\n🔍 Kiểm tra giá trị cực đoan không thực tế:")

        # Precipitation > 500mm/day (extremely rare)
        if self.target_col in self.df.columns:
            extreme_rain = (self.df[self.target_col] > 500).sum()
            if extreme_rain > 0:
                print(f"   ⚠️ {self.target_col}: {extreme_rain} ngày có mưa >500mm (cực đoan)")
                max_rain = self.df[self.target_col].max()
                print(f"      - Lượng mưa tối đa: {max_rain:.2f}mm")
                consistency_issues.append(f"Extreme precipitation: {extreme_rain} days >500mm")
            else:
                print(f"   ✅ {self.target_col}: Không có giá trị cực đoan")

        # Temperature extremes
        if 'Nhiệt độ 2m' in self.df.columns:
            temp_data = self.df['Nhiệt độ 2m'].dropna()
            extreme_hot = (temp_data > 45).sum()  # >45°C
            extreme_cold = (temp_data < 10).sum()  # <10°C for tropical climate

            if extreme_hot > 0 or extreme_cold > 0:
                print(f"   ⚠️ Nhiệt độ cực đoan:")
                if extreme_hot > 0:
                    print(f"      - Quá nóng (>45°C): {extreme_hot} ngày")
                if extreme_cold > 0:
                    print(f"      - Quá lạnh (<10°C): {extreme_cold} ngày")
                consistency_issues.append(f"Extreme temperatures: {extreme_hot} hot, {extreme_cold} cold days")
            else:
                print(f"   ✅ Nhiệt độ: Không có giá trị cực đoan")

        return {
            'consistency_issues': consistency_issues,
            'total_issues': len(consistency_issues)
        }

    def temporal_quality_check(self):
        """
        5. Kiểm tra chất lượng dữ liệu thời gian
        """
        print("\n" + "="*60)
        print("📅 5. TEMPORAL DATA QUALITY CHECK")
        print("="*60)

        if self.date_col not in self.df.columns:
            print(f"❌ Không tìm thấy cột ngày '{self.date_col}'")
            return {
                'invalid_dates': 0,
                'date_range_days': 0,
                'missing_dates_count': 0,
                'duplicate_dates': 0
            }

        # Ensure datetime format
        df_temp = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_temp[self.date_col]):
            print(f"🔄 Chuyển đổi cột '{self.date_col}' sang datetime...")
            df_temp[self.date_col] = pd.to_datetime(df_temp[self.date_col], errors='coerce')

        # Check for invalid dates
        invalid_dates = df_temp[self.date_col].isnull().sum()
        if invalid_dates > 0:
            print(f"⚠️ Phát hiện {invalid_dates} ngày không hợp lệ")
        else:
            print(f"✅ Tất cả ngày đều hợp lệ")

        # Date range analysis
        valid_dates = df_temp[self.date_col].dropna()
        missing_dates_count = 0
        duplicate_dates = 0
        date_range_days = 0

        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            date_range_days = (max_date - min_date).days

            print(f"\n📊 Phạm vi thời gian:")
            print(f"   - Ngày bắt đầu: {min_date.strftime('%Y-%m-%d')}")
            print(f"   - Ngày kết thúc: {max_date.strftime('%Y-%m-%d')}")
            print(f"   - Tổng khoảng: {date_range_days:,} ngày ({date_range_days/365.25:.1f} năm)")

            # Check for missing dates in sequence
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            actual_dates = set(valid_dates.dt.date)
            expected_dates_set = set(expected_dates.date)
            missing_dates = expected_dates_set - actual_dates
            missing_dates_count = len(missing_dates)

            if missing_dates:
                print(f"\n⚠️ Thiếu {len(missing_dates)} ngày trong chuỗi thời gian:")
                if len(missing_dates) <= 10:
                    for date in sorted(missing_dates)[:10]:
                        print(f"   - {date}")
                else:
                    print(f"   - Hiển thị 10 ngày đầu: {sorted(missing_dates)[:10]}")
            else:
                print(f"\n✅ Chuỗi thời gian liên tục, không thiếu ngày nào")

            # Check for duplicate dates
            duplicate_dates = valid_dates[valid_dates.duplicated()].nunique()
            if duplicate_dates > 0:
                print(f"\n⚠️ Phát hiện {duplicate_dates} ngày bị trùng lặp")
            else:
                print(f"\n✅ Không có ngày trùng lặp")

        return {
            'invalid_dates': invalid_dates,
            'date_range_days': date_range_days,
            'missing_dates_count': missing_dates_count,
            'duplicate_dates': duplicate_dates
        }

    def generate_quality_report(self):
        # Run all assessments
        basic_info = self.basic_info_analysis()
        missing_df, missing_summary = self.missing_values_analysis()
        duplicate_summary = self.duplicate_analysis()
        consistency_summary = self.data_consistency_check()
        temporal_summary = self.temporal_quality_check()

        # Overall quality score calculation
        quality_score = 100.0

        # Deduct points for issues
        if missing_summary['overall_missing_pct'] > 0:
            quality_score -= min(missing_summary['overall_missing_pct'] * 2, 30)

        if duplicate_summary['full_duplicate_pct'] > 0:
            quality_score -= min(duplicate_summary['full_duplicate_pct'] * 3, 20)

        if consistency_summary['total_issues'] > 0:
            quality_score -= min(consistency_summary['total_issues'] * 5, 25)

        if temporal_summary and temporal_summary['invalid_dates'] > 0:
            invalid_date_pct = (temporal_summary['invalid_dates'] / self.n_rows) * 100
            quality_score -= min(invalid_date_pct * 2, 15)

        quality_score = max(quality_score, 0)  # Ensure non-negative

        print(f"\n🏆 OVERALL DATA QUALITY SCORE: {quality_score:.1f}/100")

        # Quality interpretation
        if quality_score >= 90:
            quality_level = "🟢 EXCELLENT"
            recommendation = "Dữ liệu có chất lượng rất tốt, có thể tiến hành phân tích ngay."
        elif quality_score >= 80:
            quality_level = "🟡 GOOD"
            recommendation = "Dữ liệu có chất lượng tốt, có thể cần một số xử lý nhỏ."
        elif quality_score >= 70:
            quality_level = "🟠 FAIR"
            recommendation = "Dữ liệu cần được xử lý và làm sạch trước khi phân tích."
        elif quality_score >= 60:
            quality_level = "🔴 POOR"
            recommendation = "Dữ liệu có nhiều vấn đề, cần xử lý kỹ lưỡng."
        else:
            quality_level = "⛔ CRITICAL"
            recommendation = "Dữ liệu có vấn đề nghiêm trọng, cần xem xét lại nguồn dữ liệu."

        print(f"📊 Quality Level: {quality_level}")
        print(f"💡 Recommendation: {recommendation}")

        # Detailed summary
        print(f"\n📋 Chi tiết các vấn đề:")
        print(f"   - Missing Data: {missing_summary['overall_missing_pct']:.2f}%")
        print(f"   - Duplicate Rows: {duplicate_summary['full_duplicate_pct']:.2f}%")
        print(f"   - Consistency Issues: {consistency_summary['total_issues']}")
        print(f"   - Invalid Dates: {temporal_summary['invalid_dates']}")
        print(f"   - Missing Dates: {temporal_summary['missing_dates_count']}")



        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'recommendation': recommendation,
            'basic_info': basic_info,
            'missing_summary': missing_summary,
            'duplicate_summary': duplicate_summary,
            'consistency_summary': consistency_summary,
            'temporal_summary': temporal_summary,
        }

# =============================================================================
# USAGE FUNCTION
# =============================================================================

def assess_data_quality(df, target_col='Lượng mưa', date_col='Ngày'):
    """
    Chạy đánh giá chất lượng dữ liệu toàn diện

    Args:
        df: DataFrame cần đánh giá
        target_col: Tên cột biến mục tiêu
        date_col: Tên cột ngày tháng

    Returns:
        dict: Báo cáo chi tiết về chất lượng dữ liệu
    """

    print("🚀 STARTING COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print("="*80)

    # Initialize assessment
    dqa = DataQualityAssessment(df, target_col, date_col)

    # Generate comprehensive report
    quality_report = dqa.generate_quality_report()

    print("\n✅ DATA QUALITY ASSESSMENT COMPLETED")
    print("="*80)

    return quality_report

# =============================================================================
# RUN ASSESSMENT ON YOUR DATA
# =============================================================================

# Chạy đánh giá chất lượng dữ liệu cho DataFrame df_all
quality_report = assess_data_quality(df_all, target_col='Lượng mưa', date_col='Ngày')


