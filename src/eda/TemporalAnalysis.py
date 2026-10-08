# =============================================================================
# TEMPORAL STRUCTURE ANALYSIS — REFACTORED (PLAN v6)
# Pure Analysis-Visualization Suite: Calendar/Seasonal, Dual FFT, and MSTL
# Wavelet/CWT removed completely. Deterministic Leap-Day & Energy Conservation.
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional, Sequence
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from statsmodels.tsa.seasonal import MSTL

from ..config.constants import Config

warnings.filterwarnings('ignore')


class TemporalStructureAnalyzer:
    """Khảo sát cấu trúc thời gian: Mùa vụ, Phổ tần số FFT, và Phân rã MSTL.
    
    Cung cấp 3 cặp Phân tích - Trực quan hóa thuần túy (Pure Analysis-Visualization Pairs):
    1. Seasonal & Calendar Patterns (Climatology theo cặp Month-Day với làm mượt tuần hoàn 15 ngày)
    2. Dual FFT Frequency Analysis (Raw long-term [7, 730] + 30d detrended short-term [7, 30])
    3. MSTL Multi-Seasonal Decomposition (trên thang đo log1p với windows=(11, 15, 19) explicit)
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None, date_col: Optional[str] = None):
        """Initialize with configurable parameters."""
        self.df = df.copy()
        self.target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
        self.date_col = date_col or 'Ngày'
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Đảm bảo datetime, kiểm tra tính liên tục hàng ngày, tạo Month, Day, DayOfWeek."""
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')

        # Sắp xếp theo ngày
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

        # Kiểm tra tính liên tục thời gian hàng ngày (daily continuity)
        diffs = self.df[self.date_col].diff().dropna()
        if len(diffs) > 0:
            days_diff = diffs.dt.days
            if (days_diff != 1).any():
                warnings.warn(
                    f"Temporal series contains non-daily step gaps: min={days_diff.min()}d, max={days_diff.max()}d."
                )

        # Trích xuất các trường thời gian
        self.df['Month'] = self.df[self.date_col].dt.month
        self.df['Day'] = self.df[self.date_col].dt.day
        self.df['Year'] = self.df[self.date_col].dt.year
        self.df['DayOfYear'] = self.df[self.date_col].dt.dayofyear
        self.df['DayOfWeek'] = self.df[self.date_col].dt.dayofweek

    # =============================================================================
    # PAIR 1: SEASONAL PATTERNS & CALENDAR CLIMATOLOGY
    # =============================================================================

    def analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """Phân tích thống kê theo Tháng, Ngày trong năm và Thứ trong tuần.
        
        Quy ước mùa dự án (Project Domain Convention):
          - Mùa mưa (Wet season): Tháng 5, 6, 7, 8, 9, 10, 11 (May–Nov)
          - Mùa khô (Dry season): Tháng 12, 1, 2, 3, 4 (Dec–Apr)
          
        Chỉ số ngày mưa:
          - wet_day_pct_tau_1mm: Tỷ lệ ngày có lượng mưa Y >= 1.0 mm (quy ước dự án)
          - positive_precip_pct: Tỷ lệ ngày có lượng mưa Y > 0.0 mm
          
        Làm mượt Climatology:
          - Chuỗi ngày trong năm tổng hợp theo cặp (Month, Day) để tránh lệch pha năm nhuận.
          - Đường làm mượt 15 ngày căn giữa áp dụng cơ chế circular wrap-around padding (+/- 7 ngày).
          
        Returns:
            Dict chứa đầy đủ kết quả phân tích mùa vụ và chuỗi khí hậu.
        """
        wet_months = [5, 6, 7, 8, 9, 10, 11]
        dry_months = [12, 1, 2, 3, 4]

        # 1. Thống kê theo từng Tháng (1..12)
        monthly_records = []
        monthly_dict = {}
        for m in range(1, 13):
            sub = self.df[self.df['Month'] == m][self.target_col].dropna()
            m_mean = float(sub.mean()) if len(sub) > 0 else 0.0
            m_median = float(sub.median()) if len(sub) > 0 else 0.0
            m_std = float(sub.std()) if len(sub) > 0 else 0.0
            m_count = int(len(sub))
            m_wet_pct = float((sub >= 1.0).mean() * 100.0) if len(sub) > 0 else 0.0
            m_pos_pct = float((sub > 0.0).mean() * 100.0) if len(sub) > 0 else 0.0

            row = {
                'Month': m,
                'mean': round(m_mean, 4),
                'median': round(m_median, 4),
                'std': round(m_std, 4),
                'count': m_count,
                'wet_day_pct_tau_1mm': round(m_wet_pct, 2),
                'positive_precip_pct': round(m_pos_pct, 2),
                'season': 'Wet' if m in wet_months else 'Dry'
            }
            monthly_records.append(row)
            monthly_dict[str(m)] = {
                'mean': round(m_mean, 4),
                'median': round(m_median, 4),
                'std': round(m_std, 4),
                'wet_day_pct_tau_1mm': round(m_wet_pct, 2),
                'positive_precip_pct': round(m_pos_pct, 2),
            }

        monthly_df = pd.DataFrame(monthly_records)
        monthly_mean_series = monthly_df.set_index('Month')['mean']

        # 2. Climatology 365 ngày chuẩn theo cặp (Month, Day)
        # Loại trừ ngày 29/02 để tạo vector 365 ngày chuẩn
        df_standard = self.df[~((self.df['Month'] == 2) & (self.df['Day'] == 29))]
        daily_clim = df_standard.groupby(['Month', 'Day'])[self.target_col].mean().reset_index()
        daily_clim['DayOfYear_Index'] = np.arange(1, 366)
        raw_clim_values = daily_clim[self.target_col].values  # Độ dài 365

        # Thống kê riêng biệt cho ngày 29/02 (năm nhuận)
        df_leap = self.df[(self.df['Month'] == 2) & (self.df['Day'] == 29)]
        feb29_stats = {
            'count': int(len(df_leap)),
            'mean': float(round(df_leap[self.target_col].mean(), 4)) if len(df_leap) > 0 else 0.0,
            'years_observed': [int(y) for y in df_leap['Year'].unique()]
        }

        # Làm mượt tuần hoàn 15 ngày (Circular Wrap-Around Smoothing)
        half_w = 15 // 2  # 7 ngày đệm
        c_padded = np.concatenate([raw_clim_values[-half_w:], raw_clim_values, raw_clim_values[:half_w]])
        s_padded = pd.Series(c_padded).rolling(window=15, center=True).mean().values
        smoothed_clim_values = s_padded[half_w:-half_w]  # Chính xác 365 ngày, 0 NaNs

        daily_clim['smoothed_15d_circular'] = smoothed_clim_values

        # 3. Phân bố theo Thứ trong tuần (DayOfWeek: 0=Mon, 6=Sun)
        dow_stats = self.df.groupby('DayOfWeek')[self.target_col].agg(['mean', 'median', 'std', 'count']).round(4)

        return {
            'monthly_table': monthly_df,
            'monthly': monthly_df,
            'monthly_mean': monthly_mean_series,
            'monthly_dict': monthly_dict,
            'wet_months': wet_months,
            'dry_months': dry_months,
            'climatology_365': {
                'raw_daily': daily_clim,
                'raw_values': raw_clim_values,
                'smoothed_15d_circular': smoothed_clim_values,
                'feb29_stats': feb29_stats
            },
            'day_of_week': dow_stats,
            'climate_convention': {
                'definition_type': 'domain_fixed_project_convention',
                'convention_basis': 'project_domain_convention',
                'wet_season_months': wet_months,
                'dry_season_months': dry_months,
            }
        }

    def plot_seasonal_patterns(self,
                               seasonal_results: Dict[str, Any],
                               figsize: Tuple[int, int] = (16, 10),
                               show_grid: bool = True,
                               return_fig: bool = True) -> Optional[plt.Figure]:
        """Trực quan hóa cấu trúc mùa vụ & khí hậu (4 panels)."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Seasonal Patterns & Climatology Analysis', fontsize=16, fontweight='bold')

        # Panel 1: Phân bố lượng mưa theo từng tháng (Boxplot)
        sns.boxplot(data=self.df, x='Month', y=self.target_col, ax=axes[0, 0], palette='Blues')
        axes[0, 0].set_title('Monthly Precipitation Distribution')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Precipitation (mm)')
        if show_grid:
            axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: Lượng mưa trung bình theo tháng và chỉ dấu mùa khí hậu
        monthly_table = seasonal_results['monthly_table']
        colors = ['#2b83ba' if row['season'] == 'Wet' else '#fdae61' for _, row in monthly_table.iterrows()]
        bars = axes[0, 1].bar(monthly_table['Month'], monthly_table['mean'], color=colors, alpha=0.85, edgecolor='black')
        axes[0, 1].set_title('Average Monthly Precipitation (Wet: May-Nov, Dry: Dec-Apr)')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Mean Rainfall (mm/day)')
        axes[0, 1].set_xticks(range(1, 13))

        # Hiển thị tỷ lệ ngày mưa tau=1mm trên đỉnh cột
        for bar, wet_pct in zip(bars, monthly_table['wet_day_pct_tau_1mm']):
            yval = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{wet_pct:.0f}%",
                            ha='center', va='bottom', fontsize=9)
        if show_grid:
            axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Climatology 365 ngày chuẩn (Month-Day based) + Làm mượt tuần hoàn 15 ngày
        clim_data = seasonal_results['climatology_365']['raw_daily']
        axes[1, 0].plot(clim_data['DayOfYear_Index'], clim_data[self.target_col],
                        color='skyblue', alpha=0.5, linewidth=1.0, label='Raw Daily Mean (Month-Day)')
        axes[1, 0].plot(clim_data['DayOfYear_Index'], clim_data['smoothed_15d_circular'],
                        color='navy', linewidth=2.0, label='15-Day Circular Smoothed')
        axes[1, 0].set_title('Day-of-Year Climatology (Circular 15-day Smoothing, No Leap Drift)')
        axes[1, 0].set_xlabel('Day of Year (Calendar Month-Day, 1-365)')
        axes[1, 0].set_ylabel('Precipitation (mm)')
        axes[1, 0].legend(loc='upper right')
        if show_grid:
            axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: Phân bố theo Thứ trong tuần
        dow_df = seasonal_results['day_of_week']
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), dow_df['mean'], color='teal', alpha=0.75, edgecolor='black')
        axes[1, 1].set_title('Day-of-Week Precipitation Profile')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Mean Precipitation (mm)')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(dow_names)
        if show_grid:
            axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    # =============================================================================
    # PAIR 2: DUAL FFT FREQUENCY ANALYSIS (SCIPY PERIODOGRAM)
    # =============================================================================

    def analyze_fft(self,
                    top_n: int = 10,
                    raw_period_range: Tuple[int, int] = (7, 730),
                    short_term_period_range: Tuple[int, int] = (7, 30),
                    rtol: float = 1e-5,
                    atol: float = 1e-8) -> Dict[str, Any]:
        """Thực thi song song 2 phân tích phổ tần số qua scipy.signal.periodogram.
        
        Quy trình đánh giá đẳng thức năng lượng chuẩn mực:
          Periodogram toàn dải -> Kiểm tra Parseval/Mean-Square trên literal full array -> Lọc band -> Peak Selection.
          
        1. 'fft_raw':
           - Preprocessing: x_raw = Y - mean(Y) (N = len(df)). Mean-centered hoàn toàn.
           - Parseval Identity: np.sum(pxx_raw) == np.mean(x_raw**2) == var(x_raw, ddof=0).
           - Dải chẩn đoán & peak selection: raw_period_range = (7, 730) ngày inclusive.
           
        2. 'fft_short_term':
           - Preprocessing: x_short = Y - rolling_30_center(Y).
           - Xử lý biên: dropna() loại chính xác 29 NaNs biên (15 đầu, 14 cuối), N_valid = N - 29.
           - Không re-mean-center; trung bình thực nghiệm được giữ nguyên.
           - Năng lượng: np.sum(pxx_short) == np.mean(x_short_valid**2) == var + mean^2 (spectral mean-square energy).
           - Dải trích xuất đỉnh: short_term_period_range = (7, 30) ngày inclusive.
        """
        target_series = self.df.set_index(self.date_col)[self.target_col].dropna()
        n_total = len(target_series)

        # -------------------------------------------------------------
        # A. FFT RAW (Mean-Centering Duy Nhất)
        # -------------------------------------------------------------
        x_raw = target_series - target_series.mean()
        x_raw_vals = x_raw.values

        freqs_raw, pxx_raw = signal.periodogram(
            x_raw_vals, fs=1.0, window='boxcar', detrend=False, return_onesided=True, scaling='spectrum'
        )

        # Đánh giá Đẳng thức Parseval trên literal full array
        total_pwr_raw = float(np.sum(pxx_raw))
        mean_sq_raw = float(np.mean(x_raw_vals ** 2))
        var_raw = float(np.var(x_raw_vals, ddof=0))
        abs_err_raw = abs(total_pwr_raw - var_raw)
        rel_err_raw = abs_err_raw / var_raw if var_raw > 0 else 0.0
        passed_raw = bool(np.isclose(total_pwr_raw, var_raw, rtol=rtol, atol=atol))

        energy_diag_raw = {
            'total_power_full_spectrum': round(total_pwr_raw, 6),
            'time_domain_mean_square': round(mean_sq_raw, 6),
            'sample_variance': round(var_raw, 6),
            'absolute_error': round(abs_err_raw, 10),
            'relative_error': round(rel_err_raw, 10),
            'validation_tolerance': {'rtol': rtol, 'atol': atol},
            'passed': passed_raw
        }

        # Lọc dải chu kỳ raw: 7 <= T <= 730
        pos_raw = freqs_raw > 0
        freqs_pos_raw = freqs_raw[pos_raw]
        pxx_pos_raw = pxx_raw[pos_raw]
        periods_raw = 1.0 / freqs_pos_raw

        raw_mask = (periods_raw >= raw_period_range[0]) & (periods_raw <= raw_period_range[1])
        raw_indices = np.where(raw_mask)[0]

        # Peak selection Option B: Top-N local spectral bins ranked by power
        peaks_raw_rel, _ = signal.find_peaks(pxx_pos_raw[raw_mask])
        peaks_raw_idx = raw_indices[peaks_raw_rel]
        sorted_raw_idx = peaks_raw_idx[np.argsort(pxx_pos_raw[peaks_raw_idx])[::-1]]

        dominant_peaks_raw = []
        for rank, idx in enumerate(sorted_raw_idx[:top_n], start=1):
            dominant_peaks_raw.append({
                'rank': rank,
                'period_days': round(float(periods_raw[idx]), 2),
                'frequency': round(float(freqs_pos_raw[idx]), 6),
                'power': round(float(pxx_pos_raw[idx]), 6)
            })

        # Nearest peak to 365 days
        nearest_annual_peak = None
        if len(dominant_peaks_raw) > 0:
            nearest_annual_peak = min(dominant_peaks_raw, key=lambda p: abs(p['period_days'] - 365.0))

        fft_raw_results = {
            'dominant_peaks': dominant_peaks_raw,
            'dominant_periods': [p['period_days'] for p in dominant_peaks_raw],
            'detected_periods': [p['period_days'] for p in dominant_peaks_raw],
            'frequencies': freqs_raw,
            'spectrum_power': pxx_raw,
            'periods_positive': periods_raw,
            'power_positive': pxx_pos_raw,
            'energy_diagnostic': energy_diag_raw,
            'frequency_bin_width_cycles_per_day': round(1.0 / n_total, 7),
            'nearest_annual_peak': nearest_annual_peak,
            'preprocessing': {
                'method': 'mean_centering_only',
                'formula': 'x = y - mean(y)',
                'sample_size': n_total,
                'zero_padding': False,
                'power_definition': 'one_sided_power_spectrum_via_periodogram_scaling_spectrum',
                'energy_conservation': 'parseval_identity_holds_full_power_sum_equals_sample_variance'
            }
        }

        # -------------------------------------------------------------
        # B. FFT SHORT-TERM (Rolling 30-Day Subtraction)
        # -------------------------------------------------------------
        rolling_30 = target_series.rolling(window=30, center=True).mean()
        x_short_raw = target_series - rolling_30
        x_short_valid = x_short_raw.dropna()
        n_valid = len(x_short_valid)

        freqs_short, pxx_short = signal.periodogram(
            x_short_valid.values, fs=1.0, window='boxcar', detrend=False, return_onesided=True, scaling='spectrum'
        )

        # Đánh giá Năng lượng trên literal full array
        total_pwr_short = float(np.sum(pxx_short))
        mean_sq_short = float(np.mean(x_short_valid.values ** 2))
        var_short = float(np.var(x_short_valid.values, ddof=0))
        mean_short = float(np.mean(x_short_valid.values))
        abs_err_short = abs(total_pwr_short - mean_sq_short)
        rel_err_short = abs_err_short / mean_sq_short if mean_sq_short > 0 else 0.0
        passed_short = bool(np.isclose(total_pwr_short, mean_sq_short, rtol=rtol, atol=atol))

        energy_diag_short = {
            'total_power_full_spectrum': round(total_pwr_short, 6),
            'time_domain_mean_square': round(mean_sq_short, 6),
            'sample_variance': round(var_short, 6),
            'sample_mean': round(mean_short, 6),
            'absolute_error': round(abs_err_short, 10),
            'relative_error': round(rel_err_short, 10),
            'validation_tolerance': {'rtol': rtol, 'atol': atol},
            'passed': passed_short
        }

        # Lọc dải trích xuất đỉnh: short_term_period_range [7, 30]
        pos_short = freqs_short > 0
        freqs_pos_short = freqs_short[pos_short]
        pxx_pos_short = pxx_short[pos_short]
        periods_short = 1.0 / freqs_pos_short

        short_mask = (periods_short >= short_term_period_range[0]) & (periods_short <= short_term_period_range[1])
        short_indices = np.where(short_mask)[0]

        # Peak selection Option B: Top-N local spectral bins trong [7, 30]
        peaks_short_rel, _ = signal.find_peaks(pxx_pos_short[short_mask])
        peaks_short_idx = short_indices[peaks_short_rel]
        sorted_short_idx = peaks_short_idx[np.argsort(pxx_pos_short[peaks_short_idx])[::-1]]

        dominant_peaks_short = []
        for rank, idx in enumerate(sorted_short_idx[:top_n], start=1):
            dominant_peaks_short.append({
                'rank': rank,
                'period_days': round(float(periods_short[idx]), 2),
                'frequency': round(float(freqs_pos_short[idx]), 6),
                'power': round(float(pxx_pos_short[idx]), 6)
            })

        fft_short_results = {
            'dominant_peaks': dominant_peaks_short,
            'dominant_periods': [p['period_days'] for p in dominant_peaks_short],
            'detected_periods': [p['period_days'] for p in dominant_peaks_short],
            'frequencies': freqs_short,
            'spectrum_power': pxx_short,
            'periods_positive': periods_short,
            'power_positive': pxx_pos_short,
            'energy_diagnostic': energy_diag_short,
            'frequency_bin_width_cycles_per_day': round(1.0 / n_valid, 7),
            'effective_date_range': {
                'start': str(x_short_valid.index.min().date()),
                'end': str(x_short_valid.index.max().date())
            },
            'boundary_nans': {
                'head_nans': int(x_short_raw.iloc[:20].isna().sum()),
                'tail_nans': int(x_short_raw.iloc[-20:].isna().sum()),
                'total_nans': int(x_short_raw.isna().sum())
            },
            'preprocessing': {
                'method': 'centered_rolling_mean_subtraction',
                'window_days': 30,
                'center': True,
                'boundary_handling': 'dropna_29_boundary_nans (15 leading NaNs, 14 trailing NaNs)',
                'effective_sample_size': n_valid,
                'filter_characteristic': 'suppresses low-frequency / longer-period variation and exhibits sinc-like frequency response; 30 days is not an ideal hard frequency cutoff',
                'power_definition': 'spectral_mean_square_energy_not_automatically_equal_to_sample_variance',
                'sample_mean_handling': 'valid short-term series is not re-mean-centered; empirical mean retained and computed at runtime'
            }
        }

        return {
            'success': True,
            'fft_raw': fft_raw_results,
            'fft_short_term': fft_short_results,
            # Backward compatibility aliases
            'dominant_periods': [p['period_days'] for p in dominant_peaks_short],
            'detected_periods': [p['period_days'] for p in dominant_peaks_short],
            'raw_dominant_periods': [p['period_days'] for p in dominant_peaks_raw],
            'parameters': {
                'top_n': top_n,
                'raw_period_range': raw_period_range,
                'short_term_period_range': short_term_period_range,
                'detrend_window_fixed': 30
            }
        }

    def plot_spectrum(self,
                      fft_results: Dict[str, Any],
                      figsize: Tuple[int, int] = (16, 8),
                      return_fig: bool = True) -> Optional[plt.Figure]:
        """Trực quan hóa phổ tần số Dual FFT (2 panels đối chiếu)."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Dual Frequency Domain Analysis (Power Spectrum via Periodogram)', fontsize=16, fontweight='bold')

        raw_res = fft_results['fft_raw']
        short_res = fft_results['fft_short_term']

        # Panel 1: Raw FFT Spectrum (Dải chu kỳ dài)
        p_raw = raw_res['periods_positive']
        pwr_raw = raw_res['power_positive']
        axes[0].plot(p_raw, pwr_raw, color='darkblue', linewidth=1.2, label='One-Sided Power Spectrum')
        axes[0].set_xlim(raw_res['parameters']['raw_period_range'][0] if 'parameters' in raw_res else 7, 730)
        axes[0].set_title('Raw FFT Spectrum (Long-Period Structure & Annual Evidence)')
        axes[0].set_xlabel('Period (days)')
        axes[0].set_ylabel('Power (mm²)')
        axes[0].axvline(x=365, color='crimson', linestyle='--', linewidth=1.5, label='Nominal Annual (365d)')
        axes[0].grid(True, alpha=0.3)

        # Highlight top 3 raw peaks
        for p in raw_res['dominant_peaks'][:3]:
            axes[0].plot(p['period_days'], p['power'], 'ro', markersize=6)
            axes[0].annotate(f"{p['period_days']:.1f}d", (p['period_days'], p['power']),
                             textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
        axes[0].legend(loc='upper right')

        # Panel 2: 30-Day Detrended Spectrum (Dải chu kỳ ngắn hạn trong tháng)
        p_short = short_res['periods_positive']
        pwr_short = short_res['power_positive']
        axes[1].plot(p_short, pwr_short, color='teal', linewidth=1.2, label='30d-Detrended Power Spectrum')
        axes[1].set_xlim(7, 100)  # Vẽ đến 100d để thấy rõ attenuation roll-off
        axes[1].set_title('Short-Term FFT Spectrum (Intra-Monthly Diagnostic [7, 30]d)')
        axes[1].set_xlabel('Period (days)')
        axes[1].set_ylabel('Power (mm²)')
        axes[1].axvline(x=30, color='gray', linestyle=':', linewidth=1.5, label='Moving Average Window (30d)')
        axes[1].grid(True, alpha=0.3)

        # Highlight top 3 short-term peaks trong [7, 30]
        for p in short_res['dominant_peaks'][:3]:
            axes[1].plot(p['period_days'], p['power'], 'go', markersize=6)
            axes[1].annotate(f"{p['period_days']:.1f}d", (p['period_days'], p['power']),
                             textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
        axes[1].legend(loc='upper right')

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    # =============================================================================
    # PAIR 3: MSTL MULTI-SEASONAL DECOMPOSITION
    # =============================================================================

    def decompose_mstl(self,
                       periods: Sequence[int] = (7, 30, 365),
                       windows: Sequence[int] = (11, 15, 19),
                       iterate: int = 2) -> Dict[str, Any]:
        """Phân rã MSTL deterministic trên z_t = log1p(Y_t) với statsmodels 0.15.0.
        
        Args:
            periods: Tuple các chu kỳ mùa vụ, mặc định (7, 30, 365)
            windows: Tuple độ dài bộ làm mượt LOESS cho từng mùa vụ, mặc định (11, 15, 19)
            iterate: Số lượt lặp tinh chỉnh, mặc định 2
            
        Returns:
            Dict chứa đầy đủ kết quả phân rã MSTL, thành phần phần dư resid (pd.Series)
            và các tỷ số phương sai mô tả component_variance_ratio.
        """
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        # Thang đo phân tích: z_t = log(1 + Y_t)
        transformed_ts = np.log1p(target_ts)

        # Khởi tạo và khớp mô hình MSTL
        periods_list = list(periods)
        windows_list = list(windows)

        mstl = MSTL(
            transformed_ts,
            periods=periods_list,
            windows=windows_list,
            iterate=iterate
        )
        mstl_result = mstl.fit()

        # Khóa cứng bất biến căn chỉnh index
        assert mstl_result.resid.index.equals(transformed_ts.index), (
            "MSTL resid index does not strictly equal input series DatetimeIndex."
        )

        # Tính toán chỉ số phương sai mô tả (Relative Component Variance)
        total_var = float(np.var(transformed_ts.values, ddof=0))
        trend_var = float(np.var(mstl_result.trend.values, ddof=0))
        resid_var = float(np.var(mstl_result.resid.values, ddof=0))

        comp_variance_ratios = {
            'trend': round(trend_var / total_var, 4) if total_var > 0 else 0.0,
            'residual': round(resid_var / total_var, 4) if total_var > 0 else 0.0,
        }

        # Tính tỷ số phương sai cho từng thành phần mùa vụ
        seasonal_df = mstl_result.seasonal
        for col in seasonal_df.columns:
            s_var = float(np.var(seasonal_df[col].values, ddof=0))
            comp_variance_ratios[col] = round(s_var / total_var, 4) if total_var > 0 else 0.0

        period_semantic_labels = {
            '7': 'weekly / calendar-scale seasonality',
            '30': '30-day sub-seasonal / intra-month recurring component (not MJO)',
            '365': 'annual-scale macro seasonality'
        }

        return {
            'success': True,
            'mstl_obj': mstl_result,
            'original_ts': target_ts,
            'transformed_ts': transformed_ts,
            'transform_method': 'log1p',
            'periods': periods_list,
            'windows': windows_list,
            'requested_windows': windows_list,
            'resolved_windows': windows_list,
            'iterate': iterate,
            'trend': mstl_result.trend,
            'seasonal': mstl_result.seasonal,
            'resid': mstl_result.resid,
            'component_variance_ratios': comp_variance_ratios,
            'variance_ratios': comp_variance_ratios,
            'period_semantic_labels': period_semantic_labels,
            'variance_diagnostics': {
                'diagnostic_type': 'relative_component_variance_descriptive',
                'interpretation_note': 'Ratios indicate marginal relative dispersion Var(C)/Var(z); components are non-orthogonal, so ratios do not represent orthogonal variance attribution and do not necessarily sum to 100% due to cross-component covariances.',
                'total_variance': round(total_var, 6),
                'component_variance_ratios': comp_variance_ratios
            }
        }

    def plot_decomposition(self,
                           mstl_results: Dict[str, Any],
                           figsize: Tuple[int, int] = (15, 12),
                           return_fig: bool = True) -> Optional[plt.Figure]:
        """Trực quan hóa phân rã MSTL đa tầng (6 panels chuẩn mực)."""
        transformed_ts = mstl_results['transformed_ts']
        trend = mstl_results['trend']
        seasonal = mstl_results['seasonal']
        resid = mstl_results['resid']

        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
        fig.suptitle('MSTL Multi-Seasonal Decomposition on log1p(Y_t)', fontsize=16, fontweight='bold')

        # 1. Chuỗi biến đổi gốc
        axes[0].plot(transformed_ts.index, transformed_ts.values, color='black', linewidth=1.0)
        axes[0].set_ylabel('log1p(Rain)')
        axes[0].set_title('Observed Series: log1p(Y_t)')
        axes[0].grid(True, alpha=0.3)

        # 2. Xu thế (Trend)
        axes[1].plot(trend.index, trend.values, color='darkorange', linewidth=1.5)
        axes[1].set_ylabel('Trend')
        axes[1].set_title('Trend Component')
        axes[1].grid(True, alpha=0.3)

        # 3. Mùa vụ tuần (Seasonal 7)
        if 'seasonal_7' in seasonal.columns:
            axes[2].plot(seasonal.index, seasonal['seasonal_7'].values, color='royalblue', linewidth=1.0)
            axes[2].set_ylabel('Seasonal 7d')
            axes[2].set_title('Weekly Seasonality (s=7)')
            axes[2].grid(True, alpha=0.3)

        # 4. Dao động tháng (Seasonal 30)
        if 'seasonal_30' in seasonal.columns:
            axes[3].plot(seasonal.index, seasonal['seasonal_30'].values, color='mediumseagreen', linewidth=1.0)
            axes[3].set_ylabel('Seasonal 30d')
            axes[3].set_title('Sub-Seasonal / Intra-Month Component (s=30, not MJO)')
            axes[3].grid(True, alpha=0.3)

        # 5. Mùa vụ năm (Seasonal 365)
        if 'seasonal_365' in seasonal.columns:
            axes[4].plot(seasonal.index, seasonal['seasonal_365'].values, color='purple', linewidth=1.2)
            axes[4].set_ylabel('Seasonal 365d')
            axes[4].set_title('Annual Macro Seasonality (s=365)')
            axes[4].grid(True, alpha=0.3)

        # 6. Phần dư (Residual)
        axes[5].plot(resid.index, resid.values, color='crimson', linewidth=0.8, alpha=0.8)
        axes[5].set_ylabel('Residual')
        axes[5].set_title('Residual Component (for 01.5 Stationarity Diagnostics)')
        axes[5].grid(True, alpha=0.3)

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    # =============================================================================
    # ORCHESTRATION & COMPATIBILITY
    # =============================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """Điều phối chạy toàn bộ 3 phân tích: Seasonal, Dual FFT, và MSTL."""
        seasonal_results = self.analyze_seasonal_patterns()
        fft_results = self.analyze_fft()
        mstl_results = self.decompose_mstl()

        results = {
            'seasonal_patterns': seasonal_results,
            'seasonal': seasonal_results,
            'wet_season_months': seasonal_results['wet_months'],
            'dry_season_months': seasonal_results['dry_months'],
            'frequency_analysis': fft_results,
            'fft': fft_results,
            'mstl_decomposition': mstl_results,
            'mstl': mstl_results,
            'component_name': 'TemporalStructureAnalyzer_DualFFT_MSTL'
        }
        return results

    def analyze(self) -> Dict[str, Any]:
        """Standard analyzer protocol alias."""
        return self.analyze_all()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_temporal_structure(df: pd.DataFrame,
                               target_col: Optional[str] = None,
                               date_col: Optional[str] = None) -> Dict[str, Any]:
    """Hàm tiện ích chạy toàn bộ khảo sát cấu trúc thời gian."""
    analyzer = TemporalStructureAnalyzer(df, target_col, date_col)
    return analyzer.analyze_all()


def get_mstl_residual_for_stationarity(df: pd.DataFrame,
                                       target_col: Optional[str] = None,
                                       date_col: Optional[str] = None,
                                       periods: Sequence[int] = (7, 30, 365),
                                       windows: Sequence[int] = (11, 15, 19)) -> Optional[pd.Series]:
    """Hàm tiện ích trích xuất chuỗi phần dư MSTL cho phân tích tính dừng của 01.5."""
    analyzer = TemporalStructureAnalyzer(df, target_col, date_col)
    mstl_res = analyzer.decompose_mstl(periods=periods, windows=windows)
    if mstl_res.get('success', False):
        return mstl_res['resid']
    return None