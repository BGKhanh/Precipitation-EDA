# =============================================================================
# BASE ANALYZER CLASS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10



class BaseAnalyzer:
    """
    Base class cho tất cả analysis components
    """
    def __init__(self, df, target_col='Lượng mưa', date_col='Ngày'):
        """Initialize base analyzer with common data preparation"""
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self._prepare_data()

    def _prepare_data(self):
        """Common data preparation for all analyzers"""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')

        # Get numerical columns (excluding coordinates)
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        coord_cols = ['Vĩ độ', 'Kinh độ']
        self.analysis_cols = [col for col in self.numerical_cols if col not in coord_cols]

        # Sort by date
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

        # Add time features
        self.df['Month'] = self.df[self.date_col].dt.month
        self.df['Year'] = self.df[self.date_col].dt.year
        self.df['DayOfYear'] = self.df[self.date_col].dt.dayofyear
        self.df['DayOfWeek'] = self.df[self.date_col].dt.dayofweek
        self.df['Season'] = self.df['Month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2,
                                                 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})

    def analyze(self):
        """Abstract method - must implement in subclasses"""
        raise NotImplementedError("Subclasses must implement analyze() method")


# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class WeatherEDAAnalyzer:
    """Updated main orchestrator with new TemporalStructureAnalyzer"""

    def __init__(self, df, target_col='Lượng mưa', date_col='Ngày'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        print("🚀 WEATHER EDA ANALYZER INITIALIZED")
        print("="*70)
        print(f"   📊 Dataset Shape: {self.df.shape}")
        print(f"   🎯 Target Variable: {self.target_col}")

    def run_temporal_structure_analysis(self):
        """Run comprehensive temporal structure analysis"""
        self.temporal_structure = TemporalStructureAnalyzer(self.df, self.target_col, self.date_col)
        return self.temporal_structure.analyze()

    # Keep other existing methods...
    def run_extreme_events_analysis(self):
        self.extreme_events = ExtremeEventsAnalyzer(self.df, self.target_col, self.date_col)
        return self.extreme_events.analyze()

    def run_stationarity_autocorr_analysis(self, mstl_results=None):
        """Run combined stationarity and autocorrelation analysis"""
        self.stationarity_autocorr = StationarityAutocorrelationAnalyzer(
            self.df, self.target_col, self.date_col, mstl_results
        )
        return self.stationarity_autocorr.analyze()
    
    
    # =============================================================================
# ENHANCED COMPONENT: TEMPORAL STRUCTURE ANALYZER
# Combines: Time Series Decomposition + Frequency Analysis + Advanced Analysis
# =============================================================================

class TemporalStructureAnalyzer(BaseAnalyzer):
    """
    Enhanced Component: Comprehensive Temporal Structure Analysis
    Combines temporal patterns, frequency domain, and advanced decomposition
    """

    # Update main analyze method to include wavelet analysis
    def analyze(self):
        """Main orchestration function - runs complete temporal analysis workflow"""
        print("\n" + "="*70)
        print("🕐 ENHANCED COMPONENT: TEMPORAL STRUCTURE ANALYSIS")
        print("="*70)

        # Steps 1-4 (existing code...)
        visual_results = self._visual_seasonal_pattern_analysis()
        frequency_results = self._frequency_domain_analysis()
        mstl_results = self._mstl_decomposition(frequency_results.get('dominant_periods', [7, 30, 365]))

        trend_analysis = {}
        seasonal_analysis = {}
        residual_analysis = {}
        wavelet_analysis = {}

        if mstl_results['success']:
            trend_analysis = self._trend_cyclical_analysis(mstl_results)
            seasonal_analysis = self._seasonal_components_analysis(mstl_results)
            residual_analysis = self._residual_analysis(mstl_results)

            # Step 5: NEW - Wavelet analysis
            wavelet_analysis = self._wavelet_analysis(mstl_results)

        # Combine all results
        results = {
            'visual_patterns': visual_results,
            'frequency_analysis': frequency_results,
            'mstl_decomposition': mstl_results,
            'trend_analysis': trend_analysis,
            'seasonal_analysis': seasonal_analysis,
            'residual_analysis': residual_analysis,
            'wavelet_analysis': wavelet_analysis,
            'component_name': 'TemporalStructureAnalyzer'
        }

        print(f"\n✅ TEMPORAL STRUCTURE ANALYSIS COMPLETED")
        return results

    def _visual_seasonal_pattern_analysis(self):
        """
        Step 1: Visual seasonal pattern analysis (from TimeSeriesAnalyzer)
        """
        print("🔍 Step 1: Visual Seasonal Pattern Analysis")
        print("-" * 50)

        # Monthly statistics
        monthly_stats = self.df.groupby('Month')[self.target_col].agg(['mean', 'std', 'median', 'count']).round(4)

        # Plot seasonal patterns
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Visual Seasonal Patterns Analysis', fontsize=16, fontweight='bold')

        # Monthly box plot
        sns.boxplot(data=self.df, x='Month', y=self.target_col, ax=axes[0,0])
        axes[0,0].set_title('Monthly Distribution')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Precipitation (mm)')
        axes[0,0].grid(True, alpha=0.3)

        # Monthly average
        monthly_mean = self.df.groupby('Month')[self.target_col].mean()
        axes[0,1].bar(monthly_mean.index, monthly_mean.values, color='skyblue', alpha=0.7)
        axes[0,1].set_title('Average Monthly Precipitation')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Average Precipitation (mm)')
        axes[0,1].grid(True, alpha=0.3)

        # Daily pattern within year
        daily_pattern = self.df.groupby('DayOfYear')[self.target_col].mean()
        axes[1,0].plot(daily_pattern.index, daily_pattern.values, 'b-', linewidth=1, alpha=0.7)
        axes[1,0].set_title('Daily Pattern Throughout Year')
        axes[1,0].set_xlabel('Day of Year')
        axes[1,0].set_ylabel('Average Precipitation (mm)')
        axes[1,0].grid(True, alpha=0.3)

        # Yearly trend
        yearly_stats = self.df.groupby('Year')[self.target_col].mean()
        axes[1,1].plot(yearly_stats.index, yearly_stats.values, 'ro-', linewidth=2, markersize=6)
        axes[1,1].set_title('Yearly Trend')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Average Precipitation (mm)')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Identify seasons
        wet_months = monthly_mean[monthly_mean > monthly_mean.quantile(0.75)].index.tolist()
        dry_months = monthly_mean[monthly_mean < monthly_mean.quantile(0.25)].index.tolist()

        print(f"   🌧️ Wet Season (months): {wet_months}")
        print(f"   ☀️ Dry Season (months): {dry_months}")

        return {
            'monthly_stats': monthly_stats,
            'monthly_mean': monthly_mean,
            'daily_pattern': daily_pattern,
            'yearly_stats': yearly_stats,
            'wet_months': wet_months,
            'dry_months': dry_months
        }

    def _frequency_domain_analysis(self):
        """
        Step 2: Frequency domain analysis to detect periods (from FrequencyAnalyzer)
        """
        print(f"\n🔊 Step 2: Frequency Domain Analysis for Period Detection")
        print("-" * 50)

        # Prepare time series data
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        if len(target_ts) < 365:
            print("   ⚠️ Insufficient data for frequency analysis")
            return {'success': False, 'dominant_periods': [7, 30, 365]}

        # Remove trend for better frequency analysis
        detrended = target_ts - target_ts.rolling(window=30, center=True).mean()
        detrended = detrended.dropna()

        # Perform FFT
        fft_values = fft(detrended.values)
        frequencies = fftfreq(len(detrended), d=1)

        # Get positive frequencies
        positive_freq_idx = frequencies > 0
        positive_frequencies = frequencies[positive_freq_idx]
        positive_fft_magnitude = np.abs(fft_values[positive_freq_idx])
        periods = 1 / positive_frequencies

        # Find dominant periods
        peak_indices = np.argsort(positive_fft_magnitude)[-10:]
        dominant_periods = periods[peak_indices]

        # Filter reasonable periods (7 days to 2 years)
        reasonable_periods = dominant_periods[(dominant_periods >= 7) & (dominant_periods <= 730)]

        print(f"   📊 Detected {len(reasonable_periods)} dominant periods for MSTL:")

        for period in reasonable_periods[::-1]:
            print(f"      - {period:.1f} days")

        dominant_periods_int = sorted(list(set(np.round(reasonable_periods).astype(int))))

        return {
            'dominant_periods': dominant_periods_int,
            'all_periods': periods,
            'magnitudes': positive_fft_magnitude,
            'success': True
        }

    def _mstl_decomposition(self, periods):
        """
        Step 3: Enhanced MSTL decomposition with detected periods
        """
        print(f"\n📊 Step 3: MSTL Decomposition with Detected Periods")
        print("-" * 50)

        try:
            from statsmodels.tsa.seasonal import MSTL

            # Prepare time series
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()

            # Apply log transformation for better decomposition
            log_ts = np.log1p(target_ts)
            print(f"   🔄 Applied log1p transformation")

            # Run MSTL decomposition
            mstl = MSTL(log_ts, periods=periods)
            mstl_result = mstl.fit()

            print(f"   ✅ MSTL decomposition completed successfully")

            # Store results
            return {
                'mstl_obj': mstl_result,
                'log_ts': log_ts,
                'original_ts': target_ts,
                'periods': periods,
                'trend': mstl_result.trend,
                'seasonal': mstl_result.seasonal,
                'resid': mstl_result.resid,
                'success': True
            }

        except Exception as e:
            print(f"   ⚠️ MSTL decomposition failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _trend_cyclical_analysis(self, mstl_results):
        """
        Step 4a: NEW - Trend and cyclical component analysis
        """
        print(f"\n📈 Step 4a: Trend & Cyclical Analysis")
        print("-" * 50)

        trend = mstl_results['trend']
        original_ts = mstl_results['original_ts']

        # Trend visualization and analysis
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Trend & Cyclical Component Analysis', fontsize=16, fontweight='bold')

        # Trend component
        axes[0].plot(trend.index, trend.values, 'g-', linewidth=2, label='Trend')
        axes[0].set_title('Long-term Trend Component')
        axes[0].set_ylabel('Log Trend')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Original vs trend
        axes[1].plot(original_ts.index, original_ts.values, 'b-', alpha=0.7, label='Original')
        trend_original_scale = np.expm1(trend)  # Convert back from log scale
        axes[1].plot(trend.index, trend_original_scale.values, 'g-', linewidth=2, label='Trend (original scale)')
        axes[1].set_title('Original Data vs Extracted Trend')
        axes[1].set_ylabel('Precipitation (mm)')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Quantitative trend analysis
        trend_clean = trend.dropna()
        if len(trend_clean) > 2:
            # Calculate trend slope (simple linear regression)
            x_numeric = np.arange(len(trend_clean))
            slope, intercept = np.polyfit(x_numeric, trend_clean.values, 1)

            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            print(f"   📊 Trend Analysis:")
            print(f"      - Direction: {trend_direction}")
            print(f"      - Slope: {slope:.6f} log-units per day")
            print(f"      - Annual change: {slope * 365:.3f} log-units per year")

        # Cyclical patterns (detrended analysis)
        detrended = original_ts - trend_original_scale
        detrended_clean = detrended.dropna()

        if len(detrended_clean) > 100:
            print(f"   🔄 Cyclical patterns (after detrending):")
            print(f"      - Detrended data range: {detrended_clean.min():.2f} to {detrended_clean.max():.2f} mm")
            print(f"      - Variability: {detrended_clean.std():.2f} mm (std)")

        return {
            'trend_component': trend,
            'trend_direction': trend_direction if 'trend_direction' in locals() else 'Unknown',
            'trend_slope': slope if 'slope' in locals() else None,
            'detrended_data': detrended_clean if 'detrended_clean' in locals() else None
        }

    def _seasonal_components_analysis(self, mstl_results):
        """
        Step 4b: NEW - Individual seasonal components analysis (COMPLETELY FIXED)
        """
        print(f"\n🌀 Step 4b: Seasonal Components Analysis")
        print("-" * 50)

        mstl_obj = mstl_results['mstl_obj']
        periods = mstl_results['periods']
        # MSTL provides multiple seasonal components
        seasonal_components = {}

        try:
            # Extract individual seasonal components
            for i, period in enumerate(periods):
                component_name = f"seasonal_{period}d"
                # Access individual seasonal components from MSTL
                if hasattr(mstl_obj, 'seasonal_dict'):
                    seasonal_components[component_name] = mstl_obj.seasonal_dict[period]
                else:
                    # If not available, show combined seasonal
                    if i == 0:
                        seasonal_components['seasonal_combined'] = mstl_obj.seasonal

            # Visualization
            n_components = len(seasonal_components)
            if n_components > 0:
                fig, axes = plt.subplots(n_components, 1, figsize=(15, 4*n_components))
                if n_components == 1:
                    axes = [axes]

                fig.suptitle('Individual Seasonal Components', fontsize=16, fontweight='bold')

                for i, (comp_name, comp_data) in enumerate(seasonal_components.items()):
                    axes[i].plot(comp_data.index, comp_data.values, 'r-', linewidth=1.5)
                    axes[i].set_title(f'{comp_name.replace("_", " ").title()}')
                    axes[i].set_ylabel('Log Seasonal Effect')
                    axes[i].grid(True, alpha=0.3)

                    # COMPLETELY FIXED: Safe scalar extraction
                    try:
                        # Method 1: Use numpy/pandas operations and convert safely
                        comp_values = comp_data.values  # Get numpy array
                        comp_max = np.max(comp_values)
                        comp_min = np.min(comp_values)
                        comp_std = np.std(comp_values)

                        # Ensure scalar conversion
                        comp_range_val = float(comp_max - comp_min)
                        comp_std_val = float(comp_std)

                        axes[i].text(0.02, 0.95, f'Range: {comp_range_val:.4f}\nStd: {comp_std_val:.4f}',
                                  transform=axes[i].transAxes,
                                  bbox=dict(boxstyle="round", facecolor='wheat'),
                                  verticalalignment='top')
                    except Exception as text_error:
                        # Fallback: Simple component label without statistics
                        axes[i].text(0.02, 0.95, f'Component: {comp_name}',
                                  transform=axes[i].transAxes,
                                  bbox=dict(boxstyle="round", facecolor='lightgray'),
                                  verticalalignment='top')
                        print(f"      ⚠️ Text annotation fallback for {comp_name}")

                axes[-1].set_xlabel('Date')
                plt.tight_layout()
                plt.show()

            print(f"   📊 Seasonal Components Summary:")
            for comp_name, comp_data in seasonal_components.items():
                try:
                    # COMPLETELY FIXED: Safe scalar extraction for summary
                    comp_values = comp_data.values  # Get numpy array
                    strength_val = float(np.std(comp_values))  # Use numpy std directly
                    print(f"      - {comp_name}: strength = {strength_val:.4f}")
                except Exception as print_error:
                    # Fallback: Basic info without strength calculation
                    print(f"      - {comp_name}: component available (strength calculation skipped)")

        except Exception as e:
            print(f"   ⚠️ Error in seasonal component analysis: {e}")
            # Fallback to combined seasonal component
            try:
                seasonal_components = {'seasonal_combined': mstl_obj.seasonal}
                print(f"   📝 Using combined seasonal component as fallback")
            except:
                seasonal_components = {}

        return {
            'seasonal_components': seasonal_components,
            'periods_analyzed': periods,
            'components_count': len(seasonal_components)
        }

    def _residual_analysis(self, mstl_results):
        """
        Step 4c: NEW & IMPORTANT - Residual diagnostic analysis
        """
        print(f"\n🔍 Step 4c: Residual Diagnostic Analysis")
        print("-" * 50)

        residuals = mstl_results['resid'].dropna()

        # Residual statistics
        print(f"   📊 Residual Statistics:")
        print(f"      - Mean: {residuals.mean():.6f}")
        print(f"      - Std: {residuals.std():.6f}")
        print(f"      - Skewness: {residuals.skew():.6f}")
        print(f"      - Kurtosis: {residuals.kurtosis():.6f}")

        # Residual diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Residual Diagnostic Analysis', fontsize=16, fontweight='bold')

        # Time series plot of residuals
        axes[0,0].plot(residuals.index, residuals.values, 'purple', alpha=0.7, linewidth=1)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_title('Residuals Over Time')
        axes[0,0].set_ylabel('Residual')
        axes[0,0].grid(True, alpha=0.3)

        # Residual histogram
        axes[0,1].hist(residuals.values, bins=50, alpha=0.7, color='purple', density=True)
        axes[0,1].set_title('Residual Distribution')
        axes[0,1].set_xlabel('Residual Value')
        axes[0,1].set_ylabel('Density')
        axes[0,1].grid(True, alpha=0.3)

        # ACF of residuals
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            plot_acf(residuals, ax=axes[1,0], lags=40, alpha=0.05)
            axes[1,0].set_title('ACF of Residuals')
        except:
            axes[1,0].set_title('ACF Plot Not Available')

        # Q-Q plot
        try:
            from scipy.stats import probplot
            probplot(residuals.values, dist="norm", plot=axes[1,1])
            axes[1,1].set_title('Q-Q Plot (Normal)')
        except:
            axes[1,1].set_title('Q-Q Plot Not Available')

        plt.tight_layout()
        plt.show()

        # Ljung-Box test for autocorrelation in residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            significant_lags = ljung_box_result[ljung_box_result['lb_pvalue'] < 0.05]

            print(f"   🧪 Ljung-Box Test Results:")
            if len(significant_lags) > 0:
                print(f"      ⚠️ Significant autocorrelation detected at {len(significant_lags)} lags")
                print(f"      - First significant lag: {significant_lags.index[0]}")
            else:
                print(f"      ✅ No significant autocorrelation in residuals")

        except Exception as e:
            print(f"   ⚠️ Ljung-Box test failed: {e}")
            significant_lags = None

        return {
            'residuals': residuals,
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis()
            },
            'ljung_box_test': significant_lags,
            'autocorr_clean': len(significant_lags) == 0 if significant_lags is not None else None
        }

    def _wavelet_analysis(self, mstl_results=None):
        """
        Step 5: Enhanced Wavelet Analysis using PyWavelets
        """
        print(f"\n🌊 Step 5: Wavelet Analysis (PyWavelets)")
        print("-" * 50)

        try:
            import pywt

            # Prepare time series data
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()

            # Trong hàm _wavelet_analysis

            analysis_duration_years = 3
            analysis_duration_days = 365 * analysis_duration_years

            if len(target_ts) > analysis_duration_days:
                target_ts = target_ts.tail(analysis_duration_days)
                print(f"   📊 Using last {analysis_duration_years} years ({analysis_duration_days} days) for analysis")

            data = target_ts.values
            dates = target_ts.index

            print(f"   🔄 Running Continuous Wavelet Transform with PyWavelets...")

            # Define scales for periods 3-120 days
            scales = np.arange(3, 121, 2)  # Simplified scale range

            # Perform CWT using Morlet wavelet
            coefficients, frequencies = pywt.cwt(data, scales, 'morl')

            # Convert frequencies to periods
            periods = 1 / frequencies

            # Calculate power
            power = np.abs(coefficients) ** 2

            print(f"   ✅ PyWavelets analysis completed")

            # Visualization
            self._plot_pywt_results(data, dates, periods, power)

            # Analysis insights
            insights = self._extract_pywt_insights(periods, power, dates)

            return {
                'implemented': True,
                'library': 'PyWavelets',
                'coefficients': coefficients,
                'power': power,
                'periods': periods,
                'scales': scales,
                'insights': insights,
                'wavelet_type': 'Morlet'
            }

        except ImportError:
            print(f"   ⚠️ PyWavelets not available, falling back to scipy.signal")
            return self._wavelet_analysis_fallback()
        except Exception as e:
            print(f"   ⚠️ PyWavelets analysis failed: {e}")
            return {'implemented': False, 'error': str(e)}

    def _plot_pywt_results(self, data, dates, periods, power):
        """Plot PyWavelets results (simplified)"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('PyWavelets Analysis', fontsize=16, fontweight='bold')

        # 1. Original time series
        axes[0].plot(dates, data, 'b-', linewidth=1.5)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Precipitation (mm)')
        axes[0].grid(True, alpha=0.3)

        # 2. Wavelet power spectrum
        dates_num = np.arange(len(dates))
        T, P = np.meshgrid(dates_num, periods)

        log_power = np.log10(power + 1e-12)
        im = axes[1].contourf(T, P, log_power, levels=30, cmap='jet')
        axes[1].set_title('Wavelet Power Spectrum')
        axes[1].set_ylabel('Period (days)')
        axes[1].set_xlabel('Time Index')

        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        plt.show()

    def _extract_pywt_insights(self, periods, power, dates):
        """Extract insights from PyWavelets analysis"""
        # Global power spectrum
        global_power = np.mean(power, axis=1)

        # Find dominant periods
        peak_indices = np.argsort(global_power)[-5:]  # Top 5
        dominant_periods = []

        for idx in peak_indices:
            if idx < len(periods):
                period = periods[idx]
                power_val = global_power[idx]
                dominant_periods.append({
                    'period': period,
                    'power': power_val
                })

        print(f"   📊 PyWavelets Insights:")
        print(f"      - Dominant periods: {len(dominant_periods)}")
        for i, p in enumerate(dominant_periods[-3:], 1):  # Top 3
            print(f"        {i}. {p['period']:.1f} days")

        return {'dominant_periods': dominant_periods}

    def _wavelet_analysis_fallback(self):
        """Fallback to scipy.signal if pywt not available"""
        print(f"   🔄 Using scipy.signal fallback...")

        try:
            from scipy.signal import cwt, morlet2

            # Simplified fallback implementation
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna().tail(365)  # Last year only

            data = target_ts.values
            scales = np.logspace(1, 2, 20)  # Simplified scales

            coefficients = cwt(data, morlet2, scales)
            power = np.abs(coefficients) ** 2

            print(f"   ✅ Scipy fallback completed")

            return {
                'implemented': True,
                'library': 'scipy.signal (fallback)',
                'power': power,
                'scales': scales,
                'simplified': True
            }

        except Exception as e:
            print(f"   ❌ Fallback also failed: {e}")
            return {'implemented': False, 'error': 'Both pywt and scipy failed'}