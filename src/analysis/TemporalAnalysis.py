# =============================================================================
# COMPLETE REFACTOR: 4 ANALYSIS-VISUALIZATION PAIRS
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.stats import probplot
import warnings

from ..config.constants import Config

warnings.filterwarnings('ignore')

class TemporalStructureAnalyzer:
    """
    ✅ REFACTORED: 4 Pure Analysis-Visualization Pairs
    1. Seasonal Patterns (groupby-based)
    2. FFT Frequency Analysis 
    3. MSTL Decomposition
    4. Component Diagnostics
    """

    def __init__(self, df: pd.DataFrame, target_col: str = None, date_col: str = None):
        """Initialize with configurable parameters"""
        self.df = df.copy()
        self.target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
        self.date_col = date_col or 'Ngày'
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for temporal analysis"""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')

        # Add time features
        self.df['Month'] = self.df[self.date_col].dt.month
        self.df['Year'] = self.df[self.date_col].dt.year
        self.df['DayOfYear'] = self.df[self.date_col].dt.dayofyear
        
        # Sort by date
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    # =============================================================================
    # PAIR 1: SEASONAL PATTERNS - PURE SEPARATION ✅
    # =============================================================================

    def analyze_seasonal_patterns(self,
                                 time_units: List[str] = ['Month', 'DayOfYear', 'Year'],
                                 agg_functions: List[str] = ['mean', 'std', 'median', 'count'],
                                 quantile_threshold: float = 0.75) -> Dict[str, Any]:
        """
        ✅ PURE ANALYSIS: Seasonal pattern analysis (NO VISUALIZATION)
        
        Args:
            time_units: Time aggregation units to analyze
            agg_functions: Statistical functions to apply
            quantile_threshold: Threshold for wet/dry season identification
            
        Returns:
            Dict containing all seasonal analysis results
        """
        print(f"🔍 Analyzing Seasonal Patterns")
        print(f"   - Time units: {time_units}")
        print(f"   - Aggregations: {agg_functions}")
        print(f"   - Quantile threshold: {quantile_threshold}")

        results = {}

        # Monthly analysis
        if 'Month' in time_units:
            monthly_stats = self.df.groupby('Month')[self.target_col].agg(agg_functions).round(4)
            monthly_mean = self.df.groupby('Month')[self.target_col].mean()
            
            # Season identification with configurable threshold
            wet_months = monthly_mean[monthly_mean > monthly_mean.quantile(quantile_threshold)].index.tolist()
            dry_months = monthly_mean[monthly_mean < monthly_mean.quantile(1 - quantile_threshold)].index.tolist()
            
            results['monthly'] = {
                'stats': monthly_stats,
                'mean': monthly_mean,
                'wet_months': wet_months,
                'dry_months': dry_months,
                'threshold_used': quantile_threshold
            }

        # Daily pattern within year
        if 'DayOfYear' in time_units:
            daily_pattern = self.df.groupby('DayOfYear')[self.target_col].agg(agg_functions)
            results['daily_pattern'] = daily_pattern

        # Yearly analysis
        if 'Year' in time_units:
            yearly_stats = self.df.groupby('Year')[self.target_col].agg(agg_functions)
            results['yearly'] = yearly_stats

        print(f"   ✅ Seasonal analysis completed")
        return results

    def plot_seasonal_patterns(self,
                              seasonal_results: Dict[str, Any],
                              figsize: Tuple[int, int] = (16, 10),
                              color_scheme: str = 'default',
                              show_grid: bool = True) -> None:
        """
        ✅ PURE VISUALIZATION: Plot seasonal patterns (NO ANALYSIS)
        
        Args:
            seasonal_results: Results from analyze_seasonal_patterns()
            figsize: Figure size
            color_scheme: Color scheme to use
            show_grid: Whether to show grid
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Seasonal Patterns Analysis', fontsize=16, fontweight='bold')

        # Monthly boxplot
        if 'monthly' in seasonal_results:
            sns.boxplot(data=self.df, x='Month', y=self.target_col, ax=axes[0,0])
            axes[0,0].set_title('Monthly Distribution')
            axes[0,0].set_xlabel('Month')
            axes[0,0].set_ylabel('Precipitation (mm)')
            if show_grid:
                axes[0,0].grid(True, alpha=0.3)

        # Monthly average
        if 'monthly' in seasonal_results:
            monthly_mean = seasonal_results['monthly']['mean']
            axes[0,1].bar(monthly_mean.index, monthly_mean.values, color='skyblue', alpha=0.7)
            axes[0,1].set_title('Average Monthly Precipitation')
            axes[0,1].set_xlabel('Month')
            axes[0,1].set_ylabel('Average Precipitation (mm)')
            if show_grid:
                axes[0,1].grid(True, alpha=0.3)

        # Daily pattern within year
        if 'daily_pattern' in seasonal_results:
            daily_pattern = seasonal_results['daily_pattern']['mean']
            axes[1,0].plot(daily_pattern.index, daily_pattern.values, 'b-', linewidth=1.5, alpha=0.8)
            axes[1,0].set_title('Daily Pattern Throughout Year')
            axes[1,0].set_xlabel('Day of Year')
            axes[1,0].set_ylabel('Average Precipitation (mm)')
            if show_grid:
                axes[1,0].grid(True, alpha=0.3)

        # Yearly trend
        if 'yearly' in seasonal_results:
            yearly_mean = seasonal_results['yearly']['mean']
            axes[1,1].plot(yearly_mean.index, yearly_mean.values, 'ro-', linewidth=2, markersize=6)
            axes[1,1].set_title('Yearly Trend')
            axes[1,1].set_xlabel('Year')
            axes[1,1].set_ylabel('Average Precipitation (mm)')
            if show_grid:
                axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print insights
        if 'monthly' in seasonal_results:
            wet_months = seasonal_results['monthly']['wet_months']
            dry_months = seasonal_results['monthly']['dry_months']
            threshold = seasonal_results['monthly']['threshold_used']
            print(f"🌧️ Wet Season (>{threshold*100:.0f}th percentile): {wet_months}")
            print(f"☀️ Dry Season (<{(1-threshold)*100:.0f}th percentile): {dry_months}")

    # =============================================================================
    # PAIR 2: FFT FREQUENCY ANALYSIS - PURE SEPARATION ✅
    # =============================================================================

    def analyze_fft(self,
                   detrend_window: int = 30,
                   top_n: int = 10,
                   period_range: Tuple[int, int] = (7, 730)) -> Dict[str, Any]:
        """
        ✅ PURE ANALYSIS: FFT-based frequency analysis (NO VISUALIZATION)
        
        Args:
            detrend_window: Window size for trend removal
            top_n: Number of top periods to find
            period_range: Valid period range (min_days, max_days)
            
        Returns:
            Dict containing FFT analysis results
        """
        print(f"🔊 FFT Frequency Analysis")
        print(f"   - Detrend window: {detrend_window}")
        print(f"   - Top periods: {top_n}")
        print(f"   - Period range: {period_range}")

        # Prepare time series
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        if len(target_ts) < period_range[1]:
            print(f"   ⚠️ Insufficient data for frequency analysis")
            return {'success': False, 'error': 'Insufficient data'}

        # Detrend with configurable window
        detrended = target_ts - target_ts.rolling(window=detrend_window, center=True).mean()
        detrended = detrended.dropna()

        # Perform FFT
        fft_values = fft(detrended.values)
        frequencies = fftfreq(len(detrended), d=1)

        # Get positive frequencies  
        positive_freq_idx = frequencies > 0
        positive_frequencies = frequencies[positive_freq_idx]
        positive_fft_magnitude = np.abs(fft_values[positive_freq_idx])
        periods = 1 / positive_frequencies

        # Find dominant periods with configurable range
        peak_indices = np.argsort(positive_fft_magnitude)[-top_n:]
        dominant_periods = periods[peak_indices]

        # Filter by period range
        valid_periods = dominant_periods[
            (dominant_periods >= period_range[0]) & 
            (dominant_periods <= period_range[1])
        ]

        # Sort and convert to integers
        dominant_periods_int = sorted(list(set(np.round(valid_periods).astype(int))))

        print(f"   📊 Found {len(dominant_periods_int)} dominant periods:")
        for period in dominant_periods_int[:5]:  # Show top 5
            print(f"      - {period} days")

        return {
            'success': True,
            'dominant_periods': dominant_periods_int,
            'all_periods': periods,
            'magnitudes': positive_fft_magnitude,
            'frequencies': positive_frequencies,
            'detrended_series': detrended,
            'parameters': {
                'detrend_window': detrend_window,
                'top_n': top_n,
                'period_range': period_range
            }
        }

    def plot_spectrum(self,
                     fft_results: Dict[str, Any],
                     figsize: Tuple[int, int] = (15, 8),
                     show_dominant: bool = True,
                     max_period: int = 365,
                     log_scale: bool = True) -> None:
        """
        ✅ PURE VISUALIZATION: Plot frequency spectrum (NO ANALYSIS)
        
        Args:
            fft_results: Results from analyze_fft()
            figsize: Figure size
            show_dominant: Whether to highlight dominant periods
            max_period: Maximum period to show
            log_scale: Whether to use log scale for power
        """
        if not fft_results['success']:
            print("❌ Cannot plot spectrum: FFT analysis failed")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')

        periods = fft_results['all_periods']
        magnitudes = fft_results['magnitudes']
        dominant_periods = fft_results['dominant_periods']

        # Filter by max_period
        valid_idx = periods <= max_period
        periods_plot = periods[valid_idx]
        magnitudes_plot = magnitudes[valid_idx]

        # Power Spectral Density
        power = magnitudes_plot ** 2
        if log_scale:
            power = np.log10(power + 1e-12)
            ylabel = 'Log Power Spectral Density'
        else:
            ylabel = 'Power Spectral Density'

        axes[0].plot(periods_plot, power, 'b-', alpha=0.7, linewidth=1)
        axes[0].set_xlabel('Period (days)')
        axes[0].set_ylabel(ylabel)
        axes[0].set_title('Power Spectral Density')
        axes[0].grid(True, alpha=0.3)

        # Highlight dominant periods
        if show_dominant:
            for period in dominant_periods[:5]:  # Top 5
                if period <= max_period:
                    period_idx = np.argmin(np.abs(periods_plot - period))
                    axes[0].axvline(x=period, color='red', linestyle='--', alpha=0.7)
                    axes[0].text(period, power[period_idx], f'{period}d',
                               rotation=90, verticalalignment='bottom')

        # Dominant periods bar chart
        top_5_periods = dominant_periods[:5]
        top_5_power = []
        for period in top_5_periods:
            if period <= max_period:
                period_idx = np.argmin(np.abs(periods_plot - period))
                top_5_power.append(power[period_idx])
            else:
                top_5_power.append(0)

        axes[1].bar(range(len(top_5_periods)), top_5_power, alpha=0.7)
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel(ylabel)
        axes[1].set_title('Top Dominant Periods')
        axes[1].set_xticks(range(len(top_5_periods)))
        axes[1].set_xticklabels([f'{p}d' for p in top_5_periods])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # =============================================================================
    # PAIR 3: MSTL DECOMPOSITION - PURE SEPARATION ✅
    # =============================================================================

    def decompose_mstl(self,
                      periods: List[int],
                      transform: str = 'log1p',
                      **stl_kwargs) -> Dict[str, Any]:
        """
        ✅ PURE ANALYSIS: MSTL decomposition (NO VISUALIZATION)
        
        Args:
            periods: List of seasonal periods for decomposition
            transform: Transformation to apply ('log1p', 'log', 'none')
            **stl_kwargs: Additional parameters for MSTL
            
        Returns:
            Dict containing MSTL decomposition results
        """
        print(f"📊 MSTL Decomposition")
        print(f"   - Periods: {periods}")
        print(f"   - Transform: {transform}")

        try:
            from statsmodels.tsa.seasonal import MSTL

            # Prepare time series
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()

            # Apply transformation
            if transform == 'log1p':
                transformed_ts = np.log1p(target_ts)
            elif transform == 'log':
                transformed_ts = np.log(target_ts + 1e-8)
            else:
                transformed_ts = target_ts

            print(f"   🔄 Applied {transform} transformation")

            # Run MSTL decomposition - FIXED parameters
            stl_kwargs_clean = {k: v for k, v in stl_kwargs.items() 
                               if k in ['seasonal_deg', 'trend_deg', 'low_pass_deg', 
                                       'seasonal_jump', 'trend_jump', 'low_pass_jump']}
            
            mstl = MSTL(transformed_ts, periods=periods, **stl_kwargs_clean)
            mstl_result = mstl.fit()

            print(f"   ✅ MSTL decomposition completed")

            return {
                'success': True,
                'mstl_obj': mstl_result,
                'original_ts': target_ts,
                'transformed_ts': transformed_ts,
                'periods': periods,
                'trend': mstl_result.trend,
                'seasonal': mstl_result.seasonal,
                'resid': mstl_result.resid,
                'transform': transform,
                'parameters': stl_kwargs_clean
            }

        except Exception as e:
            print(f"   ⚠️ MSTL decomposition failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'periods': periods
            }

    def plot_decomposition(self,
                          mstl_results: Dict[str, Any],
                          figsize: Tuple[int, int] = (15, 12),
                          show_components: List[str] = ['trend', 'seasonal', 'resid'],
                          original_scale: bool = True) -> None:
        """
        ✅ PURE VISUALIZATION: Plot MSTL decomposition (NO ANALYSIS)
        
        Args:
            mstl_results: Results from decompose_mstl()
            figsize: Figure size
            show_components: Components to show
            original_scale: Whether to convert back to original scale
        """
        if not mstl_results['success']:
            print("❌ Cannot plot decomposition: MSTL failed")
            return

        transform = mstl_results['transform']
        n_plots = 1 + len(show_components)  # Original + components

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        fig.suptitle('MSTL Decomposition Results', fontsize=16, fontweight='bold')

        plot_idx = 0

        # Original series
        if original_scale and transform in ['log1p', 'log']:
            if transform == 'log1p':
                original_data = np.expm1(mstl_results['transformed_ts'])
            else:
                original_data = np.exp(mstl_results['transformed_ts'])
            axes[plot_idx].plot(original_data.index, original_data.values, 'b-', linewidth=1)
            axes[plot_idx].set_title('Original Data (Original Scale)')
            axes[plot_idx].set_ylabel('Precipitation (mm)')
        else:
            axes[plot_idx].plot(mstl_results['transformed_ts'].index,
                               mstl_results['transformed_ts'].values, 'b-', linewidth=1)
            axes[plot_idx].set_title(f'Original Data ({transform} scale)')
            axes[plot_idx].set_ylabel(f'{transform} Precipitation')

        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Trend component
        if 'trend' in show_components:
            trend = mstl_results['trend']
            axes[plot_idx].plot(trend.index, trend.values, 'g-', linewidth=2)
            axes[plot_idx].set_title('Trend Component')
            axes[plot_idx].set_ylabel(f'{transform} Trend')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Seasonal component
        if 'seasonal' in show_components:
            seasonal = mstl_results['seasonal']
            axes[plot_idx].plot(seasonal.index, seasonal.values, 'r-', linewidth=1)
            axes[plot_idx].set_title('Combined Seasonal Component')
            axes[plot_idx].set_ylabel(f'{transform} Seasonal')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Residual component
        if 'resid' in show_components:
            resid = mstl_results['resid']
            axes[plot_idx].plot(resid.index, resid.values, 'purple', linewidth=1, alpha=0.7)
            axes[plot_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[plot_idx].set_title('Residual Component')
            axes[plot_idx].set_ylabel(f'{transform} Residual')
            axes[plot_idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()

    # =============================================================================
    # PAIR 4: COMPONENT DIAGNOSTICS - PURE SEPARATION ✅
    # =============================================================================

    def diagnose_residuals(self,
                          residual_series: pd.Series,
                          ljung_box_lags: int = 10,
                          acf_lags: int = 40,
                          normality_tests: bool = True) -> Dict[str, Any]:
        """
        ✅ PURE ANALYSIS: Residual diagnostic analysis (NO VISUALIZATION)
        
        Args:
            residual_series: Residual component from MSTL
            ljung_box_lags: Number of lags for Ljung-Box test
            acf_lags: Number of lags for ACF analysis
            normality_tests: Whether to perform normality tests
            
        Returns:
            Dict containing residual diagnostic results
        """
        print(f"🔍 Residual Diagnostics")
        print(f"   - Ljung-Box lags: {ljung_box_lags}")
        print(f"   - ACF lags: {acf_lags}")

        residuals = residual_series.dropna()

        if len(residuals) < 50:
            return {
                'success': False,
                'error': f'Insufficient residual data ({len(residuals)} < 50)'
            }

        # Basic residual statistics
        residual_stats = {
            'count': len(residuals),
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'min': residuals.min(),
            'max': residuals.max()
        }

        # Ljung-Box test for autocorrelation
        ljung_box_results = None
        autocorr_clean = None

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box_result = acorr_ljungbox(residuals, lags=ljung_box_lags, return_df=True)
            significant_lags = ljung_box_result[ljung_box_result['lb_pvalue'] < 0.05]

            ljung_box_results = {
                'test_results': ljung_box_result,
                'significant_lags': len(significant_lags),
                'first_significant_lag': significant_lags.index[0] if len(significant_lags) > 0 else None
            }

            autocorr_clean = len(significant_lags) == 0

        except Exception as e:
            print(f"   ⚠️ Ljung-Box test failed: {e}")
            ljung_box_results = {'error': str(e)}

        # Normality tests
        normality_results = {}
        if normality_tests:
            try:
                from scipy.stats import shapiro, jarque_bera

                # Shapiro-Wilk test (for smaller samples)
                if len(residuals) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(residuals.values)
                    normality_results['shapiro'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }

                # Jarque-Bera test
                jb_stat, jb_p = jarque_bera(residuals.values)
                normality_results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_p,
                    'is_normal': jb_p > 0.05
                }

            except Exception as e:
                normality_results['error'] = str(e)

        print(f"   - Mean: {residual_stats['mean']:.6f}")
        print(f"   - Std: {residual_stats['std']:.6f}")
        if ljung_box_results and 'significant_lags' in ljung_box_results:
            if autocorr_clean:
                print(f"   ✅ No significant autocorrelation")
            else:
                print(f"   ⚠️ Autocorrelation detected at {ljung_box_results['significant_lags']} lags")

        return {
            'success': True,
            'residuals': residuals,
            'residual_stats': residual_stats,
            'ljung_box_results': ljung_box_results,
            'normality_results': normality_results,
            'autocorr_clean': autocorr_clean,
            'parameters': {
                'ljung_box_lags': ljung_box_lags,
                'acf_lags': acf_lags
            }
        }

    def plot_residual_diagnostics(self,
                                 residual_results: Dict[str, Any],
                                 figsize: Tuple[int, int] = (15, 10),
                                 hist_bins: int = 50,
                                 layout: str = '2x2') -> None:
        """
        ✅ PURE VISUALIZATION: Plot residual diagnostics (NO ANALYSIS)
        
        Args:
            residual_results: Results from diagnose_residuals()
            figsize: Figure size
            hist_bins: Number of bins for histogram
            layout: Plot layout ('2x2', '3x2', '1x4')
        """
        if not residual_results['success']:
            print("❌ Cannot plot diagnostics: Residual analysis failed")
            return

        residuals = residual_results['residuals']
        acf_lags = residual_results['parameters']['acf_lags']

        # Determine subplot layout
        if layout == '2x2':
            nrows, ncols = 2, 2
        elif layout == '3x2':
            nrows, ncols = 3, 2
        elif layout == '1x4':
            nrows, ncols = 1, 4
        else:
            nrows, ncols = 2, 2

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle('Residual Diagnostic Analysis', fontsize=16, fontweight='bold')

        plot_idx = 0

        # Time series plot
        axes[plot_idx].plot(residuals.index, residuals.values, 'purple', alpha=0.7, linewidth=1)
        axes[plot_idx].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[plot_idx].set_title('Residuals Over Time')
        axes[plot_idx].set_ylabel('Residual')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Histogram
        axes[plot_idx].hist(residuals.values, bins=hist_bins, alpha=0.7,
                           color='purple', density=True, edgecolor='black')
        axes[plot_idx].set_title('Residual Distribution')
        axes[plot_idx].set_xlabel('Residual Value')
        axes[plot_idx].set_ylabel('Density')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # ACF plot
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, ax=axes[plot_idx], lags=acf_lags, alpha=0.05)
            axes[plot_idx].set_title(f'ACF of Residuals (lags={acf_lags})')
        except Exception as e:
            axes[plot_idx].set_title('ACF Plot Not Available')
            axes[plot_idx].text(0.5, 0.5, f'Error: {str(e)}',
                               transform=axes[plot_idx].transAxes, ha='center')
        plot_idx += 1

        # Q-Q plot
        try:
            probplot(residuals.values, dist="norm", plot=axes[plot_idx])
            axes[plot_idx].set_title('Q-Q Plot (Normal)')
        except Exception as e:
            axes[plot_idx].set_title('Q-Q Plot Not Available')
            axes[plot_idx].text(0.5, 0.5, f'Error: {str(e)}',
                               transform=axes[plot_idx].transAxes, ha='center')
        plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print diagnostic summary
        stats = residual_results['residual_stats']
        print(f"\n📊 Residual Summary:")
        print(f"   - Count: {stats['count']:,}")
        print(f"   - Mean: {stats['mean']:.6f}")
        print(f"   - Std: {stats['std']:.6f}")
        print(f"   - Skewness: {stats['skewness']:.4f}")
        print(f"   - Kurtosis: {stats['kurtosis']:.4f}")

    # =============================================================================
    # PAIR 5: WAVELET ANALYSIS - PURE SEPARATION ✅
    # =============================================================================

    def analyze_wavelet(self,
                       wavelet_name: str = 'morl',
                       scales: Optional[np.ndarray] = None,
                       analysis_duration_years: int = 3,
                       period_range: Tuple[int, int] = (3, 365)) -> Dict[str, Any]:
        """
        ✅ PURE ANALYSIS: Wavelet-based time-frequency analysis (NO VISUALIZATION)
        
        Args:
            wavelet_name: Wavelet to use ('morl', 'cmor', 'gaus')
            scales: Custom scales array (if None, auto-generated)
            analysis_duration_years: Years of recent data to analyze
            period_range: Period range for scale generation (min_days, max_days)
            
        Returns:
            Dict containing wavelet analysis results
        """
        print(f"🌊 Wavelet Analysis")
        print(f"   - Wavelet: {wavelet_name}")
        print(f"   - Analysis duration: {analysis_duration_years} years")
        print(f"   - Period range: {period_range}")

        try:
            import pywt
            
            # Prepare time series data
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()

            # Limit analysis to recent years for performance
            analysis_duration_days = 365 * analysis_duration_years
            if len(target_ts) > analysis_duration_days:
                target_ts = target_ts.tail(analysis_duration_days)
                print(f"   📊 Using last {analysis_duration_years} years ({len(target_ts)} days)")
            
            data = target_ts.values
            dates = target_ts.index

            # Generate scales if not provided
            if scales is None:
                # Convert period range to scale range
                min_scale = max(1, period_range[0] // 2)
                max_scale = min(200, period_range[1] // 2)
                scales = np.arange(min_scale, max_scale, 2)
                print(f"   🎚️ Generated {len(scales)} scales: {min_scale} to {max_scale}")

            # Perform Continuous Wavelet Transform
            print(f"   🔄 Running CWT with {wavelet_name} wavelet...")
            coefficients, frequencies = pywt.cwt(data, scales, wavelet_name)

            # Convert frequencies to periods
            if wavelet_name == 'morl':
                # For Morlet wavelet, center frequency is approximately 1
                periods = scales / 1.0
            else:
                # General case
                periods = 1 / frequencies

            # Calculate power (magnitude squared)
            power = np.abs(coefficients) ** 2

            # Global wavelet spectrum (time-averaged)
            global_power = np.mean(power, axis=1)

            # Find dominant periods from wavelet analysis
            peak_indices = np.argsort(global_power)[-10:]  # Top 10
            dominant_periods_wav = periods[peak_indices]
            
            # Filter by period range
            valid_periods = dominant_periods_wav[
                (dominant_periods_wav >= period_range[0]) & 
                (dominant_periods_wav <= period_range[1])
            ]

            print(f"   📊 Found {len(valid_periods)} dominant wavelet periods:")
            for period in sorted(valid_periods, reverse=True)[:5]:  # Show top 5
                print(f"      - {period:.1f} days")

            return {
                'success': True,
                'coefficients': coefficients,
                'power': power,
                'periods': periods,
                'scales': scales,
                'frequencies': frequencies,
                'global_power': global_power,
                'dominant_periods': sorted(valid_periods, reverse=True),
                'dates': dates,
                'data': data,
                'parameters': {
                    'wavelet_name': wavelet_name,
                    'analysis_duration_years': analysis_duration_years,
                    'period_range': period_range,
                    'n_scales': len(scales)
                }
            }

        except ImportError:
            print(f"   ⚠️ PyWavelets not available, trying scipy fallback...")
            return self._analyze_wavelet_scipy_fallback(analysis_duration_years, period_range)
        except Exception as e:
            print(f"   ❌ Wavelet analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'parameters': {
                    'wavelet_name': wavelet_name,
                    'analysis_duration_years': analysis_duration_years,
                    'period_range': period_range
                }
            }

    def _analyze_wavelet_scipy_fallback(self, 
                                       analysis_duration_years: int,
                                       period_range: Tuple[int, int]) -> Dict[str, Any]:
        """
        Fallback wavelet analysis using scipy.signal
        """
        try:
            from scipy.signal import cwt, morlet2
            
            print(f"   🔄 Using scipy.signal CWT fallback...")
            
            # Prepare simplified time series
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()
            
            # Use last year only for scipy fallback
            target_ts = target_ts.tail(365)
            data = target_ts.values
            dates = target_ts.index
            
            # Simplified scales
            scales = np.logspace(1, 2, 20)  # 20 scales from 10 to 100
            
            # Perform CWT
            coefficients = cwt(data, morlet2, scales)
            power = np.abs(coefficients) ** 2
            
            # Approximate periods
            periods = scales * 2  # Rough approximation
            
            print(f"   ✅ Scipy fallback completed")
            
            return {
                'success': True,
                'coefficients': coefficients,
                'power': power,
                'periods': periods,
                'scales': scales,
                'global_power': np.mean(power, axis=1),
                'dominant_periods': [],
                'dates': dates,
                'data': data,
                'parameters': {
                    'wavelet_name': 'morlet2_scipy',
                    'analysis_duration_years': 1,  # Limited to 1 year
                    'period_range': period_range,
                    'fallback': True
                }
            }
            
        except Exception as e:
            print(f"   ❌ Scipy fallback also failed: {e}")
            return {
                'success': False,
                'error': f'Both PyWavelets and scipy failed: {str(e)}',
                'parameters': {'fallback_attempted': True}
            }

    def plot_scalogram(self,
                      wavelet_results: Dict[str, Any],
                      figsize: Tuple[int, int] = (15, 10),
                      cmap: str = 'jet',
                      show_coi: bool = True,
                      log_scale: bool = True) -> None:
        """
        ✅ PURE VISUALIZATION: Plot wavelet scalogram (NO ANALYSIS)
        
        Args:
            wavelet_results: Results from analyze_wavelet()
            figsize: Figure size
            cmap: Colormap for scalogram
            show_coi: Whether to show cone of influence
            log_scale: Whether to use log scale for power
        """
        if not wavelet_results['success']:
            print("❌ Cannot plot scalogram: Wavelet analysis failed")
            return

        power = wavelet_results['power']
        periods = wavelet_results['periods']
        dates = wavelet_results['dates']
        data = wavelet_results['data']
        params = wavelet_results['parameters']

        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle(f'Wavelet Analysis - {params["wavelet_name"]} Wavelet', 
                    fontsize=16, fontweight='bold')

        # 1. Original time series
        axes[0].plot(dates, data, 'b-', linewidth=1.5, alpha=0.8)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Precipitation (mm)')
        axes[0].grid(True, alpha=0.3)

        # 2. Wavelet power spectrum (scalogram)
        dates_num = np.arange(len(dates))
        T, P = np.meshgrid(dates_num, periods)

        if log_scale:
            power_plot = np.log10(power + 1e-12)
            power_label = 'Log₁₀ Power'
        else:
            power_plot = power
            power_label = 'Power'

        im = axes[1].contourf(T, P, power_plot, levels=30, cmap=cmap)
        axes[1].set_ylabel('Period (days)')
        axes[1].set_title('Wavelet Power Spectrum (Scalogram)')
        
        # Set y-axis to log scale for better period visualization
        axes[1].set_yscale('log')
        axes[1].set_ylim(periods.min(), periods.max())

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label(power_label)

        # 3. Global wavelet spectrum
        if 'global_power' in wavelet_results:
            global_power = wavelet_results['global_power']
            
            if log_scale:
                global_power_plot = np.log10(global_power + 1e-12)
            else:
                global_power_plot = global_power

            axes[2].plot(global_power_plot, periods, 'r-', linewidth=2)
            axes[2].set_xlabel(power_label)
            axes[2].set_ylabel('Period (days)')
            axes[2].set_title('Global Wavelet Spectrum')
            axes[2].set_yscale('log')
            axes[2].set_ylim(periods.min(), periods.max())
            axes[2].grid(True, alpha=0.3)

            # Highlight dominant periods
            if 'dominant_periods' in wavelet_results:
                dominant_periods = wavelet_results['dominant_periods']
                for period in dominant_periods[:5]:  # Top 5
                    if period >= periods.min() and period <= periods.max():
                        axes[2].axhline(y=period, color='blue', linestyle='--', alpha=0.7)
                        axes[2].text(global_power_plot.max() * 0.7, period, 
                                   f'{period:.1f}d', verticalalignment='bottom')

        # Format x-axis with dates for scalogram
        n_ticks = min(6, len(dates))
        tick_indices = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
        axes[1].set_xticks(tick_indices)
        axes[1].set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_indices], 
                               rotation=45)
        axes[1].set_xlabel('Date')

        # Same for original time series
        axes[0].set_xticks(tick_indices)
        axes[0].set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_indices], 
                               rotation=45)

        plt.tight_layout()
        plt.show()

        # Print analysis summary
        print(f"\n🌊 Wavelet Analysis Summary:")
        print(f"   - Wavelet: {params['wavelet_name']}")
        print(f"   - Data points: {len(data):,}")
        print(f"   - Period range: {periods.min():.1f} - {periods.max():.1f} days")
        print(f"   - Number of scales: {len(periods)}")
        
        if 'dominant_periods' in wavelet_results and wavelet_results['dominant_periods']:
            print(f"   - Top dominant periods:")
            for i, period in enumerate(wavelet_results['dominant_periods'][:3], 1):
                print(f"     {i}. {period:.1f} days")

    # =============================================================================
    # MAIN ORCHESTRATION - CONVENIENCE METHODS (UPDATED)
    # =============================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """
        Convenience method: Run all analyses with default parameters
        
        Returns:
            Dict containing all analysis results
        """
        print("\n" + "="*70)
        print("🕐 TEMPORAL STRUCTURE ANALYSIS - 5 PAIRS REFACTORED")
        print("="*70)

        # Step 1: Seasonal patterns analysis
        seasonal_results = self.analyze_seasonal_patterns()

        # Step 2: FFT frequency analysis
        fft_results = self.analyze_fft()

        # Step 3: MSTL decomposition
        periods = fft_results.get('dominant_periods', [7, 30, 365])[:3]  # Use top 3
        mstl_results = self.decompose_mstl(periods)

        # Step 4: Component diagnostics
        residual_results = {}
        if mstl_results['success']:
            residual_results = self.diagnose_residuals(mstl_results['resid'])

        # Step 5: Wavelet analysis (NEW)
        wavelet_results = self.analyze_wavelet()

        # Combine all results
        results = {
            'seasonal_patterns': seasonal_results,
            'frequency_analysis': fft_results,
            'mstl_decomposition': mstl_results,
            'residual_analysis': residual_results,
            'wavelet_analysis': wavelet_results,
            'component_name': 'TemporalStructureAnalyzer_5Pairs_Complete'
        }

        print(f"\n✅ TEMPORAL ANALYSIS COMPLETED - 5 PAIRS IMPLEMENTED")
        return results

    def visualize_all(self, results: Dict[str, Any]) -> None:
        """
        Convenience method: Visualize all components
        
        Args:
            results: Results from analyze_all()
        """
        print("\n🎨 Visualizing All Components...")

        # 1. Seasonal patterns
        self.plot_seasonal_patterns(results['seasonal_patterns'])

        # 2. Frequency spectrum  
        if results['frequency_analysis']['success']:
            self.plot_spectrum(results['frequency_analysis'])

        # 3. MSTL decomposition
        if results['mstl_decomposition']['success']:
            self.plot_decomposition(results['mstl_decomposition'])

        # 4. Residual diagnostics
        if results['residual_analysis'].get('success'):
            self.plot_residual_diagnostics(results['residual_analysis'])

        # 5. Wavelet scalogram (NEW)
        if results['wavelet_analysis']['success']:
            self.plot_scalogram(results['wavelet_analysis'])

# ✅ EXPORT CONVENIENCE FUNCTION
def analyze_temporal_structure(df: pd.DataFrame, 
                              target_col: str = None, 
                              date_col: str = None) -> Dict[str, Any]:
    """
    Convenience function for complete temporal analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        
    Returns:
        Dict containing all analysis results
    """
    analyzer = TemporalStructureAnalyzer(df, target_col, date_col)
    return analyzer.analyze_all()