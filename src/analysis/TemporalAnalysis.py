# =============================================================================
# COMPLETE REFACTOR: 4 ANALYSIS-VISUALIZATION PAIRS  
# ✅ RESIDUAL ANALYSIS REMOVED - NOW HANDLED BY STATIONARITY
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
import warnings

from ..config.constants import Config

warnings.filterwarnings('ignore')

class TemporalStructureAnalyzer:
    """
    ✅ REFACTORED: 4 Pure Analysis-Visualization Pairs
    1. Seasonal Patterns (groupby-based)
    2. FFT Frequency Analysis 
    3. MSTL Decomposition (returns residual for Stationarity analysis)
    4. Wavelet Analysis
    
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
                   period_range: Tuple[int, int] = (7, 730),
                   return_n_dominant: int = 10) -> Dict[str, Any]:
        """
        ✅ ENHANCED ANALYSIS: FFT-based frequency analysis with parameterized dominant periods
        
        Args:
            detrend_window: Window size for trend removal
            top_n: Number of top periods to find during analysis
            period_range: Valid period range (min_days, max_days)
            return_n_dominant: Number of dominant periods to return for visualization
            
        Returns:
            Dict containing FFT analysis results with configurable dominant periods
        """
        print(f"🔊 Enhanced FFT Frequency Analysis")
        print(f"   - Detrend window: {detrend_window}")
        print(f"   - Top periods to find: {top_n}")
        print(f"   - Period range: {period_range}")
        print(f"   - Dominant periods to return: {return_n_dominant}")

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
        dominant_periods_int = sorted(list(set(np.round(valid_periods).astype(int))), reverse=True)
        
        # Return only the requested number of dominant periods
        dominant_periods_return = dominant_periods_int[:return_n_dominant]

        print(f"   📊 Found {len(dominant_periods_int)} total dominant periods")
        print(f"   📊 Returning top {len(dominant_periods_return)} dominant periods:")
        for i, period in enumerate(dominant_periods_return, 1):
            print(f"      {i}. {period} days")

        return {
            'success': True,
            'dominant_periods': dominant_periods_return,
            'all_dominant_periods': dominant_periods_int,  # Full list for reference
            'all_periods': periods,
            'magnitudes': positive_fft_magnitude,
            'frequencies': positive_frequencies,
            'detrended_series': detrended,
            'parameters': {
                'detrend_window': detrend_window,
                'top_n': top_n,
                'period_range': period_range,
                'return_n_dominant': return_n_dominant
            }
        }

    def plot_spectrum(self,
                     fft_results: Dict[str, Any],
                     figsize: Tuple[int, int] = (15, 8),
                     show_dominant: bool = True,
                     max_period: int = 365,
                     log_scale: bool = True,
                     bar_chart_periods: Optional[int] = None) -> None:
        """
        ✅ ENHANCED VISUALIZATION: Plot frequency spectrum with flexible dominant periods display
        
        Args:
            fft_results: Results from analyze_fft()
            figsize: Figure size
            show_dominant: Whether to highlight dominant periods
            max_period: Maximum period to show
            log_scale: Whether to use log scale for power
            bar_chart_periods: Number of periods to show in bar chart (None = use all from results)
        """
        if not fft_results['success']:
            print("❌ Cannot plot spectrum: FFT analysis failed")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Enhanced Frequency Domain Analysis', fontsize=16, fontweight='bold')

        periods = fft_results['all_periods']
        magnitudes = fft_results['magnitudes']
        dominant_periods = fft_results['dominant_periods']

        # Determine how many periods to show in bar chart
        if bar_chart_periods is None:
            periods_to_show = dominant_periods
            n_periods = len(dominant_periods)
        else:
            periods_to_show = dominant_periods[:bar_chart_periods]
            n_periods = min(bar_chart_periods, len(dominant_periods))

        print(f"   📊 Displaying {n_periods} dominant periods in bar chart")

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
            for period in periods_to_show:
                if period <= max_period:
                    period_idx = np.argmin(np.abs(periods_plot - period))
                    axes[0].axvline(x=period, color='red', linestyle='--', alpha=0.7)
                    axes[0].text(period, power[period_idx], f'{period}d',
                               rotation=90, verticalalignment='bottom',
                               fontsize=9, ha='center')

        # Enhanced dominant periods bar chart
        periods_power = []
        for period in periods_to_show:
            if period <= max_period:
                period_idx = np.argmin(np.abs(periods_plot - period))
                periods_power.append(power[period_idx])
            else:
                periods_power.append(0)

        # Create bar chart with better formatting
        x_positions = range(len(periods_to_show))
        bars = axes[1].bar(x_positions, periods_power, alpha=0.7, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(periods_to_show))))
        
        axes[1].set_xlabel('Dominant Period Rank')
        axes[1].set_ylabel(ylabel)
        axes[1].set_title(f'Top {n_periods} Dominant Periods')
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels([f'{p}d' for p in periods_to_show], 
                               rotation=45 if n_periods > 8 else 0,
                               ha='right' if n_periods > 8 else 'center')
        axes[1].grid(True, alpha=0.3)

        # Add value labels on bars if not too many
        if n_periods <= 15:
            for i, (bar, period) in enumerate(zip(bars, periods_to_show)):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{period}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.show()
    # =============================================================================
    # PAIR 3: MSTL DECOMPOSITION - PURE SEPARATION ✅
    # =============================================================================

    def decompose_mstl(self,
                      periods: List[int],
                      # === MSTL-Specific Parameters ===
                      windows: Optional[List[int]] = None,
                      lmbda: Optional[float] = None,
                      iterate: int = 2,
                      # === STL Parameters (passed through stl_kwargs) ===
                      **stl_kwargs) -> Dict[str, Any]:
        """
        ✅ SIMPLIFIED MSTL DECOMPOSITION: log1p when lmbda=None, Box-Cox otherwise
        ✅ RETURNS RESIDUAL: For Stationarity analysis
        ✅ AUTO-HANDLES ZEROS: log1p naturally handles zeros
        
        Args:
            periods (List[int]): The period of each seasonal component
                               (e.g., [7, 30, 365] for daily data with weekly, monthly, annual patterns)
            
            === MSTL-Specific Parameters ===
            windows (Optional[List[int]]): The lengths of each seasonal smoother with respect to each period.
                                         Must be odd. If None, default values from original paper are used.
                                         Large values → less seasonal variability over time.
            
            lmbda (Optional[float]): Transformation parameter:
                                   - None: Apply log1p transform (handles zeros naturally)
                                   - "auto": Box-Cox auto-select lambda (with zero-handling)
                                   - float: Box-Cox with specific lambda value
            
            iterate (int): Number of iterations to use to refine the seasonal component.
                          Default = 2
            
        Returns:
            Dict containing MSTL decomposition results including residual
        """
        print(f"📊 MSTL Decomposition")
        print(f"   - Periods: {periods}")
        print(f"   - Windows: {windows}")
        print(f"   - Lambda: {lmbda}")
        print(f"   - Iterations: {iterate}")
        print(f"   - STL kwargs: {stl_kwargs}")

        try:
            from statsmodels.tsa.seasonal import MSTL

            # Prepare time series
            ts_data = self.df.set_index(self.date_col).sort_index()
            target_ts = ts_data[self.target_col].dropna()

            # Handle transformation based on lmbda value
            if lmbda is None:
                # Apply log1p transformation (handles zeros naturally)
                print(f"   🔄 Applying log1p transformation (lmbda=None)")
                transformed_ts = np.log1p(target_ts)
                transform_method = 'log1p'
                mstl_lmbda = None  # No additional Box-Cox in MSTL
                
            else:
                # Use Box-Cox transformation within MSTL
                print(f"   🔄 Using MSTL Box-Cox transformation (lmbda={lmbda})")
                
                # Check for zeros if using Box-Cox
                zero_count = (target_ts == 0).sum()
                if zero_count > 0:
                    print(f"   📊 Zero values detected: {zero_count}")
                    print(f"   ⚠️ Box-Cox requires positive data, zeros detected")
                
                transformed_ts = target_ts
                transform_method = 'box_cox'
                mstl_lmbda = lmbda

            # First attempt: Try MSTL with transformed data
            try:
                mstl = MSTL(transformed_ts, 
                           periods=periods,
                           windows=windows,
                           lmbda=mstl_lmbda,  # Only use Box-Cox if not log1p
                           iterate=iterate,
                           stl_kwargs=stl_kwargs)
                
                mstl_result = mstl.fit()
                
                print(f"   ✅ MSTL decomposition completed successfully")
                print(f"   📊 Transform method: {transform_method}")
                print(f"   📊 Components extracted: trend, seasonal, residual")
                print(f"   📤 Residual component available for Stationarity analysis")

                return {
                    'success': True,
                    'mstl_obj': mstl_result,
                    'original_ts': target_ts,
                    'transformed_ts': transformed_ts,
                    'transform_method': transform_method,
                    'adjusted_ts': None,  # No adjustment needed
                    'epsilon_added': None,
                    'periods': periods,
                    'windows': windows,
                    'lmbda': lmbda,
                    'mstl_lmbda': mstl_lmbda,
                    'iterate': iterate,
                    'stl_kwargs': stl_kwargs,
                    'trend': mstl_result.trend,
                    'seasonal': mstl_result.seasonal,
                    'resid': mstl_result.resid,
                }
                
            except Exception as first_error:
                # Check if this is a Box-Cox related error and if we can apply fallback
                if (transform_method == 'box_cox' and lmbda == "auto" and 
                    ("positive" in str(first_error).lower() or 
                     "must be positive" in str(first_error).lower())):
                    
                    print(f"   🔧 Box-Cox failed with original data: {first_error}")
                    print(f"   🚀 Applying auto-fallback: Adding epsilon to zeros")
                    
                    # Apply fallback mechanism
                    epsilon = 1e-8  # Very small constant
                    zero_mask = target_ts == 0
                    adjusted_ts = target_ts.copy()
                    adjusted_ts[zero_mask] = epsilon
                    
                    zeros_adjusted = zero_mask.sum()
                    print(f"   📊 Adjusted {zeros_adjusted} zero values with epsilon={epsilon}")
                    
                    try:
                        # Retry MSTL with adjusted data
                        mstl_adjusted = MSTL(adjusted_ts, 
                                           periods=periods,
                                           windows=windows,
                                           lmbda=lmbda,
                                           iterate=iterate,
                                           stl_kwargs=stl_kwargs)
                        
                        mstl_result = mstl_adjusted.fit()
                        
                        print(f"   ✅ MSTL decomposition completed with adjusted data")
                        print(f"   📊 Transform method: {transform_method} (with epsilon)")
                        print(f"   📊 Components extracted: trend, seasonal, residual")
                        print(f"   ⚠️ Note: {zeros_adjusted} zero values adjusted with epsilon={epsilon}")
                        print(f"   📤 Residual component available for Stationarity analysis")

                        return {
                            'success': True,
                            'mstl_obj': mstl_result,
                            'original_ts': target_ts,
                            'transformed_ts': adjusted_ts,
                            'transform_method': f'{transform_method}_epsilon',
                            'adjusted_ts': adjusted_ts,
                            'epsilon_added': epsilon,
                            'zeros_adjusted': zeros_adjusted,
                            'periods': periods,
                            'windows': windows,
                            'lmbda': lmbda,
                            'mstl_lmbda': lmbda,
                            'iterate': iterate,
                            'stl_kwargs': stl_kwargs,
                            'trend': mstl_result.trend,
                            'seasonal': mstl_result.seasonal,
                            'resid': mstl_result.resid,
                            'fallback_used': True
                        }
                        
                    except Exception as second_error:
                        print(f"   ❌ MSTL failed even after adjustment: {second_error}")
                        raise second_error
                else:
                    # Not a Box-Cox zero error, re-raise original error
                    raise first_error

        except Exception as e:
            print(f"   ⚠️ MSTL decomposition failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'periods': periods,
                'windows': windows,
                'lmbda': lmbda,
                'transform_method': transform_method if 'transform_method' in locals() else 'unknown'
            }

    def plot_decomposition(self,
                          mstl_results: Dict[str, Any],
                          figsize: Tuple[int, int] = (15, 12),
                          show_components: List[str] = ['trend', 'seasonal', 'resid'],
                          original_scale: bool = True) -> None:
        """
        ✅ ENHANCED VISUALIZATION: Handles log1p and Box-Cox transformations
        
        Args:
            mstl_results: Results from decompose_mstl()
            figsize: Figure size
            show_components: Components to show
            original_scale: Whether to convert back to original scale
        """
        if not mstl_results['success']:
            print("❌ Cannot plot decomposition: MSTL failed")
            return

        transform_method = mstl_results.get('transform_method', 'unknown')
        n_plots = 1 + len(show_components)  # Original + components

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(f'MSTL Decomposition Results ({transform_method})', 
                    fontsize=16, fontweight='bold')

        plot_idx = 0

        # Original series
        original_data = mstl_results['original_ts']
        if original_scale and transform_method == 'log1p':
            # For log1p, we can show both scales
            axes[plot_idx].plot(original_data.index, original_data.values, 'b-', linewidth=1)
            axes[plot_idx].set_title('Original Data (Original Scale)')
            axes[plot_idx].set_ylabel('Precipitation (mm)')
        elif original_scale and 'box_cox' in transform_method:
            # For Box-Cox, show original scale
            axes[plot_idx].plot(original_data.index, original_data.values, 'b-', linewidth=1)
            axes[plot_idx].set_title('Original Data (Original Scale)')
            axes[plot_idx].set_ylabel('Precipitation (mm)')
        else:
            # Show transformed scale
            transformed_data = mstl_results['transformed_ts']
            axes[plot_idx].plot(transformed_data.index, transformed_data.values, 'b-', linewidth=1)
            axes[plot_idx].set_title(f'Original Data ({transform_method} scale)')
            axes[plot_idx].set_ylabel(f'{transform_method} Precipitation')
        
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Trend component
        if 'trend' in show_components:
            trend = mstl_results['trend']
            axes[plot_idx].plot(trend.index, trend.values, 'g-', linewidth=2)
            axes[plot_idx].set_title('Trend Component')
            axes[plot_idx].set_ylabel(f'{transform_method} Trend')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Seasonal component
        if 'seasonal' in show_components:
            seasonal = mstl_results['seasonal']
            axes[plot_idx].plot(seasonal.index, seasonal.values, 'r-', linewidth=1)
            axes[plot_idx].set_title('Combined Seasonal Component')
            axes[plot_idx].set_ylabel(f'{transform_method} Seasonal')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Residual component
        if 'resid' in show_components:
            resid = mstl_results['resid']
            axes[plot_idx].plot(resid.index, resid.values, 'purple', linewidth=1, alpha=0.7)
            axes[plot_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[plot_idx].set_title('Residual Component (→ Stationarity Analysis)')
            axes[plot_idx].set_ylabel(f'{transform_method} Residual')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()

        print(f"   📤 Residual component ready for Stationarity analysis")
        print(f"   🔄 Transform method used: {transform_method}")

    # =============================================================================
    # PAIR 4: WAVELET ANALYSIS - PURE SEPARATION ✅
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
    # MAIN ORCHESTRATION - CONVENIENCE METHODS (UPDATED - NO RESIDUAL ANALYSIS)
    # =============================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """
        Convenience method: Run all analyses with default parameters
        ✅ UPDATED: No residual analysis (now handled by Stationarity)
        
        Returns:
            Dict containing all analysis results + residual for Stationarity
        """
        print("\n" + "="*70)
        print("🕐 TEMPORAL STRUCTURE ANALYSIS - 4 PAIRS REFACTORED")
        print("="*70)

        # Step 1: Seasonal patterns analysis
        seasonal_results = self.analyze_seasonal_patterns()

        # Step 2: FFT frequency analysis
        fft_results = self.analyze_fft()

        # Step 3: MSTL decomposition (returns residual for Stationarity)
        periods = fft_results.get('dominant_periods', [7, 30, 365])[:3]  # Use top 3
        mstl_results = self.decompose_mstl(periods)

        # Step 4: Wavelet analysis
        wavelet_results = self.analyze_wavelet()

        # Combine all results
        results = {
            'seasonal_patterns': seasonal_results,
            'frequency_analysis': fft_results,
            'mstl_decomposition': mstl_results,  # ✅ Contains residual for Stationarity
            'wavelet_analysis': wavelet_results,
            'component_name': 'TemporalStructureAnalyzer_4Pairs_NoResidualAnalysis'
        }

        print(f"\n✅ TEMPORAL ANALYSIS COMPLETED - 4 PAIRS IMPLEMENTED")
        print(f"   📤 MSTL residual ready for Stationarity analysis")
        return results

    def visualize_all(self, results: Dict[str, Any]) -> None:
        """
        Convenience method: Visualize all components
        ✅ UPDATED: No residual diagnostics visualization
        
        Args:
            results: Results from analyze_all()
        """
        print("\n🎨 Visualizing All Components...")

        # 1. Seasonal patterns
        self.plot_seasonal_patterns(results['seasonal_patterns'])

        # 2. Frequency spectrum  
        if results['frequency_analysis']['success']:
            self.plot_spectrum(results['frequency_analysis'])

        # 3. MSTL decomposition (shows residual for reference)
        if results['mstl_decomposition']['success']:
            self.plot_decomposition(results['mstl_decomposition'])

        # 4. Wavelet scalogram
        if results['wavelet_analysis']['success']:
            self.plot_scalogram(results['wavelet_analysis'])

        print(f"\n📋 Note: Residual analysis is now handled by StationarityAutocorrelationAnalyzer")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_temporal_structure(df: pd.DataFrame, 
                              target_col: str = None, 
                              date_col: str = None) -> Dict[str, Any]:
    """
    Convenience function for complete temporal analysis
    ✅ UPDATED: Returns residual for Stationarity analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        
    Returns:
        Dict containing all analysis results including residual for Stationarity
    """
    analyzer = TemporalStructureAnalyzer(df, target_col, date_col)
    return analyzer.analyze_all()

def get_mstl_residual_for_stationarity(df: pd.DataFrame,
                                      target_col: str = None,
                                      date_col: str = None,
                                      periods: List[int] = [7, 30, 365]) -> Optional[pd.Series]:
    """
    Convenience function to get MSTL residual for Stationarity analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        periods: Seasonal periods for MSTL
        
    Returns:
        MSTL residual series or None if failed
    """
    analyzer = TemporalStructureAnalyzer(df, target_col, date_col)
    mstl_results = analyzer.decompose_mstl(periods)
    
    if mstl_results['success']:
        return mstl_results['resid']
    else:
        print(f"⚠️ MSTL decomposition failed: {mstl_results.get('error', 'Unknown error')}")
        return None