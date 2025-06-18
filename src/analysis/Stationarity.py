# =============================================================================
# COMPONENT 3: STATIONARITY & AUTOCORRELATION DIAGNOSTICS (REFACTORED)
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot

warnings.filterwarnings('ignore')


class StationarityAutocorrelationAnalyzer:
    """
    Phần 3: Chẩn đoán Tính dừng và Tự tương quan
    ✅ REFACTORED: Loại bỏ BaseAnalyzer, thêm residual analysis từ TemporalAnalysis
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Lượng mưa', 
                 date_col: str = 'Ngày', mstl_results: Optional[Dict] = None):
        """
        Initialize with configurable parameters
        
        Args:
            df: DataFrame containing time series data
            target_col: Target column name for analysis
            date_col: Date column name
            mstl_results: Optional MSTL decomposition results from TemporalAnalysis
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.mstl_results = mstl_results
        
        # Prepare data
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for analysis"""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        
        # Sort by date
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    def analyze(self) -> Dict[str, Any]:
        """
        Main workflow: Stationarity → Autocorrelation → Residual → Synthesis
        
        Returns:
            Dict containing all analysis results
        """
        print("\n" + "="*80)
        print("🔍 PHẦN 3: STATIONARITY & AUTOCORRELATION DIAGNOSTICS")
        print("="*80)
        print("🎯 Mục tiêu: Chẩn đoán tính dừng và cấu trúc tự tương quan")

        # Prepare time series data
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        if len(target_ts) < 50:
            print("   ⚠️ Insufficient data for comprehensive diagnostics")
            return {'success': False, 'error': 'Insufficient data'}

        # Step 1: Stationarity Diagnostics
        stationarity_results = self._stationarity_diagnostics(target_ts)

        # Step 2: Autocorrelation Analysis
        autocorr_results = self._autocorrelation_analysis(target_ts)

        # Step 3: Residual Diagnostics (if MSTL results available)
        residual_results = self._residual_diagnostics() if self.mstl_results else None

        # Step 4: Comprehensive Synthesis
        synthesis = self._comprehensive_synthesis(stationarity_results, autocorr_results, residual_results)

        # Combine all results
        results = {
            'stationarity': stationarity_results,
            'autocorrelation': autocorr_results,
            'residual_diagnostics': residual_results,
            'synthesis': synthesis,
            'component_name': 'StationarityAutocorrelationAnalyzer_Refactored'
        }

        return results

    # =============================================================================
    # STATIONARITY ANALYSIS METHODS
    # =============================================================================

    def _stationarity_diagnostics(self, target_ts: pd.Series) -> Dict[str, Any]:
        """Bước 1: Chẩn đoán Tính dừng"""
        print(f"\n🔍 BƯỚC 1: CHẨN ĐOÁN TÍNH DỪNG")
        print("="*60)

        # ADF Test
        adf_results = self._perform_adf_test(target_ts)

        # KPSS Test
        kpss_results = self._perform_kpss_test(target_ts)

        # Combined Assessment
        assessment = self._stationarity_assessment(adf_results, kpss_results)

        # Visual Analysis
        self._visual_stationarity_analysis(target_ts)

        return {
            'adf_test': adf_results,
            'kpss_test': kpss_results,
            'assessment': assessment,
            'success': True
        }

    def _perform_adf_test(self, ts: pd.Series) -> Dict[str, Any]:
        """ADF Test implementation"""
        try:
            result = adfuller(ts, regression='ct', autolag='AIC')

            print(f"   📊 ADF Test:")
            print(f"      - Statistic: {result[0]:.6f}")
            print(f"      - P-value: {result[1]:.6f}")
            print(f"      - Result: {'✅ Stationary' if result[1] < 0.05 else '❌ Non-stationary'}")

            return {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _perform_kpss_test(self, ts: pd.Series) -> Dict[str, Any]:
        """KPSS Test implementation"""
        try:
            result = kpss(ts, regression='ct')

            print(f"   📊 KPSS Test:")
            print(f"      - Statistic: {result[0]:.6f}")
            print(f"      - P-value: {result[1]:.6f}")
            print(f"      - Result: {'✅ Stationary' if result[1] > 0.05 else '❌ Non-stationary'}")

            return {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[3],
                'is_stationary': result[1] > 0.05,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _stationarity_assessment(self, adf: Dict, kpss: Dict) -> Dict[str, Any]:
        """Combined stationarity assessment"""
        adf_stat = adf.get('is_stationary', False)
        kpss_stat = kpss.get('is_stationary', False)

        if adf_stat and kpss_stat:
            conclusion = "✅ STATIONARY"
            stationarity_type = "Stationary"
        elif adf_stat and not kpss_stat:
            conclusion = "🟡 DIFFERENCE STATIONARY"
            stationarity_type = "Difference Stationary"
        elif not adf_stat and kpss_stat:
            conclusion = "🟡 TREND STATIONARY"
            stationarity_type = "Trend Stationary"
        else:
            conclusion = "❌ NON-STATIONARY"
            stationarity_type = "Non-stationary"

        print(f"   🎯 Kết luận: {conclusion}")

        return {
            'conclusion': conclusion,
            'stationarity_type': stationarity_type,
            'tests_agree': adf_stat == kpss_stat,
            'adf_agrees': adf_stat,
            'kpss_agrees': kpss_stat
        }

    def _visual_stationarity_analysis(self, target_ts: pd.Series) -> None:
        """Visual stationarity analysis"""
        window = min(365, len(target_ts) // 4)
        rolling_mean = target_ts.rolling(window=window).mean()
        rolling_std = target_ts.rolling(window=window).std()

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stationarity Visual Diagnostics', fontsize=14, fontweight='bold')

        # Original + Rolling Mean
        axes[0, 0].plot(target_ts.index, target_ts.values, 'b-', alpha=0.6, label='Original')
        axes[0, 0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, 
                       label=f'Rolling Mean ({window}d)')
        axes[0, 0].set_title('Time Series with Rolling Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Rolling Std
        axes[0, 1].plot(rolling_std.index, rolling_std.values, 'g-', linewidth=2)
        axes[0, 1].set_title(f'Rolling Standard Deviation ({window}d)')
        axes[0, 1].grid(True, alpha=0.3)

        # First Difference
        first_diff = target_ts.diff().dropna()
        axes[1, 0].plot(first_diff.index, first_diff.values, 'purple', alpha=0.7)
        axes[1, 0].set_title('First Difference')
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution Comparison
        axes[1, 1].hist(target_ts.values, bins=30, alpha=0.7, label='Original', density=True)
        axes[1, 1].hist(first_diff.values, bins=30, alpha=0.7, label='First Difference', density=True)
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # =============================================================================
    # AUTOCORRELATION ANALYSIS METHODS
    # =============================================================================

    def _autocorrelation_analysis(self, target_ts: pd.Series) -> Dict[str, Any]:
        """Bước 2: Phân tích Tự tương quan"""
        print(f"\n🔄 BƯỚC 2: PHÂN TÍCH TỰ TƯƠNG QUAN")
        print("="*60)

        # ACF Analysis on original series
        acf_results = self._perform_acf_analysis(target_ts, "Original Series")

        # PACF Analysis on original series
        pacf_results = self._perform_pacf_analysis(target_ts, "Original Series")

        # SARIMA Suggestions
        sarima_suggestions = self._generate_sarima_suggestions(acf_results, pacf_results)

        # Visualization
        self._autocorrelation_visualization(target_ts, "Original Series")

        return {
            'acf_results': acf_results,
            'pacf_results': pacf_results,
            'sarima_suggestions': sarima_suggestions,
            'success': True
        }

    def _perform_acf_analysis(self, ts: pd.Series, label: str) -> Dict[str, Any]:
        """ACF Analysis implementation"""
        try:
            max_lags = min(100, len(ts) // 4)
            acf_values, acf_confint = acf(ts, nlags=max_lags, alpha=0.05, fft=True)

            # Find significant lags
            significant_lags = []
            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) > abs(acf_confint[lag, 1] - acf_values[lag]):
                    significant_lags.append((lag, acf_values[lag]))

            print(f"   📈 ACF ({label}) - Significant lags: {len(significant_lags)}")

            return {
                'acf_values': acf_values,
                'significant_lags': significant_lags,
                'max_lags': max_lags,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _perform_pacf_analysis(self, ts: pd.Series, label: str) -> Dict[str, Any]:
        """PACF Analysis implementation"""
        try:
            max_lags = min(100, len(ts) // 4)
            pacf_values, pacf_confint = pacf(ts, nlags=max_lags, alpha=0.05)

            # Find significant lags
            significant_lags = []
            for lag in range(1, len(pacf_values)):
                if abs(pacf_values[lag]) > abs(pacf_confint[lag, 1] - pacf_values[lag]):
                    significant_lags.append((lag, pacf_values[lag]))

            print(f"   📈 PACF ({label}) - Significant lags: {len(significant_lags)}")

            return {
                'pacf_values': pacf_values,
                'significant_lags': significant_lags,
                'max_lags': max_lags,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_sarima_suggestions(self, acf_results: Dict, pacf_results: Dict) -> Dict[str, Any]:
        """Generate SARIMA parameter suggestions"""
        if not (acf_results['success'] and pacf_results['success']):
            return {'success': False}

        # Extract significant lags
        pacf_lags = [lag for lag, _ in pacf_results['significant_lags'][:5]]
        acf_lags = [lag for lag, _ in acf_results['significant_lags'][:5]]

        # Simple heuristics for SARIMA orders
        p = min(3, len(pacf_lags)) if pacf_lags else 1
        q = min(3, len(acf_lags)) if acf_lags else 1

        # Seasonal parameters (check for seasonal lags)
        seasonal_lags = [lag for lag in pacf_lags + acf_lags if lag in [7, 14, 30, 365]]
        P = 1 if seasonal_lags else 0
        Q = 1 if seasonal_lags else 0
        s = max(seasonal_lags) if seasonal_lags else 0

        suggestions = {
            'nonseasonal_ar': p,
            'nonseasonal_ma': q,
            'seasonal_ar': P,
            'seasonal_ma': Q,
            'seasonal_period': s,
            'suggested_models': [
                f'SARIMA({p},1,{q})x({P},1,{Q},{s})' if s > 0 else f'ARIMA({p},1,{q})',
                f'SARIMA({p},0,{q})x({P},0,{Q},{s})' if s > 0 else f'ARIMA({p},0,{q})'
            ],
            'success': True
        }

        print(f"   🎯 SARIMA Suggestions: {suggestions['suggested_models']}")
        return suggestions

    def _autocorrelation_visualization(self, ts: pd.Series, label: str) -> None:
        """ACF/PACF visualization"""
        max_lags = min(40, len(ts) // 4)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f'Autocorrelation Analysis - {label}', fontsize=14, fontweight='bold')

        # ACF plot
        try:
            plot_acf(ts, lags=max_lags, ax=axes[0, 0], alpha=0.05)
            axes[0, 0].set_title(f'ACF - {label}')
            axes[0, 0].grid(True, alpha=0.3)
        except Exception:
            axes[0, 0].text(0.5, 0.5, 'ACF Plot Error', ha='center', va='center', 
                           transform=axes[0, 0].transAxes)

        # PACF plot
        try:
            plot_pacf(ts, lags=max_lags, ax=axes[0, 1], alpha=0.05)
            axes[0, 1].set_title(f'PACF - {label}')
            axes[0, 1].grid(True, alpha=0.3)
        except Exception:
            axes[0, 1].text(0.5, 0.5, 'PACF Plot Error', ha='center', va='center', 
                           transform=axes[0, 1].transAxes)

        # First difference ACF
        first_diff = ts.diff().dropna()
        if len(first_diff) > max_lags:
            try:
                plot_acf(first_diff, lags=max_lags, ax=axes[1, 0], alpha=0.05)
                axes[1, 0].set_title('ACF - First Difference')
                axes[1, 0].grid(True, alpha=0.3)
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'ACF Diff Error', ha='center', va='center', 
                               transform=axes[1, 0].transAxes)

        # First difference PACF
        if len(first_diff) > max_lags:
            try:
                plot_pacf(first_diff, lags=max_lags, ax=axes[1, 1], alpha=0.05)
                axes[1, 1].set_title('PACF - First Difference')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception:
                axes[1, 1].text(0.5, 0.5, 'PACF Diff Error', ha='center', va='center', 
                               transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.show()

    # =============================================================================
    # RESIDUAL ANALYSIS METHODS (MIGRATED FROM TEMPORALANALYSIS)
    # =============================================================================

    def _residual_diagnostics(self) -> Dict[str, Any]:
        """Bước 3: Chẩn đoán Residual từ MSTL"""
        if not self.mstl_results or 'resid' not in self.mstl_results:
            print(f"\n⚠️ MSTL results not available for residual diagnostics")
            return {'success': False, 'error': 'No MSTL results'}

        print(f"\n🧪 BƯỚC 3: CHẨN ĐOÁN RESIDUAL MSTL")
        print("="*60)

        residual = self.mstl_results['resid'].dropna()

        if len(residual) < 50:
            return {'success': False, 'error': 'Insufficient residual data'}

        # Use the enhanced residual analysis from TemporalAnalysis
        residual_analysis = self.diagnose_residuals(residual)
        
        if residual_analysis['success']:
            # Add visualization
            self.plot_residual_diagnostics(residual_analysis)
            
            # Legacy compatibility - add old format fields
            residual_analysis.update({
                'residual_acf': self._perform_acf_analysis(residual, "MSTL Residual"),
                'residual_pacf': self._perform_pacf_analysis(residual, "MSTL Residual"),
                'ljung_box': residual_analysis['ljung_box_results'],
                'quality_assessment': self._assess_mstl_quality(residual_analysis)
            })

        return residual_analysis

    def diagnose_residuals(self,
                          residual_series: pd.Series,
                          ljung_box_lags: int = 10,
                          acf_lags: int = 40,
                          normality_tests: bool = True) -> Dict[str, Any]:
        """
        ✅ MIGRATED FROM TEMPORALANALYSIS: Pure residual diagnostic analysis
        
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
        ✅ MIGRATED FROM TEMPORALANALYSIS: Pure visualization of residual diagnostics
        
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

    def _assess_mstl_quality(self, residual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess MSTL decomposition quality"""
        if not residual_analysis.get('success'):
            return {'overall_quality': 'Unknown', 'reasons': ['Analysis failed']}

        ljung_box = residual_analysis.get('ljung_box_results')
        autocorr_clean = residual_analysis.get('autocorr_clean', False)
        
        # Count significant lags in residuals (simplified estimation)
        significant_residual_lags = ljung_box.get('significant_lags', 0) if ljung_box else 0

        if autocorr_clean and significant_residual_lags < 5:
            quality = 'Good'
            reasons = ['Residuals appear random', 'Few significant autocorrelations']
        elif autocorr_clean or significant_residual_lags < 10:
            quality = 'Moderate'
            reasons = ['Some structure may remain', 'Consider additional AR/MA terms']
        else:
            quality = 'Poor'
            reasons = ['Significant structure in residuals', 'MSTL may be insufficient']

        print(f"   📊 MSTL Quality: {quality}")
        for reason in reasons:
            print(f"      • {reason}")

        return {
            'overall_quality': quality,
            'reasons': reasons,
            'significant_lags_count': significant_residual_lags,
            'residuals_random': autocorr_clean
        }

    # =============================================================================
    # LEGACY METHODS (for backward compatibility)
    # =============================================================================

    def _ljung_box_test(self, residual: pd.Series) -> Dict[str, Any]:
        """Legacy Ljung-Box test wrapper"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lbvalue = acorr_ljungbox(residual, lags=min(20, len(residual)//4), return_df=True)
            pvalue_min = lbvalue['lb_pvalue'].min()

            print(f"   🧪 Ljung-Box Test:")
            print(f"      - Min p-value: {pvalue_min:.6f}")
            print(f"      - Result: {'✅ Random residuals' if pvalue_min > 0.05 else '⚠️ Some structure remains'}")

            return {
                'ljung_box_results': lbvalue,
                'min_pvalue': float(pvalue_min),
                'is_random': pvalue_min > 0.05,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _residual_visualization(self, residual: pd.Series) -> None:
        """Legacy residual visualization wrapper"""
        # Create a residual analysis result dict and call the new method
        residual_analysis = {
            'success': True,
            'residuals': residual,
            'parameters': {'acf_lags': 40}
        }
        self.plot_residual_diagnostics(residual_analysis, layout='2x2')

    # =============================================================================
    # SYNTHESIS METHODS
    # =============================================================================

    def _comprehensive_synthesis(self, stationarity_results: Dict, 
                                autocorr_results: Dict, 
                                residual_results: Optional[Dict]) -> Dict[str, Any]:
        """Bước 4: Tổng hợp & Kết luận"""
        print(f"\n📋 BƯỚC 4: TỔNG HỢP & KẾT LUẬN")
        print("="*60)

        # Extract key findings
        is_stationary = stationarity_results['assessment']['tests_agree']
        stationarity_type = stationarity_results['assessment']['stationarity_type']
        significant_lags = autocorr_results['acf_results']['significant_lags'][:5]
        seasonal_lags = [lag for lag, _ in significant_lags if lag in [7, 14, 30, 365]]

        # Generate synthesis report
        synthesis_report = {
            'stationarity_conclusion': stationarity_type,
            'differencing_needed': not is_stationary,
            'significant_lags': [lag for lag, _ in significant_lags],
            'seasonal_lags_detected': seasonal_lags,
            'model_recommendations': []
        }

        # Model recommendations
        if is_stationary:
            synthesis_report['model_recommendations'].append("ARMA models suitable")
        else:
            synthesis_report['model_recommendations'].append("ARIMA models recommended (d=1)")

        if seasonal_lags:
            synthesis_report['model_recommendations'].append("SARIMA models for seasonal patterns")

        # MSTL Quality Assessment
        if residual_results and residual_results.get('success'):
            mstl_quality = residual_results.get('quality_assessment')
            if mstl_quality:
                synthesis_report['mstl_decomposition_quality'] = mstl_quality

                if mstl_quality['overall_quality'] == 'Good':
                    synthesis_report['model_recommendations'].append("MSTL decomposition captured most patterns")
                else:
                    synthesis_report['model_recommendations'].append("Additional AR/MA terms may be needed")

        # Print synthesis
        print(f"   📊 PHÁT HIỆN CHÍNH:")
        print(f"      🔹 Tính dừng: {stationarity_type}")
        print(f"      🔹 Cần sai phân: {'Có' if synthesis_report['differencing_needed'] else 'Không'}")
        print(f"      🔹 Lags quan trọng: {synthesis_report['significant_lags'][:5]}")
        print(f"      🔹 Lags mùa vụ: {seasonal_lags}")

        print(f"\n   💡 KHUYẾN NGHỊ MÔ HÌNH:")
        for rec in synthesis_report['model_recommendations']:
            print(f"      • {rec}")

        return synthesis_report


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_stationarity_autocorrelation(df: pd.DataFrame,
                                        target_col: str = 'Lượng mưa',
                                        date_col: str = 'Ngày',
                                        mstl_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for complete stationarity and autocorrelation analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        mstl_results: Optional MSTL decomposition results
        
    Returns:
        Dict containing all analysis results
    """
    analyzer = StationarityAutocorrelationAnalyzer(df, target_col, date_col, mstl_results)
    return analyzer.analyze()
