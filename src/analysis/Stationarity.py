# =============================================================================
# COMPONENT 3: STATIONARITY & AUTOCORRELATION DIAGNOSTICS 
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
    Phần 3: Chẩn đoán Tính dừng và Tự tương quan với User-Curated Periods
    
    Key Enhancement:
    - Uses user-curated representative periods (e.g., [7, 30, 122, 365])
    - Simple, clean approach without FFT complexity
    - Theory-driven parameter selection based on known periods
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Lượng mưa', 
                 date_col: str = 'Ngày', 
                 mstl_results: Optional[Dict] = None,
                 representative_periods: Optional[List[int]] = None):
        """
        Initialize with configurable parameters
        
        Args:
            df: DataFrame containing time series data
            target_col: Target column name for analysis
            date_col: Date column name
            mstl_results: Optional MSTL decomposition results from TemporalAnalysis
            representative_periods: User-curated list of periods 
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.mstl_results = mstl_results
        
        # Use representative periods instead of complex FFT results
        self.representative_periods = representative_periods or []

        # Prepare data
        self._prepare_data()
        
        # Extract theory-driven parameters from representative periods
        self.theory_driven_params = self._extract_theory_driven_parameters()

    def _prepare_data(self) -> None:
        """Prepare data for analysis"""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        
        # Sort by date
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)


    def _extract_theory_driven_parameters(self) -> Dict[str, Any]:
        """        
        Returns:
            Dict containing theory-driven parameters for ACF/PACF analysis
        """
        if not self.representative_periods:
            print("   ⚠️ No representative periods provided - using default parameters")
            return {
                'has_periods': False,
                'max_lags_recommended': 100,  # Default
                'seasonal_periods': [],
                'representative_periods': [],
                'parameter_source': 'default'
            }

        # max_lags should be >= max(representative periods) + buffer
        max_period = max(self.representative_periods)
        buffer = max(20, max_period // 4)  # Adaptive buffer
        max_lags_recommended = max_period + buffer

        seasonal_periods = [p for p in self.representative_periods if p >= 7]  # Weekly or longer

        theory_params = {
            'has_periods': True,
            'max_lags_recommended': min(max_lags_recommended, 200),  # Cap at 200
            'seasonal_periods': seasonal_periods,
            'representative_periods': self.representative_periods,
            'parameter_source': 'theory_driven',
            'max_period': max_period,
            'buffer_used': buffer
        }

        print(f"   🎯 Theory-Driven Parameters Extracted:")
        print(f"      • Representative periods: {self.representative_periods}")
        print(f"      • Max lags recommended: {theory_params['max_lags_recommended']}")
        print(f"      • Seasonal periods: {seasonal_periods}")
        print(f"      • Parameter source: {theory_params['parameter_source']}")

        return theory_params


    def _autocorrelation_analysis(self, target_ts: pd.Series) -> Dict[str, Any]:
        """                
        Bước 2: Phân tích Tự tương quan với Representative Periods
        """
        print(f"\n🔄 BƯỚC 2: PHÂN TÍCH TỰ TƯƠNG QUAN (THEORY-DRIVEN)")
        print("="*60)

        # Use theory-driven max_lags
        max_lags = min(
            self.theory_driven_params['max_lags_recommended'],
            len(target_ts) // 4  # Safety constraint
        )

        print(f"   🎯 Theory-driven max_lags: {max_lags}")
        print(f"      Source: {self.theory_driven_params['parameter_source']}")
        if self.theory_driven_params['has_periods']:
            print(f"      Based on max period: {self.theory_driven_params['max_period']} + buffer: {self.theory_driven_params['buffer_used']}")

        # ACF Analysis
        acf_results = self._perform_acf_analysis(target_ts, max_lags, "Original Series")

        # PACF Analysis  
        pacf_results = self._perform_pacf_analysis(target_ts, max_lags, "Original Series")

        # Theory-driven SARIMA Suggestions
        sarima_suggestions = self._generate_theory_driven_sarima_suggestions(acf_results, pacf_results)

        # Visualization with theory context
        self._theory_driven_autocorrelation_visualization(target_ts, max_lags, "Original Series")

        return {
            'acf_results': acf_results,
            'pacf_results': pacf_results,
            'sarima_suggestions': sarima_suggestions,
            'theory_driven_params': self.theory_driven_params,
            'max_lags_used': max_lags,
            'success': True
        }

    def _perform_acf_analysis(self, ts: pd.Series, max_lags: int, label: str) -> Dict[str, Any]:
        """ACF Analysis"""
        try:
            acf_values, acf_confint = acf(ts, nlags=max_lags, alpha=0.05, fft=True)

            # Find significant lags
            significant_lags = []
            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) > abs(acf_confint[lag, 1] - acf_values[lag]):
                    significant_lags.append((lag, acf_values[lag]))

            # Find lags near representative periods (optional validation)
            periods_validated_lags = self._find_lags_near_periods(significant_lags) if self.theory_driven_params['has_periods'] else []

            print(f"   📈 ACF ({label}):")
            print(f"      • Total significant lags: {len(significant_lags)}")
            if periods_validated_lags:
                print(f"      • Lags near representative periods: {len(periods_validated_lags)}")

            return {
                'acf_values': acf_values,
                'significant_lags': significant_lags,
                'periods_validated_lags': periods_validated_lags,
                'max_lags': max_lags,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _perform_pacf_analysis(self, ts: pd.Series, max_lags: int, label: str) -> Dict[str, Any]:
        """PACF Analysis"""
        try:
            pacf_values, pacf_confint = pacf(ts, nlags=max_lags, alpha=0.05)

            # Find significant lags
            significant_lags = []
            for lag in range(1, len(pacf_values)):
                if abs(pacf_values[lag]) > abs(pacf_confint[lag, 1] - pacf_values[lag]):
                    significant_lags.append((lag, pacf_values[lag]))

            # Find lags near representative periods
            periods_validated_lags = self._find_lags_near_periods(significant_lags) if self.theory_driven_params['has_periods'] else []

            print(f"   📈 PACF ({label}):")
            print(f"      • Total significant lags: {len(significant_lags)}")
            if periods_validated_lags:
                print(f"      • Lags near representative periods: {len(periods_validated_lags)}")

            return {
                'pacf_values': pacf_values,
                'significant_lags': significant_lags,
                'periods_validated_lags': periods_validated_lags,
                'max_lags': max_lags,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _find_lags_near_periods(self, significant_lags: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """        
        Args:
            significant_lags: List of (lag, value) tuples from ACF/PACF
        
        Returns:
            List of lags that are near representative periods
        """
        if not self.representative_periods:
            return significant_lags

        # Simple approach: find lags within ±5 days of representative periods
        validated_lags = []
        tolerance = 5

        for lag, value in significant_lags:
            for period in self.representative_periods:
                if abs(lag - period) <= tolerance:
                    validated_lags.append((lag, value))
                    break

        return validated_lags

    def _generate_theory_driven_sarima_suggestions(self, acf_results: Dict, pacf_results: Dict) -> Dict[str, Any]:
        """        
        Theory-driven approach:
        1. Use representative periods as seasonal period candidates
        2. Standard ACF/PACF analysis for p,q parameters
        3. Clean, simple logic
        """
        if not (acf_results['success'] and pacf_results['success']):
            return {'success': False}

        # Standard approach for p, q parameters
        acf_lags = [lag for lag, _ in acf_results['significant_lags'][:5]]
        pacf_lags = [lag for lag, _ in pacf_results['significant_lags'][:5]]

        # Enhanced SARIMA parameter selection
        p = min(3, len(pacf_lags)) if pacf_lags else 1
        q = min(3, len(acf_lags)) if acf_lags else 1

        # Use representative periods directly for seasonal parameters
        seasonal_periods = self.theory_driven_params['seasonal_periods']
        
        if seasonal_periods:
            P = 1
            Q = 1
            s = max(seasonal_periods)  # Primary seasonal period
            all_seasonal_periods = seasonal_periods
        else:
            # Fallback to traditional detection
            all_acf_lags = [lag for lag, _ in acf_results['significant_lags']]
            all_pacf_lags = [lag for lag, _ in pacf_results['significant_lags']]
            
            seasonal_lags = []
            for lag in all_acf_lags + all_pacf_lags:
                for period in [7, 14, 30, 365]:
                    if abs(lag - period) <= 2:
                        seasonal_lags.append(period)
                        break
            
            all_seasonal_periods = list(set(seasonal_lags))
            P = 1 if all_seasonal_periods else 0
            Q = 1 if all_seasonal_periods else 0
            s = max(all_seasonal_periods) if all_seasonal_periods else 0

        # Generate model suggestions
        model_suggestions = []
        model_suggestions.append(f'SARIMA({p},1,{q})x({P},1,{Q},{s})' if s > 0 else f'ARIMA({p},1,{q})')
        model_suggestions.append(f'SARIMA({p},0,{q})x({P},0,{Q},{s})' if s > 0 else f'ARIMA({p},0,{q})')
        
        # Add multiple seasonal models if multiple periods available
        if len(seasonal_periods) > 1:
            s2 = sorted(seasonal_periods, reverse=True)[1]
            model_suggestions.append(f'SARIMA({p},1,{q})x(1,1,1,{s2})')

        suggestions = {
            'nonseasonal_ar': p,
            'nonseasonal_ma': q,
            'seasonal_ar': P,
            'seasonal_ma': Q,
            'seasonal_period': s,
            'all_seasonal_periods_detected': all_seasonal_periods,
            'representative_periods_used': self.representative_periods,  
            'suggested_models': model_suggestions,
            'methodology': 'theory_driven' if self.theory_driven_params['has_periods'] else 'traditional',
            'success': True
        }

        print(f"   🎯 Theory-Driven SARIMA Parameter Selection:")
        print(f"      • ACF lags (p): {acf_lags}")
        print(f"      • PACF lags (q): {pacf_lags}")
        print(f"      • Representative periods: {self.representative_periods}")
        print(f"   📊 SARIMA Suggestions: {suggestions['suggested_models']}")
        
        return suggestions

    def _theory_driven_autocorrelation_visualization(self, ts: pd.Series, max_lags: int, label: str) -> None:
        """ACF/PACF visualization with representative periods context"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Theory-Driven Autocorrelation Analysis - {label}', 
                     fontsize=14, fontweight='bold')

        # Add periods information to plots
        periods_info_text = ""
        if self.theory_driven_params['has_periods']:
            periods_info_text = f"Representative Periods: {self.representative_periods}"

        # ACF plot with periods context
        try:
            plot_acf(ts, lags=max_lags, ax=axes[0, 0], alpha=0.05)
            axes[0, 0].set_title(f'ACF - {label}\n{periods_info_text}')
            
            # Highlight representative periods
            if self.theory_driven_params['has_periods']:
                for period in self.representative_periods[:5]:  # Top 5 to avoid clutter
                    if period <= max_lags:
                        axes[0, 0].axvline(x=period, color='red', linestyle='--', alpha=0.6, 
                                         label=f'{period}d' if period == self.representative_periods[0] else "")
                if self.representative_periods:
                    axes[0, 0].legend()
            
            axes[0, 0].grid(True, alpha=0.3)
        except Exception:
            axes[0, 0].text(0.5, 0.5, 'ACF Plot Error', ha='center', va='center', 
                           transform=axes[0, 0].transAxes)

        # PACF plot with periods context
        try:
            plot_pacf(ts, lags=max_lags, ax=axes[0, 1], alpha=0.05)
            axes[0, 1].set_title(f'PACF - {label}\nMax lags: {max_lags} (theory-driven)')
            
            # Highlight representative periods
            if self.theory_driven_params['has_periods']:
                for period in self.representative_periods[:5]:
                    if period <= max_lags:
                        axes[0, 1].axvline(x=period, color='red', linestyle='--', alpha=0.6)
            
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


    def _comprehensive_synthesis(self, stationarity_results: Dict, 
                                autocorr_results: Dict, 
                                residual_results: Optional[Dict]) -> Dict[str, Any]:
        """        
        Bước 4: Tổng hợp & Kết luận với Theory-Driven Methodology
        """
        print(f"\n📋 BƯỚC 4: TỔNG HỢP & KẾT LUẬN (THEORY-DRIVEN)")
        print("="*60)

        # Extract key findings
        is_stationary = stationarity_results['assessment']['tests_agree']
        stationarity_type = stationarity_results['assessment']['stationarity_type']
        
        # Extract autocorrelation findings
        all_significant_lags = autocorr_results['acf_results']['significant_lags']
        periods_validated_lags = autocorr_results['acf_results'].get('periods_validated_lags', [])
        
        # Theory-driven insights
        theory_params = autocorr_results.get('theory_driven_params', {})
        sarima_suggestions = autocorr_results['sarima_suggestions']

        # Generate synthesis report
        synthesis_report = {
            'stationarity_conclusion': stationarity_type,
            'differencing_needed': not is_stationary,
            'significant_lags': [lag for lag, _ in all_significant_lags[:5]],
            'all_significant_lags_count': len(all_significant_lags),
            'theory_driven_approach': {  
                'periods_available': theory_params.get('has_periods', False),
                'methodology': sarima_suggestions.get('methodology', 'traditional'),
                'periods_validated_lags_count': len(periods_validated_lags),
                'representative_periods': theory_params.get('representative_periods', []),
                'max_lags_used': autocorr_results.get('max_lags_used', 100),
                'parameter_source': theory_params.get('parameter_source', 'default')
            },
            'seasonal_lags_detected': sarima_suggestions['all_seasonal_periods_detected'],
            'model_recommendations': []
        }

        # Model recommendations
        if is_stationary:
            synthesis_report['model_recommendations'].append("ARMA models suitable")
        else:
            synthesis_report['model_recommendations'].append("ARIMA models recommended (d=1)")

        # Theory-driven seasonal recommendations
        if synthesis_report['seasonal_lags_detected']:
            if theory_params.get('has_periods'):
                synthesis_report['model_recommendations'].append("SARIMA models for theory-driven seasonal patterns")
                synthesis_report['model_recommendations'].append(f"Representative periods: {theory_params['representative_periods']}")
            else:
                synthesis_report['model_recommendations'].append("SARIMA models for detected seasonal patterns")
            
            synthesis_report['model_recommendations'].append(f"All seasonal periods: {synthesis_report['seasonal_lags_detected']}")

        # MSTL Quality Assessment
        if residual_results and residual_results.get('success'):
            mstl_quality = residual_results.get('quality_assessment')
            if mstl_quality:
                synthesis_report['mstl_decomposition_quality'] = mstl_quality

                if mstl_quality['overall_quality'] == 'Good':
                    synthesis_report['model_recommendations'].append("MSTL decomposition captured most patterns")
                else:
                    synthesis_report['model_recommendations'].append("Additional AR/MA terms may be needed")

        # Theory-driven insights
        if theory_params.get('has_periods'):
            synthesis_report['model_recommendations'].append(f"Theory-driven analysis: {len(periods_validated_lags)} lags validated against representative periods")

        # Print synthesis
        print(f"   📊 PHÁT HIỆN CHÍNH:")
        print(f"      🔹 Tính dừng: {stationarity_type}")
        print(f"      🔹 Cần sai phân: {'Có' if synthesis_report['differencing_needed'] else 'Không'}")
        print(f"      🔹 Methodology: {synthesis_report['theory_driven_approach']['methodology']}")
        print(f"      🔹 Lags quan trọng (top 5): {synthesis_report['significant_lags']}")
        print(f"      🔹 Theory-validated lags: {synthesis_report['theory_driven_approach']['periods_validated_lags_count']}")
        print(f"      🔹 Max lags used: {synthesis_report['theory_driven_approach']['max_lags_used']} ({synthesis_report['theory_driven_approach']['parameter_source']})")

        if theory_params.get('has_periods'):
            print(f"      🎯 Representative periods: {theory_params['representative_periods']}")

        print(f"\n   💡 THEORY-DRIVEN MODEL RECOMMENDATIONS:")
        for rec in synthesis_report['model_recommendations']:
            print(f"      • {rec}")

        return synthesis_report

    def analyze(self) -> Dict[str, Any]:
        """        
        Main workflow: Stationarity → Autocorrelation → Residual → Synthesis
        """
        print("\n" + "="*80)
        print("🔍 PHẦN 3: THEORY-DRIVEN STATIONARITY & AUTOCORRELATION DIAGNOSTICS")
        print("="*80)
        print("🎯 Mục tiêu: Theory-driven chẩn đoán với representative periods")

        # Prepare time series data
        ts_data = self.df.set_index(self.date_col).sort_index()
        target_ts = ts_data[self.target_col].dropna()

        if len(target_ts) < 50:
            print("   ⚠️ Insufficient data for comprehensive diagnostics")
            return {'success': False, 'error': 'Insufficient data'}

        # Step 1: Stationarity Diagnostics (unchanged)
        stationarity_results = self._stationarity_diagnostics(target_ts)

        # Step 2: Autocorrelation Analysis (with theory-driven periods)
        autocorr_results = self._autocorrelation_analysis(target_ts)

        # Step 3: Residual Diagnostics (unchanged)
        residual_results = self._residual_diagnostics() if self.mstl_results else None

        # Step 4: Comprehensive Synthesis (with theory-driven approach)
        synthesis = self._comprehensive_synthesis(stationarity_results, autocorr_results, residual_results)

        # Combine all results
        results = {
            'stationarity': stationarity_results,
            'autocorrelation': autocorr_results,
            'residual_diagnostics': residual_results,
            'synthesis': synthesis,
            'theory_driven_approach': autocorr_results.get('theory_driven_params', {}), 
            'component_name': 'StationarityAutocorrelationAnalyzer_Theory_Driven',
            'success': True
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

        residual_analysis_results = self.diagnose_residuals(residual)
        
        if residual_analysis_results['success']:
            # Vẽ biểu đồ từ kết quả
            self.plot_residual_diagnostics(residual_analysis_results)
            
            # Đánh giá chất lượng dựa trên kết quả có sẵn
            quality_assessment = self._assess_mstl_quality(residual_analysis_results)
            residual_analysis_results['quality_assessment'] = quality_assessment
            
            
        return residual_analysis_results

    def diagnose_residuals(self,
                          residual_series: pd.Series,
                          ljung_box_lags: int = 10,
                          acf_lags: int = 40,
                          normality_tests: bool = True) -> Dict[str, Any]:
        """
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
        
        significant_residual_lags = 0
        if ljung_box and isinstance(ljung_box, dict):
            if 'significant_lags' in ljung_box:
                significant_residual_lags = ljung_box['significant_lags']
            elif 'test_results' in ljung_box:
                # Fallback: count significant p-values < 0.05
                test_results = ljung_box['test_results']
                if hasattr(test_results, 'lb_pvalue'):
                    significant_residual_lags = (test_results['lb_pvalue'] < 0.05).sum()

        # Quality assessment logic
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
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_stationarity_autocorrelation(df: pd.DataFrame,
                                        target_col: str = 'Lượng mưa',
                                        date_col: str = 'Ngày',
                                        mstl_results: Optional[Dict] = None,
                                        representative_periods: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Convenience function for complete stationarity and autocorrelation analysis
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        date_col: Date column name
        mstl_results: Optional MSTL decomposition results
        representative_periods: User-curated list of periods [7, 30, 122, 365]
        
    Returns:
        Dict containing all analysis results
    """
    analyzer = StationarityAutocorrelationAnalyzer(df, target_col, date_col, mstl_results, representative_periods)
    return analyzer.analyze()
