"""## Stationarity & Autocorrelation"""

# =============================================================================
# COMPONENT 3: STATIONARITY & AUTOCORRELATION DIAGNOSTICS (COMBINED)
# =============================================================================

class StationarityAutocorrelationAnalyzer(BaseAnalyzer):
    """
    Phần 3: Chẩn đoán Tính dừng và Tự tương quan
    Kết hợp StationarityAnalyzer + AutocorrelationAnalyzer
    """

    def __init__(self, df, target_col='Lượng mưa', date_col='Ngày', mstl_results=None):
        """Initialize with optional MSTL results from Part 2"""
        super().__init__(df, target_col, date_col)
        self.mstl_results = mstl_results

    def analyze(self):
        """
        Main workflow: Stationarity → Autocorrelation → Diagnostics
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
            'component_name': 'StationarityAutocorrelationAnalyzer'
        }

        return results

    def _stationarity_diagnostics(self, target_ts):
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

    def _autocorrelation_analysis(self, target_ts):
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

    def _residual_diagnostics(self):
        """Bước 2b: Chẩn đoán Residual từ MSTL"""
        if not self.mstl_results or 'residual' not in self.mstl_results:
            print(f"\n⚠️ MSTL results not available for residual diagnostics")
            return {'success': False, 'error': 'No MSTL results'}

        print(f"\n🧪 BƯỚC 2B: CHẨN ĐOÁN RESIDUAL MSTL")
        print("="*60)

        residual = self.mstl_results['residual'].dropna()

        if len(residual) < 50:
            return {'success': False, 'error': 'Insufficient residual data'}

        # ACF/PACF of residuals
        residual_acf = self._perform_acf_analysis(residual, "MSTL Residual")
        residual_pacf = self._perform_pacf_analysis(residual, "MSTL Residual")

        # Ljung-Box Test for residual randomness
        ljung_box_results = self._ljung_box_test(residual)

        # Residual visualization
        self._residual_visualization(residual)

        return {
            'residual_acf': residual_acf,
            'residual_pacf': residual_pacf,
            'ljung_box': ljung_box_results,
            'quality_assessment': self._assess_mstl_quality(residual_acf, ljung_box_results),
            'success': True
        }

    def _comprehensive_synthesis(self, stationarity_results, autocorr_results, residual_results):
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
        if residual_results and residual_results['success']:
            mstl_quality = residual_results['quality_assessment']
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

    # Helper methods from original components
    def _perform_adf_test(self, ts):
        """ADF Test implementation"""
        try:
            from statsmodels.tsa.stattools import adfuller
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

    def _perform_kpss_test(self, ts):
        """KPSS Test implementation"""
        try:
            from statsmodels.tsa.stattools import kpss
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

    def _stationarity_assessment(self, adf, kpss):
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

    def _perform_acf_analysis(self, ts, label):
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

    def _perform_pacf_analysis(self, ts, label):
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

    def _generate_sarima_suggestions(self, acf_results, pacf_results):
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

    def _ljung_box_test(self, residual):
        """Ljung-Box test for residual randomness"""
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

    def _assess_mstl_quality(self, residual_acf, ljung_box):
        """Assess MSTL decomposition quality"""
        if not (residual_acf['success'] and ljung_box['success']):
            return {'overall_quality': 'Unknown', 'reasons': ['Analysis failed']}

        significant_residual_lags = len(residual_acf['significant_lags'])
        is_random = ljung_box['is_random']

        if is_random and significant_residual_lags < 5:
            quality = 'Good'
            reasons = ['Residuals appear random', 'Few significant autocorrelations']
        elif is_random or significant_residual_lags < 10:
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
            'residuals_random': is_random
        }

    def _visual_stationarity_analysis(self, target_ts):
        """Visual stationarity analysis"""
        window = min(365, len(target_ts) // 4)
        rolling_mean = target_ts.rolling(window=window).mean()
        rolling_std = target_ts.rolling(window=window).std()

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Stationarity Visual Diagnostics', fontsize=14, fontweight='bold')

        # Original + Rolling Mean
        axes[0,0].plot(target_ts.index, target_ts.values, 'b-', alpha=0.6, label='Original')
        axes[0,0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'Rolling Mean ({window}d)')
        axes[0,0].set_title('Time Series with Rolling Mean')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Rolling Std
        axes[0,1].plot(rolling_std.index, rolling_std.values, 'g-', linewidth=2)
        axes[0,1].set_title(f'Rolling Standard Deviation ({window}d)')
        axes[0,1].grid(True, alpha=0.3)

        # First Difference
        first_diff = target_ts.diff().dropna()
        axes[1,0].plot(first_diff.index, first_diff.values, 'purple', alpha=0.7)
        axes[1,0].set_title('First Difference')
        axes[1,0].grid(True, alpha=0.3)

        # Distribution Comparison
        axes[1,1].hist(target_ts.values, bins=30, alpha=0.7, label='Original', density=True)
        axes[1,1].hist(first_diff.values, bins=30, alpha=0.7, label='First Difference', density=True)
        axes[1,1].set_title('Distribution Comparison')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _autocorrelation_visualization(self, ts, label):
        """ACF/PACF visualization"""
        max_lags = min(40, len(ts) // 4)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f'Autocorrelation Analysis - {label}', fontsize=14, fontweight='bold')

        # ACF plot
        try:
            plot_acf(ts, lags=max_lags, ax=axes[0,0], alpha=0.05)
            axes[0,0].set_title(f'ACF - {label}')
            axes[0,0].grid(True, alpha=0.3)
        except:
            axes[0,0].text(0.5, 0.5, 'ACF Plot Error', ha='center', va='center', transform=axes[0,0].transAxes)

        # PACF plot
        try:
            plot_pacf(ts, lags=max_lags, ax=axes[0,1], alpha=0.05)
            axes[0,1].set_title(f'PACF - {label}')
            axes[0,1].grid(True, alpha=0.3)
        except:
            axes[0,1].text(0.5, 0.5, 'PACF Plot Error', ha='center', va='center', transform=axes[0,1].transAxes)

        # First difference ACF
        first_diff = ts.diff().dropna()
        if len(first_diff) > max_lags:
            try:
                plot_acf(first_diff, lags=max_lags, ax=axes[1,0], alpha=0.05)
                axes[1,0].set_title('ACF - First Difference')
                axes[1,0].grid(True, alpha=0.3)
            except:
                axes[1,0].text(0.5, 0.5, 'ACF Diff Error', ha='center', va='center', transform=axes[1,0].transAxes)

        # First difference PACF
        if len(first_diff) > max_lags:
            try:
                plot_pacf(first_diff, lags=max_lags, ax=axes[1,1], alpha=0.05)
                axes[1,1].set_title('PACF - First Difference')
                axes[1,1].grid(True, alpha=0.3)
            except:
                axes[1,1].text(0.5, 0.5, 'PACF Diff Error', ha='center', va='center', transform=axes[1,1].transAxes)

        plt.tight_layout()
        plt.show()

    def _residual_visualization(self, residual):
        """Residual diagnostics visualization"""
        max_lags = min(40, len(residual) // 4)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('MSTL Residual Diagnostics', fontsize=14, fontweight='bold')

        # Residual time series
        axes[0,0].plot(residual.index, residual.values, 'b-', alpha=0.7)
        axes[0,0].set_title('MSTL Residuals')
        axes[0,0].grid(True, alpha=0.3)

        # Residual distribution
        axes[0,1].hist(residual.values, bins=30, alpha=0.7, density=True)
        axes[0,1].set_title('Residual Distribution')
        axes[0,1].grid(True, alpha=0.3)

        # Residual ACF
        try:
            plot_acf(residual, lags=max_lags, ax=axes[1,0], alpha=0.05)
            axes[1,0].set_title('Residual ACF')
            axes[1,0].grid(True, alpha=0.3)
        except:
            axes[1,0].text(0.5, 0.5, 'ACF Error', ha='center', va='center', transform=axes[1,0].transAxes)

        # Residual PACF
        try:
            plot_pacf(residual, lags=max_lags, ax=axes[1,1], alpha=0.05)
            axes[1,1].set_title('Residual PACF')
            axes[1,1].grid(True, alpha=0.3)
        except:
            axes[1,1].text(0.5, 0.5, 'PACF Error', ha='center', va='center', transform=axes[1,1].transAxes)

        plt.tight_layout()
        plt.show()

# Initialize analyzer
analyzer = WeatherEDAAnalyzer(df_all, target_col='Lượng mưa', date_col='Ngày')

# Run Phần 3: Stationarity & Autocorrelation Diagnostics
# (Có thể truyền mstl_results từ Phần 2 nếu có)
diagnostics_results = analyzer.run_stationarity_autocorr_analysis()

# Access comprehensive results
if diagnostics_results.get('synthesis'):
    synthesis = diagnostics_results['synthesis']
    print(f"\n🎯 KẾT QUẢ TỔNG HỢP:")
    print(f"   • Tình trạng: {synthesis['stationarity_conclusion']}")
    print(f"   • Cần sai phân: {synthesis['differencing_needed']}")
    print(f"   • Lags quan trọng: {synthesis['significant_lags'][:5]}")
    print(f"   • Khuyến nghị: {synthesis['model_recommendations']}")

"""## Extreme Even Analysis"""

# =============================================================================
# COMPONENT 2: EXTREME EVENTS ANALYZER
# =============================================================================

class ExtremeEventsAnalyzer(BaseAnalyzer):
    """
    Component 2: Extreme Events Analysis
    """

    def analyze(self):
        """Run comprehensive extreme events analysis"""
        print("\n" + "="*70)
        print("⛈️ COMPONENT 2: EXTREME EVENTS ANALYSIS")
        print("="*70)

        # Run extreme events analysis
        extreme_definition = self._define_extreme_events()
        seasonal_patterns = self._seasonal_extreme_patterns()

        # Combine results
        results = {
            'extreme_definition': extreme_definition,
            'seasonal_patterns': seasonal_patterns,
            'component_name': 'ExtremeEventsAnalyzer'
        }

        return results

    def _define_extreme_events(self):
        """2.1 Define extreme events and thresholds"""
        print("🔍 2.1 Extreme Events Definition")
        print("-" * 50)

        # Define thresholds
        p95 = self.df[self.target_col].quantile(0.95)
        p99 = self.df[self.target_col].quantile(0.99)
        p99_9 = self.df[self.target_col].quantile(0.999)

        print(f"   📊 Extreme Event Thresholds:")
        print(f"      - 95th percentile: {p95:.2f}mm")
        print(f"      - 99th percentile: {p99:.2f}mm")
        print(f"      - 99.9th percentile: {p99_9:.2f}mm")

        # Count extreme events
        extreme_95 = (self.df[self.target_col] > p95).sum()
        extreme_99 = (self.df[self.target_col] > p99).sum()
        extreme_99_9 = (self.df[self.target_col] > p99_9).sum()

        print(f"\n   📈 Extreme Event Counts:")
        print(f"      - > 95th percentile: {extreme_95} events ({extreme_95/len(self.df)*100:.2f}%)")
        print(f"      - > 99th percentile: {extreme_99} events ({extreme_99/len(self.df)*100:.2f}%)")
        print(f"      - > 99.9th percentile: {extreme_99_9} events ({extreme_99_9/len(self.df)*100:.2f}%)")

        return {
            'thresholds': {'p95': p95, 'p99': p99, 'p99_9': p99_9},
            'extreme_counts': {'p95': extreme_95, 'p99': extreme_99, 'p99_9': extreme_99_9}
        }

    def _seasonal_extreme_patterns(self):
        """2.2 Seasonal patterns of extreme events"""
        print(f"\n🌪️ 2.2 Seasonal Patterns of Extreme Events")
        print("-" * 50)

        # Use 95th percentile as extreme threshold
        p95 = self.df[self.target_col].quantile(0.95)
        extreme_by_month = self.df[self.df[self.target_col] > p95].groupby('Month').size()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Extreme Events Analysis', fontsize=16, fontweight='bold')

        # Monthly extreme events
        axes[0,0].bar(extreme_by_month.index, extreme_by_month.values, color='red', alpha=0.7)
        axes[0,0].set_title(f'Extreme Events by Month (> {p95:.1f}mm)')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Number of Events')
        axes[0,0].grid(True, alpha=0.3)

        # Extreme events time series
        extreme_events = self.df[self.df[self.target_col] > p95]
        if len(extreme_events) > 0:
            axes[0,1].scatter(extreme_events[self.date_col], extreme_events[self.target_col],
                            color='red', alpha=0.7, s=30)
            axes[0,1].set_title('Extreme Events Over Time')
            axes[0,1].set_xlabel('Date')
            axes[0,1].set_ylabel('Precipitation (mm)')
            axes[0,1].grid(True, alpha=0.3)

        # Distribution of extreme events
        if len(extreme_events) > 0:
            axes[1,0].hist(extreme_events[self.target_col], bins=20, color='orange', alpha=0.7)
            axes[1,0].set_title('Distribution of Extreme Events')
            axes[1,0].set_xlabel('Precipitation (mm)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)

        # Return period analysis
        sorted_values = self.df[self.target_col].sort_values(ascending=False)
        return_periods = len(self.df) / (np.arange(1, len(sorted_values) + 1))
        axes[1,1].loglog(return_periods[:100], sorted_values.iloc[:100], 'bo-', markersize=4)
        axes[1,1].set_title('Return Period Analysis')
        axes[1,1].set_xlabel('Return Period (days)')
        axes[1,1].set_ylabel('Precipitation (mm)')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Peak extreme month
        peak_month = extreme_by_month.idxmax() if len(extreme_by_month) > 0 else None
        if peak_month:
            print(f"   📅 Peak extreme events month: {peak_month}")

        return {
            'extreme_by_month': extreme_by_month,
            'extreme_events_data': extreme_events,
            'peak_month': peak_month,
            'return_periods': return_periods[:100].tolist(),
            'sorted_values': sorted_values.iloc[:100].tolist()
        }

# Initialize the analyzer
analyzer = WeatherEDAAnalyzer(df_all, target_col='Lượng mưa', date_col='Ngày')

# Run Component 2: Extreme Events Analysis
extreme_results = analyzer.run_extreme_events_analysis()

# Access results
thresholds = extreme_results['extreme_definition']['thresholds']
extreme_counts = extreme_results['extreme_definition']['extreme_counts']
peak_month = extreme_results['seasonal_patterns']['peak_month']

print(f"P95 threshold: {thresholds['p95']:.2f}mm")
print(f"Extreme events (>P95): {extreme_counts['p95']}")
print(f"Peak extreme month: {peak_month}")

"""# Cross-Correlation & Multicollinearity Analysis"""

# =============================================================================
# ENHANCED PART B: COMPREHENSIVE CORRELATION ANALYSIS (CLEANED)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("RdYlBu_r")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class CityLevelCorrelationAnalyzer:
    """
    Enhanced Comprehensive Correlation Analysis for City-Level Weather Data
    Unified: Static, Dynamic, Lagged, Multicollinearity & Network Analysis
    """

    def __init__(self, df, target_col='Lượng mưa', date_col='Ngày'):
        """Initialize City-Level Correlation Analyzer"""
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        # Exclude non-predictive columns for city-level data
        exclude_cols = ['Vĩ độ', 'Kinh độ', 'Ngày', 'Nhóm']

        # Get numerical columns for analysis
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.analysis_cols = [col for col in self.numerical_cols if col not in exclude_cols]

        # Separate predictors from target
        self.predictor_cols = [col for col in self.analysis_cols if col != self.target_col]

        print("🔍 CITY-LEVEL ADVANCED CORRELATION ANALYZER INITIALIZED")
        print("="*70)
        print(f"   📊 Dataset Shape: {self.df.shape}")
        print(f"   🎯 Target Variable: {self.target_col}")
        print(f"   📍 Geographic Scope: Ho Chi Minh City (Single Location)")
        print(f"   🔢 Total Features: {len(self.analysis_cols)}")
        print(f"   📈 Predictor Features: {len(self.predictor_cols)}")

    def meteorological_correlation_matrix(self):
        """1. Specialized correlation matrix for meteorological variables"""
        print("\n" + "="*70)
        print("🌤️ 1. METEOROLOGICAL CORRELATION MATRIX ANALYSIS")
        print("="*70)

        # Group features by meteorological categories
        feature_groups = {
            'Temperature': [col for col in self.analysis_cols if 'Nhiệt độ' in col or 'Điểm sương' in col or 'bầu ướt' in col],
            'Humidity': [col for col in self.analysis_cols if 'Độ ẩm' in col],
            'Wind': [col for col in self.analysis_cols if 'gió' in col or 'Hướng' in col or 'Tốc độ' in col],
            'Pressure_Radiation': [col for col in self.analysis_cols if 'Áp suất' in col or 'Bức xạ' in col],
            'Precipitation': [self.target_col]
        }

        print(f"📊 Meteorological Feature Groups:")
        for group, features in feature_groups.items():
            print(f"   - {group}: {len(features)} features")

        # Calculate correlations
        correlations = {}
        correlations['pearson'] = self.df[self.analysis_cols].corr(method='pearson')
        correlations['spearman'] = self.df[self.analysis_cols].corr(method='spearman')

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Meteorological Cross-Correlation Analysis - Ho Chi Minh City',
                     fontsize=16, fontweight='bold')

        # Pearson correlation
        mask = np.triu(np.ones_like(correlations['pearson'], dtype=bool))
        sns.heatmap(correlations['pearson'], mask=mask, annot=True,
                   cmap='RdBu_r', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[0,0])
        axes[0,0].set_title('Pearson Correlation (Linear Relationships)', fontweight='bold')

        # Spearman correlation
        sns.heatmap(correlations['spearman'], mask=mask, annot=True,
                   cmap='RdBu_r', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[0,1])
        axes[0,1].set_title('Spearman Correlation (Monotonic Relationships)', fontweight='bold')

        # Target variable focus
        target_corr = correlations['pearson'][self.target_col].drop(self.target_col).sort_values(key=abs, ascending=False)
        bars = axes[1,0].barh(range(len(target_corr)), target_corr.values,
                             color=['red' if x > 0 else 'blue' for x in target_corr.values], alpha=0.7)
        axes[1,0].set_yticks(range(len(target_corr)))
        axes[1,0].set_yticklabels(target_corr.index, fontsize=9)
        axes[1,0].set_title(f'Correlations with {self.target_col}', fontweight='bold')
        axes[1,0].set_xlabel('Correlation Coefficient')
        axes[1,0].grid(True, alpha=0.3)

        # Nonlinearity detection
        diff_matrix = correlations['spearman'] - correlations['pearson']
        sns.heatmap(diff_matrix, mask=mask, annot=True,
                   cmap='RdYlGn', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[1,1])
        axes[1,1].set_title('Nonlinearity Detection (Spearman - Pearson)', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return correlations, feature_groups

    def temporal_correlation_dynamics(self):
        """
        2. UNIFIED Temporal Correlation Dynamics Analysis
        Combines: Seasonal, Rolling, Lagged Analysis
        """
        print("\n" + "="*70)
        print("📅 2. ENHANCED TEMPORAL CORRELATION DYNAMICS")
        print("="*70)

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        results = {}

        # 2.1 Seasonal Analysis
        print(f"🌤️ 2.1 Seasonal Correlation Analysis")
        print("-" * 50)

        df_temp = self.df.copy()
        df_temp['Month'] = df_temp[self.date_col].dt.month
        df_temp['Season'] = df_temp['Month'].map({
            12: 'Dry', 1: 'Dry', 2: 'Dry',
            3: 'Pre-wet', 4: 'Pre-wet', 5: 'Pre-wet',
            6: 'Wet', 7: 'Wet', 8: 'Wet',
            9: 'Post-wet', 10: 'Post-wet', 11: 'Post-wet'
        })

        seasonal_correlations = {}
        seasons = ['Dry', 'Pre-wet', 'Wet', 'Post-wet']

        for season in seasons:
            season_data = df_temp[df_temp['Season'] == season]
            if len(season_data) > 30:
                seasonal_correlations[season] = season_data[self.analysis_cols].corr()[self.target_col].drop(self.target_col)

        if seasonal_correlations:
            correlation_df = pd.DataFrame(seasonal_correlations)

            plt.figure(figsize=(16, 10))

            # Seasonal correlation heatmap
            plt.subplot(2, 2, 1)
            sns.heatmap(correlation_df.T, annot=True, cmap='RdBu_r', center=0,
                       cbar_kws={"shrink": .8}, fmt='.2f')
            plt.title('Seasonal Correlation Patterns', fontweight='bold')

            # Top predictors seasonal changes
            plt.subplot(2, 2, 2)
            top_predictors = correlation_df.abs().max(axis=1).nlargest(5).index
            for predictor in top_predictors:
                plt.plot(seasons, [correlation_df.loc[predictor, season] for season in seasons],
                        'o-', linewidth=2, label=predictor[:15], alpha=0.8)
            plt.title('Top Predictors Seasonal Changes', fontweight='bold')
            plt.xlabel('Season')
            plt.ylabel('Correlation')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

        results['seasonal_correlations'] = seasonal_correlations

        # 2.2 Rolling Correlations (Simplified)
        print(f"\n📈 2.2 Rolling Correlation Analysis")
        print("-" * 50)

        daily_agg = self.df.groupby(self.date_col)[self.analysis_cols].mean().reset_index()
        daily_agg = daily_agg.set_index(self.date_col).sort_index()
        daily_agg_clean = daily_agg.dropna()

        rolling_results = {}
        if len(daily_agg_clean) >= 60:
            static_corr = daily_agg_clean.corr()[self.target_col].sort_values(ascending=False)
            top_vars = static_corr.head(3).index.tolist()[1:3]  # Top 2 excluding target

            rolling_window = 30
            for var in top_vars:
                if var in daily_agg_clean.columns:
                    combined_df = daily_agg_clean[[self.target_col, var]].dropna()
                    rolling_corr = combined_df[self.target_col].rolling(window=rolling_window).corr(combined_df[var])
                    rolling_corr = rolling_corr.dropna()

                    if len(rolling_corr) > 0:
                        rolling_results[var] = {
                            'mean_correlation': rolling_corr.mean(),
                            'std_correlation': rolling_corr.std()
                        }

        results['rolling_correlations'] = rolling_results

        # 2.3 Lagged Correlations (Simplified)
        print(f"\n⏰ 2.3 Lagged Correlation Analysis")
        print("-" * 50)

        lagged_results = {}
        if len(daily_agg_clean) >= 60:
            max_lags = 5
            static_corr = daily_agg_clean.corr()[self.target_col].sort_values(ascending=False)
            top_vars = static_corr.head(3).index.tolist()[1:3]  # Top 2 excluding target

            for var in top_vars:
                if var in daily_agg_clean.columns:
                    correlations = []
                    lags = range(-max_lags, max_lags + 1)

                    target_series = daily_agg_clean[self.target_col]
                    var_series = daily_agg_clean[var]

                    for lag in lags:
                        if lag == 0:
                            corr = target_series.corr(var_series)
                        elif lag > 0:
                            shifted_var = var_series.shift(lag)
                            valid_idx = target_series.index.intersection(shifted_var.dropna().index)
                            corr = target_series.loc[valid_idx].corr(shifted_var.loc[valid_idx])
                        else:
                            shifted_target = target_series.shift(-lag)
                            corr = shifted_target.corr(var_series)

                        correlations.append(corr if not np.isnan(corr) else 0)

                    # Find best lag
                    abs_correlations = [abs(c) for c in correlations]
                    best_lag_idx = np.argmax(abs_correlations)
                    best_lag = lags[best_lag_idx]
                    best_corr = correlations[best_lag_idx]

                    lagged_results[var] = {
                        'best_lag': best_lag,
                        'best_correlation': best_corr
                    }

        results['lagged_correlations'] = lagged_results

        return results

    def multicollinearity_advanced_analysis(self):
        """3. Advanced multicollinearity analysis"""
        print("\n" + "="*70)
        print("🔍 3. ADVANCED MULTICOLLINEARITY ANALYSIS")
        print("="*70)

        X = self.df[self.predictor_cols].dropna()
        vif_data = []

        try:
            from statsmodels.tools.tools import add_constant
            X_with_const = add_constant(X)

            for i, col in enumerate(X.columns):
                vif_score = variance_inflation_factor(X_with_const.values, i+1)
                vif_data.append({
                    'Feature': col,
                    'VIF_Score': vif_score,
                    'Category': self._categorize_feature(col),
                    'Risk_Level': self._interpret_vif(vif_score)
                })

        except Exception as e:
            print(f"   ⚠️ VIF calculation error: {e}")
            # Alternative: correlation-based detection
            corr_matrix = X.corr()
            for col in X.columns:
                max_corr = corr_matrix[col].drop(col).abs().max()
                vif_score = 1/(1-max_corr**2) if max_corr < 0.99 else 100
                vif_data.append({
                    'Feature': col,
                    'VIF_Score': vif_score,
                    'Category': self._categorize_feature(col),
                    'Risk_Level': self._interpret_vif(vif_score)
                })

        if vif_data:
            vif_df = pd.DataFrame(vif_data).sort_values('VIF_Score', ascending=False)
            print(f"📊 Multicollinearity Analysis Results:")
            print(vif_df.head(10).to_string(index=False))

        return vif_df if vif_data else None

    def feature_interaction_network(self):
        """4. Network analysis focusing on feature interactions"""
        print("\n" + "="*70)
        print("🕸️ 4. FEATURE INTERACTION NETWORK")
        print("="*70)

        corr_matrix = self.df[self.analysis_cols].corr()
        G = nx.Graph()

        # Add nodes
        for feature in self.analysis_cols:
            category = self._categorize_feature(feature)
            G.add_node(feature, category=category, is_target=(feature == self.target_col))

        # Add edges for significant correlations
        correlation_threshold = 0.25
        for i, feature1 in enumerate(self.analysis_cols):
            for j, feature2 in enumerate(self.analysis_cols):
                if i < j:
                    corr_val = corr_matrix.loc[feature1, feature2]
                    if abs(corr_val) >= correlation_threshold:
                        G.add_edge(feature1, feature2, weight=abs(corr_val), correlation=corr_val)

        print(f"🕸️ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"   Density: {nx.density(G):.3f}")

        return G, []

    def advanced_correlation_clustering_analysis(self):
        """
        6. Advanced Correlation Analysis using Clustering and PCA
        Utilizes: scipy.cluster, sklearn.preprocessing, sklearn.decomposition
        """
        print("\n" + "="*70)
        print("🔬 6. ADVANCED CORRELATION CLUSTERING & PCA ANALYSIS")
        print("="*70)

        # Prepare data
        X = self.df[self.predictor_cols].dropna()

        # 6.1 Standardize features
        print("📊 6.1 Feature Standardization")
        print("-" * 50)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.predictor_cols, index=X.index)

        # 6.2 Hierarchical Clustering of Features
        print("🌳 6.2 Hierarchical Feature Clustering")
        print("-" * 50)

        # Calculate correlation distance matrix
        corr_matrix = X_scaled_df.corr()
        distance_matrix = 1 - np.abs(corr_matrix)
        condensed_distances = squareform(distance_matrix.values)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='ward')

        # Get clusters
        n_clusters = min(5, len(self.predictor_cols)//3)  # Adaptive cluster number
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Create feature clusters
        feature_clusters = {}
        for i, feature in enumerate(self.predictor_cols):
            cluster_id = cluster_labels[i]
            if cluster_id not in feature_clusters:
                feature_clusters[cluster_id] = []
            feature_clusters[cluster_id].append(feature)

        print(f"   📊 Features grouped into {n_clusters} clusters:")
        for cluster_id, features in feature_clusters.items():
            print(f"   Cluster {cluster_id}: {len(features)} features")

        # 6.3 PCA Analysis
        print("\n🎯 6.3 Principal Component Analysis")
        print("-" * 50)

        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)

        # Calculate correlation of PCs with target
        target_values = self.df.loc[X.index, self.target_col]
        pc_target_correlations = []

        for i in range(min(5, len(self.predictor_cols))):  # Top 5 PCs
            pc_corr = np.corrcoef(pca_result[:, i], target_values)[0, 1]
            pc_target_correlations.append({
                'PC': f'PC{i+1}',
                'Explained_Variance': pca.explained_variance_ratio_[i],
                'Target_Correlation': pc_corr
            })

        pc_df = pd.DataFrame(pc_target_correlations)
        print("   📈 Principal Components vs Target:")
        print(pc_df.to_string(index=False, float_format='%.3f'))

        # 6.4 Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Correlation Analysis: Clustering & PCA', fontsize=16, fontweight='bold')

        # Dendrogram
        dendrogram(linkage_matrix, labels=self.predictor_cols, ax=axes[0,0],
                  orientation='top', leaf_rotation=90)
        axes[0,0].set_title('Feature Hierarchical Clustering', fontweight='bold')
        axes[0,0].tick_params(axis='x', labelsize=8)

        # Clustered correlation heatmap
        cluster_order = []
        for cluster_id in sorted(feature_clusters.keys()):
            cluster_order.extend(feature_clusters[cluster_id])

        reordered_corr = corr_matrix.loc[cluster_order, cluster_order]
        sns.heatmap(reordered_corr, ax=axes[0,1], cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        axes[0,1].set_title('Clustered Correlation Matrix', fontweight='bold')
        axes[0,1].tick_params(axis='both', labelsize=8)

        # PCA explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[1,0].bar(range(1, len(cumsum_var[:10])+1), pca.explained_variance_ratio_[:10],
                      alpha=0.7, color='skyblue')
        axes[1,0].plot(range(1, len(cumsum_var[:10])+1), cumsum_var[:10],
                      'ro-', linewidth=2, markersize=6)
        axes[1,0].set_title('PCA Explained Variance', fontweight='bold')
        axes[1,0].set_xlabel('Principal Component')
        axes[1,0].set_ylabel('Explained Variance Ratio')
        axes[1,0].grid(True, alpha=0.3)

        # PC correlation with target
        pc_corrs = [abs(corr['Target_Correlation']) for corr in pc_target_correlations]
        axes[1,1].bar(range(1, len(pc_corrs)+1), pc_corrs, alpha=0.7, color='lightcoral')
        axes[1,1].set_title('Principal Components vs Target Correlation', fontweight='bold')
        axes[1,1].set_xlabel('Principal Component')
        axes[1,1].set_ylabel('|Correlation| with Target')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'feature_clusters': feature_clusters,
            'pca_results': pc_df,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'linkage_matrix': linkage_matrix
        }

    def generate_city_level_insights_report(self):
        """5. Generate comprehensive insights report"""
        print("\n" + "="*70)
        print("📋 5. ENHANCED CITY-LEVEL CORRELATION INSIGHTS REPORT")
        print("="*70)

        # Run all analyses
        correlations, feature_groups = self.meteorological_correlation_matrix()
        temporal_results = self.temporal_correlation_dynamics()
        vif_results = self.multicollinearity_advanced_analysis()
        network, edges = self.feature_interaction_network()

        # NEW: Add advanced clustering analysis
        clustering_results = self.advanced_correlation_clustering_analysis()

        # Extract results
        seasonal_correlations = temporal_results['seasonal_correlations']
        rolling_correlations = temporal_results['rolling_correlations']
        lagged_correlations = temporal_results['lagged_correlations']

        # Executive summary
        print(f"\n🎯 HO CHI MINH CITY WEATHER CORRELATION EXECUTIVE SUMMARY:")
        print("="*60)

        target_corr = correlations['pearson'][self.target_col].drop(self.target_col).abs().sort_values(ascending=False)
        strong_predictors = target_corr[target_corr > 0.3]
        moderate_predictors = target_corr[(target_corr > 0.2) & (target_corr <= 0.3)]

        print(f"🌧️ PRECIPITATION PREDICTION INSIGHTS:")
        print(f"   - Strong predictors (|r| > 0.3): {len(strong_predictors)}")
        print(f"   - Moderate predictors (0.2 < |r| ≤ 0.3): {len(moderate_predictors)}")

        if rolling_correlations:
            print(f"\n📈 DYNAMIC CORRELATION INSIGHTS:")
            for var, results in rolling_correlations.items():
                stability = "Stable" if results['std_correlation'] < 0.1 else "Variable"
                print(f"   - {var}: {stability} correlation")

        if lagged_correlations:
            print(f"\n⏰ LAGGED CORRELATION INSIGHTS:")
            for var, results in lagged_correlations.items():
                print(f"   - {var}: best at {results['best_lag']} days (r={results['best_correlation']:.3f})")

        # NEW: Add clustering insights
        print(f"\n🔬 ADVANCED CLUSTERING INSIGHTS:")
        print(f"   - Feature clusters identified: {len(clustering_results['feature_clusters'])}")
        top_pc = clustering_results['pca_results'].iloc[0]
        print(f"   - Top PC explains {top_pc['Explained_Variance']:.1%} variance")
        print(f"   - Top PC correlation with target: {abs(top_pc['Target_Correlation']):.3f}")

        return {
            'correlations': correlations,
            'feature_groups': feature_groups,
            'seasonal_correlations': seasonal_correlations,
            'rolling_correlations': rolling_correlations,
            'lagged_correlations': lagged_correlations,
            'vif_results': vif_results,
            'network': network,
            'strong_predictors': strong_predictors,
            'moderate_predictors': moderate_predictors,
            'clustering_results': clustering_results  # NEW
        }

    def _categorize_feature(self, feature_name):
        """Helper function to categorize meteorological features"""
        if any(term in feature_name for term in ['Nhiệt độ', 'Điểm sương', 'bầu ướt']):
            return 'Temperature'
        elif 'Độ ẩm' in feature_name:
            return 'Humidity'
        elif any(term in feature_name for term in ['gió', 'Hướng', 'Tốc độ']):
            return 'Wind'
        elif any(term in feature_name for term in ['Áp suất', 'Bức xạ']):
            return 'Pressure_Radiation'
        elif feature_name == self.target_col:
            return 'Precipitation'
        else:
            return 'Other'

    def _interpret_vif(self, vif_score):
        """Helper function to interpret VIF scores"""
        if vif_score < 5:
            return "Low"
        elif vif_score < 10:
            return "Moderate"
        else:
            return "High"

# =============================================================================
# EXECUTION
# =============================================================================

def run_city_level_correlation_analysis(df, target_col='Lượng mưa', date_col='Ngày'):
    """Run comprehensive correlation analysis for city-level data"""
    print("🚀 STARTING CLEANED CITY-LEVEL CORRELATION ANALYSIS")
    print("="*80)

    analyzer = CityLevelCorrelationAnalyzer(df, target_col, date_col)
    results = analyzer.generate_city_level_insights_report()

    print("\n✅ CLEANED ANALYSIS COMPLETED")
    return results

# =============================================================================
# RUN CITY-LEVEL ANALYSIS
# =============================================================================

# Chạy phân tích correlation cho dữ liệu TP.HCM
city_correlation_results = run_city_level_correlation_analysis(
    df_all,
    target_col='Lượng mưa',
    date_col='Ngày'
)

print(f"\n🎉 ENHANCED HO CHI MINH CITY CORRELATION ANALYSIS SUMMARY:")
print(f"   💪 Strong Predictors: {len(city_correlation_results['strong_predictors'])}")
print(f"   📊 Moderate Predictors: {len(city_correlation_results['moderate_predictors'])}")
print(f"   🌤️ Meteorological Categories: {len(city_correlation_results['feature_groups'])}")
print(f"   📈 Dynamic Correlations: {len(city_correlation_results['rolling_correlations'])}")  # NEW
print(f"   ⏰ Lagged Correlations: {len(city_correlation_results['lagged_correlations'])}")    # NEW
print(f"   🕸️ Feature Network Density: {nx.density(city_correlation_results['network']):.3f}")

"""# Feature Engineering

## Feature Selection
"""

selected_features = [
    'Nhiệt độ tối đa 2m',      # T2M_MAX
    'Nhiệt độ tối thiểu 2m',    # T2M_MIN
    'Độ ẩm tương đối 2m',       # RH2M
    'Độ ẩm đất bề mặt',         # GWETTOP
    'Hướng gió 10m',            # WD10M
    'Áp suất bề mặt',           # PS
    'Bức xạ sóng dài xuống'     # ALLSKY_SFC_LW_DWN
]

# Keep essential columns (date, target) + selected features
essential_columns = ['Ngày', 'Lượng mưa']  # Date and target variable
final_columns = essential_columns + selected_features

# Filter dataset
df_selected = df_all[final_columns].copy()

print(f"📊 Feature Selection completed:")
print(f"   Original shape: {df_all.shape}")
print(f"   Selected shape: {df_selected.shape}")
print(f"   Features selected: {len(selected_features)}")
print(f"   Selected features: {selected_features}")

"""## Temporal feature"""

import numpy as np
import pandas as pd

# Convert date column to datetime if not already
df_selected['Ngày'] = pd.to_datetime(df_selected['Ngày'])

# Extract basic time features
df_selected['Year'] = df_selected['Ngày'].dt.year
df_selected['Month'] = df_selected['Ngày'].dt.month
df_selected['DayofMonth'] = df_selected['Ngày'].dt.day
df_selected['DayofYear'] = df_selected['Ngày'].dt.dayofyear

# Create cyclical features for Month (1-12)
df_selected['Month_sin'] = np.sin(2 * np.pi * df_selected['Month'] / 12)
df_selected['Month_cos'] = np.cos(2 * np.pi * df_selected['Month'] / 12)

# Create cyclical features for Day of Year (1-365/366)
df_selected['DayOfYear_sin'] = np.sin(2 * np.pi * df_selected['DayofYear'] / 365.25)
df_selected['DayOfYear_cos'] = np.cos(2 * np.pi * df_selected['DayofYear'] / 365.25)

# Create Wet Season feature (May-November = months 5,6,7,8,9,10,11)
df_selected['Is_Wet_Season'] = df_selected['Month'].isin([5, 6, 7, 8, 9, 10, 11]).astype(int)

print("🕒 Time-based Feature Engineering completed:")
print(f"   Dataset shape: {df_selected.shape}")
print(f"   New time features added: 8")
print(f"   Wet season months (value=1): May-November")
print(f"   Dry season months (value=0): December-April")

# Display sample of new features
print("\n📊 Sample of new time features:")
print(df_selected[['Ngày', 'Year', 'Month', 'DayofMonth', 'DayofYear',
                   'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                   'Is_Wet_Season']].head(10))

# Check wet season distribution
print(f"\n🌧️ Wet Season Distribution:")
print(df_selected['Is_Wet_Season'].value_counts())

"""## Lag and Rolling window Engineering

"""

# ==============================================================================
# LAG FEATURES & ROLLING WINDOW FEATURES
# ==============================================================================

# A. TARGET VARIABLE LAG & ROLLING FEATURES (HIGHEST PRIORITY)
print("🎯 Creating Target Variable Features (Lượng mưa)...")

# Lag features for rainfall (1, 2, 3, 4, 5, 7 days)
for lag in [1, 2, 3, 4, 5, 7]:
    df_selected[f'Rainfall_lag_{lag}'] = df_selected['Lượng mưa'].shift(lag)

# Rolling sum features (cumulative rainfall)
for window in [3, 7, 14, 30]:
    df_selected[f'Rainfall_sum_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).sum()

# Rolling statistical features
for window in [7, 14]:
    df_selected[f'Rainfall_mean_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).mean()
    df_selected[f'Rainfall_max_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).max()
    df_selected[f'Rainfall_std_{window}d'] = df_selected['Lượng mưa'].rolling(window=window).std()

# B. IMPORTANT PREDICTOR VARIABLES LAG & ROLLING FEATURES
print("🌡️ Creating Temperature Features...")

# Temperature features (T2M_MAX, T2M_MIN)
temp_vars = ['Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m']
for var in temp_vars:
    # Lag 1
    df_selected[f'{var}_lag_1'] = df_selected[var].shift(1)

    # Rolling statistics
    for window in [3, 7, 14]:
        df_selected[f'{var}_mean_{window}d'] = df_selected[var].rolling(window=window).mean()
        df_selected[f'{var}_max_{window}d'] = df_selected[var].rolling(window=window).max()
        df_selected[f'{var}_min_{window}d'] = df_selected[var].rolling(window=window).min()
        df_selected[f'{var}_std_{window}d'] = df_selected[var].rolling(window=window).std()

print("💧 Creating Humidity Features...")

# Humidity features (RH2M, GWETTOP)
humidity_vars = ['Độ ẩm tương đối 2m', 'Độ ẩm đất bề mặt']
for var in humidity_vars:
    # Lag 1
    df_selected[f'{var}_lag_1'] = df_selected[var].shift(1)

    # Rolling statistics
    for window in [3, 7, 14]:
        df_selected[f'{var}_mean_{window}d'] = df_selected[var].rolling(window=window).mean()
        df_selected[f'{var}_std_{window}d'] = df_selected[var].rolling(window=window).std()

print("✅ Lag & Rolling Window Features completed:")
print(f"   Final dataset shape: {df_selected.shape}")
print(f"   Target lag features: 6")
print(f"   Target rolling features: 10")
print(f"   Temperature features: 24")
print(f"   Humidity features: 12")
print(f"   Total new features: 52")

# Check for missing values (expected due to lag/rolling)
missing_count = df_selected.isnull().sum().sum()
print(f"   Missing values created: {missing_count} (due to lag/rolling operations)")

# ==============================================================================
# CHECK MISSING VALUES BY DATE
# ==============================================================================

# Find rows with any missing values
missing_rows = df_selected.isnull().any(axis=1)
missing_dates = df_selected[missing_rows]['Ngày']

print("📅 Dates with Missing Values:")
print(f"   Total dates with missing values: {len(missing_dates)}")
print(f"   Date range: {missing_dates.min()} to {missing_dates.max()}")

print("\n🗓️ First 20 dates with missing values:")
print(missing_dates.head(20).tolist())

if len(missing_dates) > 20:
    print(f"\n... and {len(missing_dates) - 20} more dates")

# Check which features have the most missing values
print(f"\n🔍 Missing values by feature (top 10):")
missing_by_feature = df_selected.isnull().sum().sort_values(ascending=False)
print(missing_by_feature[missing_by_feature > 0].head(10))

# ==============================================================================
# DROP MISSING VALUES
# ==============================================================================

# Store original shape for comparison
original_shape = df_selected.shape

# Drop rows with any missing values
df_selected = df_selected.dropna()

# Reset index after dropping rows
df_selected = df_selected.reset_index(drop=True)

print("🗑️ Missing Values Removal completed:")
print(f"   Original shape: {original_shape}")
print(f"   Final shape: {df_selected.shape}")
print(f"   Rows dropped: {original_shape[0] - df_selected.shape[0]}")
print(f"   Remaining data: {df_selected.shape[0]} days")
print(f"   Missing values remaining: {df_selected.isnull().sum().sum()}")

# Check date range of final dataset
print(f"\n📅 Final dataset date range:")
print(f"   Start date: {df_selected['Ngày'].min()}")
print(f"   End date: {df_selected['Ngày'].max()}")

"""## Multicolinear and cross-correlation checking for new feature"""

# ==============================================================================
# MULTICOLLINEARITY & CROSS-CORRELATION CHECKING (4 PARTS)
# ==============================================================================

# Define feature groups
print("🔍 Defining Feature Groups for Correlation Analysis...")

# 1. Original selected features (after feature selection)
original_features = ['Ngày', 'Lượng mưa'] + [
    'Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m', 'Độ ẩm tương đối 2m',
    'Độ ẩm đất bề mặt', 'Hướng gió 10m', 'Áp suất bề mặt', 'Bức xạ sóng dài xuống'
]

# 2. Temporal features
temporal_features = ['Ngày', 'Lượng mưa'] + [
    'Year', 'Month', 'DayofMonth', 'DayofYear',
    'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Is_Wet_Season'
]

# 3. Lag features
lag_columns = ['Ngày', 'Lượng mưa'] + [col for col in df_selected.columns if '_lag_' in col]

# 4. Rolling window features
rolling_columns = ['Ngày', 'Lượng mưa'] + [col for col in df_selected.columns
                   if any(pattern in col for pattern in ['_sum_', '_mean_', '_max_', '_std_'])
                   and col.endswith('d')]

print(f"📊 Feature Groups Summary:")
print(f"   Original features: {len(original_features)-2}")
print(f"   Temporal features: {len(temporal_features)-2}")
print(f"   Lag features: {len(lag_columns)-2}")
print(f"   Rolling features: {len(rolling_columns)-2}")

# Part 1: Original Features Analysis
print("\n" + "="*50)
print("PART 1: ORIGINAL FEATURES CORRELATION ANALYSIS")
print("="*50)
df_original = df_selected[original_features].copy()
original_results = run_city_level_correlation_analysis(
    df_original,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 2: Temporal Features Analysis
print("\n" + "="*50)
print("PART 2: TEMPORAL FEATURES CORRELATION ANALYSIS")
print("="*50)
df_temporal = df_selected[temporal_features].copy()
temporal_results = run_city_level_correlation_analysis(
    df_temporal,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 3: Lag Features Analysis
print("\n" + "="*50)
print("PART 3: LAG FEATURES CORRELATION ANALYSIS")
print("="*50)
df_lag = df_selected[lag_columns].copy()
lag_results = run_city_level_correlation_analysis(
    df_lag,
    target_col='Lượng mưa',
    date_col='Ngày'
)

# Part 4: Rolling Window Features Analysis
print("\n" + "="*50)
print("PART 4: ROLLING WINDOW FEATURES CORRELATION ANALYSIS")
print("="*50)
df_rolling = df_selected[rolling_columns].copy()
rolling_results = run_city_level_correlation_analysis(
    df_rolling,
    target_col='Lượng mưa',
    date_col='Ngày'
)

print("\n✅ All correlation analyses completed!")

"""# Modeling

## Tree-based model
"""

!pip install -q optuna

# ==============================================================================
# LIGHTGBM TWO-STAGE MODEL WITH OPTUNA OPTIMIZATION (FIXED)
# ==============================================================================

import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

print("🚀 LIGHTGBM TWO-STAGE MODEL EXPERIMENTATION (FIXED)")
print("="*60)

print(f"📊 Data Setup:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Total samples: {len(X)}")
print(f"   Rain days: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
print(f"   No rain days: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")

# Fixed Optuna optimization functions
def objective_classification_fixed(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1
    }

    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=100)  # Removed verbose_eval

        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def objective_regression_fixed(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1
    }

    mae_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        rain_mask_train = y_clf.iloc[train_idx] == 1
        rain_mask_test = y_clf.iloc[test_idx] == 1

        if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
            X_train_rain = X_train[rain_mask_train]
            y_train_rain = y_train[rain_mask_train]
            X_test_rain = X_test[rain_mask_test]
            y_test_rain = y_test[rain_mask_test]

            train_data = lgb.Dataset(X_train_rain, label=y_train_rain)
            model = lgb.train(params, train_data, num_boost_round=100)  # Removed verbose_eval

            y_pred = model.predict(X_test_rain)
            mae = mean_absolute_error(y_test_rain, y_pred)
            mae_scores.append(mae)

    return np.mean(mae_scores) if mae_scores else float('inf')

# Optimize Classification Model
print("\n🔍 Optimizing Classification Model...")
study_clf_fixed = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_clf_fixed.optimize(objective_classification_fixed, n_trials=20)
best_params_clf_fixed = study_clf_fixed.best_params
print(f"   Best AUC: {study_clf_fixed.best_value:.4f}")

# Optimize Regression Model
print("\n🔍 Optimizing Regression Model...")
study_reg_fixed = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_reg_fixed.optimize(objective_regression_fixed, n_trials=20)
best_params_reg_fixed = study_reg_fixed.best_params
print(f"   Best MAE: {study_reg_fixed.best_value:.4f}")

# Final evaluation with best parameters
print("\n📊 Final Evaluation with Best Parameters:")
clf_results_fixed = {'AUC': []}
reg_results_fixed = {'MAE': [], 'RMSE': [], 'R2': []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n🔄 Fold {fold + 1}/3")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # Classification Model
    best_params_clf_fixed.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})
    train_data_clf = lgb.Dataset(X_train, label=y_train_clf)
    clf_model = lgb.train(best_params_clf_fixed, train_data_clf, num_boost_round=100)

    clf_pred = clf_model.predict(X_test)
    auc = roc_auc_score(y_test_clf, clf_pred)
    clf_results_fixed['AUC'].append(auc)

    # Regression Model
    rain_mask_train = y_train_clf == 1
    rain_mask_test = y_test_clf == 1

    if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
        X_train_rain = X_train[rain_mask_train]
        y_train_rain = y_train_reg[rain_mask_train]
        X_test_rain = X_test[rain_mask_test]
        y_test_rain = y_test_reg[rain_mask_test]

        best_params_reg_fixed.update({'objective': 'regression', 'metric': 'mae', 'verbosity': -1})
        train_data_reg = lgb.Dataset(X_train_rain, label=y_train_rain)
        reg_model = lgb.train(best_params_reg_fixed, train_data_reg, num_boost_round=100)

        reg_pred = reg_model.predict(X_test_rain)
        mae = mean_absolute_error(y_test_rain, reg_pred)
        rmse = np.sqrt(mean_squared_error(y_test_rain, reg_pred))
        r2 = r2_score(y_test_rain, reg_pred)

        reg_results_fixed['MAE'].append(mae)
        reg_results_fixed['RMSE'].append(rmse)
        reg_results_fixed['R2'].append(r2)

    print(f"   Classification AUC: {auc:.4f}")
    if rain_mask_test.sum() > 0:
        print(f"   Regression MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Summary Results
print("\n" + "="*60)
print("📋 LIGHTGBM FINAL RESULTS SUMMARY (FIXED)")
print("="*60)
print(f"Classification Model:")
print(f"   AUC: {np.mean(clf_results_fixed['AUC']):.4f} ± {np.std(clf_results_fixed['AUC']):.4f}")

if reg_results_fixed['MAE']:
    print(f"\nRegression Model (on rainy days only):")
    print(f"   MAE:  {np.mean(reg_results_fixed['MAE']):.4f} ± {np.std(reg_results_fixed['MAE']):.4f}")
    print(f"   RMSE: {np.mean(reg_results_fixed['RMSE']):.4f} ± {np.std(reg_results_fixed['RMSE']):.4f}")
    print(f"   R2:   {np.mean(reg_results_fixed['R2']):.4f} ± {np.std(reg_results_fixed['R2']):.4f}")

# ==============================================================================
# XGBOOST TWO-STAGE MODEL WITH OPTUNA OPTIMIZATION
# ==============================================================================

import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

print("🚀 XGBOOST TWO-STAGE MODEL EXPERIMENTATION")
print("="*60)

# Use same data setup as LightGBM
print(f"📊 Data Setup:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Total samples: {len(X)}")
print(f"   Rain days: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
print(f"   No rain days: {(1-y_clf).sum()} ({(1-y_clf.mean())*100:.1f}%)")

# Optuna optimization functions for XGBoost
def objective_classification_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'verbosity': 0,
        'random_state': 42
    }

    auc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def objective_regression_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'verbosity': 0,
        'random_state': 42
    }

    mae_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        # Only use rainy days for regression training
        rain_mask_train = y_clf.iloc[train_idx] == 1
        rain_mask_test = y_clf.iloc[test_idx] == 1

        if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
            X_train_rain = X_train[rain_mask_train]
            y_train_rain = y_train[rain_mask_train]
            X_test_rain = X_test[rain_mask_test]
            y_test_rain = y_test[rain_mask_test]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_rain, y_train_rain, verbose=False)

            y_pred = model.predict(X_test_rain)
            mae = mean_absolute_error(y_test_rain, y_pred)
            mae_scores.append(mae)

    return np.mean(mae_scores) if mae_scores else float('inf')

# Optimize Classification Model
print("\n🔍 Optimizing XGBoost Classification Model...")
study_clf_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_clf_xgb.optimize(objective_classification_xgb, n_trials=20)
best_params_clf_xgb = study_clf_xgb.best_params
print(f"   Best AUC: {study_clf_xgb.best_value:.4f}")

# Optimize Regression Model
print("\n🔍 Optimizing XGBoost Regression Model...")
study_reg_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_reg_xgb.optimize(objective_regression_xgb, n_trials=20)
best_params_reg_xgb = study_reg_xgb.best_params
print(f"   Best MAE: {study_reg_xgb.best_value:.4f}")

# Final evaluation with best parameters
print("\n📊 Final Evaluation with Best Parameters:")
clf_results_xgb = {'AUC': []}
reg_results_xgb = {'MAE': [], 'RMSE': [], 'R2': []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n🔄 Fold {fold + 1}/3")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_clf, y_test_clf = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # Classification Model
    best_params_clf_xgb.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'random_state': 42
    })
    clf_model_xgb = xgb.XGBClassifier(**best_params_clf_xgb)
    clf_model_xgb.fit(X_train, y_train_clf, verbose=False)

    # Classification predictions
    clf_pred_xgb = clf_model_xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_clf, clf_pred_xgb)
    clf_results_xgb['AUC'].append(auc)

    # Regression Model (only on rainy days)
    rain_mask_train = y_train_clf == 1
    rain_mask_test = y_test_clf == 1

    if rain_mask_train.sum() > 0 and rain_mask_test.sum() > 0:
        X_train_rain = X_train[rain_mask_train]
        y_train_rain = y_train_reg[rain_mask_train]
        X_test_rain = X_test[rain_mask_test]
        y_test_rain = y_test_reg[rain_mask_test]

        best_params_reg_xgb.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
            'random_state': 42
        })
        reg_model_xgb = xgb.XGBRegressor(**best_params_reg_xgb)
        reg_model_xgb.fit(X_train_rain, y_train_rain, verbose=False)

        # Regression predictions
        reg_pred_xgb = reg_model_xgb.predict(X_test_rain)
        mae = mean_absolute_error(y_test_rain, reg_pred_xgb)
        rmse = np.sqrt(mean_squared_error(y_test_rain, reg_pred_xgb))
        r2 = r2_score(y_test_rain, reg_pred_xgb)

        reg_results_xgb['MAE'].append(mae)
        reg_results_xgb['RMSE'].append(rmse)
        reg_results_xgb['R2'].append(r2)

    print(f"   Classification AUC: {auc:.4f}")
    if rain_mask_test.sum() > 0:
        print(f"   Regression MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Summary Results
print("\n" + "="*60)
print("📋 XGBOOST FINAL RESULTS SUMMARY")
print("="*60)
print(f"Classification Model:")
print(f"   AUC: {np.mean(clf_results_xgb['AUC']):.4f} ± {np.std(clf_results_xgb['AUC']):.4f}")

if reg_results_xgb['MAE']:
    print(f"\nRegression Model (on rainy days only):")
    print(f"   MAE:  {np.mean(reg_results_xgb['MAE']):.4f} ± {np.std(reg_results_xgb['MAE']):.4f}")
    print(f"   RMSE: {np.mean(reg_results_xgb['RMSE']):.4f} ± {np.std(reg_results_xgb['RMSE']):.4f}")
    print(f"   R2:   {np.mean(reg_results_xgb['R2']):.4f} ± {np.std(reg_results_xgb['R2']):.4f}")

"""## Times series model"""

# ==============================================================================
# STATIONARITY TESTING FOR EXOGENOUS FEATURES (ADF & KPSS)
# ==============================================================================

from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Define exogenous features (original selected features)
exogenous_features = [
    'Nhiệt độ tối đa 2m',      # T2M_MAX
    'Nhiệt độ tối thiểu 2m',    # T2M_MIN
    'Độ ẩm tương đối 2m',       # RH2M
    'Độ ẩm đất bề mặt',         # GWETTOP
    'Hướng gió 10m',            # WD10M
    'Áp suất bề mặt',           # PS
    'Bức xạ sóng dài xuống'     # ALLSKY_SFC_LW_DWN
]

print("📊 STATIONARITY TESTING FOR EXOGENOUS FEATURES")
print("="*60)

stationarity_results = {}

for feature in exogenous_features:
    print(f"\n🔍 Testing: {feature}")

    # ADF Test
    adf_result = adfuller(df_selected[feature].dropna())
    adf_stationary = adf_result[1] < 0.05

    # KPSS Test
    kpss_result = kpss(df_selected[feature].dropna())
    kpss_stationary = kpss_result[1] > 0.05

    # Overall conclusion
    if adf_stationary and kpss_stationary:
        conclusion = "STATIONARY"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "NON-STATIONARY"
    else:
        conclusion = "INCONCLUSIVE"

    # Store results
    stationarity_results[feature] = {
        'ADF_pvalue': adf_result[1],
        'ADF_stationary': adf_stationary,
        'KPSS_pvalue': kpss_result[1],
        'KPSS_stationary': kpss_stationary,
        'Conclusion': conclusion
    }

    print(f"   ADF p-value: {adf_result[1]:.6f} ({'Stationary' if adf_stationary else 'Non-stationary'})")
    print(f"   KPSS p-value: {kpss_result[1]:.6f} ({'Stationary' if kpss_stationary else 'Non-stationary'})")
    print(f"   ➤ Conclusion: {conclusion}")

# Summary
print("\n" + "="*60)
print("📋 STATIONARITY SUMMARY:")
stationary_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'STATIONARY')
non_stationary_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'NON-STATIONARY')
inconclusive_count = sum(1 for r in stationarity_results.values() if r['Conclusion'] == 'INCONCLUSIVE')

print(f"   Stationary features: {stationary_count}")
print(f"   Non-stationary features: {non_stationary_count}")
print(f"   Inconclusive features: {inconclusive_count}")

# Chỉ cần import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Prepare data
print("🚀 TIME SERIES MODELS EXPERIMENTATION")
print("="*60)

# Use original features for exogenous variables (7 features)
exog_features = [
    'Nhiệt độ tối đa 2m', 'Nhiệt độ tối thiểu 2m', 'Độ ẩm tương đối 2m',
    'Độ ẩm đất bề mặt', 'Hướng gió 10m', 'Áp suất bề mặt', 'Bức xạ sóng dài xuống'
]

# Target variable
target = df_selected['Lượng mưa']
exog_data_original = df_selected[exog_features]

# ==============================================================================
# >>>>>>>>>>>> START OF FIX 1: MAKE EXOGENOUS VARIABLES STATIONARY <<<<<<<<<<<<
# ==============================================================================
exog_data_stationary = pd.DataFrame(index=exog_data_original.index)

print("🩺 Checking and transforming exogenous variables for stationarity...")
for col in exog_data_original.columns:
    # ADF test to check for unit root
    adf_pvalue = adfuller(exog_data_original[col].dropna())[1]
    if adf_pvalue >= 0.05:
        # If not stationary, apply first difference
        print(f"   - Column '{col}' is non-stationary (p={adf_pvalue:.3f}). Applying differencing.")
        exog_data_stationary[col] = exog_data_original[col].diff()
    else:
        # If stationary, use as is
        print(f"   - Column '{col}' is stationary (p={adf_pvalue:.3f}).")
        exog_data_stationary[col] = exog_data_original[col]

# Drop NaNs created by differencing
exog_data_stationary = exog_data_stationary.dropna()
# Align target variable with the new stationary exogenous data
target = target.loc[exog_data_stationary.index]
print("   ✅ All exogenous variables are now stationary.")
# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> END OF FIX 1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================


# Time Series Cross-Validation setup
tscv = TimeSeriesSplit(n_splits=5)
print(f"\n📊 Data Setup:")
print(f"   Total samples after alignment: {len(target)}")
print(f"   CV splits: 5")

# Results storage
results = {
    'ARIMA': {'MAE': [], 'RMSE': [], 'R2': []},
    'SARIMA': {'MAE': [], 'RMSE': [], 'R2': []},
    'ARIMAX': {'MAE': [], 'RMSE': [], 'R2': []},
    'SARIMAX': {'MAE': [], 'RMSE': [], 'R2': []}
}

# Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(tscv.split(target)):
    print(f"\n🔄 Fold {fold + 1}/5")

    # Split data
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    # Use the stationary exogenous data
    X_train, X_test = exog_data_stationary.iloc[train_idx], exog_data_stationary.iloc[test_idx]

    # Define model orders
    order = (3, 0, 3)
    seasonal_order = (1, 1, 1, 7)

    # ==============================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>> START OF FIX 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    # Use SARIMAX for all models for consistency

    # 1. ARIMA
    try:
        model = SARIMAX(y_train, order=order).fit(disp=False)
        pred = model.forecast(steps=len(y_test))
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['ARIMA']['MAE'].append(mae); results['ARIMA']['RMSE'].append(rmse); results['ARIMA']['R2'].append(r2)
        print(f"   ARIMA: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   ARIMA: Failed - {e}")

    # 2. SARIMA
    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
        pred = model.forecast(steps=len(y_test))
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['SARIMA']['MAE'].append(mae); results['SARIMA']['RMSE'].append(rmse); results['SARIMA']['R2'].append(r2)
        print(f"   SARIMA: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   SARIMA: Failed - {e}")

    # 3. ARIMAX
    try:
        model = SARIMAX(y_train, exog=X_train, order=order).fit(disp=False)
        pred = model.forecast(steps=len(y_test), exog=X_test)
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['ARIMAX']['MAE'].append(mae); results['ARIMAX']['RMSE'].append(rmse); results['ARIMAX']['R2'].append(r2)
        print(f"   ARIMAX: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   ARIMAX: Failed - {e}")

    # 4. SARIMAX
    try:
        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
        pred = model.forecast(steps=len(y_test), exog=X_test)
        # ... (evaluation code is the same)
        mae = mean_absolute_error(y_test, pred); rmse = np.sqrt(mean_squared_error(y_test, pred)); r2 = r2_score(y_test, pred)
        results['SARIMAX']['MAE'].append(mae); results['SARIMAX']['RMSE'].append(rmse); results['SARIMAX']['R2'].append(r2)
        print(f"   SARIMAX: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    except Exception as e:
        print(f"   SARIMAX: Failed - {e}")

