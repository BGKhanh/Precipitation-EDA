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
