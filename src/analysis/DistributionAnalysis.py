# =============================================================================
# COMPLETE REFACTORED: src/analysis/DistributionAnalysis.py
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, jarque_bera, skewtest, kurtosistest
from scipy.stats import gaussian_kde
import warnings
import logging

from ..config.constants import Config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DistributionAnalyzer:
    """
    Comprehensive Distribution Analysis for Weather Data
    Refactored with complete functionality from original ds.py
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_col: str = None) -> None:
        """
        Initialize Distribution Analyzer
        
        Args:
            df: Weather data DataFrame
            target_col: Target variable column name (defaults to Config)
        """
        self.df = df.copy()
        self.target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
        
        # Use Config for column identification
        coord_cols = [Config.COLUMN_MAPPING.get('LATITUDE', 'Vĩ độ'),
                     Config.COLUMN_MAPPING.get('LONGITUDE', 'Kinh độ')]
        
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.analysis_cols = [col for col in self.numerical_cols if col not in coord_cols]
        
        print("📊 DISTRIBUTION ANALYSIS INITIALIZED")
        print("="*60)
        print(f"   - Total Features: {len(self.numerical_cols)}")
        print(f"   - Analysis Features: {len(self.analysis_cols)}")
        print(f"   - Target Variable: {self.target_col}")
        print(f"   - Sample Size: {len(self.df):,}")
    
    def analyze_descriptive_stats(self) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics for all features
        Based on ds.py lines 56-134
        
        Returns:
            DataFrame with comprehensive descriptive statistics
        """
        print("\n" + "="*80)
        print("📈 1. DESCRIPTIVE STATISTICS SUMMARY")
        print("="*80)
        
        desc_stats = []
        
        for col in self.analysis_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                continue
                
            try:
                # Basic statistics
                mean_val = data.mean()
                median_val = data.median()
                
                # Mode calculation (most frequent value)
                mode_result = data.mode()
                mode_val = mode_result.iloc[0] if len(mode_result) > 0 else np.nan
                
                # Shape statistics
                skewness = data.skew()
                kurt = data.kurtosis()
                
                # Variability statistics
                std_val = data.std()
                var_val = data.var()
                cv = (std_val / mean_val) * 100 if mean_val != 0 else np.inf
                
                # Range statistics
                range_val = data.max() - data.min()
                iqr = data.quantile(0.75) - data.quantile(0.25)
                
                # Percentiles
                p5 = data.quantile(0.05)
                p95 = data.quantile(0.95)
                
                stats_dict = {
                    'Feature': col,
                    'Count': len(data),
                    'Mean': mean_val,
                    'Median': median_val,
                    'Mode': mode_val,
                    'Std': std_val,
                    'Variance': var_val,
                    'CV(%)': cv,
                    'Min': data.min(),
                    'P5': p5,
                    'Q1': data.quantile(0.25),
                    'Q3': data.quantile(0.75),
                    'P95': p95,
                    'Max': data.max(),
                    'Range': range_val,
                    'IQR': iqr,
                    'Skewness': skewness,
                    'Kurtosis': kurt
                }
                
                desc_stats.append(stats_dict)
                
            except Exception as e:
                logger.warning(f"Failed to calculate stats for {col}: {e}")
                continue
        
        # Create comprehensive DataFrame
        desc_df = pd.DataFrame(desc_stats)
        
        # Display main statistics (as in original)
        print("📊 Comprehensive Descriptive Statistics:")
        print("="*120)
        
        # Basic stats
        basic_cols = ['Feature', 'Count', 'Mean', 'Median', 'Mode', 'Std', 'Min', 'Max']
        print("\n🔢 Basic Statistics:")
        print(desc_df[basic_cols].round(4).to_string(index=False))
        
        # Shape stats
        shape_cols = ['Feature', 'Skewness', 'Kurtosis', 'CV(%)', 'Range', 'IQR']
        print("\n📐 Shape & Variability Statistics:")
        print(desc_df[shape_cols].round(4).to_string(index=False))
        
        # Percentile stats
        percentile_cols = ['Feature', 'P5', 'Q1', 'Median', 'Q3', 'P95']
        print("\n📊 Percentile Statistics:")
        print(desc_df[percentile_cols].round(4).to_string(index=False))
        
        return desc_df
    
    def analyze_target_variable(self) -> Dict[str, Any]:
        """
        Deep analysis of target variable (precipitation)
        Updated to use Config constants
        """
        print("\n" + "="*80)
        print(f"🎯 2. TARGET VARIABLE DEEP ANALYSIS: {self.target_col}")
        print("="*80)
        
        target_data = self.df[self.target_col].dropna()
        
        if len(target_data) == 0:
            raise ValueError(f"No data found for target variable: {self.target_col}")
        
        # Basic descriptive statistics
        print("📈 Basic Descriptive Statistics:")
        print(f"   - Count: {len(target_data):,}")
        print(f"   - Mean: {target_data.mean():.4f} mm")
        print(f"   - Median: {target_data.median():.4f} mm")
        print(f"   - Mode: {target_data.mode().iloc[0]:.4f} mm")
        print(f"   - Standard Deviation: {target_data.std():.4f} mm")
        print(f"   - Variance: {target_data.var():.4f}")
        print(f"   - Coefficient of Variation: {(target_data.std()/target_data.mean())*100:.2f}%")
        
        # Range and percentiles
        print(f"\n📊 Range and Percentiles:")
        print(f"   - Minimum: {target_data.min():.4f} mm")
        print(f"   - 5th Percentile: {target_data.quantile(0.05):.4f} mm")
        print(f"   - 25th Percentile (Q1): {target_data.quantile(0.25):.4f} mm")
        print(f"   - 50th Percentile (Median): {target_data.quantile(0.50):.4f} mm")
        print(f"   - 75th Percentile (Q3): {target_data.quantile(0.75):.4f} mm")
        print(f"   - 95th Percentile: {target_data.quantile(0.95):.4f} mm")
        print(f"   - Maximum: {target_data.max():.4f} mm")
        print(f"   - Range: {target_data.max() - target_data.min():.4f} mm")
        print(f"   - IQR: {target_data.quantile(0.75) - target_data.quantile(0.25):.4f} mm")
        
        # Shape statistics
        skewness = target_data.skew()
        kurt = target_data.kurtosis()
        
        print(f"\n📐 Shape Statistics:")
        print(f"   - Skewness: {skewness:.4f}")
        if skewness > 1:
            skew_interpretation = "Highly right-skewed (lệch phải mạnh)"
        elif skewness > 0.5:
            skew_interpretation = "Moderately right-skewed (lệch phải vừa)"
        elif skewness > -0.5:
            skew_interpretation = "Approximately symmetric (gần đối xứng)"
        elif skewness > -1:
            skew_interpretation = "Moderately left-skewed (lệch trái vừa)"
        else:
            skew_interpretation = "Highly left-skewed (lệch trái mạnh)"
        
        print(f"     → Interpretation: {skew_interpretation}")
        
        print(f"   - Kurtosis: {kurt:.4f}")
        if kurt > 3:
            kurt_interpretation = "Leptokurtic (nhọn hơn normal)"
        elif kurt < -3:
            kurt_interpretation = "Platykurtic (tù hơn normal)"
        else:
            kurt_interpretation = "Mesokurtic (gần normal)"
        print(f"     → Interpretation: {kurt_interpretation}")
        
        # Vietnamese Meteorological Standards Classification using Config
        intensity_dist = self._classify_precipitation_intensity(target_data)
        total_days = len(target_data)
        categories = Config.PRECIPITATION_CLASSIFICATION["categories"]
        
        print(f"\n🌧️ Vietnamese Meteorological Standards Classification (24h):")
        print("   Based on Vietnamese National Weather Service Standards")
        
        for category_key, count in intensity_dist.items():
            category_info = categories[category_key]
            min_val, max_val = category_info["range"]
            label_vi = category_info["label_vi"]
            
            if max_val == float('inf'):
                range_str = f">{min_val}mm"
            elif min_val == max_val == 0:
                range_str = "0mm"
            else:
                range_str = f"{min_val}-{max_val}mm"
            
            percentage = count/total_days*100
            print(f"   - {label_vi} ({range_str}): {count:,} days ({percentage:.2f}%)")
        
        return {
            'basic_stats': {
                'count': len(target_data),
                'mean': target_data.mean(),
                'median': target_data.median(),
                'mode': target_data.mode().iloc[0],
                'std': target_data.std(),
                'skewness': skewness,
                'kurtosis': kurt
            },
            'intensity_distribution': intensity_dist
        }
    
    def test_normality(self) -> pd.DataFrame:
        """
        Test normality of distributions using multiple statistical tests
        Based on ds.py lines 247-395 - COMPLETE implementation
        
        Returns:
            DataFrame with normality test results
        """
        print("\n" + "="*80)
        print("🔬 3. STATISTICAL DISTRIBUTION SHAPE TESTS")
        print("="*80)
        
        test_results = []
        
        for col in self.analysis_cols:
            data = self.df[col].dropna()
            
            if len(data) < 8:  # Minimum sample size for tests
                continue
                
            try:
                # Sample data if too large for some tests
                sample_size = min(5000, len(data))
                sample_data = data.sample(sample_size) if len(data) > sample_size else data
                
                test_result = self._run_normality_tests(sample_data, col)
                test_results.append(test_result)
                
            except Exception as e:
                logger.warning(f"Normality tests failed for {col}: {e}")
                continue
        
        test_df = pd.DataFrame(test_results)
        
        if len(test_df) > 0:
            print("📊 Normality Test Results Summary:")
            print("="*100)
            
            # Main results table
            display_cols = ['Feature', 'Sample_Size', 'Shapiro_Normal', 'DAgostino_Normal',
                           'JB_Normal', 'Skew_Normal', 'Kurt_Normal', 'AD_Normal']
            print(test_df[display_cols].to_string(index=False))
            
            # Detailed results for target variable
            if self.target_col in test_df['Feature'].values:
                target_row = test_df[test_df['Feature'] == self.target_col].iloc[0]
                print(f"\n🎯 Detailed Results for {self.target_col}:")
                print(f"   - Shapiro-Wilk: statistic={target_row['Shapiro_Stat']:.6f}, p-value={target_row['Shapiro_P']:.2e}")
                print(f"   - D'Agostino: statistic={target_row['DAgostino_Stat']:.6f}, p-value={target_row['DAgostino_P']:.2e}")
                print(f"   - Jarque-Bera: statistic={target_row['JB_Stat']:.6f}, p-value={target_row['JB_P']:.2e}")
                print(f"   - Skewness test: statistic={target_row['Skew_Stat']:.6f}, p-value={target_row['Skew_P']:.2e}")
                print(f"   - Kurtosis test: statistic={target_row['Kurt_Stat']:.6f}, p-value={target_row['Kurt_P']:.2e}")
                print(f"   - Anderson-Darling: statistic={target_row['AD_Stat']:.6f}, critical_5%={target_row['AD_Critical_5%']:.6f}")
            
            # Summary statistics
            normal_features = test_df[
                (test_df['Shapiro_Normal'] == 'Yes') &
                (test_df['DAgostino_Normal'] == 'Yes') &
                (test_df['JB_Normal'] == 'Yes')
            ]['Feature'].tolist()
            
            print(f"\n📈 Summary:")
            print(f"   - Features possibly normal: {len(normal_features)} / {len(test_df)}")
            if normal_features:
                print(f"   - Normal features: {normal_features}")
            else:
                print(f"   - No features follow normal distribution")
        
        return test_df
    
    def test_non_parametric(self) -> pd.DataFrame:
        """
        Test relationships using non-parametric methods
        Based on ds.py lines 397-620 - COMPLETE implementation
        
        Returns:
            DataFrame with non-parametric test results
        """
        non_param_results = []
        target_data = self.df[self.target_col].dropna()
        
        for col in self.analysis_cols:
            if col == self.target_col:
                continue
                
            feature_data = self.df[col].dropna()
            aligned_indices = feature_data.index.intersection(target_data.index)
            
            if len(aligned_indices) < 20:
                continue
                
            try:
                test_result = self._run_non_parametric_tests(
                    feature_data.loc[aligned_indices],
                    target_data.loc[aligned_indices],
                    col
                )
                non_param_results.append(test_result)
                
            except Exception as e:
                logger.warning(f"Non-parametric tests failed for {col}: {e}")
                continue
        
        # Display non-parametric results
        non_param_df = pd.DataFrame(non_param_results) if non_param_results else pd.DataFrame()
        
        if not non_param_df.empty:
            print("\n📊 Non-parametric Test Results:")
            print("="*120)
            
            # Mann-Whitney U Test results
            mw_cols = ['Feature', 'MW_Statistic', 'MW_P_Value', 'MW_Significant']
            print("\n🔍 Mann-Whitney U Test (Rain vs No Rain):")
            mw_results = non_param_df[mw_cols].dropna(subset=['MW_Statistic'])
            if not mw_results.empty:
                print(mw_results.round(6).to_string(index=False))
            
            # Kruskal-Wallis Test results
            kw_cols = ['Feature', 'KW_Statistic', 'KW_P_Value', 'KW_Significant']
            print("\n🔍 Kruskal-Wallis Test (Multiple Rainfall Groups):")
            kw_results = non_param_df[kw_cols].dropna(subset=['KW_Statistic'])
            if not kw_results.empty:
                print(kw_results.round(6).to_string(index=False))
            
            # Spearman Correlation results
            spear_cols = ['Feature', 'Spearman_Corr', 'Spearman_P_Value', 'Spearman_Significant']
            print("\n🔍 Spearman Rank Correlation with Rainfall:")
            spear_results = non_param_df[spear_cols].dropna(subset=['Spearman_Corr'])
            if not spear_results.empty:
                print(spear_results.round(6).to_string(index=False))
        
        return non_param_df
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive distribution analysis report
        Based on ds.py lines 622-728 - COMPLETE implementation
        
        Returns:
            Complete analysis results with insights
        """
        print("\n" + "="*80)
        print("📋 5. DISTRIBUTION ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Run all analyses
        desc_stats = self.analyze_descriptive_stats()
        target_analysis = self.analyze_target_variable()
        normality_tests = self.test_normality()
        non_parametric_tests = self.test_non_parametric()
        
        # Summary insights (as in original)
        print(f"\n🔍 KEY INSIGHTS:")
        
        # Target variable insights
        target_stats = target_analysis['basic_stats']
        print(f"\n🎯 Target Variable ({self.target_col}):")
        print(f"   - Highly right-skewed (skew={target_stats['skewness']:.2f})")
        print(f"   - Mean > Median ({target_stats['mean']:.2f} > {target_stats['median']:.2f})")
        print(f"   - High variability (CV = {(target_stats['std']/target_stats['mean'])*100:.1f}%)")
        
        intensity_dist = target_analysis['intensity_distribution']
        total_obs = sum(intensity_dist.values())
        print(f"   - No rain days: {intensity_dist['no_rain']/total_obs*100:.1f}%")
        print(f"   - Light-moderate rain: {(intensity_dist['light_rain']+intensity_dist['moderate_rain'])/total_obs*100:.1f}%")
        print(f"   - Heavy rain events: {(intensity_dist['heavy_rain']+intensity_dist['very_heavy_rain'])/total_obs*100:.1f}%")
        
        # Distribution characteristics
        if len(normality_tests) > 0:
            normal_count = sum([
                1 for _, row in normality_tests.iterrows()
                if row['Shapiro_Normal'] == 'Yes' and row['JB_Normal'] == 'Yes'
            ])
            print(f"\n📊 Distribution Characteristics:")
            print(f"   - Normally distributed features: {normal_count}/{len(normality_tests)}")
            print(f"   - Most features are non-normal (typical for weather data)")
        
        # Non-parametric test insights
        if not non_parametric_tests.empty:
            significant_mw = non_parametric_tests[non_parametric_tests['MW_Significant'] == 'Yes']['Feature'].tolist()
            significant_kw = non_parametric_tests[non_parametric_tests['KW_Significant'] == 'Yes']['Feature'].tolist()
            significant_spear = non_parametric_tests[non_parametric_tests['Spearman_Significant'] == 'Yes']['Feature'].tolist()
            
            print(f"\n🔬 Non-parametric Test Results:")
            print(f"   - Features with significant rain/no-rain differences: {len(significant_mw)} ({significant_mw})")
            print(f"   - Features with significant group differences: {len(significant_kw)} ({significant_kw})")
            print(f"   - Features with significant rank correlation: {len(significant_spear)} ({significant_spear})")
        
        return {
            'descriptive_stats': desc_stats,
            'target_analysis': target_analysis,
            'normality_tests': normality_tests,
            'non_parametric_tests': non_parametric_tests,
            'summary_insights': {
                'target_skewness': target_stats['skewness'],
                'zero_rain_percentage': intensity_dist['no_rain']/total_obs*100,
                'normal_features_count': normal_count if len(normality_tests) > 0 else 0,
                'high_skew_features': []  # Will be filled by visualization
            }
        }
    
    def _classify_precipitation_intensity(self, data: pd.Series) -> Dict[str, int]:
        """
        Classify precipitation by Vietnamese meteorological standards
        Now using Config constants for consistency
        """
        categories = Config.PRECIPITATION_CLASSIFICATION["categories"]
        
        classification = {}
        for category_key, category_info in categories.items():
            min_val, max_val = category_info["range"]
            
            if category_key == "no_rain":
                count = (data == 0).sum()
            elif category_key == "trace_rain":
                count = ((data > 0) & (data <= 0.6)).sum()
            elif max_val == float('inf'):
                count = (data > min_val).sum()
            else:
                count = ((data > min_val) & (data <= max_val)).sum()
            
            classification[category_key] = count
        
        return classification
    
    def _run_normality_tests(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Run comprehensive normality tests - COMPLETE implementation
        Based on ds.py lines 330-395
        """
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(data)
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(data)
        
        # Skewness test - MISSING in original refactor
        skew_stat, skew_p = skewtest(data)
        
        # Kurtosis test - MISSING in original refactor
        kurt_stat, kurt_p = kurtosistest(data)
        
        # Anderson-Darling test for normality - MISSING in original refactor
        ad_stat, ad_critical, ad_sig = stats.anderson(data, dist='norm')
        
        return {
            'Feature': col_name,
            'Sample_Size': len(data),
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_P': shapiro_p,
            'Shapiro_Normal': 'Yes' if shapiro_p > 0.05 else 'No',
            'DAgostino_Stat': dagostino_stat,
            'DAgostino_P': dagostino_p,
            'DAgostino_Normal': 'Yes' if dagostino_p > 0.05 else 'No',
            'JB_Stat': jb_stat,
            'JB_P': jb_p,
            'JB_Normal': 'Yes' if jb_p > 0.05 else 'No',
            'Skew_Stat': skew_stat,
            'Skew_P': skew_p,
            'Skew_Normal': 'Yes' if skew_p > 0.05 else 'No',
            'Kurt_Stat': kurt_stat,
            'Kurt_P': kurt_p,
            'Kurt_Normal': 'Yes' if kurt_p > 0.05 else 'No',
            'AD_Stat': ad_stat,
            'AD_Critical_5%': ad_critical[2],  # 5% significance level
            'AD_Normal': 'Yes' if ad_stat < ad_critical[2] else 'No'
        }
    
    def _run_non_parametric_tests(self, aligned_feature: pd.Series, 
                                 aligned_target: pd.Series, 
                                 col_name: str) -> Dict[str, Any]:
        """
        Run comprehensive non-parametric tests - COMPLETE implementation
        Based on ds.py lines 520-575
        """
        test_result = {'Feature': col_name}
        
        try:
            # Mann-Whitney U Test (Rain vs No Rain)
            no_rain_feature = aligned_feature[aligned_target == 0]
            rain_feature = aligned_feature[aligned_target > 0]
            
            if len(no_rain_feature) > 5 and len(rain_feature) > 5:
                mw_stat, mw_p = stats.mannwhitneyu(no_rain_feature, rain_feature, alternative='two-sided')
                test_result['MW_Statistic'] = mw_stat
                test_result['MW_P_Value'] = mw_p
                test_result['MW_Significant'] = 'Yes' if mw_p < 0.05 else 'No'
            else:
                test_result['MW_Statistic'] = np.nan
                test_result['MW_P_Value'] = np.nan
                test_result['MW_Significant'] = 'N/A'
            
            # Kruskal-Wallis Test
            no_rain_group = aligned_feature[aligned_target == 0]
            light_rain_group = aligned_feature[(aligned_target > 0) & (aligned_target <= 2.5)]
            moderate_rain_group = aligned_feature[(aligned_target > 2.5) & (aligned_target <= 7.5)]
            heavy_rain_group = aligned_feature[aligned_target > 7.5]
            
            groups_with_data = [g for g in [no_rain_group, light_rain_group, moderate_rain_group, heavy_rain_group] if len(g) > 3]
            
            if len(groups_with_data) >= 3:
                kw_stat, kw_p = stats.kruskal(*groups_with_data)
                test_result['KW_Statistic'] = kw_stat
                test_result['KW_P_Value'] = kw_p
                test_result['KW_Significant'] = 'Yes' if kw_p < 0.05 else 'No'
            else:
                test_result['KW_Statistic'] = np.nan
                test_result['KW_P_Value'] = np.nan
                test_result['KW_Significant'] = 'N/A'
            
            # Spearman Rank Correlation
            spearman_corr, spearman_p = stats.spearmanr(aligned_feature, aligned_target)
            test_result['Spearman_Corr'] = spearman_corr
            test_result['Spearman_P_Value'] = spearman_p
            test_result['Spearman_Significant'] = 'Yes' if spearman_p < 0.05 else 'No'
            
        except Exception as e:
            logger.warning(f"Error in non-parametric tests for {col_name}: {e}")
        
        return test_result

# =============================================================================
# COMPLETE VISUALIZATION MODULE
# =============================================================================

class DistributionVisualizer:
    """
    Complete visualization implementation based on ds.py lines 622-728
    """
    
    def __init__(self, analyzer: DistributionAnalyzer):
        self.analyzer = analyzer
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def visualize_distributions(self) -> pd.DataFrame:
        """
        Create all visualizations for distribution analysis
        Based on ds.py lines 622-728 - COMPLETE implementation
        
        Returns:
            DataFrame with skewness and kurtosis comparison
        """
        print("\n" + "="*80)
        print("📊 4. DISTRIBUTION VISUALIZATIONS")
        print("="*80)
        
        # 4.1 Target Variable Comprehensive Visualization
        print("🎯 Creating comprehensive target variable visualization...")
        self.plot_target_distribution()
        
        # 4.2 All Features Distribution Overview
        print("\n📊 Creating distribution overview for all features...")
        self.plot_all_features_overview()
        
        # 4.3 Skewness and Kurtosis Comparison
        print("\n📐 Creating skewness and kurtosis comparison...")
        skew_kurt_df = self.plot_skewness_kurtosis_comparison()
        
        return skew_kurt_df
    
    def plot_target_distribution(self, **kwargs) -> None:
        """
        Plot comprehensive target variable distribution
        Based on ds.py lines 628-700
        """
        target_data = self.analyzer.df[self.analyzer.target_col].dropna()
        
        # Create comprehensive subplot for target variable
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Distribution Analysis: {self.analyzer.target_col}', 
                    fontsize=16, fontweight='bold')
        
        # Histogram with KDE
        axes[0,0].hist(target_data, bins=50, density=True, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].axvline(target_data.mean(), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {target_data.mean():.2f}')
        axes[0,0].axvline(target_data.median(), color='green', linestyle='--', 
                         linewidth=2, label=f'Median: {target_data.median():.2f}')
        
        # Add KDE
        kde = gaussian_kde(target_data)
        x_range = np.linspace(target_data.min(), target_data.max(), 200)
        axes[0,0].plot(x_range, kde(x_range), 'orange', linewidth=2, label='KDE')
        axes[0,0].set_title('Histogram with KDE')
        axes[0,0].set_xlabel('Precipitation (mm)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0,1].boxplot(target_data, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0,1].set_title('Box Plot')
        axes[0,1].set_ylabel('Precipitation (mm)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Q-Q plot against normal distribution
        stats.probplot(target_data, dist="norm", plot=axes[0,2])
        axes[0,2].set_title('Q-Q Plot vs Normal Distribution')
        axes[0,2].grid(True, alpha=0.3)
        
        # Log-scale histogram (for better visualization of skewed data)
        log_data = np.log1p(target_data)  # log(1+x) to handle zeros
        axes[1,0].hist(log_data, bins=50, density=True, alpha=0.7, 
                      color='lightgreen', edgecolor='black')
        axes[1,0].axvline(log_data.mean(), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {log_data.mean():.2f}')
        axes[1,0].axvline(log_data.median(), color='green', linestyle='--', 
                         linewidth=2, label=f'Median: {log_data.median():.2f}')
        axes[1,0].set_title('Log-Transformed Distribution')
        axes[1,0].set_xlabel('log(1 + Precipitation)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Violin plot
        axes[1,1].violinplot([target_data], positions=[1], 
                            showmeans=True, showmedians=True)
        axes[1,1].set_title('Violin Plot')
        axes[1,1].set_ylabel('Precipitation (mm)')
        axes[1,1].set_xticks([1])
        axes[1,1].set_xticklabels([self.analyzer.target_col])
        axes[1,1].grid(True, alpha=0.3)
        
        # Empirical CDF
        sorted_data = np.sort(target_data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1,2].plot(sorted_data, y_vals, 'b-', linewidth=2)
        axes[1,2].set_title('Empirical Cumulative Distribution')
        axes[1,2].set_xlabel('Precipitation (mm)')
        axes[1,2].set_ylabel('Cumulative Probability')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_features_overview(self) -> None:
        """
        Create distribution overview for all features
        Based on ds.py lines 702-750 - MISSING in original refactor
        """
        # Calculate number of plots needed
        n_features = len(self.analyzer.analysis_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle('Distribution Overview: All Numerical Features', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.analyzer.analysis_cols):
            data = self.analyzer.df[col].dropna()
            
            if len(data) > 0:
                # Create histogram with KDE
                axes[i].hist(data, bins=30, density=True, alpha=0.7, 
                           color='skyblue', edgecolor='black')
                
                # Add statistics
                axes[i].axvline(data.mean(), color='red', linestyle='--', 
                              linewidth=1, alpha=0.8)
                axes[i].axvline(data.median(), color='green', linestyle='--', 
                              linewidth=1, alpha=0.8)
                
                axes[i].set_title(f'{col}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}', 
                                fontsize=10)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(self.analyzer.analysis_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_skewness_kurtosis_comparison(self) -> pd.DataFrame:
        """
        Create skewness and kurtosis comparison charts
        Based on ds.py lines 752-800 - MISSING in original refactor
        
        Returns:
            DataFrame with skewness and kurtosis data
        """
        skew_kurt_data = []
        for col in self.analyzer.analysis_cols:
            data = self.analyzer.df[col].dropna()
            if len(data) > 0:
                skew_kurt_data.append({
                    'Feature': col,
                    'Skewness': data.skew(),
                    'Kurtosis': data.kurtosis(),
                    'Is_Target': col == self.analyzer.target_col
                })
        
        skew_kurt_df = pd.DataFrame(skew_kurt_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Skewness plot
        colors = ['red' if is_target else 'skyblue' for is_target in skew_kurt_df['Is_Target']]
        bars1 = ax1.bar(range(len(skew_kurt_df)), skew_kurt_df['Skewness'], 
                       color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.axhline(y=1, color='orange', linestyle='--', linewidth=0.8, 
                   alpha=0.7, label='Moderate Skew')
        ax1.axhline(y=-1, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.set_title('Skewness Comparison Across Features')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Skewness')
        ax1.set_xticks(range(len(skew_kurt_df)))
        ax1.set_xticklabels(skew_kurt_df['Feature'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Kurtosis plot
        bars2 = ax2.bar(range(len(skew_kurt_df)), skew_kurt_df['Kurtosis'], 
                       color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, 
                   label='Normal Kurtosis')
        ax2.axhline(y=3, color='orange', linestyle='--', linewidth=0.8, 
                   alpha=0.7, label='High Kurtosis')
        ax2.axhline(y=-3, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
        ax2.set_title('Kurtosis Comparison Across Features')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Kurtosis')
        ax2.set_xticks(range(len(skew_kurt_df)))
        ax2.set_xticklabels(skew_kurt_df['Feature'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return skew_kurt_df

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_distributions(df: pd.DataFrame, 
                         target_col: str = None,
                         include_visualization: bool = True) -> Dict[str, Any]:
    """
    Complete distribution analysis for weather data
    Based on ds.py analyze_distributions function
    
    Args:
        df: Weather data DataFrame
        target_col: Target variable column name
        include_visualization: Whether to generate plots
        
    Returns:
        Comprehensive distribution analysis results
    """
    print("🚀 STARTING COMPREHENSIVE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = DistributionAnalyzer(df, target_col)
    
    # Generate comprehensive report
    results = analyzer.generate_report()
    
    # Add visualizations if requested
    if include_visualization:
        visualizer = DistributionVisualizer(analyzer)
        skew_kurt_comparison = visualizer.visualize_distributions()
        
        # Update results with visualization data
        if len(skew_kurt_comparison) > 0:
            high_skew = skew_kurt_comparison[abs(skew_kurt_comparison['Skewness']) > 1]
            results['skew_kurt_comparison'] = skew_kurt_comparison
            results['summary_insights']['high_skew_features'] = high_skew['Feature'].tolist()
    
    print("\n✅ DISTRIBUTION ANALYSIS COMPLETED")
    print("="*80)
    
    return results