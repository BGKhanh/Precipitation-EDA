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