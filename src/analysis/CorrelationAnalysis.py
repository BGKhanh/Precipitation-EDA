# =============================================================================
# COMPONENT 4: CORRELATION ANALYSIS (REFACTORED)
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
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

from ..config.constants import Config

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("RdYlBu_r")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class CorrelationAnalyzer:
    """
    Comprehensive Correlation Analysis for Weather Data
    ✅ REFACTORED: Loại bỏ BaseAnalyzer, standalone module
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 target_col: str = None,
                 date_col: str = None) -> None:
        """
        Initialize Correlation Analyzer
        
        Args:
            df: Weather data DataFrame
            target_col: Target variable column name (defaults to Config)
            date_col: Date column name (defaults to Config)
        """
        self.df = df.copy()
        self.target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
        self.date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')

        # Use Config for column identification
        coord_cols = [Config.COLUMN_MAPPING.get('LATITUDE', 'Vĩ độ'),
                     Config.COLUMN_MAPPING.get('LONGITUDE', 'Kinh độ')]
        exclude_cols = coord_cols + [self.date_col, 'Nhóm']

        # Get numerical columns for analysis
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.analysis_cols = [col for col in self.numerical_cols if col not in exclude_cols]
        self.predictor_cols = [col for col in self.analysis_cols if col != self.target_col]

        print("🔍 CORRELATION ANALYZER INITIALIZED")
        print("="*70)
        print(f"   📊 Dataset Shape: {self.df.shape}")
        print(f"   🎯 Target Variable: {self.target_col}")
        print(f"   🔢 Total Features: {len(self.analysis_cols)}")
        print(f"   📈 Predictor Features: {len(self.predictor_cols)}")

    def analyze_meteorological_correlations(self) -> Dict[str, Any]:
        """
        Analyze meteorological variable correlations
        
        Returns:
            Dictionary with correlation matrices and feature groups
        """
        print("\n" + "="*70)
        print("🌤️ 1. METEOROLOGICAL CORRELATION MATRIX ANALYSIS")
        print("="*70)

        # Group features by meteorological categories using Config
        feature_groups = self._categorize_meteorological_features()

        print(f"📊 Meteorological Feature Groups:")
        for group, features in feature_groups.items():
            print(f"   - {group}: {len(features)} features")

        # Calculate correlations
        correlations = {
            'pearson': self.df[self.analysis_cols].corr(method='pearson'),
            'spearman': self.df[self.analysis_cols].corr(method='spearman')
        }

        return {
            'correlations': correlations,
            'feature_groups': feature_groups,
            'target_correlations': correlations['pearson'][self.target_col].drop(self.target_col)
        }

    def analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """
        Analyze temporal correlation dynamics
        
        Returns:
            Dictionary with seasonal, rolling, and lagged correlation results
        """
        print("\n" + "="*70)
        print("📅 2. TEMPORAL CORRELATION DYNAMICS")
        print("="*70)

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        results = {}

        # Seasonal analysis
        results['seasonal_correlations'] = self._analyze_seasonal_correlations()
        
        # Rolling correlations
        results['rolling_correlations'] = self._analyze_rolling_correlations()
        
        # Lagged correlations
        results['lagged_correlations'] = self._analyze_lagged_correlations()

        return results

    def analyze_multicollinearity(self) -> pd.DataFrame:
        """
        Analyze multicollinearity using VIF scores
        
        Returns:
            DataFrame with VIF scores and risk levels
        """
        print("\n" + "="*70)
        print("🔍 3. MULTICOLLINEARITY ANALYSIS")
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
            # Fallback: correlation-based detection
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
            return vif_df
        
        return pd.DataFrame()

    def analyze_feature_clustering(self) -> Dict[str, Any]:
        """
        Analyze feature clustering using hierarchical clustering and PCA
        
        Returns:
            Dictionary with clustering and PCA results
        """
        print("\n" + "="*70)
        print("🔬 4. FEATURE CLUSTERING & PCA ANALYSIS")
        print("="*70)

        X = self.df[self.predictor_cols].dropna()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.predictor_cols, index=X.index)

        # Hierarchical clustering
        corr_matrix = X_scaled_df.corr()
        distance_matrix = 1 - np.abs(corr_matrix)
        condensed_distances = squareform(distance_matrix.values)
        linkage_matrix = linkage(condensed_distances, method='ward')

        # Get clusters
        n_clusters = min(5, len(self.predictor_cols)//3)
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

        # PCA analysis
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)

        # Calculate PC correlations with target
        target_values = self.df.loc[X.index, self.target_col]
        pc_target_correlations = []

        for i in range(min(5, len(self.predictor_cols))):
            pc_corr = np.corrcoef(pca_result[:, i], target_values)[0, 1]
            pc_target_correlations.append({
                'PC': f'PC{i+1}',
                'Explained_Variance': pca.explained_variance_ratio_[i],
                'Target_Correlation': pc_corr
            })

        pc_df = pd.DataFrame(pc_target_correlations)
        print("   📈 Principal Components vs Target:")
        print(pc_df.to_string(index=False, float_format='%.3f'))

        return {
            'feature_clusters': feature_clusters,
            'pca_results': pc_df,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'linkage_matrix': linkage_matrix
        }

    def analyze_feature_network(self) -> Tuple[nx.Graph, List]:
        """
        Analyze feature interaction network
        
        Returns:
            NetworkX graph and edge list
        """
        print("\n" + "="*70)
        print("🕸️ 5. FEATURE INTERACTION NETWORK")
        print("="*70)

        corr_matrix = self.df[self.analysis_cols].corr()
        G = nx.Graph()

        # Add nodes
        for feature in self.analysis_cols:
            category = self._categorize_feature(feature)
            G.add_node(feature, category=category, is_target=(feature == self.target_col))

        # Add edges for significant correlations
        correlation_threshold = 0.25
        edges = []
        for i, feature1 in enumerate(self.analysis_cols):
            for j, feature2 in enumerate(self.analysis_cols):
                if i < j:
                    corr_val = corr_matrix.loc[feature1, feature2]
                    if abs(corr_val) >= correlation_threshold:
                        G.add_edge(feature1, feature2, weight=abs(corr_val), correlation=corr_val)
                        edges.append((feature1, feature2, corr_val))

        print(f"🕸️ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"   Density: {nx.density(G):.3f}")

        return G, edges

    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive correlation analysis report
        
        Returns:
            Complete analysis results with insights
        """
        print("\n" + "="*70)
        print("📋 6. CORRELATION ANALYSIS INSIGHTS REPORT")
        print("="*70)

        # Run all analyses
        meteorological_results = self.analyze_meteorological_correlations()
        temporal_results = self.analyze_temporal_dynamics()
        multicollinearity_results = self.analyze_multicollinearity()
        clustering_results = self.analyze_feature_clustering()
        network, edges = self.analyze_feature_network()

        # Generate insights
        correlations = meteorological_results['correlations']
        target_corr = correlations['pearson'][self.target_col].drop(self.target_col).abs().sort_values(ascending=False)
        
        strong_predictors = target_corr[target_corr > 0.3]
        moderate_predictors = target_corr[(target_corr > 0.2) & (target_corr <= 0.3)]

        print(f"\n🎯 CORRELATION ANALYSIS EXECUTIVE SUMMARY:")
        print("="*60)
        print(f"🌧️ PRECIPITATION PREDICTION INSIGHTS:")
        print(f"   - Strong predictors (|r| > 0.3): {len(strong_predictors)}")
        print(f"   - Moderate predictors (0.2 < |r| ≤ 0.3): {len(moderate_predictors)}")

        # Temporal insights
        if temporal_results['rolling_correlations']:
            print(f"\n📈 DYNAMIC CORRELATION INSIGHTS:")
            for var, results in temporal_results['rolling_correlations'].items():
                stability = "Stable" if results['std_correlation'] < 0.1 else "Variable"
                print(f"   - {var}: {stability} correlation")

        # Clustering insights
        print(f"\n🔬 CLUSTERING INSIGHTS:")
        print(f"   - Feature clusters identified: {len(clustering_results['feature_clusters'])}")
        if len(clustering_results['pca_results']) > 0:
            top_pc = clustering_results['pca_results'].iloc[0]
            print(f"   - Top PC explains {top_pc['Explained_Variance']:.1%} variance")

        return {
            'meteorological_results': meteorological_results,
            'temporal_results': temporal_results,
            'multicollinearity_results': multicollinearity_results,
            'clustering_results': clustering_results,
            'network': network,
            'strong_predictors': strong_predictors,
            'moderate_predictors': moderate_predictors
        }

    def _categorize_meteorological_features(self) -> Dict[str, List[str]]:
        """Categorize features by meteorological types using Config"""
        feature_groups = {
            'Temperature': [],
            'Humidity': [],
            'Wind': [],
            'Pressure_Radiation': [],
            'Precipitation': [self.target_col]
        }

        for col in self.analysis_cols:
            if col == self.target_col:
                continue
            category = self._categorize_feature(col)
            if category in feature_groups:
                feature_groups[category].append(col)

        return feature_groups

    def _categorize_feature(self, feature_name: str) -> str:
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

    def _interpret_vif(self, vif_score: float) -> str:
        """Helper function to interpret VIF scores"""
        if vif_score < 5:
            return "Low"
        elif vif_score < 10:
            return "Moderate"
        else:
            return "High"

    def _analyze_seasonal_correlations(self) -> Dict[str, pd.Series]:
        """Analyze seasonal correlation patterns"""
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

        return seasonal_correlations

    def _analyze_rolling_correlations(self) -> Dict[str, Dict[str, float]]:
        """Analyze rolling correlation patterns"""
        daily_agg = self.df.groupby(self.date_col)[self.analysis_cols].mean().reset_index()
        daily_agg = daily_agg.set_index(self.date_col).sort_index()
        daily_agg_clean = daily_agg.dropna()

        rolling_results = {}
        if len(daily_agg_clean) >= 60:
            static_corr = daily_agg_clean.corr()[self.target_col].sort_values(ascending=False)
            top_vars = static_corr.head(3).index.tolist()[1:3]

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

        return rolling_results

    def _analyze_lagged_correlations(self) -> Dict[str, Dict[str, float]]:
        """Analyze lagged correlation patterns"""
        daily_agg = self.df.groupby(self.date_col)[self.analysis_cols].mean().reset_index()
        daily_agg = daily_agg.set_index(self.date_col).sort_index()
        daily_agg_clean = daily_agg.dropna()

        lagged_results = {}
        if len(daily_agg_clean) >= 60:
            max_lags = 5
            static_corr = daily_agg_clean.corr()[self.target_col].sort_values(ascending=False)
            top_vars = static_corr.head(3).index.tolist()[1:3]

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

        return lagged_results


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class CorrelationVisualizer:
    """
    Visualization module for correlation analysis
    """

    def __init__(self, analyzer: CorrelationAnalyzer):
        self.analyzer = analyzer
        self.setup_style()

    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("RdYlBu_r")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10

    def visualize_meteorological_correlations(self, correlations: Dict[str, pd.DataFrame],
                                           feature_groups: Dict[str, List[str]]) -> None:
        """Visualize meteorological correlation matrices"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Meteorological Cross-Correlation Analysis',
                     fontsize=16, fontweight='bold')

        # Pearson correlation
        mask = np.triu(np.ones_like(correlations['pearson'], dtype=bool))
        sns.heatmap(correlations['pearson'], mask=mask, annot=True,
                   cmap='RdBu_r', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[0,0])
        axes[0,0].set_title('Pearson Correlation (Linear)', fontweight='bold')

        # Spearman correlation
        sns.heatmap(correlations['spearman'], mask=mask, annot=True,
                   cmap='RdBu_r', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[0,1])
        axes[0,1].set_title('Spearman Correlation (Monotonic)', fontweight='bold')

        # Target correlations
        target_corr = correlations['pearson'][self.analyzer.target_col].drop(self.analyzer.target_col).sort_values(key=abs, ascending=False)
        bars = axes[1,0].barh(range(len(target_corr)), target_corr.values,
                             color=['red' if x > 0 else 'blue' for x in target_corr.values], alpha=0.7)
        axes[1,0].set_yticks(range(len(target_corr)))
        axes[1,0].set_yticklabels(target_corr.index, fontsize=9)
        axes[1,0].set_title(f'Correlations with {self.analyzer.target_col}', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)

        # Nonlinearity detection
        diff_matrix = correlations['spearman'] - correlations['pearson']
        sns.heatmap(diff_matrix, mask=mask, annot=True,
                   cmap='RdYlGn', center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[1,1])
        axes[1,1].set_title('Nonlinearity Detection', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def visualize_clustering_analysis(self, clustering_results: Dict[str, Any]) -> None:
        """Visualize clustering and PCA results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Clustering & PCA Analysis', fontsize=16, fontweight='bold')

        # Dendrogram
        dendrogram(clustering_results['linkage_matrix'], 
                  labels=self.analyzer.predictor_cols, ax=axes[0,0],
                  orientation='top', leaf_rotation=90)
        axes[0,0].set_title('Feature Hierarchical Clustering', fontweight='bold')
        axes[0,0].tick_params(axis='x', labelsize=8)

        # PCA explained variance
        explained_var = clustering_results['explained_variance_ratio']
        cumsum_var = np.cumsum(explained_var)
        axes[1,0].bar(range(1, len(explained_var[:10])+1), explained_var[:10],
                      alpha=0.7, color='skyblue')
        axes[1,0].plot(range(1, len(cumsum_var[:10])+1), cumsum_var[:10],
                      'ro-', linewidth=2, markersize=6)
        axes[1,0].set_title('PCA Explained Variance', fontweight='bold')
        axes[1,0].set_xlabel('Principal Component')
        axes[1,0].set_ylabel('Explained Variance Ratio')
        axes[1,0].grid(True, alpha=0.3)

        # PC correlation with target
        pc_results = clustering_results['pca_results']
        pc_corrs = [abs(row['Target_Correlation']) for _, row in pc_results.iterrows()]
        axes[1,1].bar(range(1, len(pc_corrs)+1), pc_corrs, alpha=0.7, color='lightcoral')
        axes[1,1].set_title('PC vs Target Correlation', fontweight='bold')
        axes[1,1].set_xlabel('Principal Component')
        axes[1,1].set_ylabel('|Correlation| with Target')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_correlations(df: pd.DataFrame, 
                        target_col: str = None,
                        date_col: str = None,
                        include_visualization: bool = True) -> Dict[str, Any]:
    """
    Complete correlation analysis for weather data
    
    Args:
        df: Weather data DataFrame
        target_col: Target variable column name
        date_col: Date column name
        include_visualization: Whether to generate plots
        
    Returns:
        Comprehensive correlation analysis results
    """
    print("🚀 STARTING COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*80)

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(df, target_col, date_col)

    # Generate comprehensive report
    results = analyzer.generate_insights_report()

    # Add visualizations if requested
    if include_visualization:
        visualizer = CorrelationVisualizer(analyzer)
        
        # Visualize meteorological correlations
        meteorological_results = results['meteorological_results']
        visualizer.visualize_meteorological_correlations(
            meteorological_results['correlations'],
            meteorological_results['feature_groups']
        )
        
        # Visualize clustering results
        visualizer.visualize_clustering_analysis(results['clustering_results'])

    print("\n✅ CORRELATION ANALYSIS COMPLETED")
    print("="*80)

    return results