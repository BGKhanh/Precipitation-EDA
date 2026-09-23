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

    def visualize_temporal_dynamics(self, temporal_results: Dict[str, Any]) -> None:
        """
        Visualize temporal correlation dynamics
        
        Args:
            temporal_results: Results from analyze_temporal_dynamics()
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Temporal Correlation Dynamics Analysis', 
                     fontsize=16, fontweight='bold')

        # 1. Seasonal correlations
        seasonal_corrs = temporal_results.get('seasonal_correlations', {})
        if seasonal_corrs:
            seasons = list(seasonal_corrs.keys())
            season_data = []
            feature_names = []
            
            # Get top 10 features for clarity
            if seasons:
                first_season = seasons[0]
                top_features = seasonal_corrs[first_season].abs().sort_values(ascending=False).head(10).index
                feature_names = top_features.tolist()
                
                for season in seasons:
                    if season in seasonal_corrs:
                        season_values = [seasonal_corrs[season].get(feat, 0) for feat in feature_names]
                        season_data.append(season_values)
            
            if season_data:
                season_df = pd.DataFrame(season_data, index=seasons, columns=feature_names)
                sns.heatmap(season_df, annot=True, cmap='RdBu_r', center=0,
                           fmt='.2f', ax=axes[0,0], cbar_kws={"shrink": .8})
                axes[0,0].set_title('Seasonal Correlation Patterns', fontweight='bold')
                axes[0,0].set_xlabel('Features')
                axes[0,0].set_ylabel('Seasons')
        else:
            axes[0,0].text(0.5, 0.5, 'No seasonal data available', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Seasonal Correlation Patterns', fontweight='bold')

        # 2. Rolling correlation stability
        rolling_corrs = temporal_results.get('rolling_correlations', {})
        if rolling_corrs:
            features = list(rolling_corrs.keys())
            mean_corrs = [rolling_corrs[feat]['mean_correlation'] for feat in features]
            std_corrs = [rolling_corrs[feat]['std_correlation'] for feat in features]
            
            bars = axes[0,1].bar(range(len(features)), mean_corrs, 
                                yerr=std_corrs, alpha=0.7, capsize=5,
                                color=['green' if std < 0.1 else 'orange' for std in std_corrs])
            axes[0,1].set_xticks(range(len(features)))
            axes[0,1].set_xticklabels(features, rotation=45, ha='right')
            axes[0,1].set_title('Rolling Correlation Stability', fontweight='bold')
            axes[0,1].set_ylabel('Mean Correlation ± Std')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add stability legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Stable (std < 0.1)'),
                             Patch(facecolor='orange', label='Variable (std ≥ 0.1)')]
            axes[0,1].legend(handles=legend_elements)
        else:
            axes[0,1].text(0.5, 0.5, 'No rolling correlation data available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Rolling Correlation Stability', fontweight='bold')

        # 3. Lagged correlations
        lagged_corrs = temporal_results.get('lagged_correlations', {})
        if lagged_corrs:
            features = list(lagged_corrs.keys())
            best_lags = [lagged_corrs[feat]['best_lag'] for feat in features]
            best_corrs = [abs(lagged_corrs[feat]['best_correlation']) for feat in features]
            
            scatter = axes[1,0].scatter(best_lags, best_corrs, s=100, alpha=0.7, c=best_corrs, 
                                      cmap='viridis')
            
            for i, feat in enumerate(features):
                axes[1,0].annotate(feat, (best_lags[i], best_corrs[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            axes[1,0].set_xlabel('Best Lag (days)')
            axes[1,0].set_ylabel('|Best Correlation|')
            axes[1,0].set_title('Optimal Lag Correlations', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1,0], label='|Correlation|')
        else:
            axes[1,0].text(0.5, 0.5, 'No lagged correlation data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Optimal Lag Correlations', fontweight='bold')

        # 4. Temporal summary statistics
        axes[1,1].axis('off')
        summary_text = "📊 TEMPORAL DYNAMICS SUMMARY\n" + "="*40 + "\n"
        
        if seasonal_corrs:
            summary_text += f"🌤️ Seasonal Analysis:\n"
            summary_text += f"   - Analyzed seasons: {len(seasonal_corrs)}\n"
            
        if rolling_corrs:
            stable_count = sum(1 for feat in rolling_corrs 
                             if rolling_corrs[feat]['std_correlation'] < 0.1)
            summary_text += f"\n📈 Rolling Correlations:\n"
            summary_text += f"   - Analyzed features: {len(rolling_corrs)}\n"
            summary_text += f"   - Stable correlations: {stable_count}\n"
            
        if lagged_corrs:
            summary_text += f"\n⏰ Lagged Correlations:\n"
            summary_text += f"   - Analyzed features: {len(lagged_corrs)}\n"
            avg_lag = np.mean([abs(lagged_corrs[feat]['best_lag']) for feat in lagged_corrs])
            summary_text += f"   - Average optimal lag: {avg_lag:.1f} days\n"
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.show()

    def visualize_multicollinearity(self, vif_df: pd.DataFrame) -> None:
        """
        Visualize multicollinearity analysis results
        
        Args:
            vif_df: DataFrame with VIF scores from analyze_multicollinearity()
        """
        if vif_df.empty:
            print("⚠️ No multicollinearity data to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multicollinearity Analysis', fontsize=16, fontweight='bold')

        # 1. VIF scores bar plot
        vif_sorted = vif_df.sort_values('VIF_Score', ascending=True)
        colors = ['red' if score > 10 else 'orange' if score > 5 else 'green' 
                 for score in vif_sorted['VIF_Score']]
        
        bars = axes[0,0].barh(range(len(vif_sorted)), vif_sorted['VIF_Score'], 
                             color=colors, alpha=0.7)
        axes[0,0].set_yticks(range(len(vif_sorted)))
        axes[0,0].set_yticklabels(vif_sorted['Feature'], fontsize=9)
        axes[0,0].set_xlabel('VIF Score')
        axes[0,0].set_title('Variance Inflation Factors', fontweight='bold')
        axes[0,0].axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Moderate (5)')
        axes[0,0].axvline(x=10, color='red', linestyle='--', alpha=0.7, label='High (10)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Risk level distribution
        risk_counts = vif_df['Risk_Level'].value_counts()
        colors_pie = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        pie_colors = [colors_pie.get(level, 'gray') for level in risk_counts.index]
        
        wedges, texts, autotexts = axes[0,1].pie(risk_counts.values, labels=risk_counts.index,
                                                autopct='%1.1f%%', colors=pie_colors,
                                                explode=[0.05]*len(risk_counts))
        axes[0,1].set_title('Multicollinearity Risk Distribution', fontweight='bold')

        # 3. Category analysis
        if 'Category' in vif_df.columns:
            category_vif = vif_df.groupby('Category')['VIF_Score'].agg(['mean', 'max', 'count']).reset_index()
            category_vif = category_vif.sort_values('mean', ascending=False)
            
            x_pos = range(len(category_vif))
            bars1 = axes[1,0].bar([x-0.2 for x in x_pos], category_vif['mean'], 
                                 width=0.4, label='Mean VIF', alpha=0.7, color='skyblue')
            bars2 = axes[1,0].bar([x+0.2 for x in x_pos], category_vif['max'], 
                                 width=0.4, label='Max VIF', alpha=0.7, color='lightcoral')
            
            axes[1,0].set_xticks(x_pos)
            axes[1,0].set_xticklabels(category_vif['Category'], rotation=45, ha='right')
            axes[1,0].set_ylabel('VIF Score')
            axes[1,0].set_title('VIF by Feature Category', fontweight='bold')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Add count annotations
            for i, (mean_val, max_val, count) in enumerate(zip(category_vif['mean'], 
                                                              category_vif['max'], 
                                                              category_vif['count'])):
                axes[1,0].text(i, max_val + 0.5, f'n={count}', ha='center', fontsize=9)
        else:
            axes[1,0].text(0.5, 0.5, 'No category information available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('VIF by Feature Category', fontweight='bold')

        # 4. Summary statistics
        axes[1,1].axis('off')
        
        high_vif = vif_df[vif_df['VIF_Score'] > 10]
        moderate_vif = vif_df[(vif_df['VIF_Score'] > 5) & (vif_df['VIF_Score'] <= 10)]
        low_vif = vif_df[vif_df['VIF_Score'] <= 5]
        
        summary_text = "📊 MULTICOLLINEARITY SUMMARY\n" + "="*40 + "\n"
        summary_text += f"🔍 Total Features: {len(vif_df)}\n\n"
        summary_text += f"🔴 High Risk (VIF > 10): {len(high_vif)}\n"
        if len(high_vif) > 0:
            summary_text += f"   Max VIF: {vif_df['VIF_Score'].max():.2f}\n"
            summary_text += f"   Features: {', '.join(high_vif['Feature'].head(3).tolist())}\n"
        
        summary_text += f"\n🟡 Moderate Risk (5 < VIF ≤ 10): {len(moderate_vif)}\n"
        summary_text += f"\n🟢 Low Risk (VIF ≤ 5): {len(low_vif)}\n\n"
        
        summary_text += f"📈 Statistics:\n"
        summary_text += f"   Mean VIF: {vif_df['VIF_Score'].mean():.2f}\n"
        summary_text += f"   Median VIF: {vif_df['VIF_Score'].median():.2f}\n"
        summary_text += f"   Std VIF: {vif_df['VIF_Score'].std():.2f}\n"
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.show()

    def visualize_feature_network(self, network: nx.Graph, edges: List) -> None:
        """
        Visualize feature interaction network
        
        Args:
            network: NetworkX graph from analyze_feature_network()
            edges: Edge list with correlations
        """
        if network.number_of_nodes() == 0:
            print("⚠️ No network data to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Feature Interaction Network Analysis', 
                     fontsize=16, fontweight='bold')

        # 1. Network graph
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Node colors by category
        node_colors = []
        for node in network.nodes():
            category = network.nodes[node].get('category', 'Other')
            is_target = network.nodes[node].get('is_target', False)
            if is_target:
                node_colors.append('red')
            elif category == 'Temperature':
                node_colors.append('orange')
            elif category == 'Humidity':
                node_colors.append('blue')
            elif category == 'Wind':
                node_colors.append('green')
            elif category == 'Pressure_Radiation':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
        
        # Edge weights
        edge_weights = [network[u][v]['weight'] for u, v in network.edges()]
        
        nx.draw_networkx_nodes(network, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=axes[0,0])
        nx.draw_networkx_edges(network, pos, width=[w*3 for w in edge_weights], 
                              alpha=0.6, ax=axes[0,0])
        nx.draw_networkx_labels(network, pos, font_size=8, ax=axes[0,0])
        
        axes[0,0].set_title('Feature Correlation Network', fontweight='bold')
        axes[0,0].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Target'),
            Patch(facecolor='orange', label='Temperature'),
            Patch(facecolor='blue', label='Humidity'),
            Patch(facecolor='green', label='Wind'),
            Patch(facecolor='purple', label='Pressure/Radiation'),
            Patch(facecolor='gray', label='Other')
        ]
        axes[0,0].legend(handles=legend_elements, loc='upper right')

        # 2. Degree distribution
        degrees = [network.degree(node) for node in network.nodes()]
        axes[0,1].hist(degrees, bins=max(1, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Node Degree')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Node Degree Distribution', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)

        # 3. Edge weight distribution
        if edge_weights:
            axes[1,0].hist(edge_weights, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1,0].set_xlabel('Edge Weight (|Correlation|)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Edge Weight Distribution', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No edges in network', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Edge Weight Distribution', fontweight='bold')

        # 4. Network summary
        axes[1,1].axis('off')
        
        # Calculate network metrics
        if network.number_of_edges() > 0:
            avg_clustering = nx.average_clustering(network)
            density = nx.density(network)
            
            # Find most connected nodes
            degree_centrality = nx.degree_centrality(network)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            avg_clustering = 0
            density = 0
            top_nodes = []
        
        summary_text = "📊 NETWORK SUMMARY\n" + "="*35 + "\n"
        summary_text += f"🕸️ Network Structure:\n"
        summary_text += f"   Nodes: {network.number_of_nodes()}\n"
        summary_text += f"   Edges: {network.number_of_edges()}\n"
        summary_text += f"   Density: {density:.3f}\n"
        summary_text += f"   Avg Clustering: {avg_clustering:.3f}\n\n"
        
        if top_nodes:
            summary_text += f"🌟 Most Connected Features:\n"
            for i, (node, centrality) in enumerate(top_nodes[:3]):
                summary_text += f"   {i+1}. {node[:20]}...\n"
                summary_text += f"      (centrality: {centrality:.3f})\n"
        
        if edge_weights:
            summary_text += f"\n🔗 Edge Statistics:\n"
            summary_text += f"   Mean |correlation|: {np.mean(edge_weights):.3f}\n"
            summary_text += f"   Max |correlation|: {np.max(edge_weights):.3f}\n"
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

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
        
        # NEW: Visualize temporal dynamics
        visualizer.visualize_temporal_dynamics(results['temporal_results'])
        
        # NEW: Visualize multicollinearity
        visualizer.visualize_multicollinearity(results['multicollinearity_results'])
        
        # NEW: Visualize feature network
        visualizer.visualize_feature_network(results['network'], [])

    print("\n✅ CORRELATION ANALYSIS COMPLETED")
    print("="*80)

    return results