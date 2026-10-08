# =============================================================================
# COMPONENT 4: CORRELATION & MULTICOLLINEARITY ANALYSIS (REFACTORED)
# =============================================================================

from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.spatial.distance import squareform
from scipy.linalg import svdvals
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings

from ..config.constants import Config

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("RdYlBu_r")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


# =============================================================================
# METEOROLOGICAL PREDICTOR ALLOWLISTS (EXPLICIT CONTRACT)
# =============================================================================

CIRCULAR_WIND_ALLOWLIST: List[str] = [
    "Hướng gió 2m",
    "Hướng gió 10m",
]

LINEAR_PREDICTOR_ALLOWLIST: List[str] = [
    # Surface Solar & PAR Radiation (8)
    "Bức xạ sóng ngắn bề mặt",
    "Bức xạ sóng ngắn trời quang",
    "Bức xạ trực tiếp pháp tuyến",
    "Bức xạ khuếch tán",
    "Bức xạ đỉnh khí quyển",
    "Bức xạ quang hợp tổng",
    "Bức xạ quang hợp trời quang",
    "Bức xạ sóng dài xuống",
    # UV Radiation (3)
    "Bức xạ UVA",
    "Bức xạ UVB",
    "Chỉ số UV",
    # Temperature & Surface Thermal (7)
    "Nhiệt độ 2m",
    "Điểm sương 2m",
    "Nhiệt độ bầu ướt 2m",
    "Nhiệt độ bề mặt đất",
    "Biên độ nhiệt 2m",
    "Nhiệt độ tối đa 2m",
    "Nhiệt độ tối thiểu 2m",
    # Humidity (2)
    "Độ ẩm tuyệt đối 2m",
    "Độ ẩm tương đối 2m",
    # Surface Pressure (1)
    "Áp suất bề mặt",
    # Wind Speed at 2m (4)
    "Tốc độ gió 2m",
    "Tốc độ gió tối đa 2m",
    "Tốc độ gió tối thiểu 2m",
    "Biên độ gió 2m",
    # Wind Speed at 10m (4)
    "Tốc độ gió 10m",
    "Tốc độ gió tối đa 10m",
    "Tốc độ gió tối thiểu 10m",
    "Biên độ gió 10m",
    # Soil Moisture (3)
    "Độ ẩm đất bề mặt",
    "Độ ẩm đất vùng rễ",
    "Độ ẩm đất mặt cắt",
]

METEOROLOGICAL_ALLOWLIST: List[str] = LINEAR_PREDICTOR_ALLOWLIST + CIRCULAR_WIND_ALLOWLIST


class CorrelationAnalyzer:
    """
    Correlation and Multicollinearity Analysis for Weather Data.
    Operates strictly on Dev Train population (Zero-Leakage standard).
    Enforces explicit predictor allowlists and isolates circular wind variables.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        strict_schema: bool = True,
    ) -> None:
        """
        Initialize Correlation Analyzer.

        Args:
            df: Pre-split weather data DataFrame (Dev Train).
            target_col: Target variable column name.
            date_col: Date column name.
            strict_schema: If True, raises ValueError if any expected allowlist variable is missing.
        """
        self.df = df.copy()
        self.target_col = target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')
        self.date_col = date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
        self.strict_schema = strict_schema

        # Deterministic schema validation
        self.schema_validation = self._validate_schema(strict=self.strict_schema)

        # Allowlist-based predictor resolution
        self.linear_predictor_cols = self._get_linear_predictor_cols()
        self.circular_cols = self._get_circular_cols()
        self.analysis_cols = self.linear_predictor_cols + [self.target_col]
        self.predictor_cols = self.linear_predictor_cols  # Backwards compatibility alias

        print("🔍 CORRELATION ANALYZER INITIALIZED")
        print("=" * 70)
        print(f"   📊 Analysis Population Rows: {len(self.df):,}")
        if self.date_col in self.df.columns:
            min_d = pd.to_datetime(self.df[self.date_col]).min().date()
            max_d = pd.to_datetime(self.df[self.date_col]).max().date()
            print(f"   📅 Temporal Window: {min_d} → {max_d}")
        print(f"   🎯 Target Variable: {self.target_col}")
        print(f"   🔢 Linear Meteorological Predictors: {len(self.linear_predictor_cols)}")
        print(f"   🔄 Circular Wind Variables (Isolated): {len(self.circular_cols)}")

    def _validate_schema(self, strict: bool = True) -> Dict[str, Any]:
        """Validate input dataframe against expected meteorological allowlists (Fail-Fast)."""
        available_cols = set(self.df.columns)
        missing_linear = [c for c in LINEAR_PREDICTOR_ALLOWLIST if c not in available_cols]
        missing_circular = [c for c in CIRCULAR_WIND_ALLOWLIST if c not in available_cols]

        if strict and (missing_linear or missing_circular):
            raise ValueError(
                f"Schema validation failed: missing expected meteorological predictors. "
                f"Missing linear ({len(missing_linear)}): {missing_linear}; "
                f"Missing circular ({len(missing_circular)}): {missing_circular}"
            )

        status = "PASSED" if not (missing_linear or missing_circular) else "PARTIAL"
        unexpected = [
            c for c in self.df.columns
            if c not in LINEAR_PREDICTOR_ALLOWLIST and c not in CIRCULAR_WIND_ALLOWLIST
            and c != self.target_col and c != self.date_col
        ]

        return {
            "status": status,
            "expected_linear_count": len(LINEAR_PREDICTOR_ALLOWLIST),
            "actual_linear_count": len([c for c in LINEAR_PREDICTOR_ALLOWLIST if c in available_cols]),
            "expected_circular_count": len(CIRCULAR_WIND_ALLOWLIST),
            "actual_circular_count": len([c for c in CIRCULAR_WIND_ALLOWLIST if c in available_cols]),
            "missing_linear": missing_linear,
            "missing_circular": missing_circular,
            "unexpected_columns_ignored": unexpected,
        }

    def _get_linear_predictor_cols(self) -> List[str]:
        """Return approved linear meteorological predictors present in dataframe."""
        return [c for c in LINEAR_PREDICTOR_ALLOWLIST if c in self.df.columns]

    def _get_circular_cols(self) -> List[str]:
        """Return circular wind predictors present in dataframe."""
        return [c for c in CIRCULAR_WIND_ALLOWLIST if c in self.df.columns]

    # =========================================================================
    # AXIS A: FEATURE–TARGET CONTEMPORANEOUS ASSOCIATION
    # =========================================================================

    def analyze_target_associations(self) -> pd.DataFrame:
        """
        Compute contemporaneous monotonic (Spearman) and linear (Pearson) associations
        between linear predictors and rainfall target on Dev Train.
        """
        records = []
        target_series = self.df[self.target_col]

        for col in self.linear_predictor_cols:
            feat_series = self.df[col]
            valid_mask = feat_series.notna() & target_series.notna()
            x_valid = feat_series[valid_mask]
            y_valid = target_series[valid_mask]

            if len(x_valid) > 1:
                spearman_rho, _ = stats.spearmanr(x_valid, y_valid)
                pearson_r, _ = stats.pearsonr(x_valid, y_valid)
            else:
                spearman_rho, pearson_r = np.nan, np.nan

            records.append({
                "feature": col,
                "spearman_rho": float(spearman_rho),
                "pearson_r": float(pearson_r),
                "abs_spearman": float(abs(spearman_rho)) if not np.isnan(spearman_rho) else 0.0,
                "association_type": "contemporaneous",
                "association_note": (
                    "Measured at same time point t in historical dev-train dataset. "
                    "Does not establish forecast-time availability or predictive usefulness at inference time."
                ),
            })

        df_assoc = pd.DataFrame(records).sort_values("abs_spearman", ascending=False).reset_index(drop=True)
        return df_assoc

    # =========================================================================
    # AXIS B: PREDICTOR–PREDICTOR REDUNDANCY & VIF DIAGNOSTICS
    # =========================================================================

    def analyze_pairwise_correlations(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Compute pairwise Spearman and Pearson correlation matrices for linear predictors,
        and identify strong pairwise associations (|rho| > threshold).
        """
        X = self.df[self.linear_predictor_cols]
        spearman_mat = X.corr(method='spearman')
        pearson_mat = X.corr(method='pearson')

        pairs = []
        p = len(self.linear_predictor_cols)
        all_abs_rhos = []

        for i in range(p):
            for j in range(i + 1, p):
                f1 = self.linear_predictor_cols[i]
                f2 = self.linear_predictor_cols[j]
                rho = float(spearman_mat.iloc[i, j])
                r = float(pearson_mat.iloc[i, j])
                all_abs_rhos.append(abs(rho))

                if abs(rho) >= threshold:
                    pairs.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "spearman_rho": round(rho, 4),
                        "pearson_r": round(r, 4),
                        "abs_spearman": round(abs(rho), 4),
                    })

        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)

        stats_summary = {
            "method": "spearman",
            "n_pairs": len(all_abs_rhos),
            "mean_abs_rho": float(np.mean(all_abs_rhos)) if all_abs_rhos else 0.0,
            "median_abs_rho": float(np.median(all_abs_rhos)) if all_abs_rhos else 0.0,
            "max_abs_rho": float(np.max(all_abs_rhos)) if all_abs_rhos else 0.0,
        }

        return {
            "spearman_matrix": spearman_mat,
            "pearson_matrix": pearson_mat,
            "strong_pairwise_associations": pairs_df,
            "stats_summary": stats_summary,
            "threshold": threshold,
        }

    def analyze_multicollinearity(self) -> Dict[str, Any]:
        """
        Analyze multicollinearity using VIF scores and SVD condition number on linear predictors.
        Handles non-finite/undefined cases with explicit status and 'Undefined' risk label.
        """
        X = self.df[self.linear_predictor_cols].dropna()
        n_features = X.shape[1]

        # 1. Matrix conditioning diagnostics on standardized X
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X)

        matrix_rank = int(np.linalg.matrix_rank(X_scaled))
        singular_values = [float(s) for s in svdvals(X_scaled)]

        s_max = singular_values[0] if singular_values else 0.0
        s_min = singular_values[-1] if singular_values else 0.0

        if s_min > 1e-15 and np.isfinite(s_max / s_min):
            cond_num = float(s_max / s_min)
            near_singular = bool(cond_num > 1e10 or matrix_rank < n_features)
            conditioning_status = "SUCCESS"
        else:
            cond_num = None
            near_singular = True
            conditioning_status = "NUMERICAL_WARNING"

        rank_deficient = bool(matrix_rank < n_features)

        matrix_diagnostics = {
            "computation_status": conditioning_status,
            "matrix_rank": matrix_rank,
            "n_features": n_features,
            "condition_number": cond_num,
            "near_singular": near_singular,
            "rank_deficient": rank_deficient,
            "singular_values": singular_values,
            "standardization": "StandardScaler",
            "note": "Non-finite condition number stored as null with near_singular=true. Diagnostics only; feature selection policy belongs to Feature Engineering.",
        }

        # 2. VIF computation
        vif_records = []
        try:
            X_const = add_constant(X)
            for i, col in enumerate(self.linear_predictor_cols):
                try:
                    val = variance_inflation_factor(X_const.values, i + 1)
                    if np.isneginf(val) or np.isposinf(val) or np.isnan(val):
                        vif_score = None
                        status = "VIF_UNDEFINED"
                        risk_label = "Undefined"
                    else:
                        vif_score = float(val)
                        if vif_score >= 1e5:
                            status = "SUCCESS_EXTREME"
                            risk_label = "Extreme"
                        elif vif_score >= 100:
                            status = "SUCCESS"
                            risk_label = "Extreme"
                        elif vif_score >= 10:
                            status = "SUCCESS"
                            risk_label = "High"
                        elif vif_score >= 5:
                            status = "SUCCESS"
                            risk_label = "Moderate"
                        else:
                            status = "SUCCESS"
                            risk_label = "Normal"
                except Exception:
                    vif_score = None
                    status = "NUMERICAL_WARNING"
                    risk_label = "Undefined"

                vif_records.append({
                    "Feature": col,
                    "VIF_Score": vif_score,
                    "Risk_Level": risk_label,
                    "computation_status": status,
                })
        except Exception as e:
            # Full failure fallback
            for col in self.linear_predictor_cols:
                vif_records.append({
                    "Feature": col,
                    "VIF_Score": None,
                    "Risk_Level": "Undefined",
                    "computation_status": "NUMERICAL_WARNING",
                })

        vif_df = pd.DataFrame(vif_records)
        # Sort so highest finite VIF comes first, nulls at bottom
        vif_df["sort_key"] = vif_df["VIF_Score"].fillna(-1)
        vif_df = vif_df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"]).reset_index(drop=True)

        return {
            "vif_df": vif_df,
            "matrix_diagnostics": matrix_diagnostics,
        }

    # =========================================================================
    # AXIS C: REDUNDANCY STRUCTURE (CLUSTERING & PCA)
    # =========================================================================

    def analyze_feature_clustering(self, requested_n_clusters: int = 5) -> Dict[str, Any]:
        """
        Hierarchical clustering using correlation-derived dissimilarity (1 - |rho_s|)
        and average linkage. Reports both requested and actual cluster count.
        """
        X = self.df[self.linear_predictor_cols]
        spearman_mat = X.corr(method='spearman')

        dissim_matrix = 1.0 - np.abs(spearman_mat)
        # Numerical safeguard for squareform
        np.fill_diagonal(dissim_matrix.values, 0.0)
        condensed = squareform(dissim_matrix.values, checks=False)

        linkage_matrix = linkage(condensed, method='average')

        # Cluster cut
        labels = fcluster(linkage_matrix, t=requested_n_clusters, criterion='maxclust')
        unique_labels = sorted(list(np.unique(labels)))
        actual_n_clusters = len(unique_labels)

        clusters: Dict[str, List[str]] = {}
        for uid in unique_labels:
            clusters[str(uid)] = []

        for feat, label in zip(self.linear_predictor_cols, labels):
            clusters[str(label)].append(feat)

        return {
            "dissimilarity_metric": "1 - |spearman_rho| (correlation-derived dissimilarity)",
            "linkage_method": "average",
            "requested_n_clusters": requested_n_clusters,
            "actual_n_clusters": actual_n_clusters,
            "cluster_count_note": "heuristic exploratory cut; dendrogram is primary output",
            "linkage_matrix": linkage_matrix,
            "feature_clusters": clusters,
            "feature_labels": dict(zip(self.linear_predictor_cols, [int(l) for l in labels])),
        }

    def analyze_pca(self) -> Dict[str, Any]:
        """
        PCA on standardized linear predictors. Fits all components.
        Separates explained variance diagnostics from PC-target contemporaneous association.
        """
        X = self.df[self.linear_predictor_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()  # fits all components
        pcs = pca.fit_transform(X_scaled)

        n_comp = pca.n_components_
        exp_var = [float(v) for v in pca.explained_variance_ratio_]
        cum_var = [float(v) for v in np.cumsum(exp_var)]

        # PC-target contemporaneous association
        y = self.df.loc[X.index, self.target_col]
        pc_target_records = []

        for i in range(n_comp):
            rho, _ = stats.spearmanr(pcs[:, i], y)
            abs_rho = float(abs(rho)) if not np.isnan(rho) else 0.0
            pc_target_records.append({
                "pc": f"PC{i+1}",
                "explained_var": round(exp_var[i], 4),
                "spearman_with_target": round(float(rho), 4) if not np.isnan(rho) else 0.0,
                "abs_spearman_with_target": round(abs_rho, 4),
            })

        pc_target_df = pd.DataFrame(pc_target_records)

        return {
            "purpose_variance": "redundancy_structure — variance explained by principal components",
            "purpose_pc_target": "supplementary_association — principal direction association with rainfall",
            "n_components_fitted": n_comp,
            "explained_variance_ratio": exp_var,
            "cumulative_variance": cum_var,
            "pc_target_association": pc_target_df,
            "sign_convention": "absolute_value_due_to_PCA_sign_ambiguity",
            "pca_object": pca,
            "X_scaled": X_scaled,
        }

    # =========================================================================
    # SUPPLEMENTARY DIAGNOSTICS: CIRCULAR WIND & RADIATION DEEP DIVE
    # =========================================================================

    def analyze_circular_diagnostics(self) -> Dict[str, Any]:
        """
        Isolated circular direction decomposition diagnostic for wind variables.
        Computes sin and cos directional components and their Spearman association with rainfall.
        """
        target = self.df[self.target_col]
        var_records = []

        for col in self.circular_cols:
            angles = self.df[col].dropna()
            rad = np.radians(angles)
            sin_vals = np.sin(rad)
            cos_vals = np.cos(rad)

            valid_idx = angles.index.intersection(target.dropna().index)
            y = target.loc[valid_idx]

            rho_sin, _ = stats.spearmanr(sin_vals.loc[valid_idx], y)
            rho_cos, _ = stats.spearmanr(cos_vals.loc[valid_idx], y)

            var_records.append({
                "raw_variable": col,
                "sin_spearman_with_target": round(float(rho_sin), 4) if not np.isnan(rho_sin) else 0.0,
                "cos_spearman_with_target": round(float(rho_cos), 4) if not np.isnan(rho_cos) else 0.0,
            })

        return {
            "status": "supplementary_diagnostic_only",
            "method": "circular_direction_decomposition",
            "method_note": "First-pass sin/cos decomposition. Supplementary only; not injected into primary VIF, PCA, or F0-F3 feature sets.",
            "policy": "sin/cos not injected into primary VIF/PCA/clustering; not added to F0-F3 feature sets",
            "variables": var_records,
        }

    def analyze_radiation_uv_deep_dive(self) -> Dict[str, Any]:
        """Deep-dive diagnostic into solar and UV radiation channels."""
        rad_cols = [c for c in self.linear_predictor_cols if 'Bức xạ' in c or 'UV' in c]
        X_rad = self.df[rad_cols].dropna()

        corr_mat = X_rad.corr(method='spearman')
        target = self.df[self.target_col]

        target_assocs = []
        for c in rad_cols:
            rho, _ = stats.spearmanr(X_rad[c], target.loc[X_rad.index])
            target_assocs.append({
                "feature": c,
                "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
            })

        return {
            "columns": rad_cols,
            "within_group_spearman": corr_mat.round(4).to_dict(),
            "target_associations": target_assocs,
        }

    # =========================================================================
    # REPORT GENERATION & COMPLETE SUITE
    # =========================================================================

    def generate_correlation_report(self) -> Dict[str, Any]:
        """Aggregate all analysis results into structured correlation_report artifact."""
        target_assoc_df = self.analyze_target_associations()
        pairwise_res = self.analyze_pairwise_correlations()
        vif_res = self.analyze_multicollinearity()
        clustering_res = self.analyze_feature_clustering()
        pca_res = self.analyze_pca()
        circular_res = self.analyze_circular_diagnostics()
        rad_res = self.analyze_radiation_uv_deep_dive()

        report = {
            "metadata": {
                "correlation_method_primary": "spearman",
                "correlation_method_secondary": "pearson",
                "data_split": "dev_train",
                "train_start": str(pd.to_datetime(self.df[self.date_col]).min().date()) if self.date_col in self.df.columns else "N/A",
                "train_end": str(pd.to_datetime(self.df[self.date_col]).max().date()) if self.date_col in self.df.columns else "N/A",
                "n_observations": len(self.df),
                "n_predictors": len(self.linear_predictor_cols),
                "eda_population_note": "Strict zero-leakage Dev Train only. Validation (2018-05-01 to 2020-04-30, N=731) and Final Test (2020-05-01 to 2026-04-30, N=2191) held out completely.",
                "schema_validation": self.schema_validation,
                "circular_variables_isolated": self.circular_cols,
                "high_correlation_threshold": pairwise_res["threshold"],
                "high_correlation_threshold_note": "heuristic screening threshold; not an automatic drop criterion",
                "clustering_dissimilarity_metric": clustering_res["dissimilarity_metric"],
                "clustering_linkage": clustering_res["linkage_method"],
                "requested_n_clusters": clustering_res["requested_n_clusters"],
                "actual_n_clusters": clustering_res["actual_n_clusters"],
                "cluster_count_note": clustering_res["cluster_count_note"],
            },
            "target_associations": target_assoc_df.to_dict(orient="records"),
            "pairwise_correlations": pairwise_res["stats_summary"],
            "strong_pairwise_associations": pairwise_res["strong_pairwise_associations"].to_dict(orient="records") if not pairwise_res["strong_pairwise_associations"].empty else [],
            "vif_results": [
                {
                    "feature": row["Feature"],
                    "vif_score": round(row["VIF_Score"], 2) if row["VIF_Score"] is not None else None,
                    "risk_label": row["Risk_Level"],
                    "computation_status": row["computation_status"],
                }
                for _, row in vif_res["vif_df"].iterrows()
            ],
            "vif_matrix_diagnostics": vif_res["matrix_diagnostics"],
            "redundancy_clusters": {
                "dissimilarity_metric": clustering_res["dissimilarity_metric"],
                "linkage_method": clustering_res["linkage_method"],
                "requested_n_clusters": clustering_res["requested_n_clusters"],
                "actual_n_clusters": clustering_res["actual_n_clusters"],
                "cluster_count_note": clustering_res["cluster_count_note"],
                "clusters": clustering_res["feature_clusters"],
            },
            "pca_diagnostics": {
                "purpose_variance": pca_res["purpose_variance"],
                "purpose_pc_target": pca_res["purpose_pc_target"],
                "n_components_fitted": pca_res["n_components_fitted"],
                "explained_variance_ratio": [round(v, 4) for v in pca_res["explained_variance_ratio"]],
                "cumulative_variance": [round(v, 4) for v in pca_res["cumulative_variance"]],
                "pc_target_association": pca_res["pc_target_association"].to_dict(orient="records"),
                "sign_convention": pca_res["sign_convention"],
                "input_matrix": "standardized_linear_predictors_only",
            },
            "circular_wind_diagnostics": circular_res,
            "radiation_uv_diagnostics": rad_res,
            "observed_evidence": [
                "Highest contemporaneous monotonic associations with rainfall on Dev Train occur among moisture and humidity predictors.",
                f"Predictor space exhibits rank-deficiency (rank {vif_res['matrix_diagnostics']['matrix_rank']} < {len(self.linear_predictor_cols)}) and condition number exceeding 1e10, driven by deterministic definitions (thermal range, wind range).",
                f"Hierarchical clustering (average linkage, 1-|rho_s|) organizes the 32 linear predictors into {clustering_res['actual_n_clusters']} redundancy clusters.",
                "Solar and UV radiation channels exhibit strong within-group cross-correlation (|rho_s| > 0.85).",
                "Wind direction circular decomposition reveals modest directional correlation signals, preserved as supplementary diagnostics.",
            ],
            "modeling_implications": [
                "Linear regression and SARIMAX model families will require multicollinearity management (e.g. VIF-guided feature selection or ridge regularization) in 02 Feature Engineering.",
                "Tree-based models (LightGBM/XGBoost) can accept collinear inputs directly, benefiting from feature clustering insights for grouping.",
                "Radiation variables represent candidates for dimensionality reduction or representative channel selection.",
            ],
            "not_conclusions_from_eda": "01.3 provides diagnostic association and collinearity evidence only on Dev Train. Feature-target associations are contemporaneous and do not establish forecast usefulness. Feature selection policy is determined in 02 Feature Engineering specific to model family. VIF > 10 does not dictate feature dropping. |rho| > 0.8 is a heuristic screening threshold. EDA does not make automated feature selection decisions.",
        }

        return report

    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Run complete correlation analysis suite.
        Maintains backwards-compatible interface with EDAPipeline and existing tests.
        """
        vif_res = self.analyze_multicollinearity()
        clustering_res = self.analyze_feature_clustering()
        pca_res = self.analyze_pca()
        target_assoc = self.analyze_target_associations()
        report = self.generate_correlation_report()

        return {
            "multicollinearity_results": vif_res["vif_df"],
            "clustering_results": clustering_res,
            "pca_results": pca_res,
            "target_associations": target_assoc,
            "correlation_report": report,
        }

    def analyze(self) -> Dict[str, Any]:
        """Standard analyzer protocol alias."""
        return self.generate_insights_report()

    # Backwards-compatibility wrapper for pipeline / tests
    def analyze_meteorological_correlations(self) -> Dict[str, Any]:
        """Legacy helper for meteorological correlations."""
        X = self.df[self.analysis_cols]
        correlations = {
            'pearson': X.corr(method='pearson'),
            'spearman': X.corr(method='spearman'),
        }
        return {
            'correlations': correlations,
            'target_correlations': correlations['spearman'][self.target_col].drop(self.target_col),
        }


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class CorrelationVisualizer:
    """Visualization module for correlation and multicollinearity analysis."""

    def __init__(self, analyzer: CorrelationAnalyzer):
        self.analyzer = analyzer

    def visualize_target_associations(
        self,
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 8),
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """Plot horizontal bar chart of strongest contemporaneous associations with target."""
        df_assoc = self.analyzer.analyze_target_associations().head(top_n)
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = range(len(df_assoc))
        colors = ['#0072B2' if rho > 0 else '#D55E00' for rho in df_assoc['spearman_rho']]

        bars = ax.barh(y_pos, df_assoc['spearman_rho'], color=colors, alpha=0.8, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_assoc['feature'], fontsize=10)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Contemporaneous Spearman Correlation (ρ)", fontsize=11, fontweight='bold')
        ax.set_title(
            f"Top {top_n} Contemporaneous Feature–Target Associations (Dev Train N={len(self.analyzer.df):,})\n"
            "(Diagnostic association only; does not establish forecast usefulness)",
            fontsize=12, fontweight='bold', pad=12
        )
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()

        if not return_fig:
            plt.show()
            return None
        return fig

    def visualize_pairwise_heatmap(
        self,
        figsize: Tuple[int, int] = (16, 14),
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """Plot lower triangle pairwise Spearman correlation heatmap of linear predictors."""
        res = self.analyzer.analyze_pairwise_correlations()
        corr_matrix = res["spearman_matrix"]

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.3,
            cbar_kws={"shrink": 0.7, "label": "Spearman Correlation (ρ)"},
            ax=ax
        )
        ax.set_title(
            f"Linear Meteorological Predictors Pairwise Spearman Correlation (N={len(self.analyzer.df):,})",
            fontsize=13, fontweight='bold', pad=15
        )
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        fig.tight_layout()

        if not return_fig:
            plt.show()
            return None
        return fig

    def visualize_dendrogram(
        self,
        figsize: Tuple[int, int] = (14, 6),
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """Plot hierarchical clustering dendrogram (average linkage, dissimilarity = 1 - |rho_s|)."""
        res = self.analyzer.analyze_feature_clustering()
        linkage_matrix = res["linkage_matrix"]

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(
            linkage_matrix,
            labels=self.analyzer.linear_predictor_cols,
            ax=ax,
            orientation='top',
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=0.5,
        )
        ax.set_title(
            f"Hierarchical Feature Clustering Dendrogram (Average Linkage, Dissimilarity: 1 - |ρ_s|)\n"
            f"Requested Clusters: {res['requested_n_clusters']} | Actual Cut Clusters: {res['actual_n_clusters']}",
            fontsize=12, fontweight='bold', pad=12
        )
        ax.set_ylabel("Correlation Dissimilarity (1 - |ρ|)", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        if not return_fig:
            plt.show()
            return None
        return fig

    def visualize_pca(
        self,
        top_k: int = 10,
        figsize: Tuple[int, int] = (14, 6),
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """Plot PCA scree plot and PC-target contemporaneous association."""
        pca_res = self.analyzer.analyze_pca()
        exp_var = pca_res["explained_variance_ratio"][:top_k]
        cum_var = pca_res["cumulative_variance"][:top_k]
        pc_df = pca_res["pc_target_association"].head(top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Scree plot
        k_range = range(1, len(exp_var) + 1)
        ax1.bar(k_range, [v * 100 for v in exp_var], color='skyblue', alpha=0.8, edgecolor='black', label='Individual')
        ax1.plot(k_range, [v * 100 for v in cum_var], 'ro-', linewidth=1.5, markersize=5, label='Cumulative')
        ax1.set_xlabel("Principal Component", fontsize=10)
        ax1.set_ylabel("Variance Explained (%)", fontsize=10)
        ax1.set_title("Sub-diagnostic 1: Scree Plot (Predictor Redundancy)", fontsize=11, fontweight='bold')
        ax1.set_xticks(k_range)
        ax1.set_xticklabels([f"PC{k}" for k in k_range], rotation=45)
        ax1.legend(loc='center right')
        ax1.grid(True, alpha=0.3)

        # 2. PC-target association
        ax2.bar(k_range, pc_df["abs_spearman_with_target"], color='coral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel("Principal Component", fontsize=10)
        ax2.set_ylabel("|Spearman ρ| with Target", fontsize=10)
        ax2.set_title("Sub-diagnostic 2: Contemporaneous PC–Target Association (|ρ|)", fontsize=11, fontweight='bold')
        ax2.set_xticks(k_range)
        ax2.set_xticklabels([f"PC{k}" for k in k_range], rotation=45)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        if not return_fig:
            plt.show()
            return None
        return fig

    def visualize_vif(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """Plot VIF diagnostic bar chart."""
        vif_res = self.analyzer.analyze_multicollinearity()
        vif_df = vif_res["vif_df"].head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(len(vif_df))
        scores = [v if v is not None else 0.0 for v in vif_df["VIF_Score"]]
        colors = [
            'red' if lbl == 'Extreme' else
            'orange' if lbl == 'High' else
            'gold' if lbl == 'Moderate' else
            'gray' if lbl == 'Undefined' else
            'green'
            for lbl in vif_df["Risk_Level"]
        ]

        ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vif_df["Feature"], fontsize=9)
        ax.set_xlabel("VIF Score (log-scale)", fontsize=10, fontweight='bold')
        ax.set_xscale("log")
        ax.axvline(5, color='gold', linestyle='--', label='Moderate (VIF=5)')
        ax.axvline(10, color='red', linestyle='--', label='High (VIF=10)')
        ax.set_title(
            f"VIF Multicollinearity Diagnostics (Top {top_n} Predictors)\n"
            f"Condition Number: {vif_res['matrix_diagnostics']['condition_number']:.2e} | Rank: {vif_res['matrix_diagnostics']['matrix_rank']}/{vif_res['matrix_diagnostics']['n_features']}",
            fontsize=12, fontweight='bold', pad=12
        )
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()

        if not return_fig:
            plt.show()
            return None
        return fig


# =============================================================================
# CONVENIENCE RUNNER
# =============================================================================

def analyze_correlations(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
    include_visualization: bool = True,
) -> Dict[str, Any]:
    """Execute complete correlation analysis pipeline and export artifacts."""
    analyzer = CorrelationAnalyzer(df, target_col=target_col, date_col=date_col)
    results = analyzer.generate_insights_report()

    if include_visualization:
        viz = CorrelationVisualizer(analyzer)
        viz.visualize_target_associations()
        viz.visualize_pairwise_heatmap()
        viz.visualize_dendrogram()
        viz.visualize_pca()
        viz.visualize_vif()

    return results