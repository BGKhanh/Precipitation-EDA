"""Publication-Ready Visualizations for Post-Modelling Result Analysis.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and CR-11:
- Consistent color palettes and typography.
- High-resolution outputs (300 DPI PNG & PDF).
- All figures saved centrally to reports/validation_benchmark/figures/.
- Mandatory footer/annotation on Hurdle Strategy B indicating validation-stage optimism.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_FIG_DIR = Path("reports/validation_benchmark/figures")

# Styling configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

FAMILY_COLORS = {
    "Heuristic Baseline": "#7f7f7f",
    "Classical Statistical": "#1f77b4",
    "Intermittent Demand": "#aec7e8",
    "Tabular ML Direct": "#2ca02c",
    "Tabular ML Tweedie": "#98df8a",
    "Deep Neural": "#d62728",
    "Two-Stage Hurdle": "#ff7f0e",
}

HURDLE_OPTIMISM_NOTE = (
    "Note: Hurdle Strategy B results reflect validation-stage optimism due to threshold tau*\n"
    "being tuned on the validation set; unbiased comparison is established on the test set."
)


def _ensure_dir(save_path: Optional[Path | str]) -> Optional[Path]:
    if save_path is None:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_official_leaderboard(
    df_ranked: pd.DataFrame,
    top_n: int = 20,
    save_name: Optional[str] = "fig01_top_leaderboard.png",
) -> plt.Figure:
    """Plot horizontal bar chart of the Top N models on the official leaderboard."""
    top_df = df_ranked.head(top_n).copy().sort_values("MAE", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), dpi=300)
    colors = [FAMILY_COLORS.get(fam, "#333333") for fam in top_df["family"]]

    bars = ax.barh(top_df["scenario_id"] + " (" + top_df["model"] + ")", top_df["MAE"], color=colors, alpha=0.85, edgecolor="#222222")

    # Add numeric labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f} mm", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Pooled Validation MAE (mm, lower is better)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top {top_n} Forecasters — Official Validation Leaderboard (724 Origins, H=7)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="x")
    ax.set_xlim(0, top_df["MAE"].max() * 1.15)

    # Add note if Hurdle B is present
    has_hurdle_b = (top_df["strategy"] == "Hard-Gated Regime").any()
    if has_hurdle_b:
        fig.text(0.12, -0.02, HURDLE_OPTIMISM_NOTE, fontsize=7, color="#555555", style="italic")

    plt.tight_layout()
    if save_name:
        out_p = _ensure_dir(DEFAULT_FIG_DIR / save_name)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    return fig


def plot_horizon_metric_curves(
    df_horizons: pd.DataFrame,
    scenarios: List[str],
    metric: str = "MAE",
    save_name: Optional[str] = "fig02_horizon_curves.png",
) -> plt.Figure:
    """Plot error progression across horizons h=1..7 for selected representative scenarios."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    horizons = list(range(1, 8))

    for sc in scenarios:
        sub = df_horizons[df_horizons["scenario_id"] == sc]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        y_vals = [row[f"{metric}@h{h}"] for h in horizons]
        fam = row.get("family", "Unknown")
        color = FAMILY_COLORS.get(fam, "#333333")
        label = f"{sc} ({row['model']})"
        ax.plot(horizons, y_vals, marker="o", label=label, color=color, linewidth=1.8, markersize=5)

    ax.set_xlabel("Forecast Horizon h (Days Ahead)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{metric} (mm)", fontsize=10, fontweight="bold")
    ax.set_title(f"Error Accumulation by Horizon ({metric}@h1 to {metric}@h7)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(horizons)
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    if save_name:
        out_p = _ensure_dir(DEFAULT_FIG_DIR / save_name)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_ablation_matrix(
    df_ablation: pd.DataFrame,
    save_name: Optional[str] = "fig03_feature_ablation.png",
) -> plt.Figure:
    """Plot marginal contribution delta MAE across successive feature sets F0..F3."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_ablation) * 0.45)), dpi=300)

    disp = df_ablation.copy().sort_values("best_MAE", ascending=False)
    y_labels = disp["algorithm"] + " (" + disp["strategy"] + ")"

    x = np.arange(len(disp))
    w = 0.22

    ax.barh(x + w, disp["delta_F1_minus_F0"], height=w, label="Δ(F1 - F0): Exogenous Weather", color="#2ca02c", alpha=0.85)
    ax.barh(x, disp["delta_F2_minus_F1"], height=w, label="Δ(F2 - F1): Rolling & Lags", color="#1f77b4", alpha=0.85)
    ax.barh(x - w, disp["delta_F3_minus_F2"], height=w, label="Δ(F3 - F2): Calendar & Season", color="#ff7f0e", alpha=0.85)

    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_yticks(x)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Marginal Δ MAE (mm, negative indicates improvement)", fontsize=10, fontweight="bold")
    ax.set_title("Feature Set Ablation — Marginal Contributions F0 → F1 → F2 → F3", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="x")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    if save_name:
        out_p = _ensure_dir(DEFAULT_FIG_DIR / save_name)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    return fig


def plot_hurdle_ablation_comparison(
    df_hurdle: pd.DataFrame,
    save_name: Optional[str] = "fig04_hurdle_ablation.png",
) -> plt.Figure:
    """Plot pairwise comparison between Hurdle Strategy A and Strategy B across 16 matched pairs."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    disp = df_hurdle.copy().sort_values("delta_MAE_B_minus_A")
    x = np.arange(len(disp))
    w = 0.38

    ax.bar(x - w / 2, disp["MAE_StratA"], width=w, label="Strategy A (Mixture Expectation)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, disp["MAE_StratB"], width=w, label="Strategy B (Hard-Gated Regime)", color="#ff7f0e", alpha=0.85)

    labels = disp["base_family"] + "_" + disp["feature_set"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pooled Validation MAE (mm)", fontsize=10, fontweight="bold")
    ax.set_title("Hurdle Strategy Head-to-Head Comparison (16 Matched Pairs)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="y")
    ax.legend(fontsize=9, loc="upper left")

    fig.text(0.12, -0.05, HURDLE_OPTIMISM_NOTE, fontsize=7, color="#555555", style="italic")

    plt.tight_layout()
    if save_name:
        out_p = _ensure_dir(DEFAULT_FIG_DIR / save_name)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    return fig


def plot_pareto_frontiers(
    df_res: pd.DataFrame,
    x_col: str = "latency_ms_per_origin",
    y_col: str = "MAE",
    save_name: Optional[str] = "fig07_pareto.png",
) -> plt.Figure:
    """Plot 2D Accuracy vs Cost Pareto trade-off scatter plot with step frontier line."""
    from src.analysis.pareto import identify_pareto_frontier

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Scatter of all valid models
    for fam, grp in df_res.groupby("family"):
        color = FAMILY_COLORS.get(fam, "#333333")
        ax.scatter(grp[x_col], grp[y_col], label=fam, color=color, alpha=0.7, s=40, edgecolors="none")

    # Overlay Pareto frontier
    pareto_df = identify_pareto_frontier(df_res, x_col=x_col, y_col=y_col)
    if len(pareto_df) > 0:
        ax.step(pareto_df[x_col], pareto_df[y_col], where="post", color="#e41a1c", linewidth=2.0, linestyle="--", label="Pareto Frontier")
        # Annotate frontier models
        for _, row in pareto_df.iterrows():
            ax.annotate(row["scenario_id"], (row[x_col], row[y_col]), textcoords="offset points", xytext=(4, 4), fontsize=7, fontweight="bold")

    x_label = "Inference Latency (ms/origin)" if "latency" in x_col else "Training Time (seconds)"
    ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Pooled Validation MAE (mm)", fontsize=10, fontweight="bold")
    ax.set_title(f"Accuracy vs Cost Pareto Frontier ({y_col} vs {x_label})", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_name:
        out_p = _ensure_dir(DEFAULT_FIG_DIR / save_name)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    return fig

