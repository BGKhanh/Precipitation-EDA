"""Official Leaderboard Ranking Engine according to Protocol v1.0.21 §9.3 & §18.

DS108 Rainfall Forecasting Project.
Strictly implements the official candidate tiering and tie-breaking contracts:
1. Primary ranking metric: Pooled MAE across 724 validation origins and 7 horizons (N=5,068 points).
2. Candidate tiering: ε = 10^-4 mm (0.0001 mm) tolerance relative to the minimum Pooled MAE of remaining candidates.
3. Within-tier tie-breaking: Pooled RMSE ascending on unrounded floating point values (no secondary epsilon).
4. Absolute RMSE tie-breaking: Canonical Model ID lexicographical ascending order (case-insensitive).
5. Shared rank assignment: Models in the same tier receive the rank equal to the actual starting index of that tier.
6. Scenario ID is strictly for display/presentation; NEVER used as a ranking criterion.
7. Only COMPLETED scenarios with finite MAE and RMSE are admitted to S_valid.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_TIER_EPSILON = 1e-4  # 0.0001 mm per Protocol §9.3 & §18


def compute_official_leaderboard(
    df_results: pd.DataFrame,
    epsilon: float = CANONICAL_TIER_EPSILON,
) -> pd.DataFrame:
    """Compute the official ranking leaderboard strictly adhering to Protocol §9.3 & §18.
    
    Args:
        df_results: DataFrame containing benchmark results with at least:
                    ['scenario_id', 'canonical_model_id', 'family', 'strategy', 'feature_set',
                     'MAE', 'MSE', 'RMSE', 'MASE', 'status']
        epsilon: Tiering tolerance on Pooled MAE (default 1e-4 mm).
        
    Returns:
        DataFrame with columns:
            ['rank', 'tier_id', 'scenario_id', 'model', 'canonical_model_id',
             'family', 'strategy', 'feature_set', 'MAE', 'RMSE', 'MASE', 'status']
    """
    df = df_results.copy()

    # 1. Filter valid completed scenarios (S_valid)
    valid_mask = (
        (df["status"] == "COMPLETED")
        & df["MAE"].notna()
        & df["RMSE"].notna()
        & np.isfinite(df["MAE"])
        & np.isfinite(df["RMSE"])
    )
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        logger.warning("No valid completed scenarios found to rank!")
        return pd.DataFrame()

    # Ensure canonical_model_id exists
    if "canonical_model_id" not in df_valid.columns:
        if "model" in df_valid.columns:
            df_valid["canonical_model_id"] = df_valid["model"].str.lower()
        else:
            df_valid["canonical_model_id"] = df_valid["scenario_id"].str.lower()

    # 2. Iterative candidate tiering algorithm
    ranked_records: List[Dict[str, Any]] = []
    remaining = df_valid.copy()
    current_rank = 1
    tier_idx = 1

    while len(remaining) > 0:
        # Find minimum MAE in remaining set
        min_mae = remaining["MAE"].min()
        tier_cutoff = min_mae + epsilon

        # Identify all scenarios in this tier
        tier_mask = remaining["MAE"] <= tier_cutoff
        tier_df = remaining[tier_mask].copy()

        # Sort within tier:
        # 1. Pooled RMSE ascending (unrounded float)
        # 2. Canonical Model ID lexicographical ascending
        tier_df["sort_id"] = tier_df["canonical_model_id"].astype(str).str.lower()
        tier_df = tier_df.sort_values(
            by=["RMSE", "sort_id"],
            ascending=[True, True],
        )

        # Assign ranks and record
        tier_size = len(tier_df)
        for _, row in tier_df.iterrows():
            rec = row.to_dict()
            rec["rank"] = current_rank  # Shared rank for tier
            rec["tier_id"] = f"Tier-{tier_idx}"
            rec["tier_size"] = tier_size
            rec["is_tied"] = tier_size > 1
            ranked_records.append(rec)

        # Advance rank by number of models in this tier
        current_rank += tier_size
        tier_idx += 1

        # Drop evaluated tier from remaining
        remaining = remaining[~tier_mask].copy()

    df_ranked = pd.DataFrame(ranked_records)

    # Order columns cleanly
    disp_cols = [
        "rank",
        "tier_id",
        "is_tied",
        "scenario_id",
        "model",
        "canonical_model_id",
        "family",
        "strategy",
        "feature_set",
        "MAE",
        "MSE",
        "RMSE",
        "MASE",
        "training_time",
        "inference_time",
        "selected_hurdle_threshold",
        "status",
    ]
    present_cols = [c for c in disp_cols if c in df_ranked.columns]
    df_ranked = df_ranked[present_cols].reset_index(drop=True)

    return df_ranked


def format_leaderboard_display(
    df_ranked: pd.DataFrame,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Format numeric columns of the leaderboard for clean tabular presentation."""
    df_disp = df_ranked.copy()
    if top_n is not None:
        df_disp = df_disp.head(top_n).copy()

    float_cols = {
        "MAE": "{:.4f}",
        "MSE": "{:.4f}",
        "RMSE": "{:.4f}",
        "MASE": "{:.4f}",
        "training_time": "{:.2f}s",
        "inference_time": "{:.4f}s",
    }
    for col, fmt in float_cols.items():
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "N/A")

    return df_disp

