"""Pareto Efficiency Frontier Analysis for Accuracy vs Cost Trade-offs.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and CR-06.

IMPORTANT NOTE:
Pareto frontiers are descriptive exploratory visualizations only. Per Protocol PROP-07,
resource metrics are never used to disqualify or down-rank any model.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def identify_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "MAE",
) -> pd.DataFrame:
    """Identify non-dominated scenarios on the 2D Pareto frontier (minimizing both x and y).
    
    A scenario (x, y) is on the Pareto frontier if no other scenario has both:
        x_other <= x  AND  y_other <= y (with at least one strict inequality).
        
    Args:
        df: Input DataFrame.
        x_col: Cost metric column (e.g. 'latency_ms_per_origin', 'training_time_sec', 'parameter_count').
        y_col: Error metric column (default: 'MAE').
        
    Returns:
        DataFrame containing only scenarios situated on the Pareto frontier,
        sorted by x_col ascending.
    """
    valid = df[[x_col, y_col]].dropna()
    valid_mask = np.isfinite(valid[x_col]) & np.isfinite(valid[y_col])
    sub = df.loc[valid.index[valid_mask]].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    # Sort primarily by x ascending, then by y ascending
    sub = sub.sort_values(by=[x_col, y_col], ascending=[True, True])

    pareto_indices = []
    min_y = float("inf")

    for idx, row in sub.iterrows():
        y_val = row[y_col]
        # Strictly lower error at this or higher cost
        if y_val < min_y:
            pareto_indices.append(idx)
            min_y = y_val

    df_pareto = sub.loc[pareto_indices].copy()
    df_pareto["pareto_rank"] = range(1, len(df_pareto) + 1)
    return df_pareto.reset_index(drop=True)

