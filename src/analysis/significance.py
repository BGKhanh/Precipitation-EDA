"""Diebold-Mariano Statistical Significance Testing with HLN and Holm Adjustments.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and CR-08:
1. Dynamic per-horizon evaluation: h = 1..7.
2. Autocorrelation truncation lag: k = h - 1.
3. Finite-sample correction: Harvey, Leybourne, and Newbold (1997) (HLN).
4. Degenerate case handling: If d_t == 0, return (stat=0.0, p_value=1.0).
5. Multiple testing adjustment: Holm-Bonferroni method across comparisons.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test_by_horizon(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int,
    loss_type: str = "absolute",
) -> Tuple[float, float]:
    """Compute the Diebold-Mariano test statistic with HLN finite-sample correction for horizon h.
    
    Args:
        e1: Forecast error vector of model 1 at horizon h (length T=724).
        e2: Forecast error vector of model 2 at horizon h (length T=724).
        h: Forecast horizon (1 <= h <= 7).
        loss_type: 'absolute' (|e|) or 'squared' (e^2). Default 'absolute' (matching MAE).
        
    Returns:
        Tuple of (dm_stat_hln, p_value_two_sided).
    """
    assert len(e1) == len(e2), f"Length mismatch: {len(e1)} vs {len(e2)}"
    T = len(e1)
    assert T > 0, "Empty error arrays"

    # Loss differential series
    if loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    elif loss_type == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Check for degeneracy (identical forecasts or zero variance)
    d_std = np.std(d, ddof=1)
    if d_std < 1e-12 or np.max(np.abs(d)) < 1e-9:
        return 0.0, 1.0

    d_mean = np.mean(d)

    # Autocovariance up to lag k = h - 1
    k = max(0, h - 1)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0

    for l in range(1, k + 1):
        gamma_l = np.mean((d[l:] - d_mean) * (d[:-l] - d_mean))
        gamma_sum += gamma_l

    # Spectral variance estimate
    var_d = gamma_0 + 2.0 * gamma_sum
    if var_d <= 0:
        var_d = gamma_0 if gamma_0 > 0 else 1e-8

    # Standard Diebold-Mariano statistic
    dm_stat = d_mean / np.sqrt(var_d / T)

    # Harvey, Leybourne, Newbold (1997) finite sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm_stat * hln_factor

    # Two-sided p-value from Student's t distribution with T-1 degrees of freedom
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=T - 1)))
    p_value = min(1.0, max(0.0, p_value))

    return float(dm_hln), p_value


def compute_champion_pairwise_dm(
    scenarios: List[str],
    forecasts_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    horizons: List[int] = [1, 3, 7],
    p_adjust: str = "holm",
) -> pd.DataFrame:
    """Compute pairwise Diebold-Mariano significance matrix across champion models for given horizons.
    
    Args:
        scenarios: List of scenario IDs to compare (e.g. ['SC-04', 'SC-13', 'SC-30', 'SC-39', 'SC-42']).
        forecasts_dict: Dictionary of forecast arrays of shape (724, 7).
        y_true: Ground truth target array of shape (724, 7).
        horizons: List of horizons to evaluate (default: [1, 3, 7]).
        p_adjust: Multiple comparison correction method ('holm' or 'none').
        
    Returns:
        DataFrame with pairwise comparison results, test statistics, and adjusted p-values.
    """
    records = []

    for h in horizons:
        h_idx = h - 1
        y_true_h = y_true[:, h_idx]

        for i in range(len(scenarios)):
            sc1 = scenarios[i]
            e1 = y_true_h - forecasts_dict[sc1][:, h_idx]

            for j in range(i + 1, len(scenarios)):
                sc2 = scenarios[j]
                e2 = y_true_h - forecasts_dict[sc2][:, h_idx]

                stat, p_val = diebold_mariano_test_by_horizon(e1, e2, h=h)
                records.append({
                    "horizon": f"h={h}",
                    "model_1": sc1,
                    "model_2": sc2,
                    "DM_HLN_stat": stat,
                    "p_value_raw": p_val,
                })

    df_dm = pd.DataFrame(records)

    # Multiple testing correction per horizon
    if p_adjust == "holm" and len(df_dm) > 0:
        adj_p_list = []
        for h_label, grp in df_dm.groupby("horizon", sort=False):
            # Sort p-values ascending for Holm step-down
            p_vals = grp["p_value_raw"].values
            m = len(p_vals)
            order = np.argsort(p_vals)
            sorted_p = p_vals[order]

            # Holm adjusted p-values
            adj_p = np.zeros(m)
            cum_max = 0.0
            for rank_idx, p in enumerate(sorted_p):
                val = (m - rank_idx) * p
                cum_max = max(cum_max, val)
                adj_p[rank_idx] = min(1.0, cum_max)

            # Invert back to original order
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(m)
            orig_adj_p = adj_p[inv_order]
            adj_p_list.extend(orig_adj_p)

        df_dm["p_value_holm"] = adj_p_list
        df_dm["significant_at_05"] = df_dm["p_value_holm"] < 0.05
    else:
        df_dm["significant_at_05"] = df_dm["p_value_raw"] < 0.05

    return df_dm

