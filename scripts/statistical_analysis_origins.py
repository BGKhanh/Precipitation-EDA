"""
scripts/statistical_analysis_origins.py

Comprehensive origin-level statistical significance testing and robustness analysis
across 50 weekly forecast origins (h=1..7, covering 2020-05-01 to 2021-04-15):
- Paired Student's t-test and Wilcoxon signed-rank test
- Multiplicity corrections: Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR)
- Robustness: IID Bootstrap 95% CI vs Moving Block Bootstrap (MBB, b=4 weeks) 95% CI
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy import stats

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down method for family-wise error rate control."""
    m = len(pvals)
    pvals = np.array(pvals, dtype=float)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    adj_sorted = np.zeros(m)
    cur_max = 0.0
    for i in range(m):
        val = (m - i) * p_sorted[i]
        cur_max = max(cur_max, val)
        adj_sorted[i] = min(cur_max, 1.0)
    adj = np.zeros(m)
    adj[order] = adj_sorted
    return adj


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure for false discovery rate (FDR) control."""
    m = len(pvals)
    pvals = np.array(pvals, dtype=float)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    adj_sorted = np.zeros(m)
    cur_min = 1.0
    for i in range(m - 1, -1, -1):
        val = (m / (i + 1)) * p_sorted[i]
        cur_min = min(cur_min, val)
        adj_sorted[i] = min(cur_min, 1.0)
    adj = np.zeros(m)
    adj[order] = adj_sorted
    return adj


def iid_bootstrap_ci(series: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI assuming independent origins."""
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < 3:
        return np.nan, np.nan
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(series[boot_idx], axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def moving_block_bootstrap_ci(series: np.ndarray, block_size: int = 4, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    """Moving Block Bootstrap (MBB) percentile CI preserving temporal autocorrelation."""
    n = len(series)
    if n <= block_size:
        return iid_bootstrap_ci(series, n_boot, seed)
    k = n - block_size + 1
    blocks = np.array([series[i:i + block_size] for i in range(k)])  # shape: (k, block_size)
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_size))
    # Vectorized block resampling
    rand_idx = rng.integers(0, k, size=(n_boot, n_blocks_needed))
    sampled_blocks = blocks[rand_idx]  # shape: (n_boot, n_blocks_needed, block_size)
    sampled_series = sampled_blocks.reshape(n_boot, -1)[:, :n]
    boot_means = np.mean(sampled_series, axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def moving_block_bootstrap_wet_ci(sub_df: pd.DataFrame, diff_col: str, block_size: int = 4, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    """MBB for wet-day evaluations, resampling temporal windows and averaging over wet events."""
    sub_df = sub_df.sort_values("window").reset_index(drop=True)
    n = len(sub_df)
    k = n - block_size + 1
    is_rain_arr = sub_df["is_rain"].values
    diff_arr = sub_df[diff_col].values  # contains nan for dry days, float for wet days

    rain_blocks = np.array([is_rain_arr[i:i + block_size] for i in range(k)])
    diff_blocks = np.array([diff_arr[i:i + block_size] for i in range(k)])

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_size))
    rand_idx = rng.integers(0, k, size=(n_boot, n_blocks_needed))

    sampled_rain = rain_blocks[rand_idx].reshape(n_boot, -1)[:, :n]
    sampled_diff = diff_blocks[rand_idx].reshape(n_boot, -1)[:, :n]

    # Compute mean over wet events in each resample
    # sum of diffs where rain == 1 / count of rain == 1
    wet_counts = np.sum(sampled_rain == 1, axis=1)
    diff_clean = np.nan_to_num(sampled_diff, nan=0.0)
    wet_sums = np.sum(diff_clean * (sampled_rain == 1), axis=1)

    valid_mask = wet_counts > 0
    boot_means = wet_sums[valid_mask] / wet_counts[valid_mask]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def analyze_family(
    df: pd.DataFrame,
    family_name: str,
    scope: str,
    pred_col_a: str,
    pred_col_b: str,
    target_col: str,
    is_wet_only: bool = False,
    block_size: int = 4,
) -> pd.DataFrame:
    """Evaluate pairwise difference between Model A and Model B across h=1..7."""
    rows = []
    diff_col_name = f"diff_{pred_col_a}_vs_{pred_col_b}"
    if is_wet_only:
        df[diff_col_name] = np.where(
            df["is_rain"] == 1,
            np.abs(df[pred_col_a] - df[target_col]) - np.abs(df[pred_col_b] - df[target_col]),
            np.nan,
        )
    else:
        df[diff_col_name] = np.abs(df[pred_col_a] - df[target_col]) - np.abs(df[pred_col_b] - df[target_col])

    for h in range(1, 8):
        sub = df[df["horizon"] == h].sort_values("window")
        if is_wet_only:
            sub_eval = sub[sub["is_rain"] == 1]
            diffs = sub_eval[diff_col_name].dropna().values
            iid_ci = iid_bootstrap_ci(diffs)
            block_ci = moving_block_bootstrap_wet_ci(sub, diff_col_name, block_size=block_size)
            n_samples = len(sub_eval)
        else:
            diffs = sub[diff_col_name].values
            iid_ci = iid_bootstrap_ci(diffs)
            block_ci = moving_block_bootstrap_ci(diffs, block_size=block_size)
            n_samples = len(sub)

        mean_diff = float(np.mean(diffs))

        # Paired tests
        t_stat, p_t = stats.ttest_1samp(diffs, popmean=0.0)

        # Wilcoxon
        non_zero = diffs[diffs != 0]
        if len(non_zero) > 5:
            _, p_w = stats.wilcoxon(non_zero)
        else:
            p_w = np.nan

        rows.append({
            "Family": family_name,
            "Scope": scope,
            "Horizon": h,
            "N": n_samples,
            "Delta_MAE": mean_diff,
            "IID_CI_Low": iid_ci[0],
            "IID_CI_High": iid_ci[1],
            "Block_CI_Low": block_ci[0],
            "Block_CI_High": block_ci[1],
            "t_stat": t_stat,
            "raw_p_t": p_t,
            "raw_p_w": p_w,
        })

    res_df = pd.DataFrame(rows)
    # Apply Multiple Comparison Corrections across the 7 horizons
    res_df["holm_p_t"] = holm_adjust(res_df["raw_p_t"].values)
    res_df["bh_p_t"] = bh_adjust(res_df["raw_p_t"].values)
    if not res_df["raw_p_w"].isna().all():
        res_df["holm_p_w"] = holm_adjust(res_df["raw_p_w"].values)
    else:
        res_df["holm_p_w"] = np.nan

    return res_df


def run_full_statistical_analysis():
    pred_path = Path("data/processed/hurdle_vs_tweedie_predictions_per_window.csv")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}. Run benchmark first.")

    df = pd.read_csv(pred_path)
    print("=" * 115)
    print("STATISTICAL INFERENCE & ROBUSTNESS ANALYSIS (50 Forecast Origins, Weekly Step)")
    print(f"Data Source: {pred_path}")
    print("Test Period: 2020-05-01 to 2021-04-15 (1 complete out-of-sample annual cycle)")
    print("Moving Block Bootstrap (MBB): block_size = 4 weeks (atmospheric regime persistence)")
    print("=" * 115)

    # 1. Expected vs Hard-Gate on Overall Target Y*
    res_exp_hg_overall = analyze_family(
        df,
        family_name="Expected vs Hard-Gate",
        scope="Overall Y*",
        pred_col_a="pred_expected",
        pred_col_b="pred_hard_gate",
        target_col="y_hurdle_target",
        is_wet_only=False,
    )

    # 2. Expected vs Hard-Gate on End-to-End Wet Days
    res_exp_hg_wet = analyze_family(
        df,
        family_name="Expected vs Hard-Gate",
        scope="End-to-End Wet Days",
        pred_col_a="pred_expected",
        pred_col_b="pred_hard_gate",
        target_col="y_true",
        is_wet_only=True,
    )

    # 3. Direct Tweedie vs Hurdle Hard-Gate on Overall Target Y*
    res_tw_hg_overall = analyze_family(
        df,
        family_name="Tweedie vs Hard-Gate",
        scope="Overall Y*",
        pred_col_a="pred_tweedie",
        pred_col_b="pred_hard_gate",
        target_col="y_hurdle_target",
        is_wet_only=False,
    )

    # 4. Direct Tweedie vs Hurdle Expected on Overall Target Y*
    res_tw_exp_overall = analyze_family(
        df,
        family_name="Tweedie vs Expected",
        scope="Overall Y*",
        pred_col_a="pred_tweedie",
        pred_col_b="pred_expected",
        target_col="y_hurdle_target",
        is_wet_only=False,
    )

    all_results = pd.concat([res_exp_hg_overall, res_exp_hg_wet, res_tw_hg_overall, res_tw_exp_overall], ignore_index=True)

    def print_table(title: str, sub_df: pd.DataFrame):
        print(f"\n--- {title} ---")
        print(f"{'Horizon':<8}{'Delta_MAE':<12}{'95% IID CI':<24}{'95% Block CI (b=4)':<26}{'Raw p (t)':<12}{'Holm p (t)':<12}{'BH FDR (t)':<12}{'Raw p (W)':<12}{'Holm p (W)':<12}")
        print("-" * 126)
        for _, r in sub_df.iterrows():
            h_str = f"h={int(r['Horizon'])}"
            d_str = f"{r['Delta_MAE']:>+8.4f} mm"
            iid_str = f"[{r['IID_CI_Low']:>+7.4f}, {r['IID_CI_High']:>+7.4f}]"
            block_str = f"[{r['Block_CI_Low']:>+7.4f}, {r['Block_CI_High']:>+7.4f}]"
            raw_t_str = f"{r['raw_p_t']:>10.4f}"
            holm_t_str = f"{r['holm_p_t']:>10.4f}" + (" *" if r['holm_p_t'] < 0.05 else "")
            bh_t_str = f"{r['bh_p_t']:>10.4f}" + (" *" if r['bh_p_t'] < 0.05 else "")
            raw_w_str = f"{r['raw_p_w']:>10.4f}"
            holm_w_str = f"{r['holm_p_w']:>10.4f}" + (" *" if r['holm_p_w'] < 0.05 else "")
            print(f"{h_str:<8}{d_str:<12}{iid_str:<24}{block_str:<26}{raw_t_str:<12}{holm_t_str:<12}{bh_t_str:<12}{raw_w_str:<12}{holm_w_str:<12}")

    print_table("FAMILY 1: EXPECTED vs HARD-GATE on OVERALL TARGET Y* (Delta < 0 => Expected better)", res_exp_hg_overall)
    print_table("FAMILY 2: EXPECTED vs HARD-GATE on END-TO-END WET DAYS (Delta < 0 => Expected better)", res_exp_hg_wet)
    print_table("FAMILY 3: DIRECT TWEEDIE vs HURDLE HARD-GATE on OVERALL TARGET Y* (Delta < 0 => Tweedie better)", res_tw_hg_overall)
    print_table("FAMILY 4: DIRECT TWEEDIE vs HURDLE EXPECTED on OVERALL TARGET Y* (Delta < 0 => Tweedie better)", res_tw_exp_overall)

    out_path = Path("data/processed/hurdle_vs_tweedie_statistical_tests.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\nSaved audited statistical testing results to {out_path}")
    return all_results


if __name__ == "__main__":
    run_full_statistical_analysis()
