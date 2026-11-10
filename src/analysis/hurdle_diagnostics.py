"""Stage 1 Classification Diagnostics for Two-Stage Hurdle Forecasters.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and CR-05.
Evaluates binary rain occurrence discrimination (Stage 1 probability p_hat)
against the canonical WMO RR1 ground truth threshold (Y >= 1.0 mm).

IMPORTANT NOTE:
Stage 1 diagnostics are descriptive quality indicators only; they are NOT evaluation
metrics and are strictly prohibited from participating in model ranking or selection.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

CANONICAL_RAIN_THRESHOLD_MM = 1.0  # WMO RR1 standard for wet day definition


def evaluate_stage1_classification(
    p_hat: np.ndarray,
    y_true: np.ndarray,
    decision_threshold: float,
    rain_threshold_mm: float = CANONICAL_RAIN_THRESHOLD_MM,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification diagnostics for Stage 1 Hurdle probabilities.
    
    Args:
        p_hat: Array of predicted rainfall probabilities, shape (724, 7).
        y_true: Array of actual rainfall depths, shape (724, 7).
        decision_threshold: Optimal classification gate threshold tau_M* (e.g. 0.65, 0.70, 0.80).
        rain_threshold_mm: Depth cutoff defining a wet day (default: 1.0 mm per WMO RR1).
        
    Returns:
        Dict containing ROC-AUC, Brier score, PR-AUC, Precision, Recall, F1, Specificity,
        Confusion Matrix, and curve coordinates.
    """
    assert p_hat.shape == y_true.shape, f"Shape mismatch: {p_hat.shape} vs {y_true.shape}"

    # Flatten to pooled 1D vector (N = 5,068)
    p_flat = p_hat.flatten().astype(np.float64)
    y_flat = y_true.flatten().astype(np.float64)

    # Ground truth binary label
    y_bin = (y_flat >= rain_threshold_mm).astype(int)
    # Predicted binary label at decision threshold tau
    y_pred_bin = (p_flat >= decision_threshold).astype(int)

    # 1. Discrimination and calibration metrics
    roc_auc = float(roc_auc_score(y_bin, p_flat))
    brier = float(brier_score_loss(y_bin, p_flat))
    pr_auc = float(average_precision_score(y_bin, p_flat))

    # 2. Threshold-dependent classification metrics
    prec = float(precision_score(y_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_bin, y_pred_bin, zero_division=0))

    # 3. Confusion matrix
    cm = confusion_matrix(y_bin, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 4. Curve coordinates for plotting
    fpr, tpr, _ = roc_curve(y_bin, p_flat)
    p_curve, r_curve, _ = precision_recall_curve(y_bin, p_flat)

    return {
        "roc_auc": roc_auc,
        "brier_score": brier,
        "pr_auc": pr_auc,
        "decision_threshold": decision_threshold,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "wet_ratio_actual": float(y_bin.mean()),
        "wet_ratio_predicted": float(y_pred_bin.mean()),
        "roc_curve": (fpr, tpr),
        "pr_curve": (p_curve, r_curve),
    }


def compute_all_hurdle_stage1_diagnostics(
    hurdle_probs: Dict[str, Any],
    threshold_meta: Dict[str, Any],
    rain_threshold_mm: float = CANONICAL_RAIN_THRESHOLD_MM,
) -> pd.DataFrame:
    """Run Stage 1 classification evaluation across all 16 Hurdle combinations.
    
    Args:
        hurdle_probs: Dict loaded from ml_hurdle_val_probabilities.npz.
        threshold_meta: Dict loaded from ml_hurdle_threshold_selection.json.
        rain_threshold_mm: Cutoff for wet day ground truth (1.0 mm).
        
    Returns:
        DataFrame summarizing discrimination and calibration metrics for 16 Hurdle models.
    """
    y_true = hurdle_probs["y_true"]

    def get_tau(fam_key: str, default_tau: float) -> float:
        if fam_key in threshold_meta and isinstance(threshold_meta[fam_key], dict):
            return float(threshold_meta[fam_key].get("selected_tau_m", default_tau))
        opt_fam = threshold_meta.get("optimal_thresholds_by_family", {})
        return float(opt_fam.get(fam_key, {}).get("optimal_tau", default_tau))

    tau_mapping = {
        "logistic_linear": get_tau("logistic_linear", 0.65),
        "lightgbm": get_tau("lightgbm", 0.80),
        "random_forest": get_tau("random_forest", 0.70),
        "xgboost": get_tau("xgboost", 0.70),
    }

    records = []
    families = ["logistic_linear", "lightgbm", "random_forest", "xgboost"]
    feature_sets = ["F0", "F1", "F2", "F3"]

    for fam in families:
        tau_star = tau_mapping.get(fam, 0.50)
        for f_set in feature_sets:
            key_phat = f"{fam}_{f_set}_phat"
            if key_phat not in hurdle_probs:
                continue

            p_hat = hurdle_probs[key_phat]
            diag = evaluate_stage1_classification(
                p_hat=p_hat,
                y_true=y_true,
                decision_threshold=tau_star,
                rain_threshold_mm=rain_threshold_mm,
            )

            records.append({
                "hurdle_family": fam,
                "feature_set": f_set,
                "optimal_tau": tau_star,
                "ROC_AUC": diag["roc_auc"],
                "Brier_Score": diag["brier_score"],
                "PR_AUC": diag["pr_auc"],
                "Precision": diag["precision"],
                "Recall": diag["recall"],
                "F1_Score": diag["f1_score"],
                "Specificity": diag["specificity"],
                "TP": diag["tp"],
                "FP": diag["fp"],
                "FN": diag["fn"],
                "TN": diag["tn"],
            })

    df_diag = pd.DataFrame(records)
    return df_diag.sort_values(by=["ROC_AUC"], ascending=False).reset_index(drop=True)
