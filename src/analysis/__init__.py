"""Result Analysis Package for Post-Modelling Validation Benchmark.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21, Spec v2.0.0, and Project Rules:
- data-leakage-prevention.md
- evaluation-standards.md
"""

from src.analysis.loaders import (
    load_all_validation_results,
    load_all_horizon_breakdowns,
    load_all_forecast_matrices,
    load_hurdle_probabilities,
    load_hurdle_threshold_metadata,
    verify_provenance_hashes,
)
from src.analysis.ranking import (
    compute_official_leaderboard,
    format_leaderboard_display,
)
from src.analysis.integrity import (
    verify_benchmark_health,
    check_hurdle_boundary_hits,
    assert_no_purged_metrics,
)
from src.analysis.aggregate import (
    compute_family_aggregates,
    compute_feature_set_ablation,
    compute_hurdle_ablation,
    compute_seasonal_breakdown,
)
from src.analysis.hurdle_diagnostics import (
    evaluate_stage1_classification,
    compute_all_hurdle_stage1_diagnostics,
)
from src.analysis.resources import (
    normalize_resource_metrics,
)
from src.analysis.pareto import (
    identify_pareto_frontier,
)
from src.analysis.significance import (
    diebold_mariano_test_by_horizon,
    compute_champion_pairwise_dm,
)

__all__ = [
    "load_all_validation_results",
    "load_all_horizon_breakdowns",
    "load_all_forecast_matrices",
    "load_hurdle_probabilities",
    "load_hurdle_threshold_metadata",
    "verify_provenance_hashes",
    "compute_official_leaderboard",
    "format_leaderboard_display",
    "verify_benchmark_health",
    "check_hurdle_boundary_hits",
    "assert_no_purged_metrics",
    "compute_family_aggregates",
    "compute_feature_set_ablation",
    "compute_hurdle_ablation",
    "compute_seasonal_breakdown",
    "evaluate_stage1_classification",
    "compute_all_hurdle_stage1_diagnostics",
    "normalize_resource_metrics",
    "identify_pareto_frontier",
    "diebold_mariano_test_by_horizon",
    "compute_champion_pairwise_dm",
]

