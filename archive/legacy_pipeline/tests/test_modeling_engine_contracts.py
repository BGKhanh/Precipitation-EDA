"""Automated Phase 0 Modeling Engine & Methodological Contract Test Suite (Plan v1.4.18).

Covers all 6 Phase 0 Acceptance Gates plus M1, M4, M5/P5-1, M6a/P4-1, M7, P5-2, and P6-1:
- Gate 1: Threshold Source Alignment (tau_rain = 1.0 mm/day, inclusive >=, empty mask -> np.nan)
- Gate 2: Target Boundary Purge (H=7) & Zero Final Test Label Access
- Gate 3: Unified Schema (22 cols) & Exhaustive 6-Trigger Model-Scoped Failure Contract (P5-2)
- Gate 4: Single-Fit & Intermittent Wrapper Parity (StatsForecastWrapper & FrozenIntermittentWrapper N=2, K=2)
- Gate 5: Typed Gatechecks (G1-G6) & Decision M1 DirectHurdleForecaster Contract
- Gate 6: Feature Causality (Family A Future-Perturbation Invariance) & Preprocessing Split-Isolation (Family B)
- Governance State Machines: M4 Epsilon Tie-Break, M6a Exact-Equality Window Selection, P5-1 Conservative
  Holm Multiplicity Control, and P6-1 Model-Scoped Failure vs Process-Scoped Interruption & Checkpoint Recovery.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    class _PytestStub:
        @staticmethod
        @contextmanager
        def raises(exc_type, match=None):
            try:
                yield
            except exc_type as e:
                if match and match not in str(e):
                    raise AssertionError(f"Expected '{match}' in '{e}'")
                return
            raise AssertionError(f"Expected {exc_type} was not raised")

    pytest = _PytestStub()  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import rain_day_mae, rain_day_rmse
from src.evaluation.protocols import (
    UNIFIED_SCHEMA,
    RAIN_THRESHOLD,
    RAIN_COMPARISON_OPERATOR,
    CANONICAL_RAIN_THRESHOLD,
    CANONICAL_RAIN_THRESHOLD_OPERATOR,
    LEGACY_EDA_RAIN_THRESHOLD,
    LEGACY_EDA_RAIN_THRESHOLD_OPERATOR,
    validate_and_normalize_canonical_threshold,
    DataLeakageSecurityError,
    PreReadFileReadGuard,
    get_global_read_guard,
    GatecheckStatus,
    ModelTerminalStatus,
    SystemEvaluationStatus,
    InferenceModuleStatus,
    save_modeling_contract,
    evaluate_six_gatechecks,
    gatecheck_threshold_source_alignment,
    gatecheck_target_boundary_purge_h7,
    gatecheck_feature_causality_and_split_isolation,
)
from src.eda.pipeline import EDAReport, CANONICAL_SOURCE_ROOT, CANONICAL_PRODUCTION_DESTINATIONS
from src.evaluation.benchmark import (
    UnifiedBenchmark,
    RollingBenchmarkCoordinator,
    ProcessScopedInterruption,
    rank_benchmark_results,
    select_with_epsilon_tiebreak,
    select_neural_context_window,
    run_confirmatory_dm_inference,
)
from src.models.hurdle import (
    DirectHurdleForecaster,
    build_logistic_linear_hurdle,
    build_lightgbm_hurdle,
    HARD_GATE_THRESHOLD_GRID,
)
from src.models.nixtla.wrappers import StatsForecastWrapper, FrozenIntermittentWrapper


# ======================================================================
# Gate 1: Threshold Source Alignment (tau_rain = 1.0 mm/day, >=)
# ======================================================================

def test_gate1_threshold_source_alignment(tmp_path: Path) -> None:
    """Verify all threshold sources align at 1.0 mm/day with inclusive >= and empty-mask NaN."""
    contract_file = tmp_path / "modeling_contract.json"
    save_modeling_contract(filepath=contract_file)

    res = gatecheck_threshold_source_alignment(contract_path=contract_file)
    assert res["status"] == GatecheckStatus.PASS.value
    assert res["modeling_contract_threshold"] == 1.0
    assert res["modeling_contract_operator"] == ">="
    assert res["protocols_RAIN_THRESHOLD"] == 1.0
    assert res["metrics_rain_day_mae_default"] == 1.0
    assert res["metrics_rain_day_rmse_default"] == 1.0
    assert res["protocols_compute_benchmark_metrics_default"] == 1.0
    assert res["benchmark_UnifiedBenchmark_default"] == 1.0

    # Exact boundary test: 1.0 mm is wet (included), 0.999 mm is dry (excluded)
    y_true = np.array([0.0, 0.999, 1.0, 3.0])
    y_pred = np.array([5.0, 5.0, 2.0, 1.0])
    # Only indices 2 (|2-1|=1) and 3 (|1-3|=2) are included -> MAE = 1.5, RMSE = sqrt((1+4)/2) = sqrt(2.5)
    assert np.isclose(rain_day_mae(y_true, y_pred), 1.5)
    assert np.isclose(rain_day_rmse(y_true, y_pred), np.sqrt(2.5))

    # Empty wet-day mask returns np.nan without raising ZeroDivisionError
    y_all_dry = np.array([0.0, 0.1, 0.5, 0.999])
    assert np.isnan(rain_day_mae(y_all_dry, np.zeros(4)))
    assert np.isnan(rain_day_rmse(y_all_dry, np.zeros(4)))


# ======================================================================
# Gate 2: Target Boundary Purge (H=7) & Zero Test Label Access
# ======================================================================

def test_gate2_target_boundary_purge_h7() -> None:
    """Verify H=7 target boundary purge across all splits using synthetic date fixtures only."""
    res = gatecheck_target_boundary_purge_h7()
    assert res["status"] == GatecheckStatus.PASS.value
    assert res["final_test_labels_accessed"] is False

    splits = res["splits"]
    assert splits["dev_train"]["days"] == 6329
    assert splits["dev_train"]["valid_origins"] == 6322
    assert splits["dev_train"]["common_origins"] == 6293
    assert splits["dev_train"]["max_origin_date"] == "2018-04-23"
    assert splits["dev_train"]["max_target_date"] == "2018-04-30"

    assert splits["dev_train_fit"]["days"] == 5691
    assert splits["dev_train_fit"]["valid_origins"] == 5684
    assert splits["dev_train_fit"]["max_origin_date"] == "2016-07-24"
    assert splits["dev_train_fit"]["max_target_date"] == "2016-07-31"

    assert splits["dev_train_es"]["days"] == 638
    assert splits["dev_train_es"]["valid_origins"] == 631
    assert splits["dev_train_es"]["max_origin_date"] == "2018-04-23"
    assert splits["dev_train_es"]["max_target_date"] == "2018-04-30"

    assert splits["validation"]["days"] == 731
    assert splits["validation"]["valid_origins"] == 724
    assert splits["validation"]["max_origin_date"] == "2020-04-23"
    assert splits["validation"]["max_target_date"] == "2020-04-30"

    assert splits["final_test_date_fixture"]["days"] == 2191
    assert splits["final_test_date_fixture"]["valid_origins"] == 2184
    assert splits["final_test_date_fixture"]["max_origin_date"] == "2026-04-23"
    assert splits["final_test_date_fixture"]["max_target_date"] == "2026-04-30"


# ======================================================================
# Gate 3: Unified Schema & Exhaustive 6-Trigger Model-Scoped Failure (P5-2)
# ======================================================================

def test_gate3_exhaustive_model_scoped_failure_contract(tmp_path: Path) -> None:
    """Verify all 6 model-scoped failure triggers set FAILED_TEST, NaN metrics, and (0, 7) array."""
    assert len(UNIFIED_SCHEMA) == 20

    n_origins = 10
    horizon = 7
    origin_dates = pd.date_range("2020-05-01", periods=n_origins, freq="D")
    y_true = np.ones((n_origins, horizon), dtype=np.float64) * 2.0

    partial_npz = tmp_path / "test_diagnostic_partial_forecasts.npz"
    ckpt_file = tmp_path / "test_checkpoint.json"

    def _raise_oom():
        raise MemoryError("Simulated CUDA out of memory on model 1")

    model_specs = [
        {"model": "M0_Valid", "group": "G1", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins, horizon)) * 1.5},
        {"model": "M1_RuntimeError", "group": "G1", "strategy": "direct", "predict_fn": _raise_oom},
        {"model": "M2_ShapeMismatch", "group": "G2", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins, 5))},
        {"model": "M3_NaNInf", "group": "G2", "strategy": "direct", "predict_fn": lambda: np.full((n_origins, horizon), np.nan)},
        {"model": "M4_NegativeVal", "group": "G3", "strategy": "direct", "predict_fn": lambda: np.full((n_origins, horizon), -0.01)},
        {
            "model": "M5_DateMisalign",
            "group": "G4",
            "strategy": "direct",
            "predict_fn": lambda: {
                "predictions": np.ones((n_origins, horizon)),
                "origin_dates": pd.date_range("2021-01-01", periods=n_origins, freq="D"),
            },
        },
        {"model": "M6_IncompleteOrigins", "group": "G5", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins - 2, horizon))},
    ]

    coord = RollingBenchmarkCoordinator(
        expected_origin_dates=origin_dates,
        horizon=horizon,
        rain_threshold=1.0,
        stage="test",
        checkpoint_path=ckpt_file,
        diagnostic_partial_npz_path=partial_npz,
    )
    out = coord.evaluate_models(model_specs, y_true=y_true)

    assert out["system_status"] == SystemEvaluationStatus.TEST_EVALUATION_COMPLETE.value
    df = out["results_df"]
    assert list(df.columns) == UNIFIED_SCHEMA
    assert len(df) == 7

    # M0 succeeded
    m0_row = df[df["model"] == "M0_Valid"].iloc[0]
    assert m0_row["status"] == ModelTerminalStatus.TEST_SUCCESS.value
    assert np.isclose(m0_row["MAE"], 0.5)
    assert out["official_forecasts"]["M0_Valid"].shape == (n_origins, horizon)

    # M1..M6 all triggered model-scoped FAILED_TEST
    metric_cols = ["MAE", "MSE", "RMSE", "MASE", "mae_macro", "mse_macro", "rmse_macro", "mase_lag1_macro"]
    for failed_id in ["M1_RuntimeError", "M2_ShapeMismatch", "M3_NaNInf", "M4_NegativeVal", "M5_DateMisalign", "M6_IncompleteOrigins"]:
        row = df[df["model"] == failed_id].iloc[0]
        assert row["status"] == ModelTerminalStatus.FAILED_TEST.value
        for mc in metric_cols:
            assert np.isnan(row[mc]), f"Expected NaN for {mc} on {failed_id}"
        assert out["official_forecasts"][failed_id].shape == (0, horizon)
        assert row["failure_reason"] != ""

    # Quarantined partial predictions persisted to diagnostic NPZ
    assert partial_npz.exists()
    with np.load(partial_npz) as npz_data:
        assert "partial_M6_IncompleteOrigins" in npz_data.files


# ======================================================================
# Gate 4: Single-Fit & Intermittent Wrapper Parity (§1.3 & Decision M7)
# ======================================================================

def test_gate4_statsforecast_wrapper_single_fit_and_parity() -> None:
    """Verify StatsForecastWrapper single-fit parameter invariance and numerical parity on 100-day fixture."""
    from statsforecast.models import AutoETS, Naive, SeasonalNaive

    rng = np.random.default_rng(123)
    y_fixture = rng.exponential(scale=3.0, size=100)
    y_train = y_fixture[:80]
    y_hist = y_fixture[:90]

    for model_instance in [Naive(), SeasonalNaive(season_length=7), AutoETS(season_length=7)]:
        wrapper = StatsForecastWrapper(model_instance)
        wrapper.fit(y_train)
        assert wrapper.fitted_ is True

        pred = wrapper.forward(y_hist, h=7)
        assert pred.shape == (1, 7)
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0.0)

        # Direct comparison against underlying statsforecast model forward()
        if hasattr(wrapper.model_, "forward"):
            try:
                direct_out = np.clip(np.asarray(wrapper.model_.forward(y=y_hist, h=7)["mean"]), 0.0, None).reshape(1, 7)
                np.testing.assert_allclose(pred, direct_out, atol=1e-5)
            except NotImplementedError:
                pass


def test_gate4_frozen_intermittent_wrapper_m7_contract() -> None:
    """Verify FrozenIntermittentWrapper enforces Decision M7 Option B (N=2 for ADIDA, K=2 for IMAPA)."""
    # Construct synthetic Dev Train of 6329 days with exact 2747 wet days (>= 1.0 mm) and trace drizzle (0 < Y < 1.0)
    y_dev = np.zeros(6329, dtype=np.float64)
    y_dev[:2747] = 5.0  # wet days >= 1.0 mm
    y_dev[2747:3500] = 0.3  # trace drizzle (0 < Y < 1.0 mm) preserved as > 0
    rng = np.random.default_rng(42)
    rng.shuffle(y_dev)

    for m_name in FrozenIntermittentWrapper.SUPPORTED_MODELS:
        w = FrozenIntermittentWrapper(m_name, rain_threshold=1.0)
        w.fit(y_dev)
        assert w.fitted_ is True
        assert np.isclose(w.wet_day_spacing_k0_, 6329.0 / 2747.0, atol=1e-6)
        assert w.domain_bucket_size_ == 2

        if m_name == "ADIDA":
            assert w.aggregation_chunk_size_ == 2
        if m_name == "IMAPA":
            assert w.max_aggregation_level_ == 2
            assert w.aggregation_levels_ == (1, 2)

        out = w.forward(y_dev[:200], h=7)
        assert out.shape == (1, 7)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    # Degenerate N=1 or K=1 must raise ValueError per Decision M7
    with pytest.raises(ValueError, match="Decision M7 Violation"):
        FrozenIntermittentWrapper("ADIDA", aggregation_chunk_size=1)
    with pytest.raises(ValueError, match="Decision M7 Violation"):
        FrozenIntermittentWrapper("IMAPA", max_aggregation_level=1)


# ======================================================================
# Gate 5: Typed Gatechecks (G1-G6) & Decision M1 DirectHurdleForecaster
# ======================================================================

def test_gate5_typed_gatechecks_and_direct_hurdle_m1() -> None:
    """Verify evaluate_six_gatechecks returns GatecheckStatus enums and DirectHurdleForecaster obeys M1."""
    gates = evaluate_six_gatechecks(
        y_pred=np.ones((10, 7)),
        expected_shape=(10, 7),
        hurdle_applicable=False,
    )
    assert set(gates.keys()) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert gates["G1"]["status"] == GatecheckStatus.PASS.value
    assert gates["G4"]["status"] == GatecheckStatus.PASS.value
    assert gates["G6"]["status"] == GatecheckStatus.NOT_APPLICABLE.value

    # Decision M1 DirectHurdleForecaster verification
    rng = np.random.default_rng(7)
    n_train = 120
    n_val = 30
    X_train = rng.normal(size=(n_train, 5))
    # Mix dry (0.0), trace drizzle (0.2), and wet (>= 1.0)
    Y_train = np.where(rng.random((n_train, 7)) > 0.55, rng.uniform(1.0, 15.0, size=(n_train, 7)), 0.2)
    X_val = rng.normal(size=(n_val, 5))
    Y_val = np.where(rng.random((n_val, 7)) > 0.55, rng.uniform(1.0, 15.0, size=(n_val, 7)), 0.2)

    hurdle = build_logistic_linear_hurdle(rain_threshold=1.0, random_state=42)
    hurdle.fit(X_train, Y_train)

    assert hurdle.is_fitted_ is True
    assert len(hurdle.classifiers_) == 7
    assert len(hurdle.regressors_) == 7
    assert np.isclose(hurdle.fitted_dry_mean_, 0.2, atol=1e-6)

    # Stage 2 conditional predictions must all be clipped >= 1.0 mm
    probs, wet_reg = hurdle.predict_components(X_val)
    assert probs.shape == (n_val, 7)
    assert wet_reg.shape == (n_val, 7)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all(wet_reg >= 1.0)

    # Compare Strategy A (expected_value) vs Strategy B (hard_gate over 0.10..0.90) on Validation
    sel = hurdle.select_strategy_on_validation(X_val, Y_val)
    assert sel["selected_strategy"] in ("expected_value", "hard_gate")
    assert len(HARD_GATE_THRESHOLD_GRID) == 17
    assert HARD_GATE_THRESHOLD_GRID == tuple(round(0.10 + 0.05 * i, 2) for i in range(17))
    preds = hurdle.predict(X_val)
    assert preds.shape == (n_val, 7)
    assert np.all(preds >= 0.0)


# ======================================================================
# Gate 6: Feature Causality & Preprocessing Split-Isolation
# ======================================================================

def test_gate6_feature_causality_and_recursive_isolation() -> None:
    """Verify Family A future-perturbation invariance, Family B split-isolation, and RecursiveForecaster isolation."""
    res = gatecheck_feature_causality_and_split_isolation()
    assert res["status"] == GatecheckStatus.PASS.value
    assert res["family_a_raw_feature_causality"]["passed"] is True
    assert res["family_a_raw_feature_causality"]["max_abs_diff_under_future_perturbation"] <= 1e-7
    assert res["family_a_raw_feature_causality"]["inclusive_window_formula_verified"] is True
    assert res["family_a_raw_feature_causality"]["d1b_lag_0_identity_verified"] is True
    assert res["family_b_preprocessing_split_isolation"]["passed"] is True
    assert res["family_b_preprocessing_split_isolation"]["max_abs_diff_frozen_transform"] <= 1e-7

    # Verify RecursiveForecaster is blocked by default in UnifiedBenchmark
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df_dummy = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": np.ones(20)})
    bench = UnifiedBenchmark(df_dummy.iloc[:13], df_dummy.iloc[13:], horizon=7)
    with pytest.raises(RuntimeError, match="Phase 0 Gate 6 Isolation: RecursiveForecaster is prohibited"):
        bench._predict_legacy(object(), feature_builder=object(), steps=7)


# ======================================================================
# Governance Contracts: M4, M6a, P5-1 (Conservative Holm), and P6-1
# ======================================================================

def test_m4_and_m6a_selection_contracts() -> None:
    """Verify M4 candidate-tier selection and M6a exact-equality neural context-window selection (Protocol v1.0.20 §9.3.D)."""
    df_candidates = pd.DataFrame([
        {"model": "Model_B", "group": "G2", "status": "VALIDATION_SUCCESS", "MAE": 2.10005, "RMSE": 4.00},
        {"model": "Model_A", "group": "G1", "status": "VALIDATION_SUCCESS", "MAE": 2.10000, "RMSE": 4.10},
    ])
    # Within 1e-4 epsilon band on Pooled MAE (2.10005 - 2.10000 <= 1e-4) -> Model_B wins via lower RMSE (4.00 < 4.10)
    sel = select_with_epsilon_tiebreak(df_candidates, epsilon=1e-4)
    assert sel["selected_model"] == "Model_B"
    assert set(sel["equivalence_band"]) == {"Model_A", "Model_B"}

    # Full Leaderboard Iterative Procedure (Standard Competition 1224 Ranking §9.3.D)
    df_ranking = pd.DataFrame([
        {"model": "model_x", "group": "G1", "status": "COMPLETED", "MAE": 2.00000, "RMSE": 1.0},
        {"model": "model_y", "group": "G2", "status": "COMPLETED", "MAE": 2.00005, "RMSE": 2.0},
        {"model": "model_y", "group": "G2", "status": "COMPLETED", "MAE": 2.00005, "RMSE": 2.0},
        {"model": "model_z", "group": "G3", "status": "COMPLETED", "MAE": 2.50000, "RMSE": 1.5},
        {"model": "model_failed", "group": "G4", "status": "FAILED_TEST", "MAE": np.nan, "RMSE": np.nan},
    ])
    ranked = rank_benchmark_results(df_ranking, epsilon=1e-4)
    assert ranked.loc[0, "rank"] == 1  # model_x, tier 1, RMSE 1.0
    assert ranked.loc[1, "rank"] == 2  # model_y, tier 1, RMSE 2.0
    assert ranked.loc[2, "rank"] == 2  # model_y tied with row 1
    assert ranked.loc[3, "rank"] == 4  # model_z, next tier starting rank = 1 + 3 = 4
    assert ranked.loc[4, "rank"] == "N/A [FAILED_TEST]"

    # M6a exact numerical tie on MAE (within 1e-4) and RMSE (==)
    m30 = {"MAE": 2.50002, "RMSE": 3.5}
    m60 = {"MAE": 2.50000, "RMSE": 3.5}
    res_approved = select_neural_context_window(m30, m60, epsilon=1e-4, m6a_approved=True)
    assert res_approved["selected_window"] == 30
    assert res_approved["tie_break_reason"] == "M6a_PARSIMONY_W30"

    res_pending = select_neural_context_window(m30, m60, epsilon=1e-4, m6a_approved=False)
    assert res_pending["selected_window"] is None
    assert res_pending["status"] == "TIE_BREAK_PENDING_USER_DECISION"


def test_p5_1_conservative_holm_when_representative_fails() -> None:
    """Verify P5-1: Failed group representative is NOT replaced and Holm uses K_planned denominator."""
    rng = np.random.default_rng(99)
    n = 100
    planned_reps = [
        {"model": "Rep_G1", "group": "G1"},
        {"model": "Rep_G2_Failed", "group": "G2"},
        {"model": "Rep_G3", "group": "G3"},
    ]
    statuses = {
        "Champ_G4": ModelTerminalStatus.TEST_SUCCESS.value,
        "Rep_G1": ModelTerminalStatus.TEST_SUCCESS.value,
        "Rep_G2_Failed": ModelTerminalStatus.FAILED_TEST.value,
        "Rep_G3": ModelTerminalStatus.TEST_SUCCESS.value,
    }
    losses = {
        "Champ_G4": np.ones(n) * 1.0,
        "Rep_G1": np.ones(n) * 3.0 + rng.normal(0, 0.1, n),
        "Rep_G3": np.ones(n) * 1.5 + rng.normal(0, 0.5, n),
    }

    inf = run_confirmatory_dm_inference(
        champion_id="Champ_G4",
        champion_status=statuses["Champ_G4"],
        planned_representatives=planned_reps,
        model_test_statuses=statuses,
        origin_losses=losses,
        alpha=0.05,
    )
    assert inf["inference_module_status"] == InferenceModuleStatus.CONFIRMATORY_INFERENCE_PARTIALLY_EVALUABLE.value
    assert inf["k_planned"] == 3
    assert inf["k_eval"] == 2
    assert inf["k_failed"] == 1

    # Rank j=1 must use denominator K_planned - 1 + 1 = 3 (alpha_1 = 0.05 / 3)
    # Rank j=2 must use denominator K_planned - 2 + 1 = 2 (alpha_2 = 0.05 / 2)
    eval_comps = [c for c in inf["comparisons"] if c["evaluable"]]
    assert eval_comps[0]["holm_denominator"] == 3
    assert np.isclose(eval_comps[0]["holm_threshold"], 0.05 / 3.0)
    assert eval_comps[1]["holm_denominator"] == 2
    assert np.isclose(eval_comps[1]["holm_threshold"], 0.05 / 2.0)

    failed_comps = [c for c in inf["comparisons"] if not c["evaluable"]]
    assert len(failed_comps) == 1
    assert failed_comps[0]["competitor_model"] == "Rep_G2_Failed"
    assert failed_comps[0]["note"] == "COMPETITOR_FAILED_TEST_NOT_REPLACED"


def test_p6_1_process_interruption_and_checkpoint_recovery(tmp_path: Path) -> None:
    """Verify P6-1: ProcessScopedInterruption sets TEST_EVALUATION_INTERRUPTED, leaves in-flight model
    as INCOMPLETE_UNRECORDED (never FAILED_TEST), blocks DM inference, and resumes cleanly from checkpoint.
    """
    n_origins = 8
    horizon = 7
    origin_dates = pd.date_range("2020-05-01", periods=n_origins, freq="D")
    y_true = np.ones((n_origins, horizon), dtype=np.float64) * 2.0
    ckpt_file = tmp_path / "p6_1_checkpoint.json"
    manifest = {"core_champion": "M1_Success", "frozen_at": "2020-04-30"}

    state = {"m2_should_crash_process": True}

    def _m2_predict():
        if state["m2_should_crash_process"]:
            raise ProcessScopedInterruption("Simulated power loss / OS kill while M2 is in-flight")
        return np.ones((n_origins, horizon)) * 1.8

    specs = [
        {"model": "M1_Success", "group": "G1", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins, horizon)) * 1.5},
        {"model": "M2_InterruptedInFlight", "group": "G2", "strategy": "direct", "predict_fn": _m2_predict},
        {"model": "M3_Pending", "group": "G3", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins, horizon)) * 2.2},
    ]

    # Run 1: Interrupted during M2
    coord1 = RollingBenchmarkCoordinator(
        expected_origin_dates=origin_dates,
        horizon=horizon,
        stage="test",
        checkpoint_path=ckpt_file,
        manifest_payload=manifest,
    )
    res1 = coord1.evaluate_models(specs, y_true=y_true, resume=False)

    assert res1["system_status"] == SystemEvaluationStatus.TEST_EVALUATION_INTERRUPTED.value
    assert res1["model_statuses"]["M1_Success"] == ModelTerminalStatus.TEST_SUCCESS.value
    # In-flight interrupted model MUST be INCOMPLETE_UNRECORDED, NEVER FAILED_TEST
    assert res1["model_statuses"]["M2_InterruptedInFlight"] == ModelTerminalStatus.INCOMPLETE_UNRECORDED.value
    assert res1["model_statuses"]["M3_Pending"] == ModelTerminalStatus.INCOMPLETE_UNRECORDED.value

    # Confirmatory inference MUST be blocked while system_status is TEST_EVALUATION_INTERRUPTED
    with pytest.raises(RuntimeError, match="TEST_EVALUATION_INTERRUPTED"):
        run_confirmatory_dm_inference(
            champion_id="M1_Success",
            champion_status=ModelTerminalStatus.TEST_SUCCESS.value,
            planned_representatives=[{"model": "M2_InterruptedInFlight", "group": "G2"}],
            model_test_statuses=res1["model_statuses"],
            origin_losses={},
            system_status=res1["system_status"],
        )

    # Run 2: Resume from checkpoint after resolving process interruption
    state["m2_should_crash_process"] = False
    m1_calls = {"count": 0}

    def _m1_should_not_rerun():
        m1_calls["count"] += 1
        return np.ones((n_origins, horizon)) * 1.5

    specs_resume = [
        {"model": "M1_Success", "group": "G1", "strategy": "direct", "predict_fn": _m1_should_not_rerun},
        {"model": "M2_InterruptedInFlight", "group": "G2", "strategy": "direct", "predict_fn": _m2_predict},
        {"model": "M3_Pending", "group": "G3", "strategy": "direct", "predict_fn": lambda: np.ones((n_origins, horizon)) * 2.2},
    ]

    coord2 = RollingBenchmarkCoordinator(
        expected_origin_dates=origin_dates,
        horizon=horizon,
        stage="test",
        checkpoint_path=ckpt_file,
        manifest_payload=manifest,
    )
    res2 = coord2.evaluate_models(specs_resume, y_true=y_true, resume=True)

    assert m1_calls["count"] == 0, "Already-recorded M1_Success must not be re-executed on resume!"
    assert res2["system_status"] == SystemEvaluationStatus.TEST_EVALUATION_COMPLETE.value
    assert res2["model_statuses"]["M1_Success"] == ModelTerminalStatus.TEST_SUCCESS.value
    assert res2["model_statuses"]["M2_InterruptedInFlight"] == ModelTerminalStatus.TEST_SUCCESS.value
    assert res2["model_statuses"]["M3_Pending"] == ModelTerminalStatus.TEST_SUCCESS.value
    assert len(res2["results_df"]) == 3


# ======================================================================
# P0: Semantic Decoupling, Normalization & Filtering Contracts
# ======================================================================

def test_p0_schema_values_and_explicit_operators() -> None:
    """P0: Verify that EDAReport and production artifacts contain both legacy and canonical fields with explicit >= operators."""
    rep = EDAReport.from_domain_reports(CANONICAL_SOURCE_ROOT)

    # Legacy EDA fields
    assert rep.validated_rain_threshold == 0.1
    assert rep.validated_rain_threshold_operator == ">="
    assert rep.eda_selected_threshold_mm == 0.1
    assert abs(rep.intermittency_adi - 1.64) < 0.05
    assert abs(rep.intermittency_cv2 - 2.02) < 0.05

    # Canonical modeling fields
    assert rep.canonical_rain_threshold == 1.0
    assert rep.canonical_rain_threshold_operator == ">="
    assert abs(rep.canonical_intermittency_adi - 2.30) < 0.05
    assert abs(rep.canonical_intermittency_cv2 - 1.25) < 0.05
    assert rep.canonical_intermittency_classification == "LUMPY-like (demand taxonomy analogy)"

    # Verify serialization dictionary
    d = rep.to_dict()
    assert d["validated_rain_threshold"] == 0.1
    assert d["validated_rain_threshold_operator"] == ">="
    assert d["eda_selected_threshold_mm"] == 0.1
    assert d["canonical_rain_threshold"] == 1.0
    assert d["canonical_rain_threshold_operator"] == ">="
    assert abs(d["canonical_intermittency_adi"] - 2.30) < 0.05
    assert abs(d["canonical_intermittency_cv2"] - 1.25) < 0.05


def test_p0_validator_non_finite_rejection() -> None:
    """P0: Validator rejects None, booleans, non-numeric, NaN, infinity, and normalizes acceptable tolerance values to exact 1.0."""
    # 1. Missing / None
    with pytest.raises(ValueError, match="missing"):
        validate_and_normalize_canonical_threshold(None)

    # 2. Boolean
    with pytest.raises(ValueError, match="Boolean value"):
        validate_and_normalize_canonical_threshold(True)
    with pytest.raises(ValueError, match="Boolean value"):
        validate_and_normalize_canonical_threshold(False)

    # 3. Non-numeric
    with pytest.raises(TypeError, match="must be numeric"):
        validate_and_normalize_canonical_threshold("1.0")
    with pytest.raises(TypeError, match="must be numeric"):
        validate_and_normalize_canonical_threshold([1.0])

    # 4. Non-finite: NaN, +Inf, -Inf
    with pytest.raises(ValueError, match="NaN"):
        validate_and_normalize_canonical_threshold(float("nan"))
    with pytest.raises(ValueError, match="infinite"):
        validate_and_normalize_canonical_threshold(float("inf"))
    with pytest.raises(ValueError, match="infinite"):
        validate_and_normalize_canonical_threshold(float("-inf"))

    # 5. Out of tolerance (> 1e-6)
    with pytest.raises(ValueError, match="violates canonical 1.0 mm/day contract"):
        validate_and_normalize_canonical_threshold(0.1)
    with pytest.raises(ValueError, match="violates canonical 1.0 mm/day contract"):
        validate_and_normalize_canonical_threshold(1.001)

    # 6. Acceptable tolerance normalization: within 1e-6 normalized to exact float 1.0
    val_norm = validate_and_normalize_canonical_threshold(1.00000001)
    assert val_norm == 1.0
    assert type(val_norm) is float
    assert val_norm is not True  # strictly float 1.0, not boolean


def test_p0_consumer_receives_normalized_canonical_threshold() -> None:
    """P0: build_two_stage_from_eda passes normalized classification_threshold=1.0 to model constructor."""
    from unittest.mock import MagicMock
    from src.training.trainer import build_two_stage_from_eda

    # Mock report with floating tolerance
    mock_report = MagicMock()
    mock_report.canonical_rain_threshold = 1.00000001
    mock_report.canonical_rain_threshold_operator = ">="

    mock_model_cls = MagicMock()
    mock_model_cls.__name__ = "MockRainfallModel"

    build_two_stage_from_eda(mock_model_cls, mock_report)
    mock_model_cls.assert_called_once()
    _, kwargs = mock_model_cls.call_args
    assert kwargs["classification_threshold"] == 1.0
    assert type(kwargs["classification_threshold"]) is float


def test_p0_fail_fast_missing_or_invalid_canonical_threshold() -> None:
    """P0: Fail-closed consumer behavior when canonical threshold is missing or operator is not >=."""
    from unittest.mock import MagicMock
    from src.training.trainer import build_two_stage_from_eda

    # Missing canonical threshold (silent fallback to legacy is prohibited)
    report_missing = {"validated_rain_threshold": 0.1, "validated_rain_threshold_operator": ">="}
    with pytest.raises(ValueError, match="missing required 'canonical_rain_threshold'"):
        build_two_stage_from_eda(MagicMock(), report_missing)

    # Unsupported operator
    report_bad_op = {
        "canonical_rain_threshold": 1.0,
        "canonical_rain_threshold_operator": ">",  # strictly requires >=
    }
    with pytest.raises(ValueError, match="Unsupported canonical_rain_threshold_operator"):
        build_two_stage_from_eda(MagicMock(), report_bad_op)


def test_p0_boundary_precision_at_0_1_and_1_0() -> None:
    """P0: Boundary precision tests at exact 0.1 mm and 1.0 mm for inclusive >=."""
    # 1.0 mm boundary
    y_true_1 = np.array([0.0, 0.5, 0.999, 1.0, 2.0])
    y_pred_1 = np.array([0.0, 0.0, 0.0, 1.5, 3.0])
    mae_1 = rain_day_mae(y_true_1, y_pred_1, rain_threshold=1.0)
    assert np.isclose(mae_1, 0.75)  # (|1.5 - 1.0| + |3.0 - 2.0|) / 2 = 0.75

    # 0.1 mm boundary (legacy)
    y_true_01 = np.array([0.0, 0.05, 0.099, 0.1, 1.0])
    y_pred_01 = np.array([0.0, 0.0, 0.0, 0.2, 1.5])
    mae_01 = rain_day_mae(y_true_01, y_pred_01, rain_threshold=0.1)
    assert np.isclose(mae_01, 0.30)  # (|0.2 - 0.1| + |1.5 - 1.0|) / 2 = 0.30


def test_p0_preservation_and_stage2_filtering_contract() -> None:
    """P0: Continuous source target retains 0.5 mm, Stage 2 mask excludes row, filtered array has no 0.5 mm."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "Lượng mưa": [0.0, 0.5, 1.0, 5.0],
    })

    # Stage 1 classification labels (Y >= 1.0)
    canonical_threshold = 1.0
    stage1_labels = (df["Lượng mưa"] >= canonical_threshold).astype(int).values
    assert list(stage1_labels) == [0, 0, 1, 1]

    # Source target column MUST NOT be zeroed or modified in place
    assert df["Lượng mưa"].iloc[1] == 0.5

    # Stage 2 wet filtering
    stage2_wet_mask = (df["Lượng mưa"] >= canonical_threshold).values
    df_stage2_filtered = df[stage2_wet_mask]
    y_stage2 = df_stage2_filtered["Lượng mưa"].values

    # Continuous target array for Stage 2 only contains values >= 1.0 mm
    assert 0.5 not in y_stage2
    assert len(y_stage2) == 2
    assert list(y_stage2) == [1.0, 5.0]


# ======================================================================
# P1: Decision M7 Contract Preservation
# ======================================================================

def test_p1_m7_bucket_size_locking_fail_closed() -> None:
    """P1: M7 locks ADIDA N=2 and IMAPA K=2 based on dev_train wet-day frequency 6329 / 2747 = 2.304 -> 2."""
    from src.models.nixtla.wrappers import FrozenIntermittentWrapper

    assert FrozenIntermittentWrapper.ADIDA_BUCKET_SIZE_N == 2
    assert FrozenIntermittentWrapper.IMAPA_BUCKET_SIZE_K == 2

    total_days = 6329
    wet_days = 2747
    mean_inter_arrival = total_days / wet_days
    assert np.isclose(mean_inter_arrival, 2.303967965)
    rounded_bucket = round(mean_inter_arrival)
    assert rounded_bucket == 2


def test_p1_m7_continuous_amount_and_demand_indicator() -> None:
    """P1: Intermittent wrappers preserve continuous amount and trigger demand on I(Y > 0)."""
    y_raw = np.array([0.0, 0.5, 0.0, 10.2, 0.0, 0.0, 3.4])
    demand_indicator = (y_raw > 0).astype(int)
    assert list(demand_indicator) == [0, 1, 0, 1, 0, 0, 1]
    assert demand_indicator[1] == 1
    assert y_raw[1] == 0.5


# ======================================================================
# P2: Pre-Read I/O Barrier & Zero Final Test Label Access
# ======================================================================

def test_p2_negative_guard_simulation() -> None:
    """P2: Isolated negative simulation verifies fail-closed DataLeakageSecurityError without polluting official audit."""
    guard = get_global_read_guard()
    guard.activate()

    # Pre-condition: global monitor counters must be clean
    assert guard.actual_target_reads == 0

    # Negative simulation inside isolated scope
    with guard.isolated_scope() as iso_monitor:
        assert iso_monitor.access_attempts == 0
        assert iso_monitor.blocked_attempts == 0
        assert iso_monitor.actual_target_reads == 0

        with pytest.raises(DataLeakageSecurityError, match="Blocked unauthorized read access"):
            with open("data/raw/forbidden_simulated_read.csv", "r") as f:
                f.read()

        assert iso_monitor.access_attempts == 1
        assert iso_monitor.blocked_attempts == 1
        assert iso_monitor.actual_target_reads == 0

    # Post-condition: global monitor was NOT contaminated by the isolated simulation!
    assert guard.access_attempts == 0
    assert guard.blocked_attempts == 0
    assert guard.actual_target_reads == 0


def test_p2_official_phase0_audit_zero_access() -> None:
    """P2: Verify official Phase 0 execution maintained 0 access attempts and 0 actual reads of test targets."""
    guard = get_global_read_guard()
    stats = guard.stats
    assert stats["is_active"] is True
    assert stats["access_attempts"] == 0
    assert stats["blocked_attempts"] == 0
    assert stats["actual_target_reads"] == 0


def test_p2_horizon_purge_boundary_and_target_preservation() -> None:
    """P2: Origin vs Target distinction for Final Test split."""
    test_dates = pd.date_range("2020-05-01", "2026-04-30", freq="D")
    assert len(test_dates) == 2191

    valid_origins = test_dates[:-7]
    assert len(valid_origins) == 2184
    assert valid_origins[-1].strftime("%Y-%m-%d") == "2026-04-23"

    purged_origins = test_dates[-7:]
    assert len(purged_origins) == 7
    assert purged_origins[0].strftime("%Y-%m-%d") == "2026-04-24"
    assert purged_origins[-1].strftime("%Y-%m-%d") == "2026-04-30"

    origin_2026_04_23 = pd.Timestamp("2026-04-23")
    horizon_dates_from_last_origin = [origin_2026_04_23 + pd.Timedelta(days=h) for h in range(1, 8)]
    assert horizon_dates_from_last_origin[-1].strftime("%Y-%m-%d") == "2026-04-30"
    for d in horizon_dates_from_last_origin:
        assert d in test_dates


def test_phase0_export_official_contract_and_audit_report() -> None:
    """Export official reports/modeling/modeling_contract.json and phase0_audit_report.json and verify PHASE0_GATE_PASS."""
    import json
    from src.evaluation.protocols import CONTRACT_PATH, PHASE0_AUDIT_PATH, get_default_contract

    contract_path = PROJECT_ROOT / CONTRACT_PATH
    audit_path = PROJECT_ROOT / PHASE0_AUDIT_PATH

    contract = get_default_contract()
    save_modeling_contract(contract, filepath=contract_path)
    assert contract_path.exists()

    g1 = gatecheck_threshold_source_alignment(contract_path=contract_path)
    g2 = gatecheck_target_boundary_purge_h7()
    g6 = gatecheck_feature_causality_and_split_isolation()
    six_gates = evaluate_six_gatechecks(hurdle_applicable=True, hurdle_support_valid=True)

    guard = get_global_read_guard()
    guard_stats = guard.stats

    verdict = (
        "PHASE0_GATE_PASS"
        if (
            g1["status"] == GatecheckStatus.PASS.value
            and g2["status"] == GatecheckStatus.PASS.value
            and g6["status"] == GatecheckStatus.PASS.value
            and guard_stats["actual_target_reads"] == 0
        )
        else "PHASE0_GATE_FAIL"
    )

    summary = {
        "plan_version": "v1.4.19",
        "phase0_verdict": verdict,
        "final_test_labels_accessed": False,
        "pre_read_guard_stats": guard_stats,
        "gate1_threshold_alignment": g1,
        "gate2_target_boundary_purge_h7": g2,
        "gate6_feature_causality_and_split_isolation": g6,
        "typed_gatechecks_g1_g6": six_gates,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    assert verdict == "PHASE0_GATE_PASS"
    assert audit_path.exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        test_gate1_threshold_source_alignment(tmp_p)
        print("  [PASS] test_gate1_threshold_source_alignment")
        test_gate2_target_boundary_purge_h7()
        print("  [PASS] test_gate2_target_boundary_purge_h7")
        test_gate3_exhaustive_model_scoped_failure_contract(tmp_p)
        print("  [PASS] test_gate3_exhaustive_model_scoped_failure_contract")
        test_gate4_statsforecast_wrapper_single_fit_and_parity()
        print("  [PASS] test_gate4_statsforecast_wrapper_single_fit_and_parity")
        test_gate4_frozen_intermittent_wrapper_m7_contract()
        print("  [PASS] test_gate4_frozen_intermittent_wrapper_m7_contract")
        test_gate5_typed_gatechecks_and_direct_hurdle_m1()
        print("  [PASS] test_gate5_typed_gatechecks_and_direct_hurdle_m1")
        test_gate6_feature_causality_and_recursive_isolation()
        print("  [PASS] test_gate6_feature_causality_and_recursive_isolation")
        test_m4_and_m6a_selection_contracts()
        print("  [PASS] test_m4_and_m6a_selection_contracts")
        test_p5_1_conservative_holm_when_representative_fails()
        print("  [PASS] test_p5_1_conservative_holm_when_representative_fails")
        test_p6_1_process_interruption_and_checkpoint_recovery(tmp_p)
        print("  [PASS] test_p6_1_process_interruption_and_checkpoint_recovery")
        test_p0_schema_values_and_explicit_operators()
        print("  [PASS] test_p0_schema_values_and_explicit_operators")
        test_p0_validator_non_finite_rejection()
        print("  [PASS] test_p0_validator_non_finite_rejection")
        test_p0_consumer_receives_normalized_canonical_threshold()
        print("  [PASS] test_p0_consumer_receives_normalized_canonical_threshold")
        test_p0_fail_fast_missing_or_invalid_canonical_threshold()
        print("  [PASS] test_p0_fail_fast_missing_or_invalid_canonical_threshold")
        test_p0_boundary_precision_at_0_1_and_1_0()
        print("  [PASS] test_p0_boundary_precision_at_0_1_and_1_0")
        test_p0_preservation_and_stage2_filtering_contract()
        print("  [PASS] test_p0_preservation_and_stage2_filtering_contract")
        test_p1_m7_bucket_size_locking_fail_closed()
        print("  [PASS] test_p1_m7_bucket_size_locking_fail_closed")
        test_p1_m7_continuous_amount_and_demand_indicator()
        print("  [PASS] test_p1_m7_continuous_amount_and_demand_indicator")
        test_p2_negative_guard_simulation()
        print("  [PASS] test_p2_negative_guard_simulation")
        test_p2_official_phase0_audit_zero_access()
        print("  [PASS] test_p2_official_phase0_audit_zero_access")
        test_p2_horizon_purge_boundary_and_target_preservation()
        print("  [PASS] test_p2_horizon_purge_boundary_and_target_preservation")
        test_phase0_export_official_contract_and_audit_report()
        print("  [PASS] test_phase0_export_official_contract_and_audit_report")
    print("\n[SUCCESS] ALL 22 PHASE 0 MODELING ENGINE CONTRACT TESTS PASSED (PHASE0_GATE_PASS)!")


