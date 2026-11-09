"""Single-fit causal rolling-origin wrappers for Nixtla StatsForecast & Intermittent models.

Enforces Modeling Plan v1.4.18:
- §1.3.C & M2 Single-Fit Policy: Fit once on Dev Train (or Full Train in Phase 6), freeze parameters,
  and evaluate causally at each rolling origin t via forward(y_history, h=7).
- §1.3.D & Decision M7 (Option B): FrozenIntermittentWrapper preserves continuous target Y_t >= 0,
  internal event indicator I(Y_t > 0), freezes smoothing parameters on Dev Train, and locks
  domain-informed bucket sizes N = 2 for ADIDA and K = 2 (levels 1, 2) for IMAPA derived from
  Dev Train wet-day spacing k_{0, >= 1.0} = 6329 / 2747 = 2.304 -> 2 (strictly rejecting N=1 or K=1).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def _deep_snapshot_state(obj: Any) -> Dict[str, Any]:
    """Create a deterministic snapshot of fitted model attributes for invariance verification."""
    snap: Dict[str, Any] = {}
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k == "_state_snapshot":
                continue
            if isinstance(v, np.ndarray):
                snap[k] = v.copy()
            elif isinstance(v, (int, float, str, bool, tuple, list, dict)):
                snap[k] = copy.deepcopy(v)
    return snap


def _values_equal(v_before: Any, v_after: Any) -> bool:
    """Recursively compare arbitrary Python / NumPy structures for exact state invariance."""
    if isinstance(v_before, np.ndarray) or isinstance(v_after, np.ndarray):
        return bool(np.array_equal(np.asarray(v_before), np.asarray(v_after), equal_nan=True))
    if isinstance(v_before, dict) and isinstance(v_after, dict):
        if set(v_before.keys()) != set(v_after.keys()):
            return False
        return all(_values_equal(v_before[k], v_after[k]) for k in v_before)
    if isinstance(v_before, (list, tuple)) and isinstance(v_after, (list, tuple)):
        if len(v_before) != len(v_after):
            return False
        return all(_values_equal(x, y) for x, y in zip(v_before, v_after))
    if isinstance(v_before, (float, np.floating)) and isinstance(v_after, (float, np.floating)):
        if np.isnan(v_before) and np.isnan(v_after):
            return True
        return bool(v_before == v_after)
    return bool(v_before == v_after)


def _assert_snapshots_equal(snap_before: Dict[str, Any], snap_after: Dict[str, Any], context: str = "") -> None:
    """Assert two state snapshots are identical (no parameter mutation during forward())."""
    if set(snap_before.keys()) != set(snap_after.keys()):
        raise RuntimeError(f"Model state keys mutated during forward() [{context}]")
    for k, v_before in snap_before.items():
        v_after = snap_after[k]
        if not _values_equal(v_before, v_after):
            raise RuntimeError(f"Model attribute '{k}' mutated during forward() [{context}]")


def _fit_ses_alpha(series: np.ndarray, grid_size: int = 99) -> float:
    """Fit Simple Exponential Smoothing (SES) smoothing parameter alpha in [0.01, 0.99] via one-step MSE."""
    y = np.asarray(series, dtype=np.float64).ravel()
    if len(y) <= 1:
        return 0.2
    alphas = np.linspace(0.01, 0.99, grid_size)
    best_alpha = 0.2
    best_mse = float("inf")
    for a in alphas:
        level = y[0]
        sq_err = 0.0
        for t in range(1, len(y)):
            err = y[t] - level
            sq_err += err * err
            level = a * y[t] + (1.0 - a) * level
        mse = sq_err / (len(y) - 1)
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(a)
    return best_alpha


def _apply_ses_level(series: np.ndarray, alpha: float) -> float:
    """Filter 1D series with frozen SES parameter alpha and return terminal level."""
    y = np.asarray(series, dtype=np.float64).ravel()
    if len(y) == 0:
        return 0.0
    level = float(y[0])
    for t in range(1, len(y)):
        level = alpha * float(y[t]) + (1.0 - alpha) * level
    return max(0.0, float(level))


def _extract_croston_demands_and_intervals(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract non-zero demand sizes (Y_t > 0) and inter-demand intervals (strictly I(Y_t > 0))."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    nz_indices = np.flatnonzero(y_arr > 0.0)
    if len(nz_indices) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    demands = y_arr[nz_indices]
    intervals = np.empty(len(nz_indices), dtype=np.float64)
    intervals[0] = float(nz_indices[0] + 1)
    if len(nz_indices) > 1:
        intervals[1:] = np.diff(nz_indices).astype(np.float64)
    return demands, intervals


class StatsForecastWrapper:
    """Single-fit causal rolling-origin wrapper for Nixtla StatsForecast models (§1.3.C).

    Fits the underlying StatsForecast model once on the training series (`fit(y_train)`),
    freezes `self.model_`, and evaluates causally at any forecast origin via
    `forward(y_history, h=7)` without refitting.
    """

    def __init__(self, model: Any, alias: Optional[str] = None) -> None:
        self.base_model = copy.deepcopy(model)
        self.alias = alias or getattr(model, "alias", None) or type(model).__name__
        self.model_: Optional[Any] = None
        self.fitted_: bool = False
        self.n_train_: int = 0
        self._state_snapshot: Dict[str, Any] = {}

    def fit(self, y_train: Sequence[float], X_train: Optional[np.ndarray] = None) -> "StatsForecastWrapper":
        """Fit model once on training series and freeze parameters."""
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        if len(y_arr) == 0:
            raise ValueError("y_train cannot be empty.")
        if np.any(~np.isfinite(y_arr)):
            raise ValueError("y_train contains NaN or Inf.")

        self.model_ = copy.deepcopy(self.base_model)
        self.model_.fit(y=y_arr, X=X_train)
        self.fitted_ = True
        self.n_train_ = int(len(y_arr))
        self._state_snapshot = _deep_snapshot_state(self.model_)
        return self

    def forward(
        self,
        y_history: Sequence[float],
        h: int = 7,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply frozen model causally to observed history up to origin t.

        Returns:
            np.ndarray of shape (1, h), finite and clipped at >= 0.0 mm.
        """
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("StatsForecastWrapper must be fitted before calling forward().")
        y_hist = np.asarray(y_history, dtype=np.float64).ravel()
        if len(y_hist) == 0:
            raise ValueError("y_history cannot be empty.")
        if np.any(~np.isfinite(y_hist)):
            raise ValueError("y_history contains NaN or Inf.")

        try:
            res = self.model_.forward(y=y_hist, h=h, X=X, X_future=X_future)
            preds = np.asarray(res["mean"], dtype=np.float64).ravel()
        except NotImplementedError:
            # Stateless baseline fallback (e.g. Naive / SeasonalNaive if forward not implemented)
            cls_name = type(self.model_).__name__
            if cls_name == "Naive":
                preds = np.repeat(float(y_hist[-1]), h)
            elif cls_name == "SeasonalNaive":
                m = int(getattr(self.model_, "season_length", 7))
                if len(y_hist) >= m:
                    tail = y_hist[-m:]
                    reps = int(np.ceil(h / m))
                    preds = np.tile(tail, reps)[:h].astype(np.float64)
                else:
                    preds = np.repeat(float(y_hist[-1]), h)
            else:
                raise

        # Verify fitted model state was not mutated by forward()
        current_snap = _deep_snapshot_state(self.model_)
        _assert_snapshots_equal(self._state_snapshot, current_snap, context=self.alias)

        preds = np.clip(preds, 0.0, None)
        if len(preds) != h or np.any(~np.isfinite(preds)):
            raise ValueError(f"Invalid forecast output from {self.alias}: shape={preds.shape}")
        return preds.reshape(1, h)

    def predict_rolling(
        self,
        y_full: Sequence[float],
        origin_indices: Sequence[int],
        h: int = 7,
    ) -> np.ndarray:
        """Generate causal rolling-origin forecasts across specified origin indices.

        Each origin index `idx` uses strictly causal history `y_full[:idx + 1]`.
        """
        if not self.fitted_:
            raise RuntimeError("StatsForecastWrapper must be fitted before predict_rolling().")
        y_arr = np.asarray(y_full, dtype=np.float64).ravel()
        rows = []
        for idx in origin_indices:
            y_hist = y_arr[: int(idx) + 1]
            rows.append(self.forward(y_hist, h=h)[0])
        if not rows:
            return np.empty((0, h), dtype=np.float64)
        return np.vstack(rows)


class FrozenIntermittentWrapper:
    """Single-fit causal wrapper for Group 2 Intermittent Demand models (§1.3.D & Decision M7 Option B).

    Supported models:
    - 'CrostonOptimized'
    - 'CrostonSBA'
    - 'TSB'
    - 'ADIDA'
    - 'IMAPA'

    Contract invariants (Decision M7 Option B):
    1. Continuous rainfall target Y_t >= 0.0 is preserved untouched (no pre-zeroing of 0 < Y_t < 1.0).
    2. Internal demand-event indicator I(Y_t > 0) is preserved.
    3. Smoothing parameters are fitted once on Dev Train and frozen for causal forward(y_history, h=7).
    4. ADIDA and IMAPA use domain-informed temporal bucket size derived from wet-day spacing at
       tau_rain = 1.0 mm/day:
           k_{0, >= 1.0} = N_total / N_{wet, >= 1.0}  (6329 / 2747 = 2.304 -> 2)
       locking N = 2 for ADIDA and K = 2 (aggregation levels k in {1, 2}) for IMAPA.
       Degenerate N = 1 or K = 1 is explicitly rejected.
    """

    SUPPORTED_MODELS = ("CrostonOptimized", "CrostonSBA", "TSB", "ADIDA", "IMAPA")
    ADIDA_BUCKET_SIZE_N: int = 2
    IMAPA_BUCKET_SIZE_K: int = 2

    def __init__(
        self,
        model_name: str,
        rain_threshold: float = 1.0,
        alpha_d: float = 0.2,
        alpha_p: float = 0.2,
        aggregation_chunk_size: Optional[int] = None,
        max_aggregation_level: Optional[int] = None,
        alias: Optional[str] = None,
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported intermittent model '{model_name}'. Must be one of {self.SUPPORTED_MODELS}."
            )
        if aggregation_chunk_size is not None and int(aggregation_chunk_size) < 2:
            raise ValueError("Decision M7 Violation: ADIDA aggregation_chunk_size N must be >= 2 (N=1 is degenerate).")
        if max_aggregation_level is not None and int(max_aggregation_level) < 2:
            raise ValueError("Decision M7 Violation: IMAPA max_aggregation_level K must be >= 2 (K=1 is degenerate).")

        self.model_name = model_name
        self.rain_threshold = float(rain_threshold)
        self.alpha_d_init = float(alpha_d)
        self.alpha_p_init = float(alpha_p)
        self.requested_chunk_size = aggregation_chunk_size
        self.requested_max_level = max_aggregation_level
        self.alias = alias or model_name

        # Fitted state attributes (frozen after fit())
        self.fitted_: bool = False
        self.n_train_: int = 0
        self.wet_day_spacing_k0_: float = np.nan
        self.domain_bucket_size_: int = 2
        self.alpha_d_: float = np.nan
        self.alpha_p_: float = np.nan
        self.sba_factor_: float = 1.0
        self.aggregation_chunk_size_: Optional[int] = None
        self.max_aggregation_level_: Optional[int] = None
        self.aggregation_levels_: Optional[Tuple[int, ...]] = None
        self.level_alphas_: Optional[Dict[int, float]] = None
        self._state_snapshot: Dict[str, Any] = {}

    def fit(self, y_train: Sequence[float], X_train: Optional[np.ndarray] = None) -> "FrozenIntermittentWrapper":
        """Fit smoothing and temporal aggregation parameters once on Dev Train and freeze them."""
        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        if len(y_arr) == 0:
            raise ValueError("y_train cannot be empty.")
        if np.any(~np.isfinite(y_arr)):
            raise ValueError("y_train contains NaN or Inf.")
        if np.any(y_arr < 0.0):
            raise ValueError("y_train contains negative rainfall values.")

        self.n_train_ = int(len(y_arr))
        n_wet_ge_tau = int(np.sum(y_arr >= self.rain_threshold))
        if n_wet_ge_tau > 0:
            self.wet_day_spacing_k0_ = float(len(y_arr) / n_wet_ge_tau)
            computed_bucket = int(max(2, round(self.wet_day_spacing_k0_)))
        else:
            self.wet_day_spacing_k0_ = 2.0
            computed_bucket = 2
        self.domain_bucket_size_ = computed_bucket

        demands, intervals = _extract_croston_demands_and_intervals(y_arr)

        if self.model_name == "CrostonOptimized":
            alpha_star = _fit_ses_alpha(demands) if len(demands) > 1 else 0.1
            self.alpha_d_ = float(alpha_star)
            self.alpha_p_ = float(alpha_star)
            self.sba_factor_ = 1.0

        elif self.model_name == "CrostonSBA":
            alpha_star = _fit_ses_alpha(demands) if len(demands) > 1 else 0.1
            self.alpha_d_ = float(alpha_star)
            self.alpha_p_ = float(alpha_star)
            self.sba_factor_ = float(1.0 - (alpha_star / 2.0))

        elif self.model_name == "TSB":
            self.alpha_d_ = float(self.alpha_d_init)
            self.alpha_p_ = float(self.alpha_p_init)
            self.sba_factor_ = 1.0

        elif self.model_name == "ADIDA":
            chunk_n = int(self.requested_chunk_size if self.requested_chunk_size is not None else self.domain_bucket_size_)
            if chunk_n < 2:
                raise ValueError("Decision M7 Violation: ADIDA chunk size N must be >= 2.")
            self.aggregation_chunk_size_ = chunk_n
            n_chunks = len(y_arr) // chunk_n
            if n_chunks >= 2:
                trimmed = y_arr[len(y_arr) - n_chunks * chunk_n :]
                agg_series = trimmed.reshape(n_chunks, chunk_n).sum(axis=1)
                alpha_agg = _fit_ses_alpha(agg_series)
            else:
                alpha_agg = 0.2
            self.alpha_d_ = float(alpha_agg)
            self.alpha_p_ = float(alpha_agg)

        elif self.model_name == "IMAPA":
            max_k = int(self.requested_max_level if self.requested_max_level is not None else self.domain_bucket_size_)
            if max_k < 2:
                raise ValueError("Decision M7 Violation: IMAPA max aggregation level K must be >= 2.")
            self.max_aggregation_level_ = max_k
            self.aggregation_levels_ = tuple(range(1, max_k + 1))
            level_alphas: Dict[int, float] = {}
            for k in self.aggregation_levels_:
                n_chunks = len(y_arr) // k
                if n_chunks >= 2:
                    trimmed = y_arr[len(y_arr) - n_chunks * k :]
                    agg_k = trimmed.reshape(n_chunks, k).sum(axis=1)
                    level_alphas[k] = float(_fit_ses_alpha(agg_k))
                else:
                    level_alphas[k] = 0.2
            self.level_alphas_ = level_alphas
            self.alpha_d_ = float(np.mean(list(level_alphas.values())))
            self.alpha_p_ = self.alpha_d_

        self.fitted_ = True
        self._state_snapshot = _deep_snapshot_state(self)
        return self

    def forward(
        self,
        y_history: Sequence[float],
        h: int = 7,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply frozen intermittent filter causally to observed history up to origin t.

        Preserves continuous values in y_history and internal event indicator I(Y_t > 0).
        Returns shape (1, h), non-negative and finite.
        """
        if not self.fitted_:
            raise RuntimeError("FrozenIntermittentWrapper must be fitted before calling forward().")
        y_hist = np.asarray(y_history, dtype=np.float64).ravel()
        if len(y_hist) == 0:
            raise ValueError("y_history cannot be empty.")
        if np.any(~np.isfinite(y_hist)):
            raise ValueError("y_history contains NaN or Inf.")

        if self.model_name in ("CrostonOptimized", "CrostonSBA"):
            demands, intervals = _extract_croston_demands_and_intervals(y_hist)
            if len(demands) == 0:
                rate = 0.0
            else:
                z_hat = _apply_ses_level(demands, self.alpha_d_)
                p_hat = max(1.0, _apply_ses_level(intervals, self.alpha_p_))
                rate = self.sba_factor_ * (z_hat / p_hat)
            preds = np.repeat(max(0.0, float(rate)), h)

        elif self.model_name == "TSB":
            # Teunter-Syntetos-Babai causal filter over I(Y_t > 0)
            nz_mask = y_hist > 0.0
            if not np.any(nz_mask):
                rate = 0.0
            else:
                first_nz = int(np.flatnonzero(nz_mask)[0])
                z_t = float(y_hist[first_nz])
                p_t = float(np.mean(nz_mask[: first_nz + 1]))
                ad = self.alpha_d_
                ap = self.alpha_p_
                for t in range(len(y_hist)):
                    if y_hist[t] > 0.0:
                        p_t = p_t + ap * (1.0 - p_t)
                        z_t = z_t + ad * (float(y_hist[t]) - z_t)
                    else:
                        p_t = p_t + ap * (0.0 - p_t)
                rate = max(0.0, p_t * z_t)
            preds = np.repeat(float(rate), h)

        elif self.model_name == "ADIDA":
            chunk_n = int(self.aggregation_chunk_size_ or 2)
            if chunk_n < 2:
                raise RuntimeError("Decision M7 Violation: ADIDA chunk_n < 2.")
            n_chunks = len(y_hist) // chunk_n
            if n_chunks >= 1:
                trimmed = y_hist[len(y_hist) - n_chunks * chunk_n :]
                agg_series = trimmed.reshape(n_chunks, chunk_n).sum(axis=1)
                agg_forecast = _apply_ses_level(agg_series, self.alpha_d_)
                daily_rate = agg_forecast / float(chunk_n)
            else:
                daily_rate = float(np.mean(y_hist))
            preds = np.repeat(max(0.0, float(daily_rate)), h)

        elif self.model_name == "IMAPA":
            levels = self.aggregation_levels_ or (1, 2)
            if max(levels) < 2:
                raise RuntimeError("Decision M7 Violation: IMAPA max aggregation level < 2.")
            level_rates: List[float] = []
            for k in levels:
                n_chunks = len(y_hist) // k
                alpha_k = (self.level_alphas_ or {}).get(k, self.alpha_d_)
                if n_chunks >= 1:
                    trimmed = y_hist[len(y_hist) - n_chunks * k :]
                    agg_k = trimmed.reshape(n_chunks, k).sum(axis=1)
                    agg_forecast_k = _apply_ses_level(agg_k, float(alpha_k))
                    level_rates.append(agg_forecast_k / float(k))
                else:
                    level_rates.append(float(np.mean(y_hist)))
            daily_rate = float(np.mean(level_rates))
            preds = np.repeat(max(0.0, daily_rate), h)

        else:
            raise ValueError(f"Unknown intermittent model: {self.model_name}")

        # Verify fitted state invariance
        current_snap = _deep_snapshot_state(self)
        _assert_snapshots_equal(self._state_snapshot, current_snap, context=self.alias)

        preds = np.clip(np.asarray(preds, dtype=np.float64), 0.0, None)
        if len(preds) != h or np.any(~np.isfinite(preds)):
            raise ValueError(f"Invalid forecast output from {self.alias}: shape={preds.shape}")
        return preds.reshape(1, h)

    def predict_rolling(
        self,
        y_full: Sequence[float],
        origin_indices: Sequence[int],
        h: int = 7,
    ) -> np.ndarray:
        """Generate causal rolling-origin forecasts across specified origin indices."""
        if not self.fitted_:
            raise RuntimeError("FrozenIntermittentWrapper must be fitted before predict_rolling().")
        y_arr = np.asarray(y_full, dtype=np.float64).ravel()
        rows = []
        for idx in origin_indices:
            y_hist = y_arr[: int(idx) + 1]
            rows.append(self.forward(y_hist, h=h)[0])
        if not rows:
            return np.empty((0, h), dtype=np.float64)
        return np.vstack(rows)

