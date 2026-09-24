"""
Recursive multi-step forecaster for ML-based rainfall models.

.. deprecated:: 2.0
   This class is legacy-only. It is preserved strictly for running retrospective
   comparisons of the legacy two-stage models against direct multi-horizon models.
   New forecasting pipelines should use `src.models.nixtla.MLForecastAdapter`
   which utilizes direct multi-horizon forecasting (`max_horizon=H`) to avoid
   recursive error compounding.

Generates an N-step-ahead forecast by iterating one step at a time:
predict → append prediction to history → recompute features → predict
again.  Uses ``FeatureBuilder.build_single_step()`` to guarantee the
exact same feature code path as batch training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from ...config.resolve import resolve_target_col, resolve_date_col


class RecursiveForecaster:
    """Multi-step recursive forecaster.

    Usage::

        forecaster = RecursiveForecaster(model, feature_builder)
        forecast_df = forecaster.forecast(history, steps=7)
        # forecast_df has columns: [date, prediction, horizon]
    """

    def __init__(
        self,
        model,
        feature_builder,
        clf_threshold: float = None,
        strategy: str = "hard_gate",
    ):
        """
        Args:
            model: A fitted ``BaseRainfallModel`` or ``BaseTimeSeriesModel``.
            feature_builder: A fitted ``FeatureBuilder`` instance.
            clf_threshold: Decision threshold tau for hard_gate two-stage models.
                If ``None``, uses ``model.optimal_threshold_`` or 0.5.
            strategy: Inference policy for two-stage models ('hard_gate' or 'expected').
                - 'hard_gate': Uses binary cutoff (p >= tau -> mu, else 0.0).
                - 'expected': Deterministic plug-in mean approximation (p * mu).
                  Caveat: The predicted mean is fed back into lag/rolling features;
                  therefore predictive uncertainty is not explicitly propagated and
                  the recursive approximation is not a fully probabilistically coherent
                  rollout (E[f(Y)] != f(E[Y])). The expected strategy may reduce the
                  discontinuity and error amplification caused specifically by hard
                  binary gating, while recursive error propagation still remains.
        """
        self.model = model
        self.feature_builder = feature_builder
        self.clf_threshold = (
            clf_threshold
            or getattr(model, 'optimal_threshold_', None)
            or 0.5
        )
        self.strategy = strategy
        self.target_col = feature_builder.target_col
        self.date_col = feature_builder.date_col

    def forecast(
        self,
        history: pd.DataFrame,
        steps: int = 7,
        future_exog: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate a recursive N-step forecast.

        Args:
            history: Historical DataFrame (must include target + date +
                all feature source columns).  Should have at least
                ``max(lag_periods + rolling_windows) + 10`` rows.
            steps: Number of future steps (days) to predict.
            future_exog: Optional DataFrame of known future exogenous
                variables (e.g. satellite-derived temperature forecasts).
                Must have *steps* rows.  If ``None``, the last known
                exogenous values are forward-filled (climatology fallback).
            strategy: Optional override for inference policy ('hard_gate' or 'expected').

        Returns:
            DataFrame with columns ``['date', 'prediction', 'horizon']``.
        """
        current_strategy = strategy or self.strategy
        history = history.copy()
        predictions = []
        target_col = self.target_col
        date_col = self.date_col

        # Determine the last date in history
        last_date = history[date_col].max()

        for step in range(1, steps + 1):
            # --- 1. Compute features for the next timestep ---
            try:
                feature_row = self.feature_builder.build_single_step(history)
            except Exception as e:
                print(f"   [h={step}] build_single_step failed: {e}")
                break

            # --- 2. Drop target and date before prediction ---
            feature_cols = [
                c for c in feature_row.columns
                if c != target_col and c != date_col
            ]
            X = feature_row[feature_cols]

            # --- 3. Predict ---
            prob_value = None
            mu_value = None
            model = self.model
            if hasattr(model, 'predict') and getattr(model, 'use_two_stage', False):
                clf_probs, reg_preds = model.predict(X)
                prob_value = float(clf_probs[0])
                mu_value = max(0.0, float(reg_preds[0]))
                if current_strategy == "expected":
                    pred_value = max(0.0, prob_value * mu_value)
                else:
                    pred_value = mu_value if prob_value >= self.clf_threshold else 0.0
            else:
                pred_value = self._predict_one(X, strategy=current_strategy)

            # Clamp to non-negative (rainfall cannot be negative)
            pred_value = max(0.0, float(pred_value))

            next_date = last_date + pd.Timedelta(days=step)

            rec = {
                'date': next_date,
                'prediction': pred_value,
                'horizon': step,
            }
            if prob_value is not None:
                rec['prob'] = prob_value
                rec['mu'] = mu_value
            predictions.append(rec)

            # --- 4. Append predicted row to history ---
            new_row = self._assemble_row(
                history, pred_value, next_date, future_exog, step
            )
            history = pd.concat([history, new_row], ignore_index=True)

        return pd.DataFrame(predictions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_one(self, X: pd.DataFrame, strategy: str = "hard_gate") -> float:
        """Predict a single value, handling two-stage vs single-stage."""
        model = self.model

        if hasattr(model, 'predict_combined') and getattr(model, 'use_two_stage', False):
            return float(
                model.predict_combined(
                    X,
                    threshold=self.clf_threshold,
                    strategy=strategy,
                )[0]
            )
        elif hasattr(model, 'predict'):
            result = model.predict(X)
            if isinstance(result, tuple):
                # Two-stage returns (clf_probs, reg_preds)
                clf_probs, reg_preds = result
                if strategy == "expected":
                    return float(max(0.0, clf_probs[0] * max(0.0, reg_preds[0])))
                else:
                    return float(max(0.0, reg_preds[0]) if clf_probs[0] >= self.clf_threshold else 0.0)
            return float(np.atleast_1d(result)[0])
        else:
            raise TypeError(f"Model {type(model)} does not have a predict method")

    def _assemble_row(
        self,
        history: pd.DataFrame,
        pred_value: float,
        pred_date: pd.Timestamp,
        future_exog: Optional[pd.DataFrame],
        step: int,
    ) -> pd.DataFrame:
        """Construct the next history row from prediction + exog.

        Strategy for exogenous variables:
        - If ``future_exog`` is provided, use the values for this step.
        - Otherwise, forward-fill from the last row of history
          (climatology / persistence fallback).
        """
        last_row = history.iloc[-1:].copy()
        new_row = last_row.copy()

        # Set the predicted target
        new_row[self.target_col] = pred_value
        new_row[self.date_col] = pred_date

        # Fill exogenous variables
        if future_exog is not None and step <= len(future_exog):
            exog_row = future_exog.iloc[step - 1]
            for col in exog_row.index:
                if col in new_row.columns and col not in (self.target_col, self.date_col):
                    new_row[col] = exog_row[col]
        # else: keeps forward-filled values from last_row (persistence)

        return new_row.reset_index(drop=True)
