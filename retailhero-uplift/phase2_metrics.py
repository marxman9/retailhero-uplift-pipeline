from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RankingEvaluation:
    summary: dict[str, float]
    grid_curve: pd.DataFrame
    qini_curve: pd.DataFrame


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def evaluate_ranking(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    avg_basket_value: float,
    sms_cost: float,
    n_grid_points: int = 100,
) -> RankingEvaluation:
    y_true = np.asarray(y_true, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if not (len(y_true) == len(treatment) == len(scores)):
        raise ValueError("y_true, treatment, and scores must have the same length.")

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    treatment_sorted = treatment[order]

    treated_mask = treatment_sorted == 1
    control_mask = ~treated_mask

    cum_y = np.cumsum(y_sorted)
    cum_treated = np.cumsum(treated_mask.astype(int))
    cum_control = np.cumsum(control_mask.astype(int))
    cum_y_treated = np.cumsum(y_sorted * treated_mask)
    cum_y_control = np.cumsum(y_sorted * control_mask)

    n_obs = len(y_sorted)
    overall_response_rate = float(y_true.mean())

    overall_treatment_rate = float(y_true[treatment == 1].mean())
    overall_control_rate = float(y_true[treatment == 0].mean())
    baseline_uplift = overall_treatment_rate - overall_control_rate

    x_full = np.arange(1, n_obs + 1, dtype=float) / n_obs
    qini_gain = cum_y_treated - (cum_y_control * _safe_divide(cum_treated, cum_control))
    qini_gain = qini_gain / n_obs
    total_qini_gain = float(qini_gain[-1])
    qini_baseline = x_full * total_qini_gain
    qini_auc = float(np.trapezoid(qini_gain - qini_baseline, x_full))

    grid_indices = np.unique(
        np.clip(
            np.ceil(np.linspace(1, n_obs, n_grid_points)).astype(int),
            1,
            n_obs,
        )
        - 1
    )
    alpha = (grid_indices + 1) / n_obs
    targeted_counts = grid_indices + 1

    subset_response_rate = cum_y[grid_indices] / targeted_counts
    subset_treatment_rate = _safe_divide(cum_y_treated[grid_indices], cum_treated[grid_indices])
    subset_control_rate = _safe_divide(cum_y_control[grid_indices], cum_control[grid_indices])
    subset_uplift = subset_treatment_rate - subset_control_rate
    lift = _safe_divide(subset_response_rate, np.full_like(subset_response_rate, overall_response_rate))
    if baseline_uplift != 0:
        liftup = subset_uplift / baseline_uplift
    else:
        liftup = np.zeros_like(subset_uplift)

    predictive_profit = alpha * (subset_response_rate * avg_basket_value - sms_cost)
    incremental_profit = alpha * (subset_uplift * avg_basket_value - sms_cost)

    grid_curve = pd.DataFrame(
        {
            "target_share": alpha,
            "target_count": targeted_counts,
            "response_rate": subset_response_rate,
            "treatment_rate": subset_treatment_rate,
            "control_rate": subset_control_rate,
            "uplift": subset_uplift,
            "lift": lift,
            "liftup": liftup,
            "predictive_profit": predictive_profit,
            "incremental_profit": incremental_profit,
        }
    )

    qini_curve = pd.DataFrame(
        {
            "target_share": np.concatenate([[0.0], x_full]),
            "qini_gain": np.concatenate([[0.0], qini_gain]),
            "qini_baseline": np.concatenate([[0.0], qini_baseline]),
        }
    )

    decile_idx = int(np.abs(alpha - 0.10).argmin())
    best_predictive_idx = int(np.argmax(predictive_profit))
    best_incremental_idx = int(np.argmax(incremental_profit))

    summary = {
        "overall_response_rate": overall_response_rate,
        "baseline_uplift": float(baseline_uplift),
        "qini_auc": qini_auc,
        "top_decile_lift": float(lift[decile_idx]),
        "top_decile_liftup": float(liftup[decile_idx]),
        "max_liftup": float(np.nanmax(liftup)),
        "predictive_mp": float(predictive_profit[best_predictive_idx]),
        "predictive_optimal_target_share": float(alpha[best_predictive_idx]),
        "mpu": float(incremental_profit[best_incremental_idx]),
        "mpu_optimal_target_share": float(alpha[best_incremental_idx]),
    }

    return RankingEvaluation(
        summary=summary,
        grid_curve=grid_curve,
        qini_curve=qini_curve,
    )


def apply_profit_curve(
    grid_curve: pd.DataFrame,
    avg_basket_value: float,
    sms_cost: float,
) -> pd.DataFrame:
    updated = grid_curve.copy()
    updated["predictive_profit"] = (
        updated["target_share"] * (updated["response_rate"] * avg_basket_value - sms_cost)
    )
    updated["incremental_profit"] = (
        updated["target_share"] * (updated["uplift"] * avg_basket_value - sms_cost)
    )
    return updated


def summarize_profit_curve(grid_curve: pd.DataFrame) -> dict[str, float]:
    predictive_idx = int(grid_curve["predictive_profit"].to_numpy().argmax())
    incremental_idx = int(grid_curve["incremental_profit"].to_numpy().argmax())
    return {
        "predictive_mp": float(grid_curve.iloc[predictive_idx]["predictive_profit"]),
        "predictive_optimal_target_share": float(grid_curve.iloc[predictive_idx]["target_share"]),
        "mpu": float(grid_curve.iloc[incremental_idx]["incremental_profit"]),
        "mpu_optimal_target_share": float(grid_curve.iloc[incremental_idx]["target_share"]),
    }
