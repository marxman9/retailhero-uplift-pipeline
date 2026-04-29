from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2_features import (
    CAMPAIGN_DATE,
    build_phase2_feature_bundle,
    validate_feature_bundle,
)
from phase2_metrics import apply_profit_curve, evaluate_ranking, summarize_profit_curve
from phase2_models import (
    split_train_validation,
    train_ccp_xgboost,
    train_lo_logistic,
    train_two_model_logistic,
    train_uplift_random_forest,
)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {key: _to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_plot_qini(qini_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for model_name, model_df in qini_df.groupby("model_name"):
        plt.plot(model_df["target_share"], model_df["qini_gain"], label=model_name)
    baseline_df = qini_df[qini_df["model_name"] == qini_df["model_name"].iloc[0]]
    plt.plot(
        baseline_df["target_share"],
        baseline_df["qini_baseline"],
        linestyle="--",
        color="gray",
        label="Random baseline",
    )
    plt.xlabel("Target share")
    plt.ylabel("Incremental purchasers per customer")
    plt.title("Qini Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_plot_liftup(curve_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for model_name, model_df in curve_df.groupby("model_name"):
        plt.plot(model_df["target_share"], model_df["liftup"], label=model_name)
    plt.axhline(1.0, linestyle="--", color="gray", label="Baseline uplift")
    plt.xlabel("Target share")
    plt.ylabel("Liftup")
    plt.title("Liftup Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_plot_profit(curve_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, model_df in curve_df.groupby("model_name"):
        axes[0].plot(model_df["target_share"], model_df["incremental_profit"], label=model_name)
    axes[0].set_title("Incremental Profit vs Target Share")
    axes[0].set_xlabel("Target share")
    axes[0].set_ylabel("Profit per customer in base")
    axes[0].axhline(0.0, linestyle="--", color="gray")
    axes[0].legend()

    ccp_df = curve_df[curve_df["model_name"] == "Model A - CCP Baseline"]
    axes[1].plot(ccp_df["target_share"], ccp_df["predictive_profit"], color="tab:red")
    axes[1].set_title("Predictive MP Curve (CCP)")
    axes[1].set_xlabel("Target share")
    axes[1].set_ylabel("Profit per customer in base")
    axes[1].axhline(0.0, linestyle="--", color="gray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    default_basket_value: float,
    default_sms_cost: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sms_slice = sensitivity_df[np.isclose(sensitivity_df["avg_basket_value"], default_basket_value)]
    for model_name, model_df in sms_slice.groupby("model_name"):
        axes[0].plot(model_df["sms_cost"], model_df["mpu"], marker="o", label=model_name)
    axes[0].set_title("Max Incremental Profit vs SMS Cost")
    axes[0].set_xlabel("SMS cost")
    axes[0].set_ylabel("MPU")
    axes[0].legend()

    basket_slice = sensitivity_df[np.isclose(sensitivity_df["sms_cost"], default_sms_cost)]
    for model_name, model_df in basket_slice.groupby("model_name"):
        axes[1].plot(model_df["avg_basket_value"], model_df["mpu"], marker="o", label=model_name)
    axes[1].set_title("Max Incremental Profit vs Basket Value")
    axes[1].set_xlabel("Average basket value")
    axes[1].set_ylabel("MPU")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_sensitivity_table(
    curve_df: pd.DataFrame,
    default_basket_value: float,
    default_sms_cost: float,
) -> pd.DataFrame:
    basket_values = default_basket_value * np.array([0.50, 0.75, 1.00, 1.25, 1.50])
    sms_costs = np.array([0.50, 1.00, 2.00, 3.00, 5.00]) * default_sms_cost

    rows: list[dict[str, float | str]] = []
    for (model_name, model_df), basket_value, sms_cost in (
        (item, basket_value, sms_cost)
        for item in curve_df.groupby("model_name")
        for basket_value in basket_values
        for sms_cost in sms_costs
    ):
        profit_df = apply_profit_curve(model_df, basket_value, sms_cost)
        summary = summarize_profit_curve(profit_df)
        rows.append(
            {
                "model_name": model_name,
                "avg_basket_value": float(basket_value),
                "sms_cost": float(sms_cost),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase II CCP vs CCU pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "phase2")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--sms-cost", type=float, default=1.0)
    parser.add_argument("--avg-basket-value", type=float, default=None)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--skip-sensitivity-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_phase2_feature_bundle(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        campaign_date=CAMPAIGN_DATE,
        reuse_cache=args.reuse_features,
    )
    feature_checks = validate_feature_bundle(bundle)

    avg_basket_value = (
        args.avg_basket_value
        if args.avg_basket_value is not None
        else bundle.metadata["avg_basket_value_empirical"]
    )

    outer_train, outer_valid = split_train_validation(
        bundle.train,
        seed=args.random_seed,
        validation_size=0.20,
    )

    model_results = [
        train_ccp_xgboost(
            outer_train,
            outer_valid,
            bundle.test,
            bundle.feature_columns,
            args.random_seed,
            avg_basket_value,
            args.sms_cost,
        ),
        train_two_model_logistic(
            outer_train,
            outer_valid,
            bundle.test,
            bundle.feature_columns,
            args.random_seed,
            avg_basket_value,
            args.sms_cost,
        ),
        train_lo_logistic(
            outer_train,
            outer_valid,
            bundle.test,
            bundle.feature_columns,
            args.random_seed,
            avg_basket_value,
            args.sms_cost,
        ),
        train_uplift_random_forest(
            outer_train,
            outer_valid,
            bundle.test,
            bundle.feature_columns,
            args.random_seed,
            avg_basket_value,
            args.sms_cost,
        ),
    ]

    scored_validation = outer_valid[["client_id", "treatment_flg", "target"]].copy()
    scored_test = bundle.test[["client_id"]].copy()

    metric_rows: list[dict[str, object]] = []
    qini_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    params_payload = {
        "run_config": {
            "data_dir": str(args.data_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "random_seed": args.random_seed,
            "sms_cost": args.sms_cost,
            "avg_basket_value": avg_basket_value,
            "reuse_features": args.reuse_features,
            "skip_sensitivity_sweep": args.skip_sensitivity_sweep,
        },
        "feature_checks": feature_checks,
        "models": {},
    }

    for result in model_results:
        score_column = result.model_name.lower().replace(" ", "_").replace("-", "").replace("__", "_") + "_score"
        rank_column = score_column.replace("_score", "_rank")

        scored_validation[score_column] = result.validation_scores
        scored_validation[rank_column] = (
            scored_validation[score_column].rank(ascending=False, method="first").astype(int)
        )

        scored_test[score_column] = result.test_scores
        scored_test[rank_column] = scored_test[score_column].rank(ascending=False, method="first").astype(int)

        evaluation = evaluate_ranking(
            y_true=outer_valid["target"].to_numpy(dtype=int),
            treatment=outer_valid["treatment_flg"].to_numpy(dtype=int),
            scores=result.validation_scores,
            avg_basket_value=avg_basket_value,
            sms_cost=args.sms_cost,
        )

        summary_values = dict(evaluation.summary)
        if result.score_kind != "propensity":
            summary_values["predictive_mp"] = np.nan
            summary_values["predictive_optimal_target_share"] = np.nan

        metric_rows.append(
            {
                "model_name": result.model_name,
                "score_kind": result.score_kind,
                "tuning_metric": result.tuning_metric,
                **summary_values,
            }
        )

        qini_frames.append(evaluation.qini_curve.assign(model_name=result.model_name))
        curve_frames.append(evaluation.grid_curve.assign(model_name=result.model_name))

        params_payload["models"][result.model_name] = {
            "score_kind": result.score_kind,
            "best_params": _to_builtin(result.best_params),
            "tuning_metric": float(result.tuning_metric),
        }

    summary_metrics = pd.DataFrame(metric_rows).sort_values("mpu", ascending=False).reset_index(drop=True)
    qini_curves = pd.concat(qini_frames, ignore_index=True)
    model_curves = pd.concat(curve_frames, ignore_index=True)

    sensitivity_df = pd.DataFrame()
    if not args.skip_sensitivity_sweep:
        sensitivity_df = _build_sensitivity_table(
            model_curves,
            default_basket_value=avg_basket_value,
            default_sms_cost=args.sms_cost,
        )

    scored_validation_path = args.output_dir / "phase2_scored_validation.csv"
    scored_test_path = args.output_dir / "phase2_scored_test.csv"
    summary_metrics_path = args.output_dir / "phase2_summary_metrics.csv"
    qini_curve_path = args.output_dir / "phase2_qini_curves.csv"
    liftup_curve_path = args.output_dir / "phase2_liftup_curves.csv"
    sensitivity_path = args.output_dir / "phase2_profit_sensitivity.csv"
    params_path = args.output_dir / "phase2_model_params.json"

    scored_validation.to_csv(scored_validation_path, index=False)
    scored_test.to_csv(scored_test_path, index=False)
    summary_metrics.to_csv(summary_metrics_path, index=False)
    qini_curves.to_csv(qini_curve_path, index=False)
    model_curves.to_csv(liftup_curve_path, index=False)
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(sensitivity_path, index=False)
    params_path.write_text(json.dumps(_to_builtin(params_payload), indent=2), encoding="utf-8")

    _save_plot_qini(qini_curves, args.output_dir / "qini_curves.png")
    _save_plot_liftup(model_curves, args.output_dir / "liftup_curves.png")
    _save_plot_profit(model_curves, args.output_dir / "profit_vs_target_depth.png")
    if not sensitivity_df.empty:
        _save_plot_sensitivity(
            sensitivity_df,
            default_basket_value=avg_basket_value,
            default_sms_cost=args.sms_cost,
            output_path=args.output_dir / "profit_sensitivity.png",
        )

    print("Phase II pipeline completed")
    print("===========================")
    print(f"Validation rows: {len(scored_validation):,}")
    print(f"Test rows:       {len(scored_test):,}")
    print(f"Feature count:   {len(bundle.feature_columns):,}")
    print(f"Avg basket:      {avg_basket_value:,.3f}")
    print(f"SMS cost:        {args.sms_cost:,.3f}")
    print()
    print(summary_metrics[["model_name", "qini_auc", "mpu", "predictive_mp"]].to_string(index=False))


if __name__ == "__main__":
    main()
