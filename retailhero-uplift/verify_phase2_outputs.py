from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "phase2"


def status(ok: bool) -> str:
    return "PASS" if ok else "CHECK"


def main() -> None:
    required_files = [
        OUTPUT_ROOT / "phase2_scored_validation.csv",
        OUTPUT_ROOT / "phase2_scored_test.csv",
        OUTPUT_ROOT / "phase2_summary_metrics.csv",
        OUTPUT_ROOT / "phase2_qini_curves.csv",
        OUTPUT_ROOT / "phase2_liftup_curves.csv",
        OUTPUT_ROOT / "phase2_model_params.json",
        OUTPUT_ROOT / "qini_curves.png",
        OUTPUT_ROOT / "liftup_curves.png",
        OUTPUT_ROOT / "profit_vs_target_depth.png",
        OUTPUT_ROOT / "profit_sensitivity.png",
        OUTPUT_ROOT / "phase2_profit_sensitivity.csv",
        OUTPUT_ROOT / "phase2_feature_cache" / "phase2_train_features.csv",
        OUTPUT_ROOT / "phase2_feature_cache" / "phase2_test_features.csv",
        OUTPUT_ROOT / "phase2_feature_cache" / "phase2_feature_metadata.json",
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing expected Phase II outputs: {missing_files}")

    validation = pd.read_csv(OUTPUT_ROOT / "phase2_scored_validation.csv")
    test = pd.read_csv(OUTPUT_ROOT / "phase2_scored_test.csv")
    summary = pd.read_csv(OUTPUT_ROOT / "phase2_summary_metrics.csv")
    qini = pd.read_csv(OUTPUT_ROOT / "phase2_qini_curves.csv")
    curves = pd.read_csv(OUTPUT_ROOT / "phase2_liftup_curves.csv")
    sensitivity = pd.read_csv(OUTPUT_ROOT / "phase2_profit_sensitivity.csv")
    train_features = pd.read_csv(OUTPUT_ROOT / "phase2_feature_cache" / "phase2_train_features.csv")
    test_features = pd.read_csv(OUTPUT_ROOT / "phase2_feature_cache" / "phase2_test_features.csv")
    metadata = json.loads(
        (OUTPUT_ROOT / "phase2_feature_cache" / "phase2_feature_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    feature_columns = metadata["feature_columns"]
    level1_categories = metadata["level1_categories"]
    score_columns = [col for col in validation.columns if col.endswith("_score")]

    share_columns = [f"level1_share_{category}" for category in level1_categories] + [
        "unknown_level1_spend_share"
    ]
    share_error = (train_features[share_columns].sum(axis=1) - 1.0).abs().max()

    print("Phase II verification")
    print("=====================")
    print()
    print(f"[{status(len(train_features) == 200039)}] Train feature rows: {len(train_features):,}")
    print(f"[{status(len(test_features) == 200123)}] Test feature rows:  {len(test_features):,}")
    print(
        f"[{status(train_features[feature_columns].isna().sum().sum() == 0)}] "
        f"Train feature missing values: {int(train_features[feature_columns].isna().sum().sum()):,}"
    )
    print(
        f"[{status(test_features[feature_columns].isna().sum().sum() == 0)}] "
        f"Test feature missing values:  {int(test_features[feature_columns].isna().sum().sum()):,}"
    )
    print(
        f"[{status(share_error < 1e-8)}] "
        f"Category-share max abs error: {share_error:.3e}"
    )
    print(
        f"[{status(validation['treatment_flg'].value_counts().min() > 0)}] "
        f"Validation treatment/control coverage: {validation['treatment_flg'].value_counts().to_dict()}"
    )
    print(
        f"[{status(np.isfinite(validation[score_columns].to_numpy()).all())}] "
        f"Finite validation scores across {len(score_columns)} models"
    )
    print(
        f"[{status(np.isfinite(test[[col for col in test.columns if col.endswith('_score')]].to_numpy()).all())}] "
        f"Finite test scores across {len(score_columns)} models"
    )
    print(
        f"[{status(summary['qini_auc'].notna().all())}] "
        f"Summary metrics rows: {len(summary):,}"
    )
    print(
        f"[{status(qini['target_share'].between(0, 1).all())}] "
        f"Qini target-share bounds valid"
    )
    print(
        f"[{status(curves['target_share'].between(0, 1).all())}] "
        f"Liftup/profit curve target-share bounds valid"
    )
    print(
        f"[{status(sensitivity['mpu_optimal_target_share'].between(0, 1).all())}] "
        f"Sensitivity optimal target-share bounds valid"
    )


if __name__ == "__main__":
    main()
