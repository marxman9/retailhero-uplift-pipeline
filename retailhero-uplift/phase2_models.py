from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phase2_metrics import evaluate_ranking


@dataclass
class Phase2ModelResult:
    model_name: str
    score_kind: str
    best_params: dict[str, Any]
    tuning_metric: float
    validation_scores: np.ndarray
    test_scores: np.ndarray


def make_stratification_labels(df: pd.DataFrame) -> pd.Series:
    # Preserve the joint treatment/response mix in every split.
    return df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)


def split_train_validation(
    train_df: pd.DataFrame,
    seed: int,
    validation_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, valid_idx = train_test_split(
        train_df.index,
        test_size=validation_size,
        random_state=seed,
        stratify=make_stratification_labels(train_df),
    )
    return (
        train_df.loc[train_idx].reset_index(drop=True),
        train_df.loc[valid_idx].reset_index(drop=True),
    )


def _inner_split(train_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_train_idx, inner_valid_idx = train_test_split(
        train_df.index,
        test_size=0.20,
        random_state=seed + 17,
        stratify=make_stratification_labels(train_df),
    )
    return (
        train_df.loc[inner_train_idx].reset_index(drop=True),
        train_df.loc[inner_valid_idx].reset_index(drop=True),
    )


def _numpy_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_columns].to_numpy(dtype=float)
    treatment = df["treatment_flg"].to_numpy(dtype=int)
    y = df["target"].to_numpy(dtype=int)
    return X, treatment, y


class TwoModelLogisticUplift:
    def __init__(self, C: float, random_state: int, max_iter: int = 1000) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.treatment_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.control_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "TwoModelLogisticUplift":
        # Fit the treated and control response surfaces independently, then
        # compare their predicted probabilities at scoring time.
        self.treatment_model.fit(X[treatment == 1], y[treatment == 1])
        self.control_model.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        treated_scores = self.treatment_model.predict_proba(X)[:, 1]
        control_scores = self.control_model.predict_proba(X)[:, 1]
        return treated_scores - control_scores


class LoInteractionLogisticUplift:
    def __init__(self, C: float, random_state: int, max_iter: int = 1000) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _design_matrix(X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        # Lo-style uplift uses one shared linear model with explicit treatment
        # interactions rather than two separate conditional models.
        treatment = treatment.reshape(-1, 1).astype(float)
        return np.hstack([X, treatment, X * treatment])

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "LoInteractionLogisticUplift":
        self.model.fit(self._design_matrix(X, treatment), y)
        return self

    def predict_counterfactual(self, X: np.ndarray, treatment_value: int) -> np.ndarray:
        treatment = np.full(X.shape[0], treatment_value, dtype=int)
        return self.model.predict_proba(self._design_matrix(X, treatment))[:, 1]

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.predict_counterfactual(X, 1) - self.predict_counterfactual(X, 0)


class ControlOnlyXGBModel:
    def __init__(self, random_state: int, **params: Any) -> None:
        self.random_state = random_state
        self.params = params
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **self.params,
        )

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "ControlOnlyXGBModel":
        # Estimate the organic purchase propensity from untreated behavior only.
        control_mask = treatment == 0
        self.model.fit(X[control_mask], y[control_mask])
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CausalMLUpliftRandomForest:
    def __init__(self, random_state: int, **params: Any) -> None:
        self.random_state = random_state
        self.params = params
        self.model = UpliftRandomForestClassifier(
            control_name="control",
            random_state=self.random_state,
            n_jobs=-1,
            evaluationFunction="KL",
            **self.params,
        )

    @staticmethod
    def _encode_treatment(treatment: np.ndarray) -> np.ndarray:
        # causalml expects string labels for the control/treatment arms.
        return np.where(treatment == 1, "treatment", "control")

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "CausalMLUpliftRandomForest":
        self.model.fit(
            X.astype(np.float32),
            self._encode_treatment(treatment),
            y.astype(int),
        )
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.astype(np.float32)).reshape(-1)


def _score_mpu(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    avg_basket_value: float,
    sms_cost: float,
) -> float:
    return evaluate_ranking(
        y_true,
        treatment,
        scores,
        avg_basket_value=avg_basket_value,
        sms_cost=sms_cost,
    ).summary["mpu"]


def _score_mp(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    avg_basket_value: float,
    sms_cost: float,
) -> float:
    return evaluate_ranking(
        y_true,
        treatment,
        scores,
        avg_basket_value=avg_basket_value,
        sms_cost=sms_cost,
    ).summary["predictive_mp"]


def train_ccp_xgboost(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    avg_basket_value: float,
    sms_cost: float,
) -> Phase2ModelResult:
    inner_train, inner_valid = _inner_split(outer_train, random_state)
    X_inner_train, t_inner_train, y_inner_train = _numpy_xy(inner_train, feature_columns)
    X_inner_valid, t_inner_valid, y_inner_valid = _numpy_xy(inner_valid, feature_columns)
    X_outer_train, t_outer_train, y_outer_train = _numpy_xy(outer_train, feature_columns)
    X_outer_valid, _, _ = _numpy_xy(outer_valid, feature_columns)
    X_test = test_df[feature_columns].to_numpy(dtype=float)

    # Keep the search space deliberately small so the comparison remains
    # reproducible and tractable on the full dataset.
    param_grid = [
        {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 450,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        },
    ]

    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for params in param_grid:
        model = ControlOnlyXGBModel(random_state=random_state, **params)
        model.fit(X_inner_train, t_inner_train, y_inner_train)
        scores = model.predict_score(X_inner_valid)
        metric = _score_mp(y_inner_valid, t_inner_valid, scores, avg_basket_value, sms_cost)
        if metric > best_score:
            best_score = metric
            best_params = params

    assert best_params is not None
    final_model = ControlOnlyXGBModel(random_state=random_state, **best_params)
    final_model.fit(X_outer_train, t_outer_train, y_outer_train)

    return Phase2ModelResult(
        model_name="Model A - CCP Baseline",
        score_kind="propensity",
        best_params=best_params,
        tuning_metric=best_score,
        validation_scores=final_model.predict_score(X_outer_valid),
        test_scores=final_model.predict_score(X_test),
    )


def train_two_model_logistic(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    avg_basket_value: float,
    sms_cost: float,
) -> Phase2ModelResult:
    inner_train, inner_valid = _inner_split(outer_train, random_state)
    X_inner_train, t_inner_train, y_inner_train = _numpy_xy(inner_train, feature_columns)
    X_inner_valid, t_inner_valid, y_inner_valid = _numpy_xy(inner_valid, feature_columns)
    X_outer_train, t_outer_train, y_outer_train = _numpy_xy(outer_train, feature_columns)
    X_outer_valid, _, _ = _numpy_xy(outer_valid, feature_columns)
    X_test = test_df[feature_columns].to_numpy(dtype=float)

    # Regularization strength is the primary lever for this simple baseline.
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for C_value in [0.1, 1.0, 10.0]:
        params = {"C": C_value}
        model = TwoModelLogisticUplift(C=C_value, random_state=random_state)
        model.fit(X_inner_train, t_inner_train, y_inner_train)
        scores = model.predict_uplift(X_inner_valid)
        metric = _score_mpu(y_inner_valid, t_inner_valid, scores, avg_basket_value, sms_cost)
        if metric > best_score:
            best_score = metric
            best_params = params

    assert best_params is not None
    final_model = TwoModelLogisticUplift(C=best_params["C"], random_state=random_state)
    final_model.fit(X_outer_train, t_outer_train, y_outer_train)

    return Phase2ModelResult(
        model_name="Model B1 - CCU Two-Model Logistic",
        score_kind="uplift",
        best_params=best_params,
        tuning_metric=best_score,
        validation_scores=final_model.predict_uplift(X_outer_valid),
        test_scores=final_model.predict_uplift(X_test),
    )


def train_lo_logistic(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    avg_basket_value: float,
    sms_cost: float,
) -> Phase2ModelResult:
    inner_train, inner_valid = _inner_split(outer_train, random_state)
    X_inner_train, t_inner_train, y_inner_train = _numpy_xy(inner_train, feature_columns)
    X_inner_valid, t_inner_valid, y_inner_valid = _numpy_xy(inner_valid, feature_columns)
    X_outer_train, t_outer_train, y_outer_train = _numpy_xy(outer_train, feature_columns)
    X_outer_valid, _, _ = _numpy_xy(outer_valid, feature_columns)
    X_test = test_df[feature_columns].to_numpy(dtype=float)

    # Reuse the same tuning budget as the two-model logit for a fair comparison.
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for C_value in [0.1, 1.0, 10.0]:
        params = {"C": C_value}
        model = LoInteractionLogisticUplift(C=C_value, random_state=random_state)
        model.fit(X_inner_train, t_inner_train, y_inner_train)
        scores = model.predict_uplift(X_inner_valid)
        metric = _score_mpu(y_inner_valid, t_inner_valid, scores, avg_basket_value, sms_cost)
        if metric > best_score:
            best_score = metric
            best_params = params

    assert best_params is not None
    final_model = LoInteractionLogisticUplift(C=best_params["C"], random_state=random_state)
    final_model.fit(X_outer_train, t_outer_train, y_outer_train)

    return Phase2ModelResult(
        model_name="Model B2 - CCU Lo Interaction Logistic",
        score_kind="uplift",
        best_params=best_params,
        tuning_metric=best_score,
        validation_scores=final_model.predict_uplift(X_outer_valid),
        test_scores=final_model.predict_uplift(X_test),
    )


def train_uplift_random_forest(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    avg_basket_value: float,
    sms_cost: float,
) -> Phase2ModelResult:
    inner_train, inner_valid = _inner_split(outer_train, random_state)
    X_inner_train, t_inner_train, y_inner_train = _numpy_xy(inner_train, feature_columns)
    X_inner_valid, t_inner_valid, y_inner_valid = _numpy_xy(inner_valid, feature_columns)
    X_outer_train, t_outer_train, y_outer_train = _numpy_xy(outer_train, feature_columns)
    X_outer_valid, _, _ = _numpy_xy(outer_valid, feature_columns)
    X_test = test_df[feature_columns].to_numpy(dtype=float)

    # The forest receives a light grid because training cost is materially higher
    # than the linear baselines on the full feature matrix.
    param_grid = [
        {
            "n_estimators": 25,
            "max_features": min(12, len(feature_columns)),
            "max_depth": 5,
            "min_samples_leaf": 100,
            "min_samples_treatment": 50,
            "n_reg": 10,
        },
        {
            "n_estimators": 40,
            "max_features": min(14, len(feature_columns)),
            "max_depth": 6,
            "min_samples_leaf": 75,
            "min_samples_treatment": 40,
            "n_reg": 25,
        },
    ]

    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for params in param_grid:
        model = CausalMLUpliftRandomForest(random_state=random_state, **params)
        model.fit(X_inner_train, t_inner_train, y_inner_train)
        scores = model.predict_uplift(X_inner_valid)
        metric = _score_mpu(y_inner_valid, t_inner_valid, scores, avg_basket_value, sms_cost)
        if metric > best_score:
            best_score = metric
            best_params = params

    assert best_params is not None
    final_model = CausalMLUpliftRandomForest(random_state=random_state, **best_params)
    final_model.fit(X_outer_train, t_outer_train, y_outer_train)

    return Phase2ModelResult(
        model_name="Model B3 - CCU Uplift Random Forest",
        score_kind="uplift",
        best_params=best_params,
        tuning_metric=best_score,
        validation_scores=final_model.predict_uplift(X_outer_valid),
        test_scores=final_model.predict_uplift(X_test),
    )
