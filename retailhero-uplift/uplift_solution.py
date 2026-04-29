from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test):
    """Fit separate treatment and control models and return their probability gap."""

    X_treatment = X_train[treatment_train == 1, :]
    y_treatment = target_train[treatment_train == 1]
    X_control = X_train[treatment_train == 0, :]
    y_control = target_train[treatment_train == 0]

    model_treatment = clone(model).fit(X_treatment, y_treatment)
    model_control = clone(model).fit(X_control, y_control)

    predict_treatment = model_treatment.predict_proba(X_test)[:, 1]
    predict_control = model_control.predict_proba(X_test)[:, 1]
    return predict_treatment - predict_control


def uplift_score(prediction, treatment, target, rate=0.3):
    """Evaluate uplift on the top-ranked fraction of treated and control users."""

    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    control_n = int((treatment == 0).sum() * rate)

    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    return treatment_p - control_p


def _datetime_to_unix(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    unix_seconds = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = timestamps.notna()
    unix_seconds.loc[valid_mask] = timestamps.loc[valid_mask].astype("int64") / 10**9
    return unix_seconds


def build_client_features(df_clients: pd.DataFrame) -> pd.DataFrame:
    first_issue_unixtime = _datetime_to_unix(df_clients["first_issue_date"])
    first_redeem_unixtime = _datetime_to_unix(df_clients["first_redeem_date"])

    return pd.DataFrame(
        {
            "gender_M": (df_clients["gender"] == "M").astype(int),
            "gender_F": (df_clients["gender"] == "F").astype(int),
            "gender_U": (df_clients["gender"] == "U").astype(int),
            "age": df_clients["age"],
            "first_issue_time": first_issue_unixtime,
            "first_redeem_time": first_redeem_unixtime,
            "issue_redeem_delay": first_redeem_unixtime - first_issue_unixtime,
        },
        index=df_clients.index,
    ).fillna(0)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_clients = pd.read_csv(DATA_DIR / "clients.csv", index_col="client_id")
    df_train = pd.read_csv(DATA_DIR / "uplift_train.csv", index_col="client_id")
    df_test = pd.read_csv(DATA_DIR / "uplift_test.csv", index_col="client_id")

    df_features = build_client_features(df_clients)

    indices_train = df_train.index
    indices_test = df_test.index
    indices_learn, indices_valid = train_test_split(
        df_train.index,
        test_size=0.3,
        random_state=123,
    )

    model = GradientBoostingClassifier(random_state=123)

    valid_uplift = uplift_fit_predict(
        model=model,
        X_train=df_features.loc[indices_learn, :].values,
        treatment_train=df_train.loc[indices_learn, "treatment_flg"].values,
        target_train=df_train.loc[indices_learn, "target"].values,
        X_test=df_features.loc[indices_valid, :].values,
    )
    valid_score = uplift_score(
        valid_uplift,
        treatment=df_train.loc[indices_valid, "treatment_flg"].values,
        target=df_train.loc[indices_valid, "target"].values,
    )
    print(f"Validation score: {valid_score:.6f}")

    test_uplift = uplift_fit_predict(
        model=model,
        X_train=df_features.loc[indices_train, :].values,
        treatment_train=df_train.loc[indices_train, "treatment_flg"].values,
        target_train=df_train.loc[indices_train, "target"].values,
        X_test=df_features.loc[indices_test, :].values,
    )

    submission_path = OUTPUT_DIR / "submission.csv"
    df_submission = pd.DataFrame({"uplift": test_uplift}, index=df_test.index)
    df_submission.to_csv(submission_path)
    print(f"Saved: {submission_path}")


if __name__ == "__main__":
    main()
