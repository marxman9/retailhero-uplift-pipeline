from __future__ import annotations

from pathlib import Path

import pandas as pd

from retailhero_features import build_purchase_features

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CAMPAIGN_DATE = "2019-03-19"


def status(ok: bool) -> str:
    return "PASS" if ok else "CHECK"


def main() -> None:
    clients = pd.read_csv(
        DATA_DIR / "clients.csv",
        parse_dates=["first_issue_date", "first_redeem_date"],
    )
    products = pd.read_csv(DATA_DIR / "products.csv")
    uplift_train = pd.read_csv(DATA_DIR / "uplift_train.csv")
    uplift_test = pd.read_csv(DATA_DIR / "uplift_test.csv")

    purchase_features = build_purchase_features(DATA_DIR / "purchases.csv", CAMPAIGN_DATE)
    master_train = uplift_train.merge(clients, on="client_id", how="left").merge(
        purchase_features,
        on="client_id",
        how="left",
    )
    master_test = uplift_test.merge(clients, on="client_id", how="left").merge(
        purchase_features,
        on="client_id",
        how="left",
    )

    treatment_counts = uplift_train["treatment_flg"].value_counts().sort_index()
    treatment_ratio = treatment_counts.loc[1] / treatment_counts.loc[0]

    baseline_rates = uplift_train.groupby("treatment_flg")["target"].mean().sort_index()
    baseline_uplift_pp = (baseline_rates.loc[1] - baseline_rates.loc[0]) * 100

    rfm_cols = ["total_spend", "n_trips", "avg_basket_spend", "recency_days"]
    rfm_means = master_train.groupby("treatment_flg")[rfm_cols].mean()
    rfm_relative_diff_pct = ((rfm_means.loc[1] - rfm_means.loc[0]) / rfm_means.loc[0]) * 100

    segments = uplift_train.copy()
    segments["segment"] = "Unknown"
    segments.loc[
        (segments["treatment_flg"] == 1) & (segments["target"] == 1),
        "segment",
    ] = "Persuadable"
    segments.loc[
        (segments["treatment_flg"] == 1) & (segments["target"] == 0),
        "segment",
    ] = "Lost Cause"
    segments.loc[
        (segments["treatment_flg"] == 0) & (segments["target"] == 1),
        "segment",
    ] = "Sure Thing"
    segments.loc[
        (segments["treatment_flg"] == 0) & (segments["target"] == 0),
        "segment",
    ] = "Do-Not-Disturb"
    segment_counts = segments["segment"].value_counts()
    sure_thing_pct = segment_counts["Sure Thing"] / len(segments) * 100

    master_train["freq_decile"] = pd.qcut(
        master_train["n_trips"].fillna(0),
        q=10,
        labels=False,
        duplicates="drop",
    )
    decile_rates = master_train.groupby(["freq_decile", "treatment_flg"])["target"].mean().unstack()
    decile_rates.columns = ["control_rate", "treatment_rate"]
    decile_rates["uplift_pp"] = (decile_rates["treatment_rate"] - decile_rates["control_rate"]) * 100

    low_freq_uplift_pp = decile_rates.loc[[0, 1, 2, 3], "uplift_pp"].mean()
    high_freq_uplift_pp = decile_rates.loc[[6, 7, 8, 9], "uplift_pp"].mean()

    age_min = int(clients["age"].min())
    age_max = int(clients["age"].max())
    age_anomaly_count = int(((clients["age"] < 0) | (clients["age"] > 100)).sum())

    unique_level_1 = int(products["level_1"].nunique(dropna=True))
    test_clients_with_history = int(master_test["total_spend"].notna().sum())

    print("Phase I verification")
    print("====================")
    print()

    print(
        f"[{status(abs(treatment_ratio - 1.0) <= 0.01)}] "
        f"Experiment randomization ratio: {treatment_ratio:.6f} "
        f"(control={treatment_counts.loc[0]:,}, treatment={treatment_counts.loc[1]:,})"
    )
    print(
        f"      RFM relative mean deltas vs. control (%): "
        f"total_spend={rfm_relative_diff_pct['total_spend']:.3f}, "
        f"n_trips={rfm_relative_diff_pct['n_trips']:.3f}, "
        f"avg_basket_spend={rfm_relative_diff_pct['avg_basket_spend']:.3f}, "
        f"recency_days={rfm_relative_diff_pct['recency_days']:.3f}"
    )

    print(
        f"[{status(abs(baseline_uplift_pp - 3.32) <= 0.05)}] "
        f"Baseline uplift: {baseline_uplift_pp:.3f} pp"
    )

    print(
        f"[{status(low_freq_uplift_pp > high_freq_uplift_pp)}] "
        f"Low-frequency uplift dominates: low deciles 0-3 = {low_freq_uplift_pp:.3f} pp, "
        f"high deciles 6-9 = {high_freq_uplift_pp:.3f} pp"
    )
    print(
        f"      Decile uplift range: min={decile_rates['uplift_pp'].min():.3f} pp "
        f"(decile {int(decile_rates['uplift_pp'].idxmin())}), "
        f"max={decile_rates['uplift_pp'].max():.3f} pp "
        f"(decile {int(decile_rates['uplift_pp'].idxmax())})"
    )

    print(
        f"[{status(abs(sure_thing_pct - 30.2) <= 0.2)}] "
        f"Sure Things share: {sure_thing_pct:.3f}%"
    )

    print(
        f"[{status(age_min == -7491)}] "
        f"Age anomalies: min={age_min:,}, max={age_max:,}, flagged={age_anomaly_count:,}"
    )

    print(
        f"[{status(unique_level_1 == 3)}] "
        f"Unique level_1 categories: {unique_level_1}"
    )

    print(
        f"[{status(test_clients_with_history == len(master_test))}] "
        f"Test-set clients with purchase history: {test_clients_with_history:,} / {len(master_test):,}"
    )


if __name__ == "__main__":
    main()
