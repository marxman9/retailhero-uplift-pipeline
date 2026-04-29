from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retailhero_features import (
    PRODUCT_COLUMNS_SQL,
    PURCHASE_COLUMNS_SQL,
    _connect,
    _normalize_path,
    build_purchase_features,
)

CAMPAIGN_DATE = pd.Timestamp("2019-03-19")
FEATURE_CACHE_DIRNAME = "phase2_feature_cache"


@dataclass
class Phase2FeatureBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    metadata: dict[str, Any]


def _cache_dir(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / FEATURE_CACHE_DIRNAME


def _cache_paths(output_dir: str | Path) -> dict[str, Path]:
    cache_dir = _cache_dir(output_dir)
    return {
        "dir": cache_dir,
        "train": cache_dir / "phase2_train_features.csv",
        "test": cache_dir / "phase2_test_features.csv",
        "meta": cache_dir / "phase2_feature_metadata.json",
    }


def _read_clients(clients_path: str | Path, campaign_date: pd.Timestamp) -> pd.DataFrame:
    clients = pd.read_csv(
        clients_path,
        parse_dates=["first_issue_date", "first_redeem_date"],
    )

    # Preserve the original age field but add a bounded version for modeling and
    # an explicit anomaly flag so bad source values are still auditable.
    clients["age_flagged"] = ((clients["age"] < 0) | (clients["age"] > 100)).astype(int)
    clients["age_clean"] = clients["age"].clip(lower=0, upper=100)

    # Expand the anonymized gender code into a model-friendly indicator set.
    clients["gender_M"] = (clients["gender"] == "M").astype(int)
    clients["gender_F"] = (clients["gender"] == "F").astype(int)
    clients["gender_U"] = (clients["gender"] == "U").astype(int)
    clients["gender_known"] = (clients["gender"] != "U").astype(int)

    # Redemption timing is structurally missing for customers who never redeemed,
    # so we pair every filled value with a missingness indicator.
    clients["ever_redeemed"] = clients["first_redeem_date"].notna().astype(int)
    clients["days_to_redeem"] = (
        clients["first_redeem_date"] - clients["first_issue_date"]
    ).dt.days
    clients["days_to_redeem_missing"] = clients["days_to_redeem"].isna().astype(int)
    clients["days_to_redeem"] = clients["days_to_redeem"].fillna(-1)

    clients["issue_age_days"] = (campaign_date - clients["first_issue_date"]).dt.days
    clients["redeem_recency_days"] = (
        campaign_date - clients["first_redeem_date"]
    ).dt.days
    clients["redeem_recency_missing"] = clients["redeem_recency_days"].isna().astype(int)
    clients["redeem_recency_days"] = clients["redeem_recency_days"].fillna(-1)

    return clients[
        [
            "client_id",
            "age",
            "age_clean",
            "age_flagged",
            "gender_M",
            "gender_F",
            "gender_U",
            "gender_known",
            "ever_redeemed",
            "days_to_redeem",
            "days_to_redeem_missing",
            "issue_age_days",
            "redeem_recency_days",
            "redeem_recency_missing",
        ]
    ].copy()


def compute_empirical_avg_basket_value(purchases_path: str | Path) -> float:
    # Use the empirical average trip value as the default business value in the
    # profit simulations so the pipeline stays data-driven by default.
    conn = _connect()
    try:
        avg_value = conn.execute(
            f"""
            select
              sum(coalesce(purchase_sum, 0)) / count(distinct transaction_id) as avg_trip_value
            from read_csv(?, header=true, columns={PURCHASE_COLUMNS_SQL})
            """,
            [_normalize_path(purchases_path)],
        ).fetchone()[0]
    finally:
        conn.close()
    return float(avg_value)


def _load_product_metadata(products_path: str | Path) -> dict[str, Any]:
    products = pd.read_csv(products_path, usecols=["level_1", "netto"])
    level1_categories = sorted(products["level_1"].dropna().unique().tolist())
    netto_p99 = float(products["netto"].quantile(0.99))
    return {
        "level1_categories": level1_categories,
        "netto_p99": netto_p99,
    }


def _build_product_behavior_features(
    purchases_path: str | Path,
    products_path: str | Path,
    product_meta: dict[str, Any],
) -> pd.DataFrame:
    netto_p99 = product_meta["netto_p99"]
    level1_categories = product_meta["level1_categories"]

    # Build one spend accumulator per observed hashed top-level category so the
    # final client-level matrix captures category mix without assuming semantics.
    category_spend_cols = ",\n".join(
        [
            f"    sum(case when level_1 = '{category}' then purchase_sum else 0 end) "
            f"as level1_spend_{category}"
            for category in level1_categories
        ]
    )

    conn = _connect()
    try:
        # Push the wide product-behavior aggregation into DuckDB so the largest
        # table never has to be materialized in pandas at full granularity.
        query = f"""
        with products_clean as (
          select
            product_id,
            level_1,
            brand_id,
            coalesce(is_alcohol, 0) as is_alcohol,
            coalesce(is_own_trademark, 0) as is_own_trademark,
            case
              when netto is null then null
              else least(netto, {netto_p99})
            end as netto_clean
          from read_csv(?, header=true, columns={PRODUCT_COLUMNS_SQL})
        ),
        joined as (
          select
            purchases.client_id,
            coalesce(purchases.purchase_sum, 0) as purchase_sum,
            coalesce(purchases.product_quantity, 0) as product_quantity,
            products_clean.level_1,
            products_clean.brand_id,
            products_clean.is_alcohol,
            products_clean.is_own_trademark,
            products_clean.netto_clean
          from read_csv(?, header=true, columns={PURCHASE_COLUMNS_SQL}) purchases
          left join products_clean
          using (product_id)
        ),
        client_rollup as (
          select
            client_id,
            sum(purchase_sum) as product_join_total_spend,
            sum(case when is_alcohol = 1 then purchase_sum else 0 end) as alcohol_spend,
            sum(case when is_own_trademark = 1 then purchase_sum else 0 end) as own_trademark_spend,
            sum(case when level_1 is null then purchase_sum else 0 end) as unknown_level1_spend,
            count(distinct brand_id) as n_brands,
            sum(coalesce(netto_clean, 0) * product_quantity) as netto_total,
            sum(case when netto_clean is null then product_quantity else 0 end) as missing_netto_quantity,
{category_spend_cols}
          from joined
          group by client_id
        ),
        brand_spend as (
          select
            client_id,
            brand_id,
            sum(purchase_sum) as brand_spend
          from joined
          where brand_id is not null
          group by client_id, brand_id
        ),
        brand_hhi as (
          select
            brand_spend.client_id,
            sum(power(brand_spend.brand_spend / nullif(client_rollup.product_join_total_spend, 0), 2)) as brand_hhi
          from brand_spend
          inner join client_rollup
            on brand_spend.client_id = client_rollup.client_id
          group by brand_spend.client_id
        )
        select
          client_rollup.*,
          coalesce(brand_hhi.brand_hhi, 0) as brand_hhi
        from client_rollup
        left join brand_hhi
          on client_rollup.client_id = brand_hhi.client_id
        order by client_rollup.client_id
        """
        features = conn.execute(
            query,
            [
                _normalize_path(products_path),
                _normalize_path(purchases_path),
            ],
        ).df()
    finally:
        conn.close()

    return features


def _add_share_features(df: pd.DataFrame, level1_categories: list[str]) -> pd.DataFrame:
    # Shares normalize raw spend totals into comparable behavioral signals.
    total_spend = df["total_spend"].replace(0, np.nan)

    df["alcohol_spend_share"] = (df["alcohol_spend"] / total_spend).fillna(0)
    df["own_trademark_spend_share"] = (df["own_trademark_spend"] / total_spend).fillna(0)
    df["unknown_level1_spend_share"] = (df["unknown_level1_spend"] / total_spend).fillna(0)

    for category in level1_categories:
        spend_col = f"level1_spend_{category}"
        share_col = f"level1_share_{category}"
        df[share_col] = (df[spend_col] / total_spend).fillna(0)

    item_denom = df["n_items"].replace(0, np.nan)
    df["avg_netto_per_item"] = (df["netto_total"] / item_denom).fillna(0)
    df["missing_netto_share"] = (df["missing_netto_quantity"] / item_denom).fillna(0)
    return df


def _fill_missing_with_indicators(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_columns: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # All downstream models expect a fully numeric matrix. Missingness indicators
    # preserve information while deterministic sentinel filling removes NaNs.
    combined = pd.concat(
        [
            train.assign(_dataset="train"),
            test.assign(_dataset="test"),
        ],
        ignore_index=True,
        sort=False,
    )

    feature_columns = [
        column
        for column in combined.columns
        if column not in id_columns and column != "_dataset"
    ]

    for column in feature_columns:
        if combined[column].isna().any():
            indicator_col = f"{column}_missing"
            combined[indicator_col] = combined[column].isna().astype(int)

            observed = combined[column].dropna()
            if observed.empty:
                fill_value = -1.0
            elif (observed >= 0).all():
                # Use a shared negative sentinel for naturally non-negative fields.
                fill_value = -1.0
            else:
                fill_value = float(observed.min()) - 1.0

            combined[column] = combined[column].fillna(fill_value)
        combined[column] = pd.to_numeric(combined[column], errors="raise")

    train_out = combined.loc[combined["_dataset"] == "train"].drop(columns="_dataset").reset_index(drop=True)
    test_out = combined.loc[combined["_dataset"] == "test"].drop(columns="_dataset").reset_index(drop=True)

    feature_columns = [
        column
        for column in train_out.columns
        if column not in id_columns
    ]
    return train_out, test_out, feature_columns


def validate_feature_bundle(bundle: Phase2FeatureBundle) -> dict[str, Any]:
    train = bundle.train
    test = bundle.test
    feature_columns = bundle.feature_columns
    level1_categories = bundle.metadata["level1_categories"]

    category_share_cols = [f"level1_share_{category}" for category in level1_categories]
    share_sum = train[category_share_cols + ["unknown_level1_spend_share"]].sum(axis=1)

    checks = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "train_missing_values": int(train[feature_columns].isna().sum().sum()),
        "test_missing_values": int(test[feature_columns].isna().sum().sum()),
        "category_share_max_abs_error": float((share_sum - 1.0).abs().max()),
        "purchase_history_missing_train": int(train["total_spend"].isna().sum()),
        "purchase_history_missing_test": int(test["total_spend"].isna().sum()),
    }
    return checks


def build_phase2_feature_bundle(
    data_dir: str | Path,
    output_dir: str | Path,
    campaign_date: str | pd.Timestamp = CAMPAIGN_DATE,
    reuse_cache: bool = False,
) -> Phase2FeatureBundle:
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cache_paths = _cache_paths(output_dir)

    if reuse_cache and all(path.exists() for key, path in cache_paths.items() if key != "dir"):
        train = pd.read_csv(cache_paths["train"])
        test = pd.read_csv(cache_paths["test"])
        metadata = json.loads(cache_paths["meta"].read_text(encoding="utf-8"))
        return Phase2FeatureBundle(
            train=train,
            test=test,
            feature_columns=metadata["feature_columns"],
            metadata=metadata,
        )

    cache_paths["dir"].mkdir(parents=True, exist_ok=True)

    campaign_date = pd.Timestamp(campaign_date)
    clients_path = data_dir / "clients.csv"
    products_path = data_dir / "products.csv"
    purchases_path = data_dir / "purchases.csv"
    train_path = data_dir / "uplift_train.csv"
    test_path = data_dir / "uplift_test.csv"

    client_features = _read_clients(clients_path, campaign_date)
    purchase_features = build_purchase_features(purchases_path, campaign_date)
    product_meta = _load_product_metadata(products_path)
    product_features = _build_product_behavior_features(
        purchases_path,
        products_path,
        product_meta,
    )

    # Join every feature family at the client level before appending treatment
    # labels or exporting the cached train/test matrices.
    feature_core = (
        client_features.merge(purchase_features, on="client_id", how="left")
        .merge(product_features, on="client_id", how="left")
    )
    feature_core = _add_share_features(feature_core, product_meta["level1_categories"])

    train = pd.read_csv(train_path).merge(feature_core, on="client_id", how="left")
    test = pd.read_csv(test_path).merge(feature_core, on="client_id", how="left")

    id_columns = {"client_id", "treatment_flg", "target"}
    train, test, feature_columns = _fill_missing_with_indicators(train, test, id_columns)

    metadata = {
        "campaign_date": campaign_date.date().isoformat(),
        "level1_categories": product_meta["level1_categories"],
        "netto_p99": product_meta["netto_p99"],
        "feature_columns": feature_columns,
        "avg_basket_value_empirical": compute_empirical_avg_basket_value(purchases_path),
    }

    train.to_csv(cache_paths["train"], index=False)
    test.to_csv(cache_paths["test"], index=False)
    cache_paths["meta"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return Phase2FeatureBundle(
        train=train,
        test=test,
        feature_columns=feature_columns,
        metadata=metadata,
    )
