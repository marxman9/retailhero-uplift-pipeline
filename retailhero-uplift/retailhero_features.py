from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - exercised only in a broken env
    duckdb = None
    _DUCKDB_IMPORT_ERROR = exc
else:
    _DUCKDB_IMPORT_ERROR = None


PURCHASE_COLUMNS = {
    "client_id": "VARCHAR",
    "transaction_id": "VARCHAR",
    "transaction_datetime": "TIMESTAMP",
    "regular_points_received": "DOUBLE",
    "express_points_received": "DOUBLE",
    "regular_points_spent": "DOUBLE",
    "express_points_spent": "DOUBLE",
    "purchase_sum": "DOUBLE",
    "store_id": "VARCHAR",
    "product_id": "VARCHAR",
    "product_quantity": "DOUBLE",
    "trn_sum_from_iss": "DOUBLE",
    "trn_sum_from_red": "DOUBLE",
}

PRODUCT_COLUMNS = {
    "product_id": "VARCHAR",
    "level_1": "VARCHAR",
    "level_2": "VARCHAR",
    "level_3": "VARCHAR",
    "level_4": "VARCHAR",
    "segment_id": "DOUBLE",
    "brand_id": "VARCHAR",
    "vendor_id": "VARCHAR",
    "netto": "DOUBLE",
    "is_own_trademark": "DOUBLE",
    "is_alcohol": "DOUBLE",
}


def _sql_map(mapping: dict[str, str]) -> str:
    pairs = ", ".join(f"'{key}': '{value}'" for key, value in mapping.items())
    return "{" + pairs + "}"


PURCHASE_COLUMNS_SQL = _sql_map(PURCHASE_COLUMNS)
PRODUCT_COLUMNS_SQL = _sql_map(PRODUCT_COLUMNS)


def _connect() -> "duckdb.DuckDBPyConnection":
    if duckdb is None:
        raise RuntimeError(
            "duckdb is required for the local retailhero helpers. "
            "Install the packages from requirements.txt."
        ) from _DUCKDB_IMPORT_ERROR
    return duckdb.connect(database=":memory:")


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def summarize_purchase_history(purchases_path: str | Path) -> dict[str, Any]:
    conn = _connect()
    try:
        result = conn.execute(
            f"""
            select
              min(transaction_datetime)::date as min_date,
              max(transaction_datetime)::date as max_date,
              sum(coalesce(purchase_sum, 0)) as total_spend,
              sum(coalesce(product_quantity, 0)) as total_items
            from read_csv(?, header=true, columns={PURCHASE_COLUMNS_SQL})
            """,
            [_normalize_path(purchases_path)],
        ).fetchone()
    finally:
        conn.close()

    return {
        "min_date": result[0],
        "max_date": result[1],
        "total_spend": float(result[2]),
        "total_items": float(result[3]),
    }


def build_purchase_features(
    purchases_path: str | Path,
    campaign_date: str | pd.Timestamp,
) -> pd.DataFrame:
    campaign_date = pd.Timestamp(campaign_date).date().isoformat()
    conn = _connect()
    try:
        features = conn.execute(
            f"""
            with client_rollup as (
              select
                client_id,
                sum(coalesce(purchase_sum, 0)) as total_spend,
                count(*) as n_transactions,
                count(distinct transaction_id) as n_trips,
                sum(coalesce(product_quantity, 0)) as n_items,
                count(distinct store_id) as n_stores,
                count(distinct product_id) as n_products,
                sum(coalesce(regular_points_received, 0)) as points_earned,
                sum(coalesce(regular_points_spent, 0)) as points_spent,
                min(transaction_datetime)::date as first_date,
                max(transaction_datetime)::date as last_date
              from read_csv(?, header=true, columns={PURCHASE_COLUMNS_SQL})
              group by client_id
            )
            select
              client_id,
              total_spend,
              n_transactions,
              n_trips,
              n_items,
              total_spend / greatest(n_trips, 1) as avg_basket_spend,
              n_items / greatest(n_trips, 1) as avg_basket_items,
              n_stores,
              n_products,
              points_earned,
              points_spent,
              date_diff('day', last_date, cast(? as date)) as recency_days,
              date_diff('day', first_date, last_date) as tenure_days
            from client_rollup
            order by client_id
            """,
            [_normalize_path(purchases_path), campaign_date],
        ).df()
    finally:
        conn.close()

    return features


def category_spend_by_level(
    purchases_path: str | Path,
    products_path: str | Path,
) -> pd.Series:
    conn = _connect()
    try:
        category_spend = conn.execute(
            f"""
            select
              level_1,
              sum(coalesce(purchase_sum, 0)) as total_spend
            from read_csv(?, header=true, columns={PURCHASE_COLUMNS_SQL}) purchases
            left join read_csv(?, header=true, columns={PRODUCT_COLUMNS_SQL}) products
            using (product_id)
            group by level_1
            order by total_spend desc
            """,
            [_normalize_path(purchases_path), _normalize_path(products_path)],
        ).df()
    finally:
        conn.close()

    category_spend = category_spend.dropna(subset=["level_1"])
    return category_spend.set_index("level_1")["total_spend"]
