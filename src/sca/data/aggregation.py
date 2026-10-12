"""Demand time-series aggregation by day and by week per category and per market."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_demand_daily(
    df: pd.DataFrame,
    date_col: str = "order_date",
    category_col: str = "Category Name",
    market_col: str = "Market",
    quantity_col: str = "Order Item Quantity",
    sales_col: str = "Sales",
) -> pd.DataFrame:
    """Aggregate demand daily per category and per market."""
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    agg_dict = {
        quantity_col: "sum",
        "Order Id": "nunique",
    }
    if sales_col in df.columns:
        agg_dict[sales_col] = "sum"

    grouped = (
        df.groupby([date_col, category_col, market_col], as_index=False)
        .agg(agg_dict)
        .rename(
            columns={
                date_col: "date",
                category_col: "category",
                market_col: "market",
                quantity_col: "demand",
                "Order Id": "order_count",
                sales_col: "sales",
            }
        )
    )

    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped = grouped.sort_values(["category", "market", "date"]).reset_index(drop=True)
    return grouped


def aggregate_demand_weekly(
    df: pd.DataFrame,
    week_col: str = "order_week",
    category_col: str = "Category Name",
    market_col: str = "Market",
    quantity_col: str = "Order Item Quantity",
    sales_col: str = "Sales",
) -> pd.DataFrame:
    """Aggregate demand weekly per category and per market."""
    if week_col not in df.columns:
        raise KeyError(f"Week column '{week_col}' not found in DataFrame.")

    agg_dict = {
        quantity_col: "sum",
        "Order Id": "nunique",
    }
    if sales_col in df.columns:
        agg_dict[sales_col] = "sum"

    grouped = (
        df.groupby([week_col, category_col, market_col], as_index=False)
        .agg(agg_dict)
        .rename(
            columns={
                week_col: "date",
                category_col: "category",
                market_col: "market",
                quantity_col: "demand",
                "Order Id": "order_count",
                sales_col: "sales",
            }
        )
    )

    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped = grouped.sort_values(["category", "market", "date"]).reset_index(drop=True)
    return grouped


def extract_single_series(
    df_agg: pd.DataFrame,
    category: Optional[str] = None,
    market: Optional[str] = None,
    fill_zeros: bool = True,
    freq: str = "D",
) -> pd.DataFrame:
    """Extract a continuous time series filtered by optional category and market.

    Fills missing dates with 0 demand if fill_zeros is True.
    """
    filtered = df_agg.copy()
    if category is not None:
        filtered = filtered[filtered["category"] == category]
    if market is not None:
        filtered = filtered[filtered["market"] == market]

    if filtered.empty:
        logger.warning("No records found for category='%s' and market='%s'", category, market)
        return pd.DataFrame(columns=["date", "demand", "order_count", "sales"])

    series = (
        filtered.groupby("date", as_index=False)
        .agg({"demand": "sum", "order_count": "sum", "sales": "sum"})
        .sort_values("date")
    )

    if fill_zeros and not series.empty:
        full_range = pd.date_range(series["date"].min(), series["date"].max(), freq=freq, name="date")
        series = (
            series.set_index("date")
            .reindex(full_range, fill_value=0)
            .reset_index()
        )

    return series


def save_demand_parquets(
    df: pd.DataFrame,
    processed_dir: Union[str, Path] = "data/processed",
    daily_filename: str = "demand_daily.parquet",
    weekly_filename: str = "demand_weekly.parquet",
) -> Tuple[Path, Path]:
    """Aggregate and save daily and weekly demand parquets to processed directory."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / daily_filename
    weekly_path = out_dir / weekly_filename

    logger.info("Computing daily demand aggregation...")
    daily_df = aggregate_demand_daily(df)
    daily_df.to_parquet(daily_path, index=False, engine="pyarrow")
    logger.info("Saved daily demand (%d rows) to %s", len(daily_df), daily_path)

    logger.info("Computing weekly demand aggregation...")
    weekly_df = aggregate_demand_weekly(df)
    weekly_df.to_parquet(weekly_path, index=False, engine="pyarrow")
    logger.info("Saved weekly demand (%d rows) to %s", len(weekly_df), weekly_path)

    return daily_path, weekly_path
