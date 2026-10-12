"""Statistical analysis of lead times, delivery performance, and time-series patterns."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_lead_time_stats(
    df: pd.DataFrame,
    shipping_mode_col: str = "Shipping Mode",
    actual_col: str = "Days for shipping (real)",
    scheduled_col: str = "Days for shipment (scheduled)",
) -> pd.DataFrame:
    """Compute lead-time summary statistics and gap per shipping mode.

    Returns DataFrame with:
    - order count
    - actual shipping days (mean, std, median, min, max, p25, p75)
    - scheduled shipping days (mean, std)
    - lead-time gap = actual - scheduled (mean, std, min, max)
    - distribution: frequency count of actual shipping days
    """
    df = df.copy()
    if "lead_time_gap" not in df.columns:
        df["lead_time_gap"] = df[actual_col] - df[scheduled_col]

    stats_list = []
    modes = sorted(df[shipping_mode_col].unique())

    for mode in modes:
        sub = df[df[shipping_mode_col] == mode]
        actual = sub[actual_col]
        scheduled = sub[scheduled_col]
        gap = sub["lead_time_gap"]

        # Distribution of actual shipping days
        day_counts = actual.value_counts().to_dict()
        dist_str = ", ".join(f"{int(d)}d:{count}" for d, count in sorted(day_counts.items()))

        stats_list.append({
            "shipping_mode": mode,
            "order_count": int(len(sub)),
            "actual_mean": round(float(actual.mean()), 3),
            "actual_std": round(float(actual.std()), 3),
            "actual_median": float(actual.median()),
            "actual_min": int(actual.min()),
            "actual_max": int(actual.max()),
            "actual_p25": float(actual.quantile(0.25)),
            "actual_p75": float(actual.quantile(0.75)),
            "scheduled_mean": round(float(scheduled.mean()), 3),
            "scheduled_std": round(float(scheduled.std()), 3),
            "gap_mean": round(float(gap.mean()), 3),
            "gap_std": round(float(gap.std()), 3),
            "gap_min": int(gap.min()),
            "gap_max": int(gap.max()),
            "actual_days_dist": dist_str,
        })

    return pd.DataFrame(stats_list)


def compute_late_delivery_rates(
    df: pd.DataFrame,
    late_col: str = "is_late",
    mode_col: str = "Shipping Mode",
    month_col: str = "order_year_month",
    market_col: str = "Market",
) -> Dict[str, Any]:
    """Compute late-delivery rates overall, per shipping mode, per month, and per market."""
    df = df.copy()
    if late_col not in df.columns:
        if "Late_delivery_risk" in df.columns:
            df[late_col] = df["Late_delivery_risk"].astype(int)
        elif "Delivery Status" in df.columns:
            df[late_col] = (df["Delivery Status"] == "Late delivery").astype(int)
        else:
            raise KeyError("Late delivery indicator column not found.")

    if month_col not in df.columns and "order date (DateOrders)" in df.columns:
        df[month_col] = pd.to_datetime(df["order date (DateOrders)"]).dt.to_period("M").astype(str)

    total_orders = len(df)
    total_late = int(df[late_col].sum())
    overall_rate = float(total_late / total_orders) if total_orders > 0 else 0.0

    # Rate per Shipping Mode
    mode_group = (
        df.groupby(mode_col, as_index=False)[late_col]
        .agg(total_orders="count", late_orders="sum", late_rate="mean")
        .sort_values("total_orders", ascending=False)
        .reset_index(drop=True)
    )
    mode_group["late_rate"] = mode_group["late_rate"].round(4)

    # Rate per Month
    month_group = (
        df.groupby(month_col, as_index=False)[late_col]
        .agg(total_orders="count", late_orders="sum", late_rate="mean")
        .sort_values(month_col)
        .reset_index(drop=True)
    )
    month_group["late_rate"] = month_group["late_rate"].round(4)

    # Rate per Market
    market_group = (
        df.groupby(market_col, as_index=False)[late_col]
        .agg(total_orders="count", late_orders="sum", late_rate="mean")
        .sort_values("total_orders", ascending=False)
        .reset_index(drop=True)
    )
    market_group["late_rate"] = market_group["late_rate"].round(4)

    return {
        "overall_rate": round(overall_rate, 4),
        "total_orders": total_orders,
        "total_late": total_late,
        "by_mode": mode_group,
        "by_month": month_group,
        "by_market": market_group,
    }


def evaluate_series_patterns(
    df: pd.DataFrame,
    top_n_categories: int = 10,
    min_volume_threshold: int = 5000,
) -> pd.DataFrame:
    """Evaluate continuity, sparsity, coefficient of variation, and seasonality across series."""
    df = df.copy()
    if "order_date" not in df.columns:
        df["order_date"] = pd.to_datetime(df["order date (DateOrders)"]).dt.floor("D")

    all_dates = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="D")
    total_days = len(all_dates)

    top_cats = (
        df.groupby("Category Name")["Order Item Quantity"]
        .sum()
        .nlargest(top_n_categories)
        .index.tolist()
    )
    markets = sorted(df["Market"].unique().tolist())

    records = []

    # 1. Overall Category Series
    for cat in top_cats:
        cat_sub = df[df["Category Name"] == cat]
        daily = (
            cat_sub.groupby("order_date")["Order Item Quantity"]
            .sum()
            .reindex(all_dates, fill_value=0)
        )
        vol = int(daily.sum())
        mean_d = float(daily.mean())
        std_d = float(daily.std())
        cv = float(std_d / (mean_d + 1e-9))
        zero_pct = float((daily == 0).mean() * 100)
        ac_lag7 = float(daily.autocorr(lag=7))

        records.append({
            "scope": "Category Overall",
            "category": cat,
            "market": "ALL",
            "total_volume": vol,
            "daily_mean": round(mean_d, 2),
            "daily_std": round(std_d, 2),
            "cv": round(cv, 2),
            "zero_days_pct": round(zero_pct, 1),
            "weekly_ac_lag7": round(ac_lag7, 3) if not np.isnan(ac_lag7) else 0.0,
            "history_days": total_days,
        })

    # 2. Category x Market Slices
    for cat in top_cats:
        for mkt in markets:
            slice_sub = df[(df["Category Name"] == cat) & (df["Market"] == mkt)]
            if slice_sub.empty:
                continue
            daily = (
                slice_sub.groupby("order_date")["Order Item Quantity"]
                .sum()
                .reindex(all_dates, fill_value=0)
            )
            vol = int(daily.sum())
            if vol < min_volume_threshold:
                continue
            mean_d = float(daily.mean())
            std_d = float(daily.std())
            cv = float(std_d / (mean_d + 1e-9))
            zero_pct = float((daily == 0).mean() * 100)
            ac_lag7 = float(daily.autocorr(lag=7))

            records.append({
                "scope": "Category x Market",
                "category": cat,
                "market": mkt,
                "total_volume": vol,
                "daily_mean": round(mean_d, 2),
                "daily_std": round(std_d, 2),
                "cv": round(cv, 2),
                "zero_days_pct": round(zero_pct, 1),
                "weekly_ac_lag7": round(ac_lag7, 3) if not np.isnan(ac_lag7) else 0.0,
                "history_days": total_days,
            })

    results_df = pd.DataFrame(records).sort_values("total_volume", ascending=False).reset_index(drop=True)
    return results_df
