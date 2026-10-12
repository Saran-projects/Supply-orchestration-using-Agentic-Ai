"""Tests for DataCo loading, date parsing, aggregation, lead-time stats, and EDA."""

from pathlib import Path

import pandas as pd
import pytest

from sca.data import (
    aggregate_demand_daily,
    aggregate_demand_weekly,
    compute_late_delivery_rates,
    compute_lead_time_stats,
    extract_single_series,
    load_raw_data,
)


@pytest.fixture(scope="module")
def loaded_df(project_root: Path):
    csv_path = project_root / "data" / "raw" / "DataCoSupplyChainDataset.csv"
    assert csv_path.exists(), f"Raw CSV not found at {csv_path}"
    df, encoding = load_raw_data(csv_path, parse_dates=True)
    return df


class TestDataCoLoading:
    def test_date_parsing_and_enrichment(self, loaded_df: pd.DataFrame):
        assert "order_dt" in loaded_df.columns
        assert "order_date" in loaded_df.columns
        assert "order_year_month" in loaded_df.columns
        assert "order_week" in loaded_df.columns
        assert "lead_time_actual" in loaded_df.columns
        assert "lead_time_scheduled" in loaded_df.columns
        assert "lead_time_gap" in loaded_df.columns
        assert "is_late" in loaded_df.columns

        # Verify date range
        min_date = loaded_df["order_dt"].min()
        max_date = loaded_df["order_dt"].max()
        assert min_date.year == 2015
        assert max_date.year == 2018


class TestDemandAggregation:
    def test_daily_aggregation_structure(self, loaded_df: pd.DataFrame):
        daily = aggregate_demand_daily(loaded_df)
        assert set(daily.columns) == {"date", "category", "market", "demand", "order_count", "sales"}
        assert len(daily) > 0
        assert daily["demand"].sum() == loaded_df["Order Item Quantity"].sum()

    def test_weekly_aggregation_structure(self, loaded_df: pd.DataFrame):
        weekly = aggregate_demand_weekly(loaded_df)
        assert set(weekly.columns) == {"date", "category", "market", "demand", "order_count", "sales"}
        assert len(weekly) > 0
        assert weekly["demand"].sum() == loaded_df["Order Item Quantity"].sum()

    def test_extract_single_series(self, loaded_df: pd.DataFrame):
        daily = aggregate_demand_daily(loaded_df)
        series = extract_single_series(daily, category="Cleats", market="Europe", fill_zeros=True)
        assert not series.empty
        assert "date" in series.columns
        assert "demand" in series.columns
        # Should have continuous days
        date_diffs = series["date"].diff().dropna()
        assert (date_diffs == pd.Timedelta(days=1)).all()


class TestLeadTimeAndLateRates:
    def test_lead_time_stats(self, loaded_df: pd.DataFrame):
        stats = compute_lead_time_stats(loaded_df)
        assert len(stats) == 4  # 4 shipping modes
        modes = stats["shipping_mode"].tolist()
        assert "First Class" in modes
        assert "Same Day" in modes
        assert "Second Class" in modes
        assert "Standard Class" in modes

        # Specific empirical property: First class actual is 2.0 days
        fc_row = stats[stats["shipping_mode"] == "First Class"].iloc[0]
        assert fc_row["actual_mean"] == 2.0
        assert fc_row["actual_std"] == 0.0

    def test_late_delivery_rates(self, loaded_df: pd.DataFrame):
        rates = compute_late_delivery_rates(loaded_df)
        assert 0.50 < rates["overall_rate"] < 0.60
        assert len(rates["by_mode"]) == 4
        assert len(rates["by_month"]) >= 36  # At least 36 months in 2015-2018


class TestGeneratedArtifacts:
    def test_processed_parquets_exist(self, project_root: Path):
        daily_p = project_root / "data" / "processed" / "demand_daily.parquet"
        weekly_p = project_root / "data" / "processed" / "demand_weekly.parquet"
        assert daily_p.is_file(), "demand_daily.parquet not generated"
        assert weekly_p.is_file(), "demand_weekly.parquet not generated"

        # Verify reading parquet
        df_p = pd.read_parquet(daily_p)
        assert len(df_p) > 0

    def test_eda_artifacts_exist(self, project_root: Path):
        summary_md = project_root / "results" / "eda" / "summary.md"
        combined_plot = project_root / "results" / "eda" / "top10_categories_demand_rolling.png"
        assert summary_md.is_file(), "summary.md not generated"
        assert combined_plot.is_file(), "top10 plot not generated"
        assert summary_md.stat().st_size > 500
        assert combined_plot.stat().st_size > 10000
