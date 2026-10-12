"""Data ingestion, preprocessing, aggregation, and EDA for DataCo supply chain dataset."""

from sca.data.aggregation import (
    aggregate_demand_daily,
    aggregate_demand_weekly,
    extract_single_series,
    save_demand_parquets,
)
from sca.data.analysis import (
    compute_late_delivery_rates,
    compute_lead_time_stats,
    evaluate_series_patterns,
)
from sca.data.eda import generate_eda_plots, generate_eda_summary
from sca.data.loader import detect_date_columns, load_raw_data, parse_and_enrich_dates
from sca.data.pipeline import run_data_pipeline
from sca.data.scenarios import (
    LeadTimeSampler,
    Scenario,
    find_scenario_windows,
    generate_scenarios,
    load_chosen_series,
    plot_all_scenarios,
    plot_scenario,
)

__all__ = [
    "load_raw_data",
    "parse_and_enrich_dates",
    "detect_date_columns",
    "aggregate_demand_daily",
    "aggregate_demand_weekly",
    "extract_single_series",
    "save_demand_parquets",
    "compute_lead_time_stats",
    "compute_late_delivery_rates",
    "evaluate_series_patterns",
    "generate_eda_plots",
    "generate_eda_summary",
    "run_data_pipeline",
    "LeadTimeSampler",
    "Scenario",
    "find_scenario_windows",
    "generate_scenarios",
    "load_chosen_series",
    "plot_scenario",
    "plot_all_scenarios",
]
