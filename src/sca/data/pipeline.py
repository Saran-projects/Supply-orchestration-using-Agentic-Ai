"""End-to-end data processing, aggregation, and EDA pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sca.config import DataConfig, get_config
from sca.data.aggregation import save_demand_parquets
from sca.data.analysis import compute_late_delivery_rates, compute_lead_time_stats
from sca.data.eda import generate_eda_plots, generate_eda_summary
from sca.data.loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_data_pipeline(data_config: Optional[DataConfig] = None) -> Dict[str, Any]:
    """Execute the full DataCo processing, aggregation, and EDA pipeline.

    Steps:
    1. Parse and enrich raw CSV data with multi-encoding handling.
    2. Aggregate demand daily and weekly; write parquets to data/processed/.
    3. Compute empirical lead-time and scheduled gap statistics.
    4. Compute late delivery rates overall, by mode, and by month.
    5. Generate time-series PNG plots for top 10 categories with rolling mean.
    6. Write EDA findings and recommendations to results/eda/summary.md.
    """
    if data_config is None:
        data_config = get_config().data

    logger.info("Starting SCA Data Pipeline with raw path: %s", data_config.raw_dataset_path)

    # 1. Load and parse raw dataset
    df, encoding_used = load_raw_data(
        file_path=data_config.raw_dataset_path,
        encoding=data_config.encoding,
        parse_dates=True,
    )
    logger.info("Raw data loaded successfully: %d rows (encoding: %s)", len(df), encoding_used)

    # 2. Daily & Weekly Aggregation
    daily_parquet, weekly_parquet = save_demand_parquets(
        df=df,
        processed_dir=data_config.processed_data_dir,
        daily_filename=data_config.processed_daily_filename,
        weekly_filename=data_config.processed_weekly_filename,
    )

    # 3. Statistical Analysis
    lead_time_df = compute_lead_time_stats(df)
    late_delivery_dict = compute_late_delivery_rates(df)

    # 4. Generate EDA Plots
    eda_plots = generate_eda_plots(
        df=df,
        output_dir=data_config.eda_dir,
        top_n=10,
        rolling_window=data_config.rolling_window,
    )

    # 5. Generate EDA Summary Report
    summary_path = generate_eda_summary(
        df=df,
        output_dir=data_config.eda_dir,
    )

    logger.info("Data pipeline completed successfully.")
    return {
        "status": "success",
        "rows_processed": len(df),
        "encoding_used": encoding_used,
        "daily_parquet": daily_parquet,
        "weekly_parquet": weekly_parquet,
        "lead_time_stats": lead_time_df,
        "late_delivery_stats": late_delivery_dict,
        "eda_plots": eda_plots,
        "summary_report": summary_path,
    }


if __name__ == "__main__":
    run_data_pipeline()
