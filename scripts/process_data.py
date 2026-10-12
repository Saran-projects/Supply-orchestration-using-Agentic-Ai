#!/usr/bin/env python3
"""Run DataCo data processing pipeline, generate parquet aggregations, plots, and summary."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sca.data.pipeline import run_data_pipeline


def main():
    print("[+] Starting DataCo Supply Chain Processing & EDA Pipeline...")
    results = run_data_pipeline()
    print("\n[+] Pipeline completed successfully!")
    print(f"    - Processed rows: {results['rows_processed']:,}")
    print(f"    - Daily Parquet:  {results['daily_parquet']}")
    print(f"    - Weekly Parquet: {results['weekly_parquet']}")
    print(f"    - Summary Report: {results['summary_report']}")
    print(f"    - EDA Plots generated: {len(results['eda_plots'])} files")


if __name__ == "__main__":
    main()
