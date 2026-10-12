#!/usr/bin/env python3
"""Initialize DuckDB schema and verify dataset integrity."""

import argparse
import sys
from pathlib import Path

# Add src to path if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sca.config import get_config
from sca.db import DuckDBManager, verify_schema


def main():
    parser = argparse.ArgumentParser(description="Initialize SCA DuckDB schema & verify data")
    parser.add_argument("--check-data", action="store_true", help="Check DataCo raw CSV dataset")
    parser.add_argument("--db-path", type=str, default=None, help="Custom DuckDB file path")
    args = parser.parse_args()

    cfg = get_config()
    db_path = args.db_path or cfg.experiment.db_path
    print(f"[+] Initializing DuckDB at: {db_path}")

    with DuckDBManager(db_path=db_path) as db:
        status = verify_schema(db.conn)
        print("[+] Schema status:")
        for tbl, exists in status.items():
            print(f"    - {tbl:15s}: {'OK' if exists else 'MISSING'}")

    if args.check_data:
        csv_path = cfg.data.raw_dataset_path
        print(f"[+] Checking raw dataset: {csv_path}")
        if csv_path.exists():
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"    Found {csv_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"    [!] Warning: dataset not found at {csv_path}")


if __name__ == "__main__":
    main()
