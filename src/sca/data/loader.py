"""Data loading, encoding handling, and date parsing for DataCo dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATE_CANDIDATES = [
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "order_date",
    "shipping_date",
    "order date",
]


def detect_date_columns(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> List[str]:
    """Detect available date columns in DataFrame based on names or content."""
    candidates = candidates or DEFAULT_DATE_CANDIDATES
    detected = [c for c in candidates if c in df.columns]
    if not detected:
        # Heuristic fallback: check any column with 'date' in lowercase name
        detected = [c for c in df.columns if "date" in c.lower()]
    return detected


def load_raw_data(
    file_path: Union[str, Path],
    encoding: Optional[str] = "latin1",
    parse_dates: bool = True,
    date_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load DataCo raw CSV with robust multi-encoding detection and date parsing.

    Args:
        file_path: Path to the CSV file.
        encoding: Initial encoding to attempt (default 'latin1').
        parse_dates: Whether to parse date columns into standardized datetime types.
        date_columns: Specific date column names to parse (optional).

    Returns:
        Tuple of (loaded and enriched DataFrame, successful encoding used).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"DataCo dataset not found at: {path.resolve()}")

    encodings_to_try = [encoding, "latin1", "utf-8", "cp1252", "iso-8859-1"]
    # Deduplicate while preserving order
    seen = set()
    encodings_to_try = [e for e in encodings_to_try if e and not (e in seen or seen.add(e))]

    df = None
    successful_encoding = None
    for enc in encodings_to_try:
        try:
            logger.info("Attempting to load CSV with encoding: %s", enc)
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            successful_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError) as err:
            logger.warning("Failed loading with %s: %s. Trying next...", enc, err)

    if df is None or successful_encoding is None:
        raise ValueError(f"Failed to read CSV at {path} with any of: {encodings_to_try}")

    logger.info("Successfully loaded %d rows and %d columns using %s", len(df), len(df.columns), successful_encoding)

    if parse_dates:
        df = parse_and_enrich_dates(df, date_columns=date_columns)

    return df, successful_encoding


def parse_and_enrich_dates(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parse date columns, normalize them, and compute derived time and lead-time attributes."""
    df = df.copy()
    candidate_date_cols = date_columns or detect_date_columns(df)

    # 1. Parse primary order date
    order_col = None
    for cand in candidate_date_cols:
        if "order" in cand.lower():
            order_col = cand
            break

    if order_col:
        df["order_dt"] = pd.to_datetime(df[order_col], errors="coerce")
        df["order_date"] = df["order_dt"].dt.floor("D")
        df["order_year_month"] = df["order_dt"].dt.to_period("M").astype(str)
        df["order_week"] = df["order_dt"].dt.to_period("W").dt.start_time
        df["order_day_of_week"] = df["order_dt"].dt.day_name()
    else:
        logger.warning("Primary order date column not detected.")

    # 2. Parse primary shipping date
    ship_col = None
    for cand in ["shipping date (DateOrders)", "shipping_date", "shipping date"]:
        if cand in df.columns:
            ship_col = cand
            break

    if ship_col:
        df["shipping_dt"] = pd.to_datetime(df[ship_col], errors="coerce")

    # 3. Derive lead-time actual, scheduled, and gap
    if "Days for shipping (real)" in df.columns:
        df["lead_time_actual"] = df["Days for shipping (real)"].astype(float)
    if "Days for shipment (scheduled)" in df.columns:
        df["lead_time_scheduled"] = df["Days for shipment (scheduled)"].astype(float)
    if "lead_time_actual" in df.columns and "lead_time_scheduled" in df.columns:
        df["lead_time_gap"] = df["lead_time_actual"] - df["lead_time_scheduled"]

    # 4. Standardize late delivery flag
    if "Late_delivery_risk" in df.columns:
        df["is_late"] = df["Late_delivery_risk"].astype(int)
    elif "Delivery Status" in df.columns:
        df["is_late"] = (df["Delivery Status"] == "Late delivery").astype(int)

    return df
