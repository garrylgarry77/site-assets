"""Common helpers for OFZ event-impact analysis scripts.

Functions in this module are intentionally lightweight and have no I/O side
effects beyond reading CSV files from the ``data/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def load_candles(path: Optional[Path] = None, timeframe: str = "D1") -> pd.DataFrame:
    """Load candle CSV (filters by timeframe) and normalise types.

    Returns a DataFrame with columns: Ticker, Series (str), Date (datetime64[D]),
    Open, High, Low, Close, Volume.
    """
    if path is None:
        path = DATA_DIR / "candles.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Timeframe"] == timeframe].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # "OFZ 26230" -> "26230"
    df["Series"] = df["Ticker"].astype(str).str.replace(r"^OFZ\s+", "", regex=True)
    df = df.sort_values(["Series", "Date"]).reset_index(drop=True)
    return df


def load_auctions(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "auctions.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Series"] = df["Series"].astype(str)
    return df


def load_rate_decisions(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "rate_decisions.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    for col in ("Rate", "Forecast", "Previous"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Surprise_bp"] = (df["Rate"] - df["Forecast"]) * 100.0
    df["Change_bp"] = (df["Rate"] - df["Previous"]) * 100.0
    return df


def load_bond_reference(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Optional bond reference (Series, Maturity, Coupon, Duration, YTM, ...).

    Returns ``None`` if a real (non-template) reference file is not present, so
    that the merge in the scripts becomes a no-op for missing columns.
    """
    if path is None:
        path = DATA_DIR / "bond_reference.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Series"] = df["Series"].astype(str)
    return df


def attach_reference(df: pd.DataFrame, on: str = "Bond Series") -> pd.DataFrame:
    """Left-join optional bond reference data onto a matrix DataFrame."""
    ref = load_bond_reference()
    if ref is None:
        return df
    return df.merge(ref, left_on=on, right_on="Series", how="left").drop(columns=["Series"])


def per_series_date_index(candles: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    """For each series, return sorted list of available trading dates."""
    out: Dict[str, List[pd.Timestamp]] = {}
    for series, grp in candles.groupby("Series", sort=False):
        out[series] = list(grp["Date"].sort_values().unique())
    return out


def trading_day_offset(
    dates: List[pd.Timestamp],
    anchor: pd.Timestamp,
    offset: int,
) -> Optional[pd.Timestamp]:
    """Return ``dates[idx(anchor) + offset]`` where ``idx(anchor)`` is the index
    of the first available trading day ``>= anchor``.

    Returns ``None`` when out of range, the anchor has no available date in the
    series, or the resulting position is past the end of the available range.
    """
    if not dates:
        return None
    arr = np.array(dates, dtype="datetime64[ns]")
    pos = int(np.searchsorted(arr, np.datetime64(anchor)))
    # If anchor itself is past the last date, only negative offsets make sense.
    if pos >= len(arr):
        pos = len(arr) - 1
        # only allow stepping further back from the last available day
        if offset >= 0:
            return None
    target = pos + offset
    if target < 0 or target >= len(arr):
        return None
    return pd.Timestamp(arr[target])


def get_bar(
    candles_by_series: Dict[str, pd.DataFrame],
    series: str,
    date: pd.Timestamp,
) -> Optional[pd.Series]:
    df = candles_by_series.get(series)
    if df is None:
        return None
    row = df.loc[df["Date"] == date]
    if row.empty:
        return None
    return row.iloc[0]


def candles_by_series_dict(candles: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {s: g.reset_index(drop=True) for s, g in candles.groupby("Series", sort=False)}


def safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        if np.isnan(a) or np.isnan(b) or b == 0:
            return None
    except TypeError:
        return None
    return (a / b - 1.0) * 100.0
