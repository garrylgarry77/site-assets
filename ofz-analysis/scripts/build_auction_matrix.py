"""Build the auction-impact matrix and XLSX report.

Usage::

    python ofz-analysis/scripts/build_auction_matrix.py

Reads ``data/candles.csv`` (D1 filter) and ``data/auctions.csv`` and writes
``output/auction_matrix.xlsx`` with sheets ``Matrix``, ``Summary_by_Series``
and ``Sensitivity_Ranking``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Make the script runnable both as ``python ofz-analysis/scripts/build_auction_matrix.py``
# and as ``python -m ofz-analysis.scripts.build_auction_matrix``.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ofz_common import (  # noqa: E402
    OUTPUT_DIR,
    attach_reference,
    candles_by_series_dict,
    get_bar,
    load_auctions,
    load_candles,
    per_series_date_index,
    safe_pct,
    trading_day_offset,
)


def build_matrix(candles: pd.DataFrame, auctions: pd.DataFrame) -> pd.DataFrame:
    by_series = candles_by_series_dict(candles)
    date_index = per_series_date_index(candles)
    all_series = sorted(by_series.keys())

    rows = []
    for _, auc in auctions.iterrows():
        a_date = auc["Date"]
        a_series = str(auc["Series"])
        for bond in all_series:
            dates = date_index.get(bond, [])
            # trading-day offsets from anchor (auction date) within this series
            d_m2 = trading_day_offset(dates, a_date, -2)
            d_m1 = trading_day_offset(dates, a_date, -1)
            d_0 = trading_day_offset(dates, a_date, 0)
            d_p1 = trading_day_offset(dates, a_date, 1)
            d_p2 = trading_day_offset(dates, a_date, 2)
            d_p3 = trading_day_offset(dates, a_date, 3)
            d_p4 = trading_day_offset(dates, a_date, 4)
            d_p5 = trading_day_offset(dates, a_date, 5)
            d_p6 = trading_day_offset(dates, a_date, 6)

            def bar(d):
                return get_bar(by_series, bond, d) if d is not None else None

            b_m2, b_m1, b_0, b_p1, b_p2 = bar(d_m2), bar(d_m1), bar(d_0), bar(d_p1), bar(d_p2)
            window_post = [bar(d) for d in (d_p1, d_p2, d_p3, d_p4, d_p5, d_p6)]
            highs = [b["High"] for b in window_post if b is not None]
            lows = [b["Low"] for b in window_post if b is not None]
            max_post = max(highs) if highs else np.nan
            min_post = min(lows) if lows else np.nan

            close_m1 = b_m1["Close"] if b_m1 is not None else np.nan
            close_0 = b_0["Close"] if b_0 is not None else np.nan
            close_p6 = window_post[-1]["Close"] if window_post and window_post[-1] is not None else np.nan
            # Return D-1 -> D+6 (use last available D+N close in window)
            close_last_post = next(
                (b["Close"] for b in reversed(window_post) if b is not None),
                np.nan,
            )
            ret_m1_to_p6 = safe_pct(close_last_post, close_m1)
            # Max drawdown D..D+6 from highest close seen so far down to subsequent low
            window_all = [b_0] + window_post
            closes_all = [b["Close"] for b in window_all if b is not None]
            lows_all = [b["Low"] for b in window_all if b is not None]
            if closes_all and lows_all:
                run_max = np.maximum.accumulate(closes_all)
                # align lows to same length as run_max (skip leading None of close)
                # safest: drawdown = min(lows_all)/max(closes_all)-1 conservatively
                # but proper running max-drawdown:
                running_peak = -np.inf
                max_dd = 0.0
                # iterate in order over window_all
                for b in window_all:
                    if b is None:
                        continue
                    running_peak = max(running_peak, b["Close"], b["High"])
                    dd = b["Low"] / running_peak - 1.0
                    if dd < max_dd:
                        max_dd = dd
                max_dd_pct = max_dd * 100.0
            else:
                max_dd_pct = np.nan

            rows.append({
                "Auction Date": a_date.date().isoformat(),
                "Auction Series": a_series,
                "Bond Series": bond,
                "Open D-2": b_m2["Open"] if b_m2 is not None else np.nan,
                "Low D-2": b_m2["Low"] if b_m2 is not None else np.nan,
                "Open D-1": b_m1["Open"] if b_m1 is not None else np.nan,
                "Low D-1": b_m1["Low"] if b_m1 is not None else np.nan,
                "Close D-1": close_m1,
                "Open D": b_0["Open"] if b_0 is not None else np.nan,
                "Close D": close_0,
                "Close D+1": b_p1["Close"] if b_p1 is not None else np.nan,
                "Close D+2": b_p2["Close"] if b_p2 is not None else np.nan,
                "Max(D+1..D+6)": max_post,
                "Min(D+1..D+6)": min_post,
                "Return D-1->D+6 (%)": ret_m1_to_p6,
                "Return D-1->D+2 (%)": safe_pct(b_p2["Close"] if b_p2 is not None else None, close_m1),
                "Return D-1->D (%)": safe_pct(close_0, close_m1),
                "Max Drawdown D..D+6 (%)": max_dd_pct,
                "Is Auction Series": bond == a_series,
            })

    df = pd.DataFrame(rows)
    df = attach_reference(df, on="Bond Series")
    return df


def build_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Return D-1->D (%)",
        "Return D-1->D+2 (%)",
        "Return D-1->D+6 (%)",
        "Max Drawdown D..D+6 (%)",
    ]
    agg = (
        matrix
        .groupby(["Bond Series", "Is Auction Series"], dropna=False)[metric_cols]
        .agg(["mean", "median", "count"])
    )
    agg.columns = [f"{m} {stat}" for m, stat in agg.columns]
    return agg.reset_index()


def build_sensitivity(matrix: pd.DataFrame) -> pd.DataFrame:
    sub = matrix.copy()
    sub["abs_D_to_D2"] = sub["Return D-1->D+2 (%)"].abs()
    rank = (
        sub.groupby("Bond Series")["abs_D_to_D2"]
        .agg(["mean", "median", "count"])
        .rename(columns={
            "mean": "Mean |Return D-1->D+2| (%)",
            "median": "Median |Return D-1->D+2| (%)",
            "count": "N Observations",
        })
        .sort_values("Mean |Return D-1->D+2| (%)", ascending=False)
        .reset_index()
    )
    return rank


def main() -> None:
    candles = load_candles()
    auctions = load_auctions()
    matrix = build_matrix(candles, auctions)
    summary = build_summary(matrix)
    sensitivity = build_sensitivity(matrix)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "auction_matrix.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        matrix.to_excel(writer, sheet_name="Matrix", index=False)
        summary.to_excel(writer, sheet_name="Summary_by_Series", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity_Ranking", index=False)
    print(f"Wrote {out_path}  rows={len(matrix)}  series={matrix['Bond Series'].nunique()}  auctions={matrix['Auction Date'].nunique()}")


if __name__ == "__main__":
    main()
