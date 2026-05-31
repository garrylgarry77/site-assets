"""Build the key-rate decision impact matrix and XLSX report.

Usage::

    python ofz-analysis/scripts/build_rate_matrix.py

Reads ``data/candles.csv`` (D1 filter) and ``data/rate_decisions.csv`` and
writes ``output/rate_matrix.xlsx`` with sheets ``Matrix``, ``Summary_by_Series``
and ``Sensitivity_Ranking`` plus a ``Surprise_Correlation`` sheet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ofz_common import (  # noqa: E402
    OUTPUT_DIR,
    attach_reference,
    candles_by_series_dict,
    get_bar,
    load_candles,
    load_rate_decisions,
    per_series_date_index,
    safe_pct,
    trading_day_offset,
)


def build_matrix(candles: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    by_series = candles_by_series_dict(candles)
    date_index = per_series_date_index(candles)
    all_series = sorted(by_series.keys())

    rows = []
    for _, dec in decisions.iterrows():
        d_date = dec["Date"]
        for bond in all_series:
            dates = date_index.get(bond, [])

            def bar(off):
                t = trading_day_offset(dates, d_date, off)
                return get_bar(by_series, bond, t) if t is not None else None

            b_m5, b_m1, b_0, b_p1, b_p5, b_p10 = bar(-5), bar(-1), bar(0), bar(1), bar(5), bar(10)
            close_m1 = b_m1["Close"] if b_m1 is not None else np.nan
            close_p5 = b_p5["Close"] if b_p5 is not None else np.nan
            close_p10 = b_p10["Close"] if b_p10 is not None else np.nan

            rows.append({
                "Decision Date": d_date.date().isoformat(),
                "Bond Series": bond,
                "Rate (%)": float(dec["Rate"]),
                "Forecast (%)": float(dec["Forecast"]),
                "Previous (%)": float(dec["Previous"]),
                "Surprise (bp)": float(dec["Surprise_bp"]),
                "Change (bp)": float(dec["Change_bp"]),
                "Close D-5": b_m5["Close"] if b_m5 is not None else np.nan,
                "Close D-1": close_m1,
                "Open D": b_0["Open"] if b_0 is not None else np.nan,
                "Close D": b_0["Close"] if b_0 is not None else np.nan,
                "Close D+1": b_p1["Close"] if b_p1 is not None else np.nan,
                "Close D+5": close_p5,
                "Close D+10": close_p10,
                "Return D-1->D (%)": safe_pct(b_0["Close"] if b_0 is not None else None, close_m1),
                "Return D-1->D+1 (%)": safe_pct(b_p1["Close"] if b_p1 is not None else None, close_m1),
                "Return D-1->D+5 (%)": safe_pct(close_p5, close_m1),
                "Return D-1->D+10 (%)": safe_pct(close_p10, close_m1),
            })

    df = pd.DataFrame(rows)
    df = attach_reference(df, on="Bond Series")
    return df


def build_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Return D-1->D (%)",
        "Return D-1->D+1 (%)",
        "Return D-1->D+5 (%)",
        "Return D-1->D+10 (%)",
    ]
    agg = (
        matrix
        .groupby("Bond Series")[metric_cols]
        .agg(["mean", "median", "count"])
    )
    agg.columns = [f"{m} {stat}" for m, stat in agg.columns]
    return agg.reset_index()


def build_sensitivity(matrix: pd.DataFrame) -> pd.DataFrame:
    sub = matrix.copy()
    sub["abs_D_to_D10"] = sub["Return D-1->D+10 (%)"].abs()
    rank = (
        sub.groupby("Bond Series")["abs_D_to_D10"]
        .agg(["mean", "median", "count"])
        .rename(columns={
            "mean": "Mean |Return D-1->D+10| (%)",
            "median": "Median |Return D-1->D+10| (%)",
            "count": "N Observations",
        })
        .sort_values("Mean |Return D-1->D+10| (%)", ascending=False)
        .reset_index()
    )
    return rank


def build_surprise_correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-series Pearson correlation between Surprise (bp) and Return D-1->D+5 (%)."""
    out = []
    for bond, grp in matrix.groupby("Bond Series"):
        valid = grp.dropna(subset=["Surprise (bp)", "Return D-1->D+5 (%)"])
        if len(valid) >= 2 and valid["Surprise (bp)"].nunique() > 1:
            corr = valid["Surprise (bp)"].corr(valid["Return D-1->D+5 (%)"])
        else:
            corr = np.nan
        out.append({
            "Bond Series": bond,
            "Corr(Surprise, Return D-1->D+5)": corr,
            "N": len(valid),
            "Mean Return D-1->D+5 (%)": valid["Return D-1->D+5 (%)"].mean(),
        })
    return pd.DataFrame(out).sort_values("Corr(Surprise, Return D-1->D+5)").reset_index(drop=True)


def main() -> None:
    candles = load_candles()
    decisions = load_rate_decisions()
    matrix = build_matrix(candles, decisions)
    summary = build_summary(matrix)
    sensitivity = build_sensitivity(matrix)
    surprise_corr = build_surprise_correlation(matrix)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rate_matrix.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        matrix.to_excel(writer, sheet_name="Matrix", index=False)
        summary.to_excel(writer, sheet_name="Summary_by_Series", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity_Ranking", index=False)
        surprise_corr.to_excel(writer, sheet_name="Surprise_Correlation", index=False)
    print(f"Wrote {out_path}  rows={len(matrix)}  series={matrix['Bond Series'].nunique()}  decisions={matrix['Decision Date'].nunique()}")


if __name__ == "__main__":
    main()
