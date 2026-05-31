"""Simple back-tests of two event-driven OFZ strategies.

Strategy A — Auctions: buy long-duration OFZ series 2 trading days before each
Minfin auction and sell 5 trading days after. P&L is computed per auction.

Strategy B — Rate decisions: buy long-duration OFZ series 5 trading days
before each CBR rate decision *when consensus expects a cut* (Forecast <
Previous) and hold for 10 trading days.

Usage::

    python ofz-analysis/scripts/strategy_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ofz_common import (  # noqa: E402
    OUTPUT_DIR,
    candles_by_series_dict,
    get_bar,
    load_auctions,
    load_candles,
    load_rate_decisions,
    per_series_date_index,
    safe_pct,
    trading_day_offset,
)

# Long-duration series chosen per problem statement (26248-like analogues
# included via 26240 / 26244 / 26245 / 26249, which are also long bonds).
AUCTION_BASKET = ["26230", "26238", "26240", "26244", "26245", "26249"]
RATE_BASKET = ["26230", "26238", "26221", "26225"]


def _entry_exit_pl(
    candles: pd.DataFrame,
    series_list,
    anchor_date: pd.Timestamp,
    entry_offset: int,
    exit_offset: int,
):
    by_series = candles_by_series_dict(candles)
    date_index = per_series_date_index(candles)
    rows = []
    for s in series_list:
        dates = date_index.get(s, [])
        d_in = trading_day_offset(dates, anchor_date, entry_offset)
        d_out = trading_day_offset(dates, anchor_date, exit_offset)
        b_in = get_bar(by_series, s, d_in) if d_in is not None else None
        b_out = get_bar(by_series, s, d_out) if d_out is not None else None
        if b_in is None or b_out is None:
            ret = np.nan
            entry = exit_ = np.nan
        else:
            entry = float(b_in["Close"])
            exit_ = float(b_out["Close"])
            ret = safe_pct(exit_, entry)
        rows.append({
            "Series": s,
            "Entry Date": d_in.date().isoformat() if d_in is not None else None,
            "Exit Date": d_out.date().isoformat() if d_out is not None else None,
            "Entry Close": entry,
            "Exit Close": exit_,
            "Return (%)": ret,
        })
    return rows


def backtest_auctions(candles: pd.DataFrame, auctions: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, auc in auctions.iterrows():
        a_date = auc["Date"]
        a_series = str(auc["Series"])
        legs = _entry_exit_pl(candles, AUCTION_BASKET, a_date, entry_offset=-2, exit_offset=5)
        basket_returns = [r["Return (%)"] for r in legs if r["Return (%)"] is not None and not (isinstance(r["Return (%)"], float) and np.isnan(r["Return (%)"]))]
        basket_avg = float(np.mean(basket_returns)) if basket_returns else np.nan
        for leg in legs:
            out_rows.append({
                "Strategy": "A_Auctions",
                "Event Date": a_date.date().isoformat(),
                "Auction Series": a_series,
                **leg,
                "Basket Avg Return (%)": basket_avg,
            })
    return pd.DataFrame(out_rows)


def backtest_rates(candles: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, dec in decisions.iterrows():
        expects_cut = float(dec["Forecast"]) < float(dec["Previous"])
        if not expects_cut:
            continue
        d_date = dec["Date"]
        legs = _entry_exit_pl(candles, RATE_BASKET, d_date, entry_offset=-5, exit_offset=10)
        basket_returns = [r["Return (%)"] for r in legs if r["Return (%)"] is not None and not (isinstance(r["Return (%)"], float) and np.isnan(r["Return (%)"]))]
        basket_avg = float(np.mean(basket_returns)) if basket_returns else np.nan
        for leg in legs:
            out_rows.append({
                "Strategy": "B_Rates_Dovish_Consensus",
                "Event Date": d_date.date().isoformat(),
                "Rate (%)": float(dec["Rate"]),
                "Forecast (%)": float(dec["Forecast"]),
                "Previous (%)": float(dec["Previous"]),
                **leg,
                "Basket Avg Return (%)": basket_avg,
            })
    return pd.DataFrame(out_rows)


def _summarise(df: pd.DataFrame, label: str) -> dict:
    valid = df.dropna(subset=["Return (%)"])
    if valid.empty:
        return {"strategy": label, "n_trades": 0}
    per_event = (
        valid.groupby("Event Date")["Return (%)"].mean()
    )
    return {
        "strategy": label,
        "n_legs": int(len(valid)),
        "n_events": int(per_event.shape[0]),
        "mean_leg_return_pct": float(valid["Return (%)"].mean()),
        "median_leg_return_pct": float(valid["Return (%)"].median()),
        "win_rate_pct": float((valid["Return (%)"] > 0).mean() * 100.0),
        "mean_event_basket_return_pct": float(per_event.mean()),
        "sum_event_basket_return_pct": float(per_event.sum()),
        "best_event": f"{per_event.idxmax()}: {per_event.max():+.2f}%",
        "worst_event": f"{per_event.idxmin()}: {per_event.min():+.2f}%",
    }


def main() -> None:
    candles = load_candles()
    auctions = load_auctions()
    decisions = load_rate_decisions()

    a_df = backtest_auctions(candles, auctions)
    b_df = backtest_rates(candles, decisions)

    full = pd.concat([a_df, b_df], ignore_index=True, sort=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "backtest_results.csv"
    full.to_csv(out_path, index=False)

    summary_a = _summarise(a_df, "A_Auctions")
    summary_b = _summarise(b_df, "B_Rates_Dovish_Consensus")
    print("=" * 72)
    print("Strategy A — Auctions (buy long basket D-2, sell D+5)")
    print("-" * 72)
    for k, v in summary_a.items():
        print(f"  {k:<32} {v}")
    print()
    print("Strategy B — Rate decisions (buy long basket D-5 if Forecast<Previous, sell D+10)")
    print("-" * 72)
    for k, v in summary_b.items():
        print(f"  {k:<32} {v}")
    print()
    print(f"Wrote {out_path}  legs={len(full)}")


if __name__ == "__main__":
    main()
