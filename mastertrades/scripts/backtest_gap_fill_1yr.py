"""1-year backtest of the Gap Fill signal that's surfacing as the 0.2-edge play.

For each ticker (SPY, QQQ, IWM, AAPL) we:
  1. Pull ~2 years of daily OHLC from Polygon (1y test + 1y prior history for stats).
  2. Walk forward day-by-day. Each morning we recompute the same
     `today_gap_status` the live app calls, using ONLY data prior to that day.
  3. If the signal is WATCH_FILL (>=70% historical fill rate) or NEAR_FILL
     (>=50%) we open a "fill bet" at the open: long underlying-direction toward
     prev_close (puts on a gap up, calls on a gap down).
  4. Outcome:
        - If gap_filled that session  -> P&L = abs(open - prev_close) / open
          (we captured the full distance to the fill target).
        - If NOT filled                -> P&L = -(close - open in our direction)/open
          (we rode the underlying against us into the close).
  5. Aggregate: fill rate, avg P&L, equity curve at $1k per trade
     (leverage = 1, just the underlying %), and a rough 5x options-leverage view.

No look-ahead: stats_by_bucket is recomputed each day on the trailing window only.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.scanner import fetch_or_load_daily
from src.gap_analysis import (
    add_gap_features,
    gap_stats_by_bucket,
    today_gap_status,
    DEFAULT_MIN_GAP_PCT,
)

TICKERS = ("SPY", "QQQ", "IWM", "AAPL")
TEST_DAYS = 252           # ~1 year
HISTORY_DAYS = 252        # prior history for stats
MIN_BUCKET_N = 5          # don't trade buckets with fewer than this many priors

OPTIONS_LEVERAGE = 5.0    # rough 1-week ATM option delta-leverage approximation
PER_TRADE_DOLLARS = 1000.0


def backtest_ticker(tkr: str) -> pd.DataFrame:
    # Use ~2y of cached daily bars (defaults pull plenty of history).
    daily = fetch_or_load_daily(tkr)
    if daily is None or len(daily) < (TEST_DAYS + HISTORY_DAYS):
        print(f"  {tkr}: insufficient data ({0 if daily is None else len(daily)} bars)")
        return pd.DataFrame()

    feat = add_gap_features(daily).dropna(subset=["prev_close"])

    rows: list[dict] = []
    test_start_idx = max(HISTORY_DAYS, len(feat) - TEST_DAYS)

    for i in range(test_start_idx, len(feat)):
        hist = feat.iloc[:i]                 # everything BEFORE today
        today_slice = feat.iloc[: i + 1]     # everything up to & including today
        stats = gap_stats_by_bucket(hist)
        tg = today_gap_status(tkr, today_slice, stats)

        if tg.signal not in ("WATCH_FILL", "NEAR_FILL"):
            continue
        if tg.hist_n_similar < MIN_BUCKET_N:
            continue

        today_row = feat.iloc[i]
        open_px = float(today_row["Open"])
        prev_cl = float(today_row["prev_close"])
        filled = bool(today_row["gap_filled"])
        close_px = float(today_row["Close"])

        # Direction of the bet: gap up -> bet DOWN to prev_close ; gap down -> bet UP
        if tg.gap_dir == "up":
            target_move = (open_px - prev_cl)              # positive distance to win
            actual_move = (open_px - close_px) if not filled else (open_px - prev_cl)
        else:
            target_move = (prev_cl - open_px)
            actual_move = (close_px - open_px) if not filled else (prev_cl - open_px)

        # If filled, we win the full target move (closing at the fill level).
        # If not filled, we lose the underlying move against us into the close.
        underlying_pct = (actual_move / open_px) if filled else -abs(actual_move / open_px) \
            if (close_px - open_px) * (1 if tg.gap_dir == "down" else -1) < 0 else (actual_move / open_px)

        # Cleaner: just compute realised intraday move IN OUR DIRECTION.
        if tg.gap_dir == "up":
            realised_pct = (open_px - (prev_cl if filled else close_px)) / open_px
        else:
            realised_pct = ((prev_cl if filled else close_px) - open_px) / open_px

        rows.append({
            "date":           today_row.name.date(),
            "ticker":         tkr,
            "signal":         tg.signal,
            "gap_dir":        tg.gap_dir,
            "gap_pct":        round(tg.gap_pct * 100, 3),
            "hist_fill_rate": round(tg.hist_fill_rate * 100, 1) if tg.hist_fill_rate else None,
            "hist_n":         tg.hist_n_similar,
            "filled":         filled,
            "realised_pct":   round(realised_pct * 100, 3),
            "underlying_$":   round(realised_pct * PER_TRADE_DOLLARS, 2),
            "options_$":      round(realised_pct * PER_TRADE_DOLLARS * OPTIONS_LEVERAGE, 2),
        })

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for t in TICKERS:
        print(f"Backtesting {t} …")
        df = backtest_ticker(t)
        if not df.empty:
            print(f"  {t}: {len(df)} signals fired")
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    if full.empty:
        print("No signals fired in the test window.")
        return

    full = full.sort_values("date").reset_index(drop=True)

    # Per-ticker summary
    print("\n" + "=" * 78)
    print("PER-TICKER SUMMARY (1-year walk-forward)")
    print("=" * 78)
    grp = full.groupby("ticker").agg(
        n=("filled", "size"),
        fill_rate=("filled", lambda s: round(s.mean() * 100, 1)),
        avg_realised_pct=("realised_pct", lambda s: round(s.mean(), 3)),
        med_realised_pct=("realised_pct", lambda s: round(s.median(), 3)),
        underlying_pnl_dollars=("underlying_$", lambda s: round(s.sum(), 2)),
        options_pnl_dollars=("options_$", lambda s: round(s.sum(), 2)),
    )
    print(grp.to_string())

    # Per-signal-tier summary
    print("\n" + "=" * 78)
    print("BY SIGNAL TIER")
    print("=" * 78)
    by_sig = full.groupby("signal").agg(
        n=("filled", "size"),
        fill_rate=("filled", lambda s: round(s.mean() * 100, 1)),
        avg_realised_pct=("realised_pct", lambda s: round(s.mean(), 3)),
        underlying_pnl_dollars=("underlying_$", lambda s: round(s.sum(), 2)),
        options_pnl_dollars=("options_$", lambda s: round(s.sum(), 2)),
    )
    print(by_sig.to_string())

    # Total
    print("\n" + "=" * 78)
    print("TOTAL — 1 year, all 4 tickers, all WATCH_FILL + NEAR_FILL signals")
    print("=" * 78)
    n = len(full)
    fr = full["filled"].mean() * 100
    avg = full["realised_pct"].mean()
    med = full["realised_pct"].median()
    win_rate = (full["realised_pct"] > 0).mean() * 100
    underlying_total = full["underlying_$"].sum()
    options_total = full["options_$"].sum()
    print(f"Signals fired:         {n}")
    print(f"Gap fill rate:         {fr:.1f}%")
    print(f"Trade win rate:        {win_rate:.1f}%   (realised > 0)")
    print(f"Avg realised move:     {avg:+.3f}% per trade")
    print(f"Median realised move:  {med:+.3f}% per trade")
    print(f"Per-trade $ at $1k notional, underlying-only: avg ${full['underlying_$'].mean():+.2f}")
    print(f"Total P&L on $1k/trade, UNDERLYING leg:       ${underlying_total:+,.0f}")
    print(f"Total P&L on $1k/trade, OPTIONS (~5× lev):    ${options_total:+,.0f}")
    print(f"Worst single trade:   {full['realised_pct'].min():+.2f}%")
    print(f"Best  single trade:   {full['realised_pct'].max():+.2f}%")

    # Save the trade log for inspection
    out = "/tmp/gap_backtest_1yr.csv"
    full.to_csv(out, index=False)
    print(f"\nFull trade log -> {out}")


if __name__ == "__main__":
    main()
