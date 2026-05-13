"""4-week backtest of algo-recommended MA-bounce trades against real Polygon options.

For every TOUCHING event in the last 28 calendar days (cutoff at -5 trading days
so we have 5d forward returns), we simulate buying an ATM weekly call and holding
5 trading days. For winners, we also evaluate a strike grid (ATM-2..ATM+3) and
the next-week expiry to see if a different contract would have profited more.
For losers, we capture features useful for loss-minimization filters.

Run:
    cd mastertrades && python scripts/backtest_4weeks.py
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure mastertrades/ is on path so `from src.X import Y` works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scanner import fetch_or_load_daily
from src.weekly_levels import compute_weekly_mas
from src.ma_setups_universe import HIGH_EDGE_SETUPS
from src.options_history import (
    list_expired_contracts, fetch_option_daily_bars, next_friday,
)

DATA_DIR = ROOT / "data"
TODAY = date.today()
WINDOW_START = TODAY - timedelta(days=35)   # generous buffer
WINDOW_END   = TODAY - timedelta(days=5)    # need 5d forward
TOUCH_PCT = 0.006   # within 0.6% counts as a touch


def load_setup_daily_with_ma(ticker: str, ma_label: str) -> pd.DataFrame:
    """Return daily df with `ma_value` column for the given MA label."""
    daily = fetch_or_load_daily(ticker, data_dir=DATA_DIR, refresh=False)
    if daily is None or daily.empty:
        return pd.DataFrame()
    daily = daily.sort_index()
    daily.index = pd.to_datetime(daily.index)
    mas = compute_weekly_mas(daily)
    if ma_label not in mas:
        return pd.DataFrame()
    series = mas[ma_label]
    daily = daily.copy()
    daily["ma_value"] = series.reindex(daily.index, method="ffill")
    return daily


def find_touches(daily: pd.DataFrame, start: date, end: date) -> list[dict]:
    """A touch = day where Low <= MA*(1+TOUCH_PCT) AND Close >= MA*(1-TOUCH_PCT)."""
    touches = []
    for ts, row in daily.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        if d < start or d > end:
            continue
        ma = row.get("ma_value")
        if pd.isna(ma) or ma <= 0:
            continue
        low, high, close = row["Low"], row["High"], row["Close"]
        # touch from above: low pierced or got within 0.6% of MA, close still near
        if low <= ma * (1 + TOUCH_PCT) and close >= ma * (1 - TOUCH_PCT):
            touches.append({
                "date":  d.isoformat(),
                "low":   float(low),
                "high":  float(high),
                "close": float(close),
                "ma":    float(ma),
                "dist_at_close_pct": float((close - ma) / ma * 100),
            })
    return touches


def get_forward_close(daily: pd.DataFrame, entry_date_iso: str, days_forward: int = 5) -> tuple[str | None, float | None]:
    idx = daily.index.searchsorted(pd.Timestamp(entry_date_iso))
    target_idx = idx + days_forward
    if target_idx >= len(daily):
        return None, None
    ts = daily.index[target_idx]
    return ts.date().isoformat(), float(daily.iloc[target_idx]["Close"])


def simulate_call_trade(
    underlying:    str,
    entry_date:    str,
    entry_price:   float,
    days_forward:  int = 5,
    strike_offset: int = 0,        # 0 = ATM, +1 = ATM+$1, etc.
    expiry_weeks_out: int = 1,     # 1 = next Friday, 2 = following Friday
) -> dict:
    """Buy ATM (or offset) call on entry date, sell 5 trading days later.

    Returns dict with entry, exit, return_pct, contract details, or {'error': ...}.
    """
    ed = datetime.fromisoformat(entry_date).date()
    target_friday = next_friday(ed, min_days_out=3)
    if expiry_weeks_out > 1:
        target_friday = target_friday + timedelta(days=7 * (expiry_weeks_out - 1))
    exp_window_lo = (target_friday - timedelta(days=2)).isoformat()
    exp_window_hi = (target_friday + timedelta(days=2)).isoformat()

    # Strike grid: ATM ± wide window, then pick the contract closest to target.
    # entry_price * 0.05 absorbs typical strike intervals ($1, $2.50, $5, $10).
    target_strike = entry_price + strike_offset
    half_band = max(3.0, entry_price * 0.05)
    try:
        contracts = list_expired_contracts(
            underlying       = underlying,
            expiration_gte   = exp_window_lo,
            expiration_lte   = exp_window_hi,
            contract_type    = "call",
            strike_lo        = target_strike - half_band,
            strike_hi        = target_strike + half_band,
        )
    except Exception as e:
        return {"error": f"contract list failed: {e}"}

    if not contracts:
        return {"error": "no contracts in band"}

    # Pick contract with strike closest to target_strike, then expiry closest to Friday
    contracts.sort(key=lambda c: (abs(c.strike - target_strike),
                                   abs((datetime.fromisoformat(c.expiration).date()
                                        - target_friday).days)))
    chosen = contracts[0]

    # Fetch option bars from entry to exit (or expiry)
    exit_window = (ed + timedelta(days=days_forward + 7)).isoformat()
    try:
        bars = fetch_option_daily_bars(chosen.ticker, entry_date, exit_window)
    except Exception as e:
        return {"error": f"bar fetch failed: {e}"}
    if not bars:
        return {"error": "no bars for contract"}

    bars_by_date = {b.date: b for b in bars}
    if entry_date not in bars_by_date:
        return {"error": f"no entry bar on {entry_date}"}
    entry_bar = bars_by_date[entry_date]
    if entry_bar.close <= 0:
        return {"error": "zero entry price"}

    # Find exit: day +5 trading sessions, or last available bar before expiry
    sorted_dates = sorted(bars_by_date.keys())
    try:
        entry_pos = sorted_dates.index(entry_date)
    except ValueError:
        return {"error": "entry date not in sorted bars"}
    exit_pos = min(entry_pos + days_forward, len(sorted_dates) - 1)
    exit_bar = bars_by_date[sorted_dates[exit_pos]]
    exit_price = exit_bar.close

    # Also compute peak return during hold (max high)
    hold_dates = sorted_dates[entry_pos:exit_pos + 1]
    peak_price = max(bars_by_date[d].high for d in hold_dates) if hold_dates else exit_bar.high

    return {
        "ticker":         chosen.ticker,
        "strike":         chosen.strike,
        "expiration":     chosen.expiration,
        "entry_date":     entry_date,
        "entry_price":    entry_bar.close,
        "exit_date":      sorted_dates[exit_pos],
        "exit_price":     exit_price,
        "peak_price":     peak_price,
        "return_pct":     (exit_price - entry_bar.close) / entry_bar.close * 100,
        "peak_return_pct":(peak_price - entry_bar.close) / entry_bar.close * 100,
        "days_held":      exit_pos - entry_pos,
    }


def main():
    print(f"Backtest window: {WINDOW_START} → {WINDOW_END}")
    print(f"Setups to scan: {len(HIGH_EDGE_SETUPS)}\n")

    all_trades = []     # primary ATM weekly trades (the algo's recommendation)
    all_winners_grid = []   # for each winner, results across the strike/expiry grid
    setup_meta = {}     # setup_id -> historical pct_pos_5d, avg_5d, n

    for row in HIGH_EDGE_SETUPS:
        ticker, ma_label, n_hist, pct_pos, avg_5d, *_rest = row
        setup_id = f"{ticker}_{ma_label}"
        setup_meta[setup_id] = {"pct_pos": pct_pos, "avg_5d": avg_5d, "n": n_hist}

        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty:
            print(f"  [skip] {setup_id}: no daily data")
            continue

        touches = find_touches(daily, WINDOW_START, WINDOW_END)
        if not touches:
            continue
        print(f"  {setup_id}: {len(touches)} touch(es) in window")

        for t in touches:
            entry_date  = t["date"]
            entry_close = t["close"]

            # Algo-recommended trade: ATM weekly call
            primary = simulate_call_trade(ticker, entry_date, entry_close,
                                          days_forward=5, strike_offset=0,
                                          expiry_weeks_out=1)
            if "error" in primary:
                print(f"    {entry_date}: {primary['error']}")
                continue

            # Realized 5d underlying return for context
            exit_d, exit_close = get_forward_close(daily, entry_date, 5)
            underlying_5d_pct = ((exit_close - entry_close) / entry_close * 100
                                 if exit_close else None)

            trade = {
                "setup_id":         setup_id,
                "ticker":           ticker,
                "ma_label":         ma_label,
                "entry_date":       entry_date,
                "entry_underlying": entry_close,
                "ma_at_entry":      t["ma"],
                "dist_pct":         t["dist_at_close_pct"],
                "underlying_5d_pct": underlying_5d_pct,
                "hist_pct_pos":     pct_pos,
                "hist_avg_5d":      avg_5d,
                "hist_n":           n_hist,
                "weekday":          datetime.fromisoformat(entry_date).strftime("%A"),
                **primary,
            }
            all_trades.append(trade)

            # If winner, scan a grid: strikes ATM-2..ATM+3 × expiries 1w & 2w
            if primary.get("return_pct", 0) > 0:
                grid = []
                # Algo pick: ATM 1-week (already simulated)
                grid.append({**primary, "offset": 0, "weeks_out": 1, "is_algo_pick": True})
                # Alts: ITM-2 (-2), OTM+2 (+2), OTM+5 (+5), and ATM 2-week
                for offset, weeks_out, label in [
                    (-2, 1, "ITM-2"), (2, 1, "OTM+2"),
                    (5, 1, "OTM+5"), (0, 2, "ATM 2w"),
                ]:
                    alt = simulate_call_trade(ticker, entry_date, entry_close,
                                              days_forward=5, strike_offset=offset,
                                              expiry_weeks_out=weeks_out)
                    if "error" in alt:
                        continue
                    alt["offset"] = offset
                    alt["weeks_out"] = weeks_out
                    alt["label"] = label
                    alt["is_algo_pick"] = False
                    grid.append(alt)
                all_winners_grid.append({"trade": trade, "grid": grid})

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"Total trades simulated: {len(all_trades)}")
    if all_trades:
        winners = [t for t in all_trades if t["return_pct"] > 0]
        losers  = [t for t in all_trades if t["return_pct"] <= 0]
        avg_ret = sum(t["return_pct"] for t in all_trades) / len(all_trades)
        win_rate = len(winners) / len(all_trades) * 100
        # equal $1k per trade
        total_pnl = sum(t["return_pct"] / 100 * 1000 for t in all_trades)
        print(f"Win rate: {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
        print(f"Avg return per trade: {avg_ret:+.1f}%")
        print(f"Total P&L (equal $1k per trade): ${total_pnl:+.0f}")
        print(f"\nBest trade: {max(all_trades, key=lambda t: t['return_pct'])['return_pct']:+.1f}%")
        print(f"Worst trade: {min(all_trades, key=lambda t: t['return_pct'])['return_pct']:+.1f}%")

    out = {
        "window_start": WINDOW_START.isoformat(),
        "window_end":   WINDOW_END.isoformat(),
        "trades":       all_trades,
        "winners_grid": all_winners_grid,
    }
    out_path = Path("/tmp/backtest_4weeks.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
