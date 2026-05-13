"""Compare smart-entry strategies vs the algo's current "buy on touch close" rule.

For every TOUCHING signal in the last 4 weeks, simulate 5 entry strategies:

  S1 (current):  Market buy on touch-day close
  S2 (MA limit): Limit order at the MA value (set on touch day, valid 3 days)
  S3 (MA -0.5%): Limit at MA * 0.995 (deeper pullback)
  S4 (Fib 38%): Limit at 38.2% retracement of last 20-day swing
  S5 (Fib 62%): Limit at 61.8% retracement (deepest, lowest fill rate)

For each, fetch real Polygon weekly call prices on the actual fill day.
Measure: fill rate, entry-price discount vs S1, option P&L over 5 trading days.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

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
WINDOW_START = TODAY - timedelta(days=35)
WINDOW_END   = TODAY - timedelta(days=10)   # need 5d forward + 3d for fill window
TOUCH_PCT = 0.006
FILL_LOOKAHEAD_DAYS = 3   # limit order valid for N trading days after signal
SWING_LOOKBACK = 20       # bars to compute swing high/low for Fib


def load_setup_daily_with_ma(ticker: str, ma_label: str) -> pd.DataFrame:
    daily = fetch_or_load_daily(ticker, data_dir=DATA_DIR, refresh=False)
    if daily is None or daily.empty:
        return pd.DataFrame()
    daily = daily.sort_index()
    daily.index = pd.to_datetime(daily.index)
    mas = compute_weekly_mas(daily)
    if ma_label not in mas:
        return pd.DataFrame()
    daily = daily.copy()
    daily["ma_value"] = mas[ma_label].reindex(daily.index, method="ffill")
    return daily


def find_touches(daily: pd.DataFrame, start: date, end: date) -> list[dict]:
    out = []
    for ts, row in daily.iterrows():
        d = ts.date()
        if d < start or d > end:
            continue
        ma = row.get("ma_value")
        if pd.isna(ma) or ma <= 0:
            continue
        if row["Low"] <= ma * (1 + TOUCH_PCT) and row["Close"] >= ma * (1 - TOUCH_PCT):
            out.append({
                "ts": ts, "date": d.isoformat(),
                "low": float(row["Low"]), "high": float(row["High"]),
                "close": float(row["Close"]), "ma": float(ma),
            })
    return out


def simulate_limit_fill(daily: pd.DataFrame, signal_ts: pd.Timestamp,
                        limit_price: float, max_days: int) -> tuple[str | None, float | None]:
    """If price's Low <= limit_price within next max_days bars (inclusive of signal day),
    return (fill_date_iso, fill_price=limit). Otherwise (None, None)."""
    idx = daily.index.get_loc(signal_ts) if signal_ts in daily.index else None
    if idx is None:
        return None, None
    end_idx = min(idx + max_days, len(daily) - 1)
    for i in range(idx, end_idx + 1):
        row = daily.iloc[i]
        if row["Low"] <= limit_price:
            # Fill at limit (assume sufficient liquidity), or at Open if Open already below limit
            fill = min(limit_price, float(row["Open"]))
            return daily.index[i].date().isoformat(), float(fill)
    return None, None


def buy_atm_call_on(underlying: str, fill_date: str, fill_price: float,
                    days_forward: int = 5) -> dict:
    """Buy ATM 1-week call on fill_date at fill_price; sell after days_forward sessions."""
    fd = datetime.fromisoformat(fill_date).date()
    target_friday = next_friday(fd, min_days_out=3)
    exp_lo = (target_friday - timedelta(days=2)).isoformat()
    exp_hi = (target_friday + timedelta(days=2)).isoformat()

    target_strike = fill_price
    half_band = max(3.0, fill_price * 0.05)
    try:
        contracts = list_expired_contracts(
            underlying=underlying, expiration_gte=exp_lo, expiration_lte=exp_hi,
            contract_type="call",
            strike_lo=target_strike - half_band, strike_hi=target_strike + half_band,
        )
    except Exception as e:
        return {"error": f"contract list: {e}"}
    if not contracts:
        return {"error": "no contracts"}
    contracts.sort(key=lambda c: (abs(c.strike - target_strike),
                                   abs((datetime.fromisoformat(c.expiration).date()
                                        - target_friday).days)))
    chosen = contracts[0]

    end_window = (fd + timedelta(days=days_forward + 7)).isoformat()
    try:
        bars = fetch_option_daily_bars(chosen.ticker, fill_date, end_window)
    except Exception as e:
        return {"error": f"bars: {e}"}
    if not bars:
        return {"error": "no bars"}
    by_d = {b.date: b for b in bars}
    if fill_date not in by_d:
        return {"error": f"no entry bar on {fill_date}"}
    sorted_d = sorted(by_d.keys())
    epos = sorted_d.index(fill_date)
    xpos = min(epos + days_forward, len(sorted_d) - 1)
    entry_px = by_d[fill_date].close
    if entry_px <= 0:
        return {"error": "zero entry"}
    exit_px = by_d[sorted_d[xpos]].close
    peak_px = max(by_d[d].high for d in sorted_d[epos:xpos + 1])
    return {
        "strike": chosen.strike, "expiration": chosen.expiration,
        "entry_date": fill_date, "entry_price": entry_px,
        "exit_date": sorted_d[xpos], "exit_price": exit_px,
        "return_pct": (exit_px - entry_px) / entry_px * 100,
        "peak_return_pct": (peak_px - entry_px) / entry_px * 100,
    }


def main():
    print(f"Smart-entry backtest window: {WINDOW_START} → {WINDOW_END}\n")

    rows = []   # one row per (signal, strategy)

    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, n_hist, pct_pos, avg_5d, *_ = setup
        setup_id = f"{ticker}_{ma_label}"
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty:
            continue
        touches = find_touches(daily, WINDOW_START, WINDOW_END)
        if not touches:
            continue
        print(f"  {setup_id}: {len(touches)} touch(es)")
        for t in touches:
            ts = t["ts"]
            ma = t["ma"]
            close = t["close"]

            # Compute swing high/low over prior SWING_LOOKBACK bars
            sig_idx = daily.index.get_loc(ts)
            window_start = max(0, sig_idx - SWING_LOOKBACK)
            swing = daily.iloc[window_start:sig_idx + 1]
            swing_high = float(swing["High"].max())
            swing_low_so_far = float(swing["Low"].min())
            # Fib levels: retracements DOWN from swing_high.
            #   higher level = shallower pullback = easier fill
            fib_38 = swing_high - (swing_high - swing_low_so_far) * 0.382
            fib_62 = swing_high - (swing_high - swing_low_so_far) * 0.618

            strategies = [
                ("S1_market",   close),                # buy on signal close
                ("S2_MA",       ma),                   # limit at MA
                ("S3_MA_-0.5%", ma * 0.995),           # 0.5% below MA
                ("S4_Fib_38",   fib_38),               # 38.2% retracement
                ("S5_Fib_62",   fib_62),               # 61.8% retracement
            ]
            for name, limit_px in strategies:
                if name == "S1_market":
                    fill_date = t["date"]
                    fill_px = close
                else:
                    fill_date, fill_px = simulate_limit_fill(daily, ts, limit_px, FILL_LOOKAHEAD_DAYS)
                row = {
                    "setup_id": setup_id, "signal_date": t["date"],
                    "strategy": name, "limit_price": float(limit_px),
                    "fill_date": fill_date, "fill_price": fill_px,
                    "filled": fill_date is not None,
                    "discount_vs_market_pct": (
                        (close - fill_px) / close * 100 if fill_px else None
                    ),
                }
                if fill_date is not None:
                    res = buy_atm_call_on(ticker, fill_date, fill_px, days_forward=5)
                    if "error" in res:
                        row["error"] = res["error"]
                    else:
                        row.update(res)
                rows.append(row)

    # ── Aggregate per strategy ──
    print(f"\n{'='*78}")
    print(f"SMART-ENTRY COMPARISON — same {len([r for r in rows if r['strategy']=='S1_market'])} signals, 5 entry rules")
    print(f"{'='*78}")
    print(f"{'Strategy':<14} {'Fills':>6} {'FillRt':>8} {'AvgDisc%':>10} {'Win%':>6} {'AvgRet%':>9} {'TotalP&L':>10}")
    by_strat = {}
    for r in rows:
        s = r["strategy"]
        by_strat.setdefault(s, []).append(r)
    for s in ["S1_market", "S2_MA", "S3_MA_-0.5%", "S4_Fib_38", "S5_Fib_62"]:
        recs = by_strat.get(s, [])
        if not recs:
            continue
        n_total = len(recs)
        filled = [r for r in recs if r["filled"] and "return_pct" in r]
        n_fill = len(filled)
        fill_rate = n_fill / n_total * 100 if n_total else 0
        avg_disc = (sum(r["discount_vs_market_pct"] for r in filled if r.get("discount_vs_market_pct") is not None)
                    / max(n_fill, 1))
        wins = [r for r in filled if r["return_pct"] > 0]
        win_rate = len(wins) / n_fill * 100 if n_fill else 0
        avg_ret = sum(r["return_pct"] for r in filled) / max(n_fill, 1)
        # P&L if we deploy $1k per filled signal
        pnl = sum(r["return_pct"] / 100 * 1000 for r in filled)
        print(f"{s:<14} {n_fill:>6} {fill_rate:>7.0f}% {avg_disc:>+9.2f}% {win_rate:>5.0f}% "
              f"{avg_ret:>+8.1f}% {'$'+format(pnl,'+,.0f'):>10}")

    out = {"window_start": WINDOW_START.isoformat(),
           "window_end":   WINDOW_END.isoformat(),
           "rows":         rows}
    Path("/tmp/smart_entries.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved → /tmp/smart_entries.json")


if __name__ == "__main__":
    main()
