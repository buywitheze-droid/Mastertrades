"""Per-ticker edge mining for the Gap Fill signal.

For each of SPY/QQQ/IWM/AAPL we walk forward over 1 year and grid-search:
  - min absolute gap %                    (0.20, 0.30, 0.40, 0.50, 0.70, 1.00)
  - direction filter                       (any, up only, down only)
  - weekday filter                         (any, mon, tue, wed, thu, fri)

For every cell we compute, on $1k notional/trade with no leverage:
  n, fill_rate, win_rate, avg_pct, total_pnl_$, profit_factor, max_drawdown
Output: top per-ticker configs (best risk-adjusted), and a recommended config
dict ready to paste into src/gap_per_ticker_config.py.
"""
from __future__ import annotations

import os
import sys
from itertools import product

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.scanner import fetch_or_load_daily
from src.gap_analysis import add_gap_features, gap_stats_by_bucket, today_gap_status

TICKERS = ("SPY", "QQQ", "IWM", "AAPL")
TEST_DAYS = 252
HISTORY_DAYS = 252
MIN_BUCKET_N = 5

GAP_THRESHOLDS = [0.0020, 0.0030, 0.0040, 0.0050, 0.0070, 0.0100]
DIRS = ["any", "up", "down"]
WEEKDAYS = ["any", "Mon", "Tue", "Wed", "Thu", "Fri"]
WD_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

PER_TRADE = 1000.0


def collect_signals(tkr: str) -> pd.DataFrame:
    """Walk-forward: emit every WATCH_FILL/NEAR_FILL signal with realised P&L."""
    daily = fetch_or_load_daily(tkr)
    if daily is None or len(daily) < TEST_DAYS + HISTORY_DAYS:
        return pd.DataFrame()
    feat = add_gap_features(daily).dropna(subset=["prev_close"])

    rows = []
    start_i = max(HISTORY_DAYS, len(feat) - TEST_DAYS)
    for i in range(start_i, len(feat)):
        hist = feat.iloc[:i]
        slc = feat.iloc[: i + 1]
        stats = gap_stats_by_bucket(hist)
        tg = today_gap_status(tkr, slc, stats)
        if tg.signal not in ("WATCH_FILL", "NEAR_FILL"):
            continue
        if tg.hist_n_similar < MIN_BUCKET_N:
            continue
        row = feat.iloc[i]
        open_px = float(row["Open"])
        prev_cl = float(row["prev_close"])
        close_px = float(row["Close"])
        filled = bool(row["gap_filled"])
        if tg.gap_dir == "up":
            realised = (open_px - (prev_cl if filled else close_px)) / open_px
        else:
            realised = ((prev_cl if filled else close_px) - open_px) / open_px
        rows.append({
            "date":     row.name.date(),
            "weekday":  WD_NAMES.get(row.name.weekday(), "?"),
            "signal":   tg.signal,
            "gap_dir":  tg.gap_dir,
            "abs_gap":  abs(tg.gap_pct),
            "filled":   filled,
            "realised": realised,
        })
    return pd.DataFrame(rows)


def evaluate_cell(df: pd.DataFrame, gap_thr: float, dir_f: str, wd_f: str) -> dict | None:
    f = df[df["abs_gap"] >= gap_thr]
    if dir_f != "any":
        f = f[f["gap_dir"] == dir_f]
    if wd_f != "any":
        f = f[f["weekday"] == wd_f]
    if len(f) < 10:
        return None
    pnl = f["realised"] * PER_TRADE
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "n":        len(f),
        "fill_rate":  round(f["filled"].mean() * 100, 1),
        "win_rate":   round((f["realised"] > 0).mean() * 100, 1),
        "avg_pct":    round(f["realised"].mean() * 100, 4),
        "total_$":    round(pnl.sum(), 2),
        "pf":         round(pf, 2),
        "max_dd_$":   round(dd, 2),
    }


def main() -> None:
    print("Collecting signal universe per ticker (walk-forward, 1 yr)...")
    by_ticker = {}
    for t in TICKERS:
        df = collect_signals(t)
        by_ticker[t] = df
        print(f"  {t}: {len(df)} raw signals")

    print("\n" + "=" * 92)
    print("PER-TICKER GRID SEARCH — top 5 configs by total $ P&L (n>=10)")
    print("=" * 92)

    recommended = {}

    for t, df in by_ticker.items():
        if df.empty:
            continue
        cells = []
        for thr, d, wd in product(GAP_THRESHOLDS, DIRS, WEEKDAYS):
            r = evaluate_cell(df, thr, d, wd)
            if r is None:
                continue
            r.update({"ticker": t, "min_gap%": thr * 100, "dir": d, "wd": wd})
            cells.append(r)
        if not cells:
            print(f"\n{t}: no cell met n>=10")
            continue
        c = pd.DataFrame(cells)
        # Sort by total $ then profit factor
        c = c.sort_values(["total_$", "pf"], ascending=False)

        print(f"\n── {t} ──────────────────────────────────────────────────────────────────")
        print(c.head(8).to_string(index=False))

        # Recommended = best total_$ with profit factor >= 1.2 and n >= 15
        viable = c[(c["pf"] >= 1.2) & (c["n"] >= 15) & (c["total_$"] > 0)]
        if viable.empty:
            print(f"  >> {t}: NO config meets edge bar (pf>=1.2, n>=15, P&L>0).  Recommend: DROP from gap-fill universe.")
            recommended[t] = None
        else:
            best = viable.iloc[0]
            print(f"  >> {t} RECOMMENDED: min_gap>={best['min_gap%']:.2f}%, dir={best['dir']}, wd={best['wd']}  "
                  f"({best['n']} trades, pf={best['pf']}, ${best['total_$']:.0f})")
            recommended[t] = {
                "min_gap_pct": float(best["min_gap%"]) / 100,
                "dir":         best["dir"],
                "weekday":     best["wd"],
                "n":           int(best["n"]),
                "win_rate":    float(best["win_rate"]),
                "avg_pct":     float(best["avg_pct"]),
                "total_$":     float(best["total_$"]),
                "pf":          float(best["pf"]),
            }

    print("\n" + "=" * 92)
    print("RECOMMENDED PER-TICKER CONFIG (paste into src/gap_per_ticker_config.py)")
    print("=" * 92)
    print("GAP_FILL_PER_TICKER = {")
    for t, cfg in recommended.items():
        if cfg is None:
            print(f'    "{t}": None,  # DROP — no validated edge')
        else:
            print(f'    "{t}": {{"min_gap_pct": {cfg["min_gap_pct"]:.4f}, '
                  f'"dir": "{cfg["dir"]}", "weekday": "{cfg["weekday"]}", '
                  f'"backtest": {{"n": {cfg["n"]}, "win_rate": {cfg["win_rate"]}, '
                  f'"avg_pct": {cfg["avg_pct"]:.4f}, "pf": {cfg["pf"]}, "pnl_per_1k": {cfg["total_$"]:.0f}}}}},')
    print("}")


if __name__ == "__main__":
    main()
