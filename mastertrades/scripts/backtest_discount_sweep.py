"""Sweep MA-discount levels to find the optimal limit-order entry depth.

Tests: MA, MA-0.25%, MA-0.5%, MA-0.75%, MA-1%, MA-1.5%, MA-2%, MA-3%
With fill windows of 3 and 5 trading days.

Reuses helpers from backtest_smart_entries.py.
"""
from __future__ import annotations
import json, sys
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_smart_entries import (
    load_setup_daily_with_ma, find_touches, simulate_limit_fill,
    buy_atm_call_on, WINDOW_START, WINDOW_END,
)
from src.ma_setups_universe import HIGH_EDGE_SETUPS

DISCOUNTS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
FILL_WINDOWS = [3, 5]


def main():
    print(f"Discount sweep: {WINDOW_START} → {WINDOW_END}\n")
    rows = []

    # Collect all touches once (per setup)
    all_signals = []
    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, *_ = setup
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty: continue
        touches = find_touches(daily, WINDOW_START, WINDOW_END)
        for t in touches:
            all_signals.append({"ticker": ticker, "ma_label": ma_label,
                                "setup_id": f"{ticker}_{ma_label}",
                                "daily": daily, "touch": t})
    print(f"Total signals across universe: {len(all_signals)}\n")

    # Baseline: market-on-close
    print("Computing baseline (market on close)...")
    base_pnl = 0.0; base_wins = 0; base_n = 0
    for sig in all_signals:
        t = sig["touch"]
        res = buy_atm_call_on(sig["ticker"], t["date"], t["close"], days_forward=5)
        if "error" in res: continue
        base_n += 1
        if res["return_pct"] > 0: base_wins += 1
        base_pnl += res["return_pct"] / 100 * 1000
        rows.append({"strategy": "S1_market", "discount_pct": 0.0, "fill_window": 0,
                     "setup_id": sig["setup_id"], "signal_date": t["date"],
                     "fill_date": t["date"], "fill_price": t["close"], **res})
    print(f"  Baseline: {base_n} fills, {base_wins}/{base_n} wins, ${base_pnl:+,.0f}\n")

    # Sweep discounts × fill windows
    for fw in FILL_WINDOWS:
        for d in DISCOUNTS:
            label = f"MA-{d*100:.2f}%_fill{fw}d"
            n_fill=0; wins=0; pnl=0.0; ret_sum=0.0
            for sig in all_signals:
                t = sig["touch"]
                ma = t["ma"]
                limit_px = ma * (1 - d)
                fill_date, fill_px = simulate_limit_fill(sig["daily"], t["ts"], limit_px, fw)
                if fill_date is None: continue
                res = buy_atm_call_on(sig["ticker"], fill_date, fill_px, days_forward=5)
                if "error" in res: continue
                n_fill += 1
                if res["return_pct"] > 0: wins += 1
                pnl += res["return_pct"] / 100 * 1000
                ret_sum += res["return_pct"]
                rows.append({"strategy": label, "discount_pct": d*100, "fill_window": fw,
                             "setup_id": sig["setup_id"], "signal_date": t["date"],
                             "fill_date": fill_date, "fill_price": fill_px, **res})
            fill_rate = n_fill / len(all_signals) * 100 if all_signals else 0
            win_rate = wins / n_fill * 100 if n_fill else 0
            avg_ret = ret_sum / n_fill if n_fill else 0
            print(f"  {label:<22} fills={n_fill:>3}/{len(all_signals)} ({fill_rate:>3.0f}%)  "
                  f"win={win_rate:>3.0f}%  avg={avg_ret:>+7.1f}%  P&L=${pnl:>+8,.0f}")

    # Sorted summary table
    print(f"\n{'='*92}")
    print("SORTED BY P&L (best edge per dollar deployed)")
    print(f"{'='*92}")
    print(f"{'Strategy':<22} {'Fills':>6} {'FillRt':>7} {'Win%':>6} {'AvgRet%':>9} {'P&L($1k/sig)':>14} {'P&L/fill':>10}")
    summary = {}
    for r in rows:
        s = r["strategy"]
        if s not in summary:
            summary[s] = {"n":0, "w":0, "ret":0.0, "pnl":0.0, "discount":r["discount_pct"], "fw":r["fill_window"]}
        summary[s]["n"] += 1
        if r["return_pct"] > 0: summary[s]["w"] += 1
        summary[s]["ret"] += r["return_pct"]
        summary[s]["pnl"] += r["return_pct"] / 100 * 1000
    for s, v in sorted(summary.items(), key=lambda x: -x[1]["pnl"]):
        n = v["n"]
        fr = n / len(all_signals) * 100
        wr = v["w"]/n*100 if n else 0
        ar = v["ret"]/n if n else 0
        ppf = v["pnl"]/n if n else 0
        print(f"  {s:<22} {n:>4} {fr:>6.0f}% {wr:>5.0f}% {ar:>+8.1f}% "
              f"  ${v['pnl']:>+10,.0f}  ${ppf:>+8,.0f}")

    Path("/tmp/discount_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved → /tmp/discount_sweep.json")


if __name__ == "__main__":
    main()
