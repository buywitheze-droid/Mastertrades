"""Chunked backtest — tests 3 key depths on a custom date window.

Usage: python scripts/backtest_chunk.py YYYY-MM-DD YYYY-MM-DD chunk_label

Tests only the empirically-informed depths (−0.10%, −0.50%, −1.00%) so each
chunk fits in ~90 seconds. Run multiple chunks in sequence, then aggregate.
"""
from __future__ import annotations
import json, sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_smart_entries import (
    load_setup_daily_with_ma, find_touches, simulate_limit_fill,
)
from backtest_stop_variants import fetch_option_trade, apply_stop
from src.ma_setups_universe import HIGH_EDGE_SETUPS

DEPTHS = [0.0010, 0.0050, 0.0100]   # -0.10%, -0.50%, -1.00%


def run(window_start: date, window_end: date, label: str) -> dict:
    print(f"\n=== Chunk {label}: {window_start} → {window_end} ===")
    all_signals = []
    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, *_ = setup
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty: continue
        for t in find_touches(daily, window_start, window_end):
            all_signals.append({"ticker": ticker, "setup_id": f"{ticker}_{ma_label}",
                                "daily": daily, "touch": t})
    print(f"  Signals: {len(all_signals)}")

    out = {"label": label, "window_start": window_start.isoformat(),
           "window_end": window_end.isoformat(),
           "n_signals": len(all_signals), "depths": {}}

    for d in DEPTHS:
        trades = []
        for sig in all_signals:
            t = sig["touch"]
            limit_px = t["ma"] * (1 - d)
            fill_date, fill_px = simulate_limit_fill(sig["daily"], t["ts"], limit_px, 5)
            if fill_date is None: continue
            res = fetch_option_trade(sig["ticker"], fill_date, fill_px, strike_offset=5.0)
            if "error" in res: continue
            trades.append({
                "ticker": sig["ticker"], "setup_id": sig["setup_id"],
                "signal_date": t["date"], "fill_date": res["entry_date"],
                "strike": res["strike"], "return_pct": apply_stop(res, "none"),
            })
        n = len(trades)
        wins = sum(1 for tr in trades if tr["return_pct"] > 0)
        avg = sum(tr["return_pct"] for tr in trades) / max(n, 1)
        pnl = sum(tr["return_pct"]/100*1000 for tr in trades)
        depth_key = f"-{d*100:.2f}%"
        out["depths"][depth_key] = {
            "fills": n, "wins": wins, "avg_ret_pct": avg, "pnl": pnl,
            "trades": trades,
        }
        print(f"  {depth_key}: fills={n}/{len(all_signals)}  win={wins}/{n}  "
              f"avg={avg:+.0f}%  P&L=${pnl:+,.0f}")
    return out


def main():
    if len(sys.argv) != 4:
        print("Usage: backtest_chunk.py YYYY-MM-DD YYYY-MM-DD chunk_label")
        sys.exit(1)
    ws = datetime.fromisoformat(sys.argv[1]).date()
    we = datetime.fromisoformat(sys.argv[2]).date()
    label = sys.argv[3]
    result = run(ws, we, label)
    out_path = Path(f"/tmp/chunk_{label}.json")
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
