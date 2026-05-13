"""Final entry-depth sweep — NO stop, OTM+$5 fixed.

Confirms the optimal MA-discount depth without stop-loss confounding.
Tests: 0.10%, 0.15%, 0.25%, 0.35%, 0.50%, 0.65%, 0.80%, 1.00%
Fill window: 5 trading days. Hold to expiry. OTM+$5 1-week call.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_smart_entries import (
    load_setup_daily_with_ma, find_touches, simulate_limit_fill,
    WINDOW_START, WINDOW_END,
)
from backtest_stop_variants import fetch_option_trade, apply_stop
from src.ma_setups_universe import HIGH_EDGE_SETUPS


DEPTHS = [0.0010, 0.0015, 0.0025, 0.0035, 0.0050, 0.0065, 0.0080, 0.0100]


def main():
    print(f"Clean depth sweep (no stop, OTM+$5): {WINDOW_START} → {WINDOW_END}\n")

    all_signals = []
    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, *_ = setup
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty: continue
        for t in find_touches(daily, WINDOW_START, WINDOW_END):
            all_signals.append({"ticker": ticker, "setup_id": f"{ticker}_{ma_label}",
                                "daily": daily, "touch": t})
    print(f"Total signals: {len(all_signals)}\n")

    print(f"{'Depth':<10} {'Fills':>6} {'FillRt':>7} {'Win%':>6} {'AvgRet%':>9} "
          f"{'Median%':>9} {'Best%':>7} {'Worst%':>8} {'Total P&L':>11} {'$/fill':>9}")

    rows = []
    for d in DEPTHS:
        trades = []
        for sig in all_signals:
            t = sig["touch"]
            limit_px = t["ma"] * (1 - d)
            fill_date, fill_px = simulate_limit_fill(sig["daily"], t["ts"], limit_px, 5)
            if fill_date is None: continue
            res = fetch_option_trade(sig["ticker"], fill_date, fill_px, strike_offset=5.0)
            if "error" in res: continue
            trades.append(res)

        rets = [apply_stop(tr, "none") for tr in trades]
        n = len(trades)
        if n == 0:
            print(f"  -{d*100:.2f}%   {'(no fills)':>40}")
            continue
        wins = sum(1 for r in rets if r > 0)
        avg = sum(rets) / n
        med = sorted(rets)[n // 2]
        best = max(rets); worst = min(rets)
        pnl = sum(r/100*1000 for r in rets)
        ppf = pnl / n
        fr = n / len(all_signals) * 100
        print(f"  -{d*100:.2f}%  {n:>6} {fr:>6.0f}% {wins/n*100:>5.0f}% "
              f"{avg:>+8.1f}% {med:>+8.1f}% {best:>+6.0f}% {worst:>+7.0f}% "
              f"  ${pnl:>+8,.0f}  ${ppf:>+7,.0f}")
        rows.append({"depth_pct": d*100, "fills": n, "fill_rate_pct": fr,
                     "win_rate_pct": wins/n*100, "avg_return_pct": avg,
                     "median_return_pct": med, "best_pct": best, "worst_pct": worst,
                     "pnl": pnl, "per_fill": ppf, "returns": rets})

    print(f"\n{'='*92}")
    print("RANKED BY TOTAL P&L")
    print(f"{'='*92}")
    rows.sort(key=lambda r: -r["pnl"])
    for r in rows:
        print(f"  -{r['depth_pct']:.2f}%  fills={r['fills']:>3}  win={r['win_rate_pct']:>3.0f}%  "
              f"avg={r['avg_return_pct']:>+6.0f}%  P&L=${r['pnl']:>+8,.0f}  "
              f"$/fill=${r['per_fill']:>+6,.0f}")

    print(f"\n{'='*92}")
    print("RANKED BY $/FILL (capital efficiency)")
    print(f"{'='*92}")
    rows.sort(key=lambda r: -r["per_fill"])
    for r in rows:
        print(f"  -{r['depth_pct']:.2f}%  fills={r['fills']:>3}  win={r['win_rate_pct']:>3.0f}%  "
              f"$/fill=${r['per_fill']:>+6,.0f}  total P&L=${r['pnl']:>+8,.0f}")

    Path("/tmp/depth_clean.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
