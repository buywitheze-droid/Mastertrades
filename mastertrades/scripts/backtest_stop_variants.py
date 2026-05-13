"""Test stop-loss variants on top of smart-entry + OTM+$5 strikes.

For each (smart-entry depth) combination we fetch option bars ONCE per fill,
then apply 4 different stop rules in post: none / -50% EOD / -75% EOD / trailing.

Strategies (all use OTM+$5 calls, 1-week expiry, 5d hold):
  Group A: MA-0.5% entry (fewer fills, deepest pullback)
    A_none    : no stop                                  (= STACK_D-no-stop)
    A_50      : -50% EOD stop                            (= STACK_D from prior run)
    A_75      : -75% EOD stop                            (looser)
    A_trail   : trailing -40% from peak                  (lock in winners)

  Group B: MA-0.25% entry (more fills)
    B_none    : no stop
    B_50      : -50% EOD stop
    B_75      : -75% EOD stop
    B_trail   : trailing -40% from peak
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
    WINDOW_START, WINDOW_END,
)
from src.ma_setups_universe import HIGH_EDGE_SETUPS
from src.options_history import (
    list_expired_contracts, fetch_option_daily_bars, next_friday,
)


def fetch_option_trade(underlying: str, fill_date: str, fill_price: float,
                       strike_offset: float = 5.0,
                       days_forward: int = 5) -> dict:
    """Fetch option contract + bars, return raw bar series (no stop applied yet)."""
    fd = datetime.fromisoformat(fill_date).date()
    target_friday = next_friday(fd, min_days_out=3)
    exp_lo = (target_friday - timedelta(days=2)).isoformat()
    exp_hi = (target_friday + timedelta(days=2)).isoformat()
    target_strike = fill_price + strike_offset
    half_band = max(3.0, fill_price * 0.05)
    try:
        contracts = list_expired_contracts(
            underlying=underlying, expiration_gte=exp_lo, expiration_lte=exp_hi,
            contract_type="call",
            strike_lo=target_strike - half_band, strike_hi=target_strike + half_band,
        )
    except Exception as e:
        return {"error": f"contracts: {e}"}
    if not contracts: return {"error": "no contracts"}
    contracts.sort(key=lambda c: (abs(c.strike - target_strike),
                                   abs((datetime.fromisoformat(c.expiration).date()
                                        - target_friday).days)))
    chosen = contracts[0]
    end_window = (fd + timedelta(days=days_forward + 7)).isoformat()
    try:
        bars = fetch_option_daily_bars(chosen.ticker, fill_date, end_window)
    except Exception as e:
        return {"error": f"bars: {e}"}
    if not bars: return {"error": "no bars"}
    by_d = {b.date: b for b in bars}
    if fill_date not in by_d: return {"error": f"no entry on {fill_date}"}
    sorted_d = sorted(by_d.keys())
    epos = sorted_d.index(fill_date)
    xpos = min(epos + days_forward, len(sorted_d) - 1)
    entry_px = by_d[fill_date].close
    if entry_px <= 0: return {"error": "zero entry"}
    return {
        "strike": chosen.strike, "expiration": chosen.expiration,
        "entry_date": fill_date, "entry_price": entry_px,
        "bars": [(d, by_d[d].close, by_d[d].high, by_d[d].low)
                 for d in sorted_d[epos:xpos + 1]],
    }


def apply_stop(trade: dict, stop_type: str) -> float:
    """stop_type ∈ {'none', '50_eod', '75_eod', 'trail40'}.
    Returns final return_pct after applying stop."""
    entry = trade["entry_price"]
    bars = trade["bars"]   # list of (date, close, high, low)
    if not bars: return 0.0

    if stop_type == "none":
        return (bars[-1][1] - entry) / entry * 100

    if stop_type == "50_eod":
        thresh = entry * 0.5
        for i, (_, close, _, _) in enumerate(bars):
            if i == 0: continue   # don't stop on entry day
            if close <= thresh:
                return (close - entry) / entry * 100
        return (bars[-1][1] - entry) / entry * 100

    if stop_type == "75_eod":
        thresh = entry * 0.25
        for i, (_, close, _, _) in enumerate(bars):
            if i == 0: continue
            if close <= thresh:
                return (close - entry) / entry * 100
        return (bars[-1][1] - entry) / entry * 100

    if stop_type == "trail40":
        # Lock in profits: once trade is profitable, exit if it gives back 40% from peak close
        peak = entry
        for i, (_, close, _, _) in enumerate(bars):
            if i == 0:
                peak = close
                continue
            peak = max(peak, close)
            # Only activate trailing stop once we've been up at least 50%
            if peak >= entry * 1.5 and close <= peak * 0.6:
                return (close - entry) / entry * 100
        return (bars[-1][1] - entry) / entry * 100

    return 0.0


def main():
    print(f"Stop-variant backtest: {WINDOW_START} → {WINDOW_END}\n")

    all_signals = []
    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, *_ = setup
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty: continue
        for t in find_touches(daily, WINDOW_START, WINDOW_END):
            all_signals.append({"ticker": ticker, "setup_id": f"{ticker}_{ma_label}",
                                "daily": daily, "touch": t})
    print(f"Total signals: {len(all_signals)}\n")

    # Two entry depths: -0.5% (deep) and -0.25% (moderate)
    groups = [("A_deep_-0.5%", 0.005), ("B_med_-0.25%", 0.0025)]
    stop_types = ["none", "50_eod", "75_eod", "trail40"]
    results = {}   # (group_label, stop_type) -> list of returns

    for label, disc in groups:
        print(f"\n=== Entry: MA × (1 - {disc*100}%), 5-day fill window ===")
        trades = []
        for sig in all_signals:
            t = sig["touch"]
            limit_px = t["ma"] * (1 - disc)
            fill_date, fill_px = simulate_limit_fill(sig["daily"], t["ts"], limit_px, 5)
            if fill_date is None: continue
            res = fetch_option_trade(sig["ticker"], fill_date, fill_px, strike_offset=5.0)
            if "error" in res: continue
            trades.append(res)
        print(f"  Fills: {len(trades)} / {len(all_signals)}")
        for stop in stop_types:
            rets = [apply_stop(t, stop) for t in trades]
            results[(label, stop)] = rets

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"STOP-LOSS VARIANT COMPARISON (OTM+$5, 5d hold) — does the stop add or subtract value?")
    print(f"{'='*100}")
    print(f"{'Strategy':<28} {'Fills':>6} {'Win%':>6} {'AvgRet%':>9} {'Median%':>9} "
          f"{'Best%':>7} {'Worst%':>8} {'Total P&L':>11}")
    for label, _ in groups:
        for stop in stop_types:
            rets = results[(label, stop)]
            n = len(rets)
            if n == 0: continue
            wins = sum(1 for r in rets if r > 0)
            avg = sum(rets) / n
            med = sorted(rets)[n // 2]
            best = max(rets); worst = min(rets)
            pnl = sum(r / 100 * 1000 for r in rets)
            name = f"{label} + {stop}"
            print(f"{name:<28} {n:>6} {wins/n*100:>5.0f}% {avg:>+8.1f}% {med:>+8.1f}% "
                  f"{best:>+6.0f}% {worst:>+7.0f}%   ${pnl:>+8,.0f}")

    # ── Best-of comparison ──
    print(f"\n{'='*100}")
    print(f"VERDICT: Does the stop loss help when stacked with smart entry + OTM+$5?")
    print(f"{'='*100}")
    for label, _ in groups:
        none_pnl = sum(r/100*1000 for r in results[(label, 'none')])
        print(f"\n  Group {label}:")
        for stop in ["50_eod", "75_eod", "trail40"]:
            stop_pnl = sum(r/100*1000 for r in results[(label, stop)])
            delta = stop_pnl - none_pnl
            pct = delta / none_pnl * 100 if none_pnl else 0
            verdict = "✓ ADDS VALUE" if delta > 0 else "✗ HURTS"
            print(f"    + {stop:<10}  P&L=${stop_pnl:>+8,.0f}  vs no-stop ${none_pnl:>+8,.0f}  "
                  f"Δ=${delta:>+7,.0f} ({pct:>+5.0f}%)  {verdict}")

    Path("/tmp/stop_variants.json").write_text(json.dumps(
        {k[0]+"_"+k[1]: v for k, v in results.items()}, indent=2, default=str))
    print(f"\nSaved → /tmp/stop_variants.json")


if __name__ == "__main__":
    main()
