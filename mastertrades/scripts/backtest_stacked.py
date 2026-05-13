"""Stack all three edges and measure compounded improvement.

Edges:
  1. SMART ENTRY: limit at MA × (1 - discount), valid 5 trading days
  2. STRIKE OFFSET: ATM, OTM+$2, or OTM+$5
  3. STOP LOSS: intraday -50% (checked against option's daily Low)

Strategies compared (all on same 31-signal window):
  BASE     : market close, ATM, no stop                (current algo)
  STOP     : market close, ATM, -50% stop
  SMART    : MA-0.25%, ATM, no stop
  OTM5     : market close, OTM+$5, no stop
  STACK_A  : MA-0.25%, ATM,    -50% stop
  STACK_B  : MA-0.25%, OTM+$5, no stop
  STACK_C  : MA-0.25%, OTM+$5, -50% stop                (full stack)
  STACK_D  : MA-0.50%, OTM+$5, -50% stop                (high-conviction stack)
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


def buy_call(underlying: str, fill_date: str, fill_price: float,
             strike_offset: float = 0.0, days_forward: int = 5,
             stop_loss_pct: float | None = None,
             expiry_weeks_out: int = 1) -> dict:
    """Buy a call with given strike offset; simulate intraday stop using Low.

    stop_loss_pct: e.g. 50 means exit if option's Low ≤ entry * 0.5 on any day.
    """
    fd = datetime.fromisoformat(fill_date).date()
    target_friday = next_friday(fd, min_days_out=3) + timedelta(days=7 * (expiry_weeks_out - 1))
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

    stop_threshold = (entry_px * (1 - stop_loss_pct / 100)) if stop_loss_pct else None
    stopped = False; stop_date = None; stop_px = None
    peak_px = entry_px
    final_px = by_d[sorted_d[xpos]].close
    final_date = sorted_d[xpos]
    for i in range(epos, xpos + 1):
        b = by_d[sorted_d[i]]
        peak_px = max(peak_px, b.high)
        # End-of-day stop: check Close, not intraday Low (avoids whipsaw stops on
        # options that wick down intraday but recover by close)
        if stop_threshold is not None and b.close <= stop_threshold and i > epos:
            stopped = True
            stop_date = sorted_d[i]
            stop_px = b.close
            break
    if stopped:
        exit_px = stop_px; exit_date = stop_date
    else:
        exit_px = final_px; exit_date = final_date

    return {
        "strike": chosen.strike, "strike_offset": strike_offset,
        "expiration": chosen.expiration,
        "entry_date": fill_date, "entry_price": entry_px,
        "exit_date": exit_date, "exit_price": exit_px,
        "stopped": stopped,
        "return_pct": (exit_px - entry_px) / entry_px * 100,
        "peak_return_pct": (peak_px - entry_px) / entry_px * 100,
    }


# Strategy specs: (name, ma_discount, fill_window, strike_offset, stop_loss_pct)
STRATEGIES = [
    ("BASE",    None,   0, 0.0, None),
    ("STOP",    None,   0, 0.0, 50.0),
    ("STACK_A", 0.0025, 5, 0.0, 50.0),
    ("STACK_C", 0.0025, 5, 5.0, 50.0),
    ("STACK_D", 0.0050, 5, 5.0, 50.0),
]


def main():
    print(f"Stacked backtest: {WINDOW_START} → {WINDOW_END}\n")

    # Collect all signals
    all_signals = []
    for setup in HIGH_EDGE_SETUPS:
        ticker, ma_label, *_ = setup
        daily = load_setup_daily_with_ma(ticker, ma_label)
        if daily.empty: continue
        for t in find_touches(daily, WINDOW_START, WINDOW_END):
            all_signals.append({"ticker": ticker, "ma_label": ma_label,
                                "setup_id": f"{ticker}_{ma_label}",
                                "daily": daily, "touch": t})
    print(f"Total signals: {len(all_signals)}\n")

    summary = {}
    all_rows = []

    for name, disc, fw, off, sl in STRATEGIES:
        print(f"Running {name} (disc={disc}, fw={fw}, offset=${off}, stop={sl}%)...")
        n_fill=0; wins=0; ret_sum=0.0; pnl=0.0; stops=0
        max_loss=0; max_win=0
        for sig in all_signals:
            t = sig["touch"]
            # Determine fill date / price
            if disc is None:
                fill_date = t["date"]; fill_px = t["close"]
            else:
                limit_px = t["ma"] * (1 - disc)
                fill_date, fill_px = simulate_limit_fill(sig["daily"], t["ts"], limit_px, fw)
                if fill_date is None: continue
            res = buy_call(sig["ticker"], fill_date, fill_px,
                           strike_offset=off, days_forward=5, stop_loss_pct=sl)
            if "error" in res: continue
            n_fill += 1
            if res["return_pct"] > 0: wins += 1
            if res.get("stopped"): stops += 1
            ret_sum += res["return_pct"]
            pnl += res["return_pct"] / 100 * 1000
            max_loss = min(max_loss, res["return_pct"])
            max_win = max(max_win, res["return_pct"])
            all_rows.append({"strategy": name, "setup_id": sig["setup_id"],
                             "signal_date": t["date"], **res})
        summary[name] = {
            "n_signals": len(all_signals), "n_fills": n_fill,
            "wins": wins, "stops": stops,
            "avg_ret": ret_sum / max(n_fill, 1),
            "pnl": pnl, "max_win": max_win, "max_loss": max_loss,
        }

    # ── Print comparison table ──
    print(f"\n{'='*100}")
    print(f"STACKED EDGE COMPARISON — {len(all_signals)} signals, $1k per fill")
    print(f"{'='*100}")
    print(f"{'Strategy':<10} {'Fills':>6} {'FillRt':>7} {'Win%':>6} {'Stops':>6} "
          f"{'AvgRet%':>9} {'BestTr':>8} {'WorstTr':>9} {'Total P&L':>11} {'$/fill':>9}")
    for name, _, _, _, _ in STRATEGIES:
        v = summary[name]
        fr = v["n_fills"] / v["n_signals"] * 100
        wr = v["wins"]/v["n_fills"]*100 if v["n_fills"] else 0
        ppf = v["pnl"]/v["n_fills"] if v["n_fills"] else 0
        print(f"{name:<10} {v['n_fills']:>6} {fr:>6.0f}% {wr:>5.0f}% {v['stops']:>6} "
              f"{v['avg_ret']:>+8.1f}% {v['max_win']:>+7.0f}% {v['max_loss']:>+8.0f}% "
              f"  ${v['pnl']:>+8,.0f}  ${ppf:>+7,.0f}")

    # ── Marginal contribution of each edge ──
    print(f"\n{'='*100}")
    print("MARGINAL EDGE CONTRIBUTION  (each row = added benefit on top of BASE)")
    print(f"{'='*100}")
    base_pnl = summary["BASE"]["pnl"]
    base_avg = summary["BASE"]["avg_ret"]
    for name in [n for n, *_ in STRATEGIES if n != "BASE"]:
        v = summary[name]
        d_pnl = v["pnl"] - base_pnl
        d_avg = v["avg_ret"] - base_avg
        d_pnl_pct = d_pnl / base_pnl * 100 if base_pnl else 0
        print(f"  {name:<10}  ΔP&L=${d_pnl:>+8,.0f}  ({d_pnl_pct:>+5.0f}% vs BASE)   "
              f"Δavg/trade={d_avg:>+6.1f}pp")

    Path("/tmp/stacked_backtest.json").write_text(json.dumps(
        {"summary": summary, "rows": all_rows}, indent=2, default=str))
    print(f"\nSaved → /tmp/stacked_backtest.json")


if __name__ == "__main__":
    main()
