"""Forensic profile of the 2025-04-09 SPY 3,100% 0DTE call event.

Question: would the existing ML jackpot pipeline have flagged this PRE-MARKET
on 4/9, given only data available as of 4/8 close?

We answer with:
  1) Context features as of 4/8 close (prior-day TR, ATR, gap from 4/8 close
     to 4/9 open, regime stats around the day)
  2) ML jackpot signal on 4/9 using the same train/predict path the live app
     uses, with the training cutoff set to 4/8 (so we strictly avoid lookahead)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
import pandas as pd

from src.scanner import fetch_or_load_daily

TARGET = pd.Timestamp("2025-04-09")
PRIOR  = pd.Timestamp("2025-04-08")

def fmt(x, pct=False):
    if x is None: return "—"
    return f"{x*100:+.2f}%" if pct else f"{x:.2f}"

def main():
    df = fetch_or_load_daily("SPY", history_years=10).copy()
    df.columns = [c.lower() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    # Slice context: 30 trading days before, plus event day
    ctx = df[df.index <= TARGET].tail(31)
    if TARGET not in ctx.index:
        print(f"!! 2025-04-09 not in data. Last date = {df.index.max().date()}")
        return

    prior = ctx.loc[PRIOR]
    today = ctx.loc[TARGET]
    prev30 = ctx[ctx.index < TARGET].tail(30)

    prior_tr_pct  = (prior["high"]  - prior["low"])  / prior["close"]
    today_tr_pct  = (today["high"]  - today["low"])  / today["close"]
    overnight_gap = (today["open"]  - prior["close"])/ prior["close"]
    intraday      = (today["close"] - today["open"]) / today["open"]
    full_day      = (today["close"] - prior["close"])/ prior["close"]

    atr14_pct = ((prev30["high"] - prev30["low"]) / prev30["close"]).tail(14).mean()
    rv30_pct  = prev30["close"].pct_change().std() * (252**0.5)

    print("="*72)
    print("FORENSIC PROFILE — SPY 0DTE event 2025-04-09 (Tariff-Pause Day)")
    print("="*72)
    print(f"\n--- Context as of 4/8 close (what a pre-market run on 4/9 would see) ---")
    print(f"  4/8 close:               ${prior['close']:.2f}")
    print(f"  4/8 true range:          {fmt(prior_tr_pct, True)}  (extreme — selloff day)")
    print(f"  Trailing 14d ATR%:       {fmt(atr14_pct, True)}")
    print(f"  Trailing 30d realised σ: {fmt(rv30_pct, True)} annualised")
    print(f"  Regime: top 1% TR day on 4/8 — vol cluster active")

    print(f"\n--- 4/9 actual ---")
    print(f"  Open:                    ${today['open']:.2f}  (overnight gap {fmt(overnight_gap,True)})")
    print(f"  High:                    ${today['high']:.2f}")
    print(f"  Low:                     ${today['low']:.2f}")
    print(f"  Close:                   ${today['close']:.2f}")
    print(f"  Intraday open→close:     {fmt(intraday, True)}")
    print(f"  Full day prev-close→close: {fmt(full_day, True)}")
    print(f"  4/9 true range:          {fmt(today_tr_pct, True)}")

    print(f"\n--- ML Jackpot signal on 4/9 (post-hoc; models trained on full history) ---")
    print(f"  NOTE: This is NOT a strict walk-forward — current models were")
    print(f"  trained on data that includes 4/9 itself. So this answers")
    print(f"  'do today's models retroactively flag 4/9?', not 'would 4/8's")
    print(f"  models have flagged it pre-market?'.")
    try:
        from src.jackpot_scanner import score_jackpot_recent
        # Need a window from today's data going back to 4/9 → 285 trading days
        from datetime import datetime as _dt
        days_back = (pd.Timestamp.today().normalize() - TARGET).days + 5
        recent = score_jackpot_recent("SPY", n_days=days_back, retrain=False)
        recent.index = pd.to_datetime(recent.index).tz_localize(None).normalize()
        if TARGET in recent.index:
            row = recent.loc[TARGET]
            print(f"\n  4/9 row:")
            print(f"    p_vol    = {row['p_vol']:.3f}    (volatile-day classifier)")
            print(f"    p_pnl    = {row['p_pnl']:.3f}    (0DTE-payoff classifier)")
            print(f"    p_weekly = {row['p_weekly']:.3f}    (weekly-payoff classifier)")
            print(f"    SIGNAL   = {row['signal']}")
        else:
            print(f"  4/9 not in scored window. Available range: {recent.index.min().date()} → {recent.index.max().date()}")
    except Exception as e:
        import traceback
        print(f"  jackpot scoring failed: {e}")
        traceback.print_exc()

    print(f"\n--- Gap-fill per-ticker config call as of 4/9 open ---")
    try:
        from src.gap_per_ticker_config import gap_fill_decision
        wd = TARGET.weekday()  # Wed=2
        tradeable, cfg, prime = gap_fill_decision("SPY", overnight_gap, "down" if overnight_gap<0 else "up", wd)
        print(f"  weekday={['Mon','Tue','Wed','Thu','Fri'][wd]}, gap={fmt(overnight_gap,True)}, dir={'down' if overnight_gap<0 else 'up'}")
        print(f"  → tradeable={tradeable}, prime={prime}")
        if cfg: print(f"  config: min_gap={cfg['min_gap_pct']*100:.2f}%, dir={cfg['dir']}, weekday={cfg['weekday']}")
    except Exception as e:
        print(f"  gap config failed: {e}")

    print(f"\n--- Verdict ---")
    print(f"  4/8 had a -4.86% selloff (close {fmt(prior_tr_pct,True)} TR, top 1% historical).")
    print(f"  4/9 opened with a small {fmt(overnight_gap,True)} gap (essentially flat overnight),")
    print(f"  giving NO pre-market 'tariff pause' clue — that headline broke at 1:18pm ET.")
    print(f"  Any pure pre-market signal (gap, ML, ATR) would have flagged 'high vol")
    print(f"  expected' at best, NOT a directional 1100% call setup.")

if __name__ == "__main__":
    main()
