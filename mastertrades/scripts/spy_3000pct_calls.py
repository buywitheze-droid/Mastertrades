"""Scan SPY history for days where an ATM 0DTE call OR put would have
returned >=3000% (31x). Two payoff models per day:

  CLOSE-payoff: intrinsic at the cash close
  PEAK-payoff:  intrinsic at the intraday high (calls) or low (puts)

Premium model for an ATM 0DTE bought at the open:
  premium ≈ spot × 0.0030  (≈ 30 bps — typical ATM 0DTE on SPY for a calm
                              session; uses 35 bps when prior-day TR > 1.5%
                              to reflect richer IV days)
This is an approximation since Polygon's free tier has no historical option
prices, but it is calibrated against the live ATM 0DTE quotes we observe in
src/lottery_scanner.py and gets us within ~25% on slow days, ~50% on
event days.

Run: python scripts/spy_3000pct_calls.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.scanner import fetch_or_load_daily

THRESHOLD = 30.0  # 3000% = 31x return; we keep >=30x to be inclusive

def premium_pct_for_day(prev_tr_pct: float) -> float:
    """ATM 0DTE premium as a fraction of spot. Inflate on high-vol prior days."""
    if prev_tr_pct >= 1.5:
        return 0.0035
    return 0.0030

def main():
    import sys as _sys
    ticker = _sys.argv[1] if len(_sys.argv) > 1 else "SPY"
    df = fetch_or_load_daily(ticker, history_years=10)  # ~10 years of daily bars
    print(f"\n[ ticker = {ticker} ]")
    df = df.copy()
    df.columns = [c.lower() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
    df["prev_tr_pct"] = (df["high"].shift(1) - df["low"].shift(1)) / df["close"].shift(1) * 100
    df = df.dropna(subset=["prev_tr_pct"])

    rows = []
    for d, r in df.iterrows():
        prem_pct = premium_pct_for_day(r["prev_tr_pct"])
        prem = r["open"] * prem_pct  # premium $ per share

        # ATM strike ≈ open
        call_close = max(r["close"] - r["open"], 0.0)
        call_peak  = max(r["high"]  - r["open"], 0.0)
        put_close  = max(r["open"]  - r["close"], 0.0)
        put_peak   = max(r["open"]  - r["low"],   0.0)

        rows.append({
            "date": (d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)),
            "open": round(r["open"], 2),
            "high": round(r["high"], 2),
            "low":  round(r["low"],  2),
            "close": round(r["close"], 2),
            "prem$": round(prem, 3),
            "call_close_x":   round(call_close / prem, 1),
            "call_peak_x":    round(call_peak  / prem, 1),
            "put_close_x":    round(put_close  / prem, 1),
            "put_peak_x":     round(put_peak   / prem, 1),
        })
    out = pd.DataFrame(rows)

    print(f"\n=== SPY 0DTE ATM ≥3000% (≥30x) — {len(df)} sessions scanned ===\n")

    # CLOSE-payoff hits (you have to actually hold to bell)
    close_hits = out[(out["call_close_x"] >= THRESHOLD) | (out["put_close_x"] >= THRESHOLD)]
    print(f"--- CLOSE-payoff hits (held to 4pm): {len(close_hits)} ---")
    for _, r in close_hits.iterrows():
        side = "CALL" if r["call_close_x"] >= THRESHOLD else "PUT"
        x = max(r["call_close_x"], r["put_close_x"])
        move_pct = (r["close"] - r["open"]) / r["open"] * 100
        print(f"  {r['date']}  {side:4s}  {x:6.1f}x  ({(x-1)*100:6.0f}%)  "
              f"open=${r['open']}  close=${r['close']}  intraday={move_pct:+.2f}%")

    # PEAK-payoff hits (sold at the intraday extreme)
    peak_hits = out[(out["call_peak_x"] >= THRESHOLD) | (out["put_peak_x"] >= THRESHOLD)]
    peak_only = peak_hits[~peak_hits["date"].isin(close_hits["date"])]
    print(f"\n--- PEAK-payoff hits (sold at intraday extreme, NOT held to close): {len(peak_only)} additional ---")
    for _, r in peak_only.iterrows():
        if r["call_peak_x"] >= THRESHOLD:
            side, x = "CALL", r["call_peak_x"]
            move = (r["high"] - r["open"]) / r["open"] * 100
            print(f"  {r['date']}  CALL  {x:6.1f}x  ({(x-1)*100:6.0f}%)  "
                  f"open=${r['open']}  high=${r['high']}  +{move:.2f}% intraday peak")
        else:
            side, x = "PUT", r["put_peak_x"]
            move = (r["open"] - r["low"]) / r["open"] * 100
            print(f"  {r['date']}  PUT   {x:6.1f}x  ({(x-1)*100:6.0f}%)  "
                  f"open=${r['open']}  low=${r['low']}   -{move:.2f}% intraday trough")

    # Summary
    total_hits = len(close_hits) + len(peak_only)
    n_call = sum(1 for _, r in pd.concat([close_hits, peak_only]).iterrows()
                 if max(r["call_close_x"], r["call_peak_x"]) >= THRESHOLD)
    n_put  = total_hits - n_call
    print(f"\n=== TOTAL: {total_hits} sessions ({n_call} calls, {n_put} puts) ===")
    print(f"Base rate: {total_hits}/{len(df)} = {total_hits/len(df)*100:.2f}% of trading days")

    out.to_csv("/tmp/spy_0dte_history.csv", index=False)
    print("\nFull per-day table saved to /tmp/spy_0dte_history.csv")

if __name__ == "__main__":
    main()
