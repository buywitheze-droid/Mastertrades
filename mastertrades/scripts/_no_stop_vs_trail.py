"""Did the 15% trail-stop kill any winners?

Compares three exit strategies head-to-head over the last 30 days:
  1. NO STOP — hold every contract to the close (matches user's current behavior)
  2. TRAIL  — exit at peak × 0.85 (or initial × 0.85 if no peak)
  3. PEAK   — sell at the absolute intraday high (theoretical max, unknowable in real time)

For each trade we classify:
  - "Stop helped"   : trail-exit > close-exit (the stop captured peak, close faded)
  - "Stop hurt"     : trail-exit < close-exit (the stop killed a winner)
  - "Stop neutral"  : both exits agree (no 15% drawdown happened)
  - "AMBIGUOUS"    : opt_low ≤ entry × 0.85 AND opt_high > entry × 1.15
                     — both events happened, OHLC alone can't tell us if the
                     drawdown happened BEFORE or AFTER the peak.
                     These are the trades where minute-level data is needed
                     to resolve. They get fetched in the next phase.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import numpy as np
import pandas as pd
from src.options_scanner import OptionContract, recommend_strikes

START_CASH    = 1_000.0
TRAIL_PCT     = 15.0
TODAY         = dt.date(2026, 5, 13)
LOOKBACK_DAYS = 30

# Use the realistic fill model (10-min latency, 60% of opening recovery)
ENTRY_LATENCY = 0.60

# ── Build pick list ─────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "per_source_gate_trades_ladder.csv")
picks = []
for (ticker, date), group in df.groupby(["ticker", "date"]):
    underlying_open = float(group["underlying_open"].iloc[0])
    underlying_low  = float(group["underlying_low"].iloc[0])
    chain = [
        OptionContract(
            ticker=row["contract"], contract_type="call",
            strike=float(row["strike"]), expiration=date,
            day_open=float(row["opt_open"]), day_high=float(row["opt_high"]),
            day_low=float(row["opt_low"]),  day_close=float(row["opt_close"]),
            day_volume=1000, implied_vol=0.30,
            delta=0.5, gamma=0.05, theta=-0.10, vega=0.05,
            open_interest=500,
        )
        for _, row in group.iterrows()
    ]
    recs = recommend_strikes(underlying_open, underlying_low, chain)
    if not recs:
        continue
    top = max(recs, key=lambda r: r.leverage_score)
    match = group[group["strike"] == top.strike].iloc[0]
    picks.append({
        "ticker": ticker, "date": pd.to_datetime(date).date(),
        "strike": top.strike, "contract": top.contract_ticker,
        "opt_open": float(match["opt_open"]),
        "opt_high": float(match["opt_high"]),
        "opt_low":  float(match["opt_low"]),
        "opt_close": float(match["opt_close"]),
    })
picks_df = pd.DataFrame(picks).sort_values("date").reset_index(drop=True)
cutoff = TODAY - dt.timedelta(days=LOOKBACK_DAYS)
month = picks_df[picks_df["date"] >= cutoff].copy().reset_index(drop=True)

# ── Compute fill price + 3 exit prices per trade ───────────────────────────
def trail_exit(entry, high, low, close, trail=TRAIL_PCT/100):
    if high > entry:
        peak_stop = high * (1 - trail)
        return max(close, peak_stop) if close < peak_stop else close
    init_stop = entry * (1 - trail)
    return init_stop if low <= init_stop else close

month["fill"]      = (month["opt_low"] + ENTRY_LATENCY *
                      (month["opt_open"] - month["opt_low"])).round(2)
month["fill"]      = month["fill"].clip(lower=0.01)
month["exit_close"] = month["opt_close"]                          # NO STOP
month["exit_trail"] = month.apply(
    lambda r: trail_exit(r["fill"], r["opt_high"], r["opt_low"], r["opt_close"]),
    axis=1)
month["exit_peak"]  = month["opt_high"]                          # PEAK SELL

month["pct_close"] = (month["exit_close"] / month["fill"] - 1) * 100
month["pct_trail"] = (month["exit_trail"] / month["fill"] - 1) * 100
month["pct_peak"]  = (month["exit_peak"]  / month["fill"] - 1) * 100

# Drawdown thresholds
trail = TRAIL_PCT / 100
month["dd_pct"]      = (month["opt_low"] / month["fill"] - 1) * 100
month["had_dd_15"]   = month["opt_low"] <= month["fill"] * (1 - trail)
month["had_peak_15"] = month["opt_high"] >= month["fill"] * (1 + trail)
month["ambiguous"]   = month["had_dd_15"] & month["had_peak_15"]

# Classify
def classify(r):
    if not r["had_dd_15"]:
        return "stop-neutral"
    if r["ambiguous"]:
        return "AMBIGUOUS"
    # Drawdown happened, no peak above 15% → stop fires from initial level.
    # If close > stop level, no-stop wins; if close < stop level, stop helps.
    init_stop = r["fill"] * (1 - trail)
    if r["opt_close"] > init_stop:
        return "stop-hurt"      # stop fired but option recovered above stop by close
    return "stop-helped"
month["category"] = month.apply(classify, axis=1)

# ── Print per-trade table ──────────────────────────────────────────────────
print(f"=== Per-trade comparison (last 30 days · {len(month)} trades · "
      f"15% trail · {ENTRY_LATENCY*100:.0f}% fill latency) ===\n")
print(f"  {'#':>2} {'date':<11} {'tkr':<4} {'fill':>5} {'low':>5} {'high':>5} "
      f"{'close':>5} {'dd%':>6} | {'no-stop':>8} {'trail':>8} {'peak':>8}  category")
print(f"  " + "-" * 110)
for i, r in month.iterrows():
    print(f"  {i+1:>2} {r['date']!s:<11} {r['ticker']:<4} "
          f"${r['fill']:>4.2f} ${r['opt_low']:>4.2f} ${r['opt_high']:>4.2f} "
          f"${r['opt_close']:>4.2f} {r['dd_pct']:>+5.0f}% | "
          f"{r['pct_close']:>+6.0f}% {r['pct_trail']:>+6.0f}% {r['pct_peak']:>+6.0f}%  "
          f"{r['category']}")

# ── Aggregate $ comparison at $500/trade (level playing field) ─────────────
POS_USD = 500
month["ct"] = (POS_USD // (month["fill"] * 100)).astype(int)
for col, src in [("pnl_close", "exit_close"), ("pnl_trail", "exit_trail"),
                  ("pnl_peak",  "exit_peak")]:
    month[col] = (month[src] - month["fill"]) * month["ct"] * 100

print(f"\n=== Aggregate $ comparison ($500/trade, 15% trail, 60% latency) ===\n")
totals = {
    "NO STOP (current — hold to close)": month["pnl_close"].sum(),
    "15% TRAIL-STOP (proposed)        ": month["pnl_trail"].sum(),
    "PEAK SELL (theoretical max)      ": month["pnl_peak"].sum(),
}
for name, t in totals.items():
    print(f"  {name}: ${t:>+11,.2f}")
print(f"\n  → Trail-stop {'GAINED' if totals['15% TRAIL-STOP (proposed)        '] > totals['NO STOP (current — hold to close)'] else 'COST'} "
      f"${abs(totals['15% TRAIL-STOP (proposed)        '] - totals['NO STOP (current — hold to close)']):,.0f} "
      f"vs no-stop over {len(month)} trades.")
print(f"  → Peak-sell would have earned ${totals['PEAK SELL (theoretical max)      '] - totals['15% TRAIL-STOP (proposed)        ']:,.0f} more than trail-stop "
      f"(unattainable in real time, shown as ceiling).")

# ── Category breakdown ─────────────────────────────────────────────────────
print(f"\n=== Category breakdown ({len(month)} trades) ===\n")
cat_counts = month["category"].value_counts()
for cat, n in cat_counts.items():
    sub = month[month["category"] == cat]
    diff = sub["pnl_close"].sum() - sub["pnl_trail"].sum()
    sign = "no-stop better" if diff > 0 else "trail-stop better" if diff < 0 else "tied"
    print(f"  {cat:<14} : {n:>2} trades · "
          f"no-stop ${sub['pnl_close'].sum():>+8,.0f} vs "
          f"trail ${sub['pnl_trail'].sum():>+8,.0f} "
          f"(Δ ${diff:>+7,.0f}, {sign})")

# ── Stop-hurt detail (the user's specific concern) ─────────────────────────
hurt = month[month["category"] == "stop-hurt"]
print(f"\n=== Trades where 'trail-stop killed a clear winner' (no-ambiguity cases) ===\n")
if hurt.empty:
    print(f"  None. Every trade with a confirmed 15% drawdown also closed near or below the stop level.")
else:
    print(f"  {len(hurt)} trades dropped 15%+ then recovered above the stop level by close.")
    for _, r in hurt.iterrows():
        loss_avoided = r["pnl_close"] - r["pnl_trail"]
        print(f"    {r['date']!s} {r['ticker']:<4} fill ${r['fill']:.2f} → low ${r['opt_low']:.2f} "
              f"({r['dd_pct']:.0f}%) → close ${r['opt_close']:.2f}: "
              f"trail exited −15%, no-stop earned {r['pct_close']:+.0f}% "
              f"(missed +${loss_avoided:.0f})")

ambig = month[month["category"] == "AMBIGUOUS"]
print(f"\n=== AMBIGUOUS trades (need minute-level data to resolve) ===\n")
if ambig.empty:
    print(f"  None — every drawdown trade was unambiguously classifiable.")
else:
    print(f"  {len(ambig)} trades hit BOTH a 15% drawdown AND a 15%+ peak. "
          f"Without minute data we can't tell if the drawdown happened BEFORE")
    print(f"  the peak (trail-stop kills the trade, no-stop catches recovery)")
    print(f"  or AFTER (trail-stop captures peak, no-stop fades back).\n")
    for _, r in ambig.iterrows():
        print(f"    {r['date']!s} {r['ticker']:<4} fill ${r['fill']:.2f}  "
              f"low ${r['opt_low']:.2f} ({r['dd_pct']:+.0f}%)  "
              f"high ${r['opt_high']:.2f} ({(r['opt_high']/r['fill']-1)*100:+.0f}%)  "
              f"close ${r['opt_close']:.2f} ({(r['opt_close']/r['fill']-1)*100:+.0f}%)  "
              f"contract: {r['contract']}")

# Save the ambiguous list for the minute-data follow-up
if not ambig.empty:
    ambig.to_csv(ROOT / "data" / "ambiguous_trades_for_minute_check.csv", index=False)
    print(f"\n  → wrote {len(ambig)} trades to data/ambiguous_trades_for_minute_check.csv")
    print(f"     for the minute-resolution follow-up phase.")
