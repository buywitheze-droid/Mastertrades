"""How many alerts will fire for the rest of the month, and what would the
account look like under the user's compound rule?

Compound rule (per the user):
  - Start with $500.
  - Each trade gets a position size of: $500 + 0.50 * cumulative_profit
    (i.e. you bank 50% of running profits into every subsequent trade,
    and keep the other 50% out as 'taken cash').
  - When cumulative profit < 0, position falls back to $500 floor.

Frequency:
  - Use the live picker (cap $1 + leverage bonus) on the 90-day ladder
    cache to get the realistic per-day alert count.
  - Project that to the remaining trading days in May 2026.
  - Today is Wed 2026-05-13. Rest of month = 14, 15, 18, 19, 20, 21, 22,
    26, 27, 28, 29 (May 25 is Memorial Day) = 11 trading days.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import numpy as np
import pandas as pd

from src.options_scanner import OptionContract, recommend_strikes, MAX_PREMIUM_USD

POSITION_BASE = 500.0
TRAIL_PCT     = 15.0
COMPOUND_FRAC = 0.50

# ── 1) Build the historical pick-list with the live picker ──────────────────
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
        "strike": top.strike, "entry": top.est_entry_price,
        "opt_open": float(match["opt_open"]),
        "opt_high": float(match["opt_high"]),
        "opt_low":  float(match["opt_low"]),
        "opt_close": float(match["opt_close"]),
    })

picks_df = pd.DataFrame(picks).sort_values("date").reset_index(drop=True)

# ── 2) Trade frequency ──────────────────────────────────────────────────────
unique_days = picks_df["date"].nunique()
date_min, date_max = picks_df["date"].min(), picks_df["date"].max()
all_days = pd.bdate_range(date_min, date_max).date
trading_days_in_window = len(all_days)
days_with_alerts = unique_days
alerts_total = len(picks_df)

print(f"=== Trade frequency from 90-day backtest ===")
print(f"  Window               : {date_min} -> {date_max}")
print(f"  Trading days in span : {trading_days_in_window}")
print(f"  Days with >=1 alert  : {days_with_alerts} "
      f"({days_with_alerts/trading_days_in_window*100:.0f}%)")
print(f"  Alerts surfaced      : {alerts_total} "
      f"(across SPY/QQQ/IWM ENTRY_OPEN candidates)")
print(f"  Avg alerts / day     : {alerts_total/trading_days_in_window:.2f}")

# After the cap-$1 picker, some alerts will be filtered (no surviving strike).
# Score what fraction of historical alerts produced an actionable pick.
# Compute trades-taken (where contracts > 0 with $500 budget).
picks_df["contracts"] = (POSITION_BASE // (picks_df["opt_open"] * 100)).astype(int)
taken = picks_df[picks_df["contracts"] > 0].copy()
fill_rate = len(taken) / alerts_total
print(f"  Actionable (>=1 ct)  : {len(taken)}/{alerts_total} = "
      f"{fill_rate*100:.0f}% (rest priced > $5/contract w/ $500 budget)")

# Compute realised P&L per trade (15% trail)
f = TRAIL_PCT / 100.0
recovered = taken["opt_high"] > taken["opt_open"]
stop_after_peak = taken["opt_high"] * (1 - f)
exit_recov = stop_after_peak.where(taken["opt_close"] < stop_after_peak, taken["opt_close"])
init_stop = taken["opt_open"] * (1 - f)
init_breach = taken["opt_low"] <= init_stop
exit_norec = init_stop.where(init_breach, taken["opt_close"])
exit_p = exit_recov.where(recovered, exit_norec)
taken["pnl_500"]   = (exit_p - taken["opt_open"]) * taken["contracts"] * 100
taken["pnl_per_$"] = taken["pnl_500"] / POSITION_BASE   # P&L per dollar of position

# ── 3) Project forward to rest of May ────────────────────────────────────────
# Today: Wed 2026-05-13. Rest of trading days in May 2026:
remaining_days = [
    dt.date(2026,5,14), dt.date(2026,5,15),
    dt.date(2026,5,18), dt.date(2026,5,19), dt.date(2026,5,20),
    dt.date(2026,5,21), dt.date(2026,5,22),
    # May 25 = Memorial Day (closed)
    dt.date(2026,5,26), dt.date(2026,5,27), dt.date(2026,5,28), dt.date(2026,5,29),
]
rem_n_days = len(remaining_days)
projected_alerts = alerts_total / trading_days_in_window * rem_n_days
projected_actionable = projected_alerts * fill_rate

print(f"\n=== Rest-of-May projection ({rem_n_days} trading days) ===")
print(f"  Expected alerts          : {projected_alerts:.0f} "
      f"(~{alerts_total/trading_days_in_window:.2f}/day × {rem_n_days} days)")
print(f"  Expected actionable plays: {projected_actionable:.0f} "
      f"(rest filtered by ${POSITION_BASE:.0f} budget cap)")

# ── 4) Compound simulation (deterministic — replay historical sequence) ─────
# The user's rule:
#   size_n = $500 + 0.50 * max(0, cumulative_profit_so_far)
# Each trade scales position by (size_n / $500) since we have $/$ P&L.
def compound_simulation(pnl_per_dollar_seq: np.ndarray,
                         base: float, frac: float) -> dict:
    cum_profit = 0.0
    sizes = []
    pnls  = []
    for r in pnl_per_dollar_seq:
        size = base + frac * max(0.0, cum_profit)
        pnl  = r * size
        cum_profit += pnl
        sizes.append(size); pnls.append(pnl)
    return dict(
        final_balance = base + cum_profit,    # treat base as starting cash
        cum_profit    = cum_profit,
        avg_size      = float(np.mean(sizes)),
        max_size      = float(np.max(sizes)),
        peak_profit   = float(np.maximum.accumulate(np.cumsum(pnls)).max()),
    )

# 4a) Replay the entire 90-day historical sequence (sanity)
seq = taken["pnl_per_$"].values
total_realistic = compound_simulation(seq, POSITION_BASE, COMPOUND_FRAC)

# 4b) Bootstrap-sample the most-recent N trades as a proxy for the next N
#     (using the most recent matches the late-window market regime)
recent_n = int(round(projected_actionable))
recent_trades = taken.tail(recent_n)["pnl_per_$"].values

# 4c) Monte-Carlo: 1000 random draws of `recent_n` trades from the full pool
rng = np.random.default_rng(42)
mc_results = []
for _ in range(1000):
    sample_idx = rng.choice(len(seq), size=recent_n, replace=True)
    sample = seq[sample_idx]
    res = compound_simulation(sample, POSITION_BASE, COMPOUND_FRAC)
    mc_results.append(res["cum_profit"])
mc_arr = np.array(mc_results)

# 4d) Same MC but with FIXED $500 (no compounding) for comparison
mc_fixed = []
for _ in range(1000):
    sample = seq[rng.choice(len(seq), size=recent_n, replace=True)]
    mc_fixed.append(np.sum(sample * POSITION_BASE))
mc_fixed_arr = np.array(mc_fixed)

# 4e) MC with 100% compounding (all-in reinvest) for upper-bound comparison
mc_full = []
for _ in range(1000):
    sample = seq[rng.choice(len(seq), size=recent_n, replace=True)]
    res = compound_simulation(sample, POSITION_BASE, 1.0)
    mc_full.append(res["cum_profit"])
mc_full_arr = np.array(mc_full)

print(f"\n=== Compound-projection (1,000 Monte-Carlo paths · {recent_n} trades) ===")
print(f"  Rule: position = $500 + {COMPOUND_FRAC*100:.0f}% × cumulative profit\n")

def percentile_block(name, arr, base):
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f"  {name}")
    print(f"    Median final cash : ${base + p50:>9,.0f}  (profit ${p50:>+8,.0f})")
    print(f"    p25 - p75 range   : ${base + p25:>9,.0f} - ${base + p75:,.0f}")
    print(f"    p10 (downside)    : ${base + p10:>9,.0f}  (profit ${p10:>+8,.0f})")
    print(f"    p90 (upside)      : ${base + p90:>9,.0f}  (profit ${p90:>+8,.0f})")
    print(f"    Worst-case path   : ${base + arr.min():>9,.0f}")
    print(f"    Best-case path    : ${base + arr.max():>9,.0f}")
    pct_lose = (arr < 0).mean() * 100
    print(f"    Probability red   : {pct_lose:.0f}%")
    print()

percentile_block(f"Strategy: $500 fixed (no compounding)", mc_fixed_arr, POSITION_BASE)
percentile_block(f"Strategy: +50% compound (your rule)",  mc_arr,       POSITION_BASE)
percentile_block(f"Strategy: +100% compound (all-in)",    mc_full_arr,  POSITION_BASE)

# ── 5) Walk through the most-recent N trades as a concrete example ──────────
print(f"=== Concrete walkthrough: replay the LAST {recent_n} historical trades ===\n")
print(f"  This is what the +50% compound rule would have done on the most recent")
print(f"  {recent_n} actionable trades from the cached backtest.\n")
print(f"  {'#':>3} {'date':<12} {'tkr':<5} {'$ entry':>8} {'ct':>4} "
      f"{'size':>8} {'P&L':>9} {'cum P&L':>10} {'cash':>9}")
cum_p = 0.0
cash  = POSITION_BASE
for i, (_, row) in enumerate(taken.tail(recent_n).iterrows(), 1):
    size = POSITION_BASE + COMPOUND_FRAC * max(0.0, cum_p)
    contracts = int(size // (row["opt_open"] * 100))
    if contracts == 0:
        # Position too small for even 1 contract. Skip.
        print(f"  {i:>3} {row['date']!s:<12} {row['ticker']:<5} "
              f"${row['opt_open']:>6.2f}  - ${size:>6,.0f}  SKIP (size < 1 ct)")
        continue
    # Approximate scaling: pnl scales with position size
    f = TRAIL_PCT / 100.0
    if row["opt_high"] > row["opt_open"]:
        stop = row["opt_high"] * (1 - f)
        exit_p = stop if row["opt_close"] < stop else row["opt_close"]
    else:
        init_s = row["opt_open"] * (1 - f)
        exit_p = init_s if row["opt_low"] <= init_s else row["opt_close"]
    pnl = (exit_p - row["opt_open"]) * contracts * 100
    cum_p += pnl
    cash   = POSITION_BASE + cum_p
    print(f"  {i:>3} {row['date']!s:<12} {row['ticker']:<5} "
          f"${row['opt_open']:>6.2f} {contracts:>4} ${size:>6,.0f} "
          f"${pnl:>+8,.0f} ${cum_p:>+9,.0f} ${cash:>8,.0f}")

print(f"\n  Final cash: ${cash:,.0f}  (started with ${POSITION_BASE:,.0f})")
