"""Full-balance compounding sim — last 30 days, $1,000 start.

Question being answered:
  If we had taken EVERY alert from the last 30 days, given ourselves ~10 min
  to enter (so a degraded fill vs the absolute low), gone all-in on each trade
  using full account balance, starting from $1,000 — where would we be now
  and did we ever blow up?

Modelling choices:
  - Universe: actionable picks from the live picker (cap $1 + leverage_score).
  - Entry: 10-min-delayed fill modelled as
        entry = opt_low + ENTRY_LATENCY * (opt_open - opt_low)
    With ENTRY_LATENCY = 0.5 we assume a fill HALFWAY between the option's
    intraday low and its session open. This is conservative — an alert that
    fires at the underlying's low and we fill ~10 min later typically gets
    a price somewhere between the option low and the open (the low can be
    a momentary spike to $0.01).
  - Exit: 15% trail-stop (matches the dashboard default)
        if peak > entry: exit = max(close, peak * 0.85)
        else:            exit = max(close, entry * 0.85) if low <= entry*0.85
                                         else close
  - Sizing: integer contracts purchasable with FULL balance at the moment
        of the trade. ALL profits reinvested into the next trade.
  - Order: chronological. Multiple alerts on the same day are sequenced
        in the order Polygon returns them (deterministic per-ticker).
  - Bankruptcy: if balance drops below the cheapest contract cost, no more
        trades possible (game over).

Side question: could ANY single trade have wiped the account?
  With the trail-stop model, max single-trade loss is bounded at ~15% of
  position (initial stop). Account can't go to zero in one trade UNLESS:
    (a) the option gaps through the stop (slippage), or
    (b) the trail-stop fails to execute (illiquid 0DTE).
  We report the worst observed historical trade and the theoretical worst
  case under perfect stop execution.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import numpy as np
import pandas as pd
from src.options_scanner import OptionContract, recommend_strikes, MAX_PREMIUM_USD

START_CASH      = 1_000.0
TRAIL_PCT       = 15.0
ENTRY_LATENCY   = 0.5    # 0 = perfect fill at low, 1 = fill at session open
TODAY           = dt.date(2026, 5, 13)
LOOKBACK_DAYS   = 30

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
        "strike": top.strike,
        "opt_open": float(match["opt_open"]),
        "opt_high": float(match["opt_high"]),
        "opt_low":  float(match["opt_low"]),
        "opt_close": float(match["opt_close"]),
    })
picks_df = pd.DataFrame(picks).sort_values("date").reset_index(drop=True)

# ── 2) Filter to last 30 days ────────────────────────────────────────────────
cutoff = TODAY - dt.timedelta(days=LOOKBACK_DAYS)
month_picks = picks_df[picks_df["date"] >= cutoff].copy()
month_picks = month_picks.sort_values(["date", "ticker"]).reset_index(drop=True)
print(f"=== Window: {cutoff} -> {TODAY} ({len(month_picks)} alerts) ===")
print(f"  Across {month_picks['date'].nunique()} unique alert days "
      f"({month_picks['ticker'].value_counts().to_dict()})\n")

# ── 3) Compute the modelled fill price (10-min latency) ─────────────────────
month_picks["fill"] = (
    month_picks["opt_low"] + ENTRY_LATENCY *
    (month_picks["opt_open"] - month_picks["opt_low"])
).round(2)
# Per-dollar return under 15% trail-stop
trail = TRAIL_PCT / 100.0
def trail_exit(entry, high, low, close):
    """Return the exit price under the 15% trail-stop rule."""
    if high > entry:
        peak_stop = high * (1 - trail)
        return max(close, peak_stop) if close < peak_stop else close
    init_stop = entry * (1 - trail)
    return init_stop if low <= init_stop else close
month_picks["exit"] = month_picks.apply(
    lambda r: trail_exit(r["fill"], r["opt_high"], r["opt_low"], r["opt_close"]),
    axis=1)
month_picks["pct_return"] = (month_picks["exit"] / month_picks["fill"]) - 1

# ── 4) Full-balance compound walk ───────────────────────────────────────────
balance = START_CASH
peak    = START_CASH
trough  = START_CASH
journal = []
went_broke = False
broke_reason = ""

print(f"=== Full-balance compounding · ${START_CASH:,.0f} start · "
      f"15% trail · entry latency {ENTRY_LATENCY*100:.0f}% ===\n")
print(f"  {'#':>2} {'date':<11} {'tkr':<4} {'fill':>5} {'exit':>5} "
      f"{'%':>6} {'ct':>5} {'cost':>7} {'pnl':>9} {'balance':>10}")
print(f"  " + "-" * 84)

for i, row in month_picks.iterrows():
    fill = row["fill"]
    cost_per_contract = fill * 100
    contracts = int(balance // cost_per_contract)
    if contracts == 0:
        # Cannot afford even 1 contract — sit out, balance unchanged
        print(f"  {i+1:>2} {row['date']!s:<11} {row['ticker']:<4} "
              f"${fill:>4.2f}    -      -     -        -        -  ${balance:>8,.2f} (skip)")
        continue
    cost = contracts * cost_per_contract
    pnl  = (row["exit"] - fill) * contracts * 100
    balance_after = balance + pnl
    pct_ret = pnl / cost * 100
    print(f"  {i+1:>2} {row['date']!s:<11} {row['ticker']:<4} "
          f"${fill:>4.2f} ${row['exit']:>4.2f} {pct_ret:>+5.0f}% "
          f"{contracts:>4}  ${cost:>5,.0f} ${pnl:>+8,.2f} ${balance_after:>9,.2f}")
    journal.append({
        "date": row["date"], "ticker": row["ticker"],
        "fill": fill, "exit": row["exit"], "contracts": contracts,
        "pct_return": pct_ret, "pnl": pnl,
        "balance_before": balance, "balance_after": balance_after,
    })
    balance = balance_after
    peak    = max(peak, balance)
    trough  = min(trough, balance)
    if balance < 1.0:
        went_broke = True
        broke_reason = f"balance < $1 after trade #{i+1}"
        break

print(f"  " + "-" * 84)

# ── 5) Summary ──────────────────────────────────────────────────────────────
total_return = (balance / START_CASH - 1) * 100
n_trades = len(journal)
n_wins = sum(1 for j in journal if j["pnl"] > 0)
n_losses = n_trades - n_wins
worst = min((j["pct_return"] for j in journal), default=0)
best  = max((j["pct_return"] for j in journal), default=0)
worst_dollar = min((j["pnl"] for j in journal), default=0)
best_dollar  = max((j["pnl"] for j in journal), default=0)
max_dd_pct = (1 - trough / peak) * 100 if peak else 0

print(f"\n=== Summary ===")
print(f"  Starting balance     : ${START_CASH:,.2f}")
print(f"  Ending balance       : ${balance:,.2f}")
print(f"  Total return         : {total_return:+.0f}%  "
      f"({balance / START_CASH:.1f}x)")
print(f"  Trades taken         : {n_trades}  ({n_wins}W / {n_losses}L · "
      f"win rate {n_wins / n_trades * 100 if n_trades else 0:.0f}%)")
print(f"  Best single trade    : {best:+.0f}%  (${best_dollar:+,.0f})")
print(f"  Worst single trade   : {worst:+.0f}%  (${worst_dollar:+,.0f})")
print(f"  Peak balance         : ${peak:,.2f}")
print(f"  Trough after peak    : ${trough:,.2f}")
print(f"  Max drawdown         : {max_dd_pct:.0f}%")

print(f"\n=== Did we ever go from full position to ~$0? ===")
print(f"  Bankrupt during run  : {'YES — ' + broke_reason if went_broke else 'NO'}")
print(f"  Worst single trade %: {worst:+.0f}%  "
      f"(15% trail-stop floor = -15% / trade; matched: "
      f"{'yes' if abs(worst + 15) < 5 else 'no'})")
print(f"")

# ── 6) Stress test — what if a trade WAS a total loss? ──────────────────────
print(f"=== Stress test: 'what if the trail-stop failed on a single trade?' ===")
print(f"  Scenario: assume one of the 1...n trades suffered a -100% wipe")
print(f"  (option expired worthless, no fill at the trail level).\n")
print(f"  {'after trade #':<14} {'balance before':>15} {'balance if -100%':>18}")
print(f"  " + "-" * 50)
running = START_CASH
for i, j in enumerate(journal, 1):
    running_before = j["balance_before"]
    if running_before > 0:
        # simulating: we'd lose ALL of cost (= contracts * fill * 100)
        cost = j["contracts"] * j["fill"] * 100
        balance_if_wiped = running_before - cost
        print(f"  trade #{i:<8} ${running_before:>13,.2f}  "
              f"${balance_if_wiped:>15,.2f}")
print(f"  (each row asks: had THAT specific trade gone to zero, what would")
print(f"   balance be? trail-stop normally caps loss at ~-15%/trade; this is")
print(f"   the worst-case 'stop failed' downside.)")
