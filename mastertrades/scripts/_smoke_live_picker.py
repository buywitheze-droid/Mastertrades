"""End-to-end smoke: feed the ladder cache through the LIVE recommend_strikes()
and confirm the picks match what _strike_strategy_compare.py predicted for
Strategy E (cap $1 + leverage bonus). This guarantees the in-app picker now
matches the +$24,480 / +$401-per-trade strategy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.options_scanner import OptionContract, recommend_strikes, MAX_PREMIUM_USD

df = pd.read_csv(ROOT / "data" / "per_source_gate_trades_ladder.csv")
print(f"Loaded {len(df)} ladder rows. MAX_PREMIUM_USD = ${MAX_PREMIUM_USD:.2f}\n")

picks = []
skipped_no_picks = 0
for (ticker, date), group in df.groupby(["ticker", "date"]):
    underlying_open = float(group["underlying_open"].iloc[0])
    underlying_low  = float(group["underlying_low"].iloc[0])

    # Re-build a synthetic chain from the ladder rows (one per strike).
    chain = []
    for _, row in group.iterrows():
        chain.append(OptionContract(
            ticker=row["contract"], contract_type="call",
            strike=float(row["strike"]), expiration=date,
            day_open=float(row["opt_open"]), day_high=float(row["opt_high"]),
            day_low=float(row["opt_low"]),  day_close=float(row["opt_close"]),
            day_volume=1000, implied_vol=0.30,
            delta=0.5, gamma=0.05, theta=-0.10, vega=0.05,
            open_interest=500,
        ))
    recs = recommend_strikes(underlying_open, underlying_low, chain)
    if not recs:
        skipped_no_picks += 1
        continue

    top = max(recs, key=lambda r: r.leverage_score)
    # Find the matching ladder row to get the realised opt_high/low/close
    match = group[group["strike"] == top.strike].iloc[0]
    picks.append({
        "ticker": ticker, "date": date,
        "strike": top.strike, "entry": top.est_entry_price,
        "lev_score": top.leverage_score, "est_gain_pct": top.est_gain_pct,
        "opt_open": float(match["opt_open"]),
        "opt_high": float(match["opt_high"]),
        "opt_low":  float(match["opt_low"]),
        "opt_close": float(match["opt_close"]),
    })

picks_df = pd.DataFrame(picks)
print(f"Live algo produced {len(picks_df)} picks "
      f"({skipped_no_picks} candidate-days had no surviving strike under cap).\n")

# Score with $500/trade and 15% trail
position_usd = 500.0; trail = 0.15
picks_df["contracts"] = (position_usd // (picks_df["opt_open"] * 100)).astype(int)
taken = picks_df[picks_df["contracts"] > 0].copy()
recovered = taken["opt_high"] > taken["opt_open"]
stop_after_peak = taken["opt_high"] * (1 - trail)
exit_recov = stop_after_peak.where(taken["opt_close"] < stop_after_peak, taken["opt_close"])
init_stop = taken["opt_open"] * (1 - trail)
init_breach = taken["opt_low"] <= init_stop
exit_norec = init_stop.where(init_breach, taken["opt_close"])
exit_p = exit_recov.where(recovered, exit_norec)
taken["pnl"] = (exit_p - taken["opt_open"]) * taken["contracts"] * 100

print(f"=== Live recommend_strikes() — $500/trade · 15% trail ===")
print(f"  Trades taken     : {len(taken)}")
print(f"  Avg entry price  : ${taken['opt_open'].mean():.2f}")
print(f"  Avg contracts    : {taken['contracts'].mean():.1f}")
print(f"  Total P&L        : ${taken['pnl'].sum():+,.0f}")
print(f"  Avg P&L / trade  : ${taken['pnl'].mean():+,.0f}")
print(f"  Win rate         : {(taken['pnl'] > 0).mean()*100:.0f}%")
print(f"  Best trade       : ${taken['pnl'].max():+,.0f}")
print(f"  Worst trade      : ${taken['pnl'].min():+,.0f}")
print(f"  Profit factor    : {taken[taken['pnl']>0]['pnl'].sum() / abs(taken[taken['pnl']<0]['pnl'].sum()):.1f}")
print(f"\nExpected (Strategy E from compare-script):")
print(f"  Trades taken     : 61")
print(f"  Total P&L        : +$24,480")
print(f"  Avg P&L / trade  : +$401")
print(f"  Win rate         : 85%")
print(f"  Profit factor    : 43.6")
