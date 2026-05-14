"""How would a minimum-premium filter change 0DTE Drop performance?

Reads the strike-ladder cache (per_source_gate_trades_ladder.csv) and re-runs
the live-algo strike selection (max est_gain_pct in offsets 1..9) under
several premium-floor scenarios. Reports REALISTIC P&L (15% trail-stop) for
each so you can decide whether to implement.

Scenarios:
  - No floor       (current behavior — all contracts admitted)
  - >= $0.02       (block sub-penny chains)
  - >= $0.05       (only contracts with at least 5¢ entry)
  - >= $0.10       (only "real" liquid options)
  - $0.04-$0.05    (the narrow band the user asked about)
  - $0.05-$0.20    (sweet spot for cheap-but-tradeable)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

POSITION_USD = 500.0
TRAIL_PCT    = 15.0  # matches dashboard default

ladder = ROOT / "data" / "per_source_gate_trades_ladder.csv"
df_full = pd.read_csv(ladder).copy()
print(f"Loaded {len(df_full)} ladder rows across "
      f"{df_full[['ticker','date']].drop_duplicates().shape[0]} candidate days.\n")


# ── Replay live-algo strike selection (mirror of money_ladder_phase) ────────
def pick_and_score(df: pd.DataFrame) -> dict:
    """Given a (filtered) ladder df, pick the live-algo strike per day and
    score $POSITION_USD per trade with realistic mid-exit + 15% trail."""
    if df.empty:
        return {"days_with_pick": 0, "trades_taken": 0,
                "total_pnl_real": 0.0, "total_pnl_trail": 0.0,
                "avg_per_trade": 0.0, "win_rate": 0.0,
                "avg_contracts": 0.0, "skipped_premium": 0,
                "no_pick": 0}

    # Mirror options_scanner.recommend_strikes() logic
    df = df.copy()
    df["entry_est"]      = df["opt_low"].clip(lower=0.01)
    df["recovery_target"] = df["underlying_open"]
    intrinsic_at_target  = (df["recovery_target"] - df["strike"]).clip(lower=0.0)
    target_itm           = intrinsic_at_target + 0.05
    target_otm           = (0.5 * (df["recovery_target"] - df["strike"] + 1.0)).clip(lower=0.05)
    df["target_price_est"] = target_itm.where(df["recovery_target"] >= df["strike"], target_otm)
    df["est_gain_pct"]   = (df["target_price_est"] - df["entry_est"]) / df["entry_est"] * 100

    # Live algo only considers offsets 1..9 above the low
    df = df[(df["offset_from_low"] >= 1) & (df["offset_from_low"] <= 9)].copy()

    # Group by candidate-day and pick highest est_gain_pct
    if df.empty:
        return {"days_with_pick": 0, "trades_taken": 0,
                "total_pnl_real": 0.0, "total_pnl_trail": 0.0,
                "avg_per_trade": 0.0, "win_rate": 0.0,
                "avg_contracts": 0.0, "skipped_premium": 0,
                "no_pick": 0}

    pick_idx = df.groupby(["ticker", "date"])["est_gain_pct"].idxmax()
    picks = df.loc[pick_idx].copy()

    n_days = len(picks)

    # Position sizing: $500 / (opt_open * 100) → integer contracts
    picks["contracts"] = (POSITION_USD // (picks["opt_open"] * 100)).astype(int)
    skipped = int((picks["contracts"] == 0).sum())
    taken = picks[picks["contracts"] > 0].copy()

    # REALISTIC exit: midpoint of (open, high)
    taken["exit_mid"]   = (taken["opt_open"] + taken["opt_high"]) / 2.0
    taken["pnl_real"]   = (taken["exit_mid"] - taken["opt_open"]) * taken["contracts"] * 100

    # TRAIL exit: max(opt_close, opt_high * (1-trail)) for recovered, else
    # max(opt_close, opt_open*(1-trail)) for non-recovered (initial-stop guard).
    f = TRAIL_PCT / 100.0
    recovered = taken["opt_high"] > taken["opt_open"]
    stop_after_peak = taken["opt_high"] * (1 - f)
    exit_recovered  = stop_after_peak.where(taken["opt_close"] < stop_after_peak, taken["opt_close"])
    initial_stop    = taken["opt_open"] * (1 - f)
    initial_breach  = taken["opt_low"] <= initial_stop
    exit_no_recov   = initial_stop.where(initial_breach, taken["opt_close"])
    exit_trail      = exit_recovered.where(recovered, exit_no_recov)
    taken["pnl_trail"] = (exit_trail - taken["opt_open"]) * taken["contracts"] * 100

    return {
        "days_with_pick":   n_days,
        "trades_taken":     len(taken),
        "skipped_premium":  skipped,
        "total_pnl_real":   float(taken["pnl_real"].sum()),
        "total_pnl_trail":  float(taken["pnl_trail"].sum()),
        "avg_per_trade":    float(taken["pnl_trail"].mean()) if len(taken) else 0.0,
        "win_rate":         float((taken["pnl_trail"] > 0).mean() * 100) if len(taken) else 0.0,
        "avg_contracts":    float(taken["contracts"].mean()) if len(taken) else 0.0,
        "avg_entry":        float(taken["opt_open"].mean()) if len(taken) else 0.0,
    }


scenarios = [
    ("Current (no floor)",       lambda r: True),
    ("Floor >= $0.02",           lambda r: r["opt_open"] >= 0.02),
    ("Floor >= $0.04",           lambda r: r["opt_open"] >= 0.04),
    ("Floor >= $0.05",           lambda r: r["opt_open"] >= 0.05),
    ("Floor >= $0.10",           lambda r: r["opt_open"] >= 0.10),
    ("Floor >= $0.20",           lambda r: r["opt_open"] >= 0.20),
    ("Window $0.04 - $0.05",     lambda r: 0.04 <= r["opt_open"] <= 0.05),
    ("Window $0.05 - $0.20",     lambda r: 0.05 <= r["opt_open"] <= 0.20),
    ("Window $0.10 - $0.50",     lambda r: 0.10 <= r["opt_open"] <= 0.50),
]

print(f"=== Premium-floor / window sweep — REALISTIC exit & 15% trail-stop ===\n")
print(f"  Scenario                    Days  Taken  AvgEntry  AvgCt  TotalP&L(trail) AvgPnl   Win%")
print(f"  " + "-" * 96)
for label, predicate in scenarios:
    mask = df_full.apply(predicate, axis=1)
    sub  = df_full[mask].copy()
    r = pick_and_score(sub)
    entry = f"${r['avg_entry']:.2f}" if r["trades_taken"] else "—"
    print(f"  {label:<26} {r['days_with_pick']:>4}  {r['trades_taken']:>5}  "
          f"{entry:>7}  {r['avg_contracts']:>5.1f}  "
          f"{r['total_pnl_trail']:>+13,.0f}$  "
          f"{r['avg_per_trade']:>+6,.0f}$  {r['win_rate']:>4.0f}%")

print()
print(f"=== Honest read ===")
print(f"  • POSITION_USD = ${POSITION_USD:.0f} per trade · TRAIL = {TRAIL_PCT:.0f}%")
print(f"  • REALISTIC exit assumes you sell mid(open,high); trail uses 15% peak-stop.")
print(f"  • 'Days' = candidate days where ≥1 strike survives the filter.")
print(f"  • 'Taken' = days where the picked strike's premium ≤ ${POSITION_USD:.0f} per contract.")
print(f"  • Window scenarios IGNORE strikes outside the band even if they had higher est_gain_pct,")
print(f"    so you may see fewer 'days' than the unfiltered case.")
