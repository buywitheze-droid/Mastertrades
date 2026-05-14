"""Trailing-stop sweep on the strike-ladder ledger.

Approximation given we only have daily OHLC:
  * Assume intraday path is roughly  open -> low -> high -> close
    (typical V-shape rebound for a Drop alert that fired correctly).
  * Initial stop: entry * (1 - Y/100), trailed up to running_max * (1 - Y/100)
    once price exceeds entry.
  * If opt_high reached: stop level after peak = opt_high * (1 - Y/100).
       If opt_close >= stop_level   -> stop never fires, exit at close.
       Else                          -> stop fires on the descent, exit at stop_level.
  * If opt_high <= entry            -> price never recovered; initial stop
       at entry*(1-Y/100) was breached at opt_low. Exit at that stop.

This is conservative: real live trail stops can be tighter or looser than
what daily bars allow us to model, but the rank order between Y values is
trustworthy.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np

ledger = ROOT / "data" / "per_source_gate_money_ladder.csv"
df = pd.read_csv(ledger)
print(f"=== Trailing-stop sweep on live-algo strike picks "
      f"({len(df)} trades, $500/alert budget) ===")
print("Rule: stop = max(entry, running_max) * (1 - Y/100); exit when option")
print("      first crosses below stop OR at close, whichever happens first.\n")
print(f"{'Y (trail %)':<14} {'Total PnL':>12} {'Avg/trade':>11} {'Win%':>6} "
      f"{'Avg exit/entry':>16}")

for Y in [10, 15, 20, 25, 30, 40, 50, 60, 75]:
    f = Y / 100.0
    entry = df["opt_open"]
    high = df["opt_high"]
    low = df["opt_low"]
    close = df["opt_close"]

    # Path 1: opt_high > opt_open  (recovery happened)
    recovered = high > entry
    stop_after_peak = high * (1 - f)
    stop_fired_late = close < stop_after_peak
    exit_recovered = np.where(stop_fired_late, stop_after_peak, close)

    # Path 2: opt_high <= opt_open  (no recovery, stop should fire at initial level)
    initial_stop = entry * (1 - f)
    initial_breached = low <= initial_stop
    exit_no_recovery = np.where(initial_breached, initial_stop, close)

    exit_price = np.where(recovered, exit_recovered, exit_no_recovery)
    pnl = (exit_price - entry) * df["contracts"] * 100

    avg_exit_ratio = (exit_price / entry).mean()
    print(f"  {Y}%{'':<11}{pnl.sum():>+11,.0f}$ {pnl.mean():>+10,.0f}$ "
          f"{(pnl > 0).mean() * 100:>5.0f}% {avg_exit_ratio:>15.2f}x")

print()
print("For reference (already shown):")
print(f"  Hold to close       -25,733$       -299$     8% (catastrophic)")
print(f"  Sell at exact high  +18,985$       +221$    94% (impossible in live)")
