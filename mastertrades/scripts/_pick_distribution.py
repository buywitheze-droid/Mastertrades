"""What's the distribution of premiums the live algo currently picks?"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

df = pd.read_csv(ROOT / "data" / "per_source_gate_money_ladder.csv")
print(f"=== Live-algo strike picks: premium distribution ({len(df)} trades) ===\n")
buckets = [
    ("$0.01 (penny)",     df["opt_open"] <= 0.01),
    ("$0.02 - $0.05",     (df["opt_open"] > 0.01) & (df["opt_open"] <= 0.05)),
    ("$0.05 - $0.10",     (df["opt_open"] > 0.05) & (df["opt_open"] <= 0.10)),
    ("$0.10 - $0.20",     (df["opt_open"] > 0.10) & (df["opt_open"] <= 0.20)),
    ("$0.20 - $0.50",     (df["opt_open"] > 0.20) & (df["opt_open"] <= 0.50)),
    ("$0.50 - $1.00",     (df["opt_open"] > 0.50) & (df["opt_open"] <= 1.00)),
    ("$1.00 - $2.00",     (df["opt_open"] > 1.00) & (df["opt_open"] <= 2.00)),
    ("$2.00 - $5.00",     (df["opt_open"] > 2.00) & (df["opt_open"] <= 5.00)),
    ("$5.00+",            df["opt_open"] > 5.00),
]
print(f"  {'bucket':<18} {'n':>4} {'%':>6} {'avg gain':>10} "
      f"{'realised P&L (15% trail)':>26}")
for label, mask in buckets:
    sub = df[mask]
    n = len(sub)
    pct = n / len(df) * 100
    if n == 0:
        print(f"  {label:<18} {n:>4} {pct:>5.0f}% {'—':>10} {'—':>26}")
        continue
    # Realistic 15% trail P&L (recompute on the fly)
    f = 0.15
    high = sub["opt_high"]; low = sub["opt_low"]; close = sub["opt_close"]; ent = sub["opt_open"]
    recovered = high > ent
    stop_after_peak = high * (1 - f)
    exit_recov = stop_after_peak.where(close < stop_after_peak, close)
    init_stop = ent * (1 - f)
    init_breach = low <= init_stop
    exit_norec = init_stop.where(init_breach, close)
    exit_p = exit_recov.where(recovered, exit_norec)
    pnl = (exit_p - ent) * sub["contracts"] * 100
    avg_gain = ((high / ent) - 1.0).mean() * 100
    print(f"  {label:<18} {n:>4} {pct:>5.0f}% {avg_gain:>+9.0f}% "
          f"{pnl.sum():>+12,.0f}$ tot · {pnl.mean():>+5,.0f}$/trade")
