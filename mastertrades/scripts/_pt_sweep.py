"""One-shot profit-take rule sweep on the cached money-simulation ledger.
Reads data/per_source_gate_money_ladder.csv (live-algo strike selection)
produced by scripts/backtest_per_source_gate.py --money 500 --ladder.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

ledger = ROOT / "data" / "per_source_gate_money_ladder.csv"
if not ledger.exists():
    ledger = ROOT / "data" / "per_source_gate_money.csv"
df = pd.read_csv(ledger)
print(f'=== Profit-take rule sweep (source: {ledger.name}, {len(df)} trades) ===')
print('Rule: SELL all contracts when option HIGH crosses opt_open * (1 + pct).')
print('      If HIGH never reaches the target -> HOLD to close.')
print()
print(f'{"PT rule":<14} {"Total PnL":>12} {"Avg/trade":>11} {"Win%":>6} {"# fired":>8} {"# held":>7}')

scenarios = [
    ('No PT (close)', None),
    ('+25%',          0.25),
    ('+50%',          0.50),
    ('+75%',          0.75),
    ('+100%',         1.00),
    ('+150%',         1.50),
    ('+200%',         2.00),
    ('+300%',         3.00),
    ('+500%',         5.00),
    ('+1000%',       10.00),
    ('+2000%',       20.00),
    ('Sell at high',  None, 'high'),
]

for sc in scenarios:
    label = sc[0]
    pct   = sc[1]
    mode  = sc[2] if len(sc) > 2 else None
    if mode == 'high':
        df['realised'] = df['opt_high']
        n_fired = int((df['opt_high'] > df['opt_open']).sum())
        n_held  = len(df) - n_fired
    elif pct is None:
        df['realised'] = df['opt_close']
        n_fired = 0
        n_held  = len(df)
    else:
        target = df['opt_open'] * (1 + pct)
        fired_mask = df['opt_high'] >= target
        df['realised'] = target.where(fired_mask, df['opt_close'])
        n_fired = int(fired_mask.sum())
        n_held  = len(df) - n_fired
    pnl = (df['realised'] - df['opt_open']) * df['contracts'] * 100
    print(f'  {label:<12} {pnl.sum():>+11,.0f}$ {pnl.mean():>+10,.0f}$ '
          f'{(pnl > 0).mean() * 100:>5.0f}% {n_fired:>8} {n_held:>7}')
