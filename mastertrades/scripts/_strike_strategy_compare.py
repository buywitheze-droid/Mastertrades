"""Strike-selection strategy comparison.

For each candidate strategy, simulate $500/trade with a 15% trail-stop on
the 90-day strike-ladder dataset. Reports total P&L, avg/trade, win rate,
worst single trade, max consecutive drawdown, and Sharpe-ish risk-adjusted
return so we can pick the one that maximises profits and minimises losses.

Strategies:
  A. Current               — algo picks max est_gain_pct in offsets 1..9.
  B. Hard cap $1.00        — same as A, but skip strikes with opt_open > $1.
  C. Hard cap $0.50        — even tighter cap.
  D. Leverage bonus        — re-rank by est_gain_pct * sqrt(1 / entry).
                              Boosts cheap strikes without hard-filtering.
  E. Both (cap $1 + bonus) — aggressive cheap-bias.
  F. Floor only ($0.05)    — defensive (verifies it's a no-op).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import math
import numpy as np
import pandas as pd

POSITION_USD = 500.0
TRAIL_PCT    = 15.0

ladder = ROOT / "data" / "per_source_gate_trades_ladder.csv"
df_full = pd.read_csv(ladder).copy()


def base_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the live-algo's est_gain_pct column to a ladder df."""
    df = df.copy()
    df["entry_est"]      = df["opt_low"].clip(lower=0.01)
    df["recovery_target"] = df["underlying_open"]
    intrinsic_at_target  = (df["recovery_target"] - df["strike"]).clip(lower=0.0)
    target_itm           = intrinsic_at_target + 0.05
    target_otm           = (0.5 * (df["recovery_target"] - df["strike"] + 1.0)).clip(lower=0.05)
    df["target_price_est"] = target_itm.where(df["recovery_target"] >= df["strike"], target_otm)
    df["est_gain_pct"]   = (df["target_price_est"] - df["entry_est"]) / df["entry_est"] * 100
    df = df[(df["offset_from_low"] >= 1) & (df["offset_from_low"] <= 9)].copy()
    return df


def pick_with_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Pick one strike per (ticker, date) according to strategy."""
    if df.empty:
        return df
    if strategy == "current":
        rank_col = "est_gain_pct"
    elif strategy == "max_cap_1":
        df = df[df["opt_open"] <= 1.00].copy()
        rank_col = "est_gain_pct"
    elif strategy == "max_cap_050":
        df = df[df["opt_open"] <= 0.50].copy()
        rank_col = "est_gain_pct"
    elif strategy == "leverage_bonus":
        df = df.copy()
        df["leverage_score"] = df["est_gain_pct"] * (1.0 / df["opt_open"]).pow(0.5)
        rank_col = "leverage_score"
    elif strategy == "cap_and_bonus":
        df = df[df["opt_open"] <= 1.00].copy()
        df["leverage_score"] = df["est_gain_pct"] * (1.0 / df["opt_open"]).pow(0.5)
        rank_col = "leverage_score"
    elif strategy == "floor_only":
        df = df[df["opt_open"] >= 0.05].copy()
        rank_col = "est_gain_pct"
    else:
        raise ValueError(strategy)
    if df.empty:
        return df
    pick_idx = df.groupby(["ticker", "date"])[rank_col].idxmax()
    return df.loc[pick_idx].copy()


def score(picks: pd.DataFrame) -> dict:
    if picks.empty:
        return dict(days=0, taken=0, total=0, avg=0, win_pct=0, best=0,
                    worst=0, max_dd=0, total_loss=0, total_win=0,
                    risk_adj=0, avg_entry=0, avg_contracts=0)
    picks = picks.copy()
    picks["contracts"] = (POSITION_USD // (picks["opt_open"] * 100)).astype(int)
    taken = picks[picks["contracts"] > 0].copy()
    if taken.empty:
        return dict(days=len(picks), taken=0, total=0, avg=0, win_pct=0, best=0,
                    worst=0, max_dd=0, total_loss=0, total_win=0,
                    risk_adj=0, avg_entry=0, avg_contracts=0)

    f = TRAIL_PCT / 100.0
    high = taken["opt_high"]; low = taken["opt_low"]
    close = taken["opt_close"]; ent = taken["opt_open"]
    recovered = high > ent
    stop_after_peak = high * (1 - f)
    exit_recov = stop_after_peak.where(close < stop_after_peak, close)
    init_stop = ent * (1 - f)
    init_breach = low <= init_stop
    exit_norec = init_stop.where(init_breach, close)
    exit_p = exit_recov.where(recovered, exit_norec)
    pnl = (exit_p - ent) * taken["contracts"] * 100

    # Sort by date for drawdown
    taken_sorted = taken.assign(pnl=pnl).sort_values("date")
    equity = taken_sorted["pnl"].cumsum()
    drawdowns = equity.cummax() - equity
    max_dd = float(drawdowns.max())

    losses  = pnl[pnl < 0]
    wins    = pnl[pnl > 0]

    # Sharpe-ish: total / stdev of trade P&L (annualised vs 1)
    sharpe_ish = pnl.mean() / (pnl.std(ddof=1) + 1e-9) * math.sqrt(len(pnl))

    return dict(
        days=len(picks), taken=len(taken),
        total=float(pnl.sum()),
        avg=float(pnl.mean()),
        win_pct=float((pnl > 0).mean() * 100),
        best=float(pnl.max()),
        worst=float(pnl.min()),
        max_dd=max_dd,
        total_loss=float(losses.sum()),
        total_win=float(wins.sum()),
        risk_adj=sharpe_ish,
        avg_entry=float(taken["opt_open"].mean()),
        avg_contracts=float(taken["contracts"].mean()),
    )


df = base_compute(df_full)
strategies = [
    ("A. Current (no change)",   "current"),
    ("B. Hard cap $1.00",        "max_cap_1"),
    ("C. Hard cap $0.50",        "max_cap_050"),
    ("D. Leverage bonus",        "leverage_bonus"),
    ("E. Cap $1 + bonus",        "cap_and_bonus"),
    ("F. Floor $0.05 only",      "floor_only"),
]

print(f"=== Strike-selection strategy backtest ($500/trade · 15% trail · 90 days) ===\n")
print(f"  {'Strategy':<24} {'Take':>5} {'Total':>11} {'Avg':>7} {'Win%':>5} "
      f"{'Best':>9} {'Worst':>9} {'MaxDD':>9} {'Wins$':>9} {'Loss$':>9} {'Rsk-Adj':>8}")
print(f"  " + "-" * 130)

results = {}
for label, key in strategies:
    picks = pick_with_strategy(df, key)
    r = score(picks)
    results[key] = (label, r)
    print(f"  {label:<24} {r['taken']:>5} "
          f"{r['total']:>+10,.0f}$ {r['avg']:>+6,.0f}$ {r['win_pct']:>4.0f}% "
          f"{r['best']:>+8,.0f}$ {r['worst']:>+8,.0f}$ {r['max_dd']:>8,.0f}$ "
          f"{r['total_win']:>+8,.0f}$ {r['total_loss']:>+8,.0f}$ {r['risk_adj']:>7.2f}")

print()
print(f"=== Risk-adjusted ranking (higher Rsk-Adj = better profits per unit volatility) ===\n")
ranked = sorted(results.items(), key=lambda kv: kv[1][1]["risk_adj"], reverse=True)
for i, (key, (label, r)) in enumerate(ranked, 1):
    print(f"  {i}. {label}")
    print(f"     Total ${r['total']:+,.0f} on {r['taken']} trades · "
          f"avg ${r['avg']:+.0f}/trade · {r['win_pct']:.0f}% win · "
          f"max single-trade loss ${r['worst']:,.0f} · max drawdown ${r['max_dd']:,.0f}")
    print(f"     Wins totalled ${r['total_win']:+,.0f} · losses totalled ${r['total_loss']:+,.0f}")
    print()

# ── Net-after-losses comparison ─────────────────────────────────────────────
print("=== Net cash if we follow the strategy (sum of P&L) ===")
for key in [k for _, k in strategies]:
    label, r = results[key]
    pf = (r["total_win"] / abs(r["total_loss"])) if r["total_loss"] < 0 else float("inf")
    print(f"  {label:<24}  net: {r['total']:>+9,.0f}$  · profit factor: {pf:.2f}")
