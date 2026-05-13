"""Per-ticker configuration for the Gap Fill signal.

Derived from a 1-year walk-forward backtest (scripts/edge_mine_gap_per_ticker.py,
2026-05-13). For each ticker we encode:
  - whether the ticker is tradeable at all
  - minimum absolute gap %
  - direction filter (any / up / down)
  - optional weekday bonus tier ("PRIME") with its own strict criteria
  - validated backtest stats so the UI can display realised edge, not just fill rate

Edge philosophy: fill rate without payoff is a vanity stat. We require:
  - profit factor >= 1.2
  - n >= 15 trades in the year
  - positive total $ P&L

QQQ failed all gates and is NOT tradeable here.
"""
from __future__ import annotations
from typing import Optional


# Validated config — see edge_mine_gap_per_ticker.py output 2026-05-13
GAP_FILL_PER_TICKER: dict[str, Optional[dict]] = {
    "SPY": {
        "min_gap_pct": 0.0020,        # 0.20%
        "dir":         "down",         # only gap-DOWN setups; gap-up puts are dead money
        "weekday":     "any",
        "backtest": {
            "n":          27,
            "fill_rate":  70.4,
            "win_rate":   77.8,
            "avg_pct":    0.1355,      # avg realised % per trade
            "pf":         2.75,
            "pnl_per_1k": 37,          # $ per year on $1k notional
        },
    },
    # QQQ — DROPPED. Best 1-yr config produced only $8 P&L on $1k notional
    # (PF 1.25, n=21). Below transaction-cost noise floor. Fill rate alone
    # was misleading; payoff is too small/asymmetric.
    "QQQ": None,
    "IWM": {
        "min_gap_pct": 0.0040,        # 0.40% — smaller IWM gaps don't pay
        "dir":         "any",
        "weekday":     "any",
        "backtest": {
            "n":          69,
            "fill_rate":  66.7,
            "win_rate":   69.6,
            "avg_pct":    0.0776,
            "pf":         1.27,
            "pnl_per_1k": 54,
        },
    },
    "AAPL": {
        "min_gap_pct": 0.0020,        # 0.20%
        "dir":         "any",
        "weekday":     "any",
        "backtest": {
            "n":          120,
            "fill_rate":  76.7,
            "win_rate":   78.3,
            "avg_pct":    0.1482,
            "pf":         1.83,
            "pnl_per_1k": 178,
        },
        # PRIME tier: AAPL on Thursdays with ≥0.20% gap.
        # 24 trades, 88% win, PF 19.5, $95 on $1k notional. The single
        # best risk-adjusted slice we found in the entire sweep.
        "prime": {
            "weekday":   "Thu",
            "min_gap":   0.0020,
            "backtest": {
                "n":          24,
                "fill_rate":  87.5,
                "win_rate":   91.7,
                "avg_pct":    0.3954,
                "pf":         19.46,
                "pnl_per_1k": 95,
            },
        },
    },
}


WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def gap_fill_decision(ticker: str, gap_pct: float, gap_dir: str,
                      weekday_idx: int) -> tuple[bool, Optional[dict], bool]:
    """Decide whether to publish a gap-fill play for `ticker`.

    Returns (is_tradeable, ticker_cfg, is_prime).
      - is_tradeable: True if the per-ticker filters pass
      - ticker_cfg: the config dict for this ticker (or None if dropped)
      - is_prime: True if this signal also qualifies for the PRIME tier
    """
    cfg = GAP_FILL_PER_TICKER.get(ticker)
    if cfg is None:
        return False, None, False
    if abs(gap_pct) < cfg["min_gap_pct"]:
        return False, cfg, False
    if cfg["dir"] != "any" and gap_dir != cfg["dir"]:
        return False, cfg, False
    if cfg["weekday"] != "any" and WEEKDAY_NAMES.get(weekday_idx) != cfg["weekday"]:
        return False, cfg, False

    is_prime = False
    prime = cfg.get("prime")
    if prime is not None:
        if (abs(gap_pct) >= prime["min_gap"]
                and WEEKDAY_NAMES.get(weekday_idx) == prime["weekday"]):
            is_prime = True

    return True, cfg, is_prime
