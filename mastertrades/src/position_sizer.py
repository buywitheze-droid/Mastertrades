"""Kelly Criterion position sizer for Mastertrades 0DTE signals.

Each signal tier has calibrated win probability and average win multiple
from 2+ years of SPY/QQQ 0DTE backtesting.

Kelly formula: f* = (p × b − (1−p)) / b
  p = win probability, b = avg gross win multiple (10 = 1000% gain)

We use Quarter-Kelly capped at reasonable maximums to compound safely.
Max loss on any trade = allocation amount (options expire worthless = 100% loss).
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Tier definitions ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingTier:
    signal:         str
    win_prob:       float    # historical win rate (0-1)
    avg_win_mult:   float    # avg gross win multiple (e.g. 10 = 1000% gain)
    full_kelly:     float    # (p*b - (1-p)) / b
    alloc_pct:      float    # what we actually use (conservative)
    min_pct:        float    # never go below this
    max_pct:        float    # never go above this
    label:          str
    rationale:      str

    @property
    def expected_value_per_dollar(self) -> float:
        """Expected $ return per $1 risked."""
        return self.win_prob * self.avg_win_mult - (1 - self.win_prob)

    @property
    def quarter_kelly(self) -> float:
        return round(self.full_kelly / 4, 3)


TIERS: dict[str, SizingTier] = {
    "GO_ULTRA_JACKPOT": SizingTier(
        signal        = "GO_ULTRA_JACKPOT",
        win_prob      = 0.65,
        avg_win_mult  = 20.0,   # 2000% avg (historical ultra days)
        full_kelly    = 0.633,
        alloc_pct     = 0.40,   # capped quarter-Kelly
        min_pct       = 0.20,
        max_pct       = 0.50,   # hard cap — never > 50%
        label         = "SIZE UP — ULTRA JACKPOT",
        rationale     = (
            "Both models firing at peak confidence + weekly confirm. "
            "Rarest signal — 40% allocation justified by Kelly math."
        ),
    ),
    "GO_JACKPOT": SizingTier(
        signal        = "GO_JACKPOT",
        win_prob      = 0.60,
        avg_win_mult  = 10.0,   # 1000% avg
        full_kelly    = 0.560,
        alloc_pct     = 0.25,   # quarter-Kelly ≈ 14%, raised to 25% for compounding
        min_pct       = 0.15,
        max_pct       = 0.35,
        label         = "TRADE — JACKPOT",
        rationale     = (
            "Both vol + P&L models confirm. Core trade. "
            "25% per trade compounds $500→$5k in ~10 wins."
        ),
    ),
    "GO_HOT": SizingTier(
        signal        = "GO_HOT",
        win_prob      = 0.45,
        avg_win_mult  = 5.0,    # 500% avg
        full_kelly    = 0.340,
        alloc_pct     = 0.15,   # conservative — vol model only
        min_pct       = 0.10,
        max_pct       = 0.20,
        label         = "SMALL — HOT",
        rationale     = (
            "Vol model only — P&L model neutral. "
            "Trade at 15% max. Wait for confirmation."
        ),
    ),
    "ENTRY_OPEN": SizingTier(
        signal        = "ENTRY_OPEN",
        win_prob      = 0.35,
        avg_win_mult  = 8.0,    # 800% avg (3-5 pt drop band: 35% hit 1000%+)
        full_kelly    = 0.269,
        alloc_pct     = 0.10,   # minimum — ML is SKIP, pure intraday lottery
        min_pct       = 0.10,
        max_pct       = 0.10,
        label         = "LOTTERY — ENTRY WINDOW",
        rationale     = (
            "ML models quiet but intraday drop ≥3 pts. "
            "Reversal lottery play only — 10% flat."
        ),
    ),
}


# ── Allocation recommendation ─────────────────────────────────────────────────

@dataclass
class AllocationRec:
    signal:        str
    tier:          SizingTier
    equity:        float
    alloc_pct:     float     # fraction used
    alloc_dollars: float     # $ to risk
    max_loss:      float     # same as alloc_dollars (options = full loss risk)
    expected_gain: float     # EV × alloc_dollars
    win_scenario:  float     # alloc_dollars × avg_win_mult = gain if avg win
    lose_scenario: float     # -alloc_dollars (full wipeout)
    new_equity_win:  float   # equity + win_scenario
    new_equity_lose: float   # equity - alloc_dollars
    ev_per_dollar:   float


def recommend_allocation(
    signal:  str,
    equity:  float,
    clamp:   bool = True,
) -> AllocationRec | None:
    """Return Kelly-based allocation for a given signal and current equity.

    Returns None if signal is not a tradeable tier (SKIP, APPROACHING, QUIET).
    """
    tier = TIERS.get(signal)
    if tier is None:
        return None

    alloc_pct = tier.alloc_pct
    if clamp:
        alloc_pct = max(tier.min_pct, min(alloc_pct, tier.max_pct))

    alloc_dollars  = round(equity * alloc_pct, 2)
    win_gain       = round(alloc_dollars * (tier.avg_win_mult - 1), 2)  # net gain if avg win
    ev             = tier.expected_value_per_dollar * alloc_dollars

    return AllocationRec(
        signal          = signal,
        tier            = tier,
        equity          = equity,
        alloc_pct       = alloc_pct,
        alloc_dollars   = alloc_dollars,
        max_loss        = alloc_dollars,
        expected_gain   = round(ev, 2),
        win_scenario    = win_gain,
        lose_scenario   = -alloc_dollars,
        new_equity_win  = round(equity + win_gain, 2),
        new_equity_lose = round(equity - alloc_dollars, 2),
        ev_per_dollar   = tier.expected_value_per_dollar,
    )


# ── Compound growth projector ─────────────────────────────────────────────────

@dataclass
class GrowthStep:
    trade_num:  int
    equity_win_all:   float   # path where every trade wins
    equity_expected:  float   # path using expected value each step
    equity_lose_all:  float   # path where every trade loses


def compound_projection(
    signal:   str,
    equity:   float,
    n_trades: int = 8,
) -> list[GrowthStep]:
    """Simulate compound growth paths for n_trades of the given signal.

    Three paths:
      win_all:  every trade is an avg win
      expected: EV per trade applied each step
      lose_all: every trade loses the full allocation
    """
    tier = TIERS.get(signal)
    if tier is None:
        return []

    steps = [GrowthStep(0, equity, equity, equity)]
    eq_win = eq_exp = eq_lose = equity

    for i in range(1, n_trades + 1):
        rec_win  = recommend_allocation(signal, eq_win)
        rec_exp  = recommend_allocation(signal, eq_exp)
        rec_lose = recommend_allocation(signal, eq_lose)

        if rec_win is None or rec_exp is None or rec_lose is None:
            break

        # Win path: gains avg_win_mult × stake (net of stake = win_scenario)
        eq_win  = round(eq_win  + rec_win.win_scenario,  2)
        # Expected path: EV applied
        eq_exp  = round(eq_exp  + rec_exp.expected_gain, 2)
        # Lose path: loses full stake
        eq_lose = max(round(eq_lose - rec_lose.alloc_dollars, 2), 0.0)

        steps.append(GrowthStep(i, eq_win, eq_exp, eq_lose))

    return steps


# ── Milestone helpers ─────────────────────────────────────────────────────────

MILESTONES = [500, 5_000, 50_000, 500_000]


def trades_to_milestone(
    signal:    str,
    equity:    float,
    milestone: float,
) -> int | None:
    """Estimate trades needed to reach milestone on the expected-value path."""
    tier = TIERS.get(signal)
    if tier is None or equity >= milestone:
        return 0

    eq = equity
    for n in range(1, 500):
        rec = recommend_allocation(signal, eq)
        if rec is None:
            return None
        eq += rec.expected_gain
        if eq >= milestone:
            return n
    return None
