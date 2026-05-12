"""Monte Carlo simulation of model-driven 0DTE option strategies.

Question being answered:
    Starting with $500, what is the probability of reaching $5,000 by day N
    under each strategy, and what does the equity-curve distribution look like?

Two strategies, both filtered by the volatility classifier:

  CALM_SELL  — when P(vol) is in the bottom of the historical scores, sell
               a 0DTE iron condor. Defined-risk, high win-rate, small wins.
               Maps to "premium-selling environment."

  HOT_BUY    — when P(vol) is in the top of the historical scores, buy
               a 0DTE ATM straddle. Defined-loss (premium), unbounded gain.
               Maps to "premium-buying environment."

  COMBINED   — do both, depending on tier.

Iron-condor model (defined risk):
    short_dist = open_ * short_dist_pct        # short strikes ± 0.5% by default
    wing_width = open_ * wing_pct              # long wings 1.0% past short
    credit     = wing_width * credit_frac      # ~30% of wing collected
    max_loss   = wing_width - credit
    pnl_per_share = credit - clip(|close-open| - short_dist, 0, wing_width)
    return_per_unit_risked = pnl_per_share / max_loss   # in [-1, +credit_frac/(1-credit_frac)]

Straddle model (debit, hold to close):
    premium  = open_ * premium_pct             # ~0.7% of spot
    payoff   = |close - open|                  # exercise value at close
    pnl_per_share = payoff - premium
    return_per_unit_risked = pnl_per_share / premium    # in [-1, +infty]

Sizing:
    For each trade, risk a fixed fraction `risk_frac` of current equity.
    Equity update: equity_{t+1} = equity_t * (1 + risk_frac * return_t)
    `risk_frac = 0.05` ≈ "5% per trade" (Kelly-conservative for our edge).

Monte Carlo:
    Bootstrap (sample with replacement) from the historical sequence of trade
    returns. For each path, record time-to-target, max drawdown, and final equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


Strategy = Literal["calm_sell", "hot_buy", "directional_buy", "combined"]


# ---------------------------------------------------------------------------
# Per-trade P&L models (return per dollar risked)
# ---------------------------------------------------------------------------


def ic_return(
    open_: float,
    high: float,
    low: float,
    close: float,
    short_dist_pct: float = 0.007,
    wing_pct: float = 0.010,
    credit_frac: float = 0.13,
    cost_pct: float = 0.06,
) -> float:
    """0DTE iron-condor return per dollar of max loss risked.

    Conservative model: if the underlying TOUCHES the long wing at any point
    during the session (using the day's high/low), assume max loss — by the
    time price came back the IC was already deep negative due to gamma /
    assignment risk and you would have managed out at max-loss-equivalent.

    Realistic defaults:
      short_dist_pct = 0.007  → short strikes ~10-delta on a typical SPY day
      wing_pct       = 0.010  → $7-wide wings on $700 SPY
      credit_frac    = 0.18   → credit is ~18% of wing width
      cost_pct       = 0.04   → 4% drag from bid-ask + commissions
    """
    short_dist = open_ * short_dist_pct
    wing_width = open_ * wing_pct
    credit = wing_width * credit_frac
    max_loss = wing_width - credit
    if max_loss <= 0:
        return -1.0

    long_wing_dist = short_dist + wing_width
    intraday_max_excursion = max(high - open_, open_ - low)

    if intraday_max_excursion >= long_wing_dist:
        pnl = -max_loss
    else:
        close_move = abs(close - open_)
        if close_move <= short_dist:
            pnl = credit
        elif close_move >= long_wing_dist:
            pnl = -max_loss
        else:
            pnl = credit - (close_move - short_dist)

    raw = pnl / max_loss
    return float(max(raw - cost_pct, -1.0))


def straddle_return(
    open_: float,
    high: float,
    low: float,
    close: float,
    premium_pct: float = 0.011,
    cost_pct: float = 0.06,
) -> float:
    """0DTE ATM straddle return per dollar of premium paid.

    Realistic management: long straddle benefits from any intraday excursion;
    we assume you scalp 50% of the day's best directional move (not the full
    high-to-low which would be cherry-picking).
    """
    premium = open_ * premium_pct
    if premium <= 0:
        return -1.0
    intraday_max_excursion = max(high - open_, open_ - low)
    close_move = abs(close - open_)
    payoff = max(close_move, intraday_max_excursion * 0.5)
    raw = (payoff - premium) / premium
    return float(max(raw - cost_pct, -1.0))


def weekly_straddle_return(
    open_: float,
    high_window: float,
    low_window: float,
    close_window: float,
    premium_pct: float = 0.025,
    cost_pct: float = 0.06,
) -> float:
    """Long weekly straddle held over a 5-trading-day window from entry.

    Premium is ~2.5% of spot for SPY weeklies (much higher than 0DTE
    because of additional time value). Realistic management:

    - Best plausible exit captures ~55% of the max favorable excursion
      observed during the holding window (we have 5 days to time the
      exit instead of 1, slightly better than 0DTE's 50% factor but not
      perfect because theta is paying down by the time you peak).
    - Held-to-expiry alternative: payoff = |close_friday - open_monday|
      with a 30% theta haircut.
    - Floor at -1.0 (premium is the maximum loss).

    All inputs are levels (dollars), not returns:
        open_           — spot at entry
        high_window     — max of high over 5 trading days
        low_window      — min of low over 5 trading days
        close_window    — close on day 5
    """
    premium = open_ * premium_pct
    if premium <= 0:
        return -1.0
    favorable = max(high_window - open_, open_ - low_window)
    close_move = abs(close_window - open_)
    payoff = max(close_move * 0.7, favorable * 0.55)
    raw = (payoff - premium) / premium
    return float(max(raw - cost_pct, -1.0))


def directional_return(
    open_: float,
    high: float,
    low: float,
    close: float,
    direction: int,                       # +1 = bought a CALL, -1 = bought a PUT
    premium_pct: float = 0.006,
    cost_pct: float = 0.05,
) -> float:
    """0DTE single-leg long call/put return per dollar of premium paid.

    Premium for a single leg is ~half a straddle (default 0.6% of spot).
    Payoff: directional capture of the high (call) or low (put), discounted
    50% to model that you can't perfectly time the exit.
    """
    if direction not in (1, -1):
        return 0.0
    premium = open_ * premium_pct
    if premium <= 0:
        return -1.0
    if direction == 1:
        favorable_excursion = max(high - open_, 0.0)
        close_payoff = max(close - open_, 0.0)
    else:
        favorable_excursion = max(open_ - low, 0.0)
        close_payoff = max(open_ - close, 0.0)
    payoff = max(close_payoff, favorable_excursion * 0.5)
    raw = (payoff - premium) / premium
    return float(max(raw - cost_pct, -1.0))


# ---------------------------------------------------------------------------
# Per-day signal -> trade returns
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    strategy: Strategy = "calm_sell"
    calm_threshold: float = 0.10   # P(vol) below this → CALM signal (sell premium)
    hot_threshold: float = 0.40    # P(vol) above this → HOT signal (buy premium)

    # IC pricing
    short_dist_pct: float = 0.005
    wing_pct: float = 0.010
    credit_frac: float = 0.30

    # Straddle pricing
    premium_pct: float = 0.007

    # Single-leg directional (for "directional_buy"): premium per leg
    directional_premium_pct: float = 0.006


@dataclass
class TradeStats:
    n_signals: int
    n_trades_per_year: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expected_return_per_unit: float   # mean return per dollar risked
    median_return_per_unit: float
    pct_zero_loss: float


def compute_per_day_returns(
    daily: pd.DataFrame,
    p_vol: pd.Series,
    cfg: StrategyConfig,
    direction_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """For every aligned trading day, return ret per dollar risked (or 0 if no trade).

    This preserves the FULL chronological sequence so historical-walk bootstraps
    keep regime correlations (vol clusters, drawdowns) intact.

    direction_signal: optional series of +1 / -1 / 0 per date, used by the
        ``directional_buy`` strategy to pick CALL (+1) or PUT (-1).
    """
    common = sorted(daily.index.intersection(p_vol.index))
    if len(common) == 0:
        return pd.DataFrame(columns=["p_vol", "side", "ret"])

    rows = []
    for date in common:
        bar = daily.loc[date]
        p = float(p_vol.loc[date])
        open_ = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        side = "NONE"
        ret = 0.0
        if cfg.strategy in ("calm_sell", "combined") and p < cfg.calm_threshold:
            side = "SELL_IC"
            ret = ic_return(open_, high, low, close, cfg.short_dist_pct, cfg.wing_pct, cfg.credit_frac)
        elif cfg.strategy in ("hot_buy", "combined") and p > cfg.hot_threshold:
            side = "BUY_STRADDLE"
            ret = straddle_return(open_, high, low, close, cfg.premium_pct)
        elif cfg.strategy == "directional_buy" and p > cfg.hot_threshold:
            d = 0
            if direction_signal is not None and date in direction_signal.index:
                d = int(np.sign(direction_signal.loc[date]))
            if d == 0:
                # No direction conviction — fall back to a straddle so we still play vol
                side = "BUY_STRADDLE"
                ret = straddle_return(open_, high, low, close, cfg.premium_pct)
            else:
                side = "BUY_CALL" if d == 1 else "BUY_PUT"
                ret = directional_return(open_, high, low, close, d, cfg.directional_premium_pct)

        rows.append({"date": date, "p_vol": p, "side": side, "ret": float(ret)})

    return pd.DataFrame(rows).set_index("date").sort_index()


# Backward-compatible alias for trade-day-only returns (used for stats)
def compute_trade_returns(
    daily: pd.DataFrame,
    p_vol: pd.Series,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Trade days only (filters out NONE rows from compute_per_day_returns)."""
    df = compute_per_day_returns(daily, p_vol, cfg)
    return df[df["side"] != "NONE"].copy()


def trade_stats(returns: pd.Series, n_years: float) -> TradeStats:
    n = len(returns)
    if n == 0 or n_years <= 0:
        return TradeStats(0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return TradeStats(
        n_signals=int(n),
        n_trades_per_year=float(n / n_years),
        win_rate=float((returns > 0).mean()),
        avg_win=float(wins.mean()) if len(wins) > 0 else 0.0,
        avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
        expected_return_per_unit=float(returns.mean()),
        median_return_per_unit=float(returns.median()),
        pct_zero_loss=float((returns == 0).mean()),
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class MCConfig:
    n_sims: int = 2000
    horizon_days: int = 252       # one year of trading days
    risk_frac: float = 0.05        # fraction of equity risked per trade
    trades_per_day: int = 1        # cash account, PDT-friendly
    start_equity: float = 500.0
    target_equity: float = 5_000.0
    floor_equity: float = 25.0     # bust threshold (below this we mark "ruined")
    seed: int | None = 42


@dataclass
class MCSummary:
    config: MCConfig
    n_paths: int

    pct_hit_target: float
    pct_ruined: float

    median_final: float
    p25_final: float
    p75_final: float
    p05_final: float
    p95_final: float

    median_max_dd: float
    p25_max_dd: float
    p75_max_dd: float

    median_time_to_target: float | None
    p25_time_to_target: float | None
    p75_time_to_target: float | None

    pct_hit_by_day: list[dict] = field(default_factory=list)
    median_curve: list[float] = field(default_factory=list)
    p25_curve: list[float] = field(default_factory=list)
    p75_curve: list[float] = field(default_factory=list)
    p05_curve: list[float] = field(default_factory=list)
    p95_curve: list[float] = field(default_factory=list)


def monte_carlo(
    per_day_returns: pd.Series,
    signal_density: float,  # kept for backwards compat / reporting; unused
    cfg: MCConfig,
) -> MCSummary:
    """Random-historical-walk simulation of the equity-curve distribution.

    Each path = a contiguous slice of the FULL per-day return series starting
    at a random date. This preserves vol-regime clustering (e.g. 2008, 2020)
    so streaks and drawdowns reflect what really happened. Avoids the IID-
    bootstrap pathology of compounding diverging when EV is positive.

    If the historical series is too short for the requested horizon, falls
    back to wrap-around sampling.
    """
    del signal_density  # only used in reporting metadata
    rng = np.random.default_rng(cfg.seed)
    arr = per_day_returns.to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return MCSummary(cfg, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None)

    horizon = cfg.horizon_days
    n_sims = cfg.n_sims
    f = cfg.risk_frac * cfg.trades_per_day

    if n < horizon + 1:
        # Wrap-around sampling — uncommon but handle gracefully
        starts = rng.integers(0, n, size=n_sims)
        samples = np.array([np.take(arr, range(s, s + horizon), mode="wrap") for s in starts])
    else:
        starts = rng.integers(0, n - horizon, size=n_sims)
        samples = np.array([arr[s:s + horizon] for s in starts])

    multipliers = 1.0 + f * samples
    multipliers = np.clip(multipliers, 0.0, None)
    eq = cfg.start_equity * np.cumprod(multipliers, axis=1)
    eq = np.maximum(eq, 0.01)

    # Hit-target index per path (first day equity >= target). -1 means never.
    hit_mask = eq >= cfg.target_equity
    first_hit = np.where(hit_mask.any(axis=1), hit_mask.argmax(axis=1), -1)

    # Final equity is whatever we have at horizon (or at target if hit, then we'd stop;
    # but we don't model "stop on target" since the question is "by day N").
    final = eq[:, -1]

    # Drawdown per path.
    running_max = np.maximum.accumulate(eq, axis=1)
    dd = (eq / running_max) - 1.0
    max_dd = dd.min(axis=1)

    # Bust = ever fell below floor.
    bust_mask = (eq <= cfg.floor_equity).any(axis=1)
    pct_ruined = float(bust_mask.mean())

    # Hit fraction by day (vector).
    hit_by_day = (eq >= cfg.target_equity).cumsum(axis=1) > 0
    pct_hit_curve = hit_by_day.mean(axis=0)

    pct_hit_by_day = [
        {"day": int(d), "pct": float(pct_hit_curve[d])}
        for d in range(0, horizon, max(1, horizon // 60))
    ]
    if pct_hit_by_day[-1]["day"] != horizon - 1:
        pct_hit_by_day.append({"day": int(horizon - 1), "pct": float(pct_hit_curve[-1])})

    # Equity-curve percentile bands
    qs = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)
    median_curve = qs[2].tolist()
    p25_curve = qs[1].tolist()
    p75_curve = qs[3].tolist()
    p05_curve = qs[0].tolist()
    p95_curve = qs[4].tolist()

    hit_paths = first_hit[first_hit >= 0]
    if len(hit_paths) > 0:
        med_t = float(np.median(hit_paths))
        p25_t = float(np.percentile(hit_paths, 25))
        p75_t = float(np.percentile(hit_paths, 75))
    else:
        med_t = p25_t = p75_t = None

    return MCSummary(
        config=cfg,
        n_paths=n_sims,
        pct_hit_target=float((first_hit >= 0).mean()),
        pct_ruined=pct_ruined,
        median_final=float(np.median(final)),
        p25_final=float(np.percentile(final, 25)),
        p75_final=float(np.percentile(final, 75)),
        p05_final=float(np.percentile(final, 5)),
        p95_final=float(np.percentile(final, 95)),
        median_max_dd=float(np.median(max_dd)),
        p25_max_dd=float(np.percentile(max_dd, 25)),
        p75_max_dd=float(np.percentile(max_dd, 75)),
        median_time_to_target=med_t,
        p25_time_to_target=p25_t,
        p75_time_to_target=p75_t,
        pct_hit_by_day=pct_hit_by_day,
        median_curve=median_curve,
        p25_curve=p25_curve,
        p75_curve=p75_curve,
        p05_curve=p05_curve,
        p95_curve=p95_curve,
    )
