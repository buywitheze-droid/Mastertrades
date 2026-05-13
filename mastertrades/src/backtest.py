"""Real historical backtest: run the Jackpot algo on actual past data.

Uses the same walk-forward expanding-window approach that all OOS stats in
this codebase use — the model NEVER sees a future date during training, so
every probability is genuinely out-of-sample.

Signal classification (mirrors live Command Center):
    GO_JACKPOT  — p_vol >= HOT_THRESHOLD  AND  p_pnl >= JACKPOT_THRESHOLD
    GO_HOT      — p_vol >= HOT_THRESHOLD  (p_pnl below threshold)
    SKIP        — p_vol < HOT_THRESHOLD

Trade execution:
    On JACKPOT days  → buy 0DTE ATM straddle, risk JACKPOT_ALLOC_PCT of equity
    On HOT days      → buy 0DTE ATM straddle, risk HOT_ALLOC_PCT of equity
    Actual straddle P&L uses real OHLCV (same model as strategy_sim.py).

Equity compounds daily:
    equity_{t+1} = equity_t + alloc_dollars * straddle_ret
    (floored at 0 to model total loss of premium)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .jackpot_scanner import (
    HOT_THRESHOLD,
    JACKPOT_THRESHOLD,
    estimate_premium_pct,
    DEFAULT_MODEL_DIR,
    _safe_ticker_filename,
)
from .scanner import fetch_or_load_daily
from .volatility_classifier import prepare_xy, walk_forward_proba, make_logreg
from .volatility_patterns import build_features
from .edge_finder import train_direct_pnl_model
from .strategy_sim import straddle_return, directional_return
from .regime_features import build_regime_features


# Kelly allocation fractions (matches TIERS in position_sizer.py)
JACKPOT_ALLOC_PCT = 0.25
HOT_ALLOC_PCT     = 0.15

BACKTEST_CACHE_AGE_DAYS = 7   # re-run if cache is stale

# ── Gap-fade strategy parameters ────────────────────────────────────────────
GAP_THRESHOLD_PCT      = 0.0025   # |gap| >= 0.25% → fade with directional
GAP_MAX_FADE_PCT       = 0.015    # gaps > 1.5% are usually news/trend, NOT fades
DIRECTIONAL_PREMIUM_PCT = 0.006   # single-leg ~0.6% of spot
TREND_LOOKBACK_DAYS    = 5        # measure 5-day trend to validate fade direction

# ── Risk-management parameters ──────────────────────────────────────────────
CIRCUIT_BREAKER_LOSSES = 3        # stop after this many consecutive losses
CIRCUIT_BREAKER_PAUSE  = 5        # pause this many sessions before resuming

# ── Smart-v2 regime filter thresholds (Tier 1 indicators) ──────────────────
VIX_BLOWOUT_5D_CHG     = 0.15     # skip if VIX up >15% in 5 days (IV crush risk)
EXTREME_Z_DOWN         = -2.5     # skip if SPY > 2.5σ below 20d mean (panic exhausted)
DIRECTIONAL_PROB_HI    = 0.60     # P(up) above this → single-leg CALL
DIRECTIONAL_PROB_LO    = 0.40     # P(up) below this → single-leg PUT
DIRECTIONAL_ALLOC_BOOST = 1.30    # boost alloc on high-conviction directional


@dataclass
class BacktestResult:
    ticker:     str
    start_date: str
    end_date:   str
    n_days:     int          # total trading days in window
    n_trades:   int          # days where a signal fired
    n_jackpot:  int
    n_hot:      int
    n_wins:     int
    n_losses:   int
    win_rate:   float
    start_equity: float
    end_equity:   float
    max_equity:   float
    min_equity:   float
    max_drawdown_pct: float  # peak-to-trough in the window
    trade_log:  pd.DataFrame # one row per trading day (including SKIP days)
    strategy_mode: str = "straddle"   # "straddle" | "gapfade" | "smart" | "smart_v2"
    n_gap_fades:  int = 0             # days routed to directional fade
    n_straddles:  int = 0             # days kept as straddle
    n_breaker_skips: int = 0          # signals skipped by circuit-breaker
    n_regime_skips:  int = 0          # signals skipped by regime filter (smart_v2)
    n_directional:   int = 0          # signals routed to single-leg directional


def _cache_path(ticker: str, n_months: int, mode: str = "straddle") -> Path:
    suffix = "" if mode == "straddle" else f"_{mode}"
    return DEFAULT_MODEL_DIR / f"{_safe_ticker_filename(ticker)}_backtest_{n_months}m{suffix}.joblib"


def run_jackpot_backtest(
    ticker: str = "SPY",
    start_equity: float = 500.0,
    n_months: int = 6,
    force_refresh: bool = False,
    strategy_mode: str = "straddle",
) -> BacktestResult:
    """Run the historical Jackpot backtest for the last ``n_months`` months.

    Results are cached in models/ so subsequent calls are instant.
    Pass ``force_refresh=True`` to re-run the walk-forward (slow ~30-60 s).

    strategy_mode:
        "straddle" — baseline: every HOT/JACKPOT day buys an ATM 0DTE
                     straddle. Premium ~1.1% of spot.
        "gapfade"  — naive gap fade: any HOT day with |gap| >= 0.25% buys a
                     CALL (gap-down) or PUT (gap-up). Otherwise straddle.
        "smart"    — three layered upgrades:
                     1) Trend-aware gap fade: only fade gaps in the SWEET
                        SPOT (0.25%–1.5%) AND only when the gap direction is
                        OPPOSITE to the 5-day trend (counter-trend fade).
                        Big gaps (>1.5%) and same-direction gaps default to
                        straddle.
                     2) Circuit breaker: pause for 5 sessions after 3
                        consecutive losses. Stops the bleed in trending
                        regimes like March 2026.
                     3) Otherwise reverts to straddle (preserves edge on
                        no-gap volatile days).
    """
    path = _cache_path(ticker, n_months, strategy_mode)
    if path.exists() and not force_refresh:
        age_days = (
            pd.Timestamp.now() - pd.Timestamp.fromtimestamp(path.stat().st_mtime)
        ).days
        if age_days < BACKTEST_CACHE_AGE_DAYS:
            cached = joblib.load(path)
            # Re-scale equity if starting balance changed
            if cached.start_equity != start_equity:
                scale = start_equity / cached.start_equity
                cached = _rescale(cached, scale, start_equity)
            return cached

    result = _run_fresh(ticker, start_equity, n_months, strategy_mode)
    joblib.dump(result, path)
    return result


def _rescale(r: BacktestResult, scale: float, new_start: float) -> BacktestResult:
    """Rescale a cached result to a different starting equity."""
    log = r.trade_log.copy()
    for col in ("alloc_dollars", "dollar_change", "equity"):
        if col in log.columns:
            log[col] = log[col] * scale
    return BacktestResult(
        ticker=r.ticker,
        start_date=r.start_date,
        end_date=r.end_date,
        n_days=r.n_days,
        n_trades=r.n_trades,
        n_jackpot=r.n_jackpot,
        n_hot=r.n_hot,
        n_wins=r.n_wins,
        n_losses=r.n_losses,
        win_rate=r.win_rate,
        start_equity=new_start,
        end_equity=r.end_equity * scale,
        max_equity=r.max_equity * scale,
        min_equity=r.min_equity * scale,
        max_drawdown_pct=r.max_drawdown_pct,
        trade_log=log,
        strategy_mode=getattr(r, "strategy_mode", "straddle"),
        n_gap_fades=getattr(r, "n_gap_fades", 0),
        n_straddles=getattr(r, "n_straddles", 0),
        n_breaker_skips=getattr(r, "n_breaker_skips", 0),
        n_regime_skips=getattr(r, "n_regime_skips", 0),
        n_directional=getattr(r, "n_directional", 0),
    )


def _walk_forward_directional(
    X: pd.DataFrame,
    daily: pd.DataFrame,
    min_train: int = 500,
    step: int = 21,
) -> pd.DataFrame:
    """Walk-forward classifier predicting P(close > open) per session.

    Reuses the same X feature matrix as the volatility classifier so we get
    a directional probability with no extra feature engineering. Returns a
    DataFrame indexed by date with columns: y_true, y_score (= P(up)).
    """
    common = X.index.intersection(daily.index)
    X = X.loc[common]
    direction = ((daily["Close"] - daily["Open"]) > 0).astype(int).loc[common]
    direction.name = "y_true"
    return walk_forward_proba(X, direction, make_logreg, min_train=min_train, step=step)


def _run_fresh(
    ticker: str,
    start_equity: float,
    n_months: int,
    strategy_mode: str = "straddle",
) -> BacktestResult:
    """Full walk-forward backtest. Slow path (~30–60 s first run)."""

    # ── 1. Load OHLCV ────────────────────────────────────────────────────────
    daily = fetch_or_load_daily(ticker)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # ── 2. Feature engineering ───────────────────────────────────────────────
    feats = build_features(daily)
    premium_pct = estimate_premium_pct(daily)

    # ── 2b. Regime features (Tier 1: VIX context, z-score, RSI(2)) ───────────
    # Used by smart_v2 as a hard-rule overlay to skip post-blowout / panic days.
    regime = build_regime_features(daily)

    # ── 3. Vol-classifier walk-forward (OOS p_vol for every date) ────────────
    X_vol, y_vol, _ = prepare_xy(feats, volatile_quantile=0.80)
    # min_train=500 gives ~14+ months of OOS predictions on a 5-year dataset
    vol_preds = walk_forward_proba(X_vol, y_vol, make_logreg, min_train=500, step=21)
    # vol_preds has columns: y_true, y_score (= p_vol)

    # ── 4. Direct-PnL classifier walk-forward (OOS p_pnl + actual straddle ret)
    pnl_result = train_direct_pnl_model(
        feats, daily, vol_preds["y_score"], premium_pct=premium_pct,
        min_train=500, step=21,
    )
    # pnl_result.preds_oos has columns: y_true, y_score (= p_pnl), straddle_ret

    # ── 4b. Directional classifier walk-forward (OOS P(up)) ──────────────────
    # Predicts whether today closes above open. Used by smart_v2 to route to
    # single-leg call/put (~1000% upside) instead of straddles when conviction
    # is high. Only computed for smart_v2 to keep other modes' cache hot.
    if strategy_mode == "smart_v2":
        dir_preds = _walk_forward_directional(X_vol, daily, min_train=500, step=21)
    else:
        dir_preds = None

    # ── 5. Align the prediction series ───────────────────────────────────────
    pnl_preds = pnl_result.preds_oos
    common = sorted(vol_preds.index.intersection(pnl_preds.index).intersection(daily.index))
    if len(common) == 0:
        raise ValueError(f"No overlapping dates for {ticker} backtest.")

    # ── 6. Extract last n_months of trading days ──────────────────────────────
    approx_days = n_months * 21
    bt_dates = common[-approx_days:]

    # ── 7. Day-by-day simulation ─────────────────────────────────────────────
    equity      = start_equity
    rows: list[dict] = []
    n_gap_fades = 0
    n_straddles = 0
    n_breaker_skips = 0
    n_regime_skips = 0
    n_directional  = 0

    # Pre-compute previous close + 5-day trend for gap-fade routing
    prev_close_series = daily["Close"].shift(1)
    trend_pct_series  = daily["Close"].pct_change(TREND_LOOKBACK_DAYS).shift(1)

    # Circuit-breaker state
    consecutive_losses = 0
    pause_remaining    = 0

    for date in bt_dates:
        p_vol  = float(vol_preds.loc[date, "y_score"])
        p_pnl  = float(pnl_preds.loc[date, "y_score"])
        straddle_ret_actual = float(pnl_preds.loc[date, "straddle_ret"])

        bar    = daily.loc[date]
        open_  = float(bar["Open"])
        close_ = float(bar["Close"])
        high_  = float(bar["High"])
        low_   = float(bar["Low"])
        day_rng_pct = (high_ - low_) / open_ * 100

        prev_close = prev_close_series.loc[date]
        gap_pct = (open_ - float(prev_close)) / float(prev_close) if pd.notna(prev_close) and prev_close > 0 else 0.0
        trend_pct = float(trend_pct_series.loc[date]) if pd.notna(trend_pct_series.loc[date]) else 0.0

        # Signal classification
        if p_vol >= HOT_THRESHOLD and p_pnl >= JACKPOT_THRESHOLD:
            signal      = "GO_JACKPOT"
            alloc_pct   = JACKPOT_ALLOC_PCT
        elif p_vol >= HOT_THRESHOLD:
            signal      = "GO_HOT"
            alloc_pct   = HOT_ALLOC_PCT
        else:
            signal      = "SKIP"
            alloc_pct   = 0.0

        # Regime context (only used by smart_v2)
        if strategy_mode == "smart_v2" and date in regime.index:
            r = regime.loc[date]
            vix_val      = float(r["vix"])      if pd.notna(r["vix"])      else float("nan")
            vix_5d_chg   = float(r["vix_5d_chg"]) if pd.notna(r["vix_5d_chg"]) else 0.0
            vix_term_val = float(r["vix_term"])  if pd.notna(r["vix_term"])  else float("nan")
            z_20d_val    = float(r["z_20d"])     if pd.notna(r["z_20d"])     else 0.0
            rsi2_val     = float(r["rsi2"])      if pd.notna(r["rsi2"])      else 50.0
        else:
            vix_val = vix_5d_chg = vix_term_val = z_20d_val = rsi2_val = float("nan")

        # P(up) for smart_v2 directional routing
        if dir_preds is not None and date in dir_preds.index:
            p_up = float(dir_preds.loc[date, "y_score"])
        else:
            p_up = 0.5

        # Trade routing: straddle / gap-fade / smart / smart_v2
        trade_type = "NONE"
        actual_ret = 0.0

        # ---- Circuit breaker (smart & smart_v2) ────────────────────────────
        if signal != "SKIP" and strategy_mode in ("smart", "smart_v2") and pause_remaining > 0:
            trade_type = "BREAKER_SKIP"
            n_breaker_skips += 1
            pause_remaining -= 1
            actual_ret = 0.0
            alloc_pct  = 0.0
            signal     = "SKIP"   # treat as a skip for accounting

        # ---- Hard regime filter (smart_v2 only) ────────────────────────────
        elif signal != "SKIP" and strategy_mode == "smart_v2" and (
            vix_5d_chg > VIX_BLOWOUT_5D_CHG or z_20d_val < EXTREME_Z_DOWN
        ):
            # VIX already blew out 5d → IV crush risk; OR price already
            # capitulated 2.5σ below mean → panic mostly priced in.
            trade_type = "REGIME_SKIP"
            n_regime_skips += 1
            actual_ret = 0.0
            alloc_pct  = 0.0
            signal     = "SKIP"

        elif signal != "SKIP":
            do_fade = False
            if strategy_mode == "gapfade":
                # Naive gap-fade: any meaningful gap
                do_fade = abs(gap_pct) >= GAP_THRESHOLD_PCT
            elif strategy_mode in ("smart", "smart_v2"):
                # Trend-aware gap-fade: counter-trend, sweet-spot only
                in_sweet_spot = GAP_THRESHOLD_PCT <= abs(gap_pct) <= GAP_MAX_FADE_PCT
                # Counter-trend: gap is opposite to recent direction
                # (gap up after downtrend = exhaustion; gap down after uptrend = panic)
                counter_trend = (gap_pct > 0 and trend_pct < 0) or (gap_pct < 0 and trend_pct > 0)
                do_fade = in_sweet_spot and counter_trend

            # ---- smart_v2: directional override ────────────────────────────
            # High-conviction directional plays take priority over the
            # gap-fade heuristic — single-leg call/put is asymmetric upside,
            # while gap-fade is a lower-conviction mean-reversion bet.
            took_directional = False
            if strategy_mode == "smart_v2":
                if p_up >= DIRECTIONAL_PROB_HI:
                    direction = 1
                    actual_ret = directional_return(
                        open_, high_, low_, close_, direction,
                        premium_pct=DIRECTIONAL_PREMIUM_PCT,
                    )
                    trade_type = "DIR_CALL"
                    alloc_pct  = min(alloc_pct * DIRECTIONAL_ALLOC_BOOST, JACKPOT_ALLOC_PCT)
                    n_directional += 1
                    took_directional = True
                elif p_up <= DIRECTIONAL_PROB_LO:
                    direction = -1
                    actual_ret = directional_return(
                        open_, high_, low_, close_, direction,
                        premium_pct=DIRECTIONAL_PREMIUM_PCT,
                    )
                    trade_type = "DIR_PUT"
                    alloc_pct  = min(alloc_pct * DIRECTIONAL_ALLOC_BOOST, JACKPOT_ALLOC_PCT)
                    n_directional += 1
                    took_directional = True

            if not took_directional:
                if do_fade:
                    direction = -1 if gap_pct > 0 else 1
                    actual_ret = directional_return(
                        open_, high_, low_, close_, direction,
                        premium_pct=DIRECTIONAL_PREMIUM_PCT,
                    )
                    trade_type = "FADE_PUT" if direction == -1 else "FADE_CALL"
                    n_gap_fades += 1
                else:
                    actual_ret = straddle_ret_actual
                    trade_type = "STRADDLE"
                    n_straddles += 1

        alloc_dollars = round(equity * alloc_pct, 2)
        dollar_change = round(alloc_dollars * actual_ret, 2) if signal != "SKIP" else 0.0
        equity        = max(round(equity + dollar_change, 2), 0.0)

        outcome = (
            "WIN"     if signal != "SKIP" and dollar_change > 0 else
            "LOSS"    if signal != "SKIP" and dollar_change < 0 else
            "PAUSED"  if trade_type == "BREAKER_SKIP" else
            "SKIP"
        )

        # Update circuit-breaker state based on actual outcome
        if strategy_mode in ("smart", "smart_v2"):
            if outcome == "LOSS":
                consecutive_losses += 1
                if consecutive_losses >= CIRCUIT_BREAKER_LOSSES and pause_remaining == 0:
                    pause_remaining = CIRCUIT_BREAKER_PAUSE
                    consecutive_losses = 0
            elif outcome == "WIN":
                consecutive_losses = 0

        rows.append({
            "date":          date,
            "signal":        signal,
            "trade_type":    trade_type,
            "p_vol":         round(p_vol,  3),
            "p_pnl":         round(p_pnl,  3),
            "gap_pct":       round(gap_pct * 100, 2),
            "trend_pct":     round(trend_pct * 100, 2),
            "open":          round(open_,  2),
            "close":         round(close_, 2),
            "range_pct":     round(day_rng_pct, 2),
            "straddle_ret":  round(actual_ret, 3),
            "p_up":          round(p_up, 3),
            "vix":           round(vix_val, 2)      if pd.notna(vix_val)      else None,
            "vix_5d_chg_%":  round(vix_5d_chg*100, 1) if pd.notna(vix_5d_chg) else None,
            "vix_term":      round(vix_term_val, 3) if pd.notna(vix_term_val) else None,
            "z_20d":         round(z_20d_val, 2)    if pd.notna(z_20d_val)    else None,
            "rsi2":          round(rsi2_val, 1)     if pd.notna(rsi2_val)     else None,
            "alloc_pct":     alloc_pct,
            "alloc_dollars": alloc_dollars,
            "dollar_change": dollar_change,
            "equity":        equity,
            "outcome":       outcome,
        })

    trade_log = pd.DataFrame(rows)
    if trade_log.empty:
        raise ValueError(f"Backtest produced no rows for {ticker}.")

    trade_log["date"] = pd.to_datetime(trade_log["date"])

    # ── 8. Compute summary stats ──────────────────────────────────────────────
    trades = trade_log[trade_log["signal"] != "SKIP"]
    wins   = trades[trades["outcome"] == "WIN"]
    losses = trades[trades["outcome"] == "LOSS"]

    equity_series = trade_log["equity"]
    running_max   = equity_series.cummax()
    drawdowns     = (equity_series - running_max) / running_max
    max_dd_pct    = float(drawdowns.min()) * 100  # negative number

    return BacktestResult(
        ticker=ticker,
        start_date=trade_log["date"].iloc[0].strftime("%Y-%m-%d"),
        end_date=trade_log["date"].iloc[-1].strftime("%Y-%m-%d"),
        n_days=len(trade_log),
        n_trades=len(trades),
        n_jackpot=int((trade_log["signal"] == "GO_JACKPOT").sum()),
        n_hot=int((trade_log["signal"] == "GO_HOT").sum()),
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate=float(len(wins) / len(trades)) if len(trades) > 0 else 0.0,
        start_equity=start_equity,
        end_equity=float(equity_series.iloc[-1]),
        max_equity=float(equity_series.max()),
        min_equity=float(equity_series.min()),
        max_drawdown_pct=max_dd_pct,
        trade_log=trade_log,
        strategy_mode=strategy_mode,
        n_gap_fades=n_gap_fades,
        n_straddles=n_straddles,
        n_breaker_skips=n_breaker_skips,
        n_regime_skips=n_regime_skips,
        n_directional=n_directional,
    )
