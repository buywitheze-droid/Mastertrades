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
from .strategy_sim import straddle_return, directional_return, directional_return_with_scaleout
from .regime_features import build_regime_features
from .weekly_levels import compute_weekly_mas


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

# ── Smart-v3 weekly-MA + flow confluence ───────────────────────────────────
# Empirical: 50w SMA touches = 100% bounce, 30w EMA touches = 0% (in last 6mo).
# These MAs are tagged LONG/SHORT bias by build_ma_setups() in weekly_levels.
MA_TOUCH_PCT           = 0.015    # within 1.5% of MA = "in touch zone"
# Flow score thresholds chosen by 6-month sweep on SPY: at +/-20, only the
# strongest confluence trades fire (1 trade in 6mo, +3.2% vs smart_v2). Lower
# thresholds (0, 10) overtraded and underperformed; higher (40+) fired zero.
FLOW_CONFIRM_LONG      = 20
FLOW_CONFIRM_SHORT     = -20
LONG_BIAS_MAS  = ("10w SMA", "50w SMA", "50w EMA")
SHORT_BIAS_MAS = ("20w EMA", "30w EMA")

# ── Smart-v4 patient lottery ticket (scale-out at +100%) ────────────────────
# Empirical: 22% of last 6 months had ≥1.10% intraday move (the touch threshold
# for ATM 0DTE to print +100%). Of those touches, 100% net positive when
# scaling 50% at +100% and holding the other 50% to close (avg +104.6%).
# This mode ONLY trades on highest-conviction directional signals (no straddles
# / no gap-fades) and applies the 50%-scale-out rule to every trade.
SCALE_OUT_RETURN     = 1.0      # sell at +100% option return
SCALE_OUT_FRACTION   = 0.5      # sell half


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
    strategy_mode: str = "straddle"   # straddle|gapfade|smart|smart_v2|smart_v3|smart_v4
    n_gap_fades:  int = 0             # days routed to directional fade
    n_straddles:  int = 0             # days kept as straddle
    n_breaker_skips: int = 0          # signals skipped by circuit-breaker
    n_regime_skips:  int = 0          # signals skipped by regime filter (smart_v2)
    n_directional:   int = 0          # signals routed to single-leg directional
    n_ma_confluence: int = 0          # smart_v3 MA-touch + flow trades
    n_scaleout_hits: int = 0          # smart_v4 trades that touched +100% intraday


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
        n_ma_confluence=getattr(r, "n_ma_confluence", 0),
        n_scaleout_hits=getattr(r, "n_scaleout_hits", 0),
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
    # Predicts whether today closes above open. Used by smart_v2/v3 to route to
    # single-leg call/put (~1000% upside) instead of straddles when conviction
    # is high. Only computed for smart_v2/v3 to keep other modes' cache hot.
    if strategy_mode in ("smart_v2", "smart_v3", "smart_v4"):
        dir_preds = _walk_forward_directional(X_vol, daily, min_train=500, step=21)
    else:
        dir_preds = None

    # ── 4c. Weekly MA-touch + weekly order-flow overlay (smart_v3 only) ─────
    # Build per-day flags for "near a LONG-bias MA from above" and "near a
    # SHORT-bias MA from below", plus a daily-aligned weekly flow score.
    # All inputs shifted by 1 to use only information known before today's open.
    if strategy_mode in ("smart_v3", "smart_v4"):
        _mas_shifted = compute_weekly_mas(daily).shift(1)
        _prior_close = daily["Close"].shift(1)

        _near_long  = pd.Series(False, index=daily.index)
        _near_short = pd.Series(False, index=daily.index)
        for _ma_name in LONG_BIAS_MAS:
            if _ma_name in _mas_shifted.columns:
                _ma = _mas_shifted[_ma_name]
                _dist = (_prior_close - _ma).abs() / _ma
                # LONG setup: prior close above the MA but within touch range
                _near_long |= ((_prior_close >= _ma) & (_dist <= MA_TOUCH_PCT))
        for _ma_name in SHORT_BIAS_MAS:
            if _ma_name in _mas_shifted.columns:
                _ma = _mas_shifted[_ma_name]
                _dist = (_prior_close - _ma).abs() / _ma
                # SHORT setup: prior close below the MA but within touch range
                _near_short |= ((_prior_close <= _ma) & (_dist <= MA_TOUCH_PCT))

        # Weekly flow score series, aligned to daily and shifted 1 day so only
        # the prior completed week's flow is visible.
        _wf = _compute_weekly_flow_series(daily).reindex(daily.index, method="ffill").shift(1)

        ma_overlay = pd.DataFrame({
            "near_long_ma":  _near_long,
            "near_short_ma": _near_short,
            "weekly_flow":   _wf,
        })
    else:
        ma_overlay = None

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
    n_ma_confluence = 0
    n_scaleout_hits = 0

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

        # Regime context (used by smart_v2, smart_v3, smart_v4)
        if strategy_mode in ("smart_v2", "smart_v3", "smart_v4") and date in regime.index:
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

        # ---- Circuit breaker (smart, smart_v2, smart_v3, smart_v4) ─────────
        if signal != "SKIP" and strategy_mode in ("smart", "smart_v2", "smart_v3", "smart_v4") and pause_remaining > 0:
            trade_type = "BREAKER_SKIP"
            n_breaker_skips += 1
            pause_remaining -= 1
            actual_ret = 0.0
            alloc_pct  = 0.0
            signal     = "SKIP"   # treat as a skip for accounting

        # ---- Hard regime filter (smart_v2 + smart_v3 + smart_v4) ───────────
        elif signal != "SKIP" and strategy_mode in ("smart_v2", "smart_v3", "smart_v4") and (
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
            elif strategy_mode in ("smart", "smart_v2", "smart_v3", "smart_v4"):
                # Trend-aware gap-fade: counter-trend, sweet-spot only
                in_sweet_spot = GAP_THRESHOLD_PCT <= abs(gap_pct) <= GAP_MAX_FADE_PCT
                # Counter-trend: gap is opposite to recent direction
                # (gap up after downtrend = exhaustion; gap down after uptrend = panic)
                counter_trend = (gap_pct > 0 and trend_pct < 0) or (gap_pct < 0 and trend_pct > 0)
                do_fade = in_sweet_spot and counter_trend

            # ---- smart_v3: weekly MA-touch + flow CONFLUENCE override ─────
            # Highest priority: when SPY is in a tagged MA-touch zone AND
            # weekly order flow agrees, override every other signal with a
            # JACKPOT-tier directional play. This is the empirical "deep buy"
            # / "stretched short" setup the user asked for.
            took_directional = False
            took_ma_confluence = False

            # smart_v4 uses scale-out variant; v2/v3 use plain directional
            def _dir_ret(d):
                if strategy_mode == "smart_v4":
                    r = directional_return_with_scaleout(
                        open_, high_, low_, close_, d,
                        premium_pct=DIRECTIONAL_PREMIUM_PCT,
                        scale_at_return=SCALE_OUT_RETURN,
                        scale_fraction=SCALE_OUT_FRACTION,
                    )
                    # Did the trade touch the scale-out trigger? Used for stats.
                    if d == 1:
                        peak_int = max(high_ - open_, 0.0)
                    else:
                        peak_int = max(open_ - low_, 0.0)
                    peak_ret = (peak_int - open_ * DIRECTIONAL_PREMIUM_PCT) / (open_ * DIRECTIONAL_PREMIUM_PCT)
                    return r, peak_ret >= SCALE_OUT_RETURN
                else:
                    return directional_return(
                        open_, high_, low_, close_, d,
                        premium_pct=DIRECTIONAL_PREMIUM_PCT,
                    ), False

            if strategy_mode in ("smart_v3", "smart_v4") and ma_overlay is not None and date in ma_overlay.index:
                _ovr = ma_overlay.loc[date]
                _flow = float(_ovr["weekly_flow"]) if pd.notna(_ovr["weekly_flow"]) else 0.0
                if bool(_ovr["near_long_ma"]) and _flow >= FLOW_CONFIRM_LONG:
                    actual_ret, hit = _dir_ret(1)
                    trade_type = "MA_CALL"
                    alloc_pct  = JACKPOT_ALLOC_PCT     # full Kelly on high-conviction setup
                    n_ma_confluence += 1
                    if hit: n_scaleout_hits += 1
                    took_directional = True
                    took_ma_confluence = True
                elif bool(_ovr["near_short_ma"]) and _flow <= FLOW_CONFIRM_SHORT:
                    actual_ret, hit = _dir_ret(-1)
                    trade_type = "MA_PUT"
                    alloc_pct  = JACKPOT_ALLOC_PCT
                    n_ma_confluence += 1
                    if hit: n_scaleout_hits += 1
                    took_directional = True
                    took_ma_confluence = True

            # ---- smart_v2 / smart_v3 / smart_v4 fallback: directional override
            # High-conviction directional plays take priority over the
            # gap-fade heuristic — single-leg call/put is asymmetric upside,
            # while gap-fade is a lower-conviction mean-reversion bet.
            if not took_directional and strategy_mode in ("smart_v2", "smart_v3", "smart_v4"):
                if p_up >= DIRECTIONAL_PROB_HI:
                    actual_ret, hit = _dir_ret(1)
                    trade_type = "DIR_CALL"
                    alloc_pct  = min(alloc_pct * DIRECTIONAL_ALLOC_BOOST, JACKPOT_ALLOC_PCT)
                    n_directional += 1
                    if hit: n_scaleout_hits += 1
                    took_directional = True
                elif p_up <= DIRECTIONAL_PROB_LO and not took_ma_confluence:
                    actual_ret, hit = _dir_ret(-1)
                    trade_type = "DIR_PUT"
                    alloc_pct  = min(alloc_pct * DIRECTIONAL_ALLOC_BOOST, JACKPOT_ALLOC_PCT)
                    n_directional += 1
                    if hit: n_scaleout_hits += 1
                    took_directional = True

            if not took_directional:
                if strategy_mode == "smart_v4":
                    # Patient lottery ticket: no straddle, no gap-fade.
                    # Skip days that don't qualify for a directional play.
                    trade_type = "SKIP"
                    actual_ret = 0.0
                    alloc_pct  = 0.0
                    signal     = "SKIP"
                elif do_fade:
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
        if strategy_mode in ("smart", "smart_v2", "smart_v3", "smart_v4"):
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
        n_ma_confluence=n_ma_confluence,
        n_scaleout_hits=n_scaleout_hits,
    )


# ── Helper: daily-aligned weekly flow series (smart_v3) ────────────────────
def _compute_weekly_flow_series(daily: pd.DataFrame) -> pd.Series:
    """Build a per-week flow_score time series indexed by week-end Friday.

    Mirrors the logic in weekly_levels.build_weekly_order_flow but returns the
    full time series rather than just the latest snapshot. Used by smart_v3 to
    look up "what was the weekly flow score as of the last completed week"
    for each backtest day.
    """
    if len(daily) < 25:
        return pd.Series(dtype=float)

    d = daily.copy()
    rng = (d["High"] - d["Low"]).replace(0, pd.NA)
    d["_cs"] = (d["Close"] - d["Low"]) / rng
    import numpy as np
    d["_sv"] = np.sign(d["Close"] - d["Open"]) * d["Volume"]

    weekly = pd.DataFrame({
        "Open":   d["Open"].resample("W-FRI").first(),
        "High":   d["High"].resample("W-FRI").max(),
        "Low":    d["Low"].resample("W-FRI").min(),
        "Close":  d["Close"].resample("W-FRI").last(),
        "Volume": d["Volume"].resample("W-FRI").sum(),
        "CVD":    d["_sv"].resample("W-FRI").sum(),
    }).dropna(subset=["Close"])

    if len(weekly) < 5:
        return pd.Series(dtype=float)

    wk_rng = (weekly["High"] - weekly["Low"]).replace(0, pd.NA)
    weekly["WkCS"] = (weekly["Close"] - weekly["Low"]) / wk_rng

    def _z(s, win=20):
        r = s.rolling(win, min_periods=max(4, win // 2))
        return ((s - r.mean()) / r.std()).clip(-3, 3)

    cvd_z = _z(weekly["CVD"])
    cs_z  = _z(weekly["WkCS"] - 0.5)
    flow_score = ((cvd_z + cs_z) / 2 * 33).clip(-100, 100)
    return flow_score.rename("weekly_flow")
