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


# Kelly allocation fractions (matches TIERS in position_sizer.py)
JACKPOT_ALLOC_PCT = 0.25
HOT_ALLOC_PCT     = 0.15

BACKTEST_CACHE_AGE_DAYS = 7   # re-run if cache is stale


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


def _cache_path(ticker: str, n_months: int) -> Path:
    return DEFAULT_MODEL_DIR / f"{_safe_ticker_filename(ticker)}_backtest_{n_months}m.joblib"


def run_jackpot_backtest(
    ticker: str = "SPY",
    start_equity: float = 500.0,
    n_months: int = 6,
    force_refresh: bool = False,
) -> BacktestResult:
    """Run the historical Jackpot backtest for the last ``n_months`` months.

    Results are cached in models/ so subsequent calls are instant.
    Pass ``force_refresh=True`` to re-run the walk-forward (slow ~30-60 s).
    """
    path = _cache_path(ticker, n_months)
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

    result = _run_fresh(ticker, start_equity, n_months)
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
    )


def _run_fresh(
    ticker: str,
    start_equity: float,
    n_months: int,
) -> BacktestResult:
    """Full walk-forward backtest. Slow path (~30–60 s first run)."""

    # ── 1. Load OHLCV ────────────────────────────────────────────────────────
    daily = fetch_or_load_daily(ticker)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # ── 2. Feature engineering ───────────────────────────────────────────────
    feats = build_features(daily)
    premium_pct = estimate_premium_pct(daily)

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

    # ── 5. Align the two prediction series ───────────────────────────────────
    pnl_preds = pnl_result.preds_oos
    common = sorted(vol_preds.index.intersection(pnl_preds.index).intersection(daily.index))
    if len(common) == 0:
        raise ValueError(f"No overlapping dates for {ticker} backtest.")

    # ── 6. Extract last n_months of trading days ──────────────────────────────
    approx_days = n_months * 21
    bt_dates = common[-approx_days:]

    # ── 7. Day-by-day simulation ─────────────────────────────────────────────
    equity = start_equity
    rows: list[dict] = []

    for date in bt_dates:
        p_vol  = float(vol_preds.loc[date, "y_score"])
        p_pnl  = float(pnl_preds.loc[date, "y_score"])
        actual_ret = float(pnl_preds.loc[date, "straddle_ret"])

        bar    = daily.loc[date]
        open_  = float(bar["Open"])
        close_ = float(bar["Close"])
        high_  = float(bar["High"])
        low_   = float(bar["Low"])
        day_rng_pct = (high_ - low_) / open_ * 100

        # Signal
        if p_vol >= HOT_THRESHOLD and p_pnl >= JACKPOT_THRESHOLD:
            signal      = "GO_JACKPOT"
            alloc_pct   = JACKPOT_ALLOC_PCT
        elif p_vol >= HOT_THRESHOLD:
            signal      = "GO_HOT"
            alloc_pct   = HOT_ALLOC_PCT
        else:
            signal      = "SKIP"
            alloc_pct   = 0.0

        alloc_dollars = round(equity * alloc_pct, 2)
        dollar_change = round(alloc_dollars * actual_ret, 2) if signal != "SKIP" else 0.0
        equity        = max(round(equity + dollar_change, 2), 0.0)

        outcome = (
            "WIN"  if signal != "SKIP" and dollar_change > 0 else
            "LOSS" if signal != "SKIP" and dollar_change < 0 else
            "SKIP"
        )

        rows.append({
            "date":          date,
            "signal":        signal,
            "p_vol":         round(p_vol,  3),
            "p_pnl":         round(p_pnl,  3),
            "open":          round(open_,  2),
            "close":         round(close_, 2),
            "range_pct":     round(day_rng_pct, 2),
            "straddle_ret":  round(actual_ret, 3),
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
    )
