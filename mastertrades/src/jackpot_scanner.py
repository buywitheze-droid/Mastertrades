"""Multi-ticker JACKPOT scanner — runs both vol classifier AND direct-P&L classifier.

This is the production scanner that powers the multi-ticker dashboard.
For each ticker in the universe (default: SPY, QQQ, IWM, AAPL — the four
that pass the edge bar), we:

  1. Fetch / refresh daily OHLCV (cached as data/<TICKER>_1d.csv).
  2. Build features and prepare the supervised target.
  3. Load (or train + cache) the per-ticker vol classifier (LogReg).
  4. Load (or train + cache) the per-ticker direct-PnL classifier (LogReg).
  5. Score the most recent session through both models.
  6. Combine into a single signal:
        GO_JACKPOT  — both fire (p_vol >= hot, p_pnl >= jackpot)
        GO_HOT      — vol classifier alone (lower confidence)
        SKIP        — calm

Caches live in:
    data/<TICKER>_1d.csv
    models/<TICKER>_logreg.joblib       ← vol classifier
    models/<TICKER>_pnl_logreg.joblib   ← direct-P&L classifier

Use as a CLI::

    python -m src.jackpot_scanner --tickers SPY,QQQ,IWM,AAPL

Or programmatically::

    from src.jackpot_scanner import scan_jackpot_universe
    rows, errors = scan_jackpot_universe()
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from src.scanner import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DATA_FRESH_HOURS_DEFAULT,
    HISTORY_YEARS_DEFAULT,
    MIN_TRAIN_ROWS,
    _safe_ticker_filename,
    fetch_or_load_daily,
    load_or_train_model,
)
from src.strategy_sim import straddle_return, weekly_straddle_return
from src.volatility_classifier import (
    BINARY_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    _one_hot_weekday,
    fit_full,
    make_logreg,
    prepare_xy,
    score_dataframe,
)
from src.volatility_patterns import build_features


# Universe selected from the multi-ticker comparison: AUC > 0.72 + win rate > 59%
DEFAULT_JACKPOT_UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "IWM", "AAPL")

# Decision thresholds (calibrated from the SPY edge_finder analysis)
HOT_THRESHOLD = 0.30          # p_vol cutoff for "elevated vol expected"
JACKPOT_THRESHOLD = 0.55      # p_pnl cutoff for "highest-conviction profit setup"
WEEKLY_CONFIRM_THRESHOLD = 0.55  # p_weekly cutoff used for the ULTRA tier
WEEKLY_DAYS = 5               # weekly model lookahead window

logger = logging.getLogger("jackpot_scanner")


# ---------------------------------------------------------------------------
# Market-phase helper (used by both dashboards to surface signal liveness)
# ---------------------------------------------------------------------------


def market_phase(now: datetime | None = None) -> dict:
    """Return current US-market phase metadata for the dashboards.

    Returns a dict with:
        phase            — "PRE_OPEN" / "OPEN_PENDING_DATA" / "OPEN_LIVE" /
                            "AFTER_HOURS" / "WEEKEND"
        label            — short user-facing string
        is_live          — True if the model's signal can be trusted as final
        is_open          — True if cash market is open right now
        next_open_in     — minutes until next 9:30 ET open (None if open now)
        minutes_since_open — minutes since today's 9:30 ET open (None if pre-open)
        as_of            — datetime used for the computation

    Uses US/Eastern via zoneinfo; falls back to naive time if zoneinfo
    fails (rare on modern Windows but possible).
    """
    if now is None:
        now = datetime.now()
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        if now.tzinfo is None:
            local_now = now.astimezone() if hasattr(now, "astimezone") else now
            et_now = local_now.astimezone(et) if local_now.tzinfo else now
            # If still naive, assume server is UTC
            if et_now.tzinfo is None:
                from datetime import timezone as _tz
                et_now = now.replace(tzinfo=_tz.utc).astimezone(et)
        else:
            et_now = now.astimezone(et)
    except Exception:
        et_now = now

    wd = et_now.weekday()
    open_dt = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_dt = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
    data_settled_dt = et_now.replace(hour=9, minute=50, second=0, microsecond=0)

    if wd >= 5:
        return {
            "phase": "WEEKEND", "label": "WEEKEND — Markets closed",
            "is_live": False, "is_open": False,
            "next_open_in": None, "minutes_since_open": None, "as_of": et_now,
        }

    if et_now < open_dt:
        return {
            "phase": "PRE_OPEN",
            "label": "PRE-MARKET — Signal pending today's open",
            "is_live": False, "is_open": False,
            "next_open_in": int((open_dt - et_now).total_seconds() // 60),
            "minutes_since_open": None, "as_of": et_now,
        }
    if et_now < data_settled_dt:
        return {
            "phase": "OPEN_PENDING_DATA",
            "label": "MARKET OPEN — Yahoo data settling (refresh in a few min)",
            "is_live": False, "is_open": True,
            "next_open_in": None,
            "minutes_since_open": int((et_now - open_dt).total_seconds() // 60),
            "as_of": et_now,
        }
    if et_now < close_dt:
        return {
            "phase": "OPEN_LIVE",
            "label": "MARKET OPEN — Signal LIVE (decision window)",
            "is_live": True, "is_open": True,
            "next_open_in": None,
            "minutes_since_open": int((et_now - open_dt).total_seconds() // 60),
            "as_of": et_now,
        }
    return {
        "phase": "AFTER_HOURS",
        "label": "AFTER-HOURS — Signal FROZEN until next open",
        "is_live": True, "is_open": False,
        "next_open_in": None,
        "minutes_since_open": int((et_now - open_dt).total_seconds() // 60),
        "as_of": et_now,
    }


# ---------------------------------------------------------------------------
# Premium estimation per ticker (rough IV proxy from realized range)
# ---------------------------------------------------------------------------


def estimate_premium_pct(daily: pd.DataFrame) -> float:
    """Rough at-the-money 0DTE straddle premium as % of spot.

    Anchor: SPY's mean daily range is ~1.05% and we price its straddle at
    1.1% of spot (a slight markup over realized to model implied vol).
    Apply the same multiplier to other names.
    """
    rng = ((daily["High"] - daily["Low"]) / daily["Open"]).dropna()
    mean_rng = float(rng.mean())
    spy_mean = 0.0105
    spy_premium = 0.011
    return spy_premium * (mean_rng / spy_mean)


def estimate_weekly_premium_pct(daily: pd.DataFrame) -> float:
    """Rough at-the-money weekly straddle premium as % of spot.

    Anchor: SPY weekly straddle is ~2.5% of spot, scaled by the same
    realized-range multiplier we use for 0DTE.
    """
    rng = ((daily["High"] - daily["Low"]) / daily["Open"]).dropna()
    mean_rng = float(rng.mean())
    spy_mean = 0.0105
    spy_weekly_premium = 0.025
    return spy_weekly_premium * (mean_rng / spy_mean)


# ---------------------------------------------------------------------------
# Direct-P&L classifier (mirrors edge_finder.train_direct_pnl_model)
# ---------------------------------------------------------------------------


def _apply_straddle(daily: pd.DataFrame, premium_pct: float) -> pd.Series:
    rets = []
    for _date, bar in daily.iterrows():
        rets.append(
            straddle_return(
                float(bar["Open"]),
                float(bar["High"]),
                float(bar["Low"]),
                float(bar["Close"]),
                premium_pct=premium_pct,
            )
        )
    return pd.Series(rets, index=daily.index, name="straddle_ret")


def _prepare_pnl_xy(features: pd.DataFrame, daily: pd.DataFrame, premium_pct: float):
    feat = features.dropna(subset=["range_pct"]).copy()
    common = feat.index.intersection(daily.index)
    feat = feat.loc[common]
    daily = daily.loc[common]
    straddle = _apply_straddle(daily, premium_pct=premium_pct)
    y = (straddle > 0).astype(int)

    base = feat[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    weekday_dummies = _one_hot_weekday(feat["weekday"])
    X = pd.concat([base, weekday_dummies], axis=1)

    keep = X.dropna().index
    X = X.loc[keep]
    y = y.loc[keep]
    realized = straddle.loc[keep]
    return X, y, realized


def _apply_weekly_straddle(daily: pd.DataFrame, premium_pct: float,
                           window: int = WEEKLY_DAYS) -> pd.Series:
    """Compute realized weekly straddle return for every day with a
    full ``window`` of forward bars."""
    rets = []
    dates = []
    n = len(daily)
    for i in range(n - window):
        bar = daily.iloc[i]
        win = daily.iloc[i:i + window]
        rets.append(weekly_straddle_return(
            float(bar["Open"]),
            float(win["High"].max()),
            float(win["Low"].min()),
            float(win["Close"].iloc[-1]),
            premium_pct=premium_pct,
        ))
        dates.append(daily.index[i])
    return pd.Series(rets, index=dates, name="weekly_ret")


def _prepare_weekly_xy(features: pd.DataFrame, daily: pd.DataFrame,
                       premium_pct: float, window: int = WEEKLY_DAYS):
    """X / y for the dedicated weekly model (target: weekly_ret > 0)."""
    feat = features.dropna(subset=["range_pct"]).copy()
    weekly = _apply_weekly_straddle(daily, premium_pct=premium_pct, window=window)
    common = feat.index.intersection(weekly.index)
    feat = feat.loc[common]
    weekly = weekly.loc[common]
    y = (weekly > 0).astype(int)

    base = feat[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    weekday_dummies = _one_hot_weekday(feat["weekday"])
    X = pd.concat([base, weekday_dummies], axis=1)

    keep = X.dropna().index
    X = X.loc[keep]
    y = y.loc[keep]
    realized = weekly.loc[keep]
    return X, y, realized


def _pnl_model_path(ticker: str, model_dir: Path) -> Path:
    return model_dir / f"{_safe_ticker_filename(ticker)}_pnl_logreg.joblib"


def _weekly_model_path(ticker: str, model_dir: Path) -> Path:
    return model_dir / f"{_safe_ticker_filename(ticker)}_weekly_logreg.joblib"


def load_or_train_pnl_model(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Path = DEFAULT_MODEL_DIR,
    retrain: bool = False,
    max_age_days: int = 7,
):
    """Load cached direct-P&L model unless missing/stale or retrain=True."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = _pnl_model_path(ticker, model_dir)

    if path.exists() and not retrain:
        age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days < max_age_days:
            return joblib.load(path)

    logger.info("Training fresh PnL model for %s on %d rows", ticker, len(X))
    model = fit_full(X, y, make_logreg)
    joblib.dump(model, path)
    return model


def load_or_train_weekly_model(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Path = DEFAULT_MODEL_DIR,
    retrain: bool = False,
    max_age_days: int = 7,
):
    """Load cached weekly model unless missing/stale or retrain=True."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = _weekly_model_path(ticker, model_dir)

    if path.exists() and not retrain:
        age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days < max_age_days:
            return joblib.load(path)

    logger.info("Training fresh weekly model for %s on %d rows", ticker, len(X))
    model = fit_full(X, y, make_logreg)
    joblib.dump(model, path)
    return model


# ---------------------------------------------------------------------------
# Per-ticker historical jackpot stats (precomputed cache for the dashboard)
# ---------------------------------------------------------------------------


def historical_jackpot_stats(
    daily: pd.DataFrame,
    feats: pd.DataFrame,
    premium_pct: float,
    weekly_premium_pct: float,
    pnl_threshold: float = JACKPOT_THRESHOLD,
    weekly_threshold: float = WEEKLY_CONFIRM_THRESHOLD,
) -> dict:
    """Walk-forward all three models; collect jackpot + ultra-jackpot stats.

    The ULTRA-JACKPOT subset = days where BOTH the 0DTE direct-P&L model
    and the dedicated weekly model fire. Empirically this lifts win rate
    from ~65% to ~83% on SPY.

    Cached on disk so repeated dashboard runs don't re-walk-forward.
    """
    from src.edge_finder import train_direct_pnl_model
    from src.volatility_classifier import walk_forward_proba

    X, y, _ = prepare_xy(feats, volatile_quantile=0.80)
    vol_preds = walk_forward_proba(X, y, make_logreg)
    direct = train_direct_pnl_model(feats, daily, vol_preds["y_score"], premium_pct=premium_pct)

    # Walk-forward the dedicated weekly model
    X_w, y_w, weekly_realized = _prepare_weekly_xy(feats, daily, premium_pct=weekly_premium_pct)
    weekly_preds = walk_forward_proba(X_w, y_w, make_logreg)

    pnl_preds = direct.preds_oos
    jackpot = pnl_preds[pnl_preds["y_score"] >= pnl_threshold].copy()

    # ULTRA = both signals fire on the same date
    weekly_aligned = weekly_preds["y_score"].reindex(pnl_preds.index)
    ultra_mask = (pnl_preds["y_score"] >= pnl_threshold) & (weekly_aligned >= weekly_threshold)
    ultra = pnl_preds[ultra_mask].copy()
    ultra_weekly_rets = weekly_realized.reindex(ultra.index)

    if len(jackpot) == 0:
        return {
            "n_jackpot": 0, "win_rate": float("nan"),
            "avg_ret": float("nan"), "trades_per_year": 0.0,
            "auc": direct.auc, "active_start": None, "active_end": None,
            "n_ultra": 0, "ultra_win_rate": float("nan"),
            "ultra_avg_ret": float("nan"), "ultra_trades_per_year": 0.0,
            "ultra_weekly_avg_ret": float("nan"),
        }

    span_years = max((jackpot.index.max() - jackpot.index.min()).days / 365.25, 0.5)
    ultra_span = max((ultra.index.max() - ultra.index.min()).days / 365.25, 0.5) if len(ultra) > 0 else 0.5

    return {
        "n_jackpot": int(len(jackpot)),
        "win_rate": float((jackpot["straddle_ret"] > 0).mean()),
        "avg_ret": float(jackpot["straddle_ret"].mean()),
        "trades_per_year": float(len(jackpot) / span_years),
        "auc": float(direct.auc),
        "active_start": jackpot.index.min().strftime("%Y-%m-%d"),
        "active_end": jackpot.index.max().strftime("%Y-%m-%d"),
        "n_ultra": int(len(ultra)),
        "ultra_win_rate": float((ultra["straddle_ret"] > 0).mean()) if len(ultra) > 0 else float("nan"),
        "ultra_avg_ret": float(ultra["straddle_ret"].mean()) if len(ultra) > 0 else float("nan"),
        "ultra_trades_per_year": float(len(ultra) / ultra_span) if len(ultra) > 0 else 0.0,
        "ultra_weekly_avg_ret": float(ultra_weekly_rets.dropna().mean()) if len(ultra) > 0 else float("nan"),
    }


def _stats_path(ticker: str, model_dir: Path) -> Path:
    return model_dir / f"{_safe_ticker_filename(ticker)}_jackpot_stats.joblib"


def load_or_compute_stats(
    ticker: str,
    daily: pd.DataFrame,
    feats: pd.DataFrame,
    premium_pct: float,
    weekly_premium_pct: float,
    model_dir: Path = DEFAULT_MODEL_DIR,
    refresh: bool = False,
    max_age_days: int = 14,
) -> dict:
    path = _stats_path(ticker, model_dir)
    if path.exists() and not refresh:
        age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days < max_age_days:
            cached = joblib.load(path)
            # Backwards-compatible: re-compute if cache predates ultra fields.
            if "n_ultra" in cached:
                return cached

    logger.info("Computing fresh jackpot stats for %s (this is the slow path)", ticker)
    stats = historical_jackpot_stats(daily, feats, premium_pct, weekly_premium_pct)
    joblib.dump(stats, path)
    return stats


# ---------------------------------------------------------------------------
# Per-ticker scoring
# ---------------------------------------------------------------------------


@dataclass
class JackpotRow:
    ticker: str
    as_of: pd.Timestamp
    last_close: float
    prev_close: float
    pct_change: float
    p_vol: float
    p_pnl: float
    p_weekly: float
    signal: str            # GO_ULTRA_JACKPOT / GO_JACKPOT / GO_HOT / SKIP
    premium_pct: float
    weekly_premium_pct: float
    estimated_premium_dollars: float
    estimated_weekly_premium_dollars: float
    base_rate_vol: float

    # Historical baseline jackpot stats
    n_jackpot_history: int
    trades_per_year: float
    win_rate_history: float
    avg_ret_history: float

    # Historical ultra-jackpot stats (BOTH models agree)
    n_ultra_history: int
    ultra_trades_per_year: float
    ultra_win_rate_history: float
    ultra_avg_ret_history: float
    ultra_weekly_avg_ret_history: float

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.strftime("%Y-%m-%d"),
            "last_close": self.last_close,
            "prev_close": self.prev_close,
            "pct_change": self.pct_change,
            "p_vol": self.p_vol,
            "p_pnl": self.p_pnl,
            "p_weekly": self.p_weekly,
            "signal": self.signal,
            "premium_pct": self.premium_pct,
            "weekly_premium_pct": self.weekly_premium_pct,
            "estimated_premium_dollars": self.estimated_premium_dollars,
            "estimated_weekly_premium_dollars": self.estimated_weekly_premium_dollars,
            "base_rate_vol": self.base_rate_vol,
            "n_jackpot_history": self.n_jackpot_history,
            "trades_per_year": self.trades_per_year,
            "win_rate_history": self.win_rate_history,
            "avg_ret_history": self.avg_ret_history,
            "n_ultra_history": self.n_ultra_history,
            "ultra_trades_per_year": self.ultra_trades_per_year,
            "ultra_win_rate_history": self.ultra_win_rate_history,
            "ultra_avg_ret_history": self.ultra_avg_ret_history,
            "ultra_weekly_avg_ret_history": self.ultra_weekly_avg_ret_history,
        }


def classify_signal(p_vol: float, p_pnl: float, p_weekly: float,
                    hot_threshold: float = HOT_THRESHOLD,
                    jackpot_threshold: float = JACKPOT_THRESHOLD,
                    weekly_threshold: float = WEEKLY_CONFIRM_THRESHOLD) -> str:
    """Four-tier signal:
        GO_ULTRA_JACKPOT  — vol HOT + 0DTE-PnL >= jp + weekly >= wkly
        GO_JACKPOT        — vol HOT + 0DTE-PnL >= jp (no weekly confirm)
        GO_HOT            — vol HOT alone
        SKIP              — calm
    """
    if p_vol >= hot_threshold and p_pnl >= jackpot_threshold and p_weekly >= weekly_threshold:
        return "GO_ULTRA_JACKPOT"
    if p_vol >= hot_threshold and p_pnl >= jackpot_threshold:
        return "GO_JACKPOT"
    if p_vol >= hot_threshold:
        return "GO_HOT"
    return "SKIP"


# ---------------------------------------------------------------------------
# Walk-forward training (leakage-free)
# ---------------------------------------------------------------------------
# The original load_or_train_* functions train one model on the FULL daily
# history and then score the most recent day with that same model. When the
# most recent day is itself an extreme outlier (e.g. 2025-04-09 SPY +11.18%),
# its label leaks into the training set and inflates the model's confidence
# on similar pre-market setups. Walk-forward retraining cuts the training
# window at the previous Friday, so any prediction on day T uses ZERO data
# from T or later.
#
# The cache key is the cutoff Friday, so within a given week every weekday
# reuses the same three models — at most ~52 model-trainings per ticker per
# year per kind (~624/yr across the 4-ticker × 3-model universe).

WF_SUBDIR = "wf"
WF_CACHE_KEEP_PER_KEY = 12  # keep most recent N cutoffs per (ticker, kind)


def _walkforward_cutoff(target_date: pd.Timestamp) -> pd.Timestamp:
    """Last Friday STRICTLY before `target_date`.

    Examples (target_date weekday → cutoff):
      Mon → previous Friday (3 days back)
      Wed → previous Friday (5 days back)
      Fri → previous Friday (7 days back)
    """
    target = pd.Timestamp(target_date).normalize()
    # Days to subtract to reach the prior Friday
    offset = (target.weekday() - 4) % 7
    if offset == 0:
        offset = 7  # if target itself is Friday, go back a full week
    return target - pd.Timedelta(days=offset)


def _data_fingerprint(daily_train: pd.DataFrame) -> str:
    """Short fingerprint of the training slice so revisions to historical
    OHLCV (Polygon corrections, dividend adjustments) invalidate the cache."""
    n = len(daily_train)
    last_close = float(daily_train["Close"].iloc[-1])
    last_date = pd.Timestamp(daily_train.index[-1]).strftime("%Y%m%d")
    # 8-char hex of n + last_close + last_date — collision risk is negligible
    import hashlib
    raw = f"{n}|{last_close:.4f}|{last_date}".encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def _wf_model_path(ticker: str, cutoff: pd.Timestamp, kind: str,
                   model_dir: Path = DEFAULT_MODEL_DIR,
                   data_fp: str = "any") -> Path:
    """Cache path for a walk-forward model. kind ∈ {vol,pnl,weekly}.
    `data_fp` makes the cache invalidate on data revisions."""
    safe = _safe_ticker_filename(ticker)
    cdate = pd.Timestamp(cutoff).strftime("%Y%m%d")
    return model_dir / WF_SUBDIR / f"{safe}_{cdate}_{data_fp}_{kind}.joblib"


def _prune_wf_cache(ticker: str, kind: str, model_dir: Path,
                    keep: int = WF_CACHE_KEEP_PER_KEY) -> None:
    """Keep only the most recent `keep` cutoffs per (ticker, kind).
    Filenames are `{safe_ticker}_{cutoff}_{fp}_{kind}.joblib`; we sort by
    cutoff (the YYYYMMDD slug) and unlink the rest."""
    safe = _safe_ticker_filename(ticker)
    wf_dir = model_dir / WF_SUBDIR
    if not wf_dir.exists():
        return
    matches = sorted(
        wf_dir.glob(f"{safe}_*_*_{kind}.joblib"),
        key=lambda p: p.name.split("_")[1],  # cutoff slug
        reverse=True,
    )
    for stale in matches[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def train_or_load_walkforward_models(
    ticker: str,
    cutoff: pd.Timestamp,
    daily_full: pd.DataFrame,
    model_dir: Path = DEFAULT_MODEL_DIR,
    retrain: bool = False,
) -> dict:
    """Train/load the three walk-forward models for `ticker` with the strict
    cutoff. `daily_full` must be the full daily DataFrame (we slice it
    internally to ≤ cutoff for training).

    Returns a dict::
        {vol_model, pnl_model, weekly_model,
         vol_cols, pnl_cols, weekly_cols,
         premium_pct, weekly_premium_pct, vol_threshold}
    """
    (model_dir / WF_SUBDIR).mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(cutoff).normalize()
    daily_full = daily_full.copy()
    # Normalise index to date (lose tz/intraday) for clean slicing
    if daily_full.index.tz is not None:
        daily_full.index = daily_full.index.tz_localize(None)
    daily_full.index = pd.to_datetime(daily_full.index).normalize()

    daily_train = daily_full[daily_full.index <= cutoff]
    if len(daily_train) < MIN_TRAIN_ROWS + 50:
        raise RuntimeError(
            f"{ticker}: only {len(daily_train)} training rows ≤ {cutoff.date()} "
            f"(need ≥ {MIN_TRAIN_ROWS + 50})"
        )

    feats_train = build_features(daily_train)
    X_vol, y_vol, vol_thresh = prepare_xy(feats_train, volatile_quantile=0.80)
    if len(X_vol) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{ticker}: only {len(X_vol)} vol-feature rows ≤ {cutoff.date()}")
    premium_pct = estimate_premium_pct(daily_train)
    weekly_premium_pct = estimate_weekly_premium_pct(daily_train)
    X_pnl, y_pnl, _ = _prepare_pnl_xy(feats_train, daily_train, premium_pct)
    X_wk, y_wk, _ = _prepare_weekly_xy(feats_train, daily_train, weekly_premium_pct)

    data_fp = _data_fingerprint(daily_train)
    out = {
        "vol_cols": list(X_vol.columns),
        "pnl_cols": list(X_pnl.columns),
        "weekly_cols": list(X_wk.columns),
        "premium_pct": premium_pct,
        "weekly_premium_pct": weekly_premium_pct,
        "vol_threshold": vol_thresh,
        "data_fingerprint": data_fp,
    }

    for kind, X, y in [("vol", X_vol, y_vol), ("pnl", X_pnl, y_pnl), ("weekly", X_wk, y_wk)]:
        path = _wf_model_path(ticker, cutoff, kind, model_dir, data_fp=data_fp)
        if path.exists() and not retrain:
            out[f"{kind}_model"] = joblib.load(path)
            continue
        logger.info("Training walk-forward %s model for %s (cutoff=%s, fp=%s) on %d rows",
                    kind, ticker, cutoff.date(), data_fp, len(X))
        model = fit_full(X, y, make_logreg)
        joblib.dump(model, path)
        out[f"{kind}_model"] = model
        _prune_wf_cache(ticker, kind, model_dir)

    return out


def score_jackpot_one_walkforward(
    ticker: str,
    refresh_data: bool = True,
    retrain: bool = False,
    refresh_stats: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
    history_years: int = HISTORY_YEARS_DEFAULT,
) -> JackpotRow:
    """Strict walk-forward scoring of `ticker` for the latest available
    session. Models are retrained at the last-Friday cutoff so the
    prediction uses ZERO knowledge of the day being scored or anything
    later.
    """
    daily = fetch_or_load_daily(
        ticker, data_dir=data_dir, refresh=refresh_data,
        data_fresh_hours=data_fresh_hours, history_years=history_years,
    )
    if len(daily) < MIN_TRAIN_ROWS + 50:
        raise RuntimeError(f"{ticker}: only {len(daily)} daily rows.")

    daily = daily.copy()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = pd.to_datetime(daily.index).normalize()
    target_date = daily.index.max()
    cutoff = _walkforward_cutoff(target_date)

    bundle = train_or_load_walkforward_models(
        ticker, cutoff, daily, model_dir=model_dir, retrain=retrain,
    )

    # Build features on the FULL daily slice; the row at target_date uses only
    # lagged (≤ target_date - 1) information by construction of build_features.
    feats = build_features(daily)
    if target_date not in feats.index:
        raise RuntimeError(f"{ticker}: target {target_date.date()} missing from feature panel")

    row = feats.loc[[target_date]].copy()
    # build_features computes days_in_month / days_left_in_month from the
    # observed slice — for the LAST row of any slice, those calendar booleans
    # are mechanically true even when the calendar disagrees. Zero them on
    # the prediction row to avoid spurious end-of-month signals.
    for spurious in ("is_turn_of_month", "is_last_trading_day_of_month",
                     "is_first_trading_day_of_month"):
        if spurious in row.columns:
            row[spurious] = 0
    base = row[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    wd_dummies = _one_hot_weekday(row["weekday"])
    X_pred_vol = pd.concat([base, wd_dummies], axis=1).reindex(columns=bundle["vol_cols"], fill_value=0.0)
    X_pred_pnl = pd.concat([base, wd_dummies], axis=1).reindex(columns=bundle["pnl_cols"], fill_value=0.0)
    X_pred_wk  = pd.concat([base, wd_dummies], axis=1).reindex(columns=bundle["weekly_cols"], fill_value=0.0)

    p_vol = float(score_dataframe(bundle["vol_model"], X_pred_vol).iloc[0])
    p_pnl = float(score_dataframe(bundle["pnl_model"], X_pred_pnl).iloc[0])
    p_weekly = float(score_dataframe(bundle["weekly_model"], X_pred_wk).iloc[0])

    last_close = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float("nan")
    pct_change = (last_close / prev_close - 1.0) if not pd.isna(prev_close) else float("nan")

    # Stats are descriptive (per-ticker historical jackpot frequency); fine to
    # compute on full history — they are NOT used as model input.
    stats = load_or_compute_stats(
        ticker, daily, feats, bundle["premium_pct"], bundle["weekly_premium_pct"],
        model_dir=model_dir, refresh=refresh_stats,
    )
    # Vol classifier base rate from the TRAINING slice (honest)
    base_rate_vol = float((feats.loc[feats.index <= cutoff, "range_pct"]
                          .dropna() >= bundle["vol_threshold"]).mean())

    return JackpotRow(
        ticker=ticker.upper(),
        as_of=target_date,
        last_close=last_close,
        prev_close=prev_close,
        pct_change=float(pct_change),
        p_vol=p_vol,
        p_pnl=p_pnl,
        p_weekly=p_weekly,
        signal=classify_signal(p_vol, p_pnl, p_weekly),
        premium_pct=bundle["premium_pct"],
        weekly_premium_pct=bundle["weekly_premium_pct"],
        estimated_premium_dollars=last_close * bundle["premium_pct"],
        estimated_weekly_premium_dollars=last_close * bundle["weekly_premium_pct"],
        base_rate_vol=base_rate_vol,
        n_jackpot_history=int(stats.get("n_jackpot", 0)),
        trades_per_year=float(stats.get("trades_per_year", 0.0)),
        win_rate_history=float(stats.get("win_rate", float("nan"))),
        avg_ret_history=float(stats.get("avg_ret", float("nan"))),
        n_ultra_history=int(stats.get("n_ultra", 0)),
        ultra_trades_per_year=float(stats.get("ultra_trades_per_year", 0.0)),
        ultra_win_rate_history=float(stats.get("ultra_win_rate", float("nan"))),
        ultra_avg_ret_history=float(stats.get("ultra_avg_ret", float("nan"))),
        ultra_weekly_avg_ret_history=float(stats.get("ultra_weekly_avg_ret", float("nan"))),
    )


def score_jackpot_one(
    ticker: str,
    refresh_data: bool = True,
    retrain: bool = False,
    refresh_stats: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
    history_years: int = HISTORY_YEARS_DEFAULT,
    walk_forward: bool = False,
) -> JackpotRow:
    """Score a single ticker through all three models. Used by the dashboard.

    When `walk_forward=True`, routes to `score_jackpot_one_walkforward`,
    which retrains at the prior-Friday cutoff to eliminate label leakage.
    """
    if walk_forward:
        return score_jackpot_one_walkforward(
            ticker, refresh_data=refresh_data, retrain=retrain,
            refresh_stats=refresh_stats, data_dir=data_dir,
            model_dir=model_dir, data_fresh_hours=data_fresh_hours,
            history_years=history_years,
        )
    daily = fetch_or_load_daily(
        ticker,
        data_dir=data_dir,
        refresh=refresh_data,
        data_fresh_hours=data_fresh_hours,
        history_years=history_years,
    )
    if len(daily) < MIN_TRAIN_ROWS + 50:
        raise RuntimeError(f"{ticker}: only {len(daily)} daily rows.")

    feats = build_features(daily)
    X, y, _ = prepare_xy(feats, volatile_quantile=0.80)
    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{ticker}: only {len(X)} feature rows after dropping NaNs.")

    vol_model = load_or_train_model(ticker, X, y, model_dir=model_dir, retrain=retrain)

    premium_pct = estimate_premium_pct(daily)
    weekly_premium_pct = estimate_weekly_premium_pct(daily)

    X_pnl, y_pnl, _realized = _prepare_pnl_xy(feats, daily, premium_pct)
    pnl_model = load_or_train_pnl_model(ticker, X_pnl, y_pnl, model_dir=model_dir, retrain=retrain)

    X_weekly, y_weekly, _ = _prepare_weekly_xy(feats, daily, weekly_premium_pct)
    weekly_model = load_or_train_weekly_model(ticker, X_weekly, y_weekly,
                                              model_dir=model_dir, retrain=retrain)

    last_X = X.tail(1)
    last_X_pnl = X_pnl.tail(1)
    last_X_weekly = X_weekly.tail(1) if len(X_weekly) > 0 else last_X_pnl

    p_vol = float(score_dataframe(vol_model, last_X).iloc[0])
    p_pnl = float(score_dataframe(pnl_model, last_X_pnl).iloc[0])
    p_weekly = float(score_dataframe(weekly_model, last_X_weekly).iloc[0])

    last_idx = last_X.index[0]
    last_close = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float("nan")
    pct_change = (last_close / prev_close - 1.0) if not pd.isna(prev_close) else float("nan")

    stats = load_or_compute_stats(
        ticker, daily, feats, premium_pct, weekly_premium_pct,
        model_dir=model_dir, refresh=refresh_stats,
    )

    return JackpotRow(
        ticker=ticker.upper(),
        as_of=last_idx,
        last_close=last_close,
        prev_close=prev_close,
        pct_change=float(pct_change),
        p_vol=p_vol,
        p_pnl=p_pnl,
        p_weekly=p_weekly,
        signal=classify_signal(p_vol, p_pnl, p_weekly),
        premium_pct=premium_pct,
        weekly_premium_pct=weekly_premium_pct,
        estimated_premium_dollars=last_close * premium_pct,
        estimated_weekly_premium_dollars=last_close * weekly_premium_pct,
        base_rate_vol=float(y.mean()),
        n_jackpot_history=int(stats.get("n_jackpot", 0)),
        trades_per_year=float(stats.get("trades_per_year", 0.0)),
        win_rate_history=float(stats.get("win_rate", float("nan"))),
        avg_ret_history=float(stats.get("avg_ret", float("nan"))),
        n_ultra_history=int(stats.get("n_ultra", 0)),
        ultra_trades_per_year=float(stats.get("ultra_trades_per_year", 0.0)),
        ultra_win_rate_history=float(stats.get("ultra_win_rate", float("nan"))),
        ultra_avg_ret_history=float(stats.get("ultra_avg_ret", float("nan"))),
        ultra_weekly_avg_ret_history=float(stats.get("ultra_weekly_avg_ret", float("nan"))),
    )


def score_jackpot_recent(
    ticker: str,
    n_days: int = 10,
    refresh_data: bool = True,
    retrain: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
    history_years: int = HISTORY_YEARS_DEFAULT,
) -> pd.DataFrame:
    """Score the last ``n_days`` sessions for a ticker through all three models.

    Returns a DataFrame indexed by date with columns:
        p_vol, p_pnl, p_weekly, signal, open, high, low, close,
        zdte_realized (NaN if not enough history), weekly_realized
    """
    daily = fetch_or_load_daily(
        ticker,
        data_dir=data_dir,
        refresh=refresh_data,
        data_fresh_hours=data_fresh_hours,
        history_years=history_years,
    )
    if len(daily) < MIN_TRAIN_ROWS + 50:
        raise RuntimeError(f"{ticker}: only {len(daily)} daily rows.")

    feats = build_features(daily)
    X, y, _ = prepare_xy(feats, volatile_quantile=0.80)
    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{ticker}: only {len(X)} feature rows after dropping NaNs.")

    premium_pct = estimate_premium_pct(daily)
    weekly_premium_pct = estimate_weekly_premium_pct(daily)

    vol_model = load_or_train_model(ticker, X, y, model_dir=model_dir, retrain=retrain)
    X_pnl, y_pnl, _ = _prepare_pnl_xy(feats, daily, premium_pct)
    pnl_model = load_or_train_pnl_model(ticker, X_pnl, y_pnl, model_dir=model_dir, retrain=retrain)
    X_weekly, y_weekly, _ = _prepare_weekly_xy(feats, daily, weekly_premium_pct)
    weekly_model = load_or_train_weekly_model(
        ticker, X_weekly, y_weekly, model_dir=model_dir, retrain=retrain
    )

    # Take the last n_days rows from the intersection of all three feature panels
    common_idx = X.index.intersection(X_pnl.index)
    recent_idx = common_idx[-n_days:]

    p_vol_series = score_dataframe(vol_model, X.loc[recent_idx])
    p_pnl_series = score_dataframe(pnl_model, X_pnl.loc[recent_idx])
    # Weekly model may have a shorter index (drops last 5 rows). Reindex with NaN.
    weekly_idx = X_weekly.index.intersection(recent_idx)
    p_weekly_raw = score_dataframe(weekly_model, X_weekly.loc[weekly_idx]) if len(weekly_idx) > 0 else pd.Series(dtype=float)
    p_weekly_series = p_weekly_raw.reindex(recent_idx)

    out = pd.DataFrame({
        "p_vol": p_vol_series,
        "p_pnl": p_pnl_series,
        "p_weekly": p_weekly_series,
        "open": daily.loc[recent_idx, "Open"],
        "high": daily.loc[recent_idx, "High"],
        "low": daily.loc[recent_idx, "Low"],
        "close": daily.loc[recent_idx, "Close"],
        "premium_pct": premium_pct,
        "weekly_premium_pct": weekly_premium_pct,
    })

    out["signal"] = [
        classify_signal(
            v,
            p,
            w if not pd.isna(w) else 0.0,
        )
        for v, p, w in zip(out["p_vol"], out["p_pnl"], out["p_weekly"])
    ]

    # Realized straddle returns for past days (today's bar is complete after close)
    zdte_rets = []
    for d in recent_idx:
        bar = daily.loc[d]
        zdte_rets.append(straddle_return(
            float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
            float(bar["Close"]), premium_pct=premium_pct,
        ))
    out["zdte_realized"] = zdte_rets

    return out


def scan_jackpot_universe(
    tickers: Iterable[str] = DEFAULT_JACKPOT_UNIVERSE,
    refresh_data: bool = True,
    retrain: bool = False,
    refresh_stats: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
    history_years: int = HISTORY_YEARS_DEFAULT,
    walk_forward: bool = False,
) -> tuple[list[JackpotRow], list[dict]]:
    rows: list[JackpotRow] = []
    errors: list[dict] = []
    for t in tickers:
        try:
            rows.append(
                score_jackpot_one(
                    t,
                    refresh_data=refresh_data,
                    retrain=retrain,
                    refresh_stats=refresh_stats,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    data_fresh_hours=data_fresh_hours,
                    history_years=history_years,
                    walk_forward=walk_forward,
                )
            )
        except Exception as exc:
            logger.warning("Failed %s: %s", t, exc)
            errors.append({"ticker": t, "error": str(exc)})
    # Rank: ULTRA first, JACKPOT second, HOT third, then SKIP
    rank_keys = {"GO_ULTRA_JACKPOT": 0, "GO_JACKPOT": 1, "GO_HOT": 2, "SKIP": 3}
    rows.sort(key=lambda r: (rank_keys.get(r.signal, 9), -r.p_pnl, -r.p_weekly, -r.p_vol))
    return rows, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tickers", default=",".join(DEFAULT_JACKPOT_UNIVERSE))
    parser.add_argument("--no-refresh-data", action="store_true")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain both models")
    parser.add_argument("--refresh-stats", action="store_true",
                        help="Force recompute the jackpot history stats (slow)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    rows, errors = scan_jackpot_universe(
        tickers=tickers,
        refresh_data=not args.no_refresh_data,
        retrain=args.retrain,
        refresh_stats=args.refresh_stats,
    )

    print()
    print(f"{'Ticker':<7} {'Signal':<18} {'p_vol':>6} {'p_pnl':>6} {'p_wkly':>7} "
          f"{'Close':>9} {'JP WR':>7} {'Ultra WR':>9} {'Ultra/yr':>9}")
    print("-" * 95)
    for r in rows:
        print(f"{r.ticker:<7} {r.signal:<18} {r.p_vol:>6.3f} {r.p_pnl:>6.3f} {r.p_weekly:>7.3f} "
              f"${r.last_close:>7.2f} {r.win_rate_history*100:>6.1f}% "
              f"{r.ultra_win_rate_history*100:>8.1f}% "
              f"{r.ultra_trades_per_year:>8.1f}")

    if errors:
        print()
        print("Errors:")
        for e in errors:
            print(f"  {e['ticker']}: {e['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
