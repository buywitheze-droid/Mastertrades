"""Gap Reversal Analysis — identify, classify, and backtest gap fill + reversal setups.

A "gap" is when today's open differs significantly from yesterday's close.
Classic tape: SPY gaps down on fear, shorts pile in, then the gap closes and
flushes them — price rallies 6+ pts through and beyond the fill.

This module answers four questions using DAILY OHLCV bars only:

  1. What is today's gap? (size, direction, fill level)
  2. Historically, how often do gaps of this size fill on the same session?
  3. When a gap fills, how often does price reverse (vs. continue)?
  4. How large are those reversals on average?

Gap detection (daily bars):
    gap_pct = (Open - prev_Close) / prev_Close
    gap_up   → gap_pct > +threshold
    gap_down → gap_pct < -threshold

Gap fill proxy (daily bars):
    Gap-up filled  → Low  ≤ prev_Close during the session
    Gap-down filled→ High ≥ prev_Close during the session
    (Imprecise because we can't see intraday path, but correct direction)

Reversal proxy (daily bars):
    After a gap-up fills (Low ≤ prev_Close), a reversal occurs when
    Close > prev_Close — i.e., price came back up above the fill level.
    After a gap-down fills (High ≥ prev_Close), reversal occurs when
    Close < prev_Close — price fell back below the fill level.

Reversal magnitude (daily bars):
    gap_up_reversal_pts   = Close - prev_Close  (>0 means closed above fill)
    gap_down_reversal_pts = prev_Close - Close  (>0 means closed below fill)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("gap_analysis")

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MIN_GAP_PCT    = 0.0020   # 0.20% minimum gap to classify as a gap day
DEFAULT_LOOKBACK_YEARS = 5        # years of history to use for stats
GAP_BUCKET_EDGES       = [0.0020, 0.0040, 0.0075, 0.0150, 0.0300, 1.0]
GAP_BUCKET_LABELS      = ["0.2–0.4%", "0.4–0.75%", "0.75–1.5%", "1.5–3%", ">3%"]


# ─── Core feature engineering ─────────────────────────────────────────────────


def add_gap_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session gap features and return an augmented DataFrame.

    Input must have columns: Open, High, Low, Close (indexed by date).
    Adds the following columns:
        prev_close         — yesterday's Close
        gap_pts            — Open - prev_close  (signed, in price units)
        gap_pct            — gap_pts / prev_close  (signed fraction)
        abs_gap_pct        — |gap_pct|
        gap_dir            — 'up', 'down', 'flat'
        gap_bucket         — size tier label (e.g. '0.4–0.75%')
        gap_filled         — bool: did price touch prev_close intraday?
        gap_fill_pct       — fraction of gap closed (0–1+)
        gap_fill_reversal  — bool: after fill, did price reverse beyond prev_close?
        reversal_pts       — signed pts from prev_close at Close (>0 = bullish reversal)
        reversal_pct       — reversal_pts / prev_close
        fill_then_reversal — bool: gap_filled AND gap_fill_reversal
        session_return_pct — (Close - Open) / Open  (intraday return)
        gap_close_pct      — (Close - prev_close) / prev_close (net move from fill level)
    """
    df = daily.copy()
    df["prev_close"] = df["Close"].shift(1)

    df["gap_pts"]    = df["Open"] - df["prev_close"]
    df["gap_pct"]    = df["gap_pts"] / df["prev_close"]
    df["abs_gap_pct"] = df["gap_pct"].abs()

    # Direction
    df["gap_dir"] = "flat"
    df.loc[df["gap_pct"] >  DEFAULT_MIN_GAP_PCT, "gap_dir"] = "up"
    df.loc[df["gap_pct"] < -DEFAULT_MIN_GAP_PCT, "gap_dir"] = "down"

    # Size bucket
    df["gap_bucket"] = pd.cut(
        df["abs_gap_pct"],
        bins=GAP_BUCKET_EDGES,
        labels=GAP_BUCKET_LABELS,
        right=False,
    )
    df.loc[df["gap_dir"] == "flat", "gap_bucket"] = None

    # Gap fill (daily proxy)
    gap_up_filled   = (df["gap_dir"] == "up")   & (df["Low"]  <= df["prev_close"])
    gap_down_filled = (df["gap_dir"] == "down")  & (df["High"] >= df["prev_close"])
    df["gap_filled"] = gap_up_filled | gap_down_filled
    df.loc[df["gap_dir"] == "flat", "gap_filled"] = False

    # Fraction of gap filled: how far did price travel toward prev_close from open
    # For gap_up:  fill_pct = (Open - min(Low, prev_close)) / gap_pts
    # For gap_down: fill_pct = (max(High, prev_close) - Open) / (-gap_pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill_frac_up   = (df["Open"] - df["Low"]) / df["gap_pts"].abs()
        fill_frac_down = (df["High"] - df["Open"]) / df["gap_pts"].abs()
    df["gap_fill_pct"] = np.where(
        df["gap_dir"] == "up",   fill_frac_up.clip(0, 2),
        np.where(df["gap_dir"] == "down", fill_frac_down.clip(0, 2), 0.0)
    )

    # Reversal: after fill, did price trade beyond prev_close in the opposite direction?
    # Gap-up filled + Close > prev_close → price bounced back above fill → bullish reversal
    # Gap-down filled + Close < prev_close → price fell back below fill → bearish reversal
    up_reversal   = gap_up_filled   & (df["Close"] > df["prev_close"])
    down_reversal = gap_down_filled & (df["Close"] < df["prev_close"])
    df["gap_fill_reversal"] = up_reversal | down_reversal
    df.loc[df["gap_dir"] == "flat", "gap_fill_reversal"] = False

    # Convenience flag
    df["fill_then_reversal"] = df["gap_filled"] & df["gap_fill_reversal"]

    # Reversal magnitude from the fill level (prev_close)
    # Positive = bullish (closed above prev_close), Negative = bearish
    df["reversal_pts"] = df["Close"] - df["prev_close"]
    df["reversal_pct"] = df["reversal_pts"] / df["prev_close"]

    # Session return
    df["session_return_pct"] = (df["Close"] - df["Open"]) / df["Open"]

    # Net close vs prev_close
    df["gap_close_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"]

    return df


# ─── Statistics tables ────────────────────────────────────────────────────────


def gap_stats_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fill rate, reversal rate, and magnitudes by gap size bucket.

    Returns a DataFrame indexed by gap_bucket with columns:
        n               — number of gap sessions
        fill_rate       — fraction that filled same session
        reversal_rate   — fraction of filled gaps that reversed
        fill_then_rev   — combined: filled AND reversed
        avg_rev_pts     — mean reversal_pts on fill+reversal days
        avg_rev_pct     — mean reversal_pct on fill+reversal days
        med_rev_pts     — median reversal_pts on fill+reversal days
        avg_gap_size    — mean abs_gap_pct for this bucket
    """
    gap_days = df[df["gap_dir"].isin(["up", "down"])].copy()
    if gap_days.empty:
        return pd.DataFrame()

    def _bucket_stats(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        filled = grp["gap_filled"]
        fill_rate = filled.mean() if n > 0 else float("nan")
        reversed_after_fill = grp.loc[filled, "gap_fill_reversal"]
        reversal_rate = reversed_after_fill.mean() if filled.sum() > 0 else float("nan")
        ftr = grp["fill_then_reversal"]
        rev_days = grp[ftr]
        avg_rev_pts = rev_days["reversal_pts"].mean() if len(rev_days) > 0 else float("nan")
        avg_rev_pct = rev_days["reversal_pct"].mean() if len(rev_days) > 0 else float("nan")
        med_rev_pts = rev_days["reversal_pts"].median() if len(rev_days) > 0 else float("nan")
        avg_gap = grp["abs_gap_pct"].mean()
        return pd.Series({
            "n":            n,
            "fill_rate":    fill_rate,
            "reversal_rate": reversal_rate,
            "fill_then_rev": ftr.mean(),
            "avg_rev_pts":  avg_rev_pts,
            "avg_rev_pct":  avg_rev_pct,
            "med_rev_pts":  med_rev_pts,
            "avg_gap_size": avg_gap,
        })

    stats = (
        gap_days.groupby("gap_bucket", observed=True)
                .apply(_bucket_stats)
                .reindex(GAP_BUCKET_LABELS)
    )
    return stats


def gap_stats_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Fill + reversal stats split by gap direction (up vs down)."""
    gap_days = df[df["gap_dir"].isin(["up", "down"])].copy()
    if gap_days.empty:
        return pd.DataFrame()

    rows = []
    for direction in ("up", "down"):
        g = gap_days[gap_days["gap_dir"] == direction]
        if g.empty:
            continue
        filled = g["gap_filled"]
        ftr = g["fill_then_reversal"]
        rev_days = g[ftr]
        rows.append({
            "Direction":      "Gap Up ↑" if direction == "up" else "Gap Down ↓",
            "Sessions":       len(g),
            "Fill Rate":      filled.mean(),
            "Reversal Rate":  g.loc[filled, "gap_fill_reversal"].mean() if filled.sum() > 0 else float("nan"),
            "Fill+Rev Rate":  ftr.mean(),
            "Avg Rev Pts":    rev_days["reversal_pts"].mean() if len(rev_days) > 0 else float("nan"),
            "Med Rev Pts":    rev_days["reversal_pts"].median() if len(rev_days) > 0 else float("nan"),
            "Avg Rev %":      rev_days["reversal_pct"].mean() if len(rev_days) > 0 else float("nan"),
            "Avg Gap Size":   g["abs_gap_pct"].mean(),
        })
    return pd.DataFrame(rows)


def gap_stats_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Fill + reversal stats by day of week."""
    gap_days = df[df["gap_dir"].isin(["up", "down"])].copy()
    if gap_days.empty:
        return pd.DataFrame()

    wd_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    gap_days["weekday"] = gap_days.index.dayofweek.map(wd_map)

    rows = []
    for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        g = gap_days[gap_days["weekday"] == wd]
        if g.empty:
            continue
        ftr = g["fill_then_reversal"]
        rev_days = g[ftr]
        rows.append({
            "Weekday":        wd,
            "Gap Sessions":   len(g),
            "Fill Rate":      g["gap_filled"].mean(),
            "Fill+Rev Rate":  ftr.mean(),
            "Avg Rev Pts":    rev_days["reversal_pts"].mean() if len(rev_days) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def recent_gap_trades(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Return the most recent N gap fill+reversal days with outcome details."""
    gap_days = df[df["gap_dir"].isin(["up", "down"])].copy()
    gap_days = gap_days.sort_index(ascending=False).head(n * 3)  # look through more to find FTR

    rows = []
    for date, row in gap_days.iterrows():
        rows.append({
            "Date":           date.strftime("%Y-%m-%d"),
            "Weekday":        date.strftime("%a"),
            "Dir":            "↑ Up" if row["gap_dir"] == "up" else "↓ Down",
            "Gap %":          row["gap_pct"],
            "Gap Pts":        row["gap_pts"],
            "Fill Level":     row["prev_close"],
            "Filled":         "✅" if row["gap_filled"] else "❌",
            "Reversed":       "✅" if row["gap_fill_reversal"] else ("—" if not row["gap_filled"] else "❌"),
            "Rev Pts":        row["reversal_pts"] if row["fill_then_reversal"] else float("nan"),
            "Close":          row["Close"],
        })

    return pd.DataFrame(rows).head(n)


# ─── Today's gap status ───────────────────────────────────────────────────────


@dataclass
class TodayGap:
    ticker:           str
    today_date:       str
    gap_dir:          str           # 'up', 'down', 'flat'
    gap_pct:          float         # signed fraction
    gap_pts:          float         # signed price units
    open_price:       float
    prev_close:       float         # fill target level
    fill_level:       float         # same as prev_close
    distance_to_fill_pct: float     # |current_open - fill_level| / fill_level
    # Historical context (from stats)
    hist_fill_rate:   Optional[float]
    hist_rev_rate:    Optional[float]
    hist_avg_rev_pts: Optional[float]
    hist_med_rev_pts: Optional[float]
    hist_n_similar:   int
    # Signal
    signal:           str           # 'WATCH_FILL', 'NEAR_FILL', 'NO_GAP', 'SMALL_GAP'
    signal_detail:    str


def today_gap_status(
    ticker: str,
    df_with_features: pd.DataFrame,
    stats_by_bucket: pd.DataFrame,
) -> TodayGap:
    """Evaluate today's gap and compute the fill signal.

    `df_with_features` must already have add_gap_features applied and
    the last row must be today (or the most recent session).
    """
    last = df_with_features.iloc[-1]
    hist = df_with_features.iloc[:-1]  # exclude today for clean stats

    gap_dir  = last["gap_dir"]
    gap_pct  = float(last["gap_pct"])
    gap_pts  = float(last["gap_pts"])
    open_px  = float(last["Open"])
    prev_cls = float(last["prev_close"])

    # Distance from open to fill level
    dist_to_fill_pct = abs(open_px - prev_cls) / prev_cls if prev_cls > 0 else 0.0

    # Look up historical stats for this bucket
    bucket = last["gap_bucket"]
    hist_fill_rate = hist_rev_rate = hist_avg_rev = hist_med_rev = None
    hist_n = 0

    if bucket is not None and not pd.isna(bucket) and not stats_by_bucket.empty:
        bkey = str(bucket)
        if bkey in stats_by_bucket.index:
            row = stats_by_bucket.loc[bkey]
            hist_fill_rate   = float(row["fill_rate"])   if pd.notna(row["fill_rate"])   else None
            hist_rev_rate    = float(row["reversal_rate"]) if pd.notna(row["reversal_rate"]) else None
            hist_avg_rev     = float(row["avg_rev_pts"])  if pd.notna(row["avg_rev_pts"])  else None
            hist_med_rev     = float(row["med_rev_pts"])  if pd.notna(row["med_rev_pts"])  else None
            hist_n           = int(row["n"]) if pd.notna(row["n"]) else 0

    # Signal classification
    if gap_dir == "flat":
        signal = "NO_GAP"
        detail = "No significant gap today."
    elif abs(gap_pct) < DEFAULT_MIN_GAP_PCT:
        signal = "SMALL_GAP"
        detail = f"Gap too small ({gap_pct*100:.2f}%) to classify."
    elif hist_fill_rate is not None and hist_fill_rate >= 0.70:
        signal = "WATCH_FILL"
        fill_rate_pct = hist_fill_rate * 100
        rev_pts_str = f"{hist_med_rev:+.2f} pts" if hist_med_rev is not None else "unknown"
        detail = (
            f"{'Gap up' if gap_dir == 'up' else 'Gap down'} {gap_pct*100:+.2f}% "
            f"({gap_pts:+.2f} pts). Fill target: {prev_cls:.2f}. "
            f"Historical fill rate: {fill_rate_pct:.0f}%. "
            f"When filled, median reversal: {rev_pts_str}."
        )
    elif hist_fill_rate is not None and hist_fill_rate >= 0.50:
        signal = "NEAR_FILL"
        detail = (
            f"{'Gap up' if gap_dir == 'up' else 'Gap down'} {gap_pct*100:+.2f}%"
            f" ({gap_pts:+.2f} pts). Fill target: {prev_cls:.2f}. "
            f"Historical fill rate: {hist_fill_rate*100:.0f}%."
        )
    else:
        signal = "MONITOR"
        detail = (
            f"{'Gap up' if gap_dir == 'up' else 'Gap down'} {gap_pct*100:+.2f}%"
            f" ({gap_pts:+.2f} pts). Fill target: {prev_cls:.2f}. "
            f"Fill rate: {hist_fill_rate*100:.0f}% — lower edge, monitor only."
            if hist_fill_rate is not None else
            f"{'Gap up' if gap_dir == 'up' else 'Gap down'} {gap_pct*100:+.2f}% — insufficient history."
        )

    return TodayGap(
        ticker=ticker,
        today_date=str(df_with_features.index[-1].date()),
        gap_dir=gap_dir,
        gap_pct=gap_pct,
        gap_pts=gap_pts,
        open_price=open_px,
        prev_close=prev_cls,
        fill_level=prev_cls,
        distance_to_fill_pct=dist_to_fill_pct,
        hist_fill_rate=hist_fill_rate,
        hist_rev_rate=hist_rev_rate,
        hist_avg_rev_pts=hist_avg_rev,
        hist_med_rev_pts=hist_med_rev,
        hist_n_similar=hist_n,
        signal=signal,
        signal_detail=detail,
    )


# ─── Full pipeline ────────────────────────────────────────────────────────────


def run_gap_analysis(
    ticker: str,
    daily: pd.DataFrame,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, "TodayGap"]:
    """Run the full gap analysis pipeline for one ticker.

    Returns:
        df_feat       — daily DataFrame with all gap feature columns
        stats_bucket  — gap stats by size bucket
        stats_dir     — gap stats by direction (up/down)
        stats_wd      — gap stats by weekday
        today         — TodayGap dataclass with today's signal
    """
    # Trim to lookback
    cutoff = daily.index[-1] - pd.DateOffset(years=lookback_years)
    df = daily[daily.index >= cutoff].copy()

    # Feature engineering
    df_feat = add_gap_features(df)

    # Stats (exclude today so we don't contaminate with partially known session)
    hist = df_feat.iloc[:-1]

    stats_bucket = gap_stats_by_bucket(hist)
    stats_dir    = gap_stats_by_direction(hist)
    stats_wd     = gap_stats_by_weekday(hist)

    # Today's signal
    today = today_gap_status(ticker, df_feat, stats_bucket)

    return df_feat, stats_bucket, stats_dir, stats_wd, today
