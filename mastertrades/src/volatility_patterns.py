"""Find structural patterns shared by SPY's most volatile days.

Engineers ~25 features per session that are knowable BEFORE the session
opens — calendar position, lagged price/volume/range, compression/expansion
flags, volatility-clustering metrics, technical position — and ranks them by
how strongly they predict whether the session will be in the top quintile of
realized intraday range.

No news, no economic-calendar event tagging, no FOMC dates. Only structure
already in the OHLCV bars and the calendar.

Usage::

    from src.volatility_patterns import build_features, find_patterns
    feats = build_features(daily)
    patterns = find_patterns(feats, volatile_quantile=0.80)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """3rd Friday of (year, month) — monthly options expiration."""
    first = pd.Timestamp(year=year, month=month, day=1)
    days_until_friday = (4 - first.weekday()) % 7
    first_friday = first + pd.Timedelta(days=days_until_friday)
    return first_friday + pd.Timedelta(days=14)


def _opex_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """All monthly OpEx Fridays roughly bracketing the window."""
    pad = pd.DateOffset(months=2)
    cur = (start - pad).to_period("M").to_timestamp()
    last = (end + pad).to_period("M").to_timestamp()
    out = []
    while cur <= last:
        out.append(_third_friday(cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    return pd.DatetimeIndex(out)


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-session feature panel. Index = session date.

    The target column is ``range_pct`` (today's intraday range). Every other
    column uses ONLY information knowable before today's session (calendar
    facts, prior bars' OHLCV, and the gap from prior close to today's open).
    """
    df = daily.sort_index().copy()

    range_pct = ((df["High"] - df["Low"]) / df["Open"]).rename("range_pct")
    body_pct = ((df["Close"] - df["Open"]).abs() / df["Open"])
    intraday_ret = (df["Close"] - df["Open"]) / df["Open"]
    closing_strength = (
        (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0.0, np.nan)
    ).fillna(0.5)

    # ---- Calendar -----------------------------------------------------------
    weekday = df.index.day_name()
    week_of_month = ((df.index.day - 1) // 7 + 1).astype(int)

    opex_dates = _opex_calendar(df.index.min(), df.index.max())
    opex_dates_norm = pd.DatetimeIndex([d.normalize() for d in opex_dates])
    next_pos = opex_dates_norm.searchsorted(df.index)
    next_pos = np.clip(next_pos, 0, len(opex_dates_norm) - 1)
    next_opex = opex_dates_norm[next_pos]
    days_to_opex = (next_opex - df.index).days

    is_opex_day = pd.Series(df.index.normalize().isin(opex_dates_norm), index=df.index).astype(int)
    is_opex_week = ((days_to_opex >= 0) & (days_to_opex <= 4)).astype(int)
    is_quarterly_opex_week = (is_opex_week & df.index.month.isin([3, 6, 9, 12])).astype(int)

    grouped = df.groupby([df.index.year, df.index.month])
    bd_in_month = grouped.cumcount() + 1
    days_in_month = grouped["Close"].transform("size")
    days_left_in_month = days_in_month - bd_in_month
    is_turn_of_month = ((bd_in_month <= 2) | (days_left_in_month <= 1)).astype(int)
    is_first_trading_day_of_month = (bd_in_month == 1).astype(int)
    is_last_trading_day_of_month = (days_left_in_month == 0).astype(int)

    # ---- Lagged price / range / color --------------------------------------
    lag1_range = range_pct.shift(1)
    lag2_range = range_pct.shift(2)
    lag5_avg_range = range_pct.shift(1).rolling(5).mean()
    lag20_avg_range = range_pct.shift(1).rolling(20).mean()
    range_compression_ratio = (lag5_avg_range / lag20_avg_range)

    lag1_body = body_pct.shift(1)
    lag1_color = np.sign(intraday_ret.shift(1)).fillna(0).astype(int)
    lag1_close_strength = closing_strength.shift(1)

    gap_pct = (df["Open"] / df["Close"].shift(1) - 1.0)
    abs_gap_pct = gap_pct.abs()

    # ---- Compression / expansion patterns ----------------------------------
    is_lag1_nr4 = (lag1_range == range_pct.rolling(4).min().shift(1)).astype(int)
    is_lag1_nr7 = (lag1_range == range_pct.rolling(7).min().shift(1)).astype(int)

    rolling_q20 = range_pct.rolling(60, min_periods=20).quantile(0.20)
    rolling_q80 = range_pct.rolling(60, min_periods=20).quantile(0.80)

    is_after_flat = (lag1_range < rolling_q20.shift(1)).astype(int)
    is_after_volatile = (lag1_range > rolling_q80.shift(1)).astype(int)

    below = (range_pct < rolling_q20).astype(int)
    flat_streak = below.groupby((1 - below).cumsum()).cumsum().shift(1).fillna(0).astype(int)
    is_after_2_flat = (flat_streak >= 2).astype(int)
    is_after_3plus_flat = (flat_streak >= 3).astype(int)

    # ---- Volume ------------------------------------------------------------
    vol_mean60 = df["Volume"].rolling(60).mean()
    vol_std60 = df["Volume"].rolling(60).std()
    volume_z = (df["Volume"] - vol_mean60) / vol_std60.replace(0.0, np.nan)
    lag1_volume_z = volume_z.shift(1)
    lag5_avg_volume_z = volume_z.shift(1).rolling(5).mean()

    # ---- Volatility clustering --------------------------------------------
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    realized_vol_5d = log_ret.shift(1).rolling(5).std() * np.sqrt(252)
    realized_vol_20d = log_ret.shift(1).rolling(20).std() * np.sqrt(252)
    vol_regime_shift = (realized_vol_5d - realized_vol_20d)  # >0 = vol expanding

    # ---- Technical position ------------------------------------------------
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    dist_ma20 = (df["Close"] / ma20 - 1.0).shift(1)
    dist_ma50 = (df["Close"] / ma50 - 1.0).shift(1)
    dist_ma200 = (df["Close"] / ma200 - 1.0).shift(1)

    rsi14 = _rsi(df["Close"], n=14).shift(1)

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    bb_pos = ((df["Close"] - (sma20 - 2.0 * std20)) / (4.0 * std20)).shift(1)

    high20 = df["High"].rolling(20).max()
    low20 = df["Low"].rolling(20).min()
    pct_in_20d_range = ((df["Close"] - low20) / (high20 - low20).replace(0.0, np.nan)).shift(1)

    return pd.DataFrame(
        {
            "range_pct": range_pct,
            "weekday": weekday,
            "week_of_month": week_of_month,
            "days_to_opex": days_to_opex,
            "is_opex_day": is_opex_day,
            "is_opex_week": is_opex_week,
            "is_quarterly_opex_week": is_quarterly_opex_week,
            "is_turn_of_month": is_turn_of_month,
            "is_first_trading_day_of_month": is_first_trading_day_of_month,
            "is_last_trading_day_of_month": is_last_trading_day_of_month,
            "lag1_range": lag1_range,
            "lag2_range": lag2_range,
            "lag5_avg_range": lag5_avg_range,
            "range_compression_ratio": range_compression_ratio,
            "lag1_body": lag1_body,
            "lag1_color": lag1_color,
            "lag1_close_strength": lag1_close_strength,
            "gap_pct": gap_pct,
            "abs_gap_pct": abs_gap_pct,
            "is_lag1_nr4": is_lag1_nr4,
            "is_lag1_nr7": is_lag1_nr7,
            "is_after_flat": is_after_flat,
            "is_after_volatile": is_after_volatile,
            "is_after_2_flat": is_after_2_flat,
            "is_after_3plus_flat": is_after_3plus_flat,
            "lag1_volume_z": lag1_volume_z,
            "lag5_avg_volume_z": lag5_avg_volume_z,
            "realized_vol_5d": realized_vol_5d,
            "realized_vol_20d": realized_vol_20d,
            "vol_regime_shift": vol_regime_shift,
            "rsi14": rsi14,
            "dist_ma20": dist_ma20,
            "dist_ma50": dist_ma50,
            "dist_ma200": dist_ma200,
            "bb_pos": bb_pos,
            "pct_in_20d_range": pct_in_20d_range,
        }
    )


# ---------------------------------------------------------------------------
# Pattern ranking
# ---------------------------------------------------------------------------


CATEGORICAL = {"weekday"}
BINARY = {
    "is_opex_day", "is_opex_week", "is_quarterly_opex_week",
    "is_turn_of_month", "is_first_trading_day_of_month",
    "is_last_trading_day_of_month", "is_lag1_nr4", "is_lag1_nr7",
    "is_after_flat", "is_after_volatile", "is_after_2_flat",
    "is_after_3plus_flat",
}
ORDINAL = {"week_of_month", "lag1_color"}


def find_patterns(
    features: pd.DataFrame,
    volatile_quantile: float = 0.80,
    min_count: int = 5,
) -> dict:
    """Rank features by how strongly they distinguish volatile days.

    Volatile = today's ``range_pct`` is at or above the quantile threshold
    over the input window.

    For binary/categorical features we report **lift** = P(class | volatile)
    / P(class). Lift > 1 means that class is over-represented on volatile
    days. We also report the conditional volatile rate, base rate, and
    sample size per class.

    For continuous features we report the mean on volatile vs non-volatile
    days, plus a Welch t-statistic so big differences with small samples
    don't dominate the ranking.
    """
    feat = features.dropna(subset=["range_pct"]).copy()
    threshold = feat["range_pct"].quantile(volatile_quantile)
    feat["is_volatile"] = (feat["range_pct"] >= threshold).astype(int)
    base_rate = float(feat["is_volatile"].mean())

    cat_rows: list[dict] = []
    cont_rows: list[dict] = []

    for col in feat.columns:
        if col in {"range_pct", "is_volatile"}:
            continue
        s = feat[col]

        if col in CATEGORICAL or col in BINARY or col in ORDINAL:
            valid = feat[~s.isna()]
            for cls, group in valid.groupby(col):
                n = len(group)
                if n < min_count:
                    continue
                rate = float(group["is_volatile"].mean())
                cat_rows.append({
                    "feature": col,
                    "class": str(cls),
                    "n": int(n),
                    "p_volatile_given": rate,
                    "base_rate": base_rate,
                    "lift": rate / base_rate if base_rate > 0 else float("nan"),
                })
        else:
            valid = feat[~s.isna()]
            if valid.empty:
                continue
            mu_v = float(valid.loc[valid["is_volatile"] == 1, col].mean())
            mu_n = float(valid.loc[valid["is_volatile"] == 0, col].mean())
            sd_v = float(valid.loc[valid["is_volatile"] == 1, col].std(ddof=1))
            sd_n = float(valid.loc[valid["is_volatile"] == 0, col].std(ddof=1))
            n_v = int((valid["is_volatile"] == 1).sum())
            n_n = int((valid["is_volatile"] == 0).sum())
            denom = np.sqrt((sd_v**2 / max(n_v, 1)) + (sd_n**2 / max(n_n, 1)))
            tstat = (mu_v - mu_n) / denom if denom and not np.isnan(denom) else float("nan")
            cont_rows.append({
                "feature": col,
                "mean_volatile": mu_v,
                "mean_non_volatile": mu_n,
                "diff": mu_v - mu_n,
                "tstat": float(tstat),
                "n_volatile": n_v,
                "n_non_volatile": n_n,
            })

    cat_df = pd.DataFrame(cat_rows).sort_values("lift", ascending=False)
    cont_df = pd.DataFrame(cont_rows)
    cont_df["abs_t"] = cont_df["tstat"].abs()
    cont_df = cont_df.sort_values("abs_t", ascending=False).drop(columns=["abs_t"])

    return {
        "threshold_range_pct": float(threshold),
        "base_rate": base_rate,
        "n_total": int(len(feat)),
        "n_volatile": int(feat["is_volatile"].sum()),
        "categorical": cat_df.reset_index(drop=True),
        "continuous": cont_df.reset_index(drop=True),
    }
