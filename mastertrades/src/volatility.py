"""Daily volatility estimators that use OHLC efficiently.

Close-to-close standard deviation throws away the High/Low information that
sits inside every bar. The estimators below extract that information so we
get a much sharper read on per-day volatility — important when we want to
classify days as "volatile" vs "flat".

References:
    Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991),
    Yang & Zhang (2000).

All `*_vol` functions return *annualized* volatility (sqrt(periods_per_year)
applied) computed over a rolling window. Use ``periods_per_year=252`` for
daily SPY data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range, in price units.

    TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|).
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    return pd.concat(
        [(h - l), (h - pc).abs(), (l - pc).abs()],
        axis=1,
    ).max(axis=1).rename("TrueRange")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range, simple moving average of TR."""
    return true_range(df).rolling(n).mean().rename(f"ATR_{n}")


def daily_range_pct(df: pd.DataFrame) -> pd.Series:
    """(High - Low) / Open, the simplest per-day volatility proxy."""
    return ((df["High"] - df["Low"]) / df["Open"]).rename("RangePct")


def body_pct(df: pd.DataFrame) -> pd.Series:
    """|Close - Open| / Open. How much *directional* move happened today."""
    return ((df["Close"] - df["Open"]).abs() / df["Open"]).rename("BodyPct")


def efficiency_ratio(df: pd.DataFrame) -> pd.Series:
    """Body / Range. 1.0 = pure trend day, ~0 = pure chop with no net move.

    A "flat" day has BOTH a small range and a small body (or, equivalently,
    a small range — the body is bounded above by the range). A "rangy but
    flat" day (high range, near-zero body) shows up as a small efficiency
    ratio, which is useful for distinguishing chop from quiet.
    """
    rng = (df["High"] - df["Low"])
    body = (df["Close"] - df["Open"]).abs()
    er = body / rng.replace(0.0, np.nan)
    return er.fillna(0.0).rename("EfficiencyRatio")


def parkinson_vol(
    df: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Parkinson (1980) high-low estimator. Ignores overnight moves."""
    hl = np.log(df["High"] / df["Low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    var = factor * hl.rolling(window).mean()
    return np.sqrt(periods_per_year * var).rename(f"ParkinsonVol_{window}")


def garman_klass_vol(
    df: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Garman-Klass (1980) estimator. Uses OHLC; ignores overnight gaps."""
    hl = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    co = (2.0 * np.log(2.0) - 1.0) * np.log(df["Close"] / df["Open"]) ** 2
    var = (hl - co).rolling(window).mean()
    return np.sqrt(periods_per_year * var.clip(lower=0)).rename(f"GarmanKlassVol_{window}")


def rogers_satchell_vol(
    df: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rogers-Satchell (1991). Drift-independent, ignores overnight gaps."""
    log_h_o = np.log(df["High"] / df["Open"])
    log_h_c = np.log(df["High"] / df["Close"])
    log_l_o = np.log(df["Low"] / df["Open"])
    log_l_c = np.log(df["Low"] / df["Close"])
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    var = rs.rolling(window).mean()
    return np.sqrt(periods_per_year * var.clip(lower=0)).rename(f"RogersSatchellVol_{window}")


def yang_zhang_vol(
    df: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Yang-Zhang (2000) — best general-purpose estimator for stocks/ETFs.

    Decomposes daily variance into overnight, open-to-close, and Rogers-
    Satchell components, then blends them with the optimal weight ``k``.
    Captures gap risk *and* intraday range. This is the one to default to.
    """
    log_o_pc = np.log(df["Open"] / df["Close"].shift(1))     # overnight
    log_c_o = np.log(df["Close"] / df["Open"])               # open-to-close
    log_h_o = np.log(df["High"] / df["Open"])
    log_h_c = np.log(df["High"] / df["Close"])
    log_l_o = np.log(df["Low"] / df["Open"])
    log_l_c = np.log(df["Low"] / df["Close"])

    sigma_o2 = log_o_pc.rolling(window).var()
    sigma_c2 = log_c_o.rolling(window).var()
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    sigma_rs2 = rs.rolling(window).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return np.sqrt(periods_per_year * var.clip(lower=0)).rename(f"YangZhangVol_{window}")


def realized_vol_from_intraday(
    intraday: pd.DataFrame,
    column: str = "Close",
    periods_per_year: int = 252,
) -> pd.Series:
    """Per-day realized volatility from intraday bars.

    Sums squared intraday log returns within each session, takes sqrt, and
    annualizes. This is a model-free volatility estimate that's much sharper
    than any OHLC-based estimator if you can afford the data.

    Parameters
    ----------
    intraday:
        DataFrame indexed by intraday timestamps with the price column.
    column:
        Price column to use ("Close" by default).
    """
    if not isinstance(intraday.index, pd.DatetimeIndex):
        raise TypeError("intraday must have a DatetimeIndex.")

    px = intraday[column].astype(float)
    log_ret = np.log(px / px.shift(1))
    daily_rv = (log_ret ** 2).groupby(log_ret.index.date).sum()
    daily_rv.index = pd.to_datetime(daily_rv.index)
    return np.sqrt(daily_rv * periods_per_year).rename("RealizedVol")


def all_daily_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute the full per-day volatility feature set in one call."""
    return pd.concat(
        [
            true_range(df),
            atr(df, n=window),
            daily_range_pct(df),
            body_pct(df),
            efficiency_ratio(df),
            parkinson_vol(df, window=window),
            garman_klass_vol(df, window=window),
            yang_zhang_vol(df, window=window),
        ],
        axis=1,
    )
