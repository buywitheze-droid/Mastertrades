"""Order-flow *proxies* derived from intraday OHLCV bars.

True order flow analysis (footprint charts, CVD from print-by-print bid/ask
classification, Lee-Ready trade signing, VPIN, market-by-order replay) needs
tick data with bid/ask quotes. Yahoo Finance does not provide that.

What we *can* do with hourly or 5-minute OHLCV is build proxies that
correlate reasonably well with the real thing for ETFs that trade close to
NAV (SPY, QQQ, IWM):

- **Signed volume**: assign each bar's volume a sign from its return
  (sign(close - open) * volume). This is a *bar-level* tick-rule proxy.
- **Cumulative Volume Delta (CVD)**: cumulative signed volume across a
  session. Diverges from price during absorption/distribution.
- **VWAP** and **VWAP deviation** at session close. Persistent closes far
  from VWAP suggest one-sided flow.
- **Closing strength**: where the close sits inside the day's range.
  (Close - Low) / (High - Low). High = buyers won, low = sellers won.
- **Volume profile**: volume traded at each price bucket within a session,
  giving Point of Control (POC), Value Area High/Low.

These proxies are NOT a substitute for true tick-level order flow. They are
useful for regime classification but should not be used to call individual
trade prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _session_groupby(intraday: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    if not isinstance(intraday.index, pd.DatetimeIndex):
        raise TypeError("intraday must have a DatetimeIndex.")
    return intraday.groupby(intraday.index.normalize())


def signed_volume(intraday: pd.DataFrame) -> pd.Series:
    """Per-bar signed volume from the bar's open->close direction."""
    direction = np.sign(intraday["Close"] - intraday["Open"]).replace(0, np.nan).ffill().fillna(0)
    return (direction * intraday["Volume"]).rename("SignedVolume")


def cumulative_volume_delta(intraday: pd.DataFrame) -> pd.Series:
    """Per-bar CVD that resets each session."""
    sv = signed_volume(intraday)
    return sv.groupby(intraday.index.normalize()).cumsum().rename("CVD")


def session_cvd_close(intraday: pd.DataFrame) -> pd.Series:
    """Final CVD value of each session, indexed by date.

    Positive = buyers dominated for the day, negative = sellers.
    Useful as a daily order-flow imbalance score.
    """
    sv = signed_volume(intraday)
    daily = sv.groupby(intraday.index.normalize()).sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.rename("CVD_close")


def session_vwap(intraday: pd.DataFrame) -> pd.Series:
    """Per-bar running VWAP that resets each session."""
    typical = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3.0
    pv = typical * intraday["Volume"]
    cum_pv = pv.groupby(intraday.index.normalize()).cumsum()
    cum_v = intraday["Volume"].groupby(intraday.index.normalize()).cumsum()
    return (cum_pv / cum_v).rename("VWAP")


def session_vwap_close_deviation(intraday: pd.DataFrame) -> pd.Series:
    """End-of-session (close - vwap) / vwap, indexed by date.

    Strong positive = persistent buying pressure into the close.
    Strong negative = persistent selling pressure.
    """
    vwap = session_vwap(intraday)
    last = pd.DataFrame(
        {"close": intraday["Close"], "vwap": vwap}
    ).groupby(intraday.index.normalize()).tail(1)
    last.index = pd.to_datetime(last.index.normalize())
    dev = (last["close"] - last["vwap"]) / last["vwap"]
    return dev.rename("VWAPDeviation")


def closing_strength(daily: pd.DataFrame) -> pd.Series:
    """(Close - Low) / (High - Low), per day. 1=close at high, 0=close at low.

    A simple but durable proxy for end-of-day buying vs selling pressure.
    """
    rng = daily["High"] - daily["Low"]
    return ((daily["Close"] - daily["Low"]) / rng.replace(0.0, np.nan)).fillna(0.5).rename(
        "ClosingStrength"
    )


def volume_profile(
    intraday: pd.DataFrame,
    bins: int = 50,
) -> pd.DataFrame:
    """Volume traded in each price bucket across the entire intraday frame.

    Returns a DataFrame with columns ``price`` (bucket center) and ``volume``,
    sorted by volume descending so row 0 is the Point of Control (POC).
    """
    px = intraday["Close"].astype(float)
    vol = intraday["Volume"].astype(float)
    edges = np.linspace(px.min(), px.max(), bins + 1)
    bucket = np.digitize(px, edges) - 1
    bucket = np.clip(bucket, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = pd.DataFrame({"price": centers[bucket], "volume": vol.values})
    return (
        profile.groupby("price", as_index=False)["volume"]
        .sum()
        .sort_values("volume", ascending=False, ignore_index=True)
    )


def value_area(profile: pd.DataFrame, coverage: float = 0.70) -> dict:
    """Compute Point of Control and Value Area from a volume profile.

    The value area is the contiguous range around the POC that contains
    ``coverage`` (default 70%) of total traded volume.
    """
    if profile.empty:
        return {"poc": np.nan, "val": np.nan, "vah": np.nan}

    sorted_by_price = profile.sort_values("price").reset_index(drop=True)
    poc_idx = sorted_by_price["volume"].idxmax()
    poc_price = sorted_by_price.loc[poc_idx, "price"]
    total = sorted_by_price["volume"].sum()
    target = total * coverage

    lo = hi = poc_idx
    accum = sorted_by_price.loc[poc_idx, "volume"]
    while accum < target and (lo > 0 or hi < len(sorted_by_price) - 1):
        up_vol = sorted_by_price["volume"].iloc[hi + 1] if hi < len(sorted_by_price) - 1 else -1
        down_vol = sorted_by_price["volume"].iloc[lo - 1] if lo > 0 else -1
        if up_vol >= down_vol:
            hi += 1
            accum += up_vol
        else:
            lo -= 1
            accum += down_vol

    return {
        "poc": float(poc_price),
        "val": float(sorted_by_price.loc[lo, "price"]),
        "vah": float(sorted_by_price.loc[hi, "price"]),
        "coverage": float(accum / total),
    }


def daily_order_flow_features(
    daily: pd.DataFrame,
    intraday: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bundle of per-day order-flow proxies suitable for joining onto a daily frame.

    Always returns ``ClosingStrength`` (needs daily only). When ``intraday``
    is supplied, also returns ``CVD_close`` and ``VWAPDeviation``.
    """
    feats = {"ClosingStrength": closing_strength(daily)}
    if intraday is not None and not intraday.empty:
        feats["CVD_close"] = session_cvd_close(intraday)
        feats["VWAPDeviation"] = session_vwap_close_deviation(intraday)
    return pd.concat(feats.values(), axis=1)
