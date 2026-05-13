"""
Regime / volatility-context features.

Adds Tier-1 indicators that the data showed would have flipped many of the
losing trades in the 6-month walk-forward backtest:

  * vix              — absolute VIX close
  * vix_5d_chg       — 5-day % change in VIX  (>15% = post-blowout, IV crush risk)
  * vix_term         — VIX / VIX3M             (>1.0 = backwardation = stress)
  * z_20d            — SPY distance from 20d SMA in std-devs (extension)
  * rsi2             — Connors 2-period RSI    (<10 = overdone)

VIX-family series are pulled from yfinance and cached to disk with a short TTL.
SPY-derived features (z_20d, rsi2) are computed from the OHLCV frame already
loaded by the scanner, so they cost nothing.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VIX_CACHE_TTL_HOURS = 6


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "").replace("/", "_")
    return DATA_DIR / f"regime_{safe}.csv"


def _fetch_vix_series(symbol: str, lookback_years: int = 8) -> pd.Series:
    """Pull a VIX-family close series from yfinance, with disk cache."""
    cache = _cache_path(symbol)
    if cache.exists():
        age = datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)
        if age < timedelta(hours=VIX_CACHE_TTL_HOURS):
            try:
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
                return df["Close"].astype(float)
            except Exception:
                pass

    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=365 * lookback_years)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name="Close")

    # Flatten potential multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    closes = df["Close"].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes[~closes.index.duplicated(keep="last")].dropna()

    out = closes.to_frame(name="Close")
    try:
        out.to_csv(cache)
    except Exception:
        pass
    return out["Close"].astype(float)


def build_regime_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of regime features aligned to ``daily.index``.

    Parameters
    ----------
    daily : DataFrame
        SPY (or other ticker) daily OHLCV with DatetimeIndex.

    Returns
    -------
    DataFrame indexed by date with columns:
        vix, vix_5d_chg, vix_term, z_20d, rsi2

    All values use only information available BEFORE the session open
    (each value is shifted by 1 trading day) so the features can be safely
    fed into the walk-forward classifier without lookahead.
    """
    df = daily.sort_index().copy()
    idx = pd.to_datetime(df.index).tz_localize(None).normalize()

    # ---- VIX-family ---------------------------------------------------------
    vix   = _fetch_vix_series("^VIX")
    vix3m = _fetch_vix_series("^VIX3M")

    if vix.empty:
        vix   = pd.Series(np.nan, index=idx)
    if vix3m.empty:
        vix3m = pd.Series(np.nan, index=idx)

    # Reindex to daily session dates (forward-fill across non-trading anomalies)
    vix   = vix.reindex(idx, method="ffill")
    vix3m = vix3m.reindex(idx, method="ffill")

    vix_5d_chg = vix.pct_change(5)
    vix_term   = vix / vix3m.replace(0.0, np.nan)

    # ---- SPY-derived (cheap, from existing OHLCV) ---------------------------
    close = df["Close"].copy()
    close.index = idx

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    z_20d = (close - sma20) / std20.replace(0.0, np.nan)

    # Connors RSI(2)
    delta = close.diff()
    up    = delta.clip(lower=0.0)
    down  = -delta.clip(upper=0.0)
    avg_up   = up.rolling(2).mean()
    avg_down = down.rolling(2).mean()
    rs   = avg_up / avg_down.replace(0.0, np.nan)
    rsi2 = 100.0 - 100.0 / (1.0 + rs)

    feats = pd.DataFrame(
        {
            "vix":         vix,
            "vix_5d_chg":  vix_5d_chg,
            "vix_term":    vix_term,
            "z_20d":       z_20d,
            "rsi2":        rsi2,
        },
        index=idx,
    )

    # Shift by one session — features must be knowable BEFORE the trade day open
    feats = feats.shift(1)

    # Re-attach original (possibly tz-aware) index from `daily`
    feats.index = df.index
    return feats
