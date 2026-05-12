"""Polygon.io data feed for Mastertrades.

What works with standard Polygon keys
--------------------------------------
✅  /v2/aggs/ticker/{T}/range/1/day/{from}/{to}  — historical OHLCV (all history)
✅  /v2/aggs/ticker/{T}/prev                      — previous trading day OHLCV
✅  /v3/reference/tickers/{T}                     — ticker metadata

❌  /v2/snapshot/...      — requires paid plan (returns 403)
❌  /v2/last/trade/...    — requires paid plan (returns 403)
❌  Today's intraday bar  — requires paid plan (returns 403)

Integration strategy
--------------------
Use Polygon for historical OHLCV (replaces yfinance for daily bars, giving
more reliable, exchange-quality data).  For today's intraday/real-time price
the app falls back to yfinance as-is, since Polygon doesn't provide that on
this key tier.

All public functions return None / empty DataFrame on failure so callers never
crash — they just fall back to yfinance.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"
_LAST_CALL   = 0.0          # simple in-process rate limiter
_MIN_GAP_S   = 0.25         # ≥250 ms between calls (free tier: 5 req/min limit)


# ── Internals ─────────────────────────────────────────────────────────────────

def _key() -> Optional[str]:
    return os.environ.get("POLYGON_API_KEY") or None


def _get(path: str, params: dict | None = None) -> dict:
    global _LAST_CALL
    key = _key()
    if not key:
        raise RuntimeError("POLYGON_API_KEY not configured")

    # Throttle to avoid 429 on free tier
    gap = time.monotonic() - _LAST_CALL
    if gap < _MIN_GAP_S:
        time.sleep(_MIN_GAP_S - gap)

    p = dict(params or {})
    p["apiKey"] = key
    _LAST_CALL = time.monotonic()

    resp = requests.get(f"{POLYGON_BASE}{path}", params=p, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "")
    if status not in ("OK", "DELAYED", "ok", ""):
        raise RuntimeError(
            f"Polygon [{status}]: {data.get('error') or data.get('message', '')}"
        )
    return data


# ── Public helpers ────────────────────────────────────────────────────────────

def has_polygon_key() -> bool:
    """True when POLYGON_API_KEY is present in the environment."""
    return bool(_key())


def fetch_daily_bars(ticker: str, days: int = 365 * 7) -> Optional[pd.DataFrame]:
    """Fetch historical daily OHLCV from Polygon (adjusted for splits/dividends).

    Returns a DataFrame with:
      index   — DatetimeIndex named 'Date' (tz-naive, UTC-aligned)
      columns — Open, High, Low, Close, Volume

    Schema matches yfinance output, so it works as a drop-in replacement.
    Returns None on any failure.
    """
    try:
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=days)
        path = (
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/day"
            f"/{start_dt.isoformat()}/{end_dt.isoformat()}"
        )
        all_bars: list[dict] = []
        next_url: Optional[str] = None

        while True:
            if next_url:
                k = _key()
                resp = requests.get(f"{next_url}&apiKey={k}", timeout=20)
                resp.raise_for_status()
                data = resp.json()
            else:
                data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})

            all_bars.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break

        if not all_bars:
            log.warning("Polygon: no bars returned for %s", ticker)
            return None

        df = pd.DataFrame(all_bars)
        df["Date"] = (
            pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        )
        df = (
            df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                "c": "Close", "v": "Volume"})
            .set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            .sort_index()
        )
        df.index.name = "Date"
        log.info("Polygon: fetched %d daily bars for %s", len(df), ticker)
        return df

    except Exception as exc:
        log.warning("Polygon daily bars %s: %s", ticker, exc)
        return None


def fetch_prev_close(ticker: str) -> Optional[dict]:
    """Previous trading day OHLCV for a ticker.

    Returns dict {date, open, high, low, close, volume} or None.
    """
    try:
        data = _get(f"/v2/aggs/ticker/{ticker.upper()}/prev",
                    {"adjusted": "true"})
        results = data.get("results") or []
        if not results:
            return None
        r = results[0]
        return {
            "date":   pd.to_datetime(r["t"], unit="ms").date(),
            "open":   r.get("o"),
            "high":   r.get("h"),
            "low":    r.get("l"),
            "close":  r.get("c"),
            "volume": r.get("v"),
        }
    except Exception as exc:
        log.warning("Polygon prev_close %s: %s", ticker, exc)
        return None


def fetch_and_cache_daily(
    ticker: str,
    data_dir: Path,
    days: int = 365 * 7,
    max_age_hours: float = 1.0,
) -> Optional[pd.DataFrame]:
    """Fetch daily bars from Polygon and cache them to CSV.

    If a fresh enough cache file exists it is returned without hitting the API.
    Falls back to the stale cache if the API call fails.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / f"polygon_{ticker.upper()}_daily.csv"

    # Load existing cache
    cached_df: Optional[pd.DataFrame] = None
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached_df.index.name = "Date"
            age_hours = (
                pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            ).total_seconds() / 3600
            if age_hours < max_age_hours:
                return cached_df          # fresh enough
        except Exception:
            cached_df = None

    # Fetch from API
    df = fetch_daily_bars(ticker, days=days)
    if df is not None and not df.empty:
        try:
            df.to_csv(cache_path)
        except Exception:
            pass
        return df

    # API failed — return stale cache if available
    if cached_df is not None:
        log.warning("Polygon: using stale cache for %s", ticker)
        return cached_df

    return None
