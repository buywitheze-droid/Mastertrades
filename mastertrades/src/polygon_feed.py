"""Polygon.io data feed for Mastertrades.

Confirmed working endpoints (upgraded plan)
--------------------------------------------
✅  /v2/aggs/ticker/{T}/range/1/day/{from}/{to}  — full historical OHLCV
✅  /v2/aggs/ticker/{T}/prev                      — previous trading day OHLCV
✅  /v2/snapshot/locale/us/markets/stocks/tickers — multi-ticker live snapshot
    Fields available: day (OHLCV, VWAP), prevDay, todaysChangePerc
    Note: lastTrade / lastQuote may be None — day.c used as current price

Integration strategy
--------------------
• Historical OHLCV  → fetch_daily_bars()  (replaces yfinance, used for ML + gap)
• Live quotes       → fetch_multi_snapshot()  (ticker cards, scanner, gap progress)
• Graceful fallback: every function returns None / {} on failure
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
_LAST_CALL   = 0.0
_MIN_GAP_S   = 0.12          # ~8 req/s; snapshot calls are batched so this is fine


# ── Internals ─────────────────────────────────────────────────────────────────

def _key() -> Optional[str]:
    return os.environ.get("POLYGON_API_KEY") or None


def _get(path: str, params: dict | None = None) -> dict:
    global _LAST_CALL
    key = _key()
    if not key:
        raise RuntimeError("POLYGON_API_KEY not configured")
    elapsed = time.monotonic() - _LAST_CALL
    if elapsed < _MIN_GAP_S:
        time.sleep(_MIN_GAP_S - elapsed)
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


def _parse_snap_item(snap: dict, api_status: str) -> dict:
    """Convert a raw Polygon snapshot ticker dict into a normalised quote dict."""
    day        = snap.get("day")      or {}
    prev       = snap.get("prevDay")  or {}
    min_bar    = snap.get("min")      or {}
    last_trade = snap.get("lastTrade") or {}   # None on some plan tiers — guard with {}
    last_quote = snap.get("lastQuote") or {}

    # Best available current price:
    #   real-time trade → latest minute bar → today's daily close
    last_price: float = float(
        last_trade.get("p")
        or min_bar.get("c")
        or last_quote.get("P")
        or day.get("c")
        or 0.0
    )
    prev_close: float = float(prev.get("c") or 0.0)

    raw_chg = snap.get("todaysChangePerc")
    if raw_chg is not None:
        change_pct = float(raw_chg) / 100.0
    elif prev_close:
        change_pct = (last_price - prev_close) / prev_close
    else:
        change_pct = 0.0

    is_delayed   = api_status == "DELAYED"
    status_label = "DELAYED" if is_delayed else "LIVE"

    return {
        "ticker":       snap.get("ticker", ""),
        "last_price":   last_price,
        "prev_close":   prev_close,
        "change_pct":   change_pct,
        "change_pts":   last_price - prev_close,
        "day_open":     float(day.get("o") or 0.0),
        "day_high":     float(day.get("h") or 0.0),
        "day_low":      float(day.get("l") or 0.0),
        "day_close":    float(day.get("c") or 0.0),
        "day_volume":   int(day.get("v") or 0),
        "day_vwap":     float(day.get("vw") or 0.0),
        "prev_open":    float(prev.get("o") or 0.0),
        "prev_high":    float(prev.get("h") or 0.0),
        "prev_low":     float(prev.get("l") or 0.0),
        "prev_vwap":    float(prev.get("vw") or 0.0),
        "is_delayed":   is_delayed,
        "status_label": status_label,
    }


# ── Public helpers ────────────────────────────────────────────────────────────

def has_polygon_key() -> bool:
    return bool(_key())


def fetch_snapshot(ticker: str) -> Optional[dict]:
    """Live/delayed snapshot for a single ticker. Returns None on failure."""
    try:
        data = _get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
        )
        return _parse_snap_item(data.get("ticker") or {}, data.get("status", ""))
    except Exception as exc:
        log.warning("Polygon snapshot %s: %s", ticker, exc)
        return None


def fetch_multi_snapshot(tickers: list[str]) -> dict[str, dict]:
    """Batch live snapshots for multiple tickers.

    Returns {TICKER: quote_dict}. Missing / failed tickers are silently omitted.
    """
    if not tickers:
        return {}
    try:
        joined = ",".join(t.upper() for t in tickers)
        data = _get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": joined},
        )
        api_status = data.get("status", "")
        return {
            item["ticker"]: _parse_snap_item(item, api_status)
            for item in (data.get("tickers") or [])
            if item.get("ticker")
        }
    except Exception as exc:
        log.warning("Polygon multi-snapshot: %s", exc)
        return {}


def fetch_daily_bars(ticker: str, days: int = 365 * 7) -> Optional[pd.DataFrame]:
    """Historical daily OHLCV from Polygon (split/dividend adjusted).

    Returns DataFrame(index=DatetimeIndex, columns=[Open,High,Low,Close,Volume])
    matching yfinance schema. Returns None on failure.
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
                resp = requests.get(
                    f"{next_url}&apiKey={_key()}", timeout=20
                )
                resp.raise_for_status()
                data = resp.json()
            else:
                data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})

            all_bars.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break

        if not all_bars:
            log.warning("Polygon: no bars for %s", ticker)
            return None

        df = pd.DataFrame(all_bars)
        df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        df = (
            df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                "c": "Close", "v": "Volume"})
            .set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            .sort_index()
        )
        df.index.name = "Date"
        log.info("Polygon: %d bars for %s", len(df), ticker)
        return df

    except Exception as exc:
        log.warning("Polygon daily bars %s: %s", ticker, exc)
        return None


def fetch_prev_close(ticker: str) -> Optional[dict]:
    """Previous trading day OHLCV. Returns {date,open,high,low,close,volume} or None."""
    try:
        data = _get(f"/v2/aggs/ticker/{ticker.upper()}/prev", {"adjusted": "true"})
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
    """Fetch + cache daily bars. Returns stale cache on API failure."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / f"polygon_{ticker.upper()}_daily.csv"

    cached_df: Optional[pd.DataFrame] = None
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached_df.index.name = "Date"
            age_h = (
                pd.Timestamp.now()
                - pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            ).total_seconds() / 3600
            if age_h < max_age_hours:
                return cached_df
        except Exception:
            cached_df = None

    df = fetch_daily_bars(ticker, days=days)
    if df is not None and not df.empty:
        try:
            df.to_csv(cache_path)
        except Exception:
            pass
        return df

    if cached_df is not None:
        log.warning("Polygon: using stale cache for %s", ticker)
        return cached_df
    return None
