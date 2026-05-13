"""Historical Polygon options helpers.

Polygon paid plan provides:
  ✓ /v3/reference/options/contracts (with expired=true) — list contracts that existed
  ✓ /v2/aggs/ticker/O:SYMBOL/range/1/day/{from}/{to}    — daily OHLCV per contract
  ✗ /v3/trades, /v3/quotes (tick tape — not entitled on this plan)

Used by `scripts/backtest_4weeks.py` to replay algo-recommended trades against
real historical option prices.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests

POLYGON_BASE = "https://api.polygon.io"
_MIN_GAP_S = 0.12  # ~8 req/s — well under 100/min limit on paid tier
_LAST_CALL = 0.0


def _key() -> str:
    k = os.environ.get("POLYGON_API_KEY")
    if not k:
        raise RuntimeError("POLYGON_API_KEY not configured")
    return k


def _get(path: str, params: dict | None = None) -> dict:
    global _LAST_CALL
    elapsed = time.monotonic() - _LAST_CALL
    if elapsed < _MIN_GAP_S:
        time.sleep(_MIN_GAP_S - elapsed)
    p = dict(params or {})
    p["apiKey"] = _key()
    _LAST_CALL = time.monotonic()
    resp = requests.get(f"{POLYGON_BASE}{path}", params=p, timeout=20)
    resp.raise_for_status()
    return resp.json()


@dataclass
class ExpiredContract:
    ticker:        str       # e.g. "O:SPY260515C00580000"
    underlying:    str
    contract_type: str       # "call" / "put"
    strike:        float
    expiration:    str       # "YYYY-MM-DD"


def list_expired_contracts(
    underlying:        str,
    expiration_gte:    str,
    expiration_lte:    str,
    contract_type:     str = "call",
    strike_lo:         Optional[float] = None,
    strike_hi:         Optional[float] = None,
    limit:             int = 1000,
) -> list[ExpiredContract]:
    """List option contracts (expired or live) that existed in a date window."""
    params: dict = {
        "underlying_ticker":   underlying,
        "contract_type":       contract_type,
        "expiration_date.gte": expiration_gte,
        "expiration_date.lte": expiration_lte,
        "expired":             "true",
        "limit":               limit,
    }
    if strike_lo is not None:
        params["strike_price.gte"] = strike_lo
    if strike_hi is not None:
        params["strike_price.lte"] = strike_hi

    out: list[ExpiredContract] = []
    data = _get("/v3/reference/options/contracts", params)
    for item in data.get("results", []) or []:
        out.append(ExpiredContract(
            ticker        = item.get("ticker", ""),
            underlying    = item.get("underlying_ticker", underlying),
            contract_type = item.get("contract_type", contract_type),
            strike        = float(item.get("strike_price", 0) or 0),
            expiration    = item.get("expiration_date", ""),
        ))
    return out


@dataclass
class OptionBar:
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int
    vwap:   float


def fetch_option_daily_bars(
    option_ticker: str,
    from_date:     str,
    to_date:       str,
) -> list[OptionBar]:
    """Daily OHLCV bars for one option contract."""
    path = f"/v2/aggs/ticker/{option_ticker}/range/1/day/{from_date}/{to_date}"
    data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 5000})
    bars: list[OptionBar] = []
    for r in data.get("results", []) or []:
        ts_ms = int(r.get("t", 0))
        d = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
        bars.append(OptionBar(
            date   = d,
            open   = float(r.get("o", 0) or 0),
            high   = float(r.get("h", 0) or 0),
            low    = float(r.get("l", 0) or 0),
            close  = float(r.get("c", 0) or 0),
            volume = int(r.get("v", 0) or 0),
            vwap   = float(r.get("vw", 0) or 0),
        ))
    return bars


def find_atm_strike(price: float, available_strikes: list[float]) -> Optional[float]:
    """Closest available strike to the underlying price."""
    if not available_strikes:
        return None
    return min(available_strikes, key=lambda s: abs(s - price))


def next_friday(d: date, min_days_out: int = 3) -> date:
    """Next Friday at least `min_days_out` calendar days from `d`."""
    days_ahead = (4 - d.weekday()) % 7   # 4 = Friday
    if days_ahead < min_days_out:
        days_ahead += 7
    return d + timedelta(days=days_ahead)
