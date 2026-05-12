"""0DTE Options Scanner — Polygon.io live options chain analysis.

Fetches the full 0DTE options chain for a ticker, identifies contracts
that have already moved 1000%+ (or are set up for it), and computes
the "sweet spot" strikes based on intraday price action.

Pattern:
  1. Underlying drops X pts from open to intraday low
  2. Calls at strikes near the OPEN price (OTM by X at the low) get very cheap
  3. When price reverses back through the open, those calls explode
  4. Sweet spot: strikes +1 to +8 pts above the intraday low
     = was ATM at open, now cheap OTM → if reversal happens → 1000%+
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests


POLYGON_BASE = "https://api.polygon.io"


def _key() -> str:
    return os.environ.get("POLYGON_API_KEY", "")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    ticker: str
    contract_type: str          # "call" or "put"
    strike: float
    expiration: str             # "YYYY-MM-DD"
    day_open:  float
    day_high:  float
    day_low:   float
    day_close: float
    day_volume: int
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega:  float
    open_interest: int


@dataclass
class ContractAnalysis:
    contract: OptionContract
    day_gain_pct: float         # (high - low) / low * 100
    dist_from_underlying_low: float   # strike - underlying_low
    dist_from_underlying_open: float  # strike - underlying_open
    is_1000_plus: bool
    is_sweet_spot: bool         # dist from low between 0 and 9 pts
    category: str               # "JACKPOT", "WATCH", "DEEP_ITM", "FAR_OTM"


# ── Fetch chain ───────────────────────────────────────────────────────────────

def fetch_0dte_chain(
    ticker: str,
    exp_date: Optional[str] = None,
    contract_type: Optional[str] = None,
    limit: int = 250,
) -> list[OptionContract]:
    """Fetch live 0DTE options chain snapshot from Polygon.

    exp_date: "YYYY-MM-DD", defaults to today.
    contract_type: "call", "put", or None (both).
    """
    if exp_date is None:
        exp_date = date.today().isoformat()

    params: dict = {
        "expiration_date": exp_date,
        "limit": limit,
        "apiKey": _key(),
    }
    if contract_type:
        params["contract_type"] = contract_type

    url = f"{POLYGON_BASE}/v3/snapshot/options/{ticker}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    contracts: list[OptionContract] = []
    for item in resp.json().get("results", []):
        day = item.get("day", {})
        det = item.get("details", {})
        gr  = item.get("greeks", {})
        contracts.append(OptionContract(
            ticker        = det.get("ticker", ""),
            contract_type = det.get("contract_type", ""),
            strike        = float(det.get("strike_price", 0)),
            expiration    = det.get("expiration_date", exp_date),
            day_open      = float(day.get("open",   0) or 0),
            day_high      = float(day.get("high",   0) or 0),
            day_low       = float(day.get("low",    0) or 0),
            day_close     = float(day.get("close",  0) or 0),
            day_volume    = int(day.get("volume",   0) or 0),
            implied_vol   = float(item.get("implied_volatility", 0) or 0),
            delta         = float(gr.get("delta", 0) or 0),
            gamma         = float(gr.get("gamma", 0) or 0),
            theta         = float(gr.get("theta", 0) or 0),
            vega          = float(gr.get("vega",  0) or 0),
            open_interest = int(item.get("open_interest", 0) or 0),
        ))

    return contracts


# ── Analyse chain ─────────────────────────────────────────────────────────────

def analyze_chain(
    contracts: list[OptionContract],
    underlying_open:  float,
    underlying_low:   float,
    underlying_high:  float,
    underlying_close: float,
    strike_window:    float = 20.0,
) -> list[ContractAnalysis]:
    """Compute gain% and categorise every contract in the chain."""
    results: list[ContractAnalysis] = []
    for c in contracts:
        if c.day_low <= 0 or c.day_high <= 0:
            continue
        # Filter to relevant strike range
        if abs(c.strike - underlying_open) > strike_window:
            continue

        gain_pct = (c.day_high - c.day_low) / c.day_low * 100
        dist_low  = c.strike - underlying_low
        dist_open = c.strike - underlying_open

        is_1000  = gain_pct >= 1000
        # Sweet spot for calls: strike is 0-9 pts above the intraday low
        # Sweet spot for puts:  strike is 0-9 pts below the intraday high
        if c.contract_type == "call":
            sweet = 0 <= dist_low <= 9.0
        else:
            sweet = -9.0 <= dist_low <= 0

        if gain_pct >= 1000:
            cat = "JACKPOT 🌟"
        elif gain_pct >= 500:
            cat = "FIRE 🔥"
        elif gain_pct >= 100:
            cat = "HOT 📈"
        else:
            cat = "BASE"

        results.append(ContractAnalysis(
            contract=c,
            day_gain_pct=gain_pct,
            dist_from_underlying_low=dist_low,
            dist_from_underlying_open=dist_open,
            is_1000_plus=is_1000,
            is_sweet_spot=sweet,
            category=cat,
        ))

    results.sort(key=lambda x: -x.day_gain_pct)
    return results


# ── Strike recommendation ─────────────────────────────────────────────────────

@dataclass
class StrikeRecommendation:
    strike: float
    dist_from_low: float        # pts above low
    dist_from_open: float       # pts above open (negative = ITM at open)
    est_entry_price: float      # estimated call price at current low
    est_target_price: float     # if underlying returns to open
    est_gain_pct: float         # (target - entry) / entry * 100
    risk_category: str          # "LOTTERY", "AGGRESSIVE", "MODERATE"
    note: str


def recommend_strikes(
    underlying_open:   float,
    underlying_low:    float,
    contracts:         list[OptionContract],
    recovery_target:   Optional[float] = None,
) -> list[StrikeRecommendation]:
    """Given the current intraday low, recommend which call strikes to buy.

    recovery_target: expected recovery level (default = open price).
    Looks at actual live contract prices from the chain.
    """
    if recovery_target is None:
        recovery_target = underlying_open

    drop_pts = underlying_open - underlying_low  # positive number

    recs: list[StrikeRecommendation] = []
    call_map = {c.strike: c for c in contracts if c.contract_type == "call"}

    # Sweet spot: strikes +1 to +8 pts above the low
    for offset in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        target_strike = round(underlying_low + offset)
        c = call_map.get(target_strike)
        if c is None:
            continue
        entry  = max(c.day_low, 0.01)  # current price at low
        dist_open = target_strike - underlying_open

        # Estimate target price using intrinsic + small premium
        if recovery_target >= target_strike:
            intrinsic = recovery_target - target_strike
            target_price = intrinsic + 0.05  # small remaining theta
        else:
            # Still OTM at recovery target — use delta estimate
            target_price = max(c.delta * (recovery_target - target_strike + 1.0), 0.05)

        gain_pct = (target_price - entry) / entry * 100

        if offset <= 3:
            risk_cat = "MODERATE — deeper ITM, smaller % gain"
        elif offset <= 6:
            risk_cat = "AGGRESSIVE — sweet spot 🎯"
        else:
            risk_cat = "LOTTERY — tiny entry, explosive if recovered"

        cost_note = f"${entry:.2f} entry"
        if entry <= 0.05:
            cost_note += " (penny)"
        elif entry <= 0.15:
            cost_note += " (cheap)"

        recs.append(StrikeRecommendation(
            strike=target_strike,
            dist_from_low=float(offset),
            dist_from_open=dist_open,
            est_entry_price=entry,
            est_target_price=round(target_price, 2),
            est_gain_pct=gain_pct,
            risk_category=risk_cat,
            note=cost_note,
        ))

    recs.sort(key=lambda x: -x.est_gain_pct)
    return recs


# ── Historical multiplier estimates ──────────────────────────────────────────

def drop_band_multiplier_table() -> list[dict]:
    """Hardcoded 2-year SPY historical estimates (pre-computed from BSM model).

    Columns: drop_band, n_sessions, pct_1000plus, avg_recovery_needed_pts,
             median_max_gain_pct.
    """
    return [
        {"band": "0–1 pts",    "n": 136, "pct_1000plus":  2, "recovery_needed":  2.0, "note": "No setup"},
        {"band": "1–2 pts",    "n":  84, "pct_1000plus":  5, "recovery_needed":  3.5, "note": "No setup"},
        {"band": "2–3 pts",    "n":  69, "pct_1000plus": 12, "recovery_needed":  5.0, "note": "Marginal"},
        {"band": "3–5 pts",    "n":  78, "pct_1000plus": 35, "recovery_needed":  6.9, "note": "WATCH 👀"},
        {"band": "5–7 pts",    "n":  40, "pct_1000plus": 28, "recovery_needed":  8.1, "note": "WATCH 👀"},
        {"band": "7–10 pts",   "n":  33, "pct_1000plus":  9, "recovery_needed": 16.6, "note": "Hard recovery"},
        {"band": "10+ pts",    "n":  12, "pct_1000plus":  0, "recovery_needed":   0,  "note": "Skip — gap day"},
    ]
