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


# ── Single contract live snapshot (for trail-stop tracking) ──────────────────

def fetch_option_quote(
    underlying: str,
    contract_ticker: str,
    timeout: int = 10,
) -> Optional[dict]:
    """Fetch a live snapshot for ONE option contract.

    Returns a dict with the fields needed by the live trail-stop tracker:
        {
          "last_price":   float — last trade price (or mid quote fallback),
          "day_open":     float,
          "day_high":     float,   # running high since today's open
          "day_low":      float,   # running low since today's open
          "day_close":    float,   # last trade close (== last_price intraday)
          "bid":          float,
          "ask":          float,
          "fetched_at":   datetime ISO string,
        }
    Returns None if the snapshot is unavailable (no key, http error, no
    trades yet, etc.). Callers must handle None gracefully.

    Endpoint: GET /v3/snapshot/options/{underlying}/{contract_ticker}
    """
    if not _key():
        return None
    url = f"{POLYGON_BASE}/v3/snapshot/options/{underlying.upper()}/{contract_ticker}"
    try:
        resp = requests.get(url, params={"apiKey": _key()}, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None
    item = (resp.json() or {}).get("results") or {}
    day = item.get("day") or {}
    quote = item.get("last_quote") or {}
    trade = item.get("last_trade") or {}
    bid = float(quote.get("bid") or 0.0)
    ask = float(quote.get("ask") or 0.0)
    last_trade_price = float(trade.get("price") or 0.0)
    day_close = float(day.get("close") or 0.0)
    # Prefer last trade; fall back to mid-quote; fall back to day_close.
    if last_trade_price > 0:
        last_price = last_trade_price
    elif bid > 0 and ask > 0:
        last_price = (bid + ask) / 2.0
    else:
        last_price = day_close
    from datetime import datetime as _dt
    return {
        "last_price": last_price,
        "day_open":   float(day.get("open")  or 0.0),
        "day_high":   float(day.get("high")  or 0.0),
        "day_low":    float(day.get("low")   or 0.0),
        "day_close":  day_close,
        "bid":        bid,
        "ask":        ask,
        "fetched_at": _dt.now().isoformat(timespec="seconds"),
    }


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
    est_entry_price: float      # algo's entry estimate (= max(day_low, 0.01)).
                                # Used for ranking math. Can be $0.01 if option
                                # touched the price floor intraday — DO NOT
                                # display this as the user-facing entry price;
                                # use display_entry_price instead.
    est_target_price: float     # if underlying returns to open
    est_gain_pct: float         # (target - est_entry) / est_entry * 100.
                                # Reflects upside FROM the day's low. Used for
                                # ranking (matches backtest assumption).
    risk_category: str          # "LOTTERY", "AGGRESSIVE", "MODERATE"
    note: str
    contract_ticker: str = ""   # Polygon option ticker, e.g. "O:SPY260513C00744000"
                                # Required for live-quote follow-up (trail stops).
    leverage_score: float = 0.0 # est_gain_pct × √(1/entry). Empirically picks
                                # 3× more profitable strikes than est_gain_pct
                                # alone — see scripts/_strike_strategy_compare.py.

    # ── Display fields (use these in the UI, NOT est_entry_price) ───────────
    display_entry_price: float = 0.0   # Realistic current-fill estimate.
                                       # = max(day_close, day_open, day_low, 0.01).
                                       # Use for: action line, sizing, sell
                                       # trigger, max-risk display.
    display_gain_pct: float = 0.0      # Upside from display_entry_price, NOT
                                       # from intraday low. Honest "what you'll
                                       # actually capture" estimate.


# ── Strike-selection knobs (validated 2026-05-13) ────────────────────────────
# Backtest: scripts/_strike_strategy_compare.py against 90-day Polygon data
# proved that a $1 max-premium cap + leverage_score ranking earns
# +$24,480 vs +$11,402 for the previous "max est_gain_pct alone" picker
# (115% improvement on $500/trade, 85% win rate vs 72%, profit factor 43×).
# Bump this only after re-running the backtest.

MAX_PREMIUM_USD = 1.00          # skip strikes priced above this at alert time
STRIKE_SELECTION_VERSION = "2026-05-13-cap1-leverage-bonus"


def recommend_strikes(
    underlying_open:   float,
    underlying_low:    float,
    contracts:         list[OptionContract],
    recovery_target:   Optional[float] = None,
    max_premium_usd:   Optional[float] = None,
    apply_leverage_bonus: bool = True,
) -> list[StrikeRecommendation]:
    """Given the current intraday low, recommend which call strikes to buy.

    recovery_target: expected recovery level (default = open price).
    max_premium_usd: per-share premium cap. Strikes priced above this at
                    alert time are SKIPPED. Defaults to MAX_PREMIUM_USD ($1.00),
                    validated 2026-05-13 against 90-day Polygon backtest.
    apply_leverage_bonus: when True, populates `leverage_score` with
                    est_gain_pct × √(1/entry). The dashboard ranks by this
                    instead of est_gain_pct alone — empirically yields 3×
                    per-trade P&L by biasing toward cheap, leveraged OTM.

    Looks at actual live contract prices from the chain.
    """
    if recovery_target is None:
        recovery_target = underlying_open
    if max_premium_usd is None:
        max_premium_usd = MAX_PREMIUM_USD

    drop_pts = underlying_open - underlying_low  # positive number

    recs: list[StrikeRecommendation] = []
    call_map = {c.strike: c for c in contracts if c.contract_type == "call"}

    # Sweet spot: strikes +1 to +9 pts above the low
    for offset in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        target_strike = round(underlying_low + offset)
        c = call_map.get(target_strike)
        if c is None:
            continue
        entry  = max(c.day_low, 0.01)  # current price at low (for est_gain_pct)
        # Premium-cap filter: filters on the **actual fill price** the user
        # will pay (≈ day_open for an alert that fires shortly after open),
        # NOT on day_low. day_low can spike to $0.01 intraday on any chain,
        # which would let pricey strikes leak through. day_open matches the
        # backtest assumption in scripts/_strike_strategy_compare.py and is
        # what produced the validated +$401/trade · 85% win-rate result.
        # Falls back to day_low only if day_open is missing (defensive).
        cap_ref = c.day_open if c.day_open and c.day_open > 0 else entry
        if cap_ref > max_premium_usd:
            continue
        dist_open = target_strike - underlying_open

        # Estimate target price using intrinsic + small premium
        if recovery_target >= target_strike:
            intrinsic = recovery_target - target_strike
            target_price = intrinsic + 0.05  # small remaining theta
        else:
            # Still OTM at recovery target — use delta estimate
            target_price = max(c.delta * (recovery_target - target_strike + 1.0), 0.05)

        gain_pct = (target_price - entry) / entry * 100

        # Leverage-weighted ranking score. Square-root prevents the cheapest
        # strike from always winning regardless of est_gain_pct (a $0.01 entry
        # with even tiny target would otherwise dominate). With sqrt, a $0.10
        # cost basis with 200% est_gain beats a $1.00 cost basis with 50%
        # est_gain by roughly 4× — matches what the backtest shows.
        # Uses cap_ref (≈ actual fill price), NOT entry (which is opt_low and
        # can spike to $0.01 intraday). cap_ref is the cost-basis reference
        # consistent with the cap filter and with backtest sizing.
        if apply_leverage_bonus and cap_ref > 0:
            leverage_score = gain_pct * (1.0 / cap_ref) ** 0.5
        else:
            leverage_score = gain_pct

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

        # Realistic current-price estimate for DISPLAY (not for ranking).
        # day_close in a Polygon snapshot is the close of the most recent
        # 1-min bar ≈ the option's current price. Falls back gracefully if
        # day_close is missing or zero. Always clamped to the $0.01 floor
        # because the OCC clears option contracts at penny tick increments
        # and exchanges will not show a print below $0.01.
        display_entry = max(c.day_close, c.day_open, c.day_low, 0.01)
        # Honest gain-pct from where you'd actually fill, not from intraday low.
        if display_entry > 0:
            display_gain = (target_price - display_entry) / display_entry * 100
        else:
            display_gain = 0.0

        recs.append(StrikeRecommendation(
            strike=target_strike,
            dist_from_low=float(offset),
            dist_from_open=dist_open,
            est_entry_price=entry,
            est_target_price=round(target_price, 2),
            est_gain_pct=gain_pct,
            risk_category=risk_cat,
            note=cost_note,
            contract_ticker=c.ticker,
            leverage_score=leverage_score,
            display_entry_price=display_entry,
            display_gain_pct=display_gain,
        ))

    # Sort by leverage_score (the new ranking) so the top entry is what the
    # dashboard would surface. Callers that still ranks by est_gain_pct will
    # see the same recs, just in a different order — no breakage.
    recs.sort(key=lambda x: -x.leverage_score)
    return recs


# ── Historical multiplier estimates ──────────────────────────────────────────

# Date the realised numbers below were last validated against Polygon.
# Re-run scripts/backtest_per_source_gate.py periodically and bump this if
# the regime has shifted (e.g. quarterly).
DROP_BAND_TABLE_SCAN_DATE = "2026-05-13"


def drop_band_multiplier_table() -> list[dict]:
    """Drop-to-recovery probabilities used by the live 0DTE Drop alert.

    The 3-5 / 5-7 / 7-10 / 10+ bands were re-calibrated 2026-05-13 from 110
    real Polygon 0DTE option trades on SPY/QQQ/IWM over the trailing 90
    sessions (scripts/backtest_per_source_gate.py). The original numbers
    were "pre-computed from BSM model" at an unknown earlier date and
    understated the true win rate by a factor of 2-7×, which combined with
    the unified-edge gate in app.py ("Today's Plays") to silence the entire
    source. See the analysis chat dated 2026-05-13 for the validation.

    The 0-1 / 1-2 / 2-3 bands were NOT recalibrated — they're below the
    ENTRY_OPEN trigger of 3.0 pts so they never reach the live alert path.

    Columns:
      band              — drop range in points (open - low)
      n                 — sample size in the calibration window
      pct_1000plus      — % of sessions where the algo's recommended
                          0DTE strike produced ≥1000% intraday gain
                          (entry near opt low, exit near opt high)
      recovery_needed   — typical underlying recovery required for a 1000% move
      note              — UI hint (kept short for the dashboard banner)
    """
    return [
        # 0-1 / 1-2 / 2-3 — below trigger, original BSM-era values retained
        {"band": "0–1 pts",    "n": 136, "pct_1000plus":  2, "recovery_needed":  2.0, "note": "No setup"},
        {"band": "1–2 pts",    "n":  84, "pct_1000plus":  5, "recovery_needed":  3.5, "note": "No setup"},
        {"band": "2–3 pts",    "n":  69, "pct_1000plus": 12, "recovery_needed":  5.0, "note": "Marginal"},
        # 3+ bands — recalibrated 2026-05-13 from real Polygon 90-day data
        {"band": "3–5 pts",    "n":  48, "pct_1000plus": 54, "recovery_needed":  6.0, "note": "TRADE 🎯"},
        {"band": "5–7 pts",    "n":  28, "pct_1000plus": 61, "recovery_needed":  7.2, "note": "TRADE 🎯"},
        {"band": "7–10 pts",   "n":  21, "pct_1000plus": 62, "recovery_needed":  9.6, "note": "TRADE 🎯"},
        {"band": "10+ pts",    "n":  13, "pct_1000plus": 69, "recovery_needed": 12.9, "note": "TRADE 🎯 (high variance)"},
    ]
