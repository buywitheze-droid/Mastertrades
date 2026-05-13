"""High-edge MA-touch setups discovered via universe scan (50 tickers x 4 MAs).

Each setup was validated on 1 year of daily data: ≥5 touches AND ≥75% of touches
produced positive 5-day forward returns. Sorted by win rate, then average return.

Source: scripts/scan_universe.py (one-off scan, results frozen here).
Recompute by re-running the scan and updating HIGH_EDGE_SETUPS below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

SCAN_DATE = "2026-05-13"   # When HIGH_EDGE_SETUPS was last refreshed
SCAN_LOOKBACK_DAYS = 365
import numpy as np

from src.scanner import fetch_or_load_daily
from src.weekly_levels import compute_weekly_mas, find_touches


# Frozen scan results: 22 high-edge setups (n>=5, %pos>=75%, lookback=365d)
# (Originally 24; HYG_30w EMA and XLV_50w SMA dropped 2026-05-13 after real-options
# backtest showed 30%/29% win rates — historical edge did not replicate.)
HIGH_EDGE_SETUPS = [
    # ticker, ma_label, n_touches, pct_pos_5d, avg_5d, med_5d, pct_pos_10d, avg_10d, best_5d, worst_5d
    ("AVGO",  "50w EMA",  5, 100.0, 12.21, 13.29, 100.0, 24.38, 18.1,   4.6),
    ("GOOGL", "50w EMA",  9, 100.0,  4.42,  5.11,  78.0,  3.19,  7.5,   1.4),
    ("GOOGL", "30w EMA", 11, 100.0,  4.18,  4.45,  82.0,  4.93, 10.3,   0.3),
    ("XLI",   "30w SMA",  5, 100.0,  3.77,  3.96, 100.0,  6.56,  5.4,   1.2),
    ("XLI",   "30w EMA",  5, 100.0,  3.77,  3.96, 100.0,  6.56,  5.4,   1.2),
    ("SHOP",  "30w SMA", 10,  90.0,  4.39,  4.72,  90.0,  2.69, 10.2,  -4.8),
    ("XOM",   "30w EMA", 18,  89.0,  2.01,  2.14,  67.0,  1.12,  4.8,  -2.6),
    ("XOM",   "30w SMA", 15,  87.0,  2.23,  2.44,  60.0,  1.33,  4.5,  -2.6),
    ("SOXX",  "50w EMA",  7,  86.0,  3.38,  2.07, 100.0,  5.95,  6.9,  -0.4),
    ("DIA",   "50w SMA",  7,  86.0,  2.17,  3.06, 100.0,  4.33,  3.5,  -1.0),
    ("AMZN",  "30w SMA", 20,  85.0,  4.17,  5.04,  90.0,  4.52, 12.3,  -2.2),
    ("IWM",   "50w EMA", 13,  85.0,  1.46,  1.90,  62.0,  2.63,  5.6,  -3.5),
    ("AMZN",  "50w SMA",  6,  83.0,  2.29,  2.85,  83.0,  4.91,  6.0,  -4.3),
    # HYG 30w EMA: REMOVED 2026-05-13. Historical 83% win rate did NOT replicate.
    # Real Polygon options backtest (Apr-May 2026): 20 trades, 30% win, -37% avg.
    # The MA went sideways and price drifted along it instead of bouncing off.
    ("MA",    "30w SMA", 21,  81.0,  1.01,  1.15,  57.0, -0.37,  4.3,  -6.1),
    ("INTC",  "30w SMA", 10,  80.0,  5.02,  5.21,  90.0,  7.01, 16.0,  -5.8),
    ("LLY",   "50w SMA",  5,  80.0,  4.50,  2.82, 100.0,  7.16, 11.5,  -3.2),
    ("AMZN",  "30w EMA", 15,  80.0,  3.49,  2.91,  93.0,  5.44, 12.3,  -1.6),
    ("ORCL",  "30w SMA",  5,  80.0,  3.00,  5.06, 100.0, 18.07,  6.4,  -4.1),
    ("ORCL",  "50w SMA", 10,  80.0,  1.55,  3.06,  40.0, -2.22, 10.2, -14.0),
    # XLV 50w SMA: REMOVED 2026-05-13. Historical 80% win rate did NOT replicate.
    # Real Polygon options backtest (Apr-May 2026): 7 trades, 29% win, -39% avg.
    # 3 of the 5 worst losses across the entire universe came from this single setup.
    ("XLE",   "30w EMA", 31,  77.0,  1.33,  1.11,  71.0,  1.61,  6.1,  -1.8),
    ("XOM",   "50w EMA", 13,  77.0,  1.05,  1.36,  54.0,  0.20,  4.8,  -5.1),
    ("ABNB",  "50w SMA", 32,  75.0,  1.85,  2.57,  66.0,  1.80, 10.2,  -9.7),
]


@dataclass
class LiveSetupStatus:
    ticker: str
    ma_label: str
    n_touches: int
    pct_pos_5d: float
    avg_5d: float
    avg_10d: float
    best_5d: float
    worst_5d: float
    last_close: float
    ma_value: float
    distance_pct: float        # (price-ma)/ma*100. Positive = above MA.
    above_ma: bool
    drift_5d: float            # 5-day return to detect "approaching from above"
    state: str                 # APPROACHING | TOUCHING | EXTENDED | BELOW
    state_color: str
    edge_score: float          # composite ranking score


def _classify(distance_pct: float, above_ma: bool, drift_5d: float) -> tuple[str, str]:
    """Classify a setup based on price-to-MA distance + recent drift."""
    if not above_ma:
        return ("BELOW", "#f85149")               # red — MA broke
    if abs(distance_pct) <= 0.6:
        return ("TOUCHING", "#ffd633")            # yellow — at the level NOW
    if distance_pct <= 2.5 and drift_5d < 0:
        return ("APPROACHING", "#3fb950")         # green — falling toward support
    if distance_pct <= 2.5:
        return ("APPROACHING", "#79c0ff")         # blue — close, but not yet falling
    return ("EXTENDED", "#8b949e")                # gray — far from level


def get_live_setup_status(ticker: str, ma_label: str,
                          n: int, pct_pos: float, avg_5d: float, avg_10d: float,
                          best_5d: float, worst_5d: float) -> Optional[LiveSetupStatus]:
    """Compute the live state of a single high-edge setup."""
    try:
        d = fetch_or_load_daily(ticker)
        if d is None or len(d) < 250:
            return None
        d = d.sort_index()
        d.index = pd.to_datetime(d.index)
        mas = compute_weekly_mas(d)
        if ma_label not in mas:
            return None
        ma_series = mas[ma_label].dropna()
        if len(ma_series) < 1:
            return None
        ma_val = float(ma_series.iloc[-1])
        last_close = float(d["Close"].iloc[-1])
        distance_pct = (last_close - ma_val) / ma_val * 100
        above_ma = last_close >= ma_val
        if len(d) >= 6:
            drift_5d = (last_close - float(d["Close"].iloc[-6])) / float(d["Close"].iloc[-6]) * 100
        else:
            drift_5d = 0.0
        state, color = _classify(distance_pct, above_ma, drift_5d)
        # Composite edge score: pct_pos * avg_5d, boosted if currently TOUCHING/APPROACHING
        edge = pct_pos * max(avg_5d, 0.1) / 100
        if state == "TOUCHING": edge *= 2.5
        elif state == "APPROACHING" and drift_5d < 0: edge *= 1.6
        elif state == "APPROACHING": edge *= 1.2
        elif state == "BELOW": edge *= 0.3
        return LiveSetupStatus(
            ticker=ticker, ma_label=ma_label, n_touches=n,
            pct_pos_5d=pct_pos, avg_5d=avg_5d, avg_10d=avg_10d,
            best_5d=best_5d, worst_5d=worst_5d,
            last_close=last_close, ma_value=ma_val,
            distance_pct=distance_pct, above_ma=above_ma,
            drift_5d=drift_5d, state=state, state_color=color,
            edge_score=edge,
        )
    except Exception:
        return None


def get_all_live_setups() -> tuple[list[LiveSetupStatus], list[tuple[str, str]]]:
    """Compute live status for all HIGH_EDGE_SETUPS.

    Returns (loaded_setups, failed_setups) where failed_setups is a list of
    (ticker, ma_label) tuples that could not be priced (data fetch error).
    """
    out: list[LiveSetupStatus] = []
    failed: list[tuple[str, str]] = []
    for row in HIGH_EDGE_SETUPS:
        ticker, ma_label, n, pct_pos, avg_5d, _med, _pct10, avg_10d, best_5d, worst_5d = row
        s = get_live_setup_status(ticker, ma_label, n, pct_pos, avg_5d, avg_10d, best_5d, worst_5d)
        if s is not None:
            out.append(s)
        else:
            failed.append((ticker, ma_label))
    state_order = {"TOUCHING": 0, "APPROACHING": 1, "EXTENDED": 2, "BELOW": 3}
    out.sort(key=lambda s: (state_order.get(s.state, 99), -s.edge_score))
    return out, failed
