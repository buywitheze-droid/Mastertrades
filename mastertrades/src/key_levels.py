"""Key Reversal Levels — Pivot points + historical intraday bounce analysis.

Strategy: Identify price levels where SPY (or any ticker) historically reverses
from its intraday extreme back toward the open.  Three converging signals:

  1. Pivot levels  — S3/S2/S1/PP/R1/R2/R3 from previous session H/L/C
  2. Drop band     — when open→low drop matches today's magnitude, how often bounce?
  3. VWAP deviance — when low is X pts below VWAP, bounce probability

Combined, these let us say: "At $732, three signals align — 56-64% bounce rate,
median recovery +3.7 pts, average return-to-open +5.1 pts.  BUY CALLS."
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Pivot computation ──────────────────────────────────────────────────────────

@dataclass
class PivotLevels:
    pp: float
    r1: float; r2: float; r3: float
    s1: float; s2: float; s3: float
    m_r1: float; m_r2: float
    m_s1: float; m_s2: float; m_s3: float


def compute_pivots(prev_high: float, prev_low: float, prev_close: float) -> PivotLevels:
    """Standard floor-trader pivot points from the previous session."""
    pp  = (prev_high + prev_low + prev_close) / 3
    r1  = 2 * pp - prev_low
    r2  = pp + (prev_high - prev_low)
    r3  = prev_high + 2 * (pp - prev_low)
    s1  = 2 * pp - prev_high
    s2  = pp - (prev_high - prev_low)
    s3  = prev_low - 2 * (prev_high - pp)
    return PivotLevels(
        pp=pp, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3,
        m_r1=(pp + r1) / 2, m_r2=(r1 + r2) / 2,
        m_s1=(pp + s1) / 2, m_s2=(s1 + s2) / 2, m_s3=(s2 + s3) / 2,
    )


def pivot_list(pv: PivotLevels) -> list[tuple[str, float]]:
    """All pivot levels as (label, price) sorted descending."""
    return sorted([
        ("R3", pv.r3), ("R2", pv.r2), ("Mid R1-R2", pv.m_r2), ("R1", pv.r1),
        ("Mid PP-R1", pv.m_r1), ("PP", pv.pp), ("Mid PP-S1", pv.m_s1),
        ("S1", pv.s1), ("Mid S1-S2", pv.m_s2), ("S2", pv.s2),
        ("Mid S2-S3", pv.m_s3), ("S3", pv.s3),
    ], key=lambda x: x[1], reverse=True)


# ── Drop-band reversal analysis ────────────────────────────────────────────────

@dataclass
class DropBandStats:
    drop_lo_pts: float      # band lower bound (e.g. -6)
    drop_hi_pts: float      # band upper bound (e.g. -4)
    n_sessions: int
    close_above_open_rate: float   # fully recovered to open
    recovery_50pct_rate: float     # bounced ≥50% back toward open
    recovery_75pct_rate: float     # bounced ≥75%
    median_bounce_pts: float       # close - low
    avg_bounce_pts: float
    avg_open_to_low_pts: float     # typical open→low distance in this band


def drop_band_analysis(df: pd.DataFrame) -> list[DropBandStats]:
    """Compute intraday bounce stats grouped by magnitude of open→low drop."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["_drop"] = df["Low"] - df["Open"]     # always ≤ 0

    bands = [
        (-1, 0), (-2, -1), (-3, -2), (-4, -3), (-5, -4),
        (-7, -5), (-10, -7), (-999, -10),
    ]
    results: list[DropBandStats] = []
    for lo, hi in bands:
        mask = (df["_drop"] > lo) & (df["_drop"] <= hi)
        sub  = df[mask]
        if len(sub) < 3:
            continue
        bounce    = sub["Close"] - sub["Low"]
        ret_open  = (sub["Open"] - sub["Low"]).replace(0, np.nan)
        pct_rec   = bounce / ret_open
        results.append(DropBandStats(
            drop_lo_pts=lo,
            drop_hi_pts=hi,
            n_sessions=len(sub),
            close_above_open_rate=float((sub["Close"] >= sub["Open"]).mean()),
            recovery_50pct_rate=float((pct_rec >= 0.50).mean()),
            recovery_75pct_rate=float((pct_rec >= 0.75).mean()),
            median_bounce_pts=float(bounce.median()),
            avg_bounce_pts=float(bounce.mean()),
            avg_open_to_low_pts=float(ret_open.mean()),
        ))
    return results


def matching_drop_band(bands: list[DropBandStats], actual_drop_pts: float) -> Optional[DropBandStats]:
    """Find the band that matches today's actual open→low drop."""
    for b in bands:
        if b.drop_lo_pts < actual_drop_pts <= b.drop_hi_pts:
            return b
    return None


# ── VWAP deviation analysis ───────────────────────────────────────────────────

@dataclass
class VwapDeviationStats:
    deviation_pts: float       # how far low was below VWAP (negative)
    n_sessions: int
    close_above_open_rate: float
    median_bounce_pts: float
    avg_bounce_pts: float


def vwap_deviation_analysis(
    df: pd.DataFrame, bands: tuple = (-2, -3, -4, -6, -8, -12)
) -> list[VwapDeviationStats]:
    """Bounce stats based on how far the day's Low fell below VWAP proxy.

    VWAP proxy = (High + Low + Close) / 3  (typical price).
    """
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["_vwap_proxy"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_low_vs_vwap"] = df["Low"] - df["_vwap_proxy"]

    edges = list(bands) + [-999]
    results: list[VwapDeviationStats] = []
    for i in range(len(edges) - 1):
        hi_dev = edges[i]        # e.g. -2  (less negative end of band)
        lo_dev = edges[i + 1]    # e.g. -3
        mask = (df["_low_vs_vwap"] >= lo_dev) & (df["_low_vs_vwap"] < hi_dev)
        sub  = df[mask]
        if len(sub) < 3:
            continue
        bounce = sub["Close"] - sub["Low"]
        results.append(VwapDeviationStats(
            deviation_pts=(hi_dev + lo_dev) / 2,
            n_sessions=len(sub),
            close_above_open_rate=float((sub["Close"] >= sub["Open"]).mean()),
            median_bounce_pts=float(bounce.median()),
            avg_bounce_pts=float(bounce.mean()),
        ))
    return results


# ── S3 pivot historical analysis ──────────────────────────────────────────────

@dataclass
class PivotTouchStats:
    pivot_label: str
    n_touches: int
    tolerance_pts: float
    close_above_open_rate: float
    median_bounce_pts: float
    avg_return_to_open_pts: float


def pivot_touch_analysis(df: pd.DataFrame, tolerance: float = 1.5) -> list[PivotTouchStats]:
    """For each pivot level (S1–S3, R1–R3), compute bounce stats when Low (or High) touches it."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["prev_h"] = df["High"].shift(1)
    df["prev_l"] = df["Low"].shift(1)
    df["prev_c"] = df["Close"].shift(1)
    df = df.dropna(subset=["prev_h", "prev_l", "prev_c"])

    pp  = (df["prev_h"] + df["prev_l"] + df["prev_c"]) / 3
    df["_s1"] = 2 * pp - df["prev_h"]
    df["_s2"] = pp - (df["prev_h"] - df["prev_l"])
    df["_s3"] = df["prev_l"] - 2 * (df["prev_h"] - pp)
    df["_r1"] = 2 * pp - df["prev_l"]
    df["_r2"] = pp + (df["prev_h"] - df["prev_l"])
    df["_pp"] = pp

    results: list[PivotTouchStats] = []

    # Support levels — Low touches
    for label, col in [("S1", "_s1"), ("S2", "_s2"), ("S3", "_s3")]:
        mask = abs(df["Low"] - df[col]) <= tolerance
        sub  = df[mask]
        if len(sub) < 3:
            continue
        bounce     = sub["Close"] - sub["Low"]
        ret_open   = sub["Open"] - sub["Low"]
        results.append(PivotTouchStats(
            pivot_label=label,
            n_touches=len(sub),
            tolerance_pts=tolerance,
            close_above_open_rate=float((sub["Close"] >= sub["Open"]).mean()),
            median_bounce_pts=float(bounce.median()),
            avg_return_to_open_pts=float(ret_open.mean()),
        ))

    # Resistance levels — High touches
    for label, col in [("R1", "_r1"), ("R2", "_r2")]:
        mask = abs(df["High"] - df[col]) <= tolerance
        sub  = df[mask]
        if len(sub) < 3:
            continue
        drop     = sub["High"] - sub["Close"]
        ret_open = sub["High"] - sub["Open"]
        results.append(PivotTouchStats(
            pivot_label=label,
            n_touches=len(sub),
            tolerance_pts=tolerance,
            close_above_open_rate=float((sub["Close"] < sub["Open"]).mean()),  # "bounce" = close below open
            median_bounce_pts=float(drop.median()),
            avg_return_to_open_pts=float(ret_open.mean()),
        ))

    return results


# ── Combined signal score ─────────────────────────────────────────────────────

def reversal_signal_score(
    drop_band: Optional[DropBandStats],
    vwap_dev_pts: float,
    vwap_stats: list[VwapDeviationStats],
    pivot_touch: Optional[PivotTouchStats],
) -> tuple[str, float, str]:
    """Returns (signal, confidence_0_to_1, description) for a reversal at the current price.

    Combines three independent signals with weights:
      - Drop band recovery rate  (weight 0.40)
      - VWAP deviation rate      (weight 0.35)
      - Pivot touch rate         (weight 0.25)
    """
    score = 0.0
    weight_sum = 0.0
    parts: list[str] = []

    if drop_band and drop_band.n_sessions >= 5:
        score      += drop_band.recovery_50pct_rate * 0.40
        weight_sum += 0.40
        parts.append(
            f"Drop band ({drop_band.drop_hi_pts:+.0f} to {drop_band.drop_lo_pts:+.0f} pts): "
            f"{drop_band.recovery_50pct_rate*100:.0f}% bounce rate ({drop_band.n_sessions} sessions)"
        )

    # Find matching VWAP band
    for vs in vwap_stats:
        if abs(vs.deviation_pts - vwap_dev_pts) <= 1.5:
            score      += vs.close_above_open_rate * 0.35
            weight_sum += 0.35
            parts.append(
                f"VWAP dev {vwap_dev_pts:+.1f} pts: {vs.close_above_open_rate*100:.0f}% "
                f"recovered to open ({vs.n_sessions} sessions)"
            )
            break

    if pivot_touch and pivot_touch.n_touches >= 5:
        score      += pivot_touch.close_above_open_rate * 0.25
        weight_sum += 0.25
        parts.append(
            f"{pivot_touch.pivot_label} pivot touch: {pivot_touch.close_above_open_rate*100:.0f}% "
            f"closed above open ({pivot_touch.n_touches} touches)"
        )

    confidence = score / weight_sum if weight_sum > 0 else 0.0

    if confidence >= 0.58:
        signal = "STRONG REVERSAL"
    elif confidence >= 0.48:
        signal = "PROBABLE REVERSAL"
    elif confidence >= 0.38:
        signal = "WATCH"
    else:
        signal = "WEAK / SKIP"

    return signal, confidence, " · ".join(parts)
