"""Weekly moving-average touch detection + weekly order-flow synthesis.

This module finds the highest-quality reversal levels on the weekly timeframe
and combines them with weekly order-flow proxies (CVD, closing strength) to
produce confluence-based reversal setups.

Empirical finding (SPY, last 6 months):
  - 50w SMA touches: 100% bounced (+3.93% avg 5d, max DD only -0.78%)
  - 30w EMA touches in downtrends: bearish (avg -1.42% next 5d, 0% positive)
  - Faster MAs (10w, 20w): mixed but useful for short-term context

Order-flow proxies (CVD, closing strength) on their own do NOT beat baseline
on next-week returns in this sample. They are used as CONFLUENCE/context for
MA-touch setups, not as standalone signals.

All MAs are computed on daily bars but represent weekly-equivalent periods
(e.g. 50w SMA = 250-day SMA), which avoids resampling artifacts and gives a
clean per-day touch detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ── Configuration ────────────────────────────────────────────────────────────

WEEKLY_MA_PERIODS = (10, 20, 30, 50, 100, 200)  # in weeks

# A "touch" = day's intraday range crosses the MA value
# An "approach" = price is within APPROACH_PCT of the MA
APPROACH_PCT = 0.015        # within 1.5% counts as "approaching"
TOUCH_LOOKBACK_DAYS = 183   # ~6 months for outcome stats
FWD_DAYS_DEFAULT = 5        # measure outcome over next 5 trading days

# Bias rules (from empirical analysis on SPY)
LONG_BIAS_MA = "50w SMA"     # touches historically bounce (+3.93% avg)
SHORT_BIAS_MA = "30w EMA"    # touches in downtrend historically drop (-1.42% avg)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MASetup:
    """A single weekly-MA setup: which level to watch and what to expect."""
    ma_label: str               # e.g. "50w SMA"
    ma_value: float             # current MA price
    distance_pct: float         # current_price/ma - 1, positive = price above
    distance_dollars: float     # current_price - ma_value
    bias: Literal["LONG", "SHORT", "NEUTRAL"]
    n_touches_6m: int
    avg_5d_ret_after_touch: float       # % return 5d after a touch
    pct_positive_after_touch: float     # % of touches with positive 5d return
    avg_5d_max_up: float                # avg max 5d high above touch close
    avg_5d_max_dn: float                # avg max 5d low below touch close


@dataclass
class WeeklyOrderFlow:
    """Weekly order-flow snapshot for the most recent completed week."""
    week_ending: pd.Timestamp
    close: float
    weekly_cvd: float            # sum of daily signed-volume for the week
    cvd_4w_avg: float            # 4-week average of weekly CVD
    closing_strength: float      # (close - low) / (high - low), 0..1
    closing_strength_4w_avg: float
    flow_score: float            # composite -100..+100 (negative bearish)
    z_20w: float                 # close vs 20w mean, in std-devs
    interpretation: str          # short human-readable verdict


@dataclass
class ReversalVerdict:
    """Combined MA + order-flow verdict for what to watch next."""
    headline: str                # e.g. "Watch for 50w SMA touch at $661 (-10.4%)"
    bias: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    primary_setup: MASetup | None
    flow: WeeklyOrderFlow | None
    confluence_note: str         # how flow agrees/disagrees with the MA bias
    play_suggestion: str         # e.g. "Buy CALLs on touch of $661 zone"


# ── Moving-average computation ───────────────────────────────────────────────

def compute_weekly_mas(daily: pd.DataFrame,
                       periods_weeks: tuple = WEEKLY_MA_PERIODS) -> pd.DataFrame:
    """Compute weekly-equivalent SMAs and EMAs on daily bars.

    Returns a DataFrame with columns like '10w SMA', '10w EMA', '20w SMA', ...
    indexed by the same dates as `daily`.
    """
    if "Close" not in daily.columns:
        raise ValueError("daily must have a 'Close' column.")
    out = pd.DataFrame(index=daily.index)
    close = daily["Close"]
    for w in periods_weeks:
        n = w * 5  # ~5 trading days per week
        out[f"{w}w SMA"] = close.rolling(n, min_periods=max(20, n // 2)).mean()
        out[f"{w}w EMA"] = close.ewm(span=n, adjust=False, min_periods=max(20, n // 2)).mean()
    return out


# ── Touch detection + outcome stats ──────────────────────────────────────────

def find_touches(daily: pd.DataFrame, ma: pd.Series,
                 lookback_days: int = TOUCH_LOOKBACK_DAYS) -> pd.DatetimeIndex:
    """A 'touch' day = daily Low <= MA <= daily High (price kissed it intraday)."""
    aligned = ma.reindex(daily.index)
    mask = (daily["Low"] <= aligned) & (daily["High"] >= aligned) & aligned.notna()
    if lookback_days is not None and len(daily) > 0:
        cutoff = daily.index.max() - pd.Timedelta(days=lookback_days)
        mask = mask & (daily.index >= cutoff)
    return daily.index[mask]


def touch_outcome_stats(daily: pd.DataFrame,
                        touches: pd.DatetimeIndex,
                        fwd_days: int = FWD_DAYS_DEFAULT) -> dict:
    """Measure forward-N-day outcomes after each touch.

    Returns keys: n, avg_ret_pct, pct_positive, avg_max_up_pct, avg_max_dn_pct.
    Empty stats if no touches.
    """
    if len(touches) == 0:
        return {"n": 0, "avg_ret_pct": 0.0, "pct_positive": 0.0,
                "avg_max_up_pct": 0.0, "avg_max_dn_pct": 0.0}

    rets, max_ups, max_dns = [], [], []
    for d in touches:
        try:
            i = daily.index.get_loc(d)
        except KeyError:
            continue
        if i + fwd_days >= len(daily):
            continue
        c0 = float(daily["Close"].iloc[i])
        c_fwd = float(daily["Close"].iloc[i + fwd_days])
        window = daily.iloc[i + 1: i + 1 + fwd_days]
        if window.empty:
            continue
        rets.append((c_fwd - c0) / c0 * 100)
        max_ups.append((float(window["High"].max()) - c0) / c0 * 100)
        max_dns.append((float(window["Low"].min()) - c0) / c0 * 100)

    if not rets:
        return {"n": 0, "avg_ret_pct": 0.0, "pct_positive": 0.0,
                "avg_max_up_pct": 0.0, "avg_max_dn_pct": 0.0}

    return {
        "n": len(rets),
        "avg_ret_pct": float(np.mean(rets)),
        "pct_positive": float(100 * np.mean(np.array(rets) > 0)),
        "avg_max_up_pct": float(np.mean(max_ups)),
        "avg_max_dn_pct": float(np.mean(max_dns)),
    }


def build_ma_setups(daily: pd.DataFrame,
                    periods_weeks: tuple = WEEKLY_MA_PERIODS,
                    lookback_days: int = TOUCH_LOOKBACK_DAYS,
                    fwd_days: int = FWD_DAYS_DEFAULT) -> list[MASetup]:
    """Build a MASetup for every weekly MA (SMA + EMA), sorted by abs distance."""
    mas = compute_weekly_mas(daily, periods_weeks)
    last_close = float(daily["Close"].iloc[-1])
    setups: list[MASetup] = []

    for col in mas.columns:
        ma_series = mas[col]
        ma_now = ma_series.iloc[-1]
        if pd.isna(ma_now):
            continue
        ma_now = float(ma_now)
        touches = find_touches(daily, ma_series, lookback_days=lookback_days)
        stats = touch_outcome_stats(daily, touches, fwd_days=fwd_days)

        # Bias by empirical asymmetry (computed from the actual touch outcomes).
        # Require either a meaningful sample (n>=3) with strong skew, OR a tiny
        # sample (n=2) with a perfect record — single coin-flips don't qualify.
        n, pct_pos, avg_ret = stats["n"], stats["pct_positive"], stats["avg_ret_pct"]
        long_strong  = (n >= 3 and pct_pos >= 70 and avg_ret >  1.0)
        long_perfect = (n == 2 and pct_pos == 100 and avg_ret >  1.0)
        short_strong  = (n >= 3 and pct_pos <= 30 and avg_ret < -0.5)
        short_perfect = (n == 2 and pct_pos == 0  and avg_ret < -0.5)
        if long_strong or long_perfect:
            bias = "LONG"
        elif short_strong or short_perfect:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        setups.append(MASetup(
            ma_label=col,
            ma_value=ma_now,
            distance_pct=(last_close / ma_now - 1) * 100,
            distance_dollars=last_close - ma_now,
            bias=bias,
            n_touches_6m=stats["n"],
            avg_5d_ret_after_touch=stats["avg_ret_pct"],
            pct_positive_after_touch=stats["pct_positive"],
            avg_5d_max_up=stats["avg_max_up_pct"],
            avg_5d_max_dn=stats["avg_max_dn_pct"],
        ))

    # Sort by absolute distance (closest = most actionable)
    setups.sort(key=lambda s: abs(s.distance_pct))
    return setups


# ── Weekly order-flow synthesis (daily-bar derived) ──────────────────────────

def _zscore(series: pd.Series, win: int = 20) -> pd.Series:
    rolling = series.rolling(win, min_periods=max(4, win // 2))
    return ((series - rolling.mean()) / rolling.std()).clip(-3, 3)


def build_weekly_order_flow(daily: pd.DataFrame) -> WeeklyOrderFlow | None:
    """Aggregate daily bars into a weekly order-flow snapshot.

    Uses daily proxies (signed volume from close-open direction, daily closing
    strength) since true intraday order flow needs tick data we don't have.
    The result is a coarse but consistent weekly accumulation/distribution
    signal.
    """
    if len(daily) < 25:
        return None

    daily = daily.copy()
    rng = (daily["High"] - daily["Low"]).replace(0, np.nan)
    daily["_cs"] = (daily["Close"] - daily["Low"]) / rng
    daily["_sv"] = np.sign(daily["Close"] - daily["Open"]) * daily["Volume"]

    weekly = pd.DataFrame({
        "Open":   daily["Open"].resample("W-FRI").first(),
        "High":   daily["High"].resample("W-FRI").max(),
        "Low":    daily["Low"].resample("W-FRI").min(),
        "Close":  daily["Close"].resample("W-FRI").last(),
        "Volume": daily["Volume"].resample("W-FRI").sum(),
        "CVD":    daily["_sv"].resample("W-FRI").sum(),
    }).dropna(subset=["Close"])

    if len(weekly) < 5:
        return None

    wk_rng = (weekly["High"] - weekly["Low"]).replace(0, np.nan)
    weekly["WkCS"] = (weekly["Close"] - weekly["Low"]) / wk_rng
    weekly["CVD_4w"] = weekly["CVD"].rolling(4, min_periods=2).mean()
    weekly["CS_4w"] = weekly["WkCS"].rolling(4, min_periods=2).mean()
    weekly["z_20w"] = (
        (weekly["Close"] - weekly["Close"].rolling(20, min_periods=8).mean())
        / weekly["Close"].rolling(20, min_periods=8).std()
    )

    # Composite flow score: -100 (heavy distribution) to +100 (heavy accumulation)
    cvd_z = _zscore(weekly["CVD"], 20)
    cs_z = _zscore(weekly["WkCS"] - 0.5, 20)
    weekly["flow_score"] = ((cvd_z + cs_z) / 2 * 33).clip(-100, 100)

    last = weekly.iloc[-1]
    if pd.isna(last["WkCS"]) or pd.isna(last["flow_score"]):
        return None

    score = float(last["flow_score"])
    cs = float(last["WkCS"])
    cvd = float(last["CVD"])
    cvd_avg = float(last["CVD_4w"]) if pd.notna(last["CVD_4w"]) else 0.0
    cs_avg = float(last["CS_4w"]) if pd.notna(last["CS_4w"]) else 0.5
    z20 = float(last["z_20w"]) if pd.notna(last["z_20w"]) else 0.0

    # Human verdict
    if score >= 30:
        interp = "Strong accumulation — buyers in control"
    elif score >= 10:
        interp = "Mild accumulation"
    elif score <= -30:
        interp = "Strong distribution — sellers in control"
    elif score <= -10:
        interp = "Mild distribution"
    else:
        interp = "Neutral / mixed flow"

    return WeeklyOrderFlow(
        week_ending=weekly.index[-1],
        close=float(last["Close"]),
        weekly_cvd=cvd,
        cvd_4w_avg=cvd_avg,
        closing_strength=cs,
        closing_strength_4w_avg=cs_avg,
        flow_score=score,
        z_20w=z20,
        interpretation=interp,
    )


# ── Combined verdict ─────────────────────────────────────────────────────────

def build_reversal_verdict(daily: pd.DataFrame) -> ReversalVerdict:
    """Pick the best actionable setup and combine it with order-flow context."""
    setups = build_ma_setups(daily)
    flow = build_weekly_order_flow(daily)

    # Find the most actionable directional setup:
    # 1. Closest LONG-bias MA below current price (touch = bounce setup)
    # 2. Closest SHORT-bias MA above current price (touch = breakdown setup)
    long_candidates = [s for s in setups
                       if s.bias == "LONG" and s.distance_pct > 0]
    short_candidates = [s for s in setups
                        if s.bias == "SHORT" and s.distance_pct < 0]

    primary = None
    play = ""
    bias: Literal["LONG", "SHORT", "NEUTRAL"] = "NEUTRAL"
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    headline = "No high-confidence MA setup nearby — wait for price to approach a key level."

    if long_candidates:
        primary = min(long_candidates, key=lambda s: s.distance_pct)
        bias = "LONG"
        headline = (
            f"Watch for {primary.ma_label} touch at ${primary.ma_value:,.2f} "
            f"({primary.distance_pct:+.1f}% from here) — historically a HIGH-CONVICTION LONG setup "
            f"({primary.pct_positive_after_touch:.0f}% bounced, avg +{primary.avg_5d_ret_after_touch:.2f}% over 5d)."
        )
        play = (
            f"If SPY pulls back to ${primary.ma_value:,.2f}, buy CALLs (5–10d expiry, ATM or "
            f"slightly OTM). Stop if close breaks below the MA by more than 0.5%."
        )
    elif short_candidates:
        primary = max(short_candidates, key=lambda s: s.distance_pct)  # closest above
        bias = "SHORT"
        headline = (
            f"Watch for {primary.ma_label} test at ${primary.ma_value:,.2f} "
            f"({primary.distance_pct:+.1f}% from here) — historically a SHORT setup "
            f"({100 - primary.pct_positive_after_touch:.0f}% dropped, avg {primary.avg_5d_ret_after_touch:.2f}% over 5d)."
        )
        play = (
            f"If SPY rallies to ${primary.ma_value:,.2f} and stalls, buy PUTs (5–10d expiry). "
            f"Confluence with bearish flow score required."
        )

    # Confluence with weekly flow
    confluence = "Weekly flow data unavailable."
    if flow is not None and primary is not None:
        if bias == "LONG":
            if flow.flow_score >= 10:
                confidence = "HIGH"
                confluence = f"Confirmed: weekly flow is {flow.flow_score:+.0f} (accumulation supports the bounce)."
            elif flow.flow_score <= -30:
                confidence = "LOW"
                confluence = f"Conflict: weekly flow is {flow.flow_score:+.0f} (distribution warns the MA may break)."
            else:
                confidence = "MEDIUM"
                confluence = f"Neutral confluence: weekly flow {flow.flow_score:+.0f} — wait for flow to confirm."
        elif bias == "SHORT":
            if flow.flow_score <= -10:
                confidence = "HIGH"
                confluence = f"Confirmed: weekly flow is {flow.flow_score:+.0f} (distribution supports the drop)."
            elif flow.flow_score >= 30:
                confidence = "LOW"
                confluence = f"Conflict: weekly flow is {flow.flow_score:+.0f} (accumulation warns the resistance may break)."
            else:
                confidence = "MEDIUM"
                confluence = f"Neutral confluence: weekly flow {flow.flow_score:+.0f} — wait for flow to confirm."
    elif flow is not None:
        confluence = f"Current weekly flow: {flow.flow_score:+.0f} ({flow.interpretation.lower()})."

    return ReversalVerdict(
        headline=headline,
        bias=bias,
        confidence=confidence,
        primary_setup=primary,
        flow=flow,
        confluence_note=confluence,
        play_suggestion=play,
    )
