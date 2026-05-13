"""Mastertrades — Command Center & Trading Dashboard.

Pages:
  1. Command Center   — Today's jackpot signals for SPY/QQQ/IWM/AAPL
  2. Scanner          — Multi-ticker volatility scanner (broader universe)
  3. Gap Reversal     — Gap fill detection and reversal signal scanner
  4. Account Tracker  — Equity curve, trades, milestones
  5. Weekday Patterns — Volatility patterns by day of week
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import json
import logging
import math
from datetime import datetime, date

import pandas as pd
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mastertrades",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

DATA_DIR   = APP_DIR / "data"
MODEL_DIR  = APP_DIR / "models"
ACCT_PATH  = DATA_DIR / "account_state.json"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
<style>
  /* ── Material Symbols icon font (Streamlit 1.57 internal icons) ── */
  .material-symbols-rounded,
  .material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    display: inline-block !important;
    font-size: inherit;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
  }

  /* ── Load Inter and override Streamlit's default Source Sans Pro everywhere ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body,
  [class*="css"],
  .stMarkdown, .stMarkdown *,
  .stText, .stText *,
  p, span, div, label, button, input, select, textarea,
  h1, h2, h3, h4, h5, h6,
  [data-testid] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
    font-feature-settings: "kern" 1, "liga" 1 !important;
  }

  /* ── Streamlit metric widget ── */
  [data-testid="metric-container"] {
    background: #161b22;
    border-radius: 10px;
    padding: 12px 16px !important;
    border: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stMetricValue"] > div {
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
  }
  [data-testid="stMetricLabel"] > div {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #8b949e !important;
  }
  [data-testid="stMetricDelta"] > div { font-size: 12px !important; font-weight: 600 !important; }

  /* ── Dataframe headers ── */
  thead tr th {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
    color: #8b949e !important;
  }
  tbody tr td { font-size: 12px !important; font-weight: 500 !important; }

  /* ── Sidebar labels ── */
  .stSidebar label, .stSidebar .stMarkdown { font-size: 12px !important; font-weight: 500 !important; }

  /* ── Layout ── */
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

  /* ── Hide chrome ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* ── Force Inter onto every inline HTML card (div with style attr) ── */
  .stMarkdown div[style], .stMarkdown span[style] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
  }
  /* ── Sharper section headers injected via st.markdown ── */
  .stMarkdown > div > div > div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Helvetica, Arial, sans-serif !important;
  }
  /* ── Sidebar widgets ── */
  .stSidebar * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Helvetica, Arial, sans-serif !important;
  }
  /* ── Section divider ── */
  hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.4rem 0 !important; }
</style>
""")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def signal_color(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "#ffd633",
        "GO_JACKPOT":       "#3fb950",
        "GO_HOT":           "#d29922",
        "SKIP":             "#8b949e",
    }.get(signal or "SKIP", "#8b949e")


def signal_label(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "🌟 ULTRA JACKPOT",
        "GO_JACKPOT":       "✅ JACKPOT",
        "GO_HOT":           "🔥 HOT",
        "SKIP":             "⏭ SKIP",
    }.get(signal or "SKIP", "⏭ SKIP")


def signal_action(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "Both models firing at peak confidence — size up, trade aggressively.",
        "GO_JACKPOT":       "Both vol + P&L models confirm — this is a trade day.",
        "GO_HOT":           "Volatility model firing, P&L model neutral — trade smaller or wait for confirmation.",
        "SKIP":             "Models expect a calm session — no setup, stand by.",
    }.get(signal or "SKIP", "Stand by.")


def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_dollar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"${v:,.2f}"


def section(title: str, subtitle: str = ""):
    sub_html = f'<div style="color:#8b949e;font-size:12px;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.html(
        f"""<div style="margin:1.6rem 0 0.8rem;">
          <span style="font-size:17px;font-weight:800;color:#e6edf3;
                       letter-spacing:.01em;">{title}</span>
          {sub_html}
        </div>""")


# ─── Cached data loaders ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_jackpot_scan(tickers=("SPY", "QQQ", "IWM", "AAPL")):
    from src.jackpot_scanner import scan_jackpot_universe
    rows, errors = scan_jackpot_universe(
        tickers=list(tickers),
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        refresh_data=True,
    )
    return rows, errors


@st.cache_data(ttl=900, show_spinner=False)
def load_scanner(tickers):
    from src.scanner import scan_universe
    df, errors = scan_universe(
        tickers=list(tickers),
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        refresh_data=True,
    )
    return df, errors


@st.cache_data(ttl=1800, show_spinner=False)
def load_weekday_data(ticker="SPY", lookback_days=504):
    from src.scanner import fetch_or_load_daily
    from src.volatility import all_daily_features
    from src.order_flow_proxies import daily_order_flow_features

    daily = fetch_or_load_daily(ticker, data_dir=DATA_DIR, refresh=True)
    vol_feats = all_daily_features(daily)
    daily = daily.join(vol_feats, how="left")
    cutoff = daily.index[-1] - pd.Timedelta(days=lookback_days)
    daily = daily[daily.index >= cutoff].copy()
    try:
        flow = daily_order_flow_features(daily)
        daily = daily.join(flow, how="left")
    except Exception:
        pass
    return daily


@st.cache_data(ttl=900, show_spinner=False)
def load_gap_analysis(ticker: str, lookback_years: int = 5):
    from src.scanner import fetch_or_load_daily
    from src.gap_analysis import run_gap_analysis
    daily = fetch_or_load_daily(ticker, data_dir=DATA_DIR, refresh=True)
    df_feat, stats_bucket, stats_dir, stats_wd, today = run_gap_analysis(
        ticker, daily, lookback_years=lookback_years
    )
    return df_feat, stats_bucket, stats_dir, stats_wd, today


@st.cache_data(ttl=30, show_spinner=False)
def load_live_quotes(tickers: tuple) -> dict:
    """Fetch live Polygon snapshots for a tuple of tickers.
    Returns {TICKER: snap_dict}. TTL=30 s so quotes auto-refresh.
    Falls back to empty dict if Polygon unavailable.
    """
    try:
        from src.polygon_feed import fetch_multi_snapshot, has_polygon_key
        if not has_polygon_key():
            return {}
        return fetch_multi_snapshot(list(tickers))
    except Exception:
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def load_0dte_alert(ticker: str = "SPY") -> dict:
    """Live 0DTE entry alert — auto-refreshes every 30 s.

    Returns a dict:
        status:   "ENTRY_OPEN" | "APPROACHING" | "QUIET" | "UNAVAILABLE"
        ticker:   str
        drop_pts: float   (open - low, positive = sold off)
        rise_pts: float   (high - open, positive = ran up)
        day_open: float
        day_low:  float
        day_high: float
        day_vwap: float
        recs:     list[StrikeRecommendation]  — sorted by est_gain_pct desc
        hist_pct_1000plus: int  — from drop-band table
    """
    try:
        from src.polygon_feed import has_polygon_key
        from src.options_scanner import (
            fetch_0dte_chain, recommend_strikes, drop_band_multiplier_table
        )
        from datetime import datetime

        if not has_polygon_key():
            return {"status": "UNAVAILABLE", "ticker": ticker, "drop_pts": 0.0,
                    "rise_pts": 0.0, "day_open": 0.0, "day_low": 0.0,
                    "day_high": 0.0, "day_vwap": 0.0, "recs": [], "hist_pct_1000plus": 0}

        snaps  = fetch_0dte_alert._live_cache if False else {}  # just call live quotes
        snaps  = __import__("src.polygon_feed", fromlist=["fetch_multi_snapshot"]).fetch_multi_snapshot([ticker])
        snap   = snaps.get(ticker, {})

        day_open = float(snap.get("day_open",  0.0) or 0.0)
        day_low  = float(snap.get("day_low",   0.0) or 0.0)
        day_high = float(snap.get("day_high",  0.0) or 0.0)
        day_vwap = float(snap.get("day_vwap",  0.0) or 0.0)

        if day_open <= 0 or day_low <= 0:
            return {"status": "UNAVAILABLE", "ticker": ticker, "drop_pts": 0.0,
                    "rise_pts": 0.0, "day_open": day_open, "day_low": day_low,
                    "day_high": day_high, "day_vwap": day_vwap, "recs": [], "hist_pct_1000plus": 0}

        drop_pts = day_open - day_low    # positive = sold off from open
        rise_pts = day_high - day_open   # positive = ran up from open

        # Historical probability from drop-band table
        table = drop_band_multiplier_table()
        hist_pct = 0
        for row in table:
            band = row["band"]
            if "0–1" in band   and drop_pts < 1:   hist_pct = row["pct_1000plus"]; break
            if "1–2" in band   and drop_pts < 2:   hist_pct = row["pct_1000plus"]; break
            if "2–3" in band   and drop_pts < 3:   hist_pct = row["pct_1000plus"]; break
            if "3–5" in band   and drop_pts < 5:   hist_pct = row["pct_1000plus"]; break
            if "5–7" in band   and drop_pts < 7:   hist_pct = row["pct_1000plus"]; break
            if "7–10" in band  and drop_pts < 10:  hist_pct = row["pct_1000plus"]; break
            if "10+" in band:                       hist_pct = row["pct_1000plus"]; break

        # Determine status
        if drop_pts >= 3.0:
            status = "ENTRY_OPEN"
        elif drop_pts >= 1.5:
            status = "APPROACHING"
        else:
            status = "QUIET"

        # Only fetch options chain when a real setup is in play
        recs = []
        if drop_pts >= 2.0:
            try:
                exp_date = datetime.now().strftime("%Y-%m-%d")
                contracts = fetch_0dte_chain(ticker, exp_date=exp_date, contract_type="call")
                recs = recommend_strikes(day_open, day_low, contracts)
            except Exception:
                recs = []

        return {
            "status":              status,
            "ticker":              ticker,
            "drop_pts":            round(drop_pts, 2),
            "rise_pts":            round(rise_pts, 2),
            "day_open":            day_open,
            "day_low":             day_low,
            "day_high":            day_high,
            "day_vwap":            day_vwap,
            "recs":                recs,
            "hist_pct_1000plus":   hist_pct,
        }
    except Exception:
        return {"status": "UNAVAILABLE", "ticker": ticker, "drop_pts": 0.0,
                "rise_pts": 0.0, "day_open": 0.0, "day_low": 0.0,
                "day_high": 0.0, "day_vwap": 0.0, "recs": [], "hist_pct_1000plus": 0}


@st.cache_data(ttl=1800, show_spinner=False)
def load_key_levels(ticker: str, lookback_years: int = 2):
    """Pre-compute all reversal-level statistics for a ticker."""
    from src.scanner import fetch_or_load_daily
    from src.key_levels import drop_band_analysis, vwap_deviation_analysis, pivot_touch_analysis
    daily = fetch_or_load_daily(ticker, data_dir=DATA_DIR, refresh=True)
    cutoff = daily.index[-1] - pd.DateOffset(years=lookback_years)
    daily  = daily[daily.index >= cutoff].dropna()
    bands      = drop_band_analysis(daily)
    vwap_stats = vwap_deviation_analysis(daily)
    pivot_hist = pivot_touch_analysis(daily)
    return daily, bands, vwap_stats, pivot_hist


@st.cache_data(ttl=60, show_spinner=False)
def load_account():
    from src.account_state import load_state
    return load_state(path=ACCT_PATH)


# ─── Sidebar navigation ───────────────────────────────────────────────────────

PAGE_META = {
    "Command Center":   "Today's trade signals",
    "Scanner":          "Ranked volatility universe",
    "Gap Reversal":     "Gap fill & reversal setups",
    "Reversal Levels":  "Intraday low/high reversal zones",
    "0DTE Lottery":     "1000%+ options plays & sweet spots",
    "Account Tracker":  "Equity curve & trade log",
    "Weekday Patterns": "Vol by day of week",
}

with st.sidebar:
    st.html(
        """<div style="font-size:20px;font-weight:800;color:#fff;
                       letter-spacing:-.01em;margin-bottom:4px;">📈 Mastertrades</div>
           <div style="color:#8b949e;font-size:11px;margin-bottom:16px;">
             0DTE Options Intelligence</div>""")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        list(PAGE_META.keys()),
        label_visibility="collapsed",
        format_func=lambda p: p,
    )
    st.caption(PAGE_META[page])
    st.markdown("---")

    # ── Account equity (always visible) ───────────────────────────────────────
    try:
        from src.account_state import load_state as _load_acct
        _acct = _load_acct()
        _default_eq = float(_acct.current_equity or 500.0)
    except Exception:
        _default_eq = 500.0

    st.html(
        """<div style="font-size:10px;font-weight:800;color:#8b949e;
                       text-transform:uppercase;letter-spacing:.08em;
                       margin-bottom:4px;">Account Balance</div>""")
    equity_input = st.number_input(
        "Account equity ($)",
        value=_default_eq,
        min_value=10.0,
        step=50.0,
        label_visibility="collapsed",
    )
    st.html(
        f"""<div style="background:#0d1f14;border:1px solid #3fb950;
                       border-radius:6px;padding:6px 10px;margin-top:2px;
                       margin-bottom:12px;">
             <span style="color:#3fb950;font-size:18px;font-weight:800;
                          font-variant-numeric:tabular-nums;">
               ${equity_input:,.2f}</span>
             <span style="color:#8b949e;font-size:10px;margin-left:6px;">current equity</span>
           </div>""")
    st.markdown("---")

    # Data source status
    try:
        from src.polygon_feed import has_polygon_key, fetch_prev_close
        _poly_ok = has_polygon_key()
    except Exception:
        _poly_ok = False

    if _poly_ok:
        st.html(
            f"""<div style="background:#0d1f14;border:1px solid #3fb950;
                           border-radius:8px;padding:10px 12px;margin-bottom:10px;">
                 <div style="font-size:10px;font-weight:800;color:#3fb950;
                             text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                   ✅ Polygon.io Live</div>
                 <div style="font-size:10px;color:#8b949e;line-height:1.6;">
                   Exchange-quality OHLCV<br>
                   Live quotes &amp; snapshots<br>
                   <span style="color:#ffd633;">Refreshes every 30 s</span>
                 </div>
               </div>""")
    else:
        st.html(
            """<div style="background:#1a1208;border:1px solid #6e7681;
                           border-radius:8px;padding:10px 12px;margin-bottom:10px;">
                 <div style="font-size:10px;font-weight:800;color:#6e7681;
                             text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                   ○ Yahoo Finance (fallback)</div>
                 <div style="font-size:10px;color:#8b949e;">
                   Add POLYGON_API_KEY to enable<br>exchange-quality data</div>
               </div>""")

    st.caption(f"Updated: {datetime.now().strftime('%b %d, %H:%M')}")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "Command Center":
    JACKPOT_TICKERS = ("SPY", "QQQ", "IWM", "AAPL")

    with st.spinner("Running scanner… (first run trains ML models, ~60 s)"):
        try:
            rows, errors = load_jackpot_scan(JACKPOT_TICKERS)
        except Exception as e:
            st.error(f"Scanner error: {e}")
            rows, errors = [], [str(e)]

    if errors:
        with st.expander(f"⚠️ {len(errors)} warning(s)"):
            for err in errors:
                st.text(err if isinstance(err, str) else f"{err.get('ticker','?')}: {err.get('error','unknown')}")

    if not rows:
        st.warning("No scan results yet. Try refreshing.")
        st.stop()

    # ── Market phase — computed once, used throughout this page ───────────────
    from src.jackpot_scanner import market_phase as _market_phase
    _ph = _market_phase()
    _ph_p = _ph["phase"]

    _noi = _ph.get("next_open_in") or 0   # minutes until next open (None when open)
    _mso = _ph.get("minutes_since_open") or 0  # minutes since open (None pre-open)
    _ph_banner_cfg = {
        "PRE_OPEN": (
            "#d29922", "#1c1600", "#2a1e00",
            "🟡 PRE-MARKET — Signal Preview",
            (f"Cash market opens in {_noi//60}h {_noi%60}m (9:30 AM ET). "
             f"Signal is 95% ready — only the opening gap is still missing. "
             f"<strong style='color:#ffd633;'>Wait for 9:50 AM ET before trading.</strong>"),
        ),
        "OPEN_PENDING_DATA": (
            "#d29922", "#1c1600", "#2a1e00",
            "🟡 OPEN · DATA SETTLING (9:30–9:50 AM ET)",
            (f"Market opened {_mso} min ago. "
             f"Yahoo's daily bar takes ~15–20 min to reflect the opening print. "
             f"<strong style='color:#ffd633;'>Signal may still be from yesterday — "
             f"refresh at 9:50 AM ET for the confirmed read.</strong>"),
        ),
        "OPEN_LIVE": (
            "#3fb950", "#0d1f14", "#0a1a0f",
            "🟢 MARKET OPEN · SIGNAL LIVE",
            (f"Decision window open · {_mso} min into the session · "
             f"Signal is final for the day. Closes at 4:00 PM ET."),
        ),
        "AFTER_HOURS": (
            "#58a6ff", "#0a1428", "#0d1f36",
            "🔵 AFTER-HOURS · SESSION CLOSED",
            ("Today's trading window ended at 4:00 PM ET. "
             "All 0DTE options expired worthless at close. "
             "<strong style='color:#a5d6ff;'>Signal below is today's FINAL result — "
             "use it as context for tomorrow's plan.</strong>"),
        ),
        "WEEKEND": (
            "#8b949e", "#161b22", "#1c2128",
            "⚫ WEEKEND · MARKETS CLOSED",
            ("Markets reopen Monday 9:30 AM ET. "
             "Signal below is a <strong style='color:#e6edf3;'>preview for Monday</strong>, "
             "computed from Friday's close — the opening gap will refine it at the bell."),
        ),
    }
    _ph_color, _ph_bg, _ph_border_bg, _ph_title, _ph_sub = _ph_banner_cfg.get(
        _ph_p, _ph_banner_cfg["AFTER_HOURS"]
    )
    st.html(
        f"""<div style="background:{_ph_bg};border:1px solid {_ph_color};
                        border-radius:10px;padding:12px 18px;margin-bottom:12px;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;
                        display:flex;align-items:flex-start;gap:10px;">
          <div style="flex:1;">
            <div style="font-size:11px;font-weight:800;color:{_ph_color};
                        text-transform:uppercase;letter-spacing:.09em;margin-bottom:4px;">
              {_ph_title}</div>
            <div style="font-size:12px;color:#8b949e;line-height:1.6;">{_ph_sub}</div>
          </div>
          <div style="color:#30363d;font-size:10px;white-space:nowrap;padding-top:2px;">
            {datetime.now().strftime('%H:%M ET')}</div>
        </div>""")

    # ── Hero verdict ──────────────────────────────────────────────────────────
    rank_order = {"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2, "SKIP": 1}
    best_row = max(rows, key=lambda r: rank_order.get(r.signal, 0))
    sig = best_row.signal
    hero_border = signal_color(sig)

    hero_bg = {
        "GO_ULTRA_JACKPOT": "linear-gradient(135deg,#1a1208,#2d2008)",
        "GO_JACKPOT":       "linear-gradient(135deg,#0d1f14,#12311e)",
        "GO_HOT":           "linear-gradient(135deg,#1f1808,#2e2210)",
        "SKIP":             "linear-gradient(135deg,#0d1117,#131920)",
    }.get(sig, "linear-gradient(135deg,#0d1117,#131920)")

    # Count trade vs skip tickers
    trade_tickers = [r.ticker for r in rows if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")]
    hot_tickers   = [r.ticker for r in rows if r.signal == "GO_HOT"]

    trade_tickers_html = (
        f'<div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.1);'
        f'font-size:12px;color:#8b949e;">Trade: <strong style="color:#3fb950;">'
        f'{", ".join(trade_tickers)}</strong></div>'
    ) if trade_tickers else ""

    st.html(
        f"""
        <div style="background:{hero_bg};border:2px solid {hero_border};
                    border-radius:16px;padding:28px 32px;margin-bottom:8px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="color:#8b949e;font-size:11px;letter-spacing:.14em;
                          text-transform:uppercase;font-weight:700;margin-bottom:8px;">
                {datetime.now().strftime('%A, %B %d')} · Today's Verdict
              </div>
              <div style="font-size:38px;font-weight:800;color:{hero_border};
                          margin-bottom:10px;line-height:1;">{signal_label(sig)}</div>
              <div style="font-size:14px;color:#c9d1d9;max-width:520px;line-height:1.5;">
                {signal_action(sig)}
              </div>
            </div>
            <div style="text-align:right;min-width:140px;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          margin-bottom:4px;">Lead ticker</div>
              <div style="font-size:32px;font-weight:800;color:#fff;">{best_row.ticker}</div>
              <div style="color:{hero_border};font-size:13px;font-weight:700;">
                P(vol) {fmt_pct(best_row.p_vol)} · P(pnl) {fmt_pct(best_row.p_pnl)}
              </div>
            </div>
          </div>
          {trade_tickers_html}
        </div>
        """
    )

    # ── Live quotes ───────────────────────────────────────────────────────────
    live_q = load_live_quotes(JACKPOT_TICKERS)

    # ── Ticker cards ──────────────────────────────────────────────────────────
    section("Per-Ticker Signals")
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        sig_c = signal_color(row.signal)
        snap  = live_q.get(row.ticker, {})

        # Price: prefer Polygon live, fall back to ML model's last close
        price     = snap.get("last_price") or row.last_close
        chg_pct   = snap.get("change_pct") or getattr(row, "pct_change", None)
        chg_pts   = snap.get("change_pts", 0.0)
        day_high  = snap.get("day_high") or 0.0
        day_low   = snap.get("day_low")  or 0.0
        day_vwap  = snap.get("day_vwap") or 0.0
        # Only show LIVE badge when cash market is actually open
        s_label   = snap.get("status_label", "") if _ph["is_open"] else ""
        rsi_v     = getattr(row, "rsi14", None)

        price_str = fmt_dollar(price)
        chg_c     = "#3fb950" if (chg_pct or 0) >= 0 else "#f85149"
        chg_str   = f"{chg_pct*100:+.2f}%" if chg_pct is not None and not math.isnan(float(chg_pct)) else "—"
        chg_pts_s = f"{chg_pts:+.2f}" if chg_pts else ""
        rsi_str   = f"{rsi_v:.0f}" if rsi_v is not None and not math.isnan(float(rsi_v)) else "—"
        is_trade  = row.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")
        glow      = f"box-shadow:0 0 20px {sig_c}55;" if is_trade else ""
        live_badge = (
            f'<span style="background:#0d1f14;color:#3fb950;font-size:8px;'
            f'font-weight:800;padding:2px 5px;border-radius:3px;'
            f'letter-spacing:.05em;margin-left:4px;">{s_label}</span>'
            if s_label else ""
        )
        hl_row = (
            f'<div><div style="color:#8b949e;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.07em;">Day H / L</div>'
            f'<div style="color:#e6edf3;font-weight:700;font-size:11px;">'
            f'{fmt_dollar(day_high)} / {fmt_dollar(day_low)}</div></div>'
            if day_high else ""
        )
        vwap_row = (
            f'<div><div style="color:#8b949e;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.07em;">VWAP</div>'
            f'<div style="color:#58a6ff;font-weight:700;">{fmt_dollar(day_vwap)}</div></div>'
            if day_vwap else ""
        )

        with col:
            st.html(
                f"""
                <div style="background:#161b22;border:2px solid {sig_c};
                            border-radius:14px;padding:20px 16px;{glow}
                            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;">
                  <div style="display:flex;justify-content:space-between;align-items:center;
                              margin-bottom:4px;">
                    <span style="font-size:24px;font-weight:800;color:#fff;">{row.ticker}</span>
                    <span style="background:{sig_c};color:#0c1117;font-size:11px;
                                 font-weight:800;padding:3px 8px;border-radius:4px;
                                 letter-spacing:.05em;">{row.signal.replace("GO_","")}</span>
                  </div>
                  <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:14px;">
                    <span style="font-size:28px;font-weight:800;color:#fff;">{price_str}</span>
                    {live_badge}
                  </div>
                  <div style="font-size:15px;font-weight:700;color:{chg_c};margin-bottom:14px;">
                    {chg_str}
                    <span style="font-size:11px;font-weight:500;color:{chg_c};">{chg_pts_s}</span>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;row-gap:10px;
                              column-gap:8px;font-size:11px;">
                    <div>
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;">P(vol)</div>
                      <div style="color:{sig_c};font-size:16px;font-weight:800;">{fmt_pct(row.p_vol)}</div>
                    </div>
                    <div>
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;">P(pnl)</div>
                      <div style="color:{sig_c};font-size:16px;font-weight:800;">{fmt_pct(row.p_pnl)}</div>
                    </div>
                    {hl_row}
                    {vwap_row}
                    <div>
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;">RSI(14)</div>
                      <div style="color:#e6edf3;font-weight:700;">{rsi_str}</div>
                    </div>
                  </div>
                </div>
                """
            )

    # ── Live 0DTE Entry Alert ─────────────────────────────────────────────────
    st.markdown("---")
    alert = load_0dte_alert("SPY")
    a_status   = alert["status"]
    a_drop     = alert["drop_pts"]
    a_open     = alert["day_open"]
    a_low      = alert["day_low"]
    a_high     = alert["day_high"]
    a_vwap     = alert["day_vwap"]
    a_recs     = alert["recs"]
    a_hist_pct = alert["hist_pct_1000plus"]

    # Override 0DTE display when the cash session is closed.
    # All 0DTE options expire at 4 PM ET — there is nothing actionable after close.
    if not _ph["is_open"]:
        a_status = "SESSION_CLOSED"

    if a_status == "SESSION_CLOSED":
        _next_open_label = (
            "Monday 9:30 AM ET" if _ph_p == "WEEKEND"
            else "9:50 AM ET" if _ph_p in ("PRE_OPEN", "OPEN_PENDING_DATA")
            else "next session"
        )
        alert_border = "#30363d"
        alert_bg     = "#0d1117"
        if _ph_p in ("PRE_OPEN", "OPEN_PENDING_DATA", "WEEKEND"):
            alert_title = "🔒 0DTE NOT YET ACTIVE — Awaiting Market Open"
            alert_sub   = (
                f"No 0DTE session has started yet today. "
                f"The entry alert will activate once the cash market opens and SPY data flows in. "
                f"Check back at {_next_open_label}."
            )
        else:  # AFTER_HOURS
            alert_title = "🔒 0DTE SESSION CLOSED — All Options Expired at 4:00 PM ET"
            alert_sub   = (
                f"Today's cash session ended at 4:00 PM ET. 0DTE contracts are now worthless. "
                + (f"Session review: Open ${a_open:.2f} · Low ${a_low:.2f} · "
                   f"Drop {a_drop:.1f} pts · High ${a_high:.2f}. "
                   if a_open > 0 else "")
                + f"Next entry window opens tomorrow at 9:30 AM ET."
            )
    elif a_status == "ENTRY_OPEN":
        alert_border = "#ffd633"
        alert_bg     = "linear-gradient(135deg,#1a1208,#131008)"
        alert_title  = "🎯 0DTE ENTRY WINDOW OPEN — SPY REVERSAL SETUP ACTIVE"
        alert_sub    = (f"SPY dropped {a_drop:.1f} pts from open · "
                        f"Historical 1000%+ probability: {a_hist_pct}% of sessions")
    elif a_status == "APPROACHING":
        alert_border = "#d29922"
        alert_bg     = "linear-gradient(135deg,#1a1508,#161008)"
        alert_title  = "👀 APPROACHING SETUP — SPY DROP BUILDING"
        alert_sub    = (f"SPY down {a_drop:.1f} pts from open · "
                        f"Setup activates at 3 pts · Need {max(0, 3-a_drop):.1f} more pts of drop")
    elif a_status == "QUIET":
        alert_border = "#30363d"
        alert_bg     = "#0d1117"
        alert_title  = "⏸ QUIET — No 0DTE Setup Yet"
        alert_sub    = (f"SPY within {a_drop:.1f} pts of open · "
                        f"Setup fires when drop ≥ 3 pts from open (${a_open:.2f})")
    else:
        alert_border = "#30363d"
        alert_bg     = "#0d1117"
        alert_title  = "⏸ 0DTE ALERT — Awaiting Live Data"
        alert_sub    = "Live Polygon data required — check Polygon API key"

    # ── Kelly allocation for the 0DTE alert ───────────────────────────────────
    from src.position_sizer import recommend_allocation as _rec_alloc
    _0dte_alloc = _rec_alloc("ENTRY_OPEN", equity_input) if a_status == "ENTRY_OPEN" else None
    _0dte_alloc_html = ""
    if _0dte_alloc and a_status == "ENTRY_OPEN":
        _0dte_alloc_html = (
            f'<div style="background:rgba(0,0,0,0.4);border:1px solid #ffd633;'
            f'border-radius:8px;padding:10px 14px;margin-bottom:12px;'
            f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
            f'<div>'
            f'<div style="color:#ffd633;font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:.07em;">Kelly Allocation — 10% Lottery Play</div>'
            f'<div style="color:#8b949e;font-size:11px;margin-top:2px;">'
            f'ML is SKIP · Intraday reversal only · Risk only what you can lose</div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;">'
            f'<div style="text-align:center;">'
            f'<div style="color:#8b949e;font-size:9px;text-transform:uppercase;">Spend</div>'
            f'<div style="color:#ffd633;font-size:22px;font-weight:900;">'
            f'${_0dte_alloc.alloc_dollars:,.2f}</div></div>'
            f'<div style="text-align:center;">'
            f'<div style="color:#8b949e;font-size:9px;text-transform:uppercase;">If 1000% hit</div>'
            f'<div style="color:#3fb950;font-size:22px;font-weight:900;">'
            f'+${_0dte_alloc.alloc_dollars * 9:,.2f}</div></div>'
            f'<div style="text-align:center;">'
            f'<div style="color:#8b949e;font-size:9px;text-transform:uppercase;">Max Loss</div>'
            f'<div style="color:#f85149;font-size:22px;font-weight:900;">'
            f'-${_0dte_alloc.max_loss:,.2f}</div></div>'
            f'</div></div>'
        )

    # Build strike recommendation section
    if a_recs and a_status == "ENTRY_OPEN":
        top3     = a_recs[:3]
        best_rec = top3[0]
        recs_html = ""
        for idx, rec in enumerate(top3):
            is_best = idx == 0
            rec_bg  = "#0d1f14" if is_best else "#161b22"
            rec_bdr = "#3fb950" if is_best else "#30363d"
            best_tag = (
                '<div style="font-size:9px;background:#0d1f14;border:1px solid #3fb950;'
                'color:#3fb950;padding:2px 7px;border-radius:4px;display:inline-block;'
                'margin-bottom:6px;">★ BEST R/R</div><br>' if is_best else ""
            )
            gain_c = "#ffd633" if rec.est_gain_pct >= 1000 else "#3fb950" if rec.est_gain_pct >= 500 else "#58a6ff"
            recs_html += f"""
            <div style="background:{rec_bg};border:2px solid {rec_bdr};border-radius:12px;
                        padding:16px;flex:1;min-width:160px;text-align:center;">
              {best_tag}
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.07em;margin-bottom:4px;">
                CALL ${rec.strike:.0f}</div>
              <div style="color:#8b949e;font-size:10px;margin-bottom:8px;">
                +{rec.dist_from_low:.0f} pts above low</div>
              <div style="font-size:11px;color:#8b949e;margin-bottom:3px;">
                Entry: <strong style="color:#e6edf3;font-size:14px;">${rec.est_entry_price:.2f}</strong></div>
              <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">
                Target: <strong style="color:#3fb950;font-size:14px;">${rec.est_target_price:.2f}</strong></div>
              <div style="font-size:22px;font-weight:800;color:{gain_c};">
                +{rec.est_gain_pct:,.0f}%</div>
              <div style="font-size:10px;color:#8b949e;margin-top:4px;">{rec.risk_category.split("—")[0].strip()}</div>
            </div>"""

        reentry_note = (
            f"Buy at intraday low (~${a_low:.2f}) · "
            f"Target open price ${a_open:.2f} (+{a_drop:.1f} pts recovery) · "
            f"Best strike: ${best_rec.strike:.0f} call (est. ${best_rec.est_entry_price:.2f} entry → "
            f"${best_rec.est_target_price:.2f} target)"
        )
        strike_section_html = f"""
        <div style="margin-top:16px;padding-top:14px;
                    border-top:1px solid rgba(255,255,255,0.08);">
          <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:10px;">Strike Recommendations — Buy at Low</div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;">{recs_html}</div>
          <div style="margin-top:10px;color:#8b949e;font-size:11px;
                      line-height:1.6;">{reentry_note}</div>
        </div>"""
    elif a_status == "SESSION_CLOSED":
        strike_section_html = f"""
        <div style="margin-top:12px;padding-top:10px;
                    border-top:1px solid rgba(255,255,255,0.06);
                    font-size:11px;color:#8b949e;line-height:1.7;">
          {"Session stats: Open $" + f"{a_open:.2f}" + " · Low $" + f"{a_low:.2f}" + " · Drop " + f"{a_drop:.1f}" + " pts · High $" + f"{a_high:.2f}" if a_open > 0 else "No intraday data available for this session."}
          The 0DTE window opens again at the next cash session open.
        </div>"""
    elif a_status == "APPROACHING":
        pts_needed  = max(0, 3.0 - a_drop)
        strike_level = round(a_low - pts_needed)  # approx strike if drop continues
        strike_section_html = f"""
        <div style="margin-top:14px;padding-top:12px;
                    border-top:1px solid rgba(255,255,255,0.07);
                    font-size:12px;color:#8b949e;line-height:1.7;">
          <strong style="color:#d29922;">Watching:</strong>
          If SPY drops {pts_needed:.1f} more pts to ~${a_low - pts_needed:.2f},
          setup activates. Sweet-spot calls would be around the
          <strong style="color:#e6edf3;">${round(a_open):.0f}–${round(a_open + 3):.0f} strikes</strong>
          (which will be +1 to +8 pts above the new low).
          Historical 1000%+ rate at a 3 pt drop: <strong style="color:#ffd633;">35%</strong>.
        </div>"""
    else:
        pts_to_setup = max(0, 3.0 - a_drop)
        strike_section_html = f"""
        <div style="margin-top:12px;padding-top:10px;
                    border-top:1px solid rgba(255,255,255,0.06);
                    font-size:11px;color:#8b949e;line-height:1.7;">
          Setup activates when SPY drops ≥3 pts from open
          (${a_open:.2f} − 3 = ${a_open - 3:.2f} trigger).
          Currently {a_drop:.1f} pts below open — need {pts_to_setup:.1f} more pts.
          Refreshes every 30 s automatically.
        </div>"""

    st.html(
        f"""<div style="background:{alert_bg};border:2px solid {alert_border};
                        border-radius:16px;padding:22px 26px;margin-bottom:4px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;
                      flex-wrap:wrap;gap:12px;">
            <div>
              <div style="font-size:14px;font-weight:800;color:{alert_border};
                          text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                {alert_title}</div>
              <div style="font-size:12px;color:#8b949e;">{alert_sub}</div>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;">
              <div style="text-align:center;">
                <div style="color:#8b949e;font-size:9px;text-transform:uppercase;">Open</div>
                <div style="font-size:16px;font-weight:800;color:#e6edf3;">${a_open:.2f}</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#8b949e;font-size:9px;text-transform:uppercase;">Low</div>
                <div style="font-size:16px;font-weight:800;color:#f85149;">${a_low:.2f}</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#8b949e;font-size:9px;text-transform:uppercase;">Drop</div>
                <div style="font-size:16px;font-weight:800;color:{alert_border};">
                  -{a_drop:.1f} pts</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#8b949e;font-size:9px;text-transform:uppercase;">VWAP</div>
                <div style="font-size:16px;font-weight:800;color:#58a6ff;">${a_vwap:.2f}</div>
              </div>
            </div>
          </div>
          {_0dte_alloc_html}
          {strike_section_html}
        </div>""")

    # ── Trade tickets ─────────────────────────────────────────────────────────
    from src.position_sizer import recommend_allocation, compound_projection, TIERS
    from src.report_jackpot_dashboard import trade_ticket

    trade_rows = [r for r in rows if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")]
    hot_rows   = [r for r in rows if r.signal == "GO_HOT"]
    all_signal_rows = trade_rows + hot_rows

    if trade_rows or hot_rows:
        st.markdown("---")
        section("Kelly Criterion Trade Plan", f"Position sizing for ${equity_input:,.0f} account — risk only what you allocate")

        # Warn user when the trading session is not currently open
        if not _ph["is_open"]:
            _session_note = (
                "Monday 9:30 AM ET" if _ph_p == "WEEKEND"
                else "9:50 AM ET after the open" if _ph_p in ("PRE_OPEN", "OPEN_PENDING_DATA")
                else "next session"
            )
            st.html(
                f"""<div style="background:#0a1428;border:1px solid #58a6ff;
                                border-radius:8px;padding:10px 16px;margin-bottom:12px;
                                font-size:12px;color:#8b949e;line-height:1.6;">
                  <strong style="color:#58a6ff;">ℹ️ Session not open.</strong>
                  These are <strong style="color:#e6edf3;">planning tickets</strong> for when the
                  market is live. Entry window opens <strong style="color:#a5d6ff;">{_session_note}</strong>.
                  Do not place 0DTE orders outside cash market hours (9:30–4:00 PM ET).
                </div>""")

        for r in all_signal_rows:
            alloc = recommend_allocation(r.signal, equity_input)
            if alloc is None:
                continue
            ticket   = trade_ticket(r, equity_input, alloc.alloc_pct)
            tier     = alloc.tier
            is_ultra = r.signal == "GO_ULTRA_JACKPOT"
            is_hot   = r.signal == "GO_HOT"
            border_c = "#ffd633" if is_ultra else "#3fb950" if not is_hot else "#d29922"
            card_bg  = (
                "linear-gradient(135deg,#1a1208,#2d2008)" if is_ultra
                else "linear-gradient(135deg,#1f1808,#2e2210)" if is_hot
                else "linear-gradient(135deg,#0d1f14,#12311e)"
            )

            # Kelly explanation bar  (quarter-kelly vs full-kelly)
            full_k_pct  = int(tier.full_kelly * 100)
            used_k_pct  = int(alloc.alloc_pct * 100)
            bar_fill_w  = min(int(used_k_pct / full_k_pct * 100), 100) if full_k_pct else 0

            # Win / lose scenario strings
            win_new_fmt  = f"${alloc.new_equity_win:,.2f}"
            lose_new_fmt = f"${alloc.new_equity_lose:,.2f}"

            ev_color = "#3fb950" if alloc.expected_gain > 0 else "#f85149"
            ev_label = f"+${alloc.expected_gain:,.2f}" if alloc.expected_gain >= 0 else f"-${abs(alloc.expected_gain):,.2f}"

            # Quick growth milestone line
            from src.position_sizer import trades_to_milestone
            next_milestone = next(
                (m for m in [5_000, 50_000, 500_000] if m > equity_input),
                None,
            )
            milestone_html = ""
            if next_milestone:
                n_to_ms = trades_to_milestone(r.signal, equity_input, next_milestone)
                if n_to_ms is not None:
                    milestone_html = (
                        f'<div style="margin-top:10px;color:#8b949e;font-size:11px;">'
                        f'📈 Expected path to <strong style="color:#ffd633;">'
                        f'${next_milestone:,.0f}</strong>: '
                        f'<strong style="color:#e6edf3;">{n_to_ms} wins</strong> '
                        f'at this allocation</div>'
                    )

            st.html(
                f"""
                <div style="background:{card_bg};border:2px solid {border_c};
                            border-radius:14px;padding:22px 26px;margin-bottom:16px;
                            box-shadow:0 0 24px {border_c}22;">

                  <!-- Header row -->
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
                    <div>
                      <div style="color:{border_c};font-size:13px;font-weight:800;
                                  text-transform:uppercase;letter-spacing:.07em;">
                        {tier.label} — {r.ticker}</div>
                      <div style="color:#8b949e;font-size:11px;margin-top:3px;
                                  line-height:1.5;">{tier.rationale}</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">Win Probability</div>
                      <div style="color:{border_c};font-size:28px;font-weight:900;
                                  line-height:1;">{int(tier.win_prob*100)}%</div>
                      <div style="color:#8b949e;font-size:10px;">{int(tier.avg_win_mult*100-100)}% avg win</div>
                    </div>
                  </div>

                  <!-- Kelly allocation bar -->
                  <div style="background:rgba(0,0,0,0.3);border-radius:8px;
                              padding:12px 16px;margin-bottom:16px;">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:6px;">
                      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                  letter-spacing:.07em;">Kelly Allocation
                        <span style="color:#30363d;margin:0 4px;">·</span>
                        <span style="color:#8b949e;">Full Kelly = {full_k_pct}% · Using {used_k_pct}%</span>
                      </div>
                      <div style="color:{border_c};font-size:22px;font-weight:900;
                                  font-variant-numeric:tabular-nums;">
                        ${alloc.alloc_dollars:,.2f}
                        <span style="font-size:12px;color:#8b949e;">({used_k_pct}%)</span>
                      </div>
                    </div>
                    <div style="height:8px;background:rgba(139,148,158,0.15);
                                border-radius:4px;overflow:hidden;">
                      <div style="width:{bar_fill_w}%;height:100%;
                                  background:{border_c};border-radius:4px;
                                  box-shadow:0 0 8px {border_c}88;"></div>
                    </div>
                  </div>

                  <!-- 4-cell stats grid -->
                  <div style="display:grid;grid-template-columns:repeat(4,1fr);
                              gap:12px;margin-bottom:14px;">
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;
                                padding:12px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.06em;margin-bottom:4px;">Spend</div>
                      <div style="color:#fff;font-size:20px;font-weight:800;">
                        ${alloc.alloc_dollars:,.0f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;
                                padding:12px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.06em;margin-bottom:4px;">If Avg Win</div>
                      <div style="color:#3fb950;font-size:20px;font-weight:800;">
                        +${alloc.win_scenario:,.0f}</div>
                      <div style="color:#8b949e;font-size:9px;">→ {win_new_fmt}</div>
                    </div>
                    <div style="background:rgba(248,81,73,0.08);border-radius:8px;
                                padding:12px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.06em;margin-bottom:4px;">Max Loss</div>
                      <div style="color:#f85149;font-size:20px;font-weight:800;">
                        -${alloc.max_loss:,.0f}</div>
                      <div style="color:#8b949e;font-size:9px;">→ {lose_new_fmt}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;
                                padding:12px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.06em;margin-bottom:4px;">EV per Trade</div>
                      <div style="color:{ev_color};font-size:20px;font-weight:800;">
                        {ev_label}</div>
                      <div style="color:#8b949e;font-size:9px;">
                        ${alloc.ev_per_dollar:.1f} per $1 risked</div>
                    </div>
                  </div>

                  <!-- Strike / contract info from old ticket -->
                  <div style="display:flex;gap:12px;flex-wrap:wrap;
                              padding-top:12px;border-top:1px solid rgba(255,255,255,0.07);">
                    <div style="color:#8b949e;font-size:11px;">
                      Suggested strike: <strong style="color:#e6edf3;">{fmt_dollar(ticket["strike"])}</strong>
                    </div>
                    <div style="color:#30363d;">·</div>
                    <div style="color:#8b949e;font-size:11px;">
                      Contracts: <strong style="color:#e6edf3;">{ticket["n_contracts"]}</strong>
                    </div>
                    <div style="color:#30363d;">·</div>
                    <div style="color:#8b949e;font-size:11px;">
                      Premium/contract: <strong style="color:#e6edf3;">
                        ${ticket["premium_per_contract"]:.2f}</strong>
                    </div>
                  </div>
                  {milestone_html}
                </div>
                """)
    else:
        # ── SKIP state — rich context card ─────────────────────────────────
        from src.jackpot_scanner import HOT_THRESHOLD, JACKPOT_THRESHOLD
        st.markdown("---")
        section("Model Intelligence", "Why SKIP — and what to watch for")

        # Per-ticker p_vol gauge bars
        gauge_html = ""
        for r in rows:
            pv   = r.p_vol
            pp   = r.p_pnl
            pv_w = min(int(pv / HOT_THRESHOLD * 100), 100)
            pp_w = min(int(pp / JACKPOT_THRESHOLD * 100), 100)
            pv_c = "#3fb950" if pv >= HOT_THRESHOLD else "#ffd633" if pv >= HOT_THRESHOLD * 0.7 else "#8b949e"
            pp_c = "#3fb950" if pp >= JACKPOT_THRESHOLD else "#ffd633" if pp >= JACKPOT_THRESHOLD * 0.7 else "#8b949e"
            pv_pct_of_thresh = int(pv / HOT_THRESHOLD * 100)
            pp_pct_of_thresh = int(pp / JACKPOT_THRESHOLD * 100)
            snap_r = live_q.get(r.ticker, {})
            price_r = snap_r.get("last_price") or r.last_close
            chg_r   = snap_r.get("change_pct") or 0.0
            chg_c_r = "#3fb950" if chg_r >= 0 else "#f85149"
            gauge_html += f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                        padding:16px 18px;">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:12px;">
                <span style="font-size:16px;font-weight:800;color:#e6edf3;">{r.ticker}</span>
                <span style="font-size:14px;font-weight:700;color:{chg_c_r};">
                  ${price_r:,.2f}
                  <span style="font-size:11px;">{chg_r*100:+.2f}%</span></span>
              </div>
              <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                  <span style="color:#8b949e;font-size:10px;text-transform:uppercase;
                               letter-spacing:.06em;">P(vol) — need ≥{HOT_THRESHOLD*100:.0f}%</span>
                  <span style="font-size:11px;font-weight:800;color:{pv_c};">
                    {pv*100:.1f}% ({pv_pct_of_thresh}% of threshold)</span>
                </div>
                <div style="height:6px;background:rgba(139,148,158,0.15);border-radius:3px;
                            overflow:hidden;">
                  <div style="width:{pv_w}%;height:100%;background:{pv_c};border-radius:3px;">
                  </div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                  <span style="color:#8b949e;font-size:10px;text-transform:uppercase;
                               letter-spacing:.06em;">P(pnl) — need ≥{JACKPOT_THRESHOLD*100:.0f}%</span>
                  <span style="font-size:11px;font-weight:800;color:{pp_c};">
                    {pp*100:.1f}% ({pp_pct_of_thresh}% of threshold)</span>
                </div>
                <div style="height:6px;background:rgba(139,148,158,0.15);border-radius:3px;
                            overflow:hidden;">
                  <div style="width:{pp_w}%;height:100%;background:{pp_c};border-radius:3px;">
                  </div>
                </div>
              </div>
            </div>"""

        st.html(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">'
            f'{gauge_html}</div>')

        # ── What would trigger ─────────────────────────────────────────────
        closest = max(rows, key=lambda r: r.p_vol / HOT_THRESHOLD)
        gap_to_hot  = max(HOT_THRESHOLD - closest.p_vol, 0.0)
        gap_to_jack = max(JACKPOT_THRESHOLD - closest.p_pnl, 0.0)
        st.html(
            f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;
                            padding:18px 24px;margin-top:12px;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:10px;">What would flip to GO_HOT</div>
              <div style="display:flex;gap:24px;flex-wrap:wrap;">
                <div>
                  <div style="color:#8b949e;font-size:11px;">Closest ticker</div>
                  <div style="font-size:20px;font-weight:800;color:#fff;">{closest.ticker}</div>
                </div>
                <div>
                  <div style="color:#8b949e;font-size:11px;">P(vol) gap to trigger</div>
                  <div style="font-size:20px;font-weight:800;color:#ffd633;">
                    +{gap_to_hot*100:.1f}% needed</div>
                </div>
                <div>
                  <div style="color:#8b949e;font-size:11px;">P(pnl) gap to trigger</div>
                  <div style="font-size:20px;font-weight:800;color:#a5d6ff;">
                    +{gap_to_jack*100:.1f}% needed</div>
                </div>
                <div>
                  <div style="color:#8b949e;font-size:11px;">Thresholds</div>
                  <div style="font-size:13px;font-weight:700;color:#8b949e;">
                    P(vol) ≥ {HOT_THRESHOLD*100:.0f}% <span style="color:#30363d;">·</span>
                    P(pnl) ≥ {JACKPOT_THRESHOLD*100:.0f}%</div>
                </div>
              </div>
            </div>""")

        # ── Today's intraday opportunity (only meaningful during OPEN_LIVE) ──
        live_spy = live_q.get("SPY", {})
        spy_open = live_spy.get("day_open", 0.0)
        spy_low  = live_spy.get("day_low",  0.0)
        spy_high = live_spy.get("day_high", 0.0)
        spy_drop = spy_open - spy_low if spy_open > 0 else 0.0

        if spy_drop >= 2.0 and _ph["is_open"]:
            opp_color = "#ffd633" if spy_drop >= 4.0 else "#d29922"
            opp_bg    = "#1c1a0a" if spy_drop >= 4.0 else "#1a1208"
            st.html(
                f"""<div style="background:{opp_bg};border:2px solid {opp_color};
                                border-radius:12px;padding:18px 24px;margin-top:12px;">
                  <div style="font-size:13px;font-weight:800;color:{opp_color};
                              text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">
                    🎰 Intraday Opportunity Active — ML is SKIP but market is moving</div>
                  <div style="color:#c9d1d9;font-size:13px;line-height:1.7;">
                    SPY has dropped <strong style="color:#fff;">{spy_drop:.1f} pts</strong>
                    from open (${spy_open:.2f}) to low (${spy_low:.2f}).
                    {"<strong style='color:#ffd633;'>This is a potential 1000%+ 0DTE reversal setup.</strong>" if spy_drop >= 4.0 else ""}
                    Calls near the open strike are cheap — check the 0DTE Lottery and Reversal Levels pages
                    for entry levels, sweet-spot strikes, and historical bounce rates.
                  </div>
                  <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;">
                    <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                                padding:10px 16px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;">Drop from open</div>
                      <div style="font-size:18px;font-weight:800;color:#f85149;">
                        -{spy_drop:.1f} pts</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                                padding:10px 16px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;">SPY Low</div>
                      <div style="font-size:18px;font-weight:800;color:#e6edf3;">
                        ${spy_low:.2f}</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                                padding:10px 16px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;">Recovery target</div>
                      <div style="font-size:18px;font-weight:800;color:#3fb950;">
                        ${spy_open:.2f} (+{spy_drop:.1f} pts)</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                                padding:10px 16px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;">Check pages</div>
                      <div style="font-size:13px;font-weight:800;color:#ffd633;">
                        0DTE Lottery · Reversal Levels</div>
                    </div>
                  </div>
                </div>""")
        elif _ph["is_open"]:
            # Only show "quiet" message during live session, not after hours
            st.html(
                f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                                padding:14px 20px;margin-top:12px;color:#8b949e;font-size:12px;
                                line-height:1.7;">
                  <strong style="color:#e6edf3;">No intraday setup active.</strong>
                  SPY drop from open: {spy_drop:.1f} pts (need ≥2 pts for a reversal setup).
                  Use the <strong style="color:#ffd633;">0DTE Lottery</strong> and
                  <strong style="color:#ffd633;">Reversal Levels</strong> pages to monitor
                  live intraday action. Signals refresh every 5 minutes — use the
                  🔄 button in the sidebar to force a refresh.
                </div>""")

    # ── Compound Growth Projector ──────────────────────────────────────────────
    st.markdown("---")
    section("Compound Growth Projector",
            f"What happens if you trade every signal — ${equity_input:,.0f} starting balance")

    from src.position_sizer import compound_projection, TIERS, recommend_allocation as _ra

    # Determine which signal to project (best active signal today)
    _proj_signal = sig  # best ML signal (GO_ULTRA_JACKPOT / GO_JACKPOT / GO_HOT / SKIP)
    if _proj_signal == "SKIP" and a_status == "ENTRY_OPEN":
        _proj_signal = "ENTRY_OPEN"

    _proj_tier = TIERS.get(_proj_signal)
    if _proj_tier:
        _proj_steps = compound_projection(_proj_signal, equity_input, n_trades=8)
        _proj_alloc  = _ra(_proj_signal, equity_input)

        if _proj_steps and _proj_alloc:
            _pa_pct = int(_proj_alloc.alloc_pct * 100)
            _pa_wp  = int(_proj_tier.win_prob * 100)
            _pa_wm  = int(_proj_tier.avg_win_mult * 100 - 100)

            # Build the table rows
            rows_html = ""
            for step in _proj_steps:
                if step.trade_num == 0:
                    rows_html += (
                        f'<tr>'
                        f'<td style="color:#8b949e;text-align:center;">Start</td>'
                        f'<td style="text-align:right;font-weight:700;color:#e6edf3;">'
                        f'${step.equity_expected:,.0f}</td>'
                        f'<td style="text-align:right;color:#3fb950;">—</td>'
                        f'<td style="text-align:right;color:#f85149;">—</td>'
                        f'</tr>'
                    )
                    continue

                # Color the expected column based on vs starting equity
                gain_vs_start = step.equity_expected - equity_input
                exp_color = "#3fb950" if step.equity_expected > equity_input else "#f85149"

                # Check milestones
                prev_exp = _proj_steps[step.trade_num - 1].equity_expected
                ms_tag = ""
                for ms in [5_000, 50_000, 500_000]:
                    if prev_exp < ms <= step.equity_expected:
                        ms_label = f"${'5k' if ms==5_000 else '50k' if ms==50_000 else '500k'} ✓"
                        ms_tag = (
                            f'<span style="background:#ffd633;color:#0d1117;'
                            f'font-size:8px;font-weight:800;padding:1px 5px;'
                            f'border-radius:3px;margin-left:4px;">{ms_label}</span>'
                        )
                        break

                rows_html += (
                    f'<tr>'
                    f'<td style="color:#8b949e;text-align:center;">{step.trade_num}</td>'
                    f'<td style="text-align:right;font-weight:800;color:{exp_color};">'
                    f'${step.equity_expected:,.0f}{ms_tag}</td>'
                    f'<td style="text-align:right;color:#3fb950;">'
                    f'${step.equity_win_all:,.0f}</td>'
                    f'<td style="text-align:right;color:#f85149;">'
                    f'${max(step.equity_lose_all, 0):,.0f}</td>'
                    f'</tr>'
                )

            st.html(
                f"""<div style="background:#0d1117;border:1px solid #30363d;
                                border-radius:14px;padding:22px 26px;margin-bottom:16px;">

                  <!-- Summary row -->
                  <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:18px;">
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:10px 18px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">Signal</div>
                      <div style="color:#ffd633;font-size:15px;font-weight:800;">
                        {_proj_signal.replace("_"," ")}</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:10px 18px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">Allocation / trade</div>
                      <div style="color:#fff;font-size:15px;font-weight:800;">
                        {_pa_pct}% · ${_proj_alloc.alloc_dollars:,.0f}</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:10px 18px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">Win Rate (hist)</div>
                      <div style="color:#3fb950;font-size:15px;font-weight:800;">{_pa_wp}%</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:10px 18px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">Avg Win (hist)</div>
                      <div style="color:#3fb950;font-size:15px;font-weight:800;">+{_pa_wm}%</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:10px 18px;text-align:center;">
                      <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                                  letter-spacing:.07em;">EV / trade</div>
                      <div style="color:#58a6ff;font-size:15px;font-weight:800;">
                        +${_proj_alloc.expected_gain:,.0f}</div>
                    </div>
                  </div>

                  <!-- Growth table -->
                  <table style="width:100%;border-collapse:collapse;font-size:12px;
                                font-variant-numeric:tabular-nums;">
                    <thead>
                      <tr style="border-bottom:1px solid #30363d;">
                        <th style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                   letter-spacing:.06em;padding:6px 0;text-align:center;
                                   font-weight:700;">Trade #</th>
                        <th style="color:#58a6ff;font-size:10px;text-transform:uppercase;
                                   letter-spacing:.06em;padding:6px 0;text-align:right;
                                   font-weight:700;">Expected Equity</th>
                        <th style="color:#3fb950;font-size:10px;text-transform:uppercase;
                                   letter-spacing:.06em;padding:6px 0;text-align:right;
                                   font-weight:700;">All Wins Path</th>
                        <th style="color:#f85149;font-size:10px;text-transform:uppercase;
                                   letter-spacing:.06em;padding:6px 0;text-align:right;
                                   font-weight:700;">All Losses Path</th>
                      </tr>
                    </thead>
                    <tbody style="line-height:2;">
                      {rows_html}
                    </tbody>
                  </table>

                  <div style="margin-top:14px;padding-top:12px;
                              border-top:1px solid rgba(255,255,255,0.06);
                              color:#8b949e;font-size:11px;line-height:1.7;">
                    <strong style="color:#e6edf3;">How to read this:</strong>
                    Expected equity applies win probability to each trade (the realistic path).
                    "All Wins" shows max compounding power.
                    "All Losses" shows Kelly's downside protection — you keep {100 - _pa_pct}% of equity even on every loss.
                    <strong style="color:#ffd633;">Risk is capped at your allocation — options can only go to $0.</strong>
                  </div>
                </div>""")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: MULTI-TICKER SCANNER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Scanner":
    DEFAULT_UNIVERSE = [
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "MSFT", "NVDA", "GOOGL",
        "AMZN", "META", "TSLA", "AMD",
    ]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Universe**")
        ticker_input = st.text_area(
            "Tickers (one per line)",
            value="\n".join(DEFAULT_UNIVERSE),
            height=200,
        )
        user_tickers = [
            t.strip().upper()
            for t in ticker_input.replace(",", "\n").split("\n")
            if t.strip()
        ]

    # ── Scanner market phase banner ───────────────────────────────────────────
    from src.jackpot_scanner import market_phase as _sc_market_phase
    _sc_ph = _sc_market_phase()
    _sc_ph_p = _sc_ph["phase"]
    _sc_ph_cfgs = {
        "PRE_OPEN":          ("#d29922", "#1c1600", "🟡 PRE-MARKET",
                              f"Signals are based on yesterday's close. "
                              f"Opening gap feature locks at 9:30 AM ET — "
                              f"<strong style='color:#ffd633;'>confirm at 9:50 AM before acting.</strong>"),
        "OPEN_PENDING_DATA": ("#d29922", "#1c1600", "🟡 OPEN · DATA SETTLING",
                              "Market opened but data hasn't settled yet (~9:50 AM ET). "
                              "<strong style='color:#ffd633;'>Scores may still reflect yesterday — refresh shortly.</strong>"),
        "OPEN_LIVE":         ("#3fb950", "#0d1f14", "🟢 MARKET OPEN · LIVE",
                              f"Scores are final for today. "
                              f"{_sc_ph.get('minutes_since_open') or 0} min into session · closes 4:00 PM ET."),
        "AFTER_HOURS":       ("#58a6ff", "#0a1428", "🔵 AFTER-HOURS · SESSION CLOSED",
                              "Today's session ended at 4:00 PM ET. "
                              "Scores below are today's final values — use for tomorrow's planning."),
        "WEEKEND":           ("#8b949e", "#161b22", "⚫ WEEKEND · CLOSED",
                              "Markets reopen Monday 9:30 AM ET. "
                              "Scores are a preview for Monday based on Friday's close."),
    }
    _sc_c, _sc_bg, _sc_lbl, _sc_sub = _sc_ph_cfgs.get(_sc_ph_p, _sc_ph_cfgs["AFTER_HOURS"])
    st.html(
        f"""<div style="background:{_sc_bg};border:1px solid {_sc_c};
                        border-radius:10px;padding:11px 16px;margin-bottom:12px;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;
                        display:flex;align-items:flex-start;gap:10px;">
          <div style="flex:1;">
            <div style="font-size:11px;font-weight:800;color:{_sc_c};
                        text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px;">{_sc_lbl}</div>
            <div style="font-size:12px;color:#8b949e;line-height:1.5;">{_sc_sub}</div>
          </div>
          <div style="color:#30363d;font-size:10px;white-space:nowrap;padding-top:2px;">
            {datetime.now().strftime('%H:%M ET')}</div>
        </div>""")

    with st.spinner(f"Scanning {len(user_tickers)} tickers…"):
        try:
            df, errors = load_scanner(tuple(user_tickers))
        except Exception as e:
            st.error(f"Scanner error: {e}")
            df, errors = pd.DataFrame(), [str(e)]

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker(s) failed"):
            for err in errors:
                st.text(err if isinstance(err, str) else f"{err.get('ticker','?')}: {err.get('error','unknown')}")

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        st.warning("No results. Try refreshing or check your ticker list.")
        st.stop()

    def lift_label(lift):
        if pd.isna(lift): return ("—", "#8b949e")
        if lift >= 3.0:   return ("EXTREME", "#f85149")
        if lift >= 2.0:   return ("HIGH",    "#d29922")
        if lift >= 1.3:   return ("ELEVATED","#ffd633")
        if lift >= 0.8:   return ("NORMAL",  "#58a6ff")
        return ("CALM", "#8b949e")

    df_s = df.copy()
    if "p_vol" in df_s.columns:
        df_s = df_s.sort_values("p_vol", ascending=False).reset_index(drop=True)

    # ── Live quotes for scanner universe ─────────────────────────────────────
    scanner_tickers = tuple(df_s["ticker"].tolist()) if "ticker" in df_s.columns else ()
    scanner_live = load_live_quotes(scanner_tickers) if scanner_tickers else {}

    # ── Top picks (top 3) ─────────────────────────────────────────────────────
    section("Top Picks Today", "Ranked by P(volatile day) — highest conviction first")

    top3 = df_s.head(3)
    cols = st.columns(len(top3))
    for i, (_, row) in enumerate(top3.iterrows()):
        tkr    = row.get("ticker", "?")
        p_vol  = row.get("p_vol",  float("nan"))
        lift   = row.get("lift",   float("nan"))
        rsi    = row.get("rsi14",  float("nan"))
        snap   = scanner_live.get(tkr, {})

        # Prefer live Polygon price/change, fall back to model data
        live_price = snap.get("last_price") or row.get("last_close", float("nan"))
        live_chg   = snap.get("change_pct")
        if live_chg is None:
            live_chg = row.get("pct_change", float("nan"))
        day_high   = snap.get("day_high", 0.0)
        day_low    = snap.get("day_low",  0.0)
        day_vwap   = snap.get("day_vwap", 0.0)
        # Only show LIVE badge when cash market is actually open
        s_label    = snap.get("status_label", "") if _sc_ph["is_open"] else ""

        lift_txt, lift_c = lift_label(lift)
        chg_c  = "#3fb950" if (live_chg is not None and not math.isnan(float(live_chg)) and float(live_chg) >= 0) else "#f85149"
        pvol_c = "#ffd633" if (not math.isnan(p_vol) and p_vol >= 0.65) else (
                  "#3fb950" if (not math.isnan(p_vol) and p_vol >= 0.55) else "#58a6ff")
        rank_badge = ["#1", "#2", "#3"][i]
        live_badge = (
            f'<span style="background:#0d1f14;color:#3fb950;font-size:8px;font-weight:800;'
            f'padding:2px 5px;border-radius:3px;letter-spacing:.05em;margin-left:6px;">{s_label}</span>'
            if s_label else ""
        )
        hl_html = (
            f'<div><div style="color:#8b949e;font-size:11px;text-transform:uppercase;">H / L</div>'
            f'<div style="color:#e6edf3;font-weight:700;font-size:11px;">{fmt_dollar(day_high)} / {fmt_dollar(day_low)}</div></div>'
            if day_high else ""
        )
        vwap_html = (
            f'<div><div style="color:#8b949e;font-size:11px;text-transform:uppercase;">VWAP</div>'
            f'<div style="color:#58a6ff;font-weight:700;">{fmt_dollar(day_vwap)}</div></div>'
            if day_vwap else ""
        )

        with cols[i]:
            st.html(
                f"""
                <div style="background:#161b22;border:2px solid {pvol_c};
                            border-radius:14px;padding:20px 18px;
                            box-shadow:0 0 16px {pvol_c}33;">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:6px;">
                    <span style="font-size:22px;font-weight:800;color:#fff;">{tkr}</span>
                    <span style="color:#8b949e;font-size:13px;font-weight:700;">{rank_badge}</span>
                  </div>
                  <div style="display:flex;align-items:baseline;gap:4px;margin-bottom:2px;">
                    <span style="font-size:24px;font-weight:800;color:#fff;">{fmt_dollar(live_price)}</span>
                    {live_badge}
                  </div>
                  <div style="font-size:14px;font-weight:700;color:{chg_c};margin-bottom:12px;">
                    {f"{float(live_chg)*100:+.2f}%" if live_chg is not None and not math.isnan(float(live_chg)) else "—"}
                  </div>
                  <div style="font-size:36px;font-weight:800;color:{pvol_c};
                              line-height:1;margin-bottom:4px;">{fmt_pct(p_vol)}</div>
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                              letter-spacing:.07em;margin-bottom:12px;">P(volatile day)</div>
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:12px;">
                    <span style="background:{lift_c}22;color:{lift_c};font-size:10px;
                                 font-weight:800;padding:3px 8px;border-radius:4px;
                                 letter-spacing:.05em;">{lift_txt}</span>
                    <span style="color:#8b949e;font-size:11px;">Lift {lift:.2f}x</span>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;">
                    {hl_html}
                    {vwap_html}
                    <div>
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;">RSI(14)</div>
                      <div style="color:#e6edf3;font-weight:700;">
                        {f"{rsi:.0f}" if not math.isnan(rsi) else "—"}</div>
                    </div>
                  </div>
                </div>
                """)

    # ── Full ranked list ──────────────────────────────────────────────────────
    st.markdown("---")
    section("Full Ranked List", "All tickers sorted by P(volatile day) — higher = more expected movement")

    tbl_rows_html = ""
    for rank, (_, row) in enumerate(df_s.iterrows(), 1):
        p_vol  = row.get("p_vol",  float("nan"))
        lift   = row.get("lift",   float("nan"))
        close  = row.get("last_close", float("nan"))
        chg    = row.get("pct_change", float("nan"))
        rsi    = row.get("rsi14",  float("nan"))
        tkr    = row.get("ticker", "?")
        lift_txt, lift_c = lift_label(lift)
        pvol_c = ("#ffd633" if (not math.isnan(p_vol) and p_vol >= 0.65)
                  else "#3fb950" if (not math.isnan(p_vol) and p_vol >= 0.30)
                  else "#8b949e")
        chg_c  = "#3fb950" if (not math.isnan(chg) and chg >= 0) else "#f85149"
        bar_w  = max(3, min(int(p_vol * 200), 100)) if not math.isnan(p_vol) else 3
        snap_r = scanner_live.get(tkr, {})
        live_px = snap_r.get("last_price") or close
        live_chg = snap_r.get("change_pct")
        if live_chg is not None and not math.isnan(float(live_chg)):
            chg_str = f"{float(live_chg)*100:+.2f}%"
            chg_c   = "#3fb950" if float(live_chg) >= 0 else "#f85149"
        else:
            chg_str = f"{chg*100:+.2f}%" if not math.isnan(chg) else "—"
        tbl_rows_html += f"""
        <tr>
          <td style="padding:9px 14px;color:#8b949e;font-size:11px;">{rank}</td>
          <td style="padding:9px 14px;font-size:14px;font-weight:800;color:#e6edf3;">{tkr}</td>
          <td style="padding:9px 14px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="background:{pvol_c};height:6px;width:{bar_w}px;
                          border-radius:3px;min-width:3px;"></div>
              <span style="font-size:13px;font-weight:800;color:{pvol_c};">
                {fmt_pct(p_vol)}</span>
            </div>
          </td>
          <td style="padding:9px 14px;font-size:12px;color:#e6edf3;">
            {f"{lift:.2f}x" if not math.isnan(lift) else "—"}</td>
          <td style="padding:9px 14px;">
            <span style="background:{lift_c}22;color:{lift_c};font-size:10px;font-weight:800;
                         padding:2px 7px;border-radius:4px;">{lift_txt}</span></td>
          <td style="padding:9px 14px;font-size:13px;font-weight:700;color:#e6edf3;">
            {fmt_dollar(live_px)}</td>
          <td style="padding:9px 14px;font-size:12px;font-weight:700;color:{chg_c};">
            {chg_str}</td>
          <td style="padding:9px 14px;font-size:12px;color:#8b949e;">
            {f"{rsi:.0f}" if not math.isnan(rsi) else "—"}</td>
        </tr>"""

    st.html(
        f"""<div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                      background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
          <thead>
            <tr style="background:#161b22;border-bottom:1px solid #30363d;">
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">#</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Ticker</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">P(vol)</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Lift</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Verdict</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Price</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Chg %</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">RSI(14)</th>
            </tr>
          </thead>
          <tbody>{tbl_rows_html}</tbody>
        </table></div>""")
    st.caption("Lift = P(vol) ÷ base-rate   ·   Extreme >3.0  ·  High >2.0  ·  Elevated >1.3")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: GAP REVERSAL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Gap Reversal":
    GAP_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Settings**")
        gap_tickers = st.multiselect(
            "Tickers to scan",
            options=GAP_UNIVERSE + ["META", "AMZN", "GOOGL"],
            default=["SPY", "QQQ", "IWM", "AAPL", "NVDA"],
        )
        lookback_yrs = st.selectbox("History lookback", [3, 5, 10], index=1,
                                    format_func=lambda v: f"{v} years")
        detail_ticker = st.selectbox(
            "Deep-dive ticker", gap_tickers if gap_tickers else ["SPY"], index=0
        )

    def gap_signal_color(sig: str) -> str:
        return {
            "WATCH_FILL": "#3fb950",
            "NEAR_FILL":  "#d29922",
            "MONITOR":    "#58a6ff",
            "NO_GAP":     "#8b949e",
            "SMALL_GAP":  "#6e7681",
        }.get(sig, "#8b949e")

    def fill_progress_bar(open_price: float, fill_level: float, current: float | None) -> str:
        """Visual bar from open toward fill level."""
        try:
            if current is None or open_price == fill_level:
                pct = 0.0
            else:
                dist_total = abs(fill_level - open_price)
                dist_done  = abs(current - open_price)
                pct = min(dist_done / dist_total * 100, 100)
        except Exception:
            pct = 0.0
        bar_c = "#3fb950" if pct >= 80 else "#d29922" if pct >= 40 else "#58a6ff"
        return (
            f'<div style="height:6px;background:rgba(139,148,158,0.15);'
            f'border-radius:3px;overflow:hidden;margin:6px 0 2px;">'
            f'  <div style="width:{pct:.0f}%;height:100%;background:{bar_c};'
            f'border-radius:3px;transition:width .3s;"></div></div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:11px;color:#8b949e;">'
            f'<span>Open ${open_price:.2f}</span>'
            f'<span style="color:{bar_c};font-weight:700;">{pct:.0f}% to fill</span>'
            f'<span>Fill ${fill_level:.2f}</span></div>'
        )

    # ── Load tickers ──────────────────────────────────────────────────────────
    if not gap_tickers:
        st.warning("Select at least one ticker in the sidebar.")
        st.stop()

    today_rows = []
    load_errors = []
    for tkr in gap_tickers:
        with st.spinner(f"Loading {tkr}…"):
            try:
                _, sb, _, _, tod = load_gap_analysis(tkr, lookback_yrs)
                today_rows.append((tkr, tod, sb))
            except Exception as exc:
                load_errors.append(f"{tkr}: {exc}")

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} error(s)"):
            for e in load_errors:
                st.text(e)

    MIN_GAP_PCT = 0.01   # require ≥1% gap to show a setup
    gap_today = [
        (tkr, tod, sb) for tkr, tod, sb in today_rows
        if tod.gap_dir in ("up", "down") and abs(tod.gap_pct) >= MIN_GAP_PCT
    ]
    no_gap    = [(tkr, tod, sb) for tkr, tod, sb in today_rows if tod.gap_dir not in ("up", "down")]

    # Live quotes for fill progress bars (TTL=30 s)
    gap_live = load_live_quotes(tuple(gap_tickers))

    # ── Trade logic helpers ────────────────────────────────────────────────────
    def gap_trade_action(gap_dir: str, phase: int = 1) -> tuple[str, str]:
        """Returns (action_label, action_color) for a gap trade.
        phase=1 → ride the fill; phase=2 → post-fill reversal flip.
        """
        if phase == 1:
            if gap_dir == "up":
                return "BUY PUTS", "#f85149"    # gap up → sell off → puts
            else:
                return "BUY CALLS", "#3fb950"   # gap down → rally to fill → calls
        else:
            if gap_dir == "up":
                return "FLIP → CALLS", "#3fb950"
            else:
                return "FLIP → PUTS", "#f85149"

    def fill_status(pct: float) -> tuple[str, str]:
        """Returns (status_label, color) based on fill progress %."""
        if pct >= 90:
            return "FILLED — EXIT NOW", "#ffd633"
        if pct >= 70:
            return "NEAR FILL — HOLD", "#3fb950"
        if pct >= 40:
            return "FILLING — In Motion", "#d29922"
        if pct >= 10:
            return "In Progress", "#58a6ff"
        return "Approaching", "#8b949e"

    def calc_fill_pct(open_price: float, fill_level: float, current: float | None) -> float:
        if current is None or open_price == fill_level:
            return 0.0
        try:
            dist_total = abs(fill_level - open_price)
            dist_done  = abs(current - open_price)
            return min(dist_done / dist_total * 100, 100)
        except Exception:
            return 0.0

    # ── Alert header ──────────────────────────────────────────────────────────
    n_gaps  = len(gap_today)
    n_watch = sum(1 for _, tod, _ in gap_today if tod.signal == "WATCH_FILL")
    n_near  = sum(1 for _, tod, _ in gap_today if tod.signal == "NEAR_FILL")
    actionable = [
        (tkr, tod, sb) for tkr, tod, sb in gap_today
        if tod.signal in ("WATCH_FILL", "NEAR_FILL")
    ]

    if n_gaps == 0:
        st.html(
            f"""<div style="background:#161b22;border:1px solid #30363d;
                            border-radius:12px;padding:20px 24px;margin-bottom:20px;
                            text-align:center;">
              <div style="font-size:15px;font-weight:700;color:#8b949e;">
                No Gaps Today — {datetime.now().strftime('%A %b %d')}</div>
              <div style="font-size:13px;color:#6e7681;margin-top:6px;">
                All {len(today_rows)} tickers opened near yesterday's close. Nothing to trade. Stand by.</div>
            </div>""")
    else:
        alert_c = "#3fb950" if n_watch > 0 else "#d29922" if n_near > 0 else "#58a6ff"
        alert_bg = (
            "linear-gradient(135deg,#0d1f14,#0a1a0f)" if n_watch > 0
            else "linear-gradient(135deg,#1a1208,#131008)"
            if n_near > 0 else "linear-gradient(135deg,#0a1428,#0d1f36)"
        )
        alert_title = (
            f"TRADE ALERT — {n_watch} High-Probability Gap{'s' if n_watch>1 else ''}" if n_watch > 0
            else f"MONITOR — {n_near} Gap Setup{'s' if n_near>1 else ''}" if n_near > 0
            else f"{n_gaps} Gap{'s' if n_gaps>1 else ''} Detected — Low Probability"
        )
        best_setups = [tkr for tkr, tod, _ in gap_today if tod.signal in ("WATCH_FILL","NEAR_FILL")]
        summary_line = (
            f"High-confidence setups active: <strong style='color:#fff;'>{', '.join(best_setups)}</strong>"
            if best_setups else "Gaps open but historical fill rate below threshold."
        )
        st.html(
            f"""<div style="background:{alert_bg};border:2px solid {alert_c};
                            border-radius:14px;padding:20px 28px;margin-bottom:24px;">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <div style="font-size:11px;font-weight:800;color:{alert_c};
                            text-transform:uppercase;letter-spacing:.1em;">
                  {datetime.now().strftime('%A %b %d · %H:%M')} ET</div>
              </div>
              <div style="font-size:22px;font-weight:800;color:{alert_c};
                          margin-bottom:8px;line-height:1.1;">{alert_title}</div>
              <div style="font-size:13px;color:#c9d1d9;">{summary_line}</div>
            </div>""")

    # ── Action trade cards ─────────────────────────────────────────────────────
    if gap_today:
        section("Trade Setups", f"{datetime.now().strftime('%A %b %d')} · Gap fill & reversal plays")
        n_cols = min(len(gap_today), 3)
        cols = st.columns(n_cols)

        for i, (tkr, tod, _) in enumerate(gap_today):
            is_up     = tod.gap_dir == "up"
            sig_c     = gap_signal_color(tod.signal)
            dir_c     = "#f85149" if is_up else "#3fb950"
            dir_str   = f"↑ GAP UP  {tod.gap_pct*100:+.2f}%" if is_up else f"↓ GAP DOWN  {tod.gap_pct*100:+.2f}%"
            action_lbl, action_c = gap_trade_action(tod.gap_dir, 1)

            # Live price data
            live_snap  = gap_live.get(tkr, {})
            live_cur   = live_snap.get("last_price") or None
            live_badge = (
                f'<span style="font-size:11px;font-weight:800;color:#3fb950;'
                f'background:#0d1f14;padding:2px 5px;border-radius:3px;'
                f'margin-left:6px;">LIVE</span>'
                if live_snap else ""
            )

            # Fill progress
            fp = calc_fill_pct(tod.open_price, tod.fill_level, live_cur)
            fp_status, fp_color = fill_status(fp)
            bar_c = fp_color

            # Remaining points to fill
            if live_cur is not None:
                pts_to_fill = abs(live_cur - tod.fill_level)
                entry_price = live_cur
            else:
                pts_to_fill = abs(tod.open_price - tod.fill_level)
                entry_price = tod.open_price

            atm_strike = round(entry_price)  # nearest $1 strike for 0DTE

            # Phase 2 (post-fill reversal) recommendation
            show_phase2 = (
                tod.hist_rev_rate is not None and tod.hist_rev_rate >= 0.55
                and tod.hist_med_rev_pts is not None
            )
            phase2_lbl, phase2_c = gap_trade_action(tod.gap_dir, 2)
            phase2_pts = abs(tod.hist_med_rev_pts) if tod.hist_med_rev_pts is not None else 0.0
            phase2_target = (
                tod.fill_level + phase2_pts if is_up      # gap-up fills → price may bounce up
                else tod.fill_level - phase2_pts           # gap-down fills → price may drop
            )

            # Confidence badge
            fill_r_str  = f"{tod.hist_fill_rate*100:.0f}%" if tod.hist_fill_rate is not None else "—"
            rev_r_str   = f"{tod.hist_rev_rate*100:.0f}%" if tod.hist_rev_rate is not None else "—"
            conf_str    = (
                "HIGH" if (tod.hist_fill_rate or 0) >= 0.70
                else "MED" if (tod.hist_fill_rate or 0) >= 0.50
                else "LOW"
            )
            conf_c = "#3fb950" if conf_str == "HIGH" else "#d29922" if conf_str == "MED" else "#6e7681"

            phase2_html = (
                f"""<div style="background:rgba(255,255,255,0.04);border-radius:8px;
                                padding:10px 12px;margin-top:10px;border-left:3px solid {phase2_c};">
                      <div style="font-size:11px;font-weight:800;color:#8b949e;
                                  text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                        Phase 2 · After Fill</div>
                      <div style="font-size:14px;font-weight:800;color:{phase2_c};">
                        {phase2_lbl}</div>
                      <div style="font-size:11px;color:#c9d1d9;margin-top:2px;">
                        at fill level ${tod.fill_level:.2f} →
                        target ${phase2_target:.2f}
                        <span style="color:{phase2_c};">(+{phase2_pts:.1f} pts)</span></div>
                      <div style="font-size:10px;color:#8b949e;margin-top:2px;">
                        Reversal after fill: {rev_r_str} historically</div>
                    </div>"""
                if show_phase2 else ""
            )

            with cols[i % n_cols]:
                st.html(
                    f"""
                    <div style="background:#0d1117;border:2px solid {sig_c};
                                border-radius:16px;padding:20px 18px;margin-bottom:12px;
                                box-shadow:0 0 24px {sig_c}33;">

                      <!-- Header row -->
                      <div style="display:flex;justify-content:space-between;
                                  align-items:center;margin-bottom:10px;">
                        <div>
                          <span style="font-size:26px;font-weight:800;color:#fff;
                                       letter-spacing:-.01em;">{tkr}</span>
                          {live_badge}
                        </div>
                        <div style="text-align:right;">
                          <span style="background:{conf_c}22;color:{conf_c};font-size:11px;
                                       font-weight:800;padding:3px 8px;border-radius:4px;
                                       letter-spacing:.07em;">{conf_str} CONF</span>
                        </div>
                      </div>

                      <!-- Gap direction -->
                      <div style="font-size:13px;font-weight:800;color:{dir_c};
                                  margin-bottom:12px;">{dir_str} · {tod.gap_pts:+.2f} pts</div>

                      <!-- Action box -->
                      <div style="background:{action_c}18;border:2px solid {action_c};
                                  border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                        <div style="font-size:10px;font-weight:800;color:#8b949e;
                                    text-transform:uppercase;letter-spacing:.1em;
                                    margin-bottom:6px;">Phase 1 · Ride the Fill</div>
                        <div style="font-size:28px;font-weight:800;color:{action_c};
                                    line-height:1;margin-bottom:10px;">{action_lbl}</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                          <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;">
                            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;
                                        font-weight:700;letter-spacing:.06em;margin-bottom:3px;">Entry</div>
                            <div style="font-size:18px;font-weight:800;color:#fff;">
                              ${entry_price:.2f}</div>
                            <div style="font-size:11px;color:#8b949e;">ATM ~${atm_strike}</div>
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;">
                            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;
                                        font-weight:700;letter-spacing:.06em;margin-bottom:3px;">Target</div>
                            <div style="font-size:18px;font-weight:800;color:{action_c};">
                              ${tod.fill_level:.2f}</div>
                            <div style="font-size:11px;color:#8b949e;">prev close</div>
                          </div>
                          <div style="background:rgba(0,0,0,0.3);border-radius:6px;padding:8px;">
                            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;
                                        font-weight:700;letter-spacing:.06em;margin-bottom:3px;">Potential</div>
                            <div style="font-size:18px;font-weight:800;color:#ffd633;">
                              {pts_to_fill:.2f}</div>
                            <div style="font-size:11px;color:#8b949e;">pts remaining</div>
                          </div>
                        </div>
                      </div>

                      <!-- Fill progress bar -->
                      <div style="margin-bottom:4px;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:4px;">
                          <span style="font-size:10px;font-weight:800;color:{fp_color};
                                       text-transform:uppercase;letter-spacing:.07em;">{fp_status}</span>
                          <span style="font-size:10px;color:#8b949e;">{fp:.0f}% filled</span>
                        </div>
                        <div style="height:8px;background:rgba(139,148,158,0.15);
                                    border-radius:4px;overflow:hidden;">
                          <div style="width:{fp:.0f}%;height:100%;background:{bar_c};
                                      border-radius:4px;"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    font-size:11px;color:#8b949e;margin-top:3px;">
                          <span>Open ${tod.open_price:.2f}</span>
                          <span>Fill target ${tod.fill_level:.2f}</span>
                        </div>
                      </div>

                      <!-- Phase 2 -->
                      {phase2_html}

                      <!-- Stats footer -->
                      <div style="display:flex;justify-content:space-between;
                                  margin-top:12px;padding-top:10px;
                                  border-top:1px solid rgba(255,255,255,0.06);
                                  font-size:10px;color:#8b949e;">
                        <span>Fill rate <strong style="color:#ffd633;">{fill_r_str}</strong></span>
                        <span>Rev after fill <strong style="color:#ffd633;">{rev_r_str}</strong></span>
                        <span>n={tod.hist_n_similar}</span>
                      </div>
                    </div>
                    """)

    if no_gap:
        st.caption(f"No gap today ({datetime.now().strftime('%b %d')}): {', '.join(t for t, _, _ in no_gap)} — all opened near yesterday's close.")

    # ── Deep-dive ─────────────────────────────────────────────────────────────
    st.markdown("---")
    section(f"Deep Dive — {detail_ticker}", "Historical stats, recent gap log, and backtest")

    try:
        df_feat, stats_bucket, stats_dir, stats_wd, tod_detail = load_gap_analysis(
            detail_ticker, lookback_yrs
        )
    except Exception as exc:
        st.error(f"Could not load {detail_ticker}: {exc}")
        st.stop()

    sig_c = gap_signal_color(tod_detail.signal)
    gap_all = df_feat[df_feat["gap_dir"].isin(["up","down"])].copy()
    ftr_days = df_feat[df_feat["fill_then_reversal"]].copy()

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Today's Gap", f"{tod_detail.gap_pct*100:+.2f}%" if tod_detail.gap_dir in ("up","down") else "None")
    k2.metric("Fill Target", fmt_dollar(tod_detail.fill_level))
    k3.metric("Hist Fill Rate", fmt_pct(tod_detail.hist_fill_rate) if tod_detail.hist_fill_rate else "—")
    k4.metric("Rev After Fill", fmt_pct(tod_detail.hist_rev_rate) if tod_detail.hist_rev_rate else "—")
    k5.metric("Median Rev", f"{tod_detail.hist_med_rev_pts:+.2f} pts" if tod_detail.hist_med_rev_pts else "—")

    # Today's signal detail
    st.html(
        f"""<div style="background:rgba(22,27,34,0.9);border:1px solid {sig_c};
                        border-radius:10px;padding:14px 18px;margin:12px 0 4px;">
          <span style="background:{sig_c};color:#0c1117;font-size:11px;font-weight:800;
                       padding:2px 7px;border-radius:3px;margin-right:8px;">{tod_detail.signal}</span>
          <span style="color:#c9d1d9;font-size:13px;">{tod_detail.signal_detail}</span>
        </div>""")

    # Backtest equity curve
    if len(gap_all) > 5:
        st.markdown("**Gap Fill+Reversal Strategy Equity Curve**")
        gap_all = gap_all.copy()
        gap_all["strategy_pnl"] = 0.0
        # Winners: use this row's own reversal_pts (correct index alignment)
        mask_down_win = (gap_all["gap_dir"] == "down") & gap_all["fill_then_reversal"]
        mask_up_win   = (gap_all["gap_dir"] == "up")   & gap_all["fill_then_reversal"]
        gap_all.loc[mask_down_win, "strategy_pnl"] = gap_all.loc[mask_down_win, "reversal_pts"].abs()
        gap_all.loc[mask_up_win,   "strategy_pnl"] = gap_all.loc[mask_up_win,   "reversal_pts"].abs()
        # Losers: small fixed loss proportional to gap size
        mask_loss = ~gap_all["fill_then_reversal"]
        gap_all.loc[mask_loss, "strategy_pnl"] = -(
            gap_all.loc[mask_loss, "abs_gap_pct"].fillna(0) *
            gap_all.loc[mask_loss, "Open"].fillna(0) * 0.5
        )
        gap_all["strategy_pnl"] = gap_all["strategy_pnl"].fillna(0)
        cum_pnl = gap_all["strategy_pnl"].cumsum().rename("Cumulative P&L (pts)")
        # Only chart if we have valid finite data
        if cum_pnl.notna().any() and cum_pnl.replace([float("inf"), float("-inf")], float("nan")).notna().any():
            st.line_chart(cum_pnl, use_container_width=True, height=180)

        win_rate  = gap_all["fill_then_reversal"].mean()
        avg_win   = gap_all.loc[gap_all["fill_then_reversal"], "strategy_pnl"].mean()
        avg_loss  = gap_all.loc[~gap_all["fill_then_reversal"], "strategy_pnl"].mean()
        total_pnl = gap_all["strategy_pnl"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate",  fmt_pct(win_rate))
        c2.metric("Avg Win",   f"{avg_win:+.2f} pts" if pd.notna(avg_win) else "—")
        c3.metric("Avg Loss",  f"{avg_loss:+.2f} pts" if pd.notna(avg_loss) else "—")
        c4.metric("Total P&L", f"{total_pnl:+.1f} pts")

    # Stats tables in expander
    with st.expander("Historical Stats Tables"):
        if not stats_dir.empty:
            st.markdown("**By Direction**")
            fmt_dir = stats_dir.copy()
            for c in ["Fill Rate", "Reversal Rate", "Fill+Rev Rate", "Avg Gap Size", "Avg Rev %"]:
                if c in fmt_dir.columns:
                    fmt_dir[c] = fmt_dir[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
            for c in ["Avg Rev Pts", "Med Rev Pts"]:
                if c in fmt_dir.columns:
                    fmt_dir[c] = fmt_dir[c].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            st.dataframe(fmt_dir, use_container_width=True, hide_index=True)

        if not stats_bucket.empty:
            st.markdown("**By Gap Size Bucket**")
            sb_d = stats_bucket.copy().reset_index()
            sb_d.columns = ["Gap Size","Sessions","Fill Rate","Rev Rate","Fill+Rev","Avg Rev Pts","Avg Rev %","Med Rev Pts","Avg Gap"]
            for c in ["Fill Rate","Rev Rate","Fill+Rev","Avg Rev %","Avg Gap"]:
                if c in sb_d.columns:
                    sb_d[c] = sb_d[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
            for c in ["Avg Rev Pts","Med Rev Pts"]:
                if c in sb_d.columns:
                    sb_d[c] = sb_d[c].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            if "Sessions" in sb_d.columns:
                sb_d["Sessions"] = sb_d["Sessions"].apply(lambda v: int(v) if pd.notna(v) else 0)
            st.dataframe(sb_d, use_container_width=True, hide_index=True)

        if not stats_wd.empty:
            st.markdown("**By Weekday**")
            wd_d = stats_wd.copy()
            for c in ["Fill Rate","Fill+Rev Rate"]:
                if c in wd_d.columns:
                    wd_d[c] = wd_d[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
            if "Avg Rev Pts" in wd_d.columns:
                wd_d["Avg Rev Pts"] = wd_d["Avg Rev Pts"].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            st.dataframe(wd_d, use_container_width=True, hide_index=True)

    # Fill rate chart
    if not stats_bucket.empty:
        fill_chart = stats_bucket[["fill_rate", "fill_then_rev"]].copy().dropna()
        if not fill_chart.empty:
            st.markdown("**Fill Rate & Fill+Reversal Rate by Gap Size**")
            fill_chart.columns = ["Fill Rate %", "Fill+Reversal %"]
            st.bar_chart(fill_chart * 100, use_container_width=True, height=180)

    # Recent gap log
    from src.gap_analysis import recent_gap_trades
    recent = recent_gap_trades(df_feat, n=30)
    if not recent.empty:
        with st.expander("Recent Gap Sessions (last 30)"):
            for c in ["Gap %"]:
                if c in recent.columns:
                    recent[c] = recent[c].apply(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
            for c in ["Gap Pts", "Rev Pts"]:
                if c in recent.columns:
                    recent[c] = recent[c].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            for c in ["Fill Level", "Close"]:
                if c in recent.columns:
                    recent[c] = recent[c].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
            st.dataframe(recent, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: ACCOUNT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Account Tracker":
    try:
        state = load_account()
    except Exception as e:
        st.error(f"Could not load account state: {e}")
        st.stop()

    pnl = state.total_pnl()
    pnl_pct = state.total_pnl_pct()
    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    section("Account Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting Equity", fmt_dollar(state.starting_equity))
    k2.metric("Current Equity",  fmt_dollar(state.current_equity),
              delta=f"{'+' if pnl >= 0 else ''}{fmt_dollar(pnl)}")
    k3.metric("Total Return",    f"{pnl_pct*100:+.1f}%")
    k4.metric("Trades Logged",   state.trade_count())
    k5.metric("Win Rate",
              fmt_pct(state.win_count() / state.trade_count())
              if state.trade_count() > 0 else "—")

    # ── Milestones ────────────────────────────────────────────────────────────
    st.markdown("---")
    section("Milestone Tracker", "Journey from starting equity to each goal")
    for ms in state.milestones:
        pct = min(state.current_equity / ms, 1.0) if ms > 0 else 0.0
        bar_c = "#3fb950" if pct >= 1.0 else "#ffd633" if pct >= 0.5 else "#58a6ff"
        done_label = "✅ REACHED" if pct >= 1.0 else f"{pct*100:.1f}% there"
        st.html(
            f"""
            <div style="background:#161b22;border-radius:10px;
                        padding:14px 18px;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:13px;margin-bottom:8px;">
                <span style="color:#e6edf3;font-weight:700;">
                  {fmt_dollar(state.starting_equity)}
                  <span style="color:#8b949e;font-weight:400;"> → </span>
                  {fmt_dollar(ms)}</span>
                <span style="color:{bar_c};font-weight:800;">{done_label}</span>
              </div>
              <div style="height:8px;background:rgba(139,148,158,0.12);
                          border-radius:4px;overflow:hidden;">
                <div style="width:{pct*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#3fb950,{bar_c});
                            border-radius:4px;"></div>
              </div>
            </div>
            """)

    # ── Log trade form ────────────────────────────────────────────────────────
    st.markdown("---")
    section("Log a Trade")
    with st.form("log_trade_form", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1.2, 1, 1])
        trade_date   = c1.date_input("Date", value=date.today())
        trade_ticker = c2.text_input("Ticker", value="SPY")
        trade_tier   = c3.selectbox("Signal tier", ["GO_ULTRA_JACKPOT", "GO_JACKPOT", "GO_HOT"])
        trade_risk   = c4.number_input("Risk ($)", value=50.0, min_value=0.0, step=10.0)
        trade_pnl    = c5.number_input("P&L ($)", value=0.0, step=1.0)
        trade_note   = st.text_input("Note (optional)", value="")
        submitted    = st.form_submit_button("Log Trade ›", use_container_width=True, type="primary")

    if submitted:
        from src.account_state import load_state, log_trade, save_state, snapshot_equity, TradeEntry
        s = load_state(path=ACCT_PATH)
        entry = TradeEntry(
            date=str(trade_date),
            ticker=trade_ticker.upper(),
            tier=trade_tier,
            risk=float(trade_risk),
            pnl=float(trade_pnl),
            note=trade_note,
        )
        s = log_trade(s, entry)
        s = snapshot_equity(s)
        save_state(s, path=ACCT_PATH)
        st.cache_data.clear()
        pnl_sign = "+" if trade_pnl >= 0 else ""
        st.success(f"Logged: {trade_ticker.upper()} {trade_tier} — P&L {pnl_sign}{fmt_dollar(trade_pnl)}")
        st.rerun()

    # ── Update equity form ────────────────────────────────────────────────────
    with st.expander("Update equity balance"):
        with st.form("set_equity_form", clear_on_submit=True):
            new_equity = st.number_input("Current equity ($)", value=float(state.current_equity),
                                         min_value=0.0, step=10.0)
            if st.form_submit_button("Update", use_container_width=True):
                from src.account_state import load_state, save_state, snapshot_equity
                s = load_state(path=ACCT_PATH)
                s.current_equity = float(new_equity)
                s = snapshot_equity(s)
                save_state(s, path=ACCT_PATH)
                st.cache_data.clear()
                st.success(f"Equity updated to {fmt_dollar(new_equity)}")
                st.rerun()

    # ── Equity curve ──────────────────────────────────────────────────────────
    if state.history and len(state.history) > 1:
        st.markdown("---")
        section("Equity Curve")
        hist_df = pd.DataFrame(state.history)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.set_index("date").sort_index()
        st.line_chart(hist_df["equity"], use_container_width=True, height=200)

    # ── Trade log ─────────────────────────────────────────────────────────────
    if state.trades:
        st.markdown("---")
        section("Trade Log", f"{len(state.trades)} trade{'s' if len(state.trades)>1 else ''} recorded")
        trades_data = []
        for t in reversed(state.trades):
            trades_data.append({
                "Date":   t.date,
                "Ticker": t.ticker,
                "Tier":   t.tier.replace("GO_", ""),
                "Risk":   fmt_dollar(t.risk),
                "P&L":    fmt_dollar(t.pnl),
                "Note":   t.note or "",
            })
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: WEEKDAY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Weekday Patterns":
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Settings**")
        ticker_wd = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "AAPL"], index=0)
        lookback  = st.selectbox("Lookback", [504, 252, 756], index=0,
                                 format_func=lambda v: {504:"2 years",252:"1 year",756:"3 years"}[v])

    with st.spinner(f"Loading {ticker_wd} history…"):
        try:
            daily = load_weekday_data(ticker_wd, lookback)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    if daily is None or len(daily) == 0:
        st.warning("No data available.")
        st.stop()

    wd_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    daily["Weekday"] = daily.index.dayofweek.map(wd_map)

    vol_col  = "YangZhang" if "YangZhang" in daily.columns else ("ATR" if "ATR" in daily.columns else None)
    rng_col  = "RangePct" if "RangePct" in daily.columns else None
    body_col = "BodyPct"  if "BodyPct"  in daily.columns else None

    if vol_col is None and rng_col is None:
        daily["RangePct"] = (daily["High"] - daily["Low"]) / daily["Open"]
        rng_col = "RangePct"

    agg_dict: dict = {"Close": "count"}
    if vol_col:  agg_dict[vol_col]  = "mean"
    if rng_col:  agg_dict[rng_col]  = "mean"
    if body_col: agg_dict[body_col] = "mean"

    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd_agg = daily.groupby("Weekday").agg(agg_dict).reindex(wd_order).dropna(how="all")
    wd_agg.rename(columns={"Close": "Sessions"}, inplace=True)

    # Best / worst day
    metric_col = vol_col or rng_col
    if metric_col and metric_col in wd_agg.columns:
        best_day  = wd_agg[metric_col].idxmax()
        worst_day = wd_agg[metric_col].idxmin()
    else:
        best_day = worst_day = "—"

    # ── Day summary banners ───────────────────────────────────────────────────
    section(f"Weekday Volatility — {ticker_wd}",
            f"{lookback//252}-year lookback · {len(daily)} sessions")

    col_best, col_worst = st.columns(2)
    with col_best:
        st.html(
            f"""<div style="background:#0d1f14;border:2px solid #3fb950;
                            border-radius:12px;padding:20px;text-align:center;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:6px;">Most Volatile Day</div>
              <div style="font-size:30px;font-weight:800;color:#3fb950;">{best_day}</div>
              <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                {'Avg range: ' + fmt_pct(wd_agg.loc[best_day, rng_col]) if rng_col and best_day in wd_agg.index else ""}</div>
            </div>""")
    with col_worst:
        st.html(
            f"""<div style="background:#1a1208;border:2px solid #8b949e;
                            border-radius:12px;padding:20px;text-align:center;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:6px;">Calmest Day</div>
              <div style="font-size:30px;font-weight:800;color:#8b949e;">{worst_day}</div>
              <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                {'Avg range: ' + fmt_pct(wd_agg.loc[worst_day, rng_col]) if rng_col and worst_day in wd_agg.index else ""}</div>
            </div>""")

    # ── Charts ────────────────────────────────────────────────────────────────
    if rng_col and rng_col in wd_agg.columns:
        st.markdown("---")
        section("Average Intraday Range % by Weekday")
        chart_df = (wd_agg[[rng_col]] * 100).rename(columns={rng_col: "Avg Range %"})
        st.bar_chart(chart_df, use_container_width=True, height=200)

    if vol_col and vol_col in wd_agg.columns and vol_col != rng_col:
        st.markdown("---")
        section(f"Average {vol_col} by Weekday")
        st.bar_chart(wd_agg[[vol_col]].rename(columns={vol_col: vol_col}),
                     use_container_width=True, height=180)

    # ── Data table ────────────────────────────────────────────────────────────
    st.markdown("---")
    section("Weekday Statistics Table")
    display_wd = wd_agg.copy()
    display_wd["Sessions"] = display_wd["Sessions"].apply(lambda v: int(v) if pd.notna(v) else 0)
    for c in [rng_col, body_col]:
        if c and c in display_wd.columns:
            display_wd[c] = display_wd[c].apply(
                lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—"
            )
    if vol_col and vol_col in display_wd.columns:
        display_wd[vol_col] = display_wd[vol_col].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
    st.dataframe(display_wd.reset_index(), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVERSAL LEVELS
# ══════════════════════════════════════════════════════════════════════════════

if page == "Reversal Levels":
    from src.key_levels import (
        compute_pivots, pivot_list,
        drop_band_analysis, matching_drop_band,
        vwap_deviation_analysis, pivot_touch_analysis,
        reversal_signal_score,
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        rl_ticker = st.selectbox(
            "Ticker", ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMZN"], index=0,
            key="rl_ticker"
        )
        rl_direction = st.radio(
            "Trade Direction",
            ["🟢 Low → Calls (bounce up)", "🔴 High → Puts (fade down)"],
            index=0, key="rl_dir",
        )
        is_long = rl_direction.startswith("🟢")

    section("Reversal Level Scanner", f"{rl_ticker} · Intraday extreme → open reversal analysis")

    # ── Live quote ───────────────────────────────────────────────────────────
    live = load_live_quotes((rl_ticker,))
    snap = live.get(rl_ticker, {})

    day_open  = snap.get("day_open",  0.0)
    day_high  = snap.get("day_high",  0.0)
    day_low   = snap.get("day_low",   0.0)
    day_close = snap.get("day_close", 0.0)
    day_vwap  = snap.get("day_vwap",  0.0)
    price     = day_close or day_open

    has_live  = (day_open > 0 and day_low > 0)
    drop_pts  = (day_low  - day_open) if has_live else 0.0   # negative = sold off
    rise_pts  = (day_high - day_open) if has_live else 0.0   # positive = ran up
    extreme_pts = drop_pts if is_long else rise_pts

    # ── Previous close for pivots ────────────────────────────────────────────
    prev = {}
    try:
        from src.polygon_feed import fetch_prev_close
        prev = fetch_prev_close(rl_ticker)
    except Exception:
        pass

    pivots = None
    if prev.get("high"):
        pivots = compute_pivots(prev["high"], prev["low"], prev["close"])

    # ── Historical data ───────────────────────────────────────────────────────
    try:
        daily, hist_bands, hist_vwap, hist_pivots = load_key_levels(rl_ticker, lookback_years=2)
    except Exception as exc:
        st.error(f"Could not load historical data: {exc}")
        st.stop()

    # ── TODAY'S CONTEXT ROW ───────────────────────────────────────────────────
    if has_live:
        vwap_dev = (day_low - day_vwap) if is_long else (day_high - day_vwap)
        c1, c2, c3, c4, c5 = st.columns(5)
        def metric_card(col, label, value, sub="", color="#fff"):
            with col:
                st.html(
                    f"""<div style="background:#161b22;border:1px solid #30363d;
                                    border-radius:10px;padding:14px 10px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                  letter-spacing:.08em;margin-bottom:4px;">{label}</div>
                      <div style="font-size:22px;font-weight:800;color:{color};
                                  letter-spacing:-.02em;">{value}</div>
                      {"<div style='color:#8b949e;font-size:10px;margin-top:4px;'>"+sub+"</div>" if sub else ""}
                    </div>"""
                )
        metric_card(c1, "Open",  f"${day_open:.2f}",  color="#e6edf3")
        metric_card(c2, "Today's Low",  f"${day_low:.2f}",
                    sub=f"{drop_pts:+.2f} from open", color="#f85149" if drop_pts < 0 else "#3fb950")
        metric_card(c3, "Today's High", f"${day_high:.2f}",
                    sub=f"{rise_pts:+.2f} from open", color="#3fb950" if rise_pts > 0 else "#f85149")
        metric_card(c4, "VWAP",  f"${day_vwap:.2f}",
                    sub=f"Low vs VWAP: {day_low-day_vwap:+.2f}", color="#ffd633")
        metric_card(c5, "Price", f"${price:.2f}",
                    sub="LIVE", color="#a5d6ff")
        st.html("<div style='margin-top:6px'></div>")

    # ── PIVOT LEVELS TABLE ────────────────────────────────────────────────────
    st.markdown("---")
    section("Today's Pivot Levels", "Calculated from yesterday's High / Low / Close")

    if pivots:
        all_levels = pivot_list(pivots)
        extreme_price = day_low if is_long else day_high
        near_label, near_price, near_dist = None, 0.0, 9999.0
        for lbl, lv in all_levels:
            d = abs(lv - extreme_price) if has_live else abs(lv - price)
            if d < near_dist:
                near_dist, near_label, near_price = d, lbl, lv

        rows_html = ""
        for lbl, lv in all_levels:
            if price > 0:
                dist = lv - price
                dist_str = f"{dist:+.2f}"
                is_near = lbl == near_label
                if dist > 0.5:
                    bar_col = "#238636"; side = "resistance"
                elif dist < -0.5:
                    bar_col = "#da3633"; side = "support"
                else:
                    bar_col = "#ffd633"; side = "at price"
            else:
                dist_str, is_near, bar_col, side = "—", False, "#30363d", ""

            highlight = "border-left:3px solid #ffd633;background:#1c1a0a;" if is_near else ""
            rows_html += f"""
            <tr style="{highlight}">
              <td style="padding:8px 12px;font-weight:{'800' if is_near else '600'};
                         color:{'#ffd633' if is_near else '#e6edf3'};font-size:12px;">{lbl}</td>
              <td style="padding:8px 12px;font-weight:800;color:{bar_col};font-size:14px;">
                ${lv:.2f}</td>
              <td style="padding:8px 12px;color:#8b949e;font-size:11px;">{dist_str}</td>
              <td style="padding:8px 12px;color:#8b949e;font-size:10px;text-transform:uppercase;
                         letter-spacing:.06em;">{side}</td>
              {"<td style='padding:8px 12px;font-size:10px;color:#ffd633;font-weight:800;'>← NEAREST TO LOW</td>" if is_near and is_long else ""}
              {"<td style='padding:8px 12px;font-size:10px;color:#ffd633;font-weight:800;'>← NEAREST TO HIGH</td>" if is_near and not is_long else ""}
              {"<td></td>" if not is_near else ""}
            </tr>"""

        st.html(
            f"""<div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                          background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
              <thead>
                <tr style="background:#161b22;">
                  <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Level</th>
                  <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Price</th>
                  <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Dist from Price</th>
                  <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Role</th>
                  <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Note</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table></div>""")
        st.html("<div style='margin-top:6px'></div>")

    # ── DROP BAND BOUNCE TABLE ────────────────────────────────────────────────
    st.markdown("---")
    section(
        "Historical Bounce Rates by Drop Magnitude",
        f"2-year {rl_ticker} daily data · When open→low drops X pts, how often does price bounce back?"
    )

    today_band = matching_drop_band(hist_bands, drop_pts) if has_live else None
    rows_html  = ""
    for b in hist_bands:
        is_today = (b is today_band)
        hi_lbl   = f"{b.drop_hi_pts:+.0f}" if b.drop_hi_pts > -900 else "flat"
        lo_lbl   = f"{b.drop_lo_pts:+.0f}" if b.drop_lo_pts > -900 else "—"
        r50      = int(b.recovery_50pct_rate * 100)
        r75      = int(b.recovery_75pct_rate * 100)
        r100     = int(b.close_above_open_rate * 100)
        med_b    = b.median_bounce_pts
        bar_w    = max(4, r50)

        if r50 >= 65:   bar_c = "#3fb950"
        elif r50 >= 50: bar_c = "#ffd633"
        else:           bar_c = "#f85149"

        hl = "border-left:3px solid #ffd633;background:#1c1a0a;" if is_today else ""
        rows_html += f"""
        <tr style="{hl}">
          <td style="padding:9px 12px;color:{'#ffd633' if is_today else '#8b949e'};
                     font-size:11px;font-weight:{'800' if is_today else '400'};">
            {lo_lbl} to {hi_lbl} pts
            {"&nbsp;<span style='font-size:9px;background:#1c1a0a;border:1px solid #ffd633;"
             "color:#ffd633;padding:1px 5px;border-radius:4px;'>TODAY</span>" if is_today else ""}</td>
          <td style="padding:9px 12px;color:#8b949e;font-size:11px;">{b.n_sessions}</td>
          <td style="padding:9px 12px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="background:{bar_c};height:8px;width:{bar_w}px;border-radius:4px;
                          min-width:4px;"></div>
              <span style="font-size:13px;font-weight:800;color:{bar_c};">{r50}%</span>
            </div>
          </td>
          <td style="padding:9px 12px;font-size:12px;font-weight:800;color:#a5d6ff;">{r75}%</td>
          <td style="padding:9px 12px;font-size:12px;font-weight:800;color:#3fb950;">{r100}%</td>
          <td style="padding:9px 12px;font-size:13px;font-weight:800;color:#e6edf3;">
            +{med_b:.1f} pts</td>
          <td style="padding:9px 12px;font-size:11px;color:#8b949e;">
            +{b.avg_open_to_low_pts:.1f} pts to open</td>
        </tr>"""

    st.html(
        f"""<div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                      background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
          <thead>
            <tr style="background:#161b22;">
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">Drop from Open</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">Sessions</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">≥50% Recovery</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">≥75% Recovery</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">Back to Open</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">Med Bounce</th>
              <th style="padding:8px 12px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.08em;">Avg to Open</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table></div>""")

    # ── PIVOT HISTORICAL TOUCH STATS ─────────────────────────────────────────
    st.markdown("---")
    section("Pivot Level Historical Touch Analysis",
            "When the daily Low (or High) has touched each pivot — historical bounce outcomes")

    if hist_pivots:
        rows_html = ""
        for pt in hist_pivots:
            r = int(pt.close_above_open_rate * 100)
            if r >= 60:  bc = "#3fb950"
            elif r >= 45: bc = "#ffd633"
            else:          bc = "#f85149"
            rows_html += f"""
            <tr>
              <td style="padding:9px 14px;font-size:13px;font-weight:800;color:#e6edf3;">{pt.pivot_label}</td>
              <td style="padding:9px 14px;font-size:12px;color:#8b949e;">{pt.n_touches}</td>
              <td style="padding:9px 14px;">
                <span style="font-size:15px;font-weight:800;color:{bc};">{r}%</span></td>
              <td style="padding:9px 14px;font-size:13px;font-weight:800;color:#3fb950;">
                +{pt.median_bounce_pts:.2f} pts</td>
              <td style="padding:9px 14px;font-size:12px;color:#8b949e;">
                +{pt.avg_return_to_open_pts:.2f} pts to open avg</td>
            </tr>"""
        st.html(
            f"""<div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                          background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
              <thead>
                <tr style="background:#161b22;">
                  <th style="padding:8px 14px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Pivot</th>
                  <th style="padding:8px 14px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Touches (2yr)</th>
                  <th style="padding:8px 14px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Closed Above Open</th>
                  <th style="padding:8px 14px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Med Bounce</th>
                  <th style="padding:8px 14px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.08em;">Avg Return to Open</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table></div>""")
    else:
        st.info("Not enough historical pivot touches to compute stats (need ≥3 touches per level).")

    # ── COMBINED SIGNAL + TRADE CARD ─────────────────────────────────────────
    st.markdown("---")
    section("Reversal Signal & Trade Setup",
            "Weighted combination of drop-band, VWAP deviation, and pivot touch signals")

    if has_live and pivots:
        vwap_dev_pts = (day_low - day_vwap) if is_long else (day_high - day_vwap)

        # Find matching pivot touch stat
        target_label = "S3" if is_long else "R1"
        pivot_stat = next((p for p in hist_pivots if p.pivot_label == target_label), None)
        if pivot_stat is None and hist_pivots:
            pivot_stat = next(
                (p for p in hist_pivots if p.pivot_label in ("S2", "S1")),
                hist_pivots[0]
            )

        sig, confidence, desc = reversal_signal_score(
            today_band, vwap_dev_pts, hist_vwap, pivot_stat
        )

        if sig == "STRONG REVERSAL":
            sig_col, sig_bg, sig_border = "#3fb950", "#0d1f14", "#3fb950"
        elif sig == "PROBABLE REVERSAL":
            sig_col, sig_bg, sig_border = "#ffd633", "#1a1208", "#ffd633"
        elif sig == "WATCH":
            sig_col, sig_bg, sig_border = "#a5d6ff", "#0d1724", "#a5d6ff"
        else:
            sig_col, sig_bg, sig_border = "#8b949e", "#161b22", "#30363d"

        conf_pct = int(confidence * 100)
        trade_type = "BUY CALLS" if is_long else "BUY PUTS"
        entry    = day_low if is_long else day_high
        target1  = day_vwap  if is_long else day_vwap
        target2  = day_open  if is_long else day_open
        move1    = target1 - entry if is_long else entry - target1
        move2    = target2 - entry if is_long else entry - target2
        stop     = entry - 1.0 if is_long else entry + 1.0
        near_pvt = near_label if pivots and near_label else "—"
        near_prc = near_price if pivots else 0.0

        st.html(
            f"""<div style="background:{sig_bg};border:2px solid {sig_border};
                            border-radius:16px;padding:28px;margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;
                          flex-wrap:wrap;gap:16px;">
                <div>
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                              letter-spacing:.1em;margin-bottom:6px;">Combined Signal</div>
                  <div style="font-size:28px;font-weight:800;color:{sig_col};
                              letter-spacing:-.02em;">{sig}</div>
                  <div style="color:#8b949e;font-size:10px;margin-top:6px;max-width:520px;
                              line-height:1.7;">{desc}</div>
                </div>
                <div style="text-align:right;">
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                              letter-spacing:.1em;margin-bottom:6px;">Confidence</div>
                  <div style="font-size:36px;font-weight:800;color:{sig_col};">{conf_pct}%</div>
                  <div style="color:#8b949e;font-size:10px;margin-top:4px;">
                    Nearest pivot: {near_pvt} @ ${near_prc:.2f}
                    (Δ{near_prc-entry:+.2f})</div>
                </div>
              </div>

              <div style="border-top:1px solid {sig_border}33;margin:20px 0 18px;"></div>

              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                            padding:14px;text-align:center;">
                  <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:6px;">Trade</div>
                  <div style="font-size:15px;font-weight:800;color:{sig_col};">{trade_type}</div>
                  <div style="color:#8b949e;font-size:9px;margin-top:4px;">0DTE ATM</div>
                </div>
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                            padding:14px;text-align:center;">
                  <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:6px;">Entry Zone</div>
                  <div style="font-size:17px;font-weight:800;color:#e6edf3;">${entry:.2f}</div>
                  <div style="color:#8b949e;font-size:9px;margin-top:4px;">
                    {near_pvt} pivot zone</div>
                </div>
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                            padding:14px;text-align:center;">
                  <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:6px;">Target 1 (VWAP)</div>
                  <div style="font-size:17px;font-weight:800;color:#ffd633;">${target1:.2f}</div>
                  <div style="color:#8b949e;font-size:9px;margin-top:4px;">
                    {'+' if move1>=0 else ''}{move1:.2f} pts</div>
                </div>
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                            padding:14px;text-align:center;">
                  <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:6px;">Target 2 (Open)</div>
                  <div style="font-size:17px;font-weight:800;color:#3fb950;">${target2:.2f}</div>
                  <div style="color:#8b949e;font-size:9px;margin-top:4px;">
                    {'+' if move2>=0 else ''}{move2:.2f} pts</div>
                </div>
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                            padding:14px;text-align:center;">
                  <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:6px;">Stop Loss</div>
                  <div style="font-size:17px;font-weight:800;color:#f85149;">${stop:.2f}</div>
                  <div style="color:#8b949e;font-size:9px;margin-top:4px;">$1 beyond entry</div>
                </div>
              </div>
            </div>""")

        # ── Historical context banner ─────────────────────────────────────────
        if today_band:
            b = today_band
            st.html(
                f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;
                                padding:18px 24px;display:flex;flex-wrap:wrap;gap:20px;
                                align-items:center;">
                  <div>
                    <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px;">Historical Precedents</div>
                    <div style="font-size:22px;font-weight:800;color:#e6edf3;">
                      {b.n_sessions} sessions</div>
                    <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                      with same drop magnitude</div>
                  </div>
                  <div style="width:1px;height:40px;background:#30363d;"></div>
                  <div>
                    <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px;">≥50% Recovery Rate</div>
                    <div style="font-size:22px;font-weight:800;color:#ffd633;">
                      {int(b.recovery_50pct_rate*100)}%</div>
                  </div>
                  <div style="width:1px;height:40px;background:#30363d;"></div>
                  <div>
                    <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px;">Median Bounce</div>
                    <div style="font-size:22px;font-weight:800;color:#3fb950;">
                      +{b.median_bounce_pts:.2f} pts</div>
                  </div>
                  <div style="width:1px;height:40px;background:#30363d;"></div>
                  <div>
                    <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px;">Avg Return to Open</div>
                    <div style="font-size:22px;font-weight:800;color:#a5d6ff;">
                      +{b.avg_open_to_low_pts:.2f} pts</div>
                  </div>
                  <div style="width:1px;height:40px;background:#30363d;"></div>
                  <div>
                    <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px;">Fully to Open</div>
                    <div style="font-size:22px;font-weight:800;color:#e6edf3;">
                      {int(b.close_above_open_rate*100)}%</div>
                  </div>
                </div>""")
    else:
        st.info("Waiting for today's intraday data from Polygon to compute the reversal signal.")

    # ── Recent S3 touch dates ─────────────────────────────────────────────────
    st.markdown("---")
    section("Recent Sessions Near Today's Setup",
            f"Last 12 sessions where {rl_ticker} low dropped 4–7 pts from open")

    recent_df = daily.copy()
    recent_df["_drop"] = recent_df["Low"] - recent_df["Open"]
    similar   = recent_df[(recent_df["_drop"] >= -7.0) & (recent_df["_drop"] <= -4.0)].tail(12)
    if len(similar) > 0:
        disp = similar[["Open","High","Low","Close"]].copy()
        disp["Drop pts"]   = (disp["Low"]   - disp["Open"]).map("{:+.2f}".format)
        disp["Bounce pts"] = (disp["Close"] - disp["Low"]).map("+{:.2f}".format)
        disp["Recovered?"] = (similar["Close"] >= similar["Open"]).map(
            {True: "✅ Yes", False: "❌ No"})
        disp["% Recovered"] = (
            (similar["Close"] - similar["Low"]) /
            (similar["Open"]  - similar["Low"]).replace(0, float("nan"))
        ).map(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—")
        for c in ["Open","High","Low","Close"]:
            disp[c] = disp[c].map("${:.2f}".format)
        disp.index = disp.index.strftime("%Y-%m-%d")
        st.dataframe(disp, use_container_width=True)
    else:
        st.info("No historical sessions found matching today's drop magnitude.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 0DTE LOTTERY
# ══════════════════════════════════════════════════════════════════════════════

if page == "0DTE Lottery":
    from src.options_scanner import (
        fetch_0dte_chain, analyze_chain, recommend_strikes,
        drop_band_multiplier_table,
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        lot_ticker = st.selectbox(
            "Ticker", ["SPY", "QQQ", "IWM"], index=0, key="lot_ticker"
        )
        lot_ctype = st.radio(
            "Direction", ["Calls (bounce from low)", "Puts (fade from high)"],
            index=0, key="lot_ctype",
        )
        is_calls = lot_ctype.startswith("Calls")

    # ── Live data ─────────────────────────────────────────────────────────────
    live = load_live_quotes((lot_ticker,))
    snap = live.get(lot_ticker, {})

    day_open  = snap.get("day_open",  0.0)
    day_high  = snap.get("day_high",  0.0)
    day_low   = snap.get("day_low",   0.0)
    day_close = snap.get("day_close", 0.0)
    day_vwap  = snap.get("day_vwap",  0.0)
    price     = day_close or day_open
    has_live  = day_open > 0 and day_low > 0

    drop_pts = (day_open - day_low)   if has_live else 0.0   # positive = sold off
    rise_pts = (day_high - day_open)  if has_live else 0.0   # positive = ran up

    section("0DTE Lottery Scanner",
            f"{lot_ticker} · Live 1000%+ options plays — today's intraday reversal setups")

    # ── Context strip ─────────────────────────────────────────────────────────
    if has_live:
        c1, c2, c3, c4 = st.columns(4)
        def lcard(col, lbl, val, sub="", color="#e6edf3"):
            with col:
                st.html(
                    f"""<div style="background:#161b22;border:1px solid #30363d;
                                    border-radius:10px;padding:14px 10px;text-align:center;">
                      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                  letter-spacing:.08em;margin-bottom:4px;">{lbl}</div>
                      <div style="font-size:22px;font-weight:800;color:{color};">{val}</div>
                      {"<div style='color:#8b949e;font-size:10px;margin-top:4px;'>"+sub+"</div>" if sub else ""}
                    </div>"""
                )
        lcard(c1, "Open",  f"${day_open:.2f}",  color="#e6edf3")
        lcard(c2, "Intraday Low", f"${day_low:.2f}",
              sub=f"Drop: {drop_pts:+.2f} pts from open",
              color="#f85149" if drop_pts > 0 else "#8b949e")
        lcard(c3, "Intraday High", f"${day_high:.2f}",
              sub=f"Rise: +{rise_pts:.2f} pts from open",
              color="#3fb950")
        lcard(c4, "VWAP", f"${day_vwap:.2f}",
              sub=f"Price vs VWAP: {price-day_vwap:+.2f}",
              color="#ffd633")
        st.html("<div style='margin-top:6px'></div>")

    # ── Fetch live options chain ──────────────────────────────────────────────
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with st.spinner(f"Loading {lot_ticker} 0DTE options chain…"):
            @st.cache_data(ttl=30, show_spinner=False)
            def _fetch_chain(ticker, exp):
                return fetch_0dte_chain(ticker, exp_date=exp)
            contracts = _fetch_chain(lot_ticker, today_str)

        ctype_filter = "call" if is_calls else "put"
        analyses = analyze_chain(
            [c for c in contracts if c.contract_type == ctype_filter],
            underlying_open  = day_open  or price,
            underlying_low   = day_low   or price,
            underlying_high  = day_high  or price,
            underlying_close = day_close or price,
            strike_window    = 20.0,
        )
    except Exception as exc:
        st.error(f"Options chain error: {exc}")
        st.stop()

    over_1000  = [a for a in analyses if a.is_1000_plus]
    over_500   = [a for a in analyses if a.day_gain_pct >= 500 and not a.is_1000_plus]
    sweet_spot = [a for a in analyses if a.is_sweet_spot]

    # ── Hero banner ───────────────────────────────────────────────────────────
    hero_count = len(over_1000)
    hero_color = "#ffd633" if hero_count > 0 else "#8b949e"
    hero_bg    = "linear-gradient(135deg,#1a1208,#131008)" if hero_count > 0 else "#161b22"
    hero_border = "#ffd633" if hero_count > 0 else "#30363d"
    hero_title = (
        f"🌟 {hero_count} CONTRACT{'S' if hero_count!=1 else ''} HIT 1000%+ TODAY"
        if hero_count > 0
        else "No 1000%+ Contracts Yet Today"
    )

    st.html(
        f"""<div style="background:{hero_bg};border:2px solid {hero_border};
                        border-radius:14px;padding:22px 28px;margin-bottom:20px;">
          <div style="font-size:24px;font-weight:800;color:{hero_color};
                      margin-bottom:10px;">{hero_title}</div>
          <div style="display:flex;gap:28px;flex-wrap:wrap;">
            <div>
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;">500%+ movers</div>
              <div style="font-size:28px;font-weight:800;color:#f85149;">
                {len(over_500)+hero_count}</div>
            </div>
            <div>
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;">Sweet-spot contracts</div>
              <div style="font-size:28px;font-weight:800;color:#3fb950;">
                {len(sweet_spot)}</div>
            </div>
            <div>
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;">Total 0DTE contracts</div>
              <div style="font-size:28px;font-weight:800;color:#8b949e;">
                {len(analyses)}</div>
            </div>
            <div>
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;">Today's drop</div>
              <div style="font-size:28px;font-weight:800;color:#a5d6ff;">
                {drop_pts:+.1f} pts</div>
            </div>
          </div>
        </div>""")

    # ── Live options chain table ──────────────────────────────────────────────
    st.markdown("---")
    section(
        "Live 0DTE Chain — Today's Movers",
        f"{lot_ticker} {'calls' if is_calls else 'puts'} · day low → high gain% · sweet spot = strikes +1 to +8 pts from intraday low"
    )

    if analyses:
        rows_html = ""
        for a in analyses:
            c = a.contract
            if c.day_low <= 0: continue
            gain_pct = a.day_gain_pct
            if gain_pct >= 1000:
                gain_c = "#ffd633"; badge = "🌟 JACKPOT"
            elif gain_pct >= 500:
                gain_c = "#f85149"; badge = "🔥 FIRE"
            elif gain_pct >= 100:
                gain_c = "#3fb950"; badge = "📈 HOT"
            else:
                gain_c = "#8b949e"; badge = ""
            sw = "border-left:3px solid #ffd633;" if a.is_sweet_spot else ""
            dist_c = "#ffd633" if a.is_sweet_spot else "#8b949e"
            rows_html += f"""
            <tr style="{sw}">
              <td style="padding:9px 12px;font-weight:800;font-size:14px;
                         color:#e6edf3;">${c.strike:.0f}</td>
              <td style="padding:9px 12px;font-size:12px;color:{dist_c};font-weight:700;">
                {a.dist_from_underlying_low:+.1f} pts</td>
              <td style="padding:9px 12px;font-size:12px;color:#8b949e;">
                {a.dist_from_underlying_open:+.1f} pts</td>
              <td style="padding:9px 12px;font-size:12px;color:#8b949e;">
                ${c.day_low:.3f}</td>
              <td style="padding:9px 12px;font-size:13px;font-weight:800;color:#3fb950;">
                ${c.day_high:.2f}</td>
              <td style="padding:9px 12px;font-size:15px;font-weight:800;color:{gain_c};">
                {gain_pct:,.0f}%</td>
              <td style="padding:9px 12px;font-size:11px;color:{gain_c};">{badge}</td>
              <td style="padding:9px 12px;font-size:11px;color:#8b949e;">
                Δ{c.delta:.2f}  IV{c.implied_vol*100:.0f}%</td>
            </tr>"""

        st.html(
            f"""<div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                          background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
              <thead>
                <tr style="background:#161b22;border-bottom:1px solid #30363d;">
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Strike</th>
                  <th style="padding:9px 12px;color:#ffd633;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Dist from Low ★</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Dist from Open</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Day Low $</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Day High $</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Max Gain %</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Signal</th>
                  <th style="padding:9px 12px;color:#8b949e;font-size:10px;text-align:left;
                             text-transform:uppercase;letter-spacing:.07em;">Greeks</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table></div>""")
    else:
        st.info("No contracts in range — options chain may not be available yet or market is closed.")

    # ── Strike recommendation card ────────────────────────────────────────────
    if has_live and analyses:
        st.markdown("---")
        section(
            "Strike Recommendation — What to Buy at the Low",
            f"If {lot_ticker} reverses from ${day_low:.2f} back toward open ${day_open:.2f} — "
            f"expected +{drop_pts:.1f} pt recovery"
        )

        recs = recommend_strikes(day_open, day_low, contracts)
        if recs:
            rec_html = ""
            for r in recs[:6]:
                is_best = r.est_gain_pct == max(x.est_gain_pct for x in recs)
                bg  = "#0d1f14" if is_best else "#0d1117"
                bdr = "#3fb950" if is_best else "#30363d"
                tag = f"""<div style="font-size:9px;background:#0d1f14;border:1px solid #3fb950;
                              color:#3fb950;padding:2px 6px;border-radius:4px;margin-bottom:6px;
                              display:inline-block;">BEST R/R</div><br>""" if is_best else ""
                rec_html += f"""
                <div style="background:{bg};border:2px solid {bdr};border-radius:12px;
                            padding:18px;flex:1;min-width:140px;text-align:center;">
                  {tag}
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                              letter-spacing:.07em;margin-bottom:4px;">
                    Call ${r.strike:.0f}</div>
                  <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">
                    +{r.dist_from_low:.0f} pts from low</div>
                  <div style="font-size:13px;color:#8b949e;margin-bottom:2px;">
                    Entry: <span style="color:#e6edf3;font-weight:800;">
                      ${r.est_entry_price:.2f}</span></div>
                  <div style="font-size:13px;color:#8b949e;margin-bottom:8px;">
                    Target: <span style="color:#3fb950;font-weight:800;">
                      ${r.est_target_price:.2f}</span></div>
                  <div style="font-size:22px;font-weight:800;color:#ffd633;">
                    {r.est_gain_pct:,.0f}%</div>
                  <div style="color:#8b949e;font-size:10px;margin-top:6px;">
                    {r.note}</div>
                </div>"""
            st.html(
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{rec_html}</div>')
        st.html("<div style='margin-top:8px'></div>")

    # ── Historical pattern table ──────────────────────────────────────────────
    st.markdown("---")
    section(
        "Historical 1000%+ Probability by Drop Magnitude",
        "2-year SPY daily data · BSM model estimates · When intraday drop is X pts from open — "
        "how often does the call reversal produce 1000%+"
    )

    mult_table = drop_band_multiplier_table()
    today_band_lbl = None
    if has_live:
        for row in mult_table:
            band = row["band"]
            lo_str, hi_str = band.split("–")
            try:
                lo_v = float(lo_str.strip().replace("+",""))
                hi_v = float(hi_str.split()[0].strip().replace("+",""))
                if lo_v <= drop_pts < hi_v:
                    today_band_lbl = band
                    break
            except Exception:
                pass

    rows_html = ""
    for row in mult_table:
        is_today = (row["band"] == today_band_lbl)
        pct = row["pct_1000plus"]
        n   = row["n"]
        rec = row["recovery_needed"]
        note = row["note"]
        if pct >= 25:   bar_c = "#ffd633"
        elif pct >= 10: bar_c = "#d29922"
        else:            bar_c = "#30363d"
        bar_w = max(4, pct * 3)
        hl = "border-left:3px solid #ffd633;background:#1c1a0a;" if is_today else ""
        today_badge = (
            "&nbsp;<span style='font-size:9px;background:#1c1a0a;border:1px solid #ffd633;"
            "color:#ffd633;padding:1px 5px;border-radius:4px;'>TODAY</span>"
            if is_today else ""
        )
        rows_html += f"""
        <tr style="{hl}">
          <td style="padding:9px 14px;font-size:12px;font-weight:{'800' if is_today else '400'};
                     color:{'#ffd633' if is_today else '#e6edf3'};">
            {row['band']}{today_badge}</td>
          <td style="padding:9px 14px;font-size:12px;color:#8b949e;">{n}</td>
          <td style="padding:9px 14px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="background:{bar_c};height:8px;width:{bar_w}px;
                          border-radius:4px;"></div>
              <span style="font-size:14px;font-weight:800;color:{bar_c};">{pct}%</span>
            </div>
          </td>
          <td style="padding:9px 14px;font-size:12px;color:#8b949e;">
            {f'+{rec:.1f} pts' if rec > 0 else '—'}</td>
          <td style="padding:9px 14px;font-size:11px;color:#8b949e;">{note}</td>
        </tr>"""

    st.html(
        f"""<div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                      background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden;">
          <thead>
            <tr style="background:#161b22;border-bottom:1px solid #30363d;">
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Drop from Open</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Sessions (2yr)</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">1000%+ Probability</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Avg Recovery Needed</th>
              <th style="padding:9px 14px;color:#8b949e;font-size:10px;text-align:left;
                         text-transform:uppercase;letter-spacing:.07em;">Status</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table></div>""")

    # ── How it works explainer ────────────────────────────────────────────────
    st.markdown("---")
    section("How the 1000%+ Setup Works",
            "The math behind today's SPY $737 call: $0.04 → $1.84 = 4,500%")
    st.html(
        f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;
                        padding:24px 28px;font-family:Inter,sans-serif;line-height:1.8;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
            <div>
              <div style="color:#ffd633;font-size:13px;font-weight:800;
                          margin-bottom:12px;text-transform:uppercase;letter-spacing:.06em;">
                The Pattern</div>
              <div style="color:#c9d1d9;font-size:13px;">
                <span style="color:#a5d6ff;font-weight:700;">1.</span>
                  Underlying opens (e.g. SPY at $736.89)<br>
                <span style="color:#a5d6ff;font-weight:700;">2.</span>
                  Early sell-off drives price to intraday low ($731.83 = −5 pts)<br>
                <span style="color:#a5d6ff;font-weight:700;">3.</span>
                  Calls near the OPEN strike become deep OTM → worth pennies<br>
                <span style="color:#a5d6ff;font-weight:700;">4.</span>
                  Reversal happens — price snaps back to open or higher<br>
                <span style="color:#a5d6ff;font-weight:700;">5.</span>
                  Those same calls go from pennies to $1–$6 → 1000–6000%
              </div>
            </div>
            <div>
              <div style="color:#ffd633;font-size:13px;font-weight:800;
                          margin-bottom:12px;text-transform:uppercase;letter-spacing:.06em;">
                Today's Example</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#161b22;border-radius:8px;padding:10px;text-align:center;">
                  <div style="color:#8b949e;font-size:10px;">SPY Open</div>
                  <div style="color:#e6edf3;font-weight:800;font-size:15px;">${day_open:.2f}</div>
                </div>
                <div style="background:#161b22;border-radius:8px;padding:10px;text-align:center;">
                  <div style="color:#8b949e;font-size:10px;">SPY Low</div>
                  <div style="color:#f85149;font-weight:800;font-size:15px;">${day_low:.2f}</div>
                </div>
                <div style="background:#161b22;border-radius:8px;padding:10px;text-align:center;">
                  <div style="color:#8b949e;font-size:10px;">$737 Call @ Low</div>
                  <div style="color:#8b949e;font-weight:800;font-size:15px;">$0.04</div>
                </div>
                <div style="background:#161b22;border-radius:8px;padding:10px;text-align:center;">
                  <div style="color:#8b949e;font-size:10px;">$737 Call High</div>
                  <div style="color:#3fb950;font-weight:800;font-size:15px;">$1.84</div>
                </div>
              </div>
              <div style="background:#1c1a0a;border:1px solid #ffd633;border-radius:8px;
                          padding:12px;text-align:center;margin-top:10px;">
                <div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Actual gain</div>
                <div style="color:#ffd633;font-weight:800;font-size:28px;">4,500%</div>
                <div style="color:#8b949e;font-size:11px;">$0.04 → $1.84 per contract</div>
              </div>
            </div>
          </div>
          <div style="border-top:1px solid #30363d;margin-top:20px;padding-top:16px;">
            <div style="color:#8b949e;font-size:12px;line-height:1.7;">
              <span style="color:#ffd633;font-weight:700;">Sweet spot rule:</span>
              Buy calls at strikes <strong style="color:#fff;">+1 to +8 pts above the intraday low</strong>
              (= near the open price, OTM by the drop amount).
              These are cheap because they need the full reversal to become ATM/ITM.
              Risk: if no reversal, you lose 100% of a small position.<br>
              <span style="color:#ffd633;font-weight:700;">Timing:</span>
              Best entries are when the low is confirmed (price starts moving back up).
              Use the Reversal Levels page S3 pivot + VWAP as confluence.
            </div>
          </div>
        </div>""")
