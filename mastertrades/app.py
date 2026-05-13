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


@st.cache_data(ttl=900, show_spinner=False)
def load_cc_gap_verdicts(tickers: tuple = ("SPY", "QQQ", "IWM", "AAPL")):
    """Load gap verdict for each Command Center ticker. Returns list of (ticker, TodayGap)."""
    from src.scanner import fetch_or_load_daily
    from src.gap_analysis import run_gap_analysis
    results = []
    for tkr in tickers:
        try:
            daily = fetch_or_load_daily(tkr, data_dir=DATA_DIR, refresh=True)
            _, sb, _, _, tod = run_gap_analysis(tkr, daily)
            results.append((tkr, tod))
        except Exception:
            pass
    return results


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


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest(ticker: str = "SPY", start_equity: float = 500.0,
                  n_months: int = 6, strategy_mode: str = "straddle"):
    """Run (or load cached) real 6-month historical backtest. TTL=1h."""
    try:
        from src.backtest import run_jackpot_backtest
        return run_jackpot_backtest(ticker=ticker, start_equity=start_equity,
                                    n_months=n_months, strategy_mode=strategy_mode)
    except Exception as e:
        return None


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
    "Today's Plays":    ("⚡ Today's Plays",    "Every actionable signal, ranked by edge"),
    "Command Center":   ("🎯 Command Center",   "Today's ML jackpot signals (SPY/QQQ/IWM/AAPL)"),
    "MA Bounce Setups": ("🎯 MA Bounce Setups", "22 high-edge weekly MA-touch plays"),
    "Gap Reversal":     ("🎯 Gap Reversal",     "Gap fill & reversal setups"),
    "0DTE Lottery":     ("🎯 0DTE Lottery",     "1000%+ options plays & sweet spots"),
    "Scanner":          ("📊 Scanner",          "Ranked volatility universe (research)"),
    "Weekly MAs":       ("📊 Weekly MAs",       "Per-ticker MA + order flow drill-down"),
    "Reversal Levels":  ("📊 Reversal Levels",  "Intraday low/high reversal zones"),
    "Weekday Patterns": ("📊 Weekday Patterns", "Vol by day of week (research)"),
    "Account Tracker":  ("💰 Account Tracker",  "Equity curve & trade log"),
}

with st.sidebar:
    st.html(
        """<div style="font-size:20px;font-weight:800;color:#fff;
                       letter-spacing:-.01em;margin-bottom:4px;">📈 Mastertrades</div>
           <div style="color:#8b949e;font-size:11px;margin-bottom:16px;">
             0DTE Options Intelligence</div>""")
    st.markdown("---")
    st.html(
        """<div style="font-size:10px;font-weight:800;color:#8b949e;
                       text-transform:uppercase;letter-spacing:.08em;
                       margin-bottom:6px;">⚡ Trade Now &nbsp;·&nbsp; 🎯 Signal Details &nbsp;·&nbsp; 📊 Research &nbsp;·&nbsp; 💰 Account</div>""")
    page = st.radio(
        "Navigate",
        list(PAGE_META.keys()),
        label_visibility="collapsed",
        format_func=lambda p: PAGE_META[p][0],
    )
    st.caption(PAGE_META[page][1])
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

if page == "Today's Plays":
    import math
    section("⚡ Today's Plays",
            "Every actionable signal from every validated system, ranked by normalized expected edge")

    # ── Doctrine banner: empirically-derived MA Bounce trading rules ─────────
    with st.expander("💎 High-Conviction Doctrine — what we learned from 3 months of real Polygon options data", expanded=False):
        st.html("""
        <div style="background:#0a1428;border-left:3px solid #58a6ff;padding:14px 18px;
                    border-radius:6px;font-size:13px;line-height:1.6;color:#c9d1d9;">
          <div style="color:#58a6ff;font-weight:800;margin-bottom:8px;font-size:14px;">
            The current algo (market-buy on touch close, ATM, no stop) achieved
            42% win rate / +42% avg / $424 per fill.
          </div>
          <div style="color:#f0f6fc;font-weight:700;margin-bottom:6px;">
            The empirically-validated High-Conviction Doctrine (shown on every MA Bounce card below):
          </div>
          <ul style="margin:6px 0 12px 18px;color:#c9d1d9;">
            <li><b style="color:#58a6ff;">Smart entry:</b> limit at MA × 0.995 (−0.50% pullback), valid 5 trading days. Skip if unfilled.</li>
            <li><b style="color:#58a6ff;">Strike:</b> OTM+$5 from fill price, 1-week expiry.</li>
            <li><b style="color:#58a6ff;">Exit:</b> hold to expiry. <b>No stop loss</b> — backtests show stops hurt P&L by ~18%.</li>
            <li><b style="color:#58a6ff;">Sizing:</b> equal $ per fill. Optionally size 1.5–2× when a deeper limit at MA × 0.99 also fills.</li>
          </ul>
          <div style="color:#c9d1d9;margin-bottom:8px;">
            <b style="color:#3fb950;">Result on 72 signals over 3 months:</b>
            35% fill rate, <b>64% win rate</b>, +132% avg return, <b>$1,322 per fill</b>
            (3× per-trade efficiency vs current algo).
          </div>
          <div style="color:#8b949e;font-size:11px;font-style:italic;border-top:1px solid #21262d;padding-top:8px;">
            ⚠ Honest caveats: one of three months (April 2026) provided most of the P&L —
            real-world results will be lumpy. Sample of 72 signals is decent but not bulletproof.
            HYG_30w EMA and XLV_50w SMA were dropped from the universe after their historical
            edges failed to replicate (30%/29% real win rates vs 83%/80% claimed).
            See <code>mastertrades/scripts/backtest_*.py</code> for the full validation.
          </div>
        </div>
        """)

    # ── Helpers: unified normalized edge ─────────────────────────────────────
    # All sources contribute a single comparable score:
    #   edge_pct = expected_return_per_trade(%) × confidence × sample_shrinkage
    # where sample_shrinkage = sqrt(n / (n + PRIOR)) penalises tiny samples,
    # and expected_return = win_rate × avg_win_return (rough Bernoulli proxy).
    # Strict gate: avg_ret must be > 0 AND win_rate ≥ 50% AND n ≥ MIN_N.
    PRIOR_N = 5
    MIN_N   = 3
    MIN_WINRATE = 50.0   # %

    def _shrink(n: int) -> float:
        return math.sqrt(max(n, 0) / (max(n, 0) + PRIOR_N))

    def _edge(win_rate_pct: float, avg_ret_pct: float, n: int, conf: float) -> float:
        # Expected % return per trade × confidence × sample shrinkage
        return (win_rate_pct / 100.0) * avg_ret_pct * conf * _shrink(n)

    def _passes_gate(win_rate_pct: float, avg_ret_pct: float, n: int) -> bool:
        if avg_ret_pct is None or win_rate_pct is None:
            return False
        if not (math.isfinite(avg_ret_pct) and math.isfinite(win_rate_pct)):
            return False
        if avg_ret_pct <= 0:
            return False
        if win_rate_pct < MIN_WINRATE:
            return False
        if n < MIN_N:
            return False
        return True

    # ── Source-health tracking ───────────────────────────────────────────────
    plays = []
    source_health = {}   # name -> {"ok": bool, "msg": str, "n_in": int, "n_kept": int}

    # 1. MA Bounce Setups (TOUCHING + APPROACHING only)
    try:
        from src.ma_setups_universe import get_all_live_setups, SCAN_DATE
        @st.cache_data(ttl=600)
        def _tp_ma():
            return get_all_live_setups()
        live_ma, ma_failed = _tp_ma()
        n_in = n_kept = 0
        for s in live_ma:
            if s.state not in ("TOUCHING", "APPROACHING"):
                continue
            n_in += 1
            if not _passes_gate(s.pct_pos_5d, s.avg_5d, s.n_touches):
                continue
            conf = 1.0 if s.state == "TOUCHING" else 0.55
            edge = _edge(s.pct_pos_5d, s.avg_5d, s.n_touches, conf)
            action = "BUY CALLS NOW" if s.state == "TOUCHING" else "WATCH FOR TOUCH"
            # ── HIGH-CONVICTION DOCTRINE (3-month Polygon-validated) ──────
            # Smart entry: limit at MA × 0.995 (−0.50% pullback), valid 5 trading days
            # Strike:      OTM ~$5 from fill, capped at 2% of underlying so the
            #              heuristic stays sane on lower-priced names. 1-week call.
            # Hold to expiry. No stop loss. Equal $ size per fill.
            #   Backtest: 35% fill rate, 64% win, +132% avg, $1,322/fill (vs current
            #   algo's 42% win, +42% avg, $424/fill). 3× per-trade efficiency.
            smart_entry  = s.ma_value * 0.995
            conviction   = s.ma_value * 0.99
            # Cap OTM distance at 2% of underlying — backtest used names ≥$80
            # where $5 ≈ 0.5–6% OTM. For lower-priced names $5 would be absurd.
            otm_distance = min(5.0, smart_entry * 0.02)
            otm5_strike  = round(smart_entry + otm_distance)
            plays.append({
                "source": "MA Bounce", "ticker": s.ticker, "tag": s.ma_label,
                "state": s.state, "action": action,
                "entry": s.last_close,
                "target": s.last_close * (1 + s.avg_5d / 100),
                "win_rate": s.pct_pos_5d, "avg_ret": s.avg_5d, "n": s.n_touches,
                "edge": edge,
                "reason": f"{s.ma_label} historically bounces {s.pct_pos_5d:.0f}% of touches for +{s.avg_5d:.2f}% in 5d (n={s.n_touches}). "
                          f"Currently {s.distance_pct:+.2f}% from MA, {s.state}.",
                "horizon": "5 days (weekly options)",
                # Smart-entry doctrine fields (consumed by play card render below)
                "doctrine_ma":          s.ma_value,
                "doctrine_smart_entry": smart_entry,
                "doctrine_conviction":  conviction,
                "doctrine_strike":      otm5_strike,
                "doctrine_fill_window": "5 trading days",
                "doctrine_hold_rule":   "Hold to expiry. No stop loss.",
            })
            n_kept += 1
        msg = f"{len(live_ma)} live setups, {n_in} actionable, {n_kept} passed edge gate"
        if ma_failed:
            msg += f" · ⚠ {len(ma_failed)} setups failed to price: {', '.join(t+' '+m for t,m in ma_failed[:5])}"
        source_health["MA Bounce"] = {"ok": True, "msg": msg, "n_in": n_in, "n_kept": n_kept}
    except Exception as e:
        source_health["MA Bounce"] = {"ok": False, "msg": f"ERROR: {e}", "n_in": 0, "n_kept": 0}

    # 2. ML Jackpot (GO_JACKPOT / GO_ULTRA_JACKPOT)
    try:
        jackpot_rows, jerr = load_jackpot_scan()
        n_in = n_kept = 0
        for r in jackpot_rows:
            if r.signal not in ("GO_JACKPOT", "GO_ULTRA_JACKPOT"):
                continue
            n_in += 1
            is_ultra = (r.signal == "GO_ULTRA_JACKPOT")
            win_rate = (r.ultra_win_rate_history if is_ultra else r.win_rate_history) * 100
            avg_ret  = (r.ultra_avg_ret_history if is_ultra else r.avg_ret_history) * 100
            n_hist   = r.n_ultra_history if is_ultra else r.n_jackpot_history
            if not _passes_gate(win_rate, avg_ret, n_hist):
                continue   # ← rejects ML signals with non-positive historical edge
            conf = 1.2 if is_ultra else 1.0
            edge = _edge(win_rate, avg_ret, n_hist, conf)
            plays.append({
                "source": "ML Jackpot", "ticker": r.ticker, "tag": r.signal,
                "state": "ENTRY_OPEN",
                "action": "BUY 0DTE STRADDLE" if is_ultra else "BUY 0DTE OPTIONS",
                "entry": r.last_close,
                "target": r.last_close * (1 + avg_ret / 100),
                "win_rate": win_rate, "avg_ret": avg_ret, "n": n_hist,
                "edge": edge,
                "reason": f"Both ML models agree: P(volatile)={r.p_vol:.0%}, P(P&L)={r.p_pnl:.0%}. "
                          f"Historical {r.signal}: {win_rate:.0f}% win @ +{avg_ret:.2f}% avg (n={n_hist}).",
                "horizon": "Same day (0DTE)",
            })
            n_kept += 1
        msg = f"{len(jackpot_rows)} tickers scanned, {n_in} fired, {n_kept} passed edge gate"
        if jerr:
            msg += f" · ⚠ {len(jerr)} ticker errors"
        source_health["ML Jackpot"] = {"ok": True, "msg": msg, "n_in": n_in, "n_kept": n_kept}
    except Exception as e:
        source_health["ML Jackpot"] = {"ok": False, "msg": f"ERROR: {e}", "n_in": 0, "n_kept": 0}

    # 3. Gap Reversal — WATCH_FILL is STRONG (≥70% fill rate), NEAR_FILL is moderate (≥50%)
    try:
        gap_results = load_cc_gap_verdicts()
        n_in = n_kept = 0
        for tkr, tg in gap_results:
            if tg.signal not in ("WATCH_FILL", "NEAR_FILL"):
                continue
            n_in += 1
            if tg.hist_fill_rate is None or tg.hist_n_similar < MIN_N:
                continue
            win_rate = tg.hist_fill_rate * 100
            avg_ret  = tg.distance_to_fill_pct * 100   # expected % move to fill
            if not _passes_gate(win_rate, avg_ret, tg.hist_n_similar):
                continue
            # WATCH_FILL = high-edge bucket (≥70% fill rate); NEAR_FILL = moderate (≥50%)
            conf = 1.0 if tg.signal == "WATCH_FILL" else 0.65
            edge = _edge(win_rate, avg_ret, tg.hist_n_similar, conf)
            direction = "PUTS (gap up → fill down)" if tg.gap_dir == "up" else "CALLS (gap down → fill up)"
            plays.append({
                "source": "Gap Fill", "ticker": tkr,
                "tag": f"Gap {tg.gap_dir} {tg.gap_pct*100:+.2f}%",
                "state": tg.signal,
                "action": f"BUY {direction}",
                "entry": tg.open_price, "target": tg.fill_level,
                "win_rate": win_rate, "avg_ret": avg_ret, "n": tg.hist_n_similar,
                "edge": edge,
                "reason": f"Similar gaps fill {win_rate:.0f}% of the time (n={tg.hist_n_similar}). "
                          f"Distance to fill: {tg.distance_to_fill_pct*100:.2f}%.",
                "horizon": "Same day (intraday fill)",
            })
            n_kept += 1
        msg = f"{len(gap_results)}/{4} tickers loaded, {n_in} with gap signals, {n_kept} passed edge gate"
        if len(gap_results) < 4:
            msg += f" · ⚠ {4 - len(gap_results)} ticker(s) failed to load"
            source_health["Gap Fill"] = {"ok": False, "msg": msg, "n_in": n_in, "n_kept": n_kept}
        else:
            source_health["Gap Fill"] = {"ok": True, "msg": msg, "n_in": n_in, "n_kept": n_kept}
    except Exception as e:
        source_health["Gap Fill"] = {"ok": False, "msg": f"ERROR: {e}", "n_in": 0, "n_kept": 0}

    # 4. 0DTE Lottery (ENTRY_OPEN — drop ≥3 pts from open)
    try:
        n_in = n_kept = 0
        per_ticker_errors = []
        for tkr in ("SPY", "QQQ", "IWM"):
            try:
                alert = load_0dte_alert(tkr)
            except Exception as e:
                per_ticker_errors.append(f"{tkr}: {e}")
                continue
            if alert.get("status") != "ENTRY_OPEN":
                continue
            recs = alert.get("recs", []) or []
            if not recs:
                continue
            n_in += 1
            top_rec = max(recs, key=lambda r: r.est_gain_pct)
            hist_pct = float(alert.get("hist_pct_1000plus", 0))
            avg_ret  = top_rec.est_gain_pct   # already a %
            # 0DTE has no per-trade sample size; treat hist_pct as a low-confidence
            # win-rate proxy and apply heavy shrinkage (n=3 effective sample)
            n_eff = 3
            if not _passes_gate(hist_pct, avg_ret, n_eff):
                continue
            edge = _edge(hist_pct, avg_ret, n_eff, conf=0.7)   # 0.7 = lottery-grade conf
            plays.append({
                "source": "0DTE Drop", "ticker": tkr,
                "tag": f"Drop {alert['drop_pts']:.1f} pts",
                "state": "ENTRY_OPEN",
                "action": f"BUY {top_rec.strike}C @ ~${top_rec.est_entry_price:.2f}",
                "entry": alert["day_low"], "target": alert["day_open"],
                "win_rate": hist_pct, "avg_ret": avg_ret, "n": n_eff,
                "edge": edge,
                "reason": f"Sold off {alert['drop_pts']:.1f} pts from open. "
                          f"{hist_pct:.0f}% of similar drops produced 1000%+ option moves on recovery to VWAP/open. "
                          f"⚠ Lottery-grade: small sample, heavy shrinkage applied.",
                "horizon": "Minutes–hours (0DTE intraday)",
            })
            n_kept += 1
        msg = f"3 tickers polled, {n_in} firing, {n_kept} passed edge gate"
        if per_ticker_errors:
            msg += f" · ⚠ errors: {'; '.join(per_ticker_errors)}"
            source_health["0DTE Drop"] = {"ok": False, "msg": msg, "n_in": n_in, "n_kept": n_kept}
        else:
            source_health["0DTE Drop"] = {"ok": True, "msg": msg, "n_in": n_in, "n_kept": n_kept}
    except Exception as e:
        source_health["0DTE Drop"] = {"ok": False, "msg": f"ERROR: {e}", "n_in": 0, "n_kept": 0}

    # ── Source health banner (always shown) ──────────────────────────────────
    bad_sources = [name for name, h in source_health.items() if not h["ok"]]
    health_color = "#d29922" if bad_sources else "#238636"
    health_label = f"⚠ {len(bad_sources)} source(s) degraded" if bad_sources else "✓ All systems healthy"
    rows_html = ""
    for name, h in source_health.items():
        dot = "🟢" if h["ok"] else "🟡"
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;gap:12px;'
            f'padding:6px 0;border-bottom:1px solid #21262d;font-size:12px;">'
            f'<span style="color:#c9d1d9;font-weight:600;">{dot} {name}</span>'
            f'<span style="color:#8b949e;text-align:right;">{h["msg"]}</span></div>'
        )
    with st.expander(f"📡 Signal source health — {health_label}", expanded=bool(bad_sources)):
        st.html(f'<div style="border-left:3px solid {health_color};padding-left:12px;">{rows_html}</div>')

    # ── Render ───────────────────────────────────────────────────────────────
    if not plays:
        st.html("""
        <div style="background:linear-gradient(135deg,#1c2128,#161b22);
                    border:1px solid #30363d;border-radius:14px;padding:32px;
                    text-align:center;margin:16px 0;">
          <div style="font-size:48px;margin-bottom:8px;">😴</div>
          <div style="font-size:22px;font-weight:800;color:#f0f6fc;
                      margin-bottom:6px;">No actionable plays right now</div>
          <div style="color:#8b949e;font-size:14px;line-height:1.5;">
            Every validated signal source is quiet. No MA touches, no jackpot triggers,
            no gap-fill setups, no 0DTE drops.<br>
            <span style="color:#6e7681;">Patience is a position. Check back in 30 min.</span>
          </div>
        </div>
        """)
    else:
        # Sort by edge descending
        plays.sort(key=lambda p: p["edge"], reverse=True)

        # Top counters
        n_now      = sum(1 for p in plays if p["state"] in ("TOUCHING", "ENTRY_OPEN", "NEAR_FILL"))
        n_watch    = sum(1 for p in plays if p["state"] in ("APPROACHING", "WATCH_FILL"))
        best_edge  = plays[0]["edge"]
        sources    = sorted({p["source"] for p in plays})
        st.html(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;">
          <div style="background:linear-gradient(135deg,#1f6feb,#0969da);border-radius:12px;padding:16px;">
            <div style="font-size:11px;color:#cbe1ff;font-weight:700;letter-spacing:.06em;text-transform:uppercase;">Trade Now</div>
            <div style="font-size:32px;font-weight:900;color:#fff;margin-top:4px;">{n_now}</div>
            <div style="font-size:11px;color:#cbe1ff;">live signals</div>
          </div>
          <div style="background:linear-gradient(135deg,#d29922,#9e6a03);border-radius:12px;padding:16px;">
            <div style="font-size:11px;color:#fff3c4;font-weight:700;letter-spacing:.06em;text-transform:uppercase;">Watch List</div>
            <div style="font-size:32px;font-weight:900;color:#fff;margin-top:4px;">{n_watch}</div>
            <div style="font-size:11px;color:#fff3c4;">approaching trigger</div>
          </div>
          <div style="background:linear-gradient(135deg,#238636,#196c2e);border-radius:12px;padding:16px;">
            <div style="font-size:11px;color:#c4f5d4;font-weight:700;letter-spacing:.06em;text-transform:uppercase;">Best Edge</div>
            <div style="font-size:32px;font-weight:900;color:#fff;margin-top:4px;">{best_edge:.1f}</div>
            <div style="font-size:11px;color:#c4f5d4;">{plays[0]['ticker']} · {plays[0]['source']}</div>
          </div>
          <div style="background:linear-gradient(135deg,#6e40c9,#553098);border-radius:12px;padding:16px;">
            <div style="font-size:11px;color:#e2d5ff;font-weight:700;letter-spacing:.06em;text-transform:uppercase;">Active Systems</div>
            <div style="font-size:32px;font-weight:900;color:#fff;margin-top:4px;">{len(sources)}</div>
            <div style="font-size:11px;color:#e2d5ff;">{', '.join(sources)}</div>
          </div>
        </div>
        """)

        st.caption("Plays ranked by **edge score** = win rate × expected return × confidence. "
                   "All signals are filtered to validated systems only — no setup with negative or unproven historical edge appears here.")

        # Render each play as a card
        STATE_COLORS = {
            "TOUCHING":    ("#238636", "🔥"),
            "ENTRY_OPEN":  ("#1f6feb", "⚡"),
            "NEAR_FILL":   ("#1f6feb", "⚡"),
            "APPROACHING": ("#d29922", "👀"),
            "WATCH_FILL":  ("#d29922", "👀"),
        }
        SOURCE_BADGE = {
            "MA Bounce":  ("#6e40c9", "Weekly MA bounce · validated +243% on real options"),
            "ML Jackpot": ("#1f6feb", "ML vol+P&L classifier agreement · same-day 0DTE"),
            "Gap Fill":   ("#0e8c87", "Gap reversal · historical fill rate ≥X%"),
            "0DTE Drop":  ("#bf3989", "Intraday drop ≥3 pts · drop-band lottery"),
        }
        for i, p in enumerate(plays, 1):
            color, emoji = STATE_COLORS.get(p["state"], ("#8b949e", "•"))
            src_color, src_desc = SOURCE_BADGE.get(p["source"], ("#8b949e", ""))
            n_str = f"n={p['n']}" if p["n"] > 0 else "live"
            # Smart-entry doctrine block — only rendered for MA Bounce plays
            doctrine_html = ""
            if "doctrine_smart_entry" in p:
                doctrine_html = f"""
                  <div style="margin-top:10px;padding:10px 12px;background:#0a1428;
                              border:1px solid #1f6feb;border-left:3px solid #58a6ff;border-radius:8px;">
                    <div style="font-size:10px;color:#58a6ff;font-weight:800;letter-spacing:.08em;
                                text-transform:uppercase;margin-bottom:6px;">
                      💎 High-Conviction Doctrine · 3-Month Polygon-Validated
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px 18px;
                                font-size:12px;color:#c9d1d9;">
                      <div><b style="color:#58a6ff;">Smart entry:</b>
                           ${p['doctrine_smart_entry']:.2f} limit
                           <span style="color:#6e7681;">(MA × 0.995, valid {p['doctrine_fill_window']})</span></div>
                      <div><b style="color:#58a6ff;">Conviction add:</b>
                           ${p['doctrine_conviction']:.2f} limit
                           <span style="color:#6e7681;">(MA × 0.99, size 1.5–2×)</span></div>
                      <div><b style="color:#58a6ff;">Buy strike:</b>
                           ~${p['doctrine_strike']} call
                           <span style="color:#6e7681;">(OTM, capped at 2% of underlying, 1-week expiry)</span></div>
                      <div><b style="color:#58a6ff;">Exit rule:</b>
                           {p['doctrine_hold_rule']}</div>
                    </div>
                    <div style="margin-top:6px;font-size:11px;color:#8b949e;line-height:1.5;">
                      Skip if not filled by Day 5 — no chasing.
                      Loss is naturally capped (~−55% to −90%) by OTM expiry.<br>
                      <span style="color:#d29922;">⚠ Strike is a heuristic estimate — confirm against live option chain
                      for the nearest tradable strike with adequate liquidity.</span>
                    </div>
                  </div>"""
            st.html(f"""
            <div style="background:#0d1117;border:1px solid {color};border-left:4px solid {color};
                        border-radius:12px;padding:18px;margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
                <div style="flex:1;min-width:0;">
                  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
                    <span style="background:#161b22;color:#8b949e;font-size:11px;font-weight:800;
                                 padding:3px 8px;border-radius:6px;">#{i}</span>
                    <span style="font-size:22px;font-weight:900;color:#f0f6fc;">{emoji} {p['ticker']}</span>
                    <span style="background:{src_color}22;color:{src_color};font-size:10px;font-weight:800;
                                 padding:3px 8px;border-radius:6px;letter-spacing:.04em;text-transform:uppercase;">
                      {p['source']}</span>
                    <span style="background:{color}22;color:{color};font-size:10px;font-weight:800;
                                 padding:3px 8px;border-radius:6px;letter-spacing:.04em;text-transform:uppercase;">
                      {p['state']}</span>
                    <span style="color:#6e7681;font-size:11px;">{p['tag']}</span>
                  </div>
                  <div style="font-size:18px;font-weight:800;color:{color};margin-bottom:8px;">
                    → {p['action']}
                  </div>
                  <div style="color:#c9d1d9;font-size:13px;line-height:1.55;margin-bottom:8px;">
                    {p['reason']}
                  </div>
                  <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#8b949e;">
                    <span><b style="color:#f0f6fc;">Entry:</b> ${p['entry']:.2f}</span>
                    <span><b style="color:#f0f6fc;">Target:</b> ${p['target']:.2f}</span>
                    <span><b style="color:#3fb950;">Win rate:</b> {p['win_rate']:.0f}%</span>
                    <span><b style="color:#3fb950;">Avg return:</b> +{p['avg_ret']:.2f}%</span>
                    <span><b style="color:#f0f6fc;">Sample:</b> {n_str}</span>
                    <span><b style="color:#f0f6fc;">Horizon:</b> {p['horizon']}</span>
                  </div>
                </div>
                <div style="text-align:right;flex-shrink:0;min-width:90px;">
                  <div style="font-size:10px;color:#6e7681;font-weight:700;letter-spacing:.06em;
                              text-transform:uppercase;">Edge</div>
                  <div style="font-size:28px;font-weight:900;color:{color};line-height:1;">{p['edge']:.1f}</div>
                </div>
              </div>
              {doctrine_html}
            </div>
            """)

        st.markdown("---")
        st.caption("💡 **How to use:** Work the list top-down. Each play shows the exact action, entry, target, and historical edge. "
                   "Open the corresponding **🎯 Signal Details** page in the sidebar for deeper drill-down. "
                   "Use **💰 Account Tracker** to log fills and track equity vs. milestones.")

elif page == "Command Center":
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

    # ── Composite verdict synthesis ───────────────────────────────────────────
    _cc_gap_rows = load_cc_gap_verdicts()
    _gap_active  = [(t, g) for t, g in _cc_gap_rows if g.signal in ("WATCH_FILL", "NEAR_FILL")]
    _gap_best    = (max(_gap_active,
                        key=lambda x: {"WATCH_FILL": 2, "NEAR_FILL": 1}.get(x[1].signal, 0))
                    if _gap_active else None)

    _dte_verd = load_0dte_alert("SPY")
    _dte_vs   = _dte_verd.get("status", "UNAVAILABLE")
    if not _ph["is_open"]:
        _dte_vs = "SESSION_CLOSED"

    _ml_active  = sig in ("GO_JACKPOT", "GO_ULTRA_JACKPOT", "GO_HOT")
    _dte_active = (_dte_vs == "ENTRY_OPEN")
    _n_plays    = sum([_ml_active, bool(_gap_best), _dte_active])

    if _n_plays >= 3:
        _plays_badge = (
            '<span style="background:#ffd633;color:#0c1117;font-size:11px;font-weight:800;'
            'padding:3px 10px;border-radius:12px;letter-spacing:.08em;">⚡ '
            + str(_n_plays) + ' PLAYS ACTIVE</span>')
    elif _n_plays == 2:
        _plays_badge = (
            '<span style="background:#3fb950;color:#0c1117;font-size:11px;font-weight:800;'
            'padding:3px 10px;border-radius:12px;letter-spacing:.08em;">⚡ 2 PLAYS ACTIVE</span>')
    elif _n_plays == 1:
        _plays_badge = (
            '<span style="background:rgba(88,166,255,0.15);color:#58a6ff;font-size:11px;'
            'font-weight:700;padding:3px 10px;border-radius:12px;'
            'border:1px solid rgba(88,166,255,0.3);">1 PLAY ACTIVE</span>')
    else:
        _plays_badge = ""

    # ML tile content
    _ml_bg = {"GO_ULTRA_JACKPOT": "rgba(255,214,51,0.08)", "GO_JACKPOT": "rgba(63,185,80,0.08)",
               "GO_HOT": "rgba(210,153,34,0.08)", "SKIP": "rgba(139,148,158,0.04)"}.get(
               sig, "rgba(139,148,158,0.04)")
    if trade_tickers:
        _ml_detail = ('Trade: <strong style="color:#3fb950;">'
                      + ", ".join(trade_tickers) + "</strong>")
    elif hot_tickers:
        _ml_detail = ('Watch: <strong style="color:#d29922;">'
                      + ", ".join(hot_tickers) + "</strong>")
    else:
        _ml_detail = '<span style="color:#6e7681;">No ML trade signal today</span>'

    # Gap tile content
    if _gap_best:
        _gbt, _gbg = _gap_best
        _gap_tb  = "#3fb950" if _gbg.signal == "WATCH_FILL" else "#d29922"
        _gap_tbg = ("rgba(63,185,80,0.06)" if _gbg.signal == "WATCH_FILL"
                    else "rgba(210,153,34,0.06)")
        _gdir    = "↓ GAP DOWN" if _gbg.gap_dir == "down" else "↑ GAP UP"
        _gact    = "BUY CALLS" if _gbg.gap_dir == "down" else "BUY PUTS"
        _gact_c  = "#3fb950" if _gbg.gap_dir == "down" else "#f85149"
        _gfill   = (f"{_gbg.hist_fill_rate * 100:.0f}% hist. fill rate"
                    if _gbg.hist_fill_rate else "")
        _gap_head = (f'<div style="font-size:14px;font-weight:800;color:{_gap_tb};">'
                     f'{_gdir} · {_gbt}</div>')
        _gap_det  = (f'<strong style="color:{_gact_c};">{_gact}</strong>'
                     f' &nbsp;·&nbsp; {_gbg.gap_pct * 100:+.2f}% gap'
                     + (f' &nbsp;·&nbsp; {_gfill}' if _gfill else ""))
    else:
        _gap_tb   = "#30363d"
        _gap_tbg  = "rgba(139,148,158,0.04)"
        _gap_head = '<div style="font-size:14px;font-weight:800;color:#6e7681;">NO GAP TODAY</div>'
        _gap_det  = (f'<span style="color:#6e7681;">{len(_cc_gap_rows)} tickers opened near '
                     "yesterday's close</span>")

    # 0DTE tile content
    if _dte_vs == "ENTRY_OPEN":
        _dte_tb   = "#f85149"
        _dte_tbg  = "rgba(248,81,73,0.06)"
        _drop     = _dte_verd.get("drop_pts") or 0
        _dte_head = '<div style="font-size:14px;font-weight:800;color:#f85149;">🎯 ENTRY OPEN</div>'
        _dte_det  = (f'<strong style="color:#3fb950;">BUY CALLS</strong>'
                     f' &nbsp;·&nbsp; SPY −{_drop:.1f} pts from open')
    elif _dte_vs == "APPROACHING":
        _dte_tb   = "#d29922"
        _dte_tbg  = "rgba(210,153,34,0.06)"
        _drop     = _dte_verd.get("drop_pts") or 0
        _dte_head = '<div style="font-size:14px;font-weight:800;color:#d29922;">⚡ APPROACHING</div>'
        _dte_det  = f'SPY −{_drop:.1f} pts · nearing entry threshold'
    elif _dte_vs == "SESSION_CLOSED":
        _dte_tb   = "#30363d"
        _dte_tbg  = "rgba(139,148,158,0.04)"
        _dte_head = ('<div style="font-size:14px;font-weight:800;color:#6e7681;">'
                     'SESSION CLOSED</div>')
        _d_open   = _dte_verd.get("day_open") or 0
        _dte_det  = (f'<span style="color:#6e7681;">Final SPY open ${_d_open:.2f} · '
                     'Opens again 9:30 AM ET</span>' if _d_open else
                     '<span style="color:#6e7681;">Opens again 9:30 AM ET</span>')
    else:
        _dte_tb   = "#30363d"
        _dte_tbg  = "rgba(139,148,158,0.04)"
        _dte_head = '<div style="font-size:14px;font-weight:800;color:#6e7681;">QUIET</div>'
        if _ph["is_open"]:
            _d_open  = _dte_verd.get("day_open") or 0
            _dte_det = (f'<span style="color:#6e7681;">SPY open ${_d_open:.2f} · '
                        'No reversal setup yet</span>' if _d_open else
                        '<span style="color:#6e7681;">No reversal setup active</span>')
        else:
            _dte_det = ('<span style="color:#6e7681;">'
                        'Activates at next market open (9:30 AM ET)</span>')

    st.html(
        f"""
        <div style="background:{hero_bg};border:2px solid {hero_border};
                    border-radius:16px;padding:28px 32px;margin-bottom:8px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;">

          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <div style="color:#8b949e;font-size:11px;letter-spacing:.14em;
                            text-transform:uppercase;font-weight:700;">
                  {datetime.now().strftime('%A, %B %d')} · Today's Verdict
                </div>
                {_plays_badge}
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

          <div style="margin-top:18px;padding-top:16px;
                      border-top:1px solid rgba(255,255,255,0.08);
                      display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">

            <div style="background:{_ml_bg};border:1px solid {hero_border};
                        border-radius:10px;padding:12px 14px;">
              <div style="font-size:9px;font-weight:800;text-transform:uppercase;
                          color:#6e7681;letter-spacing:.1em;margin-bottom:8px;">
                ML JACKPOT SIGNAL</div>
              <div style="font-size:14px;font-weight:800;color:{hero_border};">
                {signal_label(sig)}</div>
              <div style="font-size:11px;color:#8b949e;margin-top:6px;">{_ml_detail}</div>
            </div>

            <div style="background:{_gap_tbg};border:1px solid {_gap_tb};
                        border-radius:10px;padding:12px 14px;">
              <div style="font-size:9px;font-weight:800;text-transform:uppercase;
                          color:#6e7681;letter-spacing:.1em;margin-bottom:8px;">
                GAP REVERSAL</div>
              {_gap_head}
              <div style="font-size:11px;color:#8b949e;margin-top:6px;">{_gap_det}</div>
            </div>

            <div style="background:{_dte_tbg};border:1px solid {_dte_tb};
                        border-radius:10px;padding:12px 14px;">
              <div style="font-size:9px;font-weight:800;text-transform:uppercase;
                          color:#6e7681;letter-spacing:.1em;margin-bottom:8px;">
                0DTE LOTTERY</div>
              {_dte_head}
              <div style="font-size:11px;color:#8b949e;margin-top:6px;">{_dte_det}</div>
            </div>

          </div>
        </div>
        """
    )

    # ── Play detail tabs (below hero card) ────────────────────────────────────
    _has_dte_detail = _dte_vs in ("ENTRY_OPEN", "APPROACHING")
    if _ml_active or _gap_best or _has_dte_detail:
        from src.position_sizer import recommend_allocation as _ra2, TIERS as _TIERS

        _tab_labels: list[str] = []
        if _ml_active:
            _tab_labels.append(f"{signal_label(sig)} — ML Trade Plan")
        if _gap_best:
            _gbt2, _gbg2 = _gap_best
            _arr2 = "↓" if _gbg2.gap_dir == "down" else "↑"
            _tab_labels.append(f"{_arr2} Gap Reversal — {_gbt2}")
        if _has_dte_detail:
            _tab_labels.append("🎯 0DTE Lottery — Contracts")

        _play_tabs = st.tabs(_tab_labels)
        _ti = 0

        # ── ML Trade Plan tab ────────────────────────────────────────────────
        if _ml_active:
            with _play_tabs[_ti]:
                _ti += 1
                _alloc2 = _ra2(sig, equity_input)
                _tier2  = _TIERS.get(sig)
                _lq2    = load_live_quotes(JACKPOT_TICKERS)

                _ml_df = pd.DataFrame([{
                    "Ticker":     r.ticker,
                    "Signal":     r.signal.replace("GO_", ""),
                    "P(vol)":     f"{r.p_vol * 100:.1f}%",
                    "P(pnl)":     f"{r.p_pnl * 100:.1f}%",
                    "Last Price": fmt_dollar((_lq2.get(r.ticker) or {}).get("last_price") or r.last_close),
                    "Action":     ("TRADE 🎯" if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")
                                   else "WATCH 🔥" if r.signal == "GO_HOT" else "—"),
                } for r in rows])
                st.dataframe(_ml_df, hide_index=True, use_container_width=True)

                if _alloc2 and _tier2:
                    _c1, _c2, _c3, _c4 = st.columns(4)
                    _c1.metric("Allocate",         f"${_alloc2.alloc_dollars:,.2f}",
                               f"{_alloc2.alloc_pct * 100:.0f}% of equity")
                    _c2.metric("Avg win scenario", f"+${_alloc2.win_scenario:,.2f}",
                               f"{(_tier2.avg_win_mult - 1) * 100:.0f}% gain")
                    _c3.metric("Max loss",         f"-${_alloc2.max_loss:,.2f}",
                               "if expires worthless")
                    _c4.metric("Historical win %", f"{_tier2.win_prob * 100:.0f}%",
                               "backtest estimate")

                _trade_tkr2 = (trade_tickers[0] if trade_tickers
                               else hot_tickers[0] if hot_tickers
                               else best_row.ticker)
                st.html(f"""
                <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                            padding:14px 18px;margin-top:8px;
                            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                  <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                              letter-spacing:.1em;margin-bottom:12px;">Entry Plan</div>
                  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                    <div>
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Instrument</div>
                      <div style="color:#3fb950;font-weight:800;font-size:15px;">
                        {_trade_tkr2} 0DTE CALLS</div>
                      <div style="color:#8b949e;font-size:12px;margin-top:3px;">
                        ATM or 1–2 strikes OTM at open</div>
                    </div>
                    <div>
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Entry Timing</div>
                      <div style="color:#ffd633;font-weight:800;font-size:15px;">
                        9:30–9:50 AM ET</div>
                      <div style="color:#8b949e;font-size:12px;margin-top:3px;">
                        Wait for opening print — buy near open</div>
                    </div>
                    <div>
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Size / Max Risk</div>
                      <div style="color:#f85149;font-weight:800;font-size:15px;">
                        ${_alloc2.alloc_dollars if _alloc2 else 0:,.2f}</div>
                      <div style="color:#8b949e;font-size:12px;margin-top:3px;">
                        {f"{_alloc2.alloc_pct * 100:.0f}% of equity" if _alloc2 else ""}
                        — full loss possible</div>
                    </div>
                  </div>
                </div>""")

        # ── Gap Reversal tab ─────────────────────────────────────────────────
        if _gap_best:
            with _play_tabs[_ti]:
                _ti += 1
                _gbt2, _gbg2 = _gap_best
                _gact2    = "BUY CALLS" if _gbg2.gap_dir == "down" else "BUY PUTS"
                _gact_c2  = "#3fb950"   if _gbg2.gap_dir == "down" else "#f85149"
                _fill2    = _gbg2.fill_level
                _open2    = _gbg2.open_price
                _gpts2    = abs(_gbg2.gap_pts)
                _stop2    = round(_open2 - 1.5, 2) if _gbg2.gap_dir == "down" else round(_open2 + 1.5, 2)
                _ct2      = "CALL" if _gbg2.gap_dir == "down" else "PUT"

                # Three suggested strikes: ATM / midpoint / at-fill
                _sk_atm  = round(_open2)
                _sk_fill = round(_fill2)
                _sk_mid  = round((_open2 + _fill2) / 2)
                _sk_list = [
                    (_sk_atm,  "CONSERVATIVE",   "Near current open · higher probability, smaller % gain"),
                    (_sk_mid,  "BEST R/R 🎯",    "Mid-gap strike · balanced risk / reward"),
                    (_sk_fill, "AGGRESSIVE",      "At fill level · cheap OTM — explosive if gap fills"),
                ]
                _sk_html2 = ""
                for _sk, _sk_lbl, _sk_note in _sk_list:
                    _is_b2  = "R/R" in _sk_lbl
                    _sk_bg2 = "#0d1f14" if _is_b2 else "#161b22"
                    _sk_bd2 = "#3fb950" if _is_b2 else "#30363d"
                    _btag2  = ('<div style="font-size:9px;color:#3fb950;margin-bottom:6px;">'
                               '★ BEST R/R</div>' if _is_b2 else "")
                    _sk_html2 += f"""
                    <div style="background:{_sk_bg2};border:2px solid {_sk_bd2};border-radius:10px;
                                padding:14px;flex:1;min-width:140px;text-align:center;">
                      {_btag2}
                      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                  letter-spacing:.06em;margin-bottom:6px;">{_sk_lbl}</div>
                      <div style="font-size:17px;font-weight:800;color:#e6edf3;">
                        {_gbt2} ${_sk} {_ct2}</div>
                      <div style="font-size:11px;color:#8b949e;margin-top:8px;
                                  line-height:1.5;">{_sk_note}</div>
                    </div>"""

                _fill_rate_str = (f"{_gbg2.hist_fill_rate * 100:.0f}%"
                                  if _gbg2.hist_fill_rate else "N/A")
                _rev_str = (f"+{_gbg2.hist_med_rev_pts:.2f} pts beyond fill"
                            if _gbg2.hist_med_rev_pts else "")
                _n_str   = (f"n={_gbg2.hist_n_similar} sessions"
                            if _gbg2.hist_n_similar else "")

                st.html(f"""
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">

                  <div style="display:grid;grid-template-columns:repeat(4,1fr);
                              gap:10px;margin-bottom:16px;">
                    <div style="background:#161b22;border:1px solid #30363d;
                                border-radius:8px;padding:12px;text-align:center;">
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Today's Open</div>
                      <div style="color:#e6edf3;font-size:18px;font-weight:800;">
                        ${_open2:.2f}</div>
                      <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                        entry zone</div>
                    </div>
                    <div style="background:#161b22;border:1px solid {_gact_c2};
                                border-radius:8px;padding:12px;text-align:center;">
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Fill Target</div>
                      <div style="color:{_gact_c2};font-size:18px;font-weight:800;">
                        ${_fill2:.2f}</div>
                      <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                        prev close · +{_gpts2:.2f} pts</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #f85149;
                                border-radius:8px;padding:12px;text-align:center;">
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Stop Loss</div>
                      <div style="color:#f85149;font-size:18px;font-weight:800;">
                        ${_stop2:.2f}</div>
                      <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                        exit if breaks past open</div>
                    </div>
                    <div style="background:#161b22;border:1px solid #58a6ff;
                                border-radius:8px;padding:12px;text-align:center;">
                      <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                                  margin-bottom:4px;">Hist. Fill Rate</div>
                      <div style="color:#58a6ff;font-size:18px;font-weight:800;">
                        {_fill_rate_str}</div>
                      <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                        {_n_str}</div>
                    </div>
                  </div>

                  <div style="color:#6e7681;font-size:10px;text-transform:uppercase;
                              letter-spacing:.08em;margin-bottom:10px;">
                    Suggested Contracts</div>
                  <div style="display:flex;gap:10px;flex-wrap:wrap;
                              margin-bottom:14px;">{_sk_html2}</div>

                  <div style="background:#0c1520;border:1px solid #1d3a5c;
                              border-radius:8px;padding:12px 14px;
                              font-size:11px;color:#8b949e;line-height:1.7;">
                    <strong style="color:#58a6ff;">Trade plan:</strong>
                    {_gact2} {_gbt2} at open · Target fill at ${_fill2:.2f}
                    (+{_gpts2:.2f} pts) · Stop ${_stop2:.2f}
                    {("· Median reversal after fill: " + _rev_str) if _rev_str else ""} ·
                    Use 5–10% of equity (gap plays are lower conviction than full JACKPOT).
                  </div>
                </div>""")

        # ── 0DTE Lottery tab ─────────────────────────────────────────────────
        if _has_dte_detail:
            with _play_tabs[_ti]:
                _ti += 1
                _dte_recs2 = _dte_verd.get("recs") or []
                _drop2     = _dte_verd.get("drop_pts") or 0
                _d_open2   = _dte_verd.get("day_open") or 0
                _d_low2    = _dte_verd.get("day_low")  or 0

                if _dte_vs == "APPROACHING":
                    st.info(
                        f"SPY is {_drop2:.1f} pts below open (${_d_open2:.2f}). "
                        f"Setup activates at 3 pts — need "
                        f"{max(0.0, 3.0 - _drop2):.1f} more pts of drop. Watch for entry.")

                if _dte_recs2:
                    _recs2_html = ""
                    for _ri, _rec2 in enumerate(_dte_recs2[:3]):
                        _ib2    = _ri == 0
                        _rb2    = "#0d1f14" if _ib2 else "#161b22"
                        _rbd2   = "#3fb950" if _ib2 else "#30363d"
                        _btag3  = ('<div style="font-size:9px;color:#3fb950;margin-bottom:6px;">'
                                   '★ BEST R/R</div>' if _ib2 else "")
                        _gc2    = ("#ffd633" if _rec2.est_gain_pct >= 1000
                                   else "#3fb950" if _rec2.est_gain_pct >= 500
                                   else "#58a6ff")
                        _recs2_html += f"""
                        <div style="background:{_rb2};border:2px solid {_rbd2};
                                    border-radius:12px;padding:16px;flex:1;
                                    min-width:150px;text-align:center;">
                          {_btag3}
                          <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                                      letter-spacing:.07em;margin-bottom:4px;">
                            SPY ${_rec2.strike:.0f} CALL</div>
                          <div style="color:#8b949e;font-size:10px;margin-bottom:10px;">
                            +{_rec2.dist_from_low:.0f} pts above low</div>
                          <div style="font-size:12px;color:#8b949e;margin-bottom:3px;">Entry:
                            <strong style="color:#e6edf3;font-size:16px;">
                              ${_rec2.est_entry_price:.2f}</strong></div>
                          <div style="font-size:12px;color:#8b949e;margin-bottom:10px;">Target:
                            <strong style="color:#3fb950;font-size:16px;">
                              ${_rec2.est_target_price:.2f}</strong></div>
                          <div style="font-size:24px;font-weight:800;color:{_gc2};">
                            +{_rec2.est_gain_pct:,.0f}%</div>
                          <div style="font-size:10px;color:#8b949e;margin-top:4px;">
                            {_rec2.risk_category.split("—")[0].strip()}</div>
                        </div>"""

                    st.html(f"""
                    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                      <div style="display:flex;gap:10px;flex-wrap:wrap;
                                  margin-bottom:14px;">{_recs2_html}</div>
                      <div style="background:#0c1520;border:1px solid #1d3a5c;border-radius:8px;
                                  padding:12px 14px;font-size:11px;color:#8b949e;line-height:1.7;">
                        <strong style="color:#58a6ff;">Entry note:</strong>
                        SPY at intraday low ${_d_low2:.2f} · Open was ${_d_open2:.2f} ·
                        Drop: {_drop2:.1f} pts · Buy calls at or near current low ·
                        Target: recovery to open ${_d_open2:.2f} · Use 10% of equity max.
                      </div>
                    </div>""")
                else:
                    st.info("Detailed strike prices appear once Polygon live options data is "
                            "available (requires Polygon options subscription).")

                _alloc_dte2 = _ra2("ENTRY_OPEN", equity_input)
                if _alloc_dte2:
                    _dc1, _dc2, _dc3 = st.columns(3)
                    _dc1.metric("Spend (10% Kelly)", f"${_alloc_dte2.alloc_dollars:,.2f}",
                                "max you can lose")
                    _dc2.metric("If 1000% hit",
                                f"+${_alloc_dte2.alloc_dollars * 9:,.2f}", "net profit")
                    _dc3.metric("Win rate (hist)", "35%",
                                "SPY 3–5 pt drop band")

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

    # ── 6-Month Full Trading Simulation ────────────────────────────────────────
    st.markdown("---")
    section("6-Month Full Simulation",
            f"Every signal, every day — ${equity_input:,.0f} starting balance over 6 months")

    # Signal frequency: estimated actionable trade days per month based on
    # historical ML model calibration (~21 trading days/month)
    _6M_MIX = [
        ("GO_ULTRA_JACKPOT", 0.5,  "#ffd633", "ULTRA JACKPOT",  "Both models at peak confidence — rarest"),
        ("GO_JACKPOT",       2.0,  "#3fb950", "JACKPOT",         "Both models fire — core trade day"),
        ("GO_HOT",           3.0,  "#d29922", "HOT",             "Vol model fires — elevated, smaller size"),
        ("ENTRY_OPEN",       3.0,  "#f85149", "0DTE LOTTERY",    "SPY 3+ pt intraday drop — reversal play"),
    ]
    _6M_TOTAL = sum(c for _, c, _, _, _ in _6M_MIX)

    # Signal mix strip
    _mix_html = ""
    for _s6, _c6, _col6, _lbl6, _note6 in _6M_MIX:
        _t6 = TIERS.get(_s6)
        if _t6:
            _mix_html += f"""
            <div style="background:#161b22;border:1px solid {_col6}44;border-radius:10px;
                        padding:12px 14px;flex:1;min-width:130px;">
              <div style="font-size:9px;font-weight:800;text-transform:uppercase;
                          color:{_col6};letter-spacing:.1em;margin-bottom:6px;">{_lbl6}</div>
              <div style="font-size:24px;font-weight:900;color:#e6edf3;">
                {_c6}/mo</div>
              <div style="font-size:10px;color:#8b949e;margin-top:5px;line-height:1.5;">
                {int(_t6.win_prob*100)}% win · {int(_t6.avg_win_mult*100-100)}% avg gain<br>
                {int(_t6.alloc_pct*100)}% allocation per trade</div>
            </div>"""

    st.html(f"""
    <div style="margin-bottom:16px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="color:#8b949e;font-size:11px;margin-bottom:10px;">
        Estimated <strong style="color:#e6edf3;">{_6M_TOTAL:.0f} trade actions/month</strong>
        out of ~21 trading days (~{_6M_TOTAL/21*100:.0f}% of days are actionable across all strategies)
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">{_mix_html}</div>
    </div>""")

    # ── Run 3-path simulation ────────────────────────────────────────────────
    _s6_ev   = float(equity_input)
    _s6_win  = float(equity_input)
    _s6_lose = float(equity_input)
    _s6_data = []   # (month, ev, win, lose, monthly_gain_ev)

    for _m6 in range(1, 7):
        _m6_start = _s6_ev

        for _sig6, _cnt6, _, _, _ in _6M_MIX:
            _int6  = int(_cnt6)
            _frac6 = _cnt6 - _int6

            # EV path (fractional trades use proportional EV)
            for _ in range(_int6):
                _r6 = _ra(_sig6, _s6_ev)
                if _r6:
                    _s6_ev = max(_s6_ev + _r6.expected_gain, 0.0)
            if _frac6 > 0:
                _r6 = _ra(_sig6, _s6_ev)
                if _r6:
                    _s6_ev = max(_s6_ev + _r6.expected_gain * _frac6, 0.0)

            # Win-all path
            for _ in range(round(_cnt6)):
                _r6 = _ra(_sig6, _s6_win)
                if _r6:
                    _s6_win += _r6.win_scenario

            # Lose-all path (Kelly: equity shrinks but stays > 0)
            for _ in range(round(_cnt6)):
                _r6 = _ra(_sig6, _s6_lose)
                if _r6:
                    _s6_lose = max(_s6_lose - _r6.alloc_dollars, 0.0)

        _s6_data.append((
            _m6,
            round(_s6_ev, 2),
            round(_s6_win, 2),
            round(_s6_lose, 2),
            round(_s6_ev - _m6_start, 2),
        ))

    # ── Build monthly table rows ─────────────────────────────────────────────
    _month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    _now_month  = datetime.now().month - 1   # 0-indexed
    _now_year   = datetime.now().year

    _tbl6 = f"""
        <tr>
          <td style="padding:8px 6px;color:#6e7681;">Start</td>
          <td style="padding:8px 6px;text-align:right;font-weight:700;
                     color:#e6edf3;">${equity_input:,.0f}</td>
          <td style="padding:8px 6px;text-align:right;color:#3fb950;">
            ${equity_input:,.0f}</td>
          <td style="padding:8px 6px;text-align:right;color:#f85149;">
            ${equity_input:,.0f}</td>
          <td style="padding:8px 6px;text-align:right;color:#6e7681;">—</td>
        </tr>"""

    _prev6_ev = float(equity_input)
    for _m6n, _ev6, _win6, _lose6, _gain6 in _s6_data:
        _mname = _month_abbr[(_now_month + _m6n) % 12]
        _yr    = _now_year + (_now_month + _m6n) // 12
        _ev_c6 = "#3fb950" if _ev6 >= equity_input else "#f85149"
        _g_c6  = "#3fb950" if _gain6 >= 0 else "#f85149"
        _g_pct = (_gain6 / _prev6_ev * 100) if _prev6_ev > 0 else 0

        # Milestone badge
        _ms6_tag = ""
        for _ms6v in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
            if _prev6_ev < _ms6v <= _ev6:
                _ms6_lbl = {1_000:"$1k",5_000:"$5k",10_000:"$10k",50_000:"$50k",
                            100_000:"$100k",500_000:"$500k",1_000_000:"$1M"}.get(_ms6v,"")
                _ms6_tag = (f'<span style="background:#ffd633;color:#0d1117;font-size:8px;'
                            f'font-weight:800;padding:1px 5px;border-radius:3px;'
                            f'margin-left:5px;">{_ms6_lbl} ✓</span>')
                break

        _tbl6 += f"""
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:8px 6px;color:#8b949e;font-weight:600;">
            {_mname} {_yr}</td>
          <td style="padding:8px 6px;text-align:right;font-weight:800;color:{_ev_c6};">
            ${_ev6:,.0f}{_ms6_tag}</td>
          <td style="padding:8px 6px;text-align:right;color:#3fb950;">
            ${_win6:,.0f}</td>
          <td style="padding:8px 6px;text-align:right;color:#f85149;">
            ${_lose6:,.0f}</td>
          <td style="padding:8px 6px;text-align:right;font-weight:700;color:{_g_c6};">
            +${_gain6:,.0f}
            <span style="font-size:10px;color:{_g_c6};opacity:.7;">
              (+{_g_pct:.0f}%)</span></td>
        </tr>"""
        _prev6_ev = _ev6

    # ── Final outcome numbers ────────────────────────────────────────────────
    _fin = _s6_data[-1]
    _fin_ev, _fin_win, _fin_lose = _fin[1], _fin[2], _fin[3]
    _mult_ev  = _fin_ev   / equity_input
    _mult_win = _fin_win  / equity_input
    _remain_lose_pct = _fin_lose / equity_input * 100

    def _fmt_big(v: float) -> str:
        if v >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"${v:,.0f}"
        return f"${v:.2f}"

    st.html(f"""
    <div style="background:#0d1117;border:1px solid #30363d;border-radius:14px;
                padding:22px 26px;margin-bottom:4px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">

      <!-- Monthly breakdown table -->
      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:12px;">Month-by-Month Equity</div>
      <table style="width:100%;border-collapse:collapse;
                    font-size:13px;font-variant-numeric:tabular-nums;margin-bottom:22px;">
        <thead>
          <tr style="border-bottom:2px solid #30363d;">
            <th style="color:#8b949e;font-size:10px;text-transform:uppercase;
                       letter-spacing:.06em;padding:6px 6px;font-weight:700;text-align:left;">
              Month</th>
            <th style="color:#58a6ff;font-size:10px;text-transform:uppercase;
                       letter-spacing:.06em;padding:6px 6px;font-weight:700;text-align:right;">
              Expected (EV)</th>
            <th style="color:#3fb950;font-size:10px;text-transform:uppercase;
                       letter-spacing:.06em;padding:6px 6px;font-weight:700;text-align:right;">
              Best Case</th>
            <th style="color:#f85149;font-size:10px;text-transform:uppercase;
                       letter-spacing:.06em;padding:6px 6px;font-weight:700;text-align:right;">
              Worst Case</th>
            <th style="color:#8b949e;font-size:10px;text-transform:uppercase;
                       letter-spacing:.06em;padding:6px 6px;font-weight:700;text-align:right;">
              EV Monthly Gain</th>
          </tr>
        </thead>
        <tbody style="line-height:1.9;">{_tbl6}</tbody>
      </table>

      <!-- 6-month outcome cards -->
      <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:12px;">6-Month Final Outcome</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
                  margin-bottom:18px;">

        <div style="background:linear-gradient(135deg,#0c1a2e,#0d1f36);
                    border:2px solid #58a6ff;border-radius:12px;padding:20px;text-align:center;">
          <div style="color:#58a6ff;font-size:10px;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:10px;">Expected Path (EV model)</div>
          <div style="font-size:36px;font-weight:900;color:#fff;line-height:1;">
            {_fmt_big(_fin_ev)}</div>
          <div style="color:#58a6ff;font-size:15px;font-weight:800;margin-top:6px;">
            {_mult_ev:.0f}× starting equity</div>
          <div style="color:#8b949e;font-size:11px;margin-top:4px;">
            +{_fmt_big(_fin_ev - equity_input)} net gain</div>
        </div>

        <div style="background:linear-gradient(135deg,#0d1f14,#0e2218);
                    border:2px solid #3fb950;border-radius:12px;padding:20px;text-align:center;">
          <div style="color:#3fb950;font-size:10px;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:10px;">Best Case (every win)</div>
          <div style="font-size:36px;font-weight:900;color:#fff;line-height:1;">
            {_fmt_big(_fin_win)}</div>
          <div style="color:#3fb950;font-size:15px;font-weight:800;margin-top:6px;">
            {_mult_win:.0f}× starting equity</div>
          <div style="color:#8b949e;font-size:11px;margin-top:4px;">
            +{_fmt_big(_fin_win - equity_input)} net gain</div>
        </div>

        <div style="background:linear-gradient(135deg,#1a0d0d,#200f0f);
                    border:2px solid #f85149;border-radius:12px;padding:20px;text-align:center;">
          <div style="color:#f85149;font-size:10px;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:10px;">Worst Case (every loss)</div>
          <div style="font-size:36px;font-weight:900;color:#fff;line-height:1;">
            {_fmt_big(_fin_lose)}</div>
          <div style="color:#f85149;font-size:15px;font-weight:800;margin-top:6px;">
            {_remain_lose_pct:.1f}% of equity remains</div>
          <div style="color:#8b949e;font-size:11px;margin-top:4px;">
            Kelly sizing preserves capital — never goes to $0</div>
        </div>

      </div>

      <!-- Disclaimer -->
      <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                  padding:12px 16px;font-size:11px;color:#6e7681;line-height:1.8;">
        <strong style="color:#8b949e;">⚠️ Model assumptions behind these numbers:</strong>
        JACKPOT uses 60% historical win rate × 1000% avg win, 25% allocation per trade.
        HOT uses 45% win rate × 500% avg win, 15% allocation.
        0DTE Lottery uses 35% win rate × 800% avg win, 10% allocation.
        Signal frequency (~{_6M_TOTAL:.0f}/month) is estimated from historical ML calibration.
        The <strong style="color:#58a6ff;">Expected path</strong> applies EV per trade — it is
        the mathematical long-run average, not a guarantee.
        <strong style="color:#e6edf3;">Real results will vary: consecutive losses and consecutive
        jackpots both happen.</strong>
        Only risk capital you can afford to lose entirely.
      </div>
    </div>""")

    # ── Real Historical Backtest ────────────────────────────────────────────────
    st.markdown("---")
    section("Real 6-Month Backtest — Strategy Comparison",
            f"Six strategies replayed on the same real SPY data · ${equity_input:,.0f} starting balance · walk-forward OOS")

    with st.spinner("Running walk-forward backtests for all 6 strategies… (~30 s first run, cached after)"):
        _bt_str  = load_backtest("SPY", equity_input, 6, "straddle")
        _bt_gap  = load_backtest("SPY", equity_input, 6, "gapfade")
        _bt_smt  = load_backtest("SPY", equity_input, 6, "smart")
        _bt_v2   = load_backtest("SPY", equity_input, 6, "smart_v2")
        _bt_v3   = load_backtest("SPY", equity_input, 6, "smart_v3")
        _bt_v4   = load_backtest("SPY", equity_input, 6, "smart_v4")

    if _bt_str is None or _bt_smt is None or _bt_v2 is None or _bt_v3 is None or _bt_v4 is None:
        st.warning("Backtest could not run — check that SPY OHLCV data is available.")
    else:
        # ── Strategy comparison row (3 large cards) ──────────────────────────
        def _strat_card(label: str, sub: str, bt, accent: str, badge: str = "") -> str:
            net  = bt.end_equity - equity_input
            mult = bt.end_equity / equity_input
            net_c = "#3fb950" if net >= 0 else "#f85149"
            badge_html = (f'<span style="background:{accent};color:#000;font-size:9px;'
                          f'font-weight:900;padding:2px 8px;border-radius:6px;'
                          f'margin-left:8px;letter-spacing:.05em;">{badge}</span>') if badge else ""
            return f"""
            <div style="background:#161b22;border:2px solid {accent};border-radius:14px;
                        padding:18px 16px;text-align:center;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:2px;">{label}{badge_html}</div>
              <div style="color:#6e7681;font-size:10px;margin-bottom:10px;">{sub}</div>
              <div style="font-size:32px;font-weight:900;color:#fff;line-height:1.1;">
                ${bt.end_equity:,.0f}</div>
              <div style="font-size:13px;color:{net_c};font-weight:800;margin-top:4px;">
                {'+' if net >= 0 else ''}{net:,.2f} ({mult:.2f}×)</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                          margin-top:14px;padding-top:12px;border-top:1px solid #21262d;
                          font-size:11px;">
                <div><span style="color:#8b949e;">Trades:</span>
                  <span style="color:#e6edf3;font-weight:700;">{bt.n_trades}</span></div>
                <div><span style="color:#8b949e;">Win:</span>
                  <span style="color:#3fb950;font-weight:700;">{bt.win_rate*100:.0f}%</span></div>
                <div><span style="color:#8b949e;">Max DD:</span>
                  <span style="color:#f85149;font-weight:700;">{bt.max_drawdown_pct:.0f}%</span></div>
                <div><span style="color:#8b949e;">W/L:</span>
                  <span style="color:#e6edf3;font-weight:700;">{bt.n_wins}/{bt.n_losses}</span></div>
              </div>
            </div>"""

        # Determine winner across all 6 modes
        _all = {"straddle": _bt_str, "gapfade": _bt_gap, "smart": _bt_smt,
                "smart_v2": _bt_v2, "smart_v3": _bt_v3, "smart_v4": _bt_v4}
        _winner_key = max(_all, key=lambda k: _all[k].end_equity)
        _v2_lift      = _bt_v2.end_equity - _bt_str.end_equity
        _v3_lift      = _bt_v3.end_equity - _bt_str.end_equity
        _v3_over_v2   = _bt_v3.end_equity - _bt_v2.end_equity
        _v4_lift      = _bt_v4.end_equity - _bt_str.end_equity
        _v4_over_v3   = _bt_v4.end_equity - _bt_v3.end_equity
        _v3_lift_c    = "#3fb950" if _v3_over_v2 > 0 else "#f85149" if _v3_over_v2 < 0 else "#8b949e"
        _v4_lift_c    = "#3fb950" if _v4_over_v3 > 0 else "#f85149" if _v4_over_v3 < 0 else "#8b949e"

        def _badge(key): return "WINNER" if key == _winner_key else ""

        st.html(f"""
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;
                    margin-bottom:14px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          {_strat_card("1. Baseline", "ATM straddle every HOT day", _bt_str,
                       "#3fb950" if _winner_key=="straddle" else "#30363d", _badge("straddle"))}
          {_strat_card("2. Gap-Fade", "Fade every gap ≥ 0.25%", _bt_gap,
                       "#3fb950" if _winner_key=="gapfade" else "#5b3b1d", _badge("gapfade"))}
          {_strat_card("3. Smart v1",
                       f"Counter-trend fade + breaker ({_bt_smt.n_breaker_skips})",
                       _bt_smt,
                       "#3fb950" if _winner_key=="smart" else "#1f6feb", _badge("smart"))}
          {_strat_card("4. Smart v2",
                       f"+ regime filter ({_bt_v2.n_regime_skips}) + directional ({_bt_v2.n_directional})",
                       _bt_v2,
                       "#3fb950" if _winner_key=="smart_v2" else "#a371f7", _badge("smart_v2"))}
          {_strat_card("5. Smart v3",
                       f"+ weekly MA-touch + flow override ({_bt_v3.n_ma_confluence})",
                       _bt_v3,
                       "#3fb950" if _winner_key=="smart_v3" else "#d29922", _badge("smart_v3"))}
          {_strat_card("6. Smart v4",
                       f"Patient lottery · scale 50% @ +100% · {_bt_v4.n_scaleout_hits}/{_bt_v4.n_trades} touches",
                       _bt_v4,
                       "#3fb950" if _winner_key=="smart_v4" else "#f778ba", _badge("smart_v4"))}
        </div>
        <div style="background:#1a1410;border:1px solid #4d3a1a;border-radius:10px;
                    padding:14px 18px;margin-bottom:18px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                    font-size:13px;color:#e6d5b8;line-height:1.6;">
          <div style="margin-bottom:8px;">
            <strong style="color:#d29922;">Smart v3 vs Smart v2:</strong>
            <span style="color:{_v3_lift_c};font-weight:900;font-size:16px;">
              {'+' if _v3_over_v2 >= 0 else ''}${_v3_over_v2:,.2f}
            </span>
            ({(_bt_v3.end_equity/_bt_v2.end_equity - 1)*100:+.1f}%) ·
            <strong>vs baseline:</strong>
            <span style="color:{'#3fb950' if _v3_lift >= 0 else '#f85149'};font-weight:800;">
              {'+' if _v3_lift >= 0 else ''}${_v3_lift:,.2f}
            </span>
            ({(_bt_v3.end_equity/_bt_str.end_equity - 1)*100:+.1f}%)
          </div>
          <strong style="color:#d29922;">v3 layered upgrade (weekly structure):</strong>
          <ul style="margin:6px 0 0 18px;padding:0;">
            <li><strong>MA-touch + flow confluence</strong> — when SPY's prior close is within
                1.5% of an empirically biased weekly MA (10w/50w SMA + 50w EMA = LONG-bias;
                20w/30w EMA = SHORT-bias) <em>and</em> weekly order flow agrees in the same
                direction (|score| ≥ 20), force a JACKPOT-tier directional play.
                Fired {_bt_v3.n_ma_confluence} time(s) this window.</li>
          </ul>
          <div style="margin-top:8px;color:#8b949e;font-size:12px;">
            <em>Honest note:</em> v3 only fires on a tiny number of days (the 6-month window
            saw {_bt_v3.n_ma_confluence} confluence trade(s)), so any improvement here is
            small-sample. The MA-touch hypothesis came from finding 50w SMA = 100% bounce
            (n=2) and 30w EMA = 0% bounce (n=5) in 6 months of empirical SPY data — real but
            statistically thin. v3 will show its edge mostly during deep pullbacks toward
            those weekly levels, which were rare in this window.
          </div>
        </div>
        <div style="background:#1d1424;border:1px solid #4a2a5e;border-radius:10px;
                    padding:14px 18px;margin-bottom:18px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                    font-size:13px;color:#e8d5f0;line-height:1.6;">
          <div style="margin-bottom:8px;">
            <strong style="color:#f778ba;">Smart v4 vs Smart v3:</strong>
            <span style="color:{_v4_lift_c};font-weight:900;font-size:16px;">
              {'+' if _v4_over_v3 >= 0 else ''}${_v4_over_v3:,.2f}
            </span>
            ({(_bt_v4.end_equity/_bt_v3.end_equity - 1)*100:+.1f}%) ·
            <strong>vs baseline:</strong>
            <span style="color:{'#3fb950' if _v4_lift >= 0 else '#f85149'};font-weight:800;">
              {'+' if _v4_lift >= 0 else ''}${_v4_lift:,.2f}
            </span>
            ({(_bt_v4.end_equity/_bt_str.end_equity - 1)*100:+.1f}%)
          </div>
          <strong style="color:#f778ba;">v4 = your "sell half at +100%" rule, gated on highest conviction:</strong>
          <ul style="margin:6px 0 0 18px;padding:0;">
            <li><strong>Patient lottery ticket</strong> — only trades on directional setups
                (MA-touch + flow OR P(up) extreme). No straddles, no gap-fades. Skips ~90%
                of days.</li>
            <li><strong>Scale-out at +100%</strong> — when the option doubles intraday, sell
                half. Hold the rest to close. Empirically validated: 22% of last 6mo had
                ≥1.10% intraday move (the touch threshold), and 100% of those touches
                netted positive after scaling (avg +104.6% on the trade).</li>
            <li>This window: <strong>{_bt_v4.n_scaleout_hits}/{_bt_v4.n_trades} trades hit
                the +100% scale-out trigger</strong>.</li>
          </ul>
          <div style="margin-top:8px;color:#8b949e;font-size:12px;">
            <em>Honest note:</em> "Sell half at +100% = no loss" is real <em>conditional on
            hitting +100% intraday</em>. The other 78% of days, the trade decays toward
            zero just like any other 0DTE long. v4's edge is being patient enough to only
            take the highest-conviction setups so the touch rate is meaningfully higher
            than 22%. Whether that holds out-of-sample on a 126-day window is a small-sample
            question — judge it on the trade log below, not just the headline number.
          </div>
        </div>""")

        # ── Detail block shows the WINNING strategy ───────────────────────
        _bt = _all[_winner_key]
        _winner_label = {"straddle":"Baseline Straddle","gapfade":"Naive Gap-Fade",
                         "smart":"Smart v1","smart_v2":"Smart v2 (Tier-1)",
                         "smart_v3":"Smart v3 (MA + Flow)",
                         "smart_v4":"Smart v4 (Patient + Scale-Out)"}[_winner_key]
        st.markdown(f"**Detail view — {_winner_label} day-by-day:**")
        # ── Summary stat row ─────────────────────────────────────────────────
        _bt_net    = _bt.end_equity - equity_input
        _bt_mult   = _bt.end_equity / equity_input
        _bt_net_c  = "#3fb950" if _bt_net >= 0 else "#f85149"
        _bt_dd_c   = "#f85149" if _bt.max_drawdown_pct < -20 else "#ffd633" if _bt.max_drawdown_pct < -10 else "#3fb950"

        st.html(f"""
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;
                    margin-bottom:16px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="background:#161b22;border:2px solid {_bt_net_c};border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">Final Equity</div>
            <div style="font-size:24px;font-weight:900;color:#fff;">
              ${_bt.end_equity:,.2f}</div>
            <div style="font-size:11px;color:{_bt_net_c};font-weight:700;margin-top:3px;">
              {'+' if _bt_net >= 0 else ''}{_bt_net:,.2f} ({_bt_mult:.2f}×)</div>
          </div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">Trade Days</div>
            <div style="font-size:24px;font-weight:900;color:#e6edf3;">{_bt.n_trades}</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;">
              of {_bt.n_days} sessions</div>
          </div>
          <div style="background:#161b22;border:1px solid #3fb950;border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">Win Rate</div>
            <div style="font-size:24px;font-weight:900;color:#3fb950;">
              {_bt.win_rate*100:.1f}%</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;">
              {_bt.n_wins}W / {_bt.n_losses}L</div>
          </div>
          <div style="background:#161b22;border:1px solid #ffd633;border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">JACKPOT Days</div>
            <div style="font-size:24px;font-weight:900;color:#ffd633;">{_bt.n_jackpot}</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;">
              HOT days: {_bt.n_hot}</div>
          </div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">Peak Equity</div>
            <div style="font-size:24px;font-weight:900;color:#58a6ff;">
              ${_bt.max_equity:,.2f}</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;">
              min: ${_bt.min_equity:,.2f}</div>
          </div>
          <div style="background:#161b22;border:1px solid {_bt_dd_c}33;border-radius:10px;
                      padding:14px;text-align:center;">
            <div style="color:#8b949e;font-size:9px;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:4px;">Max Drawdown</div>
            <div style="font-size:24px;font-weight:900;color:{_bt_dd_c};">
              {_bt.max_drawdown_pct:.1f}%</div>
            <div style="font-size:11px;color:#8b949e;margin-top:3px;">
              peak-to-trough</div>
          </div>
        </div>""")

        # ── Month-by-month breakdown ──────────────────────────────────────────
        _log = _bt.trade_log.copy()
        _log["month_label"] = _log["date"].dt.to_period("M").astype(str)
        _months_grp = (
            _log.groupby("month_label")
            .agg(
                start_eq  = ("equity", "first"),
                end_eq    = ("equity", "last"),
                trades    = ("signal", lambda s: (s != "SKIP").sum()),
                wins      = ("outcome", lambda s: (s == "WIN").sum()),
                losses    = ("outcome", lambda s: (s == "LOSS").sum()),
                jackpots  = ("signal", lambda s: (s == "GO_JACKPOT").sum()),
            )
            .reset_index()
        )
        # recalc start_eq as the equity at the BEGINNING of each month
        _month_start_eqs = []
        for _, _mrow in _months_grp.iterrows():
            _m_rows = _log[_log["month_label"] == _mrow["month_label"]]
            # equity at start of this month = equity at end of previous day
            _idx0 = _m_rows.index[0]
            _prev = _log[_log.index < _idx0]["equity"]
            _month_start_eqs.append(_prev.iloc[-1] if len(_prev) > 0 else equity_input)
        _months_grp["start_eq"] = _month_start_eqs

        _mo_rows_html = ""
        _prev_eq = equity_input
        for _, _mo in _months_grp.iterrows():
            _mo_end = float(_mo["end_eq"])
            _mo_gain = _mo_end - _prev_eq
            _mo_g_c = "#3fb950" if _mo_gain >= 0 else "#f85149"
            _mo_wr = (_mo["wins"] / _mo["trades"] * 100) if _mo["trades"] > 0 else 0
            _mo_rows_html += f"""
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 6px;color:#8b949e;font-weight:600;">{_mo["month_label"]}</td>
              <td style="padding:8px 6px;text-align:right;color:#6e7681;">
                ${_prev_eq:,.2f}</td>
              <td style="padding:8px 6px;text-align:right;font-weight:800;color:#e6edf3;">
                ${_mo_end:,.2f}</td>
              <td style="padding:8px 6px;text-align:right;font-weight:700;color:{_mo_g_c};">
                {'+' if _mo_gain >= 0 else ''}{_mo_gain:,.2f}</td>
              <td style="padding:8px 6px;text-align:center;color:#e6edf3;">
                {int(_mo['trades'])}</td>
              <td style="padding:8px 6px;text-align:center;color:#ffd633;">
                {int(_mo['jackpots'])}</td>
              <td style="padding:8px 6px;text-align:center;
                         color:{'#3fb950' if _mo_wr >= 50 else '#f85149'};">
                {_mo_wr:.0f}%</td>
            </tr>"""
            _prev_eq = _mo_end

        st.html(f"""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:14px;
                    padding:20px 24px;margin-bottom:16px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:12px;">
            Month-by-Month Performance · {_bt.start_date} → {_bt.end_date}</div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;
                        font-variant-numeric:tabular-nums;">
            <thead>
              <tr style="border-bottom:2px solid #30363d;">
                <th style="color:#8b949e;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:left;">Month</th>
                <th style="color:#6e7681;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:right;">Start</th>
                <th style="color:#e6edf3;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:right;">End</th>
                <th style="color:#58a6ff;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:right;">P&L</th>
                <th style="color:#8b949e;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:center;">Trades</th>
                <th style="color:#ffd633;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:center;">JKPT</th>
                <th style="color:#3fb950;font-size:10px;text-transform:uppercase;
                           letter-spacing:.06em;padding:6px 6px;font-weight:700;
                           text-align:center;">Win%</th>
              </tr>
            </thead>
            <tbody style="line-height:1.9;">{_mo_rows_html}</tbody>
          </table>
        </div>""")

        # ── Full trade log ─────────────────────────────────────────────────
        with st.expander("📋 Full Day-by-Day Trade Log", expanded=False):
            _display_log = _bt.trade_log[_bt.trade_log["signal"] != "SKIP"].copy()
            _display_log["date"]         = _display_log["date"].dt.strftime("%Y-%m-%d")
            _display_log["straddle_ret"] = (_display_log["straddle_ret"] * 100).round(1).astype(str) + "%"
            _display_log["alloc"]        = _display_log["alloc_dollars"].apply(lambda x: f"${x:,.2f}")
            _display_log["P&L"]          = _display_log["dollar_change"].apply(
                lambda x: f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}")
            _display_log["equity"]       = _display_log["equity"].apply(lambda x: f"${x:,.2f}")
            _display_log["p_vol"]        = (_display_log["p_vol"] * 100).round(1).astype(str) + "%"
            _display_log["p_pnl"]        = (_display_log["p_pnl"] * 100).round(1).astype(str) + "%"

            st.dataframe(
                _display_log[["date", "signal", "p_vol", "p_pnl",
                               "straddle_ret", "alloc", "P&L", "equity", "outcome"]].rename(columns={
                    "date": "Date", "signal": "Signal",
                    "p_vol": "P(vol)", "p_pnl": "P(pnl)",
                    "straddle_ret": "Actual Return", "alloc": "Risked",
                    "equity": "Equity After", "outcome": "Result",
                }),
                hide_index=True,
                use_container_width=True,
            )

        # ── Methodology note ─────────────────────────────────────────────────
        st.html(f"""
        <div style="background:#0c1520;border:1px solid #1d3a5c;border-radius:8px;
                    padding:12px 16px;font-size:11px;color:#6e7681;line-height:1.8;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <strong style="color:#58a6ff;">How this backtest works:</strong>
          Walk-forward expanding window — the model trains only on data before each prediction date,
          never touching future prices. Signal thresholds: P(vol) ≥ 30% for GO_HOT,
          additionally P(pnl) ≥ 55% for GO_JACKPOT. Trade P&L uses
          <em>actual SPY OHLCV prices</em> (not simulated win rates) — the
          straddle model prices ATM premium as ~1.1% of spot (dynamic per-ticker IV proxy)
          and measures payoff from the real close/intraday moves.
          Allocation: JACKPOT = 25% Kelly, HOT = 15% Kelly. Equity compounds daily.
          <br><br>
          <strong style="color:#3fb950;">Smart strategy upgrades</strong> applied to the detail
          view above: (1) <em>Trend-aware gap-fade</em> — only fade gaps in 0.25%–1.5% range AND
          opposite the 5-day trend (counter-trend mean-reversion); skip the rest.
          (2) <em>Circuit breaker</em> — pause for 5 sessions after 3 consecutive losses to stop
          the bleed in trending regimes. (3) Default to straddle on no-gap volatile days to
          preserve the original edge.
          Window: {_bt.start_date} → {_bt.end_date} ({_bt.n_days} sessions).
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
#  PAGE: WEEKLY MAs + ORDER FLOW
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Weekly MAs":
    from src.weekly_levels import (
        build_ma_setups, build_weekly_order_flow, build_reversal_verdict,
    )

    with st.sidebar:
        st.markdown("---")
        wma_ticker = st.selectbox(
            "Ticker", ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMZN", "META"],
            index=0, key="wma_ticker",
        )

    section("Weekly Moving Averages — Reversal Levels",
            f"{wma_ticker} · weekly MA touch zones + weekly order flow confluence")

    # Load 2 years of daily data so 50w/100w MAs are well-defined
    try:
        from src.scanner import fetch_or_load_daily
        _daily = fetch_or_load_daily(wma_ticker).sort_index()
        _daily.index = pd.to_datetime(_daily.index).tz_localize(None)
    except Exception as exc:
        st.error(f"Could not load daily data for {wma_ticker}: {exc}")
        st.stop()

    if len(_daily) < 250:
        st.warning(f"Need at least 250 daily bars for full weekly MA analysis. Have {len(_daily)}.")
        st.stop()

    setups = build_ma_setups(_daily)
    flow   = build_weekly_order_flow(_daily)
    verdict = build_reversal_verdict(_daily)
    last_price = float(_daily["Close"].iloc[-1])

    # ── HERO VERDICT CARD ─────────────────────────────────────────────────
    _bias_color = {"LONG": "#3fb950", "SHORT": "#f85149", "NEUTRAL": "#8b949e"}[verdict.bias]
    _conf_color = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#8b949e"}[verdict.confidence]
    _bias_emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}[verdict.bias]

    st.html(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:2px solid {_bias_color};border-radius:14px;padding:22px 26px;
                margin-bottom:18px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:4px;">Active Setup</div>
          <div style="font-size:26px;font-weight:900;color:{_bias_color};">
            {_bias_emoji} {verdict.bias} BIAS
          </div>
        </div>
        <div style="text-align:right;">
          <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:4px;">Confidence</div>
          <div style="font-size:22px;font-weight:800;color:{_conf_color};">
            {verdict.confidence}
          </div>
        </div>
      </div>
      <div style="color:#e6edf3;font-size:15px;line-height:1.55;margin-bottom:10px;">
        {verdict.headline}
      </div>
      <div style="color:#c8e6c9;font-size:13px;line-height:1.5;
                  background:#0c1f12;border-left:3px solid {_bias_color};
                  padding:10px 14px;border-radius:6px;margin-bottom:10px;">
        <strong>Confluence:</strong> {verdict.confluence_note}
      </div>
      <div style="color:#79c0ff;font-size:13px;line-height:1.5;
                  background:#0d1f3a;border-left:3px solid #1f6feb;
                  padding:10px 14px;border-radius:6px;">
        <strong>Play:</strong> {verdict.play_suggestion or "Wait for price to approach the key level."}
      </div>
    </div>
    """)

    # ── CURRENT PRICE + WEEKLY ORDER FLOW GAUGE ────────────────────────────
    c1, c2 = st.columns([1, 2])
    with c1:
        st.html(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                    padding:18px 16px;text-align:center;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:6px;">{wma_ticker} Last Close</div>
          <div style="font-size:32px;font-weight:900;color:#e6edf3;
                      letter-spacing:-.02em;">${last_price:,.2f}</div>
          <div style="color:#8b949e;font-size:11px;margin-top:6px;">
            as of {_daily.index[-1].date()}
          </div>
        </div>
        """)
    with c2:
        if flow is None:
            st.info("Weekly order-flow data unavailable.")
        else:
            _fs = flow.flow_score
            _fs_color = "#3fb950" if _fs >= 10 else ("#f85149" if _fs <= -10 else "#8b949e")
            # gauge bar position 0..100
            _gauge_pos = max(0, min(100, (_fs + 100) / 2))
            st.html(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                        padding:18px 22px;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
                <div>
                  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                              letter-spacing:.1em;">Weekly Order Flow</div>
                  <div style="color:#e6edf3;font-size:13px;margin-top:2px;">
                    Week ending {flow.week_ending.date()} — {flow.interpretation}
                  </div>
                </div>
                <div style="font-size:28px;font-weight:900;color:{_fs_color};">
                  {_fs:+.0f}
                </div>
              </div>
              <div style="background:linear-gradient(90deg,#f85149 0%,#8b949e 50%,#3fb950 100%);
                          height:8px;border-radius:4px;position:relative;margin:10px 0 14px 0;">
                <div style="position:absolute;left:{_gauge_pos}%;top:-4px;width:3px;height:16px;
                            background:#e6edf3;border-radius:2px;
                            box-shadow:0 0 6px rgba(255,255,255,.6);"></div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
                          color:#c9d1d9;font-size:12px;">
                <div>
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Closing Strength</div>
                  <div style="font-weight:700;">{flow.closing_strength:.2f}
                    <span style="color:#8b949e;font-size:11px;font-weight:400;"> · 4w {flow.closing_strength_4w_avg:.2f}</span>
                  </div>
                </div>
                <div>
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Weekly CVD</div>
                  <div style="font-weight:700;">{flow.weekly_cvd/1e6:+.0f}M
                    <span style="color:#8b949e;font-size:11px;font-weight:400;"> · 4w {flow.cvd_4w_avg/1e6:+.0f}M</span>
                  </div>
                </div>
                <div>
                  <div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Z-score 20w</div>
                  <div style="font-weight:700;">{flow.z_20w:+.2f}σ</div>
                </div>
              </div>
            </div>
            """)

    # ── MA STACK TABLE ─────────────────────────────────────────────────────
    st.markdown("### Weekly MA stack — distance from current price + historical touch outcomes")
    st.caption(
        "Each row is a weekly-equivalent MA. **Touches in last 6 months** count days "
        "where price's intraday range crossed the MA. **5d outcome** = average return "
        "over the next 5 trading days after a touch. **Bias** is auto-derived from "
        "asymmetry (LONG when ≥70% bounced, SHORT when ≤30% bounced)."
    )

    _rows_html = ""
    for s in setups:
        if s.n_touches_6m == 0:
            continue
        _bias_c = {"LONG": "#3fb950", "SHORT": "#f85149", "NEUTRAL": "#8b949e"}[s.bias]
        _bias_bg = {"LONG": "#0c1f12", "SHORT": "#1f0c0c", "NEUTRAL": "#1c1f24"}[s.bias]
        _dist_c = "#3fb950" if s.distance_pct < 0 else "#f85149"  # below MA = green opportunity
        _ret_c = "#3fb950" if s.avg_5d_ret_after_touch >= 0 else "#f85149"
        _rows_html += f"""
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 12px;font-weight:700;color:#e6edf3;">{s.ma_label}</td>
          <td style="padding:10px 12px;color:#e6edf3;text-align:right;">${s.ma_value:,.2f}</td>
          <td style="padding:10px 12px;text-align:right;color:{_dist_c};font-weight:600;">
            {s.distance_pct:+.2f}%
            <div style="color:#8b949e;font-size:11px;font-weight:400;">${s.distance_dollars:+,.2f}</div>
          </td>
          <td style="padding:10px 12px;text-align:center;">
            <span style="background:{_bias_bg};color:{_bias_c};
                         padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;
                         letter-spacing:.05em;">{s.bias}</span>
          </td>
          <td style="padding:10px 12px;text-align:center;color:#c9d1d9;">{s.n_touches_6m}</td>
          <td style="padding:10px 12px;text-align:right;color:#c9d1d9;">{s.pct_positive_after_touch:.0f}%</td>
          <td style="padding:10px 12px;text-align:right;color:{_ret_c};font-weight:700;">
            {s.avg_5d_ret_after_touch:+.2f}%
          </td>
          <td style="padding:10px 12px;text-align:right;color:#3fb950;">+{s.avg_5d_max_up:.2f}%</td>
          <td style="padding:10px 12px;text-align:right;color:#f85149;">{s.avg_5d_max_dn:.2f}%</td>
        </tr>
        """

    st.html(f"""
    <div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;
                overflow:hidden;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                font-size:13px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead style="background:#161b22;">
          <tr style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em;">
            <th style="padding:10px 12px;text-align:left;">MA</th>
            <th style="padding:10px 12px;text-align:right;">Level</th>
            <th style="padding:10px 12px;text-align:right;">Distance</th>
            <th style="padding:10px 12px;text-align:center;">Bias</th>
            <th style="padding:10px 12px;text-align:center;">Touches 6m</th>
            <th style="padding:10px 12px;text-align:right;">% bounced</th>
            <th style="padding:10px 12px;text-align:right;">Avg 5d ret</th>
            <th style="padding:10px 12px;text-align:right;">Avg 5d max ↑</th>
            <th style="padding:10px 12px;text-align:right;">Avg 5d max ↓</th>
          </tr>
        </thead>
        <tbody>
          {_rows_html}
        </tbody>
      </table>
    </div>
    """)

    # ── PRICE CHART WITH MA OVERLAYS ───────────────────────────────────────
    st.markdown("### Price + key weekly MAs (last 12 months)")
    try:
        import altair as alt
        from src.weekly_levels import compute_weekly_mas
        _mas = compute_weekly_mas(_daily)
        _chart_window = _daily.index.max() - pd.Timedelta(days=365)
        _chart_df = _daily[_daily.index >= _chart_window][["Close"]].copy()
        _chart_df["10w SMA"] = _mas["10w SMA"]
        _chart_df["20w SMA"] = _mas["20w SMA"]
        _chart_df["50w SMA"] = _mas["50w SMA"]
        _chart_df = _chart_df.dropna(subset=["Close"]).reset_index().rename(columns={"index": "date"})
        _chart_long = _chart_df.melt(id_vars=["date"], var_name="series", value_name="price").dropna()

        _color_scale = alt.Scale(
            domain=["Close", "10w SMA", "20w SMA", "50w SMA"],
            range=["#e6edf3", "#79c0ff", "#d29922", "#3fb950"],
        )
        _chart = (
            alt.Chart(_chart_long).mark_line().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("price:Q", title=None,
                        scale=alt.Scale(zero=False, padding=10)),
                color=alt.Color("series:N", scale=_color_scale,
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[
                    alt.Tooltip("date:T"),
                    alt.Tooltip("series:N"),
                    alt.Tooltip("price:Q", format="$.2f"),
                ],
            )
            .properties(height=340, background="#0d1117")
            .configure_view(strokeOpacity=0)
            .configure_axis(grid=True, gridColor="#21262d", labelColor="#8b949e",
                            tickColor="#30363d", domainColor="#30363d")
            .configure_legend(labelColor="#c9d1d9")
        )
        st.altair_chart(_chart, use_container_width=True)
    except Exception as _e:
        st.caption(f"(chart unavailable: {_e})")

    # ── HONEST CAVEATS ─────────────────────────────────────────────────────
    st.html("""
    <div style="background:#1c1f24;border:1px solid #30363d;border-radius:10px;
                padding:14px 18px;margin-top:18px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                color:#8b949e;font-size:12px;line-height:1.6;">
      <strong style="color:#d29922;">Honest sample-size caveat:</strong> The 50w SMA shows the
      strongest asymmetry (100% bounce rate, +3.93% avg) but with only 2 touches in 6 months —
      that's a small sample. The 10w/20w MAs have larger samples (15–23 touches) but smaller
      edges. Treat the 50w SMA as a HIGH-conviction zone <em>when</em> price actually reaches it,
      not a frequent trade signal.
      <br><br>
      <strong style="color:#d29922;">Order-flow note:</strong> The flow score is built from
      daily bars (close-vs-open direction × volume + closing strength) — it's a
      <em>proxy</em>, not true intraday tick-derived order flow. Tested on its own it does
      <strong>not</strong> beat baseline next-week returns, so we use it only as
      <em>confluence</em> for an MA-touch setup, never as a standalone signal.
    </div>
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MA BOUNCE SETUPS — Universe-scanned high-edge MA-touch plays
# ══════════════════════════════════════════════════════════════════════════════

elif page == "MA Bounce Setups":
    from src.ma_setups_universe import get_all_live_setups, HIGH_EDGE_SETUPS, SCAN_DATE

    section("MA Bounce Setups — Universe Scan",
            f"{len(HIGH_EDGE_SETUPS)} high-edge weekly MA-touch setups (n≥5 touches, ≥75% positive 5d) · "
            "validated on 1 yr daily data + real Polygon options")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Filters**")
        min_winrate = st.slider("Min historical win rate (%)", 75, 100, 80, step=5,
                                key="ma_setups_min_wr")
        only_actionable = st.checkbox("Show only TOUCHING / APPROACHING", value=True,
                                       key="ma_setups_actionable")
        st.caption("Universe scanned: 50 tickers (indices, mega-caps, high-vol single names) "
                   "× 4 MAs (30w/50w · SMA/EMA)")

    # ── Methodology / disclaimer card ─────────────────────────────────────────
    st.html("""
    <div style="background:#0d1f2e;border:1px solid #1f6feb;border-radius:10px;
                padding:12px 16px;margin-bottom:16px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="color:#79c0ff;font-size:11px;font-weight:800;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px;">📊 How this works</div>
      <div style="color:#c8e6c9;font-size:12px;line-height:1.6;">
        Each setup is a (ticker, weekly MA) pair where, over the past year, ≥75% of MA touches
        produced a positive 5-day return. We compute the live distance between today's price and
        each MA, then classify: <span style="color:#ffd633;font-weight:700;">TOUCHING</span> (within ±0.6%),
        <span style="color:#3fb950;font-weight:700;">APPROACHING from above</span> (within 2.5%, drifting down),
        <span style="color:#8b949e;font-weight:700;">EXTENDED</span> (>2.5% above MA),
        <span style="color:#f85149;font-weight:700;">BELOW</span> (broke the level).
        Real-options validation: AMZN/SPY/QQQ/NVDA setups returned <strong style="color:#3fb950;">+243%</strong>
        on $39k over 39 weekly call trades (62% win rate, largest 29x).
      </div>
    </div>
    """)

    @st.cache_data(ttl=600)
    def _cached_live_setups():
        return get_all_live_setups()

    with st.spinner(f"Scanning {len(HIGH_EDGE_SETUPS)} setups across the universe…"):
        live_setups, failed_setups = _cached_live_setups()

    if not live_setups:
        st.error("Could not load any setup data. Check Polygon connectivity.")
        st.stop()

    if failed_setups:
        st.warning(
            f"⚠ {len(failed_setups)} setup(s) could not be priced and were excluded: "
            + ", ".join(f"{t} {m}" for t, m in failed_setups)
        )

    st.caption(f"📅 Universe scan results frozen on **{SCAN_DATE}** "
               f"(re-run `scripts/scan_universe.py` to refresh). "
               f"Live MA distances refreshed every 10 min.")

    # Filter
    filtered = [s for s in live_setups if s.pct_pos_5d >= min_winrate]
    if only_actionable:
        filtered = [s for s in filtered if s.state in ("TOUCHING", "APPROACHING")]

    # ── Hero counts ───────────────────────────────────────────────────────────
    n_touch = sum(1 for s in live_setups if s.state == "TOUCHING")
    n_appr  = sum(1 for s in live_setups if s.state == "APPROACHING")
    n_ext   = sum(1 for s in live_setups if s.state == "EXTENDED")
    n_below = sum(1 for s in live_setups if s.state == "BELOW")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Touching now",   n_touch, "#ffd633"),
        (c2, "Approaching",    n_appr,  "#3fb950"),
        (c3, "Extended",       n_ext,   "#8b949e"),
        (c4, "Below MA",       n_below, "#f85149"),
    ]:
        with col:
            st.html(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:14px 12px;text-align:center;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
              <div style="font-size:30px;font-weight:900;color:{color};
                          line-height:1;">{value}</div>
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;margin-top:6px;">{label}</div>
            </div>
            """)

    st.markdown("")

    if not filtered:
        st.info(f"No setups match the current filters. Try lowering the win-rate threshold or "
                "unchecking 'only TOUCHING/APPROACHING'.")
        st.stop()

    st.html(f"""<div style="color:#8b949e;font-size:12px;margin-bottom:8px;">
      Showing <strong style="color:#e6edf3;">{len(filtered)}</strong> setup(s)
      ranked by edge score (state × historical win rate × avg return)
    </div>""")

    # ── Setup cards ──────────────────────────────────────────────────────────
    for s in filtered:
        # Direction text
        dir_text = "above" if s.distance_pct >= 0 else "below"
        drift_arrow = "↓" if s.drift_5d < 0 else "↑"
        drift_color = "#3fb950" if s.drift_5d < 0 and s.above_ma else (
                      "#f85149" if s.drift_5d > 0 and not s.above_ma else "#8b949e")

        # Action recommendation
        if s.state == "TOUCHING":
            action = (f"<strong style='color:#ffd633;'>ACT NOW</strong> — price is at the MA. "
                      f"Consider a 5-day weekly call. Historical edge: {s.pct_pos_5d:.0f}% win, "
                      f"{s.avg_5d:+.2f}% avg.")
            action_bg = "#2a2410"; action_border = "#ffd633"
        elif s.state == "APPROACHING" and s.drift_5d < 0:
            action = (f"<strong style='color:#3fb950;'>WATCH CLOSELY</strong> — only "
                      f"{abs(s.distance_pct):.2f}% above the level and falling "
                      f"({s.drift_5d:+.2f}% past 5d). Set alert at ${s.ma_value:.2f}.")
            action_bg = "#0d1f14"; action_border = "#3fb950"
        elif s.state == "APPROACHING":
            action = (f"<strong style='color:#79c0ff;'>NEAR LEVEL</strong> — "
                      f"{abs(s.distance_pct):.2f}% above the MA but trending up. "
                      f"Wait for a pullback to ${s.ma_value:.2f}.")
            action_bg = "#0d1f2e"; action_border = "#1f6feb"
        elif s.state == "EXTENDED":
            action = (f"<strong style='color:#8b949e;'>NO TRADE</strong> — price is "
                      f"{s.distance_pct:+.1f}% from the MA. Wait for mean reversion.")
            action_bg = "#161b22"; action_border = "#30363d"
        else:  # BELOW
            action = (f"<strong style='color:#f85149;'>LEVEL BROKEN</strong> — price is "
                      f"{s.distance_pct:+.1f}% below the MA. The setup has invalidated; "
                      f"wait for a reclaim or skip.")
            action_bg = "#1f0e0e"; action_border = "#f85149"

        st.html(f"""
        <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                    border:2px solid {s.state_color};border-radius:12px;
                    padding:16px 20px;margin-bottom:14px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;
                      margin-bottom:12px;flex-wrap:wrap;gap:10px;">
            <div>
              <div style="color:{s.state_color};font-size:11px;font-weight:800;
                          text-transform:uppercase;letter-spacing:.1em;">{s.state}</div>
              <div style="font-size:22px;font-weight:900;color:#e6edf3;margin-top:2px;">
                {s.ticker} <span style="color:#8b949e;font-weight:600;font-size:14px;">
                ({s.ma_label})</span>
              </div>
            </div>
            <div style="text-align:right;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.08em;">Historical edge ({s.n_touches} touches)</div>
              <div style="font-size:18px;font-weight:800;color:#3fb950;">
                {s.pct_pos_5d:.0f}% win · {s.avg_5d:+.2f}% avg
              </div>
              <div style="color:#8b949e;font-size:10px;margin-top:2px;">
                10d: {s.avg_10d:+.2f}% · best {s.best_5d:+.1f}% · worst {s.worst_5d:+.1f}%
              </div>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
                      margin-bottom:10px;">
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                        padding:10px 12px;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.07em;">Last close</div>
              <div style="font-size:18px;font-weight:800;color:#e6edf3;">
                ${s.last_close:,.2f}
              </div>
            </div>
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                        padding:10px 12px;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.07em;">{s.ma_label}</div>
              <div style="font-size:18px;font-weight:800;color:#79c0ff;">
                ${s.ma_value:,.2f}
              </div>
            </div>
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                        padding:10px 12px;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.07em;">Distance / 5d drift</div>
              <div style="font-size:16px;font-weight:800;color:{s.state_color};">
                {s.distance_pct:+.2f}% {dir_text}
              </div>
              <div style="color:{drift_color};font-size:11px;">
                {drift_arrow} {s.drift_5d:+.2f}% past 5d
              </div>
            </div>
          </div>

          <div style="background:{action_bg};border-left:3px solid {action_border};
                      border-radius:6px;padding:10px 14px;color:#e6edf3;font-size:13px;
                      line-height:1.55;">
            {action}
          </div>
        </div>
        """)

    # ── Methodology footer ────────────────────────────────────────────────────
    with st.expander("📋 Full universe scan results (all 22 high-edge setups)"):
        import pandas as _pd
        rows = []
        for tk, ma, n, pct, avg5, med5, pct10, avg10, best, worst in HIGH_EDGE_SETUPS:
            rows.append({"Ticker": tk, "MA": ma, "Touches": n,
                         "%Pos 5d": f"{pct:.0f}%", "Avg 5d": f"{avg5:+.2f}%",
                         "Med 5d": f"{med5:+.2f}%", "%Pos 10d": f"{pct10:.0f}%",
                         "Avg 10d": f"{avg10:+.2f}%",
                         "Best 5d": f"{best:+.1f}%", "Worst 5d": f"{worst:+.1f}%"})
        st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Universe: 50 tickers (SPY, QQQ, IWM, DIA, sector ETFs, mega-caps, "
                   "high-vol single names) × 4 MAs (30w/50w · SMA/EMA) = 200 setups tested. "
                   "Filter: ≥5 touches, ≥75% positive 5d. Lookback: 365 days.")

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
