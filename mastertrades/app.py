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
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
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
""", unsafe_allow_html=True)


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
    st.markdown(
        f"""<div style="margin:1.6rem 0 0.8rem;">
          <span style="font-size:17px;font-weight:800;color:#e6edf3;
                       letter-spacing:.01em;">{title}</span>
          {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Cached data loaders ─────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
def load_account():
    from src.account_state import load_state
    return load_state(path=ACCT_PATH)


# ─── Sidebar navigation ───────────────────────────────────────────────────────

PAGE_META = {
    "Command Center":  "Today's trade signals",
    "Scanner":         "Ranked volatility universe",
    "Gap Reversal":    "Gap fill & reversal setups",
    "Account Tracker": "Equity curve & trade log",
    "Weekday Patterns":"Vol by day of week",
}

with st.sidebar:
    st.markdown(
        """<div style="font-size:20px;font-weight:800;color:#fff;
                       letter-spacing:-.01em;margin-bottom:4px;">📈 Mastertrades</div>
           <div style="color:#8b949e;font-size:11px;margin-bottom:16px;">
             0DTE Options Intelligence</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        list(PAGE_META.keys()),
        label_visibility="collapsed",
        format_func=lambda p: p,
    )
    st.caption(PAGE_META[page])
    st.markdown("---")

    # Data source status
    try:
        from src.polygon_feed import has_polygon_key, fetch_prev_close
        _poly_ok = has_polygon_key()
    except Exception:
        _poly_ok = False

    if _poly_ok:
        st.markdown(
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
               </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="background:#1a1208;border:1px solid #6e7681;
                           border-radius:8px;padding:10px 12px;margin-bottom:10px;">
                 <div style="font-size:10px;font-weight:800;color:#6e7681;
                             text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                   ○ Yahoo Finance (fallback)</div>
                 <div style="font-size:10px;color:#8b949e;">
                   Add POLYGON_API_KEY to enable<br>exchange-quality data</div>
               </div>""",
            unsafe_allow_html=True,
        )

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

    st.markdown(
        f"""
        <div style="background:{hero_bg};border:2px solid {hero_border};
                    border-radius:16px;padding:28px 32px;margin-bottom:24px;">
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
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;margin-bottom:4px;">Lead ticker</div>
              <div style="font-size:32px;font-weight:800;color:#fff;">{best_row.ticker}</div>
              <div style="color:{hero_border};font-size:13px;font-weight:700;">
                P(vol) {fmt_pct(best_row.p_vol)} · P(pnl) {fmt_pct(best_row.p_pnl)}
              </div>
            </div>
          </div>
          {
            f'<div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.1);'
            f'font-size:12px;color:#8b949e;">Trade: <strong style="color:#3fb950;">'
            f'{", ".join(trade_tickers)}</strong></div>' if trade_tickers else ""
          }
        </div>
        """,
        unsafe_allow_html=True,
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
        s_label   = snap.get("status_label", "")
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
            st.markdown(
                f"""
                <div style="background:#161b22;border:2px solid {sig_c};
                            border-radius:14px;padding:20px 16px;{glow}">
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
                """,
                unsafe_allow_html=True,
            )

    # ── Trade tickets ─────────────────────────────────────────────────────────
    trade_rows = [r for r in rows if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")]
    if trade_rows:
        st.markdown("---")
        section("Trade Tickets", "Suggested sizing based on your account settings below")

        with st.sidebar:
            st.markdown("---")
            st.markdown("**Trade Sizing**")
            equity_input = st.number_input("Account equity ($)", value=500.0, min_value=10.0, step=50.0)
            risk_frac = st.slider("Risk per trade (%)", 5, 25, 10) / 100

        from src.report_jackpot_dashboard import trade_ticket

        for r in trade_rows:
            ticket = trade_ticket(r, equity_input, risk_frac)
            border_c = "#ffd633" if r.signal == "GO_ULTRA_JACKPOT" else "#3fb950"
            action_label = "⚡ SIZE UP — ULTRA JACKPOT" if r.signal == "GO_ULTRA_JACKPOT" else "✅ TRADE — JACKPOT"

            st.markdown(
                f"""
                <div style="background:rgba(22,27,34,0.9);border:1px solid {border_c};
                            border-radius:12px;padding:20px 24px;margin-bottom:14px;
                            box-shadow:0 0 20px {border_c}22;">
                  <div style="display:flex;justify-content:space-between;align-items:center;
                              margin-bottom:16px;">
                    <div>
                      <div style="color:{border_c};font-size:13px;font-weight:800;
                                  text-transform:uppercase;letter-spacing:.07em;">{action_label}</div>
                      <div style="color:#8b949e;font-size:11px;margin-top:2px;">{r.ticker} · 0DTE ATM option</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;">Win Prob</div>
                      <div style="color:{border_c};font-size:24px;font-weight:800;">{fmt_pct(ticket["win_prob"])}</div>
                    </div>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;">
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;margin-bottom:4px;">Strike</div>
                      <div style="color:#fff;font-size:22px;font-weight:800;">{fmt_dollar(ticket["strike"])}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;">
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;margin-bottom:4px;">Contracts</div>
                      <div style="color:#fff;font-size:22px;font-weight:800;">{ticket["n_contracts"]}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;">
                      <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                                  letter-spacing:.07em;margin-bottom:4px;">Max Risk</div>
                      <div style="color:#f85149;font-size:22px;font-weight:800;">{fmt_dollar(ticket["actual_risk"])}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # No trade signals — show skip message
        skip_tickers = [r.ticker for r in rows if r.signal == "SKIP"]
        if skip_tickers:
            st.markdown(
                f"""<div style="background:#161b22;border:1px solid #8b949e;border-radius:10px;
                               padding:16px 20px;margin-top:12px;color:#8b949e;font-size:13px;">
                  ⏭ No trade setups today — {", ".join(skip_tickers)} all showing SKIP.
                  Check back at open or after any pre-market catalyst.
                </div>""",
                unsafe_allow_html=True,
            )


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
        s_label    = snap.get("status_label", "")

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
            st.markdown(
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
                """,
                unsafe_allow_html=True,
            )

    # ── Full ranked list ──────────────────────────────────────────────────────
    st.markdown("---")
    section("Full Ranked List", "All tickers sorted by P(volatile day) — higher = more expected movement")

    table_rows = []
    for rank, (_, row) in enumerate(df_s.iterrows(), 1):
        p_vol  = row.get("p_vol",  float("nan"))
        lift   = row.get("lift",   float("nan"))
        close  = row.get("last_close", float("nan"))
        chg    = row.get("pct_change", float("nan"))
        rsi    = row.get("rsi14",  float("nan"))
        lift_txt, _ = lift_label(lift)
        table_rows.append({
            "#":         rank,
            "Ticker":    row.get("ticker", "?"),
            "P(vol)":    fmt_pct(p_vol),
            "Lift":      f"{lift:.2f}x" if not math.isnan(lift) else "—",
            "Verdict":   lift_txt,
            "Close":     fmt_dollar(close),
            "Chg %":     f"{chg*100:+.2f}%" if not math.isnan(chg) else "—",
            "RSI(14)":   f"{rsi:.0f}" if not math.isnan(rsi) else "—",
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
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

    gap_today = [(tkr, tod, sb) for tkr, tod, sb in today_rows if tod.gap_dir in ("up", "down")]
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
        st.markdown(
            f"""<div style="background:#161b22;border:1px solid #30363d;
                            border-radius:12px;padding:20px 24px;margin-bottom:20px;
                            text-align:center;">
              <div style="font-size:15px;font-weight:700;color:#8b949e;">
                No Gaps Today — {datetime.now().strftime('%A %b %d')}</div>
              <div style="font-size:13px;color:#6e7681;margin-top:6px;">
                All {len(today_rows)} tickers opened near yesterday's close. Nothing to trade. Stand by.</div>
            </div>""",
            unsafe_allow_html=True,
        )
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
        st.markdown(
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
            </div>""",
            unsafe_allow_html=True,
        )

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
                st.markdown(
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
                    """,
                    unsafe_allow_html=True,
                )

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
    st.markdown(
        f"""<div style="background:rgba(22,27,34,0.9);border:1px solid {sig_c};
                        border-radius:10px;padding:14px 18px;margin:12px 0 4px;">
          <span style="background:{sig_c};color:#0c1117;font-size:11px;font-weight:800;
                       padding:2px 7px;border-radius:3px;margin-right:8px;">{tod_detail.signal}</span>
          <span style="color:#c9d1d9;font-size:13px;">{tod_detail.signal_detail}</span>
        </div>""",
        unsafe_allow_html=True,
    )

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
        st.markdown(
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
            """,
            unsafe_allow_html=True,
        )

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
        st.markdown(
            f"""<div style="background:#0d1f14;border:2px solid #3fb950;
                            border-radius:12px;padding:20px;text-align:center;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:6px;">Most Volatile Day</div>
              <div style="font-size:30px;font-weight:800;color:#3fb950;">{best_day}</div>
              <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                {'Avg range: ' + fmt_pct(wd_agg.loc[best_day, rng_col]) if rng_col and best_day in wd_agg.index else ""}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_worst:
        st.markdown(
            f"""<div style="background:#1a1208;border:2px solid #8b949e;
                            border-radius:12px;padding:20px;text-align:center;">
              <div style="color:#8b949e;font-size:10px;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:6px;">Calmest Day</div>
              <div style="font-size:30px;font-weight:800;color:#8b949e;">{worst_day}</div>
              <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                {'Avg range: ' + fmt_pct(wd_agg.loc[worst_day, rng_col]) if rng_col and worst_day in wd_agg.index else ""}</div>
            </div>""",
            unsafe_allow_html=True,
        )

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
