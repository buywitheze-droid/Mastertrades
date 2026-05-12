"""Mastertrades — Command Center & Trading Dashboard.

A Streamlit web app wrapping the Mastertrades Python trading analytics engine.
Pages:
  1. Command Center   — Today's jackpot signals for SPY/QQQ/IWM/AAPL
  2. Scanner          — Multi-ticker volatility scanner (broader universe)
  3. Account Tracker  — Equity curve, trades, milestones
  4. Weekday Patterns — Volatility patterns by day of week
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure 'src' is importable from this directory
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def signal_color(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "#ffd633",
        "GO_JACKPOT":       "#3fb950",
        "GO_HOT":           "#d29922",
        "SKIP":             "#8b949e",
    }.get(signal or "SKIP", "#8b949e")


def signal_emoji(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "🌟 ULTRA JACKPOT",
        "GO_JACKPOT":       "✅ JACKPOT",
        "GO_HOT":           "🔥 HOT",
        "SKIP":             "⏭ SKIP",
    }.get(signal or "SKIP", "⏭ SKIP")


def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_dollar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"${v:,.2f}"


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


@st.cache_data(ttl=60, show_spinner=False)
def load_account():
    from src.account_state import load_state
    return load_state(path=ACCT_PATH)


# ─── Sidebar navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Mastertrades")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Command Center", "Scanner", "Account Tracker", "Weekday Patterns"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"As of: {datetime.now().strftime('%b %d %Y, %H:%M')}")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "Command Center":
    st.title("Command Center")
    st.caption("Daily jackpot signals — SPY · QQQ · IWM · AAPL")

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
    # Pick the strongest signal across all tickers
    rank_order = {"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2, "SKIP": 1}
    best_row = max(rows, key=lambda r: rank_order.get(r.signal, 0))
    sig = best_row.signal

    hero_bg = {
        "GO_ULTRA_JACKPOT": "linear-gradient(135deg,#1a1208,#3d2f10,#1a1208)",
        "GO_JACKPOT":       "linear-gradient(135deg,#0d1f14,#1c4a30)",
        "GO_HOT":           "linear-gradient(135deg,#1f1808,#463812)",
        "SKIP":             "linear-gradient(135deg,#0d1117,#1a2133)",
    }.get(sig, "")
    hero_border = signal_color(sig)

    st.markdown(
        f"""
        <div style="background:{hero_bg};border:2px solid {hero_border};
                    border-radius:16px;padding:28px 32px;margin-bottom:20px;">
          <div style="color:#8b949e;font-size:11px;letter-spacing:.14em;
                      text-transform:uppercase;font-weight:700;margin-bottom:6px;">
            Today's verdict — {datetime.now().strftime('%A %b %d')}
          </div>
          <h1 style="font-size:42px;margin:0 0 8px;font-weight:900;
                     color:{hero_border};">{signal_emoji(sig)}</h1>
          <div style="color:#8b949e;font-size:14px;">
            Strongest signal: <strong style="color:#fff;">{best_row.ticker}</strong>
            &nbsp;·&nbsp; P(vol)&nbsp;<strong style="color:#fff;">
            {fmt_pct(best_row.p_vol)}</strong>
            &nbsp;·&nbsp; P(pnl)&nbsp;<strong style="color:#fff;">
            {fmt_pct(best_row.p_pnl)}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Ticker cards ──────────────────────────────────────────────────────────
    st.subheader("Per-Ticker Signals")
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        sig_c = signal_color(row.signal)
        with col:
            st.markdown(
                f"""
                <div style="background:#161b22;border:2px solid {sig_c};
                            border-radius:12px;padding:18px 16px;text-align:center;">
                  <div style="font-size:26px;font-weight:900;color:#fff;">{row.ticker}</div>
                  <div style="display:inline-block;margin:8px 0;padding:5px 12px;
                              border-radius:5px;background:{sig_c};
                              color:#0c1117;font-size:11px;font-weight:800;
                              letter-spacing:.05em;">{row.signal}</div>
                  <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                    Close: <strong style="color:#fff;">{fmt_dollar(row.last_close)}</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Detailed score table ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Scores")

    table_rows = []
    for r in rows:
        table_rows.append({
            "Ticker":        r.ticker,
            "Signal":        r.signal,
            "P(vol)":        fmt_pct(r.p_vol),
            "P(pnl)":        fmt_pct(r.p_pnl),
            "P(weekly)":     fmt_pct(getattr(r, "p_weekly", float("nan"))),
            "Last Close":    fmt_dollar(r.last_close),
            "WR (hist)":     fmt_pct(getattr(r, "win_rate_history", float("nan"))),
            "Avg Ret (hist)": fmt_pct(getattr(r, "avg_ret_history", float("nan"))),
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Trade tickets (only for GO_JACKPOT / GO_ULTRA_JACKPOT) ───────────────
    trade_rows = [r for r in rows if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")]
    if trade_rows:
        st.markdown("---")
        st.subheader("Trade Tickets")

        with st.sidebar:
            st.markdown("### Trade sizing")
            equity_input = st.number_input("Account equity ($)", value=500.0, min_value=10.0, step=50.0)
            risk_frac = st.slider("Risk per trade (%)", 5, 25, 10) / 100

        from src.report_jackpot_dashboard import trade_ticket

        for r in trade_rows:
            ticket = trade_ticket(r, equity_input, risk_frac)
            border_c = "#ffd633" if r.signal == "GO_ULTRA_JACKPOT" else "#3fb950"
            st.markdown(
                f"""
                <div style="background:rgba(63,185,80,0.1);border:1px dashed {border_c};
                            border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                  <div style="font-weight:800;color:{border_c};font-size:13px;
                              text-transform:uppercase;letter-spacing:.06em;
                              margin-bottom:10px;">{r.ticker} — {r.signal}</div>
                  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
                    <div><div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Strike</div>
                         <div style="color:#fff;font-size:18px;font-weight:700;">{fmt_dollar(ticket["strike"])}</div></div>
                    <div><div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Contracts</div>
                         <div style="color:#fff;font-size:18px;font-weight:700;">{ticket["n_contracts"]}</div></div>
                    <div><div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Risk $</div>
                         <div style="color:#fff;font-size:18px;font-weight:700;">{fmt_dollar(ticket["actual_risk"])}</div></div>
                    <div><div style="color:#8b949e;font-size:10px;text-transform:uppercase;">Win Prob</div>
                         <div style="color:#3fb950;font-size:18px;font-weight:700;">{fmt_pct(ticket["win_prob"])}</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: MULTI-TICKER SCANNER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Scanner":
    st.title("Multi-Ticker Scanner")
    st.caption("P(volatile day) ranked across the full universe")

    DEFAULT_UNIVERSE = [
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "MSFT", "NVDA", "GOOGL",
        "AMZN", "META", "TSLA", "AMD",
    ]

    with st.sidebar:
        st.markdown("### Universe")
        ticker_input = st.text_area(
            "Tickers (one per line or comma-separated)",
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

    # Lift badges
    def lift_badge(lift):
        if pd.isna(lift):
            return "—"
        if lift >= 3.0:
            return "🔴 EXTREME"
        if lift >= 2.0:
            return "🟠 HIGH"
        if lift >= 1.3:
            return "🟡 ELEVATED"
        if lift >= 0.8:
            return "⚪ NORMAL"
        return "🔵 CALM"

    display_df = df.copy()
    display_df["Verdict"] = display_df["lift"].apply(lift_badge)

    cols_to_show = ["ticker", "p_vol", "lift", "Verdict", "last_close",
                    "pct_change", "rsi14", "bb_pos", "lag1_range",
                    "range_compression", "abs_gap_pct"]
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]

    rename_map = {
        "ticker": "Ticker",
        "p_vol": "P(vol)",
        "lift": "Lift",
        "last_close": "Close",
        "pct_change": "Chg %",
        "rsi14": "RSI(14)",
        "bb_pos": "BB Pos",
        "lag1_range": "Lag1 Range",
        "range_compression": "Range Cmpr",
        "abs_gap_pct": "Gap %",
    }

    display_df = display_df[cols_to_show].rename(columns=rename_map)

    # Format numeric cols
    for c in ["P(vol)", "Chg %", "Lag1 Range", "Range Cmpr", "Gap %"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"
            )
    for c in ["RSI(14)", "BB Pos"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—"
            )
    if "Lift" in display_df.columns:
        display_df["Lift"] = display_df["Lift"].apply(
            lambda v: f"{v:.2f}x" if pd.notna(v) else "—"
        )
    if "Close" in display_df.columns:
        display_df["Close"] = display_df["Close"].apply(
            lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Lift = P(vol) ÷ base-rate. >1.3 = Elevated · >2.0 = High · >3.0 = Extreme")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: ACCOUNT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Account Tracker":
    st.title("Account Tracker")
    st.caption("Track your equity, milestones, and trade log")

    try:
        state = load_account()
    except Exception as e:
        st.error(f"Could not load account state: {e}")
        st.stop()

    pnl = state.total_pnl()
    pnl_pct = state.total_pnl_pct()
    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting Equity", fmt_dollar(state.starting_equity))
    k2.metric("Current Equity", fmt_dollar(state.current_equity),
              delta=f"{'+' if pnl >= 0 else ''}{fmt_dollar(pnl)}")
    k3.metric("Total P&L %", f"{pnl_pct*100:+.1f}%")
    k4.metric("Trades", state.trade_count())
    k5.metric("Win Rate",
              fmt_pct(state.win_count() / state.trade_count())
              if state.trade_count() > 0 else "—")

    # ── Milestones ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Milestones")
    for ms in state.milestones:
        pct = min(state.current_equity / ms, 1.0) if ms > 0 else 0
        st.markdown(
            f"""
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:13px;margin-bottom:4px;">
                <span style="color:#8b949e;">
                  {fmt_dollar(state.starting_equity)} → {fmt_dollar(ms)}
                </span>
                <span style="color:#ffd633;font-weight:700;">
                  {pct*100:.1f}% there
                </span>
              </div>
              <div style="height:12px;background:rgba(139,148,158,0.15);
                          border-radius:6px;overflow:hidden;">
                <div style="width:{pct*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#3fb950,#ffd633);">
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Add trade form ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Log a Trade")
    with st.form("log_trade_form", clear_on_submit=True):
        cols = st.columns([1, 1.5, 1, 1, 1])
        trade_date  = cols[0].date_input("Date", value=date.today())
        trade_ticker = cols[1].text_input("Ticker", value="SPY")
        trade_tier  = cols[2].selectbox("Tier", ["GO_ULTRA_JACKPOT", "GO_JACKPOT", "GO_HOT"])
        trade_risk  = cols[3].number_input("Risk ($)", value=50.0, min_value=0.0, step=10.0)
        trade_pnl   = cols[4].number_input("P&L ($)", value=0.0, step=1.0)
        trade_note  = st.text_input("Note (optional)", value="")
        submitted   = st.form_submit_button("Log Trade", use_container_width=True)

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
        st.success(f"Logged: {trade_ticker} {trade_tier} — P&L {'+' if trade_pnl >= 0 else ''}{fmt_dollar(trade_pnl)}")
        st.rerun()

    # ── Set equity form ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Update Equity")
    with st.form("set_equity_form", clear_on_submit=True):
        new_equity = st.number_input("Current equity ($)", value=float(state.current_equity),
                                     min_value=0.0, step=10.0)
        set_eq = st.form_submit_button("Update", use_container_width=True)
    if set_eq:
        from src.account_state import load_state, save_state, snapshot_equity
        s = load_state(path=ACCT_PATH)
        s.current_equity = float(new_equity)
        s = snapshot_equity(s)
        save_state(s, path=ACCT_PATH)
        st.cache_data.clear()
        st.success(f"Equity updated to {fmt_dollar(new_equity)}")
        st.rerun()

    # ── Trade log ─────────────────────────────────────────────────────────────
    if state.trades:
        st.markdown("---")
        st.subheader("Trade Log")
        trades_data = []
        for t in reversed(state.trades):
            trades_data.append({
                "Date":   t.date,
                "Ticker": t.ticker,
                "Tier":   t.tier,
                "Risk":   fmt_dollar(t.risk),
                "P&L":    fmt_dollar(t.pnl),
                "Note":   t.note,
            })
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)

    # ── Equity history ────────────────────────────────────────────────────────
    if state.history and len(state.history) > 1:
        st.markdown("---")
        st.subheader("Equity Curve")
        hist_df = pd.DataFrame(state.history)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.set_index("date").sort_index()
        st.line_chart(hist_df["equity"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: WEEKDAY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Weekday Patterns":
    st.title("Weekday Volatility Patterns")
    st.caption("Which day of the week is most volatile? Most flat?")

    with st.sidebar:
        st.markdown("### Settings")
        ticker_wd = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "AAPL"], index=0)
        lookback  = st.selectbox("Lookback", [504, 252, 756], index=0,
                                 format_func=lambda v: {504: "2 years", 252: "1 year", 756: "3 years"}[v])

    with st.spinner(f"Loading {ticker_wd} history…"):
        try:
            daily = load_weekday_data(ticker_wd, lookback)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    if daily is None or len(daily) == 0:
        st.warning("No data available.")
        st.stop()

    # ── Build weekday aggregates ───────────────────────────────────────────────
    wd_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    daily["Weekday"] = daily.index.dayofweek.map(wd_map)

    vol_col = "YangZhang" if "YangZhang" in daily.columns else (
              "ATR" if "ATR" in daily.columns else None)
    rng_col = "RangePct" if "RangePct" in daily.columns else None
    body_col = "BodyPct" if "BodyPct" in daily.columns else None

    if vol_col is None and rng_col is None:
        # Compute basic range
        daily["RangePct"] = (daily["High"] - daily["Low"]) / daily["Open"]
        rng_col = "RangePct"

    agg_dict: dict = {"Close": "count"}
    if rng_col:   agg_dict[rng_col]   = "mean"
    if body_col:  agg_dict[body_col]  = "mean"
    if vol_col:   agg_dict[vol_col]   = "mean"

    wd_agg = (
        daily.groupby("Weekday")
             .agg(agg_dict)
             .rename(columns={"Close": "Sessions"})
             .reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
    )

    # ── Answer cards ──────────────────────────────────────────────────────────
    if rng_col and rng_col in wd_agg.columns:
        most_vol_day = wd_agg[rng_col].idxmax()
        most_flat_day = wd_agg[rng_col].idxmin()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1a0c0c,#3d1010);
                            border:2px solid #f85149;border-radius:14px;padding:22px 24px;">
                  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                              letter-spacing:.1em;font-weight:700;">Most Volatile Day</div>
                  <div style="font-size:44px;font-weight:900;color:#f85149;margin:6px 0;">
                    {most_vol_day}
                  </div>
                  <div style="color:#8b949e;font-size:13px;">
                    Avg range: <strong style="color:#fff;">
                    {wd_agg.loc[most_vol_day, rng_col]*100:.2f}%</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#0c1220,#10204d);
                            border:2px solid #58a6ff;border-radius:14px;padding:22px 24px;">
                  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
                              letter-spacing:.1em;font-weight:700;">Flattest Day</div>
                  <div style="font-size:44px;font-weight:900;color:#58a6ff;margin:6px 0;">
                    {most_flat_day}
                  </div>
                  <div style="color:#8b949e;font-size:13px;">
                    Avg range: <strong style="color:#fff;">
                    {wd_agg.loc[most_flat_day, rng_col]*100:.2f}%</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Bar charts ────────────────────────────────────────────────────────────
    if rng_col and rng_col in wd_agg.columns:
        st.subheader("Average Intraday Range by Weekday")
        chart_data = wd_agg[[rng_col]].copy()
        chart_data.columns = ["Avg Range %"]
        chart_data["Avg Range %"] = chart_data["Avg Range %"] * 100
        st.bar_chart(chart_data, use_container_width=True)

    if vol_col and vol_col in wd_agg.columns:
        st.subheader("Average Yang-Zhang Volatility by Weekday")
        yz_data = wd_agg[[vol_col]].copy()
        yz_data.columns = ["Avg YZ Vol"]
        st.bar_chart(yz_data, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Weekday Summary")
    show_cols = [c for c in [rng_col, body_col, vol_col, "Sessions"] if c and c in wd_agg.columns]
    tbl = wd_agg[show_cols].copy()
    for c in show_cols:
        if c != "Sessions":
            tbl[c] = tbl[c].apply(lambda v: f"{v*100:.3f}%" if pd.notna(v) else "—")
    st.dataframe(tbl, use_container_width=True)

    # ── Individual session scatter (using native Streamlit) ──────────────────
    if rng_col:
        st.markdown("---")
        st.subheader("All Sessions — Range Distribution")
        scatter_data = daily[["Weekday", rng_col]].copy()
        scatter_data[rng_col] = scatter_data[rng_col] * 100
        scatter_data.columns = ["Weekday", "Range %"]
        st.scatter_chart(scatter_data, x="Weekday", y="Range %", use_container_width=True)
