"""COMMAND CENTER — multi-ticker JACKPOT scanner daily decision board.

Single-page mission control for your trading day. Everything you need to
decide whether to trade — and what to trade — is on one screen:

  - Sticky top bar with live JS countdown to the next decision moment
  - Big hero card with TODAY'S verdict
  - Compact 5-day mini calendar (no need to open a second URL)
  - Trade tickets that appear ONLY when a signal fires
  - Per-ticker cards with three model scores + progress bars
  - Account tracker (reads ``data/account_state.json`` if it exists)
  - Quick-jump nav to the deep reports
  - Collapsible "How it works" (closed by default)

Daily ritual:
  1. Open this dashboard each morning. The hero card tells you what to do.
  2. If a GO_* signal fires: a trade ticket appears with size + strike.
  3. Mini calendar shows the surrounding 5 days for context.
  4. Account tracker shows progress toward the next $5k / $50k / $500k milestone.

Usage::

    python -m src.report_jackpot_dashboard
    python -m src.report_jackpot_dashboard --equity 500 --risk-frac 0.15
    python -m src.report_jackpot_dashboard --watch --watch-interval 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.account_state import AccountState, load_state, save_state, snapshot_equity
from src.jackpot_scanner import (
    DEFAULT_JACKPOT_UNIVERSE,
    HOT_THRESHOLD,
    JACKPOT_THRESHOLD,
    WEEKLY_CONFIRM_THRESHOLD,
    JackpotRow,
    market_phase,
    scan_jackpot_universe,
    score_jackpot_recent,
)
from src.live_quotes import LiveQuote, get_live_quotes


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
logger = logging.getLogger("report_jackpot_dashboard")


# ---------------------------------------------------------------------------
# Trade tickets
# ---------------------------------------------------------------------------


def trade_ticket(row: JackpotRow, equity: float, risk_frac: float) -> dict:
    """Concrete trade plan for a GO_JACKPOT or GO_ULTRA_JACKPOT signal."""
    is_ultra = row.signal == "GO_ULTRA_JACKPOT"
    effective_risk = risk_frac if is_ultra else risk_frac * 0.5
    risk_dollars = equity * effective_risk
    spot = row.last_close

    if spot >= 200:
        strike = round(spot)
    elif spot >= 50:
        strike = round(spot * 2) / 2.0
    else:
        strike = round(spot * 4) / 4.0

    premium_per_share = spot * row.premium_pct
    premium_per_contract = premium_per_share * 100
    n_contracts = max(int(risk_dollars / premium_per_contract), 0)

    win_rate = row.ultra_win_rate_history if is_ultra else row.win_rate_history
    avg_ret = row.ultra_avg_ret_history if is_ultra else row.avg_ret_history
    expected_value = risk_dollars * avg_ret if not pd.isna(avg_ret) else 0.0

    win_multiple = 1.0 + (avg_ret + 1.0) * 0.5 / max(win_rate, 0.01)
    expected_win = (n_contracts * premium_per_contract) * (win_multiple - 1.0)

    return {
        "spot": spot,
        "strike": strike,
        "risk_dollars": risk_dollars,
        "premium_per_contract": premium_per_contract,
        "n_contracts": n_contracts,
        "actual_risk": n_contracts * premium_per_contract,
        "expected_value": expected_value,
        "expected_win": expected_win,
        "expected_loss": -(n_contracts * premium_per_contract),
        "win_prob": win_rate,
        "is_ultra": is_ultra,
        "effective_risk_pct": effective_risk,
    }


def weekly_ticket(row: JackpotRow, equity: float, risk_frac: float) -> dict:
    """Suggested companion weekly straddle for ULTRA signals (half-size)."""
    risk_dollars = equity * risk_frac * 0.5
    spot = row.last_close

    if spot >= 200:
        strike = round(spot)
    elif spot >= 50:
        strike = round(spot * 2) / 2.0
    else:
        strike = round(spot * 4) / 4.0

    premium_per_share = spot * row.weekly_premium_pct
    premium_per_contract = premium_per_share * 100
    n_contracts = max(int(risk_dollars / premium_per_contract), 0)

    return {
        "strike": strike,
        "risk_dollars": risk_dollars,
        "premium_per_contract": premium_per_contract,
        "n_contracts": n_contracts,
        "actual_risk": n_contracts * premium_per_contract,
        "expected_avg_ret": row.ultra_weekly_avg_ret_history,
    }


# ---------------------------------------------------------------------------
# Mini calendar (5-day strip around today)
# ---------------------------------------------------------------------------


def build_mini_calendar(
    tickers: list[str],
    n_before: int = 2,
    n_after: int = 2,
    refresh_data: bool = True,
) -> list[dict]:
    """Build a compact strip: [t-n_before .. today .. t+n_after] of CALENDAR days.

    For each calendar day we collect: the strongest signal across the tickers,
    the realized outcome (only for past JACKPOT/ULTRA), and a flag for today.

    We fetch ~10 trading sessions of scoring data once, then map by calendar date.
    Future days are returned as "pending".
    """
    today = pd.Timestamp(datetime.now().date())
    per_ticker: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            per_ticker[t] = score_jackpot_recent(t, n_days=10, refresh_data=refresh_data)
        except Exception as e:
            logger.warning("Mini calendar — %s failed: %s", t, e)

    rank = {"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2, "SKIP": 1, None: 0}
    strip: list[dict] = []
    for offset in range(-n_before, n_after + 1):
        d = today + timedelta(days=offset)
        is_today = (offset == 0)
        is_future = (offset > 0)

        strongest = None
        strongest_rank = 0
        per: list[dict] = []
        for t, df in per_ticker.items():
            if d in df.index:
                row = df.loc[d]
                sig = row["signal"]
                zret = float(row["zdte_realized"]) if "zdte_realized" in row else float("nan")
                per.append({"ticker": t, "signal": sig, "zdte_realized": zret})
                rk = rank.get(sig, 0)
                if rk > strongest_rank:
                    strongest_rank = rk
                    strongest = sig
            else:
                per.append({"ticker": t, "signal": None, "zdte_realized": float("nan")})

        try:
            day_label = d.strftime("%b %#d") if sys.platform.startswith("win") else d.strftime("%b %-d")
        except Exception:
            day_label = str(d)
        strip.append({
            "date": d,
            "weekday": d.strftime("%a"),
            "day_label": day_label,
            "is_today": is_today,
            "is_future": is_future,
            "is_past": (offset < 0),
            "is_weekend": d.weekday() >= 5,
            "strongest": strongest,
            "tickers": per,
        })
    return strip


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Command Center · {as_of}</title>
<meta http-equiv="refresh" content="{refresh_seconds}"/>
<style>
  :root {{
    --bg:#0a0d12; --panel:#161b22; --panel-2:#1c232c; --line:#30363d;
    --text:#e6edf3; --muted:#8b949e; --dim:#6e7681;
    --good:#3fb950; --bad:#f85149; --warn:#d29922; --gold:#ffd633; --blue:#58a6ff;
    --gradient: linear-gradient(135deg, #58a6ff, #d2a8ff);
    --gradient-gold: linear-gradient(90deg, #ffd633, #ff8c33, #ffd633);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font: 14px/1.55 -apple-system, "Segoe UI", Inter, Arial, sans-serif;
    padding: 0 0 80px;
  }}
  a {{ color: var(--blue); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ background: #0d1117; padding: 2px 6px; border-radius: 4px; font-size: 0.92em; }}

  /* ------- Sticky top bar ------- */
  .topbar {{
    position: sticky; top: 0; z-index: 50;
    background: rgba(10, 13, 18, 0.95); backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
    padding: 10px 24px;
    display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
  }}
  .topbar .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .topbar .ph-label {{ font-weight: 700; font-size: 13px; letter-spacing: 0.04em; }}
  .topbar .ph-countdown {{
    font-variant-numeric: tabular-nums;
    font-size: 13px; color: var(--muted);
  }}
  .topbar .ph-countdown strong {{ color: #fff; }}
  .topbar .spacer {{ flex: 1; }}
  .topbar .equity-pill {{
    font-size: 12px; color: var(--muted);
    background: var(--panel); border: 1px solid var(--line);
    padding: 6px 12px; border-radius: 20px;
    font-variant-numeric: tabular-nums;
  }}
  .topbar .equity-pill strong {{ color: #fff; font-size: 13px; }}
  .topbar.live .dot {{ background: var(--good);
                       box-shadow: 0 0 8px var(--good);
                       animation: pulse 1.6s ease-in-out infinite; }}
  .topbar.live .ph-label {{ color: var(--good); }}
  .topbar.pending .dot {{ background: var(--warn); }}
  .topbar.pending .ph-label {{ color: var(--warn); }}
  .topbar.closed .dot {{ background: var(--blue); }}
  .topbar.closed .ph-label {{ color: var(--blue); }}
  .topbar.weekend .dot {{ background: var(--muted); }}
  .topbar.weekend .ph-label {{ color: var(--muted); }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

  /* ------- Live price strip ------- */
  .liveprice {{
    background: var(--panel); border-bottom: 1px solid var(--line);
    padding: 10px 24px; display: flex; gap: 18px; align-items: center;
    flex-wrap: wrap; overflow-x: auto;
  }}
  .liveprice .label {{
    font-size: 10px; color: var(--muted); letter-spacing: 0.1em;
    text-transform: uppercase; font-weight: 700; flex-shrink: 0;
  }}
  .liveprice .label .freshness {{ color: var(--warn); margin-left: 6px; }}
  .liveprice .lp-card {{
    display: flex; align-items: baseline; gap: 6px;
    font-variant-numeric: tabular-nums; flex-shrink: 0;
  }}
  .liveprice .lp-card .lp-tkr {{ font-weight: 700; color: #fff; font-size: 13px; }}
  .liveprice .lp-card .lp-last {{ color: #fff; font-size: 14px; }}
  .liveprice .lp-card .lp-chg.up {{ color: var(--good); font-size: 11px; }}
  .liveprice .lp-card .lp-chg.down {{ color: var(--bad); font-size: 11px; }}
  .liveprice .lp-card .lp-chg.flat {{ color: var(--muted); font-size: 11px; }}
  .liveprice .lp-asof {{
    font-size: 10px; color: var(--dim); margin-left: auto; flex-shrink: 0;
  }}

  /* ------- Layout container ------- */
  .container {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}

  /* ------- Hero verdict ------- */
  .hero {{
    background: linear-gradient(135deg, #0d1117 0%, #1a2133 100%);
    border: 1px solid #2c3a5e; border-radius: 16px;
    padding: 32px 36px; margin-bottom: 20px;
    position: relative; overflow: hidden;
  }}
  .hero.is-ultra {{
    background: linear-gradient(135deg, #1a1208 0%, #3d2f10 50%, #1a1208 100%);
    border: 2px solid var(--gold);
    box-shadow: 0 0 40px rgba(255,214,51,0.30);
  }}
  .hero.is-jackpot {{
    background: linear-gradient(135deg, #0d1f14 0%, #1c4a30 100%);
    border: 2px solid var(--good);
    box-shadow: 0 0 28px rgba(63,185,80,0.22);
  }}
  .hero.is-hot {{
    background: linear-gradient(135deg, #1f1808 0%, #463812 100%);
    border: 2px solid var(--warn);
  }}
  .hero .verdict-lab {{
    color: var(--muted); font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; font-weight: 700; margin-bottom: 6px;
  }}
  .hero h1 {{
    font-size: 46px; line-height: 1.05; margin: 0 0 12px;
    font-weight: 900; letter-spacing: -0.02em; color: #fff;
  }}
  .hero h1.is-ultra {{
    background: var(--gradient-gold); -webkit-background-clip: text;
    background-clip: text; color: transparent; background-size: 200% 100%;
    animation: gold-shimmer 4s linear infinite;
  }}
  .hero h1.is-jackpot {{ color: var(--good); }}
  .hero h1.is-hot {{ color: var(--warn); }}
  .hero h1.is-skip {{ color: var(--muted); }}
  @keyframes gold-shimmer {{
    0% {{ background-position: 0% 50%; }}
    100% {{ background-position: 200% 50%; }}
  }}
  .hero .sub {{ color: var(--muted); font-size: 16px; margin: 0 0 6px;
                line-height: 1.45; }}
  .hero .sub strong {{ color: #fff; }}

  .hero-stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 18px; margin-top: 20px;
    padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.10);
  }}
  .hero-stats .stat .num {{ font-size: 22px; font-weight: 800;
                             font-variant-numeric: tabular-nums; line-height: 1; }}
  .hero-stats .stat .num.gold {{ color: var(--gold); }}
  .hero-stats .stat .num.good {{ color: var(--good); }}
  .hero-stats .stat .num.warn {{ color: var(--warn); }}
  .hero-stats .stat .num.bad {{ color: var(--bad); }}
  .hero-stats .stat .num.muted {{ color: var(--muted); }}
  .hero-stats .stat .lab {{ color: var(--muted); font-size: 11px;
                              text-transform: uppercase; letter-spacing: 0.06em;
                              margin-top: 6px; }}

  /* ------- Mini calendar strip ------- */
  .mini-cal {{
    display: grid; gap: 8px; margin-bottom: 24px;
    grid-template-columns: repeat(5, 1fr);
  }}
  @media (max-width: 720px) {{ .mini-cal {{ grid-template-columns: repeat(3, 1fr); }} }}
  .mc-day {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 10px; text-align: center;
    min-height: 92px; display: flex; flex-direction: column; justify-content: space-between;
  }}
  .mc-day.today {{
    border: 2px solid var(--blue);
    box-shadow: 0 0 18px rgba(88,166,255,0.30);
  }}
  .mc-day.future {{ opacity: 0.55; }}
  .mc-day.weekend {{ opacity: 0.35; }}
  .mc-day .mc-wd {{ font-size: 10px; color: var(--muted);
                     letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }}
  .mc-day .mc-dt {{ font-size: 20px; font-weight: 700; color: #fff;
                     line-height: 1; margin-top: 2px; }}
  .mc-sig {{
    padding: 3px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 800; letter-spacing: 0.04em;
    text-transform: uppercase; margin-top: 8px;
  }}
  .mc-sig.go-ultra-jackpot {{
    background: var(--gradient-gold); background-size: 200% 100%;
    color: #0c1117; animation: gold-shimmer 3s linear infinite;
  }}
  .mc-sig.go-jackpot {{ background: var(--good); color: #0c1117; }}
  .mc-sig.go-hot {{ background: var(--warn); color: #0c1117; }}
  .mc-sig.skip {{ background: rgba(139,148,158,0.20); color: var(--muted); }}
  .mc-sig.pending {{ background: rgba(88,166,255,0.15); color: var(--blue); }}
  .mc-sig.weekend-tag {{ background: transparent; color: var(--dim); }}
  .mc-out {{ margin-top: 4px; font-size: 10px; font-weight: 700;
              font-variant-numeric: tabular-nums; }}
  .mc-out.win {{ color: var(--good); }}
  .mc-out.lose {{ color: var(--bad); }}

  /* ------- Trade tickets (only visible when a signal fires) ------- */
  .tickets {{ margin-bottom: 24px; }}
  .ticket {{
    background: rgba(63,185,80,0.10);
    border: 1px dashed var(--good);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
  }}
  .ticket.ultra {{
    background: rgba(255,214,51,0.12); border: 1px dashed var(--gold);
  }}
  .ticket.weekly-companion {{
    background: rgba(88,166,255,0.10); border: 1px dashed var(--blue);
    margin-top: -4px;
  }}
  .ticket h3 {{ margin: 0 0 12px; font-size: 14px; color: var(--good);
                text-transform: uppercase; letter-spacing: 0.06em; font-weight: 800; }}
  .ticket.ultra h3 {{ color: var(--gold); }}
  .ticket.weekly-companion h3 {{ color: var(--blue); }}
  .ticket-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                   gap: 14px; }}
  .ticket-grid .cell .lab {{ color: var(--muted); font-size: 10px;
                              text-transform: uppercase; letter-spacing: 0.05em; }}
  .ticket-grid .cell .val {{ color: #fff; font-size: 16px; font-weight: 700;
                              font-variant-numeric: tabular-nums; margin-top: 3px; }}
  .ticket-grid .cell .val.good {{ color: var(--good); }}
  .ticket-grid .cell .val.bad {{ color: var(--bad); }}
  .ticket-grid .cell .val.gold {{ color: var(--gold); }}

  /* ------- Section heads ------- */
  .section-head {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 24px 0 12px;
  }}
  .section-head h2 {{
    font-size: 13px; margin: 0; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;
  }}
  .section-head .meta {{ font-size: 12px; color: var(--dim); }}

  /* ------- Ticker grid ------- */
  .ticker-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;
  }}
  @media (max-width: 900px) {{ .ticker-grid {{ grid-template-columns: 1fr; }} }}
  .ticker-card {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px;
  }}
  .ticker-card.go-ultra-jackpot {{
    background: linear-gradient(135deg, #1a2f0c 0%, #2d5510 50%, #3d7515 100%);
    border: 2px solid var(--gold);
    box-shadow: 0 0 22px rgba(255,214,51,0.20);
  }}
  .ticker-card.go-jackpot {{
    background: linear-gradient(135deg, #0f2818 0%, #1c4a30 100%);
    border: 2px solid var(--good);
  }}
  .ticker-card.go-hot {{
    background: linear-gradient(135deg, #2d2410 0%, #463812 100%);
    border: 2px solid var(--warn);
  }}
  .tc-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .tc-head h3 {{ font-size: 24px; margin: 0; font-weight: 800; color: #fff;
                  letter-spacing: -0.01em; }}
  .tc-head .price {{ font-variant-numeric: tabular-nums; font-size: 16px;
                      color: #c9d1d9; text-align: right; }}
  .tc-head .price small {{ color: var(--muted); font-weight: normal; }}
  .tc-head .price .live-asof {{ color: var(--dim); font-size: 10px;
                                 display: block; margin-top: 2px; }}
  .tc-intraday {{
    margin-top: 8px; padding: 6px 10px;
    background: rgba(13, 17, 23, 0.6); border-radius: 6px;
    display: flex; gap: 12px; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .tc-intraday span strong {{ color: #c9d1d9; font-weight: 600; }}
  .tc-sig {{ display: inline-block; padding: 5px 10px; border-radius: 5px;
              font-size: 11px; font-weight: 800; letter-spacing: 0.06em;
              text-transform: uppercase; margin-top: 10px; }}
  .tc-sig.go-ultra-jackpot {{
    background: var(--gradient-gold); background-size: 200% 100%;
    color: #0c1117; animation: gold-shimmer 3s linear infinite;
  }}
  .tc-sig.go-jackpot {{ background: var(--good); color: #0c1117; }}
  .tc-sig.go-hot {{ background: var(--warn); color: #0c1117; }}
  .tc-sig.skip {{ background: rgba(139,148,158,0.20); color: var(--muted); }}

  .scores {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px;
              margin-top: 14px; }}
  .score .lab {{ color: var(--muted); font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .score .val {{ font-size: 18px; font-weight: 700; margin-top: 2px;
                  font-variant-numeric: tabular-nums; }}
  .score .bar {{ height: 5px; background: rgba(139,148,158,0.18); border-radius: 3px;
                  overflow: hidden; margin-top: 6px; }}
  .score .bar-fill {{ height: 100%; transition: width 0.4s; }}
  .score .bar-fill.vol {{ background: var(--warn); }}
  .score .bar-fill.pnl {{ background: var(--good); }}
  .score .bar-fill.weekly {{ background: var(--blue); }}
  .score .bar-fill.over {{ background: var(--good); box-shadow: 0 0 6px var(--good); }}
  .score .bar-fill.over-gold {{ background: var(--gold); box-shadow: 0 0 8px var(--gold); }}

  .tc-meta {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 14px;
              padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.06);
              color: var(--muted); font-size: 11px; }}
  .tc-meta strong {{ color: #c9d1d9; }}

  /* ------- Account tracker ------- */
  .account {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 20px 24px; margin-top: 24px;
  }}
  .account-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .account-head h2 {{ font-size: 13px; margin: 0; color: var(--muted);
                       text-transform: uppercase; letter-spacing: 0.1em; }}
  .account-head .pl {{ font-size: 18px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .account-head .pl.good {{ color: var(--good); }}
  .account-head .pl.bad {{ color: var(--bad); }}
  .account-head .pl.flat {{ color: var(--muted); }}

  .progress-row {{ margin-top: 16px; }}
  .progress-row .ms-lab {{ display: flex; justify-content: space-between;
                            font-size: 12px; margin-bottom: 6px; }}
  .progress-row .ms-lab .from {{ color: var(--muted); }}
  .progress-row .ms-lab .to {{ color: var(--gold); font-weight: 700; }}
  .progress-bar {{ height: 14px; background: rgba(139,148,158,0.10);
                    border-radius: 7px; overflow: hidden; position: relative; }}
  .progress-bar .fill {{
    height: 100%; background: linear-gradient(90deg, var(--good), var(--gold));
    transition: width 0.5s;
  }}
  .progress-bar .fill-pct {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    color: #fff; font-size: 11px; font-weight: 700;
    text-shadow: 0 0 4px rgba(0,0,0,0.8);
  }}

  .account-stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 14px; margin-top: 18px;
  }}
  .account-stats .s .lab {{ color: var(--muted); font-size: 10px;
                              text-transform: uppercase; letter-spacing: 0.05em; }}
  .account-stats .s .val {{ font-size: 18px; font-weight: 700; margin-top: 2px;
                              font-variant-numeric: tabular-nums; color: #fff; }}

  .account-edit {{
    margin-top: 16px; font-size: 12px; color: var(--muted);
    padding: 10px 14px; background: rgba(13,17,23,0.6); border-radius: 6px;
    border: 1px dashed var(--line);
  }}
  .account-edit code {{ color: var(--blue); }}

  /* ------- Quick-jump nav ------- */
  .quickjump {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px; margin-top: 18px;
  }}
  .qj-card {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 14px 16px;
    transition: border-color 0.2s, transform 0.2s;
  }}
  .qj-card:hover {{ border-color: var(--blue); transform: translateY(-2px); text-decoration: none; }}
  .qj-card .qj-title {{ color: #fff; font-weight: 700; font-size: 13px; }}
  .qj-card .qj-desc {{ color: var(--muted); font-size: 11px; margin-top: 4px; }}

  /* ------- Collapsible "How it works" ------- */
  details.howto {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 18px; margin-top: 24px;
  }}
  details.howto summary {{ cursor: pointer; color: var(--blue); font-weight: 700;
                            font-size: 13px; }}
  details.howto summary:hover {{ text-decoration: underline; }}
  details.howto[open] {{ padding-bottom: 18px; }}
  details.howto .howto-body {{ margin-top: 14px; color: var(--muted); font-size: 13px;
                                line-height: 1.7; }}
  details.howto .howto-body ul {{ margin: 0; padding-left: 22px; }}
  details.howto .howto-body strong {{ color: #c9d1d9; }}

  /* ------- Footer ------- */
  .footer {{ color: var(--dim); font-size: 11px; text-align: center;
              margin-top: 30px; padding: 20px; }}
</style>
</head><body>

<div class="topbar {phase_class}" id="topbar">
  <div class="dot"></div>
  <div class="ph-label" id="ph-label">{phase_label}</div>
  <div class="ph-countdown" id="ph-countdown">{phase_sub}</div>
  <div class="spacer"></div>
  {equity_pill}
</div>

{liveprice_strip}

<div class="container">

  {hero_html}

  <div class="section-head"><h2>Last 2 · Today · Next 2</h2>
    <span class="meta">All times in your local timezone · <a href="JACKPOT_weekly_calendar.html">full week view ↗</a></span></div>
  <div class="mini-cal">
    {mini_cal_html}
  </div>

  {tickets_html}

  <div class="section-head"><h2>Ticker board · ranked by signal strength</h2></div>
  <div class="ticker-grid">
    {tickers_html}
  </div>

  {account_html}

  <div class="section-head"><h2>Deep dives</h2></div>
  <div class="quickjump">
    <a class="qj-card" href="JACKPOT_weekly_calendar.html">
      <div class="qj-title">Full weekly calendar</div>
      <div class="qj-desc">Mon-Fri grids for last week + this week with realized outcomes</div>
    </a>
    <a class="qj-card" href="SPY_jackpot_compound.html">
      <div class="qj-title">$500 → $5k / $50k / $500k</div>
      <div class="qj-desc">Compounding sim — time-to-milestone projections</div>
    </a>
    <a class="qj-card" href="SPY_edge_finder.html">
      <div class="qj-title">Edge finder</div>
      <div class="qj-desc">Five conditional filters that sharpen the signal</div>
    </a>
    <a class="qj-card" href="SPY_continuous_18mo.html">
      <div class="qj-title">18-month sim</div>
      <div class="qj-desc">Continuous compounding from $500</div>
    </a>
    <a class="qj-card" href="SPY_moonshot_signal.html">
      <div class="qj-title">SPY moonshot signal</div>
      <div class="qj-desc">Single-ticker traffic light (legacy)</div>
    </a>
    <a class="qj-card" href="SPY_volatility_patterns.html">
      <div class="qj-title">Volatility patterns</div>
      <div class="qj-desc">Feature importances + pattern catalog</div>
    </a>
  </div>

  <details class="howto">
    <summary>How the four signals work</summary>
    <div class="howto-body">
      <ul>
        <li><strong style="color:var(--gold);">★★★ ULTRA JACKPOT</strong> — all three models fire (vol classifier ≥ {hot_thr}, 0DTE-PnL ≥ {jp_thr}, weekly-PnL ≥ {wkly_thr}). Historical SPY: ~83% win rate, avg +114%/$.
            <strong>Full size 0DTE straddle + companion weekly straddle.</strong></li>
        <li><strong style="color:var(--good);">★ JACKPOT</strong> — vol + 0DTE-PnL fire but the weekly model does NOT confirm. Historical: ~65% win rate. <strong>Half size; no companion weekly.</strong></li>
        <li><strong style="color:var(--warn);">HOT</strong> — only the vol classifier fires. Lower confidence; usually skip during early-stage compounding.</li>
        <li><strong style="color:var(--muted);">SKIP</strong> — no model fires. No trade today. Most days are SKIP — that's the model working correctly.</li>
      </ul>
      <p style="margin-top: 14px;"><strong>Signal timing</strong> — 29 of 30 features are locked the previous afternoon; the final feature (gap_pct) fixes at the 9:30 AM ET opening print. After ~9:50 AM ET (Yahoo data settles), the signal is final for the rest of the day. The dashboard auto-refreshes every {refresh_seconds}s and refetches Yahoo data every 15 minutes.</p>
      <p><strong>Honest caveats</strong> — backtest stats are walk-forward, monthly-retrained, out-of-sample. Real-world fills will be 20-40% worse than the model assumes due to slippage. Position-size as if the live edge is ~70% of the backtest edge.</p>
    </div>
  </details>

  <div class="footer">
    Refreshed {as_of_full} · auto-refresh every {refresh_seconds}s · data freshness 15min
  </div>

</div>

<script>
// ---- Live countdown to the next decision-window milestone ----
// Phase data is injected as JSON for the JS to read.
const PHASE_DATA = {phase_json};

function fmtCountdown(ms) {{
  if (ms <= 0) return "now";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${{h}}h ${{m}}m ${{String(sec).padStart(2,'0')}}s`;
  if (m > 0) return `${{m}}m ${{String(sec).padStart(2,'0')}}s`;
  return `${{sec}}s`;
}}

// Target Date objects parsed from ISO strings, in the user's local timezone.
const target = PHASE_DATA.target_iso ? new Date(PHASE_DATA.target_iso) : null;

function update() {{
  if (!target) return;
  const now = new Date();
  const remain = target - now;
  const label = document.getElementById("ph-countdown");
  if (!label) return;

  if (PHASE_DATA.phase === "PRE_OPEN") {{
    label.innerHTML = `Cash market opens in <strong>${{fmtCountdown(remain)}}</strong> (9:30 AM ET).`;
  }} else if (PHASE_DATA.phase === "OPEN_PENDING_DATA") {{
    label.innerHTML = `Data settles in <strong>${{fmtCountdown(remain)}}</strong> — signal not yet final.`;
  }} else if (PHASE_DATA.phase === "OPEN_LIVE") {{
    label.innerHTML = `Closing bell in <strong>${{fmtCountdown(remain)}}</strong> · signal is final, decide soon.`;
  }} else if (PHASE_DATA.phase === "AFTER_HOURS") {{
    label.innerHTML = `Next open in <strong>${{fmtCountdown(remain)}}</strong> · signal frozen until then.`;
  }} else if (PHASE_DATA.phase === "WEEKEND") {{
    label.innerHTML = `Markets reopen in <strong>${{fmtCountdown(remain)}}</strong> (Mon 9:30 AM ET).`;
  }}

  // When the countdown crosses zero, full-page refresh picks up new phase.
  if (remain <= 0) {{ window.location.reload(); }}
}}
update();
setInterval(update, 1000);
</script>

</body></html>
"""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------


def _signal_pretty(signal: str) -> str:
    return {
        "GO_ULTRA_JACKPOT": "★★★ ULTRA JACKPOT",
        "GO_JACKPOT": "★ JACKPOT",
        "GO_HOT": "HOT",
        "SKIP": "SKIP",
    }.get(signal, signal)


def _liveprice_strip_html(quotes: dict[str, LiveQuote], tickers: list[str]) -> str:
    """Top-of-page strip showing live (15-min delayed) prices for each ticker."""
    if not quotes:
        return ""

    cards = []
    latest_asof = None
    sources = set()
    for t in tickers:
        q = quotes.get(t.upper())
        if q is None:
            continue
        sources.add(q.source)
        if latest_asof is None or q.as_of > latest_asof:
            latest_asof = q.as_of
        chg = q.change_pct() * 100
        if chg > 0.02:
            cls = "up"
            sign = "+"
        elif chg < -0.02:
            cls = "down"
            sign = ""
        else:
            cls = "flat"
            sign = ""
        cards.append(
            f"<div class='lp-card'>"
            f"<span class='lp-tkr'>{q.ticker}</span>"
            f"<span class='lp-last'>${q.last:,.2f}</span>"
            f"<span class='lp-chg {cls}'>{sign}{chg:.2f}%</span>"
            f"</div>"
        )

    if not cards:
        return ""

    src_label = "+".join(sorted(sources))
    asof_label = latest_asof.strftime("%H:%M ET") if latest_asof else "—"
    return (
        f"<div class='liveprice'>"
        f"<div class='label'>LIVE QUOTES "
        f"<span class='freshness'>delayed ~15 min · {src_label}</span></div>"
        + "".join(cards)
        + f"<div class='lp-asof'>last bar {asof_label}</div>"
        f"</div>"
    )


def _signal_class(signal: str | None) -> str:
    return (signal or "skip").lower().replace("_", "-")


def _hero_html(rows: list[JackpotRow], phase: dict, equity: float) -> str:
    """The big top verdict block."""
    n_ultra = sum(1 for r in rows if r.signal == "GO_ULTRA_JACKPOT")
    n_jp = sum(1 for r in rows if r.signal == "GO_JACKPOT")
    n_hot = sum(1 for r in rows if r.signal == "GO_HOT")
    n_skip = sum(1 for r in rows if r.signal == "SKIP")

    strongest_rows = [r for r in rows if r.signal in ("GO_ULTRA_JACKPOT", "GO_JACKPOT", "GO_HOT")]
    strongest_rows.sort(
        key=lambda r: ({"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2}.get(r.signal, 0), r.p_pnl),
        reverse=True,
    )

    if n_ultra > 0:
        klass = "is-ultra"
        h1_class = "is-ultra"
        verdict_lab = "TODAY'S VERDICT"
        s = strongest_rows[0]
        h1 = f"GO — ULTRA on {s.ticker}"
        sub = (f"All three models fire. <strong>{s.ticker} P(vol) {s.p_vol*100:.0f}% · "
               f"P(0DTE) {s.p_pnl*100:.0f}% · P(weekly) {s.p_weekly*100:.0f}%.</strong> "
               f"Take the full-size 0DTE straddle and the companion weekly. Historical edge: "
               f"~{s.ultra_win_rate_history*100:.0f}% win rate, avg +{s.ultra_avg_ret_history*100:.0f}%/$.")
    elif n_jp > 0:
        klass = "is-jackpot"
        h1_class = "is-jackpot"
        verdict_lab = "TODAY'S VERDICT"
        s = strongest_rows[0]
        h1 = f"GO — JACKPOT on {s.ticker}"
        sub = (f"Vol + 0DTE-PnL fire but the weekly model does not confirm. <strong>Half-size 0DTE straddle on {s.ticker}.</strong> "
               f"Historical: ~{s.win_rate_history*100:.0f}% win rate. In early compounding you can also skip this tier.")
    elif n_hot > 0:
        klass = "is-hot"
        h1_class = "is-hot"
        verdict_lab = "TODAY'S VERDICT"
        s = strongest_rows[0]
        h1 = f"CAUTION — {s.ticker} HOT only"
        sub = (f"Only the vol classifier fires on <strong>{s.ticker}</strong>. Lower confidence. "
               f"Skip in early-stage compounding; consider a tiny exploratory size if you're already profitable.")
    else:
        klass = ""
        h1_class = "is-skip"
        verdict_lab = "TODAY'S VERDICT"
        h1 = "STAND ASIDE"
        # Show the strongest non-firing score as context.
        if rows:
            best_pnl = max(rows, key=lambda r: r.p_pnl)
            sub = (f"All four tickers SKIP. Strongest read is <strong>{best_pnl.ticker}</strong> "
                   f"at P(0DTE) {best_pnl.p_pnl*100:.1f}% (needs ≥ {JACKPOT_THRESHOLD*100:.0f}% to fire). "
                   f"No trade today — this is most days. The model wins by being patient.")
        else:
            sub = "No tickers scored — check the errors panel below."

    # Hero stats
    total_at_risk = 0.0
    total_expected = 0.0
    for r in rows:
        if r.signal in ("GO_ULTRA_JACKPOT", "GO_JACKPOT"):
            t = trade_ticket(r, equity, 0.15)
            total_at_risk += t["actual_risk"]
            total_expected += t["expected_value"]
            if r.signal == "GO_ULTRA_JACKPOT":
                wt = weekly_ticket(r, equity, 0.15)
                total_at_risk += wt["actual_risk"]

    decide = phase["phase"] in ("OPEN_LIVE",)
    decide_str = "<span style='color:var(--good);'>NOW</span>" if decide else (
        "<span style='color:var(--warn);'>after 9:50 AM ET</span>" if phase["phase"] in ("PRE_OPEN", "OPEN_PENDING_DATA")
        else "<span style='color:var(--blue);'>next session</span>" if phase["phase"] == "AFTER_HOURS"
        else "<span style='color:var(--muted);'>Mon 9:50 AM ET</span>"
    )

    stats_html = f"""
    <div class="hero-stats">
      <div class="stat">
        <div class="num gold">{n_ultra}</div>
        <div class="lab">★★★ ULTRA</div>
      </div>
      <div class="stat">
        <div class="num good">{n_jp}</div>
        <div class="lab">★ JACKPOT</div>
      </div>
      <div class="stat">
        <div class="num warn">{n_hot}</div>
        <div class="lab">HOT only</div>
      </div>
      <div class="stat">
        <div class="num muted">{n_skip}</div>
        <div class="lab">SKIP</div>
      </div>
      <div class="stat">
        <div class="num">${total_at_risk:,.0f}</div>
        <div class="lab">Total at risk if you take them all</div>
      </div>
      <div class="stat">
        <div class="num">{decide_str}</div>
        <div class="lab">Decide</div>
      </div>
    </div>
    """

    return f"""
    <div class="hero {klass}">
      <div class="verdict-lab">{verdict_lab}</div>
      <h1 class="{h1_class}">{h1}</h1>
      <p class="sub">{sub}</p>
      {stats_html}
    </div>
    """


def _mini_calendar_html(strip: list[dict]) -> str:
    if not strip:
        return "<div class='mc-day skip'><div class='mc-wd'>—</div><div class='mc-dt'>—</div></div>" * 5

    cards = []
    for day in strip:
        klass = ["mc-day"]
        if day["is_today"]:
            klass.append("today")
        if day["is_future"] and not day["is_today"]:
            klass.append("future")
        if day["is_weekend"]:
            klass.append("weekend")

        sig = day["strongest"]
        if day["is_weekend"]:
            sig_html = "<div class='mc-sig weekend-tag'>—</div>"
        elif day["is_future"]:
            sig_html = "<div class='mc-sig pending'>PENDING</div>"
        elif sig:
            label_map = {
                "GO_ULTRA_JACKPOT": "★★★ ULTRA",
                "GO_JACKPOT": "★ JP",
                "GO_HOT": "HOT",
                "SKIP": "SKIP",
            }
            sig_html = f"<div class='mc-sig {_signal_class(sig)}'>{label_map.get(sig, sig)}</div>"
        else:
            sig_html = "<div class='mc-sig skip'>—</div>"

        # Outcome on past days where a JACKPOT or ULTRA fired
        outcome_html = ""
        if day["is_past"] and sig in ("GO_JACKPOT", "GO_ULTRA_JACKPOT"):
            zret = next((t["zdte_realized"] for t in day["tickers"]
                          if t["signal"] in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")), float("nan"))
            if not pd.isna(zret):
                cls = "win" if zret > 0 else "lose"
                sign = "+" if zret > 0 else ""
                outcome_html = f"<div class='mc-out {cls}'>{sign}{zret*100:.0f}%</div>"

        try:
            day_num = day["date"].strftime("%#d") if sys.platform.startswith("win") else day["date"].strftime("%-d")
        except Exception:
            day_num = str(day["date"].day if hasattr(day["date"], "day") else day["date"])
        cards.append(
            f"<div class='{' '.join(klass)}'>"
            f"<div><div class='mc-wd'>{day['weekday']}</div>"
            f"<div class='mc-dt'>{day_num}</div></div>"
            f"{sig_html}"
            f"{outcome_html}"
            f"</div>"
        )
    return "\n".join(cards)


def _trade_tickets_html(rows: list[JackpotRow], equity: float, risk_frac: float) -> str:
    """Render trade-ticket panels. Returns empty string when no signals fire."""
    actionable = [r for r in rows if r.signal in ("GO_JACKPOT", "GO_ULTRA_JACKPOT")]
    if not actionable:
        return ""

    cards = []
    for r in actionable:
        ticket = trade_ticket(r, equity, risk_frac)
        is_ultra = ticket["is_ultra"]
        title = f"★★★ ULTRA · {r.ticker} ${ticket['strike']:g} 0DTE straddle" if is_ultra \
                else f"★ JACKPOT · {r.ticker} ${ticket['strike']:g} 0DTE straddle (half-size)"
        klass = "ultra" if is_ultra else ""
        wr = ticket["win_prob"]
        wr_str = f"{wr*100:.0f}%" if not pd.isna(wr) else "—"
        cards.append(f"""
        <div class="ticket {klass}">
          <h3>{title}</h3>
          <div class="ticket-grid">
            <div class="cell"><div class="lab">Contracts</div>
              <div class="val {'gold' if is_ultra else 'good'}">{ticket['n_contracts']} × call + {ticket['n_contracts']} × put</div></div>
            <div class="cell"><div class="lab">Strike</div>
              <div class="val">${ticket['strike']:g}</div></div>
            <div class="cell"><div class="lab">Premium / pair</div>
              <div class="val">${ticket['premium_per_contract']:.0f}</div></div>
            <div class="cell"><div class="lab">Total at risk</div>
              <div class="val">${ticket['actual_risk']:,.0f}</div></div>
            <div class="cell"><div class="lab">Expected win</div>
              <div class="val good">+${ticket['expected_win']:,.0f}</div></div>
            <div class="cell"><div class="lab">Max loss</div>
              <div class="val bad">−${ticket['actual_risk']:,.0f}</div></div>
            <div class="cell"><div class="lab">Hist. win rate</div>
              <div class="val">{wr_str}</div></div>
          </div>
        </div>
        """)
        # Companion weekly for ULTRA
        if is_ultra and not pd.isna(r.ultra_weekly_avg_ret_history):
            wt = weekly_ticket(r, equity, risk_frac)
            avg_str = f"{wt['expected_avg_ret']*100:+.0f}% per $" if not pd.isna(wt["expected_avg_ret"]) else "—"
            cards.append(f"""
            <div class="ticket weekly-companion">
              <h3>+ Companion weekly straddle</h3>
              <div class="ticket-grid">
                <div class="cell"><div class="lab">Contracts</div>
                  <div class="val">{wt['n_contracts']} × call + {wt['n_contracts']} × put</div></div>
                <div class="cell"><div class="lab">Strike · Expiry</div>
                  <div class="val">${wt['strike']:g} · next Friday</div></div>
                <div class="cell"><div class="lab">Premium / pair</div>
                  <div class="val">${wt['premium_per_contract']:.0f}</div></div>
                <div class="cell"><div class="lab">Total at risk</div>
                  <div class="val">${wt['actual_risk']:,.0f}</div></div>
                <div class="cell"><div class="lab">Hist. avg return</div>
                  <div class="val good">{avg_str}</div></div>
              </div>
            </div>
            """)
    return f'<div class="section-head"><h2>Trade tickets · take these positions</h2></div><div class="tickets">{"".join(cards)}</div>'


def _ticker_card_html(row: JackpotRow, quote: LiveQuote | None = None) -> str:
    p_vol, p_pnl, p_wkly = row.p_vol, row.p_pnl, row.p_weekly
    is_ultra = row.signal == "GO_ULTRA_JACKPOT"

    def bar_class(val: float, thr: float, ultra: bool) -> str:
        if ultra and val >= thr:
            return "over-gold"
        if val >= thr:
            return "over"
        return ""

    def val_color(val: float, thr: float, ultra: bool) -> str:
        if ultra and val >= thr:
            return "var(--gold)"
        if val >= thr:
            return "var(--good)"
        return "#c9d1d9"

    vol_class = bar_class(p_vol, HOT_THRESHOLD, is_ultra) or "vol"
    pnl_class = bar_class(p_pnl, JACKPOT_THRESHOLD, is_ultra) or "pnl"
    wkly_class = bar_class(p_wkly, WEEKLY_CONFIRM_THRESHOLD, is_ultra) or "weekly"

    wr_jp = f"{row.win_rate_history*100:.0f}%" if not pd.isna(row.win_rate_history) else "—"
    wr_ultra = f"{row.ultra_win_rate_history*100:.0f}%" if not pd.isna(row.ultra_win_rate_history) else "—"
    avg_ultra = f"{row.ultra_avg_ret_history*100:+.0f}%" if not pd.isna(row.ultra_avg_ret_history) else "—"

    # Live-quote-aware price line + intraday range chip.
    if quote is not None:
        live_pct = quote.change_pct() * 100
        price_block = (
            f"<div class='price'>${quote.last:,.2f} "
            f"<small style='color:{'var(--good)' if live_pct >= 0 else 'var(--bad)'};'>"
            f"{live_pct:+.2f}%</small>"
            f"<span class='live-asof'>as of {quote.as_of:%H:%M} ET · {quote.source}</span>"
            f"</div>"
        )
        intraday_html = (
            f"<div class='tc-intraday'>"
            f"<span>Open <strong>${quote.day_open:,.2f}</strong></span>"
            f"<span>High <strong>${quote.day_high:,.2f}</strong></span>"
            f"<span>Low <strong>${quote.day_low:,.2f}</strong></span>"
            f"<span>Range <strong>{quote.day_range_pct()*100:.2f}%</strong></span>"
            f"</div>"
        )
    else:
        price_block = (
            f"<div class='price'>${row.last_close:.2f} "
            f"<small>{row.pct_change*100:+.2f}%</small>"
            f"<span class='live-asof'>daily bar · no live quote</span>"
            f"</div>"
        )
        intraday_html = ""

    return f"""
    <div class="ticker-card {_signal_class(row.signal)}">
      <div class="tc-head">
        <h3>{row.ticker}</h3>
        {price_block}
      </div>
      <div class="tc-sig {_signal_class(row.signal)}">{_signal_pretty(row.signal)}</div>
      {intraday_html}
      <div class="scores">
        <div class="score">
          <div class="lab">P(vol) · ≥ {HOT_THRESHOLD*100:.0f}%</div>
          <div class="val" style="color:{val_color(p_vol, HOT_THRESHOLD, is_ultra)};">{p_vol*100:.0f}%</div>
          <div class="bar"><div class="bar-fill {vol_class}" style="width:{min(p_vol*100, 100):.1f}%"></div></div>
        </div>
        <div class="score">
          <div class="lab">P(0DTE PnL) · ≥ {JACKPOT_THRESHOLD*100:.0f}%</div>
          <div class="val" style="color:{val_color(p_pnl, JACKPOT_THRESHOLD, is_ultra)};">{p_pnl*100:.0f}%</div>
          <div class="bar"><div class="bar-fill {pnl_class}" style="width:{min(p_pnl*100, 100):.1f}%"></div></div>
        </div>
        <div class="score">
          <div class="lab">P(weekly) · ≥ {WEEKLY_CONFIRM_THRESHOLD*100:.0f}%</div>
          <div class="val" style="color:{val_color(p_wkly, WEEKLY_CONFIRM_THRESHOLD, is_ultra)};">{p_wkly*100:.0f}%</div>
          <div class="bar"><div class="bar-fill {wkly_class}" style="width:{min(p_wkly*100, 100):.1f}%"></div></div>
        </div>
      </div>
      <div class="tc-meta">
        <span><strong>JP WR:</strong> {wr_jp}</span>
        <span><strong>Ultra WR:</strong> {wr_ultra}</span>
        <span><strong>Ultra avg:</strong> {avg_ultra}</span>
        <span><strong>Ultra/yr:</strong> {row.ultra_trades_per_year:.1f}</span>
      </div>
    </div>
    """


def _account_html(state: AccountState) -> str:
    pnl = state.total_pnl()
    pnl_pct = state.total_pnl_pct() * 100
    pl_class = "good" if pnl > 0 else "bad" if pnl < 0 else "flat"
    pl_str = f"{'+' if pnl >= 0 else ''}${pnl:,.0f}  ({pnl_pct:+.1f}%)"

    nxt = state.next_milestone()
    prev = state.previous_milestone()
    pct = state.progress_to_next() * 100

    if nxt is None:
        progress_html = (
            "<div class='progress-row'>"
            "<div class='ms-lab'><span class='from'>You've passed every milestone.</span></div>"
            "</div>"
        )
    else:
        progress_html = f"""
        <div class="progress-row">
          <div class="ms-lab">
            <span class="from">${prev:,.0f} → next milestone</span>
            <span class="to">${nxt:,.0f}</span>
          </div>
          <div class="progress-bar">
            <div class="fill" style="width:{pct:.1f}%;"></div>
            <div class="fill-pct">{pct:.1f}%</div>
          </div>
        </div>
        """

    n_trades = state.trade_count()
    wr = state.win_rate() * 100 if n_trades > 0 else 0.0
    wr_str = f"{wr:.0f}%" if n_trades > 0 else "—"

    return f"""
    <div class="account">
      <div class="account-head">
        <h2>Account tracker</h2>
        <div class="pl {pl_class}">{pl_str}</div>
      </div>
      {progress_html}
      <div class="account-stats">
        <div class="s"><div class="lab">Start</div><div class="val">${state.starting_equity:,.0f}</div></div>
        <div class="s"><div class="lab">Now</div><div class="val">${state.current_equity:,.0f}</div></div>
        <div class="s"><div class="lab">Trades logged</div><div class="val">{n_trades}</div></div>
        <div class="s"><div class="lab">Win rate</div><div class="val">{wr_str}</div></div>
        <div class="s"><div class="lab">Wins / Losses</div><div class="val">{state.win_count()} / {state.loss_count()}</div></div>
      </div>
      <div class="account-edit">
        Tracker reads from <code>data/account_state.json</code>. Update equity after each trading day, e.g.:<br>
        <code>python -m src.account_state_cli set-equity 562.50</code> &nbsp;·&nbsp;
        <code>python -m src.account_state_cli log-trade SPY GO_ULTRA_JACKPOT 75 38.5</code>
      </div>
    </div>
    """


def _phase_topbar_data(phase: dict, now: datetime) -> tuple[str, str, str, dict]:
    """Return (css_class, label, sub_text, json_payload_for_js)."""
    p = phase["phase"]

    # Compute the next milestone Date in user-local time.
    # We approximate ET → local by trusting the OS clock (datetime.now() is local).
    et_now = phase["as_of"]  # ET-aware datetime if zoneinfo worked
    today_local = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Construct ET key moments as datetimes today, then convert to a JS-readable ISO.
    # Because the OS clock is in local time and the dashboard always renders on the
    # same machine, we can rebuild the target moment using "today in ET" + the hour.

    def et_today_at(hour: int, minute: int) -> datetime:
        """Return today's ET hh:mm as an ISO timestamp the JS Date() can parse."""
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            base = datetime.now(et).replace(hour=hour, minute=minute, second=0, microsecond=0)
            return base
        except Exception:
            return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if p == "PRE_OPEN":
        klass = "pending"
        target = et_today_at(9, 30)
        label = "PRE-MARKET"
        sub = (f"Cash market opens in {phase['next_open_in']//60}h "
               f"{phase['next_open_in']%60}m (9:30 AM ET).")
    elif p == "OPEN_PENDING_DATA":
        klass = "pending"
        target = et_today_at(9, 50)
        label = "OPEN · DATA SETTLING"
        sub = "Yahoo data settles ~9:50 AM ET — signal not yet final."
    elif p == "OPEN_LIVE":
        klass = "live"
        target = et_today_at(16, 0)
        label = "OPEN · SIGNAL LIVE"
        sub = "Decision window open · signal is final for the day."
    elif p == "AFTER_HOURS":
        klass = "closed"
        # next open is tomorrow 9:30 ET (or Monday if Friday)
        from zoneinfo import ZoneInfo
        try:
            et = ZoneInfo("America/New_York")
            base = datetime.now(et).replace(hour=9, minute=30, second=0, microsecond=0)
            base += timedelta(days=1)
            while base.weekday() >= 5:
                base += timedelta(days=1)
            target = base
        except Exception:
            target = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        label = "AFTER-HOURS · FROZEN"
        sub = "Signal frozen until next session."
    else:  # WEEKEND
        klass = "weekend"
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            base = datetime.now(et).replace(hour=9, minute=30, second=0, microsecond=0)
            while base.weekday() != 0:
                base += timedelta(days=1)
            target = base
        except Exception:
            target = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        label = "WEEKEND · CLOSED"
        sub = "Markets reopen Monday 9:30 AM ET."

    payload = {
        "phase": p,
        "target_iso": target.isoformat() if target else None,
    }
    return klass, label, sub, payload


def render(
    rows: list[JackpotRow],
    errors: list[dict],
    mini_cal: list[dict],
    state: AccountState,
    quotes: dict[str, LiveQuote],
    equity: float,
    risk_frac: float,
    refresh_seconds: int,
) -> str:
    now = datetime.now()
    as_of = max((r.as_of for r in rows), default=now)
    phase = market_phase(now)
    phase_class, phase_label, phase_sub, phase_payload = _phase_topbar_data(phase, now)

    rank = {"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2, "SKIP": 1}
    rows_sorted = sorted(rows, key=lambda r: (rank.get(r.signal, 0), r.p_pnl), reverse=True)

    err_html = ""
    if errors:
        err_html = "<div class='ticker-card skip'><strong>Errors:</strong><ul>" + "".join(
            f"<li>{e['ticker']}: {e['error']}</li>" for e in errors
        ) + "</ul></div>"

    equity_pill = (
        f"<div class='equity-pill'>Equity <strong>${state.current_equity:,.0f}</strong>"
        f" · next milestone <strong>${state.next_milestone():,.0f}</strong></div>"
        if state.next_milestone() else
        f"<div class='equity-pill'>Equity <strong>${state.current_equity:,.0f}</strong></div>"
    )

    ticker_order = [r.ticker for r in rows_sorted]
    return HTML_TEMPLATE.format(
        as_of=as_of.strftime("%Y-%m-%d") if isinstance(as_of, pd.Timestamp) else now.strftime("%Y-%m-%d"),
        as_of_full=now.strftime("%A, %B %d, %Y %H:%M"),
        equity=equity,
        risk_pct=f"{risk_frac * 100:.0f}%",
        refresh_seconds=refresh_seconds,
        phase_class=phase_class,
        phase_label=phase_label,
        phase_sub=phase_sub,
        phase_json=json.dumps(phase_payload),
        equity_pill=equity_pill,
        liveprice_strip=_liveprice_strip_html(quotes, ticker_order),
        hero_html=_hero_html(rows, phase, equity),
        mini_cal_html=_mini_calendar_html(mini_cal),
        tickets_html=_trade_tickets_html(rows_sorted, equity, risk_frac),
        tickers_html=err_html + "\n".join(
            _ticker_card_html(r, quotes.get(r.ticker.upper())) for r in rows_sorted
        ),
        account_html=_account_html(state),
        hot_thr=f"{HOT_THRESHOLD * 100:.0f}%",
        jp_thr=f"{JACKPOT_THRESHOLD * 100:.0f}%",
        wkly_thr=f"{WEEKLY_CONFIRM_THRESHOLD * 100:.0f}%",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_once(args, out_path: Path) -> tuple[list[JackpotRow], list[dict]]:
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    rows, errors = scan_jackpot_universe(
        tickers=tickers,
        refresh_data=not args.no_refresh_data,
        retrain=args.retrain,
        refresh_stats=args.refresh_stats,
    )

    mini_cal = build_mini_calendar(tickers, n_before=2, n_after=2, refresh_data=False)
    state = load_state()
    snapshot_equity(state, args.equity if args.set_equity else state.current_equity)

    quotes: dict[str, LiveQuote] = {}
    if not args.no_live:
        try:
            quotes = get_live_quotes(tickers, prefer=args.live_source)
            logger.info("Live quotes: %d/%d (%s)", len(quotes), len(tickers), args.live_source)
        except Exception as e:
            logger.warning("Live quotes failed: %s", e)

    html = render(
        rows=rows, errors=errors, mini_cal=mini_cal, state=state, quotes=quotes,
        equity=state.current_equity, risk_frac=args.risk_frac,
        refresh_seconds=args.refresh_seconds,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "as_of": datetime.now().isoformat(),
        "equity": state.current_equity,
        "risk_frac": args.risk_frac,
        "rows": [r.to_dict() for r in rows],
        "errors": errors,
    }, indent=2, default=float), encoding="utf-8")

    if args.set_equity:
        save_state(state)
    return rows, errors


def _format_console(rows: list[JackpotRow]) -> str:
    """Tightly-aligned console table — no f-string newlines for clean tail."""
    headers = ["Ticker", "Signal", "P(vol)", "P(0DTE)", "P(week)", "Close", "JP-WR", "Ultra-WR"]
    widths = [7, 19, 7, 8, 8, 9, 7, 9]
    lines = ["  ".join(h.rjust(w) if i > 1 else h.ljust(w) for i, (h, w) in enumerate(zip(headers, widths)))]
    lines.append("─" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        row = [
            r.ticker.ljust(7),
            r.signal.ljust(19),
            f"{r.p_vol*100:6.1f}%".rjust(7),
            f"{r.p_pnl*100:6.1f}%".rjust(8),
            f"{r.p_weekly*100:6.1f}%".rjust(8),
            f"${r.last_close:>7.2f}".rjust(9),
            (f"{r.win_rate_history*100:5.1f}%" if not pd.isna(r.win_rate_history) else "    —").rjust(7),
            (f"{r.ultra_win_rate_history*100:5.1f}%" if not pd.isna(r.ultra_win_rate_history) else "    —").rjust(9),
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tickers", default=",".join(DEFAULT_JACKPOT_UNIVERSE))
    parser.add_argument("--equity", type=float, default=None,
                        help="Override equity in account_state.json (writes the file).")
    parser.add_argument("--risk-frac", type=float, default=0.15)
    parser.add_argument("--refresh-seconds", type=int, default=300)
    parser.add_argument("--no-refresh-data", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--refresh-stats", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=300)
    parser.add_argument("--no-live", action="store_true",
                        help="Skip live (delayed) quote fetch — useful offline or to speed up renders.")
    parser.add_argument("--live-source", choices=["yahoo", "stooq"], default="yahoo",
                        help="Preferred live-quote source. Falls back to the other if it fails.")
    args = parser.parse_args(argv)

    args.set_equity = args.equity is not None
    if args.equity is None:
        st = load_state()
        args.equity = st.current_equity

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    out_path = Path(args.out_dir) / "JACKPOT_dashboard.html"
    rows, errors = _run_once(args, out_path)

    n_ultra = sum(1 for r in rows if r.signal == "GO_ULTRA_JACKPOT")
    n_jp = sum(1 for r in rows if r.signal == "GO_JACKPOT")
    n_hot = sum(1 for r in rows if r.signal == "GO_HOT")
    logger.info("Wrote dashboard -> %s", out_path)
    logger.info("ULTRA: %d  JACKPOT: %d  HOT: %d  SKIP: %d",
                n_ultra, n_jp, n_hot, len(rows) - n_ultra - n_jp - n_hot)
    print()
    print(_format_console(rows))
    print()

    if not args.no_open:
        webbrowser.open(out_path.as_uri())

    if args.watch:
        logger.info("Watch mode: refreshing every %d s. Ctrl+C to stop.", args.watch_interval)
        try:
            while True:
                time.sleep(args.watch_interval)
                rows, _ = _run_once(args, out_path)
                logger.info("Refreshed %s  (%d ULTRA / %d JP)",
                            datetime.now().strftime("%H:%M:%S"),
                            sum(1 for r in rows if r.signal == "GO_ULTRA_JACKPOT"),
                            sum(1 for r in rows if r.signal == "GO_JACKPOT"))
        except KeyboardInterrupt:
            logger.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
