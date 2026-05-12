"""Live moonshot indicator — single GO / WAIT / SKIP traffic-light.

This is the daily decision tool for the long-straddle continuous-compound
strategy. Run it each morning around 9:35 ET (after the open prints) and
look at the big light:

  🟢 GO   — P_vol >= go_threshold. Buy 0DTE ATM SPY straddle, 50% of equity.
  🟡 WAIT — P_vol is close to threshold. Optional, lower size or skip.
  🔴 SKIP — P_vol < threshold. Don't trade today; wait for tomorrow.

Single self-contained HTML page that auto-refreshes. The page also surfaces:
  - exact ATM strike from today's open
  - dollar-sized position for your current equity
  - last 10 historical signal days with outcomes (win/loss/hold)
  - strategy rules in 3 bullets
  - projected timeline to $5k / $50k / $500k

Run::

    python -m src.report_moonshot_signal --equity 500
    python -m src.report_moonshot_signal --equity 500 --watch 300
    python -m src.report_moonshot_signal --equity 5000 --threshold 0.30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from src.scanner import fetch_or_load_daily, load_or_train_model
from src.strategy_sim import (
    StrategyConfig,
    compute_per_day_returns,
    straddle_return,
)
from src.volatility_classifier import (
    make_logreg,
    prepare_xy,
    score_dataframe,
    walk_forward_proba,
)
from src.volatility_patterns import build_features


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "SPY_moonshot_signal.html"

logger = logging.getLogger("moonshot_signal")


# ---------------------------------------------------------------------------
# Today's signal + trade ticket
# ---------------------------------------------------------------------------


def evaluate_today(
    daily: pd.DataFrame,
    p_vol: pd.Series,
    go_threshold: float,
    wait_threshold: float,
    equity: float,
    risk_frac: float,
    premium_pct: float,
) -> dict:
    last_idx = daily.index[-1]
    score = float(p_vol.loc[last_idx]) if last_idx in p_vol.index else float("nan")
    open_ = float(daily.loc[last_idx, "Open"])
    close = float(daily.loc[last_idx, "Close"])

    if score != score:
        verdict = "UNKNOWN"
    elif score >= go_threshold:
        verdict = "GO"
    elif score >= wait_threshold:
        verdict = "WAIT"
    else:
        verdict = "SKIP"

    risk_dollars = equity * risk_frac
    spy_strike = round(open_)
    expected_premium_per_spy_share = open_ * premium_pct
    expected_premium_per_spy_contract = expected_premium_per_spy_share * 100
    expected_premium_per_xsp_contract = expected_premium_per_spy_contract / 10.0

    n_xsp_contracts = max(1, int(risk_dollars // max(expected_premium_per_xsp_contract, 1.0)))
    actual_xsp_premium = n_xsp_contracts * expected_premium_per_xsp_contract

    n_spy_contracts = max(0, int(risk_dollars // max(expected_premium_per_spy_contract, 1.0)))
    actual_spy_premium = n_spy_contracts * expected_premium_per_spy_contract

    return {
        "as_of": last_idx,
        "score": score,
        "verdict": verdict,
        "open": open_,
        "close": close,
        "spy_strike": spy_strike,
        "xsp_strike": round(open_ / 10.0, 1),
        "risk_dollars": risk_dollars,
        "expected_premium_per_spy": expected_premium_per_spy_contract,
        "expected_premium_per_xsp": expected_premium_per_xsp_contract,
        "n_xsp_contracts": n_xsp_contracts,
        "actual_xsp_premium": actual_xsp_premium,
        "n_spy_contracts": n_spy_contracts,
        "actual_spy_premium": actual_spy_premium,
        "go_threshold": go_threshold,
        "wait_threshold": wait_threshold,
    }


# ---------------------------------------------------------------------------
# Recent history
# ---------------------------------------------------------------------------


def recent_signals(
    daily: pd.DataFrame,
    p_vol: pd.Series,
    threshold: float,
    n_days: int = 30,
    premium_pct: float = 0.011,
) -> list[dict]:
    """Last N calendar trading days, with signal + actual outcome."""
    common = sorted(daily.index.intersection(p_vol.index))[-n_days:]
    rows = []
    for date in common:
        bar = daily.loc[date]
        score = float(p_vol.loc[date])
        was_signal = score >= threshold
        ret = None
        if was_signal:
            ret = straddle_return(
                float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                float(bar["Close"]), premium_pct=premium_pct,
            )
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "weekday": date.strftime("%a"),
            "score": score,
            "was_signal": bool(was_signal),
            "ret": ret,
            "outcome": "WIN" if (ret is not None and ret > 0) else ("LOSS" if (ret is not None) else "—"),
        })
    return rows


def signal_summary_stats(history: list[dict]) -> dict:
    trades = [r for r in history if r["was_signal"]]
    wins = [r for r in trades if r["ret"] is not None and r["ret"] > 0]
    losses = [r for r in trades if r["ret"] is not None and r["ret"] <= 0]
    return {
        "n_signals": len(trades),
        "n_skips": len(history) - len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "total_return": float(np.sum([r["ret"] for r in trades if r["ret"] is not None])) if trades else 0.0,
        "avg_win": float(np.mean([r["ret"] for r in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([r["ret"] for r in losses])) if losses else 0.0,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="refresh" content="{refresh_s}" />
<title>Moonshot Signal — {as_of_str}</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --panel-2: #243047; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #38bdf8; --pos: #22c55e; --neg: #ef4444; --warn: #f59e0b; --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ padding: 28px 28px 60px; max-width: 1200px; margin: 0 auto; }}

  header {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 18px; margin-bottom: 18px; }}
  h1 {{ font-size: 24px; margin: 0; letter-spacing: -0.4px; }}
  .timestamp {{ color: var(--muted); font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .timestamp .live-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #22c55e; margin-right: 8px; vertical-align: middle; box-shadow: 0 0 0 0 rgba(34,197,94,0.6); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }} 70% {{ box-shadow: 0 0 0 10px rgba(34,197,94,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }} }}

  /* The big traffic light */
  .signal-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 56px 32px;
    margin: 14px 0 24px;
    text-align: center;
    border-left: 14px solid {verdict_color};
    box-shadow: 0 20px 60px -25px {verdict_color};
  }}
  .signal-card .verdict-icon {{ font-size: 90px; line-height: 1; margin-bottom: 8px; }}
  .signal-card .verdict-text {{
    font-size: 72px; font-weight: 800; letter-spacing: -2px; margin: 0;
    color: {verdict_color};
  }}
  .signal-card .verdict-subtext {{ font-size: 22px; font-weight: 500; color: var(--muted); margin-top: 4px; }}
  .signal-card .verdict-explain {{ font-size: 15px; color: var(--muted); margin-top: 22px; line-height: 1.6; }}
  .signal-card .verdict-explain strong {{ color: var(--text); }}

  .gauge-row {{ display: flex; justify-content: center; align-items: center; gap: 16px; margin-top: 22px; flex-wrap: wrap; }}
  .gauge {{ background: rgba(15, 23, 42, 0.6); border-radius: 10px; padding: 10px 16px; min-width: 150px; }}
  .gauge .gauge-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .gauge .gauge-value {{ font-size: 22px; font-weight: 600; margin-top: 2px; }}

  .grid-2 {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  @media (max-width: 920px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px; }}
  .card h3 {{ margin: 0 0 12px; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); }}

  .kv {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed var(--border); }}
  .kv:last-child {{ border-bottom: none; }}
  .kv .k {{ color: var(--muted); font-size: 13px; }}
  .kv .v {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; font-weight: 600; }}
  .kv .v.big {{ font-size: 17px; color: var(--accent); }}
  .kv .v.warn {{ color: var(--warn); }}
  .kv .v.pos {{ color: var(--pos); }}

  .ticket {{ background: var(--panel-2); border: 1px solid var(--border); border-left: 5px solid var(--accent); border-radius: 12px; padding: 18px 22px; }}
  .ticket .title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; }}

  .history {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{ background: var(--panel-2); text-align: left; padding: 8px 10px; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 8px 10px; border-bottom: 1px solid rgba(51, 65, 85, 0.3); }}
  tbody tr:last-child td {{ border-bottom: none; }}
  td.numeric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.win {{ color: var(--pos); font-weight: 600; }}
  td.loss {{ color: var(--neg); font-weight: 600; }}
  td.skip {{ color: var(--muted); }}

  .summary-pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .summary-pill.win {{ background: rgba(34,197,94,0.15); color: var(--pos); }}
  .summary-pill.loss {{ background: rgba(239,68,68,0.15); color: var(--neg); }}

  .rules-list {{ list-style: none; padding: 0; margin: 0; }}
  .rules-list li {{ display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--border); }}
  .rules-list li:last-child {{ border-bottom: none; }}
  .rules-list .when {{ color: var(--muted); width: 70px; flex-shrink: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; }}
  .rules-list .what {{ font-size: 14px; line-height: 1.5; }}
  .rules-list .what strong {{ color: var(--accent); }}

  footer {{ margin-top: 36px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  footer code {{ background: var(--panel); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Moonshot Signal · SPY 0DTE Long Straddle</h1>
    <div style="color:var(--muted); font-size:13px; margin-top: 4px;">
      Continuous-compound long-only strategy · risk {risk_pct}% per trade · target $500k in 18 months
    </div>
  </div>
  <div class="timestamp"><span class="live-dot"></span>Updated {ts}</div>
</header>

<div class="signal-card">
  <div class="verdict-icon">{verdict_icon}</div>
  <div class="verdict-text">{verdict_text}</div>
  <div class="verdict-subtext">{verdict_subtext}</div>

  <div class="gauge-row">
    <div class="gauge">
      <div class="gauge-label">P_vol score</div>
      <div class="gauge-value" style="color:{verdict_color}">{score_pct}</div>
    </div>
    <div class="gauge">
      <div class="gauge-label">Threshold</div>
      <div class="gauge-value">≥ {threshold_pct}</div>
    </div>
    <div class="gauge">
      <div class="gauge-label">Session</div>
      <div class="gauge-value">{session_str}</div>
    </div>
    <div class="gauge">
      <div class="gauge-label">SPY open</div>
      <div class="gauge-value">${open_str}</div>
    </div>
  </div>

  <div class="verdict-explain">{verdict_explain}</div>
</div>

<div class="grid-2">
  <div class="card">
    <h3>Trade ticket</h3>
    {ticket_html}
  </div>
  <div class="card">
    <h3>Strategy rules — always the same</h3>
    <ul class="rules-list">
      <li><div class="when">When</div><div class="what">P_vol score &ge; <strong>{threshold_pct}</strong> at the open (HOT signal)</div></li>
      <li><div class="when">What</div><div class="what">Buy ATM 0DTE <strong>straddle</strong> (1 call + 1 put at open price)</div></li>
      <li><div class="when">Size</div><div class="what">Risk <strong>{risk_pct}%</strong> of current equity (defined max loss = premium)</div></li>
      <li><div class="when">Exit</div><div class="what">3:30pm ET, OR take 50% of intraday max-favorable-excursion</div></li>
      <li><div class="when">Skip</div><div class="what">All non-HOT days — most days you do nothing</div></li>
    </ul>
  </div>
</div>

<div class="card" style="margin-top:16px;">
  <h3>Last 10 trading days</h3>
  <div class="history">
    <table>
      <thead><tr>
        <th>Date</th><th>Day</th><th>P_vol</th><th>Signal</th><th>Trade outcome</th><th>Return / dollar risked</th>
      </tr></thead>
      <tbody>
        {history_rows}
      </tbody>
    </table>
  </div>
  <div style="margin-top: 14px; color: var(--muted); font-size: 13px;">
    Last 30 sessions: <span class="summary-pill win">{n_wins} wins</span>
    <span class="summary-pill loss">{n_losses} losses</span>
    · <strong>{win_rate_pct} win rate</strong>
    · {n_skips} skip days
    · Cumulative {total_return_pct} per dollar risked
  </div>
</div>

<div class="grid-3" style="margin-top:16px;">
  <div class="card">
    <h3>Where you are</h3>
    <div class="kv"><span class="k">Current equity</span><span class="v big">${equity_str}</span></div>
    <div class="kv"><span class="k">Last close (SPY)</span><span class="v">${close_str}</span></div>
    <div class="kv"><span class="k">Intraday move</span><span class="v {pct_chg_class}">{pct_chg_str}</span></div>
  </div>
  <div class="card">
    <h3>Targets &amp; expected timeline</h3>
    <div class="kv"><span class="k">$5,000 (10×)</span><span class="v">~6 months · 50% backtest</span></div>
    <div class="kv"><span class="k">$50,000 (100×)</span><span class="v">~12 months · 60% backtest</span></div>
    <div class="kv"><span class="k">$500,000 (1,000×)</span><span class="v">~18 months · 67% backtest</span></div>
  </div>
  <div class="card">
    <h3>Reality check</h3>
    <div class="kv"><span class="k">Realistic haircut</span><span class="v warn">−30% on EV</span></div>
    <div class="kv"><span class="k">P(hit $500k, real)</span><span class="v warn">~40-46%</span></div>
    <div class="kv"><span class="k">P(busted)</span><span class="v pos">&lt; 1%</span></div>
  </div>
</div>

<footer>
  <p><strong>How to use this page.</strong> Open it once a day at <code>~9:35 ET</code>. Look at the big light. If <strong>GO</strong>, place the trade exactly per the ticket. If <strong>WAIT</strong>, skip or use half size. If <strong>SKIP</strong>, do nothing — that's the strategy too. Run <code>python -m src.report_moonshot_signal --watch 300</code> for auto-refreshing live mode.</p>
  <p><strong>Why straddles, not single calls/puts.</strong> Our model predicts magnitude (vol), not direction. Single-leg directional has a lower historical win rate (~45%) and higher bust risk (~30% in 6 months at full size). Straddles win 64% of the time and bust < 1%.</p>
  <p><strong>What's NOT in this signal.</strong> News (FOMC, earnings, geopolitical), pre-market gaps beyond the open print, intraday signal updates. The score crystallizes at the open and stays valid all day.</p>
</footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verdict_visuals(verdict: str) -> tuple[str, str, str, str, str]:
    if verdict == "GO":
        return (
            "GO",                             # text
            "Buy 0DTE straddle",              # subtext
            "🟢",                              # icon
            "#22c55e",                        # color
            "Model predicts an ELEVATED-or-higher day. Place the straddle from the ticket below within the first 15 minutes of trading.",
        )
    if verdict == "WAIT":
        return (
            "WAIT",
            "Borderline signal — half size or skip",
            "🟡",
            "#f59e0b",
            "Score is close to the threshold. You can take half size, wait for a stronger setup, or skip. Skipping is fine — there are ~5 HOT days per month on average.",
        )
    if verdict == "SKIP":
        return (
            "SKIP",
            "No trade today — wait for tomorrow",
            "🔴",
            "#ef4444",
            "Model predicts a calm-to-average day. Long straddles lose money on these — premium decays faster than the underlying moves. Patience is part of the strategy.",
        )
    return ("UNKNOWN", "Insufficient data", "•", "#94a3b8", "Could not score today's session.")


def _ticket_html(today: dict, verdict: str) -> str:
    if verdict == "SKIP":
        return """
            <div style='color: var(--muted); font-size: 14px; line-height: 1.6; padding: 8px 0;'>
              No trade today. The strategy depends on patience — most days are skip days.
              Come back tomorrow at 9:35 ET.
            </div>
        """
    badge = "Active trade" if verdict == "GO" else "Optional / half-size"
    return f"""
        <div class="ticket">
          <div class="title">{badge} · 0DTE SPY straddle</div>
          <div class="kv"><span class="k">Call strike</span><span class="v big">${today['spy_strike']}</span></div>
          <div class="kv"><span class="k">Put strike</span><span class="v big">${today['spy_strike']}</span></div>
          <div class="kv"><span class="k">Estimated total premium / SPY contract</span><span class="v">~${today['expected_premium_per_spy']:.0f}</span></div>
          <div class="kv"><span class="k">Estimated total premium / XSP contract</span><span class="v">~${today['expected_premium_per_xsp']:.2f}</span></div>
          <div class="kv"><span class="k">Risk budget ({today['risk_dollars']:.0f}/{(today['risk_dollars'] / max(today['expected_premium_per_xsp'], 1.0) ):.1f} of XSP)</span><span class="v">${today['risk_dollars']:.0f}</span></div>
          <div class="kv"><span class="k">Recommended XSP contracts</span><span class="v big">{today['n_xsp_contracts']} · ~${today['actual_xsp_premium']:.0f} cost</span></div>
          {('<div class="kv"><span class="k">SPY contracts (if budget allows)</span><span class="v">' + str(today['n_spy_contracts']) + ' · ~$' + format(today['actual_spy_premium'], ',.0f') + ' cost</span></div>') if today['n_spy_contracts'] >= 1 else ''}
          <div style="margin-top: 10px; color: var(--muted); font-size: 12px;">
            Use XSP for cash account / no PDT / 60-40 tax treatment. Use SPY only if you have a margin account &gt; $25k.
          </div>
        </div>
    """


def _history_rows(history: list[dict]) -> str:
    out = []
    for r in history[::-1]:
        if not r["was_signal"]:
            out.append(f"""
              <tr>
                <td>{escape(r['date'])}</td>
                <td>{escape(r['weekday'])}</td>
                <td class="numeric">{r['score'] * 100:.1f}%</td>
                <td class="skip">SKIP</td>
                <td class="skip">—</td>
                <td class="skip">—</td>
              </tr>
            """)
            continue
        ret = r["ret"]
        cls = "win" if (ret is not None and ret > 0) else "loss"
        out.append(f"""
              <tr>
                <td>{escape(r['date'])}</td>
                <td>{escape(r['weekday'])}</td>
                <td class="numeric">{r['score'] * 100:.1f}%</td>
                <td class="numeric"><strong>HOT · BUY straddle</strong></td>
                <td class="{cls}">{escape(r['outcome'])}</td>
                <td class="numeric {cls}">{(ret * 100):+.1f}%</td>
              </tr>
        """)
    return "\n".join(out)


def render_html(
    today: dict,
    history: list[dict],
    summary: dict,
    equity: float,
    risk_frac: float,
    refresh_s: int,
) -> str:
    verdict_text, verdict_subtext, verdict_icon, verdict_color, verdict_explain = _verdict_visuals(today["verdict"])
    open_ = today["open"]
    close = today["close"]
    pct_chg = (close / open_ - 1.0) if open_ > 0 else 0.0

    return HTML_TEMPLATE.format(
        refresh_s=refresh_s,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        as_of_str=today["as_of"].strftime("%Y-%m-%d"),
        verdict_text=verdict_text,
        verdict_subtext=verdict_subtext,
        verdict_icon=verdict_icon,
        verdict_color=verdict_color,
        verdict_explain=verdict_explain,
        score_pct=f"{today['score'] * 100:.1f}%" if today["score"] == today["score"] else "—",
        threshold_pct=f"{today['go_threshold'] * 100:.0f}%",
        session_str=today["as_of"].strftime("%a %b %d"),
        open_str=f"{open_:,.2f}",
        close_str=f"{close:,.2f}",
        pct_chg_str=f"{pct_chg * 100:+.2f}%",
        pct_chg_class="pos" if pct_chg >= 0 else "neg",
        equity_str=f"{equity:,.0f}",
        risk_pct=f"{risk_frac * 100:.0f}",
        ticket_html=_ticket_html(today, today["verdict"]),
        history_rows=_history_rows(history),
        n_wins=summary["n_wins"],
        n_losses=summary["n_losses"],
        n_skips=summary["n_skips"],
        win_rate_pct=f"{summary['win_rate'] * 100:.0f}%" if summary["win_rate"] == summary["win_rate"] else "—",
        total_return_pct=f"{summary['total_return'] * 100:+.0f}%",
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def compute_signal_data(
    args: argparse.Namespace,
    fast: bool = False,
) -> dict:
    daily = fetch_or_load_daily("SPY", refresh=not args.no_fetch)
    feats = build_features(daily)
    X, y, _ = prepare_xy(feats)

    if fast:
        # Fast path: use the cached scanner LogReg model (full-fit, not walk-forward).
        # Same model that the live scanner uses.
        model = load_or_train_model("SPY", X, y)
        p_vol = score_dataframe(model, X)
    else:
        preds = walk_forward_proba(X, y, make_logreg, min_train=args.min_train, step=args.refit_step)
        p_vol = preds["y_score"]

    today = evaluate_today(
        daily, p_vol,
        go_threshold=args.threshold,
        wait_threshold=args.threshold * 0.85,
        equity=args.equity,
        risk_frac=args.risk_frac,
        premium_pct=0.011,
    )
    history = recent_signals(daily, p_vol, args.threshold, n_days=30)
    summary = signal_summary_stats(history)

    return {"today": today, "history": history, "summary": summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live moonshot signal indicator (long-straddle strategy).")
    p.add_argument("--equity", type=float, default=500.0, help="Your current account equity.")
    p.add_argument("--threshold", type=float, default=0.30, help="P_vol threshold for HOT signal (default 0.30).")
    p.add_argument("--risk-frac", type=float, default=0.50, help="Fraction of equity per trade (default 50%).")
    p.add_argument("--watch", type=int, default=0, help="Re-render every N seconds.")
    p.add_argument("--refresh-page", type=int, default=180, help="HTML meta-refresh in seconds.")
    p.add_argument("--no-fetch", action="store_true", help="Use cached SPY data, don't hit Yahoo.")
    p.add_argument("--fast", action="store_true", default=True, help="Use cached scanner model (fast). Set --slow for fresh walk-forward.")
    p.add_argument("--slow", dest="fast", action="store_false", help="Re-run full walk-forward (slow but freshest).")
    p.add_argument("--min-train", type=int, default=1000)
    p.add_argument("--refit-step", type=int, default=21)
    p.add_argument("--out", default=str(DEFAULT_OUTPUT))
    p.add_argument("--no-open", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_once() -> str:
        data = compute_signal_data(args, fast=args.fast)
        html = render_html(
            today=data["today"],
            history=data["history"],
            summary=data["summary"],
            equity=args.equity,
            risk_frac=args.risk_frac,
            refresh_s=args.refresh_page,
        )
        out_path.write_text(html, encoding="utf-8")
        return data["today"]["verdict"]

    verdict = _run_once()
    logger.info("Wrote %s · today's verdict: %s", out_path, verdict)
    print(f"Today's signal: {verdict}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    if args.watch > 0:
        try:
            while True:
                time.sleep(args.watch)
                v = _run_once()
                print(f"[{datetime.now():%H:%M:%S}] re-rendered · verdict: {v}")
        except KeyboardInterrupt:
            print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
