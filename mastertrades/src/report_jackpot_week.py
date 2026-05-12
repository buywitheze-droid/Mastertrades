"""Weekly calendar view of jackpot alerts (Mon–Fri, this week + last week).

For each weekday in the current week and the previous week, this report
shows a single card with:

  - The date (e.g. "Mon May 11")
  - The strongest signal across all 4 tickers (★★★ ULTRA / ★ JACKPOT / HOT / SKIP)
  - A row per ticker showing that ticker's signal as a small chip
  - For past days: realized 0DTE straddle outcome (green if winning) for any
    ticker that fired a JACKPOT or ULTRA signal
  - Future days: greyed out / "pending"
  - Today: highlighted with a glowing border

Usage::

    python -m src.report_jackpot_week
    python -m src.report_jackpot_week --equity 5000 --no-open

The page auto-refreshes every 5 minutes; useful as a wall-monitor view.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.jackpot_scanner import (
    DEFAULT_JACKPOT_UNIVERSE,
    HOT_THRESHOLD,
    JACKPOT_THRESHOLD,
    WEEKLY_CONFIRM_THRESHOLD,
    market_phase,
    score_jackpot_recent,
)


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
logger = logging.getLogger("report_jackpot_week")

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WEEKDAY_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


SIGNAL_RANK = {"GO_ULTRA_JACKPOT": 4, "GO_JACKPOT": 3, "GO_HOT": 2, "SKIP": 1, None: 0}


def _monday_of_week(d: datetime) -> datetime:
    """Return the Monday of the calendar week containing d."""
    return (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def collect_week_data(
    tickers: list[str],
    n_days: int = 15,
    refresh_data: bool = True,
    retrain: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame of last n_days scoring} for each ticker."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = score_jackpot_recent(
                t, n_days=n_days, refresh_data=refresh_data, retrain=retrain,
            )
        except Exception as e:
            logger.warning("Failed %s: %s", t, e)
    return out


def build_calendar(
    data: dict[str, pd.DataFrame],
    week_start: datetime,
) -> list[dict]:
    """Build a 5-day calendar (Mon-Fri) for the week starting on ``week_start``.

    Each entry: {
        "date": pd.Timestamp,
        "weekday": "Mon"...,
        "is_today": bool,
        "is_future": bool,
        "is_past": bool,
        "strongest": "GO_ULTRA_JACKPOT" / ... / None,
        "tickers": [
            {"ticker": "SPY", "signal": "...", "p_vol":..., "p_pnl":...,
             "p_weekly":..., "zdte_realized":..., "had_data": bool},
            ...
        ]
    }
    """
    today = pd.Timestamp(datetime.now().date())
    cal: list[dict] = []
    for i in range(5):
        date = pd.Timestamp((week_start + timedelta(days=i)).date())
        is_today = date == today
        is_future = date > today
        is_past = date < today

        ticker_rows = []
        strongest_rank = 0
        strongest_signal = None

        for t, df in data.items():
            if date in df.index:
                row = df.loc[date]
                signal = row["signal"]
                ticker_rows.append({
                    "ticker": t,
                    "signal": signal,
                    "p_vol": float(row["p_vol"]),
                    "p_pnl": float(row["p_pnl"]),
                    "p_weekly": float(row["p_weekly"]) if not pd.isna(row["p_weekly"]) else float("nan"),
                    "zdte_realized": float(row["zdte_realized"]) if "zdte_realized" in row else float("nan"),
                    "had_data": True,
                    "close": float(row["close"]),
                })
                rk = SIGNAL_RANK.get(signal, 0)
                if rk > strongest_rank:
                    strongest_rank = rk
                    strongest_signal = signal
            else:
                ticker_rows.append({
                    "ticker": t, "signal": None, "p_vol": float("nan"),
                    "p_pnl": float("nan"), "p_weekly": float("nan"),
                    "zdte_realized": float("nan"), "had_data": False,
                    "close": float("nan"),
                })

        cal.append({
            "date": date,
            "weekday": WEEKDAYS[i],
            "weekday_full": WEEKDAY_FULL[i],
            "is_today": is_today,
            "is_future": is_future,
            "is_past": is_past,
            "strongest": strongest_signal,
            "tickers": ticker_rows,
        })
    return cal


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _signal_label(signal: str | None) -> str:
    return {
        "GO_ULTRA_JACKPOT": "★★★ ULTRA",
        "GO_JACKPOT": "★ JACKPOT",
        "GO_HOT": "HOT",
        "SKIP": "SKIP",
        None: "—",
    }.get(signal, str(signal))


def _signal_class(signal: str | None) -> str:
    return (signal or "EMPTY").lower().replace("_", "-")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>JACKPOT weekly calendar — {generated}</title>
<meta http-equiv="refresh" content="300"/>
<style>
  :root {{
    --bg:#0c1117; --panel:#161b22; --line:#30363d; --text:#e6edf3;
    --muted:#8b949e; --good:#3fb950; --bad:#f85149; --warn:#d29922;
    --gold:#ffd633; --blue:#58a6ff;
    --gradient: linear-gradient(135deg, #58a6ff, #d2a8ff);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, "Segoe UI", Inter, Arial, sans-serif;
    margin: 0; padding: 24px;
  }}
  h1 {{ font-size: 28px; margin: 0 0 4px;
        background: var(--gradient); -webkit-background-clip: text;
        background-clip: text; color: transparent; }}
  h2 {{ font-size: 16px; margin: 28px 0 12px; color: #c9d1d9;
        text-transform: uppercase; letter-spacing: 0.06em; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}

  .week-grid {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
  }}
  @media (max-width: 1100px) {{ .week-grid {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 700px) {{ .week-grid {{ grid-template-columns: 1fr; }} }}

  .day-card {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px;
    display: flex; flex-direction: column;
    min-height: 280px;
    position: relative;
  }}
  .day-card.today {{
    border: 2px solid var(--blue);
    box-shadow: 0 0 22px rgba(88, 166, 255, 0.28);
  }}
  .day-card.future {{ opacity: 0.55; }}

  /* The "strongest signal" tint applied at the card level */
  .day-card.go-ultra-jackpot {{
    background: linear-gradient(135deg, #1a2f0c 0%, #2d5510 60%, #3d7515 100%);
    border: 2px solid var(--gold);
    box-shadow: 0 0 22px rgba(255, 214, 51, 0.25);
  }}
  .day-card.go-jackpot {{
    background: linear-gradient(135deg, #0f2818 0%, #1c4a30 100%);
    border: 2px solid var(--good);
  }}
  .day-card.go-hot {{
    background: linear-gradient(135deg, #2d2410 0%, #463812 100%);
    border: 2px solid var(--warn);
  }}

  .day-head {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .day-head .wd {{ font-size: 14px; color: var(--muted); text-transform: uppercase;
                    letter-spacing: 0.05em; font-weight: 700; }}
  .day-head .dt {{ font-size: 22px; color: #fff; font-weight: 700; line-height: 1; }}

  .day-banner {{
    margin: 10px 0 12px;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px; font-weight: 800; letter-spacing: 0.05em;
    text-transform: uppercase;
    text-align: center;
  }}
  .day-banner.go-ultra-jackpot {{
    background: linear-gradient(90deg, var(--gold), #ff8c33, var(--gold));
    background-size: 200% 100%; color: #0c1117;
    animation: shimmer 3s linear infinite;
  }}
  @keyframes shimmer {{
    0% {{ background-position: 0% 50%; }} 100% {{ background-position: 200% 50%; }}
  }}
  .day-banner.go-jackpot {{ background: var(--good); color: #0c1117; }}
  .day-banner.go-hot {{ background: var(--warn); color: #0c1117; }}
  .day-banner.skip {{ background: rgba(139,148,158,0.18); color: var(--muted); }}
  .day-banner.empty {{ background: rgba(139,148,158,0.10); color: var(--muted); }}

  .tickers {{ display: flex; flex-direction: column; gap: 4px; flex: 1; }}
  .ticker-chip {{
    display: grid;
    grid-template-columns: 44px 80px auto;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    border-radius: 6px;
    background: rgba(13, 17, 23, 0.55);
    font-size: 12px;
  }}
  .ticker-chip .tkr {{ font-weight: 700; color: #c9d1d9; font-size: 13px; }}
  .ticker-chip .sig {{ font-size: 10px; font-weight: 800;
                        text-align: center; padding: 2px 4px; border-radius: 3px;
                        letter-spacing: 0.04em; }}
  .sig.go-ultra-jackpot {{ background: var(--gold); color: #0c1117; }}
  .sig.go-jackpot {{ background: var(--good); color: #0c1117; }}
  .sig.go-hot {{ background: var(--warn); color: #0c1117; }}
  .sig.skip {{ background: rgba(139,148,158,0.18); color: var(--muted); }}
  .sig.empty {{ background: rgba(139,148,158,0.10); color: var(--muted); }}
  .ticker-chip .scores {{
    color: var(--muted); font-size: 11px;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  .ticker-chip .scores .pv {{ color: var(--warn); }}
  .ticker-chip .scores .pp {{ color: var(--good); }}
  .ticker-chip .scores .pw {{ color: var(--blue); }}

  /* Outcome badges for past days */
  .outcome {{ font-size: 11px; font-weight: 700; margin-left: 4px;
              padding: 1px 5px; border-radius: 3px; }}
  .outcome.win {{ background: rgba(63, 185, 80, 0.20); color: var(--good); }}
  .outcome.lose {{ background: rgba(248, 81, 73, 0.20); color: var(--bad); }}
  .outcome.skip {{ background: rgba(139,148,158,0.15); color: var(--muted); }}

  .summary {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 22px; margin-bottom: 24px;
    display: flex; gap: 32px; flex-wrap: wrap;
  }}
  .summary .stat .v {{ font-size: 28px; font-weight: 800; line-height: 1; }}
  .summary .stat .v.gold {{ color: var(--gold); }}
  .summary .stat .v.good {{ color: var(--good); }}
  .summary .stat .v.warn {{ color: var(--warn); }}
  .summary .stat .v.muted {{ color: var(--muted); }}
  .summary .stat .l {{ color: var(--muted); font-size: 11px;
                        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}

  .legend {{ color: var(--muted); font-size: 12px; padding: 12px 16px;
              background: var(--panel); border-radius: 8px; margin-top: 24px; }}
  .legend code {{ background:#0d1117; padding:2px 5px; border-radius:3px; }}

  .phase-banner {{
    border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;
    display: flex; gap: 16px; align-items: center;
    border: 1px solid var(--line);
  }}
  .phase-banner .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .phase-banner.live {{ background: rgba(63, 185, 80, 0.10); border-color: var(--good); }}
  .phase-banner.live .dot {{ background: var(--good); box-shadow: 0 0 8px var(--good);
                              animation: pulse 1.6s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
  .phase-banner.pending {{ background: rgba(210, 153, 34, 0.10); border-color: var(--warn); }}
  .phase-banner.pending .dot {{ background: var(--warn); }}
  .phase-banner.closed {{ background: rgba(88, 166, 255, 0.08); border-color: var(--blue); }}
  .phase-banner.closed .dot {{ background: var(--blue); }}
  .phase-banner.weekend {{ background: rgba(139, 148, 158, 0.10); border-color: var(--muted); }}
  .phase-banner.weekend .dot {{ background: var(--muted); }}
  .phase-banner .ph {{ font-weight: 700; font-size: 14px; color: #fff; letter-spacing: 0.03em; }}
  .phase-banner .ph-meta {{ color: var(--muted); font-size: 12px; }}

  .timeline {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 18px; margin-bottom: 24px;
  }}
  .timeline .t-row {{ display: grid; grid-template-columns: 90px 1fr; gap: 12px;
                       padding: 4px 0; font-size: 12px; }}
  .timeline .t-row .t-time {{ color: var(--gold); font-weight: 700; font-variant-numeric: tabular-nums; }}
  .timeline .t-row .t-desc {{ color: var(--muted); }}
  .timeline .t-row.now {{ background: rgba(88,166,255,0.10); border-radius: 4px;
                           padding: 4px 8px; margin: 0 -8px; }}
  .timeline .t-row.now .t-time {{ color: var(--blue); }}
  .timeline .t-row.now .t-desc {{ color: #fff; }}
</style>
</head><body>

<p style="margin: 0 0 8px;"><a href="JACKPOT_dashboard.html" style="color:#58a6ff; text-decoration:none; font-size:13px;">&larr; Back to Command Center</a></p>
<h1>JACKPOT weekly calendar</h1>
<p class="sub">{generated_full} · Universe: {tickers_str} · Equity ${equity:,.0f} · Auto-refresh every 5 min</p>

{phase_html}

{timeline_html}

<div class="summary">
  <div class="stat"><div class="v gold">{n_ultra_2w}</div><div class="l">ULTRA in 2 weeks</div></div>
  <div class="stat"><div class="v good">{n_jp_2w}</div><div class="l">JACKPOT in 2 weeks</div></div>
  <div class="stat"><div class="v warn">{n_hot_2w}</div><div class="l">HOT in 2 weeks</div></div>
  <div class="stat"><div class="v muted">{n_skip_2w}</div><div class="l">SKIP days</div></div>
  <div class="stat"><div class="v">{this_week_strongest}</div><div class="l">strongest this week</div></div>
</div>

<h2>Last week — {last_week_label}</h2>
<div class="week-grid">
  {last_week_html}
</div>

<h2>This week — {this_week_label}</h2>
<div class="week-grid">
  {this_week_html}
</div>

<div class="legend">
  <strong>How to read:</strong> Each card is one trading day, Mon–Fri.
  The big banner is the strongest signal across the four tickers that day.
  Inside each card, the ticker chips show the per-ticker signal plus the three model scores
  (<span style="color:var(--warn);">P(vol)</span> /
  <span style="color:var(--good);">P(0DTE)</span> /
  <span style="color:var(--blue);">P(weekly)</span>).
  Past-day chips show a <code>WIN</code>/<code>LOSE</code> badge for tickers that fired JACKPOT or ULTRA,
  based on the actual 0DTE straddle return for that day. Today is outlined in blue; future days are dimmed.
  All scores are out-of-sample (walk-forward retrained monthly).
</div>

</body></html>
"""


def _ticker_chip(t: dict, is_past: bool, is_today: bool) -> str:
    sig = t["signal"]
    sig_class = _signal_class(sig)
    sig_label = _signal_label(sig)

    scores = "—" if not t["had_data"] else (
        f"<span class='pv'>{t['p_vol'] * 100:.0f}%</span>"
        f" / <span class='pp'>{t['p_pnl'] * 100:.0f}%</span>"
        f" / <span class='pw'>"
        f"{(t['p_weekly'] * 100 if not pd.isna(t['p_weekly']) else 0):.0f}%</span>"
    )

    outcome_html = ""
    if is_past and t["had_data"] and sig in ("GO_JACKPOT", "GO_ULTRA_JACKPOT"):
        zret = t["zdte_realized"]
        if not pd.isna(zret):
            if zret > 0:
                outcome_html = f"<span class='outcome win'>+{zret * 100:.0f}%</span>"
            else:
                outcome_html = f"<span class='outcome lose'>{zret * 100:.0f}%</span>"
        else:
            outcome_html = "<span class='outcome skip'>—</span>"

    return (
        f"<div class='ticker-chip'>"
        f"<div class='tkr'>{t['ticker']}</div>"
        f"<div class='sig {sig_class}'>{sig_label}{outcome_html}</div>"
        f"<div class='scores'>{scores}</div>"
        f"</div>"
    )


def _day_card_html(day: dict) -> str:
    klass = ["day-card"]
    if day["is_today"]:
        klass.append("today")
    if day["is_future"]:
        klass.append("future")
    if day["strongest"]:
        klass.append(_signal_class(day["strongest"]))

    chips_html = "\n".join(_ticker_chip(t, day["is_past"], day["is_today"]) for t in day["tickers"])

    strongest_label = _signal_label(day["strongest"]) if day["strongest"] else (
        "PENDING" if day["is_future"] else "—"
    )
    banner_class = _signal_class(day["strongest"]) if day["strongest"] else "empty"

    date_str = day["date"].strftime("%b %d")
    return (
        f"<div class='{' '.join(klass)}'>"
        f"<div class='day-head'>"
        f"<span class='wd'>{day['weekday']}</span>"
        f"<span class='dt'>{date_str}</span>"
        f"</div>"
        f"<div class='day-banner {banner_class}'>{strongest_label}</div>"
        f"<div class='tickers'>{chips_html}</div>"
        f"</div>"
    )


def _week_html(cal: list[dict]) -> str:
    return "\n".join(_day_card_html(d) for d in cal)


def _phase_html(phase: dict) -> str:
    p = phase["phase"]
    if p == "OPEN_LIVE":
        klass = "live"
        sub = (f"Today's signal is final. Decision window open · "
               f"{phase['minutes_since_open']} min into the session.")
    elif p == "OPEN_PENDING_DATA":
        klass = "pending"
        sub = (f"Market opened {phase['minutes_since_open']} min ago - "
               f"Yahoo's daily bar takes ~15-20 min to reflect the open. "
               f"Refresh again at 9:50 AM ET for the final signal.")
    elif p == "PRE_OPEN":
        klass = "pending"
        mins = phase['next_open_in']
        hrs = mins // 60
        mins_left = mins % 60
        sub = (f"Cash market opens in {hrs}h {mins_left}m (9:30 AM ET). "
               f"Signal is pending until the opening print sets the gap.")
    elif p == "AFTER_HOURS":
        klass = "closed"
        sub = ("Today's signal is frozen. If a signal fired and you didn't act, "
               "wait for the next session.")
    else:
        klass = "weekend"
        sub = "Markets closed. Next open: Monday 9:30 AM ET."

    return (
        f"<div class='phase-banner {klass}'>"
        f"<div class='dot'></div>"
        f"<div><div class='ph'>{phase['label']}</div>"
        f"<div class='ph-meta'>{sub}</div></div>"
        f"</div>"
    )


def _timeline_html(phase: dict) -> str:
    """Render a small fixed timeline showing the daily signal-lifecycle."""
    rows = [
        ("BEFORE", "4:00 PM ET prev day", "29 of 30 features locked - signal ~95% predictable"),
        ("PRE_OPEN", "9:00-9:30 AM ET", "Final pre-market prep · signal still pending today's open"),
        ("OPEN_PENDING_DATA", "9:30-9:50 AM ET", "Market opens · opening print fixes the last feature"),
        ("OPEN_LIVE", "9:50 AM-4:00 PM ET", "Signal LIVE & final · trade entry window"),
        ("AFTER_HOURS", "4:00 PM ET onward", "Signal frozen until next session"),
    ]
    now_key = phase["phase"]
    out = ["<div class='timeline'><strong style='font-size:13px;color:#fff;'>"
           "Daily signal lifecycle</strong><br/>"
           "<span style='color:var(--muted);font-size:11px;'>"
           "All features are pre-computed except <code style='background:#0d1117;padding:1px 4px;border-radius:3px;'>"
           "gap_pct</code>, which fixes at the opening bell. After that, the signal does not change.</span>"
           "<div style='margin-top:8px;'>"]
    for key, time, desc in rows:
        cls = " now" if key == now_key else ""
        out.append(f"<div class='t-row{cls}'><div class='t-time'>{time}</div>"
                   f"<div class='t-desc'>{desc}</div></div>")
    out.append("</div></div>")
    return "".join(out)


def render(
    last_week: list[dict],
    this_week: list[dict],
    tickers: list[str],
    equity: float,
) -> str:
    all_days = last_week + this_week
    n_ultra = sum(1 for d in all_days if d["strongest"] == "GO_ULTRA_JACKPOT")
    n_jp = sum(1 for d in all_days if d["strongest"] == "GO_JACKPOT")
    n_hot = sum(1 for d in all_days if d["strongest"] == "GO_HOT")
    n_skip = sum(1 for d in all_days if d["strongest"] == "SKIP")

    this_strongest = "—"
    rank = 0
    for d in this_week:
        if d["strongest"] and SIGNAL_RANK.get(d["strongest"], 0) > rank:
            rank = SIGNAL_RANK[d["strongest"]]
            this_strongest = _signal_label(d["strongest"])

    last_label = (f"{last_week[0]['date'].strftime('%b %d')}"
                  f" – {last_week[-1]['date'].strftime('%b %d')}")
    this_label = (f"{this_week[0]['date'].strftime('%b %d')}"
                  f" – {this_week[-1]['date'].strftime('%b %d')}")

    now = datetime.now()
    phase = market_phase(now)
    return HTML_TEMPLATE.format(
        generated=now.strftime("%Y-%m-%d %H:%M"),
        generated_full=now.strftime("%A, %B %d, %Y %H:%M"),
        tickers_str=", ".join(tickers),
        equity=equity,
        phase_html=_phase_html(phase),
        timeline_html=_timeline_html(phase),
        n_ultra_2w=n_ultra,
        n_jp_2w=n_jp,
        n_hot_2w=n_hot,
        n_skip_2w=n_skip,
        this_week_strongest=this_strongest,
        last_week_label=last_label,
        this_week_label=this_label,
        last_week_html=_week_html(last_week),
        this_week_html=_week_html(this_week),
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tickers", default=",".join(DEFAULT_JACKPOT_UNIVERSE))
    parser.add_argument("--equity", type=float, default=5_000.0)
    parser.add_argument("--no-refresh-data", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logger.info("Scoring last ~15 sessions for %d tickers ...", len(tickers))
    data = collect_week_data(tickers, n_days=15, refresh_data=not args.no_refresh_data,
                             retrain=args.retrain)

    today = datetime.now()
    this_monday = _monday_of_week(today)
    last_monday = this_monday - timedelta(days=7)

    last_week = build_calendar(data, last_monday)
    this_week = build_calendar(data, this_monday)

    html = render(last_week, this_week, tickers, args.equity)
    out_path = Path(args.out_dir) / "JACKPOT_weekly_calendar.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    summary = {
        "generated": datetime.now().isoformat(),
        "this_week": [
            {**{k: v for k, v in d.items() if k != "date"},
             "date": d["date"].strftime("%Y-%m-%d")}
            for d in this_week
        ],
        "last_week": [
            {**{k: v for k, v in d.items() if k != "date"},
             "date": d["date"].strftime("%Y-%m-%d")}
            for d in last_week
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    logger.info("Wrote calendar -> %s", out_path)

    # Console summary
    print()
    print("Last week:")
    for d in last_week:
        print(f"  {d['weekday']} {d['date'].strftime('%b %d')}: {_signal_label(d['strongest'] or 'none')}")
    print("This week:")
    for d in this_week:
        mark = "  ← TODAY" if d["is_today"] else (" (future)" if d["is_future"] else "")
        print(f"  {d['weekday']} {d['date'].strftime('%b %d')}: {_signal_label(d['strongest'] or 'pending')}{mark}")

    if not args.no_open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
