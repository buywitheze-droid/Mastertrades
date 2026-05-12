"""Render a single, very clear HTML dashboard ranking weekdays Mon-Fri by:

- **Volatility** (avg daily range, single-day Yang-Zhang vol)
- **Flatness** (combined z-score of range + body — small range *and* small body)
- **Order flow** (closing strength always; CVD + VWAP-deviation when hourly
  data is available)

Everything is shown on one page. The two answers (most volatile / most flat)
are displayed in big "answer cards" at the top. All charts are interactive
(Chart.js via CDN). The page works as a static file — no server.

Usage::

    python -m src.report_weekday_dashboard
    python -m src.report_weekday_dashboard --lookback-days 365
    python -m src.report_weekday_dashboard --ticker QQQ --no-open

If a matching hourly file (``data/<TICKER>_1h.csv``/``.parquet``) exists, the
dashboard automatically adds true intraday-derived order-flow proxies. Run
``python -m src.download_spy --interval 1h --start 2024-05-08`` to enable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_history
from . import volatility as v
from . import order_flow_proxies as ofp


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


logger = logging.getLogger("report_weekday_dashboard")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _single_day_yz_ann(df: pd.DataFrame) -> pd.Series:
    log_o_pc = np.log(df["Open"] / df["Close"].shift(1))
    log_c_o = np.log(df["Close"] / df["Open"])
    log_h_o = np.log(df["High"] / df["Open"])
    log_h_c = np.log(df["High"] / df["Close"])
    log_l_o = np.log(df["Low"] / df["Open"])
    log_l_c = np.log(df["Low"] / df["Close"])
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    yz = log_o_pc**2 + 0.34 * log_c_o**2 + 0.66 * rs
    return np.sqrt(yz.clip(lower=0) * 252)


def compute(daily: pd.DataFrame, lookback_days: int, intraday: pd.DataFrame | None) -> dict:
    today = daily.index.max()
    start = today - pd.Timedelta(days=lookback_days)
    df = daily.loc[start:].copy()

    df["RangePct"] = v.daily_range_pct(df)
    df["BodyPct"] = v.body_pct(df)
    df["EfficiencyRatio"] = v.efficiency_ratio(df)
    df["YZ_1d"] = _single_day_yz_ann(df)
    df["ClosingStrength"] = ofp.closing_strength(df)
    df["IntradayRet"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Weekday"] = df.index.day_name()
    df = df[df["Weekday"].isin(WEEKDAYS)].dropna(
        subset=["RangePct", "BodyPct", "YZ_1d"]
    )

    # Day color: doji band is ±5 bps to avoid labeling near-flat sessions
    # as "green" or "red" just from rounding noise.
    DOJI_BAND = 0.0005
    df["Color"] = np.where(
        df["IntradayRet"] > DOJI_BAND, "green",
        np.where(df["IntradayRet"] < -DOJI_BAND, "red", "doji"),
    )

    # Combined "flatness" z-score: small range AND small body.
    df["FlatScore"] = (
        (df["RangePct"] - df["RangePct"].mean()) / df["RangePct"].std(ddof=0)
        + (df["BodyPct"] - df["BodyPct"].mean()) / df["BodyPct"].std(ddof=0)
    )

    # Optional intraday-derived order-flow proxies. Hourly Yahoo data is
    # typically tz-aware while daily is naive, which would break both slicing
    # and joining — strip tz on the intraday index up front.
    if intraday is not None and not intraday.empty:
        if isinstance(intraday.index, pd.DatetimeIndex) and intraday.index.tz is not None:
            intraday = intraday.copy()
            intraday.index = intraday.index.tz_localize(None)
        intraday = intraday.loc[start:]
        cvd = ofp.session_cvd_close(intraday)
        vwap_dev = ofp.session_vwap_close_deviation(intraday)
        df = df.join(cvd).join(vwap_dev)
        # CVD raw values can be huge — normalize by daily volume so it is
        # comparable across periods of growing share count.
        if "CVD_close" in df.columns:
            df["CVDNormalized"] = df["CVD_close"] / df["Volume"].replace(0, np.nan)

    agg_map = {
        "n_days": ("RangePct", "count"),
        "avg_range": ("RangePct", "mean"),
        "median_range": ("RangePct", "median"),
        "max_range": ("RangePct", "max"),
        "avg_body": ("BodyPct", "mean"),
        "avg_yz": ("YZ_1d", "mean"),
        "avg_eff": ("EfficiencyRatio", "mean"),
        "avg_close_strength": ("ClosingStrength", "mean"),
        "avg_flat_score": ("FlatScore", "mean"),
    }
    if "CVDNormalized" in df.columns:
        agg_map["avg_cvd_norm"] = ("CVDNormalized", "mean")
    if "VWAPDeviation" in df.columns:
        agg_map["avg_vwap_dev"] = ("VWAPDeviation", "mean")

    by_dow = df.groupby("Weekday").agg(**agg_map).reindex(WEEKDAYS)

    sorted_by_range = df.sort_values("RangePct", ascending=False)
    sorted_by_flat = df.sort_values("FlatScore")

    top_volatile = sorted_by_range.head(5)
    top_flat = sorted_by_flat.head(5)

    # Per-weekday color counts, average up/down move size, and cumulative
    # return if you held SPY only on that weekday.
    direction_rows = []
    for wd in WEEKDAYS:
        sub = df[df["Weekday"] == wd]
        n = len(sub)
        green = sub[sub["Color"] == "green"]
        red = sub[sub["Color"] == "red"]
        doji = sub[sub["Color"] == "doji"]
        cum = float((1.0 + sub["IntradayRet"]).prod() - 1.0)
        direction_rows.append({
            "weekday": wd,
            "n": n,
            "n_green": int(len(green)),
            "n_red": int(len(red)),
            "n_doji": int(len(doji)),
            "pct_green": float(len(green) / n) if n else float("nan"),
            "pct_red": float(len(red) / n) if n else float("nan"),
            "avg_green": float(green["IntradayRet"].mean()) if len(green) else float("nan"),
            "avg_red": float(red["IntradayRet"].mean()) if len(red) else float("nan"),
            "cum_ret": cum,
        })
    direction = pd.DataFrame(direction_rows).set_index("weekday")

    # Best/worst sessions for the most-volatile weekday (default: the
    # answer card winner). Useful for "show me the standout sessions".
    most_volatile_name = by_dow["avg_range"].idxmax()
    winner_sessions = df[df["Weekday"] == most_volatile_name].copy()
    best_sessions = winner_sessions.nlargest(5, "IntradayRet")
    worst_sessions = winner_sessions.nsmallest(5, "IntradayRet")

    strip = [
        {
            "weekday": WEEKDAYS.index(row.Weekday),
            "range_pct": round(float(row.RangePct) * 100, 4),
            "date": idx.strftime("%Y-%m-%d"),
        }
        for idx, row in df.iterrows()
    ]

    most_volatile = by_dow["avg_range"].idxmax()
    most_flat = by_dow["avg_flat_score"].idxmin()

    return {
        "df": df,
        "by_dow": by_dow,
        "top_volatile": top_volatile,
        "top_flat": top_flat,
        "strip": strip,
        "most_volatile": most_volatile,
        "most_flat": most_flat,
        "direction": direction,
        "best_sessions": best_sessions,
        "worst_sessions": worst_sessions,
        "window_start": df.index.min().date().isoformat(),
        "window_end": df.index.max().date().isoformat(),
        "has_intraday": "CVDNormalized" in df.columns,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_pct(x: float, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _fmt_signed_pct(x: float, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:+.{digits}f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def _summary_table(by_dow: pd.DataFrame, has_intraday: bool) -> str:
    headers = [
        "Weekday", "n", "Avg range", "Avg body", "Avg YZ vol",
        "Avg efficiency", "Avg close strength",
    ]
    if has_intraday:
        headers += ["Avg CVD/Vol", "Avg VWAP dev"]

    rows = []
    for dow in WEEKDAYS:
        r = by_dow.loc[dow]
        cells = [
            dow,
            f"{int(r['n_days'])}",
            _fmt_pct(r["avg_range"]),
            _fmt_pct(r["avg_body"]),
            _fmt_pct(r["avg_yz"]),
            _fmt_num(r["avg_eff"]),
            _fmt_num(r["avg_close_strength"]),
        ]
        if has_intraday:
            cells.append(_fmt_signed_pct(r.get("avg_cvd_norm"), digits=2))
            cells.append(_fmt_signed_pct(r.get("avg_vwap_dev")))
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _moves_table(rows: pd.DataFrame, label: str) -> str:
    body = []
    for idx, r in rows.iterrows():
        body.append(
            f"<tr><td>{idx.strftime('%Y-%m-%d')}</td>"
            f"<td>{r['Weekday']}</td>"
            f"<td>{_fmt_pct(r['RangePct'])}</td>"
            f"<td>{_fmt_pct(r['BodyPct'])}</td>"
            f"<td>{r['EfficiencyRatio']:.2f}</td></tr>"
        )
    return (
        f"<h3>{label}</h3>"
        "<table><thead><tr>"
        "<th>Date</th><th>Weekday</th><th>Range</th><th>|Body|</th><th>Eff.</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _direction_table(direction: pd.DataFrame) -> str:
    rows = []
    for wd in WEEKDAYS:
        r = direction.loc[wd]
        cum_class = "pos" if r["cum_ret"] > 0 else "neg"
        rows.append(
            f"<tr><td>{wd}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td class='pos'>{int(r['n_green'])} ({r['pct_green']:.1%})</td>"
            f"<td class='neg'>{int(r['n_red'])} ({r['pct_red']:.1%})</td>"
            f"<td>{int(r['n_doji'])}</td>"
            f"<td class='pos'>{_fmt_signed_pct(r['avg_green'])}</td>"
            f"<td class='neg'>{_fmt_signed_pct(r['avg_red'])}</td>"
            f"<td class='{cum_class}'>{_fmt_signed_pct(r['cum_ret'], digits=2)}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Weekday</th><th>n</th><th>Green</th><th>Red</th><th>Doji</th>"
        "<th>Avg green</th><th>Avg red</th><th>Cum (this WD only)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _color_moves_table(rows: pd.DataFrame, label: str) -> str:
    body = []
    for idx, r in rows.iterrows():
        cls = "pos" if r["IntradayRet"] > 0 else "neg"
        body.append(
            f"<tr><td>{idx.strftime('%Y-%m-%d')}</td>"
            f"<td>{r['Open']:.2f}</td><td>{r['Close']:.2f}</td>"
            f"<td class='{cls}'>{_fmt_signed_pct(r['IntradayRet'])}</td></tr>"
        )
    return (
        f"<h3>{label}</h3>"
        "<table><thead><tr>"
        "<th>Date</th><th>Open</th><th>Close</th><th>Open→Close</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0b0d12;
      --card: #141822;
      --card-2: #1a1f2c;
      --text: #ecedef;
      --muted: #8b95a7;
      --vol: #ef4444;
      --vol-soft: rgba(239,68,68,0.18);
      --flat: #3b82f6;
      --flat-soft: rgba(59,130,246,0.18);
      --neutral: #2a3340;
      --bar: #4f8cff;
      --border: #232938;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 100px; }}
    h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -0.01em; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 36px; }}

    .answers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 36px; }}
    @media (max-width: 880px) {{ .answers {{ grid-template-columns: 1fr; }} }}
    .answer {{
      border-radius: 16px;
      padding: 28px 32px;
      border: 1px solid var(--border);
    }}
    .answer.vol {{
      background: linear-gradient(135deg, var(--vol-soft), transparent), var(--card);
      border-left: 4px solid var(--vol);
    }}
    .answer.flat {{
      background: linear-gradient(135deg, var(--flat-soft), transparent), var(--card);
      border-left: 4px solid var(--flat);
    }}
    .answer .label {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-weight: 600;
    }}
    .answer .name {{
      font-size: 56px;
      font-weight: 800;
      line-height: 1.05;
      margin: 8px 0 14px;
    }}
    .answer.vol  .name {{ color: var(--vol); }}
    .answer.flat .name {{ color: var(--flat); }}
    .stat-row {{ display: flex; gap: 22px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }}
    .stat-row .v {{ color: var(--text); font-weight: 600; font-size: 15px; }}

    .section-title {{
      font-size: 13px;
      letter-spacing: 0.1em;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 600;
      margin: 28px 0 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px 24px;
      margin-bottom: 22px;
    }}
    .card h2 {{
      font-size: 13px; margin: 0 0 14px; color: var(--muted);
      font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    @media (max-width: 880px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    canvas {{ width: 100% !important; }}
    .h-280 canvas {{ height: 280px !important; }}
    .h-340 canvas {{ height: 340px !important; }}
    .h-380 canvas {{ height: 380px !important; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:hover td {{ background: rgba(79, 140, 255, 0.06); }}
    td.pos {{ color: #22c55e; }}
    td.neg {{ color: #ef4444; }}
    .moves-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    @media (max-width: 880px) {{ .moves-grid {{ grid-template-columns: 1fr; }} }}
    .moves-grid h3 {{ font-size: 14px; margin: 0 0 10px; color: var(--text); }}

    .footer {{ color: var(--muted); font-size: 12px; margin-top: 36px; line-height: 1.7; }}
    code {{ background: #0a0c11; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
    .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; }}
    .pill.vol {{ background: var(--vol-soft); color: var(--vol); }}
    .pill.flat {{ background: var(--flat-soft); color: var(--flat); }}
  </style>
</head>
<body>
<div class="wrap">

  <h1>{ticker} · weekday volatility dashboard</h1>
  <div class="subtitle">{window_start} → {window_end} · {n_days} sessions{intraday_note}</div>

  <div class="answers">
    <div class="answer vol">
      <div class="label">Most volatile weekday</div>
      <div class="name">{most_volatile}</div>
      <div class="stat-row">
        <div>Avg range <span class="v">{vol_avg_range}</span></div>
        <div>Avg body <span class="v">{vol_avg_body}</span></div>
        <div>YZ vol (ann.) <span class="v">{vol_avg_yz}</span></div>
      </div>
    </div>
    <div class="answer flat">
      <div class="label">Most flat weekday</div>
      <div class="name">{most_flat}</div>
      <div class="stat-row">
        <div>Avg range <span class="v">{flat_avg_range}</span></div>
        <div>Avg body <span class="v">{flat_avg_body}</span></div>
        <div>YZ vol (ann.) <span class="v">{flat_avg_yz}</span></div>
      </div>
    </div>
  </div>

  <div class="section-title">Ranking</div>
  <div class="card h-340">
    <h2>Avg daily range % per weekday — sorted by volatility</h2>
    <canvas id="chart-ranking"></canvas>
  </div>

  <div class="section-title">Detail</div>
  <div class="grid-2">
    <div class="card h-280">
      <h2>Avg single-day Yang-Zhang vol (ann.)</h2>
      <canvas id="chart-yz"></canvas>
    </div>
    <div class="card h-280">
      <h2>Avg |body| % (open → close magnitude)</h2>
      <canvas id="chart-body"></canvas>
    </div>
  </div>

  <div class="section-title">Order-flow context</div>
  <div class="grid-2">
    <div class="card h-280">
      <h2>Closing strength (0 = close at low, 1 = close at high)</h2>
      <canvas id="chart-close"></canvas>
    </div>
    <div class="card h-280">
      <h2>{flow_chart_2_title}</h2>
      <canvas id="chart-flow2"></canvas>
    </div>
  </div>

  <div class="section-title">Direction (green vs red days)</div>
  <div class="grid-2">
    <div class="card h-280">
      <h2>Open → close direction count per weekday</h2>
      <canvas id="chart-direction"></canvas>
    </div>
    <div class="card h-280">
      <h2>Cumulative return if held SPY only on this weekday</h2>
      <canvas id="chart-cumret"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Direction summary</h2>
    {direction_table}
  </div>
  <div class="moves-grid">
    <div class="card">
      <span class="pill vol">5 best {most_volatile}s (open → close)</span>
      {best_sessions_table}
    </div>
    <div class="card">
      <span class="pill flat">5 worst {most_volatile}s (open → close)</span>
      {worst_sessions_table}
    </div>
  </div>

  <div class="section-title">Per-session detail</div>
  <div class="card h-380">
    <h2>Every session as a dot — range % grouped by weekday</h2>
    <canvas id="chart-strip"></canvas>
  </div>

  <div class="card">
    <h2>Per-weekday summary</h2>
    {summary_table}
  </div>

  <div class="moves-grid">
    <div class="card">
      <span class="pill vol">Top 5 volatile sessions</span>
      {top_volatile_table}
    </div>
    <div class="card">
      <span class="pill flat">Top 5 flat sessions</span>
      {top_flat_table}
    </div>
  </div>

  <div class="footer">
    <strong>Definitions.</strong>
    Range = (High − Low) / Open · Body = |Close − Open| / Open ·
    Efficiency = body ÷ range (1 = pure trend, 0 = pure chop) ·
    Yang-Zhang = single-day estimator combining overnight, open-to-close, and Rogers-Satchell variance, annualized via √252 ·
    Closing strength = (Close − Low) / (High − Low) ·
    CVD/Vol = signed hourly volume summed over the session, divided by total volume ·
    VWAP dev = (Close − session VWAP) / session VWAP.
    <br />
    Built with <code>python -m src.report_weekday_dashboard</code> on {generated_at}.
  </div>

</div>

<script>
  const labels = {labels_json};
  const volIdx = {vol_idx};
  const flatIdx = {flat_idx};

  function colorFor(i) {{
    if (i === volIdx) return '#ef4444';
    if (i === flatIdx) return '#3b82f6';
    return '#465264';
  }}
  const colors = labels.map((_, i) => colorFor(i));

  const baseOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#8b95a7' }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b95a7' }} }}
    }}
  }};

  // Sort weekdays by avg range to make the ranking chart visually obvious.
  const rangeData = {avg_range_json};
  const ranked = labels
    .map((lbl, i) => ({{ lbl, val: rangeData[i], color: colorFor(i) }}))
    .sort((a, b) => b.val - a.val);

  new Chart(document.getElementById('chart-ranking'), {{
    type: 'bar',
    data: {{
      labels: ranked.map(r => r.lbl),
      datasets: [{{
        data: ranked.map(r => r.val),
        backgroundColor: ranked.map(r => r.color),
        borderRadius: 8,
      }}]
    }},
    options: {{
      ...baseOpts,
      indexAxis: 'y',
      plugins: {{
        ...baseOpts.plugins,
        tooltip: {{ callbacks: {{ label: (ctx) => ctx.parsed.x.toFixed(3) + '%' }} }}
      }},
      scales: {{
        x: {{ ...baseOpts.scales.x, ticks: {{ ...baseOpts.scales.x.ticks, callback: (v) => v + '%' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b95a7', font: {{ size: 14 }} }} }}
      }}
    }}
  }});

  function makeBar(elId, dataArr, fmt) {{
    new Chart(document.getElementById(elId), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{ data: dataArr, backgroundColor: colors, borderRadius: 6 }}]
      }},
      options: {{
        ...baseOpts,
        plugins: {{
          ...baseOpts.plugins,
          tooltip: {{ callbacks: {{ label: (ctx) => fmt(ctx.parsed.y) }} }}
        }},
        scales: {{
          ...baseOpts.scales,
          y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: fmt }} }}
        }}
      }}
    }});
  }}

  makeBar('chart-yz',    {avg_yz_json},    (v) => v.toFixed(2) + '%');
  makeBar('chart-body',  {avg_body_json},  (v) => v.toFixed(3) + '%');
  makeBar('chart-close', {avg_close_json}, (v) => v.toFixed(3));
  makeBar('chart-flow2', {flow2_data_json}, (v) => {flow2_fmt});

  // Direction stacked bar (green / red / doji counts per weekday)
  new Chart(document.getElementById('chart-direction'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Green', data: {green_counts_json}, backgroundColor: '#22c55e', stack: 's' }},
        {{ label: 'Red',   data: {red_counts_json},   backgroundColor: '#ef4444', stack: 's' }},
        {{ label: 'Doji',  data: {doji_counts_json},  backgroundColor: '#64748b', stack: 's' }},
      ]
    }},
    options: {{
      ...baseOpts,
      plugins: {{ legend: {{ display: true, position: 'top', labels: {{ color: '#8b95a7' }} }} }},
      scales: {{
        x: {{ ...baseOpts.scales.x, stacked: true }},
        y: {{ ...baseOpts.scales.y, stacked: true, ticks: {{ ...baseOpts.scales.y.ticks, callback: (v) => v + ' days' }} }}
      }}
    }}
  }});

  // Cumulative WD-only return bar (green if positive, red if negative)
  const cumValues = {cum_ret_json};
  new Chart(document.getElementById('chart-cumret'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: cumValues,
        backgroundColor: cumValues.map(v => v >= 0 ? '#22c55e' : '#ef4444'),
        borderRadius: 6,
      }}]
    }},
    options: {{
      ...baseOpts,
      plugins: {{
        ...baseOpts.plugins,
        tooltip: {{ callbacks: {{ label: (ctx) => (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toFixed(2) + '%' }} }}
      }},
      scales: {{
        ...baseOpts.scales,
        y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: (v) => (v>=0?'+':'') + v + '%' }} }}
      }}
    }}
  }});

  const stripPoints = {strip_json}.map(p => ({{
    x: p.weekday + (Math.random() - 0.5) * 0.55,
    y: p.range_pct,
    date: p.date,
    wd: p.weekday,
  }}));

  new Chart(document.getElementById('chart-strip'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        data: stripPoints,
        backgroundColor: (ctx) => {{
          const wd = ctx.raw && ctx.raw.wd;
          if (wd === volIdx) return 'rgba(239,68,68,0.55)';
          if (wd === flatIdx) return 'rgba(59,130,246,0.55)';
          return 'rgba(140,150,170,0.45)';
        }},
        pointRadius: 3.2,
      }}]
    }},
    options: {{
      ...baseOpts,
      plugins: {{
        ...baseOpts.plugins,
        tooltip: {{ callbacks: {{
          label: (ctx) => `${{ctx.raw.date}}: ${{ctx.raw.y.toFixed(3)}}%`,
          title: () => '',
        }} }}
      }},
      scales: {{
        x: {{
          min: -0.7, max: 4.7,
          ticks: {{
            color: '#8b95a7', stepSize: 1,
            callback: (v) => labels[Math.round(v)] || '',
            font: {{ size: 13 }}
          }},
          grid: {{ display: false }}
        }},
        y: {{
          ticks: {{ color: '#8b95a7', callback: (v) => v + '%' }},
          grid: {{ color: 'rgba(255,255,255,0.06)' }}
        }}
      }}
    }}
  }});
</script>
</body>
</html>
"""


def render(stats: dict, ticker: str) -> str:
    by_dow = stats["by_dow"]
    most_volatile = stats["most_volatile"]
    most_flat = stats["most_flat"]
    direction = stats["direction"]

    if stats["has_intraday"]:
        flow2_title = "CVD bias (signed volume / total volume)"
        flow2_values = [round(float(by_dow.loc[d, "avg_cvd_norm"]) * 100, 4) for d in WEEKDAYS]
        flow2_fmt = "v.toFixed(2) + '%'"
        intraday_note = " · hourly intraday loaded"
    else:
        flow2_title = "Avg efficiency (body ÷ range)"
        flow2_values = [round(float(by_dow.loc[d, "avg_eff"]), 4) for d in WEEKDAYS]
        flow2_fmt = "v.toFixed(3)"
        intraday_note = ""

    return HTML.format(
        title=f"{ticker} weekday dashboard",
        ticker=ticker,
        window_start=stats["window_start"],
        window_end=stats["window_end"],
        n_days=int(by_dow["n_days"].sum()),
        intraday_note=intraday_note,
        most_volatile=most_volatile,
        vol_avg_range=_fmt_pct(by_dow.loc[most_volatile, "avg_range"]),
        vol_avg_body=_fmt_pct(by_dow.loc[most_volatile, "avg_body"]),
        vol_avg_yz=_fmt_pct(by_dow.loc[most_volatile, "avg_yz"], digits=2),
        most_flat=most_flat,
        flat_avg_range=_fmt_pct(by_dow.loc[most_flat, "avg_range"]),
        flat_avg_body=_fmt_pct(by_dow.loc[most_flat, "avg_body"]),
        flat_avg_yz=_fmt_pct(by_dow.loc[most_flat, "avg_yz"], digits=2),
        labels_json=json.dumps(WEEKDAYS),
        vol_idx=WEEKDAYS.index(most_volatile),
        flat_idx=WEEKDAYS.index(most_flat),
        avg_range_json=json.dumps([round(float(by_dow.loc[d, "avg_range"]) * 100, 4) for d in WEEKDAYS]),
        avg_yz_json=json.dumps([round(float(by_dow.loc[d, "avg_yz"]) * 100, 4) for d in WEEKDAYS]),
        avg_body_json=json.dumps([round(float(by_dow.loc[d, "avg_body"]) * 100, 4) for d in WEEKDAYS]),
        avg_close_json=json.dumps([round(float(by_dow.loc[d, "avg_close_strength"]), 4) for d in WEEKDAYS]),
        flow_chart_2_title=flow2_title,
        flow2_data_json=json.dumps(flow2_values),
        flow2_fmt=flow2_fmt,
        strip_json=json.dumps(stats["strip"]),
        summary_table=_summary_table(by_dow, stats["has_intraday"]),
        top_volatile_table=_moves_table(stats["top_volatile"], "Largest range %"),
        top_flat_table=_moves_table(stats["top_flat"], "Smallest range + body z-score"),
        direction_table=_direction_table(direction),
        best_sessions_table=_color_moves_table(stats["best_sessions"], "Largest open → close gain"),
        worst_sessions_table=_color_moves_table(stats["worst_sessions"], "Largest open → close loss"),
        green_counts_json=json.dumps([int(direction.loc[d, "n_green"]) for d in WEEKDAYS]),
        red_counts_json=json.dumps([int(direction.loc[d, "n_red"]) for d in WEEKDAYS]),
        doji_counts_json=json.dumps([int(direction.loc[d, "n_doji"]) for d in WEEKDAYS]),
        cum_ret_json=json.dumps([round(float(direction.loc[d, "cum_ret"]) * 100, 4) for d in WEEKDAYS]),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the weekday volatility dashboard.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--lookback-days", type=int, default=730)
    p.add_argument("--out", default=None)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    daily = load_history(args.ticker, interval="1d")
    try:
        intraday = load_history(args.ticker, interval="1h")
        logger.info("Hourly file found — including intraday-derived order-flow proxies.")
    except FileNotFoundError:
        intraday = None
        logger.info(
            "No hourly file (data/%s_1h.*) — order-flow chart falls back to "
            "efficiency ratio. Run `python -m src.download_spy --interval 1h "
            "--start 2024-05-08` to enable real intraday proxies.",
            args.ticker.upper(),
        )

    stats = compute(daily, lookback_days=args.lookback_days, intraday=intraday)
    html = render(stats, ticker=args.ticker.upper())

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_REPORTS_DIR / f"{args.ticker.upper()}_weekday_dashboard.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    by_dow = stats["by_dow"]
    logger.info(
        "Most volatile: %s (%s)  ·  Most flat: %s (%s)",
        stats["most_volatile"], _fmt_pct(by_dow.loc[stats["most_volatile"], "avg_range"]),
        stats["most_flat"], _fmt_pct(by_dow.loc[stats["most_flat"], "avg_range"]),
    )
    logger.info("Wrote %s", out_path)

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
