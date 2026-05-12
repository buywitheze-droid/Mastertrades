"""Render a self-contained HTML report ranking SPY weekdays by volatility.

For each session in the lookback window we compute:

- ``RangePct``  = (High - Low) / Open
- ``BodyPct``   = |Close - Open| / Open
- ``YZ_1d``     = single-day Yang-Zhang variance, annualized (sqrt * 252)

We then aggregate by weekday (Mon-Fri) and emit a single HTML file with
two interactive bar charts (Chart.js loaded via CDN), a per-day strip plot,
and two tables. The file is written to ``reports/weekday_volatility.html``
and opened in the default browser.

Usage::

    python -m src.report_weekday_volatility
    python -m src.report_weekday_volatility --ticker SPY --lookback-days 730
    python -m src.report_weekday_volatility --no-open      # skip opening browser

Requires that ``data/SPY_1d.csv`` (or .parquet) exists. Run
``python -m src.download_spy`` first if it doesn't.
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


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


logger = logging.getLogger("report_weekday_volatility")


def _single_day_yz_ann(df: pd.DataFrame) -> pd.Series:
    """Single-day Yang-Zhang variance, annualized (sqrt(.) * sqrt(252))."""
    log_o_pc = np.log(df["Open"] / df["Close"].shift(1))
    log_c_o = np.log(df["Close"] / df["Open"])
    log_h_o = np.log(df["High"] / df["Open"])
    log_h_c = np.log(df["High"] / df["Close"])
    log_l_o = np.log(df["Low"] / df["Open"])
    log_l_c = np.log(df["Low"] / df["Close"])
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    yz = log_o_pc**2 + 0.34 * log_c_o**2 + 0.66 * rs
    return np.sqrt(yz.clip(lower=0) * 252)


def compute_stats(daily: pd.DataFrame, lookback_days: int) -> dict:
    today = daily.index.max()
    start = today - pd.Timedelta(days=lookback_days)
    df = daily.loc[start:].copy()

    df["RangePct"] = v.daily_range_pct(df)
    df["BodyPct"] = v.body_pct(df)
    df["EfficiencyRatio"] = v.efficiency_ratio(df)
    df["YZ_1d"] = _single_day_yz_ann(df)
    df["Weekday"] = df.index.day_name()
    df = df[df["Weekday"].isin(WEEKDAYS)].dropna(
        subset=["RangePct", "BodyPct", "YZ_1d"]
    )

    by_dow = (
        df.groupby("Weekday")
        .agg(
            n_days=("RangePct", "count"),
            avg_range=("RangePct", "mean"),
            median_range=("RangePct", "median"),
            max_range=("RangePct", "max"),
            avg_body=("BodyPct", "mean"),
            avg_yz=("YZ_1d", "mean"),
            median_yz=("YZ_1d", "median"),
        )
        .reindex(WEEKDAYS)
    )

    # Most volatile single day per weekday + top-10 overall.
    sorted_by_range = df.sort_values("RangePct", ascending=False)
    top_per_dow = (
        sorted_by_range.groupby("Weekday").head(1).set_index("Weekday").reindex(WEEKDAYS)
    )
    top_overall = sorted_by_range.head(10)

    # Strip-plot data (one point per session).
    strip = [
        {
            "weekday": WEEKDAYS.index(row.Weekday),
            "range_pct": round(float(row.RangePct) * 100, 4),
            "date": idx.strftime("%Y-%m-%d"),
        }
        for idx, row in df.iterrows()
    ]

    winner = by_dow["avg_range"].idxmax()
    return {
        "df": df,
        "by_dow": by_dow,
        "top_per_dow": top_per_dow,
        "top_overall": top_overall,
        "strip": strip,
        "winner": winner,
        "window_start": df.index.min().date().isoformat(),
        "window_end": df.index.max().date().isoformat(),
    }


def _fmt_pct(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _table_html(by_dow: pd.DataFrame, top_per_dow: pd.DataFrame) -> str:
    rows = []
    for dow in WEEKDAYS:
        row = by_dow.loc[dow]
        top_row = top_per_dow.loc[dow] if dow in top_per_dow.index else None
        peak_date = (
            top_row.name.strftime("%Y-%m-%d")
            if top_row is not None and not pd.isna(top_row.get("RangePct"))
            else "—"
        )
        peak_range = (
            _fmt_pct(top_row["RangePct"])
            if top_row is not None and not pd.isna(top_row.get("RangePct"))
            else "—"
        )
        rows.append(
            f"<tr><td>{dow}</td>"
            f"<td>{int(row['n_days'])}</td>"
            f"<td>{_fmt_pct(row['avg_range'])}</td>"
            f"<td>{_fmt_pct(row['median_range'])}</td>"
            f"<td>{_fmt_pct(row['max_range'])}</td>"
            f"<td>{_fmt_pct(row['avg_body'])}</td>"
            f"<td>{_fmt_pct(row['avg_yz'])}</td>"
            f"<td>{peak_date}</td>"
            f"<td>{peak_range}</td></tr>"
        )
    body = "\n".join(rows)
    return f"""
    <table>
      <thead>
        <tr>
          <th>Weekday</th>
          <th>n</th>
          <th>Avg range</th>
          <th>Median range</th>
          <th>Max range</th>
          <th>Avg |body|</th>
          <th>Avg YZ vol (ann.)</th>
          <th>Peak day</th>
          <th>Peak range</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def _top_overall_table_html(top_overall: pd.DataFrame) -> str:
    rows = []
    for idx, r in top_overall.iterrows():
        rows.append(
            f"<tr><td>{idx.strftime('%Y-%m-%d')}</td>"
            f"<td>{r['Weekday']}</td>"
            f"<td>{_fmt_pct(r['RangePct'])}</td>"
            f"<td>{_fmt_pct(r['BodyPct'])}</td>"
            f"<td>{r['EfficiencyRatio']:.2f}</td>"
            f"<td>{_fmt_pct(r['YZ_1d'])}</td></tr>"
        )
    body = "\n".join(rows)
    return f"""
    <table>
      <thead>
        <tr>
          <th>Date</th><th>Weekday</th><th>Range %</th>
          <th>|Body| %</th><th>Efficiency</th><th>YZ vol (ann.)</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1115;
      --card: #161a22;
      --text: #e6e8eb;
      --muted: #98a2b3;
      --accent: #ef4444;
      --accent-soft: rgba(239, 68, 68, 0.15);
      --bar: #4f8cff;
      --bar-dim: #2a3a5e;
      --border: #232938;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 80px; }}
    h1 {{ font-size: 28px; margin: 0 0 4px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 28px; }}
    .winner-card {{
      background: linear-gradient(135deg, var(--accent-soft), transparent),
                  var(--card);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 12px;
      padding: 24px 28px;
      margin-bottom: 28px;
    }}
    .winner-label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .winner-name {{ font-size: 44px; font-weight: 700; margin: 4px 0 0; color: var(--accent); }}
    .winner-stat {{ color: var(--muted); margin-top: 6px; font-size: 14px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 28px;
    }}
    @media (max-width: 880px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }}
    .card h2 {{ font-size: 15px; margin: 0 0 12px; color: var(--muted); font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }}
    canvas {{ width: 100% !important; height: 280px !important; }}
    .strip-card canvas {{ height: 320px !important; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      text-align: right;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:hover td {{ background: rgba(79, 140, 255, 0.06); }}
    .footer {{ color: var(--muted); font-size: 12px; margin-top: 36px; line-height: 1.6; }}
    code {{ background: #0a0c11; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
<div class="wrap">

  <h1>{ticker} — most volatile weekday</h1>
  <div class="subtitle">Window: <strong>{window_start} → {window_end}</strong> ({n_days} sessions)</div>

  <div class="winner-card">
    <div class="winner-label">Most volatile weekday</div>
    <div class="winner-name">{winner}</div>
    <div class="winner-stat">
      Avg daily range: <strong>{winner_avg_range}</strong> &nbsp;·&nbsp;
      Avg single-day Yang-Zhang vol: <strong>{winner_avg_yz}</strong>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Avg daily range %</h2>
      <canvas id="chart-range"></canvas>
    </div>
    <div class="card">
      <h2>Avg Yang-Zhang vol (annualized)</h2>
      <canvas id="chart-yz"></canvas>
    </div>
  </div>

  <div class="card strip-card" style="margin-bottom:28px;">
    <h2>Per-session range % by weekday (every dot = one session)</h2>
    <canvas id="chart-strip"></canvas>
  </div>

  <div class="card" style="margin-bottom:28px;">
    <h2>Per-weekday summary</h2>
    {summary_table}
  </div>

  <div class="card">
    <h2>Top 10 single-session moves in the window</h2>
    {top_overall_table}
  </div>

  <div class="footer">
    <strong>Method.</strong> RangePct = (High − Low) / Open. BodyPct = |Close − Open| / Open.
    Yang-Zhang variance is the single-day decomposition (overnight + open-to-close + Rogers-Satchell)
    annualized via √252. Built from daily OHLCV downloaded with
    <code>python -m src.download_spy</code>. Generated {generated_at}.
  </div>

</div>

<script>
  const labels = {labels_json};
  const winnerIdx = {winner_idx};
  const palette = ['#4f8cff','#4f8cff','#4f8cff','#4f8cff','#4f8cff'];
  palette[winnerIdx] = '#ef4444';

  const commonOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#98a2b3' }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#98a2b3' }} }}
    }}
  }};

  new Chart(document.getElementById('chart-range'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: {avg_range_json},
        backgroundColor: palette,
        borderRadius: 6,
      }}]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        tooltip: {{ callbacks: {{ label: (ctx) => ctx.parsed.y.toFixed(3) + '%' }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, ticks: {{ ...commonOpts.scales.y.ticks, callback: (v) => v + '%' }} }} }}
    }}
  }});

  new Chart(document.getElementById('chart-yz'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: {avg_yz_json},
        backgroundColor: palette,
        borderRadius: 6,
      }}]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        tooltip: {{ callbacks: {{ label: (ctx) => ctx.parsed.y.toFixed(3) + '%' }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, ticks: {{ ...commonOpts.scales.y.ticks, callback: (v) => v + '%' }} }} }}
    }}
  }});

  const stripPoints = {strip_json}.map(p => ({{
    x: p.weekday + (Math.random() - 0.5) * 0.6,
    y: p.range_pct,
    date: p.date,
  }}));

  new Chart(document.getElementById('chart-strip'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        data: stripPoints,
        backgroundColor: 'rgba(79, 140, 255, 0.55)',
        pointRadius: 3,
      }}]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        tooltip: {{ callbacks: {{
          label: (ctx) => `${{ctx.raw.date}}: ${{ctx.raw.y.toFixed(3)}}%`,
          title: () => '',
        }} }}
      }},
      scales: {{
        x: {{
          min: -0.7, max: 4.7,
          ticks: {{
            color: '#98a2b3',
            stepSize: 1,
            callback: (v) => labels[Math.round(v)] || ''
          }},
          grid: {{ display: false }}
        }},
        y: {{
          ticks: {{ color: '#98a2b3', callback: (v) => v + '%' }},
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
    winner = stats["winner"]
    winner_idx = WEEKDAYS.index(winner)

    return HTML_TEMPLATE.format(
        title=f"{ticker} weekday volatility",
        ticker=ticker,
        window_start=stats["window_start"],
        window_end=stats["window_end"],
        n_days=int(by_dow["n_days"].sum()),
        winner=winner,
        winner_avg_range=_fmt_pct(by_dow.loc[winner, "avg_range"]),
        winner_avg_yz=_fmt_pct(by_dow.loc[winner, "avg_yz"]),
        summary_table=_table_html(by_dow, stats["top_per_dow"]),
        top_overall_table=_top_overall_table_html(stats["top_overall"]),
        labels_json=json.dumps(WEEKDAYS),
        winner_idx=winner_idx,
        avg_range_json=json.dumps([round(float(x) * 100, 4) for x in by_dow["avg_range"]]),
        avg_yz_json=json.dumps([round(float(x) * 100, 4) for x in by_dow["avg_yz"]]),
        strip_json=json.dumps(stats["strip"]),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render an HTML report ranking weekdays by volatility.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--lookback-days", type=int, default=730, help="Lookback window in calendar days (default: 730 ≈ 2y).")
    p.add_argument("--out", default=None, help="Output HTML path (default: reports/<TICKER>_weekday_volatility.html).")
    p.add_argument("--no-open", action="store_true", help="Do not open the report in the browser.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    daily = load_history(args.ticker, interval="1d")
    stats = compute_stats(daily, lookback_days=args.lookback_days)
    html = render(stats, ticker=args.ticker.upper())

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_REPORTS_DIR / f"{args.ticker.upper()}_weekday_volatility.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    by_dow = stats["by_dow"]
    logger.info("Most volatile weekday: %s (avg range %s)",
                stats["winner"], _fmt_pct(by_dow.loc[stats["winner"], "avg_range"]))
    logger.info("Per-weekday avg range: %s",
                {k: _fmt_pct(v) for k, v in by_dow["avg_range"].items()})
    logger.info("Wrote %s", out_path)

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
