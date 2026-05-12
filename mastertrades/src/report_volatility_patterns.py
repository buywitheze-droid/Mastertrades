"""Render an HTML report explaining what (structurally) the most volatile
SPY days have in common, ignoring news.

The report leads with a plain-English "signature of a volatile day" derived
from the strongest signals in ``volatility_patterns.find_patterns``, then
shows the supporting evidence as ranked tables and bar charts.

Usage::

    python -m src.report_volatility_patterns
    python -m src.report_volatility_patterns --lookback-days 365
    python -m src.report_volatility_patterns --volatile-quantile 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

from .loader import load_history
from .volatility_patterns import build_features, find_patterns


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

logger = logging.getLogger("report_volatility_patterns")


# ---------------------------------------------------------------------------
# Plain-English signature
# ---------------------------------------------------------------------------


def _build_signature(res: dict) -> list[dict]:
    """Plain-English bullet points describing the structural signature."""
    cont = res["continuous"].set_index("feature")
    bullets: list[dict] = []

    def fmt_pct(x: float, digits: int = 2) -> str:
        return f"{x * 100:+.{digits}f}%" if abs(x) < 1 else f"{x:+.{digits}f}"

    if "rsi14" in cont.index:
        v = cont.loc["rsi14"]
        bullets.append({
            "label": "Momentum is cool, not hot",
            "detail": f"RSI(14) on volatile days averages {v.mean_volatile:.0f} vs {v.mean_non_volatile:.0f} on calm days. Vol expansion happens after pullbacks, not at trend tops.",
            "tstat": v.tstat,
        })
    if "bb_pos" in cont.index:
        v = cont.loc["bb_pos"]
        bullets.append({
            "label": "Price sits in the lower third of recent range",
            "detail": f"Bollinger position averages {v.mean_volatile:.2f} on volatile days vs {v.mean_non_volatile:.2f} on calm days (0 = lower band, 1 = upper band).",
            "tstat": v.tstat,
        })
    if "dist_ma50" in cont.index:
        v = cont.loc["dist_ma50"]
        bullets.append({
            "label": "Price is below or at the 50-day moving average",
            "detail": f"Avg distance from MA50 is {fmt_pct(v.mean_volatile)} on volatile days vs {fmt_pct(v.mean_non_volatile)} on calm days.",
            "tstat": v.tstat,
        })
    if "lag5_avg_range" in cont.index and "realized_vol_5d" in cont.index:
        r = cont.loc["lag5_avg_range"]
        rv = cont.loc["realized_vol_5d"]
        bullets.append({
            "label": "Vol has already been expanding for several days",
            "detail": (
                f"5-day avg daily range entering the day is {r.mean_volatile*100:.2f}% (vs {r.mean_non_volatile*100:.2f}% on calm days), "
                f"and 5-day realized vol is {rv.mean_volatile*100:.0f}% (vs {rv.mean_non_volatile*100:.0f}%). Vol clusters in time."
            ),
            "tstat": max(r.tstat, rv.tstat),
        })
    if "lag1_volume_z" in cont.index:
        v = cont.loc["lag1_volume_z"]
        bullets.append({
            "label": "Yesterday's volume was already heavy",
            "detail": f"Prior-day volume z-score (vs 60-day mean) is {v.mean_volatile:+.2f} on volatile days vs {v.mean_non_volatile:+.2f} on calm days.",
            "tstat": v.tstat,
        })
    if "lag1_close_strength" in cont.index:
        v = cont.loc["lag1_close_strength"]
        bullets.append({
            "label": "Yesterday closed weak (in the lower half of its bar)",
            "detail": f"Prior-day closing strength is {v.mean_volatile:.2f} on volatile days vs {v.mean_non_volatile:.2f} on calm days.",
            "tstat": v.tstat,
        })

    cat = res["categorical"]
    after_vol = cat[(cat["feature"] == "is_after_volatile") & (cat["class"] == "1")]
    if not after_vol.empty:
        row = after_vol.iloc[0]
        bullets.append({
            "label": "Yesterday was itself a volatile day",
            "detail": (
                f"P(volatile | yesterday volatile) = {row['p_volatile_given']:.0%} vs {row['base_rate']:.0%} base "
                f"(lift {row['lift']:.2f}×) over n={int(row['n'])}."
            ),
            "tstat": float("inf"),  # surface this prominently
        })
    first_tom = cat[(cat["feature"] == "is_first_trading_day_of_month") & (cat["class"] == "1")]
    if not first_tom.empty:
        row = first_tom.iloc[0]
        bullets.append({
            "label": "Calendar edge: first trading day of the month",
            "detail": (
                f"First-trading-day-of-month sessions are volatile {row['p_volatile_given']:.0%} of the time vs {row['base_rate']:.0%} base "
                f"(lift {row['lift']:.2f}×). End-of-month is similar."
            ),
            "tstat": float("nan"),
        })
    thursday = cat[(cat["feature"] == "weekday") & (cat["class"] == "Thursday")]
    if not thursday.empty:
        row = thursday.iloc[0]
        bullets.append({
            "label": "Thursday over-indexes among volatile days",
            "detail": (
                f"P(volatile | Thursday) = {row['p_volatile_given']:.0%} vs {row['base_rate']:.0%} base "
                f"(lift {row['lift']:.2f}×). The other red-biased weekday."
            ),
            "tstat": float("nan"),
        })
    return bullets


# ---------------------------------------------------------------------------
# Charts data
# ---------------------------------------------------------------------------


_FEATURE_PRETTY = {
    "rsi14": "RSI(14)",
    "bb_pos": "BB position (0=lower, 1=upper)",
    "pct_in_20d_range": "% in 20d range",
    "dist_ma20": "Distance from MA20",
    "dist_ma50": "Distance from MA50",
    "dist_ma200": "Distance from MA200",
    "lag5_avg_range": "Prior 5d avg range",
    "lag1_volume_z": "Prior-day volume z",
    "lag5_avg_volume_z": "Prior 5d avg volume z",
    "realized_vol_5d": "Realized vol (5d)",
    "realized_vol_20d": "Realized vol (20d)",
    "lag1_range": "Prior-day range",
    "lag2_range": "Day-before range",
    "abs_gap_pct": "|Gap|",
    "range_compression_ratio": "5d/20d range ratio",
    "lag1_close_strength": "Prior closing strength",
    "lag1_body": "Prior |body|",
    "gap_pct": "Gap (signed)",
    "vol_regime_shift": "Vol regime shift (5d−20d)",
    "days_to_opex": "Days to OpEx Friday",
}

_CATEGORICAL_PRETTY = {
    "is_after_volatile=1": "Yesterday was a volatile day",
    "is_first_trading_day_of_month=1": "First trading day of month",
    "is_last_trading_day_of_month=1": "Last trading day of month",
    "is_turn_of_month=1": "Turn-of-month (±2 days)",
    "is_quarterly_opex_week=1": "Quarterly OpEx week",
    "is_opex_week=1": "Monthly OpEx week",
    "is_opex_day=1": "OpEx Friday itself",
    "is_lag1_nr4=1": "Yesterday was an NR4 (smallest range of last 4)",
    "is_lag1_nr7=1": "Yesterday was an NR7 (smallest of last 7)",
    "is_after_flat=1": "Yesterday was a flat day (bottom 20% range)",
    "is_after_2_flat=1": "Two flat days in a row",
    "is_after_3plus_flat=1": "Three+ flat days in a row",
    "lag1_color=1": "Yesterday was green",
    "lag1_color=-1": "Yesterday was red",
    "lag1_color=0": "Yesterday was a doji",
}


def _label_for(feature: str, cls: str) -> str:
    if feature == "weekday":
        return cls
    if feature == "week_of_month":
        return f"Week {cls} of month"
    return _CATEGORICAL_PRETTY.get(f"{feature}={cls}", f"{feature}={cls}")


def _continuous_chart_data(cont_df: pd.DataFrame, top_n: int = 10) -> dict:
    cont_df = cont_df.copy()
    cont_df["abs_t"] = cont_df["tstat"].abs()
    cont_df = cont_df.sort_values("abs_t", ascending=False).head(top_n)
    return {
        "labels": [_FEATURE_PRETTY.get(f, f) for f in cont_df["feature"]],
        "volatile": [round(float(x), 6) for x in cont_df["mean_volatile"]],
        "calm": [round(float(x), 6) for x in cont_df["mean_non_volatile"]],
        "tstat": [round(float(x), 2) for x in cont_df["tstat"]],
        "raw_features": list(cont_df["feature"]),
    }


def _categorical_chart_data(cat_df: pd.DataFrame, top_n: int = 12) -> dict:
    df = cat_df.copy()
    df = df[df["lift"] > 1.0].sort_values("lift", ascending=False).head(top_n)
    labels = [_label_for(r["feature"], r["class"]) for _, r in df.iterrows()]
    return {
        "labels": labels,
        "lift": [round(float(x), 3) for x in df["lift"]],
        "p_volatile": [round(float(x), 4) for x in df["p_volatile_given"]],
        "n": [int(x) for x in df["n"]],
    }


def _format_signature_html(bullets: list[dict]) -> str:
    items = []
    for b in bullets:
        t = b.get("tstat")
        badge = ""
        if isinstance(t, float) and t == float("inf"):
            badge = '<span class="badge strong">strong</span>'
        elif isinstance(t, float) and not pd.isna(t):
            absT = abs(t)
            if absT >= 5:
                badge = '<span class="badge strong">very strong</span>'
            elif absT >= 3:
                badge = '<span class="badge mid">strong</span>'
            else:
                badge = '<span class="badge weak">moderate</span>'
        items.append(
            f'<li><div class="sig-label">{b["label"]} {badge}</div>'
            f'<div class="sig-detail">{b["detail"]}</div></li>'
        )
    return "<ul class='signature'>" + "".join(items) + "</ul>"


def _categorical_table(cat_df: pd.DataFrame, top_n: int = 25) -> str:
    df = cat_df.head(top_n)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td>{_label_for(r['feature'], r['class'])}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['p_volatile_given']:.1%}</td>"
            f"<td>{r['base_rate']:.1%}</td>"
            f"<td><strong>{r['lift']:.2f}×</strong></td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Class</th><th>n</th><th>P(volatile | class)</th><th>Base rate</th><th>Lift</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _continuous_table(cont_df: pd.DataFrame, top_n: int = 20) -> str:
    df = cont_df.head(top_n)
    rows = []
    for _, r in df.iterrows():
        m_v = r["mean_volatile"]
        m_n = r["mean_non_volatile"]
        is_pct = r["feature"] in {
            "lag1_range", "lag2_range", "lag5_avg_range", "abs_gap_pct", "gap_pct",
            "lag1_body", "dist_ma20", "dist_ma50", "dist_ma200",
            "realized_vol_5d", "realized_vol_20d", "vol_regime_shift",
        }
        fmt = (lambda x: f"{x*100:+.2f}%") if is_pct else (lambda x: f"{x:+.2f}")
        rows.append(
            f"<tr><td>{_FEATURE_PRETTY.get(r['feature'], r['feature'])}</td>"
            f"<td>{fmt(m_v)}</td>"
            f"<td>{fmt(m_n)}</td>"
            f"<td><strong>{r['tstat']:+.1f}</strong></td>"
            f"<td>{int(r['n_volatile'])} / {int(r['n_non_volatile'])}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Feature</th><th>Mean (volatile)</th><th>Mean (calm)</th>"
        "<th>t-stat</th><th>n vol / n calm</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


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
      --text: #ecedef;
      --muted: #8b95a7;
      --vol: #ef4444;
      --vol-soft: rgba(239,68,68,0.18);
      --calm: #3b82f6;
      --border: #232938;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 100px; }}
    h1 {{ font-size: 30px; margin: 0 0 6px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 32px; }}

    .hero {{
      background: linear-gradient(135deg, var(--vol-soft), transparent), var(--card);
      border: 1px solid var(--border);
      border-left: 4px solid var(--vol);
      border-radius: 14px;
      padding: 26px 30px;
      margin-bottom: 28px;
    }}
    .hero h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .hero .meta {{ color: var(--muted); font-size: 13px; }}

    .section-title {{
      font-size: 13px; letter-spacing: 0.1em; color: var(--muted);
      text-transform: uppercase; font-weight: 600;
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

    .signature {{ list-style: none; padding: 0; margin: 0; }}
    .signature li {{
      padding: 14px 0;
      border-bottom: 1px solid var(--border);
    }}
    .signature li:last-child {{ border-bottom: 0; }}
    .sig-label {{ font-weight: 600; font-size: 16px; margin-bottom: 4px; }}
    .sig-detail {{ color: var(--muted); font-size: 14px; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-left: 6px;
      font-weight: 600;
    }}
    .badge.strong {{ background: rgba(239,68,68,0.18); color: var(--vol); }}
    .badge.mid {{ background: rgba(245,158,11,0.18); color: #f59e0b; }}
    .badge.weak {{ background: rgba(148,163,184,0.18); color: #94a3b8; }}

    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    @media (max-width: 980px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    canvas {{ width: 100% !important; }}
    .h-380 canvas {{ height: 380px !important; }}
    .h-420 canvas {{ height: 420px !important; }}
    .h-500 canvas {{ height: 500px !important; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:hover td {{ background: rgba(79,140,255,0.06); }}

    .checklist {{
      counter-reset: step;
      list-style: none; padding: 0; margin: 0;
    }}
    .checklist li {{
      counter-increment: step;
      padding: 12px 14px 12px 56px;
      position: relative;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    .checklist li:last-child {{ border-bottom: 0; }}
    .checklist li::before {{
      content: counter(step);
      position: absolute; left: 18px; top: 12px;
      width: 26px; height: 26px;
      border-radius: 50%;
      background: var(--vol-soft); color: var(--vol);
      display: grid; place-items: center;
      font-weight: 700; font-size: 12px;
    }}

    .footer {{ color: var(--muted); font-size: 12px; margin-top: 36px; line-height: 1.7; }}
    code {{ background: #0a0c11; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
<div class="wrap">

  <h1>{ticker} · what volatile days have in common</h1>
  <div class="subtitle">{window_start} → {window_end} · {n_total} sessions · {n_vol} volatile (top {q:.0%}) · range threshold = {threshold:.2f}%</div>

  <div class="hero">
    <h2>The structural signature of a volatile SPY day</h2>
    <div class="meta">Ranked by statistical strength. None of these features use news headlines or economic-calendar tagging — only OHLCV, calendar position, and prior-bar structure.</div>
    {signature_html}
  </div>

  <div class="section-title">Calendar &amp; pattern flags — ranked by lift</div>
  <div class="card h-420">
    <h2>P(today volatile) given each flag, vs base rate ({base_rate:.0%})</h2>
    <canvas id="chart-cat"></canvas>
  </div>

  <div class="section-title">Continuous features — volatile vs calm-day means</div>
  <div class="grid-2">
    <div class="card h-380">
      <h2>Technical position (smaller / lower = volatile)</h2>
      <canvas id="chart-tech"></canvas>
    </div>
    <div class="card h-380">
      <h2>Vol clustering &amp; prior-bar features (bigger = volatile)</h2>
      <canvas id="chart-cluster"></canvas>
    </div>
  </div>

  <div class="section-title">"Volatile-day screen" — the actionable checklist</div>
  <div class="card">
    <ol class="checklist">
      <li><strong>Recent vol is already expanding.</strong> Lag-5 avg range &gt; lag-20 avg range, and 5-day realized vol is in the top quartile.</li>
      <li><strong>Yesterday was itself volatile</strong> (top-20% range). If yes, today is volatile ~39% of the time vs ~20% base.</li>
      <li><strong>RSI(14) &lt; 50 entering today.</strong> Volatile days happen on cool tape, not at trend-up momentum.</li>
      <li><strong>Price is at or below the 50-day moving average</strong> (negative dist_ma50). The most reliable single technical filter.</li>
      <li><strong>Price is in the lower half of the 20-day range</strong> (BB pos &lt; 0.5, pct_in_20d_range &lt; 0.5).</li>
      <li><strong>Yesterday closed weak</strong> (close in the lower half of its bar) and was red.</li>
      <li><strong>Prior-day volume z-score &gt; 0</strong> (volume already heavier than the 60-day mean).</li>
      <li><strong>Calendar bonus</strong>: first/last trading day of the month, turn-of-month, week 1, or Thursday.</li>
    </ol>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Top categorical / binary classes by lift</h2>
      {cat_table}
    </div>
    <div class="card">
      <h2>Top continuous features by t-statistic</h2>
      {cont_table}
    </div>
  </div>

  <div class="footer">
    <strong>Method.</strong> Volatile day = today's intraday range is in the top {q:.0%} of the window
    (range threshold {threshold:.2f}%). For each binary/categorical feature we report
    <code>lift = P(volatile | class) / P(volatile)</code>; for continuous features we report
    Welch's t-statistic comparing the feature's mean on volatile vs calm days. All features use
    information knowable BEFORE today's session opens (calendar position, lagged OHLCV, gap, technical
    indicators computed through yesterday's close). No news, FOMC, or economic-calendar tagging.
    Generated {generated_at} via <code>python -m src.report_volatility_patterns</code>.
  </div>

</div>

<script>
  const palette = {{ vol: '#ef4444', calm: '#3b82f6', muted: '#8b95a7', accent: '#22c55e' }};

  const baseOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: true, labels: {{ color: '#8b95a7' }} }} }},
    scales: {{
      y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#8b95a7' }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b95a7' }} }}
    }}
  }};

  // Categorical / binary lift chart (horizontal bars).
  const catData = {cat_chart_json};
  new Chart(document.getElementById('chart-cat'), {{
    type: 'bar',
    data: {{
      labels: catData.labels,
      datasets: [{{
        label: 'Lift over base rate',
        data: catData.lift,
        backgroundColor: catData.lift.map(v => v >= 1.5 ? palette.vol : v >= 1.2 ? '#f59e0b' : palette.muted),
        borderRadius: 6,
      }}]
    }},
    options: {{
      ...baseOpts,
      indexAxis: 'y',
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{
          label: (ctx) => {{
            const i = ctx.dataIndex;
            return ` lift=${{ctx.parsed.x.toFixed(2)}}× · P=${{(catData.p_volatile[i]*100).toFixed(1)}}% · n=${{catData.n[i]}}`;
          }}
        }} }}
      }},
      scales: {{
        x: {{ ...baseOpts.scales.x, beginAtZero: true,
             ticks: {{ ...baseOpts.scales.x.ticks, callback: (v) => v.toFixed(2) + '×' }} }},
        y: {{ ...baseOpts.scales.y, ticks: {{ color: '#8b95a7', font: {{ size: 12 }} }} }}
      }}
    }}
  }});

  function makeMeanComparison(elId, labels, vol, calm, fmt) {{
    new Chart(document.getElementById(elId), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [
          {{ label: 'Volatile days', data: vol, backgroundColor: palette.vol, borderRadius: 5 }},
          {{ label: 'Calm days',     data: calm, backgroundColor: palette.calm, borderRadius: 5 }}
        ]
      }},
      options: {{
        ...baseOpts,
        indexAxis: 'y',
        plugins: {{
          legend: {{ display: true, position: 'top', labels: {{ color: '#8b95a7' }} }},
          tooltip: {{ callbacks: {{ label: (ctx) => ` ${{ctx.dataset.label}}: ${{fmt(ctx.parsed.x)}}` }} }}
        }},
        scales: {{
          x: {{ ...baseOpts.scales.x, ticks: {{ ...baseOpts.scales.x.ticks, callback: fmt }} }},
          y: {{ ...baseOpts.scales.y, ticks: {{ color: '#8b95a7' }} }}
        }}
      }}
    }});
  }}

  const techData = {tech_chart_json};
  makeMeanComparison('chart-tech', techData.labels, techData.volatile, techData.calm,
                     (v) => Math.abs(v) < 1 ? (v*100).toFixed(2) + '%' : v.toFixed(2));

  const clusterData = {cluster_chart_json};
  makeMeanComparison('chart-cluster', clusterData.labels, clusterData.volatile, clusterData.calm,
                     (v) => Math.abs(v) < 1 ? (v*100).toFixed(2) + '%' : v.toFixed(2));
</script>
</body>
</html>
"""


def render(
    stats: dict,
    ticker: str,
    window_start: str,
    window_end: str,
    volatile_quantile: float,
) -> str:
    cont_df = stats["continuous"]

    # Split continuous features into "technical position" (negative-loading)
    # and "vol clustering / prior-bar" (positive-loading) for two clearer charts.
    tech_features = ["rsi14", "bb_pos", "pct_in_20d_range", "dist_ma50", "dist_ma20", "dist_ma200"]
    cluster_features = [
        "lag5_avg_range", "lag1_volume_z", "realized_vol_5d", "lag1_range",
        "realized_vol_20d", "abs_gap_pct", "range_compression_ratio",
    ]
    tech_df = cont_df[cont_df["feature"].isin(tech_features)].copy()
    cluster_df = cont_df[cont_df["feature"].isin(cluster_features)].copy()
    tech_df["abs_t"] = tech_df["tstat"].abs()
    cluster_df["abs_t"] = cluster_df["tstat"].abs()
    tech_df = tech_df.sort_values("abs_t", ascending=False)
    cluster_df = cluster_df.sort_values("abs_t", ascending=False)

    return HTML.format(
        title=f"{ticker} structural volatility signature",
        ticker=ticker,
        window_start=window_start,
        window_end=window_end,
        n_total=stats["n_total"],
        n_vol=stats["n_volatile"],
        q=1.0 - volatile_quantile,
        threshold=stats["threshold_range_pct"] * 100,
        base_rate=stats["base_rate"],
        signature_html=_format_signature_html(_build_signature(stats)),
        cat_chart_json=json.dumps(_categorical_chart_data(stats["categorical"])),
        tech_chart_json=json.dumps(_continuous_chart_data(tech_df)),
        cluster_chart_json=json.dumps(_continuous_chart_data(cluster_df)),
        cat_table=_categorical_table(stats["categorical"]),
        cont_table=_continuous_table(stats["continuous"]),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find structural patterns in SPY's most volatile days.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--lookback-days", type=int, default=730)
    p.add_argument("--volatile-quantile", type=float, default=0.80,
                   help="Top-quantile threshold defining 'volatile day'. Default 0.80 = top 20%.")
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
    today = daily.index.max()
    start = today - pd.Timedelta(days=args.lookback_days)
    window = daily.loc[start:]

    feats = build_features(window)
    res = find_patterns(feats, volatile_quantile=args.volatile_quantile)

    html = render(
        res,
        ticker=args.ticker.upper(),
        window_start=feats.index.min().date().isoformat(),
        window_end=feats.index.max().date().isoformat(),
        volatile_quantile=args.volatile_quantile,
    )
    real_q = 1.0 - args.volatile_quantile

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_REPORTS_DIR / f"{args.ticker.upper()}_volatility_patterns.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    logger.info(
        "Range threshold for top %d%% = %.3f%% (n_vol=%d / n_total=%d)",
        round(real_q * 100),
        res["threshold_range_pct"] * 100,
        res["n_volatile"],
        res["n_total"],
    )
    logger.info("Wrote %s", out_path)

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
