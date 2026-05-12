"""HTML report: today's volatility-probability score + walk-forward backtest.

Trains both a logistic regression and a gradient-boosting classifier, runs an
expanding-window walk-forward over the full SPY daily history, and emits a
single self-contained HTML page summarizing:

- Today's predicted P(top-quintile range day) from each model
- Backtest metrics (AUC, average precision, Brier, top-decile/quintile lift)
- Reliability / calibration diagram
- Score-vs-time chart with realized volatile days marked
- Top driver features by standardized coefficient (LogReg)
- Recent 20 sessions table: score, actual outcome, hit/miss

Usage::

    python -m src.report_volatility_classifier
    python -m src.report_volatility_classifier --volatile-quantile 0.85
    python -m src.report_volatility_classifier --no-open

The walk-forward retrains every 21 sessions and takes ~3 minutes on SPY's
full history (the GBM is the slow part).
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
from .volatility_patterns import build_features
from .volatility_classifier import (
    BacktestSummary,
    evaluate_backtest,
    feature_importances,
    fit_full,
    make_gbm,
    make_logreg,
    prepare_xy,
    score_dataframe,
    walk_forward_proba,
)


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

logger = logging.getLogger("report_volatility_classifier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _verdict(prob: float, base_rate: float) -> tuple[str, str]:
    """Return (label, css class) for a given probability."""
    if pd.isna(prob):
        return "n/a", "neutral"
    lift = prob / base_rate if base_rate > 0 else 0.0
    if lift >= 2.5:
        return "Very high", "vol"
    if lift >= 1.75:
        return "Elevated", "warn"
    if lift >= 1.0:
        return "Around average", "neutral"
    if lift >= 0.5:
        return "Below average", "calm"
    return "Very calm", "calm"


def _backtest_table(s_lr: BacktestSummary, s_gbm: BacktestSummary, base_rate: float) -> str:
    rows = [
        ("Sessions evaluated",                    f"{s_lr.n:,}",                       f"{s_gbm.n:,}"),
        ("Base rate (volatile)",                  _fmt_pct(s_lr.base_rate),            _fmt_pct(s_gbm.base_rate)),
        ("AUC-ROC",                               f"{s_lr.auc:.3f}",                   f"{s_gbm.auc:.3f}"),
        ("Avg precision (PR-AUC)",                f"{s_lr.avg_precision:.3f}",         f"{s_gbm.avg_precision:.3f}"),
        ("Brier score (lower is better)",         f"{s_lr.brier:.3f}",                 f"{s_gbm.brier:.3f}"),
        ("Top-decile precision",                  _fmt_pct(s_lr.top_decile_precision), _fmt_pct(s_gbm.top_decile_precision)),
        ("Top-decile recall",                     _fmt_pct(s_lr.top_decile_recall),    _fmt_pct(s_gbm.top_decile_recall)),
        ("Top-decile lift",                       f"{s_lr.top_decile_lift:.2f}×",      f"{s_gbm.top_decile_lift:.2f}×"),
        ("Top-quintile precision",                _fmt_pct(s_lr.top_quintile_precision),_fmt_pct(s_gbm.top_quintile_precision)),
        ("Top-quintile recall",                   _fmt_pct(s_lr.top_quintile_recall),  _fmt_pct(s_gbm.top_quintile_recall)),
        ("Top-quintile lift",                     f"{s_lr.top_quintile_lift:.2f}×",    f"{s_gbm.top_quintile_lift:.2f}×"),
    ]
    body = "".join(
        f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>" for label, a, b in rows
    )
    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Logistic regression</th><th>Gradient boosting</th>"
        "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def _coef_table(imp: pd.DataFrame, top_n: int = 18) -> str:
    rows = []
    for _, r in imp.head(top_n).iterrows():
        cls = "pos" if r["weight"] > 0 else "neg"
        rows.append(
            f"<tr><td>{r['feature']}</td>"
            f"<td class='{cls}'>{r['weight']:+.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Feature</th><th>Standardized coef</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _recent_table(preds: pd.DataFrame, n: int = 20) -> str:
    recent = preds.tail(n).iloc[::-1]
    rows = []
    for d, r in recent.iterrows():
        actual = "Volatile" if r["y_true"] == 1 else "Calm"
        actual_cls = "vol" if r["y_true"] == 1 else "calm"
        score_pct = r["y_score"] * 100
        rows.append(
            f"<tr><td>{d.strftime('%Y-%m-%d')}</td>"
            f"<td>{score_pct:.1f}%</td>"
            f"<td class='{actual_cls}'>{actual}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Date</th><th>P(volatile) — LogReg</th><th>Actual</th>"
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
      --warn: #f59e0b;
      --warn-soft: rgba(245,158,11,0.18);
      --calm: #3b82f6;
      --calm-soft: rgba(59,130,246,0.18);
      --neutral: #94a3b8;
      --neutral-soft: rgba(148,163,184,0.18);
      --pos: #22c55e;
      --neg: #ef4444;
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

    .scores {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-bottom: 28px; }}
    @media (max-width: 880px) {{ .scores {{ grid-template-columns: 1fr; }} }}
    .score-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 26px 30px;
    }}
    .score-card.vol     {{ border-left: 4px solid var(--vol);     background: linear-gradient(135deg, var(--vol-soft),     transparent), var(--card); }}
    .score-card.warn    {{ border-left: 4px solid var(--warn);    background: linear-gradient(135deg, var(--warn-soft),    transparent), var(--card); }}
    .score-card.neutral {{ border-left: 4px solid var(--neutral); background: linear-gradient(135deg, var(--neutral-soft), transparent), var(--card); }}
    .score-card.calm    {{ border-left: 4px solid var(--calm);    background: linear-gradient(135deg, var(--calm-soft),    transparent), var(--card); }}
    .score-label {{ color: var(--muted); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; }}
    .score-value {{ font-size: 64px; font-weight: 800; line-height: 1.05; margin: 8px 0 6px; }}
    .score-card.vol  .score-value     {{ color: var(--vol); }}
    .score-card.warn .score-value     {{ color: var(--warn); }}
    .score-card.neutral .score-value  {{ color: var(--text); }}
    .score-card.calm .score-value     {{ color: var(--calm); }}
    .score-meta {{ color: var(--muted); font-size: 13px; }}
    .score-meta .verdict {{ color: var(--text); font-weight: 600; }}
    .score-meta .lift {{ font-weight: 600; }}

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
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    @media (max-width: 980px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    canvas {{ width: 100% !important; }}
    .h-280 canvas {{ height: 280px !important; }}
    .h-340 canvas {{ height: 340px !important; }}
    .h-380 canvas {{ height: 380px !important; }}
    .h-420 canvas {{ height: 420px !important; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tbody tr:hover td {{ background: rgba(79,140,255,0.06); }}
    td.pos {{ color: var(--pos); }}
    td.neg {{ color: var(--neg); }}
    td.vol {{ color: var(--vol); font-weight: 600; }}
    td.calm {{ color: var(--calm); }}

    .footer {{ color: var(--muted); font-size: 12px; margin-top: 36px; line-height: 1.7; }}
    code {{ background: #0a0c11; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
<div class="wrap">

  <h1>{ticker} · volatility probability score</h1>
  <div class="subtitle">
    Daily history through {as_of} · base rate (top {q:.0%} of range) = {base_rate} · range threshold = {threshold}
  </div>

  <div class="scores">
    <div class="score-card {lr_class}">
      <div class="score-label">Today's score · Logistic regression</div>
      <div class="score-value">{lr_pct}</div>
      <div class="score-meta">
        Verdict: <span class="verdict">{lr_verdict}</span>
        &nbsp;·&nbsp; Lift over base: <span class="lift">{lr_lift:.2f}×</span>
      </div>
    </div>
    <div class="score-card {gbm_class}">
      <div class="score-label">Today's score · Gradient boosting</div>
      <div class="score-value">{gbm_pct}</div>
      <div class="score-meta">
        Verdict: <span class="verdict">{gbm_verdict}</span>
        &nbsp;·&nbsp; Lift over base: <span class="lift">{gbm_lift:.2f}×</span>
      </div>
    </div>
  </div>

  <div class="section-title">Walk-forward backtest (full SPY history, expanding window, refit every 21 days)</div>
  <div class="card">
    <h2>Side-by-side metrics</h2>
    {backtest_table}
  </div>

  <div class="grid-2">
    <div class="card h-340">
      <h2>Reliability diagram (calibration) — LogReg</h2>
      <canvas id="chart-calib-lr"></canvas>
    </div>
    <div class="card h-340">
      <h2>Reliability diagram — Gradient boosting</h2>
      <canvas id="chart-calib-gbm"></canvas>
    </div>
  </div>

  <div class="section-title">Score over time (LogReg) with realized volatile days marked</div>
  <div class="card h-380">
    <h2>Predicted P(volatile) per session — red dots = days that actually were volatile</h2>
    <canvas id="chart-history"></canvas>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Top driver features (LogReg standardized coefficients)</h2>
      {coef_table}
    </div>
    <div class="card">
      <h2>Last 20 sessions — score &amp; outcome</h2>
      {recent_table}
    </div>
  </div>

  <div class="footer">
    <strong>Method.</strong> Target = today's intraday range in the top {q:.0%} of the full window
    (range threshold {threshold}). Features computed exclusively from data
    knowable BEFORE today's session opens (calendar position, lagged OHLCV, gap, technical
    indicators through prior close). Walk-forward: expanding window, initial training of 1,000 sessions,
    retrain every 21 sessions, predictions never see future data. Logistic regression standardized;
    gradient boosting via <code>HistGradientBoostingClassifier</code> (max_iter=300, lr=0.05, max_depth=4).
    Generated {generated_at} via <code>python -m src.report_volatility_classifier</code>.
  </div>

</div>

<script>
  const baseOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#8b95a7' }} }} }},
    scales: {{
      y: {{ grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#8b95a7' }} }},
      x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#8b95a7' }} }}
    }}
  }};

  function makeReliability(elId, calData, baseRate) {{
    const points = calData.map(b => ({{ x: b.mean_predicted * 100, y: b.actual_rate * 100, n: b.n }}));
    new Chart(document.getElementById(elId), {{
      type: 'scatter',
      data: {{
        datasets: [
          {{
            label: 'Bin (mean predicted vs actual)',
            data: points,
            backgroundColor: '#4f8cff',
            pointRadius: 6, pointHoverRadius: 8,
            showLine: true, borderColor: 'rgba(79,140,255,0.5)', borderWidth: 1.5,
          }},
          {{
            label: 'Perfect calibration',
            data: [{{ x: 0, y: 0 }}, {{ x: 100, y: 100 }}],
            borderColor: '#94a3b8', borderDash: [4, 4], borderWidth: 1.5,
            pointRadius: 0, showLine: true, fill: false,
          }},
          {{
            label: 'Base rate',
            data: [{{ x: 0, y: baseRate * 100 }}, {{ x: 100, y: baseRate * 100 }}],
            borderColor: 'rgba(245,158,11,0.6)', borderDash: [2, 4], borderWidth: 1,
            pointRadius: 0, showLine: true, fill: false,
          }}
        ]
      }},
      options: {{
        ...baseOpts,
        plugins: {{
          ...baseOpts.plugins,
          tooltip: {{ callbacks: {{
            label: (ctx) => {{
              if (ctx.dataset.label !== 'Bin (mean predicted vs actual)') return ctx.dataset.label;
              return ` predicted=${{ctx.parsed.x.toFixed(1)}}%, actual=${{ctx.parsed.y.toFixed(1)}}%, n=${{ctx.raw.n}}`;
            }}
          }} }}
        }},
        scales: {{
          x: {{ ...baseOpts.scales.x, title: {{ display: true, text: 'Mean predicted P(volatile) %', color: '#8b95a7' }}, min: 0, max: 100 }},
          y: {{ ...baseOpts.scales.y, title: {{ display: true, text: 'Actual rate %',                color: '#8b95a7' }}, min: 0, max: 100 }}
        }}
      }}
    }});
  }}

  makeReliability('chart-calib-lr',  {calib_lr_json},  {base_rate_num});
  makeReliability('chart-calib-gbm', {calib_gbm_json}, {base_rate_num});

  // Score-over-time chart with hits marked.
  const histDates = {hist_dates_json};
  const histScores = {hist_scores_json};
  const hitDates = {hit_dates_json};
  const hitScores = {hit_scores_json};

  new Chart(document.getElementById('chart-history'), {{
    type: 'line',
    data: {{
      labels: histDates,
      datasets: [
        {{
          label: 'P(volatile) — LogReg',
          data: histScores,
          borderColor: '#4f8cff',
          borderWidth: 1.0,
          pointRadius: 0,
          tension: 0.15,
          fill: false,
        }},
        {{
          label: 'Actual volatile days',
          data: hitDates.map((d, i) => ({{ x: d, y: hitScores[i] }})),
          backgroundColor: 'rgba(239,68,68,0.85)',
          borderColor: 'rgba(239,68,68,0.85)',
          pointRadius: 2.4,
          showLine: false,
          parsing: false,
        }}
      ]
    }},
    options: {{
      ...baseOpts,
      plugins: {{
        ...baseOpts.plugins,
        tooltip: {{ callbacks: {{ label: (ctx) => ` ${{(ctx.parsed.y*100).toFixed(1)}}% on ${{ctx.label || ctx.parsed.x}}` }} }}
      }},
      scales: {{
        x: {{ ...baseOpts.scales.x, ticks: {{ ...baseOpts.scales.x.ticks, maxTicksLimit: 12 }} }},
        y: {{ ...baseOpts.scales.y, min: 0, max: 1,
             ticks: {{ ...baseOpts.scales.y.ticks, callback: (v) => (v*100).toFixed(0) + '%' }} }}
      }}
    }}
  }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(
    daily: pd.DataFrame,
    volatile_quantile: float,
    min_train: int,
    step: int,
    history_tail: int = 1500,
) -> dict:
    feats = build_features(daily)
    X, y, threshold = prepare_xy(feats, volatile_quantile=volatile_quantile)

    logger.info("Walking forward LR (n=%d, min_train=%d, step=%d)...", len(X), min_train, step)
    preds_lr = walk_forward_proba(X, y, make_logreg, min_train=min_train, step=step)
    logger.info("Walking forward GBM...")
    preds_gbm = walk_forward_proba(X, y, make_gbm, min_train=min_train, step=step)

    s_lr = evaluate_backtest(preds_lr)
    s_gbm = evaluate_backtest(preds_gbm)

    final_lr = fit_full(X, y, make_logreg)
    final_gbm = fit_full(X, y, make_gbm)
    last_row = X.tail(1)
    today_lr = float(score_dataframe(final_lr, last_row).iloc[0])
    today_gbm = float(score_dataframe(final_gbm, last_row).iloc[0])
    coef_df = feature_importances(final_lr, X.columns)

    return {
        "X": X, "y": y, "threshold": threshold,
        "preds_lr": preds_lr, "preds_gbm": preds_gbm,
        "summary_lr": s_lr, "summary_gbm": s_gbm,
        "today_lr": today_lr, "today_gbm": today_gbm,
        "today_date": last_row.index[0],
        "coef_df": coef_df,
        "history_tail": history_tail,
    }


def render(stats: dict, ticker: str, volatile_quantile: float) -> str:
    s_lr: BacktestSummary = stats["summary_lr"]
    s_gbm: BacktestSummary = stats["summary_gbm"]
    base_rate = s_lr.base_rate

    lr_verdict, lr_class = _verdict(stats["today_lr"], base_rate)
    gbm_verdict, gbm_class = _verdict(stats["today_gbm"], base_rate)

    preds_lr = stats["preds_lr"].tail(stats["history_tail"])
    hit_idx = preds_lr.index[preds_lr["y_true"] == 1]
    hit_scores = preds_lr.loc[hit_idx, "y_score"]

    return HTML.format(
        title=f"{ticker} volatility probability",
        ticker=ticker,
        as_of=stats["today_date"].strftime("%Y-%m-%d"),
        q=1.0 - volatile_quantile,
        threshold=_fmt_pct(stats["threshold"]),
        base_rate=_fmt_pct(base_rate),
        base_rate_num=round(base_rate, 4),
        lr_class=lr_class,
        lr_pct=_fmt_pct(stats["today_lr"], digits=1),
        lr_verdict=lr_verdict,
        lr_lift=stats["today_lr"] / base_rate if base_rate > 0 else 0.0,
        gbm_class=gbm_class,
        gbm_pct=_fmt_pct(stats["today_gbm"], digits=1),
        gbm_verdict=gbm_verdict,
        gbm_lift=stats["today_gbm"] / base_rate if base_rate > 0 else 0.0,
        backtest_table=_backtest_table(s_lr, s_gbm, base_rate),
        coef_table=_coef_table(stats["coef_df"]),
        recent_table=_recent_table(stats["preds_lr"]),
        calib_lr_json=json.dumps(s_lr.calibration),
        calib_gbm_json=json.dumps(s_gbm.calibration),
        hist_dates_json=json.dumps([d.strftime("%Y-%m-%d") for d in preds_lr.index]),
        hist_scores_json=json.dumps([round(float(x), 4) for x in preds_lr["y_score"]]),
        hit_dates_json=json.dumps([d.strftime("%Y-%m-%d") for d in hit_idx]),
        hit_scores_json=json.dumps([round(float(x), 4) for x in hit_scores]),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + backtest a volatility-day classifier and render an HTML report.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--volatile-quantile", type=float, default=0.80)
    p.add_argument("--min-train", type=int, default=1000)
    p.add_argument("--step", type=int, default=21)
    p.add_argument("--history-tail", type=int, default=1500,
                   help="How many recent walk-forward days to plot (default 1500 ≈ 6y).")
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
    stats = run(
        daily,
        volatile_quantile=args.volatile_quantile,
        min_train=args.min_train,
        step=args.step,
        history_tail=args.history_tail,
    )
    html = render(stats, ticker=args.ticker.upper(), volatile_quantile=args.volatile_quantile)

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_REPORTS_DIR / f"{args.ticker.upper()}_volatility_probability.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    s_lr = stats["summary_lr"]
    s_gbm = stats["summary_gbm"]
    logger.info(
        "Today's score (LogReg) = %s  ·  (GBM) = %s",
        _fmt_pct(stats["today_lr"], digits=1),
        _fmt_pct(stats["today_gbm"], digits=1),
    )
    logger.info(
        "AUC: LR=%.3f, GBM=%.3f  ·  Top-decile precision: LR=%s GBM=%s (lift %.2fx / %.2fx)",
        s_lr.auc, s_gbm.auc,
        _fmt_pct(s_lr.top_decile_precision), _fmt_pct(s_gbm.top_decile_precision),
        s_lr.top_decile_lift, s_gbm.top_decile_lift,
    )
    logger.info("Wrote %s", out_path)

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
