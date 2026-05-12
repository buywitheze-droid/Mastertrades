"""Live HTML dashboard for the multi-ticker volatility scanner.

Generates a single self-contained HTML page that:

  - Ranks the universe by today's P(volatile day)
  - Color-codes each ticker by tier (EXTREME / HIGH / ELEVATED / AVERAGE / CALM)
  - Surfaces high-priority alerts at the top
  - Shows model & feature context (RSI, BB position, lag1 range, gap, vol regime)
  - Auto-refreshes the browser tab via <meta http-equiv="refresh">

Two modes:

    # Single shot (run once, open the browser)
    python -m src.report_scanner

    # Live mode (re-scan every N seconds; the page auto-refreshes to pick up updates)
    python -m src.report_scanner --watch 300

Designed to be paired with Windows Task Scheduler / cron for an end-to-end
production-style setup:

    Trigger:  Mon-Fri 09:31 ET
    Action:   <repo>/.venv/Scripts/python.exe -m src.report_scanner --no-open
    Then:     keep the report's HTML tab open in your browser all day.
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

import pandas as pd

from src.scanner import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_UNIVERSE,
    scan_universe,
    verdict_for,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "SPY_family_scanner.html"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="refresh" content="{refresh_s}" />
<title>Volatility Scanner — Live</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-2: #243047;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --pos: #22c55e;
    --neg: #ef4444;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  body {{ padding: 28px 40px 60px; }}
  header {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin-bottom: 24px; }}
  h1 {{ font-size: 28px; margin: 0; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  .stamp {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: var(--muted); }}
  .stamp .live {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #22c55e; margin-right: 8px; box-shadow: 0 0 0 0 rgba(34,197,94,0.6); animation: pulse 2s infinite; vertical-align: middle; }}
  @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }} 70% {{ box-shadow: 0 0 0 10px rgba(34,197,94,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }} }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }}
  .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .stat .value {{ font-size: 26px; font-weight: 600; margin-top: 6px; }}
  .stat .small {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  .alerts {{ margin-bottom: 28px; }}
  .alert-card {{ display: flex; align-items: center; gap: 16px; background: var(--panel); border: 1px solid var(--border); border-left-width: 6px; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }}
  .alert-card .ticker {{ font-size: 22px; font-weight: 700; min-width: 80px; }}
  .alert-card .pvol {{ font-size: 22px; font-weight: 700; min-width: 90px; }}
  .alert-card .why {{ color: var(--muted); font-size: 13px; }}
  .alert-card.empty {{ border-left-color: #334155; color: var(--muted); justify-content: center; padding: 18px; font-style: italic; }}

  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
  thead th {{ background: var(--panel-2); text-align: left; padding: 12px 14px; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); font-size: 14px; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(56,189,248,0.05); }}
  td.ticker {{ font-weight: 700; font-size: 16px; }}
  td.numeric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.pos {{ color: var(--pos); }}
  td.neg {{ color: var(--neg); }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #fff; }}
  .pvol-cell {{ font-weight: 700; }}
  .lift-cell {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }}
  .errors {{ margin-top: 24px; padding: 14px 18px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; color: #fca5a5; font-size: 13px; }}
  .errors h3 {{ margin: 0 0 8px; font-size: 14px; }}

  footer {{ margin-top: 36px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  footer code {{ background: var(--panel); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
  footer .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0 20px; }}
  footer .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  footer .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Volatility Scanner — Live</h1>
    <div class="subtitle">P(today is in top 20% of intraday range) · {n_universe} symbols · per-ticker logistic-regression model</div>
  </div>
  <div class="stamp">
    <span class="live"></span>Updated {ts}
    &nbsp;·&nbsp; auto-refresh every {refresh_s}s
  </div>
</header>

<section class="summary-grid">
  <div class="stat">
    <div class="label">As of session</div>
    <div class="value">{as_of_str}</div>
    <div class="small">latest bar across the universe</div>
  </div>
  <div class="stat">
    <div class="label">High-priority signals</div>
    <div class="value" style="color:{n_alert_color}">{n_alerts}</div>
    <div class="small">tier ELEVATED or above</div>
  </div>
  <div class="stat">
    <div class="label">Average lift</div>
    <div class="value">{avg_lift:.2f}x</div>
    <div class="small">universe-wide P(vol) / base-rate</div>
  </div>
  <div class="stat">
    <div class="label">Top score</div>
    <div class="value">{top_ticker} · {top_p_vol:.1f}%</div>
    <div class="small">{top_verdict}</div>
  </div>
</section>

<section class="alerts">
  <h2 style="font-size:18px; margin:0 0 12px; color:var(--muted); letter-spacing:1px; text-transform:uppercase;">Priority alerts</h2>
  {alerts_html}
</section>

<section>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Ticker</th>
        <th>P(volatile)</th>
        <th>Lift</th>
        <th>Verdict</th>
        <th>Last</th>
        <th>%Δ</th>
        <th>RSI(14)</th>
        <th>BB pos</th>
        <th>Lag-1 range</th>
        <th>Compression</th>
        <th>Gap %</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</section>

{errors_html}

<footer>
  <div class="legend">
    {legend_html}
  </div>
  <div>
    <strong>How to read:</strong>
    <code>P(volatile)</code> is the model's probability that today's intraday range will land in the top quintile of this ticker's history.
    <code>Lift</code> = P(volatile) / base-rate; lift &gt; 2 means the setup is in the top decile of historical conditions.
    <code>Compression</code> = 5-day avg range / 60-day avg range; values &lt; 1 (squeeze) precede many vol expansions.
  </div>
  <div style="margin-top:8px;">
    <strong>Data:</strong> Yahoo Finance daily bars (~15min delayed during RTH). Models cached at <code>models/&lt;TICKER&gt;_logreg.joblib</code>, retrained weekly. Re-render this page with <code>python -m src.report_scanner --watch 300</code>.
  </div>
  <div style="margin-top:8px; color:#7f8c8d;">
    For real-time tick data swap yfinance for Alpaca / Polygon. For execution, point your broker API at the same signal stream.
  </div>
</footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _row_html(rank: int, r: pd.Series) -> str:
    label, color, _ = verdict_for(r["p_vol"], r["base_rate"])
    pct_class = "pos" if r["pct_change"] >= 0 else "neg"
    pct_sign = "+" if r["pct_change"] >= 0 else ""
    return f"""
      <tr>
        <td class="numeric">#{rank}</td>
        <td class="ticker">{escape(r['ticker'])}</td>
        <td class="pvol-cell numeric" style="color:{color}">{r['p_vol'] * 100:.1f}%</td>
        <td class="lift-cell">{r['lift']:.2f}x</td>
        <td><span class="badge" style="background:{color}">{escape(label)}</span></td>
        <td class="numeric">${r['last_close']:.2f}</td>
        <td class="numeric {pct_class}">{pct_sign}{r['pct_change'] * 100:.2f}%</td>
        <td class="numeric">{r['rsi14']:.0f}</td>
        <td class="numeric">{r['bb_pos']:.2f}</td>
        <td class="numeric">{r['lag1_range'] * 100:.2f}%</td>
        <td class="numeric">{r['range_compression']:.2f}</td>
        <td class="numeric">{r['abs_gap_pct'] * 100:.2f}%</td>
      </tr>
    """


def _alerts_html(df: pd.DataFrame) -> tuple[str, int]:
    alerts: list[str] = []
    n_priority = 0
    for _, r in df.iterrows():
        label, color, blurb = verdict_for(r["p_vol"], r["base_rate"])
        if label not in ("EXTREME", "HIGH", "ELEVATED"):
            continue
        n_priority += 1
        alerts.append(
            f"""
            <div class="alert-card" style="border-left-color:{color}">
              <div class="ticker">{escape(r['ticker'])}</div>
              <div class="pvol" style="color:{color}">{r['p_vol'] * 100:.1f}%</div>
              <div>
                <span class="badge" style="background:{color}">{escape(label)}</span>
                &nbsp;·&nbsp; lift {r['lift']:.2f}x
                &nbsp;·&nbsp; <span class="why">{escape(blurb)}</span>
              </div>
            </div>
            """
        )
    if not alerts:
        return ('<div class="alert-card empty">No high-priority signals — the universe is calm today.</div>', 0)
    return ("\n".join(alerts), n_priority)


def _legend_html() -> str:
    from src.scanner import VERDICT_TIERS

    parts = []
    for _cutoff, label, color, _blurb in VERDICT_TIERS:
        parts.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{escape(label)}</div>'
        )
    return "".join(parts)


def render_html(df: pd.DataFrame, errors: list[dict], refresh_s: int) -> str:
    if df.empty:
        rows_html = '<tr><td colspan="12" style="text-align:center; padding:30px; color:var(--muted);">No tickers scored.</td></tr>'
        as_of_str = "—"
        avg_lift = 0.0
        top_ticker = "—"
        top_p_vol = 0.0
        top_verdict = "—"
        n_universe = 0
    else:
        rows_html = "\n".join(_row_html(i + 1, r) for i, r in df.iterrows())
        as_of_str = pd.Timestamp(df["as_of"].iloc[0]).strftime("%a %b %d, %Y") if not df.empty else "—"
        avg_lift = float(df["lift"].mean())
        top = df.iloc[0]
        top_ticker = str(top["ticker"])
        top_p_vol = float(top["p_vol"]) * 100.0
        top_verdict = verdict_for(top["p_vol"], top["base_rate"])[0]
        n_universe = len(df)

    alerts_html, n_alerts = _alerts_html(df) if not df.empty else ("", 0)
    errors_html = ""
    if errors:
        items = "".join(
            f'<li><strong>{escape(e["ticker"])}</strong>: {escape(str(e["error"]))}</li>' for e in errors
        )
        errors_html = f'<div class="errors"><h3>Skipped tickers</h3><ul>{items}</ul></div>'

    return HTML_TEMPLATE.format(
        refresh_s=refresh_s,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        as_of_str=as_of_str,
        n_universe=n_universe,
        n_alerts=n_alerts,
        n_alert_color="#22c55e" if n_alerts == 0 else "#f59e0b",
        avg_lift=avg_lift,
        top_ticker=top_ticker,
        top_p_vol=top_p_vol,
        top_verdict=top_verdict,
        alerts_html=alerts_html,
        rows_html=rows_html,
        errors_html=errors_html,
        legend_html=_legend_html(),
    )


def write_sidecar_json(df: pd.DataFrame, errors: list[dict], path: Path) -> None:
    """Machine-readable mirror of the report — useful for downstream bots."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe": df["ticker"].tolist() if not df.empty else [],
        "signals": (df.assign(as_of=lambda d: d["as_of"].astype(str))
                      .to_dict(orient="records")) if not df.empty else [],
        "errors": errors,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the live volatility scanner dashboard.")
    p.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE))
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output HTML path.")
    p.add_argument("--watch", type=int, default=0, help="Re-scan every N seconds (0 = single run).")
    p.add_argument("--refresh-page", type=int, default=120, help="Browser meta-refresh interval in seconds.")
    p.add_argument("--no-fetch", action="store_true", help="Use cached data only.")
    p.add_argument("--retrain", action="store_true", help="Force a fresh model train.")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the browser.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _run_once(
    tickers: list[str],
    refresh_data: bool,
    retrain: bool,
    out_path: Path,
    refresh_page: int,
    data_dir: Path,
    model_dir: Path,
) -> pd.DataFrame:
    df, errors = scan_universe(
        tickers=tickers,
        refresh_data=refresh_data,
        retrain=retrain,
        data_dir=data_dir,
        model_dir=model_dir,
    )
    html = render_html(df, errors, refresh_s=refresh_page)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    write_sidecar_json(df, errors, out_path.with_suffix(".json"))
    return df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    out_path = Path(args.out)

    df = _run_once(
        tickers=tickers,
        refresh_data=not args.no_fetch,
        retrain=args.retrain,
        out_path=out_path,
        refresh_page=args.refresh_page,
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
    )
    if df.empty:
        print("No tickers scored — check log output above.")
        return 1

    print(f"Wrote {out_path}")
    print(f"Top: {df.iloc[0]['ticker']} P(vol)={df.iloc[0]['p_vol']:.1%} lift={df.iloc[0]['lift']:.2f}x")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    if args.watch > 0:
        print(f"Watch mode: re-scanning every {args.watch}s (Ctrl-C to stop)")
        try:
            while True:
                time.sleep(args.watch)
                df = _run_once(
                    tickers=tickers,
                    refresh_data=True,
                    retrain=False,
                    out_path=out_path,
                    refresh_page=args.refresh_page,
                    data_dir=Path(args.data_dir),
                    model_dir=Path(args.model_dir),
                )
                if not df.empty:
                    print(f"[{datetime.now():%H:%M:%S}] re-scanned, top: {df.iloc[0]['ticker']} {df.iloc[0]['p_vol']:.1%}")
        except KeyboardInterrupt:
            print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
