"""HTML report: $500 → $5,000 strategy Monte Carlo on the volatility model.

Uses the walk-forward predictions from the volatility classifier (so we never
peek at future prices when choosing a trade), models 0DTE iron-condor and
straddle P&L from actual SPY OHLC, and bootstraps thousands of forward paths
to estimate the distribution of outcomes.

Run::

    python -m src.report_strategy_sim                         # full backtest
    python -m src.report_strategy_sim --sims 5000 --no-open
    python -m src.report_strategy_sim --start-equity 1000 --target 10000
    python -m src.report_strategy_sim --risk-frac 0.10 --trades-per-day 5

The output HTML is written to reports/SPY_strategy_sim.html and auto-opened.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import load_history
from src.strategy_sim import (
    MCConfig,
    MCSummary,
    StrategyConfig,
    compute_per_day_returns,
    compute_trade_returns,
    monte_carlo,
    trade_stats,
)
from src.volatility_classifier import (
    make_logreg,
    prepare_xy,
    walk_forward_proba,
)
from src.volatility_patterns import build_features


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "SPY_strategy_sim.html"

logger = logging.getLogger("strategy_sim")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_backtests(
    spy: pd.DataFrame,
    p_vol_oos: pd.Series,
    n_years: float,
    mc_cfg: MCConfig,
) -> list[dict]:
    """Run several strategy variants and return a list of result dicts."""
    variants: list[tuple[str, str, StrategyConfig]] = [
        (
            "calm_sell_strict",
            "MOST SECURE — iron condor, very-calm only (P_vol < 0.08)",
            StrategyConfig(strategy="calm_sell", calm_threshold=0.08),
        ),
        (
            "calm_sell_lenient",
            "CALM iron condor — broader filter (P_vol < 0.14)",
            StrategyConfig(strategy="calm_sell", calm_threshold=0.14),
        ),
        (
            "hot_buy_strict",
            "MOONSHOT — long straddle on hot days (P_vol > 0.45)",
            StrategyConfig(strategy="hot_buy", hot_threshold=0.45),
        ),
        (
            "combined",
            "COMBINED — sell IC on calm + buy straddle on hot",
            StrategyConfig(strategy="combined", calm_threshold=0.10, hot_threshold=0.45),
        ),
    ]

    results: list[dict] = []
    n_total_days = len(spy.index.intersection(p_vol_oos.index))
    for key, label, cfg in variants:
        per_day = compute_per_day_returns(spy, p_vol_oos, cfg)
        trades = per_day[per_day["side"] != "NONE"]
        if trades.empty:
            logger.warning("Strategy %s produced 0 trades — skipping", key)
            continue

        ts = trade_stats(trades["ret"], n_years=n_years)
        signal_density = len(trades) / n_total_days if n_total_days > 0 else 0.0
        mc = monte_carlo(per_day["ret"], signal_density=signal_density, cfg=mc_cfg)

        results.append({
            "key": key,
            "label": label,
            "config": cfg,
            "stats": ts,
            "trades": trades,
            "mc": mc,
            "signal_density": float(signal_density),
        })

    return results


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>$500 → ${target_int} strategy simulation</title>
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
    --warn: #f59e0b;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  body {{ padding: 28px 40px 60px; max-width: 1400px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  h1 {{ font-size: 28px; margin: 0; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  h2 {{ font-size: 20px; margin: 32px 0 12px; }}
  h3 {{ font-size: 16px; margin: 0 0 6px; color: var(--muted); }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 16px 0 24px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
  .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .stat .small {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  .verdict {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin: 8px 0 22px; line-height: 1.55; font-size: 15px; }}
  .verdict.win {{ border-left: 6px solid var(--pos); }}
  .verdict.warn {{ border-left: 6px solid var(--warn); }}
  .verdict.bust {{ border-left: 6px solid var(--neg); }}

  .strategy-block {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin: 18px 0; }}
  .strategy-head {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }}
  .strategy-head .name {{ font-size: 18px; font-weight: 700; }}
  .strategy-head .density {{ color: var(--muted); font-size: 12px; }}

  .row {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; margin-top: 14px; }}
  @media (max-width: 980px) {{ .row {{ grid-template-columns: 1fr; }} }}
  .panel {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px 10px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--border); }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }}
  td.numeric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.pos {{ color: var(--pos); }}
  td.neg {{ color: var(--neg); }}

  canvas {{ background: #0b1220; border-radius: 8px; padding: 4px; }}

  footer {{ margin-top: 36px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  footer code {{ background: var(--panel); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <h1>${start_int} → ${target_int} — Strategy Monte Carlo</h1>
  <div class="subtitle">
    SPY · {n_years:.1f} yrs walk-forward · {n_sims:,} bootstrap paths · horizon {horizon} trading days · risk {risk_pct:.1f}% / trade × {trades_per_day} trades/day
  </div>
</header>

<div class="verdict {verdict_cls}">
  <strong>Bottom line:</strong> {verdict_text}
</div>

<section>
  <h2>Strategy comparison</h2>
  <div class="panel" style="overflow-x:auto;">
    <table>
      <thead><tr>
        <th>Strategy</th>
        <th>Trades / yr</th>
        <th>Win rate</th>
        <th>Avg win</th>
        <th>Avg loss</th>
        <th>EV / trade</th>
        <th>P(hit ${target_int}) in {horizon}d</th>
        <th>P(ruined)</th>
        <th>Median final</th>
        <th>Median time-to-${target_int}</th>
      </tr></thead>
      <tbody>
        {compare_rows}
      </tbody>
    </table>
  </div>
</section>

{strategy_blocks}

<footer>
  <p><strong style="color:#f59e0b">⚠ MODELING CAVEATS — read before trading.</strong>
     This simulation uses synthetic option prices derived from SPY's actual OHLC, not real historical option chains.
     Iron-condor model: short strikes ±0.7% from open, long wings 1.0% past short, credit ≈ 13% of wing, 6% drag for slippage + commissions. Wing-touch intraday counts as max loss.
     Straddle model: ATM, premium ≈ 1.1% of spot, payoff = max(|close − open|, 0.5 × intraday-max-excursion), 6% drag. Both held to close.
     <strong>Real-world haircut estimate: 30–50% of backtest edge is lost to bid-ask spread, IV crush at the open, partial fills, pin risk, and emotional execution errors.</strong>
     Adjust the median final values DOWN by 50–80% to get a realistic personal expectation.</p>
  <p><strong>Sizing.</strong> Each trade risks <code>{risk_pct:.1f}%</code> of current equity. With {trades_per_day} trades/day, the effective daily risk is <code>{daily_risk_pct:.1f}%</code> per signal day. Paths that fall below <code>${floor_int}</code> are flagged "ruined."</p>
  <p><strong>Walk-forward note.</strong> Volatility scores are out-of-sample (model never trained on the day it scored). Each Monte Carlo path is a contiguous slice of historical per-day returns starting at a random date — this preserves real vol-regime clustering (2008, 2018, 2020) so streaks and drawdowns are realistic.</p>
  <p><strong>PDT &amp; account.</strong> 5 day-trades/day on a $500 account requires either (a) a cash account with cash-settled index options like <code>XSP</code>/<code>SPX</code>, (b) micro futures (<code>MES</code>, <code>M2K</code>), or (c) margin &gt; $25k. SPY 0DTE on a sub-$25k margin account is limited to 3 day-trades per 5 rolling business days.</p>
  <p><strong>What this report can and can't say.</strong> It CAN show you the <em>relative</em> ranking of strategies by risk-adjusted edge. It CAN show that the model genuinely picks calm days more reliably than chance. It CAN'T promise live P&amp;L will resemble these curves — that depends on your execution and how stable the model edge is going forward.</p>
</footer>

<script>
  const PALETTE = {{ med: '#38bdf8', band25: 'rgba(56,189,248,0.30)', band05: 'rgba(56,189,248,0.12)', target: '#22c55e', start: '#94a3b8' }};

  function drawEquity(canvasId, payload) {{
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const labels = payload.median.map((_, i) => i);
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: '95th pct', data: payload.p95, borderColor: 'transparent', backgroundColor: PALETTE.band05, fill: 1, pointRadius: 0, tension: 0.1 }},
          {{ label: '75th pct', data: payload.p75, borderColor: 'transparent', backgroundColor: PALETTE.band25, fill: 2, pointRadius: 0, tension: 0.1 }},
          {{ label: '25th pct', data: payload.p25, borderColor: 'transparent', backgroundColor: PALETTE.band25, fill: 3, pointRadius: 0, tension: 0.1 }},
          {{ label: '5th pct',  data: payload.p05, borderColor: 'transparent', backgroundColor: PALETTE.band05, fill: false, pointRadius: 0, tension: 0.1 }},
          {{ label: 'median',   data: payload.median, borderColor: PALETTE.med, borderWidth: 2, fill: false, pointRadius: 0, tension: 0.1 }},
        ],
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: true, position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
          tooltip: {{ mode: 'index', intersect: false }},
          annotation: false,
        }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Trading days', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ type: 'logarithmic', title: {{ display: true, text: 'Equity ($)', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
        }},
        interaction: {{ mode: 'nearest', intersect: false }},
      }},
    }});
  }}

  function drawHitRate(canvasId, payload) {{
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: payload.map(d => d.day),
        datasets: [{{ label: 'P(hit target)', data: payload.map(d => d.pct * 100), borderColor: PALETTE.target, backgroundColor: 'rgba(34,197,94,0.15)', fill: true, tension: 0.2, pointRadius: 0 }}],
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: (c) => 'P(hit) = ' + c.parsed.y.toFixed(1) + '%' }} }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Day', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ title: {{ display: true, text: 'P(hit ${target_int}) %', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, min: 0, max: 100 }},
        }},
      }},
    }});
  }}

  const PAYLOADS = {payloads_json};
  for (const k in PAYLOADS) {{
    drawEquity('eq_' + k, PAYLOADS[k].equity);
    drawHitRate('hit_' + k, PAYLOADS[k].hit);
  }}
</script>

</body>
</html>
"""


def _format_pct(x: float | None, places: int = 1) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 100:.{places}f}%"


def _format_dollar(x: float | None) -> str:
    if x is None or x != x:
        return "—"
    return f"${x:,.0f}"


def _format_days(x: float | None) -> str:
    if x is None or x != x:
        return "never"
    return f"{int(x)}d"


def _compare_rows_html(results: list[dict]) -> str:
    rows = []
    for r in results:
        ts = r["stats"]
        mc = r["mc"]
        rows.append(
            f"""
            <tr>
              <td><strong>{escape(r['label'])}</strong></td>
              <td class="numeric">{ts.n_trades_per_year:.0f}</td>
              <td class="numeric">{_format_pct(ts.win_rate)}</td>
              <td class="numeric pos">+{_format_pct(ts.avg_win)}</td>
              <td class="numeric neg">{_format_pct(ts.avg_loss)}</td>
              <td class="numeric {'pos' if ts.expected_return_per_unit > 0 else 'neg'}">{ts.expected_return_per_unit * 100:+.2f}%</td>
              <td class="numeric">{_format_pct(mc.pct_hit_target)}</td>
              <td class="numeric {'neg' if mc.pct_ruined > 0.10 else ''}">{_format_pct(mc.pct_ruined)}</td>
              <td class="numeric">{_format_dollar(mc.median_final)}</td>
              <td class="numeric">{_format_days(mc.median_time_to_target)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def _strategy_block_html(r: dict, target: float) -> str:
    cfg = r["config"]
    ts = r["stats"]
    mc = r["mc"]
    key = r["key"]
    return f"""
    <section class="strategy-block">
      <div class="strategy-head">
        <div class="name">{escape(r['label'])}</div>
        <div class="density">signal density: {r['signal_density'] * 100:.1f}% of trading days · {ts.n_signals} historical trades</div>
      </div>

      <div class="summary-grid">
        <div class="stat"><div class="label">P(hit ${int(target):,})</div><div class="value">{_format_pct(mc.pct_hit_target)}</div><div class="small">in {mc.config.horizon_days} days</div></div>
        <div class="stat"><div class="label">P(ruined)</div><div class="value" style="color:{'#ef4444' if mc.pct_ruined > 0.10 else '#22c55e'}">{_format_pct(mc.pct_ruined)}</div><div class="small">equity ≤ ${int(mc.config.floor_equity)}</div></div>
        <div class="stat"><div class="label">Median final</div><div class="value">{_format_dollar(mc.median_final)}</div><div class="small">25–75: {_format_dollar(mc.p25_final)} – {_format_dollar(mc.p75_final)}</div></div>
        <div class="stat"><div class="label">Median time-to-target</div><div class="value">{_format_days(mc.median_time_to_target)}</div><div class="small">25–75: {_format_days(mc.p25_time_to_target)} – {_format_days(mc.p75_time_to_target)}</div></div>
        <div class="stat"><div class="label">Median max drawdown</div><div class="value" style="color:#ef4444">{_format_pct(mc.median_max_dd)}</div><div class="small">25–75: {_format_pct(mc.p25_max_dd)} – {_format_pct(mc.p75_max_dd)}</div></div>
        <div class="stat"><div class="label">EV / trade</div><div class="value" style="color:{'#22c55e' if ts.expected_return_per_unit > 0 else '#ef4444'}">{ts.expected_return_per_unit * 100:+.2f}%</div><div class="small">per dollar risked</div></div>
      </div>

      <div class="row">
        <div class="panel">
          <h3>Equity-curve fan ({mc.n_paths:,} paths, log-scale)</h3>
          <canvas id="eq_{key}" height="240"></canvas>
        </div>
        <div class="panel">
          <h3>Probability of hitting ${int(target):,}</h3>
          <canvas id="hit_{key}" height="240"></canvas>
        </div>
      </div>
    </section>
    """


def _verdict_for(top: dict, target: float) -> tuple[str, str]:
    mc = top["mc"]
    p_hit = mc.pct_hit_target
    p_ruin = mc.pct_ruined
    horizon = mc.config.horizon_days

    if p_hit >= 0.50 and p_ruin <= 0.05:
        cls = "win"
        text = (
            f"The best strategy ({escape(top['label'])}) hits ${int(target):,} in "
            f"{horizon} days with <strong>{p_hit * 100:.0f}%</strong> probability and only "
            f"<strong>{p_ruin * 100:.0f}%</strong> chance of ruin. This is the closest "
            "thing to a &lsquo;secure&rsquo; 10&times; we found."
        )
    elif p_hit >= 0.20 and p_ruin <= 0.20:
        cls = "warn"
        text = (
            f"The best strategy ({escape(top['label'])}) gives you "
            f"<strong>{p_hit * 100:.0f}%</strong> probability of hitting ${int(target):,} in "
            f"{horizon} days, with <strong>{p_ruin * 100:.0f}%</strong> chance of getting "
            "wiped out. Real edge exists, but &lsquo;secure&rsquo; is a stretch."
        )
    else:
        cls = "bust"
        text = (
            f"Even the best strategy only hits the target with "
            f"<strong>{p_hit * 100:.0f}%</strong> probability and busts <strong>{p_ruin * 100:.0f}%</strong> "
            "of paths. The 10× goal is too aggressive for the model's edge at the chosen sizing. "
            "Lower the target, lengthen the horizon, or accept the math."
        )
    return cls, text


def render_html(results: list[dict], mc_cfg: MCConfig, n_years: float) -> str:
    payloads = {
        r["key"]: {
            "equity": {
                "median": r["mc"].median_curve,
                "p25": r["mc"].p25_curve,
                "p75": r["mc"].p75_curve,
                "p05": r["mc"].p05_curve,
                "p95": r["mc"].p95_curve,
            },
            "hit": r["mc"].pct_hit_by_day,
        }
        for r in results
    }

    # Pick "best" by P(hit) - P(ruin) trade-off
    best = max(results, key=lambda r: r["mc"].pct_hit_target - r["mc"].pct_ruined)
    verdict_cls, verdict_text = _verdict_for(best, mc_cfg.target_equity)

    return HTML_TEMPLATE.format(
        target_int=int(mc_cfg.target_equity),
        start_int=int(mc_cfg.start_equity),
        floor_int=int(mc_cfg.floor_equity),
        n_years=n_years,
        n_sims=mc_cfg.n_sims,
        horizon=mc_cfg.horizon_days,
        risk_pct=mc_cfg.risk_frac * 100,
        trades_per_day=mc_cfg.trades_per_day,
        daily_risk_pct=mc_cfg.risk_frac * mc_cfg.trades_per_day * 100,
        verdict_cls=verdict_cls,
        verdict_text=verdict_text,
        compare_rows=_compare_rows_html(results),
        strategy_blocks="\n".join(_strategy_block_html(r, mc_cfg.target_equity) for r in results),
        payloads_json=json.dumps(payloads, default=float),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strategy MC report.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start-equity", type=float, default=500.0)
    p.add_argument("--target", type=float, default=5_000.0)
    p.add_argument("--floor", type=float, default=25.0)
    p.add_argument("--horizon", type=int, default=252, help="Trading days to simulate.")
    p.add_argument("--risk-frac", type=float, default=0.05, help="Fraction of equity per trade (0.05 = 5%).")
    p.add_argument("--trades-per-day", type=int, default=1)
    p.add_argument("--sims", type=int, default=2000)
    p.add_argument("--min-train", type=int, default=1000)
    p.add_argument("--refit-step", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
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

    daily = load_history(args.ticker)
    feats = build_features(daily)
    X, y, _ = prepare_xy(feats)
    if len(X) <= args.min_train:
        raise SystemExit(f"Not enough rows ({len(X)}) — need >{args.min_train}.")

    logger.info("Walk-forward LogReg on %d rows (min_train=%d, step=%d)…", len(X), args.min_train, args.refit_step)
    preds = walk_forward_proba(X, y, make_logreg, min_train=args.min_train, step=args.refit_step)
    p_vol_oos = preds["y_score"].rename("p_vol")
    logger.info("Out-of-sample predictions: %d rows from %s to %s", len(p_vol_oos), p_vol_oos.index.min().date(), p_vol_oos.index.max().date())

    n_years = (p_vol_oos.index[-1] - p_vol_oos.index[0]).days / 365.25 if len(p_vol_oos) > 1 else 1.0

    mc_cfg = MCConfig(
        n_sims=args.sims,
        horizon_days=args.horizon,
        risk_frac=args.risk_frac,
        trades_per_day=args.trades_per_day,
        start_equity=args.start_equity,
        target_equity=args.target,
        floor_equity=args.floor,
        seed=args.seed,
    )

    results = run_backtests(daily, p_vol_oos, n_years=n_years, mc_cfg=mc_cfg)
    if not results:
        raise SystemExit("All strategies produced 0 trades. Check thresholds.")

    html = render_html(results, mc_cfg, n_years=n_years)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print()
    print(f"{'Strategy':<55}{'Trades/yr':>10}{'WinRate':>10}{'EV/trade':>10}{'P(hit)':>10}{'P(ruin)':>10}{'MedFinal':>12}")
    print("-" * 117)
    for r in results:
        ts = r["stats"]
        mc = r["mc"]
        print(
            f"{r['label']:<55}"
            f"{ts.n_trades_per_year:>10.0f}"
            f"{ts.win_rate * 100:>9.1f}%"
            f"{ts.expected_return_per_unit * 100:>+9.2f}%"
            f"{mc.pct_hit_target * 100:>9.1f}%"
            f"{mc.pct_ruined * 100:>9.1f}%"
            f"  ${mc.median_final:>9,.0f}"
        )
    print()

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
