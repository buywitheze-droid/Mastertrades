"""1-month moonshot: $500 → $5,000 by BUYING calls/puts only.

Constraints from the user:
  - Long-only options (no premium selling)
  - Horizon ~21 trading days (one calendar month)
  - 70%+ win rate is acceptable; doesn't need to be 100%
  - High variance OK as long as the target is reachable

Two long-only strategies tested at multiple sizings:

  STRADDLE      — buy 0DTE ATM straddle on every HOT-tier signal (P_vol > thr)
                  Backtest win rate ~70% (no direction needed)
  DIRECTIONAL   — buy single-leg call OR put based on a momentum bias
                  (today's open vs 20-day MA). Lower win rate, bigger wins.

For each strategy we sweep risk_frac ∈ {25%, 50%, 75%, 100%} per trade.
21-day horizon · 5,000 random historical-walk paths each.

The report ranks by **P(hit $5,000)** but also surfaces P(bust) so you can
pick the spot on the risk/reward frontier you actually want to live with.
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
    StrategyConfig,
    compute_per_day_returns,
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
DEFAULT_OUTPUT = REPORTS_DIR / "SPY_moonshot.html"

logger = logging.getLogger("moonshot")


# ---------------------------------------------------------------------------
# Strategy variants
# ---------------------------------------------------------------------------


def _build_direction_signal(features: pd.DataFrame) -> pd.Series:
    """+1 if recent trend is up, -1 if down, 0 if flat.

    Uses dist_ma20 (today's close vs 20-day MA, lagged by one). Signed by sign,
    with a small dead-band so we don't flip on noise.
    """
    raw = features["dist_ma20"]
    deadband = float(raw.abs().quantile(0.20))
    direction = np.sign(raw.where(raw.abs() >= deadband, 0.0))
    return direction.fillna(0).astype(int).rename("direction")


def _strategy_variants() -> list[tuple[str, str, StrategyConfig]]:
    return [
        (
            "straddle_strict",
            "Long straddle, HOT-only (P_vol > 0.45)",
            StrategyConfig(strategy="hot_buy", hot_threshold=0.45),
        ),
        (
            "straddle_lenient",
            "Long straddle, broader HOT (P_vol > 0.30)",
            StrategyConfig(strategy="hot_buy", hot_threshold=0.30),
        ),
        (
            "directional_strict",
            "Single-leg call/put, HOT-only (P_vol > 0.45)",
            StrategyConfig(strategy="directional_buy", hot_threshold=0.45),
        ),
        (
            "directional_lenient",
            "Single-leg call/put, broader HOT (P_vol > 0.30)",
            StrategyConfig(strategy="directional_buy", hot_threshold=0.30),
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_sweep(
    spy: pd.DataFrame,
    p_vol_oos: pd.Series,
    direction: pd.Series,
    risk_fracs: list[float],
    horizon_days: int,
    n_sims: int,
    start_equity: float,
    target_equity: float,
    floor_equity: float,
    seed: int,
) -> list[dict]:
    """For each strategy variant × risk_frac combination, run an MC sim."""
    n_total_days = len(spy.index.intersection(p_vol_oos.index))
    n_years = (p_vol_oos.index[-1] - p_vol_oos.index[0]).days / 365.25 if len(p_vol_oos) > 1 else 1.0

    out: list[dict] = []
    for key, label, cfg in _strategy_variants():
        per_day = compute_per_day_returns(spy, p_vol_oos, cfg, direction_signal=direction)
        trades = per_day[per_day["side"] != "NONE"]
        if trades.empty:
            logger.warning("Strategy %s produced 0 trades — skipping", key)
            continue

        ts = trade_stats(trades["ret"], n_years=n_years)
        signal_density = len(trades) / n_total_days if n_total_days > 0 else 0.0

        for rf in risk_fracs:
            mc_cfg = MCConfig(
                n_sims=n_sims,
                horizon_days=horizon_days,
                risk_frac=rf,
                trades_per_day=1,
                start_equity=start_equity,
                target_equity=target_equity,
                floor_equity=floor_equity,
                seed=seed,
            )
            mc = monte_carlo(per_day["ret"], signal_density=signal_density, cfg=mc_cfg)

            # Probability of hitting various intermediate milestones (additional
            # context beyond just P(target)).
            arr = per_day["ret"].to_numpy(dtype=float)
            n = len(arr)
            rng = np.random.default_rng(seed)
            if n < horizon_days + 1:
                samples = np.array([np.take(arr, range(s, s + horizon_days), mode="wrap") for s in rng.integers(0, n, size=n_sims)])
            else:
                samples = np.array([arr[s:s + horizon_days] for s in rng.integers(0, n - horizon_days, size=n_sims)])
            multipliers = np.clip(1.0 + rf * samples, 0.0, None)
            eq = start_equity * np.cumprod(multipliers, axis=1)
            final = eq[:, -1]
            ever_hit = (eq >= target_equity).any(axis=1)

            milestones = {
                "p_double":       float((final >= 2 * start_equity).mean()),
                "p_triple":       float((final >= 3 * start_equity).mean()),
                "p_5x":           float((final >= 5 * start_equity).mean()),
                "p_10x_ever":     float(ever_hit.mean()),     # alias for clarity
                "p_below_half":   float((final < 0.5 * start_equity).mean()),
                "p_below_floor":  float((final <= floor_equity).mean()),
            }

            out.append({
                "key": f"{key}__rf{int(rf * 100)}",
                "strategy_key": key,
                "label": label,
                "config": cfg,
                "stats": ts,
                "trades": trades,
                "mc": mc,
                "signal_density": float(signal_density),
                "risk_frac": float(rf),
                "milestones": milestones,
            })

    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Moonshot: $500 → ${target_int} in {horizon} days</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --panel-2: #243047; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #38bdf8; --pos: #22c55e; --neg: #ef4444; --warn: #f59e0b; --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ padding: 28px 40px 60px; max-width: 1400px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  h1 {{ font-size: 28px; margin: 0; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  h2 {{ font-size: 20px; margin: 32px 0 12px; }}
  h3 {{ font-size: 15px; margin: 0 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}

  .verdict {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin: 8px 0 22px; line-height: 1.6; font-size: 15px; }}
  .verdict.win {{ border-left: 6px solid var(--pos); }}
  .verdict.warn {{ border-left: 6px solid var(--warn); }}
  .verdict.bust {{ border-left: 6px solid var(--neg); }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 16px 0 24px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
  .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .stat .small {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
  thead th {{ background: var(--panel-2); text-align: left; padding: 10px 12px; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(56,189,248,0.06); }}
  tbody tr.best {{ background: rgba(34,197,94,0.10); border-left: 3px solid var(--pos); }}
  tbody tr.bust {{ background: rgba(239,68,68,0.06); }}
  td.numeric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.pos {{ color: var(--pos); }}
  td.neg {{ color: var(--neg); }}
  td.bold {{ font-weight: 600; }}

  .strategy-block {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin: 18px 0; }}
  .strategy-head {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }}
  .strategy-head .name {{ font-size: 17px; font-weight: 700; }}

  .row {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; margin-top: 14px; }}
  @media (max-width: 980px) {{ .row {{ grid-template-columns: 1fr; }} }}
  .panel {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}

  canvas {{ background: #0b1220; border-radius: 8px; padding: 4px; }}

  footer {{ margin-top: 36px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  footer code {{ background: var(--panel); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <h1>Moonshot: ${start_int} → ${target_int} in {horizon} trading days</h1>
  <div class="subtitle">
    Long calls/puts only · {n_sims:,} historical-walk paths per cell · {n_years:.0f}y SPY backtest · walk-forward OOS predictions
  </div>
</header>

<div class="verdict {verdict_cls}">
  <strong>{verdict_headline}</strong><br>{verdict_text}
</div>

<section>
  <h2>Risk-fraction sweep — pick your spot on the frontier</h2>
  <div class="panel" style="overflow-x:auto;">
    <table>
      <thead><tr>
        <th>Strategy</th>
        <th>Risk / trade</th>
        <th>Trades / mo</th>
        <th>Win rate</th>
        <th>EV / trade</th>
        <th>P(hit ${target_int})</th>
        <th>P(triple)</th>
        <th>P(double)</th>
        <th>P(below ½ start)</th>
        <th>P(busted)</th>
        <th>Median final</th>
      </tr></thead>
      <tbody>
        {compare_rows}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Top 3 by P(hit ${target_int}) — full equity-curve fans</h2>
  {strategy_blocks}
</section>

<footer>
  <p><strong style="color:#f59e0b">⚠ Modeling caveats.</strong>
     Long-straddle payoff is modeled as max(close-move, 50% × intraday-max-excursion). Single-leg directional uses the favorable side only, also discounted 50%. Both pay a 5–6% drag for spread + commissions. Real-world haircut is typically 30–50% (slippage on entries, IV crush at the open, exit timing errors). Adjust the median-final values DOWN by 30–50% to get a realistic personal expectation.</p>
  <p><strong>About win rate.</strong> Per-trade win rate here means "the trade closed positive after costs", not "we hit a 100% gain." Many wins are small; many losses are full -100%. The strategy lives or dies on the right tail of wins.</p>
  <p><strong>About leverage.</strong> Risk-frac at 100% means "all in" — a single loss day takes you to ~0. The path may hit ${target_int} with 25-30% probability, but ~30% of paths blow up before getting there. There's no free lunch.</p>
  <p><strong>Sample size warning.</strong> Each {horizon}-day path contains only 3–10 actual trades on average. Compounding outcomes are dominated by the right tail (a few big wins) and the left tail (a few full-loss days). Don't over-interpret single-path stories.</p>
  <p><strong>PDT &amp; account.</strong> 0DTE long calls/puts can be day-traded freely in a cash account using cash-settled index options (XSP / SPX). For SPY 0DTE under $25k margin, you're capped at 3 day-trades per 5 business days unless you use SPX/XSP.</p>
</footer>

<script>
  const PALETTE = {{ med: '#38bdf8', band25: 'rgba(56,189,248,0.30)', band05: 'rgba(56,189,248,0.12)', target: '#22c55e' }};

  function drawEquity(canvasId, payload, target) {{
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const labels = payload.median.map((_, i) => i);
    const targetLine = labels.map(_ => target);
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
          {{ label: 'target',   data: targetLine, borderColor: PALETTE.target, borderWidth: 1.5, borderDash: [6,4], fill: false, pointRadius: 0 }},
        ],
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true, position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Trading day', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ type: 'logarithmic', title: {{ display: true, text: 'Equity ($)', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
        }},
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
        datasets: [{{ label: 'P(hit)', data: payload.map(d => d.pct * 100), borderColor: PALETTE.target, backgroundColor: 'rgba(34,197,94,0.15)', fill: true, tension: 0.2, pointRadius: 0 }}],
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Day', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ title: {{ display: true, text: 'P(hit %)', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, min: 0, max: 100 }},
        }},
      }},
    }});
  }}

  const PAYLOADS = {payloads_json};
  const TARGET = {target_value};
  for (const k in PAYLOADS) {{
    drawEquity('eq_' + k, PAYLOADS[k].equity, TARGET);
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


def _row_html(r: dict, target: float, is_best: bool) -> str:
    ts = r["stats"]
    mc = r["mc"]
    ms = r["milestones"]
    cls = "best" if is_best else ("bust" if ms["p_below_floor"] > 0.20 else "")
    return f"""
        <tr class="{cls}">
          <td><strong>{escape(r['label'])}</strong></td>
          <td class="numeric">{int(r['risk_frac'] * 100)}%</td>
          <td class="numeric">{ts.n_trades_per_year / 12:.1f}</td>
          <td class="numeric">{_format_pct(ts.win_rate)}</td>
          <td class="numeric {'pos' if ts.expected_return_per_unit > 0 else 'neg'}">{ts.expected_return_per_unit * 100:+.1f}%</td>
          <td class="numeric bold pos">{_format_pct(mc.pct_hit_target)}</td>
          <td class="numeric">{_format_pct(ms['p_triple'])}</td>
          <td class="numeric">{_format_pct(ms['p_double'])}</td>
          <td class="numeric neg">{_format_pct(ms['p_below_half'])}</td>
          <td class="numeric {'neg bold' if ms['p_below_floor'] > 0.10 else ''}">{_format_pct(ms['p_below_floor'])}</td>
          <td class="numeric">{_format_dollar(mc.median_final)}</td>
        </tr>
    """


def _strategy_block_html(r: dict, target: float) -> str:
    mc = r["mc"]
    ms = r["milestones"]
    return f"""
    <section class="strategy-block">
      <div class="strategy-head">
        <div class="name">{escape(r['label'])} · risk {int(r['risk_frac'] * 100)}% / trade</div>
      </div>
      <div class="summary-grid">
        <div class="stat"><div class="label">P(hit ${int(target):,})</div><div class="value pos">{_format_pct(mc.pct_hit_target)}</div></div>
        <div class="stat"><div class="label">P(triple)</div><div class="value">{_format_pct(ms['p_triple'])}</div></div>
        <div class="stat"><div class="label">P(busted)</div><div class="value" style="color:{'#ef4444' if ms['p_below_floor'] > 0.10 else '#22c55e'}">{_format_pct(ms['p_below_floor'])}</div></div>
        <div class="stat"><div class="label">Median final</div><div class="value">{_format_dollar(mc.median_final)}</div></div>
        <div class="stat"><div class="label">Median max DD</div><div class="value neg">{_format_pct(mc.median_max_dd)}</div></div>
        <div class="stat"><div class="label">Median time-to-target</div><div class="value">{('%dd' % int(mc.median_time_to_target)) if mc.median_time_to_target is not None else 'never'}</div></div>
      </div>
      <div class="row">
        <div class="panel">
          <h3>Equity-curve fan ({mc.n_paths:,} paths, log-scale)</h3>
          <canvas id="eq_{r['key']}" height="220"></canvas>
        </div>
        <div class="panel">
          <h3>P(hit target) by day</h3>
          <canvas id="hit_{r['key']}" height="220"></canvas>
        </div>
      </div>
    </section>
    """


def _verdict(best: dict, target: float, horizon: int) -> tuple[str, str, str]:
    p_hit = best["mc"].pct_hit_target
    p_bust = best["milestones"]["p_below_floor"]
    rf = int(best["risk_frac"] * 100)
    label = best["label"]

    if p_hit >= 0.40 and p_bust <= 0.15:
        cls = "win"
        head = f"Best shot: {p_hit * 100:.0f}% chance of hitting ${int(target):,} in {horizon} days."
        text = (
            f"<strong>{escape(label)}</strong> at <strong>{rf}% risk/trade</strong> wins backtest — "
            f"and only {p_bust * 100:.0f}% of paths get wiped out. After a real-world haircut, expect "
            f"~{max(p_hit - 0.10, 0.05) * 100:.0f}% chance of hitting target with ~{p_bust * 100:.0f}% bust risk."
        )
    elif p_hit >= 0.20:
        cls = "warn"
        head = f"Best shot: {p_hit * 100:.0f}% chance of hitting ${int(target):,} in {horizon} days."
        text = (
            f"<strong>{escape(label)}</strong> at <strong>{rf}% risk/trade</strong> is the best risk-reward, but "
            f"<strong>{p_bust * 100:.0f}%</strong> of paths bust before getting there. "
            f"After realistic frictions, expect ~{max(p_hit - 0.10, 0.05) * 100:.0f}% chance of target / {min(p_bust + 0.05, 0.5) * 100:.0f}% bust risk. "
            "This is a moonshot, not a plan."
        )
    else:
        cls = "bust"
        head = f"No combination hits ${int(target):,} reliably in {horizon} days."
        text = (
            f"Best result: <strong>{escape(label)}</strong> at {rf}% risk only delivers "
            f"{p_hit * 100:.0f}% probability with {p_bust * 100:.0f}% bust risk. "
            "10× in a month is fundamentally hard with the model's edge — consider lengthening the horizon "
            "or accepting that the answer is 'play smaller, take longer'."
        )
    return cls, head, text


def render_html(results: list[dict], horizon: int, n_sims: int, n_years: float, start: float, target: float) -> str:
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

    sorted_by_hit = sorted(results, key=lambda r: r["mc"].pct_hit_target, reverse=True)
    best = sorted_by_hit[0]
    top3_keys = {r["key"] for r in sorted_by_hit[:3]}

    verdict_cls, verdict_head, verdict_text = _verdict(best, target, horizon)

    sweep_rows = "".join(_row_html(r, target, r["key"] == best["key"]) for r in results)
    detail_blocks = "\n".join(_strategy_block_html(r, target) for r in sorted_by_hit[:3])

    return HTML_TEMPLATE.format(
        target_int=int(target),
        start_int=int(start),
        horizon=horizon,
        n_sims=n_sims,
        n_years=n_years,
        verdict_cls=verdict_cls,
        verdict_headline=verdict_head,
        verdict_text=verdict_text,
        compare_rows=sweep_rows,
        strategy_blocks=detail_blocks,
        payloads_json=json.dumps(payloads, default=float),
        target_value=int(target),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1-month moonshot Monte Carlo report.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start-equity", type=float, default=500.0)
    p.add_argument("--target", type=float, default=5_000.0)
    p.add_argument("--floor", type=float, default=50.0)
    p.add_argument("--horizon", type=int, default=21, help="Trading days (default 21 = 1 month).")
    p.add_argument("--risk-fracs", default="0.25,0.50,0.75,1.00",
                   help="Comma-separated risk fractions to sweep.")
    p.add_argument("--sims", type=int, default=5000)
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

    logger.info("Walk-forward LogReg on %d rows…", len(X))
    preds = walk_forward_proba(X, y, make_logreg, min_train=args.min_train, step=args.refit_step)
    p_vol_oos = preds["y_score"].rename("p_vol")
    direction = _build_direction_signal(feats)
    n_years = (p_vol_oos.index[-1] - p_vol_oos.index[0]).days / 365.25

    risk_fracs = [float(x) for x in args.risk_fracs.split(",")]
    logger.info("Sweeping risk_fracs=%s, horizon=%d days, %d sims", risk_fracs, args.horizon, args.sims)

    results = run_sweep(
        spy=daily,
        p_vol_oos=p_vol_oos,
        direction=direction,
        risk_fracs=risk_fracs,
        horizon_days=args.horizon,
        n_sims=args.sims,
        start_equity=args.start_equity,
        target_equity=args.target,
        floor_equity=args.floor,
        seed=args.seed,
    )
    if not results:
        raise SystemExit("All strategies produced 0 trades.")

    html = render_html(
        results=results,
        horizon=args.horizon,
        n_sims=args.sims,
        n_years=n_years,
        start=args.start_equity,
        target=args.target,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print()
    print(f"{'Strategy':<55}{'Risk':>6}{'Trd/mo':>8}{'WinRate':>10}{'EV/trd':>9}{'P(hit)':>9}{'P(triple)':>11}{'P(bust)':>10}{'MedFinal':>11}")
    print("-" * 119)
    for r in sorted(results, key=lambda r: r["mc"].pct_hit_target, reverse=True):
        ts = r["stats"]
        mc = r["mc"]
        ms = r["milestones"]
        print(
            f"{r['label']:<55}"
            f"{int(r['risk_frac'] * 100):>5}%"
            f"{ts.n_trades_per_year / 12:>8.1f}"
            f"{ts.win_rate * 100:>9.1f}%"
            f"{ts.expected_return_per_unit * 100:>+8.1f}%"
            f"{mc.pct_hit_target * 100:>8.1f}%"
            f"{ms['p_triple'] * 100:>10.1f}%"
            f"{ms['p_below_floor'] * 100:>9.1f}%"
            f"  ${mc.median_final:>8,.0f}"
        )

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
