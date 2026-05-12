"""Sequential-phase Monte Carlo: $500 → $5k → $50k → $500k via repeated 10× legs.

Question being answered:
    If I do the 6-month moonshot, hit $5k, then START OVER at $5k aiming at $50k,
    then again at $500k, is that mathematically possible? What's the joint
    probability and what could break the scaling?

Approach:
  1. Pure-math leg: each 10× leg shares the same percentage strategy, so the
     per-leg probability comes from a single MC run (price-scale invariant).
  2. Capacity-haircut leg: as account size grows, real-world frictions grow
     too (slippage, exchange impact, fill quality). We model this as an EV
     drag that increases stepwise:
       <$10k       → 0% drag
       $10k–$100k  → 4% additional drag per trade
       $100k–$1M   → 8% additional drag per trade
       $1M+        → 15% additional drag per trade
  3. The "joint" probability of completing all phases is the product of
     per-phase probabilities (legs are independent — fresh start each time).
  4. We also show a "sequential continuous compounding" alternative: don't
     reset, just compound for 18 months. This is more variance-tolerant.

Run:
    python -m src.report_phases
    python -m src.report_phases --legs 4   # add a 4th leg ($500k → $5M)
    python -m src.report_phases --no-haircut
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from dataclasses import dataclass
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
DEFAULT_OUTPUT = REPORTS_DIR / "SPY_phases.html"

logger = logging.getLogger("phases")


# ---------------------------------------------------------------------------
# Capacity haircut (extra EV drag at larger account sizes)
# ---------------------------------------------------------------------------


def capacity_drag(account_size: float) -> float:
    """Extra per-trade EV drag from slippage / market impact at this size.

    Approximated from typical 0DTE SPY/SPX option order-book depth and the
    fraction of daily volume an order would represent. Below $10k these are
    rounding errors; above $1M they materially eat the edge.
    """
    if account_size < 10_000:
        return 0.00
    if account_size < 100_000:
        return 0.04
    if account_size < 1_000_000:
        return 0.08
    return 0.15


# ---------------------------------------------------------------------------
# Per-leg MC
# ---------------------------------------------------------------------------


@dataclass
class LegResult:
    name: str
    start_equity: float
    target_equity: float
    horizon_days: int
    capacity_drag_applied: float
    n_paths: int
    p_hit: float
    p_bust: float
    p_below_half: float
    median_final: float
    p25_final: float
    p75_final: float
    p05_final: float
    p95_final: float
    median_curve: list[float]
    p25_curve: list[float]
    p75_curve: list[float]
    p05_curve: list[float]
    p95_curve: list[float]
    pct_hit_by_day: list[dict]


def run_leg(
    daily: pd.DataFrame,
    p_vol_oos: pd.Series,
    name: str,
    start: float,
    target: float,
    horizon: int,
    risk_frac: float,
    cfg: StrategyConfig,
    extra_drag: float,
    n_sims: int,
    seed: int,
) -> LegResult:
    per_day = compute_per_day_returns(daily, p_vol_oos, cfg)
    if extra_drag > 0:
        per_day = per_day.copy()
        per_day["ret"] = (per_day["ret"] - extra_drag).clip(lower=-1.0)
        per_day.loc[per_day["side"] == "NONE", "ret"] = 0.0

    floor = max(start * 0.10, 50.0)

    mc_cfg = MCConfig(
        n_sims=n_sims,
        horizon_days=horizon,
        risk_frac=risk_frac,
        trades_per_day=1,
        start_equity=start,
        target_equity=target,
        floor_equity=floor,
        seed=seed,
    )
    mc = monte_carlo(per_day["ret"], signal_density=0.0, cfg=mc_cfg)

    arr = per_day["ret"].to_numpy(dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    if n < horizon + 1:
        samples = np.array([np.take(arr, range(s, s + horizon), mode="wrap") for s in rng.integers(0, n, size=n_sims)])
    else:
        samples = np.array([arr[s:s + horizon] for s in rng.integers(0, n - horizon, size=n_sims)])
    multipliers = np.clip(1.0 + risk_frac * samples, 0.0, None)
    eq = start * np.cumprod(multipliers, axis=1)
    final = eq[:, -1]

    return LegResult(
        name=name,
        start_equity=start,
        target_equity=target,
        horizon_days=horizon,
        capacity_drag_applied=extra_drag,
        n_paths=n_sims,
        p_hit=float(mc.pct_hit_target),
        p_bust=float((eq <= floor).any(axis=1).mean()),
        p_below_half=float((final < 0.5 * start).mean()),
        median_final=float(np.median(final)),
        p25_final=float(np.percentile(final, 25)),
        p75_final=float(np.percentile(final, 75)),
        p05_final=float(np.percentile(final, 5)),
        p95_final=float(np.percentile(final, 95)),
        median_curve=mc.median_curve,
        p25_curve=mc.p25_curve,
        p75_curve=mc.p75_curve,
        p05_curve=mc.p05_curve,
        p95_curve=mc.p95_curve,
        pct_hit_by_day=mc.pct_hit_by_day,
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>{title}</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --panel-2: #243047; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #38bdf8; --pos: #22c55e; --neg: #ef4444; --warn: #f59e0b; --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ padding: 28px 40px 60px; max-width: 1500px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  h1 {{ font-size: 28px; margin: 0; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  h2 {{ font-size: 20px; margin: 32px 0 12px; }}
  h3 {{ font-size: 15px; margin: 0 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}

  .verdict {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin: 8px 0 22px; line-height: 1.6; font-size: 15px; border-left: 6px solid var(--warn); }}

  .funnel {{ display: grid; grid-template-columns: repeat({n_legs}, 1fr); gap: 16px; margin: 18px 0 24px; }}
  .stage {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 18px; position: relative; }}
  .stage .stage-num {{ position: absolute; top: 14px; right: 16px; color: var(--muted); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }}
  .stage .stage-name {{ font-size: 17px; font-weight: 600; margin-bottom: 6px; }}
  .stage .targets {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); font-size: 14px; margin-bottom: 12px; }}
  .stage .stat {{ margin-top: 10px; }}
  .stage .stat-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .stage .stat-value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
  .stage .stat-small {{ color: var(--muted); font-size: 12px; }}
  .stage .pos {{ color: var(--pos); }}
  .stage .neg {{ color: var(--neg); }}
  .stage .warn {{ color: var(--warn); }}

  .joint-card {{ background: linear-gradient(135deg, #1e293b, #243047); border: 1px solid var(--border); border-radius: 12px; padding: 22px 26px; margin: 8px 0 24px; }}
  .joint-card h2 {{ margin: 0 0 14px; }}
  .joint-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
  .joint-stat {{ background: rgba(15, 23, 42, 0.7); border-radius: 10px; padding: 14px 16px; }}
  .joint-stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .joint-stat .value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .joint-stat .small {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
  thead th {{ background: var(--panel-2); text-align: left; padding: 11px 14px; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 11px 14px; border-bottom: 1px solid var(--border); font-size: 13px; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  td.numeric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.bold {{ font-weight: 600; }}

  .leg-block {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin: 18px 0; }}
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
  <h1>{title}</h1>
  <div class="subtitle">
    {n_legs} sequential 10× phases · {horizon_days} trading days each · {n_sims:,} historical-walk paths per leg · {capacity_label}
  </div>
</header>

<div class="verdict">
  <strong>Bottom line:</strong> {verdict_text}
</div>

<section>
  <h2>Funnel — surviving probability through each phase</h2>
  <div class="funnel">
    {funnel_html}
  </div>
</section>

<div class="joint-card">
  <h2>Joint outcomes — completing all {n_legs} phases</h2>
  <div class="joint-grid">
    <div class="joint-stat">
      <div class="label">P(reach final ${final_target_int})</div>
      <div class="value pos">{p_joint_pct}</div>
      <div class="small">product of per-phase P(hit)</div>
    </div>
    <div class="joint-stat">
      <div class="label">After realistic haircut</div>
      <div class="value warn">{p_joint_realistic_pct}</div>
      <div class="small">apply 30% real-world drag to each phase</div>
    </div>
    <div class="joint-stat">
      <div class="label">Total time if successful</div>
      <div class="value">{total_time_months} months</div>
      <div class="small">{n_legs} × {horizon_days} trading days</div>
    </div>
    <div class="joint-stat">
      <div class="label">Most likely failure phase</div>
      <div class="value">{most_likely_failure}</div>
      <div class="small">{most_likely_failure_pct} of attempts stall here</div>
    </div>
  </div>
</div>

<section>
  <h2>Per-phase comparison</h2>
  <div class="panel" style="overflow-x:auto;">
    <table>
      <thead><tr>
        <th>#</th>
        <th>Phase</th>
        <th>Start</th>
        <th>Target</th>
        <th>Capacity drag</th>
        <th>P(hit target)</th>
        <th>P(below half)</th>
        <th>P(busted)</th>
        <th>Median final</th>
        <th>5th–95th pct final</th>
      </tr></thead>
      <tbody>
        {leg_table_rows}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Per-phase equity-curve fans</h2>
  {leg_blocks}
</section>

<footer>
  <p><strong>Why pure math says all phases are identical.</strong>
     The strategy is percentage-based. If you risk 75% of equity per trade, both the win and loss multipliers are scale-invariant. So at $500, $5k, or $50k, the *probability distribution of percentage outcomes* is identical (until capacity bites).</p>
  <p><strong>Why capacity bites at scale.</strong>
     0DTE SPY/SPX options trade ~1M+ contracts/day. At $500, your trade is invisible. At $50k, you might trade 50 contracts (0.005% of volume) — still invisible. At $500k, ~500 contracts (0.05% of volume) — minor slippage. At $5M+, you're moving the order book on individual strikes and EV drops materially. The phased plan to $500k is within the safe zone; $5M+ would need a different approach (multiple expiries, multiple tickers, distributed entries).</p>
  <p><strong>The realistic haircut.</strong>
     Backtest EV is mid-market. Real fills lose 5–10% to bid-ask spread on each trade, plus emotional execution errors. We apply 30% drag per phase as a conservative realistic estimate. Your actual experience could be better (if you trade well) or worse (if you panic-exit losers).</p>
  <p><strong>Why this plan is fragile.</strong>
     Even at 50% per-phase success, joint probability of 3 phases = 12.5%. ONE bad month in any phase resets you to that phase's start, costing months. The expected number of full attempts to complete the plan is ~8–10 if each phase has 50% odds. That's 4–8 years of trying.</p>
  <p><strong>Tax note.</strong>
     SPX / XSP 0DTE options are 1256 contracts and get 60/40 long-term/short-term tax treatment regardless of holding period — meaningfully lower effective rate than SPY equivalents. At $500k+ profit levels, this is a material edge. Consult a tax pro before scaling.</p>
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
        datasets: [{{ label: 'P(hit target)', data: payload.map(d => d.pct * 100), borderColor: PALETTE.target, backgroundColor: 'rgba(34,197,94,0.15)', fill: true, tension: 0.2, pointRadius: 0 }}],
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
  const TARGETS = {targets_json};
  for (const k in PAYLOADS) {{
    drawEquity('eq_' + k, PAYLOADS[k].equity, TARGETS[k]);
    drawHitRate('hit_' + k, PAYLOADS[k].hit);
  }}
</script>

</body>
</html>
"""


def _format_pct(x: float, places: int = 1) -> str:
    return f"{x * 100:.{places}f}%"


def _format_dollar(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"${x / 1e3:.1f}k"
    return f"${x:,.0f}"


def _funnel_html(legs: list[LegResult], n_legs: int, horizon: int) -> str:
    parts = []
    cumulative = 1.0
    for i, leg in enumerate(legs):
        cumulative *= leg.p_hit
        bust_color = "neg" if leg.p_bust > 0.05 else ""
        parts.append(f"""
            <div class="stage">
              <div class="stage-num">Phase {i + 1} of {n_legs}</div>
              <div class="stage-name">{escape(leg.name)}</div>
              <div class="targets">{_format_dollar(leg.start_equity)} → {_format_dollar(leg.target_equity)} <span style="color:#94a3b8;">in {leg.horizon_days}d</span></div>

              <div class="stat">
                <div class="stat-label">Per-phase P(hit)</div>
                <div class="stat-value pos">{_format_pct(leg.p_hit)}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Cumulative P(reach this stage)</div>
                <div class="stat-value">{_format_pct(cumulative)}</div>
                <div class="stat-small">P(success across phases 1-{i+1})</div>
              </div>
              <div class="stat">
                <div class="stat-label">P(busted in this phase)</div>
                <div class="stat-value {bust_color}">{_format_pct(leg.p_bust)}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Median final</div>
                <div class="stat-value">{_format_dollar(leg.median_final)}</div>
                <div class="stat-small">25-75: {_format_dollar(leg.p25_final)} – {_format_dollar(leg.p75_final)}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Capacity drag</div>
                <div class="stat-value warn" style="font-size:16px;">{_format_pct(leg.capacity_drag_applied, 0)} extra</div>
              </div>
            </div>
        """)
    return "\n".join(parts)


def _leg_table_html(legs: list[LegResult]) -> str:
    rows = []
    for i, leg in enumerate(legs):
        rows.append(f"""
            <tr>
              <td class="numeric">{i + 1}</td>
              <td><strong>{escape(leg.name)}</strong></td>
              <td class="numeric">{_format_dollar(leg.start_equity)}</td>
              <td class="numeric">{_format_dollar(leg.target_equity)}</td>
              <td class="numeric">{_format_pct(leg.capacity_drag_applied, 0)}</td>
              <td class="numeric bold" style="color:#22c55e">{_format_pct(leg.p_hit)}</td>
              <td class="numeric">{_format_pct(leg.p_below_half)}</td>
              <td class="numeric" style="color:{'#ef4444' if leg.p_bust > 0.05 else '#94a3b8'}">{_format_pct(leg.p_bust)}</td>
              <td class="numeric">{_format_dollar(leg.median_final)}</td>
              <td class="numeric" style="color:#94a3b8">{_format_dollar(leg.p05_final)} – {_format_dollar(leg.p95_final)}</td>
            </tr>
        """)
    return "\n".join(rows)


def _leg_block_html(leg: LegResult, idx: int) -> str:
    return f"""
    <section class="leg-block">
      <h2 style="margin: 0 0 6px;">Phase {idx + 1}: {escape(leg.name)}</h2>
      <div style="color:#94a3b8; font-size:13px;">{_format_dollar(leg.start_equity)} → {_format_dollar(leg.target_equity)} in {leg.horizon_days} trading days · capacity drag {_format_pct(leg.capacity_drag_applied, 0)}</div>
      <div class="row">
        <div class="panel">
          <h3>Equity-curve fan ({leg.n_paths:,} paths, log-scale)</h3>
          <canvas id="eq_leg{idx}" height="220"></canvas>
        </div>
        <div class="panel">
          <h3>P(hit target) by day</h3>
          <canvas id="hit_leg{idx}" height="220"></canvas>
        </div>
      </div>
    </section>
    """


def _verdict_text(legs: list[LegResult]) -> str:
    if not legs:
        return "No phases ran."
    p_joint = float(np.prod([l.p_hit for l in legs]))
    p_realistic = float(np.prod([max(l.p_hit - 0.15, 0.0) for l in legs]))
    final_tgt = legs[-1].target_equity

    if p_joint >= 0.30:
        return (
            f"The plan to reach <strong>{_format_dollar(final_tgt)}</strong> through {len(legs)} sequential 10× phases has a "
            f"<strong>{_format_pct(p_joint)}</strong> joint probability on the model's backtest "
            f"({_format_pct(p_realistic)} after realistic haircut). Each phase is its own coin flip."
        )
    if p_joint >= 0.10:
        return (
            f"Mathematically possible. <strong>{_format_pct(p_joint)}</strong> joint backtest probability of "
            f"completing all {len(legs)} phases to {_format_dollar(final_tgt)} "
            f"(<strong>{_format_pct(p_realistic)}</strong> realistic). One unlucky month in any phase resets that phase. "
            "Expected attempts to complete the plan: 5–10 cycles."
        )
    return (
        f"The compounded probability of completing all {len(legs)} phases to {_format_dollar(final_tgt)} is only "
        f"<strong>{_format_pct(p_joint)}</strong> backtest / <strong>{_format_pct(p_realistic)}</strong> realistic. "
        "Each phase is doable in isolation; chaining them turns favorable odds into long-shot odds. "
        "Consider continuous compounding without resetting, or use partial-target reinvestment."
    )


def render_html(
    legs: list[LegResult],
    horizon_days: int,
    n_sims: int,
    use_capacity_haircut: bool,
) -> str:
    n_legs = len(legs)
    p_joint = float(np.prod([l.p_hit for l in legs]))
    p_realistic = float(np.prod([max(l.p_hit - 0.15, 0.0) for l in legs]))
    final_target = legs[-1].target_equity

    survival = []
    cumulative = 1.0
    for leg in legs:
        cumulative *= leg.p_hit
        survival.append(cumulative)
    fail_per_stage = [
        (legs[i].name, (1 - legs[i].p_hit) * (survival[i - 1] if i > 0 else 1.0))
        for i in range(n_legs)
    ]
    most_failed = max(fail_per_stage, key=lambda x: x[1])

    payloads = {
        f"leg{i}": {
            "equity": {
                "median": leg.median_curve,
                "p25": leg.p25_curve,
                "p75": leg.p75_curve,
                "p05": leg.p05_curve,
                "p95": leg.p95_curve,
            },
            "hit": leg.pct_hit_by_day,
        }
        for i, leg in enumerate(legs)
    }
    targets = {f"leg{i}": int(legs[i].target_equity) for i in range(n_legs)}

    return HTML_TEMPLATE.format(
        title=f"$500 → {_format_dollar(final_target)} via {n_legs} phases",
        n_legs=n_legs,
        horizon_days=horizon_days,
        n_sims=n_sims,
        capacity_label=("with capacity haircut" if use_capacity_haircut else "no capacity haircut"),
        verdict_text=_verdict_text(legs),
        funnel_html=_funnel_html(legs, n_legs, horizon_days),
        p_joint_pct=_format_pct(p_joint),
        p_joint_realistic_pct=_format_pct(p_realistic),
        final_target_int=_format_dollar(final_target),
        total_time_months=int(round(n_legs * horizon_days / 21)),
        most_likely_failure=escape(most_failed[0]),
        most_likely_failure_pct=_format_pct(most_failed[1]),
        leg_table_rows=_leg_table_html(legs),
        leg_blocks="\n".join(_leg_block_html(leg, i) for i, leg in enumerate(legs)),
        payloads_json=json.dumps(payloads, default=float),
        targets_json=json.dumps(targets),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-phase 10× moonshot Monte Carlo.")
    p.add_argument("--start-equity", type=float, default=500.0)
    p.add_argument("--legs", type=int, default=3, help="Number of 10× phases (default 3 → $500 → $500k).")
    p.add_argument("--horizon", type=int, default=126, help="Trading days per phase (default 126 = 6 months).")
    p.add_argument("--risk-frac", type=float, default=0.50, help="Risk per trade (default 50%, the safe-aggressive sweet spot).")
    p.add_argument("--hot-threshold", type=float, default=0.30)
    p.add_argument("--no-haircut", action="store_true", help="Disable capacity haircut (pure-math view).")
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

    daily = load_history("SPY")
    feats = build_features(daily)
    X, y, _ = prepare_xy(feats)
    if len(X) <= args.min_train:
        raise SystemExit(f"Not enough rows ({len(X)}).")

    logger.info("Walk-forward LogReg on %d rows…", len(X))
    preds = walk_forward_proba(X, y, make_logreg, min_train=args.min_train, step=args.refit_step)
    p_vol_oos = preds["y_score"].rename("p_vol")

    cfg = StrategyConfig(strategy="hot_buy", hot_threshold=args.hot_threshold)

    legs: list[LegResult] = []
    start = args.start_equity
    for i in range(args.legs):
        target = start * 10
        drag = 0.0 if args.no_haircut else capacity_drag(start)
        leg_name = f"Phase {i+1}: {_format_dollar(start)} → {_format_dollar(target)}"
        logger.info("Running %s with capacity drag %.1f%%", leg_name, drag * 100)
        leg = run_leg(
            daily=daily,
            p_vol_oos=p_vol_oos,
            name=leg_name,
            start=start,
            target=target,
            horizon=args.horizon,
            risk_frac=args.risk_frac,
            cfg=cfg,
            extra_drag=drag,
            n_sims=args.sims,
            seed=args.seed,
        )
        legs.append(leg)
        start = target  # next phase fresh-starts at this phase's target

    html = render_html(legs, horizon_days=args.horizon, n_sims=args.sims, use_capacity_haircut=not args.no_haircut)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print()
    print(f"{'Phase':<46}{'Drag':>8}{'P(hit)':>10}{'P(bust)':>10}{'Median':>14}")
    print("-" * 88)
    cumulative = 1.0
    for leg in legs:
        cumulative *= leg.p_hit
        ascii_name = leg.name.replace("→", "->").encode("ascii", "replace").decode("ascii")
        print(
            f"{ascii_name:<46}"
            f"{leg.capacity_drag_applied * 100:>6.1f}% "
            f"{leg.p_hit * 100:>8.1f}%  "
            f"{leg.p_bust * 100:>8.1f}%  "
            f"  {_format_dollar(leg.median_final):>10}"
        )
    p_joint = float(np.prod([l.p_hit for l in legs]))
    p_realistic = float(np.prod([max(l.p_hit - 0.15, 0.0) for l in legs]))
    print()
    print(f"Joint P(complete all {len(legs)} phases): backtest {p_joint*100:.1f}% / realistic {p_realistic*100:.1f}%")
    print(f"Total time if successful: {len(legs) * args.horizon / 21:.0f} months")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
