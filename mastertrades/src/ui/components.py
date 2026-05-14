"""Mastertrades reusable UI components.

These functions return plain HTML strings (no Streamlit imports), so they
can be:
  - composed inside larger st.html() blocks
  - unit-tested against snapshot fixtures
  - reused on any page (Today's Plays, Command Center, future pages)

Every visual decision routes through src/ui/tokens.py — never hand-type a
color, font-size, or radius in this file. If you need a new shade, add it
to tokens.py.

Public API:
  metric_tile(label, value, *, sub, tone, columns_compatible=True) -> str
  phase_banner(phase, title, body, *, time_str) -> str
  source_health_row(name, ok, msg) -> str
  source_health_panel(rows: dict) -> str
  contract_row(p: dict) -> str
  trail_state_card(state: dict|None, trail_pct: float) -> str
  play_card(p: dict, *, rank, first_seen_str, ago_str, now_str,
            is_fresh, trail_html='', doctrine_html='') -> str
  empty_state(title, body) -> str
  actionable_hero(p, *, ago_seconds, now_str, trail_pct) -> str
"""
from __future__ import annotations

from typing import Optional

from .tokens import (
    SURFACE, BORDER, TEXT, STATUS, SOURCE, TYPE, RADIUS, SPACE,
    StateStyle, NEUTRAL_STYLE, SUCCESS_STYLE, WARN_STYLE, DANGER_STYLE, INFO_STYLE,
    tint, CSS_TABULAR_NUMS, CSS_UPPERCASE_LABEL,
)


# ── Per-source visual identity (source name → border color, blurb) ───────────
_SOURCE_BADGE: dict[str, tuple[str, str]] = {
    "MA Bounce":  (SOURCE.MA_BOUNCE,  "Weekly MA bounce · validated +243% on real options"),
    "ML Jackpot": (SOURCE.ML_JACKPOT, "ML vol+P&L classifier agreement · same-day 0DTE"),
    "Gap Fill":   (SOURCE.GAP_FILL,   "Gap reversal · per-ticker validated config (1-yr backtest, realised avg %)"),
    "0DTE Drop":  (SOURCE.DTE_DROP,   "Intraday drop ≥3 pts · drop-band lottery"),
}

# Per-state visual identity (state machine → accent color, glyph)
_STATE_STYLE: dict[str, tuple[str, str]] = {
    "TOUCHING":    (STATUS.SUCCESS, "🔥"),
    "ENTRY_OPEN":  (STATUS.INFO,    "⚡"),
    "NEAR_FILL":   (STATUS.INFO,    "⚡"),
    "APPROACHING": (STATUS.WARN,    "👀"),
    "WATCH_FILL":  (STATUS.WARN,    "👀"),
    "💎 PRIME":    (SOURCE.DTE_DROP,"💎"),
}


# ── Atomic primitives ────────────────────────────────────────────────────────

def metric_tile(label: str, value: str, *, sub: str = "",
                tone: str = TEXT.PRIMARY) -> str:
    """A single labeled metric — value on top, uppercase label above, optional
    sub-line below. Use inside a CSS Grid for KPI strips.

    Replaces the ad-hoc `lcard()` helpers scattered through 0DTE Lottery and
    similar pages.
    """
    sub_html = (
        f'<div style="color:{TEXT.TERTIARY};font-size:{TYPE.XS}px;'
        f'margin-top:{SPACE.XS}px;">{sub}</div>'
        if sub else ""
    )
    return (
        f'<div style="background:{SURFACE.RAISED};border:1px solid {BORDER.DEFAULT};'
        f'border-radius:{RADIUS.MD}px;padding:{SPACE.LG}px {SPACE.MD}px;text-align:center;">'
        f'  <div style="color:{TEXT.TERTIARY};{CSS_UPPERCASE_LABEL}'
        f'              margin-bottom:{SPACE.XS}px;">{label}</div>'
        f'  <div style="font-size:{TYPE.XL}px;font-weight:800;color:{tone};'
        f'              {CSS_TABULAR_NUMS}">{value}</div>'
        f'  {sub_html}'
        f'</div>'
    )


def phase_banner(phase_color: str, title: str, body_html: str,
                 *, time_str: str = "") -> str:
    """Top-of-page market-phase strip (PRE_OPEN / OPEN / AFTER_HOURS / WEEKEND).

    Flat surface with a single accent border + uppercase title. Body is HTML
    so callers can inline `<strong>` highlights. `time_str` is a small right-
    aligned timestamp.
    """
    bg = tint(phase_color, 0.06)
    time_html = (
        f'<div style="color:{TEXT.MUTED};font-size:{TYPE.XS}px;'
        f'white-space:nowrap;padding-top:{SPACE.XXS}px;{CSS_TABULAR_NUMS}">'
        f'{time_str}</div>'
    ) if time_str else ""
    return (
        f'<div style="background:{bg};border:1px solid {phase_color};'
        f'border-radius:{RADIUS.MD}px;padding:{SPACE.LG}px {SPACE.XL}px;'
        f'margin-bottom:{SPACE.LG}px;display:flex;align-items:flex-start;'
        f'gap:{SPACE.MD}px;">'
        f'  <div style="flex:1;">'
        f'    <div style="color:{phase_color};{CSS_UPPERCASE_LABEL}'
        f'                margin-bottom:{SPACE.XS}px;">{title}</div>'
        f'    <div style="font-size:{TYPE.BASE}px;color:{TEXT.TERTIARY};'
        f'                line-height:1.6;">{body_html}</div>'
        f'  </div>'
        f'  {time_html}'
        f'</div>'
    )


# ── Source health row + panel ────────────────────────────────────────────────

def source_health_row(name: str, ok: bool, msg: str) -> str:
    """One row inside the source-health panel."""
    dot_color = STATUS.SUCCESS if ok else STATUS.WARN
    return (
        f'<div style="display:flex;justify-content:space-between;'
        f'gap:{SPACE.LG}px;padding:{SPACE.SM}px 0;'
        f'border-bottom:1px solid {BORDER.SUBTLE};font-size:{TYPE.BASE}px;">'
        f'  <span style="color:{TEXT.SECONDARY};font-weight:600;">'
        f'    <span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{dot_color};margin-right:8px;'
        f'vertical-align:middle;"></span>{name}'
        f'  </span>'
        f'  <span style="color:{TEXT.TERTIARY};text-align:right;">{msg}</span>'
        f'</div>'
    )


def source_health_panel(rows: dict[str, dict]) -> str:
    """Whole `<details>` collapsible panel with one row per source.

    `rows` is the existing source_health dict: {name: {ok, msg, n_in, n_kept}}.
    """
    bad = [n for n, h in rows.items() if not h.get("ok")]
    accent = STATUS.WARN if bad else STATUS.SUCCESS
    label = (f"⚠ {len(bad)} source(s) degraded" if bad
             else "✓ All systems healthy")
    open_attr = " open" if bad else ""
    rows_html = "".join(
        source_health_row(name, h.get("ok", False), h.get("msg", ""))
        for name, h in rows.items()
    )
    return (
        f'<details{open_attr} style="background:{SURFACE.CARD};'
        f'border:1px solid {BORDER.SUBTLE};border-radius:{RADIUS.SM}px;'
        f'margin-bottom:{SPACE.LG}px;">'
        f'  <summary style="cursor:pointer;padding:{SPACE.MD}px {SPACE.LG}px;'
        f'font-weight:600;color:{TEXT.SECONDARY};font-size:{TYPE.MD}px;'
        f'list-style:none;user-select:none;">'
        f'    Signal source health — {label}'
        f'  </summary>'
        f'  <div style="padding:{SPACE.SM}px {SPACE.LG}px {SPACE.LG}px {SPACE.LG}px;">'
        f'    <div style="border-left:3px solid {accent};padding-left:{SPACE.LG}px;">'
        f'      {rows_html}'
        f'    </div>'
        f'  </div>'
        f'</details>'
    )


# ── Contract row (the "what to buy" sub-card on every play) ──────────────────

def contract_row(p: dict) -> str:
    """The 'Contract to buy' sub-card. Pulls from the play dict:
    contract_strike, contract_type, contract_expiry, contract_premium, contract_notes.
    Falls back to plain Entry/Target line if no contract block was attached.
    """
    if "contract_strike" not in p:
        return (
            f'<div style="display:flex;gap:{SPACE.XL}px;flex-wrap:wrap;'
            f'font-size:{TYPE.BASE}px;color:{TEXT.TERTIARY};">'
            f'  <span><b style="color:{TEXT.PRIMARY};">Entry:</b> '
            f'    ${p["entry"]:.2f}</span>'
            f'  <span><b style="color:{TEXT.PRIMARY};">Target:</b> '
            f'    ${p["target"]:.2f}</span>'
            f'</div>'
        )
    strike   = p["contract_strike"]
    ctype    = p["contract_type"]
    expiry   = p["contract_expiry"]
    prem     = float(p["contract_premium"])
    notes    = p.get("contract_notes", "")
    n_per_1k = int(1000 // prem) if prem > 0 else 0
    notes_html = (
        f'<div style="margin-top:{SPACE.SM}px;font-size:{TYPE.SM}px;'
        f'color:{TEXT.TERTIARY};">{notes}</div>'
    ) if notes else ""
    return (
        f'<div style="background:{SURFACE.SUBTLE};border-left:3px solid {STATUS.INFO};'
        f'border-radius:{RADIUS.MD}px;padding:{SPACE.LG}px {SPACE.LG}px;'
        f'margin-bottom:{SPACE.MD}px;">'
        f'  <div style="color:{STATUS.INFO};{CSS_UPPERCASE_LABEL}'
        f'              margin-bottom:{SPACE.SM}px;">📜 Contract to buy</div>'
        f'  <div style="display:grid;grid-template-columns:repeat(2,1fr);'
        f'              gap:{SPACE.SM}px {SPACE.XL}px;'
        f'              font-size:{TYPE.BASE}px;color:{TEXT.SECONDARY};">'
        f'    <div><b style="color:{STATUS.INFO};">Strike / type:</b> '
        f'      ${strike} {ctype}</div>'
        f'    <div><b style="color:{STATUS.INFO};">Expiry:</b> {expiry}</div>'
        f'    <div><b style="color:{STATUS.INFO};">Est. cost:</b> '
        f'      ~${prem:,.0f} per contract '
        f'      <span style="color:{TEXT.MUTED};">(${prem/100:.2f} × 100)</span></div>'
        f'    <div><b style="color:{STATUS.INFO};">Per $1k capital:</b> '
        f'      {n_per_1k} contract{"s" if n_per_1k != 1 else ""}</div>'
        f'  </div>'
        f'  {notes_html}'
        f'</div>'
    )


# ── Live trail-stop sub-card (0DTE Drop only) ────────────────────────────────

def trail_state_card(state: Optional[dict], trail_pct: float) -> str:
    """Render the 3-state trail-stop sub-card.

    `state` is the dict produced by app.py:_update_trail_state. If None,
    show a degraded "tracking unavailable" notice.
    """
    if state is None:
        return (
            f'<div style="background:{SURFACE.RAISED};border-left:3px solid {TEXT.MUTED};'
            f'border-radius:{RADIUS.MD}px;padding:{SPACE.MD}px {SPACE.LG}px;'
            f'margin-top:{SPACE.MD}px;">'
            f'  <div style="font-size:{TYPE.SM}px;color:{TEXT.MUTED};line-height:1.5;">'
            f'    ⚠ Live trail-stop tracking unavailable for this play '
            f'(no contract ticker or Polygon snapshot offline). '
            f'Manually exit when option drops {trail_pct:.0f}% from your peak.'
            f'  </div>'
            f'</div>'
        )

    entry = float(state["entry_price"])
    peak  = float(state["running_max"])
    last  = float(state["last_price"]) if state.get("last_price") else entry
    raw_stop = float(state["stop_level"]) if state.get("stop_level") else entry * (1 - trail_pct / 100.0)
    # Clamp displayed stop to the $0.01 OCC tick floor. Options cannot print
    # below $0.01, so any computed stop below that is meaningless. We still
    # use the raw_stop for the "is exit triggered" decision (that math is in
    # _update_trail_state); here we just guard the displayed number.
    stop  = max(raw_stop, 0.01)
    stop_at_floor = raw_stop <= 0.01
    pnl_pct  = (last - entry) / entry * 100.0 if entry > 0 else 0.0
    peak_pct = (peak - entry) / entry * 100.0 if entry > 0 else 0.0

    # When the entry itself is at the $0.01 floor, the trail-stop concept
    # has no usable downside. Surface that honestly.
    entry_floored_msg = ""
    if entry <= 0.01 + 1e-9:
        entry_floored_msg = (
            f" <b style='color:{STATUS.WARN};'>Note: entry was at the $0.01 floor; "
            f"there is no usable trail-stop protection at this premium. "
            f"Exit on judgment (or if option never recovers above ~$0.05).</b>"
        )

    # Pick state style
    if state["exited"]:
        style: StateStyle = DANGER_STYLE
        title = "🔴 EXIT TRIGGERED — SELL NOW"
        if state["exit_reason"] == "trail_stop_after_peak":
            msg = (f"Trail stop hit at <b>${state['exit_price']:.2f}</b> "
                   f"(peak was <b>${peak:.2f}</b>, trail {trail_pct:.0f}% = "
                   f"stop ${stop:.2f}). "
                   f"Realised: <b>{pnl_pct:+.0f}%</b> on this trade.")
        else:
            msg = (f"Initial stop hit at <b>${state['exit_price']:.2f}</b> — "
                   f"option never broke above entry ${entry:.2f}. "
                   f"Realised: <b>{pnl_pct:+.0f}%</b>. "
                   f"This is the failure mode the trail stop is designed to catch early.")
    elif state.get("fetch_error"):
        style = WARN_STYLE
        title = "🟡 TRACKING — quote pending"
        manual_stop = max(entry * (1 - trail_pct/100), 0.01)
        msg   = (f"Watching ${entry:.2f} entry. Live quote unavailable "
                 f"({state['fetch_error']}); will resume on next refresh. "
                 f"Manual stop = <b>${manual_stop:.2f}</b>."
                 f"{entry_floored_msg}")
    else:
        if stop_at_floor:
            style = WARN_STYLE
            title = "🟡 LIVE TRAILING — stop at $0.01 floor"
            msg   = (f"Effective stop = <b>$0.01</b> (the OCC tick floor — "
                    f"options cannot print below this). "
                    f"<b style='color:{STATUS.WARN};'>"
                    f"The trail-stop provides no real protection at this premium. "
                    f"Exit on judgment if the option fades.</b>{entry_floored_msg}")
        else:
            style = SUCCESS_STYLE
            title = "🟢 LIVE TRAILING — hold while price > stop"
            msg   = (f"HOLD while option > <b>${stop:.2f}</b>. "
                    f"SELL the moment price prints below ${stop:.2f}."
                    f"{entry_floored_msg}")

    fetched = state.get("last_quote_at", "—")
    return (
        f'<div style="background:{style.bg};border-left:3px solid {style.border};'
        f'border-radius:{RADIUS.MD}px;padding:{SPACE.MD}px {SPACE.LG}px;'
        f'margin-top:{SPACE.MD}px;">'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline;gap:{SPACE.LG}px;'
        f'              margin-bottom:{SPACE.SM}px;">'
        f'    <div style="color:{style.accent};{CSS_UPPERCASE_LABEL}">{title}</div>'
        f'    <div style="font-size:{TYPE.XS}px;color:{TEXT.MUTED};{CSS_TABULAR_NUMS}">'
        f'      last quote: {fetched}</div>'
        f'  </div>'
        f'  <div style="display:grid;grid-template-columns:repeat(4,1fr);'
        f'              gap:{SPACE.SM}px {SPACE.XL}px;'
        f'              font-size:{TYPE.BASE}px;color:{TEXT.SECONDARY};'
        f'              {CSS_TABULAR_NUMS}">'
        f'    <div><b style="color:{style.accent};">Entry:</b> ${entry:.2f}</div>'
        f'    <div><b style="color:{style.accent};">Peak:</b> ${peak:.2f} '
        f'      <span style="color:{TEXT.TERTIARY};">({peak_pct:+.0f}%)</span></div>'
        f'    <div><b style="color:{style.accent};">Now:</b> ${last:.2f} '
        f'      <span style="color:{TEXT.TERTIARY};">({pnl_pct:+.0f}%)</span></div>'
        f'    <div><b style="color:{style.accent};">Stop:</b> ${stop:.2f}</div>'
        f'  </div>'
        f'  <div style="margin-top:{SPACE.SM}px;font-size:{TYPE.SM}px;'
        f'              color:{TEXT.SECONDARY};line-height:1.5;">{msg}</div>'
        f'</div>'
    )


# ── Empty state (no plays available) ─────────────────────────────────────────

def empty_state(title: str, body_html: str) -> str:
    """Centered empty-state block. Replaces the gradient version."""
    return (
        f'<div style="background:{SURFACE.RAISED};border:1px solid {BORDER.DEFAULT};'
        f'border-radius:{RADIUS.LG}px;padding:{SPACE.XXL + SPACE.MD}px;'
        f'text-align:center;margin:{SPACE.LG}px 0;">'
        f'  <div style="font-size:{TYPE.XL}px;font-weight:800;color:{TEXT.PRIMARY};'
        f'              margin-bottom:{SPACE.SM}px;">{title}</div>'
        f'  <div style="color:{TEXT.TERTIARY};font-size:{TYPE.MD}px;line-height:1.5;">'
        f'    {body_html}'
        f'  </div>'
        f'</div>'
    )


# ── Top-of-page summary strip (4 KPI tiles) ──────────────────────────────────

def summary_strip_html(*, n_now: int, n_watch: int, best_edge: float,
                       best_label: str, sources: list[str]) -> str:
    """The 4-tile strip at the top of Today's Plays.

    Replaces the previous 4 gradient cards with a flat KPI row using
    metric_tile + a tone hint. Source list is rendered as comma-separated
    text for now; consider replacing with chips later.
    """
    sources_str = ", ".join(sources) if sources else "—"
    return (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
        f'gap:{SPACE.LG}px;margin-bottom:{SPACE.XL}px;">'
        f'{metric_tile("Trade Now",  str(n_now),    sub="live signals",        tone=STATUS.INFO)}'
        f'{metric_tile("Watch List", str(n_watch),  sub="approaching trigger", tone=STATUS.WARN)}'
        f'{metric_tile("Best Edge",  f"{best_edge:.1f}", sub=best_label,        tone=STATUS.SUCCESS)}'
        f'{metric_tile("Active Systems", str(len(sources)), sub=sources_str,    tone=SOURCE.MA_BOUNCE)}'
        f'</div>'
    )


# ── Play card (the workhorse) ────────────────────────────────────────────────

def play_card(p: dict, *, rank: int, first_seen_str: str, ago_str: str,
              now_str: str, is_fresh: bool, trail_html: str = "",
              doctrine_html: str = "") -> str:
    """The big play card on Today's Plays.

    Required play dict fields:
      source, ticker, tag, state, action, reason, win_rate, avg_ret, n,
      edge, horizon, entry, target.
    Optional:
      contract_strike (+ contract_type/expiry/premium/notes/ticker)
      doctrine_*       (MA Bounce smart-entry doctrine)

    External slots (passed in, not computed here):
      trail_html      — output of trail_state_card() for 0DTE Drop, else ""
      doctrine_html   — pre-rendered High-Conviction Doctrine block, else ""
      first_seen_str / ago_str / now_str — timestamp footer values
      is_fresh        — True ⇒ render the green "NEW" pill
    """
    accent, emoji = _STATE_STYLE.get(p["state"], (TEXT.TERTIARY, "•"))
    src_color, _ = _SOURCE_BADGE.get(p["source"], (TEXT.TERTIARY, ""))

    n_str = f"n={p['n']}" if p.get("n", 0) > 0 else "live"

    # Compute alert age in seconds for staleness pill
    # ago_str is like "2m 13s ago" or "12s ago" — parse for a numeric proxy.
    # Conservative: only mark STALE when ago_str starts with a number > 10 mins.
    is_stale = (
        ago_str.endswith(" ago") and
        ("h " in ago_str or
         (ago_str.startswith(("1", "2", "3", "4", "5", "6", "7", "8", "9")) and
          "m " in ago_str and
          ago_str[0].isdigit() and
          int(ago_str.split("m")[0].strip()) >= 10))
    )

    if is_fresh:
        age_pill = (
            f'<span style="background:{tint(STATUS.SUCCESS, 0.15)};'
            f'color:{STATUS.SUCCESS};font-size:{TYPE.XS}px;font-weight:800;'
            f'padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;'
            f'letter-spacing:.05em;text-transform:uppercase;'
            f'{CSS_TABULAR_NUMS}">🟢 NEW · {ago_str}</span>'
        )
    elif is_stale:
        age_pill = (
            f'<span style="background:{tint(STATUS.DANGER, 0.15)};'
            f'color:{STATUS.DANGER};font-size:{TYPE.XS}px;font-weight:800;'
            f'padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;'
            f'letter-spacing:.05em;text-transform:uppercase;'
            f'{CSS_TABULAR_NUMS}">🔴 STALE · {ago_str}</span>'
        )
    else:
        age_pill = (
            f'<span style="background:{tint(STATUS.WARN, 0.15)};'
            f'color:{STATUS.WARN};font-size:{TYPE.XS}px;font-weight:800;'
            f'padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;'
            f'letter-spacing:.05em;text-transform:uppercase;'
            f'{CSS_TABULAR_NUMS}">⏱ {ago_str}</span>'
        )

    return f"""
    <div style="background:{SURFACE.CARD};border:1px solid {accent};
                border-left:4px solid {accent};border-radius:{RADIUS.LG - 2}px;
                padding:{SPACE.XL}px;margin-bottom:{SPACE.LG}px;">
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;gap:{SPACE.LG}px;">
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:{SPACE.MD}px;
                      flex-wrap:wrap;margin-bottom:{SPACE.SM}px;">
            <span style="background:{SURFACE.RAISED};color:{TEXT.TERTIARY};
                         font-size:{TYPE.SM}px;font-weight:800;
                         padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;
                         {CSS_TABULAR_NUMS}">#{rank}</span>
            <span style="font-size:{TYPE.XL}px;font-weight:900;color:{TEXT.PRIMARY};">
              {emoji} {p['ticker']}</span>
            <span style="background:{tint(src_color, 0.13)};color:{src_color};
                         font-size:{TYPE.XS}px;font-weight:800;
                         padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;
                         letter-spacing:.04em;text-transform:uppercase;">
              {p['source']}</span>
            <span style="background:{tint(accent, 0.13)};color:{accent};
                         font-size:{TYPE.XS}px;font-weight:800;
                         padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;
                         letter-spacing:.04em;text-transform:uppercase;">
              {p['state']}</span>
            <span style="color:{TEXT.MUTED};font-size:{TYPE.SM}px;">{p['tag']}</span>
            {age_pill}
          </div>
          <div style="font-size:{TYPE.LG}px;font-weight:800;color:{accent};
                      margin-bottom:{SPACE.MD}px;">
            → {p['action']}
          </div>
          <div style="color:{TEXT.SECONDARY};font-size:{TYPE.MD - 1}px;
                      line-height:1.55;margin-bottom:{SPACE.MD}px;">
            {p['reason']}
          </div>
          {contract_row(p)}
          {trail_html}
          <div style="display:flex;gap:{SPACE.XL}px;flex-wrap:wrap;
                      font-size:{TYPE.BASE}px;color:{TEXT.TERTIARY};
                      margin-top:{SPACE.MD}px;{CSS_TABULAR_NUMS}">
            <span><b style="color:{STATUS.SUCCESS};">Win rate:</b> {p['win_rate']:.0f}%</span>
            <span><b style="color:{STATUS.SUCCESS};">Avg return:</b> +{p['avg_ret']:.2f}%</span>
            <span><b style="color:{TEXT.PRIMARY};">Sample:</b> {n_str}</span>
            <span><b style="color:{TEXT.PRIMARY};">Horizon:</b> {p['horizon']}</span>
            <span style="color:{TEXT.MUTED};">Underlying ${p['entry']:.2f} → ${p['target']:.2f}</span>
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0;min-width:110px;">
          <div style="color:{TEXT.MUTED};{CSS_UPPERCASE_LABEL}">Edge</div>
          <div style="font-size:{TYPE.XXL}px;font-weight:900;color:{accent};
                      line-height:1;{CSS_TABULAR_NUMS}">{p['edge']:.1f}</div>
          <div style="font-size:{TYPE.XS}px;color:{TEXT.MUTED};
                      margin-top:{SPACE.MD + 2}px;line-height:1.55;{CSS_TABULAR_NUMS}">
            <div>⏱ alert <span style="color:{TEXT.SECONDARY};font-weight:700;">{first_seen_str}</span></div>
            <div>↻ refresh <span style="color:{TEXT.SECONDARY};font-weight:700;">{now_str}</span></div>
          </div>
        </div>
      </div>
      {doctrine_html}
    </div>
    """


# ── Actionable hero (the BIG "DO THIS NOW" block at top of page) ────────────

def actionable_hero(p: dict, *, ago_seconds: float, now_str: str,
                    trail_pct: float = 15.0) -> str:
    """Big top-of-page banner for the most actionable play.

    Renders only when there's a play in an actionable state. Surfaces:
      - The exact buy instruction (ticker + strike + count for $500/$1k/$2k)
      - Validated stats so the user knows WHY (win rate, avg per trade)
      - The exit rule (trail-stop level)
      - A freshness indicator + entry-window warning

    `ago_seconds` is the # seconds since this alert was first seen this session.
    Used to render the "STALE — verify fill is still good" warning when > 600s.
    """
    src_color, _ = _SOURCE_BADGE.get(p["source"], (STATUS.INFO, ""))
    accent = src_color

    # Freshness badge
    if ago_seconds < 60:
        freshness_label = "🟢 FRESH"
        freshness_tone  = STATUS.SUCCESS
        freshness_msg   = "Alert just fired. Best fills are in the next 5–10 min."
    elif ago_seconds < 600:
        freshness_label = "🟡 ACTIVE"
        freshness_tone  = STATUS.WARN
        freshness_msg   = (f"Alert is {int(ago_seconds // 60)}m old. Still tradeable, "
                           f"but spreads may have widened.")
    else:
        freshness_label = "🔴 STALE"
        freshness_tone  = STATUS.DANGER
        freshness_msg   = (f"Alert is {int(ago_seconds // 60)}m old. Entry edge has "
                           f"degraded. Verify the option price hasn't already run.")

    # Position-size table at $500 / $1,000 / $2,000 budgets.
    # `prem` is the per-share premium (= contract_premium / 100). The play
    # builder in app.py uses display_entry_price (≈ current quote) so this
    # is the realistic per-share fill price. Clamped to the $0.01 OCC floor
    # because option contracts cannot trade below $0.01 anywhere on US exchanges.
    prem  = max(float(p.get("contract_premium", 0)) / 100.0, 0.01)
    budgets = [500, 1_000, 2_000]
    sizing_cells = "".join(
        f'<div style="text-align:center;padding:{SPACE.SM}px {SPACE.MD}px;'
        f'background:{tint(accent, 0.08)};border-radius:{RADIUS.SM}px;">'
        f'  <div style="color:{TEXT.MUTED};font-size:{TYPE.XS}px;'
        f'              text-transform:uppercase;letter-spacing:.05em;'
        f'              margin-bottom:2px;">${b:,} budget</div>'
        f'  <div style="font-size:{TYPE.LG}px;font-weight:900;color:{accent};'
        f'              {CSS_TABULAR_NUMS}">{int(b // (prem * 100))} ct</div>'
        f'  <div style="color:{TEXT.MUTED};font-size:{TYPE.XS}px;'
        f'              {CSS_TABULAR_NUMS}">'
        f'    ${int(b // (prem * 100)) * prem * 100:,.0f} cost</div>'
        f'</div>'
        for b in budgets
    )

    # Strike / contract details
    strike_str = ""
    if "contract_strike" in p:
        strike_str = (f'${p["contract_strike"]} {p["contract_type"]} · '
                      f'exp {p["contract_expiry"]}')

    # Sell-trigger price. Floor at $0.01 (the OCC tick floor — no option can
    # print below $0.01). When the trail-stop math would put the trigger at
    # or below the floor, the trail-stop concept is meaningless: the option
    # has either decayed to the floor (effectively zero) or you're trying to
    # protect a $0.01 entry which has no usable downside.
    raw_sell_price = prem * (1 - trail_pct / 100.0)
    sell_price = max(raw_sell_price, 0.01)
    sell_at_floor = raw_sell_price <= 0.01
    if sell_at_floor:
        sell_msg = (f"<span style='color:{STATUS.WARN};'>"
                    f"⚠ trail-stop floored at $0.01 (no real protection at this "
                    f"premium — exit on judgment if option fades).</span>")
    else:
        sell_msg = ""

    # Win-rate / edge stats from the play
    wr = float(p.get("win_rate", 0))
    n  = int(p.get("n", 0))

    # Risk-per-contract math:
    # - trail_loss_dollars: dollar loss if the trail-stop fires at sell_price
    # - worst_loss_dollars: dollar loss if option goes straight to the $0.01 floor
    # We surface BOTH so the user understands their realistic vs worst-case
    # exposure per contract.
    trail_loss_dollars = (prem - sell_price) * 100   # always >= 0 because sell_price <= prem
    worst_loss_dollars = max((prem - 0.01) * 100, 0)
    sell_trigger_html  = (
        f'below ${sell_price:.2f} ({trail_pct:.0f}% trail-stop)'
        if not sell_at_floor else
        f'$0.01 floor — judgment exit only'
    )

    return (
        f'<div style="background:{tint(accent, 0.08)};border:2px solid {accent};'
        f'border-radius:{RADIUS.LG}px;padding:{SPACE.XL}px {SPACE.XXL}px;'
        f'margin-bottom:{SPACE.XL}px;box-shadow:0 4px 24px rgba(0,0,0,.4);">'
        # Top row: TAKE THIS NOW + freshness + clock
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:center;flex-wrap:wrap;gap:{SPACE.MD}px;'
        f'              margin-bottom:{SPACE.MD}px;">'
        f'    <div style="display:flex;align-items:center;gap:{SPACE.MD}px;'
        f'                flex-wrap:wrap;">'
        f'      <span style="font-size:{TYPE.XS}px;color:{accent};font-weight:900;'
        f'                   text-transform:uppercase;letter-spacing:.12em;">'
        f'        ▶ TAKE THIS NOW</span>'
        f'      <span style="background:{tint(freshness_tone, 0.15)};'
        f'                   color:{freshness_tone};'
        f'                   font-size:{TYPE.XS}px;font-weight:800;'
        f'                   padding:3px {SPACE.MD}px;border-radius:{RADIUS.SM}px;'
        f'                   text-transform:uppercase;letter-spacing:.05em;">'
        f'        {freshness_label}</span>'
        f'    </div>'
        f'    <div style="color:{TEXT.MUTED};font-size:{TYPE.SM}px;{CSS_TABULAR_NUMS}">'
        f'      ↻ {now_str}'
        f'    </div>'
        f'  </div>'
        # Action line (huge)
        f'  <div style="font-size:{TYPE.XXL + 4}px;font-weight:900;'
        f'              color:{TEXT.PRIMARY};line-height:1.15;'
        f'              margin-bottom:{SPACE.SM}px;{CSS_TABULAR_NUMS}">'
        f'    {p["ticker"]} <span style="color:{accent};">→ {p["action"]}</span>'
        f'  </div>'
        f'  <div style="color:{TEXT.SECONDARY};font-size:{TYPE.MD}px;'
        f'              line-height:1.5;margin-bottom:{SPACE.LG}px;">'
        f'    {strike_str}'
        f'  </div>'
        # Three-up sizing strip
        f'  <div style="display:grid;grid-template-columns:repeat(3,1fr);'
        f'              gap:{SPACE.MD}px;margin-bottom:{SPACE.LG}px;">'
        f'    {sizing_cells}'
        f'  </div>'
        # Stats grid (2x2). Validated avg = +$401/trade is the strategy-level
        # average from the cap-$1 + leverage-bonus picker over a 90-day
        # backtest at $500/trade. NOT this play's specific est_gain_pct,
        # which is noisy. The cohort average is honest expected value.
        f'  <div style="display:grid;grid-template-columns:1fr 1fr;'
        f'              gap:{SPACE.MD}px {SPACE.XL}px;'
        f'              padding:{SPACE.MD}px {SPACE.LG}px;'
        f'              background:{SURFACE.SUBTLE};border-radius:{RADIUS.SM}px;'
        f'              font-size:{TYPE.BASE}px;color:{TEXT.SECONDARY};'
        f'              {CSS_TABULAR_NUMS}">'
        f'    <div><b style="color:{STATUS.SUCCESS};">Validated win rate:</b> '
        f'      {wr:.0f}% (n={n}, 90d backtest)</div>'
        f'    <div><b style="color:{STATUS.SUCCESS};">Validated strategy avg:</b> '
        f'      +$401/trade on $500 budget</div>'
        f'    <div><b style="color:{STATUS.DANGER};">SELL trigger:</b> '
        f'      {sell_trigger_html}</div>'
        f'    <div><b style="color:{STATUS.DANGER};">Risk per ct:</b> '
        f'      −${trail_loss_dollars:.0f} on stop · '
        f'      <span style="color:{TEXT.MUTED};">'
        f'      −${worst_loss_dollars:.0f} max if option → $0.01</span></div>'
        f'  </div>'
        f'  {sell_msg}'
        # Freshness explanation
        f'  <div style="margin-top:{SPACE.MD}px;font-size:{TYPE.SM}px;'
        f'              color:{freshness_tone};line-height:1.5;">'
        f'    {freshness_msg}'
        f'  </div>'
        f'</div>'
    )


__all__ = [
    "metric_tile", "phase_banner",
    "source_health_row", "source_health_panel",
    "contract_row", "trail_state_card", "empty_state",
    "summary_strip_html", "play_card", "actionable_hero",
]
