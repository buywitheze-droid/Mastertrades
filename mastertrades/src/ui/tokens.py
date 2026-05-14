"""Mastertrades design tokens — semantic, namespaced, immutable.

Why this exists:
  Pre-2026-05-13 audit, app.py contained 74 unique hex colors, 21 unique
  font-size values, 9 unique border-radii, and 964 inline `style=` attributes.
  Eight different greens claimed to mean "success". Every play card had
  drifted into its own subtly-different visual style.

Rule going forward:
  Any new # + hex color, font-size, or radius that isn't in this file is a
  bug. If you need a new shade, add it here and use the constant. If you
  catch yourself reaching for `style="color: #c9d1d9"`, replace with
  `style=f"color: {TEXT.SECONDARY}"`.

Audit: re-run the count snippets in
  scripts/_audit_visual_budget.py
periodically to make sure the budget stays under control.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Tinted-overlay helper ─────────────────────────────────────────────────────
# A LOT of the pre-refactor noise was hand-typed `rgba(...)` overlays. This
# converts a hex token to an `rgba(...)` string so callers don't reinvent the
# math each time. Use sparingly — flat solid surfaces are usually clearer.

def tint(hex_color: str, opacity: float) -> str:
    """Return an `rgba(r, g, b, opacity)` string from a `#rrggbb` hex.
    Opacity must be in [0, 1]. Caller decides when a tint is appropriate."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"tint() expects #rrggbb, got: {hex_color!r}")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0,1], got: {opacity}")
    return f"rgba({r},{g},{b},{opacity:.3f})"


# ── Surfaces ──────────────────────────────────────────────────────────────────
# Three-tier elevation: PAGE (deepest) → CARD → RAISED (sub-card on a card).
# Borrowed from GitHub Primer dark theme so it composes with Streamlit's
# default dark mode without surprise contrast jumps.
class SURFACE:
    PAGE          = "#0c1117"     # body background
    CARD          = "#0d1117"     # primary card background
    RAISED        = "#161b22"     # raised tile / metric background
    SUBTLE        = "#0a1428"     # sub-card on a card (e.g. contract row)


class BORDER:
    DEFAULT       = "#30363d"     # standard 1px border
    SUBTLE        = "#21262d"     # divider line, hairline
    STRONG        = "#3a4047"     # higher-contrast border (used sparingly)


# ── Text ──────────────────────────────────────────────────────────────────────
# 4 levels covers every use case. PRIMARY for headings/values, SECONDARY for
# body, TERTIARY for labels, MUTED for footnotes/timestamps.
class TEXT:
    PRIMARY       = "#f0f6fc"
    SECONDARY     = "#c9d1d9"
    TERTIARY      = "#8b949e"
    MUTED         = "#6e7681"
    INVERSE       = "#0c1117"     # for use ON light/colored backgrounds


# ── Semantic action colors ────────────────────────────────────────────────────
# THE ONLY four colors with semantic meaning. Use these, not their cousins.
# - SUCCESS for "trade is winning / signal fired", positive P&L
# - WARN    for "approaching threshold / data settling / partial degradation"
# - DANGER  for "EXIT NOW / loss / source down"
# - INFO    for "neutral status / informational badges"
class STATUS:
    SUCCESS       = "#3fb950"     # green
    WARN          = "#d29922"     # amber
    DANGER        = "#f85149"     # red
    INFO          = "#58a6ff"     # blue (also primary ACCENT)
    GOLD          = "#ffd633"     # reserved for ULTRA JACKPOT only


# ── Source-identity colors (badges only, never as semantic signal) ────────────
# These tag WHICH algo produced a play, not whether you should trust it.
# Purposely chosen to NOT overlap with STATUS so a player never confuses
# "MA Bounce purple" with "warning amber".
class SOURCE:
    MA_BOUNCE     = "#6e40c9"     # purple
    ML_JACKPOT    = "#1f6feb"     # deep blue (distinct from STATUS.INFO)
    GAP_FILL      = "#0e8c87"     # teal
    DTE_DROP      = "#bf3989"     # magenta


# ── Type scale (px) ───────────────────────────────────────────────────────────
# 7 sizes max. If you reach for an 8th, you're inventing instead of choosing.
class TYPE:
    XS            = 10            # micro labels, timestamps
    SM            = 11            # secondary metadata
    BASE          = 12            # default body / card content
    MD            = 14            # primary content / button labels
    LG            = 18            # card titles / "BUY {strike}C" actions
    XL            = 22            # page-level KPI values
    XXL           = 28            # hero edge score, big stat
    DISPLAY       = 32            # used ONCE per page max

    FAMILY = (
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Helvetica, Arial, sans-serif"
    )


# ── Border radii (px) ────────────────────────────────────────────────────────
# 3 radii. SM for pills/badges, MD for cards, LG for hero blocks.
class RADIUS:
    SM            = 6
    MD            = 10
    LG            = 14


# ── Spacing (px) ─────────────────────────────────────────────────────────────
# Use the named values; avoid arbitrary `padding: 7px 13px` style drift.
class SPACE:
    XXS           = 2
    XS            = 4
    SM            = 6
    MD            = 8
    LG            = 12
    XL            = 18
    XXL           = 24


# ── Pre-derived state styles (DRY for the most-used combinations) ────────────
@dataclass(frozen=True)
class StateStyle:
    """Bundle of bg + border + accent for a single semantic state."""
    bg:     str
    border: str
    accent: str
    label_color: str = ""        # falls back to accent if empty

    @property
    def text_on_accent(self) -> str:
        return self.label_color or self.accent


# State styles consumed by play_card, callouts, and trail-stop sub-card.
# Filled-in values come from tinted overlays so they're cheap to recolor en
# masse later if the brand shifts.
SUCCESS_STYLE = StateStyle(
    bg=tint(STATUS.SUCCESS, 0.06), border=STATUS.SUCCESS, accent=STATUS.SUCCESS,
)
WARN_STYLE = StateStyle(
    bg=tint(STATUS.WARN, 0.06), border=STATUS.WARN, accent=STATUS.WARN,
)
DANGER_STYLE = StateStyle(
    bg=tint(STATUS.DANGER, 0.06), border=STATUS.DANGER, accent=STATUS.DANGER,
)
INFO_STYLE = StateStyle(
    bg=tint(STATUS.INFO, 0.05), border=STATUS.INFO, accent=STATUS.INFO,
)
NEUTRAL_STYLE = StateStyle(
    bg=SURFACE.RAISED, border=BORDER.DEFAULT, accent=TEXT.TERTIARY,
)


# ── Convenience CSS snippets ─────────────────────────────────────────────────
# Common style blocks that recur across components. Each is a plain string so
# components can interpolate them with f-strings cheaply.
CSS_TABULAR_NUMS    = "font-variant-numeric: tabular-nums;"
CSS_UPPERCASE_LABEL = (
    f"font-size:{TYPE.XS}px; font-weight:800; "
    f"text-transform:uppercase; letter-spacing:.08em;"
)


__all__ = [
    "SURFACE", "BORDER", "TEXT", "STATUS", "SOURCE",
    "TYPE", "RADIUS", "SPACE",
    "StateStyle",
    "SUCCESS_STYLE", "WARN_STYLE", "DANGER_STYLE", "INFO_STYLE", "NEUTRAL_STYLE",
    "tint", "CSS_TABULAR_NUMS", "CSS_UPPERCASE_LABEL",
]
