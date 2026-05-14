"""Mastertrades UI primitives.

Two tiny modules:
  tokens.py     — semantic colors, type scale, radii, spacing. Source of truth
                  for every visual decision.
  components.py — reusable HTML render helpers (play_card, metric_tile,
                  phase_banner, source_health_row, trail_state_card).

Rule: any new # + hex color, font size, or border-radius added directly to
app.py is technical debt. Add it to tokens.py instead.
"""
from . import tokens, components  # noqa: F401
