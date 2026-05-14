"""Comprehensive audit of $0.01 floor handling across the dashboard.

Checks every place where a price/sell-trigger/sizing number is computed:
  1. recommend_strikes() populates display_entry_price from realistic fields.
  2. actionable_hero floors sell-trigger at $0.01 with honest message.
  3. trail_state_card floors stop at $0.01 with honest message.
  4. Sizing calculations don't produce absurd contract counts when entry = $0.01.
  5. play_card displays the realistic premium (display_entry_price), not the
     raw day_low spike.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.options_scanner import (
    OptionContract, recommend_strikes, MAX_PREMIUM_USD,
)
from src.ui import components as UI


def print_section(title):
    print(f"\n=== {title} ===")


# ── 1) display_entry_price uses day_close/open, NOT the day_low spike ──────
print_section("Test 1: display_entry_price ignores day_low spike")
chain = [
    # SPY 744 — day_low spiked to $0.01, but day_close (current) = $0.13
    OptionContract(ticker="O:SPY260513C00744000", contract_type="call",
                   strike=744.0, expiration="2026-05-13",
                   day_open=0.20, day_high=0.30, day_low=0.01, day_close=0.13,
                   day_volume=1000, implied_vol=0.25, delta=0.18,
                   gamma=0.05, theta=-0.10, vega=0.05, open_interest=500),
    # SPY 745 — never traded above $0.05, day_close = $0.05
    OptionContract(ticker="O:SPY260513C00745000", contract_type="call",
                   strike=745.0, expiration="2026-05-13",
                   day_open=0.05, day_high=0.05, day_low=0.01, day_close=0.05,
                   day_volume=200, implied_vol=0.25, delta=0.10,
                   gamma=0.04, theta=-0.10, vega=0.05, open_interest=300),
    # SPY 743 (closer ITM)
    OptionContract(ticker="O:SPY260513C00743000", contract_type="call",
                   strike=743.0, expiration="2026-05-13",
                   day_open=0.50, day_high=0.70, day_low=0.10, day_close=0.40,
                   day_volume=800, implied_vol=0.25, delta=0.30,
                   gamma=0.05, theta=-0.10, vega=0.05, open_interest=400),
]
recs = recommend_strikes(underlying_open=748.0, underlying_low=742.0,
                          contracts=chain)
print(f"  {'strike':>7} {'day_low':>8} {'day_close':>10} "
      f"{'est_entry':>10} {'display_entry':>14} {'display_gain':>13}")
for r in recs:
    c = next(c for c in chain if c.strike == r.strike)
    print(f"  {r.strike:>7.0f} ${c.day_low:>6.2f} ${c.day_close:>8.2f} "
          f"${r.est_entry_price:>8.2f} ${r.display_entry_price:>12.2f} "
          f"{r.display_gain_pct:>+12.0f}%")

# Assertions
for r in recs:
    c = next(c for c in chain if c.strike == r.strike)
    assert r.display_entry_price >= 0.01, f"display_entry below floor for {r.strike}"
    assert r.display_entry_price >= max(c.day_close, c.day_open), \
        f"display_entry should be >= day_close/day_open for {r.strike}"
    assert r.est_entry_price == max(c.day_low, 0.01), \
        f"est_entry_price should still use day_low for ranking math"
print(f"  ✓ display_entry_price always uses realistic current price")
print(f"  ✓ est_entry_price (for ranking) still uses day_low (matches backtest)")

# ── 2) Sell-trigger floored at $0.01 in actionable_hero ────────────────────
print_section("Test 2: actionable_hero clamps sell-trigger at $0.01")
def synth_play(prem_dollars):
    """Build a play with contract_premium = prem_dollars (per contract)."""
    return {
        "source": "0DTE Drop", "ticker": "SPY",
        "tag": "Drop 5pt", "state": "ENTRY_OPEN",
        "action": f"BUY 744C @ ${prem_dollars/100:.2f}",
        "entry": 743.0, "target": 748.0,
        "win_rate": 85.0, "avg_ret": 800.0, "n": 3,
        "edge": 7.0,
        "reason": "test", "horizon": "0DTE",
        "contract_strike": 744, "contract_type": "OTM call",
        "contract_expiry": "0DTE",
        "contract_premium": prem_dollars,
        "contract_notes": "test",
        "contract_ticker": "O:SPY260513C00744000",
    }

scenarios = [
    ("$0.01 entry  (penny floor)", 1.0,    "FLOOR expected"),
    ("$0.05 entry  (penny range)", 5.0,    "$0.04 sell expected"),
    ("$0.13 entry  (cheap OTM)",   13.0,   "$0.11 sell expected"),
    ("$0.50 entry  (mid-tier)",    50.0,   "$0.42 sell expected"),
    ("$1.00 entry  (cap edge)",    100.0,  "$0.85 sell expected"),
]
for label, prem_d, expected in scenarios:
    p = synth_play(prem_d)
    html = UI.actionable_hero(p, ago_seconds=30, now_str="14:23:05",
                                trail_pct=15.0)
    has_floor_msg = "$0.01 floor" in html or "floored at $0.01" in html
    has_below_001 = "below $0.00" in html or "below $-" in html
    expected_floor = prem_d <= 1.0  # ≤ $0.01 per share
    floor_ok = (has_floor_msg == expected_floor)
    no_subpenny = not has_below_001
    print(f"  {label:<35} floor msg: {'YES' if has_floor_msg else 'no '}  "
          f"sub-penny price displayed: {'NO ' if no_subpenny else 'YES'}  "
          f"{'✓' if floor_ok and no_subpenny else '✗'}")

# ── 3) Sizing math doesn't produce absurd contract counts ──────────────────
print_section("Test 3: sizing math sane at all premium levels")
for label, prem_d, _ in scenarios:
    p = synth_play(prem_d)
    html = UI.actionable_hero(p, ago_seconds=30, now_str="14:23:05",
                                trail_pct=15.0)
    # Pull contracts-per-$500 out of the HTML
    import re
    m = re.search(r'(\d+) ct</div>\s*<div[^>]*>\s*\$([\d,]+) cost', html)
    if m:
        ct, cost = int(m.group(1)), int(m.group(2).replace(",", ""))
        prem_per_share = max(prem_d / 100.0, 0.01)
        expected_ct = int(500 // (prem_per_share * 100))
        ok = ct == expected_ct and ct < 100_000  # sanity ceiling
        print(f"  {label:<35} $500 budget → {ct:>5} contracts "
              f"(${cost:>4} cost) {'✓' if ok else '✗'}")

# ── 4) trail_state_card clamps stop display at $0.01 ───────────────────────
print_section("Test 4: trail_state_card clamps stop display at $0.01")
synthetic_states = [
    ("Healthy: $0.50 entry, $0.80 peak, $0.65 now", {
        "entry_price": 0.50, "running_max": 0.80, "last_price": 0.65,
        "stop_level": 0.68, "exited": False, "fetch_error": None,
    }, False),
    ("Floored: $0.01 entry, $0.01 peak", {
        "entry_price": 0.01, "running_max": 0.01, "last_price": 0.01,
        "stop_level": 0.0085, "exited": False, "fetch_error": None,
    }, True),
    ("Exited: trail fired at $0.40", {
        "entry_price": 0.10, "running_max": 0.50, "last_price": 0.40,
        "stop_level": 0.425, "exited": True, "exit_reason": "trail_stop_after_peak",
        "exit_price": 0.40, "fetch_error": None,
    }, False),
]
for label, state, expect_floor_msg in synthetic_states:
    html = UI.trail_state_card(state, trail_pct=15.0)
    has_floor_msg = "floored at $0.01" in html or "no usable trail-stop" in html
    has_subpenny = ("$0.0085" in html or "$0.008" in html or "$0.00" in html)
    floor_ok = (has_floor_msg == expect_floor_msg)
    print(f"  {label:<55} floor msg: {'YES' if has_floor_msg else 'no '}  "
          f"sub-penny shown: {'NO ' if not has_subpenny else 'YES'}  "
          f"{'✓' if floor_ok and not has_subpenny else '✗'}")

# ── 5) play_card uses realistic premium for the contract sub-card ──────────
print_section("Test 5: contract_row uses realistic premium")
p = synth_play(13.0)  # $0.13 entry
html = UI.contract_row(p)
assert "$0.13" in html, f"contract_row should show $0.13 per share. HTML: {html}"
assert "$13" in html or "~$13" in html, f"contract_row should show $13 per contract"
print(f"  ✓ contract_row shows $0.13/share and $13/contract correctly")

print(f"\nAll floor-edge audits complete.")
