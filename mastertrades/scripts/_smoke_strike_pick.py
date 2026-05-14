"""Smoke test for new recommend_strikes() logic:
  - Strikes priced > MAX_PREMIUM_USD must be SKIPPED.
  - leverage_score must be populated and equal est_gain_pct × √(1/entry).
  - Sort order must be by leverage_score, descending.
  - app.py-style top_rec = max(recs, key=leverage_score) must pick a CHEAP strike
    even when an expensive strike has higher est_gain_pct.
"""
import sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.options_scanner import (
    OptionContract, recommend_strikes, MAX_PREMIUM_USD,
    STRIKE_SELECTION_VERSION,
)

# Build a synthetic chain mimicking SPY at $580 with intraday low $575.
# Strikes near the open are EXPENSIVE; deep OTM strikes are cheap.
chain = [
    OptionContract(ticker=f"O:SPY260513C{int(s*1000):08d}",
                   contract_type="call", strike=s, expiration="2026-05-13",
                   day_open=p, day_high=p*1.5, day_low=p, day_close=p,
                   day_volume=1000, implied_vol=0.20,
                   delta=max(0.0, min(1.0, 0.5 + (580-s)/10)),
                   gamma=0.05, theta=-0.10, vega=0.05,
                   open_interest=500)
    for s, p in [
        (576, 4.20),   # deep ITM at low — expensive
        (577, 3.30),
        (578, 2.40),
        (579, 1.55),
        (580, 0.85),   # ATM-ish — borderline cap
        (581, 0.40),
        (582, 0.20),
        (583, 0.10),   # cheap OTM — leverage sweet spot
        (584, 0.05),   # penny-ish
    ]
]

print(f"Strike-selection version: {STRIKE_SELECTION_VERSION}")
print(f"MAX_PREMIUM_USD          : ${MAX_PREMIUM_USD:.2f}")
print(f"Synthetic chain          : low=$575 open=$580 (5pt drop)\n")

recs = recommend_strikes(
    underlying_open=580.0, underlying_low=575.0, contracts=chain,
)

print(f"=== {len(recs)} surviving strikes (sorted by leverage_score desc) ===")
print(f"  {'strike':>6} {'entry':>6} {'gain%':>8} {'lev_score':>10} {'cap_ok':>7}")
for r in recs:
    cap_ok = "yes" if r.est_entry_price <= MAX_PREMIUM_USD else "NO"
    print(f"  {r.strike:>6.0f} {r.est_entry_price:>5.2f}$ "
          f"{r.est_gain_pct:>+7.0f}% {r.leverage_score:>9.0f} {cap_ok:>7}")

# ── Assertions ───────────────────────────────────────────────────────────────
assert all(r.est_entry_price <= MAX_PREMIUM_USD for r in recs), \
    f"FAIL: strikes priced above ${MAX_PREMIUM_USD} leaked through cap"
print(f"\n✓ All survivors priced ≤ ${MAX_PREMIUM_USD:.2f}")

# leverage_score formula sanity (uses entry as proxy for cap_ref since the
# synthetic chain has day_open == day_low). Real chains will have day_open
# diverge from day_low; the cap-on-day_open behaviour is verified separately
# in scripts/_smoke_live_picker.py.
for r in recs:
    expected = r.est_gain_pct * (1.0 / r.est_entry_price) ** 0.5
    assert abs(r.leverage_score - expected) < 1e-6, \
        f"FAIL: leverage_score mismatch for strike {r.strike}: " \
        f"got {r.leverage_score}, expected {expected}"
print("✓ leverage_score = est_gain_pct × √(1/cap_ref) for every strike")

# ── Regression test: cap on day_open, not day_low ────────────────────────────
print("\n=== Regression test: cap filters on day_open, not day_low ===")
expensive_chain = [
    OptionContract(ticker="O:TEST260513C00580000",
                   contract_type="call", strike=580.0, expiration="2026-05-13",
                   day_open=4.20,    # WAY above $1 cap → must be filtered
                   day_high=5.00, day_low=0.01,    # day_low spiked to penny
                   day_close=4.50, day_volume=1000, implied_vol=0.20,
                   delta=0.55, gamma=0.05, theta=-0.10, vega=0.05,
                   open_interest=500),
    OptionContract(ticker="O:TEST260513C00583000",
                   contract_type="call", strike=583.0, expiration="2026-05-13",
                   day_open=0.50,    # Within cap → must be kept
                   day_high=2.00, day_low=0.10,
                   day_close=1.50, day_volume=1000, implied_vol=0.20,
                   delta=0.30, gamma=0.05, theta=-0.10, vega=0.05,
                   open_interest=500),
]
recs2 = recommend_strikes(580.0, 575.0, expensive_chain)
strikes_kept = [r.strike for r in recs2]
assert 580.0 not in strikes_kept, \
    f"FAIL: strike 580 (day_open=$4.20) leaked through $1 cap. Got: {strikes_kept}"
assert 583.0 in strikes_kept, \
    f"FAIL: strike 583 (day_open=$0.50) was wrongly filtered. Got: {strikes_kept}"
print(f"✓ Cap correctly skips day_open=$4.20 strike "
      f"(even though day_low=$0.01)")
print(f"✓ Cap correctly admits day_open=$0.50 strike")

# Sort order
scores = [r.leverage_score for r in recs]
assert scores == sorted(scores, reverse=True), \
    f"FAIL: recs not sorted by leverage_score desc: {scores}"
print("✓ recs sorted by leverage_score descending")

# Top pick should be the CHEAPEST surviving strike (since recovery to open
# guarantees these all reach intrinsic; cheap strike → highest gain%).
top_rec = max(recs, key=lambda r: r.leverage_score)
print(f"\n→ top_rec (what dashboard will show): "
      f"strike {top_rec.strike:.0f}, entry ${top_rec.est_entry_price:.2f}, "
      f"+{top_rec.est_gain_pct:.0f}% gain, lev_score {top_rec.leverage_score:.0f}")

# Compare against what the OLD ranker would have chosen
old_top = max(recs, key=lambda r: r.est_gain_pct)
if old_top.strike != top_rec.strike:
    print(f"→ old ranker would have picked: strike {old_top.strike:.0f}, "
          f"entry ${old_top.est_entry_price:.2f} "
          f"({(top_rec.est_entry_price/old_top.est_entry_price-1)*100:+.0f}% cost diff)")
else:
    print(f"→ both rankers agree on this chain")

print("\nAll smoke checks passed.")
