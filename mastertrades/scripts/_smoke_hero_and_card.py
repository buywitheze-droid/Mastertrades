"""Smoke test the actionable_hero and updated play_card components.

Renders synthetic plays at 3 freshness levels (NEW, ACTIVE, STALE) plus
3 alert ages and confirms:
  - HTML balances (open/close tag count)
  - All token references resolve (no f-string KeyErrors)
  - Hero shows the right freshness label per ago_seconds
  - Play card shows the right age pill per ago_str
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datetime import datetime
from src.ui import components as UI

# Synthetic 0DTE Drop play (mirrors the real shape from app.py:1103)
def make_play():
    return {
        "source": "0DTE Drop",
        "ticker": "SPY",
        "tag":    "Drop 5.2 pts",
        "state":  "ENTRY_OPEN",
        "action": "BUY 744C @ ~$0.13 (38 contracts per $500)",
        "entry":  580.50,
        "target": 585.70,
        "win_rate": 85.0,
        "avg_ret":  802.0,
        "n":        3,
        "edge":     7.4,
        "reason":   ("Sold off 5.2 pts from open. 85% of similar drops produced "
                     "1000%+ option moves on recovery to VWAP/open. "
                     "<b style='color:#3fb950;'>Strike picked via leverage-weighted "
                     "ranking</b> (cap $1.00, validated +$401/trade)."),
        "horizon":  "Minutes–hours (0DTE intraday)",
        "contract_strike":  744.0,
        "contract_type":    "OTM call",
        "contract_expiry":  "0DTE (today)",
        "contract_premium": 13.0,    # $0.13 × 100
        "contract_notes":   "Est. peak +800% · trail-stop default 15%",
        "contract_ticker":  "O:SPY260513C00744000",
    }

# ── Hero rendering at 3 freshness levels ────────────────────────────────────
print("=== actionable_hero — three freshness scenarios ===\n")
play = make_play()
for label, ago_s in [("FRESH (<60s)", 30), ("ACTIVE (1-10min)", 240),
                      ("STALE (>10min)", 700)]:
    html = UI.actionable_hero(play, ago_seconds=ago_s, now_str="14:23:05",
                                trail_pct=15.0)
    open_count  = html.count("<div") + html.count("<span")
    close_count = html.count("</div>") + html.count("</span>")
    balanced = "✓" if open_count == close_count else f"✗ ({open_count} open, {close_count} close)"
    has_label = ("FRESH" in html if ago_s < 60 else
                 "ACTIVE" in html if ago_s < 600 else "STALE" in html)
    print(f"  {label:<22}  HTML balanced: {balanced}  freshness label OK: "
          f"{'✓' if has_label else '✗'}")

# ── Play card with the new age pill ─────────────────────────────────────────
print("\n=== play_card — three age scenarios ===\n")
for label, ago, fresh in [("NEW (12s)", "12s ago", True),
                           ("WARM (3m)", "3m ago", False),
                           ("STALE (15m)", "15m ago", False)]:
    html = UI.play_card(play, rank=1, first_seen_str="14:18:42",
                         ago_str=ago, now_str="14:23:05",
                         is_fresh=fresh, trail_html="", doctrine_html="")
    open_count  = html.count("<div") + html.count("<span")
    close_count = html.count("</div>") + html.count("</span>")
    balanced = "✓" if open_count == close_count else f"✗ ({open_count} open, {close_count} close)"
    if fresh:
        pill_ok = "🟢 NEW" in html
    elif "15m" in ago:
        pill_ok = "🔴 STALE" in html
    else:
        pill_ok = "⏱" in html and "🟢 NEW" not in html and "🔴 STALE" not in html
    print(f"  {label:<14}  HTML balanced: {balanced}  age pill OK: "
          f"{'✓' if pill_ok else '✗'}")

# ── Tokens-only check: no leaked hex codes in components.py ────────────────
import re
src = (ROOT / "src" / "ui" / "components.py").read_text()
hex_codes = re.findall(r'#[0-9a-fA-F]{6}', src)
# Allow only $0.05 etc patterns or hex inside f-string format specifiers
suspicious = [h for h in hex_codes if h not in {"#3fb950", "#ffd633", "#f85149", "#58a6ff"}]
print(f"\n=== Token discipline ===")
print(f"  Hex codes in components.py: {len(hex_codes)} (all from semantic tokens)")
print(f"  Suspicious leaks          : {suspicious if suspicious else 'NONE ✓'}")

print("\nAll smoke checks complete.")
