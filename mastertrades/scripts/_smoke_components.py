"""Render every component once with synthetic data. Catches f-string errors,
missing dict keys, and broken HTML structure (open tags without close).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui import components as C
from src.ui import tokens as T

# 1. metric_tile
html = C.metric_tile("Trade Now", "3", sub="live signals", tone=T.STATUS.INFO)
assert "Trade Now" in html and "live signals" in html

# 2. phase_banner
html = C.phase_banner(T.STATUS.SUCCESS, "MARKET OPEN", "Decision window open",
                      time_str="14:32 ET")
assert "MARKET OPEN" in html and "14:32 ET" in html

# 3. source_health_panel
panel = C.source_health_panel({
    "MA Bounce":  {"ok": True,  "msg": "5 setups, 3 actionable"},
    "ML Jackpot": {"ok": False, "msg": "ERROR: ConnectionError"},
})
assert "MA Bounce" in panel and "ML Jackpot" in panel
assert "<details open" in panel  # because ML Jackpot is bad

# 4. contract_row
row = C.contract_row({
    "contract_strike": 744, "contract_type": "OTM call",
    "contract_expiry": "0DTE (today)", "contract_premium": 13.0,
    "contract_notes": "Est. peak +600%",
})
assert "744" in row and "OTM call" in row

# 5. trail_state_card — three states
for state in [None,
              {"entry_price": 0.13, "running_max": 0.66, "last_price": 0.50,
               "stop_level": 0.561, "exited": False, "fetch_error": None},
              {"entry_price": 0.13, "running_max": 0.66, "last_price": 0.40,
               "stop_level": 0.561, "exited": True,
               "exit_reason": "trail_stop_after_peak", "exit_price": 0.40,
               "last_quote_at": "14:46:18"}]:
    card = C.trail_state_card(state, 15.0)
    assert "<div" in card

# 6. empty_state
empty = C.empty_state("No actionable plays right now",
                      "Patience is a position. Check back in 30 min.")
assert "No actionable plays" in empty

# 7. summary_strip_html
strip = C.summary_strip_html(n_now=3, n_watch=1, best_edge=42.7,
                             best_label="SPY · 0DTE Drop",
                             sources=["0DTE Drop", "MA Bounce"])
assert "Trade Now" in strip and "Best Edge" in strip and "42.7" in strip

# 8. play_card — full play with trail
play = {
    "source": "0DTE Drop", "ticker": "SPY", "tag": "Drop 5.2 pts",
    "state": "ENTRY_OPEN", "action": "BUY 744C @ ~$0.13",
    "entry": 730.50, "target": 735.70,
    "win_rate": 54.0, "avg_ret": 220.0, "n": 3, "edge": 42.7,
    "reason": "Sold off 5.2 pts from open. 54% similar drops produced 1000%+ moves.",
    "horizon": "Minutes-hours (0DTE intraday)",
    "contract_strike": 744, "contract_type": "OTM call",
    "contract_expiry": "0DTE (today)", "contract_premium": 13.0,
    "contract_notes": "Est. peak +600% on recovery to open",
    "contract_ticker": "O:SPY260513C00744000",
}
trail = C.trail_state_card(
    {"entry_price": 0.13, "running_max": 0.66, "last_price": 0.50,
     "stop_level": 0.561, "exited": False, "fetch_error": None,
     "last_quote_at": "14:46:18"},
    15.0)
card = C.play_card(play, rank=1,
                   first_seen_str="14:32:08", ago_str="14m ago",
                   now_str="14:46:21", is_fresh=False, trail_html=trail)

# Sanity: roughly balanced div tags
opens  = card.count("<div")
closes = card.count("</div>")
print(f"play_card: opens={opens}, closes={closes}")
assert opens == closes, f"unbalanced divs: {opens} open vs {closes} close"

# Hex audit: components.py should NOT contain any literal hex outside tokens.py
import pathlib
src = (pathlib.Path("src") / "ui" / "components.py").read_text(encoding="utf-8")
hexes = sorted(set(re.findall(r"#[0-9a-fA-F]{6}\b", src)))
print(f"\nLiteral hex in components.py: {hexes}")
assert not hexes, f"components.py leaks hex tokens: {hexes}"

print("\nAll components smoke-tested OK.")
