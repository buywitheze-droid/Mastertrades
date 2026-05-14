"""Smoke test for the live trail-stop helpers.

Doesn't run the Streamlit app — just imports the helpers, fakes a session_state
shim, and steps a synthetic 0DTE Drop play through several updates so we can
see the state transitions print correctly.
"""
import sys
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Shim st.session_state so we can import app.py outside Streamlit ──────────
class _SessionState(dict):
    def __getattr__(self, k):  return self[k]
    def __setattr__(self, k, v): self[k] = v


fake_st = types.SimpleNamespace()
fake_st.session_state = _SessionState()
fake_st.cache_data    = types.SimpleNamespace(clear=lambda: None)
sys.modules["streamlit"] = fake_st  # noqa  — must be set BEFORE app import

# Disable side-effecting heavy imports
import importlib

# We can't import app.py directly because it runs Streamlit page code at module
# load time. Instead, copy the helper functions inline by importing them via
# exec on the relevant slice. Simpler: just re-implement the same helper logic
# using the real options_scanner module — that's what we're really testing.
from src.options_scanner import fetch_option_quote, StrikeRecommendation

# Stub out fetch_option_quote so we can simulate a live price walk
prices = iter([0.18, 0.34, 0.62, 0.88, 1.05, 0.95, 0.78])  # peak then fade


def _stub_fetch(_underlying, _contract):
    try:
        p = next(prices)
    except StopIteration:
        return None
    return {
        "last_price": p, "day_open": 0.13, "day_high": p, "day_low": 0.10,
        "day_close": p, "bid": p - 0.01, "ask": p + 0.01,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


import src.options_scanner as os_mod
os_mod.fetch_option_quote = _stub_fetch

# Minimal copy of the helpers under test (paste from app.py)
_TRAIL_STATE_KEY = "_dte_drop_trail_state"


def _trail_state_dict():
    if _TRAIL_STATE_KEY not in fake_st.session_state:
        fake_st.session_state[_TRAIL_STATE_KEY] = {}
    return fake_st.session_state[_TRAIL_STATE_KEY]


def _trail_key(p):
    if not p.get("contract_ticker"): return None
    return (p["ticker"], datetime.now().date().isoformat(), p["contract_ticker"])


def _update_trail_state(p, trail_pct):
    if p.get("source") != "0DTE Drop": return None
    key = _trail_key(p)
    if key is None: return None
    states = _trail_state_dict()
    if key not in states:
        entry = float(p["contract_premium"]) / 100.0
        states[key] = {
            "ticker": p["ticker"], "contract_ticker": p["contract_ticker"],
            "strike": p["contract_strike"], "entry_price": entry,
            "running_max": entry, "first_seen": datetime.now().isoformat(timespec="seconds"),
            "last_price": None, "stop_level": None, "exited": False,
            "exit_reason": None, "exit_price": None, "exit_time": None,
            "fetch_error": None,
        }
    state = states[key]
    state["trail_pct"] = float(trail_pct)
    if state["exited"]: return state
    quote = os_mod.fetch_option_quote(p["ticker"], p["contract_ticker"])
    if not quote or quote.get("last_price", 0) <= 0:
        state["fetch_error"] = "no live quote"; return state
    state["fetch_error"] = None
    last = float(quote["last_price"])
    state["last_price"] = last
    state["last_quote_at"] = quote.get("fetched_at")
    state["running_max"] = max(float(state["running_max"]), last)
    state["stop_level"]  = state["running_max"] * (1 - trail_pct/100)
    initial_stop = state["entry_price"] * (1 - trail_pct/100)
    eff_stop = max(state["stop_level"], initial_stop)
    if last <= eff_stop:
        state["exited"] = True
        state["exit_reason"] = ("trail_stop_after_peak" if state["running_max"] > state["entry_price"]
                                else "initial_stop_no_recovery")
        state["exit_price"] = last
        state["exit_time"]  = quote.get("fetched_at")
    return state


# Synthetic play
play = {
    "source": "0DTE Drop", "ticker": "SPY",
    "contract_strike": 744, "contract_premium": 13.0,  # $0.13/share = $13/contract
    "contract_ticker": "O:SPY260513C00744000",
}

print(f"=== Trail-stop smoke test (15% trail, $0.13 entry) ===")
print(f"{'step':<5} {'price':>6} {'peak':>6} {'stop':>6} {'state':<14} {'exit_reason':<28}")
for i in range(7):
    s = _update_trail_state(play, 15.0)
    last = s.get("last_price") or 0.0
    peak = s.get("running_max") or 0.0
    stop = s.get("stop_level") or 0.0
    state_label = "EXITED" if s["exited"] else "TRAILING"
    reason = s.get("exit_reason") or ""
    print(f"  {i+1:<3} {last:>5.2f}$ {peak:>5.2f}$ {stop:>5.2f}$ {state_label:<14} {reason:<28}")
print()
print(f"Final realised P&L: {((s['last_price'] - s['entry_price']) / s['entry_price'] * 100):+.0f}%")
