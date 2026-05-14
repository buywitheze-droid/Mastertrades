"""Minute-resolution truth: did the trail-stop kill any winners?

For each AMBIGUOUS trade from _no_stop_vs_trail.py, fetch Polygon minute bars
for both the option contract and the underlying. Determine:
  1. The minute the underlying hit its intraday low (alert moment).
  2. The fill minute = alert_minute + 10 minutes (10-min execution delay).
  3. The fill price = option price at that minute (use 'open' of the bar).
  4. Walk forward minute-by-minute:
       - Track running peak option price.
       - If price <= peak * (1 - TRAIL_PCT/100), trail-stop FIRES at that minute.
       - Otherwise, hold to close.
  5. Compare to: hold-to-close (no stop), peak-sell (theoretical max).

This gives us the ACTUAL answer to "did the trail-stop exit before a recovery?"
"""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import json
import requests
import pandas as pd

# Load env (for POLYGON_API_KEY)
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("POLYGON_API_KEY"):
    print("ERROR: POLYGON_API_KEY not set in environment or .env file.")
    sys.exit(1)

TRAIL_PCT       = 15.0
ENTRY_DELAY_MIN = 10  # minutes after the underlying low

amb_path = ROOT / "data" / "ambiguous_trades_for_minute_check.csv"
if not amb_path.exists():
    print(f"ERROR: {amb_path} not found. Run _no_stop_vs_trail.py first.")
    sys.exit(1)
amb = pd.read_csv(amb_path)
print(f"Loaded {len(amb)} ambiguous trades to resolve at minute resolution.\n")

# Polygon helpers
BASE = "https://api.polygon.io"
_LAST = 0.0
def _get(path, params=None):
    global _LAST
    elapsed = time.monotonic() - _LAST
    if elapsed < 0.12:
        time.sleep(0.12 - elapsed)
    p = dict(params or {})
    p["apiKey"] = os.environ["POLYGON_API_KEY"]
    _LAST = time.monotonic()
    r = requests.get(f"{BASE}{path}", params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_minute_bars(ticker: str, date: str) -> list[dict]:
    """Fetch 1-min bars for `ticker` on `date` (YYYY-MM-DD).
    Returns list of {ts_et: 'HH:MM', open, high, low, close, volume}."""
    path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 5000})
    bars = []
    for r in data.get("results", []) or []:
        ts_ms = int(r.get("t", 0))
        # Convert UTC -> America/New_York (UTC-4 in DST, ET market hours)
        ts_utc = dt.datetime.utcfromtimestamp(ts_ms / 1000)
        ts_et  = ts_utc - dt.timedelta(hours=4)  # EDT during late Apr / May
        bars.append({
            "ts_et": ts_et.strftime("%H:%M"),
            "minute_idx": ts_et.hour * 60 + ts_et.minute,
            "open":  float(r.get("o", 0) or 0),
            "high":  float(r.get("h", 0) or 0),
            "low":   float(r.get("l", 0) or 0),
            "close": float(r.get("c", 0) or 0),
            "volume": int(r.get("v", 0) or 0),
        })
    return bars


def trail_simulate(opt_bars: list[dict], fill_idx: int, fill_price: float,
                   trail: float = TRAIL_PCT/100) -> dict:
    """Walk forward from fill_idx and apply trail-stop. Return exit details."""
    if not opt_bars or fill_idx is None:
        return {"exit_minute": None, "exit_price": fill_price, "trigger": "none"}
    peak = fill_price
    peak_minute = fill_idx
    for bar in opt_bars:
        if bar["minute_idx"] < fill_idx:
            continue
        # Check intra-bar drawdown first (low of this bar)
        stop_level = peak * (1 - trail)
        if bar["low"] <= stop_level:
            return {
                "exit_minute": bar["ts_et"],
                "exit_price": stop_level,
                "trigger": f"trail (peak ${peak:.2f} @ minute {peak_minute})",
                "peak": peak,
            }
        # Update peak to bar's high if it's a new high
        if bar["high"] > peak:
            peak = bar["high"]
            peak_minute = bar["minute_idx"]
    # No stop trigger — held to close
    return {
        "exit_minute": opt_bars[-1]["ts_et"],
        "exit_price": opt_bars[-1]["close"],
        "trigger": "close (no stop)",
        "peak": peak,
    }


# ── Process each ambiguous trade ────────────────────────────────────────────
results = []
print(f"{'date':<12} {'tkr':<5} {'fill':>6} {'trail exit':>10} {'no-stop':>9} {'peak':>8} | verdict")
print("-" * 110)
for _, row in amb.iterrows():
    date = str(row["date"])
    underlying_ticker = str(row["ticker"])
    contract = str(row["contract"])
    print(f"  fetching {underlying_ticker}/{contract} on {date}...", end="", flush=True)

    try:
        und_bars = fetch_minute_bars(underlying_ticker, date)
        opt_bars = fetch_minute_bars(contract, date)
    except Exception as e:
        print(f" FAIL: {e}")
        continue

    if not und_bars or not opt_bars:
        print(f" no bars (und={len(und_bars)}, opt={len(opt_bars)})")
        continue

    # Find underlying intraday low minute (RTH only — 9:30 to 16:00)
    rth_und = [b for b in und_bars if 570 <= b["minute_idx"] < 960]
    if not rth_und:
        print(f" no RTH underlying bars")
        continue
    low_bar = min(rth_und, key=lambda b: b["low"])
    alert_minute = low_bar["minute_idx"]
    fill_minute  = alert_minute + ENTRY_DELAY_MIN

    # Find option price at fill_minute (use bar.open of that minute, or interpolate)
    fill_bar = next((b for b in opt_bars if b["minute_idx"] == fill_minute), None)
    if fill_bar is None:
        # Fallback: nearest later bar
        fill_bar = next((b for b in opt_bars if b["minute_idx"] >= fill_minute), None)
    if fill_bar is None:
        print(f" no option bar at fill minute")
        continue
    fill_price = max(fill_bar["open"], 0.01)

    # Run trail-stop sim from fill minute
    trail_res = trail_simulate(opt_bars, fill_minute, fill_price)

    # Comparison metrics
    last_bar = opt_bars[-1]
    no_stop_exit  = last_bar["close"]                     # hold to close
    peak_exit     = max(b["high"] for b in opt_bars
                        if b["minute_idx"] >= fill_minute)  # theoretical max

    pct_trail = (trail_res["exit_price"] / fill_price - 1) * 100
    pct_close = (no_stop_exit / fill_price - 1) * 100
    pct_peak  = (peak_exit  / fill_price - 1) * 100

    # Verdict
    if pct_trail > pct_close + 5:
        verdict = "trail HELPED (+{:.0f}% vs no-stop)".format(pct_trail - pct_close)
    elif pct_close > pct_trail + 5:
        verdict = "trail HURT  (-{:.0f}% vs no-stop) ← stop killed a winner".format(pct_close - pct_trail)
    else:
        verdict = "tie"

    results.append({
        "date": date, "ticker": underlying_ticker, "contract": contract,
        "alert_min": low_bar["ts_et"], "fill_min": fill_bar["ts_et"],
        "fill_price": fill_price,
        "trail_exit": trail_res["exit_price"], "trail_trigger": trail_res["trigger"],
        "trail_exit_min": trail_res["exit_minute"],
        "close_exit": no_stop_exit, "peak_exit": peak_exit,
        "pct_trail": pct_trail, "pct_close": pct_close, "pct_peak": pct_peak,
        "verdict": verdict,
    })

    print(f"\r{date:<12} {underlying_ticker:<5} ${fill_price:>5.2f} "
          f"${trail_res['exit_price']:>9.2f} ${no_stop_exit:>8.2f} ${peak_exit:>7.2f} | {verdict}")

# ── Summary ─────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
print(f"\n=== Minute-truth verdict over {len(res_df)} ambiguous trades ===\n")
if res_df.empty:
    print("No data — check API key / network.")
    sys.exit(0)

helpd = res_df[res_df["verdict"].str.startswith("trail HELPED")]
hurt  = res_df[res_df["verdict"].str.startswith("trail HURT")]
tie   = res_df[res_df["verdict"] == "tie"]
print(f"  trail HELPED : {len(helpd)} trades ({len(helpd)/len(res_df)*100:.0f}%)")
print(f"  trail HURT   : {len(hurt)} trades ({len(hurt)/len(res_df)*100:.0f}%)  ← user's concern")
print(f"  tie          : {len(tie)} trades")

# Aggregate at $500/trade
POS = 500.0
res_df["ct"] = (POS // (res_df["fill_price"] * 100)).astype(int)
for col, src in [("$_trail", "trail_exit"), ("$_close", "close_exit"), ("$_peak", "peak_exit")]:
    res_df[col] = (res_df[src] - res_df["fill_price"]) * res_df["ct"] * 100

print(f"\n=== Aggregate $ comparison ($500/trade across {len(res_df)} trades) ===")
print(f"  NO STOP (current — hold to close): ${res_df['$_close'].sum():>+11,.0f}")
print(f"  15% TRAIL-STOP                   : ${res_df['$_trail'].sum():>+11,.0f}")
print(f"  PEAK SELL (theoretical max)      : ${res_df['$_peak'].sum():>+11,.0f}")

# Detail on HURT trades (the user's concern)
if not hurt.empty:
    print(f"\n=== Trades where trail-stop killed a winner ===")
    for _, r in hurt.iterrows():
        print(f"  {r['date']} {r['ticker']:<4} contract {r['contract']}")
        print(f"    Alert at {r['alert_min']}, filled at {r['fill_min']} for ${r['fill_price']:.2f}")
        print(f"    Trail-stop fired at {r['trail_exit_min']} for ${r['trail_exit']:.2f} ({r['pct_trail']:+.0f}%)")
        print(f"    Held to close: ${r['close_exit']:.2f} ({r['pct_close']:+.0f}%)")
        print(f"    → {r['verdict']}")

# Save full result
out_path = ROOT / "data" / "minute_truth_results.csv"
res_df.to_csv(out_path, index=False)
print(f"\n→ Full results saved to {out_path}")
