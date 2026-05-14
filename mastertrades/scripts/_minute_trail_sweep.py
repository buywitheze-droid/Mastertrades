"""Sweep trail-stop percentage at minute resolution to find the optimum."""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import pandas as pd
import requests

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ENTRY_DELAY_MIN = 10
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

def fetch_min(t, d):
    data = _get(f"/v2/aggs/ticker/{t}/range/1/minute/{d}/{d}",
                {"adjusted": "true", "sort": "asc", "limit": 5000})
    bars = []
    for r in data.get("results", []) or []:
        ts_utc = dt.datetime.utcfromtimestamp(int(r["t"]) / 1000)
        ts_et  = ts_utc - dt.timedelta(hours=4)
        bars.append({"min_idx": ts_et.hour*60 + ts_et.minute,
                     "open": float(r["o"]), "high": float(r["h"]),
                     "low":  float(r["l"]), "close": float(r["c"])})
    return bars

def trail_sim(opt_bars, fill_idx, fill_price, trail):
    peak = fill_price
    for b in opt_bars:
        if b["min_idx"] < fill_idx: continue
        if b["low"] <= peak * (1 - trail):
            return peak * (1 - trail)
        if b["high"] > peak: peak = b["high"]
    return opt_bars[-1]["close"]

amb = pd.read_csv(ROOT / "data" / "ambiguous_trades_for_minute_check.csv")
print(f"Loading minute bars for {len(amb)} contracts...")
trades = []
for _, row in amb.iterrows():
    date = str(row["date"]); tk = str(row["ticker"]); ct = str(row["contract"])
    try:
        und = fetch_min(tk, date)
        opt = fetch_min(ct, date)
    except Exception:
        continue
    if not und or not opt: continue
    rth = [b for b in und if 570 <= b["min_idx"] < 960]
    if not rth: continue
    low_b = min(rth, key=lambda b: b["low"])
    fill_min = low_b["min_idx"] + ENTRY_DELAY_MIN
    fill_bar = next((b for b in opt if b["min_idx"] >= fill_min), None)
    if not fill_bar: continue
    fill = max(fill_bar["open"], 0.01)
    close = opt[-1]["close"]
    peak = max(b["high"] for b in opt if b["min_idx"] >= fill_min)
    trades.append({"date": date, "tk": tk, "ct": ct,
                   "opt": opt, "fill_idx": fill_min, "fill": fill,
                   "close": close, "peak": peak})
print(f"Got {len(trades)} clean trades.\n")

POS = 500.0
trail_pcts = [10, 12, 15, 18, 20, 25, 30, 40, 50]
print(f"=== Trail-stop sweep ({len(trades)} trades, $500 each, {ENTRY_DELAY_MIN}-min fill delay) ===\n")
print(f"  {'trail %':>8} | {'total $':>10} {'avg/trade':>10} {'win %':>7} {'killed winners':>15}")
print(f"  " + "-" * 65)

# Baseline: no stop
ctd = [int(POS // (t['fill']*100)) for t in trades]
no_stop_pnl = sum((t['close'] - t['fill']) * c * 100
                   for t, c in zip(trades, ctd))
peak_pnl    = sum((t['peak']  - t['fill']) * c * 100
                   for t, c in zip(trades, ctd))
print(f"  {'NO STOP':>8} | ${no_stop_pnl:>+9,.0f} ${no_stop_pnl/len(trades):>+9,.0f}    -   {'-':>15}")

for pct in trail_pcts:
    f = pct / 100
    pnls, wins, killed = [], 0, 0
    for t, c in zip(trades, ctd):
        if c == 0: continue
        ex = trail_sim(t["opt"], t["fill_idx"], t["fill"], f)
        pnl = (ex - t["fill"]) * c * 100
        pnls.append(pnl)
        if pnl > 0: wins += 1
        # Killed winner = stop fired AND close was higher than stop level
        if ex < t["close"] - 0.01 and t["close"] > t["fill"]:
            killed += 1
    total = sum(pnls)
    print(f"  {pct:>7}% | ${total:>+9,.0f} ${total/len(pnls):>+9,.0f} "
          f"{wins/len(pnls)*100:>5.0f}%   {killed:>15}")

print(f"\n  PEAK SELL (theoretical max, unattainable): ${peak_pnl:>+9,.0f} "
      f"(${peak_pnl/len(trades):>+,.0f}/trade)")
