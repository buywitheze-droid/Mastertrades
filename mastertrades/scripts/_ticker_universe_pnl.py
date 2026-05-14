"""Phase-2 ticker validation: realistic $500/trade P&L with the live picker.

For each candidate ticker that survived the recon (data/ticker_universe_recon.csv),
walk the last 90 days of daily OHLC and:
  1. Identify each drop-day candidate (matches the live algo's ENTRY_OPEN gate).
  2. Fetch the option chain for that day at offsets +1 to +9 above the low.
  3. Run the live recommend_strikes() picker on the chain.
  4. Apply $500/trade with 15% trail-stop.
  5. Aggregate total P&L, win rate, and per-trade economics.

The output is a head-to-head table comparing each ticker to the SPY/QQQ/IWM
baseline so we can recommend ADD / SKIP decisions confidently.

This is the EXPENSIVE phase — fetches one option-chain call per drop day per
ticker. With 20 drop days per ticker × 8 tickers, that's ~160 chain fetches.
Polygon's free tier rate-limits to 5/min so this can take 30+ minutes; on the
paid plan it runs in ~3-4 minutes.
"""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import requests
import pandas as pd

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("POLYGON_API_KEY"):
    print("ERROR: POLYGON_API_KEY not set.")
    sys.exit(1)

from src.options_scanner import OptionContract, recommend_strikes

POSITION_USD  = 500.0
TRAIL_PCT     = 15.0
LOOKBACK_DAYS = 120
MIN_DROP_PTS_PCT = 0.005   # 0.5% of underlying — replaces fixed 3pt threshold

# The live algo uses a fixed $1.00 premium cap, tuned for SPY/QQQ/IWM at
# ~$300-700 underlying. For a fair cross-ticker test we scale the cap by
# underlying price so high-priced tickers (TSLA, META, SMH) can also compete.
# Calibration: $1.00 cap at SPY ($688) ≈ 0.145% of underlying.
# We use 0.20% (slightly looser) with a $0.40 floor and $5.00 ceiling.
def price_scaled_cap(underlying_price: float) -> float:
    return min(max(underlying_price * 0.0020, 0.40), 5.00)

# Pull the top-ranked candidates from phase 1
recon_csv = ROOT / "data" / "ticker_universe_recon.csv"
if not recon_csv.exists():
    print(f"Run scripts/_ticker_universe_recon.py first.")
    sys.exit(1)
recon = pd.read_csv(recon_csv).sort_values("score", ascending=False)

# Test the top 12 candidates that have recovery_rate >= 20% (baseline floor)
TEST_TICKERS = recon[recon["recovery_rate_pct"] >= 20].head(15)["ticker"].tolist()
# Always include the baseline so we can compare apples-to-apples
for t in ["SPY", "QQQ", "IWM"]:
    if t not in TEST_TICKERS:
        TEST_TICKERS.append(t)
print(f"=== Phase-2 P&L validation · {len(TEST_TICKERS)} tickers ===")
print(f"  Tickers: {', '.join(TEST_TICKERS)}\n")

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


def fetch_underlying_daily(ticker: str, days: int = 120) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    data = _get(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                {"adjusted": "true", "sort": "asc", "limit": 5000})
    rows = []
    for r in data.get("results", []) or []:
        ts = dt.datetime.utcfromtimestamp(int(r["t"]) / 1000).date()
        rows.append({
            "date": ts, "open": float(r["o"]), "high": float(r["h"]),
            "low":  float(r["l"]), "close": float(r["c"]),
        })
    return pd.DataFrame(rows)


def fetch_option_daily_for_strike(option_ticker: str, day: dt.date) -> dict | None:
    """Fetch one day's OHLC for a single option contract. Returns None if no data."""
    s = day.isoformat()
    try:
        data = _get(f"/v2/aggs/ticker/{option_ticker}/range/1/day/{s}/{s}",
                    {"adjusted": "true", "limit": 1})
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return {"open": float(r["o"]), "high": float(r["h"]),
            "low":  float(r["l"]), "close": float(r["c"])}


def list_strikes_for_day(ticker: str, day: dt.date,
                          low: float, high: float) -> list[float]:
    """List call strikes available for `ticker` expiring on `day`, in the
    [low - 1, high + 9] band. Uses Polygon's expired-contracts endpoint."""
    iso = day.isoformat()
    try:
        data = _get("/v3/reference/options/contracts", {
            "underlying_ticker": ticker.upper(),
            "expiration_date": iso,
            "contract_type": "call",
            "strike_price.gte": low - 1,
            "strike_price.lte": high + 9,
            "limit": 1000,
            "expired": "true",
        })
    except Exception:
        return []
    return sorted({float(c["strike_price"]) for c in data.get("results", [])
                   if c.get("strike_price")})


def trail_exit(entry, high, low, close, trail=TRAIL_PCT/100):
    if high > entry:
        peak_stop = high * (1 - trail)
        return max(close, peak_stop) if close < peak_stop else close
    init_stop = entry * (1 - trail)
    return init_stop if low <= init_stop else close


# ── Main loop ───────────────────────────────────────────────────────────────
all_trades = []
ticker_stats = []

for tk in TEST_TICKERS:
    print(f"  {tk}: fetching underlying...", end="", flush=True)
    try:
        und = fetch_underlying_daily(tk, days=LOOKBACK_DAYS)
    except Exception as e:
        print(f" FAIL: {e}")
        continue
    if und.empty or len(und) < 30:
        print(f" not enough underlying data")
        continue

    # Identify drop-days where the live algo would fire ENTRY_OPEN.
    # Use a percentage-based threshold so smaller-priced tickers (XLE, ARKK,
    # TLT, etc.) can also qualify — a fixed 3pt threshold would never fire
    # for a $50 ETF.
    und["drop_pts"] = und["open"] - und["low"]
    und["drop_pct"] = und["drop_pts"] / und["open"]
    und["entry_open_candidate"] = (und["drop_pct"] >= MIN_DROP_PTS_PCT)
    candidates = und[und["entry_open_candidate"]].copy()
    print(f" {len(und)} days, {len(candidates)} drop candidates...", end="", flush=True)

    # For each candidate day, fetch the option ladder
    n_chains_fetched   = 0  # day had at least one strike listed in the band
    n_chains_with_data = 0  # at least one strike had actual OHLC bars
    n_picks_made       = 0  # recommend_strikes returned at least one rec
    n_trades_taken     = 0  # contracts > 0
    pnls               = []
    # Sample diagnostics: track first few days that failed at each stage
    fail_reasons = {"no_strikes": 0, "no_bars": 0, "no_recs": 0, "zero_contracts": 0}

    # Cap per-ticker chain fetches to keep runtime tractable. We sample the
    # 30 most-recent drop days, which is plenty to estimate per-ticker stats.
    sampled = candidates.sort_values("date", ascending=False).head(30)

    for _, day in sampled.iterrows():
        d = day["date"]
        strikes = list_strikes_for_day(tk, d,
                                         low=day["low"], high=day["high"])
        if not strikes:
            fail_reasons["no_strikes"] += 1
            continue
        n_chains_fetched += 1

        # Fetch each strike's daily OHLC; build OptionContract list
        chain = []
        for s in strikes:
            opt_str = f"{int(s*1000):08d}"
            otk = f"O:{tk}{d.strftime('%y%m%d')}C{opt_str}"
            bars = fetch_option_daily_for_strike(otk, d)
            if not bars:
                continue
            chain.append(OptionContract(
                ticker=otk, contract_type="call", strike=float(s),
                expiration=d.isoformat(),
                day_open=bars["open"], day_high=bars["high"],
                day_low=bars["low"],   day_close=bars["close"],
                day_volume=1000, implied_vol=0.30,
                delta=0.5, gamma=0.05, theta=-0.10, vega=0.05,
                open_interest=500,
            ))
        if not chain:
            fail_reasons["no_bars"] += 1
            continue
        n_chains_with_data += 1

        # Use a price-scaled cap so a TSLA call at $400 underlying can compete
        # alongside a SPY call at $688. Without this, the $1.00 default cap
        # filters out every OTM strike on tickers above ~$300.
        recs = recommend_strikes(
            underlying_open=float(day["open"]),
            underlying_low=float(day["low"]),
            contracts=chain,
            max_premium_usd=price_scaled_cap(float(day["open"])),
        )
        if not recs:
            fail_reasons["no_recs"] += 1
            continue
        n_picks_made += 1
        top = max(recs, key=lambda r: r.leverage_score)
        match = next(c for c in chain if c.strike == top.strike)
        # Sizing on the realistic display entry (matches live algo)
        fill = max(top.display_entry_price, 0.01)
        contracts = int(POSITION_USD // (fill * 100))
        if contracts == 0:
            fail_reasons["zero_contracts"] += 1
            continue
        exit_p = trail_exit(fill, match.day_high, match.day_low, match.day_close)
        pnl = (exit_p - fill) * contracts * 100
        n_trades_taken += 1
        pnls.append(pnl)
        all_trades.append({
            "ticker": tk, "date": d, "strike": top.strike,
            "fill": fill, "exit": exit_p,
            "contracts": contracts, "pnl": pnl,
        })

    if not pnls:
        ticker_stats.append({"ticker": tk, "drop_days": len(candidates),
                              "trades_taken": 0, "total_pnl": 0,
                              "avg_pnl": 0, "win_rate": 0,
                              "best": 0, "worst": 0,
                              "fail_reasons": fail_reasons})
        # Show why nothing tradeable came out
        why_parts = []
        if fail_reasons["no_strikes"]:
            why_parts.append(f"{fail_reasons['no_strikes']} no-chain")
        if fail_reasons["no_bars"]:
            why_parts.append(f"{fail_reasons['no_bars']} no-bars")
        if fail_reasons["no_recs"]:
            why_parts.append(f"{fail_reasons['no_recs']} all-too-pricey")
        if fail_reasons["zero_contracts"]:
            why_parts.append(f"{fail_reasons['zero_contracts']} unaffordable")
        why = ", ".join(why_parts) if why_parts else "no reason recorded"
        print(f" no tradeable picks ({why})")
        continue

    pnl_arr = pd.Series(pnls)
    stats = {
        "ticker":           tk,
        "drop_days":        len(candidates),
        "sampled_days":     len(sampled),
        "chains_OK":        n_chains_fetched,
        "chains_with_data": n_chains_with_data,
        "picks_made":       n_picks_made,
        "trades_taken":     n_trades_taken,
        "total_pnl":        float(pnl_arr.sum()),
        "avg_pnl":          float(pnl_arr.mean()),
        "win_rate":         float((pnl_arr > 0).mean() * 100),
        "best":             float(pnl_arr.max()),
        "worst":            float(pnl_arr.min()),
        "profit_factor":    float(pnl_arr[pnl_arr > 0].sum() /
                                    max(abs(pnl_arr[pnl_arr < 0].sum()), 1e-9)),
    }
    ticker_stats.append(stats)
    print(f" took {n_trades_taken}/{len(sampled)} trades, "
          f"${stats['total_pnl']:+,.0f}")

# ── Final ranked report ─────────────────────────────────────────────────────
ts_df = pd.DataFrame(ticker_stats).sort_values("total_pnl", ascending=False)
baseline_total = ts_df[ts_df["ticker"].isin(["SPY", "QQQ", "IWM"])]["total_pnl"].sum()
baseline_avg   = ts_df[ts_df["ticker"].isin(["SPY", "QQQ", "IWM"])]["avg_pnl"].mean()

print(f"\n=== Per-ticker P&L summary · 90 days · $500/trade · 15% trail ===\n")
print(f"  {'rank':>4} {'tk':<5} {'drops':>6} {'trades':>7} {'win%':>6} "
      f"{'total $':>10} {'avg/tr':>8} {'best':>8} {'worst':>7} {'PF':>5}  verdict")
print(f"  " + "-" * 105)
for i, r in ts_df.iterrows():
    is_baseline = r["ticker"] in ["SPY", "QQQ", "IWM"]
    if is_baseline:
        verdict = "✓ BASELINE"
    elif r["trades_taken"] == 0:
        verdict = "✗ SKIP (no chain data — possibly no 0DTE listings)"
    elif r["avg_pnl"] >= baseline_avg and r["total_pnl"] > 0:
        verdict = f"✓✓ ADD ({r['avg_pnl']/baseline_avg if baseline_avg > 0 else 1:.1f}× baseline avg)"
    elif r["total_pnl"] > 0:
        verdict = "✓ ADD (profitable, below baseline avg)"
    elif r["total_pnl"] < -1000:
        verdict = "✗✗ AVOID (would lose money)"
    else:
        verdict = "✗ SKIP (marginal/negative)"
    pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 99 else "∞"
    print(f"  {i+1:>4} {r['ticker']:<5} {r['drop_days']:>5}  {r['trades_taken']:>5}  "
          f"{r['win_rate']:>5.0f}% ${r['total_pnl']:>+8,.0f} ${r['avg_pnl']:>+6,.0f} "
          f"${r['best']:>+6,.0f} ${r['worst']:>+5,.0f} {pf_str:>5}  {verdict}")

print(f"\n  Baseline (SPY+QQQ+IWM combined) total: ${baseline_total:+,.0f}")
print(f"  Baseline avg/trade: ${baseline_avg:+,.0f}")

# Save trades + stats for inspection (drop dict cols before CSV)
flat_ts = ts_df.copy()
if "fail_reasons" in flat_ts.columns:
    flat_ts = flat_ts.drop(columns=["fail_reasons"])
flat_ts.to_csv(ROOT / "data" / "ticker_universe_pnl.csv", index=False)
pd.DataFrame(all_trades).to_csv(ROOT / "data" / "ticker_universe_trades.csv", index=False)
print(f"\n→ Per-ticker stats saved to data/ticker_universe_pnl.csv")
print(f"→ Individual trades saved to data/ticker_universe_trades.csv")
