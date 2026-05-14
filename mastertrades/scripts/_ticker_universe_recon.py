"""Cheap recon: which tickers behave like SPY/QQQ/IWM for the 0DTE Drop signal?

For each candidate ticker, fetches 90 days of daily OHLC and computes:
  - drop frequency       — # days with intraday drop ≥ 3 pts (or 0.5% pct)
  - recovery rate         — % of drop days where close >= open (recovered)
  - mean recovery magnitude — avg pts recovered from low
  - median absolute range — typical day's high-low spread
  - relative volatility   — daily range as % of price (vs SPY baseline)
  - drop band distribution — how often each drop magnitude bucket occurs

Tickers are ranked by a composite score that rewards similar behavior to
the validated SPY/QQQ/IWM cohort. Output is a ranked table; only the top
performers proceed to the expensive option-ladder validation phase.

Daily OHLC fetches are cheap (1 Polygon call per ticker per request, single
call for 90 days). No option-chain calls happen here.
"""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datetime as dt
import requests
import pandas as pd
import numpy as np

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("POLYGON_API_KEY"):
    print("ERROR: POLYGON_API_KEY not set.")
    sys.exit(1)

# ── Candidate universe ──────────────────────────────────────────────────────
# Index ETFs first (most likely to behave like SPY/QQQ/IWM, all have 0DTE listed).
# Sector ETFs second (more concentrated, may have wilder swings).
# Single-name mega-caps third (have weekly options, sometimes daily, but
# different microstructure — may behave very differently from index ETFs).
CANDIDATE_TICKERS = [
    # ── Already validated baseline (for sanity check) ───────────────────────
    "SPY",   # S&P 500 — current baseline
    "QQQ",   # Nasdaq-100 — current baseline
    "IWM",   # Russell 2000 — current baseline (most volatile of the 3)
    # ── Other broad-market ETFs with 0DTE listings ──────────────────────────
    "DIA",   # Dow Jones 30 — has 0DTE
    "XSP",   # mini-SPX (cash-settled) — has 0DTE
    # ── Sector ETFs with weekly+ options (may have 0DTE) ────────────────────
    "XLK",   # tech sector
    "XLF",   # financials
    "XLE",   # energy
    "SMH",   # semis
    "ARKK",  # innovation
    # ── Mega-caps (high option liquidity, daily/weekly expiries) ────────────
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "META",
    "AMZN",
    # ── Volatility-adjacent ETFs ────────────────────────────────────────────
    "TLT",   # 20-yr treasuries — different microstructure
    "GLD",   # gold
]

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


def fetch_daily(ticker: str, days: int = 120) -> pd.DataFrame:
    """Fetch daily OHLC for the last N calendar days."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    data = _get(path, {"adjusted": "true", "sort": "asc", "limit": 5000})
    rows = []
    for r in data.get("results", []) or []:
        ts = dt.datetime.utcfromtimestamp(int(r["t"]) / 1000).date()
        rows.append({
            "date": ts, "open": float(r["o"]), "high": float(r["h"]),
            "low":  float(r["l"]), "close": float(r["c"]),
            "volume": int(r.get("v") or 0),
        })
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, drop_pts_min: float = 3.0,
             drop_pct_min: float = 0.005) -> dict:
    """Drop+recovery stats for one ticker. Definitions match the live algo:
       - "drop day" = (open - low) >= drop_pts_min OR >= drop_pct_min × open
       - "recovered" = close >= open (≥ open recovery — what 0DTE Drop targets)
    """
    if df.empty or len(df) < 30:
        return {"days_analysed": 0}

    df = df.copy().sort_values("date").reset_index(drop=True)
    drop_pts = (df["open"] - df["low"]).clip(lower=0)
    drop_pct = drop_pts / df["open"]
    is_drop  = (drop_pts >= drop_pts_min) | (drop_pct >= drop_pct_min)
    drop_days = df[is_drop].copy()
    drop_days["drop_pts"] = drop_pts[is_drop]
    drop_days["drop_pct"] = drop_pct[is_drop] * 100
    # Recovery: close >= open (full recovery to open)
    drop_days["recovered_to_open"] = drop_days["close"] >= drop_days["open"]
    # Recovery magnitude (pts recovered from low)
    drop_days["recovery_pts"] = drop_days["close"] - drop_days["low"]

    # Drop-band distribution
    bands = {
        "3-5pt":  (drop_days["drop_pts"] >= 3) & (drop_days["drop_pts"] < 5),
        "5-7pt":  (drop_days["drop_pts"] >= 5) & (drop_days["drop_pts"] < 7),
        "7-10pt": (drop_days["drop_pts"] >= 7) & (drop_days["drop_pts"] < 10),
        "10pt+":  drop_days["drop_pts"] >= 10,
    }
    band_counts = {b: int(m.sum()) for b, m in bands.items()}
    band_recover_rates = {
        b: float(drop_days[m]["recovered_to_open"].mean() * 100) if int(m.sum()) > 0 else 0.0
        for b, m in bands.items()
    }

    return {
        "days_analysed":          int(len(df)),
        "avg_price":               float(df["close"].mean()),
        "drop_days":               int(len(drop_days)),
        "drop_freq_pct":           float(len(drop_days) / len(df) * 100),
        "median_drop_pts":         float(drop_days["drop_pts"].median()) if len(drop_days) else 0,
        "median_drop_pct":         float(drop_days["drop_pct"].median()) if len(drop_days) else 0,
        "recovery_rate_pct":       float(drop_days["recovered_to_open"].mean() * 100) if len(drop_days) else 0,
        "median_range_pct":        float(((df["high"] - df["low"]) / df["open"]).median() * 100),
        "avg_volume_M":            float(df["volume"].mean() / 1e6),
        "band_counts":             band_counts,
        "band_recover_rates":      band_recover_rates,
    }


# ── Run recon ──────────────────────────────────────────────────────────────
print(f"=== Daily-OHLC recon · {len(CANDIDATE_TICKERS)} tickers · 90-day window ===\n")
results = []
for tk in CANDIDATE_TICKERS:
    print(f"  fetching {tk}...", end="", flush=True)
    try:
        df = fetch_daily(tk, days=120)
        stats = analyse(df)
    except Exception as e:
        print(f" FAIL: {e}")
        continue
    if stats.get("days_analysed", 0) < 30:
        print(f" not enough data ({stats.get('days_analysed', 0)} days)")
        continue
    stats["ticker"] = tk
    results.append(stats)
    print(f" OK")

if not results:
    print("\nNo results. Check Polygon API key / network.")
    sys.exit(1)

res_df = pd.DataFrame(results)

# ── Composite score ─────────────────────────────────────────────────────────
# Reward tickers that:
#   - have many drop days (more alerts → more $)
#   - have HIGH recovery rates (the strategy assumes recovery)
#   - have DAILY ranges similar to or higher than the SPY/QQQ/IWM baseline
#     (otherwise the trail-stop has nothing to capture)
# Composite = drop_days × recovery_rate × range_pct  (higher = more tradeable)
res_df["score"] = (
    res_df["drop_days"] *
    res_df["recovery_rate_pct"] *
    res_df["median_range_pct"]
) / 1000.0

# Baseline cohort = SPY+QQQ+IWM
baseline = res_df[res_df["ticker"].isin(["SPY", "QQQ", "IWM"])]
baseline_min_score = baseline["score"].min() if not baseline.empty else 0
baseline_min_recov = baseline["recovery_rate_pct"].min() if not baseline.empty else 0
print(f"\nBaseline (SPY/QQQ/IWM): "
      f"min score = {baseline_min_score:.0f}, "
      f"min recovery rate = {baseline_min_recov:.0f}%\n")

# ── Print ranked table ─────────────────────────────────────────────────────
res_df = res_df.sort_values("score", ascending=False).reset_index(drop=True)
print(f"=== Ranked candidates ===\n")
print(f"  {'rank':>4} {'tk':<5} {'price':>7} {'drops':>6} {'freq':>6} "
      f"{'med-drop':>9} {'recov%':>7} {'range%':>7} {'score':>7}  verdict")
print(f"  " + "-" * 95)
for i, r in res_df.iterrows():
    if r["ticker"] in ["SPY", "QQQ", "IWM"]:
        verdict = "✓ BASELINE"
    elif r["score"] >= baseline_min_score and r["recovery_rate_pct"] >= baseline_min_recov:
        verdict = "✓ ADD (matches/beats baseline)"
    elif r["recovery_rate_pct"] < 30:
        verdict = "✗ SKIP (poor recovery rate)"
    elif r["drop_days"] < 10:
        verdict = "✗ SKIP (too few alerts)"
    elif r["score"] >= baseline_min_score * 0.5:
        verdict = "? CONSIDER (below baseline but tradeable)"
    else:
        verdict = "✗ SKIP (low composite score)"
    print(f"  {i+1:>4} {r['ticker']:<5} ${r['avg_price']:>5.0f} "
          f"{r['drop_days']:>5}  {r['drop_freq_pct']:>5.0f}% "
          f"{r['median_drop_pts']:>7.2f}  {r['recovery_rate_pct']:>5.0f}% "
          f"{r['median_range_pct']:>5.1f}%  {r['score']:>6.0f}  {verdict}")

# ── Drop-band breakdown for the top candidates ─────────────────────────────
print(f"\n=== Drop-band recovery rates (top 10 by score) ===\n")
print(f"  {'tk':<5} {'3-5pt':>14}  {'5-7pt':>14}  {'7-10pt':>14}  {'10pt+':>14}")
for _, r in res_df.head(10).iterrows():
    parts = []
    for b in ["3-5pt", "5-7pt", "7-10pt", "10pt+"]:
        n  = r["band_counts"][b]
        rr = r["band_recover_rates"][b]
        parts.append(f"{n}d / {rr:>3.0f}% rec")
    print(f"  {r['ticker']:<5} " + "  ".join(f"{p:>14}" for p in parts))

# ── Save ranked CSV for the next phase ──────────────────────────────────────
out = ROOT / "data" / "ticker_universe_recon.csv"
# Drop the dict columns before writing (they'd serialize awkwardly)
flat = res_df.drop(columns=["band_counts", "band_recover_rates"]).copy()
flat.to_csv(out, index=False)
print(f"\n→ Ranked recon written to {out}")
print(f"  Use the top 'ADD' candidates for the option-ladder validation phase.")
