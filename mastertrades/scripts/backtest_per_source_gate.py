"""Validate the proposed per-source edge gate for the Today's Plays page.

Background
----------
The current unified gate in `app.py` requires `win_rate >= 50%` for every
source. The hardcoded SPY drop-band table in `src/options_scanner.py` tops
out at 35% (3-5 pts band), so the 0DTE Drop source can never pass the gate
under any market condition. This is structural, not a calibration accident
— see the analysis chat dated 2026-05-13.

This script measures, for the last N trading days, how many 0DTE Drop
setups would have fired under several proposed gate variants, and what
the realised return would have been on the algo's recommended strike,
priced from real Polygon historical options OHLC.

The intent is to PRODUCE the calibration numbers we need before we patch
the gate, and then to RE-VALIDATE after the patch on a wider window.

Pipeline (mirrors scripts/backtest_signal_threshold.py)
-------------------------------------------------------
Phase 1 (`--score`)   : enumerate ENTRY_OPEN candidates per ticker per
                        day from cached daily OHLC. Cheap; no Polygon hits.
Phase 2 (`--options`) : for each candidate, fetch real same-day option
                        OHLC for the algo's recommended strike (low+5)
                        via `src.options_history`. Cached incrementally.
Phase 3 (`--report`)  : sweep gate variants (current 50% gate, lowered
                        win-rate gates, EV-based gates) and print
                        side-by-side performance.

Run incrementally:
    cd mastertrades
    python scripts/backtest_per_source_gate.py --score
    python scripts/backtest_per_source_gate.py --options --max-new 50
    python scripts/backtest_per_source_gate.py --report

Or all-in-one:
    python scripts/backtest_per_source_gate.py --days 90 --score --options --report
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd

from src.scanner import fetch_or_load_daily
from src.options_scanner import drop_band_multiplier_table

DATA_DIR = ROOT / "data"
SCORES_CSV = DATA_DIR / "per_source_gate_candidates.csv"
TRADES_CSV = DATA_DIR / "per_source_gate_trades.csv"
LADDER_TRADES_CSV = DATA_DIR / "per_source_gate_trades_ladder.csv"

# How wide a strike ladder (in pts) to fetch per candidate. The live algo
# considers offsets 1..9 above the intraday low; we fetch a slightly wider
# band so we catch all the strikes the algo would have surfaced.
LADDER_WIDTH_PTS = 12

DEFAULT_TICKERS = ("SPY", "QQQ", "IWM")
DEFAULT_DAYS = 90

# Algo's ENTRY_OPEN trigger (matches load_0dte_alert in app.py)
ENTRY_OPEN_DROP_PTS = 3.0

# Algo's recommended strike offset (sweet-spot midpoint per recommend_strikes
# in src/options_scanner.py — offsets 1..9 above the intraday low; we test
# a small grid and report the best per day to mirror the live UI which shows
# the highest-est-gain rec).
STRIKE_OFFSETS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


# ---------------------------------------------------------------------------
# Phase 1 — enumerate ENTRY_OPEN candidates from cached daily OHLC
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    ticker: str
    date: str          # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    drop_pts: float
    rise_pts: float    # high - open
    range_pts: float   # high - low
    gap_pct: float     # (open / prev_close - 1) * 100
    weekday: int


def _drop_band_for(drop_pts: float) -> tuple[str, int]:
    """Return (band_label, hist_pct_1000plus) for a given drop in pts."""
    table = drop_band_multiplier_table()
    for row in table:
        band = row["band"]
        if "0–1" in band   and drop_pts < 1:   return band, int(row["pct_1000plus"])
        if "1–2" in band   and drop_pts < 2:   return band, int(row["pct_1000plus"])
        if "2–3" in band   and drop_pts < 3:   return band, int(row["pct_1000plus"])
        if "3–5" in band   and drop_pts < 5:   return band, int(row["pct_1000plus"])
        if "5–7" in band   and drop_pts < 7:   return band, int(row["pct_1000plus"])
        if "7–10" in band  and drop_pts < 10:  return band, int(row["pct_1000plus"])
        if "10+" in band:                       return band, int(row["pct_1000plus"])
    return "unknown", 0


def enumerate_candidates(tickers: tuple[str, ...], days: int) -> pd.DataFrame:
    """One row per (ticker, day) where drop_pts >= ENTRY_OPEN_DROP_PTS."""
    rows: list[Candidate] = []
    for tkr in tickers:
        try:
            daily = fetch_or_load_daily(tkr, data_dir=DATA_DIR, refresh=False)
        except Exception as exc:
            print(f"  {tkr}: skipped — {exc}", flush=True)
            continue
        daily = daily.sort_index().tail(days + 1).copy()  # +1 for prev_close
        for i in range(1, len(daily)):
            row = daily.iloc[i]
            prev = daily.iloc[i - 1]
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            drop = o - l
            if drop < ENTRY_OPEN_DROP_PTS:
                continue
            d = daily.index[i]
            rows.append(Candidate(
                ticker=tkr,
                date=pd.Timestamp(d).strftime("%Y-%m-%d"),
                open=o, high=h, low=l, close=c,
                drop_pts=round(drop, 2),
                rise_pts=round(h - o, 2),
                range_pts=round(h - l, 2),
                gap_pct=round((o / float(prev["Close"]) - 1.0) * 100, 3),
                weekday=int(pd.Timestamp(d).weekday()),
            ))
    df = pd.DataFrame([asdict(r) for r in rows])
    if not df.empty:
        df["band"], df["hist_pct_1000plus"] = zip(*df["drop_pts"].map(_drop_band_for))
    return df


def score_phase(tickers: tuple[str, ...], days: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = enumerate_candidates(tickers, days)
    if df.empty:
        print("No ENTRY_OPEN candidates found.", flush=True)
        return
    df.sort_values(["ticker", "date"]).to_csv(SCORES_CSV, index=False)
    print(f"Wrote {len(df)} candidates → {SCORES_CSV.relative_to(ROOT)}", flush=True)
    print()
    print("Per-ticker counts:")
    print(df.groupby("ticker").size().to_string())
    print()
    print("Per-band counts:")
    print(df.groupby("band").size().to_string())


# ---------------------------------------------------------------------------
# Phase 2 — fetch real historical option OHLC for each candidate
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    ticker: str
    date: str
    drop_pts: float
    band: str
    hist_pct_1000plus: int
    underlying_low: float
    underlying_high: float
    underlying_close: float
    strike: float
    contract: str
    opt_open: float
    opt_high: float
    opt_low: float
    opt_close: float
    # Realised returns under several simple exit rules
    ret_low_to_high_pct: float    # buy at opt_low, sell at opt_high (best case)
    ret_open_to_high_pct: float   # buy at opt_open, sell at opt_high (more realistic)
    ret_open_to_close_pct: float  # buy at opt_open, sell at close (passive hold)


def _best_strike_offset(low: float, high: float) -> int:
    """Pick the strike offset that maximises low→high return assuming
    intrinsic-value pricing (matches recommend_strikes' sweet-spot logic)."""
    best_off = STRIKE_OFFSETS[0]
    best_ret = -1.0
    for off in STRIKE_OFFSETS:
        strike = round(low + off)
        # Approx intrinsic gain: max(high - strike, 0); entry is unknown
        # without the chain so we rank by raw intrinsic potential here.
        intrinsic_gain = max(high - strike, 0.0)
        if intrinsic_gain > best_ret:
            best_ret = intrinsic_gain
            best_off = off
    return best_off


def options_phase(max_new: int) -> None:
    if not SCORES_CSV.exists():
        print(f"Run --score first ({SCORES_CSV} missing).")
        return
    from src.options_history import list_expired_contracts, fetch_option_daily_bars

    cands = pd.read_csv(SCORES_CSV)
    cands["date"] = pd.to_datetime(cands["date"]).dt.strftime("%Y-%m-%d")

    if TRADES_CSV.exists():
        existing = pd.read_csv(TRADES_CSV)
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        done = set(zip(existing["ticker"], existing["date"]))
    else:
        existing = pd.DataFrame()
        done = set()

    todo = cands[~cands.apply(lambda r: (r["ticker"], r["date"]) in done, axis=1)]
    print(f"Candidates: {len(cands)} · already priced: {len(done)} · to fetch: {len(todo)}",
          flush=True)
    if todo.empty:
        return

    new_rows: list[Trade] = []
    n_done = 0
    for _, r in todo.iterrows():
        if n_done >= max_new:
            break
        offset = _best_strike_offset(r["low"], r["high"])
        strike = round(float(r["low"]) + offset)
        try:
            cs = list_expired_contracts(
                r["ticker"], r["date"], r["date"], "call",
                strike_lo=strike - 2, strike_hi=strike + 2,
            )
        except Exception as exc:
            print(f"  {r['ticker']} {r['date']}: contracts API failed: {exc}", flush=True)
            continue
        if not cs:
            print(f"  {r['ticker']} {r['date']}: no contracts at K~{strike}", flush=True)
            continue
        cs.sort(key=lambda c: abs(c.strike - strike))
        c = cs[0]
        try:
            bars = fetch_option_daily_bars(c.ticker, r["date"], r["date"])
        except Exception as exc:
            print(f"  {r['ticker']} {r['date']}: bars API failed: {exc}", flush=True)
            continue
        if not bars or bars[0].open <= 0 or bars[0].low <= 0:
            print(f"  {r['ticker']} {r['date']}: empty bar for {c.ticker}", flush=True)
            continue
        b = bars[0]
        ret_lh = (b.high / b.low - 1.0) * 100
        ret_oh = (b.high / b.open - 1.0) * 100
        ret_oc = (b.close / b.open - 1.0) * 100
        new_rows.append(Trade(
            ticker=r["ticker"], date=r["date"],
            drop_pts=float(r["drop_pts"]), band=str(r["band"]),
            hist_pct_1000plus=int(r["hist_pct_1000plus"]),
            underlying_low=float(r["low"]), underlying_high=float(r["high"]),
            underlying_close=float(r["close"]),
            strike=float(c.strike), contract=c.ticker,
            opt_open=b.open, opt_high=b.high, opt_low=b.low, opt_close=b.close,
            ret_low_to_high_pct=round(ret_lh, 1),
            ret_open_to_high_pct=round(ret_oh, 1),
            ret_open_to_close_pct=round(ret_oc, 1),
        ))
        n_done += 1
        print(f"  {r['ticker']} {r['date']}  drop={r['drop_pts']:>4.1f}  "
              f"K={c.strike:.0f}  ${b.open:.2f}→${b.high:.2f}  "
              f"low→high={ret_lh:+.0f}%  open→high={ret_oh:+.0f}%", flush=True)
        if n_done % 10 == 0:
            _save_trades(existing, new_rows)

    _save_trades(existing, new_rows)
    print(f"\nFetched {n_done} new trades. Total cached: "
          f"{len(existing) + len(new_rows)}", flush=True)


def _save_trades(existing: pd.DataFrame, new_rows: list[Trade]) -> None:
    if not new_rows and existing.empty:
        return
    df_new = pd.DataFrame([asdict(t) for t in new_rows])
    out = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    out.sort_values(["ticker", "date"]).to_csv(TRADES_CSV, index=False)


# ---------------------------------------------------------------------------
# Phase 2b — fetch full strike-ladder OHLC per candidate
# ---------------------------------------------------------------------------


def options_ladder_phase(max_new: int) -> None:
    """Fetch every call strike in [low, low + LADDER_WIDTH_PTS] per
    candidate. Lets us replay the live algo's recommend_strikes() ranking
    instead of being locked to a single strike per day."""
    if not SCORES_CSV.exists():
        print(f"Run --score first ({SCORES_CSV} missing).")
        return
    from src.options_history import list_expired_contracts, fetch_option_daily_bars

    cands = pd.read_csv(SCORES_CSV)
    cands["date"] = pd.to_datetime(cands["date"]).dt.strftime("%Y-%m-%d")

    if LADDER_TRADES_CSV.exists():
        existing = pd.read_csv(LADDER_TRADES_CSV)
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        done = set(zip(existing["ticker"], existing["date"]))
    else:
        existing = pd.DataFrame()
        done = set()

    todo = cands[~cands.apply(lambda r: (r["ticker"], r["date"]) in done, axis=1)]
    print(f"Candidate days: {len(cands)} · already laddered: {len(done)} · "
          f"to fetch: {len(todo)}", flush=True)
    if todo.empty:
        return

    new_rows: list[dict] = []
    n_done = 0
    for _, r in todo.iterrows():
        if n_done >= max_new:
            break
        low = float(r["low"])
        high = float(r["high"])
        try:
            cs = list_expired_contracts(
                r["ticker"], r["date"], r["date"], "call",
                strike_lo=int(low),
                strike_hi=int(low) + LADDER_WIDTH_PTS,
            )
        except Exception as exc:
            print(f"  {r['ticker']} {r['date']}: contracts list failed: {exc}",
                  flush=True)
            continue
        if not cs:
            print(f"  {r['ticker']} {r['date']}: no contracts in ladder", flush=True)
            n_done += 1
            continue
        # Sort by strike ascending so the print is readable
        cs.sort(key=lambda c: c.strike)
        contracts_added = 0
        for c in cs:
            offset = c.strike - low
            if offset < 0 or offset > LADDER_WIDTH_PTS:
                continue
            try:
                bars = fetch_option_daily_bars(c.ticker, r["date"], r["date"])
            except Exception:
                continue
            if not bars or bars[0].open <= 0 or bars[0].low <= 0:
                continue
            b = bars[0]
            new_rows.append({
                "ticker":          r["ticker"],
                "date":            r["date"],
                "drop_pts":        float(r["drop_pts"]),
                "underlying_open": float(r["open"]),
                "underlying_high": high,
                "underlying_low":  low,
                "underlying_close":float(r["close"]),
                "strike":          float(c.strike),
                "contract":        c.ticker,
                "offset_from_low": round(offset, 2),
                "opt_open":        b.open,
                "opt_high":        b.high,
                "opt_low":         b.low,
                "opt_close":       b.close,
            })
            contracts_added += 1
        n_done += 1
        print(f"  {r['ticker']} {r['date']}  drop={r['drop_pts']:>4.1f}  "
              f"+{contracts_added} strikes (offsets {0}-{LADDER_WIDTH_PTS})",
              flush=True)
        if n_done % 5 == 0:
            _save_ladder(existing, new_rows)
    _save_ladder(existing, new_rows)
    print(f"\nFetched {n_done} new candidate-days. Total ladder rows now: "
          f"{len(existing) + len(new_rows)}", flush=True)


def _save_ladder(existing: pd.DataFrame, new_rows: list[dict]) -> None:
    if not new_rows and existing.empty:
        return
    df_new = pd.DataFrame(new_rows)
    out = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    out.sort_values(["ticker", "date", "offset_from_low"]).to_csv(LADDER_TRADES_CSV, index=False)


# ---------------------------------------------------------------------------
# Phase 4b — money sim using the live algo's strike selection
# ---------------------------------------------------------------------------


def money_ladder_phase(position_usd: float) -> None:
    """Simulate $position_usd per alert using the strike the LIVE algo would
    have surfaced (highest est_gain_pct, mirroring recommend_strikes)."""
    if not LADDER_TRADES_CSV.exists():
        print(f"Run --options --ladder first ({LADDER_TRADES_CSV} missing).")
        return
    df = pd.read_csv(LADDER_TRADES_CSV).copy()
    if df.empty:
        print("Ladder file empty.")
        return

    # ── Mirror src/options_scanner.py:recommend_strikes() ────────────────
    # entry         = max(opt_low, 0.01)
    # recovery_target = underlying_open  (algo's default)
    # if recovery_target >= strike:
    #     target_price = (recovery_target - strike) + 0.05
    # else:
    #     target_price = max(0.05, 0.5 * (recovery_target - strike + 1.0))  ← we approx delta=0.5
    df["entry_est"] = df["opt_low"].clip(lower=0.01)
    df["recovery_target"] = df["underlying_open"]
    intrinsic_at_target = (df["recovery_target"] - df["strike"]).clip(lower=0.0)
    target_itm = intrinsic_at_target + 0.05
    target_otm = (0.5 * (df["recovery_target"] - df["strike"] + 1.0)).clip(lower=0.05)
    df["target_price_est"] = target_itm.where(df["recovery_target"] >= df["strike"], target_otm)
    df["est_gain_pct"] = (df["target_price_est"] - df["entry_est"]) / df["entry_est"] * 100

    # Live algo only considers offsets 1..9 above the low (sweet spot)
    df = df[(df["offset_from_low"] >= 1) & (df["offset_from_low"] <= 9)].copy()

    # Pick the strike with highest est_gain_pct per (ticker, date) — what the
    # dashboard's "TOP rec" would have been at alert time.
    best_idx = df.groupby(["ticker", "date"])["est_gain_pct"].idxmax()
    picks = df.loc[best_idx].copy()

    n_days = len(picks)
    # Position sizing — buy as many full contracts as $position_usd allows.
    # ENTRY assumption: opt_open (more honest than opt_low which assumes you
    # nailed the bottom). The algo's est_gain uses opt_low for the rec sort
    # but a real fill will be at the prevailing ask, which is opt_open or
    # somewhere between opt_open and opt_low.
    picks["contracts"] = (position_usd // (picks["opt_open"] * 100)).astype(int)
    n_skipped_premium = int((picks["contracts"] == 0).sum())
    taken = picks[picks["contracts"] > 0].copy()
    taken["cost_usd"] = taken["contracts"] * taken["opt_open"] * 100
    taken["idle_usd"] = position_usd - taken["cost_usd"]

    # Three exit scenarios (same definitions as money_phase)
    taken["exit_optim_usd"]   = taken["contracts"] * taken["opt_high"]  * 100
    taken["exit_real_usd"]    = taken["contracts"] * ((taken["opt_open"] + taken["opt_high"]) / 2.0) * 100
    taken["exit_conserv_usd"] = taken["contracts"] * taken["opt_close"] * 100
    taken["pnl_optim_usd"]    = taken["exit_optim_usd"]   - taken["cost_usd"]
    taken["pnl_real_usd"]     = taken["exit_real_usd"]    - taken["cost_usd"]
    taken["pnl_conserv_usd"]  = taken["exit_conserv_usd"] - taken["cost_usd"]

    print(f"\n=== ${position_usd:.0f} per alert — LIVE ALGO STRIKE SELECTION ===")
    print(f"   (replays src/options_scanner.py:recommend_strikes ranking)\n")
    print(f"  Candidate days with usable ladder: {n_days}")
    print(f"  Skipped (premium > ${position_usd:.0f}/contract):  {n_skipped_premium}")
    print(f"  Trades actually taken:            {len(taken)}")
    print(f"  Total capital cycled:             ${taken['cost_usd'].sum():,.0f}")
    print(f"  Avg capital deployed/trade:       ${taken['cost_usd'].mean():,.0f}  "
          f"(vs ${position_usd:.0f} budget)")
    print(f"  Avg contracts/trade:              {taken['contracts'].mean():.1f}")
    print(f"  Avg strike offset above low:      +{taken['offset_from_low'].mean():.1f} pts")
    print()

    print(f"{'Scenario':<14} {'Total P&L':>12} {'Avg/trade':>11} {'Win%':>6} "
          f"{'Best':>10} {'Worst':>10}")
    for label, col in [("OPTIMISTIC",   "pnl_optim_usd"),
                       ("REALISTIC",    "pnl_real_usd"),
                       ("CONSERVATIVE", "pnl_conserv_usd")]:
        p = taken[col]
        print(f"  {label:<12} {p.sum():>+11,.0f}$ {p.mean():>+10,.0f}$ "
              f"{(p > 0).mean() * 100:>5.0f}% {p.max():>+9,.0f}$ {p.min():>+9,.0f}$")

    print()
    print(f"=== Per-band breakdown (REALISTIC) ===")
    taken["band"] = taken["drop_pts"].map(lambda d: _drop_band_for(d)[0])
    print(f"{'band':<10} {'n':>4} {'total':>12} {'avg/trade':>11} {'win%':>6} "
          f"{'avg cont':>9}")
    for band, g in taken.groupby("band"):
        p = g["pnl_real_usd"]
        print(f"  {band:<8} {len(g):>4} {p.sum():>+11,.0f}$ {p.mean():>+10,.0f}$ "
              f"{(p > 0).mean() * 100:>5.0f}% {g['contracts'].mean():>8.1f}")

    print()
    print(f"=== 5 best and 5 worst single trades (REALISTIC) ===")
    sorted_real = taken.sort_values("pnl_real_usd", ascending=False)
    print(f"{'when':<14} {'tkr':<5} {'K':>5} {'off':>5} {'open':>6} {'high':>6} "
          f"{'close':>6} {'cont':>4} {'P&L':>10}")
    for r in list(sorted_real.head(5).itertuples()) + list(sorted_real.tail(5).itertuples()):
        print(f"  {r.date:<12} {r.ticker:<5} {r.strike:>5.0f} +{r.offset_from_low:>3.0f} "
              f"{r.opt_open:>5.2f} {r.opt_high:>5.2f} {r.opt_close:>5.2f} "
              f"{r.contracts:>4} {r.pnl_real_usd:>+9,.0f}$")

    out = ROOT / "data" / "per_source_gate_money_ladder.csv"
    taken.to_csv(out, index=False)
    print()
    print(f"Per-trade ledger written to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Phase 3 — sweep gate variants
# ---------------------------------------------------------------------------


def report_phase() -> None:
    if not TRADES_CSV.exists():
        print("No trades cached. Run --score then --options first.")
        return
    df = pd.read_csv(TRADES_CSV)
    n = len(df)
    if n == 0:
        print("Trades file empty.")
        return

    # Refresh hist_pct from the LIVE table — the cached value was stamped
    # at fetch time, but the table may have been recalibrated since. We
    # always want the gate sweep to reflect the current production values.
    refreshed = df["drop_pts"].map(_drop_band_for)
    df["band"] = refreshed.map(lambda t: t[0])
    df["hist_pct_1000plus"] = refreshed.map(lambda t: t[1])

    print(f"\n=== Trade universe: {n} 0DTE Drop setups ({df['ticker'].nunique()} tickers, "
          f"{df['date'].min()} → {df['date'].max()}) ===")
    print(f"   (gate sweep uses live drop_band_multiplier_table values)\n")

    # Group by drop band — replicates the structure of drop_band_multiplier_table
    # so we can see whether the hardcoded 35% number is still accurate today.
    print("Realised performance by drop band (entry at opt_low, exit at opt_high):")
    print(f"{'band':<12} {'n':>4} {'≥1000% hit':>12} {'≥100% hit':>11} "
          f"{'median ret':>11} {'mean ret':>10} {'best':>8}")
    for band, g in df.groupby("band", sort=False):
        n_b = len(g)
        hit_1000 = (g["ret_low_to_high_pct"] >= 1000).mean() * 100
        hit_100  = (g["ret_low_to_high_pct"] >= 100).mean() * 100
        med = g["ret_low_to_high_pct"].median()
        avg = g["ret_low_to_high_pct"].mean()
        best = g["ret_low_to_high_pct"].max()
        print(f"  {band:<10} {n_b:>4} {hit_1000:>10.1f}%  {hit_100:>9.1f}%  "
              f"{med:>+10.0f}%  {avg:>+9.0f}%  {best:>+7.0f}%")

    # Gate sweep — uses ret_open_to_high_pct as the realistic per-trade return
    # (entry at the option's open is a more honest proxy for an alert-triggered
    # entry than entry at the option's daily low, which requires perfect timing).
    realistic = df["ret_open_to_high_pct"]
    print()
    print("=== Gate variant sweep (return metric = open→high, the alert-time entry proxy) ===")
    print(f"{'gate':<40} {'kept':>5} {'avg ret':>9} {'win%':>6} {'big win':>8}")

    def _stats(mask):
        g = realistic[mask]
        if len(g) == 0:
            return 0, float("nan"), float("nan"), 0
        return len(g), g.mean(), (g > 0).mean() * 100, int((g >= 200).sum())

    # Current gate (50% win rate proxy). hist_pct_1000plus tops out at 35%
    # so this should never fire — verify.
    n_k, avg, win, bw = _stats(df["hist_pct_1000plus"] >= 50)
    print(f"  {'CURRENT (hist_pct >= 50%)':<40} {n_k:>5} {avg:>+8.0f}% {win:>5.0f}% {bw:>7d}")

    # Win-rate gate sweep (lower thresholds)
    for thr in [10, 20, 25, 30, 35]:
        n_k, avg, win, bw = _stats(df["hist_pct_1000plus"] >= thr)
        print(f"  {'WR-based, hist_pct >= ' + str(thr) + '%':<40} {n_k:>5} {avg:>+8.0f}% {win:>5.0f}% {bw:>7d}")

    # EV-based gate sweep — gate fires if expected % return >= MIN_EV.
    # We approximate EV from the same drop-band table the live algo uses,
    # multiplied by an assumed payoff of 1000% per win (matches the table's
    # "≥1000% multiplier" definition). MIN_EV is the dial.
    df["ev_proxy_pct"] = df["hist_pct_1000plus"] / 100.0 * 1000  # e.g. 35%×1000% = 350% EV
    print()
    print("EV proxy assumes payoff = 1000% per winner (matches table definition):")
    for thr in [50, 100, 150, 200, 250, 300, 350]:
        n_k, avg, win, bw = _stats(df["ev_proxy_pct"] >= thr)
        print(f"  {'EV-based, EV proxy >= ' + str(thr) + '%':<40} {n_k:>5} {avg:>+8.0f}% {win:>5.0f}% {bw:>7d}")

    print()
    print("=== Honest caveat ===")
    print("• low→high returns assume perfect timing of both entry AND exit.")
    print("• open→high returns assume you bought at the official open and sold at the day's peak —")
    print("  still optimistic, but a more honest upper bound for an alert-triggered trade.")
    print("• A live trade with normal slippage typically captures 30-60% of the open→high move.")
    print("• Use the EV-based gate calibration to set MIN_EV with that haircut in mind.")


# ---------------------------------------------------------------------------
# Phase 4 — money simulation: fixed $ per trade
# ---------------------------------------------------------------------------


def money_phase(position_usd: float) -> None:
    """Simulate $position_usd per alert across all cached trades.

    For each trade we model THREE exit scenarios so the realistic range is
    visible (no single number is honest on its own):

      OPTIMISTIC  — entry at opt_open, exit at opt_high (caught the peak)
      REALISTIC   — entry at opt_open, exit at midpoint of (open, high)
                    (typical alert-driven trader catches some, not all)
      CONSERVATIVE— entry at opt_open, exit at opt_close
                    (passive hold to bell, no profit-take)

    Position sizing:
      contracts = floor(position_usd / (opt_open * 100))
      If contracts == 0 (premium > position_usd / 100), trade is SKIPPED.
      Idle cash (position_usd - contracts*opt_open*100) earns 0%.
    """
    if not TRADES_CSV.exists():
        print("No trades cached. Run --score then --options first.")
        return
    df = pd.read_csv(TRADES_CSV).copy()
    n = len(df)
    if n == 0:
        print("Trades file empty.")
        return

    # Refresh band/hist_pct so the current production gate is applied
    refreshed = df["drop_pts"].map(_drop_band_for)
    df["band"] = refreshed.map(lambda t: t[0])
    df["hist_pct_1000plus"] = refreshed.map(lambda t: t[1])

    # Apply the live MIN_EV_0DTE gate (we read it from app.py to stay honest)
    # — fall back to admitting all trades if app.py changes shape.
    try:
        import re
        app_src = (ROOT / "app.py").read_text(encoding="utf-8")
        m = re.search(r"MIN_EV_0DTE\s*=\s*([\d.]+)", app_src)
        min_ev = float(m.group(1)) if m else 100.0
    except Exception:
        min_ev = 100.0

    # PATH A — high-conviction (win_rate ≥ 50)
    pa = df["hist_pct_1000plus"] >= 50
    # PATH B — asymmetric (EV ≥ min_ev). EV proxy = hist_pct × 1000% per win
    ev_proxy = df["hist_pct_1000plus"] * 10.0  # /100 then ×1000 = ×10
    pb = ev_proxy >= min_ev
    df["passes_gate"] = pa | pb
    n_admitted = int(df["passes_gate"].sum())
    n_blocked  = n - n_admitted

    print(f"\n=== ${position_usd:.0f} per alert simulation "
          f"(MIN_EV_0DTE={min_ev:.0f}%, admits {n_admitted}/{n} alerts) ===\n")

    df = df[df["passes_gate"]].copy()
    df["contracts"] = (position_usd // (df["opt_open"] * 100)).astype(int)
    skipped_premium = (df["contracts"] == 0).sum()
    df = df[df["contracts"] > 0].copy()
    df["cost_usd"] = df["contracts"] * df["opt_open"] * 100
    df["idle_usd"] = position_usd - df["cost_usd"]

    # Three exit prices
    df["exit_optim_usd"]  = df["contracts"] * df["opt_high"] * 100
    df["exit_real_usd"]   = df["contracts"] * ((df["opt_open"] + df["opt_high"]) / 2.0) * 100
    df["exit_conserv_usd"] = df["contracts"] * df["opt_close"] * 100

    df["pnl_optim_usd"]   = df["exit_optim_usd"]  - df["cost_usd"]
    df["pnl_real_usd"]    = df["exit_real_usd"]   - df["cost_usd"]
    df["pnl_conserv_usd"] = df["exit_conserv_usd"] - df["cost_usd"]

    print(f"  Alerts admitted by gate:      {len(df) + skipped_premium}")
    print(f"  Skipped (premium > ${position_usd:.0f}/contract): {skipped_premium}")
    print(f"  Trades actually taken:        {len(df)}")
    print(f"  Total capital cycled:         ${df['cost_usd'].sum():,.0f}")
    print(f"  Avg capital deployed/trade:   ${df['cost_usd'].mean():,.0f}  (vs ${position_usd:.0f} budget)")
    print()

    print(f"{'Scenario':<14} {'Total P&L':>12} {'Avg/trade':>11} "
          f"{'Win%':>6} {'Best':>10} {'Worst':>10}")
    for label, col in [("OPTIMISTIC",  "pnl_optim_usd"),
                       ("REALISTIC",   "pnl_real_usd"),
                       ("CONSERVATIVE","pnl_conserv_usd")]:
        p = df[col]
        print(f"  {label:<12} {p.sum():>+11,.0f}$ {p.mean():>+10,.0f}$ "
              f"{(p > 0).mean() * 100:>5.0f}% {p.max():>+9,.0f}$ {p.min():>+9,.0f}$")

    print()
    print(f"=== Per-ticker breakdown (REALISTIC scenario) ===")
    print(f"{'ticker':<8} {'n':>4} {'total':>12} {'avg/trade':>11} {'win%':>6}")
    for tkr, g in df.groupby("ticker"):
        p = g["pnl_real_usd"]
        print(f"  {tkr:<6} {len(g):>4} {p.sum():>+11,.0f}$ {p.mean():>+10,.0f}$ "
              f"{(p > 0).mean() * 100:>5.0f}%")

    print()
    print(f"=== Per-band breakdown (REALISTIC scenario) ===")
    print(f"{'band':<10} {'n':>4} {'total':>12} {'avg/trade':>11} {'win%':>6}")
    for band, g in df.groupby("band"):
        p = g["pnl_real_usd"]
        print(f"  {band:<8} {len(g):>4} {p.sum():>+11,.0f}$ {p.mean():>+10,.0f}$ "
              f"{(p > 0).mean() * 100:>5.0f}%")

    print()
    print(f"=== 5 best and 5 worst single trades (REALISTIC) ===")
    sorted_real = df.sort_values("pnl_real_usd", ascending=False)
    print(f"{'when':<14} {'tkr':<5} {'K':>5} {'open':>6} {'high':>6} {'close':>6} "
          f"{'cont':>4} {'P&L':>10}")
    for r in list(sorted_real.head(5).itertuples()) + list(sorted_real.tail(5).itertuples()):
        print(f"  {r.date:<12} {r.ticker:<5} {r.strike:>5.0f} "
              f"{r.opt_open:>5.2f} {r.opt_high:>5.2f} {r.opt_close:>5.2f} "
              f"{r.contracts:>4} {r.pnl_real_usd:>+9,.0f}$")

    # CSV side-output for charting later
    out = ROOT / "data" / "per_source_gate_money.csv"
    df.to_csv(out, index=False)
    print()
    print(f"Per-trade ledger written to {out.relative_to(ROOT)}")
    print()
    print("=== Honest caveats ===")
    print("• REALISTIC = mid(open, high) approximates an alert-triggered trader who")
    print("  catches roughly half of the recovery move. Real fills depend on alert latency,")
    print("  bid-ask spread, and discipline at the exit.")
    print("• Slippage of 1-3 cents per contract on entry AND exit is NOT modelled here.")
    print("  On cheap options that's 5-15% drag per round-trip — reduce all P&L by ~10%")
    print("  for a realistic out-of-sample expectation.")
    print("• Today's session is included if it's in the trades CSV. Re-run --options to refresh.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                    help="Comma-separated tickers (default: SPY,QQQ,IWM)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help="Lookback window in trading days (default: 90)")
    ap.add_argument("--score", action="store_true",
                    help="Phase 1: enumerate ENTRY_OPEN candidates from cached daily OHLC")
    ap.add_argument("--options", action="store_true",
                    help="Phase 2: fetch real Polygon historical options OHLC for each")
    ap.add_argument("--report", action="store_true",
                    help="Phase 3: print per-band stats and sweep gate variants")
    ap.add_argument("--money", type=float, metavar="USD", default=None,
                    help="Phase 4: simulate $USD per alert (e.g. --money 500)")
    ap.add_argument("--ladder", action="store_true",
                    help="Use the strike-ladder dataset (live-algo strike selection). "
                         "Combine with --options to fetch the ladder, or with --money "
                         "to simulate against it.")
    ap.add_argument("--max-new", type=int, default=200,
                    help="Cap on Polygon options fetches per --options run (default: 200)")
    args = ap.parse_args()

    tickers = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())

    if not (args.score or args.options or args.report or args.money is not None):
        ap.print_help()
        return 0
    if args.score:
        score_phase(tickers, args.days)
    if args.options:
        if args.ladder:
            options_ladder_phase(args.max_new)
        else:
            options_phase(args.max_new)
    if args.report:
        report_phase()
    if args.money is not None:
        if args.ladder:
            money_ladder_phase(args.money)
        else:
            money_phase(args.money)
    return 0


if __name__ == "__main__":
    sys.exit(main())
