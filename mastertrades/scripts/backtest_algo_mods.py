"""
Backtest two algo modifications over recent SPY signals:
  Mod A: skip if overnight gap |g| >= GAP_THRESHOLD
  Mod B: profit-take half off at +TAKE_PROFIT_PCT, let rest ride to bell

Universe: SPY days where the leakage-free walk-forward classifier fires.
Trade: 9:30 open, OTM call at strike = round(open) + $5, 0DTE, hold to bell.

Persists per-trade results to data/backtest_algo_mods_trades.csv so re-runs
are fast.
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from src.scanner import fetch_or_load_daily
from src.options_history import list_expired_contracts, fetch_option_daily_bars
from src.jackpot_scanner import (
    _walkforward_cutoff, train_or_load_walkforward_models,
    classify_signal, NUMERIC_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES,
)
from src.volatility_classifier import _one_hot_weekday
from src.volatility_patterns import build_features

CACHE = "data/backtest_algo_mods_trades.csv"


def score_window(ticker: str, days: int) -> pd.DataFrame:
    daily = fetch_or_load_daily(ticker, refresh=False).copy()
    if daily.index.tz is not None: daily.index = daily.index.tz_localize(None)
    daily.index = pd.to_datetime(daily.index).normalize()
    end = daily.index.max()
    start = end - pd.Timedelta(days=days)
    win = daily.loc[start:end].index
    print(f"Scoring {len(win)} sessions {win[0].date()} → {win[-1].date()}", flush=True)

    rows = []
    for d in win:
        cutoff = _walkforward_cutoff(d)
        sl = daily[daily.index <= d]
        if len(sl) < 300: continue
        try:
            bundle = train_or_load_walkforward_models(ticker, cutoff, sl)
        except Exception:
            continue
        feats = build_features(sl)
        if d not in feats.index: continue
        fr = feats.loc[[d]].copy()
        for sp in ("is_turn_of_month","is_last_trading_day_of_month","is_first_trading_day_of_month"):
            if sp in fr.columns: fr[sp] = 0
        base = fr[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
        wd = _one_hot_weekday(fr["weekday"])
        Xp = pd.concat([base, wd], axis=1)
        pv = float(bundle["vol_model"].predict_proba(Xp.reindex(columns=bundle["vol_cols"], fill_value=0))[0,1])
        pp = float(bundle["pnl_model"].predict_proba(Xp.reindex(columns=bundle["pnl_cols"], fill_value=0))[0,1])
        pw = float(bundle["weekly_model"].predict_proba(Xp.reindex(columns=bundle["weekly_cols"], fill_value=0))[0,1])
        sig = classify_signal(pv, pp, pw)
        if sig == "SKIP": continue
        i = daily.index.get_loc(d)
        spy_open = float(daily["Open"].iloc[i])
        prev_close = float(daily["Close"].iloc[i-1])
        gap = (spy_open / prev_close - 1) * 100
        rows.append({"date": d.strftime("%Y-%m-%d"), "signal": sig,
                     "p_vol": pv, "p_pnl": pp, "p_wk": pw,
                     "spy_open": spy_open, "gap": gap})
    return pd.DataFrame(rows)


def attach_options(sigs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in sigs.iterrows():
        d = r["date"]; spy_open = r["spy_open"]
        otm = round(spy_open) + 5
        try:
            cs = list_expired_contracts("SPY", d, d, "call",
                                        strike_lo=otm-2, strike_hi=otm+2)
        except Exception: cs = []
        if not cs:
            print(f"  {d}: no contracts", flush=True); continue
        cs.sort(key=lambda c: abs(c.strike - otm))
        c = cs[0]
        try: bars = fetch_option_daily_bars(c.ticker, d, d)
        except Exception: bars = []
        if not bars or bars[0].open <= 0:
            print(f"  {d}: no bars for {c.ticker}", flush=True); continue
        b = bars[0]
        peak = (b.high / b.open - 1) * 100
        held = (b.close / b.open - 1) * 100
        out.append({**r.to_dict(), "strike": c.strike, "contract": c.ticker,
                    "entry": b.open, "high": b.high, "close": b.close,
                    "peak%": peak, "held%": held})
        print(f"  {d}  {r['signal']:18s}  gap={r['gap']:+5.1f}  "
              f"K={c.strike:.0f}  ${b.open:.2f}→${b.high:.2f} (peak {peak:+.0f}%)  "
              f"close ${b.close:.2f} ({held:+.0f}%)", flush=True)
    return pd.DataFrame(out)


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("No trades."); return
    base = df["held%"].mean()
    print(f"\n=== BASELINE (current algo: ATM/+$5 OTM, hold to bell) ===")
    print(f"  N={len(df)}  avg={base:+.1f}%  median={df['held%'].median():+.1f}%  "
          f"win={(df['held%']>0).mean()*100:.0f}%  best={df['held%'].max():+.0f}%  "
          f"worst={df['held%'].min():+.0f}%")

    print(f"\n=== MOD A: skip if |gap| >= threshold ===")
    print(f"{'thresh':>8} {'kept':>5} {'avg held%':>11} {'win%':>6} {'vs base':>10}")
    for g in [0.5, 1.0, 1.5, 2.0]:
        keep = df[df["gap"].abs() < g]
        if len(keep) == 0: continue
        a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
        print(f"  <{g:.1f}%   {len(keep):4d} {a:+10.1f}% {w:5.0f}%  {a-base:+9.1f}%")

    print(f"\n=== MOD B: take half off at +X% peak, hold rest to bell ===")
    print(f"{'take@':>6} {'avg ret%':>9} {'win%':>6} {'vs base':>10}")
    for tp in [100, 200, 300, 500]:
        rets = [(0.5*tp + 0.5*r["held%"]) if r["peak%"] >= tp else r["held%"]
                for _, r in df.iterrows()]
        a = np.mean(rets); w = (np.array(rets)>0).mean()*100
        print(f"  +{tp:4d}% {a:+8.1f}% {w:5.0f}%  {a-base:+9.1f}%")

    print(f"\n=== A + B combined (gap<1.5% AND take half at +200%) ===")
    keep = df[df["gap"].abs() < 1.5]
    if len(keep) > 0:
        rets = [(0.5*200 + 0.5*r["held%"]) if r["peak%"] >= 200 else r["held%"]
                for _, r in keep.iterrows()]
        a = np.mean(rets); w = (np.array(rets)>0).mean()*100
        print(f"  N={len(keep)}  avg={a:+.1f}%  win={w:.0f}%  vs base {a-base:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(CACHE):
        df = pd.read_csv(CACHE)
        print(f"Loaded {len(df)} cached trades from {CACHE}")
    else:
        sigs = score_window("SPY", args.days)
        print(f"Signals fired: {len(sigs)}")
        if len(sigs) == 0: return
        print(sigs["signal"].value_counts().to_dict())
        df = attach_options(sigs)
        if not df.empty:
            os.makedirs("data", exist_ok=True)
            df.to_csv(CACHE, index=False)
            print(f"\nCached {len(df)} trades to {CACHE}")
    report(df)


if __name__ == "__main__":
    main()
