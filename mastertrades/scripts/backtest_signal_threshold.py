"""
Extended backtest of the leakage-free walk-forward jackpot signal.

Phase 1: score every SPY session in the past N years (incremental cache).
Phase 2: pull 0DTE +$5 OTM call OHLC for every fired signal (incremental cache).
Phase 3: report multiple variants:
  - Baseline (all GO_HOT/GO_JACKPOT, hold to bell)
  - Mod A: skip if |gap| >= threshold (sweep)
  - Mod B: take half off at +X% peak (sweep)
  - Mod C: tighten threshold — require p_pnl >= X (sweep)
  - Mod D: tighten threshold — require p_weekly >= X (sweep)
  - Combined best-of

Run incrementally:
    python scripts/backtest_signal_threshold.py --score --max-new 50
    python scripts/backtest_signal_threshold.py --options
    python scripts/backtest_signal_threshold.py --report
Or all-in-one with: --years 3 --score --options --report
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

SCORES_CSV = "data/wf_scores_SPY.csv"
TRADES_CSV = "data/wf_trades_SPY.csv"


def load_scores() -> pd.DataFrame:
    if os.path.exists(SCORES_CSV):
        df = pd.read_csv(SCORES_CSV)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame(columns=["date","signal","p_vol","p_pnl","p_wk","spy_open","gap"])


def save_scores(df: pd.DataFrame) -> None:
    os.makedirs("data", exist_ok=True)
    df.sort_values("date").to_csv(SCORES_CSV, index=False)


def score_phase(years: int, max_new: int) -> int:
    daily = fetch_or_load_daily("SPY", refresh=False).copy()
    if daily.index.tz is not None: daily.index = daily.index.tz_localize(None)
    daily.index = pd.to_datetime(daily.index).normalize()

    end = daily.index.max()
    start = end - pd.Timedelta(days=int(years*365))
    win = [d for d in daily.loc[start:end].index]
    cached = set(load_scores()["date"].tolist())
    todo = [d for d in win if d not in cached]
    print(f"Window: {win[0].date()} → {win[-1].date()} ({len(win)} sessions, {len(todo)} new)", flush=True)
    if not todo: return 0

    rows = load_scores().to_dict("records")
    n_done = 0
    for d in todo:
        if n_done >= max_new: break
        cutoff = _walkforward_cutoff(d)
        sl = daily[daily.index <= d]
        if len(sl) < 300: continue
        try:
            bundle = train_or_load_walkforward_models("SPY", cutoff, sl)
        except Exception as e:
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
        i = daily.index.get_loc(d)
        spy_open = float(daily["Open"].iloc[i])
        prev_close = float(daily["Close"].iloc[i-1])
        gap = (spy_open / prev_close - 1) * 100
        rows.append({"date": d, "signal": sig, "p_vol": pv, "p_pnl": pp, "p_wk": pw,
                     "spy_open": spy_open, "gap": gap})
        n_done += 1
        if n_done % 25 == 0:
            save_scores(pd.DataFrame(rows))
            print(f"  ... scored {n_done}/{max_new} (last={d.date()})", flush=True)
    save_scores(pd.DataFrame(rows))
    print(f"Scored {n_done} new sessions. Total cached: {len(rows)}", flush=True)
    return n_done


def options_phase() -> None:
    sigs = load_scores()
    sigs = sigs[sigs["signal"].isin(["GO_HOT","GO_JACKPOT","GO_ULTRA_JACKPOT"])].copy()
    print(f"Fired signals in cache: {len(sigs)}", flush=True)

    if os.path.exists(TRADES_CSV):
        existing = pd.read_csv(TRADES_CSV)
        existing["date"] = pd.to_datetime(existing["date"])
        done_dates = set(existing["date"].tolist())
    else:
        existing = pd.DataFrame()
        done_dates = set()
    todo = sigs[~sigs["date"].isin(done_dates)]
    print(f"To fetch options for: {len(todo)}", flush=True)
    if todo.empty:
        return

    new_rows = []
    for _, r in todo.iterrows():
        d = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
        spy_open = r["spy_open"]; otm = round(spy_open) + 5
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
        new_rows.append({**r.to_dict(), "strike": c.strike, "contract": c.ticker,
                         "entry": b.open, "high": b.high, "close": b.close,
                         "peak%": peak, "held%": held})
        print(f"  {d}  {r['signal']:18s}  gap={r['gap']:+5.1f}  pV={r['p_vol']:.2f} pP={r['p_pnl']:.2f}  "
              f"K={c.strike:.0f}  ${b.open:.2f}→${b.high:.2f} (peak {peak:+.0f}%)  close ${b.close:.2f} ({held:+.0f}%)",
              flush=True)
    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    out.sort_values("date").to_csv(TRADES_CSV, index=False)
    print(f"Total trades cached: {len(out)}")


def report_phase() -> None:
    if not os.path.exists(TRADES_CSV):
        print("No trades cached. Run --score then --options first."); return
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n=== Trade universe: {len(df)} fired signals over "
          f"{df['date'].min().date()} → {df['date'].max().date()} ===")
    print(f"  Signal mix: {df['signal'].value_counts().to_dict()}")

    base = df["held%"].mean()
    base_win = (df["held%"]>0).mean()*100
    big_winners = df[df["held%"]>=200]
    print(f"\n=== BASELINE ===")
    print(f"  N={len(df)}  avg={base:+.1f}%  median={df['held%'].median():+.1f}%  "
          f"win={base_win:.0f}%  big winners (≥200%)={len(big_winners)}  "
          f"best={df['held%'].max():+.0f}%  worst={df['held%'].min():+.0f}%")

    print(f"\n=== MOD A: skip if |gap| >= threshold ===")
    print(f"{'thresh':>8} {'kept':>5} {'avg held%':>11} {'win%':>6} {'big_win':>7} {'vs base':>10}")
    for g in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        keep = df[df["gap"].abs() < g]
        if len(keep)==0: continue
        a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
        bw = (keep["held%"]>=200).sum()
        print(f"  <{g:.1f}%   {len(keep):4d} {a:+10.1f}% {w:5.0f}% {bw:6d}  {a-base:+9.1f}%")

    print(f"\n=== MOD B: take half off at +X% peak, hold rest to bell ===")
    print(f"{'take@':>6} {'avg ret%':>9} {'win%':>6} {'vs base':>10}")
    for tp in [100, 200, 300, 500, 1000]:
        rets = [(0.5*tp + 0.5*r["held%"]) if r["peak%"] >= tp else r["held%"]
                for _, r in df.iterrows()]
        a = np.mean(rets); w = (np.array(rets)>0).mean()*100
        print(f"  +{tp:4d}% {a:+8.1f}% {w:5.0f}%  {a-base:+9.1f}%")

    print(f"\n=== MOD C: tighten p_pnl threshold ===")
    print(f"{'min p_pnl':>10} {'kept':>5} {'avg held%':>11} {'win%':>6} {'big_win':>7} {'vs base':>10}")
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        keep = df[df["p_pnl"] >= thr]
        if len(keep)==0: continue
        a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
        bw = (keep["held%"]>=200).sum()
        print(f"  >={thr:.2f}    {len(keep):4d} {a:+10.1f}% {w:5.0f}% {bw:6d}  {a-base:+9.1f}%")

    print(f"\n=== MOD D: tighten p_weekly threshold ===")
    print(f"{'min p_wk':>10} {'kept':>5} {'avg held%':>11} {'win%':>6} {'big_win':>7} {'vs base':>10}")
    for thr in [0.05, 0.08, 0.10, 0.13, 0.15, 0.20]:
        keep = df[df["p_wk"] >= thr]
        if len(keep)==0: continue
        a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
        bw = (keep["held%"]>=200).sum()
        print(f"  >={thr:.2f}    {len(keep):4d} {a:+10.1f}% {w:5.0f}% {bw:6d}  {a-base:+9.1f}%")

    print(f"\n=== MOD E: tighten p_vol threshold (above HOT_THRESHOLD=0.30) ===")
    print(f"{'min p_vol':>10} {'kept':>5} {'avg held%':>11} {'win%':>6} {'big_win':>7} {'vs base':>10}")
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        keep = df[df["p_vol"] >= thr]
        if len(keep)==0: continue
        a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
        bw = (keep["held%"]>=200).sum()
        print(f"  >={thr:.2f}    {len(keep):4d} {a:+10.1f}% {w:5.0f}% {bw:6d}  {a-base:+9.1f}%")

    print(f"\n=== Combined sweep: gap-skip + p_pnl tighten (no profit-take) ===")
    print(f"{'gap<':>5} {'p_pnl≥':>7} {'kept':>5} {'avg held%':>11} {'win%':>6} {'big_win':>7} {'vs base':>10}")
    for g in [2.0, 2.5, 3.0]:
        for thr in [0.10, 0.15, 0.20]:
            keep = df[(df["gap"].abs() < g) & (df["p_pnl"] >= thr)]
            if len(keep)==0: continue
            a = keep["held%"].mean(); w = (keep["held%"]>0).mean()*100
            bw = (keep["held%"]>=200).sum()
            print(f"  <{g:.1f} >={thr:.2f}   {len(keep):4d} {a:+10.1f}% {w:5.0f}% {bw:6d}  {a-base:+9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--options", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()
    if args.score: score_phase(args.years, args.max_new)
    if args.options: options_phase()
    if args.report: report_phase()
    if not (args.score or args.options or args.report):
        ap.print_help()


if __name__ == "__main__": main()
