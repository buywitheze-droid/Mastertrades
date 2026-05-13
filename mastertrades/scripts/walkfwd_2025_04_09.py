"""Strict walk-forward retrain for 2025-04-09 SPY.

Training cutoff = 2025-04-08 close. Models see ZERO data from 4/9 or later.
Then score 4/9 (whose features only use lagged 4/8-and-earlier values).

Compares to the post-hoc score:
  p_vol=0.999  p_pnl=0.895  p_weekly=0.669  → GO_ULTRA_JACKPOT
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import pandas as pd

from src.scanner import fetch_or_load_daily
from src.volatility_patterns import build_features
from src.volatility_classifier import prepare_xy, fit_full, make_logreg, NUMERIC_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES, _one_hot_weekday
from src.jackpot_scanner import (
    _prepare_pnl_xy, _prepare_weekly_xy,
    estimate_premium_pct, estimate_weekly_premium_pct,
    classify_signal, MIN_TRAIN_ROWS,
)

CUTOFF = pd.Timestamp("2025-04-08")  # last day allowed in training set
TARGET = pd.Timestamp("2025-04-09")  # day to predict

def main():
    full = fetch_or_load_daily("SPY", history_years=10).copy()
    full.index = pd.to_datetime(full.index).tz_localize(None).normalize()
    full = full.sort_index()

    train = full[full.index <= CUTOFF].copy()
    pred  = full[full.index <= TARGET].copy()  # for building 4/9's lagged features

    print(f"Training rows  (≤ {CUTOFF.date()}): {len(train)}")
    print(f"Prediction set (≤ {TARGET.date()}): {len(pred)}")

    # ── Train all three models on data ≤ 4/8 ─────────────────────────────────
    train_feats = build_features(train)
    X_vol, y_vol, vol_thresh = prepare_xy(train_feats, volatile_quantile=0.80)
    print(f"\nVol classifier: trained on {len(X_vol)} rows, vol-day threshold = {vol_thresh*100:.2f}% range")
    vol_model = fit_full(X_vol, y_vol, make_logreg)

    prem_pct = estimate_premium_pct(train)
    wkly_prem_pct = estimate_weekly_premium_pct(train)
    print(f"Estimated 0DTE premium pct: {prem_pct*100:.3f}%   weekly: {wkly_prem_pct*100:.3f}%")

    X_pnl, y_pnl, _ = _prepare_pnl_xy(train_feats, train, prem_pct)
    pnl_model = fit_full(X_pnl, y_pnl, make_logreg)
    print(f"PnL classifier: trained on {len(X_pnl)} rows, base-rate winners = {y_pnl.mean()*100:.1f}%")

    X_wk, y_wk, _ = _prepare_weekly_xy(train_feats, train, wkly_prem_pct)
    wk_model = fit_full(X_wk, y_wk, make_logreg)
    print(f"Weekly classifier: trained on {len(X_wk)} rows, base-rate winners = {y_wk.mean()*100:.1f}%")

    # ── Build 4/9 prediction row using lagged features ───────────────────────
    pred_feats = build_features(pred)
    if TARGET not in pred_feats.index:
        print(f"!! 4/9 not in feature index: {pred_feats.index.max()}")
        return

    row = pred_feats.loc[[TARGET]]
    base = row[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    wd_dum = _one_hot_weekday(row["weekday"])
    X_pred = pd.concat([base, wd_dum], axis=1)
    # Align columns to training (handle any missing weekday dummy column)
    X_pred = X_pred.reindex(columns=X_vol.columns, fill_value=0.0)

    p_vol = float(vol_model.predict_proba(X_pred)[0, 1])
    p_pnl = float(pnl_model.predict_proba(X_pred)[0, 1])
    p_wk  = float(wk_model.predict_proba(X_pred)[0, 1])
    sig   = classify_signal(p_vol, p_pnl, p_wk)

    print("\n" + "="*72)
    print("STRICT WALK-FORWARD RESULT — SPY 2025-04-09")
    print(f"Models trained on data ≤ 2025-04-08 only. Zero lookahead.")
    print("="*72)
    print(f"  p_vol    = {p_vol:.3f}")
    print(f"  p_pnl    = {p_pnl:.3f}")
    print(f"  p_weekly = {p_wk:.3f}")
    print(f"  SIGNAL   = {sig}")
    print()
    print(f"  Compare to post-hoc (full-history models):")
    print(f"    p_vol=0.999  p_pnl=0.895  p_weekly=0.669  → GO_ULTRA_JACKPOT")
    print()
    if sig == "GO_ULTRA_JACKPOT":
        print(f"  ✅ CONFIRMED: the model would have fired GO_ULTRA_JACKPOT")
        print(f"     pre-market on 4/9 with NO knowledge of 4/9 itself.")
    elif sig == "GO_JACKPOT":
        print(f"  ⚠ PARTIAL: would have fired GO_JACKPOT but missed the weekly tier.")
    elif sig == "GO_HOT":
        print(f"  ⚠ PARTIAL: vol HOT but PnL classifier did NOT fire.")
    else:
        print(f"  ❌ MISS: model would have said SKIP. Post-hoc result was data leakage.")

if __name__ == "__main__":
    main()
