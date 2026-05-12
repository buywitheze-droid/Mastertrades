"""Hunt for conditional sub-segments where our edge is much sharper than average.

There are no "glitches" in liquid SPY markets — those get arbitraged out
in milliseconds by HFTs. But there are sub-segments of trading days where
our model's edge concentrates: certain score levels, calendar slots,
cooldown periods, gap regimes, or direct-P&L predictors that beat the
indirect "predict volatility" target.

This module runs five orthogonal analyses on top of the existing
``volatility_classifier`` walk-forward predictions, so every metric is
honestly out-of-sample.

Pipeline
--------
1. ``score_bucket_calibration`` — bucket P_vol into deciles, ask whether
   higher scores pay disproportionately more.
2. ``conditional_by_weekday`` — does HOT-on-Wednesday outperform HOT-on-Friday?
3. ``conditional_by_cooldown`` — does HOT after a long calm streak win more?
4. ``conditional_by_gap`` — does HOT + big opening gap reinforce the signal?
5. ``train_direct_pnl_model`` — train a SECOND classifier whose target is
   "did the long-straddle make money", not "was the day in the top quintile".
   Different target = potentially sharper signal for the actual question.

Everything reports OOS win-rate, OOS expected return per dollar of
premium, and sample size. All computations use the existing per-day
straddle-return P&L model from ``strategy_sim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .strategy_sim import straddle_return
from .volatility_classifier import (
    BINARY_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    WEEKDAY_LEVELS,
    _one_hot_weekday,
    make_logreg,
    walk_forward_proba,
)
from .volatility_patterns import build_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_straddle(daily: pd.DataFrame, premium_pct: float = 0.011) -> pd.Series:
    """Compute realized straddle return for every aligned trading day.

    Uses the same conservative assumptions as ``strategy_sim.straddle_return``:
    payoff = max(close_move, 50% * intraday_max_excursion), minus premium and
    transaction costs.
    """
    rets = []
    for date, bar in daily.iterrows():
        rets.append(
            straddle_return(
                float(bar["Open"]),
                float(bar["High"]),
                float(bar["Low"]),
                float(bar["Close"]),
                premium_pct=premium_pct,
            )
        )
    return pd.Series(rets, index=daily.index, name="straddle_ret")


def _join_signals(
    daily: pd.DataFrame,
    p_vol_oos: pd.Series,
    feats: pd.DataFrame,
    premium_pct: float = 0.011,
) -> pd.DataFrame:
    """Join OHLC, OOS P_vol scores, features, and realized straddle return."""
    common = daily.index.intersection(p_vol_oos.index).intersection(feats.index)
    df = pd.DataFrame(index=common)
    df["open"] = daily.loc[common, "Open"]
    df["high"] = daily.loc[common, "High"]
    df["low"] = daily.loc[common, "Low"]
    df["close"] = daily.loc[common, "Close"]
    df["p_vol"] = p_vol_oos.loc[common]
    df["weekday"] = feats.loc[common, "weekday"]
    df["abs_gap_pct"] = feats.loc[common, "abs_gap_pct"]
    df["is_opex_week"] = feats.loc[common, "is_opex_week"]
    df["is_quarterly_opex_week"] = feats.loc[common, "is_quarterly_opex_week"]
    df["range_pct"] = feats.loc[common, "range_pct"]
    df["straddle_ret"] = _apply_straddle(daily.loc[common], premium_pct)
    df["win"] = (df["straddle_ret"] > 0).astype(int)
    return df.dropna(subset=["p_vol", "straddle_ret"])


def _agg(group: pd.DataFrame) -> dict:
    """Standard summary for a slice of trading days."""
    n = len(group)
    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "avg_ret": float("nan"),
                "median_ret": float("nan"), "ev": float("nan")}
    rets = group["straddle_ret"]
    return {
        "n": int(n),
        "win_rate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "median_ret": float(rets.median()),
        "ev": float(rets.mean()),  # synonym for clarity in callers
    }


# ---------------------------------------------------------------------------
# 1. Score-bucket calibration
# ---------------------------------------------------------------------------


def score_bucket_calibration(
    df: pd.DataFrame,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Bucket OOS P_vol into deciles, report straddle outcomes per bucket.

    Returns a frame with one row per decile (lowest score → highest):
        n            — number of trading days in the bucket
        p_low / p_high — score range
        win_rate     — fraction of days where straddle_ret > 0
        avg_ret      — mean return per dollar of premium
        median_ret   — median return per dollar of premium
        cum_dollar   — what $1 risked on every day in the bucket would become
                       (just sum + 1; not compounded since we treat each day
                       as an independent unit-risked trade)
    """
    df = df.dropna(subset=["p_vol", "straddle_ret"]).copy()
    df["bucket"] = pd.qcut(df["p_vol"], q=n_buckets, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby("bucket"):
        rows.append({
            "bucket": int(b),
            "p_low": float(g["p_vol"].min()),
            "p_high": float(g["p_vol"].max()),
            **_agg(g),
        })
    out = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. Calendar: weekday + OpEx
# ---------------------------------------------------------------------------


def conditional_by_weekday(df: pd.DataFrame, hot_threshold: float = 0.30) -> pd.DataFrame:
    """For HOT-signal days, break down win-rate / EV by weekday."""
    hot = df[df["p_vol"] >= hot_threshold]
    rows = []
    for wd in WEEKDAY_LEVELS:
        g = hot[hot["weekday"] == wd]
        rows.append({"weekday": wd, **_agg(g)})
    rows.append({"weekday": "ALL HOT", **_agg(hot)})
    return pd.DataFrame(rows)


def conditional_by_opex(df: pd.DataFrame, hot_threshold: float = 0.30) -> pd.DataFrame:
    """For HOT-signal days, break down win-rate / EV by OpEx-week status."""
    hot = df[df["p_vol"] >= hot_threshold].copy()
    rows = [
        {"slice": "OpEx week (any month)", **_agg(hot[hot["is_opex_week"] == 1])},
        {"slice": "Quarterly OpEx week", **_agg(hot[hot["is_quarterly_opex_week"] == 1])},
        {"slice": "Non-OpEx week", **_agg(hot[hot["is_opex_week"] == 0])},
        {"slice": "ALL HOT", **_agg(hot)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Cooldown patterns: how long since last HOT signal
# ---------------------------------------------------------------------------


def conditional_by_cooldown(df: pd.DataFrame, hot_threshold: float = 0.30) -> pd.DataFrame:
    """Among HOT-signal days, bucket by N consecutive non-HOT days BEFORE today.

    Reads as: "after we sat out N days of CALM/MID, when we finally got
    a HOT signal, did the model deliver more?"
    """
    df = df.copy().sort_index()
    is_hot = (df["p_vol"] >= hot_threshold).astype(int)
    not_hot = 1 - is_hot
    # streak counter: cumulative non-HOT days since last HOT, evaluated AT today
    cooldown = not_hot.groupby(is_hot.cumsum()).cumcount()
    df["cooldown"] = cooldown.shift(1).fillna(0).astype(int)

    bins = [-1, 0, 2, 5, 10, 1000]
    labels = ["right after HOT", "1-2 days off", "3-5 days off",
              "6-10 days off", "10+ days off"]
    df["cooldown_bucket"] = pd.cut(df["cooldown"], bins=bins, labels=labels)

    hot = df[df["p_vol"] >= hot_threshold]
    rows = []
    for lbl in labels:
        g = hot[hot["cooldown_bucket"] == lbl]
        rows.append({"slice": lbl, **_agg(g)})
    rows.append({"slice": "ALL HOT", **_agg(hot)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Gap conditioning
# ---------------------------------------------------------------------------


def conditional_by_gap(df: pd.DataFrame, hot_threshold: float = 0.30) -> pd.DataFrame:
    """For HOT-signal days, break by absolute opening gap size."""
    hot = df[df["p_vol"] >= hot_threshold].copy()
    bins = [-0.0001, 0.001, 0.003, 0.006, 0.012, 1.0]
    labels = ["~flat (<0.1%)", "small (0.1-0.3%)", "moderate (0.3-0.6%)",
              "big (0.6-1.2%)", "huge (>1.2%)"]
    hot["gap_bucket"] = pd.cut(hot["abs_gap_pct"], bins=bins, labels=labels)
    rows = []
    for lbl in labels:
        g = hot[hot["gap_bucket"] == lbl]
        rows.append({"slice": lbl, **_agg(g)})
    rows.append({"slice": "ALL HOT", **_agg(hot)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Direct-PnL classifier — predict straddle profitability directly
# ---------------------------------------------------------------------------


@dataclass
class DirectPnLResult:
    preds_oos: pd.DataFrame              # date × y_true × y_score
    base_rate: float
    auc: float
    avg_precision: float
    top_decile_precision: float
    top_decile_lift: float
    top_quintile_precision: float
    top_quintile_lift: float
    top_quintile_avg_ret: float          # mean straddle_ret on top-quintile days
    overlap_with_vol_top20: float        # how often direct-PnL top-20% overlaps vol top-20%


def _prepare_xy_pnl(features: pd.DataFrame, daily: pd.DataFrame, premium_pct: float = 0.011):
    """X / y for the direct-PnL target: y = (today's straddle_ret > 0)."""
    feat = features.dropna(subset=["range_pct"]).copy()
    common = feat.index.intersection(daily.index)
    feat = feat.loc[common]
    daily = daily.loc[common]
    straddle = _apply_straddle(daily, premium_pct=premium_pct)
    y = (straddle > 0).astype(int)

    base = feat[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    weekday_dummies = _one_hot_weekday(feat["weekday"])
    X = pd.concat([base, weekday_dummies], axis=1)

    keep = X.dropna().index
    X = X.loc[keep]
    y = y.loc[keep]
    realized = straddle.loc[keep]
    return X, y, realized


def train_direct_pnl_model(
    features: pd.DataFrame,
    daily: pd.DataFrame,
    p_vol_oos: pd.Series,
    premium_pct: float = 0.011,
    min_train: int = 1000,
    step: int = 21,
) -> DirectPnLResult:
    """Walk-forward train a logistic classifier predicting `straddle_ret > 0`.

    Compares its top-quintile precision against the existing volatility
    classifier on the SAME dates so we know whether re-targeting actually
    helps.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    X, y, realized = _prepare_xy_pnl(features, daily, premium_pct=premium_pct)
    preds = walk_forward_proba(X, y, make_logreg, min_train=min_train, step=step)
    if preds.empty:
        raise ValueError("Walk-forward produced no predictions.")
    preds = preds.assign(straddle_ret=realized.reindex(preds.index).values)
    base_rate = float(y.loc[preds.index].mean())

    auc = float(roc_auc_score(preds["y_true"], preds["y_score"]))
    ap = float(average_precision_score(preds["y_true"], preds["y_score"]))

    n = len(preds)
    sorted_p = preds["y_score"].sort_values(ascending=False)
    decile_threshold = float(sorted_p.iloc[max(int(0.10 * n) - 1, 0)])
    quintile_threshold = float(sorted_p.iloc[max(int(0.20 * n) - 1, 0)])

    p10_sel = preds["y_score"] >= decile_threshold
    p20_sel = preds["y_score"] >= quintile_threshold

    p10 = float(preds.loc[p10_sel, "y_true"].mean())
    p20 = float(preds.loc[p20_sel, "y_true"].mean())
    avg_ret_top20 = float(preds.loc[p20_sel, "straddle_ret"].mean())

    # Overlap with vol-classifier top quintile on the same dates
    vol_aligned = p_vol_oos.reindex(preds.index).dropna()
    if len(vol_aligned) > 0:
        v_thr = float(vol_aligned.sort_values(ascending=False).iloc[max(int(0.20 * len(vol_aligned)) - 1, 0)])
        vol_top = vol_aligned[vol_aligned >= v_thr].index
        pnl_top = preds[p20_sel].index
        overlap = len(pnl_top.intersection(vol_top)) / max(len(pnl_top), 1)
    else:
        overlap = float("nan")

    return DirectPnLResult(
        preds_oos=preds,
        base_rate=base_rate,
        auc=auc,
        avg_precision=ap,
        top_decile_precision=p10,
        top_decile_lift=p10 / base_rate if base_rate > 0 else float("nan"),
        top_quintile_precision=p20,
        top_quintile_lift=p20 / base_rate if base_rate > 0 else float("nan"),
        top_quintile_avg_ret=avg_ret_top20,
        overlap_with_vol_top20=float(overlap),
    )


# ---------------------------------------------------------------------------
# 6. Stacked filter — combine signals to find a sharper sub-segment
# ---------------------------------------------------------------------------


def stacked_filter(
    df: pd.DataFrame,
    pnl_preds: pd.DataFrame,
    p_vol_threshold: float = 0.30,
    p_pnl_threshold: float = 0.55,
) -> pd.DataFrame:
    """Sub-segment of days where BOTH signals agree (vol-classifier HOT
    AND direct-PnL classifier predicts profitable straddle).

    This is the candidate "exponential edge" filter: each model has its
    own ~3.5-4x lift; if their errors are partially independent, agreement
    should produce a sharper concentrated subset.

    Returns a single-row summary frame with the filter's stats and a
    comparison row for "either alone".
    """
    aligned = df.join(pnl_preds[["y_score"]].rename(columns={"y_score": "p_pnl"}), how="inner")
    aligned = aligned.dropna(subset=["p_vol", "p_pnl", "straddle_ret"])
    n_total = len(aligned)
    if n_total == 0:
        return pd.DataFrame()

    rows = [
        {"slice": "ALL DAYS (baseline)", **_agg(aligned)},
        {"slice": f"VolClass alone (p_vol≥{p_vol_threshold:.2f})",
         **_agg(aligned[aligned["p_vol"] >= p_vol_threshold])},
        {"slice": f"PnL-Class alone (p_pnl≥{p_pnl_threshold:.2f})",
         **_agg(aligned[aligned["p_pnl"] >= p_pnl_threshold])},
        {"slice": f"BOTH agree (vol≥{p_vol_threshold:.2f} AND pnl≥{p_pnl_threshold:.2f})",
         **_agg(aligned[(aligned["p_vol"] >= p_vol_threshold) &
                       (aligned["p_pnl"] >= p_pnl_threshold)])},
        {"slice": "EITHER fires",
         **_agg(aligned[(aligned["p_vol"] >= p_vol_threshold) |
                       (aligned["p_pnl"] >= p_pnl_threshold)])},
    ]
    out = pd.DataFrame(rows)
    out["pct_of_total"] = out["n"] / n_total
    return out


# ---------------------------------------------------------------------------
# Top-line ranking: which conditional is the biggest edge boost?
# ---------------------------------------------------------------------------


def rank_findings(
    df: pd.DataFrame,
    findings: dict[str, pd.DataFrame],
    hot_threshold: float = 0.30,
    min_n: int = 30,
) -> pd.DataFrame:
    """Pull every sub-slice from every analysis, compare its win-rate / EV
    to the HOT baseline, and rank by (EV uplift × log-sample-size)."""
    hot_baseline = _agg(df[df["p_vol"] >= hot_threshold])
    base_wr = hot_baseline["win_rate"]
    base_ev = hot_baseline["ev"]

    rows: list[dict] = []
    for source, frame in findings.items():
        if frame.empty:
            continue
        cols = list(frame.columns)
        # detect label column
        label_col = next((c for c in ("weekday", "slice", "bucket") if c in cols), cols[0])
        for _, r in frame.iterrows():
            n = int(r["n"]) if "n" in cols else 0
            if n < min_n:
                continue
            wr = float(r["win_rate"]) if "win_rate" in cols else float("nan")
            ev = float(r["ev"]) if "ev" in cols else float("nan")
            label = str(r[label_col])
            if "ALL" in label.upper():
                continue
            rows.append({
                "source": source,
                "slice": label,
                "n": n,
                "win_rate": wr,
                "avg_ret": ev,
                "win_rate_uplift": wr - base_wr,
                "ev_uplift": ev - base_ev,
                "score": (ev - base_ev) * np.log1p(n),
            })
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.attrs["hot_baseline_win_rate"] = base_wr
    out.attrs["hot_baseline_ev"] = base_ev
    out.attrs["hot_baseline_n"] = int(hot_baseline["n"])
    return out


# ---------------------------------------------------------------------------
# Convenience top-level runner
# ---------------------------------------------------------------------------


def run_full_analysis(
    daily: pd.DataFrame,
    p_vol_oos: pd.Series,
    premium_pct: float = 0.011,
    hot_threshold: float = 0.30,
) -> dict:
    """Run all five analyses + the ranking + the direct-PnL stacker.

    Returns a dict with keys:
        df, score_buckets, by_weekday, by_opex, by_cooldown, by_gap,
        direct_pnl, stacked, ranking
    """
    feats = build_features(daily)
    df = _join_signals(daily, p_vol_oos, feats, premium_pct=premium_pct)

    score_buckets = score_bucket_calibration(df)
    by_weekday = conditional_by_weekday(df, hot_threshold=hot_threshold)
    by_opex = conditional_by_opex(df, hot_threshold=hot_threshold)
    by_cooldown = conditional_by_cooldown(df, hot_threshold=hot_threshold)
    by_gap = conditional_by_gap(df, hot_threshold=hot_threshold)

    direct = train_direct_pnl_model(feats, daily, p_vol_oos, premium_pct=premium_pct)
    stacked = stacked_filter(df, direct.preds_oos,
                             p_vol_threshold=hot_threshold, p_pnl_threshold=0.55)

    findings = {
        "score_bucket": score_buckets.assign(
            slice=score_buckets.apply(
                lambda r: f"decile {int(r['bucket']) + 1} (p_vol {r['p_low']:.2f}-{r['p_high']:.2f})",
                axis=1,
            )
        ),
        "weekday": by_weekday,
        "opex": by_opex,
        "cooldown": by_cooldown,
        "gap": by_gap,
    }
    ranking = rank_findings(df, findings, hot_threshold=hot_threshold)

    return {
        "df": df,
        "score_buckets": score_buckets,
        "by_weekday": by_weekday,
        "by_opex": by_opex,
        "by_cooldown": by_cooldown,
        "by_gap": by_gap,
        "direct_pnl": direct,
        "stacked": stacked,
        "ranking": ranking,
        "hot_threshold": hot_threshold,
    }
