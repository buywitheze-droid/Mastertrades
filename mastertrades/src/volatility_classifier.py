"""Predict P(today is a top-quintile range day) from pre-open features.

Two models, both backtested with proper expanding-window walk-forward on the
full SPY history so we never train on the future:

- **Logistic regression** (L2, scaled features) — interpretable baseline.
- **Histogram Gradient Boosting** — captures non-linear interactions.

Pipeline:

    raw daily OHLCV  →  build_features (volatility_patterns)
                     →  prepare_xy (drop NaNs, one-hot weekday, set y)
                     →  walk_forward_proba (expanding window, periodic refit)
                     →  evaluate (AUC, calibration, top-decile lift, etc.)
                     →  fit_full (final model trained on all data)
                     →  score_today (probability for the most recent session)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


NUMERIC_FEATURES = [
    "lag1_range", "lag2_range", "lag5_avg_range", "range_compression_ratio",
    "lag1_body", "lag1_close_strength",
    "gap_pct", "abs_gap_pct",
    "lag1_volume_z", "lag5_avg_volume_z",
    "realized_vol_5d", "realized_vol_20d", "vol_regime_shift",
    "rsi14", "bb_pos", "pct_in_20d_range",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "days_to_opex",
]
BINARY_FEATURES = [
    "is_opex_day", "is_opex_week", "is_quarterly_opex_week",
    "is_turn_of_month", "is_first_trading_day_of_month",
    "is_last_trading_day_of_month",
    "is_lag1_nr4", "is_lag1_nr7",
    "is_after_flat", "is_after_volatile",
    "is_after_2_flat", "is_after_3plus_flat",
]
ORDINAL_FEATURES = ["lag1_color", "week_of_month"]
CATEGORICAL_FEATURES = ["weekday"]

WEEKDAY_LEVELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _one_hot_weekday(s: pd.Series) -> pd.DataFrame:
    """Stable one-hot encoding for weekday so column order never depends on the slice."""
    out = pd.DataFrame(0, index=s.index, columns=[f"wd_{d}" for d in WEEKDAY_LEVELS], dtype=int)
    for d in WEEKDAY_LEVELS:
        out.loc[s == d, f"wd_{d}"] = 1
    return out


def prepare_xy(features: pd.DataFrame, volatile_quantile: float = 0.80) -> tuple[pd.DataFrame, pd.Series, float]:
    """Split the feature panel into (X, y, threshold).

    The threshold uses the full input window — keep this in mind when reading
    backtest metrics: definition of "volatile day" is global to the data, but
    the *model* never sees future feature values during training.
    """
    feat = features.dropna(subset=["range_pct"]).copy()
    threshold = float(feat["range_pct"].quantile(volatile_quantile))
    y = (feat["range_pct"] >= threshold).astype(int)

    base = feat[NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES].astype(float)
    weekday_dummies = _one_hot_weekday(feat["weekday"])
    X = pd.concat([base, weekday_dummies], axis=1)

    keep = X.dropna().index
    X = X.loc[keep]
    y = y.loc[keep]
    return X, y, threshold


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def make_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
        ]
    )


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def walk_forward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    min_train: int = 1000,
    step: int = 21,
) -> pd.DataFrame:
    """Expanding-window walk-forward predictions.

    At each step:
      train on rows [0 .. t)
      predict rows [t .. t+step)
      advance t by step

    Refit cadence ``step`` is chosen to balance leakage realism with runtime.
    21 trading days ≈ monthly refit, which is more frequent than most
    production systems.
    """
    if len(X) <= min_train:
        raise ValueError(f"Need more than {min_train} rows; got {len(X)}.")

    rows: list[dict] = []
    t = min_train
    while t < len(X):
        end_train = t
        end_test = min(t + step, len(X))
        X_tr = X.iloc[:end_train]
        y_tr = y.iloc[:end_train]
        X_te = X.iloc[end_train:end_test]
        y_te = y.iloc[end_train:end_test]

        # Skip degenerate train slices (extremely rare on 1000+ rows but cheap to guard).
        if y_tr.nunique() < 2:
            t = end_test
            continue

        model = model_factory()
        model.fit(X_tr.values, y_tr.values)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_te.values)[:, 1]
        else:
            p = model.decision_function(X_te.values)

        for i, idx in enumerate(X_te.index):
            rows.append({"date": idx, "y_true": int(y_te.iloc[i]), "y_score": float(p[i])})
        t = end_test

    return pd.DataFrame(rows).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class BacktestSummary:
    n: int
    base_rate: float
    auc: float
    avg_precision: float
    brier: float
    top_decile_precision: float
    top_decile_recall: float
    top_decile_lift: float
    top_quintile_precision: float
    top_quintile_recall: float
    top_quintile_lift: float
    threshold_50: float
    confusion_at_50: dict
    threshold_for_target_precision: float | None
    target_precision: float | None
    calibration: list[dict]


def _confusion_at(threshold: float, y_true: pd.Series, y_score: pd.Series) -> dict:
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_backtest(preds: pd.DataFrame, calibration_bins: int = 10) -> BacktestSummary:
    y = preds["y_true"]
    p = preds["y_score"]
    n = len(preds)
    base_rate = float(y.mean())

    # Top-decile / top-quintile cuts: take the highest-scoring 10% / 20% of predictions.
    sorted_p = p.sort_values(ascending=False)
    decile_threshold = float(sorted_p.iloc[max(int(0.10 * n) - 1, 0)])
    quintile_threshold = float(sorted_p.iloc[max(int(0.20 * n) - 1, 0)])

    def _precision_recall(threshold: float) -> tuple[float, float]:
        sel = p >= threshold
        if sel.sum() == 0:
            return float("nan"), float("nan")
        prec = float(y[sel].mean())
        rec = float(y[sel].sum() / max(y.sum(), 1))
        return prec, rec

    p10, r10 = _precision_recall(decile_threshold)
    p20, r20 = _precision_recall(quintile_threshold)

    # Find a threshold that yields ~2× lift if it exists.
    target_precision = base_rate * 2.0
    threshold_for_target = None
    achieved_target_precision = None
    for thr in np.linspace(p.max(), p.min(), 200):
        sel = p >= thr
        if sel.sum() < max(int(0.02 * n), 5):
            continue
        prec = float(y[sel].mean())
        if prec >= target_precision:
            threshold_for_target = float(thr)
            achieved_target_precision = prec
            break

    # Reliability diagram: equal-frequency bins on predicted score.
    quantiles = np.linspace(0, 1, calibration_bins + 1)
    bins = pd.qcut(p, q=quantiles, labels=False, duplicates="drop")
    cal_rows: list[dict] = []
    for b, group in pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin"):
        cal_rows.append({
            "bin": int(b),
            "n": int(len(group)),
            "mean_predicted": float(group["p"].mean()),
            "actual_rate": float(group["y"].mean()),
            "p_low": float(group["p"].min()),
            "p_high": float(group["p"].max()),
        })

    return BacktestSummary(
        n=n,
        base_rate=base_rate,
        auc=float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        avg_precision=float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan"),
        brier=float(brier_score_loss(y, p)),
        top_decile_precision=p10,
        top_decile_recall=r10,
        top_decile_lift=p10 / base_rate if base_rate > 0 else float("nan"),
        top_quintile_precision=p20,
        top_quintile_recall=r20,
        top_quintile_lift=p20 / base_rate if base_rate > 0 else float("nan"),
        threshold_50=0.5,
        confusion_at_50=_confusion_at(0.5, y, p),
        threshold_for_target_precision=threshold_for_target,
        target_precision=achieved_target_precision,
        calibration=cal_rows,
    )


# ---------------------------------------------------------------------------
# Final model + scoring
# ---------------------------------------------------------------------------


def fit_full(X: pd.DataFrame, y: pd.Series, model_factory: Callable[[], object]):
    model = model_factory()
    model.fit(X.values, y.values)
    return model


def score_dataframe(model, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X.values)[:, 1]
    else:
        p = model.decision_function(X.values)
    return pd.Series(p, index=X.index, name="vol_prob")


def feature_importances(model, feature_names: Iterable[str]) -> pd.DataFrame:
    """Standardized linear coefficients for LogReg, gain importance for GBM."""
    feature_names = list(feature_names)
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
        coefs = np.asarray(clf.coef_).ravel()
        return (
            pd.DataFrame({"feature": feature_names, "weight": coefs})
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
            .drop(columns=["abs_weight"])
            .reset_index(drop=True)
        )
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        return (
            pd.DataFrame({"feature": feature_names, "weight": importances})
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )
    raise TypeError(f"Unsupported model type: {type(model)!r}")
