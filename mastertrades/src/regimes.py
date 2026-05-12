"""Day classification, volatility-regime detection, and historical analog search.

Three layers of analysis:

1. **Per-day labels** — given the volatility features in ``volatility.py``,
   tag each session as one of: ``flat``, ``quiet_trend``, ``rangy_chop``,
   ``trending``, ``volatile``. Pure quantile thresholds, no ML, easy to
   explain.

2. **Volatility regimes** — KMeans on a small standardized feature vector
   (range, body, efficiency, Yang-Zhang vol, volume z-score). Returns a
   regime label per day plus a centroid description so a regime can be
   given a meaningful name ("low-vol drift", "high-vol chop", etc.).

3. **Analog matching + change-point detection** — given today's recent
   volatility footprint, search ALL prior history for the most similar
   windows and report what happened in the N days that followed each.
   Plus a simple CUSUM-style change-point detector to flag when a stable
   pattern broke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import volatility as vol


# ---------------------------------------------------------------------------
# Per-day labels
# ---------------------------------------------------------------------------


def label_days(
    daily: pd.DataFrame,
    range_col: str = "RangePct",
    body_col: str = "BodyPct",
    eff_col: str = "EfficiencyRatio",
    flat_q: float = 0.20,
    volatile_q: float = 0.80,
    trend_q: float = 0.70,
    chop_q: float = 0.30,
) -> pd.Series:
    """Classify each day into one of five labels using rolling quantiles.

    The thresholds are computed over the full input frame so labels are
    relative to the regime in the data you pass in. If you want labels
    relative to long-run history, pass long history; if relative to the
    last 2 years, pass only that.
    """
    needed = {range_col, body_col, eff_col}
    missing = needed - set(daily.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rng = daily[range_col]
    body = daily[body_col]
    eff = daily[eff_col]

    rng_lo, rng_hi = rng.quantile(flat_q), rng.quantile(volatile_q)
    eff_lo, eff_hi = eff.quantile(chop_q), eff.quantile(trend_q)

    label = pd.Series("normal", index=daily.index, dtype="object", name="DayLabel")
    label[(rng <= rng_lo) & (eff <= eff_hi)] = "flat"
    label[(rng <= rng_lo) & (eff > eff_hi)] = "quiet_trend"
    label[(rng >= rng_hi) & (eff <= eff_lo)] = "rangy_chop"
    label[(rng >= rng_hi) & (eff > eff_hi)] = "trending"
    label[(rng >= rng_hi) & (eff > eff_lo) & (eff <= eff_hi)] = "volatile"
    return label


# ---------------------------------------------------------------------------
# Regime clustering
# ---------------------------------------------------------------------------


@dataclass
class RegimeModel:
    """Fitted KMeans regime model + supporting metadata."""

    kmeans: KMeans
    scaler: StandardScaler
    feature_columns: list[str]
    centroids: pd.DataFrame   # un-scaled centroids in original feature space
    labels: pd.Series         # regime label per row used in fit
    name_map: dict[int, str]  # cluster id -> human-readable name


def _build_regime_features(daily: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Standard feature set used for regime clustering."""
    yz = vol.yang_zhang_vol(daily, window=window)
    feats = pd.DataFrame(
        {
            "RangePct": vol.daily_range_pct(daily),
            "BodyPct": vol.body_pct(daily),
            "EfficiencyRatio": vol.efficiency_ratio(daily),
            "YangZhangVol": yz,
            "VolumeZ": (
                (daily["Volume"] - daily["Volume"].rolling(60).mean())
                / daily["Volume"].rolling(60).std()
            ),
        }
    )
    return feats.dropna()


def _name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Heuristic naming: rank clusters by Yang-Zhang vol then describe each."""
    ranked = centroids.sort_values("YangZhangVol")
    n = len(ranked)
    bucket_names = (
        ["calm"] if n == 1
        else ["calm", "stressed"] if n == 2
        else ["calm", "normal", "stressed"] if n == 3
        else ["calm", "normal", "active", "stressed"] if n == 4
        else ["calm", "quiet", "normal", "active", "stressed"]
    )
    names: dict[int, str] = {}
    for rank, cid in enumerate(ranked.index):
        base = bucket_names[min(rank, len(bucket_names) - 1)]
        eff = centroids.loc[cid, "EfficiencyRatio"]
        suffix = " trend" if eff > 0.55 else " chop" if eff < 0.30 else ""
        names[int(cid)] = f"{base}{suffix}"
    return names


def fit_regimes(
    daily: pd.DataFrame,
    n_regimes: int = 4,
    window: int = 20,
    random_state: int = 0,
) -> RegimeModel:
    feats = _build_regime_features(daily, window=window)
    scaler = StandardScaler().fit(feats.values)
    X = scaler.transform(feats.values)
    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=random_state).fit(X)

    labels = pd.Series(km.labels_, index=feats.index, name="Regime")
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=feats.columns,
    )
    names = _name_clusters(centroids)
    return RegimeModel(
        kmeans=km,
        scaler=scaler,
        feature_columns=list(feats.columns),
        centroids=centroids,
        labels=labels,
        name_map=names,
    )


def regime_summary(model: RegimeModel) -> pd.DataFrame:
    """Tidy summary of each regime: name, centroid features, frequency."""
    counts = model.labels.value_counts().sort_index()
    out = model.centroids.copy()
    out.insert(0, "name", [model.name_map[i] for i in out.index])
    out["count"] = counts.reindex(out.index, fill_value=0).values
    out["share"] = (out["count"] / out["count"].sum()).round(3)
    return out


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------


def cusum_change_points(
    series: pd.Series,
    threshold: float = 4.0,
    drift: float = 0.0,
) -> list[pd.Timestamp]:
    """Two-sided CUSUM change-point detector on a 1D series.

    Standardizes the input, then accumulates positive and negative deviations
    from the mean. When either accumulator exceeds ``threshold`` standard
    deviations, a change-point is recorded and both accumulators reset.

    Threshold of 4–5 is a reasonable default for monthly-scale regime breaks
    on rolling realized vol.
    """
    s = series.dropna()
    if s.empty:
        return []
    z = (s - s.mean()) / s.std(ddof=0)
    pos = neg = 0.0
    points: list[pd.Timestamp] = []
    for ts, x in z.items():
        pos = max(0.0, pos + x - drift)
        neg = min(0.0, neg + x + drift)
        if pos > threshold or neg < -threshold:
            points.append(ts)
            pos = neg = 0.0
    return points


# ---------------------------------------------------------------------------
# Analog matching
# ---------------------------------------------------------------------------


def _standardize(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def find_analog_windows(
    feature_history: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 5,
    min_gap_days: int = 60,
) -> pd.DataFrame:
    """Find the ``top_k`` historical windows most similar to ``target``.

    Both inputs must have the *same columns* and ``target`` defines the
    window length. We slide an equal-length window across ``feature_history``,
    standardize each candidate independently (so similarity is about the
    *shape* of the regime, not its absolute level), and rank by Euclidean
    distance to the standardized target.

    To avoid returning overlapping near-duplicates, we enforce a minimum
    spacing of ``min_gap_days`` between selected match start dates.

    Returns a DataFrame with columns ``start``, ``end``, ``distance``.
    """
    if feature_history.empty or target.empty:
        return pd.DataFrame(columns=["start", "end", "distance"])
    if list(feature_history.columns) != list(target.columns):
        raise ValueError("feature_history and target must share columns/order.")

    history = feature_history.dropna()
    target = target.dropna()
    L = len(target)
    if len(history) <= L:
        return pd.DataFrame(columns=["start", "end", "distance"])

    target_std = _standardize(target.values).flatten()

    history_idx = history.index
    history_arr = history.values
    distances = np.empty(len(history) - L + 1)

    for i in range(len(history) - L + 1):
        window = history_arr[i : i + L]
        window_std = _standardize(window).flatten()
        distances[i] = np.linalg.norm(window_std - target_std)

    order = np.argsort(distances)
    selected: list[tuple[int, float]] = []
    for idx in order:
        start_ts = history_idx[idx]
        if any(
            abs((start_ts - history_idx[s_idx]).days) < min_gap_days
            for s_idx, _ in selected
        ):
            continue
        selected.append((idx, distances[idx]))
        if len(selected) >= top_k:
            break

    rows = [
        {
            "start": history_idx[i],
            "end": history_idx[i + L - 1],
            "distance": float(d),
        }
        for i, d in selected
    ]
    return pd.DataFrame(rows)


def analog_forward_returns(
    matches: pd.DataFrame,
    price: pd.Series,
    horizons_days: Iterable[int] = (5, 10, 20, 60),
) -> pd.DataFrame:
    """For each analog window, compute realized forward returns at given horizons.

    Use this to summarize what tended to happen *after* historical setups
    that looked similar to the current one.
    """
    rows = []
    px = price.dropna()
    for _, m in matches.iterrows():
        end = m["end"]
        if end not in px.index:
            pos = px.index.searchsorted(end)
            if pos >= len(px):
                continue
            end = px.index[pos]
        start_px = px.loc[end]
        out = {"end": end}
        for h in horizons_days:
            target_pos = px.index.searchsorted(end) + h
            if target_pos >= len(px):
                out[f"fwd_{h}d"] = np.nan
            else:
                out[f"fwd_{h}d"] = float(px.iloc[target_pos] / start_px - 1.0)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("end")
