"""Multi-ticker live scanner built on the volatility classifier.

Pipeline per ticker:

    yfinance daily OHLCV  →  volatility_patterns.build_features
                          →  volatility_classifier.prepare_xy
                          →  load (or train+cache) per-ticker LogReg model
                          →  score the most recent session
                          →  rank universe by P(volatile day)

Caches:
    data/<TICKER>_1d.csv      ← OHLCV (refreshed once per `data_fresh_hours`)
    models/<TICKER>_logreg.joblib  ← fitted model (retrain weekly or on demand)

Outputs (per ticker):
    p_vol        — model's probability today is in the top quintile of range
    base_rate    — long-run frequency of volatile days for THIS ticker (~20%)
    lift         — p_vol / base_rate (>1 means more likely than usual)
    last_close, pct_change, rsi14, bb_pos, lag1_range, range_compression, abs_gap_pct

Use as a CLI (single shot):
    python -m src.scanner --tickers SPY,QQQ,IWM,AAPL

Or programmatically::

    from src.scanner import scan_universe
    df, errors = scan_universe(["SPY", "QQQ", "AAPL"])
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import yfinance as yf

from src.volatility_classifier import (
    fit_full,
    make_logreg,
    prepare_xy,
    score_dataframe,
)
from src.volatility_patterns import build_features


# Liquid US ETFs + the most-watched single names. All have >15y of clean daily
# history on Yahoo, which is enough to train and walk-forward the classifier.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "TSLA", "AMD",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"

DATA_FRESH_HOURS_DEFAULT = 0.25  # refresh Yahoo data every 15 minutes
HISTORY_YEARS_DEFAULT = 20
MIN_TRAIN_ROWS = 500

logger = logging.getLogger("scanner")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _safe_ticker_filename(ticker: str) -> str:
    """Sanitize ticker for use as a filename component (handles BRK.B, BF-B etc.)."""
    return ticker.upper().replace(".", "_").replace("/", "_")


def _is_stale(path: Path, max_age_hours: float) -> bool:
    age_hours = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600.0
    return age_hours >= max_age_hours


def fetch_or_load_daily(
    ticker: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    refresh: bool = True,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
    history_years: int = HISTORY_YEARS_DEFAULT,
) -> pd.DataFrame:
    """Return daily OHLCV for `ticker`, refreshing cache if missing/stale.

    Data source priority (when refreshing):
      1. Polygon.io  — if POLYGON_API_KEY is set (exchange-quality, adjusted)
      2. Yahoo Finance — fallback when Polygon is unavailable or fails
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{_safe_ticker_filename(ticker)}_1d.csv"

    need_refresh = refresh and (not path.exists() or _is_stale(path, data_fresh_hours))
    if not path.exists():
        need_refresh = True

    if need_refresh:
        # ── Try Polygon first ────────────────────────────────────────────
        polygon_ok = False
        try:
            from src.polygon_feed import fetch_daily_bars, has_polygon_key
            if has_polygon_key():
                days = history_years * 365
                logger.info("Fetching %s via Polygon.io (%d days)", ticker, days)
                poly_df = fetch_daily_bars(ticker, days=days)
                if poly_df is not None and not poly_df.empty:
                    poly_df = poly_df.rename_axis("Date").reset_index()
                    poly_df.insert(1, "Ticker", ticker.upper())
                    poly_df.to_csv(path, index=False)
                    polygon_ok = True
                    logger.info("Polygon: saved %d rows for %s", len(poly_df), ticker)
        except Exception as exc:
            logger.warning("Polygon fetch failed for %s: %s — falling back to Yahoo", ticker, exc)

        # ── Fall back to Yahoo Finance ───────────────────────────────────
        if not polygon_ok:
            end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
            start = (end - pd.DateOffset(years=history_years)).normalize()
            logger.info("Fetching %s via Yahoo Finance %s -> %s", ticker, start.date(), end.date())
            df = yf.download(
                tickers=ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            if df is None or df.empty:
                raise RuntimeError(f"No data returned from Yahoo for {ticker}.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename_axis("Date").reset_index()
            df.insert(1, "Ticker", ticker.upper())
            df.to_csv(path, index=False)

    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


def _model_path(ticker: str, model_dir: Path) -> Path:
    return model_dir / f"{_safe_ticker_filename(ticker)}_logreg.joblib"


def load_or_train_model(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Path = DEFAULT_MODEL_DIR,
    retrain: bool = False,
    max_age_days: int = 7,
):
    """Load cached model unless missing/stale or `retrain=True`."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = _model_path(ticker, model_dir)

    if path.exists() and not retrain:
        age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days < max_age_days:
            return joblib.load(path)

    logger.info("Training fresh model for %s on %d rows", ticker, len(X))
    model = fit_full(X, y, make_logreg)
    joblib.dump(model, path)
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ScanRow:
    ticker: str
    as_of: pd.Timestamp
    p_vol: float
    base_rate: float
    lift: float
    last_close: float
    prev_close: float
    pct_change: float
    rsi14: float
    bb_pos: float
    lag1_range: float
    range_compression: float
    abs_gap_pct: float
    realized_vol_5d: float
    realized_vol_20d: float

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.strftime("%Y-%m-%d"),
            "p_vol": self.p_vol,
            "base_rate": self.base_rate,
            "lift": self.lift,
            "last_close": self.last_close,
            "prev_close": self.prev_close,
            "pct_change": self.pct_change,
            "rsi14": self.rsi14,
            "bb_pos": self.bb_pos,
            "lag1_range": self.lag1_range,
            "range_compression": self.range_compression,
            "abs_gap_pct": self.abs_gap_pct,
            "realized_vol_5d": self.realized_vol_5d,
            "realized_vol_20d": self.realized_vol_20d,
        }


def score_one(
    ticker: str,
    refresh_data: bool = True,
    retrain: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
) -> ScanRow:
    daily = fetch_or_load_daily(ticker, data_dir=data_dir, refresh=refresh_data, data_fresh_hours=data_fresh_hours)
    if len(daily) < MIN_TRAIN_ROWS + 50:
        raise RuntimeError(f"{ticker}: only {len(daily)} daily rows (need >{MIN_TRAIN_ROWS}).")

    feats = build_features(daily)
    X, y, _threshold = prepare_xy(feats)
    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{ticker}: only {len(X)} feature rows after dropping NaNs.")

    model = load_or_train_model(ticker, X, y, model_dir=model_dir, retrain=retrain)

    last_X = X.tail(1)
    p_vol = float(score_dataframe(model, last_X).iloc[0])
    base_rate = float(y.mean())
    last_idx = last_X.index[0]

    last_close = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float("nan")
    pct_change = (last_close / prev_close - 1.0) if prev_close == prev_close else float("nan")

    feat_row = feats.loc[last_idx]
    return ScanRow(
        ticker=ticker.upper(),
        as_of=last_idx,
        p_vol=p_vol,
        base_rate=base_rate,
        lift=p_vol / base_rate if base_rate > 0 else float("nan"),
        last_close=last_close,
        prev_close=prev_close,
        pct_change=float(pct_change),
        rsi14=float(feat_row.get("rsi14", float("nan"))),
        bb_pos=float(feat_row.get("bb_pos", float("nan"))),
        lag1_range=float(feat_row.get("lag1_range", float("nan"))),
        range_compression=float(feat_row.get("range_compression_ratio", float("nan"))),
        abs_gap_pct=float(feat_row.get("abs_gap_pct", float("nan"))),
        realized_vol_5d=float(feat_row.get("realized_vol_5d", float("nan"))),
        realized_vol_20d=float(feat_row.get("realized_vol_20d", float("nan"))),
    )


def scan_universe(
    tickers: Iterable[str] = DEFAULT_UNIVERSE,
    refresh_data: bool = True,
    retrain: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_fresh_hours: float = DATA_FRESH_HOURS_DEFAULT,
) -> tuple[pd.DataFrame, list[dict]]:
    """Score every ticker; returns (ranked DataFrame, list of {ticker, error})."""
    rows: list[dict] = []
    errors: list[dict] = []
    for t in tickers:
        try:
            rows.append(
                score_one(
                    t,
                    refresh_data=refresh_data,
                    retrain=retrain,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    data_fresh_hours=data_fresh_hours,
                ).to_dict()
            )
        except Exception as exc:  # noqa: BLE001 — we want to keep scanning on per-ticker errors
            logger.warning("Skipping %s: %s", t, exc)
            errors.append({"ticker": t.upper(), "error": str(exc)})

    if not rows:
        return pd.DataFrame(), errors

    df = pd.DataFrame(rows).sort_values("p_vol", ascending=False).reset_index(drop=True)
    return df, errors


# ---------------------------------------------------------------------------
# Verdict / signal helper
# ---------------------------------------------------------------------------


# Tier name + hex color + 1-line trader-friendly explanation. Tiers use lift
# (p_vol / base_rate) so they're comparable across tickers with different base
# rates (e.g. TSLA's "volatile day" bar is wider than SPY's).
VERDICT_TIERS: list[tuple[float, str, str, str]] = [
    (3.0, "EXTREME",  "#b00020", "Top ~3% of historical setups. Expect outsized intraday range."),
    (2.0, "HIGH",     "#d35400", "Top ~10% setup. Premium straddles likely justified."),
    (1.3, "ELEVATED", "#f39c12", "Above-average vol risk. Tighten stops, widen targets."),
    (0.7, "AVERAGE",  "#7f8c8d", "Typical day. Trade your normal book."),
    (0.3, "CALM",     "#3498db", "Below-average vol. Premium-selling environment."),
    (0.0, "VERY CALM","#85c1e9", "Bottom decile vol setup. Expect tight, drifty tape."),
]


def verdict_for(p_vol: float, base_rate: float) -> tuple[str, str, str]:
    """Return (label, hex_color, explanation) for the given score & base rate."""
    if base_rate <= 0 or p_vol != p_vol:  # NaN check
        return ("UNKNOWN", "#bdc3c7", "Insufficient history.")
    lift = p_vol / base_rate
    for cutoff, label, color, blurb in VERDICT_TIERS:
        if lift >= cutoff:
            return (label, color, blurb)
    return VERDICT_TIERS[-1][1:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-ticker volatility scanner.")
    p.add_argument(
        "--tickers",
        default=",".join(DEFAULT_UNIVERSE),
        help="Comma-separated symbols to scan.",
    )
    p.add_argument("--no-fetch", action="store_true", help="Use cached data only; skip Yahoo.")
    p.add_argument("--retrain", action="store_true", help="Retrain models from scratch.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    df, errors = scan_universe(
        tickers=tickers,
        refresh_data=not args.no_fetch,
        retrain=args.retrain,
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
    )

    if df.empty:
        logger.error("No tickers scored successfully.")
        for err in errors:
            logger.error("  %s: %s", err["ticker"], err["error"])
        return 1

    print()
    print(f"{'Rank':<5}{'Ticker':<8}{'P(vol)':<10}{'Lift':<8}{'Verdict':<12}{'Last':<10}{'%Δ':<10}{'RSI':<6}")
    print("-" * 75)
    for i, r in df.iterrows():
        label, _color, _blurb = verdict_for(r["p_vol"], r["base_rate"])
        print(
            f"#{i + 1:<4}"
            f"{r['ticker']:<8}"
            f"{r['p_vol'] * 100:>6.1f}%   "
            f"{r['lift']:>5.2f}x  "
            f"{label:<12}"
            f"${r['last_close']:>7.2f}  "
            f"{r['pct_change'] * 100:>+6.2f}%   "
            f"{r['rsi14']:>4.0f}"
        )
    if errors:
        print()
        print("Skipped:")
        for err in errors:
            print(f"  {err['ticker']}: {err['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
