"""Load downloaded OHLCV data back into a clean pandas DataFrame.

This is the read-side companion to ``src.download_spy``. It auto-detects the
file format (CSV or Parquet), parses the timestamp column, and returns a frame
indexed by date so downstream code (backtests, plots, notebooks) can stay
short and readable.

Typical use:

    from src.loader import load_history

    df = load_history("SPY")              # auto-finds data/SPY_1d.csv|parquet
    df = load_history("SPY", interval="1h")
    df = load_history(path="data/SPY_1d.parquet")

The returned DataFrame:
    - is indexed by ``Date`` (DatetimeIndex, sorted ascending, no duplicates)
    - has float dtypes for price columns and int64 for ``Volume``
    - keeps both ``Close`` and ``Adj Close`` so callers pick the right one
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")


def _candidate_paths(ticker: str, interval: str, data_dir: Path) -> list[Path]:
    stem = f"{ticker.upper()}_{interval}"
    return [data_dir / f"{stem}.parquet", data_dir / f"{stem}.csv"]


def _resolve_path(
    ticker: str | None,
    interval: str,
    path: str | Path | None,
    data_dir: Path,
) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        return p

    if ticker is None:
        raise ValueError("Provide either `ticker` or `path`.")

    for candidate in _candidate_paths(ticker, interval, data_dir):
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in _candidate_paths(ticker, interval, data_dir))
    raise FileNotFoundError(
        f"No data file found for {ticker} ({interval}). Tried: {tried}. "
        f"Run `python -m src.download_spy --ticker {ticker} --interval {interval}` first."
    )


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError(f"Expected a 'Date' column, got {list(df.columns)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=False, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in PRICE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")

    return df


def load_history(
    ticker: str | None = None,
    interval: str = "1d",
    path: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load OHLCV data for ``ticker`` (or an explicit ``path``).

    Parameters
    ----------
    ticker:
        Symbol to load, e.g. ``"SPY"``. Ignored if ``path`` is given.
    interval:
        Bar interval used when locating the file by ticker. Default ``"1d"``.
    path:
        Explicit path to a CSV or Parquet file. Overrides ticker-based lookup.
    data_dir:
        Directory to search for ticker files. Defaults to ``<repo>/data``.
    columns:
        Optional subset of columns to return.
    """
    resolved = _resolve_path(ticker, interval, path, Path(data_dir))
    df = _coerce(_read(resolved))

    if columns is not None:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

    return df


def returns(df: pd.DataFrame, column: str = "Adj Close", log: bool = False) -> pd.Series:
    """Daily simple or log returns from the chosen price column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame.")
    px = df[column].astype(float)
    if log:
        import numpy as np

        return np.log(px / px.shift(1)).dropna().rename(f"{column} log return")
    return px.pct_change().dropna().rename(f"{column} return")
