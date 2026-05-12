"""Download historical OHLCV data for SPY (or any ticker) from Yahoo Finance.

Usage examples:
    python -m src.download_spy
    python -m src.download_spy --ticker SPY --start 1993-01-29
    python -m src.download_spy --ticker QQQ --interval 1d --format parquet
    python -m src.download_spy --interval 1h --start 2024-01-01

Notes on adjusted vs unadjusted prices:
    Yahoo's "Close" is the raw exchange close. "Adj Close" is back-adjusted for
    splits and dividends. For backtests of total-return strategies, use Adj Close.
    For strategies that need true historical traded prices, use Close. We keep
    BOTH columns in the output and leave the choice to the consumer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf


# SPY's first trading day on AMEX.
SPY_INCEPTION = "1993-01-29"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ValidInterval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
]

# Yahoo enforces history limits on intraday intervals.
INTRADAY_MAX_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
}


logger = logging.getLogger("download_spy")


def fetch_history(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: ValidInterval,
) -> pd.DataFrame:
    """Pull OHLCV history from Yahoo Finance and return a tidy DataFrame."""
    logger.info(
        "Fetching %s @ %s from %s to %s",
        ticker,
        interval,
        start or "inception",
        end or "today",
    )

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # keep raw Close AND Adj Close
        actions=True,        # include Dividends + Stock Splits
        progress=False,
        threads=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} ({interval}, {start} -> {end}). "
            "Check the ticker, your date range, or your network connection."
        )

    # yfinance sometimes returns a MultiIndex column even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Date").reset_index()
    df.insert(1, "Ticker", ticker.upper())

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns: %s", missing)

    return df


def save(df: pd.DataFrame, path: Path, fmt: Literal["csv", "parquet"]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return path


def _validate_intraday_range(interval: str, start: str | None) -> None:
    if interval not in INTRADAY_MAX_LOOKBACK_DAYS or start is None:
        return
    max_days = INTRADAY_MAX_LOOKBACK_DAYS[interval]
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - start_dt).days
    if age_days > max_days:
        logger.warning(
            "Yahoo only serves ~%d days of %s data; requested start (%s) is %d "
            "days ago. Older bars will be silently dropped.",
            max_days,
            interval,
            start,
            age_days,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download historical OHLCV from Yahoo Finance.")
    p.add_argument("--ticker", default="SPY", help="Symbol to download (default: SPY).")
    p.add_argument(
        "--start",
        default=None,
        help="ISO start date, e.g. 2000-01-01. Defaults to ticker inception for SPY, else 20 years.",
    )
    p.add_argument("--end", default=None, help="ISO end date (exclusive). Defaults to today.")
    p.add_argument(
        "--interval",
        default="1d",
        choices=list(INTRADAY_MAX_LOOKBACK_DAYS.keys()) + ["1d", "5d", "1wk", "1mo", "3mo"],
        help="Bar interval (default: 1d).",
    )
    p.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet"],
        help="Output format (default: csv).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output file path. Defaults to data/<TICKER>_<INTERVAL>.<ext>.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = args.start
    if start is None:
        start = SPY_INCEPTION if args.ticker.upper() == "SPY" else None

    _validate_intraday_range(args.interval, start)

    df = fetch_history(args.ticker, start, args.end, args.interval)

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_DATA_DIR / f"{args.ticker.upper()}_{args.interval}.{args.format}"
    )
    save(df, out_path, args.format)

    first = df["Date"].iloc[0]
    last = df["Date"].iloc[-1]
    logger.info(
        "Saved %d rows to %s (range: %s -> %s)",
        len(df),
        out_path,
        first,
        last,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
