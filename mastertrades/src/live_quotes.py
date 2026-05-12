"""No-signup live(ish) quote fetcher for the Command Center.

Both Yahoo Finance and Stooq publish free, no-authentication endpoints
that update through the trading day. The official SEC delay for free
unauthenticated US-equity data is 15 minutes, so this module is honestly
labelled as "delayed" rather than "real-time".

Sources (in order of preference):

  1. Yahoo Finance via ``yfinance`` -- minute bars + current quote
  2. Stooq.com CSV -- simple HTTP, very reliable, ~15-min delayed

Both return ``LiveQuote`` dataclasses with a consistent shape::

    LiveQuote(
        ticker="SPY",
        last=739.50,
        prev_close=738.10,
        day_open=738.42,
        day_high=740.55,
        day_low=737.20,
        day_volume=73_450_000,
        as_of=datetime(2026, 5, 11, 15, 59),  # last bar timestamp
        source="yahoo",
        delayed_minutes=15,
    )

Used by the Command Center to:

  - Show a top-bar "live prices" strip (auto-refresh)
  - Display today's intraday high/low/close inside each ticker card
  - Replace the (stale) daily ``last_close`` with the most recent quote

Honest caveats
--------------

* This is **delayed data, not real-time**. Treat it like a glance, not a fill price.
* During market hours the data is typically 10-20 minutes behind the tape.
* Outside US market hours both sources return the last close of the prior session.
* If both sources fail, we return ``None`` for that ticker -- the dashboard handles
  gracefully and shows the daily-bar close instead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger("live_quotes")


@dataclass
class LiveQuote:
    ticker: str
    last: float
    prev_close: float
    day_open: float
    day_high: float
    day_low: float
    day_volume: float
    as_of: datetime
    source: str
    delayed_minutes: int = 15

    def change(self) -> float:
        return self.last - self.prev_close

    def change_pct(self) -> float:
        if self.prev_close <= 0:
            return 0.0
        return (self.last - self.prev_close) / self.prev_close

    def day_range_pct(self) -> float:
        if self.day_open <= 0:
            return 0.0
        return (self.day_high - self.day_low) / self.day_open

    def as_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


# ---------------------------------------------------------------------------
# Yahoo Finance (via yfinance)
# ---------------------------------------------------------------------------


def fetch_yahoo_live(ticker: str) -> Optional[LiveQuote]:
    """Fetch a live(ish) quote from Yahoo Finance via yfinance.

    Pulls 2 days of 1-minute bars so we can recover yesterday's close
    (for the change % calculation) and today's intraday OHLCV.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed -- skipping Yahoo live quote")
        return None

    try:
        t = yf.Ticker(ticker)
        # period="2d" interval="1m" returns ~2 trading days of minute bars
        df = t.history(period="2d", interval="1m", prepost=False, auto_adjust=False)
        if df is None or df.empty:
            return None

        # Drop timezone for consistent comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

        # Split into per-day groups; latest day = today's intraday
        df["_date"] = df.index.normalize()
        days = list(df.groupby("_date"))
        if not days:
            return None

        today_date, today_df = days[-1]
        if len(days) >= 2:
            _, prev_df = days[-2]
            prev_close = float(prev_df["Close"].iloc[-1])
        else:
            prev_close = float(today_df["Close"].iloc[0])

        last_bar = today_df.iloc[-1]
        return LiveQuote(
            ticker=ticker.upper(),
            last=float(last_bar["Close"]),
            prev_close=prev_close,
            day_open=float(today_df["Open"].iloc[0]),
            day_high=float(today_df["High"].max()),
            day_low=float(today_df["Low"].min()),
            day_volume=float(today_df["Volume"].sum()),
            as_of=today_df.index[-1].to_pydatetime(),
            source="yahoo",
            delayed_minutes=15,
        )
    except Exception as e:
        logger.warning("Yahoo live fetch failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Stooq (CSV, no auth)
# ---------------------------------------------------------------------------


_STOOQ_URL = "https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"


def fetch_stooq_live(ticker: str) -> Optional[LiveQuote]:
    """Fetch a live(ish) quote from Stooq.com.

    Stooq accepts ``<ticker>.us`` for US equities. The CSV has columns:
    Symbol, Date, Time, Open, High, Low, Close, Volume.

    Stooq does not include the previous close in a single call, so we
    estimate change-% as (last - open)/open until a daily history pull
    refreshes it. Good enough for a glance.
    """
    sym = f"{ticker.lower()}.us"
    url = _STOOQ_URL.format(symbol=sym)
    try:
        df = pd.read_csv(url)
        if df.empty:
            return None
        row = df.iloc[0]
        if pd.isna(row["Close"]) or row["Close"] in ("N/D", "-"):
            return None

        dt_str = f"{row['Date']} {row['Time']}"
        try:
            as_of = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            as_of = datetime.now()

        last = float(row["Close"])
        open_ = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        vol = float(row["Volume"]) if not pd.isna(row["Volume"]) else 0.0

        return LiveQuote(
            ticker=ticker.upper(),
            last=last,
            prev_close=open_,  # Stooq doesn't ship prev-close in /q/l/; use open as a fallback
            day_open=open_,
            day_high=high,
            day_low=low,
            day_volume=vol,
            as_of=as_of,
            source="stooq",
            delayed_minutes=15,
        )
    except Exception as e:
        logger.warning("Stooq live fetch failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Combined fetcher (parallel, with fallback)
# ---------------------------------------------------------------------------


def fetch_quote(ticker: str, prefer: str = "yahoo") -> Optional[LiveQuote]:
    """Fetch a quote for a single ticker, trying preferred source first.

    ``prefer`` may be ``"yahoo"`` or ``"stooq"``. If the preferred source
    fails or returns nothing usable, the other source is tried as a fallback.
    """
    if prefer == "yahoo":
        q = fetch_yahoo_live(ticker)
        if q is None:
            q = fetch_stooq_live(ticker)
    else:
        q = fetch_stooq_live(ticker)
        if q is None:
            q = fetch_yahoo_live(ticker)
    return q


def get_live_quotes(tickers: list[str], prefer: str = "yahoo",
                     timeout_sec: float = 8.0) -> dict[str, LiveQuote]:
    """Fetch quotes for many tickers in parallel.

    Returns ``{ticker: LiveQuote}``. Missing tickers are simply absent
    from the dict; callers should fall back to daily-bar data.

    Threads are joined with a global timeout so a single hung HTTP call
    cannot stall the dashboard refresh.
    """
    results: dict[str, LiveQuote] = {}
    lock = threading.Lock()

    def worker(t: str) -> None:
        q = fetch_quote(t, prefer=prefer)
        if q is not None:
            with lock:
                results[t.upper()] = q

    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in tickers]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=timeout_sec)

    return results


# ---------------------------------------------------------------------------
# CLI: quick check
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Print live (delayed) quotes for one or more tickers.")
    parser.add_argument("tickers", nargs="*", default=["SPY", "QQQ", "IWM", "AAPL"])
    parser.add_argument("--source", choices=["yahoo", "stooq", "both"], default="yahoo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    if args.source == "both":
        for ticker in args.tickers:
            print(f"\n=== {ticker} ===")
            for src in ("yahoo", "stooq"):
                q = fetch_quote(ticker, prefer=src)
                if q is None:
                    print(f"  {src:6s}  (no data)")
                else:
                    print(f"  {src:6s}  last=${q.last:.2f}  open=${q.day_open:.2f}  "
                          f"high=${q.day_high:.2f}  low=${q.day_low:.2f}  "
                          f"as_of={q.as_of:%Y-%m-%d %H:%M}  src={q.source}")
        return 0

    quotes = get_live_quotes(args.tickers, prefer=args.source)
    print(f"{'Ticker':<7} {'Last':>9} {'Open':>9} {'High':>9} {'Low':>9} "
          f"{'Δ%':>7} {'Volume':>14} {'As of':<19} {'Source':<7}")
    print("-" * 100)
    for t in args.tickers:
        q = quotes.get(t.upper())
        if q is None:
            print(f"{t:<7} {'—':>9} (no data)")
            continue
        print(f"{q.ticker:<7} ${q.last:>7.2f}  ${q.day_open:>7.2f}  ${q.day_high:>7.2f}  "
              f"${q.day_low:>7.2f}  {q.change_pct()*100:>+6.2f}%  {q.day_volume:>14,.0f}  "
              f"{q.as_of:%Y-%m-%d %H:%M:%S}  {q.source:<7}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
