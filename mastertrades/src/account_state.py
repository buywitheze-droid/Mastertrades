"""Tiny persistent account state for the Command Center dashboard.

We keep a single JSON file at ``data/account_state.json`` with the user's
starting equity, current equity, milestone targets, and an optional trade
log. The Command Center reads (and optionally updates) this file so the
dashboard can show real progress vs. the moonshot goals.

Schema::

    {
      "starting_equity": 500.0,
      "current_equity": 500.0,
      "milestones": [5000, 50000, 500000],
      "history": [
        {"date": "2026-05-11", "equity": 500.0}
      ],
      "trades": [
        {
          "date": "2026-05-11",
          "ticker": "SPY",
          "tier": "GO_ULTRA_JACKPOT",
          "risk": 75.0,
          "pnl": 38.5,
          "note": "0DTE 740 straddle"
        }
      ]
    }

If the file is missing or unreadable the loader returns a sensible default
so the dashboard still renders. Writes are atomic (write to .tmp then
replace).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("account_state")


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "account_state.json"
DEFAULT_STARTING_EQUITY = 500.0
DEFAULT_MILESTONES = (5_000.0, 50_000.0, 500_000.0)


@dataclass
class TradeEntry:
    date: str
    ticker: str
    tier: str
    risk: float
    pnl: float
    note: str = ""


@dataclass
class AccountState:
    starting_equity: float = DEFAULT_STARTING_EQUITY
    current_equity: float = DEFAULT_STARTING_EQUITY
    milestones: list[float] = field(default_factory=lambda: list(DEFAULT_MILESTONES))
    history: list[dict] = field(default_factory=list)
    trades: list[TradeEntry] = field(default_factory=list)

    # -- derived helpers -----------------------------------------------------

    def total_pnl(self) -> float:
        return self.current_equity - self.starting_equity

    def total_pnl_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return self.total_pnl() / self.starting_equity

    def trade_count(self) -> int:
        return len(self.trades)

    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    def loss_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl < 0)

    def win_rate(self) -> float:
        n = self.trade_count()
        return self.win_count() / n if n > 0 else 0.0

    def next_milestone(self) -> float | None:
        """Smallest milestone strictly above current equity."""
        upcoming = [m for m in self.milestones if m > self.current_equity]
        return min(upcoming) if upcoming else None

    def previous_milestone(self) -> float:
        """Largest milestone at or below current equity, else starting."""
        below = [m for m in self.milestones if m <= self.current_equity]
        return max(below) if below else self.starting_equity

    def progress_to_next(self) -> float:
        """Progress (0..1) from previous milestone to next milestone."""
        nxt = self.next_milestone()
        if nxt is None:
            return 1.0
        prev = self.previous_milestone()
        if nxt <= prev:
            return 1.0
        return max(0.0, min(1.0, (self.current_equity - prev) / (nxt - prev)))

    def equity_curve(self, max_points: int = 60) -> list[tuple[str, float]]:
        """Return the last ``max_points`` (date, equity) pairs."""
        if not self.history:
            return [(datetime.now().strftime("%Y-%m-%d"), self.current_equity)]
        return [(h["date"], float(h["equity"])) for h in self.history[-max_points:]]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_state(path: Path | str = DEFAULT_PATH) -> AccountState:
    """Load account state from disk. Returns a default state on any failure."""
    p = Path(path)
    if not p.exists():
        logger.info("No account state at %s — using defaults.", p)
        return AccountState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read %s: %s — using defaults.", p, e)
        return AccountState()

    trades = [TradeEntry(**t) for t in raw.get("trades", [])]
    state = AccountState(
        starting_equity=float(raw.get("starting_equity", DEFAULT_STARTING_EQUITY)),
        current_equity=float(raw.get("current_equity", DEFAULT_STARTING_EQUITY)),
        milestones=[float(m) for m in raw.get("milestones", DEFAULT_MILESTONES)],
        history=list(raw.get("history", [])),
        trades=trades,
    )
    return state


def save_state(state: AccountState, path: Path | str = DEFAULT_PATH) -> None:
    """Atomic write of account state to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "starting_equity": state.starting_equity,
        "current_equity": state.current_equity,
        "milestones": state.milestones,
        "history": state.history,
        "trades": [asdict(t) for t in state.trades],
    }
    fd, tmp = tempfile.mkstemp(prefix="account_state.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def snapshot_equity(state: AccountState, equity: float, when: datetime | None = None) -> AccountState:
    """Append a daily equity snapshot, deduplicated by date."""
    when = when or datetime.now()
    date_str = when.strftime("%Y-%m-%d")
    new_hist = [h for h in state.history if h.get("date") != date_str]
    new_hist.append({"date": date_str, "equity": float(equity)})
    state.history = new_hist
    state.current_equity = float(equity)
    return state


def log_trade(state: AccountState, trade: TradeEntry, update_equity: bool = True) -> AccountState:
    """Append a trade and (optionally) roll equity forward by its P&L."""
    state.trades.append(trade)
    if update_equity:
        state.current_equity = float(state.current_equity + trade.pnl)
        snapshot_equity(state, state.current_equity, when=datetime.strptime(trade.date, "%Y-%m-%d"))
    return state
