"""Tiny CLI for updating ``data/account_state.json``.

Examples::

    python -m src.account_state_cli show
    python -m src.account_state_cli set-equity 562.50
    python -m src.account_state_cli set-start 500
    python -m src.account_state_cli log-trade SPY GO_ULTRA_JACKPOT 75 38.5 "0DTE 740 straddle"
    python -m src.account_state_cli undo-last-trade
    python -m src.account_state_cli reset

After every change the Command Center will pick up the new state on its
next auto-refresh (every 5 min) or you can re-open the HTML manually.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from src.account_state import (
    DEFAULT_PATH,
    AccountState,
    TradeEntry,
    load_state,
    log_trade,
    save_state,
    snapshot_equity,
)


def _print(state: AccountState) -> None:
    pnl = state.total_pnl()
    print(f"  Start:       ${state.starting_equity:,.2f}")
    print(f"  Now:         ${state.current_equity:,.2f}")
    print(f"  P&L:         {'+' if pnl>=0 else ''}${pnl:,.2f}  ({state.total_pnl_pct()*100:+.1f}%)")
    nxt = state.next_milestone()
    if nxt:
        print(f"  Next target: ${nxt:,.0f}  ({state.progress_to_next()*100:.1f}% there)")
    else:
        print(f"  Next target: (all milestones passed)")
    print(f"  Trades:      {state.trade_count()}  ({state.win_count()}W / {state.loss_count()}L)")
    if state.trades:
        print()
        print(f"  Last 5 trades:")
        for t in state.trades[-5:]:
            sign = "+" if t.pnl >= 0 else ""
            print(f"    {t.date}  {t.ticker:<6} {t.tier:<19} risk ${t.risk:>5.0f} → pnl {sign}${t.pnl:>6.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="Print current state")
    p_eq = sub.add_parser("set-equity", help="Set current equity")
    p_eq.add_argument("equity", type=float)
    p_st = sub.add_parser("set-start", help="Set starting equity")
    p_st.add_argument("equity", type=float)
    p_lt = sub.add_parser("log-trade", help="Log a trade outcome")
    p_lt.add_argument("ticker")
    p_lt.add_argument("tier")
    p_lt.add_argument("risk", type=float)
    p_lt.add_argument("pnl", type=float)
    p_lt.add_argument("note", nargs="?", default="")
    p_lt.add_argument("--date", default=None, help="YYYY-MM-DD, default today")
    sub.add_parser("undo-last-trade", help="Remove the most recent trade and revert its P&L")
    sub.add_parser("reset", help="Reset to defaults ($500 start, no trades)")

    args = parser.parse_args(argv)

    state = load_state()

    if args.cmd == "show":
        _print(state)
        return 0
    if args.cmd == "set-equity":
        snapshot_equity(state, args.equity)
        save_state(state)
        print(f"Set current equity to ${args.equity:,.2f}")
        _print(state)
        return 0
    if args.cmd == "set-start":
        state.starting_equity = float(args.equity)
        save_state(state)
        print(f"Set starting equity to ${args.equity:,.2f}")
        _print(state)
        return 0
    if args.cmd == "log-trade":
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        trade = TradeEntry(
            date=date, ticker=args.ticker, tier=args.tier,
            risk=float(args.risk), pnl=float(args.pnl), note=args.note,
        )
        log_trade(state, trade, update_equity=True)
        save_state(state)
        sign = "+" if trade.pnl >= 0 else ""
        print(f"Logged: {date} {args.ticker} {args.tier}  risk ${args.risk:.2f}  pnl {sign}${args.pnl:.2f}")
        _print(state)
        return 0
    if args.cmd == "undo-last-trade":
        if not state.trades:
            print("No trades to undo.")
            return 0
        last = state.trades.pop()
        state.current_equity -= float(last.pnl)
        if state.history:
            state.history = state.history[:-1]
        save_state(state)
        print(f"Undid: {last.date} {last.ticker} pnl {last.pnl}")
        _print(state)
        return 0
    if args.cmd == "reset":
        state = AccountState()
        save_state(state)
        print(f"Reset to defaults. Account state stored at: {DEFAULT_PATH}")
        _print(state)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
