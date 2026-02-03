"""
exit_engine.py
--------------
TradeExitEngine monitors open positions on every subsequent bar after entry.
It closes a trade the *instant* (i.e. on the first bar where) either short
strike is breached.

Breach definition:
    • Upper breach  →  bar High  > upper_strike
    • Lower breach  →  bar Low   < lower_strike

If neither breach fires before the expiry bar, the trade closes at expiry as
a WIN (full credit kept).

Design notes
------------
* The engine holds a *single* open trade at a time.  The runner calls
  ``check_exit`` on every bar after the entry bar.
* Frozen dataclasses mean we cannot mutate a Trade in place.  Instead we
  return a *new* Trade with the resolved fields.  The runner replaces the
  reference.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd

from models import (
    ExitReason,
    StrategyParams,
    Trade,
    TradeResult,
)


class TradeExitEngine:
    """Stateless exit evaluator.  Call ``resolve`` once per bar per open trade."""

    def __init__(self, params: StrategyParams) -> None:
        self.params = params

    # ---------------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------------

    def resolve(
        self,
        trade: Trade,
        row: pd.Series,
        timestamp: datetime,
    ) -> Trade | None:
        """Check a single bar against an open trade.

        Returns
        -------
        Trade | None
            A *new* (closed) Trade if the bar triggered an exit, else None.
        """
        high = float(row["High"])
        low = float(row["Low"])

        # ------------------------------------------------------------------
        # Upper strike breach  (short call side)
        # ------------------------------------------------------------------
        if high > trade.upper_strike:
            return self._close(trade, timestamp, ExitReason.BREACH_SHORT_CALL)

        # ------------------------------------------------------------------
        # Lower strike breach  (short put side)
        # ------------------------------------------------------------------
        if low < trade.lower_strike:
            return self._close(trade, timestamp, ExitReason.BREACH_SHORT_PUT)

        # ------------------------------------------------------------------
        # Expiry check  (bar date >= expiry date)
        # ------------------------------------------------------------------
        if timestamp.date() >= trade.expiry_date.date():
            return self._close(trade, timestamp, ExitReason.EXPIRY_WORTHLESS)

        # No exit this bar
        return None

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _close(
        self,
        trade: Trade,
        exit_ts: datetime,
        reason: ExitReason,
    ) -> Trade:
        """Create a closed copy of *trade* with P&L resolved."""
        is_loss = reason in (ExitReason.BREACH_SHORT_CALL, ExitReason.BREACH_SHORT_PUT)

        loss_realised = self.params.max_loss if is_loss else 0.0
        pnl = trade.credit_received - loss_realised

        return replace(
            trade,
            result=TradeResult.LOSS if is_loss else TradeResult.WIN,
            exit_reason=reason,
            exit_timestamp=exit_ts,
            loss_realised=loss_realised,
            pnl=pnl,
        )