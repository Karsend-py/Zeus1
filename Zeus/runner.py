"""
runner.py
---------
BacktestRunner is the orchestrator.  It owns the main bar-by-bar loop and
delegates every decision to the entry and exit engines.

State machine per iteration
---------------------------
    ┌──────────┐   entry fires   ┌──────────┐
    │  IDLE    │ ──────────────> │  OPEN    │
    └──────────┘                 └────┬─────┘
                                      │ breach / expiry
                                      ▼
                                 ┌──────────┐
                                 │  CLOSED  │  → append to results, back to IDLE
                                 └──────────┘
"""

from __future__ import annotations

import pandas as pd

from engine import AnalyticsEngine
from entry_engine import TradeEntryEngine
from exit_engine import TradeExitEngine
from models import (
    BacktestResult,
    RejectedTrade,
    StrategyParams,
    Trade,
    TradeResult,
)


class BacktestRunner:
    """Top-level runner.  Construct once, call ``run`` once.

    Parameters
    ----------
    df : DataFrame
        Price data with all indicator columns already attached.
    params : StrategyParams
        Frozen strategy configuration.
    blackout_dates : set[date]
        Expanded blackout set (buffer already applied by BlackoutFilter).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        blackout_dates: set,
    ) -> None:
        self.df = df
        self.params = params
        self._entry = TradeEntryEngine(params, blackout_dates)
        self._exit = TradeExitEngine(params)

    # ---------------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the full backtest and return a populated BacktestResult."""
        trades: list[Trade] = []
        rejected: list[RejectedTrade] = []
        open_trade: Trade | None = None

        for timestamp, row in self.df.iterrows():
            # ------------------------------------------------------------------
            # If a trade is open, check for exit *first*
            # ------------------------------------------------------------------
            if open_trade is not None:
                closed = self._exit.resolve(open_trade, row, timestamp)
                if closed is not None:
                    trades.append(closed)
                    open_trade = None
                # Whether or not we closed, skip entry evaluation this bar
                # (only one position at a time)
                continue

            # ------------------------------------------------------------------
            # No open trade → evaluate entry
            # ------------------------------------------------------------------
            result = self._entry.evaluate_bar(row, timestamp)

            if result is None:
                # Bar was completely irrelevant (shouldn't normally happen
                # after indicator warmup, but guard it)
                continue

            if isinstance(result, Trade):
                open_trade = result
            elif isinstance(result, RejectedTrade):
                rejected.append(result)

        # ----------------------------------------------------------------------
        # If a trade is still open at end-of-data, close it at expiry (WIN)
        # ----------------------------------------------------------------------
        if open_trade is not None:
            from dataclasses import replace
            from models import ExitReason

            last_ts = self.df.index[-1]
            closed = replace(
                open_trade,
                result=TradeResult.WIN,
                exit_reason=ExitReason.EXPIRY,
                exit_timestamp=last_ts,
                loss_realised=0.0,
                pnl=open_trade.credit_received,
            )
            trades.append(closed)

        # ----------------------------------------------------------------------
        # Build result object and run analytics
        # ----------------------------------------------------------------------
        result_obj = BacktestResult(trades=trades, rejected_trades=rejected)
        result_obj = AnalyticsEngine.summarise(result_obj)
        return result_obj
