"""
test_exit_engine.py
-------------------
Unit tests for TradeExitEngine.resolve().

Each test provides an open Trade and a single bar row, then asserts whether
the engine closes the trade (and with what reason) or holds.
"""

from datetime import datetime

import pandas as pd
import pytz

import pytest

from src.strategy.exit_engine import TradeExitEngine
from src.strategy.models import (
    ExitReason,
    StrategyParams,
    Trade,
    TradeResult,
)

ET = pytz.timezone("America/New_York")


def _params(credit=0.50, max_loss=2.50) -> StrategyParams:
    return StrategyParams(credit_received=credit, max_loss=max_loss)


def _open_trade(
    upper_strike=5050.0,
    lower_strike=4950.0,
    credit=0.50,
    expiry=None,
) -> Trade:
    return Trade(
        trade_id=1,
        entry_timestamp=ET.localize(datetime(2024, 1, 10, 10, 0)),
        expiry_date=expiry or ET.localize(datetime(2024, 1, 12, 16, 0)),  # Friday
        upper_strike=upper_strike,
        lower_strike=lower_strike,
        credit_received=credit,
        result=TradeResult.OPEN,
        exit_reason=ExitReason.EXPIRY,  # placeholder
    )


def _bar(high=5030.0, low=4970.0) -> pd.Series:
    return pd.Series({"High": high, "Low": low, "Close": 5000.0})


class TestExitBreach:
    """Strike-breach scenarios."""

    def test_upper_breach(self):
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade(upper_strike=5050.0)
        # High crosses above upper strike
        row = _bar(high=5060.0, low=4970.0)
        ts = ET.localize(datetime(2024, 1, 11, 11, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is not None
        assert closed.result == TradeResult.LOSS
        assert closed.exit_reason == ExitReason.UPPER_BREACH
        assert closed.loss_realised == 2.50
        assert closed.pnl == pytest.approx(0.50 - 2.50)

    def test_lower_breach(self):
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade(lower_strike=4950.0)
        # Low crosses below lower strike
        row = _bar(high=5030.0, low=4940.0)
        ts = ET.localize(datetime(2024, 1, 11, 11, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is not None
        assert closed.result == TradeResult.LOSS
        assert closed.exit_reason == ExitReason.LOWER_BREACH
        assert closed.pnl == pytest.approx(0.50 - 2.50)

    def test_both_breached_upper_wins(self):
        """If both are breached in the same bar, upper is checked first."""
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade(upper_strike=5050.0, lower_strike=4950.0)
        # Both breached
        row = _bar(high=5100.0, low=4900.0)
        ts = ET.localize(datetime(2024, 1, 11, 11, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is not None
        assert closed.exit_reason == ExitReason.UPPER_BREACH


class TestExitExpiry:
    """Expiry close scenarios."""

    def test_expiry_on_friday(self):
        params = _params()
        engine = TradeExitEngine(params)
        # Expiry is Jan 12
        trade = _open_trade()
        row = _bar(high=5030.0, low=4970.0)  # No breach
        # Bar is on Jan 12 (expiry date)
        ts = ET.localize(datetime(2024, 1, 12, 15, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is not None
        assert closed.result == TradeResult.WIN
        assert closed.exit_reason == ExitReason.EXPIRY
        assert closed.pnl == pytest.approx(0.50)
        assert closed.loss_realised == 0.0


class TestExitHold:
    """Bars that should NOT trigger an exit."""

    def test_no_breach_before_expiry(self):
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade()
        # Stays within strikes, before expiry
        row = _bar(high=5040.0, low=4960.0)
        ts = ET.localize(datetime(2024, 1, 11, 10, 0))  # Thursday, before expiry

        closed = engine.resolve(trade, row, ts)
        assert closed is None  # Still open

    def test_high_exactly_at_strike_does_not_breach(self):
        """Breach requires *strictly greater than* the strike."""
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade(upper_strike=5050.0)
        row = _bar(high=5050.0, low=4970.0)  # High == strike, not >
        ts = ET.localize(datetime(2024, 1, 11, 10, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is None

    def test_low_exactly_at_strike_does_not_breach(self):
        """Breach requires *strictly less than* the strike."""
        params = _params()
        engine = TradeExitEngine(params)
        trade = _open_trade(lower_strike=4950.0)
        row = _bar(high=5030.0, low=4950.0)  # Low == strike, not <
        ts = ET.localize(datetime(2024, 1, 11, 10, 0))

        closed = engine.resolve(trade, row, ts)
        assert closed is None
