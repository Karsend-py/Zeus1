"""
test_analytics.py
-----------------
Unit tests for AnalyticsEngine.summarise() and the internal _max_drawdown
helper.
"""

from datetime import datetime

import pytest

from src.analytics.engine import AnalyticsEngine
from src.strategy.models import (
    BacktestResult,
    ExitReason,
    Trade,
    TradeResult,
)


def _trade(trade_id, pnl, result=TradeResult.WIN, exit_reason=ExitReason.EXPIRY):
    is_loss = result == TradeResult.LOSS
    return Trade(
        trade_id=trade_id,
        entry_timestamp=datetime(2024, 1, 1 + trade_id, 10, 0),
        expiry_date=datetime(2024, 1, 5 + trade_id, 16, 0),
        upper_strike=5050.0,
        lower_strike=4950.0,
        credit_received=0.50,
        result=result,
        exit_reason=exit_reason,
        exit_timestamp=datetime(2024, 1, 5 + trade_id, 16, 0),
        loss_realised=2.50 if is_loss else 0.0,
        pnl=pnl,
    )


class TestSummariseBasic:
    """Core metric computation."""

    def test_all_wins(self):
        trades = [_trade(i, pnl=0.50, result=TradeResult.WIN) for i in range(5)]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)

        assert result.total_trades == 5
        assert result.total_wins == 5
        assert result.total_losses == 0
        assert result.win_rate == 100.0
        assert result.total_pnl == pytest.approx(2.50)
        assert result.max_drawdown == 0.0  # Never drops

    def test_all_losses(self):
        trades = [_trade(i, pnl=-2.0, result=TradeResult.LOSS) for i in range(3)]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)

        assert result.total_losses == 3
        assert result.win_rate == 0.0
        assert abs(result.total_pnl - (-6.0)) < 1e-6
        assert abs(result.max_drawdown - 6.0) < 1e-6

    def test_mixed(self):
        trades = [
            _trade(1, pnl=0.50, result=TradeResult.WIN),
            _trade(2, pnl=-2.0, result=TradeResult.LOSS),
            _trade(3, pnl=0.50, result=TradeResult.WIN),
            _trade(4, pnl=0.50, result=TradeResult.WIN),
        ]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)

        assert result.total_trades == 4
        assert result.total_wins == 3
        assert result.total_losses == 1
        assert result.total_pnl == pytest.approx(-0.5)  # 0.5 - 2.0 + 0.5 + 0.5

    def test_equity_curve_length(self):
        trades = [_trade(i, pnl=0.50) for i in range(4)]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)
        assert len(result.equity_curve) == 4

    def test_equity_curve_values(self):
        trades = [
            _trade(1, pnl=1.0),
            _trade(2, pnl=-0.5),
            _trade(3, pnl=0.3),
        ]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)
        expected = [1.0, 0.5, 0.8]
        assert len(result.equity_curve) == len(expected)
        for actual, exp in zip(result.equity_curve, expected):
            assert abs(actual - exp) < 1e-6, f"Got {actual}, expected {exp}"


class TestMaxDrawdown:
    """Isolated _max_drawdown tests.

    All series implicitly start at 0.  So [1, 2, 3] has peak=3, no dip → dd=0.
    But [-2, -4, -6] has peak=0 (implicit start), trough=-6 → dd=6.
    """

    def test_monotonic_rise(self):
        # 0 → 1 → 2 → 3 → 4  : never dips below any prior peak
        assert AnalyticsEngine._max_drawdown([1.0, 2.0, 3.0, 4.0]) == 0.0

    def test_single_dip(self):
        # 0 → 1 → 3 → 1 → 2  : peak=3 at index 1, trough=1 → dd=2
        assert AnalyticsEngine._max_drawdown([1.0, 3.0, 1.0, 2.0]) == 2.0

    def test_two_dips_picks_largest(self):
        # 0 → 1 → 5 → 3 → 6 → 1
        # First dip:  peak=5, trough=3 → dd=2
        # Second dip: peak=6, trough=1 → dd=5  ← largest
        equity = [1.0, 5.0, 3.0, 6.0, 1.0]
        assert AnalyticsEngine._max_drawdown(equity) == 5.0

    def test_empty_list(self):
        assert AnalyticsEngine._max_drawdown([]) == 0.0

    def test_single_element(self):
        # 0 → 5  : no dip
        assert AnalyticsEngine._max_drawdown([5.0]) == 0.0

    def test_single_negative_element(self):
        # 0 → -3  : peak=0, trough=-3 → dd=3
        assert AnalyticsEngine._max_drawdown([-3.0]) == 3.0

    def test_all_negative(self):
        # 0 → -2 → -4 → -6  : peak=0, trough=-6 → dd=6
        assert AnalyticsEngine._max_drawdown([-2.0, -4.0, -6.0]) == 6.0


class TestSummariseEdgeCases:
    """Edge cases: zero trades, single trade."""

    def test_zero_trades(self):
        result = BacktestResult(trades=[])
        result = AnalyticsEngine.summarise(result)
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.total_pnl == 0.0
        assert result.max_drawdown == 0.0
        assert result.equity_curve == []

    def test_single_winning_trade(self):
        trades = [_trade(1, pnl=0.50)]
        result = BacktestResult(trades=trades)
        result = AnalyticsEngine.summarise(result)
        assert result.total_trades == 1
        assert result.win_rate == 100.0
        assert result.max_drawdown == 0.0
