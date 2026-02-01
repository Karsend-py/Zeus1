"""
test_models.py
--------------
Unit tests for the frozen dataclass domain objects.
"""

import pytest
from datetime import datetime

from models import (
    StrategyParams,
    Trade,
    RejectedTrade,
    TradeResult,
    ExitReason,
    RejectionReason,
)


class TestStrategyParams:
    """Validation gate tests."""

    def test_defaults_are_valid(self):
        """Default construction must not raise."""
        p = StrategyParams()
        assert p.ema_period == 20
        assert p.credit_received == 0.50

    def test_ema_period_zero_raises(self):
        with pytest.raises(ValueError, match="ema_period"):
            StrategyParams(ema_period=0)

    def test_atr_period_negative_raises(self):
        with pytest.raises(ValueError, match="atr_period"):
            StrategyParams(atr_period=-1)

    def test_atr_multiplier_zero_raises(self):
        with pytest.raises(ValueError, match="atr_multiplier"):
            StrategyParams(atr_multiplier=0.0)

    def test_rsi_bounds_inverted_raises(self):
        with pytest.raises(ValueError, match="rsi_low must be <= rsi_high"):
            StrategyParams(rsi_low=80.0, rsi_high=20.0)

    def test_rsi_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rsi_low"):
            StrategyParams(rsi_low=-5.0)

    def test_iv_rank_out_of_range_raises(self):
        with pytest.raises(ValueError, match="iv_rank_min"):
            StrategyParams(iv_rank_min=150.0)

    def test_negative_credit_raises(self):
        with pytest.raises(ValueError, match="credit_received"):
            StrategyParams(credit_received=-1.0)

    def test_negative_max_loss_raises(self):
        with pytest.raises(ValueError, match="max_loss"):
            StrategyParams(max_loss=-0.5)

    def test_negative_blackout_buffer_raises(self):
        with pytest.raises(ValueError, match="blackout_buffer_days"):
            StrategyParams(blackout_buffer_days=-2)

    def test_frozen(self):
        """StrategyParams must be immutable."""
        p = StrategyParams()
        with pytest.raises(AttributeError):
            p.ema_period = 99  # type: ignore[misc]


class TestTrade:
    """Trade immutability + construction."""

    def _make_trade(self, **kwargs):
        defaults = dict(
            trade_id=1,
            entry_timestamp=datetime(2024, 1, 15, 10, 0),
            expiry_date=datetime(2024, 1, 19, 16, 0),
            upper_strike=5100.0,
            lower_strike=4900.0,
            credit_received=0.50,
            result=TradeResult.WIN,
            exit_reason=ExitReason.EXPIRY,
        )
        defaults.update(kwargs)
        return Trade(**defaults)

    def test_construction(self):
        t = self._make_trade()
        assert t.trade_id == 1
        assert t.result == TradeResult.WIN

    def test_frozen(self):
        t = self._make_trade()
        with pytest.raises(AttributeError):
            t.trade_id = 999  # type: ignore[misc]

    def test_pnl_defaults(self):
        t = self._make_trade()
        assert t.pnl == 0.0
        assert t.loss_realised == 0.0


class TestRejectedTrade:
    """RejectedTrade construction + immutability."""

    def test_construction(self):
        r = RejectedTrade(
            timestamp=datetime(2024, 1, 10, 11, 0),
            reason=RejectionReason.ADX_TOO_HIGH,
            detail="ADX=30 > 25",
            adx=30.0,
        )
        assert r.reason == RejectionReason.ADX_TOO_HIGH
        assert r.adx == 30.0
        assert r.rsi is None

    def test_frozen(self):
        r = RejectedTrade(
            timestamp=datetime(2024, 1, 10, 11, 0),
            reason=RejectionReason.RSI_OUT_OF_RANGE,
        )
        with pytest.raises(AttributeError):
            r.reason = RejectionReason.ADX_TOO_HIGH  # type: ignore[misc]
