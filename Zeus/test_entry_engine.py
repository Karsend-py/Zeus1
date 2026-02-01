"""
test_entry_engine.py
--------------------
Unit tests for TradeEntryEngine.evaluate_bar().

Each test constructs a minimal row Series that would pass all gates, then
tweaks exactly one value to trigger the expected rejection — or leaves it
pristine to verify a Trade is emitted.
"""

from datetime import date, datetime

import pandas as pd
import pytz

import pytest

from entry_engine import TradeEntryEngine
from models import (
    RejectedTrade,
    RejectionReason,
    StrategyParams,
    Trade,
)

ET = pytz.timezone("America/New_York")


def _params(**overrides) -> StrategyParams:
    defaults = dict(
        ema_period=20,
        atr_period=14,
        atr_multiplier=2.0,
        adx_period=14,
        adx_threshold=25.0,
        rsi_low=30.0,
        rsi_high=70.0,
        price_range_rank_min=0.3,
        blackout_buffer_days=2,
        credit_received=0.50,
        max_loss=2.50,
    )
    defaults.update(overrides)
    return StrategyParams(**defaults)


def _good_row() -> pd.Series:
    """A row that passes every filter."""
    return pd.Series(
        {
            "Open": 5000.0,
            "High": 5020.0,
            "Low": 4980.0,
            "Close": 5010.0,
            "Volume": 5000,
            "Price_Range_Rank": 0.45,
            "EMA": 5000.0,
            "ATR": 25.0,
            "ADX": 20.0,
            "RSI": 50.0,
            "KC_Upper": 5050.0,
            "KC_Lower": 4950.0,
        }
    )


def _good_timestamp() -> datetime:
    """A Wednesday at 10 AM ET (well within session)."""
    return ET.localize(datetime(2024, 1, 10, 10, 0, 0))  # Wednesday


class TestEntryGates:
    """Each test isolates one filter gate."""

    def test_nan_indicator_rejected(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()
        row["ADX"] = float("nan")
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.INDICATORS_NOT_READY

    def test_outside_session_rejected(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()
        # 8 AM ET — before open
        ts = ET.localize(datetime(2024, 1, 10, 8, 0, 0))
        result = engine.evaluate_bar(row, ts)
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.OUTSIDE_SESSION

    def test_blackout_buffer_rejected(self):
        blocked = {date(2024, 1, 10)}  # the good timestamp's date
        engine = TradeEntryEngine(_params(), blackout_dates=blocked)
        row = _good_row()
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.WITHIN_BLACKOUT_BUFFER

    def test_adx_too_high_rejected(self):
        engine = TradeEntryEngine(_params(adx_threshold=15.0), blackout_dates=set())
        row = _good_row()
        row["ADX"] = 20.0  # above threshold of 15
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.ADX_TOO_HIGH

    def test_rsi_too_low_rejected(self):
        engine = TradeEntryEngine(_params(rsi_low=40.0), blackout_dates=set())
        row = _good_row()
        row["RSI"] = 35.0  # below 40
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.RSI_OUT_OF_RANGE

    def test_rsi_too_high_rejected(self):
        engine = TradeEntryEngine(_params(rsi_high=60.0), blackout_dates=set())
        row = _good_row()
        row["RSI"] = 65.0  # above 60
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.RSI_OUT_OF_RANGE

    def test_price_range_rank_too_low_rejected(self):
        engine = TradeEntryEngine(_params(price_range_rank_min=0.5), blackout_dates=set())
        row = _good_row()
        row["Price_Range_Rank"] = 0.4  # below 0.5
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, RejectedTrade)
        assert result.reason == RejectionReason.PRICE_RANGE_RANK_TOO_LOW

    def test_duplicate_week_rejected(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()
        ts = _good_timestamp()

        # First call should succeed
        first = engine.evaluate_bar(row, ts)
        assert isinstance(first, Trade)

        # Second call same week (Thursday) should reject
        ts_thu = ET.localize(datetime(2024, 1, 11, 10, 0, 0))
        second = engine.evaluate_bar(row, ts_thu)
        assert isinstance(second, RejectedTrade)
        assert second.reason == RejectionReason.DUPLICATE_WEEK


class TestEntrySuccess:
    """Happy-path trade emission."""

    def test_trade_emitted_with_correct_strikes(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()
        result = engine.evaluate_bar(row, _good_timestamp())

        assert isinstance(result, Trade)
        assert result.upper_strike == 5050.0
        assert result.lower_strike == 4950.0
        assert result.credit_received == 0.50
        assert result.trade_id == 1

    def test_trade_ids_increment(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()

        # Week 2 (Jan 10) and week 3 (Jan 17) — different ISO weeks
        t1 = engine.evaluate_bar(row, ET.localize(datetime(2024, 1, 10, 10, 0)))
        t2 = engine.evaluate_bar(row, ET.localize(datetime(2024, 1, 17, 10, 0)))

        assert isinstance(t1, Trade)
        assert isinstance(t2, Trade)
        assert t1.trade_id == 1
        assert t2.trade_id == 2

    def test_expiry_is_next_friday(self):
        engine = TradeEntryEngine(_params(), blackout_dates=set())
        row = _good_row()
        # Wednesday Jan 10 → next Friday is Jan 12
        result = engine.evaluate_bar(row, _good_timestamp())
        assert isinstance(result, Trade)
        assert result.expiry_date.weekday() == 4  # Friday
        assert result.expiry_date.date() == date(2024, 1, 12)
