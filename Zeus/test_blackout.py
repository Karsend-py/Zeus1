"""
test_blackout.py
----------------
Unit tests for BlackoutFilter.expand().
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from blackout import BlackoutFilter


def _blackout_df(dates_and_reasons: list[tuple[date, str]]) -> pd.DataFrame:
    return pd.DataFrame(dates_and_reasons, columns=["Date", "Reason"])


class TestBlackoutExpansion:
    """Core buffer-expansion logic."""

    def test_single_event_zero_buffer(self):
        df = _blackout_df([(date(2024, 3, 15), "Earnings")])
        blocked, warnings = BlackoutFilter.expand(df, buffer_days=0)
        assert blocked == {date(2024, 3, 15)}
        assert warnings == []

    def test_single_event_buffer_3(self):
        df = _blackout_df([(date(2024, 3, 15), "Earnings")])
        blocked, _ = BlackoutFilter.expand(df, buffer_days=3)
        expected = {
            date(2024, 3, 12),
            date(2024, 3, 13),
            date(2024, 3, 14),
            date(2024, 3, 15),
            date(2024, 3, 16),
            date(2024, 3, 17),
            date(2024, 3, 18),
        }
        assert blocked == expected

    def test_multiple_non_overlapping(self):
        df = _blackout_df([
            (date(2024, 1, 10), "Event A"),
            (date(2024, 2, 20), "Event B"),
        ])
        blocked, warnings = BlackoutFilter.expand(df, buffer_days=1)
        # Event A: Jan 9, 10, 11.  Event B: Feb 19, 20, 21.
        assert date(2024, 1, 9) in blocked
        assert date(2024, 1, 11) in blocked
        assert date(2024, 2, 19) in blocked
        assert date(2024, 2, 21) in blocked
        assert warnings == []  # No overlap

    def test_overlap_detected(self):
        # Two events 2 days apart with buffer=2 → ranges overlap
        df = _blackout_df([
            (date(2024, 3, 10), "Event A"),
            (date(2024, 3, 12), "Event B"),
        ])
        blocked, warnings = BlackoutFilter.expand(df, buffer_days=2)
        assert len(warnings) == 1
        assert "overlap" in warnings[0].lower()
        # Union should still contain all dates
        assert date(2024, 3, 8) in blocked   # A - 2
        assert date(2024, 3, 14) in blocked  # B + 2

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Date", "Reason"])
        blocked, warnings = BlackoutFilter.expand(df, buffer_days=5)
        assert blocked == set()
        assert warnings == []

    def test_buffer_size_one(self):
        df = _blackout_df([(date(2024, 6, 15), "Fed Meeting")])
        blocked, _ = BlackoutFilter.expand(df, buffer_days=1)
        assert len(blocked) == 3  # day before, day of, day after


class TestBlackoutEdgeCases:
    """Edge cases and boundary conditions."""

    def test_year_boundary(self):
        """Buffer should cross year boundaries correctly."""
        df = _blackout_df([(date(2024, 1, 1), "New Year")])
        blocked, _ = BlackoutFilter.expand(df, buffer_days=2)
        assert date(2023, 12, 30) in blocked
        assert date(2024, 1, 3) in blocked

    def test_large_buffer_does_not_crash(self):
        df = _blackout_df([(date(2024, 6, 15), "Big event")])
        blocked, _ = BlackoutFilter.expand(df, buffer_days=30)
        assert len(blocked) == 61  # 30 before + event + 30 after
