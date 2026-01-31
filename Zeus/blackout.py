"""
blackout.py
-----------
BlackoutFilter takes the raw blackout DataFrame and a buffer size, and
produces a ``set[date]`` of every calendar day that is blocked for entry.

Buffer semantics
----------------
A blackout on 2024-03-15 with buffer=3 blocks:
    2024-03-12, 2024-03-13, 2024-03-14,   ← 3 days before
    2024-03-15,                             ← the event itself
    2024-03-16, 2024-03-17, 2024-03-18    ← 3 days after

Overlap validation
------------------
If two blackout events produce overlapping buffer windows, the union is kept
and a warning string is returned so the UI can surface it.  No dates are
silently dropped.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


class BlackoutFilter:
    """Stateless blackout expansion + overlap checker."""

    @staticmethod
    def expand(
        blackout_df: pd.DataFrame,
        buffer_days: int,
    ) -> tuple[set[date], list[str]]:
        """Expand raw blackout dates into a blocked set.

        Parameters
        ----------
        blackout_df : DataFrame
            Must have columns [Date (date), Reason (str)].
        buffer_days : int
            Number of calendar days to block before *and* after each event.

        Returns
        -------
        blocked_dates : set[date]
            Every date that is off-limits for trade entry.
        warnings : list[str]
            Human-readable overlap notifications (empty if none).
        """
        if blackout_df.empty:
            return set(), []

        # Build per-event ranges
        ranges: list[tuple[date, date, str]] = []
        for _, row in blackout_df.iterrows():
            event_date: date = row["Date"]
            reason: str = str(row["Reason"])
            start = event_date - timedelta(days=buffer_days)
            end = event_date + timedelta(days=buffer_days)
            ranges.append((start, end, reason))

        # Detect overlaps (O(n²) — fine for <100 events)
        warnings: list[str] = []
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                s_i, e_i, r_i = ranges[i]
                s_j, e_j, r_j = ranges[j]
                if s_i <= e_j and s_j <= e_i:
                    warnings.append(
                        f"Blackout overlap: '{r_i}' [{s_i} – {e_i}] "
                        f"and '{r_j}' [{s_j} – {e_j}]"
                    )

        # Union all dates
        blocked: set[date] = set()
        for start, end, _ in ranges:
            current = start
            while current <= end:
                blocked.add(current)
                current += timedelta(days=1)

        return blocked, warnings
