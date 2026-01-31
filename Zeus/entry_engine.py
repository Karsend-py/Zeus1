"""
entry_engine.py
---------------
TradeEntryEngine decides, bar-by-bar, whether conditions are met to open a
new short iron condor.  Every bar that *fails* a check is written to the
rejected-trades log with the *first* reason it failed (short-circuit).

Ordering of checks (fast-fail first, most expensive last):
    1. Indicator readiness  (NaN guard)
    2. NYSE session window
    3. Blackout buffer
    4. ADX threshold
    5. RSI bounds
    6. IV Rank minimum
    7. Duplicate-week guard  (only one entry per ISO week)
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd
import pytz

from src.strategy.models import (
    RejectedTrade,
    RejectionReason,
    StrategyParams,
    Trade,
    TradeResult,
    ExitReason,
)


def _next_friday(dt: datetime) -> datetime:
    """Return the *next* Friday >= dt (same day if dt is already Friday)."""
    days_ahead = (4 - dt.weekday()) % 7  # Monday=0 … Friday=4
    if days_ahead == 0 and dt.weekday() != 4:
        days_ahead = 7
    friday = dt + timedelta(days=days_ahead)
    # Normalise to end-of-day (4 PM ET) for expiry
    tz = dt.tzinfo or pytz.timezone("America/New_York")
    return friday.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=tz)


class TradeEntryEngine:
    """Stateful engine that walks through price bars and emits trades.

    Attributes
    ----------
    params : StrategyParams
        Frozen parameter snapshot for this backtest run.
    blackout_dates : set[date]
        Pre-expanded set of all blocked calendar dates (buffer already applied).
    """

    def __init__(self, params: StrategyParams, blackout_dates: set) -> None:
        self.params = params
        self.blackout_dates = blackout_dates  # already expanded by BlackoutFilter
        self._trade_counter = 0
        self._entered_weeks: set[int] = set()  # ISO week numbers already traded

        # Session boundaries in ET
        self._tz = pytz.timezone(params.timezone)
        self._session_open = time(params.session_open_hour, params.session_open_minute)
        self._session_close = time(params.session_close_hour, params.session_close_minute)

    # ---------------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------------

    def evaluate_bar(
        self,
        row: pd.Series,
        timestamp: datetime,
    ) -> Trade | RejectedTrade | None:
        """Evaluate a single bar.  Returns exactly one of:

        * Trade        — a new position was opened
        * RejectedTrade — the bar was a candidate but failed a filter
        * None         — not a candidate at all (e.g. weekend bar in daily data)
        """
        p = self.params

        # ------------------------------------------------------------------
        # 1. Indicator readiness
        # ------------------------------------------------------------------
        required = ["EMA", "ATR", "ADX", "RSI", "KC_Upper", "KC_Lower", "IV_Rank"]
        if any(pd.isna(row.get(col)) for col in required):
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.INDICATORS_NOT_READY,
                detail="One or more indicator columns contain NaN",
            )

        # ------------------------------------------------------------------
        # 2. NYSE session window
        # ------------------------------------------------------------------
        bar_time = timestamp.timetz() if timestamp.tzinfo else self._tz.localize(timestamp).timetz()
        if not (self._session_open <= bar_time.replace(tzinfo=None) <= self._session_close):
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.OUTSIDE_SESSION,
                detail=f"Bar time {bar_time} outside NYSE session",
            )

        # ------------------------------------------------------------------
        # 3. Blackout buffer
        # ------------------------------------------------------------------
        bar_date = timestamp.date()
        if bar_date in self.blackout_dates:
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.WITHIN_BLACKOUT_BUFFER,
                detail=f"Date {bar_date} falls within blackout buffer",
                adx=float(row["ADX"]),
                rsi=float(row["RSI"]),
                iv_rank=float(row["IV_Rank"]),
            )

        # ------------------------------------------------------------------
        # 4. ADX threshold
        # ------------------------------------------------------------------
        adx_val = float(row["ADX"])
        if adx_val > p.adx_threshold:
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.ADX_TOO_HIGH,
                detail=f"ADX={adx_val:.2f} > threshold={p.adx_threshold}",
                adx=adx_val,
                rsi=float(row["RSI"]),
                iv_rank=float(row["IV_Rank"]),
            )

        # ------------------------------------------------------------------
        # 5. RSI bounds
        # ------------------------------------------------------------------
        rsi_val = float(row["RSI"])
        if not (p.rsi_low <= rsi_val <= p.rsi_high):
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.RSI_OUT_OF_RANGE,
                detail=f"RSI={rsi_val:.2f} outside [{p.rsi_low}, {p.rsi_high}]",
                adx=adx_val,
                rsi=rsi_val,
                iv_rank=float(row["IV_Rank"]),
            )

        # ------------------------------------------------------------------
        # 6. IV Rank minimum
        # ------------------------------------------------------------------
        ivr_val = float(row["IV_Rank"])
        if ivr_val < p.iv_rank_min:
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.IV_RANK_TOO_LOW,
                detail=f"IV_Rank={ivr_val:.2f} < min={p.iv_rank_min}",
                adx=adx_val,
                rsi=rsi_val,
                iv_rank=ivr_val,
            )

        # ------------------------------------------------------------------
        # 7. Duplicate week guard
        # ------------------------------------------------------------------
        iso_week = timestamp.isocalendar()[1]
        if iso_week in self._entered_weeks:
            return RejectedTrade(
                timestamp=timestamp,
                reason=RejectionReason.DUPLICATE_WEEK,
                detail=f"Already entered a trade in ISO week {iso_week}",
                adx=adx_val,
                rsi=rsi_val,
                iv_rank=ivr_val,
            )

        # ------------------------------------------------------------------
        # All gates passed → emit a Trade
        # ------------------------------------------------------------------
        self._trade_counter += 1
        self._entered_weeks.add(iso_week)

        upper_strike = float(row["KC_Upper"])
        lower_strike = float(row["KC_Lower"])

        trade = Trade(
            trade_id=self._trade_counter,
            entry_timestamp=timestamp,
            expiry_date=_next_friday(timestamp),
            upper_strike=upper_strike,
            lower_strike=lower_strike,
            credit_received=p.credit_received,
            result=TradeResult.OPEN,       # will be updated by exit engine
            exit_reason=ExitReason.EXPIRY,  # default; exit engine may override
            entry_adx=adx_val,
            entry_rsi=rsi_val,
            entry_iv_rank=ivr_val,
            entry_ema=float(row["EMA"]),
        )
        return trade
