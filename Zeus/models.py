"""
models.py
---------
Immutable domain objects.  Every piece of strategy state is represented here.
Nothing outside this module should construct trades by hand; use the engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TradeResult(Enum):
    """Outcome of a completed trade."""

    WIN = "win"          # Expired worthless — full credit kept
    LOSS = "loss"        # Short strike breached — max loss realised
    OPEN = "open"        # Still live (should not appear in final results)


class ExitReason(Enum):
    """Why the trade was closed."""

    BREACH_SHORT_CALL = "breach_short_call"      # Price exceeded upper strike
    BREACH_SHORT_PUT = "breach_short_put"        # Price fell below lower strike
    EXPIRY_WORTHLESS = "expiry_worthless"        # Reached Friday expiry, expired worthless


class RejectionReason(Enum):
    """Why a candidate trade was *not* entered."""

    OUTSIDE_SESSION = "outside_nyse_session"
    ADX_TOO_HIGH = "adx_above_threshold"
    RSI_OUT_OF_RANGE = "rsi_outside_bounds"
    PRICE_RANGE_RANK_TOO_LOW = "price_range_rank_below_minimum"
    WITHIN_BLACKOUT_BUFFER = "within_blackout_buffer"
    INDICATORS_NOT_READY = "indicators_not_ready"   # NaN in required columns
    DUPLICATE_WEEK = "duplicate_entry_same_week"    # already entered this week


# ---------------------------------------------------------------------------
# StrategyParams  (the single source of truth for every tunable knob)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    """All user-configurable parameters.  Constructed once from the sidebar."""

    # Indicator lookbacks
    ema_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 2.0
    adx_period: int = 14

    # Entry filters
    adx_threshold: float = 25.0
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    price_range_rank_min: float = 0.3
    blackout_buffer_days: int = 3

    # Slippage / credit model
    credit_received: float = 0.50   # $ per contract on entry
    max_loss: float = 2.50          # $ per contract on breach

    # Session window (ET)
    session_open_hour: int = 9
    session_open_minute: int = 30
    session_close_hour: int = 16
    session_close_minute: int = 0
    timezone: str = "America/New_York"

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    def __post_init__(self) -> None:  # noqa: D105
        if self.ema_period < 1:
            raise ValueError("ema_period must be >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if self.adx_period < 1:
            raise ValueError("adx_period must be >= 1")
        if not (0 <= self.rsi_low <= 100):
            raise ValueError("rsi_low must be in [0, 100]")
        if not (0 <= self.rsi_high <= 100):
            raise ValueError("rsi_high must be in [0, 100]")
        if self.rsi_low > self.rsi_high:
            raise ValueError("rsi_low must be <= rsi_high")
        if not (0 <= self.price_range_rank_min <= 1):
            raise ValueError("price_range_rank_min must be in [0, 1]")
        if self.blackout_buffer_days < 0:
            raise ValueError("blackout_buffer_days must be >= 0")
        if self.credit_received < 0:
            raise ValueError("credit_received must be >= 0")
        if self.max_loss < 0:
            raise ValueError("max_loss must be >= 0")


# ---------------------------------------------------------------------------
# Trade  (a single executed position, frozen once created)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """One short iron condor execution.  Append-only; never mutated."""

    trade_id: int
    entry_timestamp: datetime
    expiry_date: datetime            # Nearest upcoming Friday
    upper_strike: float              # Short call strike (EMA + k*ATR)
    lower_strike: float              # Short put strike  (EMA − k*ATR)
    credit_received: float           # Fixed credit in $
    result: TradeResult              # WIN / LOSS
    exit_reason: ExitReason          # Why it closed
    exit_timestamp: Optional[datetime] = None
    loss_realised: float = 0.0       # 0 on win, max_loss on loss
    pnl: float = 0.0                 # credit_received − loss_realised

    # Metadata for the trades table
    entry_adx: float = 0.0
    entry_rsi: float = 0.0
    entry_price_range_rank: float = 0.0
    entry_ema: float = 0.0


# ---------------------------------------------------------------------------
# RejectedTrade  (a candidate that failed a pre-entry filter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectedTrade:
    """Logged every time a bar *could* have been an entry but was filtered out."""

    timestamp: datetime
    reason: RejectionReason
    detail: str = ""                 # Human-readable context (e.g. "ADX=28.3 > 25.0")
    adx: Optional[float] = None
    rsi: Optional[float] = None
    price_range_rank: Optional[float] = None


# ---------------------------------------------------------------------------
# BacktestResult  (top-level output bag)
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Everything the analytics engine and UI need after a run."""

    trades: list[Trade] = field(default_factory=list)
    rejected_trades: list[RejectedTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    # Computed summary (filled by AnalyticsEngine)
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
