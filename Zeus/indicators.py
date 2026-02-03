"""
indicators.py
-------------
Pure-function, vectorised indicator calculations.  Every method returns a
*new* DataFrame column or Series — the input DataFrame is never mutated.

Design rule: indicators are stateless.  They take a DataFrame + params and
return computed Series.  No side effects, no global state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import StrategyParams


class TechnicalIndicators:
    """Namespace for all indicator computations.

    Usage::

        df = TechnicalIndicators.compute_all(df, params)
        # df now has columns: EMA, ATR, ADX, RSI, KC_Upper, KC_Lower
    """

    # ---------------------------------------------------------------------------
    # Public façade
    # ---------------------------------------------------------------------------

    @staticmethod
    def compute_all(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        """Attach every required indicator column to *df* and return it.

        Parameters
        ----------
        df : DataFrame
            Must contain at least: Open, High, Low, Close, Volume.
        params : StrategyParams
            Drives all lookback windows and the Keltner multiplier.

        Returns
        -------
        DataFrame
            Same rows as *df*, with new columns appended.
        """
        out = df.copy()
        out = TechnicalIndicators._add_ema(out, params.ema_period)
        out = TechnicalIndicators._add_atr(out, params.atr_period)
        out = TechnicalIndicators._add_adx(out, params.adx_period)
        out = TechnicalIndicators._add_rsi(out)
        out = TechnicalIndicators._add_keltner(out, params.atr_multiplier)
        out = TechnicalIndicators._add_price_range_rank(out)
        out = TechnicalIndicators._add_side_specific_prr(out)
        return out

    # ---------------------------------------------------------------------------
    # EMA  (exponential moving average of Close)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
        df["EMA"] = df["Close"].ewm(span=period, adjust=False).mean()
        return df

    # ---------------------------------------------------------------------------
    # ATR  (Average True Range)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR"] = tr.ewm(span=period, adjust=False).mean()
        return df

    # ---------------------------------------------------------------------------
    # ADX  (Average Directional Index)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
        high = df["High"]
        low = df["Low"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = (
            pd.Series(plus_dm, index=df.index)
            .ewm(span=period, adjust=False)
            .mean()
        )
        minus_di = (
            pd.Series(minus_dm, index=df.index)
            .ewm(span=period, adjust=False)
            .mean()
        )

        # Use ATR if already computed; otherwise fall back to a quick TR calc
        if "ATR" in df.columns:
            atr = df["ATR"]
        else:
            prev_close = df["Close"].shift(1)
            tr = pd.concat(
                [
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()

        # Normalise DI
        plus_di_norm = 100 * plus_di / atr.replace(0, np.nan)
        minus_di_norm = 100 * minus_di / atr.replace(0, np.nan)

        # DX
        denom = (plus_di_norm + minus_di_norm).replace(0, np.nan)
        dx = 100 * (plus_di_norm - minus_di_norm).abs() / denom

        # ADX = smoothed DX
        df["ADX"] = dx.ewm(span=period, adjust=False).mean()
        return df

    # ---------------------------------------------------------------------------
    # RSI  (Relative Strength Index, period=14 always)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    # ---------------------------------------------------------------------------
    # Keltner Channels  (strikes)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_keltner(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
        """Upper / lower bands define the short strikes.

        Strikes are rounded to the nearest whole dollar for realism.
        """
        df["KC_Upper"] = (df["EMA"] + multiplier * df["ATR"]).round(0)
        df["KC_Lower"] = (df["EMA"] - multiplier * df["ATR"]).round(0)
        return df

    # ---------------------------------------------------------------------------
    # Price Range Rank  (proxy for IV Rank, derived purely from price action)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_price_range_rank(df: pd.DataFrame) -> pd.DataFrame:
        """Where does today's close sit inside its 252-bar price range?

        Result is in [0, 1].  Bars where the 252-bar window is not yet full,
        or where recent_high == recent_low (flat price), are left as NaN — the
        entry engine's NaN guard will skip them automatically.
        """
        close = df["Close"]
        recent_high = close.rolling(252).max()
        recent_low = close.rolling(252).min()
        price_range = recent_high - recent_low

        # Guard: where the range is zero (flat price) set to NaN so we don't
        # produce inf or 0/0.
        price_range_safe = price_range.replace(0, np.nan)

        df["Price_Range_Rank"] = (close - recent_low) / price_range_safe
        return df

    # ---------------------------------------------------------------------------
    # Side-Specific PRR  (for directional credit spreads)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_side_specific_prr(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PRR for each side using the Keltner strikes.

        PRR_upside: where does Close sit relative to the range [Close, KC_Upper]?
        PRR_downside: where does Close sit relative to the range [KC_Lower, Close]?

        These enable directional spread selection:
        - Call credit spread needs PRR_upside > threshold (room to the upside)
        - Put credit spread needs PRR_downside > threshold (room to the downside)
        - Iron condor needs both

        Uses a 252-bar rolling window on the distance metrics.
        """
        close = df["Close"]
        upper_strike = df["KC_Upper"]
        lower_strike = df["KC_Lower"]

        # Upside: how much room from Close to Upper?
        upside_distance = upper_strike - close
        upside_high = upside_distance.rolling(252).max()
        upside_low = upside_distance.rolling(252).min()
        upside_range = upside_high - upside_low
        upside_range_safe = upside_range.replace(0, np.nan)
        df["PRR_upside"] = (upside_distance - upside_low) / upside_range_safe

        # Downside: how much room from Lower to Close?
        downside_distance = close - lower_strike
        downside_high = downside_distance.rolling(252).max()
        downside_low = downside_distance.rolling(252).min()
        downside_range = downside_high - downside_low
        downside_range_safe = downside_range.replace(0, np.nan)
        df["PRR_downside"] = (downside_distance - downside_low) / downside_range_safe

        return df
