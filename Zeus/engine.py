"""
engine.py  (analytics)
----------------------
AnalyticsEngine is a pure-function namespace.  It takes a BacktestResult and
fills in the summary metrics + equity curve.  No IO, no side effects.

Metrics computed
----------------
* Total trades / wins / losses
* Win rate (%)
* Total P&L ($)
* Maximum drawdown ($)  — largest peak-to-trough on the equity curve
* Cumulative equity curve (list of floats, one entry per closed trade)
"""

from __future__ import annotations

from models import BacktestResult, TradeResult


class AnalyticsEngine:
    """Stateless analytics calculator."""

    @staticmethod
    def summarise(result: BacktestResult) -> BacktestResult:
        """Populate *result* in place and return it.

        Parameters
        ----------
        result : BacktestResult
            Must have ``trades`` populated.  ``rejected_trades`` is ignored here.

        Returns
        -------
        BacktestResult
            Same object with summary fields filled.
        """
        trades = result.trades

        # ------------------------------------------------------------------
        # Counts
        # ------------------------------------------------------------------
        total = len(trades)
        wins = sum(1 for t in trades if t.result == TradeResult.WIN)
        losses = total - wins

        # ------------------------------------------------------------------
        # Equity curve  (running P&L after each closed trade)
        # ------------------------------------------------------------------
        equity: list[float] = []
        timestamps = []
        running_pnl = 0.0

        for t in trades:
            running_pnl += t.pnl
            equity.append(round(running_pnl, 4))
            timestamps.append(t.exit_timestamp)

        # ------------------------------------------------------------------
        # Max drawdown
        # ------------------------------------------------------------------
        max_dd = AnalyticsEngine._max_drawdown(equity)

        # ------------------------------------------------------------------
        # Write back
        # ------------------------------------------------------------------
        result.total_trades = total
        result.total_wins = wins
        result.total_losses = losses
        result.total_pnl = round(running_pnl, 4)
        result.max_drawdown = round(max_dd, 4)
        result.win_rate = round((wins / total * 100) if total > 0 else 0.0, 2)
        result.equity_curve = equity
        result.timestamps = timestamps

        return result

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(equity: list[float]) -> float:
        """Classic peak-to-trough drawdown on a running equity series.

        The implicit starting equity is 0 (before any trades).  This means a
        series like [-2, -4, -6] has a drawdown of 6 (from the 0 peak to -6).

        Returns a non-negative number.  If equity is empty, returns 0.
        """
        if not equity:
            return 0.0

        # Prepend the implicit zero starting point
        full = [0.0] + equity
        peak = full[0]
        max_dd = 0.0

        for val in full[1:]:
            if val > peak:
                peak = val
            drawdown = peak - val
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd
