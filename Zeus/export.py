"""
export.py
---------
ExportEngine produces downloadable artefacts.

Outputs
-------
* ``config.json``  — full StrategyParams as a flat JSON object
* ``config.yaml``  — same, in YAML
* ``trades.csv``   — closed trades table
* ``rejected.csv`` — rejected trades table
* ``metrics.csv``  — summary KPIs as a single-row CSV

All methods return ``str`` or ``bytes`` so Streamlit can hand them directly
to ``st.download_button``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd
import yaml

from models import BacktestResult, StrategyParams


class ExportEngine:
    """Stateless export utility."""

    # ---------------------------------------------------------------------------
    # Configuration exports
    # ---------------------------------------------------------------------------

    @staticmethod
    def params_to_json(params: StrategyParams) -> str:
        """Serialise StrategyParams to a pretty-printed JSON string."""
        return json.dumps(asdict(params), indent=2)

    @staticmethod
    def params_to_yaml(params: StrategyParams) -> str:
        """Serialise StrategyParams to a YAML string."""
        return yaml.dump(asdict(params), default_flow_style=False, sort_keys=False)

    # ---------------------------------------------------------------------------
    # Result exports
    # ---------------------------------------------------------------------------

    @staticmethod
    def trades_to_csv(result: BacktestResult) -> str:
        """Convert closed trades to a CSV string."""
        if not result.trades:
            return "No trades to export.\n"
        df = ExportEngine._trades_df(result)
        return df.to_csv(index=False)

    @staticmethod
    def rejected_to_csv(result: BacktestResult) -> str:
        """Convert rejected trades to a CSV string."""
        if not result.rejected_trades:
            return "No rejected trades to export.\n"
        df = ExportEngine._rejected_df(result)
        return df.to_csv(index=False)

    @staticmethod
    def metrics_to_csv(result: BacktestResult) -> str:
        """Export summary KPIs as a one-row CSV."""
        row: dict[str, Any] = {
            "Total Trades": result.total_trades,
            "Wins": result.total_wins,
            "Losses": result.total_losses,
            "Win Rate (%)": result.win_rate,
            "Total P&L ($)": result.total_pnl,
            "Max Drawdown ($)": result.max_drawdown,
        }
        df = pd.DataFrame([row])
        return df.to_csv(index=False)

    # ---------------------------------------------------------------------------
    # Internal DataFrame builders (also used by the Streamlit UI)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _trades_df(result: BacktestResult) -> pd.DataFrame:
        rows = []
        for t in result.trades:
            rows.append(
                {
                    "Trade ID": t.trade_id,
                    "Entry Timestamp": t.entry_timestamp,
                    "Exit Timestamp": t.exit_timestamp,
                    "Expiry Date": t.expiry_date,
                    "Upper Strike": t.upper_strike,
                    "Lower Strike": t.lower_strike,
                    "Credit Received": t.credit_received,
                    "Loss Realised": t.loss_realised,
                    "P&L": t.pnl,
                    "Result": t.result.value,
                    "Exit Reason": t.exit_reason.value,
                    "ADX at Entry": round(t.entry_adx, 2),
                    "RSI at Entry": round(t.entry_rsi, 2),
                    "IV Rank at Entry": round(t.entry_iv_rank, 2),
                    "EMA at Entry": round(t.entry_ema, 2),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _rejected_df(result: BacktestResult) -> pd.DataFrame:
        rows = []
        for r in result.rejected_trades:
            rows.append(
                {
                    "Timestamp": r.timestamp,
                    "Rejection Reason": r.reason.value,
                    "Detail": r.detail,
                    "ADX": r.adx,
                    "RSI": r.rsi,
                    "IV Rank": r.iv_rank,
                }
            )
        return pd.DataFrame(rows)
