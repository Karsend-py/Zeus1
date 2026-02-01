"""
app.py
------
Streamlit entry-point for the Short Iron Condor Backtester.

Layout
------
┌────────────┬─────────────────────────────────────────────┐
│  Sidebar   │  Main area                                  │
│            │  ├─ KPI metric cards                        │
│  • Params  │  ├─ Cumulative equity curve (line chart)    │
│  • Uploads │  ├─ Closed trades table                     │
│            │  ├─ Rejected trades table                   │
└────────────┘  └─ Export buttons                          │
                └─────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from blackout import BlackoutFilter
from export import ExportEngine
from loader import DataLoader
from indicators import TechnicalIndicators
from models import StrategyParams
from runner import BacktestRunner

# ---------------------------------------------------------------------------
# Default params loader
# ---------------------------------------------------------------------------

DEFAULTS_PATH = Path(__file__).resolve().parent / "default_params.yaml"


def _load_defaults() -> dict:
    with open(DEFAULTS_PATH) as f:
        return yaml.safe_load(f)


# ===========================================================================
# Streamlit page config
# ===========================================================================

st.set_page_config(
    page_title="Short Iron Condor Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Metric card styling */
        .stMetric {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #313244;
        }
        /* Table styling */
        .stDataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        /* Download button row */
        .export-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# SIDEBAR
# ===========================================================================

defaults = _load_defaults()

st.sidebar.title("⚙️ Strategy Parameters")

# --- Indicator Settings ---
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Indicators")
ema_period = st.sidebar.number_input(
    "EMA Period", min_value=1, value=defaults["indicators"]["ema_period"], step=1
)
atr_period = st.sidebar.number_input(
    "ATR Period", min_value=1, value=defaults["indicators"]["atr_period"], step=1
)
atr_multiplier = st.sidebar.number_input(
    "Keltner Multiplier", min_value=0.1, value=defaults["indicators"]["atr_multiplier"], step=0.1
)
adx_period = st.sidebar.number_input(
    "ADX Period", min_value=1, value=defaults["indicators"]["adx_period"], step=1
)

# --- Filter Thresholds ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Entry Filters")
adx_threshold = st.sidebar.number_input(
    "Max ADX (enter below)", min_value=0.0, value=defaults["filters"]["adx_threshold"], step=1.0
)
rsi_low = st.sidebar.number_input(
    "RSI Low Bound", min_value=0.0, max_value=100.0,
    value=defaults["filters"]["rsi_low"], step=1.0
)
rsi_high = st.sidebar.number_input(
    "RSI High Bound", min_value=0.0, max_value=100.0,
    value=defaults["filters"]["rsi_high"], step=1.0
)
iv_rank_min = st.sidebar.number_input(
    "Min IV Rank", min_value=0.0, max_value=100.0,
    value=defaults["filters"]["iv_rank_min"], step=1.0
)
blackout_buffer = st.sidebar.number_input(
    "Blackout Buffer (days)", min_value=0,
    value=defaults["filters"]["blackout_buffer_days"], step=1
)

# --- Slippage / Credit Model ---
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Slippage Model")
credit_received = st.sidebar.number_input(
    "Credit Received ($)", min_value=0.0,
    value=defaults["slippage"]["credit_received"], step=0.05
)
max_loss = st.sidebar.number_input(
    "Max Loss on Breach ($)", min_value=0.0,
    value=defaults["slippage"]["max_loss"], step=0.05
)

# --- File Uploads ---
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Upload")
price_file = st.sidebar.file_uploader("Price Data CSV", type=["csv"], key="price_upload")
blackout_file = st.sidebar.file_uploader("Blackout Dates CSV", type=["csv"], key="blackout_upload")

# --- Run button ---
st.sidebar.markdown("---")
run_backtest = st.sidebar.button("▶ Run Backtest", use_container_width=True, type="primary")

# ===========================================================================
# MAIN AREA — Header
# ===========================================================================

st.title("📊 Short Iron Condor Backtester")
st.caption(
    "1-week short iron condor strategy • Keltner Channel strikes • "
    "Fixed slippage model • Intrabar breach exit"
)

# ===========================================================================
# EXECUTION
# ===========================================================================

if not run_backtest:
    st.info(
        "Upload your Price CSV and (optionally) a Blackout CSV in the sidebar, "
        "then click **▶ Run Backtest** to begin.",
        icon="ℹ️",
    )
    st.stop()

# --- Validate uploads ---
if price_file is None:
    st.error("❌ Please upload a Price Data CSV before running.", icon="🚨")
    st.stop()

# --- Build StrategyParams ---
try:
    params = StrategyParams(
        ema_period=int(ema_period),
        atr_period=int(atr_period),
        atr_multiplier=float(atr_multiplier),
        adx_period=int(adx_period),
        adx_threshold=float(adx_threshold),
        rsi_low=float(rsi_low),
        rsi_high=float(rsi_high),
        iv_rank_min=float(iv_rank_min),
        blackout_buffer_days=int(blackout_buffer),
        credit_received=float(credit_received),
        max_loss=float(max_loss),
    )
except ValueError as exc:
    st.error(f"❌ Parameter validation failed: {exc}", icon="🚨")
    st.stop()

# --- Load price data ---
with st.spinner("Loading & validating price data…"):
    try:
        price_csv_text = price_file.getvalue().decode("utf-8")
        df = DataLoader.load_price_data(price_csv_text)
    except ValueError as exc:
        st.error(f"❌ Price data error: {exc}", icon="🚨")
        st.stop()

# --- Load blackout data ---
blackout_warnings: list[str] = []
blackout_dates: set = set()

if blackout_file is not None:
    with st.spinner("Loading blackout dates…"):
        try:
            blackout_csv_text = blackout_file.getvalue().decode("utf-8")
            blackout_df = DataLoader.load_blackout_dates(blackout_csv_text)
            blackout_dates, blackout_warnings = BlackoutFilter.expand(
                blackout_df, params.blackout_buffer_days
            )
        except ValueError as exc:
            st.warning(f"⚠️ Blackout file issue: {exc}. Continuing without blackouts.")

# Surface any overlap warnings
for w in blackout_warnings:
    st.warning(w, icon="⚠️")

# --- Compute indicators ---
with st.spinner("Computing technical indicators…"):
    df = TechnicalIndicators.compute_all(df, params)

# --- Run backtest ---
with st.spinner("Running backtest…"):
    runner = BacktestRunner(df, params, blackout_dates)
    result = runner.run()

# ===========================================================================
# ANALYTICS DASHBOARD
# ===========================================================================

# --- KPI Cards ---
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", result.total_trades)
col2.metric("Wins", result.total_wins)
col3.metric("Losses", result.total_losses)

wr_display = f"{result.win_rate:.1f}%"
col4.metric("Win Rate", wr_display)

pnl_sign = "+" if result.total_pnl >= 0 else ""
col5.metric("Total P&L", f"{pnl_sign}${result.total_pnl:.2f}")

col6.metric("Max Drawdown", f"-${result.max_drawdown:.2f}")

st.markdown("---")

# --- Equity Curve ---
if result.equity_curve:
    st.subheader("📈 Cumulative Equity Curve")
    equity_df = pd.DataFrame(
        {
            "Date": result.timestamps,
            "Equity ($)": result.equity_curve,
        }
    )
    st.line_chart(
        equity_df.set_index("Date"),
        use_container_width=True,
        height=320,
    )
    st.markdown("---")
else:
    st.info("No trades were executed — equity curve is empty.", icon="📉")

# --- Trades Table ---
st.subheader("📋 Closed Trades")
trades_df = ExportEngine._trades_df(result)
if not trades_df.empty:
    # Colour-code result column
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P&L": st.column_config.NumberColumn("P&L ($)", format="$%.2f"),
            "Credit Received": st.column_config.NumberColumn("Credit ($)", format="$%.2f"),
            "Loss Realised": st.column_config.NumberColumn("Loss ($)", format="$%.2f"),
            "Upper Strike": st.column_config.NumberColumn("Upper Strike", format="%.0f"),
            "Lower Strike": st.column_config.NumberColumn("Lower Strike", format="%.0f"),
        },
    )
else:
    st.info("No trades were closed in this backtest run.", icon="📭")

st.markdown("---")

# --- Rejected Trades Table ---
st.subheader("🚫 Rejected Trade Candidates")
rejected_df = ExportEngine._rejected_df(result)
if not rejected_df.empty:
    # Show a condensed view — limit to 500 rows for performance
    display_rejected = rejected_df.head(500).copy()
    st.dataframe(
        display_rejected,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ADX": st.column_config.NumberColumn("ADX", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
            "IV Rank": st.column_config.NumberColumn("IV Rank", format="%.2f"),
        },
    )
    if len(rejected_df) > 500:
        st.caption(f"Showing 500 of {len(rejected_df)} rejected candidates. Download for full data.")
else:
    st.info("No trade candidates were rejected.", icon="✅")

st.markdown("---")

# ===========================================================================
# EXPORT BUTTONS
# ===========================================================================

st.subheader("💾 Export Results")
export_cols = st.columns(5)

export_cols[0].download_button(
    label="📥 Config (JSON)",
    data=ExportEngine.params_to_json(params),
    file_name="strategy_config.json",
    mime="application/json",
)

export_cols[1].download_button(
    label="📥 Config (YAML)",
    data=ExportEngine.params_to_yaml(params),
    file_name="strategy_config.yaml",
    mime="text/yaml",
)

export_cols[2].download_button(
    label="📥 Trades CSV",
    data=ExportEngine.trades_to_csv(result),
    file_name="trades.csv",
    mime="text/csv",
)

export_cols[3].download_button(
    label="📥 Rejected CSV",
    data=ExportEngine.rejected_to_csv(result),
    file_name="rejected_trades.csv",
    mime="text/csv",
)

export_cols[4].download_button(
    label="📥 Metrics CSV",
    data=ExportEngine.metrics_to_csv(result),
    file_name="metrics.csv",
    mime="text/csv",
)
