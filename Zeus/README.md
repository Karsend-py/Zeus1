# Short Iron Condor Backtester

A modular, class-based Streamlit backtesting engine for a 1-week short iron condor
options strategy. Designed for extensibility, clean analytics, and production-grade
engineering practices.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                     │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ Sidebar  │  │ Analytics  │  │   Tables   │  │   Export    │ │
│  │  Params  │  │  Charts    │  │  Trades /  │  │  JSON/YAML  │ │
│  │  Upload  │  │  Metrics   │  │  Rejected  │  │  CSV        │ │
│  └────┬─────┘  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘ │
│       │              │               │                 │         │
│       └──────────────┴───────┬───────┴─────────────────┘         │
│                              ▼                                    │
│                    ┌─────────────────────┐                        │
│                    │   BacktestRunner    │  ← orchestrator        │
│                    └─────────┬───────────┘                        │
│                              │                                    │
│          ┌───────────────────┼───────────────────┐               │
│          ▼                   ▼                   ▼               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │  DataLoader │   │ TechnicalInd │   │BlackoutFilter│          │
│  │  + TZ norm  │   │ (EMA/ATR/    │   │ + buffer     │          │
│  │  + validate │   │  ADX/RSI/KC) │   │   logic      │          │
│  └─────────────┘   └──────────────┘   └──────────────┘          │
│          │                   │                   │               │
│          ▼                   ▼                   ▼               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │ TradeEntry  │   │  TradeExit   │   │  Analytics   │          │
│  │  Engine     │   │  Engine      │   │  Engine      │          │
│  └─────────────┘   └──────────────┘   └──────────────┘          │
│          │                   │                   │               │
│          └───────────────────┼───────────────────┘               │
│                              ▼                                    │
│                    ┌─────────────────────┐                        │
│                    │    Trade (dataclass) │  ← immutable record   │
│                    └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
short_iron_condor_backtester/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Black / Flake8 / project config
├── config/
│   └── default_params.yaml         # Default strategy parameters
├── src/
│   ├── __init__.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── models.py               # Trade, StrategyParams, RejectedTrade dataclasses
│   │   ├── indicators.py           # TechnicalIndicators: EMA, ATR, ADX, RSI, Keltner
│   │   ├── entry_engine.py         # TradeEntryEngine: signal generation + filtering
│   │   ├── exit_engine.py          # TradeExitEngine: intrabar breach detection
│   │   └── runner.py               # BacktestRunner: main orchestration loop
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── engine.py               # AnalyticsEngine: P&L, drawdown, equity curve
│   └── io/
│       ├── __init__.py
│       ├── loader.py               # DataLoader: CSV parse, TZ normalization, validation
│       ├── blackout.py             # BlackoutFilter: buffer enforcement
│       └── export.py               # ExportEngine: JSON/YAML config, CSV results
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + test on push
└── src/tests/
    ├── __init__.py
    ├── test_models.py              # Unit tests for dataclasses
    ├── test_indicators.py          # Unit tests for technical indicators
    ├── test_blackout.py            # Unit tests for blackout filter logic
    ├── test_entry_engine.py        # Unit tests for entry signal logic
    ├── test_exit_engine.py         # Unit tests for breach detection
    └── test_analytics.py          # Unit tests for P&L / drawdown
```

## Quick Start

```bash
# 1. Clone & create venv
git clone <repo-url>
cd short_iron_condor_backtester
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## CSV Format Requirements

### Price Data (`prices.csv`)

| Column      | Type      | Description                              |
|-------------|-----------|------------------------------------------|
| Timestamp   | datetime  | ISO 8601 or `YYYY-MM-DD HH:MM:SS`       |
| Open        | float     | Opening price                            |
| High        | float     | Intrabar high                            |
| Low         | float     | Intrabar low                             |
| Close       | float     | Closing price                            |
| Volume      | int       | Tick or share volume                     |
| IV_Rank     | float     | Implied Volatility Rank (0–100)          |

### Blackout Dates (`blackouts.csv`)

| Column      | Type      | Description                              |
|-------------|-----------|------------------------------------------|
| Date        | date      | `YYYY-MM-DD` format                      |
| Reason      | str       | Human-readable event label               |

## Key Design Decisions

1. **Immutable Trade records** — `Trade` and `RejectedTrade` are frozen dataclasses.
   Once written, a trade record is never mutated. The runner appends to lists.

2. **Timezone-first** — All timestamps are normalised to `America/New_York` immediately
   on load. NYSE session filtering uses localised comparisons, never naive datetimes.

3. **Intrabar breach model** — Exit fires when `High > upper_strike` OR
   `Low < lower_strike` within a single bar. No tick-level simulation needed.

4. **Fixed slippage via bid-ask spread** — Credit received = configured premium.
   Max loss on breach = configured loss value. Spread is the same for entry and exit,
   keeping the model simple and conservative.

5. **Keltner Channel strikes** — Upper strike = EMA + multiplier × ATR.
   Lower strike = EMA − multiplier × ATR. Strikes are rounded to the nearest whole
   dollar for realism.

6. **Rejection logging** — Every candidate bar that fails any filter is logged with
   the *first* failing reason. This gives full visibility into strategy behaviour
   without requiring a debugger.
