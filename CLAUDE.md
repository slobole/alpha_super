# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read First

Before changing code, read these doctrine documents in this exact order:

1. `QUANT_PHILOSOPHY.md`
2. `ASSUMPTIONS_AND_GAPS.md`
3. `FEATURE_ROADMAP.md`

These documents are authoritative for house philosophy, realism assumptions, and future direction. This file remains the operational entrypoint for repo-specific coding behavior.

Supplementary current-strategy behavior reference: CURRENT_STRATEGIES.md.

## Quantitative Correctness Standards

This codebase is held to a strict standard of quantitative rigor. Every piece of code â€” strategies, indicators, metrics, data handling â€” must be bullet-proof against common quant pitfalls. **Simplicity is a virtue**: prefer the clearest, most direct implementation over clever abstractions.

### Non-negotiable rules

**Lookahead bias** â€” The most critical failure mode. Signals, features, and any derived data must only use information available *before* the decision point. `compute_signals()` runs on the full dataset deliberately â€” any feature computed there must use only past data (e.g. rolling windows, `shift()`). `iterate()` receives only data up to `previous_bar`; never index into future bars.

**Survivorship bias** â€” Universe construction must use point-in-time constituent membership (Norgate's `index_constituent_timeseries`), not today's index composition. Never backtest on a static list of current constituents.

**Data-mining / overfitting** â€” Strategies should be based on a clear, explainable edge with few parameters. Do not add parameters to fit historical results. Do not optimize parameters on the full backtest period without out-of-sample validation.

**Execution realism** â€” Orders placed in `iterate()` execute at the *next bar's open*, not the close that triggered the signal. This is already enforced by the engine. Never bypass this by using same-bar prices.

**Price adjustment** â€” Use `CAPITALSPECIAL` adjustment (not `TOTALRETURN`) for individual stocks to avoid forward-looking dividend bias. Use `TOTALRETURN` only for benchmark indices.

**Statistical honesty** â€” Report metrics on the full out-of-sample period. Do not cherry-pick start/end dates or exclude drawdown periods. Sharpe ratio is computed with 0 risk-free rate (clearly documented).

**Simplicity over complexity** â€” A strategy with 3 rules that works is better than one with 10. When adding logic, ask whether it is genuinely necessary or whether it is curve-fitting noise.

---

## Commands

This project uses `uv` for dependency management (Python 3.12).

```bash
# Install dependencies
uv sync

# Run a strategy script
uv run python strategies/strategy_mr_dv2.py

# Launch Jupyter notebooks
uv run jupyter notebook
```

## Architecture

This is a custom event-driven backtesting framework for quantitative trading strategies.

### Core Engine (`alpha/engine/`)

The engine follows a strict lifecycle to prevent lookahead bias:

1. **`strategy.py`** â€” Abstract base class `Strategy`. All strategies inherit from it and must implement:
   - `compute_signals(pricing_data)` â€” Called once before the backtest; precomputes all signals on the full dataset.
   - `iterate(data, close, open_prices)` â€” Called each trading day at market open with data restricted to the previous bar. Place orders here.
   - Optionally override `finalize(current_data)` for post-simulation tasks.

2. **`backtest.py`** â€” `run_daily(strategy, pricing_data, calendar)` drives the simulation. Per bar it calls: `restrict_data()` â†’ `iterate()` â†’ `process_orders()` â†’ `update_metrics()`. Orders placed in `iterate()` execute at the **next bar's open** (next-open execution model).

3. **`order.py`** â€” Order types: `MarketOrder`, `LimitOrder`, `StopOrder`, `StopLimitOrder`. Orders specify an `amount` in `'shares'`, `'value'`, or `'percent'`. Setting `target=True` makes the amount a target position rather than a delta.

4. **`metrics.py`** â€” Post-run analytics: `generate_trades()`, `generate_drawdowns()`, `generate_overall_metrics()`, `generate_trades_metrics()`, `generate_monthly_returns()`, `sharpe_ratio()`. Called automatically by `strategy.summarize()`.

5. **`indicators.py`** â€” Custom technical indicators: `dv2_indicator()` (Varadi Oscillator) and `qp_indicator()` (quantile probability indicator).

6. **`plot.py`** â€” `plot()` renders a three-panel chart: cumulative returns (log scale), drawdown, and annual return bars.

### Pricing Data Format

`pricing_data` must be a `pd.DataFrame` with:
- **Index**: `pd.DatetimeIndex` of trading dates.
- **Columns**: `pd.MultiIndex` where level 0 is the ticker symbol and level 1 is the price field. Every symbol must include at minimum `Open`, `High`, `Low`, `Close`.

### Data Loading (`data/data_loader.py`)

- `YahooDataLoader` â€” wraps `yfinance` (free, public data).
- `StooqDataLoader` â€” wraps `pandas_datareader` (free, public data).
- Norgate Data (`norgatedata`) is used in strategies for survivorship-bias-free S&P 500 constituent history; requires a paid Norgate subscription.

### Strategies (`strategies/`)

Concrete `Strategy` subclasses. The DV2 mean-reversion strategy (`strategy_mr_dv2.py`) is the primary reference implementation â€” it shows the full pattern: building a survivorship-bias-free universe via `build_index_constituent_matrix()`, loading prices with pre-computed features, and running `run_daily()`.

### Key `Strategy` Methods for Use Inside `iterate()`

| Method | Description |
|---|---|
| `order(asset, amount)` | Market/Limit/Stop order in shares |
| `order_value(asset, value)` | Order by dollar value |
| `order_percent(asset, percent)` | Order as % of portfolio |
| `order_target_value(asset, target)` | Adjust to target dollar allocation |
| `order_target_percent(asset, target)` | Adjust to target % allocation |
| `get_position(asset)` | Net shares held for an asset |
| `get_positions()` | All positions as a Series |
| `clear_orders(asset=None)` | Cancel pending orders |
| `previous_total_value` | Portfolio value at end of previous bar |

### Multi-Strategy Portfolios (`alpha/engine/portfolio.py`)

The `Portfolio` class combines multiple completed strategy runs ("pods") into a unified portfolio. This models how a real IBKR multi-pod account works: each pod receives a capital allocation and compounds independently.

**Pod model** â€” Each strategy is a self-contained pod. Pods run independently through `run_daily()` with their own capital, universe, and logic. The `Portfolio` aggregator is read-only: it takes completed pod results and reconstructs a combined equity curve over their common date range.

**Buy-and-hold math (default)** â€” Each pod gets `capital * weight` and compounds its own daily returns independently. Portfolio equity = sum of pod equities. Weights drift with performance, matching real-world behavior where you don't rebalance between strategies daily.

**Periodic rebalancing** â€” Optional `rebalance` parameter (`'monthly'`, `'quarterly'`, `'annually'`). At each rebalance date, the total portfolio value is redistributed across pods at target weights, then each pod compounds forward independently until the next rebalance. Rebalance dates snap to actual trading days.

**Cross-strategy diagnostics:**
- **Correlation matrix** â€” Pairwise correlation of pod daily returns. Low correlation between pods is the primary source of portfolio-level risk reduction.
- **Diversification ratio** â€” `weighted_sum_vol / portfolio_vol`. Ratio > 1.0 means diversification benefit exists; ratio = 1.0 means perfect correlation (no benefit).

**Quantitative correctness notes:**
- Never use `(daily_rets * weights).sum(axis=1)` for portfolio returns â€” this is daily-rebalanced math that doesn't match real multi-pod behavior.
- Pod equity curves must compound independently. The portfolio equity is the *sum* of pod equities, not a weighted-return series.
- When adding rebalancing, redistribute total portfolio value at target weights, then compound forward. Don't just reset weights on the return series.

**Running a portfolio:**
```bash
uv run python strategies/run_portfolio.py portfolios/multipod.yaml
uv run python strategies/run_portfolio.py portfolios/multipod.yaml --rebalance quarterly
```

Portfolio YAML config:
```yaml
name: MyPortfolio
capital: 100000
rebalance: quarterly  # optional: monthly, quarterly, annually, or omit for buy-and-hold
pods:
  - strategy: StrategyA
    weight: 0.5
  - strategy: StrategyB
    weight: 0.5
```

### Persistence

Completed strategy runs can be saved/loaded with `strategy.to_pickle(path)` / `Strategy.read_pickle(path)`. Portfolios can be saved/loaded with `portfolio.to_pickle(path)` / `Portfolio.read_pickle(path)`.


