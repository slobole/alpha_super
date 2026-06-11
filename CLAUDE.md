# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read First

Before changing code, read these doctrine documents in this exact order:

1. `QUANT_PHILOSOPHY.md`
2. `ASSUMPTIONS_AND_GAPS.md`

These documents are authoritative for house philosophy and realism assumptions. This file remains the operational entrypoint for repo-specific coding behavior.

After the doctrine documents above, also read `docs/ai/KARPATHY_GUIDELINES.md`. This project keeps a self-contained local adaptation of `forrestchang/andrej-karpathy-skills` so the guidance is available here without requiring external plugin files, `.cursor` rules, or `.claude-plugin` metadata.

For Norgate snapshot/API/client-VPS work, also read `docs/live/NORGATE_SNAPSHOT_V1.md` before changing live data-source, scheduler, or dashboard behavior.

## Karpathy-Derived Engineering Guardrails

These guardrails are additive to the quant doctrine:

- **Think before coding** - state assumptions explicitly, surface ambiguity, and do not silently choose semantics when they matter.
- **Simplicity first** - prefer the minimum code that solves the actual problem; avoid speculative flexibility and unnecessary abstractions.
- **Surgical changes** - change only what the task requires and avoid drive-by cleanup in unrelated areas.
- **Goal-driven execution** - define a verifiable success condition, implement against it, and verify before declaring completion.

Use the local reference file for the full text and attribution details.

## Quantitative Correctness Standards

This codebase is held to a strict standard of quantitative rigor. Every piece of code — strategies, indicators, metrics, data handling — must be bullet-proof against common quant pitfalls. **Simplicity is a virtue**: prefer the clearest, most direct implementation over clever abstractions.

The simulator is live-first. Prefer logic and assumptions that could be traded operationally at the intended capital scale, and record any unrealistic simplification in `ASSUMPTIONS_AND_GAPS.md`.

Backtests must aim to be as credible, conservative, and robust as the current data and engine allow. Optimize for auditability and realism, not cosmetic performance.

Live trading must preserve backtest semantics up to irreducible market frictions. If the live implementation changes strategy meaning, then it is not the same strategy.

For live deployment, prefer a deterministic per-pod order-clerk model: the strategy creates explicit order intent from prior-available information, and that pod's execution layer transmits, tracks, reconciles, and reports that intent without adding opaque intelligence.

Any dangerous operation, realism gap, hidden assumption, operational ambiguity, or potentially misleading simplification must fail loud: raise the flag, document it, and discuss its likely impact instead of leaving it implicit.

### Non-negotiable rules

**Lookahead bias** — The most critical failure mode. Signals, features, and any derived data must only use information available *before* the decision point. `compute_signals()` runs on the full dataset deliberately — any feature computed there must use only past data (e.g. rolling windows, `shift()`). `iterate()` receives only data up to `previous_bar`; never index into future bars.

**Survivorship bias** — Universe construction must use point-in-time constituent membership (Norgate's `index_constituent_timeseries`), not today's index composition. Never backtest on a static list of current constituents.

**Data-mining / overfitting** — Strategies should be based on a clear, explainable edge with few parameters. Do not add parameters to fit historical results. Do not optimize parameters on the full backtest period without out-of-sample validation.

**Execution realism** — Orders placed in `iterate()` execute at the *next bar's open*, not the close that triggered the signal. This is already enforced by the engine. Never bypass this by using same-bar prices.

**Price adjustment** — Use `CAPITALSPECIAL` adjustment (not `TOTALRETURN`) for individual stocks to avoid forward-looking dividend bias. Use `TOTALRETURN` only for benchmark indices.

**Statistical honesty** — Report metrics on the full out-of-sample period. Do not cherry-pick start/end dates or exclude drawdown periods. Sharpe ratio is computed with 0 risk-free rate (clearly documented).

**Simplicity over complexity** — A strategy with 3 rules that works is better than one with 10. When adding logic, ask whether it is genuinely necessary or whether it is curve-fitting noise.

**Explicit semantics** — If you change signal timing, order timing, execution assumptions, rebalance mapping, portfolio aggregation math, or cost modeling, state the old behavior, the new behavior, and the quantitative consequence.

**Live pod-account mapping** — For live trading, a pod means one independent strategy sleeve: one live pod = one strategy = one linked IBKR account/subaccount route = one ledger. By default, that sleeve must map to a real isolated broker account, subaccount, or broker-recognized sleeve with its own cash, positions, and account value. Do not assign two different live strategies to the same account route. Do not treat a pod as a soft label inside one shared raw broker account unless a first-class pod ledger exists.

**Domain naming** — Use strict `Domain_Type` naming in quantitative logic, for example `price_vec`, `return_ser`, `signal_df`, `target_weight_ser`.

**Sensitive time-series auditability** — Add a `*** CRITICAL***` comment next to sensitive time-series operations such as `shift()`, rolling windows, forward returns, and rebalance-date mapping.

**Human-readable explanations** — Keep the logic rigorous, but explain it in simple, precise language. Default to: plain-language intuition, then the exact rule, and use formulas only when they materially improve correctness, auditability, or remove ambiguity.

**Uncertainty handling** — When uncertain, do not silently choose the prettier implementation. Choose the more explicit implementation and state the uncertainty.

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

# Launch Bench — the local research control panel (http://127.0.0.1:8765)
uv run python -m alpha.bench
```

### Bench (research control panel)

`alpha/bench/` is a local, single-operator web UI that centralizes the research
loop: it lists every strategy (flagging the WIRED/live ones), surfaces recent
analyzer runs from `results/`, and exposes one-click buttons that launch the
existing `run_strategy_analysis.py` / `run_portfolio.py` /
`run_portfolio_manager.py` scripts as tracked background jobs. It adds **no quant
logic** — it only discovers, reads artifacts, and shells out — so it preserves
backtest semantics by construction. See `alpha/bench/README.md`.

## Architecture

This is a custom event-driven backtesting framework for quantitative trading strategies.

### Core Engine (`alpha/engine/`)

The engine follows a strict lifecycle to prevent lookahead bias:

1. **`strategy.py`** — Abstract base class `Strategy`. All strategies inherit from it and must implement:
   - `compute_signals(pricing_data)` — Called once before the backtest; precomputes all signals on the full dataset.
   - `iterate(data, close, open_prices)` — Called each trading day at market open with data restricted to the previous bar. Place orders here.
   - Optionally override `finalize(current_data)` for post-simulation tasks.

2. **`backtest.py`** — `run_daily(strategy, pricing_data, calendar)` drives the simulation. Per bar it calls: `restrict_data()` → `iterate()` → `process_orders()` → `update_metrics()`. Orders placed in `iterate()` execute at the **next bar's open** (next-open execution model).

3. **`order.py`** — Order types: `MarketOrder`, `LimitOrder`, `StopOrder`, `StopLimitOrder`. Orders specify an `amount` in `'shares'`, `'value'`, or `'percent'`. Setting `target=True` makes the amount a target position rather than a delta.

4. **`metrics.py`** — Post-run analytics: `generate_trades()`, `generate_drawdowns()`, `generate_overall_metrics()`, `generate_trades_metrics()`, `generate_monthly_returns()`, `sharpe_ratio()`. Called automatically by `strategy.summarize()`.

5. **`indicators.py`** — Custom technical indicators: `dv2_indicator()` (Varadi Oscillator) and `qp_indicator()` (quantile probability indicator).

6. **`plot.py`** — `plot()` renders a three-panel chart: cumulative returns (log scale), drawdown, and annual return bars.

### Pricing Data Format

`pricing_data` must be a `pd.DataFrame` with:
- **Index**: `pd.DatetimeIndex` of trading dates.
- **Columns**: `pd.MultiIndex` where level 0 is the ticker symbol and level 1 is the price field. Every symbol must include at minimum `Open`, `High`, `Low`, `Close`.

### Data Loading

- Norgate Data (`data/norgate_loader.py`) is the production source — survivorship-bias-free constituent history and point-in-time universes; requires a paid Norgate subscription. It reads either the local `norgatedata` install directly or pre-exported snapshots (`ALPHA_USE_NORGATE_SNAPSHOT_BOOL`, see `docs/live/NORGATE_SNAPSHOT_V1.md`).
- `alpha/data/` holds auxiliary loaders (FRED macro series, Kenneth French factors).

### Strategies (`strategies/`)

Concrete `Strategy` subclasses. The DV2 mean-reversion strategy (`strategy_mr_dv2.py`) is the primary reference implementation — it shows the full pattern: building a survivorship-bias-free universe via `build_index_constituent_matrix()`, loading prices with pre-computed features, and running `run_daily()`.

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

**Pod model** — Each strategy is a self-contained pod. Pods run independently through `run_daily()` with their own capital, universe, and logic. The `Portfolio` aggregator is read-only: it takes completed pod results and reconstructs a combined equity curve over their common date range.

**Live pod-account invariant** — In live deployment, the intended production model is one live pod = one strategy = one linked IBKR account/subaccount route = one ledger. If multiple pods share one raw broker account, the system needs an explicit pod ledger before overlapping symbols or pod-level reconciliation can be trusted.

**Buy-and-hold math (default)** — Each pod gets `capital * weight` and compounds its own daily returns independently. Portfolio equity = sum of pod equities. Weights drift with performance, matching real-world behavior where you don't rebalance between strategies daily.

**Periodic rebalancing** — Optional `rebalance` parameter (`'monthly'`, `'quarterly'`, `'annually'`). At each rebalance date, the total portfolio value is redistributed across pods at target weights, then each pod compounds forward independently until the next rebalance. Rebalance dates snap to actual trading days.

**Cross-strategy diagnostics:**
- **Correlation matrix** — Pairwise correlation of pod daily returns. Low correlation between pods is the primary source of portfolio-level risk reduction.
- **Diversification ratio** — `weighted_sum_vol / portfolio_vol`. Ratio > 1.0 means diversification benefit exists; ratio = 1.0 means perfect correlation (no benefit).

**Quantitative correctness notes:**
- Never use `(daily_rets * weights).sum(axis=1)` for portfolio returns — this is daily-rebalanced math that doesn't match real multi-pod behavior.
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


