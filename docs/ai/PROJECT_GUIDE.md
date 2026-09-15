# Project Guide

Use this reference for the sections relevant to the current task. [AGENTS.md](../../AGENTS.md) defines the shared working agreements; [Quant Philosophy](../../QUANT_PHILOSOPHY.md) is the authority for quantitative rules. [Assumptions and Gaps](../../ASSUMPTIONS_AND_GAPS.md) records known limitations.

The verification policy below applies to all coding agents. The commands and architecture sections retain the technical reference previously kept in CLAUDE.md. Paths in the technical reference are relative to the repository root. Run commands from that root and within the owner's approved scope.

## Review scope

Classify the actual affected behavior as well as the changed paths. For a worktree with unrelated edits, pass the complete task-owned path set to the triage helper using `--name-only`, including new files. Preserve other work.

The primary agent assigns bounded implementation tasks and resolves review findings. Required reviewers below inspect the change independently of its implementation; they do not edit files unless explicitly authorized. Their role names describe review responsibilities, not proof of specialist competence.

## Post-Change Verification

After every meaningful code change, classify the diff by blast radius before
the final response. The user should not need to run this manually; agents must
use the local helper and then apply the matching review depth.

Run:

```bash
uv run python scripts/review/triage.py
```

By default the helper reads `git diff --name-only` and also includes untracked
non-ignored files, so brand-new scripts and tests are not missed.

To classify a proposed or explicit path set:

```bash
uv run python scripts/review/triage.py --name-only alpha/live/runner.py
uv run python scripts/review/triage.py --base HEAD
```

Tests are the hard gate. Agents are a soft review gate. If a review agent finds
a real issue that tests missed, prefer adding or tightening a regression test.
The main agent patches; review agents are read-only unless the user explicitly
requests otherwise.

### Tiers

**Tier 0 - docs, comments, isolated tooling**

- Scope: docs, comments, presentation artifacts, isolated tooling.
- Required verification: relevant tests/checks only.
- Required agents: none.
- Escalate live runbooks, operator docs, or docs that change live behavior to
  Tier 3.

**Tier 1 - research and backtest-only work**

- Scope: `strategies/**`, notebooks, research scripts, backtest-only
  experiments.
- Required verification: tests plus one quant-pitfalls agent.

**Tier 2 - engine, shared utilities, indicators, metrics**

- Scope: `alpha/engine/**`, shared data utilities, indicators, metrics,
  portfolio utilities, and execution-sensitive shared helpers.
- Required verification: tests plus quant-pitfalls, parity, and coverage
  agents.

**Tier 3 - live execution, orders, sizing, reconcile, released configs**

- Scope: `alpha/live/**`, `alpha/live/releases/**`, live runner/scheduler,
  order, reconcile, reference-price, sizing, dashboard/logging consumed by
  live, and released pod YAML/config/state contract changes.
- Required verification: tests, mandatory live-impact checklist, and parity,
  failure-modes, and coverage agents.
- Add a quant-pitfalls agent too when the live change also touches strategy,
  backtest, sizing math, reference-price semantics, or quantitative behavior.

If multiple tiers match, choose the highest tier and say which lower-tier
surfaces were also touched.

### Quant-Pitfalls Agent

This is a full quant review, not just a lookahead check. It must explicitly
check lookahead, survivorship, data mining, multiple comparisons, in-sample
contamination, target leakage, regime dependence, sample size, corporate
actions, adjustment type, cost/slippage realism, and live/backtest divergence.

### Live-Impact Checklist

For Tier 3 changes, explicitly answer:

- Order timing semantics unchanged, especially next-open execution.
- Sizing math unchanged, including `amount`, `target=True`, percent versus
  value semantics.
- Reference price source unchanged, with no silent close/open substitution.
- State files, pickle files, SQLite schemas, and config formats backward
  compatible unless intentionally migrated.
- Logging fields consumed by dashboards or runbooks still present.
- No new Windows VPS failure mode around paths, encodings, file locks,
  idempotency, or process restarts.
- Released pod YAMLs still parse and produce the same intended route/intent
  unless the change explicitly targets those semantics.

### Required Final Response Fields

After code changes, the final response must state:

- tier
- agents used
- findings fixed
- tests run
- residual risk

## Commands

This project uses `uv` for dependency management (Python 3.12).

```bash
# Install dependencies
uv sync

# Run a strategy script
uv run python -m strategies.dv2.strategy_mr_dv2

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

2. **`backtest.py` / `backtester.py`** — `run_daily(strategy, pricing_data, calendar)` delegates to `VanillaBacktester`. Per bar it calls: `restrict_data()` → `iterate()` → `process_orders()` → `update_metrics()`. For session T+1, `iterate()` sees data only through Close_T; market orders then fill at Open_(T+1), the current engine bar's open.

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
