# Incubation Flow

TL;DR: Incubation is a brokerless promotion gate. It proves that a strategy can run as a real daily process through the live stack before we trust IBKR paper, a tiny live account, or scale.

## Purpose

Incubation sits between research and broker testing:

```text
research backtest -> brokerless incubation -> IBKR paper smoke test -> tiny live account -> scale
```

It answers one operational question:

```text
Can the strategy run every day through StrategyHost -> DecisionPlan -> VPlan -> submit -> reconcile -> PodState?
```

It does not prove alpha, real broker fills, queue behavior, partial fills, or IBKR reliability.

## Model

The strategy still creates deterministic order intent:

```text
order_intent_t = g(information_available_t, pod_state_t, config)
```

The incubation broker settles accepted orders with explicit, auditable prices:

```text
cash_t = cash_{t-1} - sum(quantity_i,t * fill_price_i,t) - commissions_t
position_i,t = position_i,t-1 + quantity_i,t
equity_t = cash_t + sum(position_i,t * mark_price_i,t)
```

Commission uses the same simple IBKR-style rule as the research engine unless overridden later:

```text
commission = max(commission_minimum, commission_per_share * abs(quantity))
```

## Price Semantics

For open-execution policies:

```text
sizing_reference_price = official prior signal-session Close
fill_price = IBKR ticker.open from the target session
```

This applies to:

```text
next_open_moo
next_open_market
next_month_first_open
```

For close-execution:

```text
sizing_reference_price = causal pre-close snapshot only
fill_price = official same-session Close
```

`same_day_moc` must fail loud until a real pre-close snapshot source exists. Official close must not be used as the sizing reference because that would leak the final auction price into the decision.

Open fills are still virtual broker fills. IBKR is only a price source in incubation, using a separate price-read client ID by default.

## Account Model

Incubation uses virtual account routes:

```text
SIM_<pod_id_str>
```

`SIM_` routes are valid only in `mode: incubation`. IBKR-style `DU...` and `U...` routes are valid only for paper/live.

## Carlos-Inspired Split

Carlos described two useful concepts:

- Clearinghouse: scheduling, settlement, fills, positions, cash, and accounting.
- Ledger: strategy-facing state and audit history.

This repo keeps that split small:

- `alpha.live.runner` remains the scheduler/orchestrator.
- `alpha.live.incubation.IncubationBrokerAdapter` behaves like a virtual broker.
- SQLite stores orders, fills, cash ledger rows, pod state, and reports.

No separate platform is introduced in this phase.

## Commands

The command shape is intentionally the same as paper/live:

```bash
uv run python -m alpha.live.runner status --mode incubation
uv run python -m alpha.live.runner tick --mode incubation
uv run python -m alpha.live.runner show_vplan --mode incubation
uv run python -m alpha.live.runner submit_vplan --mode incubation --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode incubation
uv run python -m alpha.live.runner execution_report --mode incubation
uv run python -m alpha.live.runner compare_reference --mode incubation --pod-id pod_x
```

## Comparison

The default comparison is against the VPlan:

```text
quantity_diff = filled_quantity - planned_order_delta
position_diff = actual_position - target_position
fill_slippage_bps = side * (fill_price - reference_price) / reference_price * 10000
```

where:

```text
side = +1 for buy, -1 for sell
```

Optional backtest comparison can use a saved `Strategy` pickle. That comparison is diagnostic only because a backtest and a live process can differ for legitimate reasons such as skipped cycles, calendar timing, commission, and real fill behavior.

## Known Gaps

- This phase supports current daily/monthly open-execution pods first.
- `same_day_moc` is blocked unless a causal pre-close snapshot provider is configured.
- Prior-close sizing and EOD close marks come from Norgate `CAPITALSPECIAL`; same-day open settlement comes from IBKR `ticker.open`.
- No PostgreSQL, Docker, OpenFIGI, QuestDB, futures, CloudWatch, partial fills, or broker microstructure simulation.
- Corporate actions beyond split-adjusted Norgate prices remain a documented gap. Missing held-symbol prices must block or warn loudly rather than silently changing positions.
