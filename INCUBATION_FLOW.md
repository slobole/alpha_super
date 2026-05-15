# Unified Incubation / Rehearsal Flow

TL;DR: Incubation is the unified client rehearsal gate. The SIM ledger is the official rehearsal accounting truth, while IBKR is used only for price/reference plumbing inside incubation. IBKR paper remains a separate probe-only broker check; paper fills are not incubation P&L.

## Purpose

The operating ladder is:

```text
research/backtest -> incubation rehearsal -> paper probe -> small live account -> bigger account
```

It answers one operational question:

```text
Can this client stack run the strategy through StrategyHost -> DecisionPlan -> VPlan -> SIM submit -> IBKR price read -> SIM settlement -> reconcile -> PodState?
```

It does not prove alpha, real broker fills, queue behavior, partial fills, broker rejects, auction imbalance handling, or IBKR execution quality.

Paper answers a different question:

```text
Can this VPS / IBC / TWS / IBKR account talk to the broker API and observe broker responses?
```

That paper evidence is useful, but it is not the source of rehearsal positions, cash, or P&L.

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
sizing_reference_price = IBKR pre-submit reference/open-price path
fill_price = IBKR ticker.open from the target session
```

This applies to:

```text
next_open_moo
next_open_market
next_month_first_open
```

For MOO-style policies, incubation uses the same reference-price path as paper/live:

```text
auctionPrice (generic tick 225)
-> reqMktData.marketPrice fallback
-> reqTickers.marketPrice fallback
```

This means incubation target shares can differ from the research/backtest target shares when the IBKR reference price differs from the prior close. That is intentional rehearsal evidence. The SIM ledger still settles at the target-session IBKR open tick; paper fills remain outside official rehearsal P&L.

For close-execution:

```text
sizing_reference_price = causal pre-close snapshot only
fill_price = official same-session Close
```

`same_day_moc` must fail loud until a real pre-close snapshot source exists. Official close must not be used as the sizing reference because that would leak the final auction price into the decision.

Open fills are still SIM ledger fills. IBKR is only a price/reference source in incubation, using a separate price-read client ID by default. Paper fills must not be imported into the SIM ledger.

## Account Model

Incubation uses one virtual account route per pod:

```text
SIM_<pod_id_str>
```

`SIM_` routes are valid only in `mode: incubation`. IBKR-style `DU...` and `U...` routes are valid only for paper/live.

This keeps multi-strategy rehearsal clean even when paper accounts would blend positions at the account level.

Incubation state is also stored per pod by default:

```text
alpha/live/state/incubation/<pod_id_str>.sqlite3
```

The dashboard is the client-level aggregation layer. A runner/scheduler command with `--pod-id pod_x` reads or mutates only that pod DB. A command with `--mode incubation` and no `--pod-id` fans out across all enabled incubation pods and returns an aggregate result. Explicit `--db-path` is reserved for legacy/manual inspection, including old shared `alpha/live/incubation_state.sqlite3` state.

## Carlos-Inspired Split

Carlos described two useful concepts:

- Clearinghouse: scheduling, settlement, fills, positions, cash, and accounting.
- Ledger: strategy-facing state and audit history.

This repo keeps that split small:

- `alpha.live.runner` remains the scheduler/orchestrator.
- `alpha.live.incubation.IncubationBrokerAdapter` behaves like a virtual broker.
- SQLite stores orders, fills, cash ledger rows, pod state, and reports.

No separate platform is introduced in this phase. This is "Clearinghouse-Lite" inside the existing live stack.

## Paper Probe Boundary

Paper is probe-only in this model.

Good paper evidence:

- IBC/TWS connection works.
- Account visibility works.
- Contract qualification works.
- Market data permissions exist.
- Optional test orders are accepted or rejected with observable broker messages.

Not paper truth:

- SIM ledger cash.
- SIM ledger positions.
- Rehearsal P&L.
- Promotion performance.

If a paper test order is run, label it as `paper_probe` evidence. Do not call it an incubation fill.

## Promotion Gate

Promotion is cycle-based, not calendar-only.

For monthly MOO pods, one useful rehearsal cycle is:

```text
month-end signal -> VPlan -> SIM submit -> IBKR open price read -> SIM settlement -> reconcile/report
```

For daily MOO pods, reports should count completed rehearsal cycles. One successful day is evidence, not proof that the sleeve is ready to scale.

## Commands

The command shape is intentionally the same as paper/live:

```bash
uv run python -m alpha.live.runner status --mode incubation
uv run python -m alpha.live.runner tick --mode incubation
uv run python -m alpha.live.runner show_vplan --mode incubation
uv run python -m alpha.live.runner submit_vplan --mode incubation --pod-id pod_x --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode incubation
uv run python -m alpha.live.runner execution_report --mode incubation
uv run python -m alpha.live.runner compare_reference --mode incubation --pod-id pod_x
```

Use no `--pod-id` when you want the normal all-pod incubation fan-out. Use `--pod-id` for a single pod, especially any command that names a local `decision_plan_id` or `vplan_id`.

## Comparison

The default comparison is against the VPlan:

```text
quantity_diff = filled_quantity - planned_order_delta
position_diff = actual_position - target_position
```

Execution reports separate two price gaps:

```text
open_slippage_bps = side * (fill_price - official_open_price) / official_open_price * 10000
reference_slippage_bps = side * (fill_price - vplan_reference_price) / vplan_reference_price * 10000
```

where `side = +1` for buys and `side = -1` for sells.

Optional backtest comparison can use a saved `Strategy` pickle. That comparison is diagnostic only because a backtest and a live process can differ for legitimate reasons such as skipped cycles, calendar timing, commission, and real fill behavior.

## Operator Report Meaning

Read-only status and dashboard surfaces should separate these facts:

- SIM ledger status: the official rehearsal state.
- Last cycle: latest DecisionPlan, VPlan, submit, fill, and reconcile state.
- IBKR price probe: reference/open-price source evidence used by the SIM ledger.
- Paper probe: optional broker connectivity/order API evidence, never SIM P&L.
- Promotion gate: incomplete until the natural strategy cycle has settled and reconciled.

## Known Gaps

- This phase supports current daily/monthly open-execution pods first.
- `same_day_moc` is blocked unless a causal pre-close snapshot provider is configured.
- Pre-open account marks and EOD close marks come from Norgate `CAPITALSPECIAL`; MOO sizing references and same-day open settlement come from IBKR evidence.
- Paper probe evidence is not yet stored as a first-class table; until then, it remains separate operator evidence.
- No PostgreSQL, Docker, OpenFIGI, QuestDB, futures, CloudWatch, partial fills, or broker microstructure simulation.
- Corporate actions beyond split-adjusted Norgate prices remain a documented gap. Missing held-symbol prices must block or warn loudly rather than silently changing positions.
