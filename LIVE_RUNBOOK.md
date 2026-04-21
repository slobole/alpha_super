# Live Runbook

TL;DR: in live v2 you do **not** freeze final share quantities after the close. You freeze a `DecisionPlan` after the approved snapshot is ready, then you build a pre-submit `VPlan` from **broker truth** near the execution window.

If you want the shortest operator version first, read:
- [LIVE_START_HERE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_START_HERE.md)

If you want the implementation reference, read:
- [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md)

The core contract is:

```text
DecisionPlan_t = f(approved_snapshot_t, strategy_memory_t)
VPlan_submit = f(DecisionPlan_t, BrokerSnapshot_submit, live_quote_snapshot)
```

That means:

1. the strategy decides **what it wants**
2. the execution layer decides **how many shares to send**
3. final share sizing uses live `NetLiq`, live broker positions, and live quote prices

## What This Is

This file is the simple operator guide for the live layer under `alpha/live/`.

Use it for:
- where the YAML files live
- what each config section means
- what commands to run
- how to inspect the live `VPlan`

Do not use this file as the canonical code-level reference.
That role now belongs to:
- [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md)

The research strategies stay in:
- `strategies/`

The live layer only hosts them.

## Main Objects

### `DecisionPlan`

Built after the approved signal snapshot is ready.

Contains:
- an explicit decision-book type
- either:
  - incremental entry weights + exits + entry priority
  - or a full target-weight book
- strategy memory
- signal / submit / target execution timestamps

It does **not** contain final overnight share quantities.

Current strategy-family mapping:
- `incremental_entry_exit_book`
  - `strategies.dv2.strategy_mr_dv2:DVO2Strategy`
  - `strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy`
- `full_target_weight_book`
  - `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash`
  - `strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy`

### `BrokerSnapshot`

Read near the submit window from the broker.

Contains:
- current broker positions
- cash
- `NetLiq`
- `AvailableFunds`
- `ExcessLiquidity`

### `VPlan`

Built near the submit window from:
- `DecisionPlan`
- `BrokerSnapshot`
- live quote snapshot

Contains:
- current shares
- target shares
- delta shares
- live reference price
- estimated target notional

This is the execution artifact for both:
- manual review
- automatic submit

## Core Formula

Each pod uses a fixed fraction of broker `NetLiq`:

```text
PodBudget = NetLiq_broker * pod_budget_fraction
TargetDollar_i = target_weight_i * PodBudget
TargetShares_i = floor(TargetDollar_i / LivePrice_i)
OrderDelta_i = TargetShares_i - BrokerShares_i
```

That is the sizing logic used inside the `VPlan`.

## High-Level Flow

The simple live flow is:

1. put one YAML file per pod under `alpha/live/releases/`
2. run the generic live runner
3. the runner waits until the snapshot is ready
4. the runner builds a `DecisionPlan`
5. near submit time the runner reads broker truth
6. the runner builds a `VPlan`
7. you either review the `VPlan` manually or auto-submit it
8. fills are stored and broker-backed pod state is updated

## Where The Config Files Live

Current examples:
- [pod_dv2_01.yaml](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases/excelence_trade_paper_001/pod_dv2_01.yaml)
- [pod_qpi_01.yaml](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases/excelence_trade_paper_001/pod_qpi_01.yaml)
- [pod_taa_01.yaml](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases/excelence_trade_paper_001/pod_taa_01.yaml)
- [pod_ndx_mo_01.yaml](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases/excelence_trade_paper_001/pod_ndx_mo_01.yaml)

General location:

```text
alpha/live/releases/<user_id>/*.yaml
```

## Recommended Main Command

This is the main command:

```bash
uv run python -m alpha.live.runner tick --mode paper
```

Later:

```bash
uv run python -m alpha.live.runner tick --mode live
```

`tick` is the main loop.

It does:
1. load enabled manifests
2. check market session timing
3. check if Norgate data is ready
4. build `DecisionPlan` objects if due
5. expire missed windows
6. build `VPlan` objects from broker truth if due
7. auto-submit ready `VPlan` objects only when `auto_submit_enabled_bool = true`
8. reconcile submitted fills

Recommended scheduler setup:
- Windows Task Scheduler
- run `tick` every minute

The scheduler stays dumb.
The Python runner decides whether anything is actually due.

## Optional Scheduler Service

If you want a portable long-running process instead of an every-minute shell loop, use:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper
```

Important rule:

```text
tick = atomic execution primitive
scheduler_service = timing wrapper around tick
```

So:
- manual `tick` still works
- Task Scheduler / shell loops still work
- the daemon is optional

Useful scheduler commands:

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper
uv run python -m alpha.live.scheduler_service run_once --mode paper
```

The scheduler service is UTC-native.
It does not depend on the host machine local timezone.

## Full CLI Reference

There are 2 public CLIs:
- `alpha.live.runner`
- `alpha.live.scheduler_service`

The relationship is:

```text
tick = atomic execution primitive
scheduler_service = timing wrapper around tick
```

### Runner commands

Base form:

```bash
uv run python -m alpha.live.runner <command> [flags...]
```

Available commands:
- `tick`
  - one full live pass
  - may build `DecisionPlan`
  - may expire stale plans
  - may build `VPlan`
  - may auto-submit
  - may reconcile fills
- `build_decision_plans`
  - builds only overnight `DecisionPlan` objects
- `build_vplan`
  - builds only pre-submit `VPlan` objects from broker truth
- `show_vplan`
  - prints the latest `VPlan` summary or a selected `VPlan`
- `submit_vplan`
  - manually submits one ready `VPlan`
- `post_execution_reconcile`
  - reads broker truth, updates fills/orders, and only completes a `VPlan` if broker positions match target shares
- `status`
  - prints current pod-level live status with unresolved exceptions first
- `execution_report`
  - prints fill-level execution details including official open and slippage when available

### Runner shared flags

These flags are accepted by every runner command:

- `--db-path`
  - SQLite path
  - default: `alpha/live/live_state.sqlite3`
- `--releases-root`
  - release manifest root
  - default: `alpha/live/releases`
- `--as-of-ts`
  - override current time with an ISO 8601 timestamp
  - useful for replay and debugging
- `--mode`
  - release mode filter
  - default: `paper`
- `--log-path`
  - JSONL event log path
- `--json`
  - print raw machine-readable JSON instead of text
- `--broker-host`
  - broker API host
  - default: `127.0.0.1`
- `--broker-port`
  - broker API port
  - default: `7497`
- `--broker-client-id`
  - broker API client id
  - default: `31`
- `--broker-timeout-seconds`
  - broker request timeout
  - default: `4.0`

### Runner command-specific flags

- `show_vplan`
  - `--vplan-id`
    - show one specific `VPlan`
  - `--pod-id`
    - restrict output to one pod
- `submit_vplan`
  - `--vplan-id`
    - submit one specific ready `VPlan`

### Runner examples

Current status:

```bash
uv run python -m alpha.live.runner status --mode paper
```

Current status as JSON:

```bash
uv run python -m alpha.live.runner status --mode paper --json
```

Overnight decision build only:

```bash
uv run python -m alpha.live.runner build_decision_plans --mode paper
```

Build `VPlan` objects against paper TWS:

```bash
uv run python -m alpha.live.runner build_vplan --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

Show the latest `VPlan` for one pod:

```bash
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_dv2_01
```

Show one exact `VPlan`:

```bash
uv run python -m alpha.live.runner show_vplan --mode paper --vplan-id 1
```

Submit one exact `VPlan` manually:

```bash
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id 1
```

Run one full live pass:

```bash
uv run python -m alpha.live.runner tick --mode paper
```

Replay one fixed timestamp:

```bash
uv run python -m alpha.live.runner tick --mode paper --as-of-ts 2026-04-10T13:20:00+00:00
```

### Scheduler commands

Base form:

```bash
uv run python -m alpha.live.scheduler_service <command> [flags...]
```

Available commands:
- `serve`
  - long-running UTC-native daemon
  - decides when to call `tick`
- `next_due`
  - inspect-only
  - prints the next scheduler phase and next wake-up
- `run_once`
  - one scheduler-aware pass
  - if work is due, calls `tick` once

### Scheduler shared flags

These flags are accepted by every scheduler command:

- `--db-path`
  - SQLite path
- `--releases-root`
  - release manifest root
- `--as-of-ts`
  - override current time
  - mainly useful for `next_due` and `run_once`
- `--mode`
  - release mode filter
  - default: `paper`
- `--log-path`
  - JSONL event log path
- `--json`
  - print raw machine-readable JSON instead of text
- `--broker-host`
  - broker API host
  - default: `127.0.0.1`
- `--broker-port`
  - broker API port
  - default: `7497`
- `--broker-client-id`
  - broker API client id
  - default: `31`
- `--broker-timeout-seconds`
  - broker request timeout
  - default: `4.0`

### Scheduler tuning flags

These are specific to `scheduler_service`:

- `--active-poll-seconds`
  - how often the daemon rechecks while work is active
  - default: `30`
- `--idle-max-sleep-seconds`
  - maximum idle sleep before checking again
  - default: `900`
- `--reconcile-grace-seconds`
  - delay after target execution before reconcile becomes due
  - default: `300`
- `--error-retry-seconds`
  - sleep after an exception before retrying in `serve`
  - default: `60`

### Scheduler examples

Inspect the next due phase:

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper
```

Inspect the next due phase as JSON:

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper --json
```

Run one scheduler-aware pass:

```bash
uv run python -m alpha.live.scheduler_service run_once --mode paper
```

Run one scheduler-aware pass at a fixed timestamp:

```bash
uv run python -m alpha.live.scheduler_service run_once --mode paper --as-of-ts 2026-04-10T13:20:00+00:00
```

Run the daemon with defaults:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper
```

`serve` always mirrors scheduler events to stdout and the JSONL log.
It also includes the related pod status, current fills, and after active phases the current `VPlan` plus broker-order snapshot.

Use the JSONL log for tailing if you want a second terminal view:

```bash
Get-Content alpha/live/logs/live_events.jsonl -Wait
```

Run the daemon with a faster active loop:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --active-poll-seconds 15 --idle-max-sleep-seconds 300
```

Run the daemon against paper IB Gateway:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --broker-port 4002
```

## Copy-Paste Connection Commands

These are the practical connection presets for the current live CLI.

Connection mapping:

```text
paper TWS      -> port 7497
paper Gateway  -> port 4002
live TWS       -> port 7496
```

Recommended safe rule:

```text
inspect first = status / next_due
mutate later  = tick / serve
```

### Paper TWS

Safe inspect commands:

```bash
uv run python -m alpha.live.runner status --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
uv run python -m alpha.live.scheduler_service next_due --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

One manual live pass:

```bash
uv run python -m alpha.live.runner tick --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

Long-running daemon:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

### Paper IB Gateway

Safe inspect commands:

```bash
uv run python -m alpha.live.runner status --mode paper --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
uv run python -m alpha.live.scheduler_service next_due --mode paper --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
```

One manual live pass:

```bash
uv run python -m alpha.live.runner tick --mode paper --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
```

Long-running daemon:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
```

### Live TWS

Use this only when:
- TWS is logged into the live account
- your manifest uses `mode: live`
- you are ready for real orders

Safe inspect commands:

```bash
uv run python -m alpha.live.runner status --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
uv run python -m alpha.live.scheduler_service next_due --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
```

One manual live pass:

```bash
uv run python -m alpha.live.runner tick --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
```

Long-running daemon:

```bash
uv run python -m alpha.live.scheduler_service serve --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
```

Important live rule:

```text
if auto_submit_enabled_bool = true
then tick or serve may submit real live orders
```

## Useful Manual Commands

### 1. Status

```bash
uv run python -m alpha.live.runner status --mode paper
```

This shows:
- enabled pods
- latest `DecisionPlan` status
- latest `VPlan` status
- next action
- latest broker snapshot timestamp
- latest fill timestamp

Raw machine output:

```bash
uv run python -m alpha.live.runner status --mode paper --json
```

### 2. Build Decision Plans

```bash
uv run python -m alpha.live.runner build_decision_plans --mode paper
```

This builds overnight `DecisionPlan` objects from approved snapshot data.

### 3. Build VPlan

```bash
uv run python -m alpha.live.runner build_vplan --mode paper
```

This reads broker truth and live prices, then builds pre-submit `VPlan` objects.

If you use a real IBKR session, you can pass broker connection args:

```bash
uv run python -m alpha.live.runner build_vplan --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

### 4. Show VPlan

```bash
uv run python -m alpha.live.runner show_vplan --mode paper
```

This is the main manual-review command.

It shows:
- decision-base shares
- current shares
- drift shares
- target shares
- delta shares
- live reference price
- estimated target notional
- warning flag

### 5. Submit VPlan

```bash
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id 1
```

This explicitly submits one ready `VPlan`.

Use this in:
- manual review mode
- debugging
- controlled paper testing

### 6. Post-Execution Reconcile

```bash
uv run python -m alpha.live.runner post_execution_reconcile --mode paper
```

This reads broker truth after the execution window and computes:

```text
residual_share_float = target_share_float - broker_share_float
```

If all touched assets are within tolerance, the `VPlan` becomes `completed`.
If not, it stays `submitted`.
If an intended exit still leaves broker shares open, the system writes a `critical` exit residual log.

### 7. Execution Report

```bash
uv run python -m alpha.live.runner execution_report --mode paper
```

This shows fill-level execution quality:
- symbol
- fill amount
- fill price
- official open price when available
- slippage per share
- slippage notional
- fill timestamp

## Manual Mode vs Auto Mode

Default behavior:

```yaml
execution:
  auto_submit_enabled_bool: true
```

### Manual mode

Set:

```yaml
execution:
  auto_submit_enabled_bool: false
```

Flow:
1. `tick` builds a `DecisionPlan`
2. `tick` builds a `VPlan`
3. you inspect it with `show_vplan`
4. you call `submit_vplan` manually if desired

### Auto mode

Use:

```yaml
execution:
  auto_submit_enabled_bool: true
```

Flow:
1. `tick` builds a `DecisionPlan`
2. `tick` builds a `VPlan`
3. if all gates pass, `tick` submits that exact `VPlan`
4. later `tick` calls `post_execution_reconcile` after the target execution time

The same `VPlan` is used for:
- manual review
- auto-submit

There is no separate "hidden auto order list".

## Manifest Structure

Example:

```yaml
identity:
  release_id: user_001.pod_dv2.daily_moo.v1
  user_id: user_001
  pod_id: pod_dv2_01

deployment:
  mode: paper
  enabled_bool: true

broker:
  account_route: DUK322077

strategy:
  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy
  data_profile_str: norgate_eod_sp500_pit
  params:
    max_positions_int: 10

market:
  session_calendar_id_str: XNYS

schedule:
  signal_clock_str: eod_snapshot_ready
  execution_policy_str: next_open_moo

execution:
  pod_budget_fraction_float: 0.03
  auto_submit_enabled_bool: true

bootstrap:
  initial_cash_float: 100000.0

risk:
  risk_profile_str: standard_equity_mr
```

## What Each Section Means

### `identity`

- `release_id`
  - exact deployment version
- `user_id`
  - owner of the pod
- `pod_id`
  - stable live sleeve id

Keep `pod_id` stable.
If you make a new version of the same pod, usually change `release_id` and keep `pod_id`.

### `deployment`

- `mode`
  - allowed: `paper`, `live`
- `enabled_bool`
  - `true` means the runner loads it
  - `false` means the runner ignores it

### `broker`

- `account_route`
  - broker account id
  - example: `DUK322077`

### `strategy`

- `strategy_import_str`
  - which research strategy to host
- `data_profile_str`
  - which data contract the pod expects
- `params`
  - strategy parameters

Currently supported `strategy_import_str` values:
- `strategies.dv2.strategy_mr_dv2:DVO2Strategy`
- `strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy`
- `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash`
- `strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy`

Currently supported `data_profile_str` values:
- `norgate_eod_sp500_pit`
- `norgate_eod_etf_plus_vix_helper`
- `norgate_eod_ndx_pit`
- `intraday_1m_plus_daily_pit`

### `market`

- `session_calendar_id_str`
  - exchange calendar controlling:
    - holidays
    - weekends
    - early closes
    - next session
    - month-end boundaries

Supported values:
- `XNYS`
- `XTSE`
- `XASX`

### `schedule`

This controls:
- when the strategy may decide
- when the execution policy may trade

`signal_clock_str` supported values:
- `eod_snapshot_ready`
- `month_end_snapshot_ready`
- `pre_close_15m`

`execution_policy_str` supported values:
- `next_open_moo`
- `same_day_moc`
- `next_month_first_open`

### `execution`

- `pod_budget_fraction_float`
  - fraction of broker `NetLiq` allocated to this pod
  - must satisfy:

```text
0 < pod_budget_fraction <= 1
```

- `auto_submit_enabled_bool`
  - `false` = manual-first mode
  - `true` = `tick` may auto-submit a ready `VPlan`

### `bootstrap`

- `initial_cash_float`
  - used only as the fallback for a brand-new pod before real broker-backed state exists

### `risk`

- `risk_profile_str`
  - simple policy label / future risk hook

## Important Operational Rules

### 1. Broker truth wins for positions

```text
LiveShares = BrokerShares
```

If the broker says your current positions differ from the `DecisionPlan` base positions, the system records a warning and still sizes from broker truth.

### 2. Cash does not block v2 preflight

Cash is stored for audit and monitoring, and share drift is also warning-only in v2.

### 3. `next_open_moo` is basket mode

For v2, `next_open_moo` means:

```text
one opening basket
```

Not:

```text
sell first, then buy
```

### 4. Missed window means expire

If the submit window is missed:

```text
DecisionPlan -> expired
```

and a stale `VPlan` is not reused later.

## Where State Lives

Live state is stored in:
- [live_state.sqlite3](C:/Users/User/Documents/workspace/alpha_super/alpha/live/live_state.sqlite3)

Important v2 objects stored there:
- `decision_plan`
- `vplan`
- `vplan_row`
- `broker_snapshot_cache`
- `vplan_reconciliation_snapshot`
- `vplan_broker_order`
- `vplan_fill`

## Recommended Daily Workflow

### Manual review workflow

```bash
uv run python -m alpha.live.runner tick --mode paper
uv run python -m alpha.live.runner show_vplan --mode paper
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper
```

### Automatic workflow

```bash
uv run python -m alpha.live.runner tick --mode paper
```

With:

```yaml
execution:
  auto_submit_enabled_bool: true
```

## Short Mental Model

Night:

```text
approved snapshot -> DecisionPlan
```

Pre-submit:

```text
DecisionPlan + BrokerSnapshot + live quote snapshot -> VPlan
```

Execution:

```text
VPlan -> broker orders -> fills
```

That is the v2 live system.

