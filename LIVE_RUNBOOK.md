# Live Runbook

TL;DR: this is the operator guide for one-client live deployments. A deployment means one VPS/repo clone/IBC session for one client. Inside it, each sleeve/pod is one independent strategy running against one real IBKR account/subaccount.

For implementation details, use [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md).

## What This System Is

The live layer hosts research strategies from `strategies/` and runs them as live sleeves.

The current operating model is deliberately simple:

```text
one client -> one VPS -> one repo clone -> one IBC/TWS session -> many sleeves/pods
```

Do not run unrelated clients from the same live deployment unless we intentionally redesign for that later.

`user_id` is still useful for release files, logs, reports, and audit trails. It is not the main runtime selector right now.

Plain model:

```text
deployment -> client -> sleeve/pod -> IBKR account/subaccount -> strategy
```

Example:

```text
Edy deployment
  one VPS / repo clone / IBC session
  DV2 sleeve -> IBKR subaccount A -> DV2 strategy
  QPI sleeve -> IBKR subaccount B -> QPI strategy
  TAA sleeve -> IBKR subaccount C -> TAA strategy
```

The sleeve owns its own state. It reads its own broker account, submits its own orders, and reconciles its own fills.

The simple operating flow is:

```text
latest approved data -> strategy decision -> order plan -> broker orders -> fills -> updated sleeve state
```

Internal names you will see:

- `DecisionPlan` = the strategy decision.
- `VPlan` = the executable order plan.

Normal operators should think in sleeves and order plans. The internal names are there for commands, logs, and debugging.

## Safe Operating Rule

Inspect first. Mutate later.

Safe inspect commands:

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.scheduler_service next_due --mode paper --pod-id pod_dv2_01
```

Commands that may change live state:

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id pod_dv2_01 --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id pod_dv2_01
```

In live mode, mutation commands can submit real orders if the release allows auto-submit.

These commands operate on all enabled sleeves in this deployment by default. Pass `--pod-id` to isolate one POD. If a second client is added later, use a separate deployment rather than relying on client filtering inside this one.

State DB rule:

```text
no --pod-id                     -> alpha/live/live_state.sqlite3
--pod-id pod_x, no --db-path     -> alpha/live/state/<mode>/pod_x.sqlite3
--db-path custom.sqlite3         -> custom.sqlite3
```

For a new isolated POD, the POD-specific DB is the clean default. For an existing POD that already has open broker positions and old strategy state in `alpha/live/live_state.sqlite3`, keep `--db-path alpha/live/live_state.sqlite3` until that POD is migrated.

Current PAPER transition example:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_caspersky_account_paper_01 --db-path alpha/live/live_state.sqlite3
```

Use the same `--db-path` on `status`, `next_due`, `tick`, `show_vplan`, `submit_vplan`, `post_execution_reconcile`, and `eod_snapshot` while this PAPER POD is still on the old DB.

## Main Commands

### Status

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01
```

Shows enabled sleeves for this deployment, latest plan state, latest broker evidence, and next action.

Machine-readable output:

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01 --json
```

### One Live Pass

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

`tick` checks what is due and runs only valid work. With `--pod-id`, it mutates only that POD. It may:

- build a strategy decision;
- build an order plan;
- submit an auto-enabled order plan;
- reconcile after execution.

### Long-Running Service

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
```

`serve` is a timing loop around `tick`. With `--pod-id`, it uses the POD-specific default DB path unless `--db-path` is supplied.

```text
serve waits -> calls tick when due -> waits again
default POD DB = alpha/live/state/<mode>/<pod_id>.sqlite3
```

It does not implement another trading path.

### Next Due

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper --pod-id pod_dv2_01
```

Use this to inspect what the service would do next.

### Show Order Plan

```bash
uv run python -m alpha.live.runner show_vplan --mode paper
```

Show one sleeve:

```bash
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_dv2_01
```

Show one exact order plan:

```bash
uv run python -m alpha.live.runner show_vplan --mode paper --vplan-id 1
```

### Submit Manually

```bash
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id pod_dv2_01 --vplan-id 1
```

Use this only after reviewing the order plan.

### Reconcile After Execution

```bash
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id pod_dv2_01
```

This reads broker truth after the expected execution time. If the broker still has residual shares where the order plan expected none, the plan stays unresolved and the system reports it.

### Record EOD Account State

```bash
uv run python -m alpha.live.scheduler_service eod_snapshot --mode paper --pod-id pod_dv2_01
```

This samples broker cash, positions, and NetLiq after the market close. It updates `broker_snapshot_cache`, writes the latest `pod_state`, and appends a `pod_state_history` row tagged:

```text
snapshot_stage_str = eod
snapshot_source_str = broker
```

The same stage/source fields are also stored on the latest `pod_state`, so the current state can be inspected without joining to history.

For incubation the source is:

```text
snapshot_source_str = virtual_broker
```

Do not use EOD to prove that orders filled. That remains the job of `post_execution_reconcile`.

### Execution Report

```bash
uv run python -m alpha.live.runner execution_report --mode paper
```

Shows fill-level details such as symbol, fill amount, fill price, official open price when available, and slippage.

## Manual Vs Automatic

### Manual Mode

Use manual mode when the release file has auto-submit disabled.

Workflow:

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id pod_dv2_01 --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id pod_dv2_01
```

Plain meaning:

```text
build -> review -> submit -> reconcile
```

Optional after close:

```bash
uv run python -m alpha.live.scheduler_service eod_snapshot --mode paper --pod-id pod_dv2_01
```

### Automatic Mode

Use automatic mode only when you trust the sleeve in the selected environment.

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

or:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
```

If auto-submit is enabled, the same order plan you would review manually is the one the system submits automatically.

`serve` also runs the EOD snapshot phase after the session close plus a short buffer, but only when higher-priority submit/reconcile work is not due.

## Broker Connection Presets

Connection mapping:

```text
paper TWS       -> port 7497
paper Gateway   -> port 4002
live TWS        -> port 7496
```

### Paper TWS

```bash
uv run python -m alpha.live.runner status --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

### Paper IB Gateway

```bash
uv run python -m alpha.live.runner status --mode paper --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 4002 --broker-client-id 31
```

### Live TWS

Use this only when TWS is logged into the live account, the release uses `mode: live`, and you are ready for real orders.

```bash
uv run python -m alpha.live.runner status --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
uv run python -m alpha.live.runner tick --mode live --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
uv run python -m alpha.live.scheduler_service serve --mode live --pod-id pod_dv2_01 --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
```

Important:

```text
if auto-submit is enabled in live mode, tick or serve may submit real orders
```

## Release Files As Operating Cards

Release files live under:

```text
alpha/live/releases/<user_id>/*.yaml
```

In the current deployment model, this path should normally contain releases for one client identity only. The folder name is identity and audit context, not a signal to run multiple unrelated clients from one process.

Read a release file like an operating card:

- owner: who this sleeve belongs to;
- pod/sleeve: stable sleeve id;
- broker account: which IBKR account/subaccount it trades;
- strategy: which research strategy runs here;
- schedule: when the sleeve decides and trades;
- execution: whether auto-submit is allowed;
- deployment: paper or live, enabled or disabled.

Human example:

```text
Client: Edy
Sleeve: DV2
Broker account/subaccount: U1234567
Strategy: DV2
Mode: paper first, live only after approval
Schedule: decide after approved daily data, trade next open
Auto-submit: allowed only after the sleeve is trusted
```

Keep the exact YAML fields used by the current release examples. This section explains the human meaning; [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md) is the implementation reference.

## Paper Vs Live

Paper is for checking:

- strategy hosting;
- scheduling;
- order-plan creation;
- submit plumbing;
- reconciliation behavior.

Paper is not proof of real auction execution quality.

For MOO/MOC sleeves:

```text
backtest -> paper flow test -> tiny live test -> scale slowly
```

For same-day MOC logic:

```text
signal from pre-close live snapshot -> submit MOC -> fill at official close
```

Not:

```text
signal from official close -> submit MOC
```

*** CRITICAL*** Do not treat final-close information as available before the MOC order cutoff.

## Stuck Submit Recovery

If you see:

```text
vplan_status = submitting
broker_order_count = 0
ack_count = 0
fill_count = 0
```

treat it as a stuck submit until proven otherwise.

Rule:

```text
duplicate-submit risk is worse than assuming nothing happened
```

Recovery checklist:

1. Run status:

```bash
uv run python -m alpha.live.runner status --mode live
```

2. Check TWS or IB Gateway manually:

- no active order;
- no partial fill;
- no hidden API order in the account/orders view.

3. Check the operator log:

```text
alpha/live/logs/live_operator.log
```

Look for:

- broker connection failures;
- submit failures;
- stuck submit messages.

4. Fix the broker connection first.

5. Only if broker truth is clearly clean, reset local state from `submitting` back to `ready`.

6. Resubmit manually:

```bash
uv run python -m alpha.live.runner submit_vplan --mode live --vplan-id <VPLAN_ID> --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31 --json
```

7. Reconcile after a successful submit:

```bash
uv run python -m alpha.live.runner post_execution_reconcile --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31 --json
```

## Logs And State

Live state:

```text
alpha/live/live_state.sqlite3                          # default without --pod-id
alpha/live/state/<mode>/<pod_id>.sqlite3               # default with --pod-id
```

Explicit `--db-path` always wins. This is useful during transition from the old shared DB to a POD-specific DB.

Important: do not run an already-trading POD from a fresh empty POD DB. `build_vplan` reads broker positions before sizing orders, but `build_decision_plan` seeds the strategy from `pod_state` and `strategy_state` in the DB. If those are empty while the broker holds positions, the decision semantics can be wrong.

Event log:

```text
alpha/live/logs/live_events.jsonl
```

Tail the event log:

```bash
Get-Content alpha/live/logs/live_events.jsonl -Wait
```

Operator log:

```text
alpha/live/logs/live_operator.log
```

## Short Mental Model

```text
One deployment runs one client. Each sleeve decides independently, trades its own IBKR account, and reconciles from broker truth.
```
