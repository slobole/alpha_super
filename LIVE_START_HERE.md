# Live Start Here

TL;DR: think in deployments and sleeves. One live deployment is for one client. That client has one or more independent strategy sleeves/pods. Each sleeve runs one strategy, trades one real IBKR account/subaccount, and keeps its own positions and state.

```text
deployment/VPS -> client -> sleeve/pod -> IBKR account/subaccount -> strategy
```

One deployment means one client:

```text
one client -> one VPS -> one repo clone -> one IBC/TWS session -> many sleeves/pods
```

Plain flow:

```text
latest data -> strategy decision -> order plan -> broker orders -> fills -> updated sleeve state
```

Internal names that appear in commands and status output:

- `DecisionPlan` = what the strategy decided.
- `VPlan` = the exact order plan built near submit time from current broker truth.

## Read This First

This file is the short operator path. Everything else lives here:

- [LIVE_RUNBOOK.md](docs/live/LIVE_RUNBOOK.md) — the full operator guide: Norgate setup, command semantics, manual vs automatic flow, recovery.
- [LIVE_TECHNICAL_REFERENCE.md](docs/live/LIVE_TECHNICAL_REFERENCE.md) — implementation truth.
- [COMMANDS.md](COMMANDS.md) — one-line cheat sheet for every live command.

If this deployment uses the private Norgate artifact server: start it on the Norgate node (`.\scripts\start_norgate_server.cmd`), then on each client VPS run `uv run python scripts\doctor_norgate_client.py` before starting `serve`. Full walkthrough: [LIVE_RUNBOOK.md](docs/live/LIVE_RUNBOOK.md#client-norgate-snapshot-check).

## The 5 Things To Remember

### 1. A Sleeve/Pod Is Real

```text
one live pod = one strategy = one linked IBKR account/subaccount route = one ledger
```

Do not assign two different live strategies to the same `account_route`, and do not treat a pod as just a label inside one shared account unless a later ledger system explicitly supports that.

### 2. The Strategy Decides, The Broker State Sizes

The strategy decides what it wants from approved prior data. Near execution time, the system reads the broker account and live prices, then prepares the exact orders — nothing is frozen overnight:

```text
PodBudget = BrokerNetLiq_at_VPlan * pod_budget_fraction
TargetShares_i = floor(TargetWeight_i * PodBudget / LiveReferencePrice_i)
```

`pod_budget_fraction` is a per-pod sizing cap on that pod's own linked account, not a shared-account allocation mechanism.

### 3. `tick` Is One Live Pass

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

`tick` only does work that is actually due. With `--pod-id`, it only checks and mutates that POD; without it, it scans all enabled PODs for the selected mode.

### 4. `serve` Is Just The Loop

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
```

`serve` waits, calls `tick` when work is due, then waits again — no second trading path. With `--pod-id` the default DB is `alpha/live/state/<mode>/<pod_id>.sqlite3`. Never start an already-trading POD from a fresh empty DB — the broker may hold positions while the strategy memory is empty. DB-path rules and the migration procedure: [LIVE_RUNBOOK.md](docs/live/LIVE_RUNBOOK.md#safe-operating-rule).

### 5. Paper Tests Flow, Not Real Auction Quality

Paper checks logic, scheduling, submit flow, and reconciliation. It does not prove real MOO/MOC execution quality. Validate execution with very small live size before scaling:

```text
incubation/rehearsal -> paper probe -> tiny live test -> scale slowly
```

## Safe First Commands

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.scheduler_service next_due --mode paper --pod-id pod_dv2_01
```

Open the local dashboard (Dashboard V3):

```bash
uv run python -m alpha.live.dashboard_v3 --host 127.0.0.1 --port 8080
```

The dashboard aggregates enabled PODs across `incubation`, `paper`, and `live`. It reads DB/log/artifact state; submitting and reconciling stay in the CLI (plus the explicit Operator Tools confirm flow).

## Where The Detail Lives

| Task | Section |
|---|---|
| Manual review flow (build → review → submit → reconcile) | [LIVE_RUNBOOK.md — Manual Mode](docs/live/LIVE_RUNBOOK.md#manual-mode) |
| Automatic flow and the auto-submit live rule | [LIVE_RUNBOOK.md — Automatic Mode](docs/live/LIVE_RUNBOOK.md#automatic-mode) |
| EOD account snapshot and equity-error basis | [LIVE_RUNBOOK.md — Record EOD Account State](docs/live/LIVE_RUNBOOK.md#record-eod-account-state) |
| Release files (what each field means, where they live) | [LIVE_RUNBOOK.md — Release Files As Operating Cards](docs/live/LIVE_RUNBOOK.md#release-files-as-operating-cards) |
| Stuck submit recovery | [LIVE_RUNBOOK.md — Stuck Submit Recovery](docs/live/LIVE_RUNBOOK.md#stuck-submit-recovery) |
| Red-alert debugging funnel | [DEBUGGING_RUNBOOK.md](docs/live/DEBUGGING_RUNBOOK.md) |

## One-Line Mental Model

```text
One client per deployment. Each sleeve decides independently, trades its own IBKR account, and reconciles from broker truth.
```
