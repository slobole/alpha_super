# Live Start Here

TL;DR: think in deployments and sleeves. One live deployment is for one client. That client has one or more independent strategy sleeves/pods. Each sleeve runs one strategy, trades one real IBKR account/subaccount, and keeps its own positions and state.

```text
deployment/VPS -> client -> sleeve/pod -> IBKR account/subaccount -> strategy
```

Example:

```text
Edy deployment
  one VPS / repo clone / IBC session
  DV2 sleeve -> IBKR subaccount A -> DV2 strategy
  QPI sleeve -> IBKR subaccount B -> QPI strategy
  TAA sleeve -> IBKR subaccount C -> TAA strategy
```

Each sleeve should be independent. It does not need to know what the other sleeves hold or trade.

## Deployment Boundary

For now, one deployment means one client.

In practice:

```text
one client -> one VPS -> one repo clone -> one IBC/TWS session -> many sleeves/pods
```

Do not run unrelated clients from the same live deployment unless the system is intentionally redesigned for that later.

`user_id` still belongs in release files and logs because it tells us whose deployment this is. It is not the main runtime selector right now. Commands such as `status`, `tick`, and `serve` operate on all enabled sleeves in this deployment by default. Pass `--pod-id` to operate on one isolated sleeve.

## What This System Does

The live system wakes up, checks which enabled sleeves in this client deployment are due, builds the next order plan, submits it if allowed, and reconciles after execution.

Plain flow:

```text
latest data -> strategy decision -> order plan -> broker orders -> fills -> updated sleeve state
```

Internal names:

- `DecisionPlan` = what the strategy decided.
- `VPlan` = the exact order plan built near submit time from current broker truth.

You do not need to think in those names for normal operation, but they appear in commands and status output.

## Read This First

Use this file for the short operator path.

Use these when you need more detail:

- [LIVE_RUNBOOK.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_RUNBOOK.md)
- [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md)
- [LIVE_TRADING_ARCHITECTURE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TRADING_ARCHITECTURE.md)

## The 5 Things To Remember

### 1. A Sleeve/Pod Is Real

A live sleeve should map to a real IBKR account, subaccount, or broker-recognized sleeve.

Do not treat a pod as just a label inside one shared account unless a later ledger system explicitly supports that.

Operational invariant:

```text
one live pod = one strategy = one linked IBKR account/subaccount route = one ledger
```

Do not assign two different live strategies to the same `account_route`.

### 2. The Strategy Decides, The Broker State Sizes

The strategy decides what it wants based on approved data.

Near execution time, the system reads the broker account and live prices, then prepares the exact orders. This avoids freezing stale share quantities overnight.

### 3. `tick` Is One Live Pass

Run:

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

`tick` only does work that is actually due. With `--pod-id`, it only checks and mutates that POD. Without `--pod-id`, it scans all enabled PODs for the selected mode.

### 4. `serve` Is Just The Loop

Run:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
```

`serve` does not create a second trading path. It waits, calls `tick` when work is due, then waits again. With `--pod-id`, its default DB path is POD-specific:

```text
alpha/live/state/<mode>/<pod_id>.sqlite3
```

If a POD already has open broker positions and was previously running from the old shared DB, keep using that DB explicitly until its state is migrated:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id <pod_id> --db-path alpha/live/live_state.sqlite3
```

Use the same `--db-path` on `status`, `next_due`, `tick`, `show_vplan`, and reconcile commands while the POD is still on the old DB.

Do not start an already-trading POD from a fresh empty POD DB. The broker may have positions, but the strategy memory in the DB would be empty.

### 5. Paper Tests Flow, Not Real Auction Quality

Paper is useful for checking logic, scheduling, submit flow, and reconciliation.

Do not assume IBKR paper fills prove real MOO/MOC execution quality. Validate real execution with very small live size before scaling.

## Safe First Commands

Inspect status:

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01
```

Inspect the next scheduler action:

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper --pod-id pod_dv2_01
```

Run one live pass:

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

Run the long-running service:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id pod_dv2_01
```

## Manual Review Flow

Use this when auto-submit is disabled in the release file.

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id pod_dv2_01 --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id pod_dv2_01
```

Meaning:

```text
build what is due -> review order plan -> submit it -> reconcile broker truth
```

## Automatic Flow

Use this when the release file allows auto-submit.

```bash
uv run python -m alpha.live.runner tick --mode paper --pod-id pod_dv2_01
```

Normal automatic path:

```text
tick builds decision -> tick builds order plan -> tick submits -> later tick reconciles
```

Important live rule:

```text
if auto-submit is enabled and mode is live, tick or serve may submit real live orders
```

## EOD Account Snapshot

After the market close, record a clean broker-backed account snapshot:

```bash
uv run python -m alpha.live.scheduler_service eod_snapshot --mode paper --pod-id pod_dv2_01
```

`serve` also schedules this after the session close plus a short buffer when no higher-priority execution/reconcile work is due.

Meaning:

```text
post_execution_reconcile = proves fills and target positions after trading
eod_snapshot = records clean end-of-day cash, positions, and NetLiq
pod_state = latest trusted broker-backed sleeve state
broker_snapshot_cache = latest raw broker snapshot for the account
```

`pod_state.snapshot_stage_str` and `pod_state.snapshot_source_str` show whether the latest trusted state came from `post_execution` or `eod`, and from `broker` or `virtual_broker`.

Sizing is unchanged. The final order plan still reads fresh broker truth near execution:

```text
PodBudget = BrokerNetLiq_at_VPlan * pod_budget_fraction
TargetShares_i = floor(TargetWeight_i * PodBudget / LiveReferencePrice_i)
```

For the normal isolated-account route, `pod_budget_fraction` is a per-pod sizing cap or risk throttle on that pod's own linked account. It is not a shared-account allocation mechanism for multiple strategies.

EOD state is mainly for the next decision's starting state and for cleaner backtest-reference comparison:

```text
equity_error_t = actual_eod_net_liq_t / reference_close_equity_t - 1
```

## Where Release Files Live

Current release files live under:

```text
alpha/live/releases/<user_id>/*.yaml
```

In the current one-client deployment model, this path should normally contain releases for one client identity only.

They say:

- who owns the sleeve;
- which pod/sleeve it is;
- which IBKR account it trades;
- which strategy runs there;
- when it decides;
- how it executes;
- whether it is enabled;
- whether auto-submit is allowed.

## Useful Files

- [LIVE_RUNBOOK.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_RUNBOOK.md)
- [LIVE_TECHNICAL_REFERENCE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TECHNICAL_REFERENCE.md)
- [alpha/live/releases](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases)

## One-Line Mental Model

```text
One client per deployment. Each sleeve decides independently, trades its own IBKR account, and reconciles from broker truth.
```
