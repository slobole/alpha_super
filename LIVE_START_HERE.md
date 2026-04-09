# Live Start Here

TL;DR: the live v2 system works in 2 stages:

```text
DecisionPlan_t = f(approved_snapshot_t, strategy_memory_t)
VPlan_submit = f(DecisionPlan_t, BrokerSnapshot_submit, live_quote_snapshot)
```

So:
- night = freeze the decision
- pre-submit = size the final shares from broker truth

## What To Read First

If you only want the short operator path, read this file first.

If you want more detail after that:
- [LIVE_RUNBOOK.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_RUNBOOK.md)
- [LIVE_TRADING_ARCHITECTURE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TRADING_ARCHITECTURE.md)

## The 4 Things To Remember

### 1. The strategy does not freeze shares overnight

The overnight artifact is a `DecisionPlan`.

It stores:
- a typed decision book
- target weights or entry/exit instructions
- strategy memory

It does **not** store final overnight share counts.

There are now two decision-book types:
- `incremental_entry_exit_book`
  - used by DV2 and QPI-style equal-slot entry / exit-to-zero systems
- `full_target_weight_book`
  - used by TAA and full rebalance momentum systems

### 2. Final shares are computed near submit time

The execution layer reads:
- broker positions
- broker `NetLiq`
- live prices

Then it computes:

```text
PodBudget = NetLiq_broker * pod_budget_fraction
TargetShares_i = floor(target_weight_i * PodBudget / LivePrice_i)
OrderDelta_i = TargetShares_i - BrokerShares_i
```

### 3. Share drift is a warning, not a block

If:

```text
DecisionBaseShares_i != BrokerShares_i
```

the system records a warning and still sizes from broker truth.

### 4. `show_vplan` is the main manual review screen

It shows:
- decision-base shares
- current broker shares
- drift shares
- target shares
- delta shares
- live reference price
- warning flag

## The Main Files

- manifests:
  - [pod_dv2_01.yaml](C:/Users/User/Documents/workspace/alpha_super/alpha/live/releases/user_001/pod_dv2_01.yaml)
- operator guide:
  - [LIVE_RUNBOOK.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_RUNBOOK.md)
- architecture:
  - [LIVE_TRADING_ARCHITECTURE.md](C:/Users/User/Documents/workspace/alpha_super/LIVE_TRADING_ARCHITECTURE.md)

## The Main Commands

### Status

```bash
uv run python -m alpha.live.runner status --mode paper
```

### Main loop

```bash
uv run python -m alpha.live.runner tick --mode paper
```

### Optional scheduler service

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper
```

This is only a timing wrapper around `tick`.
It does not replace `tick`.

### Build only the overnight decision

```bash
uv run python -m alpha.live.runner build_decision_plans --mode paper
```

### Build the pre-submit execution plan

```bash
uv run python -m alpha.live.runner build_vplan --mode paper
```

### Show the execution plan

```bash
uv run python -m alpha.live.runner show_vplan --mode paper
```

### Submit one VPlan manually

```bash
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id 1
```

### Reconcile after execution

```bash
uv run python -m alpha.live.runner post_execution_reconcile --mode paper
```

## The Simplest Daily Manual Workflow

Use this only if you explicitly set:

```yaml
execution:
  auto_submit_enabled_bool: false
```

```bash
uv run python -m alpha.live.runner tick --mode paper
uv run python -m alpha.live.runner show_vplan --mode paper
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper
```

## The Simplest Auto Workflow

Auto-submit is now the default manifest behavior.

Use:

```yaml
execution:
  auto_submit_enabled_bool: true
```

Then run:

```bash
uv run python -m alpha.live.runner tick --mode paper
```

The normal automatic path is:

```text
tick builds DecisionPlan -> tick builds VPlan -> tick submits -> later tick reconciles after execution time
```

## Optional Long-Running Service

If you do not want an every-minute shell loop, you can run:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper
```

Useful helper commands:

```bash
uv run python -m alpha.live.scheduler_service next_due --mode paper
uv run python -m alpha.live.scheduler_service run_once --mode paper
```

Mental model:

```text
scheduler_service decides when to call tick
tick still does all real work
```

## One-Line Mental Model

Night:

```text
snapshot -> DecisionPlan
```

Pre-submit:

```text
DecisionPlan + broker truth + live prices -> VPlan
```

Execution:

```text
VPlan -> orders -> fills -> new pod state
```
