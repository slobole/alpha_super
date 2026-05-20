TL;DR: one live POD is one strategy running against one linked IBKR account/subaccount route. To add another strategy for the same client deployment, keep the client `user_id`, but create a new YAML with a new `release_id`, new `pod_id`, and a different `account_route`.

Simple rule:

```text
new_strategy_pod = same client user_id + new release_id + new pod_id + different account_route
```

## 1. Add one more strategy under the same client

This is the usual same-client deployment case. `user_id` is the client or deployment identity. It is not permission to run multiple strategies inside the same linked broker account.

Keep:
- same `user_id`

Change:
- new `release_id`
- new `pod_id`
- different `account_route`
- new `strategy_import_str`
- strategy-specific `params`

Folder pattern:

```text
alpha/live/releases/<same_user_id>/
```

Example:

```text
alpha/live/releases/excelence_trade_paper_001/
  pod_qpi_01.yaml
  pod_taa_01.yaml
  pod_ndx_mo_01.yaml
```

Each file above should point at a different linked IBKR account/subaccount route.

Minimal example:

```yaml
identity:
  release_id: excelence_trade_paper_001.pod_qpi.daily_moo.v1
  user_id: excelence_trade_paper_001
  pod_id: pod_qpi_01

deployment:
  mode: paper
  enabled_bool: false

broker:
  account_route: DUK322077

strategy:
  strategy_import_str: strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy
  data_profile_str: norgate_eod_sp500_pit
  params:
    max_positions_int: 10
```

Notes:
- `release_id` must be unique.
- `pod_id` must be unique among enabled pods.
- `account_route` should be unique per live strategy POD unless the system later adds an explicit shared-account pod ledger.
- No manual SQL work is needed. The runner loads YAMLs and upserts them into `live_release`.

## 2. Add a new user with two strategies

Create a new folder:

```text
alpha/live/releases/<new_user_id>/
```

Then add two YAML files, one per pod.

Example:

```text
alpha/live/releases/user_002/
  pod_dv2_01.yaml
  pod_taa_01.yaml
```

Rule:

\[
\text{user with 2 strategies} = 2 \times \text{pod YAML files}
\]

Example structure:

```yaml
identity:
  release_id: user_002.pod_dv2.daily_moo.v1
  user_id: user_002
  pod_id: pod_dv2_01
```

```yaml
identity:
  release_id: user_002.pod_taa.monthly_open.v1
  user_id: user_002
  pod_id: pod_taa_01
```

Keep consistent per client:
- same `user_id`

Different per strategy:
- `release_id`
- `pod_id`
- `account_route`
- `strategy_import_str`
- `params`
- schedule fields if needed

## Quick safety rules

- Start new YAMLs with `enabled_bool: false`.
- For a new version of the same pod: usually change `release_id`, keep `pod_id`.
- For a new pod: change both `release_id` and `pod_id`.
- For a new strategy: use a different linked IBKR account/subaccount route.
- If you want the cleanest isolation, use a separate DB path for a new live rollout.

## Quick check command

```bash
uv run python -m alpha.live.runner status --json
```

If the YAML is valid, the runner should load it without any manual SQL step.
