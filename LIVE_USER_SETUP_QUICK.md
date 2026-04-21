TL;DR: to add another strategy under the same linked user, keep `user_id` and usually keep `account_route`, but create a new YAML with a new `release_id` and a new `pod_id`. To add a new user with two strategies, create a new folder under `alpha/live/releases/<user_id>/` and put two YAML files there, each with its own `release_id` and `pod_id`.

Simple rule:

```text
new_live_release = (user_id + account_route) + (new_release_id + new_pod_id)
```

## 1. Add one more strategy under the same user

This is the usual "same linked user" case.

Keep:
- same `user_id`
- same `account_route` if the strategy should trade in the same linked broker account

Change:
- new `release_id`
- new `pod_id`
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

Keep consistent per user:
- same `user_id`
- same `account_route` if both pods trade in the same linked account

Different per strategy:
- `release_id`
- `pod_id`
- `strategy_import_str`
- `params`
- schedule fields if needed

## Quick safety rules

- Start new YAMLs with `enabled_bool: false`.
- For a new version of the same pod: usually change `release_id`, keep `pod_id`.
- For a new pod: change both `release_id` and `pod_id`.
- If you want the cleanest isolation, use a separate DB path for a new live rollout.

## Quick check command

```bash
uv run python -m alpha.live.runner status --json
```

If the YAML is valid, the runner should load it without any manual SQL step.
