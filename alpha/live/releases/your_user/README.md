# Manual Monthly TAA Templates

TL;DR: this folder contains ready-to-edit manual manifests for `strategy_taa_df_btal_fallback_tqqq_vix_cash`.

Files:
- `pod_taa_btal_fallback_tqqq_vix_cash_paper_manual.yaml`
- `pod_taa_btal_fallback_tqqq_vix_cash_live_manual.yaml`

Use only one enabled manifest for this pod at a time.

Live sizing stays:

```text
PodBudget_t = NetLiq_t * pod_budget_fraction_float
TargetDollar_{i,t} = w_{i,t} * PodBudget_t
TargetShares_{i,t} = floor(TargetDollar_{i,t} / LivePrice_{i,t})
OrderDelta_{i,t} = TargetShares_{i,t} - BrokerShares_{i,t}
```

The monthly manual path is:

```text
month-end snapshot
-> build_decision_plans
-> near first open of next month build_vplan
-> show_vplan
-> submit_vplan
-> post_execution_reconcile
```

You can either run the manual commands yourself or leave `serve` running as
the timing wrapper around `tick`. With `auto_submit_enabled_bool: false`,
`serve` can build the `DecisionPlan` and `VPlan`, but it will stop at manual
review instead of submitting orders.

Paper/manual commands:

```bash
uv run python -m alpha.live.runner build_decision_plans --mode paper
uv run python -m alpha.live.runner build_vplan --mode paper
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_taa_btal_fallback_tqqq_vix_cash_01
uv run python -m alpha.live.runner submit_vplan --mode paper --vplan-id <VPLAN_ID>
uv run python -m alpha.live.runner post_execution_reconcile --mode paper
```

Paper/manual service:

```bash
uv run python -m alpha.live.scheduler_service serve --mode paper --broker-host 127.0.0.1 --broker-port 7497 --broker-client-id 31
```

Live/manual commands:

```bash
uv run python -m alpha.live.runner build_decision_plans --mode live
uv run python -m alpha.live.runner build_vplan --mode live
uv run python -m alpha.live.runner show_vplan --mode live --pod-id pod_taa_btal_fallback_tqqq_vix_cash_01
uv run python -m alpha.live.runner submit_vplan --mode live --vplan-id <VPLAN_ID>
uv run python -m alpha.live.runner post_execution_reconcile --mode live
```

Live/manual service:

```bash
uv run python -m alpha.live.scheduler_service serve --mode live --broker-host 127.0.0.1 --broker-port 7496 --broker-client-id 31
```
