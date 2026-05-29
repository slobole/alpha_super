# casp3rsky_user Live VPS Checklist

TL;DR: these two live POD manifests are intended to run on a client VPS where
TWS/Gateway is logged into a live IBKR user that can see both account routes.
Keep `auto_submit_enabled_bool: false` for the first real-money cycle. Let the
services build VPlans, inspect them, then submit manually.

## PODs

| POD | Strategy | Account route | Manifest |
|---|---|---|---|
| `pod_taa_btal_fallback_tqqq_vix_cash_live_01` | `strategy_taa_df_btal_fallback_tqqq_vix_cash` | `U21192795` | `pod_taa_btal_fallback_tqqq_vix_cash_live_01.yaml` |
| `pod_ndx_atr_normalized_vxn_scaled_live_01` | `strategy_mo_atr_normalized_ndx_vxn_scaled` | `U25384771` | `pod_ndx_atr_normalized_vxn_scaled_live_01.yaml` |

Both PODs are monthly:

```text
completed month-end EOD snapshot -> next month first tradable open
```

`next_month_first_open` maps to IBKR MOO-style orders: market order with
`tif=OPG`.

## 1. Copy Files To The VPS

Copy this folder to the VPS:

```text
alpha/live/releases/casp3rsky_user/
```

Expected files:

```text
pod_taa_btal_fallback_tqqq_vix_cash_live_01.yaml
pod_ndx_atr_normalized_vxn_scaled_live_01.yaml
README.md
```

## 2. Sync Code And Python Environment

```powershell
cd C:\Users\Administrator\Documents\workspace\alpha_super
git pull
uv sync
```

The VPS must include the DTB3 live policy where stale-but-present DTB3 is a
warning, not a hard block. Missing/unavailable DTB3 still blocks TAA
DecisionPlan creation.

## 3. Configure `config.env`

Set the Norgate client settings for this release folder:

```text
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true
NORGATE_CLIENT_ID=casp3rsky_user
NORGATE_RELEASES_ROOT=alpha/live/releases/casp3rsky_user
NORGATE_SNAPSHOT_ROOT=C:\alpha\norgate_snapshots
NORGATE_API_URL=http://<norgate_node_tailscale_ip>:8787
NORGATE_API_TOKEN=<token>
```

`NORGATE_API_HOST` and `NORGATE_API_PORT` can be used instead of
`NORGATE_API_URL`.

## 4. Enable Manifests

Keep auto-submit off for the first live cycle:

```yaml
deployment:
  enabled_bool: true

execution:
  auto_submit_enabled_bool: false
```

This lets `serve` create DecisionPlans and VPlans, but it will not submit
orders automatically.

## 5. Run Norgate Client Doctor

```powershell
uv run python scripts\doctor_norgate_client.py --releases-root alpha/live/releases/casp3rsky_user
```

Required ending:

```text
RESULT: PASS
```

Do not start live services until the doctor passes.

## 6. Verify IBKR Managed Accounts

TWS/Gateway must be logged into a live user that can see both routes:

```powershell
@'
from alpha.live.ibkr_socket_client import IBKRSocketClient

expected_set = {"U21192795", "U25384771"}
client_obj = IBKRSocketClient("127.0.0.1", 7496, 99, 4.0)
visible_set = client_obj.get_visible_account_route_set()

print("Managed accounts:", sorted(visible_set))
missing_set = expected_set - set(visible_set)
print("Missing:", sorted(missing_set) if missing_set else "none")
'@ | uv run python -
```

If `U25384771` is missing, do not run the momentum POD. If `U21192795` is
missing, do not run the TAA POD.

## 7. Optional First-State Broker Snapshot

If the accounts already hold strategy positions before the first DB is created,
seed the POD DBs from broker truth before the trading cycle:

```powershell
uv run python -m alpha.live.runner eod_snapshot --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_taa_btal_fallback_tqqq_vix_cash_live_01
uv run python -m alpha.live.runner eod_snapshot --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_ndx_atr_normalized_vxn_scaled_live_01
```

This does not submit orders. It records current broker cash, NetLiq, and
positions into the pod-scoped DBs.

## 8. Inspect Status And Next Due

```powershell
uv run python -m alpha.live.runner status --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_taa_btal_fallback_tqqq_vix_cash_live_01
uv run python -m alpha.live.runner status --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_ndx_atr_normalized_vxn_scaled_live_01
```

```powershell
uv run python -m alpha.live.scheduler_service next_due --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_taa_btal_fallback_tqqq_vix_cash_live_01
uv run python -m alpha.live.scheduler_service next_due --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_ndx_atr_normalized_vxn_scaled_live_01
```

## 9. Start Pod-Scoped Services

Run each command in its own terminal/session:

```powershell
uv run python -m alpha.live.scheduler_service serve --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_taa_btal_fallback_tqqq_vix_cash_live_01
```

```powershell
uv run python -m alpha.live.scheduler_service serve --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id pod_ndx_atr_normalized_vxn_scaled_live_01
```

The manifests use different IBKR client IDs:

```text
TAA:      client_id_int = 31
Momentum: client_id_int = 32
```

Do not override both services to the same `--broker-client-id`.

## 10. Review VPlans Before Manual Submit

When a service reports that a VPlan is ready, inspect it:

```powershell
uv run python -m alpha.live.runner show_decision_plan --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID>
uv run python -m alpha.live.runner show_vplan --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID>
```

Check:

- account route is correct;
- target weights make sense;
- current broker shares are expected;
- order deltas are expected;
- no unexpected unrelated holdings are being liquidated;
- DTB3/FRED and Norgate freshness are not red/blocking.

## 11. Submit Manually

Submit only after VPlan review:

```powershell
uv run python -m alpha.live.runner submit_vplan --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID> --vplan-id <VPLAN_ID>
```

## 12. Reconcile After Execution

```powershell
uv run python -m alpha.live.runner post_execution_reconcile --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID>
```

Then inspect:

```powershell
uv run python -m alpha.live.runner status --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID>
uv run python -m alpha.live.runner execution_report --mode live --releases-root alpha/live/releases/casp3rsky_user --pod-id <POD_ID>
```

## Stop Rules

Stop and inspect manually if any of these happen:

- Norgate client doctor does not end with `RESULT: PASS`;
- either required account route is missing from `managedAccounts()`;
- the VPlan targets an unexpected account;
- the VPlan tries to sell an unrelated holding;
- `status` shows `broker_not_ready`, `account_not_visible`,
  `missing_live_price`, `missing_broker_response_ack`, or
  `execution_exception_parked`;
- TWS/Gateway shows an order, fill, or position that the runner does not
  reconcile.

