# Norgate Snapshot V1

TL;DR: Norgate is centralized only as a market-data artifact service. Live and
paper trading never call remote Norgate during DecisionPlan or VPlan logic. A
client VPS auto-syncs immutable Parquet snapshots when needed, validates them
locally, then builds trading intent only from local files.

## Architecture

```text
Windows Norgate node
  norgatedata license
  scripts/serve_norgate_snapshot_api.py
  norgate_service/<client_id>/snapshots/<profile>/<YYYY-MM-DD>/

Client VPS
  local ignored release YAMLs define data_profile_str
  alpha/live/norgate_snapshot_sync.py ensures local snapshots
  C:\alpha\norgate_snapshots\<profile>\<YYYY-MM-DD>/
  data/norgate_loader.py reads local snapshots in snapshot mode
  runner/scheduler builds DecisionPlan only after local validation passes
```

The API is an artifact control/download layer, not a DataFrame query API.

```text
client-local release YAMLs
  -> required profiles
  -> Norgate API over Tailscale
  -> manifest.json + prices.parquet + optional universe.parquet
  -> local SHA256 validation
  -> DecisionPlan build from local snapshot files
```

Live trading state stays local to the client VPS:

- DecisionPlan semantics stay unchanged.
- VPlan sizing stays unchanged.
- IBKR adapters, submit, fills, reconciliation, and POD sqlite stay local.
- Snapshot mode never falls back to direct Norgate.

## Modes

Direct local Norgate mode:

```env
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=false
```

Use this on a Windows machine with a local Norgate install. The API is not used.

Client VPS snapshot mode:

```env
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true
NORGATE_SNAPSHOT_ROOT=C:\alpha\norgate_snapshots
```

If API config is present, the live tick can auto-sync missing/stale snapshots
before building a DecisionPlan:

```env
NORGATE_API_TOKEN=<shared_token>
NORGATE_API_HOST=<norgate_tailnet_ip>
NORGATE_API_PORT=8787
NORGATE_CLIENT_ID=<client_id>
NORGATE_RELEASES_ROOT=alpha/live/releases/<client_folder>
```

`NORGATE_API_URL=http://<host>:8787` can replace `NORGATE_API_HOST` and
`NORGATE_API_PORT`.

If API config is missing, snapshot mode becomes local-only. Valid local
snapshots can be used, but missing/stale snapshots block new trading intent.

Release YAMLs under `alpha/live/releases/<client_folder>/` are local per VPS
and ignored by Git. Use tracked examples from `docs/live/release_templates/`,
copy them into the local client folder, then edit account routes, mode, budgets,
and `enabled_bool` on that VPS only.

## Runtime Flow

```text
scheduler_service serve / runner tick
  -> inspect current pod cycle state
  -> ensure_norgate_snapshots_for_live_tick only when a new DecisionPlan may be needed
  -> read enabled release YAMLs for mode and optional pod_id
  -> derive required data_profile_str values
  -> validate local manifest and SHA256 hashes
  -> if needed, download/promote snapshots through the API
  -> if ready/direct: build DecisionPlan
  -> if waiting/failed/local-only missing: block only new DecisionPlan creation
  -> existing DecisionPlan/VPlan submit and post-execution reconcile continue from persisted state
```

Norgate is a pre-decision data dependency. Once a DecisionPlan exists for the
current signal cycle, the system treats that plan as the frozen trading intent
artifact for that cycle. Later VPlan build, order submission, and reconciliation
use the persisted DecisionPlan, live broker/reference-price reads, and broker
state. They do not re-run snapshot sync or strategy data loading. The scheduler
may still evaluate its local build gate/readiness view so it can decide when the
next signal cycle becomes eligible.

After a DecisionPlan reaches `completed`, the scheduler does not re-sync
Norgate again until the next signal cycle is actually eligible to form a new
DecisionPlan. For a daily EOD pod, that means after the next market session's
close plus the EOD snapshot buffer. For a month-end pod, that means after the
next month-end session close plus the same buffer.

A Norgate-gated DecisionPlan suppresses sync only when its persisted
`snapshot_metadata_dict.norgate_data_profile_str` matches the release
`data_profile_str`. Legacy or manually inserted DecisionPlans without that
profile proof do not advance to VPlan.

The auto-sync writes:

```text
NORGATE_SNAPSHOT_ROOT\.client_sync_status.json
NORGATE_SNAPSHOT_ROOT\.sync.lock
```

`.client_sync_status.json` is the client-side audit/status file. The dashboard
reads it, but does not mutate it.

`.sync.lock` prevents two local scheduler/tick processes from writing the same
snapshot root at the same time. Stale locks are recovered after the configured
TTL.

## Failure Behavior

Bad or missing market data blocks new trading intent.

```text
API unavailable
  -> sync status waiting/failed
  -> no new DecisionPlan
  -> existing DecisionPlan/VPlan submit/reconcile may continue

bad token
  -> sync failed
  -> no new DecisionPlan
  -> existing DecisionPlan/VPlan submit/reconcile may continue

missing/hash-invalid snapshot
  -> no promotion
  -> no new DecisionPlan
  -> existing DecisionPlan/VPlan submit/reconcile may continue

snapshot arrives after cutoff
  -> scheduler gate snapshot_window_expired
  -> no late trade

ready/submitted VPlan exists
  -> submit/reconcile may continue
  -> no additional DecisionPlan is created while Norgate is blocked
```

Local snapshot `ready` means the manifest and hashes are valid. A DecisionPlan
can still fail later during strategy data loading. The dashboard separates these
states so an operator can see whether the stop happened at sync, build gate, or
post-sync strategy data loading.

## Dashboard Debugging

Open the selected POD, then use the `Freshness` tab.

The `Norgate Sync` panel shows:

- data source: `direct` or `snapshot`
- status: `direct`, `ready`, `waiting`, `failed`, or `local_snapshot_only`
- stage: human label such as `Local snapshot ready`, `API config missing`,
  `Sync failed`, `Build gate waiting`, or `Snapshot window expired`
- profile and snapshot date
- build gate reason
- per-profile snapshot date, manifest hash prefix, and error
- per-release gate reason scoped to the selected POD
- last attempt, last success, status file path, and last error

The `Debug` tab also shows a compact `Norgate sync` evidence item and can show
`Post-sync data load failed` only when a DecisionPlan data-load error happened
after the latest sync success/attempt timestamp.

## Operator Commands

Start the Norgate server on the Windows Norgate node:

```powershell
.\scripts\start_norgate_server.cmd
```

Run the server doctor:

```powershell
uv run python scripts\doctor_norgate_server.py `
  --service-root C:\alpha\norgate_service `
  --api-url http://<norgate_tailnet_ip>:8787
```

Run the client doctor on a client VPS:

```powershell
uv run python scripts\doctor_norgate_client.py
```

Start a pod-scoped paper scheduler:

```powershell
uv run python -m alpha.live.scheduler_service serve `
  --mode paper `
  --releases-root alpha/live/releases/<client_folder> `
  --pod-id <pod_id>
```

Inspect status without mutating live state:

```powershell
uv run python -m alpha.live.runner status `
  --mode paper `
  --releases-root alpha/live/releases/<client_folder> `
  --pod-id <pod_id>
```

## Important Files

- `scripts/serve_norgate_snapshot_api.py`: private Norgate artifact API.
- `scripts/sync_norgate_snapshots_api.py`: manual/operator snapshot sync.
- `scripts/doctor_norgate_server.py`: Windows Norgate node readiness check.
- `scripts/doctor_norgate_client.py`: client VPS readiness check.
- `scripts/start_norgate_server.cmd`: starts API and runs server doctor.
- `alpha/live/norgate_snapshot_sync.py`: live tick auto-sync/readiness gate.
- `data/norgate_snapshot_store.py`: local manifest/hash/parquet validation.
- `data/norgate_loader.py`: direct-vs-snapshot Norgate boundary.
- `alpha/live/dashboard.py`: read-only dashboard Norgate Sync panel.

## Current Supported Profiles

```text
norgate_eod_sp500_pit
norgate_eod_etf_plus_vix_helper
norgate_eod_ndx_pit
norgate_eod_ndx_pit_plus_vxn_helper
```

`intraday_1m_plus_daily_pit` is intentionally unsupported in Snapshot V1.

## Verification

Relevant test groups:

```powershell
uv run --with pytest python -m pytest `
  tests/test_norgate_snapshot_store.py `
  tests/test_norgate_snapshot_api.py `
  tests/test_norgate_snapshot_sync.py `
  tests/test_live_dashboard.py `
  tests/test_live_scheduler_service.py `
  tests/test_live_scheduler_utils.py -q
```

Runner safety checks:

```powershell
uv run --with pytest python -m pytest `
  tests/test_live_runner.py::test_tick_does_not_build_decision_plan_when_snapshot_sync_waits `
  tests/test_live_runner.py::test_tick_blocks_only_decision_plan_build_when_snapshot_sync_not_ready -q
```
