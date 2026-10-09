# Live Runbook

TL;DR: this is the operator guide for one-client deployments. A deployment means one VPS/repo clone/IBC session for one client. Inside it, each sleeve/pod is one independent strategy. Incubation/rehearsal uses a SIM ledger; paper probes broker plumbing; live uses real IBKR accounts/subaccounts.

For implementation details, use [LIVE_TECHNICAL_REFERENCE.md](LIVE_TECHNICAL_REFERENCE.md).

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

For promotion, use one simple ladder:

```text
incubation/rehearsal -> paper probe -> small live account -> bigger account
```

In incubation, the SIM ledger is the official rehearsal accounting truth. IBKR is used for price/reference plumbing, especially open-price evidence for MOO-style flows. Paper remains a separate broker probe; paper positions and fills are not incubation P&L.

Internal names you will see:

- `DecisionPlan` = the strategy decision.
- `VPlan` = the executable order plan.

Normal operators should think in sleeves and order plans. The internal names are there for commands, logs, and debugging.

## Back Up Every Pod Database Before Updating Code

Complete this procedure **before `git pull` or checking out updated code**, and
before starting any command that could initialize/migrate a state database. It
applies to every pod in this deployment, including NDX/TAA, disabled pods and all
modes, not just the strategy being updated. These are deployment instructions;
they do not authorize a production update or broker action.

1. Stop the deployment's scheduler, watchdog/automatic restart tasks, OPS and
   other processes that can open or write pod databases. Confirm they stay
   stopped. Stopping these processes does not cancel existing broker orders;
   record their state and use an approved maintenance window.
2. Inventory every database path used by launch commands and scheduled tasks:
   `alpha/live/state/<mode>/*.sqlite3`, legacy `alpha/live/live_state.sqlite3`,
   and every explicit `--db-path`, including paths outside this checkout.
   Match every configured pod to a path; a shared database needs one backup.
   Do not infer the inventory from enabled releases alone.
3. In a new, access-controlled backup folder outside the checkout, save
   `git rev-parse HEAD`, the source-to-backup path list, and the local ignored
   release YAMLs and `config.env` (which contains secrets). Back up **every
   inventoried database** with SQLite's backup API. A plain copy of only the
   `.sqlite3` file can omit committed WAL data. For each source/backup pair,
   run the following with absolute paths and an already-created backup folder:

   ```powershell
   @'
   import sqlite3
   import sys
   from pathlib import Path
   source_path_obj, backup_path_obj = map(Path, sys.argv[1:])
   if not source_path_obj.is_file() or backup_path_obj.exists():
       raise SystemExit("Source missing or backup already exists; stop.")
   with sqlite3.connect(source_path_obj.resolve().as_uri() + "?mode=ro", uri=True) as source_obj:
       with sqlite3.connect(backup_path_obj) as backup_obj:
           source_obj.backup(backup_obj)
           check_list = backup_obj.execute("PRAGMA quick_check").fetchall()
           if check_list != [("ok",)]:
               raise SystemExit(f"Backup integrity check failed: {check_list}")
   print(f"Verified backup: {source_path_obj} -> {backup_path_obj}")
   '@ | uv run python - C:\absolute\source\pod.sqlite3 C:\absolute\backup\pod.sqlite3
   ```

4. Check the saved path list against the full inventory and confirm every
   backup passed `quick_check`. Only then proceed with the approved code update,
   validation and restart. Keep the backups until deployment verification ends.

### Check Databases Previously Opened By `4624236` Before Runtime Initialization

For any dev, paper or test database previously touched by `4624236`, run this
check on a **verified, stable offline backup copy** before opening the source
database with the restored code. Keep its writers stopped as described above.
Even `status`, `next_due` or other inspection commands can construct
`LiveStateStore` and initialize indexes; do not use them as this preflight.

The restored `54b417f` fill key is
`(vplan_id_int, broker_order_id_str, fill_timestamp_str, fill_amount_float, fill_price_float)`.
The experimental execution-ID schema could retain multiple rows for one such
tuple. Creating the restored unique index then fails. Check every affected copy:

```powershell
.\.venv\Scripts\python.exe scripts/review/preflight_legacy_fill_duplicates.py --db C:\offline_backups\pod_core5.sqlite3 C:\offline_backups\pod_capsule.sqlite3
if ($LASTEXITCODE -ne 0) { throw 'Fill compatibility is not clear; do not initialize these databases.' }
```

The utility uses raw SQLite in read-only immutable mode, never `LiveStateStore`,
and prints JSON containing the table/index schema, duplicate counts and up to
20 duplicate groups with their first/last fill row IDs. It does not migrate,
checkpoint, deduplicate or repair anything. Adjacent `-wal`, `-shm` or `-journal`
files, or a file change during inspection, make the input unsafe and fail the
check. **Never delete sidecars to make the check pass**; make a verified offline
backup through SQLite's backup API instead.

- Exit **0**: every supplied copy has `status_str="no_duplicate_tuples"`.
- Exit **1**: at least one has `status_str="duplicates_found"`; keep runtime
  initialization stopped and preserve both executions for owner review.
- Exit **2**: inspection is incomplete, unsupported or unsafe; keep initialization
  stopped and resolve the reported input error.

Exit zero proves only that the restored tuple key has no duplicates. It does not
reverse the experimental schema, establish broker reconciliation, or authorize
deployment or rollback. No automatic deletion or merging of fill evidence is
part of this procedure.

### Short Rollback Procedure

Keep all database users and automatic restarts stopped. Preserve the failed
deployment's databases and their `-wal`/`-shm` sidecars for diagnosis, then return
to the recorded pre-update code revision and matching local configuration.
If no broker orders/fills or account changes have occurred since the backup,
restore each verified database to its original inventoried path, with no stale
sidecars from the failed database left beside it. Check integrity and run the
approved validation before restarting. If broker/account state has advanced,
**do not restore an older ledger or resume trading from it**: keep automation
stopped and review reconciliation against current broker evidence first.

## Norgate Artifact Server

Use this only on the Windows Norgate node. It serves validated Parquet snapshots over Tailscale; it does not connect to IBKR or submit orders.

Generate a real token once:

```powershell
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
[Convert]::ToBase64String($bytes)
```

Put the token and server settings in ignored `config.env`:

```env
NORGATE_API_TOKEN=<paste_generated_token>
NORGATE_SERVICE_ROOT=C:\alpha\norgate_service
NORGATE_API_HOST=100.123.13.69
NORGATE_API_PORT=8787
```

Use the Norgate node Tailscale IP for `NORGATE_API_HOST`. Client machines must use the same token when syncing snapshots.

Start the server and run the doctor:

Before the checkout/pull below, complete the database backup procedure above
for any pods hosted in this deployment.

```powershell
cd C:\Users\Administrator\Documents\workspace\alpha_super
git checkout codex/norgate-snapshots-v1
git pull
.\scripts\start_norgate_server.cmd
```

Expected behavior:
- a visible `Norgate API debug` window stays open with API stdout/stderr;
- the launcher waits for `/healthz`;
- `doctor_norgate_server.py` runs and should end with `RESULT: PASS`.

The server writes per-client artifacts under:

```text
C:\alpha\norgate_service\<client_id>\snapshots\<profile>\<YYYY-MM-DD>\
```

## Client Norgate Snapshot Check

Use this on each client VPS or dev machine that should trade from Norgate snapshots.

The client does not call Norgate directly. It asks the private Norgate API for the profiles used by enabled release YAMLs, downloads Parquet snapshot files, validates hashes locally, then the live scheduler reads only the local snapshot folder.

```text
enabled release YAMLs -> Norgate API -> local snapshots -> scheduler gate -> DecisionPlan
```

Required ignored `config.env` values on the client:

```env
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true
NORGATE_API_TOKEN=<same_token_as_server>
NORGATE_API_HOST=<norgate_node_tailscale_ip>
NORGATE_API_PORT=8787
NORGATE_CLIENT_ID=client_caspersky
NORGATE_RELEASES_ROOT=alpha/live/releases/caspersky_account
NORGATE_SNAPSHOT_ROOT=C:\alpha\norgate_snapshots
```

You can use `NORGATE_API_URL=http://<norgate_node_tailscale_ip>:8787` instead of `NORGATE_API_HOST` and `NORGATE_API_PORT`.

Set `NORGATE_CLIENT_ID` and `NORGATE_RELEASES_ROOT` to the real client folder you are deploying. The values above are examples.

Release YAMLs in `alpha/live/releases/<client_id>/` are local per VPS/client
and are ignored by Git. Copy tracked examples from `docs/live/release_templates/`
when creating a new POD, then edit the local YAML only.

Run the client doctor before starting a scheduler on a new client VPS:

Before the checkout/pull below on an existing deployment, complete the database
backup procedure above for every pod.

```powershell
cd C:\Users\Administrator\Documents\workspace\alpha_super
git checkout codex/norgate-snapshots-v1
git pull
uv run python scripts\doctor_norgate_client.py
```

Expected ending:

```text
[PASS] enabled release profiles: ...
[PASS] api healthz
[PASS] api token auth
[PASS] sync snapshots
[PASS] manifest hash validation
[PASS] scheduler snapshot heartbeat
RESULT: PASS
```

If the result is `FAIL`, do not start `serve` yet. Fix the printed failing line first. Common causes are a wrong token, wrong Tailscale IP, no enabled release YAMLs, an unsupported `data_profile_str`, or a missing/invalid local snapshot.

Use `--overwrite` only when you intentionally want to replace an existing same-date local snapshot:

```powershell
uv run python scripts\doctor_norgate_client.py --overwrite
```

This doctor may create or replace files under `NORGATE_SNAPSHOT_ROOT`. It does not touch IBKR, POD DBs, orders, fills, reconciliation, or live state.

## Safe Operating Rule

Inspect first. Mutate later.

Safe inspect commands:

```bash
uv run python -m alpha.live.runner status --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id pod_dv2_01
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

Use the same `--db-path` on `status`, `next_due`, `tick`, `show_decision_plan`, `show_vplan`, `submit_vplan`, `post_execution_reconcile`, and `eod_snapshot` while this PAPER POD is still on the old DB.

## Local POD Dashboard

Start Dashboard V3 (Flask + Jinja + HTMX — no Node, no build step):

```bash
uv run python -m alpha.live.dashboard_v3 --host 127.0.0.1 --port 8080
```

Open the operator console:

```text
http://127.0.0.1:8080
```

V3 is one local web page for all enabled PODs in this deployment. Three mode pages (`/live`, `/paper`, `/incubation`) each get the full window — no tabs, no mixed-mode tables fighting for space. Above them sits a polled health strip (Norgate freshness, EOD coverage, disk), a polled cross-pod "what's next" schedule, and a top-bar verdict.

Expanding a pod shows today's cycle as a vertical timeline (DB -> Decision -> VPlan -> ACK -> Fill -> Reconcile -> EOD), each step with its evidence inline and bulkier sub-tables behind `<details>` so the narrative reads quickly. The EOD card embeds an SVG equity curve with drawdown shading and daily-PnL bars. The separate Live vs Backtest card links to the latest comparison report and `trade_fill_diff.csv` when available.

Operator Tools sits collapsed at the bottom of every expanded pod. Five buttons: Live vs Backtest / Tick / Submit VPlan / Reconcile / EOD Snapshot. Clicking shows a preview before the single Confirm. Every confirmed action goes through the same security ceremony as before: JSON POST + same-origin + server-issued action token + explicit `confirmed_bool=true`, and is logged to `alpha/live/logs/operator_journal.jsonl`. View the log at `/journal`.

Set `ALPHA_DISCORD_WEBHOOK_URL` in the environment to receive a Discord ping the first time any pod transitions to red. State persists in `alpha/live/logs/notification_state.json`, so a recovered pod that turns red again fires a fresh alert; missing env var = silent.

Dashboard V3 also shows the Live OPS Inspector verdict. The Inspector is a
read-only contract: unknown is not green, stale is not green, and it never
submits or cancels orders. Inspect the same verdict from the CLI:

```powershell
uv run python -m alpha.live.runner ops_report --mode live --json
```

### Live OPS Watchdog (scheduled)

The watchdog is the "employee on shift": one scheduled job per VPS that builds
the Inspector report, persists it, fires red-transition Discord alerts, and
pings the external dead-man switch **last** — all in one process, so silence at
the external watcher always means "the inspector did not complete a run".

Setup, once per VPS:

1. Create a check at healthchecks.io (free tier): period **5 minutes**, grace
   **15 minutes** (tolerates one missed run or a slow build). Copy the ping URL.
2. Add to the gitignored `config.env` at the repo root:

   ```ini
   ALPHA_INSPECTOR_HEARTBEAT_URL=https://hc-ping.com/<your-check-uuid>
   ALPHA_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

3. Register the Task Scheduler job (every 5 minutes, runs without logon):

   ```powershell
   .\scripts\setup_live_ops_watchdog_task.ps1            # all modes
   .\scripts\setup_live_ops_watchdog_task.ps1 -Mode live # live only (recommended)
   ```

   Scope to `-Mode live` if you run incubation/paper rehearsal pods. Rehearsal
   pods build plans but never submit, so their execution window always reads as
   "missed" → the watchdog would ping `/fail` forever, leaving the dead-man check
   permanently down and unable to distinguish "a pod is red" from "the VPS died".
   Incubation stays fully visible on the dashboard `/incubation` page either way.

4. Verify:

   ```powershell
   uv run python scripts/live_ops_watchdog.py --json
   Start-ScheduledTask -TaskName AlphaLiveOpsWatchdog
   Get-ScheduledTaskInfo -TaskName AlphaLiveOpsWatchdog
   ```

Semantics: overall **red** pings `<url>/fail` (healthchecks alerts immediately,
even if Discord is unreachable); a **fatal error or hang** pings nothing, so the
external watcher alerts after the grace window. The latest report is always at
`alpha/live/logs/ops_report_latest.json`. The watchdog keeps its own dedup state
in `alpha/live/logs/watchdog_notification_state.json`; if the dashboard also has
`ALPHA_DISCORD_WEBHOOK_URL` set, the same red transition can alert twice —
harmless, accepted.

Failed Discord deliveries with a configured webhook are remembered and retried
once on the next watchdog run while the same Pod/Inspector remains red.
Confirmed delivery stops retries; recovery clears the pending alert. This does
not change report severity, heartbeat `/fail` behavior, or the task schedule.
See [Discord notifications](DASHBOARD_V3_RUNBOOK.md#discord-red-alert-notifications-optional)
for restart, legacy-state, and missing-webhook behavior.

The low-level building block `scripts/live_ops_heartbeat.py` still exists for
ad-hoc pings, but the watchdog is the supported scheduled path: the heartbeat
must be emitted by the inspector run itself, otherwise the dead-man switch
proves the wrong thing alive.

If the external service stops receiving pings on time, treat the VPS as silent
and inspect manually even if the last dashboard page was green.

When a red alert fires, follow `docs/live/DEBUGGING_RUNBOOK.md` — the
step-by-step funnel (Discord → dashboard → doctor → logs) for diagnosing a flag
under pressure.

See `docs/live/DASHBOARD_V3_RUNBOOK.md` for the one-page systemd + Tailscale deploy recipe.

The V2 React console (`alpha/live/dashboard_v2/`) and the V1 HTTP handler (`alpha.live.dashboard.serve_dashboard`) have been removed. `alpha/live/dashboard.py` is now a pure data-builder library used by `alpha.live.dashboard_v3.*`.

Live vs Backtest is the one explicit comparison background action in the dashboard. Pressing `Live vs Backtest` starts `compare_reference` for that POD and writes analysis artifacts under:

```text
results/live_reference_compare/<mode>/<pod_id>/<timestamp>/
```

Dashboard DB routing is configured in:

```text
alpha/live/dashboard_config.yaml
```

Default DB paths:

```text
paper/live POD, no override -> alpha/live/state/<mode>/<pod_id>.sqlite3
incubation POD, no override -> alpha/live/state/incubation/<pod_id>.sqlite3
```

For incubation, a command without `--pod-id` and without explicit `--db-path` fans out across all enabled incubation PODs and aggregates the result. The old `alpha/live/incubation_state.sqlite3` file is legacy/manual only; pass it explicitly with `--db-path` if you need to inspect old shared rehearsal state.

Current PAPER transition override:

```text
pod_dv2_caspersky_account_paper_01 -> alpha/live/live_state.sqlite3
```

Keep that override until the current PAPER POD state is migrated out of the old shared DB. If you move a POD to its dedicated DB, update this config at the same time.

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

`serve` is a timing loop around `tick`. With `--pod-id`, it uses the POD-specific default DB path unless `--db-path` is supplied. In incubation, no `--pod-id` means the service fans out across enabled incubation POD DBs and the dashboard aggregates those PODs.

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

### Show Strategy Decision

```bash
uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id pod_dv2_01
```

This is read-only. It shows the latest strategy decision before broker sizing: signal time, submit window, target execution time, target weights, exits, metadata, and the linked VPlan status if one exists.

Show one exact decision:

```bash
uv run python -m alpha.live.runner show_decision_plan --mode paper --decision-plan-id 1
```

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

Quick glossary:

```text
post_execution_reconcile = proves fills and target positions after trading
eod_snapshot = records clean end-of-day cash, positions, and NetLiq
pod_state = latest trusted broker-backed sleeve state
broker_snapshot_cache = latest raw broker snapshot for the account
```

EOD state is mainly for the next decision's starting state and for cleaner backtest-reference comparison:

```text
equity_error_t = actual_eod_net_liq_t / reference_close_equity_t - 1
```

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
uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id pod_dv2_01
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id pod_dv2_01 --vplan-id 1
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id pod_dv2_01
```

Plain meaning:

```text
build -> review decision -> review order plan -> submit -> reconcile
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

### Validate Staged Releases Before Copying Them Into The Active Root

Each `serve` parses **every `*.yaml` under its configured release root**, including
disabled releases, before applying `--pod-id`. An invalid new CORE5/capsule YAML
can therefore stop an existing NDX/TAA serve that reads the same root. Keeping a
new release disabled does not protect the other serves from a parsing error.

Prepare new YAML files outside every active release root. Make an offline staging
copy of the complete root that the serves will read, add the proposed files there,
and remove replaced versions only from that staging copy. From the approved
checkout, use its installed Python environment and actual release validators:

```powershell
@'
import sys
from pathlib import Path
from alpha.live.release_manifest import (
    load_release_list, validate_release_manifest, validate_enabled_deployment_for_mode,
)

staged_root_path_obj = Path(sys.argv[1]).resolve()
if not staged_root_path_obj.is_dir() or not any(staged_root_path_obj.rglob("*.yaml")):
    raise SystemExit("Staging root is missing or contains no YAML releases; stop.")
release_list = load_release_list(str(staged_root_path_obj))
for release_obj in release_list:
    validate_release_manifest(release_obj)
    print(f"PASS: {release_obj.source_path_str}")
for mode_str in sorted({release_obj.mode_str for release_obj in release_list}):
    validate_enabled_deployment_for_mode(release_list, mode_str)
print(f"PASS: complete staged root ({len(release_list)} releases)")
'@ | .\.venv\Scripts\python.exe - C:\staging\alpha_releases
if ($LASTEXITCODE -ne 0) { throw 'Release validation failed; do not copy staged YAMLs.' }
```

This command reads YAML and runs the branch's validation functions. It opens no
pod database, contacts no broker, and does not sync data or enable a release.
The complete-root check catches duplicate enabled release/pod IDs and daily-pod
account conflicts within that staged root. Enabled CORE5 LIVE releases also
require their current account-bound qualification record. Require the final
`PASS: complete staged root` and exit code zero before copying the reviewed new
YAMLs into the active root. Revalidate after every staged change. This check is
configuration validation; enablement and deployment still require approval.

Each CORE5 or MR capsule pod must have its **own dedicated IBKR account/subaccount**
and full-account budget (`execution.pod_budget_fraction_float: 1.0`). Do not share that account
with NDX, TAA, another daily pod, or another independently managed strategy.
Inventory all running serves, scheduled tasks and broker clients, including
separate release roots: assign a **globally unique broker client ID to each
concurrent process connecting to the same TWS/IB Gateway session**. Check both
YAML `broker.client_id_int` (or flat `broker_client_id_int`) and command-line
`--broker-client-id` overrides.
Cross-serve client-ID uniqueness is an operator requirement; the release
validator does not enforce it. Separate `--pod-id` values do not isolate broker
connections that reuse the same client ID.

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

Keep the exact YAML fields used by the current release examples. This section explains the human meaning; [LIVE_TECHNICAL_REFERENCE.md](LIVE_TECHNICAL_REFERENCE.md) is the implementation reference.

## Incubation / Paper / Live

Incubation is for checking:

- strategy hosting through the live stack;
- clean pod-separated SIM cash, positions, and P&L;
- DecisionPlan/VPlan creation;
- SIM submit and reconcile flow;
- IBKR reference/open-price availability for MOO flows;
- multi-strategy rehearsal without blended paper-account positions.

Paper is probe-only. It is for checking:

- strategy hosting;
- scheduling;
- order-plan creation;
- submit plumbing;
- reconciliation behavior;
- IBC/TWS connection;
- account visibility;
- contract qualification;
- market data permissions;
- optional test-order acceptance/reject behavior.

Paper is not proof of real auction execution quality, and paper fills are not incubation accounting truth.

For MOO/MOC sleeves:

```text
backtest -> incubation rehearsal -> paper probe -> tiny live test -> scale slowly
```

For MOO rehearsal:

```text
signal from approved prior data -> VPlan -> IBKR open/reference price read -> SIM ledger fill -> reconcile/report
```

The VPlan sizing reference uses the same IBKR path as paper/live:

```text
auctionPrice (generic tick 225) -> reqMktData fallback -> reqTickers fallback
```

The SIM fill is still separate: open-policy rehearsal settles against the target-session IBKR `ticker.open` evidence. Paper fills are not imported into incubation accounting.

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

## IBKR Performance Shadow

This surface is read-only and deliberately separate from Overview. IBKR Flex
is the authority for each account's daily TWR; the dashboard also derives a
clearly labeled indicative multi-account diagnostic. That combined line is not
Fund TWR. It does not change orders, sizing, scheduler state, reconciliation,
`healthz`, or the existing Overview.

The Activity Flex Query must be XML, Account-by-Account, Breakout by Day, and
contain both LIVE accounts with exactly these fields:

- Account Information: `Account ID`, `Currency`.
- Change in NAV, Mark-to-Market: `Account ID`, `Currency`, `From Date`,
  `To Date`, `Starting Value`, `Ending Value`, `TWR`.

Add these secrets/settings to ignored `config.env`:

```text
IBKR_FLEX_TOKEN_STR=<token>
IBKR_FLEX_QUERY_ID_STR=<query id>
IBKR_FLEX_QUERY_NAME_STR=ALPHA_DAILY_TWR
ALPHA_IBKR_PERFORMANCE_DB_PATH_STR=C:\alpha\live_ops\ibkr_performance.sqlite3
```

Do not paste the token into commands, logs, git, screenshots, or the dashboard.
The sync reads it from `config.env`; the SQLite database stores only statement
data and a checksum.

Initial import from the manually downloaded historical XML:

```powershell
uv run python -m alpha.live.ibkr_performance_sync bootstrap --xml "C:\Users\User\Downloads\ALPHA_DAILY_TWR.xml"
```

If more than one historical XML file covers different ranges, import each file
once. Imports are idempotent. A different XML that overlaps stored rows is
refused unless the operator has reviewed the correction and adds `--replace`.

Verify locally:

```powershell
uv run python -m alpha.live.ibkr_performance_sync status --json
uv run python -m alpha.live.ibkr_performance_sync sync
uv run python -m alpha.live.ibkr_performance_sync status --json
```

Register the once-daily 06:15 ET task only after the manual sync succeeds. The
setup fails if Windows is not using `Eastern Standard Time`, so daylight saving
time cannot silently move the job away from New York time.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ibkr_performance_task.ps1
Start-ScheduledTask -TaskName AlphaIbkrPerformanceSync
Get-ScheduledTaskInfo -TaskName AlphaIbkrPerformanceSync
```

The task uses `StartWhenAvailable`, refuses overlapping instances, retries a
failed run up to three times at 15-minute intervals, and permits up to ten
minutes per run. Flex Activity data updates once daily; the command sends one
`SendRequest` and polls only `GetStatement` for up to two minutes while IBKR
generates that statement.

Failure policy:

- malformed XML, unknown/missing account, non-USD data, missing market session,
  or account-to-Pod mapping drift fails loud;
- the prior valid SQLite rows remain unchanged;
- the Performance page shows the problem only inside the Shadow surface;
- LIVE controls, break-glass tools, Overview and `healthz` remain available.

The indicative-composite formula is:

```text
AdjustedBase_i,D   = EndingNAV_i,D / (1 + IBKR_TWR_i,D)
CompositeReturn_D  = sum(EndingNAV_i,D) / sum(AdjustedBase_i,D) - 1
LinkedDiagnostic   = product(1 + CompositeReturn_D) - 1
```

IBKR account TWR already includes the economic effect of dividends, interest,
fees and trading P&L while adjusting that account for external cash flows. The
combined line is only an indicative Shadow diagnostic. A changed adjusted base
blocks obvious flow-sensitive dates, but offsetting intraday deposits and
withdrawals can still cancel in daily totals. Exact consolidated Fund TWR
cannot be reconstructed from this Account-by-Account daily report alone. The
official Pod rows remain useful; the combined line is not eligible for Overview
promotion without an official consolidated IBKR return or a separately approved
fund-accounting contract.

## Logs And State

Live state:

```text
alpha/live/live_state.sqlite3                          # default without --pod-id
alpha/live/state/<mode>/<pod_id>.sqlite3               # default with --pod-id
alpha/live/dashboard_config.yaml                       # dashboard DB override map
alpha/live/incubation_state.sqlite3                    # legacy/manual shared incubation DB only
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

Live vs Backtest dashboard artifacts:

```text
results/live_reference_compare/<mode>/<pod_id>/<timestamp>/
```

## Short Mental Model

```text
One deployment runs one client. Each sleeve decides independently, trades its own IBKR account, and reconciles from broker truth.
```
