# Norgate API: Git-managed background launcher

TL;DR: the CORE5 exporter/store support is already in commit `16dd21c`.
The tracked task launchers make the background API startup reproducible.
Pulling alone does not restart Python or retarget an existing scheduled task.

## Correct checkout

As verified on 2026-09-16, the VPS has two separate Git checkouts:

| Role | Checkout |
|---|---|
| Trading and dashboard | `C:\Users\Administrator\Documents\PRODUCTION\alpha_super` |
| Norgate API | `C:\alpha\norgate_server\app` |

Update the Norgate checkout for this launcher. Preserve its existing `.venv`,
ignored `config.env`, client state and cached snapshots. The task requires these
four nonempty keys in that config file: `NORGATE_API_TOKEN`,
`NORGATE_SERVICE_ROOT`, `NORGATE_API_HOST`, `NORGATE_API_PORT`. File values override
inherited environment values. Unlike the direct API, this task requires explicit
host/port settings. The history start date remains `1990-01-01`.

```text
[Git pull in Norgate checkout] --> [Tracked launcher + private config.env]
                                               |
                                    explicit task restart
                                               v
                                   [New API PID + run logs]
```

## Updating and the one-time hotfix

Arrange an API maintenance window without active exports/downloads before
changing loaded source files. The API is shared: its downtime can delay client
data sync. Stop only the owned API task and process branch; leave trading,
dashboard and Norgate Data Updater running.

**Stop-ScheduledTask alone may leave Python children alive.** Record executable,
command line, parent PID, creation time and session. The legacy branch uses the
server `.venv\Scripts\python.exe` and
`C:\alpha\norgate_server\backups\core5_20260916T201607Z\run_api.py`;
the tracked branch uses `-m scripts.run_norgate_server_task`. Recheck identity
before terminating only the verified API root and descendants. Require no owned
API branch and no listener on the configured port before starting another copy.
If ownership is unclear, investigate before terminating a process.

In `C:\alpha\norgate_server\app`, inspect `git status --short`, confirm branch
`main`, then `git fetch origin`. For a clean checkout use
`git pull --ff-only origin main`, checking each command's exit code.

On 2026-09-16 the checkout had a two-file hotfix in
`scripts/export_norgate_snapshot.py` and `data/norgate_snapshot_store.py`.
Backups are in `C:\alpha\norgate_server\backups\core5_20260916T201607Z`.
If these are still the only changed files, first verify that both working files
match the fetched, reviewed main:

```powershell
git diff --exit-code origin/main -- scripts/export_norgate_snapshot.py data/norgate_snapshot_store.py
```

Only after exit code 0, preserve those two files with a path-specific stash,
then pull. Check each command succeeds before continuing:

```powershell
git stash push -m 'Preserve CORE5 Norgate hotfix' -- scripts/export_norgate_snapshot.py data/norgate_snapshot_store.py
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

Keep the stash; do not pop it, since its verified content is upstream. If other
changes exist or the comparison differs, review them first. Do not reset/clean
the checkout or overwrite private config.

## Point the existing task at Git (one time)

The task `AlphaNorgateSnapshotApi_Core5_20260916` initially points to
`C:\alpha\norgate_server\backups\core5_20260916T201607Z\run_api.ps1`.
After stopping its owned API branch, update that existing task in Windows Task
Scheduler with the following settings; do not create a duplicate task:

- Program: `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`.
- Arguments: `-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\alpha\norgate_server\app\scripts\run_norgate_server_task.ps1"`.
- Start in: `C:\alpha\norgate_server\app`.
- Principal: the existing Norgate Windows user, highest privileges, interactive
  logon ("Run only when user is logged on").
- No triggers or automatic retry; no execution time limit; do not start a new
  instance while one is running.

Keep the original task definition and backup directory for rollback. The user
must remain logged on; disconnecting RDP differs from signing out. This task
does not provide reboot recovery or start/update the vendor application.
Future pulls need an explicit API restart, without another task migration.

## Verify an explicit start

Start the task once. The wrapper waits for Python and returns its exit code.
New stdout/stderr files appear under `results\logs\norgate_api` in the checkout.
Verify the new PID/creation time/session, correct checkout, profile list including
`norgate_eod_core5`, no startup errors, and `GET /healthz` returning 200.
Health alone does not establish vendor availability or current snapshot dates.
Also confirm existing clients retain authenticated access to cached snapshots.

CORE5 sync must use its dedicated `core5_incubation_local` client ID: requirements
replace that client's profile list. Do not reuse another live client's ID for a
CORE5-only request. The task does not run sync, doctor, incubation or trading.
The existing visible `start_norgate_server.cmd` retains its separate behavior.

## Verification boundary

Tier 3: live data-service launcher/runbook. Order timing, sizing, reference
prices, strategy logic, data adjustments, state/SQLite schemas, dashboard fields
and released YAMLs are unchanged. Windows path handling, hidden launch, separate
logs, config precedence and exit propagation have focused tests using a mocked
API/process launcher. Actual task migration and restart require operator
verification after pulling; these tests do not establish VPS deployment success.
