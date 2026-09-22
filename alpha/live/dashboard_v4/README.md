# Dashboard V4 — LIVE operations

Native Flask/Jinja implementation of Mockup D's operator pages, following
`docs/plans/DASHBOARD_V4_HANDOFF.md`. Select a Pod from the sidebar or Overview
to open `/pods/<pod_id>`. Positions, Performance, Activity and System health are
also available. Tools lists scoped commands for copying. PAPER and INCUBATION
are deferred.

## Run beside V3

From the repository root:

```powershell
.venv\Scripts\python.exe -B -m alpha.live.dashboard_v4 --demo --port 8084
```

Open `http://127.0.0.1:8084/`. Demo mode uses fixed, clearly labelled synthetic
data. Each demo provider builds isolated temporary SQLite databases once through
`LiveStateStore`; Pod and fill evidence use the production read-only readers.
The server advances a simulated clock from its fixed scenario time. It does not
load `config.env`, a real account database or a broker session.

To use the existing local saved-data configuration:

```powershell
.venv\Scripts\python.exe -B -m alpha.live.dashboard_v4 --read-only --port 8084
```

Optional paths: `--releases-root`, `--config`, `--performance-db`.
`--skip-env-file` disables the normal `config.env` load. The default bind is
localhost. There is no flag that enables trading or other executable actions.
This command starts a local development server; no service deployment or V3
cutover is part of this phase.

## Tools

`/tools` lists saved-state inspection and commands that can change state in two
panels. Selecting an enabled LIVE Pod scopes copyable PowerShell commands to its
current release and configured state path. Required arguments are entered before
copying. Copying does not run a command. Pod and System pages link to Tools.
Run copied commands from the repository folder. `submit_vplan` requires an
explicit `--vplan-id`, including when auto-submit is disabled. The existing
parameter field is prefilled only when the Pod reader verifies a current ready
plan for the selected release and account; otherwise the ID must be entered.
The runner retains its existing execution-window, readiness and ownership checks.
The chosen ID is also bound to the synthetic preview and its one-use confirmation.
A `?tool=` link
opens and scrolls to that tool once; status refreshes do not move the page.
Diagnostics remain copyable when the saved database is unavailable, provided the
current release identity is verified. No credentials or environment-file contents
are rendered.

Production execution is not connected. There is no production enable-actions flag.
For a reviewable, isolated demonstration only:

```powershell
.venv\Scripts\python.exe -B -m alpha.live.dashboard_v4 --demo --demo-tools --port 8114
```

This adds simulated previews, a 120-second one-use confirmation, and in-memory
results for Tick, Submit, Reconcile, EOD, Reference comparison and Manual order.
Confirm opens a native "Are you sure?" dialog with the action, Pod, account and
preview details. Cancel sends no execution request and keeps an unexpired preview.
Accept rechecks the selected tool and expiry before sending the one-use request;
an unavailable dialog never counts as consent. The server also requires the
explicit browser-confirmation field.
The `/api/demo-tools/` routes accept only the synthetic provider. They never invoke
the trading runner, connect to IBKR, or write trading state. Demo Activity labels
these records as simulated; restarting the demo clears them. `--demo-tools`
without `--demo` is rejected. Connecting these actions to production requires
separate, specific owner approval and is not implemented here.

The header and sidebar refresh every 15 seconds through `/tools/status`; this
does not replace typed fields, previews, or results.

## Data and display contract

The shared desktop/mobile market header shows `Premarket` (04:00 to core open),
`Market open` (normally 09:30–16:00), `Post-market` (core close to 20:00), or
`Market closed`, all in New York time. It uses the existing XNYS session calendar
for holidays, DST and early core closes. The extended session ends at 17:00 on
early-close days, following [NYSE Arca hours](https://www.nyse.com/trade/hours-calendars).
The phase and countdown refresh with the existing 15-second status refresh.
This is a scheduled equity-session indicator, not a live halt feed, broker
availability check, or permission to trade. It changes no scheduler or order gate.

```text
[Current LIVE release metadata] ----> [Scoped saved operations]
                                              |
                                  [7-step cycle + health]
                                              |
[Saved IBKR reports + bindings] ---> [Canonical V3 finance]
                                              |
                              [Overview or Pod response / 15s]
```

- LIVE targets are filtered before runtime/state acquisition. Release/config
  validation remains owned by the existing readers; no PAPER/INCUBATION runtime
  or cash DB is read.
- Pod plus account identity must match uniquely. An older unresolved VPlan is
  not replaced visually by the newer decision.
- The cycle is Data, Decide, Plan, Submit, Fill, Reconcile, EOD. Scheduled times
  remain planned times; actual timestamps are shown only when saved evidence
  supports them. Display uses New York time, to seconds. Dates stay session dates.
- Missing DB is a red attention item. Normal scheduled waiting and healthy
  monthly idle are distinct from late, failed and unknown. EOD runs independently
  of whether the strategy traded; a missing prior EOD identifies the missing day.
- Healthy monthly waits show `Waiting` and `No trade scheduled`. Current
  `Next` also considers the next EOD and decision window using the existing
  exchange-calendar and V3 schedule helpers. Calendar forecasts are marked
  `Scheduled`; decision times say `after` close and `when data is ready`, with
  no readiness countdown. Missing timing is `Time unknown`. These projections
  do not prove scheduler liveness or alter the selected historical cycle.
- Scheduler evidence is read from bounded tails of the configured event log
  and its per-Pod LIVE trace files, including fixed rotation names. Nothing
  probes processes or connects to a broker. Healthy schedulers add no widget.
  A problem uses the existing System light, Pod mark, Now/Next and one attention
  row; previous order issues remain in that row. Historical cycle results keep
  their original meaning. Unverified automation changes future Planned marks
  to Unknown, without downgrading the Pod merely because logs are unavailable.
- Sleep liveness uses the actual event write time plus its recorded sleep,
  including intentional 3600-second sleeps. Display thresholds are 60 seconds
  overdue (review) and 300 seconds overdue (action). These are heuristics:
  `Scheduler not responding` does not claim the process is confirmed stopped.
  Recent one-shot `run_once` activity alone cannot prove a continuous service.
  Activity without a wake promise expires after 120 seconds. Unattributed
  global errors are ignored; per-Pod errors use generic safe wording.
  A long data sync before the next scoped scheduler event can also overrun the
  promised wake. Its global sync event has no mode identity, so it cannot safely
  certify LIVE liveness. The warning asks for a check, never a blind restart.
- Missing or unreadable scheduler evidence weakens the System light to Unknown
  unless an existing failure is more severe. It does not invent a Pod action.
  A verified live scheduler is mentioned only inside an existing Pod issue,
  with holding/data/reconcile wording from its recorded state. No System health
  page, watchdog change, notification, scheduler control or auto-restart is added.
- The issue can expose a selectable, quoted, Pod-scoped `next_due` command.
  The dashboard never executes it. Running that command may update local release
  metadata; its copy text says so. No restart command is fabricated from unknown
  process overrides, and the operator is told to check the existing process.
- Scheduled timestamps mean eligibility. Submit stays Working through 60 seconds
  after eligibility; Reconcile and EOD through 30 seconds. Reconcile eligibility
  already includes the scheduler's 300-second grace. These display allowances
  use scheduler default constants, not inspected process/CLI overrides. Late
  requires a saved assessment after the allowance; a cached earlier assessment
  cannot by itself prove a missed poll. Explicit failures remain visible.
- Source validity is 120 seconds, inherited from V3. Server and browser both
  enforce it. Transport failure or expiry clears operational state to Unknown
  and preserves the last successful update. Restored pages reacquire evidence.
- Account value is finalized IBKR NAV. Day is dollar P&L with TWR beneath it;
  Month/Year use current ET calendar periods and show TWR with P&L beneath.
  Canonical V3 accounting retains flow adjustments, coverage, ownership and
  finalization checks. Changing chart range does not change these tiles.
- The main charts show selected-period cumulative return, starting at 0%, rather
  than account NAV. Overview uses the canonical calculated client path and its
  explicit end-of-day cash-flow convention; the existing verified single-account
  fallback remains. Pod charts use official IBKR account TWR independently of
  the client method. Linking stays in the reporting layer: `R = product(1+r) - 1`.
  Missing return evidence withholds the curve, never substitutes NAV growth or
  averages Pods. Account value remains a separate dollar tile. Allocation uses one
  same-date source: validated broker EOD, or complete finalized IBKR values.
  Unknown cash is not estimated; no residual
  "Free cash" is manufactured.
- Production has no executable action or export routes. Non-read methods return
  403 before acquisition; only the explicitly enabled synthetic demo allows
  its fixed preview, confirmation and cancellation requests.
  Fonts, HTMX and scripts are served locally. HTMX history
  storage is disabled; period changes are normal page navigation.

## Evidence limits

V3 summary counts and completed status do not prove full fills. V4 additionally
reads the selected LIVE VPlan, plan rows, orders, ACKs and executions in one
SQLite `mode=ro` transaction. The canonical request builder preserves separate
entry/exit legs, even for the same asset. For every request i, completion requires
`abs(sum(signed execution shares_i) - requested shares_i) <= 1e-9`.
Release, Pod, account, DecisionPlan and VPlan identities must agree. Duplicate,
future, partial, ambiguous retry or changed evidence cannot prove completion.
The summary assessment time is the read cutoff. Unsupported completion remains
Unknown; a verified fill shows the latest actual execution time and `N of N filled`.

A completed plan may contain terminal `Filled` order summaries whose requested,
filled and remaining quantities are all zero. For that
exact case, the dashboard checks quantities against the canonical saved requests.
It requires a unique matching broker ACK for every order, valid ACK times, and
unique nonempty execution IDs with signed quantities fully covering each affected
request. Conflicting nonzero summaries, partial quantities or missing identities
remain unverified. No saved execution data is repaired or written. A failed proof
shows a short, safe reason above the raw Fills table; passed reconciliation remains
separate from fill verification.

No orders requires finite zero intent in both aggregate and per-leg plan rows,
with no contradictory orders, fills or ACKs. Reconciliation remains independent.
A completed DecisionPlan without a VPlan is not a verified LIVE no-order cycle.
Actual decision/plan/submit times remain unavailable in the summary; they are not
manufactured from scheduled times. Pod detail additionally reads saved decision
and plan creation times.

## Pod detail

The selected cycle sits above financial context: history picker, seven steps,
then Plan vs actual, Decision, Orders, Fills, Reconcile and Events tabs.
Failed cycles show their evidence and a short review instruction.
The Pod heading always shows the current Pod state, including non-cycle gates
and database failures. The selected cycle has a separate badge and verdict.

- History reads up to 60 recent cycles, plus the selected or unresolved cycle.
  Saved historical LIVE releases must match the current release's owner, Pod
  and account. The selected release supplies its own calendar and data profile.
  Each detail table is bounded to 2,000 records; an exceeded limit is Unknown.
- A SQLite `mode=ro` transaction reads DecisionPlan, VPlan and their scoped
  children. Current operating gates are overlaid only when decision, plan and
  release identities match. Historical views are labelled Saved cycle.
- The selected cycle's Data date comes only from its saved decision snapshot.
  Current data readiness stays in the current header and Overview, even when
  the latest monthly cycle is several weeks old.
- Plan vs actual preserves each canonical order request in three columns:
  Symbol, Position, and Broker = model. The signed requested order sits above
  Before → After; it is not presented as the actual filled amount. Before uses the
  selected VPlan's broker snapshot; After uses that cycle's post-execution
  reconciliation, never today's position cache. Verified fill quantities and
  prices stay in Orders; individual executions stay in Fills. Display fill price is
  `sum(abs(fill shares) * fill price) / sum(abs(fill shares))` per broker order.
- ACK rows require a matching request, asset and broker order, positive broker
  response, recorded response time and `broker_acked` status. Contradictions
  cannot leave Submit green. Stale or failed refreshes clear step and table
  status together. Slow detail reads cannot extend the 120-second source life.
  Completed plans with legacy `not_checked` ACK history remain Unknown rather
  than claiming submission is late. A completed Submit may show the last saved
  ACK timestamp, explicitly labelled ACK, only when all timestamps are later
  than the planned boundary; equality may be a broker-refresh fallback.
- Value, Day, Month and Since start use the selected Pod's official account
  report and existing accounting checks. The chart uses that account's official
  return path; Day P&L is flow-adjusted and is not a NAV difference. These panels
  keep their own dated sources when browsing historical trading cycles.
- Positions show saved quantities and their own timestamp. Cash has its own
  financial date. Existing sources do not prove closing marks for every symbol,
  so symbol values, weights, target/New markers are omitted.
- Unconnected Slip bps, Files, Trade sheet and Live vs backtest
  controls are hidden. The V3 exporter writes state and is not used. Events
  shows the selected VPlan's broker order events, with symbols and newest first.
- The ET clock ticks locally each second, independently of operational health.
  Selecting text does not delay refreshes. A selection within one uniquely
  identified field or evidence table is restored only if its complete text and
  page/Pod/cycle/tab/period identity are unchanged. Changed or ambiguous content
  drops the selection. This never extends the source's freshness lifetime;
  stale Next labels say `Not current`, including calculated forecasts.
  Historical pages still poll so current Pod warnings stay up to date.

## Activity

`/activity` shows saved LIVE events and mode-less system events, newest first,
grouped by New York date. It defaults to seven calendar days; Load older extends
the same view to 14, 30 and 90 days. Pod/type filters, search, technical codes and
the late/failed filter work on the returned rows. Each row has inline evidence;
an exact saved cycle identity also links to the Pod evidence tab.

The reader validates one enabled LIVE owner and unique Pod/account identities
before opening files. It scans backwards in chunks through the configured event log,
`live_critical_events.jsonl`, their ten numbered rotations, and
`operator_journal.jsonl`, stopping at the selected date boundary or safety limits.
Routine scheduler polls and benign data-sync skips do not consume the material
event budget. Each request is limited to 23 fixed files, 128 MiB, 200,000 lines,
4,000 distinct material records and a cooperative five-second scan deadline;
records above 32 KiB are rejected. The newest 1,000 events feed the presenter,
which displays at most 500 top-level rows. Limits and
unreadable sources are shown, with the actual scanned log span when incomplete;
Load older cannot recover records outside retained files or scan limits. Raw
paths, account IDs, free-form errors and unapproved payload fields are withheld.
No dynamic trace-directory scan or broker call is added.
An in-memory cache reuses only unchanged files with matching scope, selected
date boundary, identity, size and modification metadata. It holds at most 32
file entries and 4,000 events. Appends/rotation invalidate affected entries;
incomplete, invalid or future evidence is not cached. No sidecar file is written.
Coverage is based on the main-log scan, not an older isolated critical record.
Retained-history limits, rotation gaps and scan/display limits remain explicit.

Healthy cycles reuse the Pod page's saved ACK, fill and reconciliation proof.
Only matching Pod, release and decision/plan identities can fold routine success
events into a cycle. Failed, late, alert and operator events stay separate. Cycle
reads are capped at 32 Pods, 12 reads per Pod and 60 reads per response. A stale
current header does not alter independently verified historical facts. Planned
times do not become actual event times; operator requests do not imply completion,
and notification delivery needs an explicit saved receipt.

The Operator category includes saved V3 dashboard requests and manual-order
request/submission/failure events. CLI commands are not comprehensively recorded
in the operator journal. A completed manual submission is not proof of a fill.
The activity log has no per-alert delivery receipts, so the demo does not invent
delivered-alert rows. The separate watchdog run receipt supplies System health
only. Unknown routine events fold into per-day,
per-Pod Other events groups with each original record available; warnings,
failures, alert and operator families remain individual rows. Missing descriptions
never imply success. PAPER and INCUBATION records remain excluded.

The page polls every 15 seconds and preserves filters, search focus and expanded
rows. A browser-local last-looked timestamp is scoped to the saved-data source
and enabled LIVE identities. It records successful visible observations, never
an unavailable/stale refresh or an earlier timestamp from a slower tab. First
visit says Recent activity; subsequent visits count only events after that
browser's prior visit. This is a navigation aid, not an acknowledgement or an
audit receipt. Demo events are synthetic and do not write operational logs.

V4 retains the existing host-disk warning thresholds (75% review, 90% action).
The header names a disk problem explicitly, without exposing paths. Disk-only
warnings do not claim saved trading evidence is missing. Tests of unrelated
status behavior substitute a fixed disk reading; dedicated health tests exercise
the real rollup at both thresholds, probe errors and stale operations.

## System health

`/system` groups current assessments into always-running services, scheduled
work, and data/storage. Below those checks, each enabled LIVE Pod shows its own
scheduler, last observed activity, promised wake, saved data, broker read and EOD.
What should run lists the current LIVE release metadata, including disabled
releases. Accounts are masked. The shared System light links to this page.

All V4 pages use the same System assessment, including the lightweight
Performance status refresh. The reader adds bounded reads of the saved watchdog
report/receipts, file metadata and bound Flex coverage/sync-attempt metadata;
it does not connect to a broker, scan full logs, build financial
reports, inspect Windows task registration, or start any operation. Scope is
validated before runtime reads: one owner, unique enabled LIVE Pod/account
identities, and matching current releases. Disabled-only metadata causes no
runtime reads. Missing, contradictory, stale or future evidence cannot be green.

Checks without a supported current observation say `Not checked here`, remain
neutral through refresh failures, and do not affect the verdict. This applies
to the broker connection and FRED feed: saved broker reads and decision dates
remain visible as history. Missing, stale or invalid evidence for a supported
check remains Unknown/Late and affects the verdict. The summary names up to
three problems, worst first; mobile shows problem rows before other checks.

A fresh, scoped watchdog report proves `Report saved`; the report can also be
produced manually and is written before notification delivery. The watchdog now
atomically writes a final sibling receipt (`ops_report_latest.run.json` by
default) after its existing heartbeat attempt. The receipt carries the actual
completion time, exact LIVE scope, canonical report SHA256 and heartbeat result.
Only a matching fresh pair proves `Run completed`. Missing legacy receipts are
neutral for the ping; malformed, mismatched or stale receipts are not accepted.
Receipt failure leaves prior alert/ping behavior and exit codes unchanged.
`Fail signal sent` means the failure heartbeat was delivered successfully.
It does not mean the saved report was healthy. This dashboard cannot detect a
dead VPS while running on that VPS; the already-configured external dead-man
service must detect missing pings. No external monitoring service is configured
by this change.

Alerts show the saved count of LIVE alerts pending retry, never invent a
delivery acknowledgement from an empty map. Old notification files without a
pending map remain Unknown. Flex uses the existing account binding, daily
coverage and `sync_attempt` receipt; a failure can be shown even before the first
import. Coverage follows the existing daily 08:00 ET rule, including weekends,
so a newly closed market does not immediately create a false Late. No financial
history or return calculation is rebuilt by these health reads.

The Event log check uses the existing scheduler idle ceiling (60 minutes) and
60-second wake allowance; the watchdog report age policy is 900 seconds, not a
claim about its task schedule. Scheduler evidence uses bounded 64 KiB log tails
and the exact per-Pod trace file. Busy shared logs may evict that Pod's event;
without its trace evidence the state is Unknown. Reason codes may contain digits;
an unsupported reason label is omitted without hiding a valid failure event.

Pages refresh status every 15 seconds. Failed refreshes and the existing
120-second source expiry clear current status labels and icons while retaining
expected configuration and unsupported neutral rows. Time spent reading service
evidence consumes that same lifetime; changed releases invalidate current claims
across all pages. PAPER and INCUBATION health remain deferred, as elsewhere in V4.
Status JSON and Diagnostic JSON download only the sanitized view model, never
raw reports, paths, account numbers, webhook addresses or exception messages.
No export file is written on the server. The demo provides explicit synthetic
service evidence and a fixed 78% disk warning; it never probes the host disk or
falls back to real service sources.

## Verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_dashboard_v4_cycle.py tests/test_dashboard_v4_evidence.py tests/test_dashboard_v4_finance.py tests/test_dashboard_v4_return_chart.py tests/test_dashboard_v4_routes.py tests/test_dashboard_v4_pod.py tests/test_dashboard_v4_pod_data.py tests/test_dashboard_v4_pod_finance.py tests/test_dashboard_v4_pod_integration.py tests/test_dashboard_v4_pod_review.py tests/test_dashboard_v4_pod_demo.py tests/test_dashboard_v4_next_operation.py tests/test_dashboard_v4_selection_markup.py tests/test_dashboard_local_workspace.py --capture=sys -p no:cacheprovider -q
node --test tests/dashboard_v4_refresh.test.cjs
uv run python -m pytest tests/test_dashboard_v4_system.py tests/test_dashboard_v4_system_data.py tests/test_dashboard_v4_system_routes.py -q
```

Tier 3. Independent parity, failure-mode and coverage reviews found and drove
regressions for idle/wait classification, missing DB, alert reasons, unsupported
fill completion, prior-session EOD, ET timestamp fallback, private browser history
and stale-page recovery. Follow-up reviews cover quantity proof, poll allowances,
ACK failures on no-order claims, alert titles and unavailable optional readers.
Pod reviews additionally cover historical release identity, selected-cycle
provenance, ACK contradictions, stale table states, and current-failure retention
across tab navigation. Production-format temporary databases verify that GETs
and refreshes, including historical Pod tabs, do not change file bytes or
modification times.

Visual checks compare the native page with Mockup D at 1440px, and verify layout
at 768px and 390px. Browser checks cover period navigation and failed-refresh
state, Pod evidence tabs and history. No production data, broker session or VPS
is used for these checks.

System health adds strict scope/release, report age, per-Pod timestamp, missing
receipt, future/corrupt evidence, locked-Flex recovery, masked export and
disabled-only metadata tests. A final response clock consumes time spent reading
cycle and scheduler evidence; the browser lifetime is capped by both workspace
and auxiliary assessment ages. Independent parity, failure-mode and coverage
reviews drove regressions for changed releases, historical FRED dates, source
expiry, contradictory market-data dates and duplicate release selection keys.
At 768px and 390px, health tables become labelled cards. Browser verification
includes a stopped local demo server: current claims become Unknown on failure
and recover when polling succeeds again.

Activity follow-up: a read-only scan of the existing 41.8 MiB local log, using
a synthetic scope that excluded all real Pod records, reached the seven-day
boundary after 6 MiB in 0.084s, and the 90-day boundary after 29.5 MiB in 0.363s.
An unchanged repeat used the memory cache in 0.001s. Only aggregate scan metrics
were inspected; this does not verify real Pod coverage or VPS performance.

## Live-impact checklist

- Order timing, next-open execution and scheduling: unchanged.
- Sizing, amount/target semantics and capital allocation: unchanged.
- Reference price sources and close/open boundaries: unchanged.
- Existing state, pickle, SQL schema, configuration and released YAML: unchanged.
- Existing logging fields and V3 routes: unchanged.
- Windows paths/locks: bounded binary log tails close files after each read;
  missing, corrupt, rotated or unavailable evidence fails to Unknown. Existing
  read-only DB readers retain their file-preservation tests.
  The new UI runs as a separate local process. No production restart or deployment
  is performed.
- Dashboard changes are confined to this package and its tests. No V3 code is
  modified. The watchdog adds only a final atomic JSON receipt and two stdout
  receipt-status fields; its notification order, heartbeat calls and exit codes
  are unchanged. Existing receipt/state/report consumers remain compatible.
