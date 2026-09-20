# Dashboard V4 — LIVE Overview and Pods

Native Flask/Jinja implementation of Mockup D's Overview and Pod pages, following
`docs/plans/DASHBOARD_V4_HANDOFF.md`. Select a Pod from the sidebar or Overview
to open `/pods/<pod_id>`. Positions, Performance, Activity, System health and
Tools pages remain disabled. PAPER and INCUBATION are deferred.

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

## Data and display contract

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
- No action/token/export/executor routes are registered. Non-read methods return
  403 before acquisition. Fonts, HTMX and scripts are served locally. HTMX history
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
- Unconnected Slip bps, Files, Trade sheet, Tools buttons and Live vs backtest
  controls are hidden. The V3 exporter writes state and is not used. Events
  shows the selected VPlan's broker order events, with symbols and newest first.
- The ET clock ticks locally each second, independently of operational health.
  Selecting text does not delay refreshes. A selection within one uniquely
  identified field or evidence table is restored only if its complete text and
  page/Pod/cycle/tab/period identity are unchanged. Changed or ambiguous content
  drops the selection. This never extends the source's freshness lifetime;
  stale Next labels say `Not current`, including calculated forecasts.
  Historical pages still poll so current Pod warnings stay up to date.

## Verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_dashboard_v4_cycle.py tests/test_dashboard_v4_evidence.py tests/test_dashboard_v4_finance.py tests/test_dashboard_v4_return_chart.py tests/test_dashboard_v4_routes.py tests/test_dashboard_v4_pod.py tests/test_dashboard_v4_pod_data.py tests/test_dashboard_v4_pod_finance.py tests/test_dashboard_v4_pod_integration.py tests/test_dashboard_v4_pod_review.py tests/test_dashboard_v4_pod_demo.py tests/test_dashboard_v4_next_operation.py tests/test_dashboard_v4_selection_markup.py tests/test_dashboard_local_workspace.py --capture=sys -p no:cacheprovider -q
node --test tests/dashboard_v4_refresh.test.cjs
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
- Changes are confined to this package and its tests. No V3 code is modified.
