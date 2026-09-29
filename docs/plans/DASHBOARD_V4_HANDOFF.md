# Dashboard V4 — implementation handoff

Date: 2026-09-19. Base: `main` @ `a6f6e97`. Author of the design: Claude, with the owner, over one design
session (mockups A → B → C → D). **The owner approved Mockup D as the V4 design.** No dashboard code was
changed. This document hands the approved design to the implementer.

This is a plan. It grants no permission to deploy, to enable actions, to run broker operations, or to change
the live engine. `AGENTS.md` and the verification policy in `docs/ai/PROJECT_GUIDE.md` apply to all work here.

## 1. What to build, in one paragraph

Rebuild the operator dashboard as **V4**: one light, sharp console for a solo operator. Goals, in the owner's
words: operator comfort, fast understanding, minimal clutter, easy debugging. One shell instead of V3's two
(client workspace + "Advanced"). Six pages plus one page per pod. One cycle component used everywhere.
Very simple English, short labels. V4 is a **presentation layer over data V3 already has**, plus a small list
of new read-only calculations (section 8).

## 2. Where the approved design lives

| What | Where |
|---|---|
| Final mockup, clickable, static HTML | `C:\Users\User\Documents\workspace\_mockups\CLAUDE_MOCKUPS\v4d\` (outside the repo since 2026-09-28) — start at `index.html` |
| Stylesheet to reuse | `_mockups/CLAUDE_MOCKUPS/v4d/ops.css` (class-based, no build step, no JS) |
| Fonts | already vendored: `alpha/live/dashboard_v3/static/fonts/` (IBM Plex Sans + Mono). `v4d/fonts.css` is for local preview only |
| Mockup generator | `_mockups/CLAUDE_MOCKUPS/v4d/_build/` (`python build.py`); page bodies in `pages_d.py`, `pages_calm.py`, `perf_c.py`, icons in `icons.py` |
| Same screens on a canvas (private to the owner) | https://claude.ai/artifact/VLDyQVARmu28hVrtcyy7LM |
| Earlier rounds, history only | `_mockups/CLAUDE_MOCKUPS/v4` (A), `v4b` (B), `v4c` (C) — do not implement these |
| Owner's operations guide (source of truth for Tools) | `output/pdf/alpha_super_operations_quick_guide_en.pdf` |
| Local preview | `.claude/launch.json` → `v4-mockup` (static server on `../_mockups`, port 8790) → `/CLAUDE_MOCKUPS/v4d/index.html` |

Mockup pages: `index.html` Overview · `pod.html` pod on track · `pod-issue.html` pod needing action ·
`positions.html` · `performance.html` (Portfolio level) · `performance-pods.html` (Pods level) ·
`activity.html` · `system.html` (System health) · `tools.html` · `cycles.html` (every cycle state, reference
sheet) · `mobile.html` (phone Overview) · `index-scheduler-stopped.html` (the Overview when one pod's scheduler
stopped — added 2026-09-20, see 6.9).

All numbers, names and accounts in the mockup are **synthetic**. They show layout and wording, not contracts.
`ops.css` is cumulative across rounds A→D; prune rules the final templates do not use (for example `.drawer`,
`.badge`, `.pods5`, `.crumb`).

## 3. Guardrails (hard rules)

1. **Build beside V3, not in place.** New package `alpha/live/dashboard_v4/`. V3 keeps running and stays the
   operator's tool until V4 passes acceptance. Reuse V3's data modules by import; do not fork data logic.
2. **Presentation and read-only view-models only.** Do not touch the live engine: runner, scheduler, order,
   sizing, reconcile, reference-price, state/SQLite schemas, release YAMLs, log fields.
3. **Read-only stays the default**, exactly as V3: server-side blocking of POSTs, previews, notification
   sends and file-generating GETs under `--read-only`; actions only with `--enable-actions`.
4. **No new executable actions.** V4 may run only what V3 already runs: `actions.py::SUPPORTED_ACTION_NAME_LIST`
   (`tick`, `submit_vplan`, `post_execution_reconcile`, `eod_snapshot`), the separate Live-vs-Backtest run
   route (`compare_reference`) and the manual ticket — with the same preview → one-use confirm (120 s) →
   journal flow. Every other command on the Tools page is **copy only**. Recommended first release: copy only
   for everything.
5. **Unknown is not green. Old is not green.** A stale or missing source renders as Unknown, never as Done.
6. **Money is never blended across modes.** LIVE, PAPER and INCUBATION totals are never added together.
7. **One VPS = one client** (unchanged). No client directory or switcher.
8. Same stack: Flask + Jinja + HTMX, no Node, no build step, vendored assets, no outbound requests.
9. All wall-clock times in **ET with seconds** (`filters.py`: `MARKET_TIMEZONE_OBJ`, `CLOCK_DATE_FORMAT_STR`).
   "ET" is stated once per page, not in every cell.
10. Never invent a value. If a number cannot be computed from saved evidence, show "—" and keep the row.

## 4. Structure

**Side menu:** Overview · **Pods** (a heading with its symbol, *not a page*; the pods of the selected mode are
nested under it, each with a status mark, name and D/M cadence letter) · "Book" label · Positions ·
Performance · Activity · System health · Tools.

**Top bar, one row:** mode switch `LIVE | PAPER | INCUBATION`, each segment carrying the worst status mark of
that mode · market status with countdown · a single light "System OK · Data <date>" linking to System health ·
`READ-ONLY` / `ACTIONS ON` tag · clock `HH:MM:SS ET` with date and "updated HH:MM:SS".

**Rule of scope:** everything on screen belongs to the selected mode. Two exceptions: the mode-switch marks
and the Overview "needs attention" list, which always cover all modes (each item tagged with its mode).

| V3 today | V4 |
|---|---|
| Client workspace (Overview, Performance, Strategies, Exposure, Activity, Diagnostics, Investor report) + Advanced (`/vps`, `/live`, `/paper`, `/incubation`, `/pods`, `/exposure`, `/performance`, `/events`, `/diagnostics`, `/journal`) | One shell |
| Strategies / Pods pages, pod detail fragment | Pod page per pod, reached from the menu or from Overview rows. **No Pods index page** (owner: it would repeat the Overview cycle block) |
| Exposure (both shells) | Positions |
| Performance (both), Investor report page | Performance with two levels: Portfolio, Pods. Investor PDF = an export button |
| Events, client Activity, Journal | Activity (one timeline) |
| Diagnostics tabs, health strip, freshness panel | System health |
| Operator Tools panel, command catalog, manual ticket | Tools |

## 5. Rules that hold on every page

**Word for the unit: "Pod".** Not "strategy" in the UI.

**One verdict sentence** opens every page, bold first clause, plain words.

**The cycle = 7 steps, always in this order:** `Data → Decide → Plan → Submit → Fill → Reconcile → EOD`.
A cycle is keyed by pod + execution session (`Open` or `Close`), so a future MOC pod
(`execution_policy_str = same_day_moc`, `signal_clock_str = pre_close_15m`) uses the same component with
close-session times. MOC is plumbed in the engine but used by no release today; do not claim it is live.

| V4 step | Built from (V3 / engine) |
|---|---|
| Data | build gate (`scheduler_utils.evaluate_build_gate_dict` reason codes) + Norgate freshness |
| Decide | DecisionPlan status (`none / planned / vplan_ready / submitted / completed / expired / blocked`) |
| Plan | VPlan built (`ready`) and its order rows |
| Submit | VPlan `submitting / submitted` **and** ACK (`submit_ack_status_str`: `not_checked / complete / missing_critical`, coverage, missing count). ACK is part of Submit, shown as "7 sent · 5 ack" |
| Fill | fills recorded vs order rows |
| Reconcile | `vplan_reconciliation_snapshot` (`passed / blocked`), `read_failed`, waiting |
| EOD | `pod_state` with `snapshot_stage_str = 'eod'` for the market date |

V3's "DB" stage is **not** a cycle step in V4. It moves to System health; a missing DB still raises a red item.
V3's lifecycle builder is `alpha/live/dashboard.py::_build_lifecycle_step_dict_list`; required actions come
from `_build_required_action_dict`.

**Seven state words**, each a shape *and* a colour (never colour alone):
Done ✓ green · Now ● blue · Planned ○ hollow · Late ◆ amber · Failed ✕ red · None – gray · Unknown dashed.
`Late` = past its due time, not failed. `None` = not needed this cycle (no-order days, idle monthly pod).
`Unknown` = source stale or missing; then the whole strip is Unknown.

**Planned vs actual, one rule: gray / hollow = planned, black / solid = actual.** A done step shows its actual
time plus a small delta against plan (`09:23:31 +1s`); a planned step shows `due 16:10:00` in gray.
Planned times must come from the scheduler's own timing functions, never from hardcoded clock times: for
example `scheduler_utils.build_submission_timestamp_ts` (open − `DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT`
= 390 s for open policies; close − `DEFAULT_SUBMISSION_BUFFER_MINUTES_INT` = 10 min for `same_day_moc`),
the snapshot-ready buffer (`DEFAULT_SNAPSHOT_READY_BUFFER_MINUTES_INT` = 10 min after the close) and
`scheduler_service.DEFAULT_RECONCILE_GRACE_SECONDS_INT` = 300 s. **If the scheduler has no planned time for
a step, show none.** The mockup's "Plan due 09:20:00" is illustrative, not a contract.

**Colour = state only.** One blue for data marks and selection. Exception, owner-approved: four identity hues
for pods (`#2a78d6`, `#1baf7a`, `#4a3aa7`, `#e87ba4`, validated for colour-blind separation in that order),
used only in the allocation donut, the Performance · Pods chart and the pod chips on Positions.

**Money is as of the last close** (no live marks exist). Say so where money is shown. Operations are live.

## 6. Pages

Source mappings below come from a read-only inventory of V3 taken on 2026-09-19. Function names were checked
against the code; **confirm field names and line positions before relying on them.**

### 6.1 Overview — "Is anything wrong, how is the money, what is next?"

1. Verdict. 2. Needs attention: one line per red/amber required action, **all modes**, with mode tag, short
text, how long, Open → pod page. Collapses to nothing when empty. 3. Four tiles: Account value, Day, Month,
Year. 4. **Pods block (the centre):** one roomy row per pod: name + status pill + "Daily · Open"; the labelled
7-step track; **Now** (two lines); **Next** (step + time + countdown). No money columns here.
5. Account value chart (1M/3M/YTD/All) beside the **Allocation donut**.

- Pill states: On track · Working · Idle · Late · Action needed · Unknown.
- Sources: `build_dashboard_summary_dict`, `build_pod_row_dict`, required-action labels, `verdict.py`,
  `schedule.py` (next action and due time), client financial summary for the LIVE tiles and chart.
- **Donut (new grouping of existing data):** inner ring = one gray Cash slice + one invested slice per pod;
  outer ring, over the cash arc only = who holds the cash (per pod, plus "Free cash" = cash in no pod).
  Centre = total cash %. Legend table: pod · Invested % · Cash %. Keep V3's valuation rules for the ring
  (`docs/plans/OPS_DASHBOARD_DESIGN_REFRESH.md`: same-date broker EOD equity and cash, whole-Flex fallback,
  never mix sources, never zero-fill missing cash). "Free cash" appears only if the data defines it
  (open decision 2); otherwise omit the slice.

### 6.2 Pod page — "What did this pod do in this cycle, and what is the evidence?"

Header: name, pill, mode tag, cadence · session, `pod_id` · masked account, verdict, buttons Trade sheet and
Tools. **Cycle panel:** session picker (`‹ previous | Fri 09-18 · Open | next ›`), the full 7-step strip (each
step: mark, name, one fact, one time), then evidence tabs: **Plan vs actual** (default) · Decision · Orders ·
Fills · Reconcile · Events · Files. Clicking a step opens its tab. Below: four tiles (Value, Day, Month, Since
start), Positions (shares, value, weight bar with a target line, "New" tag, cash row), Value chart with one
line "Live vs backtest −0.04% · Report".

- **Plan vs actual** joins, per symbol: Before · Order (gray) · Filled · Fill px · Slip bps · After ·
  "Broker = model" mark. It replaces opening four separate tables.
- A pod needing action adds a red box on top: required-action label, one sentence, Runbook link, Open tools;
  the failed step is selected and its evidence open; an "Events · this cycle" list and "Tools for this step".
  Wording follows the owner's guide symptom table, e.g. missing broker ACK → `show_vplan`, then the broker
  connection (`doctor`), then `post_execution_reconcile`; "Do not resubmit blindly."
- Sources: `build_pod_detail_dict`, `client_presentation.py` allow-listed evidence tables (targets, order plan
  rows, ACKs, fills, reconciliation — keep the allow-list and the identity/count-race checks), events tail,
  trade sheet route, reference-compare artifacts, equity chart.
- New: listing past cycles for the picker (query DecisionPlan / VPlan history per pod, read-only).
- **Positions panel = donut + table** (owner's idea, 2026-09-21, shown in `pod.html`). Left: a donut of the
  pod — each symbol's share of the pod, **cash included**; centre = cash %. Right: the table, which **keeps the
  shares** and adds Value $ and % (weight bar with the target line only when the target comparison is
  verified), then the Cash row. Rules: **one hue = that pod's identity colour** for every holding slice and
  the cash gray for cash — never one colour per symbol (colour means state or pod identity, nothing else);
  largest first, clockwise from 12 o'clock, cash last; a label only on a slice of 15 % or more, the table
  carries every number; 2 px surface gap between slices; hovering a slice marks its row and the reverse. It is
  a glance, not a comparison tool: for a 10-name equal-weight pod it shows "evenly spread, little cash" and
  the table does the rest. **It needs a closing value per symbol (8.1). Shares alone cannot draw it** — 212
  shares of one ETF and 13 of another can be the same money. Until that source exists the panel stays
  quantities-only, with no donut and no empty value columns. Value, % and the donut share one close date,
  printed in the panel head; if the saved quantities changed after that close, say so ("Holdings changed
  since this close") instead of mixing dates silently. A short position has a negative value: then no donut,
  table only.

### 6.3 Positions — "What does the whole book hold now, and how exposed am I?"

Portfolio level, all pods of the mode. Verdict; a visible note "Shares: broker HH:MM:SS · Prices: last close
<date>, **not live**"; tiles Invested (with a mini bar invested/cash), Open P&L, Best, Worst. Main table,
**default = All, sorted by value, one row per symbol with pods merged:** Symbol + company name · Pods (chips
with identity dot) · Shares · Value $ · Weight bar + % · **P&L since entry** ($ over %) · Today tag ("New
today", "Sell not filled"). Filters: All · Changed today · Off target; pod chips; Find. "By pod" table:
positions, invested, cash, open P&L, book %, plus a Free cash row and a Portfolio total.

- Sources: `dashboard.py::build_position_exposure_dict_list` and the client exposure builder (keep: distinct
  position and price timestamps, unpriced holdings excluded from totals and listed, no invented weights).
- New: P&L per position (8.1), merged rows across pods, company names (open decision 5).
- The owner declined a per-position price chart for now. Do not add one.
- **Status after `a4c0240` (2026-09-21):** quantities-only first version — merged symbol rows, pod chips,
  shares, pod filter, Find, By-pod table, Invested tile. Value $, Weight, P&L since entry, Today, Off target
  and the Open P&L / Best / Worst tiles have no source yet. **Until 8.1 lands, hide them** (columns, tiles,
  disabled filters, the "unavailable" note) exactly as the unconnected controls were hidden on the Pod page —
  no columns of "—". For a symbol held by several pods show the per-pod share split inline (a title tooltip
  does not exist on touch). "Changed today" needs no prices: build it from today's VPlan rows plus verified
  fills, as the Pod page's Before → After already does.

### 6.4 Performance — two levels behind tabs "Portfolio | Pods"

- **Portfolio** = the page the owner asked to keep exactly: period toolbar (From/To, 1W MTD YTD All, $ / %),
  exports (CSV, Investor PDF), six KPIs (Start, End, Profit / loss, Return TWR, Max drawdown, Now below high),
  account value chart, Balance bridge, monthly return grid, daily P&L bars, Risk facts.
  Sources: client financial views (`client_views.py`, `client_financial_display.py`, `client_charts.py`,
  `client_reporting.py`, `investor_report.py`). Keep every financial contract as is (Flex-based headline,
  TWR rules, withholding rules).
- **Daily P&L — readable numbers** (owner-approved 2026-09-21; never a number on every bar): (1) a readout
  in the panel header — by default the last session (weekday, date, P&L $, return %); on hover, focus or tap
  of a bar it shows that day and the other bars dim; the hit target is the whole column; (2) value labels
  only on the best and the worst bar; (3) a `Bars | Numbers` switch — Numbers shows the same 30 sessions as
  a grid, weeks as rows (**newest week on top**, the owner's correction) and Mon–Fri as columns, each cell
  tinted with the same heat steps as the monthly grid, a week-total column, "closed" for holidays; (4) one
  summary line under the panel: sessions, total,
  up / down, best, worst. Same data as the bars, no new calculation.
- **Pods** = new level: "Return by pod" (all pods indexed to start = 100, one line each, legend with values),
  "Adds to portfolio return", a Pods table (Start, End, P&L, Return, Max drawdown, vs backtest, Portfolio
  total), and "Monthly return % by pod" (same grid, one row per pod). Sources: V3's per-strategy performance
  panels. See 8.5 for the contribution caveat.

### 6.5 Activity — "What happened, when, in what order?"

One timeline: system events + alerts + operator actions. Rules agreed with the owner:
1. **A healthy cycle is one row** ("Open cycle completed · 9 of 9 filled, 0 diffs · 6 steps") that opens into
   its steps. Failures, late steps, alerts, operator actions and system events are **always their own rows**.
2. **Every row opens its evidence** (pod page at that cycle and step, job output, or System health).
3. A **"You last looked here"** line; the verdict counts only what is new.
4. The page **follows the selected mode**, plus mode-less system events.
Also: grouped by day, filters by pod and type (Cycles, Alerts, Operator, System), "Late + failed only",
technical codes hidden behind "Show codes", alert delivery status shown, 7 days by default with "Load older".

- Sources: `live_events.jsonl`, `live_critical_events.jsonl`, `operator_journal.jsonl`, per-pod
  `trace_events.jsonl`, notification / watchdog state. V3 has an event reader
  (`tests/test_dashboard_event_log_reader.py`) and `static/new_event_badge.js`.
- New: cycle grouping and a complete event → plain sentence table (8.2).

### 6.6 System health — "Is everything that should run, running?"

Its own menu item (the owner moved it back out of Tools). One table, three groups, columns
**Now · Last sign of life · Expected** (so lateness is visible, e.g. "Every 5 min · next 09:45:03"), and a link
to the tool that checks or fixes that row:
- *Runs all the time:* Schedulers · LIVE ("4 of 4 alive · one per pod"), Schedulers · PAPER, Scheduler ·
  INCUBATION ("one for all pods"), Broker gateway, Alerts · Discord, Dashboard. Scheduler rules: 6.9.
- *Runs on a schedule:* Watchdog, **Dead-man ping** (the only thing that catches a dead VPS), Data sync ·
  Norgate, Flex import. Add Backups here when they exist; do not show a permanent gray row before that.
- *Data and space:* Market data (have vs needed session), Rates · FRED, Event log, Database, Disk.
Then **Pods** (per pod: **Its scheduler** = state and what it waits for, **Last sign of life**, **Wakes** = when
it said it would wake; then data session, broker read, last EOD), then **What should run** = releases (pod,
mode, Enabled, release, how it trades, account). Releases are the *expected* side; the lights are the *actual*
side. Downloads: Status JSON, Diagnostic JSON.

- Sources: `health.py`, freshness items in `dashboard.py`, `ops_report.py` and
  `alpha/live/logs/ops_report_latest.json`, watchdog and notification state files (`notifications.py`),
  scheduler loop events (`scheduler_sleeping` and friends, see 6.9 — note that
  `DEFAULT_OPERATOR_HEARTBEAT_SECONDS_INT` = 900 s only throttles the printed wait message; an idle scheduler
  may write nothing for up to `DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT` = 3600 s, so the Event log row expects "a
  write at least every 60 min"), `scheduler_service` next-phase and reason codes, `norgate_snapshot_sync.py` status, `ibkr_performance.py` Flex import, `release_manifest.py`.
- "Expected" values come from code constants and the task registration, not from guesses.

### 6.7 Tools — "Copy or run an operator command"

Mirrors the owner's guide: same commands, same groups, same risk classes. The **real command name is the
row title**; one plain sentence under it; class text beside it. Two blocks side by side. Class legend on top:
READ = report only · INSPECT = no orders, may write diagnostic records · ACTIVE = changes state, data or
services. Pod selector in the page head; the pod page and System health deep-link here with the pod chosen.

| Block | Group | Command | Class |
|---|---|---|---|
| Read · never trades | Daily checks | `ops_report` | READ |
| | | `status` | INSPECT |
| | | `next_due` | INSPECT |
| | Diagnose and review plans | `show_decision_plan` | INSPECT |
| | | `show_vplan` | INSPECT |
| | | `execution_report` | INSPECT |
| | Reports and support | `export_trade_sheet` | INSPECT |
| | | `compare_reference` | INSPECT |
| | | `collect_vps_debug_bundle` | INSPECT |
| | | saved watchdog report (`ops_report_latest.json`, task info) | READ |
| Act · changes state | Diagnose | `doctor` | ACTIVE — queries the broker, may sync data, needs an unused `--broker-client-id`, never in a loop |
| | tick, run_once and serve | `tick` | ACTIVE · may send orders |
| | | `run_once` | ACTIVE · may send orders |
| | | `serve` | ACTIVE · **copy only**, never a second copy for a pod |
| | Submit and confirm execution | `submit_vplan` | ACTIVE · sends orders |
| | | `post_execution_reconcile` | ACTIVE |
| | | `eod_snapshot` | ACTIVE |
| | Data and watchdog | `live_ops_watchdog` | ACTIVE · may send alerts, no orders |
| | | `doctor_norgate_client` | ACTIVE · can download files |
| | Break glass | manual order ticket | ACTIVE · sends orders |

- Copied commands carry the pod, the mode and, when the service uses them, `--releases-root` and `--db-path`
  (V3's catalog builder in `operator_tools.py` already quotes these). Never auto-pick a broker client id.
- **Copy must work in read-only mode** (V3 disables it). It is the whole value of the page by default.
- Opening a row shows the command and, only for the actions V3 already runs (guardrail 4) and only with
  actions enabled, Preview → Confirm. All other ACTIVE rows (`doctor`, `run_once`, `serve`,
  `live_ops_watchdog`, `doctor_norgate_client`) are copy only.
- `ibkr_connectivity_probe` is in `COMMANDS.md` but not in the owner's guide; left out unless he asks.

### 6.8 Phone and the reference sheet

Only the Overview is mocked for the phone (`mobile.html`): mode switch, verdict, attention cards, two tiles,
pod cards with the mini strip, bottom tabs Overview · Positions · Results · Activity · System. Other pages:
apply the same rules, no horizontal scroll at 390 px. `cycles.html` shows every cycle variant (running, close
session, idle monthly day, no orders, data late, incubation SIM, source old) and is the acceptance reference
for the cycle component.

### 6.9 The scheduler (`serve`) — "Is the thing that runs this pod alive?" (added 2026-09-20, owner-approved)

**Deployment fact.** LIVE and PAPER run **one `serve` process per pod** (`serve --mode live --pod-id <pod>`,
each in its own window, own IBKR client id — `docs/operations/vps-runtime-checklist.md` section D,
`COMMANDS.md`). INCUBATION runs **one fan-out scheduler** for all its pods. In code `--pod-id` is optional; do
not infer "one per mode" from the signature. One pod's scheduler can die while the others look fine.

**What it gives the operator.** Only one scheduler fact changes what the operator does: **alive or not, per
pod.** When it is alive but not acting, the second useful fact is **why**: it holds the pod for a person
(`manual_review_pending`: parked execution exception, or a ready VPlan with auto-submit off — it does not
retry by itself), it waits for data, or it hit an error and retries. Its "next step and time" and "last step
done" **are the cycle**, which is already on screen — never show them a second time under the name "scheduler".

**Where it shows.**
- **System health is its home** (6.6): count rows per mode, and per pod its state, last sign of life and the
  time it said it would wake.
- **Nowhere else while healthy.** No top-bar item, no line on the Overview Pods block, no line on a healthy pod
  page. A permanent "4 of 4 alive" is one more green light that says nothing; the single System light already
  covers everything that must run. (The owner saw it in four places and chose this.)
- **On failure only, through what already exists** (`index-scheduler-stopped.html`): the System light turns
  red ("System needs action · Scheduler"); that pod's menu mark and pill become **Action needed**, Now =
  "Scheduler stopped" + "no sign of life since HH:MM:SS", Next = "Start the scheduler" + "you · now"; one
  attention row ("Scheduler stopped. No sign of life since … Plan and Submit did not run."); the pod page shows
  the issue banner with `next_due` and `serve` to copy. Pods whose schedulers are alive do not change.
- **On a problem pod whose scheduler is alive**, one sentence inside the existing issue banner
  (`pod-issue.html`): "The scheduler is alive. It holds this pod and will not retry by itself." The first
  debugging question for a stuck pod is "is the scheduler even running?" — answer it where the operator decides.
  **The sentence must come from the scheduler's real last state, never from the pod's problem type:**
  `manual_review_pending` → "…It holds this pod and will not retry by itself."; still polling (for example
  `post_execution_reconcile` due, active poll) → "…It checks again every 30 s."; waiting for data → "…It waits
  for data." Verified in `scheduler_service.get_scheduler_decision` and
  `runner.is_vplan_execution_exception_parked`: a **missing broker ACK alone does not hold anything** — the
  VPlan stays `submitted` and the scheduler still runs reconcile after the 300 s grace. It parks a pod only
  (a) after a post-execution reconciliation snapshot exists **and** an unresolved order is in a terminal broker
  status (`execution_exception_parked`), or (b) when a VPlan is ready and auto-submit is off
  (`manual_review_required`). The mockup's QPI story ("Reconcile is held") is illustrative wording, not an
  engine contract.

**Rule for the cycle view-model.** A pod's planned steps count as *Planned* only while its scheduler is alive.
Stopped or erroring → the pod is *Action needed* even if no step is late yet. Liveness that cannot be
determined is *Unknown* on System health (never green); it does not by itself downgrade a pod.

**Liveness, read-only, no engine change.** On every loop `serve` writes `scheduler_sleeping` to the event log
(`logging_utils.DEFAULT_LOG_PATH_STR`) with `next_phase_str`, `reason_code_str`, `next_due_timestamp_str`,
`sleep_seconds_float` and `related_pod_id_list` (= the pod, for a pod-scoped `serve`); the same facts go to the
per-pod trace files `logs/pods/<pod>/<run>/trace_events.jsonl` (`scheduler.decision`, `.sleeping`,
`.tick_result`, `.error_retry`). **Alive = it woke when it said it would:** promised wake = that event's own
timestamp + `sleep_seconds_float`. A fixed "heartbeat age" rule is wrong here, because an idle scheduler
sleeps up to 3600 s on purpose. Proposed display states (thresholds are the owner's to tune, see 12):
*Sleeping* (now ≤ promised wake + 60 s) · *Running `<step>`* (last event is `scheduler_due_now`) · *Holding*
(`manual_review_pending`) · *Late* (more than 60 s past the promised wake) · *Stopped* (more than 5 min past
it) · *Error* (`scheduler_error_retry`, show the redacted error and the retry time) · *Unknown* (no scheduler
event found in the bounded tail, or the log is unreadable). Known limit: `scheduler_started`, `scheduler_woke`
and `scheduler_error_retry` carry no pod id today, so "running since" per pod is not derivable and is not
shown; attribute errors through the per-pod trace files. Read the tail only, bounded, and remember log rotation.

**Activity.** Scheduler stopped / error / started are System rows. Routine sleep and wake loops are never rows.

**Out of scope here, recommended separately.** A dashboard light does not help at 02:00. The watchdog (runs
every 5 min, independent of `serve`) could apply the same per-pod rule and alert on Discord / fail the dead-man
ping. Today nothing checks scheduler liveness directly; a dead `serve` is noticed only after a missed window.
That is an engine/ops change (Tier 3) and needs its own owner decision.

## 7. Vocabulary (keep it this small)

Done, Now, Planned, Late, Failed, None, Unknown · On track, Working, Idle, Late, Action needed, Unknown ·
Data, Decide, Plan, Submit, Fill, Reconcile, EOD · Pod, Portfolio, Book · Open, Close, Daily, Monthly ·
READ, INSPECT, ACTIVE. V3's ~30 raw status strings map onto the seven state words; show a raw code only
behind "Show codes".

## 8. New backend work (all read-only, all display-only)

1. **Value, weight and P&L per position — source: the IBKR Flex "Open Positions" section** (revised
   2026-09-21; replaces the first idea of rebuilding average cost from `vplan_fill`, which needs too many
   assumptions: adjusted vs unadjusted prices, splits, manual tickets, positions older than the fill log).
   The owner adds the section to his existing Flex query; the importer (`alpha/live/ibkr_performance.py`,
   which already stores `flex_import.raw_xml_str`) parses it into a new additive table keyed by account,
   report date and symbol. Expected attributes — **verify against a real export, never guess:** `symbol`,
   `conid`, `position`, `markPrice`, `positionValue`, `costBasisPrice`, `costBasisMoney`,
   `fifoPnlUnrealized`, `percentOfNAV`, `currency`, `reportDate`, `levelOfDetail` (use the summary level).
   Why: broker-official closing marks **and** cost basis, on the same close date as the NAV already shown;
   it survives splits, manual trades and old positions; no trading-engine change. Rules: a missing section
   changes nothing that exists today (NAV and TWR parsing untouched) and the pages stay quantities-only; use
   a date only when Σ position values + saved cash reconciles with that date's finalized NAV within a stated
   tolerance, otherwise show quantities only and say why; "P&L since entry" is IBKR's unrealized P&L and is
   labelled as such; one account = one pod, so per-pod rows need no allocation. It feeds the Pod-page donut
   (6.2) and the Positions page fields (Value $, Weight, P&L since entry, Open P&L / Best / Worst, sort by
   value). Headline NAV, P&L and TWR stay as they are. Record the method in `ASSUMPTIONS_AND_GAPS.md`.
   This touches the Flex importer under `alpha/live/**`: Tier 3, additive only.
2. **Activity grouping and sentences.** Group events by pod + cycle identity (DecisionPlan / VPlan ids); a
   cycle folds only when every step is healthy. V3 translates ten event codes
   (`client_operations.EVENT_LABEL_DICT`); V4 needs a sentence for every routine event, with the raw code as
   fallback. Scans must stay bounded (V3's runbook notes that some log scans are not fixed-cost).
3. **"Last looked" marker.** Browser-side only (localStorage), so read-only mode writes nothing server-side.
4. **Expected rhythm on System health.** Next-run and cadence from code constants and saved state; per-pod
   `next_due`; last dead-man ping result from the watchdog output.
5. **Performance · Pods.** Indexed series per pod; contribution per pod. **Caveat:** percentage points that
   sum to the portfolio return are exact only with no external flows in the period. Simplest correct default:
   contribution in **dollars** (pod P&L sums exactly to portfolio P&L); show points only when the period has
   no flows, or label them approximate.
6. **Cycle view-model.** One function that turns V3's lifecycle, gate, ACK, fill, reconcile and EOD facts into
   the 7-step shape with state word, one fact, actual time, planned time and delta. Used by Overview rows,
   the pod page, Activity sub-rows and the phone cards. This is the single most reused piece.
7. **Merged holdings** across pods and "Off target" (drift against the latest decision targets).
8. **Past cycles per pod** for the session picker.
9. **Allocation cash split** per pod and, if defined, free cash.
10. **Tools catalog** from one table (command, class, group, sentence, runnable?) so page, pod shortcuts and
    System health links cannot drift apart.
11. **Scheduler liveness per pod** (6.9): a bounded, read-only reader of the latest `scheduler_*` events per
    pod → state, last sign of life, promised wake, next phase, hold / error reason. Feeds System health and the
    cycle view-model's "scheduler alive" input. Unit-test the states on returned dicts, including log rotation,
    an idle 3600 s sleep (must stay alive), an overdue wake, an error-retry and "no events" (Unknown).

## 9. Front-end notes

- Lift `ops.css` and the mockup markup into Jinja partials almost as is: shell, top bar, state mark, pill,
  tag, labelled track, full cycle, tile, table, donut, tool row, note. Tokens are CSS variables on `:root`.
- One refresh clock per page ("updated HH:MM:SS"); V3's pod detail runs four independent polls — avoid that.
  If polling fails, the page must say so rather than keep showing green.
- Icons are inline stroke SVG (`_build/icons.py`), own drawings. No emoji, no icon font, no CDN.
- `tests/test_theme_no_hardcoded_colors.py` guards `alpha/engine/**` presentation modules and Bench, not the
  dashboard CSS. Still keep V4 colours as tokens in one place.

## 10. Phases and acceptance

| Phase | Scope | Done when |
|---|---|---|
| 0 | Package skeleton beside V3, shell, mode scope, top bar, cycle view-model + component, read-only enforcement | `cycles.html` variants reproduced from fixtures; read-only route tests pass; V3 untouched |
| 1 | Overview, Pod page (both states), System health — existing data only | pages match the mockup at 1440 / 768 / 390; stale source renders Unknown; no cross-mode money |
| 2 | Activity (grouping, sentences, last-looked, mode scope) | healthy cycle = one row; every row links to evidence; bounded scans |
| 3 | Positions (merged rows), then the Flex Open Positions source (8.1), then values, P&L per position and the Pod-page donut (6.2) | quantities page has no dead fields; importer change is additive and a missing section changes nothing; value rules of 8.1 (reconciliation with NAV, one close date, "—" and no-donut cases) covered by tests; assumptions register updated |
| 4 | Tools (copy only; Run only if the owner enables it) | 20 commands per the guide; copy works in read-only; no new executable paths |
| 5 | Performance levels; phone pass on all pages; cutover plan | Portfolio page parity with V3 numbers; Pods level per 8.5; owner sign-off to switch the default route |

Each phase is reviewable on its own. Claude reviews each phase read-only against the mockup and this document.

## 11. Verification

- Work under `alpha/live/**` is **Tier 3**: tests, the live-impact checklist, and parity, failure-mode and
  coverage reviewers. Run `uv run python scripts/review/triage.py` after each change. This document alone is
  Tier 0.
- **Port the existing tests, do not delete them**; they encode the contracts:
  `tests/test_dashboard_v3_*.py`, `test_dashboard_operator_*.py`, `test_dashboard_local_workspace.py`,
  `test_dashboard_event_log_reader.py`, `test_client_*.py`, `test_local_ibkr_financials.py`,
  `test_live_ops_report.py`, `test_live_ops_watchdog.py`. Exact-markup assertions will need V4 equivalents.
- Develop against synthetic data: `python -m alpha.live.dashboard_v3 --demo`, `alpha/live/dashboard_v3/demo.py`,
  `scripts/review/serve_local_workspace_fixture.py --expanded`. Browser checks exist as
  `scripts/review/check_*.cjs`; keep the 1440 / 768 / 390 matrix and the no-outbound-connection check.
- New view-models get unit tests on the returned dicts (rules, not markup). Read-only routes get tests that
  prove no job, journal or notification write happens.

## 12. Open decisions for the owner

1. **Money source per mode.** LIVE uses the Flex-based financials. What feeds the tiles in PAPER and
   INCUBATION (broker EOD snapshots, SIM ledger)? Each must be labelled; none may be blended.
2. **Free cash.** Is it account cash outside pod budgets (`pod_budget_fraction_float` < 1), cash in unbound
   accounts, or not shown at all?
3. **Contribution** in dollars (always exact) or in points (exact only without flows)? See 8.5.
4. **Value and P&L per position:** source = Flex "Open Positions" (8.1). Owner action: add that section to
   the Flex query in IBKR (Claude cannot do this). Still open: the NAV reconciliation tolerance, and whether
   P&L is shown as IBKR reports it (FIFO unrealized) or not at all when the account uses another lot method.
5. **Company names** under tickers: available from the Norgate snapshot on the VPS, or ticker only?
6. **Tools, first release:** copy only (recommended), or Preview → Run for the V3 allow-list too?
7. **Route and cutover:** separate port while both run, then switch `/` to V4 — when, and what is rollback?
8. Activity default range (7 days proposed) and the bound for "Load older".
9. **Scheduler thresholds** (6.9): *Late* after 60 s past the promised wake and *Stopped* after 5 min are
   proposals. And separately: should the watchdog alert on a stopped scheduler (Discord + dead-man fail)? That
   is outside V4 and Tier 3.

## 13. Decisions already made — do not undo

- No Pods index page; "Pods" is only a menu heading with the pods nested under it.
- System health is its own page, not a tab of Tools.
- Tools shows real command names, follows the owner's guide, and splits Read from Act.
- Overview pod rows carry no money; the "Today" run sheet from mockup A was removed as too dense.
- Allocation is a donut with the nested cash split (the owner's idea), not bars.
- Prices and P&L are as of the last close and labelled "not live". No per-position price chart.
- Activity folds healthy cycles and follows the selected mode. The name stays "Activity".
- The Performance · Portfolio page stays as designed.
- Calm density: fewer columns and captions beat completeness on screen; detail lives one click away.
- The scheduler has no permanent display outside System health. It appears elsewhere only when a pod's
  scheduler is not alive, and then through the existing light, attention row, pod mark, pill, Now and Next (6.9).
