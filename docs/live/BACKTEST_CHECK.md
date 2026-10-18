# Backtest check (nightly, report only)

**Question it answers:** is each LIVE pod still running the strategy we tested? Not "is the edge alive" — that
needs years of returns and stays in the monthly Scout review.

## What runs

Scheduled by `scripts\setup_live_tasks.ps1` (one script for every Live Ops task; `-CheckOnly` = PASS / FAIL only):
`AlphaBacktestCheck` Mon-Fri 06:30 ET (stopped at 09:00 if still running) and `AlphaBacktestRuler` Saturday 10:00 ET
(`--rulers-only`: the long history backtests, away from trading hours).

The scheduled wrappers run through `alpha.live.scheduled_backtest`: weekday checks may start only
from 06:30 until 09:00 New York time and are terminated at 09:00; ruler runs may start only on
Saturday from 10:00 until 18:00 and are terminated at 18:00. Missed backtests do not catch up.
The supervisor terminates only its own diagnostic child process tree. A cutoff returns failure;
an out-of-window invocation skips without starting a backtest. These guards apply to the scheduled
wrappers; the explicit manual Python command below remains a manual diagnostic.

Setup registers previously disabled tasks with disabled settings from the outset, including when
a later registration fails. Calendar triggers use local, offset-free New York wall time so DST
does not move the scheduled hour. `-CheckOnly` validates exported task XML: action/release path,
working directory, watchdog mode, trigger, execution limit, overlap, missed-start and principal policies.
Older UTC calendar triggers or obsolete paths fail this check and require re-setup after deployment.

```
uv run python -m alpha.live.backtest_check --performance-db C:/alpha/live_ops/ibkr_performance.sqlite3
```

Once per session day, best at about 08:00 New York time: after the Flex import (07:10) has final returns for the
previous session, and before the open. The run uses the New York date; on a non-session date TAA backtests are
skipped to stay clear of the month-end DTB3 refresh. Scheduling it is an owner decision; nothing here registers
a task. Options: `--pod-id POD` (one pod; the others keep their last result and original check dates), `--no-reference` (MR cost component only),
`--refresh-ruler` (build missing or old rare-loss rulers: one long backtest per pod, cached for 35 days).

Output: `backtest_check_latest.json` next to the watchdog report (`ops_report_latest.json`), plus a
`backtest_check_cache/` folder with the backtest runs and the rulers.

## What each pod gets

| Part | How | Shows in V4 |
|---|---|---|
| Decision check | The reference starts with the first executed plan's session and budget and ends at the last finished exchange session. Full books require matching symbols and exposure directions; weights of matching symbols and total gross exposure each tolerate ±5 percentage points because reference weights are marked at that close. Small per-symbol differences cannot accumulate into an unchecked total-exposure difference. Entry/exit books check actions and equal-slot entry weights from the actual configured DV2, DV2-IND, HPI and MR reference objects (`1 / slots`, tolerance `0.000001` in decimal weight). Other entry-weight contracts remain unchecked. Exact decision-session coverage is required; older holdings never substitute. | Pod page: Backtest tab, red box when a current check differs; Overview row |
| Same-days curve | Live daily TWR (IBKR Flex) and reference daily return, including the reference's first session from initial cash. Every finished exchange session from deployment must exist in both sources. Missing dates or invalid returns make the curve/gap unavailable, with the reason displayed; dates are never silently dropped. Non-session broker rows also require matching reference evidence. Up to 400 complete dated points are displayed. | Pod Value chart: dashed second line only when the whole selected period is covered; Performance → Vs backtest |
| Rare loss | Live loss over 1 / 3 / 5 consecutive exchange sessions ending at the required last finished session, below the reference's 1-in-100 line. The historical ruler ends before deployment, uses the runner's declared historical start/config default, requires continuous session history and at least 250 windows per length, and expires after 35 days. Cause: "strategy" when the reference lost at least 70% as much, "ours" otherwise, "unknown" if same-window reference data is absent. Missing live sessions suppress that window. | Amber line on the Pod page (red when "ours"); Overview row only when "ours" |
| MR cost component | Stock fills of the MR capsule pods vs the official open, per side, worse = positive. BIL/SPMO parking fills and fills without an official open are excluded from the average/count. The deadline starts at the first stock fill, including fills without an official open: 183 calendar days (approximately six months). The 200th qualifying fill must occur by that deadline. An early completed sample remains timely when reviewed later. The average uses all valid fills available through the as-of date. | Performance → Vs backtest; Overview row when above / failed / time cap. A pass means the cost component only; unexplained halts still need separate review. |

## Boundaries

- Read-only toward live state: pod databases opened with SQLite `mode=ro`; the Flex store through
  `client_reporting` (read-only). No `LiveStateStore`, no release sync, no broker, no orders.
- Backtests run one pod at a time inside `isolated_fred_cache_context`: all FRED cache writes go under that
  reference run's `macro_cache` directory, including historical ruler runs. The existing configured cache is
  a read-only offline fallback. The series, macro mode, as-of boundary and strategy parameters do not change.
  The context restores even after exceptions and does not redirect other threads' ordinary live calls.
  TAA's first/last-two-session month-end skip is retained as a conservative scheduling guard, not as write isolation.
- CORE5 has no automatic backtest yet (`core5_accounting_bridge_required`): it shows "Not checked" with that reason.
- The backtest leaves dividends out (live reference contract `price_return_ledger_v1`) while the IBKR TWR includes
  them, so the gap drifts by roughly the dividend yield on income-heavy pods. Read the gap as a trend.
- Path dependence: after a missed fill a slot pod (DV2-IND, MR) holds something the backtest does not, and later
  sessions can show "differs" until the paths meet again. The red box links to the evidence; the Plan vs actual tab
  says why.
- A failed ruler build, malformed cached ruler or failed rare-loss calculation never discards the night's
  decision/curve. The pod has no rare-loss line and shows the ruler reason. Expired or absent rulers need `--refresh-ruler`.
- `--pod-id` never rewrites the MR component (it pools both capsule pods); a full run does. Retained pod/component
  evidence keeps its own check timestamp and as-of date, including migration of older reports' global dates.
- Malformed saved decision books, JSON containers or nonfinite/non-numeric weights are not interpreted as empty
  valid decisions. Malformed prior reports are dropped before partial-run merges.
- An unreadable contributing MR pod makes the pooled cost component `incomplete`, with the missing source
  named and the aggregate mean/deadline verdict unavailable. It cannot pass on surviving sources alone.
- Saved-cycle Backtest evidence requires the exact release, decision-plan ID and session. Legacy reports
  without the decision ID remain readable for current views, but cannot prove a saved-cycle match.
- System health warns when any currently enabled LIVE identity lacks a completed decision check, including
  skipped checks, partial coverage, release replacement and zero checked pods.
- A malformed report is dropped whole by the dashboard before Pod, Performance or System consumption. Cumulative
  returns of -100% or below are invalid for curve rebasing. Saved curves display both their start and end dates.
- Reports only. It never blocks orders and sends no alerts. The red box quotes the stop rule; disabling a pod stays
  the owner's call.
- Evidence that is not from the latest session (checked from 10:00 ET on session days) is marked stale per pod and
  per cost component: current-action alerts, including the Pod stop-rule box, and rare-loss lines are hidden. Historical
  decision details remain dated and visible. System health warns only when the check did not run, failed or is stale;
  what it found (a difference, a gap in the return curve) shows on the Pod and Vs backtest pages, not in the header.
- On Overview a difference row has a Reviewed button. It hides that one decision's difference (saved in
  `alpha/live/logs/backtest_reviewed.json`); a new decision that differs shows again. Pod and Vs backtest keep it.

## Calculations and timing

- The morning run fixes `D` to the last XNYS session strictly before the New York run date. Plans for today's
  upcoming open are excluded, even if already submitted. The underlying strategies still decide from Close_T
  and execute under their existing next-session contracts; this checker changes no strategy or execution rule.
- Reference first-day return: `r_1 = E_1 / C_0 - 1`, where `C_0` is the first executed plan's budget and `E_1` is
  reference NAV after its first session. Later returns: `r_t = E_t / E_(t-1) - 1`, requiring NAV for the exact
  preceding exchange session. A missing prior NAV never becomes a compressed multi-session "daily" return;
  rare-loss attribution for the affected window is unknown. A missing deployment-first NAV is not rebased later.
- Each complete displayed curve compounds independently: `I_0 = 1`, `I_t = I_(t-1) * (1 + r_t)`.
  Relative gap: `I_live / I_reference - 1`; monthly gap divides each curve by its own prior-month closing index first.
- Full-book gross exposure: `G = sum(abs(w_i))` across compared assets after excluding non-asset ledger keys.
  Matching requires `abs(w_live_i - w_reference_i) <= 0.05` for every matching symbol and
  `abs(G_live - G_reference) <= 0.05`; the total tolerance is not multiplied by the number of symbols.
- A rare-window return is `product(1 + r_t) - 1` over consecutive exchange sessions. The historical line is the sorted
  overlapping-window value at index `floor(0.01 * N)`. Only negative live returns and negative lines are considered.
- Fill cost in bps: `10000 * side * (fill_price / official_open - 1)`, with `side = +1` for buys and `-1` for sells.
  The displayed mean is the arithmetic fill-record mean; the 95% descriptive range is `mean ± 1.96 * sample_std / sqrt(N)`.
  Partial fills remain fill records, not independent orders. This descriptive range is not proof of full gate qualification.
- The current snapshot reconstruction still does not prove that historical/live inputs or macro vintages are identical.
  Full-book close-weight tolerance can still disagree on an extreme market move; the red row is a diagnostic requiring review.

## Safety-fix verification (2026-10-17)

- Scope: diagnostic comparison, evidence validation/freshness, curve/ruler coverage and private FRED cache writes.
  No strategy, engine, runner, scheduler, reconciliation, reference-price implementation or released config was changed.
- Review tier: Tier 3, with a Tier 2 shared-data-loader surface. Read-only parity, failure-modes, coverage and
  quantitative reviews found no remaining material defect in the inspected scope.
- Final relevant Python suite: 3,511 tests and 39 subtests passed, with 38 numerical/deprecation warnings.
  The focused checker/dashboard/FRED subset passed all 99 cases (included in the broader count), including
  55 added regression cases. All 277 dashboard JavaScript interaction tests passed separately.
  The broader suite includes V4, reference comparison, runner/scheduler/reconciliation/releases, client/Flex
  reporting, FRED, and the affected TAA/MR/DV2-IND strategy and readiness tests. This is the relevant suite,
  not a claim about every test in the repository. `git diff --check` passed.
- Live-impact checks: next-open timing and sizing rules are unchanged; official-open fill-cost inputs are unchanged.
  Live database schemas, state/config formats and existing dashboard logging fields are unchanged. Older reports
  retain their original evidence dates, or are rejected if malformed. Windows cache-path, context restoration,
  malformed-report and repeated partial-run cases are covered locally; no VPS deployment or broker action occurred.
- Remaining evidence limits: local tests do not establish real-VPS data parity. Current snapshot/macro reconstruction,
  omitted dividends, close-weight tolerance, overlapping rare-loss windows and correlated partial fills remain
  diagnostic limitations. CORE5 remains explicitly unchecked; a cost-component pass is not full MR gate approval.

## Scheduled-task and evidence follow-up verification (2026-10-17)

- Fixed all six follow-up findings: absolute diagnostic deadlines, disabled-task preservation, observed task
  contract validation, incomplete pooled MR sources, exact saved-decision matching and missing current-pod coverage.
- Tier 3 review: three independent read-only reviewers covered failure modes, parity/quantitative correctness
  and regression coverage; no material finding remained open after fixes.
- Broad relevant regression: **2,808 passed**, including all V4 Python tests, checker/task/FRED/Flex tests,
  watchdog tests and the runner, scheduler, reconciliation and release-manifest suites. The latest focused
  scheduling/script run passed **53 tests**; these overlap the broad suite and are not an additional unique total.
- Windows tests use real unregistered trigger objects and harmless diagnostic process trees; Scheduler
  registration is stubbed. Five affected PowerShell scripts parsed successfully; scoped `git diff --check` passed.
- Order timing, sizing, reference prices, live state schemas, existing log fields and released pod routes are
  unchanged. New report fields are optional; older reports remain readable, with unproven historical matches omitted.
- No push, deployment, real task registration or broker action was performed. Real-VPS Scheduler execution and
  production data parity remain unverified. The operator setup has not been run on the server.
