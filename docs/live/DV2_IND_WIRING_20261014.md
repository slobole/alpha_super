# DV2-IND wiring — 2026-10-14

## Scope and status

DV2-IND is wired to the guarded daily DecisionPlan/VPlan path through a disabled template.
This change enables engineering support and the registry's ordinary research portfolio
eligibility; it changes no existing portfolio weights, account route, scheduler or deployment.
WIRED describes plumbing. Account readiness, forward validation and LIVE enablement remain separate.

The adapter uses the existing `DVO2IndustryEtfStrategy` unchanged. There is no new parameter
search, fit, optimization, or investment-performance claim.

## Frozen strategy

- Universe: XBI, IBB, SMH, SOXX, KRE, KBE, XHB, ITB, XRT, XOP, OIH, XME, GDX, IYT,
  IGV, ITA, IHI, XSD, XPH. This is the researched fixed list, not a historical PIT ETF universe.
- History begins 2009-01-01, including warmup. An ETF needs at least 252 nonmissing
  own closes. Inception gaps remain absent; no additional forward-fill is applied.
- Native dollar liquidity: `ADV63_T = mean(Turnover[T-62:T])`, with 63 finite positive
  values, strictly greater than USD 50,000,000. Adjusted share volume is never used
  to reconstruct dollar turnover.
- Entry: eligible ETF AND `DV2_126_T < 10` AND `Close_T > SMA200_T`
  AND `Close_T / Close_(T-126) - 1 > 0.05`. Preserve the research complete-row check.
- Rank: TA-Lib NATR14 descending; retain the parent's sorting/tie behavior.
- Exit: held ETF `Close_T > High_(T-1)`. Loss of liquidity does not add an exit rule.
- At most ten long positions. A planned exit frees one slot for a replacement in
  the same next-open auction. Existing untouched holdings retain their shares.
- Research costs remain 2.5 bps slippage per side, USD 0.005/share commission,
  minimum USD 1. Actual broker charges and execution prices determine live results.

DV2 math is unchanged:

`HL2_T = (High_T + Low_T)/2`

`DV1_T = Close_T/HL2_T - 1`

`DV_T = (DV1_T + DV1_(T-1))/2`

`DV2_T = 100 * average_rank(DV_T within trailing 126 DV observations)/126`

TA-Lib's ATR14 uses `TR_T = max(High_T-Low_T, abs(High_T-Close_(T-1)),
abs(Low_T-Close_(T-1)))`, its existing Wilder initialization/recurrence, and
`NATR14_T = 100 * ATR14_T / Close_T`. SMA200 includes observations through T.

## Data contract

The new `norgate_eod_dv2_industry_etf` profile exports CAPITALSPECIAL native OHLC,
Unadjusted Close, native Turnover and Dividend for the 19 ETFs, plus TOTALRETURN
`$SPXTR` for the benchmark requested as `$SPX`. TOTALRETURN never supplies a trade price.

Historical padding stays Norgate ALLMARKETDAYS for research identity. Every current
endpoint is independently checked against an unpadded vendor observation. Schema v2,
exact adjustment pairs, source requests, native fields/dtypes, declared per-series
first/last/count coverage, continuous exchange sessions and immutable file hashes
must validate. Shortened history, gaps, missing liquidity fields, stale/future
snapshots or a changed manifest during construction fail closed. Earlier valid
profile caches cannot hide altered bytes. Existing profiles retain their behavior.

Operational decisions require local snapshot mode; there is no provider fallback.
A young ETF is legitimately ineligible. Missing required features for an ETF with
252 observed closes blocks the whole decision cycle, a deliberate conservative
operational guard rather than silently treating provider failure as no opportunity.

## Timing, account state and sizing

```text
[Native Norgate history through Close_T]
                  |
                  v
[Validated immutable local snapshot + trusted same-session EOD cash/holdings]
                  |
                  | *** CRITICAL *** no input after Close_T enters strategy
                  v
[Frozen incremental exits + NAV_Close_T/10 entry dollars]
                  |
                  v
[Pre-submit reference quotes -> whole-share MOO orders; cutoff 09:28 ET]
                  |
                  | Next XNYS open (normally 09:30 ET; holidays respected)
                  v
[Actual fills/holdings -> daily reconciliation -> atomic committed account state]
```

A dedicated margin account uses budget fraction 1.0. This template value reserves
the whole account for its own sleeve; it does not allocate existing client capital.
Bootstrap cash cannot substitute for trusted EOD state. State must match the release
identity, account, EOD date/source and timestamp. Held positions must be known long
whole-share ETFs, at most ten slots, with the saved DV2-IND strategy identity.
An unexplained pre-existing account holding cannot be adopted by a cold start.

`NAV_Close_T = EOD_cash_T + sum(actual_shares_i * UnadjustedClose_i,T)`

`entry_dollars_i = NAV_Close_T / 10`

`entry_shares_i = floor(entry_dollars_i / reference_quote_i,submit)`

Exit quantities come from actual holdings. No synthetic Open_(T+1) price enters
sizing. CAPITALSPECIAL supplies scale-invariant features; nominal Close_T supplies
account valuation. Frozen entry dollars do not resize when broker NAV/cash changes
before submission. This preserves the researched value-order intent; exact share,
fill-price and P&L parity is not claimed.

Immediately before dispatch, fresh account holdings must still match the saved
base positions and all-client open orders must be absent. The existing guarded
claim, per-order deadline, restart/no-resend protection, intraday SELL-only exit
completion, exact-owned-order cancellation, close-time actual-account settlement,
watchdog and durable exception/overdue notifications are reused. Capsule parking
ETF validation stays separate. No shared buying-power preview is introduced.
A rejected/partial buy is not chased; unresolved exits and funding can temporarily
produce exposure different from the intended ten-slot book and are reported.

## Verification and deployment boundary

Offline checks cover synthetic native provider-to-snapshot prices/features, seeded
research-to-adapter intentions, held exits, causal future-row truncation, source
mutations, frozen sizing, temporary SQLite lifecycle, restart, account drift,
notification scope and existing live regression tests. The verification summary
records the actual executed commands/counts and review findings.

Before any separately authorized activation: qualify the producer/client version
on actual native snapshots, assign a dedicated margin account, obtain trusted EOD
state with no unexplained holdings/open orders, run Doctor/reconciliation, confirm
alerts and scheduler configuration, then accumulate untouched PAPER observations.
No broker call, order, release enablement, production deployment or allocation was
performed by this wiring task.

## Research limitations

Scout's corrected native-Turnover record reports exact engine identity (4.4e-16
maximum return difference; zero differing trade dates). That is implementation
validation, not proof of edge. The fixed ETF universe was selected after research
across 23 universes with 50 prior trials; survivorship, selection and multiplicity
remain. About eleven in-sample years, expanding ETF eligibility, S3 cost coverage
1.62 (below 2), auction/capacity uncertainty and realistic fees remain constraints.
Source: `docs/research/SCOUT_P7B_DV2_INDUSTRY_ETF_20261002.md`.


## Completed evidence (2026-10-14)

- Actual local Norgate session: 2026-10-13. Loaded 4,472 historical date rows
  from the frozen 2009 start request, for all 19 ETFs plus the benchmark.
- Native direct-provider and snapshot-consumer price frames match exactly,
  including per-symbol fields/dtypes, inception gaps and benchmark lineage.
- Every full-history feature matches exactly; 21 hypothetical account-holdings
  cases reproduce the research entry/exit intentions. These are offline test
  accounts, not observed broker accounts or forward portfolio results.
- Native artifact: `results/dv2_ind_wiring_20261014/native_qualification.json`.
- Qualified manifest SHA256: `0e63c5174e2ab3c3c699678936845a7a02a72f1b252b2289fcb688fa2a66cf6e`.
- Final focused suite: 321 passed in 54.92 seconds. It includes the DV2-IND
  producer/consumer, EOD-to-decision, successful and uncertain dispatch/restart,
  disabled/account-change guards, no-order terminal settlement, both notification
  outboxes, snapshot/session mutation, normalized account collision checks,
  registry/templates, E2 and capsule release regression tests.
- Broad live/snapshot/quant run: 2,183 passed, 8 skipped. It reported one
  outdated template-budget assertion while that test module was already loaded;
  the assertion was corrected for the new dedicated-account template and the
  complete release suite passed in the final focused run. The full broad run
  is not represented as an all-green final-revision run.
- Final-revision integration gate: 372 passed in 123.20 seconds, covering
  runner, scheduler, reconciliation, strategy host, SQLite state, guarded socket
  dispatch/restart, daily shared/monthly isolation, cycle settlement, both
  notification outboxes, IBKR payload parity, E2 lifecycle, DV2-IND and templates.
  There are no outstanding failed tests or review findings.
- `git diff --check` passed. The repository triage helper classifies the actual
  diff as Tier 3, including lower-tier shared snapshot and strategy-doc surfaces.

Independent read-only reviews covered parity/quant pitfalls, failure modes and
coverage. Material findings fixed: source-history coverage, notification scope,
malformed full-target bypass, disabled saved-release submission and dedicated
account collision. No review findings remain open.

Tier 3 live-impact checklist: existing order clocks/policies, sizing semantics
and quote sources remain unchanged; the new route uses explicitly frozen
Close-T NAV entries. No DB schema or legacy configuration migration, log-field
removal, scheduler enablement or release edits. Generic SQL transactions and
existing Windows paths/locks/restart mechanisms are reused. Existing templates
retain their routes and parse behavior; the new template is disabled/manual.


## Watchdog routing follow-up (2026-10-14)

The owner relayed an independent review finding: the outer
`_run_capsule_notifications_tuple` filter in `scripts/live_ops_watchdog.py`
still reconstructed capsule/CORE5 membership, excluding DV2-IND before either
outbox was reached. Earlier direct-delivery tests bypassed this wrapper, and
therefore did not establish complete watchdog routing. This was an actual
remaining integration defect, even though downstream outboxes were correct.

The outer filter now uses `DAILY_STRATEGY_IMPORT_TUPLE`, matching the existing
daily-summary and outbox-scope gates. Mode filtering, CORE5/capsule membership,
account validation, delivery claims, pending counts and report fields remain
unchanged. NDX/TAA and other monthly pods remain outside this daily route.
No sizing, order, state-schema, release or scheduler behavior was changed.

A regression through the real wrapper and two persisted temporary SQLite
outboxes failed before the fix (zero deliveries) and passed after (execution
exception plus overdue alert, zero pending, no repeat). Discord transport is
mocked; no real notification was sent. A second wrapper test checks both
outboxes for CORE5, all six capsules and DV2-IND, while excluding NDX/TAA.
Independent read-only parity, failure-mode and coverage reviews found no
remaining issues in this bounded correction. The prior test counts above
precede this follow-up; its final affected-suite result is recorded below.


Follow-up verification: **260 passed in 71.44 seconds** across DV2-IND,
OPS watchdog, watchdog setup scripts, daily watchdog/isolation, both daily
notification suites and capsule notifications. This includes the actual
production-baseline monthly watchdog parity test. `git diff --check` passed.
Triage remains Tier 3; parity, failure-mode and coverage reviews completed.
No production merge/deployment, account enablement or real Discord delivery
was performed.


## Fresh readiness audit (2026-10-14)

The owner requested a full engineering-readiness recheck excluding account/YAML
setup. Fresh independent data/quant, lifecycle and coverage reviewers traced
the actual producer, client, scheduler, runner and operator entrypoints rather
than relying on earlier passing tests.

Additional defects reproduced and fixed:

- Client Doctor requested CAPITALSPECIAL for the DV2 `$SPXTR` benchmark.
  It now requests the profile's required TOTALRETURN representation.
- The existing daily account connectivity probe rejected DV2-IND. It now uses
  the central daily membership list, retaining the unused-client-ID guard.
- Five DV2 safety-block log codes appeared among routine Activity events.
  Dashboard V4 now displays them as ungrouped Alerts with the proper evidence tabs.
- DV2 reference comparisons lacked their explicit diagnostic sizing/accounting
  boundary. The auto reconstruction now revalidates snapshot identity and carries
  lineage/nonexact-share/P&L/dividend flags through maps, runner JSON and HTML.
  Unknown manual-reference starting capital stays unverified. Existing reference
  initialization still uses the first pre-submit broker budget; no live sizing or
  historical accounting semantics were changed to make differences disappear.

The additional readiness suite passed **13 tests in 25.03 seconds**:

- Twenty actual local snapshot smoke reads through client Doctor.
- Daily account-probe selection and own-client rejection without connecting.
- All five real INFO-level DV2 safety events remain visible in dashboard Alerts.
- Actual reference simulation executes ten next-open buys, with JSON/HTML caveats.
- Actual scheduler `run_once` and runner `tick` across blank-account EOD, decision,
  VPlan/MOO, actual-holdings close, next dated snapshot/EOD and a second decision.
- Real producer service accepts/exports the profile; real client sync stages,
  validates/promotes files; resulting prices exactly match the native fixture.
  Only HTTP transport is mocked in that compatibility test.
- Snapshot mutation during reference reconstruction fails closed.
- Actual compare-reference runner wrapper discloses a USD 200,000 first broker
  budget versus less than USD 100,000 frozen Close-T decision NAV.

Independent additional evidence: 74 adapter/corporate-action/strategy tests and
53 operator/reference tests passed. The saved real native snapshot was revalidated;
five chronological prefix/full-feature rows matched exactly, and nineteen
independent symbol split simulations preserved entry ordering, held exit and
nominal decision NAV. No provider update, broker or real notification was used.

Final comprehensive suite results follow below. Scope boundary remains unchanged:
code stays in the attached isolated worktree. A deployed producer/client version
pair, real account reconciliation/margin/permissions, actual Discord configuration,
auction fills and forward PAPER behavior are not claimed by these offline checks.


### Final readiness result

**PASS for engineering readiness, with no identified open code findings.**

The frozen final live/dashboard/operator/snapshot/registry/quant suite finished
with **4,593 passed, 12 skipped, zero failures/errors in 1,089.92 seconds**.
All DV2-IND tests ran. Six HPI native parity tests, two HPI full-history replays,
and four older direct/snapshot parity tests retain their explicit opt-in gates.
DV2 native direct/snapshot, full-feature, prefix and split checks were separately
completed. The 24 non-fatal warnings occur in the regression run, including tiny
synthetic reference/correlation windows; they do not establish investment results.

The corrected client Doctor also passed all **20 actual saved-native-snapshot
reads**. Source hashes were unchanged throughout the final suite, and the final
`git diff --check` passed. Independent parity/quant, lifecycle/failure-mode and
coverage reviews have no remaining findings.

Evidence: `results/dv2_ind_wiring_20261014/readiness_tests.xml`,
`readiness_tests.log`, `readiness_source_manifest.json` and
`readiness_summary.json`. These results supersede the prior partial/historical
suite counts for the current revision.

The scheduler test completes one trading cycle and creates the following
session's decision; it does not claim a second completed trading cycle. API
compatibility is same-source-version transport simulation, not a deployed
producer/client version audit. No account enablement, branch merge, production
deployment, real broker order or real Discord message was performed.

Deployment/account settings and on-machine qualification remain necessary:
validated fresh data, a dedicated account with appropriate currency/margin and
permissions, broker reconciliation, unused client IDs, working notifications,
and forward PAPER operation. Real quote/auction/funding/dividend/partial-fill
behavior and research selection limitations remain explicitly separate from
engineering readiness.
