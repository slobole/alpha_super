# MR capsule: independent review and local wiring

Date: 2026-10-05. Scope: the handoff review, followed by the owner's explicit
authorization to implement WIRED in local code and tests. BIL remains the main
variant; SPMO remains the alternative. No broker operation, production deployment,
capital allocation, active release change, commit or push is part of this work.

## Verdict and evidence status

**PM_READY historical evidence remains valid for its recorded source version.
The agreed local WIRED code is implemented; production qualification remains
separate and capsule LIVE is locked.** Scheduler/EOD prerequisites, bounded
snapshot retention, broker symbol mapping and execution recovery are now
implemented. The owner deferred the detailed old/new production decision and
backtest comparison. The latest execution-policy verification record below
supersedes earlier open-item lists and test counts; it is not a live account
or auction-performance claim.
All six DV2/HPI cash/BIL/SPMO identities have local host, release, data, order,
persistence and reconciliation routes. All templates remain disabled; no
production account was activated. The original parking identities retain their
research contracts; the separate cash identities keep idle money as cash.

This is infrastructure qualification. It does not establish a new investment edge,
approve capital, or demonstrate a successful live auction. The original search
included about 100 gate variants and 10 parking variants, not one unseen hypothesis.

## Findings and fixes

Locations refer to the final implementation. P1 means a trading/recovery blocker;
P2 means a material correctness or auditability issue. Some findings concern the
previous research code; others were prerequisites or defects found while wiring.

1. **P1 — stale VIX can select the wrong state or repeat a switch.**
   `strategies/mr_capsule/capsule_pod.py:85` and `:92`, `_gate_open_at_decision` and
   `_gate_switched_at_decision`; `alpha/live/mr_capsule_adapter.py:56`,
   `_validate_vix_history`. Example: prices reach Monday but VIX stops Friday;
   an as-of lookup silently treats Friday's open gate as Monday's decision, or
   repeats Friday's switch. **Fix:** require the exact decision session and full
   VIX history, and compare gate states on the previous pricing session and T.

2. **P1 — parking quantities had no supported live intent and could use the wrong
   capital basis.** `alpha/live/strategy_host.py:208`, incremental order classifier;
   `alpha/live/models.py:52`, `DecisionPlan`; `alpha/live/execution_engine.py:142`,
   `_build_incremental_entry_exit_vplan`. Example: BIL emits a target of 900
   shares; the old host rejects it. Translating that target to a weight against
   bootstrap capital, then resizing with a different broker NAV, buys the wrong
   quantity. **Fix:** persist an explicit whole-share target map and freeze stock
   entry dollars and ETF targets against the same Close_T account NAV.

3. **P1 — full-account sizing needs exclusive account ownership.**
   `alpha/live/release_manifest.py:467`, `validate_release_list`.
   Two enabled capsule pods sharing an account could each allocate the same cash.
   **Fix:** reject enabled same-mode account sharing whenever either release is
   a capsule. Require budget fraction 1.0, a trusted EOD account snapshot, and an
   explicit margin-account confirmation for enabled paper/live releases.

4. **P2 — a missing ETF open could cause a fictitious liquidation.**
   `alpha/engine/strategy.py:1164`, `_is_missing_price_liquidation_exempt` and
   missing-price handling; `strategies/hpi/stateful_long.py`, corresponding hook;
   `strategies/mr_capsule/capsule_pod.py:81`, parking exemption.
   Example: BIL is held, its next Open is absent, but its valuation Close exists.
   The old removed-stock path could liquidate it at the prior Close. **Fix:**
   exempt parking ETFs from that stock-removal rule, cancel the unfillable order,
   and retain the position. An absent held-asset decision Close blocks live intent;
   an absent valuation Close remains an error, never a made-up fill.

5. **P1 — a persisted order delta can become stale before submission.**
   `alpha/live/runner.py:3209`, `submit_ready_vplans`;
   `alpha/live/execution_engine.py:52`, `validate_mr_capsule_execution_contract`.
   Example: a manual BIL sale, split or outstanding order appears after
   the decision. Sending the old delta would change the intended portfolio.
   **Fix:** refresh broker state immediately before claiming submission; require
   exact base holdings, finite cash, matching account/strategy identity,
   and no outstanding orders. Do not silently recompute the frozen targets.

6. **P1 — matching positions alone do not prove this order batch completed.**
   `alpha/live/runner.py:3802`, `post_execution_reconcile`.
   Example: an external trade happens to produce the target holdings while a
   submitted capsule order is missing or partially filled. **Fix:** require one
   correlated broker order and reconciled signed fills for every persisted request.
   Under the latest owner policy, a conclusively terminal residual may complete
   using actual holdings; missing/open evidence cannot. Missing initial
   ACK can recover from subsequently obtained correlated order/fill evidence.
   An execution-uncertain cycle blocks the next decision.

7. **P1 — completion written in separate transactions can strand a cycle.**
   `alpha/live/state_store_v2.py:808`, capsule completion; `:750`, safe abandonment;
   `:1007` and `:1207`, atomic creation and claim.
   Example: a crash after VPlan completion but before DecisionPlan completion
   removes the VPlan from reconciliation while the next-day guard keeps blocking.
   **Fix:** complete both statuses atomically. VPlan insertion, the decision's
   ready transition, and the submission claim also check current persisted state
   under a write lock, so a stale worker cannot resurrect an abandoned cycle.
   Only a cycle proven never claimed
   or submitted may be abandoned atomically and permit a later signal; absence
   of an ACK alone is not proof that nothing was sent.

8. **P2 — the new data route must preserve padding, volume and benchmarks.**
   `data/norgate_snapshot_store.py:501`, MR profile contract; `:832`, `load_raw_prices_df`;
   `scripts/export_norgate_snapshot.py:307`.
   HPI stocks cannot inherit DV2 padding, missing SPMO Volume cannot be treated
   as a valid no-trade observation, and `$SPX` price return must not be labelled
   the DV2 total-return benchmark. **Fix:** separate profiles with per-symbol
   padding/provenance, strict helper coverage, native Dividend/Volume validation,
   and `$SPX` to `$SPXTR` benchmark mapping for the new profiles only.
   Real-data export also exposed one native VIX gap, 1991-03-01. The contract
   records exactly this historical exception and never fills it; the approved
   observed-close expanding mean remains unchanged. Additional gaps, stale data
   or a later historical backfill fail qualification instead of silently changing
   the gate history.
   The next real-data comparison found request-window sensitivity: native VLO
   with ALLMARKETDAYS from 1990 returned zero dividends, whereas the frozen
   1998 request returned 115; all other fields matched. **Fix:** pin non-VIX
   native requests to the research's 1998 boundary, with declared pre-window-only
   history for symbols whose primary request is empty. No dividends are repaired
   or merged. HPI's capsule reader also applies the direct loader's membership
   window before requesting prices (`strategies/hpi/stateful_long.py:180`):
   1,173 in-window members instead of 1,304 all-history members. This retains
   departed in-window members and removes 40 extraneous priced symbols, without
   weakening the PIT decision filter.

9. **P1 — equal snapshot prices can still change a DV2 threshold decision.**
   `data/norgate_snapshot_store.py:581`, `_restore_mr_capsule_native_fields`.
   Combining native float32 prices with
   other symbols can promote them to float64. In the actual DV2 indicator,
   float32 `105/100 - 1` is below 5%, while float64 arithmetic is above 5%:
   identical stored prices can therefore admit a stock excluded by the direct
   loader. **Fix:** persist native fields/dtypes per symbol and adjustment,
   restore them losslessly before concatenation and signal calculation, and
   require dtype parity in the data qualifier. The regression runs the actual
   DV2 indicator and opportunity filter for both new profiles. Existing profiles
   retain their prior branches.

The handoff's reference-dividend warning was investigated and **not reproduced**
on the current engine. Its disabled-ledger context already overrides constructor
enablement, and reference generation resets the reported mode. Regression tests
exercise the real capsule constructors. No reference-accounting code was changed.

## Frozen strategy and accounting contract

- **Gate:** m_T is the mean of observed VIX closes from 1990-01-02 through T,
  requiring at least 500 observations. Open when VIX_T > m_T; after opening,
  close only when VIX_T <= m_T and 15 subsequent sessions have elapsed. The
  opening close has counter zero. Exit signals are never gated.
- **DV2:** PIT S&P 500; DV2(126) < 10, Close > SMA200, Return126 > 5%; rank by
  NATR14 descending; at most 10 stock slots; entry dollars NAV_T / 10. Exit when
  Close_T > High_(T-1). Indicator formulas and tie rules remain those of the
  unchanged `DVO2Strategy` parent and `alpha/indicators.py`.
- **HPI:** PIT S&P 500; at least two negative-return HPI votes below 30 across
  horizons 2/3/5, IBS < 0.10 and Close > SMA200; rank by Turnover descending;
  10 stock slots, NAV_T / 10 entry dollars. Each HPI uses the preceding 1,260
  return observations, excluding T; for a nonpositive return, HPI is 100 times
  the count of prior returns <= today's return divided by the count <= 0.
  Exit on IBS > 0.90, RSI2 > 90, or lost PIT membership. Preserve pending exits
  and the parent's same-auction slot reuse, G-033. HPI live readiness requires
  complete features for at least 400 members and 80% of today's PIT universe.
- **Parking:** BIL/SPMO never consume stock slots or undergo stock exit rules.
  `idle_T = max(NAV_T - retained_stock_value_T - new_entry_dollars_T - .01*NAV_T, 0)`.
  Re-target on the last session of the ISO week or a gate transition, using the
  known next-session calendar. For SPMO mode with gate closed,
  `w_T = min(1, .08 / (sample_std(last 20 close returns)*sqrt(252)))`;
  unknown volatility gives zero weight; zero measured volatility is capped at
  weight 1 by the existing rule. The last 20 sessions must each have
  Volume > 0. `N_SPMO = floor(w_T*idle_T/Close_SPMO,T)` on re-target dates;
  otherwise retain SPMO, except gate-open/invalid-weight liquidation. BIL mode
  fixes the SPMO allocation at zero.
- **BIL remainder:** `N_BIL = floor(max(idle_T - SPMO_value_after_T,0)/Close_BIL,T)`.
  Buy only on re-target dates; reductions may occur daily. Trade only when the
  dollar change exceeds 1% of NAV, or when the held allocation must become zero.
  This band and whole-share rounding mean the cash buffer is not a solvency
  guarantee. Cash mode emits no parking orders and assumes 0% idle-cash return.
- **Prices and costs:** CAPITALSPECIAL tradeables; TOTALRETURN benchmarks only.
  Research slippage 2.5 bps, commission $0.005/share with $1 minimum; house
  dividend withholding 25%. BIL begins 2007-05-30; preceding cash earns zero.
  Dividend entitlement and credit timing remain the engine's declared ledger
  contract. Negative cash remains unfinanced and diagnostic (G-023).
- **Book:** the two research books remain unchanged: independent DV2/HPI pods,
  50/50 allocation and annual rebalance. Local wiring does not automate transfers
  between two broker accounts or authorize this allocation.

No parameters were optimized in this work. The copied parent iterate bodies were
reviewed and tested against the parents; only gate, parking exclusions and the
parking call remain different. A broader parent refactor is unnecessary here.

| Quantitative check | Result and boundary |
|---|---|
| Lookahead and target leakage | Trailing inputs end at T; HPI's 1,260-observation reference excludes T. Prefix tests perturb future prices. The HPI next-open tradability assumption remains G-033. |
| Survivorship | Historical PIT membership and removed names retained. HPI stocks remain unpadded; DV2 preserves its parent's native padding. |
| Data mining and multiple comparisons | About 110 prior forks; the same-window parking selection remains selected evidence. No new tuning or untouched-holdout claim. |
| In-sample contamination | No learned normalization, fitted scaler or train/test split added. This does not undo the original selection on historical data. |
| Regimes, sample size and warm-up | Research trades from 2004 with stock history from 1998 and VIX from 1990. HPI requires 1,260 prior returns plus horizon warm-up; live cross-section guard is 400/80%. Synthetic tests are correctness cases, not statistical observations. Recent-period DSR and regime weaknesses remain. |
| Corporate actions and adjustments | CAPITALSPECIAL stocks/ETFs and native dividends; total-return benchmarks only. Changed broker quantities block stale orders. Parent historical removed-stock liquidation is a model convention, not broker-fill evidence. |
| Costs and live divergence | Research uses 2.5 bps plus $0.005/share/$1 minimum; financing is absent. Auction impact, actual taxes and margin capacity need account evidence. Stock quote sizing, partial fills, G-033 and dividend-free references limit exact live parity. |

## Smallest live design implemented

```text
[Exact-date PIT prices + VIX since 1990]   [Trusted broker EOD state at T]
                   \                         /
                    [Close_T signals + NAV_T]
                  gate -> stock orders -> parking
                              |
       *** CRITICAL *** signals/dollars/ETF targets use <= Close_T
                              v
 [Persist DecisionPlan: entry dollars / exits / ETF target shares]
                              |
       [Pre-submit quote + holdings/identity checks -> VPlan]
                              | next XNYS open, MOO/OPG
       [Durable request identity -> ACK -> signed fills]
                              |
       [Broker reconciliation -> atomic cycle completion]
```

Stage A uses the two explicitly named `_cash` identities. It keeps the same gate,
stock rules, exact data and account guards; idle cash stays cash. Stage B uses
the `_bil` identities and the additive target-share intent. SPMO uses that same
intent with the additional frozen parking rule. Changing mode requires a reviewed
account/state transition; the adapter rejects silent adoption of another strategy's
positions or incompatible parking holdings.

The sizing contract `mr_capsule_close_targets_v1` is:

```
NAV_T = cash_EOD,T + sum(broker_shares_i,T * Close_i,T)
stock_entry_dollars_i = entry_weight_i * NAV_T
live_stock_target_i = floor(stock_entry_dollars_i / observed_live_reference_i)
live_ETF_target_i = persisted_integer_target_i,T
order_delta_i = live_target_i - unchanged_broker_shares_i
```

The target-share map is additive to the incremental plan. Omitted symbols remain
untouched. Only BIL/SPMO are allowed, according to the named mode. Target shares
are finite nonnegative integers and cannot overlap entry/exit intents. Existing
strategies retain their sizing route. SQLite adds a JSON column defaulting to `{}`;
old rows and callers remain compatible. Audit output includes the explicit map.

**Parity boundary:** stock dollar intent matches research, but the existing live
route converts those dollars with a pre-submit quote; research sizes with Close_T.
Therefore live stock shares and fills need not equal historical shares and fills.
ETF target quantities remain exact through VPlan construction. The HPI next-open
tradability marker is an assumption, not tomorrow's observed Open; a failed exit
can therefore coexist with a filled entry and must remain unresolved.

Owner-directed cash policy update: a finite broker cash change after the
decision no longer invalidates a capsule cycle. Credits, debits and negative
margin cash are accepted without resizing Close-T stock entry dollars, ETF
share targets or an already persisted VPlan. Broker cash remains authoritative;
no synthetic dividend credit is added. Nonfinite cash, invalid NAV, changed
holdings, wrong account identity and outstanding orders remain blocking checks.
This policy accepts cash changes without classifying their cause, including
large withdrawals; it does not prove buying power or add a financing limit.
The broker-stub regressions cover construction, persisted-plan submission,
restart, duplicate prevention and retained blockers. No broker was contacted.

Cash-policy verification (2026-10-05): the 15 new acceptance cases failed on the
old equality guard. After removal, 287 tests passed across capsule target shares
and recovery plus shared runner, order clerk, reconcile, release manifest,
scheduler service and scheduler utilities. Task-scoped triage is Tier 3;
independent parity, failure-mode and coverage reviews found no blocking issue.
Next-open timing, sizing math, quote source, schemas, active releases and logging
fields are unchanged; focused restart and duplicate-submission checks passed.

Separate malformed-input isolation gaps observed while adapting the test
fixtures remain open: NaN broker cash can fail the snapshot-cache NOT NULL
constraint before per-plan handling; an invalid sizing-contract identity can
also make abandonment raise. The final error-isolation fixture uses infinite
cash with a valid capsule identity, exercising the retained numeric guard.
Neither malformed-input path was broadened or represented as fixed here.

Parking's research trade IDs are not live order identities. Persisted VPlan and
request keys provide idempotency across restart and recovery. No new parking-ID
state machine was added.

## Scheduler and upgrade safety follow-up (2026-10-05)

This bounded phase protects the existing monthly LIVE routes while fixing the
capsule's first-day readiness. It does not complete the remaining WIRED work.

10. **P1, fixed — capsule profiles never reach the freshness gate.**
    `alpha/live/scheduler_utils.py:54`: register both new profile names with the
    existing `$SPX` heartbeat. Existing profile mappings and data rules are unchanged.
11. **P1, fixed — decision building can starve its missing EOD prerequisite.**
    `alpha/live/scheduler_service.py:207`: new capsule decisions require the same
    completed exchange session for pricing and trusted EOD, and a resolved prior
    cycle from a strictly earlier session. A missing/failed EOD can therefore use
    the existing capture phase. Preserve all shared priorities and dispatch paths;
    existing plans still expire, build, submit or reconcile as before.
12. **P1, fixed — stale broker state can be relabeled as a fresh capsule EOD.**
    `alpha/live/runner.py:3722`: apply the existing CORE5 source validation to
    capsules, including source date/close boundary, future time, account identity
    and open orders. Persist the actual broker response time. NDX/TAA retain their
    prior capture behavior; CORE5 validation is unchanged.
13. **P2, fixed — invalid saved prerequisites can look like ordinary idle.**
    `alpha/live/scheduler_service.py:243` and `:506`: an invalid same-session EOD
    already present in history produces `mr_capsule_eod_snapshot_untrusted` /
    `manual_review_pending`. It is not overwritten. Unproven terminal cycles use
    the existing parked-execution status. Valid EOD with stale pricing does not
    trigger this diagnostic. Existing immediate/future work still takes precedence.
14. **P2, fixed — concurrent first upgrade can fail existing LIVE startup.**
    `alpha/live/state_store_v2.py:362`: serialize the schema inspection and
    additions with `BEGIN IMMEDIATE`, retaining the existing context's commit or
    rollback. Two workers opening the old database previously produced
    `duplicate column name: target_share_json_str`. The regression uses two real
    SQLite connections and verifies successful upgrade and target-share roundtrip.
    An in-memory mutation removing the lock reproduces the failure. The lock
    covers local schema work only. Earlier legacy migrations preceding it are
    unchanged; no universal historical-upgrade claim is made.
15. **P2, fixed — Bench catalog regression after adding six analysis hooks.**
    `tests/test_bench.py:184`: update the expected capacity count from 48 to 54
    and explicitly require all six capsule identities. No catalog behavior changes.

Verification on the final scheduler/startup source:

- **567 passed** across 20 offline suites covering scheduler, full-year calendars,
  monthly NDX/TAA hosts, CORE5, runner, release validation, orders/reconciliation,
  persistence/migration, snapshot sync, capsule targets/recovery and concurrency.
  Artifact: `results/research/mr_capsule_review_20261005/verification/scheduler_upgrade_regression.xml`.
- The new capsule scheduler file contributes 86 cases. It uses the real profile
  lookup, gate, calendar, SQL and EOD capture; broker/data boundaries are stubbed.
  Cases include a real `run_once` capture-to-build dispatch sequence, early close,
  next-morning cutoff, prior EOD preservation and active-cycle continuation.
- Capacity/strategy-analysis compatibility: **23 passed**. Bench initially had
  **124 passes and one obsolete-count failure**; the corrected catalog case passed
  with 14 store/concurrency cases. These are separate runs, not added to 567.
- Independent parity, failure-mode and coverage/timing reviewers found no remaining
  material issue in this bounded delta. Tests remain the evidence; reviews do not
  replace them. Tier 3, including the shared-store startup surface.
- Next-open timing, sizing math and price sources are unchanged in this phase.
  Schema format/defaults remain additive, existing logging fields remain present,
  restart/idempotency and Windows file-backed SQLite cases passed. No active release
  was edited and no broker, Norgate load, VPS or production database was accessed.

Packaging remains part of release safety. The updated owned-file manifest contains
55 paths, including new untracked modules/tests and the Bench correction. A partial
bundle is unsafe: `release_manifest.py` and `strategy_host.py` import the new
`alpha/live/mr_capsule_adapter.py` at startup, even if capsules are disabled.
Any eventual commit must include required new files as well as tracked changes.
No commit or push was performed. Existing named-column database readers/writers
remain structurally compatible; rollback must not put old software in charge of
new capsule cycles. This phase does not certify the complete dirty worktree or a
particular production checkout for pull/deployment.

Open WIRED items include bounded snapshot caching, canonical broker symbol
mapping, partial/unfilled-order recovery (especially exits and parking funding),
unique deployment client IDs, and account-specific funding qualification. Budget
fractions below 1.0 still require coherent stock-plus-parking sizing. The owner
accepted finite cash postings without a cash-drift threshold; that policy remains.

## Verification record

The consolidated machine-readable record is
`results/research/mr_capsule_review_20261005/verification_summary.json`, with
source/evidence hashes, latest-per-module test counts and qualification verdicts.

- Initial review baseline: 149 focused capsule/registry tests passed.
- Final combined local verification: 797 passed, 8 skipped, one existing synthetic
  fixture warning. JUnit: `results/research/mr_capsule_review_20261005/final_verification.xml`.
  The eight skips are opt-in real-Norgate tests of the HPI parent; none of the new
  capsule tests were skipped. This includes the final recovery, competing-worker
  and account-isolation regressions.
- Separate non-overlapping suites: 51 shared-engine/timing/indicator/data-tool
  tests and 163 Bench/portfolio tests passed. These bring the checked test count
  to 1,011 passes, without adding repeated reviewer runs. See
  `shared_engine_verification.xml` and `bench_portfolio_verification.xml` in the
  same evidence directory. The shared-engine suite has one existing correlation
  warning from a degenerate synthetic fixture.
- The offline qualification tool adds 20 separate passing cases, bringing the
  total to **1,031 passed and 8 explicitly skipped**. Its JUnit artifact is
  `offline_tool_verification.xml`. These test counts exclude repeat executions
  of the same cases during independent review.
- The native-dtype correction adds nine distinct passing cases, bringing
  the total to 1,040 passed and 8 explicitly skipped. Its focused
  rerun passed all 69 cases, including repeated surrounding checks:
  `verification/native_dtype_qualification_junit.xml`. An independent reviewer
  checked lossless casts, NaNs, large integer precision and staggered histories.
- The documented VIX-gap correction adds five distinct passing cases. The
  unique total at that stage was 1,045 passed and 8 explicitly skipped.
  The last focused suite passed 87 cases; see
  `verification/known_vix_gap_qualification_junit.xml`. Its independent reviewer
  verified the real input hash, the exact missing date, unchanged gate arithmetic
  and rejection of any additional gap or later historical backfill.
- The HPI window correction adds two cases and the native-request boundary adds
  eleven. The final total is **1,058 passed, 8 explicitly skipped, zero failures**,
  using the latest complete suite for each test module (obsolete parameter-case
  names and repeated executions are not added). The last runs passed 103 and
  154 cases respectively: `verification/hpi_window_parity_junit.xml` and
  `verification/native_request_boundary_junit.xml`. The latter also covers
  existing snapshot and CORE5 behavior after the exporter changes.
- Real-indicator host tests cover all six identities, gate open/close, weekly
  timing, full slots, HPI pending exits, PIT exclusion, and future-prefix invariance.
  Inputs are deterministic synthetic histories; signal functions are not mocked.
- Target-share tests cover model/SQL roundtrip, backward compatibility, whole
  shares, unchanged ETF targets, mixed stock/ETF intents, omitted holdings,
  partial fills, ACK recovery, duplicate submission and account drift.
- The six handoff engine runs and PM determinism check run sequentially under
  `results/research/mr_capsule_review_20261005/baseline/`, with original source
  hashes and isolated outputs. They do not overwrite Bench results.
- The final snapshot-reader delta also passed 156 surrounding snapshot, CORE5,
  capsule host and signal-parity regression cases; these overlap the distinct
  total above. Artifact: `verification/snapshot_delta_regression.xml`.
- Real direct-versus-snapshot qualification is performed by
  `scripts/research/mr_capsule_build_20261004/qualify_wiring.py`. It saves raw
  input frames and manifest/source hashes and compares values/null masks with
  zero numeric tolerance. Observed decision-field dtypes must also match;
  differences and entirely empty schema/nonmember columns are explicitly
  reported, not silently dropped.

### Reproduced engine runs

The requested fresh DV2/BIL PM check passed all three checks:
capital $100,000 to $2,941,577 versus $200,000 to $5,901,662 (ratio 2.006),
declared TOTALRETURN benchmark versus actual `$SPXTR` with zero gap, and exact
repeat terminal equity $2,941,577.2735. See `baseline/pm_readiness.json`.
The four earlier 2026-10-04 variant PM artifacts also passed and were inspected.
The fresh reproductions deliberately use the captured original capsule modules;
the final-code historical-equivalence and data-route checks below qualify the
small research-behavior changes separately.

All six engine runs completed through **2026-10-02**, with 5,724 recorded sessions
each. Their NAV, cash and portfolio-value series match the saved build outputs
exactly; every economic transaction field also matches exactly. The only CSV
difference is a constant offset in the internal `order_id` counter for later
runs in the same Python process. Asset, date, shares, price, commission and
`trade_id` remain identical. `compare.py` completed successfully.

| Pod / mode | Final NAV from $100,000 | Negative-cash sessions | Lowest cash / NAV |
|---|---:|---:|---:|
| DV2 / cash | $2,846,881.60 | 133 | -9.494% |
| DV2 / BIL | $2,941,577.27 | 141 | -9.513% |
| DV2 / SPMO | $3,424,962.00 | 136 | -8.777% |
| HPI / cash | $1,373,745.33 | 127 | -8.316% |
| HPI / BIL | $1,403,311.66 | 130 | -8.325% |
| HPI / SPMO | $1,595,111.80 | 128 | -7.642% |

The main BIL variant therefore needs the financing caveat at least as much as
the SPMO variant. No financing charge was retrofitted into the frozen research.
The reference comparison also reproduces HPI's 5,998/5,998 stock events and zero
cash-mode NAV difference. DV2's research-replica event Jaccard remains 0.98654;
it is not an exact source-replica claim. No SPMO was held after a gate-open
decision, and neither parking run produced a short position.

`compare.py` retains its frozen **2026-09-24** cutoff for return/DSR windows;
its full cash-NAV comparison extends through 2026-10-02. The reproduced
SPMO-capsule DSR is 0.96562 for the full window and 0.77968 from 2018-02,
using 110 trials. These are selected-history diagnostics, not new validation.

### Real native-data qualification

`qualification_v3/qualification.json` passed every comparison for both profiles
through 2026-10-02. DV2 has 8,511 decision-input columns over 7,232 rows; HPI has
8,231 columns over the same dates. Across those 121,078,144 price/volume/turnover/
dividend cells, the mismatch count is zero, with zero observed dtype differences
and identical column order. PIT membership, complete observed VIX history and
both true total-return benchmark routes also match exactly. Source hashes were
unchanged during the run. Manifest hashes and native request provenance are saved.

The failed `qualification/` and `qualification_v2/` attempts are deliberately
retained. They exposed the single historical VIX gap, VLO's request-boundary
dividend behavior and HPI's symbol-window mismatch. No comparison tolerance was
relaxed. Four sequential raw VLO probes and their hashes are saved under
`verification/vlo_native_probe/`. All Norgate-loading jobs ran one at a time.

`offline_qualification/offline_qualification.json` also passed without skips or
source/input hash changes. It confirms all six historical reproductions, exact
gate-switch equivalence on all 7,232 input sessions (including all 5,724 baseline
decisions), and no missing ETF Open/Close on historically exposed sessions.
Thus the gate-calendar and missing-ETF-price fixes do not change those six runs.

For all six identities, the real-indicator host and an independently seeded
research strategy produced identical intents, state, sizing NAV and execution
session for the saved 2026-10-02 inputs, targeting 2026-10-05. Accounts in this
check are explicitly reconstructed fixtures, not broker truth. This date has
no stock entries, so entry/gate-transition coverage comes from the unmocked
synthetic indicator tests, not from claiming this one historical date covers
every branch. It is intent parity, not auction-fill or live-account NAV parity.

### Live-impact checklist

| Question | Answer |
|---|---|
| Timing | Close_T decisions, next XNYS open MOO; no same-close execution added. |
| Sizing | Intentionally extended for capsule only: frozen Close_T NAV for entry dollars and explicit ETF target shares; other identities unchanged. |
| Reference prices | Existing pre-submit source retained and recorded; no silent Close/Open substitution. Research share parity is bounded as above. |
| Persistence | Additive JSON field with migration/default; durable submission keys; capsule completion atomic. |
| Logs/UI | Existing fields retained; explicit share map added to decision detail/trace. |
| Windows/restart | Local paths and temp SQLite tested; sequential Norgate jobs; uncertain submissions block fresh decisions. |
| Releases | Six disabled examples; frozen parameters/profile/budget/timing/margin guards; existing active release files untouched. |

Risk tier is **Tier 3**, including Tier 2 engine/data and Tier 1 strategy changes.
Three independent reviewer agents (`gate_parking_review`, `pod_parent_review`
and `wired_design_review`) cover quant pitfalls, parity, failure modes and coverage.
Review findings are fixed and checked with regression tests, not treated as proof
by reviewer opinion alone.

## Before enabling PAPER or LIVE

1. Deploy and validate the new reader/host/execution/store code and both profile
   exporters on the intended producer and client hosts in the documented schema
   rollout order. Obtain exact-session snapshots and retain their hashes.
2. Assign one dedicated margin account per pod, full-account budget 1.0, and an
   explicitly selected cash/BIL/SPMO identity. Confirm margin eligibility and
   same-auction funding, with cash/NAV expressed in the USD pricing currency.
   Do not enable multiple modes for one account or use it from another deployment
   or external trading process; local release validation cannot police those.
3. Initialize from a trusted same-session broker EOD snapshot. A YAML bootstrap
   amount is insufficient. Existing positions need an explicit reviewed state
   migration, including strategy identity, trade IDs and pending HPI exits.
4. Run PAPER first and inspect the complete chain: data/decision, VPlan quantities,
   request identity, ACK, fills, reconciliation and next-session EOD state. Include
   a BIL funding sale plus stock entry, no-order days, partial/rejected orders,
   restart, stale data, a gate transition and a weekly parking adjustment.
5. Resolve execution-uncertain cycles from broker evidence before continuing.
   Do not mark them completed merely to clear a dashboard warning. A proven
   unsubmitted abandoned cycle does not commit its proposed strategy state.
6. Review actual financing, commissions, auction liquidity, dividends, withholding
   and account-specific capacity. Existing reference comparison is price-return
   only and cannot certify BIL's dividend-inclusive research return or official
   broker account performance. The six Bench capacity/timing hooks are now wired
   and covered by compatibility tests; they are not account-specific capacity proof.
7. Obtain separate explicit authorization for production activation and capital.
   The research book's annual 50/50 rebalance is not an account-transfer service.

Residual statistical limitations remain: selected history, weaker recent-period
DSR, regime dependence and SPMO's thin early trading history. Operational wiring
does not remove them.


## Owner execution-policy update (2026-10-05; latest)

This section supersedes earlier strict-full-fill and open-WIRED-item descriptions.
The code remains local and uncommitted; active release YAMLs were not edited.
The current repository base is `54b417f` (other work was committed separately).

- Budget remains **1.0**, with one dedicated margin account per capsule pod.
  Cash postings do not trigger a drift threshold or recompute a frozen plan.
- Capsule buy batches require actual account margin type, USD buying power
  against all buy notionals without credit for pending sales, and individual
  broker what-if checks. This is a precheck, not guaranteed basket funding/fills.
- Original request identities are built before dispatch is reordered: BIL sells,
  other sells, then buys. Orders remain next-open MOO; sales are not awaited.
- Reconciliation uses account-scoped order/fill evidence and fresh actual shares.
  A terminal missed/partial buy is accepted with a durable warning; it is never
  automatically completed. A still-open or uncertain order blocks a new cycle.
- A terminal unfilled exit-to-zero may receive **one** same-session MKT sale.
  BIL reductions may also be completed, only down to their frozen target.
  Nonzero stock/SPMO reductions do not receive automatic completion.
  Before every attempt, refresh holdings and all-client open orders:
  `Q_sell = min(original sale remaining, max(actual shares - frozen target, 0))`.
  A unique SQL claim precedes the network call; restart never blindly resends it.
- Recovery requests expire at the original XNYS session close, including early
  closes, through broker GTD expiry with no DAY fallback. Exact SMART MKT+GTD
  acceptance is an explicit PAPER prerequisite. IBKR documents GTD stock/ETF
  support and order fields, but product/order-type/destination combinations
  still need broker qualification:
  [GTD](https://www.interactivebrokers.com/en/trading/ordertypes.php?menu=B),
  [Order fields](https://www.interactivebrokers.com/docs/tws-api/ref/order).
- A final residual after that attempt is accepted and reported from actual
  holdings. Unknown transport/order outcomes remain VERIFY with no automatic
  retry. A manual repair closes a cycle when original orders are resolved and
  broker order/fill evidence explains the holdings; matching holdings alone
  cannot prove an unknown order will not execute later. No SQL edit is required
  for an evidenced manual repair.
- Recovery/manual fills are marked `late_execution`, with actual time/price and
  no opening-auction reference. Original opening fills retain target-session
  references when available; a later day's tick-open cannot relabel them.
  Reference failure is recorded diagnostically and does not stop exit recovery.
- Settlement, PodState, decision metadata, durable warning and both completion
  statuses commit atomically. Invalid account/core snapshots never replace
  trusted state; optional unavailable margin diagnostics do not strand a cycle.
- Watchdog delivery reads the durable queue independently of red transitions.
  Alerts contain symbols, requested/filled/remaining shares, action and reason,
  including unknown quantities explicitly. Stale warnings are marked historical;
  long warnings use numbered messages without dropping symbols. Delivery is
  at least once: a crash after Discord accepts a message can repeat it.
- Snapshot caching retains at most one frame per capsule profile. Legacy profile
  caching remains unchanged. Broker aliases BRK.B/BRK B and BF.B/BF B are explicit
  across qualification, quotes, positions, orders and fills. Six disabled example
  releases use distinct client IDs 41-46. Capsule `mode=live` is rejected even
  when loading persisted releases, before broker resolution.

The same-day MKT policy intentionally differs from the next-open backtest in
execution time, price and costs. Neither its frequency nor economic impact has
been measured by this implementation. No signal, research parameter or additional
parameter search was introduced by this execution-policy phase.

### Latest live-impact checklist

| Surface | Result |
|---|---|
| Existing pods | Capsule routing is opt-in. NDX/TAA remain on existing submission/reconcile paths. No claim of newly run old/new real-date comparison. |
| Timing | Original MOO timing preserved; one expiring same-session MKT recovery is an intentional capsule-only addition. |
| Sizing | Close_T entry dollars and ETF target shares remain frozen; recovery uses only the actual residual. Budget 1.0. |
| Reference prices | Original-session diagnostics retained; late fills separately marked; no later-day tick-open substitution. |
| Persistence | Additive capsule tables and optional request field; original getter shapes unchanged by default; atomic completion and unique recovery claims. |
| Operations | Existing active YAMLs unchanged; broker evidence includes all clients without order binding; no real broker/Discord operation or deployment performed. |
| Windows/restart | File-backed SQLite concurrency, restart, partial-send and rollback scenarios covered offline. |
| Remaining qualification | Exact account/TWS expiry and margin acceptance, actual paper execution/Discord transport, and the owner-deferred real-date comparison. |

Risk tier: **3**, with shared data/store/model surfaces. Review roles covered by
`capsule_broker` (quant timing/failure modes), `capsule_data_release` (coverage and
failure modes), and `wired_design_review` (shared-path parity, transactions and
notifications). Implementer and independent review contributions are distinguished
in the verification artifacts; review opinion does not replace test results.

### Final owner-policy verification

The final full repository Python run completed in 2,217.64 seconds:
**7,484 passed, 5 failed, 34 skipped**, with 190 warnings and 73 passing
subtests (7,523 collected top-level cases; JUnit includes the 73 subtests).
All **639 capsule-named cases passed**, with no skips or failures.
All **233 Node/dashboard tests passed**. The suite is not fully green.

The five failures are the two client-operations diagnostic tests, the two
saved-engine HPI Scout gates, and the PTA Good Friday test listed in the
[Claude baseline review](MR_CAPSULE_WIRING_CLAUDE_REVIEW_20261005.md#evidence).
That review records them failing on clean HEAD 5c0d48d. This final run adds
no different failing test; attribution uses that existing clean-HEAD evidence,
not a new baseline checkout run. Exact node IDs and failure messages are
retained in the verification JSON and full log.

All 59 owned Python file hashes matched the source frozen during the final
suite. All 69 owned paths exist. The git whitespace/diff check passed.
The diff for alpha/live/releases and both capsule portfolio YAMLs remains
empty at verification time. The MR task made no source changes after the
final run. Concurrent CORE5 work then changed shared files, as noted below.

Evidence under results/research/mr_capsule_review_20261005/verification:
- full_suite_owner_policy_final.log and .xml: completed full Python suite.
- dashboard_node_tests.log: completed Node suite.
- owner_policy_verification.json: source/artifact hashes, counts, failing
  node IDs, baseline attribution, review roles and remaining qualification.

The earlier full_suite_owner_policy.log is an interrupted intermediate run,
not the final gate. No broker order, real Discord delivery, deployment,
commit or push was performed. Exact SMART MKT+GTD broker qualification and
the owner-deferred real-date comparison remain explicitly unperformed.

### Concurrent work after the final MR suite

After the full suite and its source-hash check completed, concurrent CORE5 work
changed shared files, including order_clerk.py, release_manifest.py and runner.py,
plus incubation.py and reference_compare.py. These changes are outside this MR
task and were preserved. Initially removing only six new CORE5 adapter methods
in memory reproduced the exact tested order_clerk.py hash; the repository file
was not reverted. The shared changes remain in progress.

A focused run of order-clerk, socket, release, capsule execution, submission and
next-day tests completed with **213 passed in 21.85 seconds**. Source observations
before/after are saved in owner_policy_verification.json. Ongoing external edits
mean this is bounded additional evidence, not whole-tree final qualification.

The completed broad-suite result applies to its recorded frozen MR source hashes.
Run the combined integration gate after concurrent work finishes, before merging
or deploying. The owned-path manifest lists capsule-touched paths but cannot now
be treated as permission to stage every hunk of shared files as capsule work.
