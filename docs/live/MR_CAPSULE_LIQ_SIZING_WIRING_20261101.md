# MR capsule liquidity sizing: wiring and deployment prerequisites (2026-11-01)

Two new identities are classified **WIRED**: the BIL pods of the MR capsule, DV2-G and HPI-G, with each new entry
sized by the stock's liquidity. Their release templates are disabled and their account fields are unfilled. No
actual release, account allocation, VPS, broker connection or order was created by this work.

WIRED is a code integration result. Physical deployment still requires the account's forward execution evidence.
The local tests do not supply it.

| Identity | Template | Client id | Data profile |
|---|---|---|---|
| `strategies.mr_capsule.strategy_mr_dv2_vix_gated_liq_bil` | `pod_mr_dv2_vix_gated_liq_bil_daily_moo.yaml.example` | 51 | `norgate_eod_sp500_mr_capsule_pit` |
| `strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_liq_bil` | `pod_mr_hpi_vote_vix_gated_liq_bil_daily_moo.yaml.example` | 52 | `norgate_eod_sp500_hpi_mr_capsule_pit` |

The 50/50 book is `portfolios/mr_capsule_liq_bil.yaml`. Evidence and the decision record:
`docs/research/GROWTH_MR_PRODUCTS_20261101.md` and `docs/research/MR_POSITION_SIZING_20261101.md`. Plan on half of the
measured gain at book level: about +0.5 points of CAGR and +0.02 Sharpe, with about $5M instead of $2.5M of open-auction
capacity.

Those two records and the sized section of `docs/strategies/book-strategy-caveats.md` were uncommitted in the main checkout
when this was written, so they are not on this branch. Commit them with this work; adding them here would make the merge
refuse to overwrite the untracked copies.

## Frozen rule

```text
ADV63_i,T   = mean of Turnover_i over the 63 rows ending at T, at least 42 observations
typical_T   = median of sqrt(ADV63_i,T) over the pod's own candidates at T; undefined on a day without candidates
reference_T = median of typical over the 252 rows ending at T, at least 60 observations, carried over gaps
m_i,T       = clip(sqrt(ADV63_i,T) / reference_T, 0.5, 2), and 1 where that is undefined
dollars_i   = min(m_i,T * NAV_T / 10, 0.99 * NAV_T - stocks kept at Close_T - dollars already placed that day)
skip        = dollars_i < 0.5% of NAV_T; the slot stays used that day, as an order that never filled
BIL target  = the 1/N pod's rule, fed the true stock dollars
```

```text
[Exact EOD snapshot + account at Close_T]
                 |
[Same signals, ranks, exits, gate as the 1/N pod]
                 |
[m at Close_T: a signal column, every window ends at T]
                 |
[Entry weight = dollars / NAV_T; BIL whole shares frozen at Close_T]
                 |
     *** CRITICAL *** no T+1 price can change a decision
                 |
[Existing VPlan: floor(weight * NAV / reference quote); MOO at next open]
```

Signals, ranks, exit rules, the VIX gate, the BIL parking logic, the next-open route, the reference quote and the
account contract are those of the 1/N pods. The dollars of a new entry differ, and so do some names: larger entries
fill the book sooner. On runs from 2004 to 2026-10-30 sized DV2 has 4,656 entries against 4,974 at 1/N and sized HPI 2,556
against 3,026, and about a third of entry sessions buy a different set of names. Do not describe this as "the same
stocks at other sizes".

Strategy caveats are kept with the strategy, in `strategies/mr_capsule/liquidity_sizing.py`.

## Code changed

| File | Change | Reaches the 1/N, NDX or TAA paths? |
|---|---|---|
| `strategies/mr_capsule/liquidity_sizing.py` | The rule and a mixin on the wired pods | No: only the sized classes import it |
| `strategies/mr_capsule/dv2_vix_gated_liq.py`, `hpi_vote_vix_gated_liq.py`, `strategy_mr_*_liq_bil.py` | Sized pods and entry points | No |
| `alpha/live/mr_capsule_adapter.py` | Two identities added to the tuple; every other new line runs under `liquidity_bool` | The 1/N capsule path runs no new line |
| `alpha/live/backtest_check.py` | Recorded weights are read when the strategy holds them; the compare takes an optional tolerance and basis text | Defaults keep the old contract |
| `alpha/strategy_registry.py` | Two rows | No |
| `strategies/mr_capsule/timing_hook.py`, and the Timing hook function in `dv2_vix_gated.py`, `hpi_vote_vix_gated.py` and both `_liq` modules | Fix for the Bench Timing button (below) | No: only the analysis calls these functions, and live modules do not import the new file |

The runner, scheduler, execution engine, reconcile, `strategy_host`, release loader, data loaders and engine are
unchanged. New modules load only on a sized release's decision: a fresh interpreter importing the live entry points and
the 1/N pods has none of them in `sys.modules`.

## Data contract and what blocks a decision

Turnover is native dollar volume in both data profiles. Capsule snapshot validation does not require the column, so the
adapter checks it. All checks below read data through Close_T only and run on sized identities only.

| Condition at T | Result | Message |
|---|---|---|
| No stock in the frame has Turnover | Blocks | `Liquidity sizing requires Turnover; the frame has none.` |
| No finite positive reference at T | Blocks | `Liquidity sizing has no usable reference at <T>.` |
| No members at T | Blocks | `Liquidity sizing found no members at <T>.` |
| Under 80% of the members in the frame have usable Turnover: 42 finite positive values in the last 63 rows, and one at T | Blocks | `Liquidity sizing needs usable Turnover for 80% of members at <T>; it has <x>%.` |
| Any error while importing the sizing modules | Blocks before any data load | `MR capsule liquidity sizing cannot load: ...` |
| One name without the column | That name cannot be a candidate; no block | |
| One candidate under 42 Turnover rows | m = 1 by rule; no block | |

Real-history coverage since 2004 never falls under 99% of members in either frame, and no member-day holds a zero or an
infinity (`results/research/mr_capsule_liq_build_20261101/coverage.json`). The backtest has no such check, so a day live
blocks would trade there; no real session is affected. Each plan saves its coverage as `liquidity_coverage_float`.

These faults still pass:

- **Faults on up to 20% of members.** Those names trade at m = 1 with nothing marking them beyond the saved coverage.
- **A wrong scale.** Turnover x1000 at T sends most members to m = 2. Entries stay inside the 5% to 20% bounds and the 99% cap.
- **Other errors at run time,** such as a `TypeError` inside a decision, escape the runner's catch, as on the 1/N capsule path.
  In a shared tick that stops the pods after it; with one serve per pod it stops that pod only.

The runner's existing path handles the blocks above as `mr_capsule_decision_blocked`: the message is logged, other pods
continue and the watchdog quotes it. Two existing behaviours matter here:

- **A block stops exits too.** A capsule decision is one unit, as with every capsule data block at 1/N. Sending exits under
  a sizing fault would be a new mode that needs its own tests.
- **Serve sleeps 3,600 seconds after a blocked build.** A data repair just before the open can miss the auction unless serve
  is restarted. This is the shared cadence and is unchanged.

**Not verified: stock Turnover on the same evening.** Every export checked here was made days after the session. If Turnover
for T arrives late, the sized pod blocks, exits included, where the 1/N DV2 pod would send exits and no entries. Check one
same-evening export of each profile before enabling a sized pod, including PAPER.

## Moving between the 1/N and sized forms

`_validate_account_state` treats them as different strategies, and this work leaves that guard as it is.

| Move | Result |
|---|---|
| Same pod id, either direction, holdings or flat | Blocked: `CapsuleHoldingMismatchError` |
| New pod id, fresh database, any holding, BIL included | Blocked |
| New pod id, fresh database, flat account | Works |

So the 1/N pod is **not** a quick fallback. To move an account: sell everything including BIL, then start a new pod id on
a fresh database. Allowing the two forms of one pod to adopt each other's holdings would weaken the identity guard and
is an owner decision; it is not built.

## Rollback

Before reverting the code to a version without these identities, move any YAML that names them out of the releases root,
even a disabled one. Every serve parses the whole root, so a leftover file would stop the NDX and TAA serves too. The
`.yaml.example` templates are not loaded.

## Nightly check

The account can cap an entry at Close_T; m / 10 cannot be capped. So the check compares each entry's size before the cap,
saved with the plan as `liquidity_entry_wanted_weight_dict`, against the weight the reference recorded when it placed the
entry, with a tolerance of 0.1 percentage points. Names and exits still have to match. A skip or a name on one side only
shows as a difference: near the cap a small account gap can cause one. See `docs/live/BACKTEST_CHECK.md`.

## Known limits

- **Only the open route is backtested.** Capacity above about $5M of book assumes moving stock orders to the close, which has no
  backtest. Do not quote the higher figure without that study.
- **Small accounts.** A 5% entry rounds to zero shares at half the share price a 10% entry would. No order goes out, the close
  reports no exception, and BIL stays sized as if the dollars were spent until the next weekly sweep. Fund-scale pods are
  not affected. There is no capsule minimum capital, at 1/N or sized.
- **Near the cap the names can depend on the account.** Unlike 1/N, a small live gap can change a capped entry or a skip.
- **`docs/plans/MR_CAPSULE_WIRING_REVIEW_20261005.md`,** which both templates name as a prerequisite, signed off the six 1/N
  identities. This document is the record for these two.
- **Memory.** The size frame adds one column per stock to the signal frame. Peak memory of one decision on a fresh real snapshot
  is the same in both forms (below), about 4.3 GB. Several serves building at once on one VPS were not tested.

## Bench Timing button

The button failed on every capsule pod, 1/N included, with `gate_open_ser` missing. The analysis computes signals once and
builds a fresh pod per cell, so the state `compute_signals` prepares was never there. Preparing that state alone would have
run and been wrong: the analysis sets `current_bar` to the decision session, where the parking rule reads the execution
session to find the end of the week, so the weekly BIL sweep would never fire and idle cash would sit outside BIL with no
error. `prepare_pod_for_timing_analysis` does both, and the next session is a calendar label only.

The check is that the default cell, next open for entries and exits, equals the Bench run. On all four pods in
`tests/test_strategy_mr_capsule_timing.py`, and on real data for sized DV2: 10,100 of 10,100 fills, final NAV equal to
under a cent (`timing_check_strategy_mr_dv2_vix_gated_liq_bil.json`). Within a day Timing fills exits first and Bench in
placement order, so the rows are compared without the same-day order.

Two limits of the other cells, in the shared analyzer and unchanged here: when entries fill before delayed exits it caps
new entries by a free-slot count that also counts BIL as a position, and it treats a BIL buy as an entry. Read those cells as
diagnostics, and a same-close cell as an assumption that the signal is known before the broker's cutoff.

## Verification

Tier 3, with a Tier 1 strategy surface. Three independent reviews: parity and quant pitfalls
(`results/research/mr_capsule_liq_build_20261101/review_parity_quant.md`), failure modes and coverage
(`review_failure_coverage.md`), and a final read of the fixed code. The first two ran before the last fixes; the third found no
blocker and one thing to fix, the Turnover check, which then counted zeros and infinities as usable.

Results on the frozen live code, saved in `results/research/mr_capsule_liq_build_20261101/`. The Timing fix came after them and
changes only functions the analysis calls:

| Check | Result |
|---|---|
| m against the research pods, frozen sign-off inputs | Gap 0.0 on every cell: DV2 5,733 x 1,214, HPI 5,733 x 1,174 |
| Engine against the research pods, m rounded as research saved it | Trade for trade. DV2 10,056 rows, final NAV 52,149,073.38; HPI 5,973 rows, final NAV 20,893,706.86. A rerun is byte-identical. Plain float64: same rows, NAV gap 0.0003% and 0.00000006% |
| Live decisions against the backtest, real history from 2019 | Frozen code: DV2 31 of 31 sessions, HPI 34 of 34. Slightly earlier code: 103 of 103 (226 entries) and 106 of 106 (179 entries). Same exits, same entries in order, dollars within 1e-6 of NAV, same BIL shares |
| Same, real indicators on a synthetic market, in the test suite | `tests/test_live_mr_capsule_liq_parity.py`, capped sessions included, then the nightly reference path: all "same" |
| Snapshot route, fresh exports of both profiles | Decisions build on the 2026-10-29 and 2026-10-30 data. A pod's first day through the real nightly reference: "same", 7 and 2 entries. DV2 on 10-30 shows a real capped entry and a real skip |
| Peak memory of one decision, fresh process | DV2 4.39 GB in both forms; HPI 4.28 GB sized, 4.29 GB at 1/N |
| NDX and TAA decisions, base against branch | 32 of 32 identical, 4 variants x 8 dates, 0 errors |
| The six 1/N capsule identities, base against branch | 54 decisions and the nightly weights byte-identical (third review, synthetic market) |
| PM readiness, both identities | Capital scale 1.990 and 2.022; benchmark pass; the tool's determinism step skipped, covered by the rerun above. Run before the last change, which only touches the live check |
| Tests | Live, capsule, registry, HPI, DV2, portfolio manager and Bench files: 3,097 passed, 8 skipped, 4 failed. The 4 are in `tests/test_bench.py` and fail the same way on the base commit. After the last changes, including the Timing fix, the capsule, release, nightly and analysis-runner files: 569 passed |
| Deliberate code breaks, in a scratch copy | 65 small breaks of the rule, the pods, the adapter, the nightly check and the Timing fix. All caught; two needed a new test first |
| `git diff --check` | Clean |

Not covered: a same-evening export, any broker or VPS run, PAPER, and fills. These are local code results, not account evidence.

| Live-impact check | Result |
|---|---|
| Order timing | Close_T decision to next-open MOO, unchanged. m and the held value read Close_T only |
| Sizing | Changed on purpose for the two new identities, as value entries with `amount` in dollars. Exits stay `target_value = 0`, BIL stays whole-share targets. 1/N, NDX and TAA unchanged |
| Reference price | Same quote source, unchanged. BIL shares frozen at Close_T; no silent close or open substitution |
| State and config | No schema, pickle or state format change. New metadata keys only on sized plans; two new disabled templates |
| Logs and dashboards | Existing fields stay. Blocks use the existing `mr_capsule_decision_blocked`. The nightly basis text differs for sized pods only |
| Windows and VPS failure modes | Blocks run through the real runner on temporary SQLite, with another pod continuing. Broken-module cases tested. The whole-root parsing risk is in the rollback note. No VPS run |
| Released YAMLs | None created, edited or enabled |
