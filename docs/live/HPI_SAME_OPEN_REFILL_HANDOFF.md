# HPI Same-Open Refill (Both HPI Pods): Implementation Handoff

> **Historical handoff, updated 2026-09-30.** The IBS/RSI baseline was
> demoted to `PM_READY`; only HPI 2/3/5 Vote remains `WIRED`. The baseline
> cannot be loaded as a live release or dispatched by the live host. The
> two-pod rollout steps below are superseded; do not use this document as
> current deployment authorization or instructions.

Original 2026-09-28 status: implemented and tested; committed locally on branch
`claude/dazzling-maxwell-837dc4`; **not pushed, not deployed**. Risk tier: 3
(live decision plans of two LIVE/WIRED pods).

Owner decisions:

- **2026-09-28, 2/3/5-vote pod.** Option (a) of item #1 in
  `docs/research/LEAKAGE_HUNT_BOOKS_20260927.md`. This is finding C-3 / E-03 in
  `results/research/leakage_hunt_20260927/mr/MR_FINDINGS.md`. Both files are
  uncommitted in the main checkout.
- **2026-09-28, baseline pod.** The same change was applied to
  `strategy_mr_hpi_sp500_ibs_rsi_exit`.

## 1. Summary

**The problem.** Both HPI backtests sell an exiting stock at the next open and buy
its replacement **in the same opening auction**. The live host sold at the open
but bought the replacement one trading day later.

**The change.** For both HPI pods, the live plan now puts each exit and its
replacement entry into the **same MOO basket**:

- `strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote`
- `strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit`

This is what the backtests assume, and live DV2 already works this way. Nothing
else changes: signal, ranking, slot count (10), sizing, order type and timing are
all the same.

**Evidence (real Norgate data, backtest 2004-01-02 to 2026-09-25):**

| | 2/3/5 vote | Baseline (IBS/RSI exit) |
|---|---|---|
| Sessions where the live host (fed the backtest's own state) reproduced the backtest's exits, ranked entries and weights, 2006-10-02 → 2026-09-25 | 5,026 / 5,026 | 5,026 / 5,026 |
| Same-open refill days (refilled entries) | 1,062 (1,538) | 1,165 (1,746) |
| Refill days on which the old host was wrong | 1,062 / 1,062 | 1,165 / 1,165 |
| Backtest: CAGR / Sharpe / max drawdown | 16.18% / 1.041 / −17.7% | 15.44% / 0.997 / −19.9% |
| Closed-loop backtest under the **new** live rule | identical to the backtest | identical to the backtest |
| Backtest under the **old** live rule (what live was running) | 14.00% / 0.939 / −20.7% | 12.97% / 0.870 / −20.6% |
| Minimum cash, as % of NAV (backtest = new rule) | −2.26% | −5.12% |
| Most names held at once under the new rule | 10 | 10 |

The old rule cost the vote pod −2.18 pp CAGR and −0.10 Sharpe. The leakage hunt
measured −2.62 pp for the vote pod on older data, before commit fb81e86. For the
baseline pod the old rule cost −2.47 pp CAGR and −0.13 Sharpe; the leakage hunt
never measured that pod.

## 2. Where backtest and live differed

| Item | Backtest (`strategies/hpi/stateful_long.py`, `iterate`) | Live host before (`alpha/live/strategy_host.py`, `_run_hpi_strategy_for_live_decision`) | Live DV2 (reference) |
|---|---|---|---|
| Open series given to `iterate()` | Real `Open_(T+1)` row | Empty series | Empty series |
| Exit order placed | In `iterate()`, only if `Open_(T+1)` is finite | After `iterate()`, for every pending exit still held | In `iterate()`, always |
| Slot freed by an exit | Yes, at the same open (`long_slots_int += 1`), if the open is finite | **No.** Slot reused only at the next decision | Yes, always |
| Free slots | `10 − held + exits with a finite open` | `10 − held` | `10 − held + exits` |
| Candidate ranking | Turnover descending, then symbol ascending (`get_opportunity_list`) | Same | NATR descending |
| Candidate still held (including one being exited) | Skipped, uses no slot | Same | Same |
| Entry size | `previous_total_value / 10` as a value order | Same, giving weight 0.10 of pod budget | Same form |
| Exit does not print | Slot kept; exit stays pending and is retried next day | Exit stays pending | Pod can briefly hold 11 names |

The only real difference was the free-slot count. Both HPI pods use the same
backtest class and the same host function. The reviewer checked that the live
construction (entry mode, Turnover ranking, liquidity `none`, 10 slots, cost
defaults) matches each pod's `run_hpi_variant` configuration.

## 3. What changed

`alpha/live/strategy_host.py` holds the only behaviour change:

- `_run_hpi_strategy_for_live_decision` now gives `iterate()` a series holding the
  marker `1.0` (`HPI_LIVE_TRADABLE_OPEN_MARKER_FLOAT`) for every held name,
  instead of an empty series.
  - `iterate()` then places the exits itself and frees their slots.
  - The unchanged strategy code ranks and sizes the replacements.
- The old loop that added exits after `iterate()` is deleted, because it would
  now be dead code.
- The marker is a tradability flag, not a price. `iterate()` reads the open
  series only through `isfinite()`. A test proves the plan does not change when
  the marker value changes.
- New plan metadata key `hpi_exit_slot_reuse_str: same_open_moo_batch` on both
  HPI plans. The operator can see in the DecisionPlan JSON that the new rule is
  running. Nothing in the code reads this key.

Other files:

- `strategies/hpi/stateful_long.py`: comment only. Backtest code is unchanged.
- `ASSUMPTIONS_AND_GAPS.md`: new row G-033, the remaining live/backtest
  difference.
- Tests: see section 8.

Not touched: the DV2, QPI, TAA, NDX and CORE5 paths (QPI still passes an empty
series, which is correct for that family), the execution engine, the runner,
reconcile, schemas, release YAMLs and VPS config.

## 4. Before and after on real dates (2/3/5-vote pod)

Rows are the host's plans for the same backtest state, from the parity test.

| Signal date | Held | Backtest exits | Backtest entries at next open | Old live entries | New live entries |
|---|---|---|---|---|---|
| 2008-09-03 | 10 | CELG, IBM | ALTR, TMO | none | ALTR, TMO |
| 2020-02-20 | 10 | LUMN, MAS, VNO | MS, ROP | none | MS, ROP |
| 2020-02-21 | 9 | DE | AMD, ETR | AMD | AMD, ETR |
| 2025-03-06 | 10 | T | NFLX | none | NFLX |
| 2025-03-10 | 9 | TAP | FTNT, LH | FTNT | FTNT, LH |

The baseline pod shows the same pattern: 21 refill days in the three test
windows, and the old host was wrong on all 21.

## 5. Cash and margin

**Funding source.** Each replacement buy is about 10% of pod NAV per refilled
slot. It executes in the same auction as the sale that funds it, and that sale
settles on T+1. At order-submission time (09:23 ET) the buy is therefore checked
against **buying power**, before the sale has executed.

**Size of the effect.** The backtests already run cash below zero from opening
gaps and slippage: to −2.26% of NAV for the vote pod and **−5.12%** for the
baseline pod. These figures are identical with and without the new rule, so
this is not a new modeled risk. It is, however, now real in both live accounts.

**How live DV2 handles it.** It does exactly the same thing.
`docs/live/LIVE_TRADING_ARCHITECTURE.md` ("`next_open_moo` means basket mode")
says: one opening basket, not sell-first-then-buy, under a **margin-account
operating assumption**. The execution engine lists exits before entries in the
basket. No code checks AvailableFunds or ExcessLiquidity for incremental books;
cash is recorded and monitored but never blocks. IBKR's own pre-trade check is
the only gate.

**Consequence.**

- **Both HPI accounts must be margin (Reg T) accounts** with buying power for the
  refilled slots. That is typically 10–20% of NAV and occasionally more on a
  busy day.
- In a cash account, or without headroom, IBKR would reject the buy. The pod
  would then miss that entry and park for review. The book would not grow.
- The account type cannot be verified from the repo, because release YAMLs exist
  only on the VPS. **This is a pre-enable check for each pod.** It is not a known
  blocker.

No account, margin, broker or scheduler setting was changed.

## 6. Edge cases (tested for both pods in `tests/test_live_hpi_same_open_refill.py`)

| Case | Behaviour |
|---|---|
| Exit rejected or does not print, replacement fills | The pod holds one extra name (about 110% gross). The runner already logs a critical `exit_residual_detected`, reconcile blocks, and the scheduler **parks the pod for manual review**; it does not build the next cycle by itself. The pod state keeps the name in `pending_exit_symbol_list`. The next plan re-sends that exit and adds an entry only if another exit frees a slot. Held names never exceed 10 plus the number of stuck exits. A closed-loop test checks this over four cycles. |
| Several exits and several candidates | Each exit frees one slot. Candidates fill them in turnover order, with symbol as tie-break. |
| Candidate that is also being exited | It is still held at Close_T, so it is skipped and its slot goes to the next candidate. Same as the backtest. |
| Stale pending name no longer held | Frees no slot. |
| Held name with no bar on T (halt) | Features are NaN, so the marker alone never exits it. If it was already pending, it is exited and refilled, because live cannot know whether the next open prints. |
| Name removed from the index | Membership exit plus refill in the same basket. |
| Partial exit fill | Residual shares count as a held name; the exit is re-sent; no new entry. Reconcile flags it, same as before. |
| Partial entry fill | Unchanged: the smaller position counts as a slot and is not topped up (same as backtest). |
| Book below max with an exit | Slots are free slots plus freed slots, capped by the candidates available. |
| Same cycle re-run | Identical plan, `trade_id_int` advanced once, pod state not mutated. The runner's `active_decision_plan_exists` and `duplicate_submission_guard` already block duplicates. |
| Kill switch (`enabled_bool: false`) | Release not selected, so no plan is built. |
| Review-only (`auto_submit_enabled_bool: false`) | Unchanged, family-agnostic runner path: the VPlan is built but not submitted (existing runner tests). |
| Basket order | One submission key, all MOO, the sale listed before the buy. |

## 7. Remaining live/backtest difference (G-033)

- **Missing opens.** The backtest knows after the fact whether `Open_(T+1)`
  printed; live cannot. If a pending exit's open does not print, the backtest
  keeps the slot, while live refills it and briefly holds one extra name. For
  neither pod did this happen between 2004 and 2026, which is why the
  closed-loop NAVs are identical.
  - The "no tradable open" order cancellations in the backtest logs (ACS 2010,
    NVLS 2012, CBE 2012, PBCT 2022, CTRA 2026) are all entry orders, which live
    would send in the same way.
  - JP-200603 (2006) was liquidated by the engine at its last close without
    first being an exit signal.
- **Removed names.** Names removed from the index are liquidated at the last
  close in the backtest, but go through the real corporate action in live. This
  gap already existed (G-014) and is unchanged by this work.

## 8. Tests run (2026-09-28, this workstation)

- **Baseline before any change:** live, HPI and scheduler suites
  (`tests/test_live_*.py tests/test_*hpi*.py tests/test_dashboard_v4_scheduler_*.py`):
  694 passed.
- **After this change, same command:** 728 passed, 8 skipped.
  - 33 new tests in `tests/test_live_hpi_same_open_refill.py`: 16 cases for each
    pod, plus the kill-switch test.
  - One new case in `tests/test_live_strategy_host.py`, whose single-slot
    baseline cases now expect the refill.
  - The 8 skipped tests are the opt-in real-data tests.
- **Mutation check:** with the host passing an empty open series again, 26 of the
  33 new tests fail.
- **Opt-in real-data parity** (`tests/test_live_hpi_same_open_refill_parity.py`,
  both pods): 8 of 8 passed, taking 30 minutes.
  - `ALPHA_RUN_HPI_LIVE_PARITY_BOOL=true` runs three windows (2008-09..12,
    2020-02..05, 2025-03..05) per pod: 217 of 217 sessions match for each pod,
    with 23 refill days (vote) and 21 (baseline), and the old host was wrong on
    all of them.
  - `ALPHA_RUN_HPI_LIVE_PARITY_FULL_HISTORY_BOOL=true` adds the full history,
    about 12 minutes per pod; its numbers are in section 1.
  - The "old host" in these tests is a test-only replica of the deleted code.
- **Closed-loop and old-rule backtests** (scratch scripts, not in the repo): the
  figures in section 1.
- **Parity test limits:**
  - Each day is re-seeded from the backtest state (open loop); the closed-loop
    runs cover path effects.
  - Both sides share one signal frame built on the window's member symbols. The
    live loader and the full-universe readiness gate are covered by the unit
    tests.
  - The live host's XNYS calendar (`exchange_calendars`) currently starts on
    2006-09-28, so the host cannot replay earlier dates.
- **Review agents:** three independent read-only reviews:
  - parity and quant-pitfalls;
  - failure-modes and coverage;
  - a review of the two-pod change.

  Findings fixed:
  - a stricter parity rule for missing-open days;
  - a wrong comment about stuck exits;
  - a tautological test replaced by a closed-loop test;
  - missing edge cases added;
  - the gap-register row;
  - stale documentation after the two-pod change.

  No code defect was found in the two-pod change.

## 9. VPS rollout status

The original two-pod enablement checklist is superseded by the 2026-09-30
IBS/RSI demotion. Only HPI 2/3/5 Vote has a live release route. Any existing
IBS/RSI release YAML must be retired from a releases root before loading that
root, even if the release is disabled. Current Vote deployment requires its
own release and broker checks; this historical handoff does not authorize it.

## 10. Book numbers

The same-open refill change was tested for both variants. The IBS/RSI baseline
is now `PM_READY` and has no live pod route, so this handoff does not establish
current live book numbers or a completed two-pod rollout. The historical
option (b) restatement in the leakage-hunt report (HPI vote sleeve −2.6 pp
CAGR; books −0.26 to −0.52 pp) and the baseline old-rule gap are discussed
above; any current book comparison needs fresh deployment evidence.

## 11. Optional follow-ups (not done)

- A non-blocking pre-submit warning when the basket's buy notional exceeds
  AvailableFunds plus the notional of its exits. This would be shared with DV2.
- A runner-level end-to-end test with a fake broker (exit Rejected, entry Filled).
  The parking path is family-agnostic and already covered by
  `tests/test_live_runner.py`.
