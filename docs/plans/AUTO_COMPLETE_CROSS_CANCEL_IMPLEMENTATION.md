# Opening cross-cancel completion: implementation and verification

Status: implemented and verified on `codex/auto-complete-cross-cancel`; no unresolved feature findings.
Functional implementation commit: `3478d62`. The verification follow-up contains one template whitespace fix,
an intentionally changed deadline test, and this report. No deployment has occurred.
Base: `7210df3e99c9e3f77bd6a4e8f9705bec7283ad8f`. Scope: the approved v4 handoff plus the explicitly authorized
forward fix for missing cancellation messages. No push, merge, deployment, broker connection or real order.

## Behavior

An original MOO cancelled before its session open with the saved text `Order would cross related resting order`
gets at most one automatic completion opportunity. It uses the pod's existing adapter, client ID and frozen plan.
The rule is always on when this code is deployed; there is no new switch or process.

```text
[Saved matching pre-open MOO cancel]
                 |
        session open + 5 minutes
                 v
[Durable unique (plan, symbol) attempt]
                 |
                 v
[Fresh positions + all-client open orders]
       | invalid                    | valid
       v                            v
[Operator + reason]        [One capped MKT / GTD]
                                    |
                    terminal or open + 31 minutes
                                    v
                       [Fresh complete broker check]
                         | open/read failed   | settled
                         v                    v
                       [Wait]        [Done / exact remainder]
```

- Dispatch is allowed only in `[open + 300s, open + 360s)`, including a check immediately before `placeOrder`.
  If the scheduler misses that minute, there is no late send.
- The unique attempt is committed before broker validation/send. A failed check consumes the opportunity. A crash
  or uncertain send never causes an automatic retry.
- The fresh check requires the current plan, a dedicated account, a complete snapshot, no same-symbol open order,
  the original direction, and no opening/increase of a short. Deployment-wide manifests identify shared accounts.
- Let `g = target_shares - actual_shares`. Require `g * original_direction > 0`. Quantity is
  `floor(min(abs(original_cancelled_quantity), abs(g)))`; the signed order is `quantity * original_direction`.
  Targets, allocations and signals are not recomputed. Whole shares only.
- The MKT expires at `open + 30 minutes` via GTD. A terminal broker result or `open + 31 minutes` triggers a fresh
  complete check. Failed reads or any same-symbol open order keep the attempt blocked, even when holdings match.
- V4 blocks the affected symbol from `open + 4 minutes` while awaiting the attempt, and throughout an unresolved
  attempted send. The actual dispatch guard also catches a ticket opened earlier or delayed during qualification.
- Daily exit completion skips the affected leg. The existing daily close/next-day settlement rules remain in place.
- A standard-route parked pod receives one positions recheck per exchange session at `close + 10 minutes`.
  Matching positions complete its plan and allow EOD. A mismatch stays parked. This recheck sends no orders.

09:35 is the approved operating rule, not an empirically established optimal execution time.
All boundaries come from the exchange calendar, including DST and early closes. Normal-session examples are
09:34 manual blackout, 09:35 attempt, 09:36 dispatch cutoff, 10:00 GTD, 10:01 final check and 16:10 EOD recheck.

## Step 0 and capture fix

The read-only TAMIR inspection found empty messages and error code 0 in all inspected incident events; there were
zero matching messages across all 17 stored events. DBC BUY84 and UUP BUY96 were stored as cancelled. The existing
database could not trigger exact-message recovery. [Exact evidence](AUTO_COMPLETE_CROSS_CANCEL_STEP0.md).

The forward adapter fix listens for the raw IBKR error event and correlates API order ID plus account to the submitted
request. It preserves the original text/code and the permanent broker ID when available. The standard submission
path persists captured partial results even when a subsequent broker read fails. A real ib_async wrapper plus fake
transport test reproduces the completed-Trade log omission and proves persistence into a temporary application DB.
The test's broker error code is 202; it is not asserted to be the historical stored code.

MOO submission now observes callbacks for up to one second after initial ACK, within the existing overall ten-second
wait. This intentionally changes ordinary MOO return latency. It cannot guarantee delivery of arbitrarily late
callbacks after disconnection. No historical database rows were modified or inferred from generic cancellation.

## Storage and dashboard

The attempted mark lives in the lazily created `auto_completion_attempt` table in the pod's SQLite database,
with primary key `(vplan_id_int, asset_str)`. It stores original-order linkage, canonical recovery request JSON,
quantity, broker identity when acknowledged, timestamps, outcome and remainder. Existing tables are not altered.
Existing daily sell-only completion JSON remains unchanged; reporting reads the separate recovery requests.

V4 remains read-only against pod databases. It shows planned, sent, completed, not completed and not automatic
rows, including saved cancellation evidence before the first post-open snapshot. A mark without broker ACK says
`Attempt started · awaiting broker check`, without inventing a broker order ID. The header, banner, Overview,
Orders table and Activity carry the recovery outcome. Existing unrelated warnings remain visible.

Execution cost is display-only: `side * (fill_VWAP / official_open - 1) * 10,000`, with side +1 for BUY and -1 for SELL.
Positive means adverse execution. Missing official open is explicitly unavailable, never zero or a close-price proxy.
This feature deliberately introduces a five-minute delay for an otherwise cancelled trade; no performance advantage
or equality to opening-auction backtests is claimed.

## Review and verification

Classification: **Tier 3**, including sizing and reporting semantics. Independent reviews covered parity,
failure modes, coverage and quantitative pitfalls. Review findings were fixed with regression tests:

- raw cancellation reasons lost from ended Trade logs and interrupted result reads;
- a late manual refusal incorrectly displayed as an unknown sent outcome;
- saved cancellation hidden while a partially submitted plan remained `submitting`;
- an active short-cover recovery losing its manual blackout after observed holdings crossed its target;
- missing backend proof for early terminal outcomes and the actual parked-EOD scheduler transition;
- original fill evidence disappearing when an appended recovery order entered canonical opening-order validation.
- a later sparse original-order refresh stranding an already recorded attempt; final observation now follows the
  durable mark independently, and API-to-permanent-ID changes retain exact request/account/symbol linkage.

Historical checks passed using the final unchanged decision dependencies:

| Check | Result | Evidence |
|---|---|---|
| NDX/TAA old-versus-new | 32/32 identical actual decisions, 0 refusals/errors | `results/review/auto_complete/ndx_taa_compare.log` |
| CORE5 native/host replay | 45/45 sessions, including 2 DBC reversals and 3 month ends; 180 NAV/quote invariance cases | `results/review/auto_complete/core5_native_replay.json` |
| DV2 BIL capsule native/host replay | 24/24 sessions; wrong-date positive control caught | `results/review/wired_signoff_20261015/auto_complete_dv2_bil/replay.json` |
| HPI BIL capsule native/host replay | 30/30 sessions; wrong-date positive control caught | `results/review/wired_signoff_20261015/auto_complete_hpi_bil/replay.json` |
| Browser JavaScript regressions | 277 passed | `results/review/auto_complete/javascript.log` |

The native replays are decision-parity checks with account state seeded from the historical engine. They are not
closed-loop broker simulations or original-vintage historical snapshots. They test zero new strategy variants.
The separate fault suite uses real temporary SQLite databases and a fake broker/clock to test dispatch and recovery.

Actual browser verification used the local synthetic app, never live configuration: planned row and blackout,
unrelated-symbol preview, partial completion ticket (17 target, 7 held, 10 remaining), completed quantity/VWAP/open
cost, original and appended Orders evidence, Overview attention count and Activity evidence expansion all checked.
Automatic refresh retained the open ticket. No Send button was pressed. Screenshots are saved as
`results/review/auto_complete/ui_partial.png`, `ui_completed.png`, and `ui_activity.png`.
The initial server restart caused expected transient HTMX connection errors; the final pages loaded successfully.

Final affected Python suite: **831 passed**, with one ib_async event-loop deprecation warning.
It covers recovery fault simulations, adapter capture, runner/scheduler, daily reconciliation/reporting, manual
tickets, V4 evidence/Activity, and all four compatibility gates pinned to `54b417f`. Evidence:
`results/review/auto_complete/final_affected_green.xml` and `.log`.
The preceding run had two legacy tests expecting permanent parking at 16:10. Their original expectations now
remain asserted at 16:09, with the explicitly approved close+10m recheck asserted separately; no production code
was changed to satisfy those old expectations. Independent rerun of that file: 8 passed.

The broader repository run used a disposable independent clone: **10,513 passed, 8 failed, 76 skipped**, plus
73 passing subtests. It tested an earlier snapshot of the same change; the final focused runs cover subsequent
review fixes. The broad run is not reported as green. Its failures were classified as follows:

- Two scheduler tests expected permanent parking at 16:10: updated to assert the approved EOD recheck while
  retaining the old behavior at 16:09. The final 831-test run passes.
- One deadline test rejected every MKT submission deadline: updated to accept bounded recovery MKT while still
  rejecting LMT deadlines. MOC behavior remains covered. Socket/flow-vote/payload follow-up: **158 passed**
  (`final_deadline_contract.xml` and `.log`).
- Two monthly HTML golden tests detected one extra blank line on 25 Pod pages. A one-character Jinja fix
  restored the original no-recovery HTML across all 37 captured routes. **No golden hashes were updated**.
  Unchanged golden tests plus recovery UI: **31 passed**.
- Three unrelated failures reproduce on clean baseline `7210df3e` and the candidate. The two Bench engine-map
  import tests pass in isolation but fail after full collection because `norgatedata` is already imported.
  The theme test reports the same four missing cluster-color tokens in both revisions. These were not changed.
  See `results/review/auto_complete/bench_theme_baseline_comparison.md` and its four logs.

Skipped tests are missing evidence, not passes. The explicitly run historical gates above supply the relevant
local-data evidence. No second full 30-minute repository run was performed after the bounded fixes; the affected
tests were rerun instead. A redundant expanded live/dashboard run was interrupted and is not counted as a pass.

All 58 recorded CORE5/capsule dependency hashes still match the candidate source; see
`results/review/auto_complete/replay_final_source_validation.json`.

## Live-impact checklist

| Contract | Result |
|---|---|
| Normal order timing | Original MOO/OPG payload, next-open timing and 09:28 cutoff unchanged; bounded callback observation adds up to one second after ACK. Recovery deliberately uses open+5m. |
| Sizing | Existing strategy targets, amount/target and percent/value semantics unchanged. Only the bounded remainder formula above is new. |
| Reference prices | Existing sizing reference retained; actual official open used only for reporting. No close/open substitution. |
| State/schema/config | One new lazy table; no ALTER/migration of old tables, pickle or YAML formats. V4 performs no pod-DB writes. |
| Logs/readers | Existing fields preserved; five new lifecycle event types and separate recovery request reads. |
| Windows/restarts | Temporary-DB tests cover atomic claim, restart/no replay, read-only UI and paths. No new failure observed offline; production runtime was not exercised. No new process or client ID. |
| Released pods | No released YAML changes; existing route/intent contracts and four pinned compatibility gates retained. |

Quant review: no universe/estimation changes, parameter search, new strategy, train/test fit, target leakage or
corporate-action transformation. Historical replays use current-vintage data and do not prove contemporaneous
historical availability. Slippage, liquidity and real IBKR cancellation/fill timing remain live execution risks.

## Handoff questions

1. **Stored incident text/code:** empty text, raw code 0, no matching stored cross reason. Capture repaired forward;
   historical event not backfilled.
2. **Mark/readers:** separate `auto_completion_attempt` table; existing daily sell-completion readers retain their
   contract; linked recovery requests are read separately. Dashboard is read-only.
3. **Changes without a cross cancel:** up to one second of additional MOO callback observation, the approved
   close+10m parked-position recheck, and extra read-only V4 evidence/dispatch validation. No signal/target change.
   Literal runtime byte identity is therefore not claimed; historical intent parity is tested separately.

Deployment remains an owner action. Acceptance here is offline simulation and review; no live or PAPER order was
sent. The exact historical event cannot be recreated from its missing persisted text.
