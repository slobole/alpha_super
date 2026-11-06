# Auto-complete cross-cancel: step 0 result

Status: **historical prerequisite failed; owner subsequently authorized the forward capture fix and feature implementation**.
Checked: 2026-11-06 13:25:30 UTC. Host: TAMIR (100.71.78.97).
Scope: owner-authorized SSH read of persisted application state; no broker connection or operational command.

## Verified evidence

Database:
`C:\Users\Administrator\Documents\workspace\alpha_super\alpha\live\state\live\pod_defensive_btal_qqq_01.sqlite3`

SQLite was opened with URI `mode=ro`. The query inspected all DBC/UUP events for VPlan 1, without a date filter,
and separately counted matching messages across the entire broker-event table.

| Item | DBC | UUP |
|---|---|---|
| Account | U28286459 | U28286459 |
| VPlan | 1 | 1 |
| Original order ID | 19381644 | 19381647 |
| Type / quantity | MOO / BUY 84 | MOO / BUY 96 |
| Saved order status | Cancelled | Cancelled |
| Saved filled quantity | 0 | 0 |
| Cancellation event saved | Yes, 2026-11-02 14:23:40.910345 UTC | No Cancelled event in the inspected rows |
| Event message | Empty on every inspected row | Empty on every inspected row |
| Raw event error_code_int | 0 on every inspected row | 0 on every inspected row |

There are 17 rows in `vplan_broker_order_event` and **zero** messages containing
`would cross related resting order` (case-insensitive). Exact returned rows are in
[AUTO_COMPLETE_CROSS_CANCEL_STEP0_EVIDENCE.json](AUTO_COMPLETE_CROSS_CANCEL_STEP0_EVIDENCE.json).

The saved order records prove cancellation; they do not preserve this cancellation's reason.
This result does not contradict the separate IBKR audit trail.

## Consequence for v4

The exact-message trigger specified by v4 would not identify either original incident leg from this database.
An offline test seeded directly with the desired text could pass while missing this ingestion failure.

Section 0 of [the handoff](AUTO_COMPLETE_CROSS_CANCEL_HANDOFF.md) explicitly says:
> If the text is not saved, stop and report it.

Feature implementation initially stopped before code changes. The owner then explicitly instructed Codex to resolve
the issue and continue. The inspection itself does not prove the exact historical loss point. Subsequent offline
tests reproduced a concrete loss path: ib_async can emit the cancellation through errorEvent after the Trade has
already ended, without retaining the reason in its Trade log. An interrupted follow-up read can also discard the
submission result. The forward fix captures and persists the correlated raw error through both paths.

## Subsequently authorized scope

Include a focused broker-error capture/persistence fix before the existing v4 completion logic:
- reproduce a cancellation carrying the exact broker message and error code through a fake broker;
- make the message survive the real adapter-to-database path, including order-ID/perm-ID mapping;
- retain the exact-message eligibility rule; a generic Cancelled status is not sufficient;
- then implement the already agreed completion and UI, with the full fault, regression and UI checks.

No additional lifecycle states, switch or shared ownership table are proposed here.

## Preparation completed

- Isolated worktree and branch: `codex/auto-complete-cross-cancel`, based on the original checkout's HEAD.
- Current v4 handoff copied into the worktree.
- Standard/daily reconciliation, persistence, actual-dispatch deadlines and parked-pod EOD hooks mapped.
- Five UI states, header/banner, Overview counts, Orders and Activity consumers mapped.
- Existing synthetic broker fixtures and compatibility gates identified.

The statements in Preparation describe the initial stop point. Implementation and verification then continued in
the isolated branch; see [the implementation report](AUTO_COMPLETE_CROSS_CANCEL_IMPLEMENTATION.md).
No client files, database contents, services or real orders were changed. The historical records were not backfilled.
Production remains on its existing code until the owner separately deploys the change.

