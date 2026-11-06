# Cross-cancel auto-completion: v5 implementation

This changes `codex/auto-complete-cross-cancel` from `b895a33` to the owner's
[v5 minimal contract](AUTO_COMPLETE_CROSS_CANCEL_HANDOFF.md). It supersedes the
v4 behavior described in [the earlier implementation report](AUTO_COMPLETE_CROSS_CANCEL_IMPLEMENTATION.md).
Scope is local implementation and offline verification. No push, deployment, release
configuration change, live database modification, or real broker order is authorized or performed.

## Behavior

```text
Saved pre-open MOO cross-cancel
    -> first normal reconcile from open + 5m to before close - 10m
    -> atomic mark for (plan, symbol)
    -> one MKT DAY resend through the pod's existing adapter
    -> broker terminal result, or unanswered-send resolution
    -> ordinary position reconciliation
    -> matched: complete; unmatched: existing standard/daily exception path
```

The trigger still requires the saved IBKR cross-cancel message. Other cancellation
reasons and MOC orders do not enter this path. The forward raw-error capture fix,
one-second MOO observation after ACK, and interrupted-submit result persistence remain.

The signed resend quantity is:

`sign(original amount) * max(0, abs(original amount) - original filled quantity)`.

Only executions belonging to the original order contribute. The largest cumulative
filled count or saved execution total is used, without adding both representations.
Local/permanent broker ID aliases are linked through the exact request reference;
blank references never link unrelated orders. Existing fill storage is unchanged.
Current holdings do not resize the resend. Shorts are allowed.

The one-minute dispatch window, fresh-position-gap sizing, dedicated-account check,
gap-direction and short checks, 10:00 GTD expiry, 10:01 final check, and recovery-specific
manual blackout are removed. The plan must still be current. The socket checks the
session close-minus-ten-minute boundary again immediately before `placeOrder`.

The mark is written before the broker call and never replayed, including after a
crash or lost reply. Broker Filled/Cancelled/Rejected/Inactive ends the resend;
existing equivalent terminal statuses remain supported. An unanswered send ends
only after at least two minutes and a complete fresh broker observation with no
working order in that symbol. A failed/incomplete read leaves it active. An
acknowledged working order has no artificial 10:00 expiry.
The mark uses the actual recovery clock. For an unanswered call, the two-minute
timer starts conservatively when the broker call returns, so slow reads or contract
qualification cannot consume the wait before dispatch.

Active attempts prevent parking. Daily exit completion skips crossed legs. Daily
pods read order history only on affected recovery paths to observe original fills
and resend terminal status. Standard pods retain the once-per-session parked
position check at close plus ten minutes (normally 16:10); that check sends no orders.
All session boundaries use the exchange calendar, including DST and early closes.
Once the existing daily close flow has cancelled owned orders and confirmed their
absence, it also ends any unresolved recovery marks. It uses actual holdings for the
remaining gap without inventing a Filled result or adding a history prerequisite to
daily settlement.

## Dashboard

The five states, lagged-next-check wording, never-resent reason, header, banner,
Overview attention count, Orders source `re-send`, and Activity messages follow
section 2. Green filled evidence lasts until the scheduled EOD boundary. Red rows
offer Complete using the saved missing quantity; the existing manual ticket then
refreshes positions and checks the exact current gap before sending.

Complete supports the saved short target, bounded by its missing quantity. New order
retains its existing rules. Normal manual-ticket checks, confirmation expiry and
nonce behavior remain; the recovery blackout and dispatch assertion are gone.
V4 only reads pod databases.

Displayed cost remains `side * (fill VWAP / official open - 1) * 10,000`, with BUY
side +1 and SELL side -1. Positive is adverse. A missing official open is explicitly
unavailable. This measures the move from the open as well as execution friction;
it is not a pure transaction-cost estimate.

## Verification

Classification: **Tier 3**, including order quantity/timing and operator reporting.
Independent review covers parity, failure modes, coverage and quantitative pitfalls.
Artifacts are under `results/review/auto_complete_v5/` (ignored local evidence).

| Check | Evidence |
|---|---|
| Required 22 files plus golden pages and Flow Vote wiring | Initial integrated run: 935 passed; final commit rerun: `final_affected.log` / `final_affected.xml` |
| NDX/TAA against clean `b895a33` | 32/32 identical actual decisions; zero errors on both sides; `ndx_taa_compare.log` |
| Browser JavaScript regressions | 277 passed; `javascript.log` |
| Final focused backend faults | 102 passed, including late sends, cutoff, partial fills, shorts, interrupted replies, sparse refreshes and daily settlement |
| Independent UI/manual-ticket integration | 92 passed; review record in `final_review.md` |
| Independent review | `quant_review.md`, `final_review.md` and parity/failure review artifacts |

The final commit evidence includes the exact test manifest and commit identity in
the artifact directory. The NDX/TAA decision dependencies have no diff from the
baseline; no later recovery/UI correction changed those gate dependencies.

Material review findings fixed with regression tests:

- Blank order references could borrow another order's fills; original-order identity
  is now exact, including local/permanent ID aliases.
- An early sparse refresh could erase original quantity before the eligible resend;
  known quantity now survives through both recovery and ordinary persistence.
- A stale tick timestamp could age an unanswered send too early.
- A Filled resend with a remaining position gap could avoid standard parking; it now
  follows ordinary reconciliation and stays actionable in the UI.
- Successful daily settlement could leave a recovery mark working indefinitely.
- UI waiting quantities, lost-ACK submission times, mixed-state banners and short
  Complete tickets could disagree with the order or the actual remaining gap.

Browser automation failed during navigation/attachment, so there is no new visual
browser-interaction signoff for v5. Rendered pages, golden HTML, synthetic order-panel
previews, and JavaScript tests provide automated UI evidence. No real broker was used.

## Live-impact checklist and limits

- Normal MOO submission, the 09:28 deadline, MOC routing, strategy decisions,
  allocations and initial sizing remain unchanged. Recovery timing and quantity
  semantics change intentionally as described above.
- Reference price sources are unchanged; official-open reporting has no close-price
  fallback. No historical research data, universes, training, or backtests change.
- The existing recovery mark table is retained. No ALTER, migration of old tables,
  config-format change, new process, or new client ID is introduced.
- Original broker events, fills and ACKs remain available to existing consumers;
  recovery logging and appended Orders evidence remain separate from the opening batch.
- Existing SQLite unique-key claims protect restart deduplication. No new filesystem
  lock, Windows path dependency, or production release-file change is introduced.
- The owner accepts that a manual trade in the same symbol can cause the original
  remainder resend to overtrade. The removed gap check and blackout are not restored.
- This is an operational five-minute rule, not a tested optimal execution time.
  Market fills, buying power, short availability and auction/backtest parity remain
  broker-dependent. Offline tests do not prove live fill quality.
- Historical incident rows with empty messages remain unchanged and do not qualify
  retrospectively. The capture fix applies to future events.
