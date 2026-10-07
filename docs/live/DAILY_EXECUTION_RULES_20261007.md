# CORE5 and capsule daily execution rules — 2026-10-07

This is the owner's replacement for the 2026-10-05 recovery design, implemented
locally on `live-wiring-20261005` from baseline `8bce05a`. It covers CORE5 and the
supported MR capsule releases only. NDX/TAA strategy, sizing, submission,
reconciliation and parking behavior remain unchanged. Fill recording by broker
execution ID applies to every pod. No UI, release enablement, deployment, push or
real-broker action is authorized by this document.

## Decision at Close T; execution at the next open

Both daily paths require approved data through the completed XNYS session T and
a trusted same-session EOD snapshot of the correct account. Price/signal inputs
stop at Close T. The existing EOD capture is due at **exchange close + 10 minutes**
(16:10 New York time normally; 13:10 for a 13:00 early close). The next decision
uses actual holdings/cash from that capture. Missing/stale data or account
evidence blocks a new decision. The prior daily cycle must finish before the
next one can form or capture its next EOD state.

For actual signed holdings `q_i,T`, broker cash `C_T`, and observed Close-T prices
`P_i,T`, the sizing basis is:

```text
NAV_T = C_T + sum_i(q_i,T * P_i,T)
```

Cash already includes short-sale proceeds; do not subtract them again. Actual
holdings/cash always come from the account, never simulated backtest positions.

CORE5 replays its approved signal history through T on every ordinary daily
decision, using the backtest's initialization, long-state-change and month-end
rebalance events. Between events it retains the last target weights, including
the DBC short weight fixed at that event's volatility. Missing or revised cached
signal memory generates a warning and is rebuilt; it does not require a manual
resume command. Exact snapshot date, profile and manifest identity remain
required; a current revised snapshot is not proof of the original data vintage.

A separate execution receipt records the last successfully applied event, target
weights and whole-share book. If that receipt matches the replayed event/weights
and actual shares, CORE5 retains the existing shares: price and cash drift do not
cause daily rebalancing. A new event, missing/invalid receipt, missed rebalance or
partial application instead creates catch-up intent from current Close-T NAV:

```text
CORE5 target_i = trunc(NAV_T * replay_weight_i,T / P_i,T)
order_delta_i = target_i - actual_i
```

`trunc` rounds signed shares toward zero. The six assets remain SPY, IEF, GLD,
DBC, UUP and BIL, with the existing DBC reversal legs. The candidate receipt
advances only when the observed account matches the complete frozen share book;
otherwise the prior receipt remains, allowing a later ordinary daily catch-up.
No strategy parameters, source strategy rules, universe or research costs change.

Capsule stock entry dollars and BIL/SPMO share targets use the same `NAV_T`.
Parking ETF shares stay fixed from Close T. Stock entry shares use the existing
pre-submit reference quote: `floor(frozen_entry_dollars / live_reference_price)`.
The stock/ETF distinction is intentional; capsule stock quantities are not all
fixed at Close T. Existing dedicated-account, whole-share and margin requirements
remain. See [capsule prerequisites](../plans/MR_CAPSULE_WIRING_REVIEW_20261005.md)
and [CORE5 funding/borrow qualification](CORE5_WIRING_20261005.md).

## Opening dispatch and intraday completion

```text
[Close T data + actual EOD at close+10m] --> [replay / freeze intent]
             *** CRITICAL: price/signal inputs stop at Close T ***
                            |
             next session, before 09:28 ET
                            v
[preflight] --> [claim --> deadline check before EVERY MOO/OPG order]
     | transient, no attempted send          | partial/error/cutoff
     +--> [retry same frozen batch]          v
                            [persist attempts; keep daily cycle pending]
                                           |
            [intraday: fresh account + all-client orders per asset]
                                           |
              [after exchange close: cancel exact owned orders]
                                           v
        [confirmed cancellation + fresh holdings/cash --> close cycle]
```

The opening deadline is target open minus two minutes (**09:28 New York time**
for the normal 09:30 XNYS open), checked for the batch and immediately before each
socket order. A late opening batch never becomes a market batch. Only known
transient errors before any attempted send retry the frozen batch before cutoff.
An attempted/uncertain send never automatically replays the opening batch.
Attempted and never-dispatched request keys remain in durable dispatch metadata;
missed and blocked cycles stay pending until daily close reconciliation.

CORE5 quotes are non-blocking diagnostics; frozen Close-T prices are explicitly
labeled. **There is no new NAV/cash deviation guard and no quote-based resize.**
Existing account identity, holdings/open-order, validity, funding, borrow and
qualification checks still apply. A permanent capsule funding-preflight failure
drops buys while retaining verified initial sells, including BIL sells. Transient
preflight failures retain the complete batch for retry.

After the target open and strictly before its exchange close, an already-sent
cycle may make **one claimed MKT SELL attempt per decision and asset**:

- A non-BIL asset is eligible only when its frozen target is zero and actual
  positive shares remain. No buy, short cover or nonzero-target top-up is added.
- BIL may sell `max(actual_BIL - frozen_target_BIL, 0)` with a nonnegative target,
  but only if funding did not drop buys. It is never sold alone by same-day
  completion after that buy drop, even though the initial batch retained sells.
- A fresh complete all-client order list must show **no open order for that
  symbol**. An unrelated symbol does not block this asset. Refresh again before
  each asset; quantity is `max(actual_shares - frozen_target_shares, 0)`.
- Persist the claim before sending. Errors remain recorded and retryable at the
  reconciliation level, but the same claimed market sell is not sent again after
  a crash or error. Each market sell has the exchange close as its send deadline.

The exchange calendar supplies early closes and daylight-saving boundaries.
Even a no-order or fully matched daily cycle stays open until its target close.
After close, daily retry priority yields to unrelated due NDX/TAA work, including
EOD capture; a failed daily broker refresh cannot starve that work. Its own next
EOD remains blocked until its cycle finishes.

## After-close settlement and notification

Refresh account-scoped **all-client open orders**, holdings and cash. Ownership
requires an exact locally generated/saved orderRef for this pod/account, including
older-cycle refs through this cycle's target session; a prefix alone is insufficient.
Newer-cycle orders remain untouched by historical-cycle cleanup. Cancel only those owned orders,
reconnect under each positive owning client ID, recheck account/ref/order/perm IDs,
and wait for confirmed absence in another complete refresh. Never globally cancel
orders, bind foreign orders or cancel an unrelated/manual order.

If the broker is unreachable, responses are incomplete, identity is ambiguous,
the owning client ID is busy, or an owned order requires client 0/unbound manual
binding, keep the cycle pending and retry later. Client 0 is deliberately not
used because the installed adapter automatically binds orders on that connection.
A fresh order list is bracketed around holdings/account-summary collection; an
order change during collection causes a retry. Account-summary subscriptions are
cancelled after each request, including failures.

Once our orders are confirmed absent, save fresh actual holdings/cash and close
atomically as `completed` or `completed_with_exceptions`. Compare actual holdings
to frozen intent; broker fills, completed-order history, execution-history windows
and claimed-but-unsent absence proofs do not decide closure. There is no manual
resume or close command in this policy. Diagnostic fill collection can fail
without blocking the mandatory fresh account refresh. Historical abandoned
cycles may finish out of order: if a later same-pod decision exists, the older
cycle records its outcome without overwriting the newer PodState/strategy memory.

For exceptions, the same database transaction enqueues **one decision-keyed
`daily_exception` outbox item**, including asset, absolute quantity, BUY/SELL,
reason and actual/expected holdings. Decision-only cycles with no VPlan can close;
unsized quantity stays unknown and the alert shows its weight/intent. There are
no separate dispatch Discord alerts. The existing watchdog delivers the outbox;
its delivery retry semantics remain at-least-once, not an exactly-once network
guarantee. Normal completed cycles have no exception alert.

## Fill recording and reference diagnostics

`vplan_fill.broker_execution_id_str` identifies each execution within its account.
Repeated delivery is idempotent; different execution IDs preserve two rows even
when order, second, quantity and price match. Migration backfills available raw
IDs and preserves legacy tuple identity for rows without an ID. Conflicting
lineage/economic details, or ambiguous identified/unidentified observations, fail
closed without silently deleting or reallocating records. Migration cannot
recreate historical fills already collapsed; those require broker records.

Fills remain accounting/reporting evidence. An unavailable official open is not
zero slippage. CORE5's explicitly named frozen-close deviation is a diagnostic,
not opening slippage; late completion fills are labeled `late_execution` and do
not masquerade as opening-auction fills.

## Verification boundary

Offline tests cover broker failures/incomplete refresh, exact cross-client
cancellation, subscription cleanup, duplicate executions, legacy migration,
claim/send failures, deadlines, per-asset completion, restart idempotency,
transaction rollback, decision-only alerts and monthly-pod scheduler fairness.
CORE5 tests cover replay/cache gaps, future-tail invariance, event timing and
preservation of drift/receipt state. The final change report records the focused
and full-suite results for the final code revision.

These tests do not prove real auction fills, borrow availability, live slippage or
current account readiness. Claude must rerun the fault simulation, real-day
CORE5/capsule replays and NDX/TAA old-versus-new gate on the final commit. Earlier
4,801-day CORE5 and 490-day capsule parity results concern the earlier baseline,
not independent verification of this replacement. Enablement remains a separate
owner decision; see [G-035](../../ASSUMPTIONS_AND_GAPS.md).
