# CORE5 and capsule recovery before enable

This change is local to `live-wiring-20261005` in the wiring checkout, based on
`6198c8d`. It does not authorize enabling releases, broker trading, deployment or
push. NDX/TAA routing, sizing, scheduling and submission paths remain unchanged;
the execution-ID recording correction also applies to their fills.

## Frozen intent and dispatch

CORE5 and capsule MOO batches stop at **09:28 America/New_York**, two minutes
before their target XNYS opening auction. The same deadline is checked immediately
before each socket `placeOrder`. Order type/TIF remain MOO/OPG. A late batch never
becomes a market order. Known transient connectivity/timeout failures before any
attempted send retain the frozen batch for the next scheduler pass until cutoff.
No automatic batch replay follows an attempted `placeOrder`, even if it raised.

```
[Close T: frozen targets] --> [preflight before 09:28 ET]
                                  | transient, no send
                                  +--> [retry same intent before cutoff]
                                  | accepted
                                  v
                       [claim --> deadline --> each send]
                                  | partial/error/deadline
                                  v
                       [record attempts + terminal unsent legs]
                                  v
                    [fresh orders + executions + actual holdings]
                         | uncertain             | terminal
                         v                       v
                  [critical / parked]    [capsule settle residuals]
                                        [CORE5 reviewed resume]
```

CORE5 quotes are diagnostics only. VPlan estimates explicitly identify the frozen
`core5.frozen_close_t` source; an absent opening-price diagnostic does not block
execution reconciliation. No current quote changes frozen quantities.
**Owner decision: no new NAV/cash deviation guard or resizing.** Existing account
validity, holdings, open-order, funding/borrow and qualification checks remain.

Capsule transient pre-submit errors retry. A permanent funding failure suppresses
buys and retains verified exits, including BIL sells. Original request IDs and
target rows remain immutable; suppressed buys receive terminal local proof.
Same-day exit completion is assessed per asset. Unknown account-wide orders,
uncorrelated evidence or unexplained holdings still block unsafe completion.
Completion send errors are recorded and raised, not silently discarded.

## Evidence and persistence

`vplan_fill.broker_execution_id_str` identifies an execution within an account.
Repeated delivery of one execution is idempotent; two execution IDs retain two
rows even when order, second, quantity and price match. Existing raw execution
IDs are backfilled. Legacy rows without IDs retain their tuple identity. A
conflicting historical account/execution identity fails migration without
deleting or reallocating rows. Contradictory order/economic details for one
execution ID, or an identical observation mixing identified and unidentified
fills, fail closed rather than guessing which records to combine. Already
collapsed historical executions cannot be
invented by migration; recovering them requires broker execution records.

A claimed request with no order is resolved only after its **target session
close**, using refreshed, complete account-scoped open orders, completed orders
and executions. Matching orderRef, saved order/ACK identity, or ambiguous
same-asset evidence prevents an absence conclusion. Socket proof that
`placeOrder` was never entered is recorded separately and does not require the
close. Neither proof creates a fake broker order or fill.

IBKR's current-day execution query is not complete historical evidence. Set
`ALPHA_IBKR_TWS_TIMEZONE` to the **verified actual TWS login IANA timezone** for
this evidence query, for example `America/New_York` only if that is the actual
setting. Unknown timezone, a midnight crossing, partial/error responses, stale
account evidence or insufficient history coverage prevent absence closure.
Persist proof after target close while that session remains covered. A previous
day's unproved claim cannot be cleared by today's empty broker query. This is a
documented operator evidence limitation, not permission for manual SQL changes.
See [IBKR execution request documentation](https://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html).

Terminal shortfalls or failed partial dispatch park CORE5 with a durable CRITICAL
outbox alert and retain its prior committed strategy memory. The existing
watchdog delivers these alerts per configured CORE5/capsule scope. Scope/SQLite
errors remain isolated per pod so receipt and heartbeat processing continue.

## Reviewed CORE5 resume

Use the normal runner connection/release/database options for the intended pod.
The first call refreshes evidence and prints the exact proposed decision and
`review_hash_str`; it sends no orders:

```powershell
python -m alpha.live.runner resume_core5 --mode paper --pod-id POD --db-path DB --releases-root RELEASES --json
```

After reviewing the targets, source hash, prior cycles and broker evidence, apply
that exact preview with a nonempty operator and reason:

```powershell
python -m alpha.live.runner resume_core5 --mode paper --pod-id POD --db-path DB --releases-root RELEASES --review-hash HASH --operator NAME --reason "Reviewed missed session" --json
```

The command requires an enabled, validated CORE5 release, trusted broker EOD
state at the latest completed session T, matching fresh current holdings, no open
orders, and terminal or durably never-claimed prior cycles. It fails if the next
open's 09:28 cutoff has passed. After today's missed cutoff, wait for today's
completed data/EOD evidence so the resumed decision targets the following open.
Unexplained manual holdings changes require reconciliation; the command does not
pretend that an old batch explains them.

Strategy memory is rebuilt from the approved snapshot through T using the
backtest's initialization, long-state-change and month-end events. Between those
events the last target weights remain in force, including the last DBC volatility
weight. Inputs after Close T do not enter this calculation. No parameter search,
strategy rule, universe, costs or historical account simulation is introduced.

At T, for actual signed holdings `q_i`, trusted EOD cash `C_T`, observed closes
`P_i,T` and replay-retained weights `w_i,T`:

`NAV_T = C_T + sum_i(q_i * P_i,T)`

`target_i = trunc(NAV_T * w_i,T / P_i,T)`

Execution uses the existing signed whole-share delta/reversal rules at the next
open. Physical positions and cash are never replaced by simulated backtest
holdings. The reviewed apply atomically supersedes old intent, records reason,
operator, hash and evidence, and inserts a new decision revision. Same-T revision
support preserves original decision IDs/policies/history and keeps revision-zero
uniqueness for ordinary decisions. Concurrent state, evidence or claim changes
invalidate the review. Strategy memory commits only after normal verified
completion of the new cycle. Automatic submission remains governed by existing
release settings and scheduler authorization.

## Verification boundary

Offline regression tests cover distinct identical fills, legacy migration,
claim/send failures, retries/cutoffs, terminal evidence, per-asset completion,
review hashes, concurrent supersession, replay memory and future-tail invariance.
They do not establish current borrow, live auction quality, production account
readiness, or deployment approval. Claude's fault simulation, real-day replays
and NDX/TAA old-versus-new gate remain the next independent enable checks.
