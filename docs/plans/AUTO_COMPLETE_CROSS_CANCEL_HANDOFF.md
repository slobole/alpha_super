# Auto-complete an opening order that IBKR cancels as crossing a related account

Status: **v4, the simple version.** It is the single current design, approved by the owner on 2026-11-06. There is
**no switch**: the rule is always on (owner decision, 2026-11-06).
Reader: Codex implements; Claude reviews. Background: [incident record](IBKR_CROSS_CANCEL_20261102.md).
Implementation starts with step 0. Nothing here authorizes deployment, enablement, a push or a real-broker action.
The owner deploys. Once deployed, the rule runs for every pod.

Mockup (updated for v4): `workspace/_mockups/CLAUDE_MOCKUPS/auto_complete/index.html`.

- Relative to this file: `../../../_mockups/CLAUDE_MOCKUPS/auto_complete/index.html`.
- Served by the `v4-mockup` entry in `.claude/launch.json` at `http://localhost:8790/CLAUDE_MOCKUPS/auto_complete/index.html`.
- Generator: `_build/make.py`.

**Why v4.** v3 was correct but over-built: 11 ownership states, 16 screen states, V4 writing into pod databases, and
an operator release procedure. v4 keeps the same safety with far fewer parts. Two facts make that possible:

- **IBKR ends the order at 10:00 (GTD).** At 10:01 the account's positions and open orders show the whole outcome,
  whatever happened to the order or to the connection. There is no need to track "unknown" or "not found" states.
- **No overlap in time.** V4 blocks manual sends for the symbol from 09:34 until the attempt is over, and the pod
  reads positions and open orders right before it sends. Two senders can never act on the leg at the same moment,
  so there is no need for a shared ownership table or for V4 to write anywhere.

## 0. First step, before any code

Confirm that the 2026-11-02 cancel is saved with its text and error code in the client's application database:
`vplan_broker_order_event.message_str`, and `error_code_int` in the raw payload. Report the exact text and code.

- There is no local copy of that database. The owner provides one, or runs a read-only query on the client server.
- The broker's audit trail alone does not prove what reached the database.
- If the text is not saved, stop and report it.

## 1. The rule

A **MOO leg** that IBKR cancelled before the open with **"Order would cross related resting order"** always gets
**one market order at 09:35 for what is still missing**, valid until 10:00. Whatever is still missing
after that goes to the operator, with the exact share count. Everything else stays as today.

Why: when accounts count as related, IBKR cancels the later of two opposite orders in one symbol at one auction, within
about 150 ms of the send. After the auction the other side has usually filled. Today a pod on the standard route with
a cancelled leg parks and builds no new plan until a person acts.

**Which pods.** Every pod that trades at the open (MOO), on both reconcile routes, on every server that runs the new
code:

| Route | Pods | Hook |
|---|---|---|
| Standard reconcile | NDX, TAA, BTAL_QQQ, the momentum legs, and the older DV2 and HPI pods | `post_execution_reconcile` (`runner.py:4297`) |
| Daily reconcile | CORE5, DV2-IND, the MR capsule, T1 | the daily completion loop (`daily_reconcile.py:262-310`) |
| Not covered | MOC pods (Month-End Flow Vote); legs that open or increase a short; a pod whose account is shared with another pod | — |

The routes were read from `DAILY_STRATEGY_IMPORT_TUPLE` (`daily_reconcile.py:22`) and the releases'
`execution_policy_str` (`next_open_moo`, `next_month_first_open`, `next_close_moc`).

## 2. Six rules

1. **One attempt, at open + 5 minutes.** It may start only inside [open + 300 s, open + 360 s). Check the boundary
   again right before dispatch. There is no late attempt.
2. **Mark before send.** Before the broker call, write "attempted" for (plan, symbol) in the pod's database, in one
   transaction with a unique key. A marked leg is never sent automatically again, even after a crash or an
   unanswered send.
3. **Fresh check, then size.** Right before the send, read the account's positions and all open orders. Send only if
   all of these hold:
   - the plan is still current;
   - the account belongs to this pod alone, as the Complete… ticket requires; with a shared account the gap cannot be
     computed per pod;
   - there is no open order in that symbol;
   - the gap still points the original way;
   - the leg does not open or increase a short.

   The order:
   - size = `min(cancelled_qty, abs(target - actual))`, in whole shares;
   - one MKT with GTD at open + 30 min (`ibkr_socket_client.py:898-908`);
   - sent through the pod's own adapter and client id;
   - orderRef `<submission_key>:auto-complete:<ASSET>`, stored as a broker-order row of the same plan.

   If a check fails, nothing is sent and the leg goes to the operator with the reason.
4. **No overlap with manual sends.**
   - V4 does not offer Complete… or New order… for that symbol from 09:34 until the attempt is over. A manual order
     sent before 09:34 lands before the pod's check, and the check sees it. After the end, the pod never sends again.
   - The daily exit completion never acts on an affected leg.
5. **10:01 is the end.**
   - The attempt is over when IBKR reports the order filled or ended, or at open + 31 minutes in any case.
   - Then read positions and open orders:
     - nothing missing: done;
     - something missing and no order working in that symbol: it goes to the operator with the exact share count;
     - an order still working in that symbol after 10:01, which GTD should prevent: alert and send nothing.
   - Until the attempt is over, the plan does not park.
   - After a completion exists, judge the leg by it, not by a later status update of the cancelled MOO
     (`runner.py:1378` picks the latest by timestamp today).
6. **Always on, no switch.** The owner decided this on 2026-11-06. The rule runs for every pod from the deployment
   that ships it. The only way to stop a morning's attempt is to stop that pod's serve before 09:35.

## 3. One fix outside the attempt: a parked pod resumes

Today a parked pod on the standard route never re-checks by itself:

- the scheduler skips it (`scheduler_service.py:331-333`);
- the 16:10 EOD snapshot is blocked while its plan is open (`scheduler_service.py:263-270`);
- no new plan is built.

After the operator completes a leg, nothing un-parks the pod, and its next plan is never built.

**Fix.** At the EOD time (close + 10 min), a parked pod runs one position check (`post_execution_reconcile`):

- if positions now match the plan, the plan completes as today and the EOD snapshot follows;
- if not, the pod stays parked.

This applies to every parked pod, not only cross-cancels. It sends no orders.

## 4. Daily-route pods (CORE5, DV2-IND, MR capsule, T1)

- Same rule. The daily close and the next-day catch-up stay as today.
- `claim_daily_completion_request` (`daily_reconcile.py:147`) accepts only sells today, so the "attempted" mark needs
  its own path. Existing readers of `daily_completion_request` must stay correct:
  - `dashboard_v4/daily_data.py:66-72` keeps only negative amounts;
  - `runner.py:4041-4066` and `4184` parse its JSON.

  Storage is the implementer's choice.

## 5. Live Ops (V4)

**Missing trades panel.** It appears as soon as a leg has a saved IBKR cancel or reject. Today it stays hidden until a
post-execution snapshot (`dashboard_v4/missing_trades.py:131`).

**Five row states.**

| State | Missing | State text | Button |
|---|---|---|---|
| Planned | BUY 90 | Cancelled by IBKR · crosses a related account · auto at 09:35 | none |
| Sent | BUY 90 | Sent 09:35:02 · market · order 1184220461 · until 10:00 | none |
| Completed | BUY 90 | Completed 09:35 · 90 @ 22.42 · 9 bps above the open | none |
| Not completed | BUY 100 | Not completed · 100 missing · IBKR cancelled it again · MOC until 15:45 | Complete… |
| Not automatic | BUY 100 | Rejected by IBKR · not enough buying power · MOC until 15:45 | Complete… |

**Reasons.** The reason part changes with the case:

- "Not completed": "IBKR cancelled it again", "40 of 90 filled by 10:00", "not sent: open order in UUP", or
  "auto did not run by 09:36".
  - V4 shows "auto did not run by 09:36" after open + 360 s when no mark exists. The pod's boundary rule guarantees
    that no send can follow.
- "Not automatic": any other IBKR cancel or reject, such as "not enough buying power", or "opens a short".
  - A short has no Complete… button, because the ticket cannot open a short. Daily pods catch it up at the next
    session.

**Times.** All come from the exchange calendar: 09:35 = open + 5 min, 10:00 = open + 30 min, 15:45 = close − 15 min.

**bps from the open.**

- Formula: side × (fill / official open − 1) × 10,000, with side +1 for a buy and −1 for a sell.
- A positive number is always worse: "above the open" for a buy, "below the open" for a sell.
- If the official open is missing, write "open price not available". Never show 0.

**Pill, banner and Overview.**

- **Pill:** "Auto-complete 09:35" (late) while planned, "Completing" (work) while sent, "On track" (ok) when done,
  "Action needed" (act) for the two red states.
- **Banner:** amber "IBKR cancelled 2 of 5 orders … Nothing to do." only while planned.
- **Overview:** planned and sent do not add to the LIVE count.
- **Scheduler problems** show in the existing system and scheduler health line, not in the row.

**Orders tab.** Completion orders show as "BUY 90 · MKT", with source "auto-complete".

**Activity events.** `auto_complete_planned`, `auto_complete_sent`, `auto_complete_done`,
`auto_complete_not_completed`, `parked_pod_rechecked`.

## 6. Hard rules

- Without a cancel carrying that text, every pod, NDX and TAA included, behaves byte-identically. The one exception
  is the 16:10 re-check of a parked pod (section 3). Prove both with the NDX/TAA
  old-versus-new gate.
- No ALTER or migration of existing tables. Production fill-tuple behaviour (`54b417f`) is unchanged.
- No new process and no new client id.
- Never a second automatic send for a marked leg.
- No change to plan building, sizing, MOO timing or the 09:28 deadline.
- MOC legs are untouched.
- V4 writes nothing into pod databases.

## 7. Tests (offline, fake broker)

1. **Happy path.** One MKT inside the minute, with GTD open + 30 min. Nothing is sent before the minute or after it.
2. **Out of scope.** Another text, a reject, a MOC leg, or a short-opening leg: no send.
3. **Fresh check.**
   - A smaller gap gives a smaller order.
   - A zero gap, the opposite direction, an open order or a newer plan gives no send, and "Not completed" with the
     reason.
4. **Mark before send.** A crash after the mark, before and after the send: never a second send, and 10:01 decides.
5. **Lost ACK.** No resend. At 10:01, positions and open orders decide.
6. **End.** Cover four cases:
   - filled;
   - cancelled again at once;
   - partly filled, then ended by GTD;
   - an order still working after 10:01: alert, no send.
7. **Overlap.**
   - V4 blocks manual sends for the symbol from 09:34 to the end.
   - A manual order sent before 09:34 is seen by the pod's check.
   - The daily exit completion skips an affected leg.
8. **Parking.** No park before the end. A plan with one leg done and one not completed parks only after 10:01.
9. **Parked pod at 16:10.** If positions match, the plan completes and EOD runs; if not, it stays parked. A pod
   parked for any other reason behaves the same way.
10. **The rest.**
    - Calendar: an early close and a DST boundary.
    - The readers of the daily completion data stay correct.
    - V4: the five states, the pill, the banner, the Overview, and the bps sign for buys and sells.

## 8. Before LIVE

- Step 0 comes first.
- Run the NDX/TAA old-versus-new gate on real-day replays, the fault simulation, and the CORE5 and capsule real-day
  replays, all on the final commit.
- PAPER cannot reproduce a related-account cancel, so acceptance rests on the fake broker. A controlled 1-share live
  test is the owner's decision.

## 9. Report back; do not decide

1. The stored message and error code from step 0.
2. Where the mark lives, and how existing readers stay correct.
3. Anything that would change an NDX/TAA cycle without that cancel text, apart from the 16:10 re-check.

## 10. Code map (verified 2026-11-06 on main `ba39c4a`)

| What | Where |
|---|---|
| Reconcile timing, parked pods skipped | `alpha/live/scheduler_service.py:38` (`DEFAULT_RECONCILE_GRACE_SECONDS_INT = 300`), `330-345` |
| EOD blocked while a plan is open | `alpha/live/scheduler_service.py:263-270` |
| Standard reconcile | `alpha/live/runner.py:4297` `post_execution_reconcile`; terminal rows `4534`; park event about `4580`; completion `4613` |
| Which pods use the daily route | `alpha/live/daily_reconcile.py:22` `DAILY_STRATEGY_IMPORT_TUPLE` |
| Latest order per asset, picked by timestamp | `alpha/live/runner.py:1378` |
| Terminal statuses, park predicate | `alpha/live/runner.py:261`, `1700-1745` |
| Submit results saved | `alpha/live/runner.py:3614-3667`; `alpha/live/state_store_v2.py:147-185` |
| Broker events from the trade log | `alpha/live/ibkr_socket_client.py:425-455` |
| MKT with GTD | `alpha/live/ibkr_socket_client.py:898-908` |
| Daily claim table, claim, candidates | `alpha/live/daily_reconcile.py:57-62`, `147`, `262-310` |
| Readers of the daily claim JSON | `alpha/live/dashboard_v4/daily_data.py:66-72`; `alpha/live/runner.py:4041-4066`, `4184` |
| Late-execution label | `alpha/live/runner.py:4112-4113` |
| V4 Missing trades | `alpha/live/dashboard_v4/missing_trades.py:26`, `73-96`, `102-131` |
| V4 ticket: no shorts | `alpha/live/dashboard_v4/orders.py:216` |
| MOC cutoff and session window | `alpha/live/manual_order.py:23`, `206-241` |
