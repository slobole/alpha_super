# Auto-complete an opening order that IBKR cancels as crossing a related account

Status: **v5, minimal.** Approved by the owner on 2026-11-06; it replaces v4.

- **Owner principle:** robust but minimal. This is only an auto-complete.
- **Reader:** Codex adjusts its implementation on `codex/auto-complete-cross-cancel` @ `b895a33` (section 3); Claude reviews.
- **Background:** [incident record](IBKR_CROSS_CANCEL_20261102.md).
- **Always on:** there is no switch.
- **Not authorized:** deployment, a push, or any real-broker action. The owner deploys.

Mockup (updated for v5): `workspace/_mockups/CLAUDE_MOCKUPS/auto_complete/index.html`.

- Relative to this file: `../../../_mockups/CLAUDE_MOCKUPS/auto_complete/index.html`.
- Served by the `v4-mockup` entry in `.claude/launch.json` at `http://localhost:8790/CLAUDE_MOCKUPS/auto_complete/index.html`.
- Generator: `_build/make.py`.

## 1. The rule

1. **Trigger.** A MOO leg that IBKR cancelled before the open with "Order would cross related resting order", read
   from the saved IBKR message. Any other cancel or reject stays manual, as today.
2. **Action.** Re-send the same order as a market order for the day:
   - same symbol, same side;
   - same quantity: the original quantity minus anything it filled, which is normally all of it;
   - through the pod's own adapter and client id;
   - orderRef `<submission_key>:auto-complete:<ASSET>`.
3. **When.** Inside the pod's normal reconcile, at the first check at or after open + 5 min (09:35), if the leg has not
   been re-sent yet.
   - If the pod lagged, the next check sends.
   - Nothing is sent at or after close − 10 min (15:50). Check that boundary again right before `placeOrder`.
4. **Once.** Before the broker call, write the "sent" mark for (plan, symbol), atomically, with a unique key. A
   marked leg is never sent again by any check, even after a crash or an unanswered send.
5. **Then, as usual.**
   - **No park while it works:** the plan does not park while the re-sent order is working or its outcome is still
     unknown.
   - **When the re-send is over:** when IBKR reports the order Filled, Cancelled, Rejected or Inactive. If the send
     got no answer, it is over at the first check at least 2 minutes after the send that shows no working order in
     that symbol.
   - **Normal reconcile:** if positions match, the plan completes. If not, the standard route parks, and the daily
     route records its exception, both as today.
   - **One order per leg:** the daily exit completion never acts on a leg cancelled for a cross.
6. **A parked pod at 16:10.** A parked pod on the standard route runs one position check per session at close +
   10 min. If positions now match the plan, the plan completes and EOD follows. Otherwise it stays parked. No orders
   are sent.
7. **Listening for the message.** Keep Codex's capture fix: the `errorEvent` listener, plus up to 1 second of
   observation after ACK on MOO batches. The owner approved this on 2026-11-06.

**Pods.** Every MOO pod, on both routes:

- standard route: NDX, TAA, BTAL_QQQ, the momentum legs, and the older DV2 and HPI pods;
- daily route: CORE5, DV2-IND, the MR capsule, T1.

MOC pods are out of scope.

**Accepted trade-off.** If someone buys the same symbol by hand that morning, the re-send would double it. The owner
does not do that, so there is no manual blackout.

## 2. Live Ops: easy for the operator

**Principles.**

- The colour says whether you act: amber and blue mean nothing to do, green means done, red means you act.
- A red row always has Complete…, pre-filled with the exact missing quantity.
- One plain line per row, numbers first, no codes. All times come from the exchange calendar.

**Five row states**, in the Missing trades panel. They show as soon as the cancel is saved:

| State | Missing | State text | Button |
|---|---|---|---|
| Waiting to re-send | BUY 90 | Cancelled by IBKR · crosses a related order · re-sends at 09:35 | none |
| Re-sent | BUY 90 | Re-sent 09:35:02 · market · order 1184220461 · waiting for fill | none |
| Filled | BUY 90 | Re-sent and filled · 90 @ 22.42 · +8.9 bps above the open | none |
| Not filled | BUY 100 | Re-sent, not filled · 100 missing · MOC until 15:45 | Complete… |
| Other reason | BUY 100 | Rejected by IBKR · not enough buying power · MOC until 15:45 | Complete… |

**Variants.**

- **Pod lagging:** after 09:35, while the pod has not sent yet, the amber row says "re-sends at the next check".
- **Not re-sent at all:** for example, the pod did not run until 15:50, or a newer plan exists. The row is red with
  "Not re-sent · {reason}" and Complete….
- **Partly filled:** the re-send has fills but ended short. The row reads "Re-sent, partly filled · 10 missing" and
  Activity "Partly filled · 10 missing". Only fills of the re-send itself count, never holdings.
- **bps from the open:** side × (fill / official open − 1) × 10,000, with side +1 for a buy and −1 for a sell.
  Positive is always worse. If the official open is missing, write "open price not available".

**Header and banner.**

- **Waiting to re-send:** amber pill "Re-send 09:35". The banner reads "IBKR cancelled 2 of 5 orders · DBC BUY 90 and
  UUP BUY 100 cross an order in a related account · The pod re-sends them at 09:35 as market orders. Nothing to do."
  With one order: "MSFT BUY 17 crosses an order in a related account · The pod re-sends it at 09:35 as a market order."
- **Re-sent:** blue pill "Completing", no banner.
- **Filled:** green "On track".
- **Not filled:** red "Action needed". The banner reads "Complete UUP BUY 100 by 15:45 · The re-sent order was not
  filled · Use Complete… below. The pod continues by itself after the 16:10 re-check." A partial fill says "was
  partly filled".
- **Other reason:** red "Action needed" from the moment the reject is saved, with the same banner ("Some planned
  shares are still missing"). Any red row makes the pill red.
- **Parked pod:** the same banner. Its last sentence already names the 16:10 re-check, so say it once.

**Overview.** Amber and blue are not action items and do not add to the LIVE count. The green "re-sent and filled"
line stays until EOD. Red is a normal action item.

**Orders tab.** The re-sent order shows as "BUY 90 · MKT", with source "re-send".

**Activity.** Three plain lines per leg:

1. "IBKR cancelled DBC BUY 90 (cross)";
2. "Re-sent DBC BUY 90 at 09:35:02";
3. "Filled 90 @ 22.42" or "Not filled · 90 missing".

## 3. Changes to the current implementation (`b895a33`)

**Remove:**

1. **The one-minute window** [open + 5 min, open + 6 min) and "Auto did not run by 09:36". Send at the first check
   from open + 5 min until close − 10 min.
2. **The extra sizing checks:** fresh-gap sizing, and the dedicated-account, gap-direction and short checks. Use the
   original quantity minus its filled quantity. Shorts are re-sent like any other leg.
3. **The 10:00 deadline:** GTD at open + 30 min and the final check at open + 31 min. Send MKT DAY, and end per
   rule 5.
4. **The manual blackout:**
   - in V4: the blackout itself and `manual_auto_completion_block_reason_str`;
   - in `orders.py`: the recovery checks;
   - in the socket client's dispatch loop: `assert_manual_dispatch_allowed`.

**Keep:**

- the capture fix: the `errorEvent` listener, the 1-second observation after ACK, and partial-result persistence;
- the mark table `auto_completion_attempt`;
- the recovery-first selection of the latest order;
- no park while active;
- the daily exit-completion skip;
- the 16:10 parked re-check;
- reporting and Activity;
- "plan still current" as the only send-time check besides the mark and the time boundary.

**UI.** Change the texts and states to match section 2, and drop the blackout messages.

**Tests.**

- **The new window:** a lagged send at, for example, 10:30 goes out; nothing goes out at or after 15:50; never twice.
- **Sizing and order:** quantity = original minus filled; shorts allowed; MKT DAY.
- **No-answer case:** the 2-minute rule for an unanswered send.
- **Removed:** the blackout and dedicated-account tests.
- **Re-run:** the 22 affected test files, the golden pages and the Flow Vote wiring, plus the NDX/TAA
  old-versus-new gate.

## 4. Hard rules

- **Without a cross cancel, every pod behaves as today,** with two approved exceptions: the 16:10 parked re-check,
  and up to 1 second of MOO observation after ACK.
- **Storage:** no ALTER or migration of existing tables, and production fill-tuple behaviour (`54b417f`) unchanged.
- **Process:** no new process and no new client id.
- **Sending:** never a second automatic send for a marked leg.
- **Untouched:** plan building, sizing, MOO timing, the 09:28 deadline, and MOC legs.
- **V4 is read-only:** it writes nothing into pod databases.

## 5. Before LIVE

- Re-run the tests listed in section 3 on the final commit.
- PAPER cannot reproduce the related-account cancel. A controlled 1-share live test is the owner's decision.
- The owner deploys, ideally before Dec 1.

## 6. Code map (main pointers)

| What | Where |
|---|---|
| Recovery logic | `alpha/live/auto_complete.py` (branch) |
| Standard reconcile hook | `alpha/live/runner.py` `post_execution_reconcile` |
| Daily reconcile hook | `alpha/live/daily_reconcile.py` `reconcile_daily_cycle` |
| Scheduler: reconcile timing, parked pods, 16:10 re-check | `alpha/live/scheduler_service.py` |
| Message capture | `alpha/live/ibkr_socket_client.py` `_capture_cross_cancel_errors` |
| Live Ops rows | `alpha/live/dashboard_v4/auto_complete_view.py`, `missing_trades.py` |
| Manual ticket | `alpha/live/dashboard_v4/orders.py`, `alpha/live/manual_order.py` |
