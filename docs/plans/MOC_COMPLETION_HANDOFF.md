# Live Ops — "Complete a missing trade" (MOC) — implementation handoff

Date: 2026-10-16. Base: `main` @ `d588d40`. Design: Claude, with the owner. Implementer: Claude (owner, 2026-10-16; was Codex).
Reviewer: Claude (read-only, per commit), as for the other V4 pages.

This is a plan. It grants no permission to deploy, to run broker operations on LIVE, or to change the live
engine. `AGENTS.md` and the verification policy in `docs/ai/PROJECT_GUIDE.md` apply. Read
`docs/plans/DASHBOARD_V4_HANDOFF.md` first. Its guardrails hold, with the single exception written in §4.

## 1. What to build, in one paragraph

When a pod's order did not fill (the broker cancelled it, there was no ACK, or a partial fill), the operator
wants to complete it today **in a few clicks**, usually **at the close (MOC)**. Build:

1. a **Missing trades** panel on the V4 Pod page, listing plan target vs. the saved broker position;
2. a **Complete…** button per row that opens a pre-filled manual ticket as a panel in the page. The ticket re-checks the
   broker (read-only) before anything is sent, defaults to **MOC** before the cutoff, and is sent by one
   button that names the exact order (`Send · BUY 300 APX at close`);
3. **MOC** as a third order type in the existing manual ticket backend;
4. a free **New order…** panel (any symbol, §8b);
5. the manual ticket (New order and Complete) **runnable on V4 production**, behind `--enable-actions`.

Owner decisions (2026-10-16): V4 production · gap-driven pre-fill for all pods · panels in the page (no
pop-out) · **no typed confirmation** (the Send button names the order; this replaces the earlier
"type SYMBOL SHARES" decision).

## 2. Where the design lives

| What | Where |
|---|---|
| Pod page with the panel and the open ticket | `C:\Users\User\Documents\workspace\_mockups\CLAUDE_MOCKUPS\v4d\pod-missing.html` |
| Every ticket state + row states + Overview line (reference sheet) | `…\v4d\complete-states.html` |
| Styles added for this feature (additive to `ops.css`) | `…\v4d\moc.css` |
| Generator | `…\v4d\_build\moc_pages.py` (`python moc_pages.py`) |
| Local preview | `.claude/launch.json` → `v4-mockup` → `/CLAUDE_MOCKUPS/v4d/pod-missing.html` |

All numbers, symbols and accounts in the mockup are synthetic. Wording is the contract; numbers are not.

## 3. What exists today (reuse, do not fork)

| Piece | Where | Note |
|---|---|---|
| Manual ticket backend | `alpha/live/manual_order.py` | one order per ticket; `MKT`/`LMT`, `DAY`; pod's own account; `manual:<pod>:<uuid>` refs; logs `manual_order_submit_requested/completed/failed` |
| IBKR MOC order | `alpha/live/ibkr_socket_client.py:855` | `orderType="MOC", tif="DAY"` already built for `broker_order_type_str == "MOC"` |
| Real submit path (V3) | `alpha/live/dashboard_v3/data.py:147` | `pod_job_gate_obj` lock + `operator_execution_context` + `submit_manual_order_ticket_dict` |
| V4 Tools actions | `alpha/live/dashboard_v4/tools_actions.py` | preview → one-use nonce (120 s) → confirm; **hard-coded to the synthetic demo** (`ToolsActionService.__init__` raises otherwise) |
| Fresh positions + all-client open orders | `BrokerAdapter.get_daily_execution_snapshot` → `alpha/live/daily_broker.py` | read-only; rows carry `asset_str`, `side_str`, `remaining_amount_float`, `order_ref_str` |
| Saved plan / snapshot | `state_store_v2`: `get_latest_vplan_for_pod`, `get_latest_decision_plan_for_pod`, `get_latest_broker_snapshot_for_account` | `VPlan.target_share_map`, `status_str`, `target_execution_timestamp_ts`; `BrokerSnapshot.position_amount_map`, `snapshot_timestamp_ts` |
| Session times | `scheduler_utils.get_session_close_timestamp_ts` (handles half days) | |
| Activity sentences | `alpha/live/dashboard_v4/activity.py:59-61` | manual events already mapped |

## 4. Guardrails

1. **No change to strategy, sizing, order-generation, reconcile or scheduler code**, for any pod. In
   particular nothing under the NDX/TAA paths. This feature touches only `manual_order.py` (additive),
   the V4 dashboard package, and the V4 CLI flag wiring.
2. **The one exception to V4 guardrail #4 ("no new executable actions")**: the manual ticket becomes
   runnable on V4 production. Every other ACTIVE tool stays copy-only. Read-only stays the default; the
   ticket is runnable only with `--enable-actions`, exactly as in V3.
3. One order per ticket. No batch, no "complete all" in v1.
4. Unknown is not green. Any failed read (saved evidence, fresh broker read, calendar) shows no pre-fill
   and no Send button, never a guessed number.
5. Never send without a fresh broker read taken when the ticket opens (preview), and never re-use a
   preview (one-use nonce, 120 s, existing `ConfirmationStore`).
6. Generic rules only (no per-strategy branches). The same code serves monthly and daily pods.

## 5. Backend: MOC in the manual ticket (`manual_order.py`)

- `SUPPORTED_MANUAL_ORDER_TYPE_TUPLE = ("MKT", "LMT", "MOC")`; normalize `"MARKET_ON_CLOSE"` → `"MOC"`.
  MOC takes no limit price (`_parse_limit_price_float` returns `None` as for MKT). TIF stays `DAY`.
- **MOC time gate**, checked in `build_manual_order_ticket_obj` (so preview and submit both enforce it):
  - `MOC_CUTOFF_MINUTES_BEFORE_CLOSE_INT = 15` → the cutoff is 15:45 ET on a normal day and 12:45 ET on a
    half day, computed from the release's `session_calendar_id_str` and `get_session_close_timestamp_ts`.
    (NYSE and Nasdaq accept MOC entries until 15:50; the 5 minutes are our buffer for connect + qualify.)
  - Refuse MOC when today is not a session, or `now >= cutoff`. Error text: `"MOC cutoff passed at 15:45 ET."` /
    `"Market closed today."`
  - Pre-open is allowed (the order rests for today's close).
- **MKT/LMT** keep today's behaviour, plus one new refusal for all types: not after today's session close
  (a DAY order entered after 16:00 would rest into the next session; that is not what "complete today" means).
  The New order ticket (§8b) gets the same rule.
- Optional **completion reference** on the ticket: `completion_vplan_id_int`, `completion_target_share_float`,
  `completion_fresh_position_float`. Pass-through only; written into every `manual_order_*` log payload.
  The submission key stays `manual:<pod>:<uuid>`. Do not change it: the daily close logic relies on it to tell
  manual orders apart from the pod's own (`daily_reconcile.load_daily_owned_order_ref_set`).
- `reference_price` stays as today (1.0 for non-limit types); it is not used for MOC sizing.
- The confirmation constant stays in the backend unchanged (`SUBMIT MANUAL ORDER`). The V4 service supplies it
  after a valid one-use preview nonce, exactly as `tools_actions.py:112` already does. V3 keeps working untouched.

## 6. Read-only gap model (new: `alpha/live/dashboard_v4/completion.py`)

A pod gets rows only when **all** hold (otherwise the panel is hidden; when it is hidden for a reason other
than "nothing missing", show one muted line with the reason):

1. Release enabled, mode `live` or `paper`.
2. **Dedicated account**: no other enabled release under the releases root uses the same `account_route_str`.
   (Broker positions are per account. A shared account makes "target − position" wrong.)
   Reason line: "Account shared with another pod: use New order."
3. Latest VPlan for the pod has `status_str` in {`submitted`, `completed`}. `ready`, `blocked` and `expired`
   are excluded: completing an unsent plan would bypass `submit_vplan`.
4. No newer DecisionPlan than the VPlan's own (a newer decision supersedes the old target).
5. The VPlan's target session is **today or the previous session** (session calendar). Older: reason line
   "Plan is from <date> (<n> sessions ago); share counts may have changed (splits). Use New order."
6. The latest saved broker snapshot for the account is **newer than** `target_execution_timestamp_ts`.
   Otherwise: "Waiting for broker positions after the plan's execution time." (This prevents showing every
   order as missing at 09:25 when MOO orders have not executed yet.)

Per asset in `vplan.target_share_map` (skip any target `< 0`, i.e. sentinels and shorts):

```
gap = target_shares - saved_position          # saved_position from the snapshot above, 0 if absent
row if abs(gap) >= 1                          # whole shares only: qty = floor(abs(gap))
side = BUY if gap > 0 else SELL
```

**Simplicity first (owner, 2026-10-16).** The owner explicitly liked the row-state table and the Overview line
below: one status line per row says the whole state. The mockup wording is the target.

Columns: **Symbol · Missing · State · action** (no Plan/Broker columns; the ticket shows those numbers).
Panel meta: `VPlan #<id> · positions <HH:MM>`. No explanatory line under the table.

| State | Rule | Mark | State text | Action |
|---|---|---|---|---|
| Missing | as above | red ✕ | `Missing · <why>`: `broker cancelled`, `no broker ACK`, `partly filled 120 of 300`, else `differs from plan` (from the cycle evidence already on the page) | `Complete…` |
| Sent | a `manual_order_submit_completed` event for this pod with `completion_vplan_id_int` = this VPlan and the same asset, logged after the saved snapshot | blue dot, row tinted | `Sent · MOC · waits for 16:00 close · order <id>` (`Sent · MKT · order <id>`, `Sent · LMT 41.20 · order <id>`) | — |
| Closed | a newer saved snapshot shows position = target; kept for the rest of the session, then dropped | green ✓ | `Closed by manual MOC · 300 @ 41.27` (price only when a matched fill exists) | — |
| No pre-fill | eligibility rule 2 or 5 fails | gray, row muted | `Plan older than 1 session · no pre-fill` / `Account shared with another pod · no pre-fill` | `New order…` (opens §8b) |

**Overview**: when a LIVE/PAPER pod has ≥1 Missing row, add one "needs attention" line:
`[mark] [LIVE] <pod>  <n> trades missing · APX BUY 300, MDLZ SELL 41` → **Review** links to the Pod page
`#missing`. Severity: late (amber), not failed. More than 3 symbols: list 3, then `+<n>`.

## 7. Fresh check at ticket open (preview)

Opening the ticket calls preview. Preview does one **read-only** broker query through the pod's adapter:
`get_daily_execution_snapshot(account_route_str)` (positions + all open orders on the account, any client).
Then:

```
missing = target_shares - fresh_position      # the fresh number wins over the page
```

Hard stops (no Send button, message as in the reference sheet):

| Case | Message |
|---|---|
| Any open order on the symbol (any client, any ref) | `Blocked · APX order already open (BUY 300 MOC · 14:02) · wait or cancel in TWS` |
| `missing` rounds to 0 | `Nothing to complete · broker holds 300 = plan` |
| Market closed / not a session | `Market closed · next open Mon 09:30` |
| Fresh read fails or times out | `Broker check failed · nothing sent` (Unknown, no retry loop) |
| Eligibility (§6) no longer true | `No pre-fill · use New order` |

When `missing` differs from the row, the ticket uses `missing` and the hint reads `Page showed BUY 300 · checked HH:MM:SS`.

The open-order stop also protects the **daily engine's own exit completion**. Today, between the target time
and the close, daily pods send a MKT sell for any held remainder of a full exit, and they skip an asset that has
any open order (`daily_reconcile.py:248-270`). So a resting manual MOC blocks the engine, and an engine order
blocks the ticket. Do not change the engine. Cover both directions with tests (§11).

## 8. The Complete panel

**One panel structure for Complete and New order (§8b)** (owner, 2026-10-16): a normal panel in the page,
**no drawer, no pop-out, no overlay, no typed confirmation**. Complete opens **right under the Missing trades
panel**. Mockup: `pod-missing.html#ticket`. Three stacked rows:

1. **What**: `Symbol APX` and `Side BUY`, both locked (grey); three numbers `Plan 300` · `Broker now 0` ·
   `Missing BUY 300` (red cell).
2. **Order**: `At close | Now | Limit` (MOC · MKT · LMT, all DAY) · `Shares · max 300 (missing)`, pre-filled
   with `missing`, **down only**. Limit adds one price input. One hint line:
   `MOC until 15:45 · 1:42 left · broker checked 14:02:11 · VPlan #412`. From the cutoff to the close
   `At close` is struck through, `Now` is the default, and the hint turns amber: `MOC closed at 15:45 · Now until 16:00`.
   When the fresh read differs from the page, the hint turns amber: `Page showed BUY 300 · broker checked HH:MM:SS`.
3. **Footer**: `Cancel` + a red button that names the exact order: **`Send · BUY 300 APX at close`**
   (`now`, `at 41.20`). The label updates with every change.

**Guard without typing**: the click submits the one-use preview nonce (120 s). The server sends exactly the
previewed ticket (symbol, side, quantity, type, limit, VPlan). Any edit makes a new preview. On expiry the
footer shows `Expired · reopen`.

Not shown (filled automatically): reason = `Completion · VPlan #<id> · <SYM>`; operator = from the dashboard
config / session.

Send → `submit_manual_order_ticket_dict` under the pod job gate (one action per pod at a time, as V3).
Results (one line each, replacing the Send block):

| Result | Line |
|---|---|
| ACK | `Sent · BUY 300 APX at close · order <id>` |
| Exception / timeout | `Outcome unknown · check TWS before resending` |
| Validation refusal | the backend message, nothing sent |

## 8b. New order — any symbol, not from Missing trades (owner, 2026-10-16)

A **New order…** button in the Pod page header (and on `No pre-fill` rows) opens a free ticket as a
**normal panel in the page**, right under the page header, pushing the rest down. **No pop-out, no drawer,
no overlay.** Mockup: `pod-missing.html#neworder`. It replaces the V4 Tools "manual order" form; Tools links to it.

Three stacked rows, top to bottom (owner: the Order section goes below, nothing on the side):

1. **What**: `Symbol` input · `BUY | SELL` switch · three numbers from the fresh read-only broker check:
   `Broker now 18` · `Plan 18` (the pod's latest VPlan target for that symbol, `—` if none) · `Open orders 0`.
2. **Order**: `At close | Now | Limit` (the same clock rules as §8) · `Shares · max 18 (held)` for a SELL ·
   `Reason` (required, short free text). One hint line:
   `MOC until 15:45 · 1:42 left · broker checked HH:MM:SS · leaves the pod off plan`.
3. **Footer**: `Cancel` + a red button that names the exact order: **`Send · SELL 18 NVDA at close`**.

**No typed confirmation on New order** (owner, 2026-10-16). The guard is the button label plus the server
rules: the click submits the one-use preview nonce (120 s), and the server sends exactly the previewed
ticket. Any edit after the preview makes a new preview, and the button label updates with it.

Stops (same one-line style, shown in place of the footer): any open order on the symbol · market closed ·
**SELL more than held** (`Blocked · NVDA holds 18 · this pod does not go short`) · fresh read failed.
There is no gap cap on a BUY. Log payloads carry no completion reference.

## 9. Making the ticket real on V4 production

- Add a production provider for `manual_order` only, which calls the same function as V3
  (`dashboard_v3/data.py:147` logic: job gate + `operator_execution_context` + `submit_manual_order_ticket_dict`).
  Import and reuse it; do not copy it.
- `ToolsActionService`: allow `enabled_bool` outside demo **only** for `action_name == "manual_order"`, and only
  when the app runs with `--enable-actions`. Every other action keeps raising "Tools execution is available only
  in the synthetic demo." Keep the action token header, nonce, release-hash check and journal.
- The preview step for completion also carries the fresh-check result (§7) in the confirmation context, so
  confirm cannot send a different symbol, side, quantity, type or VPlan than the one previewed.
- `tools.py:288` reason text ("Manual order entry is available in Dashboard V3 only") changes to the real state.
- Demo (`--demo --demo-tools`) keeps the synthetic provider and gains the new screens with synthetic gaps.

## 10. After the send

- Activity: `manual_order_submit_completed` with a completion reference →
  "Manual MOC order: BUY 300 APX (completion of VPlan #412)". Without it → "Manual MOC order: BUY 300 APX".
  Failed → existing failed sentence.
- The Pod page row turns **Sent** (from the event), then **Closed** when the next saved snapshot (EOD) shows
  position = target (§6 row states). No new broker reads from page renders: only the ticket's preview reads
  the broker.

## 11. Tests (minimum)

Backend (`tests/…manual_order…`):
- MOC accepted before cutoff; refused at/after 15:45; half day 12:45; refused on a non-session day; pre-open accepted.
- MKT/LMT refused after the session close; unchanged before it.
- MOC has no limit price; MOC request reaches the adapter as `broker_order_type_str == "MOC"`.
- Completion fields appear in all three log payloads; submission key prefix is still `manual:`.

Gap model (`tests/…dashboard_v4…completion…`):
- Each eligibility rule in §6 on and off (shared account, VPlan status, newer decision, plan age, snapshot older
  than execution).
- BUY gap, SELL gap, partial fill, target < 0 skipped, fractional < 1 skipped.
- Row states: Sent from event, Closed from newer snapshot.

Ticket / service:
- Fresh number overrides the page; open order on symbol (own, manual, other client) blocks; nothing missing
  blocks; fresh read failure blocks.
- Shares above missing refused server-side; nonce single use and bound to the previewed ticket;
  changing any field between preview and confirm refused.
- Non-manual actions still refuse outside demo; read-only mode refuses POSTs.
- Daily interaction: a resting manual MOC on an exit asset → `reconcile_daily_cycle` sends no completion for it
  (use the existing daily test fixtures, no engine change).

Full suite stays green (`.venv` python; note the known disk-reading flakiness on this workstation).

## 12. Verify before PAPER and LIVE (report findings, do not change the engine)

1. **Daily close vs MOC fill timing**: find when daily pods first read holdings after the close and whether a
   16:00 auction fill is visible by then. If the close commit can run before the MOC print, report it with
   evidence. That is an owner decision, not a fix in this change.
2. **IBKR client ID**: the ticket connects with the release's `broker_client_id_int`. Confirm what happens if the
   pod's `serve` is connected at that moment (IBKR error 326). If it collides, report it and propose a dedicated
   operator client ID. Do not wire one without approval.
3. **PAPER first**: one real MOC completion on a PAPER pod: open → Send → ACK → fill at close → row Closed after
   EOD. Screenshots of each state. Only then LIVE.
4. Mockup parity: every state of `complete-states.html` reachable in the demo.

## 13. Out of scope for v1

Batch "complete all" · next-day MOO ticket · LOC / limit-on-close · automatic completion without the operator ·
cancelling orders from the dashboard · V3 UI changes · any change to how the strategies size their next plan.

## 14. Decisions already made — do not undo

- MOC is the default, the cutoff is 15 min before the close, and Send is disabled after the close.
- Pre-fill comes from VPlan target shares minus broker positions. The fresh read wins. Shares can only go down.
- No typed confirmation, for both panels: the Send button names the exact order; the server sends only the previewed ticket.
- Both tickets are normal panels in the page (Complete under Missing trades, New order under the page header), never a drawer or pop-out.
- Production V4 runs only the manual ticket; every other tool stays copy-only.
