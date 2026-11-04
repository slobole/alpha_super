# Month-End Flow Vote wiring, closing auction - 2026-11-04

The Month-End Flow Vote has a gated closing-auction route and is classified **WIRED**. The release template is
disabled and its account field is unfilled; once enabled it submits automatically, like the other guarded daily
pods. Nothing was merged, deployed or sent to a broker by this work. Branch `flow-vote-moc-20261103`
(worktree `alpha_super_flow_vote`) from `6a12cda`.

WIRED is a code result. A paper run on the dedicated account must show real closing-auction fills, the TLT short
borrow and the margin before any capital. Owner decision 2026-11-03: build the closing-auction (MOC) route.
Research record: [EOM_TRADABLE_20261101.md](../research/EOM_TRADABLE_20261101.md).

## Frozen strategy contract

- Strategy `strategies.taa_beyond_6040.strategy_taa_month_end_flow_vote`, default config only (`pair_top`, `close`).
  Zero variants were searched for the wiring.
- Decision at Close_T from end-of-day data: the strategy's `compute_signals` over observed prices through T, read
  at row T. A row with `rebalance_bool` names the SPY and TLT weights for the next session's close; any other row holds.
- `NAV_T = EOD cash + sum(signed held shares * Close_T)`, CAPITALSPECIAL Close_T.
  `target_i = trunc(NAV_T * w_i / Close_T_i)`, truncation toward zero for the TLT short; `delta_i = target_i - held_i`.
  No later quote or NAV resizes these shares (the backtest sizes from the prior close).
- Orders: MOC in session T+1, both legs in one batch. The backtest fills at the official close plus 2.5 bps.
- Data: profile `norgate_eod_flow_vote`, unpadded observed rows from 2002-07-26 through T (SPY and TLT
  CAPITALSPECIAL; SPY, IEF, TLT and `$SPXTR` TOTALRETURN), so the live frame equals the research frame.

```text
[Close_T: norgate_eod_flow_vote snapshot + strict read of the dedicated account]
                             |
[compute_signals through T, row T: rebalance -> weights for the NEXT close, else hold]
                             |
     *** CRITICAL *** frozen trunc(NAV_T * w / Close_T) shares; no later price resizes them
                             |
[T+1, close - 20 min: VPlan + fresh strict read (holdings, every client's open orders)]
                             |
[before close - 15 min: MOC requests, both legs in one batch]
                             |
[close + 5 min: settle from fresh holdings; the receipt advances only if actual = frozen]
```

## Execution policy `next_close_moc` (new, reserved for this strategy)

- Signal clock `eod_snapshot_ready`, as CORE5.
- Submission: the next session's close minus 20 minutes (15:40 ET; 12:40 on a 13:00 half day).
- Local cutoff: the next session's close minus 15 minutes (15:45 ET), the house's manual MOC rule; the exchanges
  stop accepting at 15:50 (NYSE), 15:55 (Nasdaq, TLT) and 15:59 (NYSE Arca, SPY). Nothing is sent at or after the cutoff.
- Target execution: the next session's close. Settlement and the post-close report wait until close + 5 minutes
  for the auction prints.
- A plan not sent by the cutoff expires; the cycle blocks as CORE5's does and the next decision catches up (G-035).

## Decisions taken here (not owner calls)

- Guarded daily pod like CORE5: dedicated USD margin account (the pair leg shorts TLT), budget fraction 1.0, every
  session evaluated, orders only when the plan changes the frozen book, strict account read before the build and
  before the send, daily settlement after the close.
- Catch-up as CORE5: an unapplied event is retried at the next session's decision (the receipt does not advance),
  so a missed exit trades one session late at the next close rather than waiting for the next scheduled fill.
- Start: a pod that has never decided waits for the next scheduled fill, as the backtest starts flat. Once it has
  decided, an empty receipt means its first fill did not complete, and that event is caught up like any other; a
  partly filled first event is re-sized from NAV and completed, not blocked. Holdings without any decision still
  need a reviewed start.
- Paper or live only: the virtual broker would fill at the next session's Norgate close, which only a later
  snapshot holds, so incubation is refused rather than left to stall.
- Auto-submit on, as CORE5 and the capsules (owner decision 2026-10-11): the send window is 15:40-15:45, a manual
  send would miss it.
- No margin, funding or borrow preview, as CORE5 (owner decision 2026-10-11): IBKR's own checks accept or reject
  each order; the daily close records a rejected leg and the next decision catches up.
- Client id 56; 53-55 stay free for the three momentum legs on the unmerged branch `mom-legs-20261103`.

## Code changed

- New: `data/flow_vote_snapshot_contract.py`, `alpha/live/flow_vote_adapter.py`, `alpha/live/flow_vote_broker.py`
  (strict account read, same rules as CORE5's), the release template.
- Strategy module: a snapshot read path for live; the research path is unchanged.
- Policy branches, each a no-op for every other policy: `scheduler_utils` (submission and target times),
  `guarded_dispatch.dispatch_deadline_ts`, `daily_reconcile.daily_settlement_timestamp_ts`, `daily_reporting`
  (report claim), `scheduler_service` (reconcile due time), `daily_watchdog` (send cutoff and its grace), `execution_engine`
  (MOC order type), `incubation` (paper fills at the close), `ibkr_socket_client` (send deadline on MOC), `doctor`.
- Flow Vote branches beside the CORE5 ones: `runner`, `dispatch_state`, `state_store_v2` (receipt, plan guard),
  `order_clerk`, `strategy_host`, `release_manifest` (reserved policy and profile, one account per pod),
  `reference_compare`; the snapshot store and exporter; V4 dashboard texts for a closing cycle.

## Proofs (2026-11-04)

| Check | Result |
|---|---|
| Decision parity, adapter vs house engine, 2003-01-07 to 2026-10-30 | 2,016 sessions (858 fill eves, 858 after a fill, 300 others): 0 share mismatches, 0 rebalance-flag mismatches. The fill eves are the evidence; the other sessions use a receipt built from the engine's own book and only test whether the next close is a fill. 202 reruns on prices cut at T also match, which confirms the adapter's own cut rather than proving causality (the strategy's causality is checked in its tests and by the quant review) |
| Snapshot identity, real export of 2026-11-03 read by the live reader vs the research loader | 6,108 rows; columns, index, dtypes, attrs, full frame and signals equal |
| NDX/TAA old-vs-new gate, 4 pods x 8 month-ends, base and branch run back to back | 32/32 identical, no errors. A first branch run had one empty `ValueError` on TAA 2026-08-31; that case alone, rerun, matched the base, and the TAA builder fetches DTB3 from FRED into a cache per checkout, so it was a fetch, not the code |
| V4 dashboard pages vs the base | 37 pages byte-identical (26 demo, 11 production format), after the last fixes |
| Live, dashboard and broker suites | 5,712 passed, 8 skipped, 3 failed; the same 3 fail on the clean base (a V3 pod-row text test and the two V4 golden-hash tests, whose pages equal the base's) |
| Every other test file | 4,213 passed after the Bench strategy count was raised to 57 (the same edit as on main); 22 failed and 6 errors, the same 28 on the clean base (study files and saved runs that a worktree lacks) |
| Flow Vote wiring tests | 73 passed; undoing each review fix fails its test, 7 of 7 |
| Decision parity rerun after the fixes | 2,016 sessions, 0 mismatches |

Scripts: `scripts/research/flow_vote_wiring_20261104_parity.py`, `..._snapshot_check.py`; outputs in
`results/research/flow_vote_wiring_20261104/` (local).

## Live-impact checklist

- Order timing: unchanged for every existing policy, including next-open MOO; every new branch keys on
  `next_close_moc`, which the manifest reserves for this strategy. The NDX/TAA gate and the existing suites confirm it.
- Sizing: unchanged for existing pods. The Flow Vote uses CORE5's frozen-share path, selected by strategy import.
- Reference price: unchanged for existing pods. The Flow Vote's VPlan carries the frozen Close_T prices
  (`flow_vote.frozen_close_t`); no close/open substitution. Paper fills `next_close_moc` at the close, as `same_day_moc`.
- State and config: no schema change. New keys only in Flow Vote decisions and state
  (`flow_vote_execution_receipt_dict`, `flow_vote_warning_code_list`); the V4 daily payload gains `execution_policy_str`.
- Logging: every existing field and reason code kept; new `flow_vote_*` reason codes added.
- Windows VPS: no new path, task or process. The scheduler loop wakes for the 15:40 send; the snapshot sync exports
  `norgate_eod_flow_vote` only once an enabled release asks for it, so merging changes nothing on the VPS.
- Released pod YAMLs: none changed; every template parses as before (tests).

## Independent reviews (2026-11-04)

Tier 3. Four read-only agents reviewed the diff against `6a12cda`: parity, failure modes, test coverage and quant
pitfalls. None found a wrong trade on the normal path. A separate check by the quant reviewer (60 fill eves, 25 of them
TLT long-to-short switches) also gave 0 share mismatches.

Fixed here:
- First fill (parity and failure-modes reviews, the one material finding): a new pod's failed first fill was recorded
  as applied at zero shares and never caught up, and a partly filled first event blocked every later decision,
  exits included. Now handled as in "Decisions taken here".
- Incubation would have stalled at its first fill: refused for this pod.
- The state store's guard against building or claiming a plan on a blocked or expired decision covered CORE5, the
  capsules and DV2-IND only: the Flow Vote contract is added.
- The post-close fill report could be read before the auction prints: it now waits for close + 5 minutes too.
- Watchdog and V4: an unsent closing plan read "Ready" until five minutes after the close; both now count the five
  minute grace from the 15:45 cutoff. Alert and dashboard wording no longer says "opening" for a closing batch.
- Two lists that name every host strategy and every template were missing the Flow Vote.
- Coverage: 34 more test cases (73 in `tests/test_live_flow_vote_wiring.py`), including a full fake-account cycle through the runner (the TLT long-to-short split into
  two signed MOC legs, settlement only after close + 5, the committed receipt read back the next session), the final
  account read blocking a changed account, the data path from exporter to decision on a written snapshot, corrupt
  snapshots, the EOD capture, signed NAV with the short truncated toward zero, and the first-fill cases. Undoing
  each fix, one at a time, fails its test (7 of 7).

Left as is, with the reason:
- The execution report has no closing-auction benchmark (quant review, its one major item). It needs the next
  session's official close after the fact, so it belongs in a report job; it is required before live capital (G-037).
- An unreadable receipt re-sizes and trades the standing book with a warning, as CORE5 does.
- A partial later fill, or IBKR positions not yet updated at close + 5 minutes, leaves the receipt unchanged; the
  next decision re-sizes from NAV and trades the few shares of difference.
- NAV_T leaves out accrued dividends and interest, as CORE5's does (about 0.2% smaller books at month end).
- `$SPXTR` is in the snapshot though only the benchmark uses it. The decision is due a full day before its send, so a
  late index print costs nothing unless it is a day late.

## Known limits

- Closing fills are assumed, not measured at size: +5 bps a side takes the Sharpe from 1.26 to 1.06; opening fills
  give 0.98, one full session late 0.76 (G-037).
- The execution report scores fills against the open and against the frozen Close_T; for this pod both show the
  market's move, not our fill. The 12-month fill review trigger cannot fire until a fill-minus-official-close line exists.
- A refused TLT short (borrow, margin) leaves +50% SPY unhedged until the next close's catch-up. An opposite order of
  another pod resting in SPY or TLT between 15:40 and 16:00 could trigger IBKR's related-account cross cancel, as on
  2026-11-02 at the open.
- Black Friday's 13:00 close carries November's switch in 7 of 24 years, often the largest order; the backtest charges
  the usual 2.5 bps there.
- Idle cash (about 76% of days) earns broker interest that the backtest, the parity proof and the S8 shadow do not
  credit: a live-versus-model reading must bridge it (+1.1 pts a year since 2003).
- The automatic reference comparison refuses this pod (`flow_vote_accounting_bridge_required`).
- The old V3 dashboard forecasts the next window only for `next_open_moo` daily pods; it shows no forecast for
  this pod rather than a wrong one. V4 shows the closing schedule.
- The live calendar reaches 20 years back; a replay before late 2006 needs a wider calendar (the parity proof widens it).
- Reason codes `opening_dispatch_incomplete` and `opening_build_parked` keep their names for the closing auction,
  since alerts and dashboards read them.

## Before a paper run (owner)

1. Review and merge the branch. It touches the same shared live files as `mom-legs-20261103`; merge one, then
   rebase the other.
2. A dedicated IBKR paper margin account in USD with short sales allowed.
3. Copy the template to `alpha/live/releases/<pod_id>.yaml`; set `mode: paper`, the account, the port and
   `enabled_bool: true`.
4. Compare each early fill with the official close before any capital decision.

Before live capital: a report of each fill minus Norgate's official close of its session and of each order's share of
the closing cross; then the review rule of G-037 can run (Scout S8 RED, or fills more than 5 bps worse than the
official close over 12 months, send the pod to the owner).

## Rollback

Disable the release or move its YAML out of the releases root before reverting the code. No other pod uses the
profile or the policy.
