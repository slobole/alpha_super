# Dashboard V4 for the daily pods (CORE5, MR capsule)

Owner request 2026-10-10: make the UI correct for the daily pods before and during PAPER. Plan written by Claude
from a read-only audit of `dee5499`. Branch `ui-daily-pods-20261010`. One commit per phase.

## Hard rules
- NDX/TAA (monthly, LIVE) render byte-identically. Phase 0 golden hashes prove it after every phase.
- One gate: a row is daily when `row["strategy_import_str"] in alpha.live.daily_reconcile.DAILY_STRATEGY_IMPORT_TUPLE`
  or it carries `daily_cycle_dict` (attached only when `is_daily_reconcile_release_bool(release)`). Never gate on
  `next_open_moo` (demo pods use it).
- New readers open SQLite `mode=ro`, bounded reads, never render free text (`error_str`, `send_error_str`,
  `last_error_str`, client IDs).
- Without new CLI flags the app builds today's single `LiveDataProvider`.
- Never suggest an order-sending command for a daily red state.

## Daily status vocabulary (first match wins; `close` from the exchange calendar, early closes included)

| # | Label | Tone | Predicate |
|---|---|---|---|
| 1 | Closed | green | decision `completed` |
| 2 | Closed with exceptions (n) | amber | `completed_with_exceptions`, n = exception rows; all `decision_intent_not_dispatched` -> "Not sent · closed with exceptions (n)" |
| 3 | Superseded | gray | `superseded` |
| 4 | Open after close | red | any non-terminal daily decision with now >= close + 1h |
| 5 | Dispatch unconfirmed: check broker, do not resend | red | VPlan `submitting` and now >= target + 5 min; or `opening_dispatch_incomplete` with attempted keys only |
| 6 | Partly sent (a of a+n) | red | `opening_dispatch_incomplete` with attempted and never-dispatched keys |
| 7 | Not sent | red | `opening_dispatch_incomplete` with no attempted keys; or VPlan None/ready and decision planned/vplan_ready at target + 5 min; or VPlan blocked/expired |
| 8 | Retrying send | working | `dispatch_retry_pending` before its deadline |
| 9 | No decision for <date> | red | mirrors the watchdog `decision_incomplete` |
| 10 | Closing | working | not terminal, close <= now < close + 1h |
| 11 | Trading | working | VPlan `submitted`, target <= now < close (+ "completing exit X (MKT)") |
| 12 | Ready · sends 09:28 | on track | VPlan `ready` before submission |
| 13 | Deciding | working | data ready, decision not built yet, before the cutoff |

Exception reasons: `decision_intent_not_dispatched` Not sent; `actual_holding_differs_from_decision` Holding differs
from decision; `actual_holding_differs_from_vplan` Not filled / holding differs from plan; `funding_buys_dropped`
Buy dropped (funding); `bil_buy_withheld_after_funding_buys_dropped` BIL buy withheld (funding);
`bil_sale_withheld_after_funding_buys_dropped` BIL sale withheld (funding).

## Phases
0. Golden hashes of the monthly/demo pages (`tests/test_dashboard_v4_monthly_golden.py`).
1. `daily_data.py` reader + `daily_cycle.py` projection; cycle/overview/pod/next_commands wired behind the gate;
   `daily_demo.py` seeding real writers (paper, `demo_daily_*`).
2. Paper section and multiple roots: repeatable `--source name=,root=,config=,log=,watchdog=,modes=`;
   `MultiSourceDataProvider` with per-source error isolation; Paper panel, rail and header tab; money LIVE-only;
   `--demo-daily`.
3. Exceptions tab and same-day completion orders in evidence (`:daily-completion:` keys).
4. Next-trade forecast after an exceptions day (`next_operation.py`, `dashboard.py:1433` status tuple).
5. System page: daily watchdog report, receipt, `.daily.sqlite3` alert keys, pod outboxes.
6. Activity labels for daily events; pending reconciles quiet; mode filter.
7. BIL/SPMO shown as "Parked" (SPMO never "Cash").
8. Deferred until a daily pod goes LIVE: money for extra-source LIVE pods.

Each phase updates `alpha/live/dashboard_v4/README.md` and passes the README test commands.
