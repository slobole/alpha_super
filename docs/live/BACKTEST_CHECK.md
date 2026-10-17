# Backtest check (nightly, report only)

**Question it answers:** is each LIVE pod still running the strategy we tested? Not "is the edge alive" — that
needs years of returns and stays in the monthly Scout review.

## What runs

```
uv run python -m alpha.live.backtest_check --performance-db C:/alpha/live_ops/ibkr_performance.sqlite3
```

Once per session day, best at about 08:00 New York time: after the Flex import (07:10) has final returns for the
previous session, and before the open. The run uses the New York date; on a non-session date TAA backtests are
skipped to stay clear of the month-end DTB3 refresh. Scheduling it is an owner decision; nothing here registers
a task. Options: `--pod-id POD` (one pod; the others keep their last result), `--no-reference` (MR cost gate only),
`--refresh-ruler` (build missing or old rare-loss rulers: one long backtest per pod, cached for 35 days).

Output: `backtest_check_latest.json` next to the watchdog report (`ops_report_latest.json`), plus a
`backtest_check_cache/` folder with the backtest runs and the rulers.

## What each pod gets

| Part | How | Shows in V4 |
|---|---|---|
| Decision check | The research backtest runs on the pod's own days (first target session and budget, up to the last finished session) through `run_auto_reference_strategy`. For the latest executed plan's session: full books compare target weights with the backtest's holdings at that close (±5 points, because close weights carry the day's move); entry/exit books compare entries and exits with the backtest's buys and sells. Symbols and weights only — never shares. | Pod page: Backtest tab, red box when it differs; Overview row |
| Same-days curve | Live daily TWR (IBKR Flex) and the backtest's daily return on the same dates; gap = live relative to the backtest. | Pod Value chart: dashed second line + gap; Performance → Vs backtest |
| Rare loss | Live 1 / 3 / 5 session loss below the backtest's 1-in-100 line for that length (ruler from a long backtest that ends before the pod went live). Cause: "strategy" when the backtest lost at least 70% as much on the same days, "ours" when it did not, "unknown" when the backtest has no data for those days. | Amber line on the Pod page (red when "ours"); Overview row only when "ours" |
| MR cost gate | Stock fills of the MR capsule pods vs the official open, per side, worse = positive. BIL/SPMO parking fills and fills without an official open are not counted. Status: collecting (< 30 fills), on track, above, passed, failed, time cap (6 months). | Performance → Vs backtest; Overview row when above / failed / time cap (a pass shows on Performance only) |

## Boundaries

- Read-only toward live state: pod databases opened with SQLite `mode=ro`; the Flex store through
  `client_reporting` (read-only). No `LiveStateStore`, no release sync, no broker, no orders.
- Backtests run one pod at a time. TAA-family backtests are skipped on the first and the last two sessions of a
  month: the TAA loader may rewrite the shared DTB3 cache that the live month-end decision also reads.
- CORE5 has no automatic backtest yet (`core5_accounting_bridge_required`): it shows "Not checked" with that reason.
- The backtest leaves dividends out (live reference contract `price_return_ledger_v1`) while the IBKR TWR includes
  them, so the gap drifts by roughly the dividend yield on income-heavy pods. Read the gap as a trend.
- Path dependence: after a missed fill a slot pod (DV2-IND, MR) holds something the backtest does not, and later
  sessions can show "differs" until the paths meet again. The red box links to the evidence; the Plan vs actual tab
  says why.
- A failed ruler build never discards the night's check; the pod just has no rare-loss line.
- `--pod-id` never rewrites the MR gate (it pools both capsule pods); a full run does.
- A malformed or hand-edited report is dropped whole by the dashboard, never half-rendered.
- Reports only. It never blocks orders and sends no alerts. The red box quotes the stop rule; disabling a pod stays
  the owner's call.
- A report that is not from the latest session (checked from 10:00 ET on session days) is marked stale: alerts and rare-loss lines are hidden, the System row turns amber.
