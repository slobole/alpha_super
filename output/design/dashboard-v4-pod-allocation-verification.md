# Dashboard V4 Pod allocation verification

Date: 2026-09-21. Scope: the existing LIVE Pod detail page's Positions panel,
its read-only saved-report adapter, and the shared donut rendering. This
supersedes the quantities-only design and Decision width restriction documented
in dashboard-v4-pod-layout-verification.md. The portfolio Positions page is not
part of this phase.

Follow-up, same date: the owner requested a working real-data source without
waiting for Flex Open Positions. The market-estimated source below is now
implemented and verified against copied real EOD account data. Flex is optional.

## Delivered

- Pod-colored holdings donut, gray cash, largest holdings clockwise from noon,
  cash last, surface-colored slice gaps, cash percentage in the center, and
  labels only for holdings with at least 15% of NAV.
- Adjacent accessible table retains every symbol and share quantity and adds
  closing Value $ and Weight. Cash has its own row. More than 12 symbols are
  retained without an Other bucket.
- One close date covers the values, weights and donut. Current saved quantities
  are compared only when complete numeric evidence exists. A separate short
  notice identifies holdings that differ from the close.
- Hover and keyboard focus pair a slice with its table row. Keyboard focus
  survives an automatic refresh only for the same Pod selection, close date,
  symbol and element kind; restoration does not override newer user focus.
- Overview and Pod use one server-rendered SVG partial. The Pod panel stacks
  on narrow screens. Decision again uses the full panel width.
- New tags and target comparison lines are withheld: same-cycle and target
  attribution have not been established. The existing return chart is retained.

## Source and accounting contract

No importer changes, new database tables, broker requests, or engine changes.
The new reader optionally parses Open Positions from the raw Flex XML already
saved in flex_import. It uses the canonical NAV import identity, account,
query, checksum and close date; it does not search older reports for values.
Only finalized prior-day closes in ET are eligible.

Supported rows are USD STK summary positions with multiplier 1 and verified
share, mark and value fields. Ambiguous, duplicate, unsupported, malformed or
inconsistent rows fail closed. IBKR percentOfNAV is not used because its
documented denominator is an asset class rather than the whole account.
Field reference: https://www.ibkrguides.com/reportingreference/reportguide/open%20positionsfq.htm

For the same close date:

    abs(sum(position values) + saved broker cash - canonical NAV) <= USD 0.01
    displayed weight = signed position value / canonical NAV

The absolute one-cent comparison uses decimal arithmetic. Cash comes from the
existing dated broker EOD cash observation, never a residual invented to make
the chart reconcile. Execution reference prices and target weights are not
used as closing marks. Geometry normalizes the accepted component total only
to close a possible rounding gap; displayed weights always use canonical NAV.

Negative positions, negative cash or weights above 100% suppress the donut
while preserving verified signed values in the table. Missing or unverified
values retain the quantities-only panel with a concise reason and no empty
value columns. Source loading is read-only, bounded, lock-aware and cached by
source identity; conflicting newer imports invalidate the older source.

Assumption: the existing validated LIVE account-to-Pod mapping owns the account
entirely. No allocation between Pods is inferred. The contract is also recorded
in ASSUMPTIONS_AND_GAPS.md.

## Verification

Risk: Tier 3, with quantitative behavior review for displayed weights and
reconciliation. Root reviewed the complete integration and agent changes.
Independent review roles: pods_finance (parity and integration coverage),
pods_history (coverage and UI behavior), scheduler_liveness_reader (failure
modes and quantitative pitfalls). Reviews covered changes outside each
reviewer's implementation responsibilities. Final failure-mode review: PASS.

Findings fixed:

- Missing, malformed or sanitized current quantities no longer imply a flat
  account or falsely prove that holdings changed.
- A tiny NAV passing the absolute cent tolerance cannot produce a donut with
  a component above 100%.
- Long numeric values wrap safely on mobile.
- Allocation keyboard focus is preserved through refresh without crossing a
  Pod, close date or selection boundary.

Results:

- All Dashboard V4 Python tests plus local-workspace tests: 847 passed.
- LIVE clerk, reconcile, release, runner and scheduler regressions: 142 passed.
- Browser refresh/expiry/selection/interaction Node tests: 78 passed.
- Additional reporting/source-cache focused run: 120 passed; overlaps the
  broad suites. New allocation, source and integration tests are included in
  the 847-case result.
- Cases include NAV match/mismatch, missing section, shorts, changed holdings,
  more than 12 symbols, single holding, cash only, malformed evidence,
  overlapping imports, historical-cycle isolation, escaping, GET-only routes,
  and unchanged CSP.
- Browser: 1440 / 768 / 390 pixels, no horizontal page overflow. Verified
  concentrated four-holding and roughly equal ten-holding demo Pods, cash,
  single-hue slices, table values, labels, hover/focus pairing and keyboard
  navigation. A selected GLD slice and row remained focused/highlighted across
  the 15-second automatic refresh.
- Scoped git diff --check passed.

## Live-impact checklist

- Order timing, next-open execution and scheduler semantics: unchanged.
- Sizing, share amount/target semantics and capital allocation: unchanged.
- Reference-price source: unchanged; no close/open substitution.
- State, pickle, SQLite schemas and configuration formats: unchanged.
- Existing logging and consumed fields: retained; display fields are additive.
- Windows: read-only SQLite access, bounded lock wait/query work/XML size and
  source-aware caching. No production restart or write is introduced. Only the
  local synthetic dashboard demo was restarted for UI verification.
- Released Pod YAMLs and execution routes: unchanged.

## Residual limits

All three local real Flex examples inspected were NAV-only and contained no
Open Positions. The source adapter is therefore validated against synthetic
expanded reports, not a real expanded export or VPS data. A matching real
export must be checked before claiming compatibility for that optional format.
The market-estimated path below supplies values independently of Flex. Demo
financial values remain explicitly synthetic; the demo includes both four- and
ten-holding cases.

This work adds no production deployment, capital action, new route, Paper or
Incubation support, or portfolio Positions-page financial fields.

## Real market-valued holdings follow-up

Implemented in pod_eod_holdings.py, pod_close_prices.py and pod_finance.py:

    saved broker EOD quantities + cash
                 |
                 + same-date unadjusted Norgate closing prices
                 |
                 v
    Estimated values, weights and cash-inclusive donut

    value_i = round(shares_i * raw_close_i, 2)
    total = sum(value_i) + saved_cash
    weight_i = value_i / total

The estimated total is not broker NAV. It never replaces account NAV, P&L or
TWR. A complete official Flex allocation keeps priority; when unavailable, the
estimate can render even with an entirely missing Flex reporting snapshot.
The header visibly says Estimated and shows its own close date. The source and
denominator are described in its title. Cash-only portfolios need no quotes.
Shorts or negative cash retain a signed table without a pie.

Source boundaries:

- One exact owned LIVE broker EOD history row supplies quantities, cash and
  saved broker NAV. It must be after the exchange close plus ten minutes and
  before the current ET date. The row timestamp describes the saved EOD sampling
  run; for legacy NDX/TAA it is not the precise broker response time.
- The bounded read-only reader handles duplicate exact and submillisecond times,
  malformed newest evidence, account conflicts, oversized JSON and SQLite locks.
- Snapshot mode stays inside the configured local root/profile/exact date and
  validates the manifest and declared data files. It reads Unadjusted Close,
  requires observed-price evidence and complete symbol coverage, and never
  substitutes adjusted Close or another date.
- Direct local mode reads the installed Norgate Updater's loopback service with
  NONE adjustment/padding, exact date, USD stock/ETF metadata, no redirects or
  proxies, and bounded time/response sizes. This avoids the installed Python
  package's external version check and cache-file writes. Protocol and field
  semantics were checked against the installed package, actual local responses
  and https://pypi.org/project/norgatedata/.
- Source caches are bounded, copied, aware of assessment time and file identity,
  and revalidate changed artifacts. No broker connection, data sync, deployment,
  database migration or trading-engine change was introduced.

Real evidence:

- Complete public finance-builder calls used .codex_tmp/vps_ndx_live.sqlite3
  and .codex_tmp/vps_taa_live.sqlite3, both copied real accounts with latest EOD
  on 2026-08-07, an empty Flex snapshot, and real local Norgate raw prices.
- NDX: all nine held symbols valued, cash included, donut available. TAA: all
  three held symbols valued, cash included, donut available. Each full valuation
  took under 0.2 seconds in the local probe. Database hashes stayed unchanged.
- Real snapshot mode also valued CSCO on 2026-07-31. Requesting 2026-08-07 from
  that root correctly failed because the exact artifact was absent.
- These are dated copied-account checks, not a current VPS deployment check.

Final verification for this follow-up:

- Full V4 and local-workspace suite: 979 passed in 246 seconds.
- Price-reader focused suite: 55 passed, including one added cache-limit/expiry
  test after broad-suite collection; production code did not change after that
  collection. Other price tests overlap the full suite.
- EOD reader: 61 passed; wrapper and adjacent finance: 65 passed. Both overlap
  the broad suite.
- LIVE clerk/runner/scheduler/reconcile/release: 142 passed.
- Node refresh/interaction suite: 78 passed.
- Updated local demo renders the existing ten-holding card with no page overflow;
  estimated header/date and unchanged security headers are covered by full-page
  and refresh response tests. Prior 1440/768/390 layout checks remain applicable.
- Tier 3. Independent parity: pods_finance; coverage: pods_history; failure and
  quant review: scheduler_liveness_reader. Final reviews PASS. Root independently
  inspected source adapters and executed the complete real-data probe.
- Findings fixed: cache reuse across an earlier assessment, all declared file
  containment/replacement bounds, microsecond EOD uniqueness, and malformed
  current-share comparisons. Scoped whitespace checks passed.

Remaining operational dependency: the VPS needs an exact-date local price
artifact in snapshot mode. Monthly signal schedules can leave those artifacts
older than the latest EOD account row. The dashboard does not silently reuse
stale marks or initiate a data sync; it reports missing prices and keeps saved
quantities. Direct mode requires the existing local Norgate Updater service.
The code is implemented locally; no VPS restart or deployment was performed.

## Missing daily snapshot fix (2026-09-21)

The owner reported Closing prices unavailable on the real VPS. Read-only SSH
inspection verified production at 726f5cc, snapshot mode enabled, and the exact
2026-09-18 client manifest absent. The same VPS's installed Norgate Updater
returned all 13 distinct held symbols for that date in 0.375 seconds. Waiting
for another market close was not the solution: the reader was tied to monthly
trading-artifact creation despite daily broker EOD observations.

The display-only reader now uses exact-date local Norgate prices when the dated
snapshot directory is absent. Valid snapshots still take priority; existing
corrupted or incomplete directories fail closed. The source is Norgate local,
and Estimated remains visible. A snapshot appearing later immediately overrides
the local-source cache. Failures explain whether the saved price artifact or
local prices could not be read. No trading setting or source is changed.

The candidate module was executed in memory in a separate read-only process on
the real VPS. Production files and the running dashboard process were untouched.
Both complete public finance-builder calls changed from unavailable to valued:

| Pod | Close | Holdings | Saved cash | Estimated total | Cash weight |
| --- | --- | ---: | ---: | ---: | ---: |
| NDX | 2026-09-18 | 9 | $1,971.43 | $12,287.16 | 16.0% |
| TAA | 2026-09-18 | 4 | $287.08 | $20,727.95 | 1.4% |

Each returned donut_available_bool=true in under 0.2 seconds. Official money
tiles, account chart, money date and cash display compared equal before/after.
The global trading snapshot flag remained true. This validates current real
inputs and the candidate behavior, not deployment of the running dashboard.

Both real Pod pages were also rendered by an isolated Flask test client on the
VPS: HTTP 200, allocation SVG/table present, Estimated and Norgate local present,
missing-price message absent, unchanged GET-only CSP. No server was started.

Verification: 93 focused source/finance/integration tests passed. The broader
Dashboard V4 plus LIVE runner/scheduler/reconcile/release suite passed all 1,093
tests in 226.50 seconds (the focused cases overlap this run). Scoped whitespace
checks passed. No JavaScript, styles or templates changed in this follow-up.

Risk: Tier 3. Independent coverage and parity: price_fallback_coverage; failure
and quantitative boundary review: price_fallback_failures. Both passed. The
initial source-options review was price_source_review. Snapshot-only wording
found by review was corrected in the accounting assumptions and source contract.

Live-impact checklist: order timing, sizing, execution reference prices,
schemas, release YAMLs and consumed logging are unchanged. Only the existing
bounded loopback read service is used; no broker connection, snapshot sync,
source write or process restart is introduced. Windows path containment and
existing artifact integrity checks remain in force. A machine without local
Norgate still needs a valid dated snapshot; missing data never becomes a stale
price or a partial donut.
