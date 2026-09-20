# Dashboard V4 Positions verification

Date: 2026-09-21. Scope: local LIVE-only V4 Positions page based on Mockup D.

## Delivered and remaining

The page shows saved broker quantities merged by symbol, per-Pod filtering,
local symbol search, and independently dated canonical account totals.
Offsetting Pod legs remain visible. Missing ownership or evidence is not zero.

This is the first Positions implementation, not the complete valuation page.
Per-position closing value, weight, entry P&L, Best/Worst, Changed today and Off
target remain unavailable. The owner has a pending choice about expanding the
data source to support closing marks and cost basis. No new source is enabled.
The demo uses isolated synthetic SQL; this change was not verified on the VPS.

## Verification

Risk classification: Tier 3 (LIVE operator display). Independent reviews:
pods_finance: parity; scheduler_liveness_reader: failure modes and financial
source safety; pods_history: coverage. All reported findings are resolved.

Findings fixed: separate financial close/source timestamps; observed versus
recorded position times; visible opposing Pod quantities; SQL payload byte caps;
server-side expiry after slow reads; inactive-account links; unknown-count labels.

- Full V4 and local workspace suite: 703 passed.
- LIVE clerk/reconcile/release/runner/scheduler regressions: 142 passed.
- Browser refresh and local search Node suite: 53 passed.
- Final Positions checks: 104 passed with one test-fixture alias error; after
  fixing that fixture, all 44 route tests passed. These checks include the
  late review fixes; counts overlap the full suite.
- Desktop and 390px browser checks passed: no horizontal overflow, actual
  hidden rows, selected-Pod quantities/counts, query/focus retained across polling.
- Scoped diff whitespace check passed.

## Live-impact checklist

- Order timing and next-open semantics: unchanged.
- Sizing, amount/target semantics and allocation: unchanged.
- Execution reference prices: unchanged; never substituted for closing marks.
- State files, SQLite schemas, pickle and config formats: unchanged.
- Existing logging fields: unchanged.
- Windows: read-only SQLite URI, bounded reads, short lock timeout, malformed
  payload/identity/time cases and byte-preservation covered by tests. No VPS
  process restart or production operation was performed.
- Released Pod YAMLs and execution routes: unchanged; release tests passed.

Security boundaries remain unchanged, including POST rejection and CSP
form-action 'none'. Search filters existing browser rows without submitting a form.

No push or deployment was performed for this phase.
