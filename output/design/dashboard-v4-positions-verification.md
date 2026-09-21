# Dashboard V4 Positions verification

Date: 2026-09-21. Scope: local LIVE-only V4 Positions page based on Mockup D.

## Delivered and remaining

The page shows saved broker quantities merged by symbol, per-Pod filtering,
local symbol search, and independently dated canonical account totals.
Each holder's quantity is visible inline on multi-Pod rows, including offsetting
legs. The verdict follows the selected Pod, while the Invested card explicitly
remains at Portfolio scope. Missing ownership or evidence is not zero.

This is the first Positions implementation, not the complete valuation page.
Per-position closing value, weight, entry P&L, Best/Worst, Changed today and Off
target remain unavailable. The owner chose to hide these unconnected fields,
tiles, filters and their note, and discuss the Flex source separately. No new
marks or cost-basis source is enabled; Changed today is deferred as well.
The demo uses isolated synthetic SQL; this change was not verified on the VPS.
Its EOD broker cache now matches the saved synthetic EOD observation. A newer
intraday reconciliation still takes precedence.

## Verification

Risk classification: Tier 3 (LIVE operator display). Independent reviews:
pods_finance: parity; scheduler_liveness_reader: failure modes and financial
source safety; pods_history: coverage. All reported findings are resolved.

Findings fixed: separate financial close/source timestamps; observed versus
recorded position times; visible opposing Pod quantities; SQL payload byte caps;
server-side expiry after slow reads; inactive-account links; unknown-count labels.
The follow-up review also fixed touch-visible share splits, selected-Pod wording,
demo EOD observations and historical other-Pod release interference. Ambiguous
account-only cache rows remain ineligible, but fully attributed reconciliation
can still be shown. Same-Pod identity conflicts remain unavailable.

- Full V4 and local workspace suite after review fixes: 724 passed.
- LIVE clerk/reconcile/release/runner/scheduler regressions: 142 passed.
- Browser refresh and local search Node suite: 53 passed.
- Focused reader suite: 47 passed; view-model suite: 25 passed; route suite:
  50 passed. These checks overlap the full suite. The original ownership bug
  and UI findings were reproduced with failing tests before their fixes.
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

Verification used local fixtures and the demo. No VPS deployment was performed.
