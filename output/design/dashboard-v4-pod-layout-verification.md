# Dashboard V4 Pod layout verification

Date: 2026-09-21. Scope: the existing LIVE Pod detail page only.

## Delivered

- Removed the cycle color legend and the full-portfolio Decision note.
- Plan vs actual has separate Symbol, Order, Position and Broker = model columns.
  Order is the saved requested share delta, including zero versus unavailable.
  Position remains the saved broker Before -> After; it is not recomputed from
  orders, fills, targets or current holdings.
- Decision is limited to 560px on desktop and fits the available mobile width.
  Incremental entry/exit and unknown-book notes, and target precision, remain.
- Positions and cash use compact calendar stamps. Holdings show full date,
  seconds and ET; cash shows its verified close date. Screen-reader text retains
  source descriptions. These independent observations are not merged by date.

No per-symbol allocation pie was added. Saved shares and account cash/NAV do
not establish each security's closing value. The owner previously deferred the
Flex marks-source decision; execution reference prices and target weights were
not substituted for actual holdings values. The portfolio Positions page is
unchanged.

## Verification

Risk: Tier 3, because the page is consumed by LIVE operators. Review roles:
pods_finance (parity), scheduler_liveness_reader (failure modes), pods_history
(coverage). Reviewers inspected production changes independently of their own
implementation responsibilities. Root reviewed the finance and test changes.

Review findings fixed: removed a nowrap rule that could overlap long position
quantities on mobile; replaced generic-span aria-labels with screen-reader text.
No remaining material finding.

- V4 and local-workspace suite: 735 cases. Initial run had 733 passes and two
  assertions still expecting the old accessibility markup. Those assertions
  were updated, and both affected tests passed on a focused rerun. Production
  code did not change between the broad run and the rerun.
- LIVE clerk, reconcile, release, runner and scheduler regressions: 142 passed.
- Browser refresh/expiry/selection Node suite: 53 passed.
- Finance-focused suite: 33 passed (overlaps the broad suite).
- Browser: 1440px desktop and 390px mobile, no horizontal overflow. Checked
  separate Order cells, compact Decision, calendar stamps and automatic refresh.
  Mobile position cells allow normal wrapping and overflow-wrap:anywhere.
- Scoped git diff --check passed.

## Live-impact checklist

- Order timing and next-open semantics: unchanged.
- Sizing, share amount/target semantics and allocation math: unchanged.
- Reference-price source: unchanged; no closing-price substitution.
- State files, pickle files, SQLite schemas and config formats: unchanged.
- Existing data/logging fields: retained; timestamp display fields are additive.
- Windows paths, encodings, file locks, idempotency and production restarts:
  unchanged. Only the existing local synthetic demo server was restarted.
- Released Pod YAMLs and intended execution routes: unchanged.

Residual limit: checked with isolated demo/test data, not VPS or broker data.
Actual per-symbol weights still require an approved closing-marks source.
