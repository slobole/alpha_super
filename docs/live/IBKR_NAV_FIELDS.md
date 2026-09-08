# Expanded IBKR NAV fields

Reporting-only contract for the default single-VPS operator workspace. It reads
saved Flex imports; it does not fetch statements, write configuration or operate
the broker. Explicit client registries keep their own existing configuration.

## ibkr-mtm-expanded-v1

`alpha/live/ibkr_nav_profile.py` supplies this versioned profile and the existing
`daily_nav_eod_v1` client-return method in memory. No additional registry or
environment variable is required for the normal local dashboard.

### Source and supported fields

Use daily XML from **Change in NAV / Mark-to-Market**, with all fields selected,
account IDs retained and no Model grouping. The parser accepts only whole-account
USD rows (`model` absent or empty), with exact account/date/query checks. The
expanded attributes are already preserved by the existing performance importer;
there is no new database schema or import format.

Economic components are exactly:

```text
mtm + dividends + withholdingTax + changeInDividendAccruals
    + interest + changeInInterestAccruals + commissions + otherFees
```

The existing capital fields remain separate: `depositsWithdrawals`,
`internalCashTransfers`, `assetTransfers`, `debitCardActivity`, `billPay`.
`linkingAdjustments` is neither capital nor investment profit. This first profile
requires `assetTransfers` to be zero: nonzero assets need a reviewed counterparty
and in-transit accounting contract before support can be extended.

All other 36 expanded monetary fields, including `assetTransfers`, must be
present, finite and exactly zero. Their authoritative list is
`IBKR_MTM_ZERO_ONLY_FIELD_TUPLE`. Their nonzero meaning is not inferred from a
zero-valued sample. In particular, broker/advisor/client fees, FX components,
alternative realized/unrealized totals and corporate-action proceeds cannot
silently enter the calculation. An unrecognized nonzero field also blocks P&L.
Checks are per field: two unsupported values cannot cancel into acceptance.

### Independent reconciliation

For each account/day, before any display rounding:

```text
economic_total = sum(the eight supported economic fields)
capital = sum(the five capital fields)
residual = ending_NAV - starting_NAV - capital - linking_adjustments - economic_total
accept only if abs(residual) <= USD 0.01
accepted_P&L = ending_NAV - starting_NAV - capital - linking_adjustments
```

Missing, empty, NaN or infinite required fields are unknown, never zero. Period
coverage, account ownership and NAV continuity must also pass. A failed dollar
bridge does not erase independently valid NAV or official account TWR.

The client return uses the existing [daily EOD convention](CLIENT_TWR.md):
verified daily portfolio P&L divided by opening portfolio NAV, then geometric
linking. This is calculated portfolio TWR, not official consolidated IBKR TWR
or exact intraday TWR. The method applies even to a one-account portfolio;
strategy returns remain official IBKR returns and can differ on flow days.
Nonzero linking adjustments or an internal-cash imbalance withhold client TWR.
Net-zero cash fields do not prove event-level transfer matching.

### Source review and limits

Reviewed on 2026-09-08 against the owner's expanded MTM XML: 42 daily rows,
two accounts, all 21 August 2026 trading sessions. All rows reconciled within
USD 0.01 using the eight fields above; all remaining monetary components were
explicit zeros. Private account data and the XML are not committed as fixtures.

The disaggregated export was checked for dividend recognition/payment and
interest accrual reversals. Cash dividends and withholding on payment can offset
a negative accrual change; adding only the cash line would double-count income
already accrued. The same check applies to interest. The
[IBKR field reference](https://www.ibkrguides.com/reportingreference/reportguide/changeinnav_fq.htm)
provides field descriptions, not permission to sum every selected total or
subcomponent indiscriminately.

This one-month, zero-capital-flow source sample validates the exercised field
classification and arithmetic, not future nonzero unsupported fields, intraday
flow timing or external charges not recorded by IBKR. Synthetic regressions
separately check deposits, withdrawals, owner payments, matching/unmatched cash
transfers, accruals, fees, missing fields and source revisions. Extending the
supported field set requires source evidence, regression tests and a new profile
version. Profile configuration already enters scope/report hashes.

### History and deployment boundary

Older seven-field exports still support their original NAV/account TWR facts,
but cannot prove P&L. Activating this profile does not backfill old imports.
Mixed old/new periods remain incomplete; selecting a complete period is allowed,
but the dashboard never silently trims the requested dates. Complete account
ownership history is also required; the first saved performance row is not
automatically funding inception. D+1 finalization gates remain unchanged.

Local acceptance does not deploy code or import history on a VPS. A page read
performs neither action. No live-readiness or broker-parity certification is
implied by the source review.

Live-impact checklist: order/next-open timing, sizing/amount/target semantics,
reference prices, released YAML routes, state/pickle/SQLite formats and consumed
logging fields are unchanged. The optional `zero_only_fields` validation is
additive; explicit legacy registries retain their prior policy. No network call,
new file write, Windows lock mechanism or service restart is introduced.

## Local acceptance - 2026-09-08

- 1,203 focused regression tests passed: client reporting, account performance,
  dashboard, investor PDF, runner, scheduler, reconcile and releases. This is
  not a full research-suite run.
- The supplied August XML passed the new profile in memory: 42 rows, 21 sessions,
  complete portfolio bridge/path and unchanged official strategy TWR. No database
  was opened and the source file hash was unchanged.
- Browser checks passed all seven client pages at 1440/768/390 pixels in three
  isolated configurations: complete expanded local data, incomplete legacy local
  data and the explicit-registry demo. The demo's seven advanced routes also
  passed. Desktop/mobile expanded Overview screenshots were visually inspected.
- Expanded normal-launcher tests verify shared dashboard/JSON/PDF values, deposit
  exclusion, loss days, mixed history, missing third-Pod history and revision
  invalidation. Fixture bytes/mtimes remain unchanged and outgoing connections
  are blocked.
- Tier 3 plus tooling/tests: quant, parity/failure-modes and coverage reviewers
  found no remaining blocker. The two coverage suggestions were added as
  regressions. Risk triage and `git diff --check` passed.
- No VPS deployment, real history import, broker operation, commit or push.
