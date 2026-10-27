# Monthly client report

The report at `/monthly-report` uses saved broker evidence. It does not place
orders, move capital, import broker data or deploy anything. Recommended cadence:
once per completed calendar month, after statements reconcile. An unfinished
month is a printable draft. Ordinary one- and two-program examples use two A4
pages; larger books may continue onto additional pages without hiding rows.

## Meaning and scope

- Headline NAV, dollar profit and TWR cover the **programs only**: the strategy
  Pods, the same scope as the Performance page. Program cards cover their own
  Pods. Other accounts such as VAULT are named on page 1 but never counted:
  IBKR translates their shekel cash to USD, so their USD change is mostly the
  exchange rate, not a result (owner decision 2026-10-27).
- "Since start" is the first Pod day. Once every Pod has closed, the months end
  with the last Pod day, and that month is final once statements reach it. A
  Pod missing a required day still blocks finality; it is not silently dropped.
- Values use the approved USD-base statement contract. A statement actually
  denominated in ILS is unsupported. Showing VAULT in shekels would need IBKR's
  cash report by currency in the Flex query, which is not configured.
- `Opening capital + net movements` includes the initial measured value and
  subsequent capital/scope movements. It is not a lifetime deposit total.
- Dollar profit includes recorded costs, income and FX. It is not TWR multiplied
  by contributions. Posted activity is explanatory: never add it to profit again.
  Accrual changes are already in the NAV bridge. Dividends and withholding tax
  are separate; not every tax posting is assumed to tax a dividend.
- Fee wording comes from the client's reviewed `fee_basis`. This feature adds
  no management/performance fee calculation.

## Returns and transfers

The approved [daily TWR contract](CLIENT_TWR.md) remains:

```
For each owned date D:
B_D = sum of starting NAV
E_D = sum of ending NAV
F_D = reviewed capital movements; inflow positive
P_D = E_D - B_D - F_D
r_D = P_D / B_D
TWR = product(1 + r_D) - 1
```

Capital uses the **end-of-day convention**: a daily approximation, not exact
intraday TWR or official consolidated IBKR TWR. Individual account rows retain
official broker account returns. Period returns are geometrically linked, never
averaged using today's allocations. Initial/final partial program months retain
their dates. Missing returns are unavailable, not zero. MWR/XIRR is not included.

A Growth-to-Steady move is a flow for each program but nets out at client level.
A subset's internal-transfer gate can be relaxed only when the **same-date
complete containing client book** independently passes the original ownership,
source, NAV/flow and daily internal-net-zero checks. This includes VAULT. The
proof enters report provenance; actual flows and official account returns remain
unchanged. Same-day net zero is not event-level matching. Cross-day money in
transit remains unsupported and withholds the return.

```
[Saved USD NAV + flows + dated ownership]
                    |
        [Full-client daily reconciliation]  (proves transfers between accounts)
             /                  \
   [Programs summary]   [Verified program subsets]
             \                  /
       [Monthly report + concise data status]
                    |
        [Explicit Save report and print PDF]
                    |
     [Frozen HTML + evidence] --> [Print that same copy]
```

## History, holdings and presentation

A Pod's program covers its full measured history. Closed programs retain their
own returns through closure and zero current allocation; they are not folded
into a surviving program. The Programs panel can change or clear a Pod's
program: the change shows in every report built afterwards, while saved reports
keep the names they were issued with. There is no dated reassignment. Corrupt
mappings withhold the report.

Holdings require same-close positions **and independently saved cash**, summing
to broker NAV within one cent. NAV minus positions is not proof of cash.
Missing/stale/unreconciled holdings are omitted with a printed note; complete
NAV-based performance can still be issued. Any short or negative cash disables
the holdings donut and uses signed holdings in Details. Current-month capital
movements use cents; more than 12 rows are folded into an explicitly labeled
earlier-movements row within that month.

## Issue, store and correct

1. Open Performance, then Monthly report, and select the completed month.
2. Review scope, balances, program identities and concise missing-data notices.
3. `Save report and print PDF` rechecks the displayed report identity, rejects
   drafts/stale previews and freezes HTML, CSS, font, optional picture, timestamp
   and source/method/program evidence.
4. The browser prints **the frozen copy**, not the earlier preview. `Saved copy`
   reopens it. If printing fails after saving, open that copy and print it.
   A failed save does not print a final report.

Files live in `monthly_reports/<sha256>.json` beside the event log. GET requests
do not write files. Saved reads verify the digest and never rebuild from current
data. Corrected source data produces a new version; existing files are never
overwritten. Back up these files with the event log. Demos are visibly synthetic
and store archives only in memory.

The digest detects changed bytes; it is not an independent audit or proof of
source authenticity. Program mappings assume one writing application process.
No GIPS compliance, administrator valuation or intraday broker parity is claimed.

## Verification and change boundary

Reporting calculations, display, program-name storage and report archival are
affected. Order timing, sizing/target semantics, price sources, execution logs,
broker connections, state/pickle formats, SQLite schemas and release YAML/routes
are unchanged. Archive writes are explicit, bounded and fail closed. Program
writes retain Windows replace retries. Deployment is a separate step.

Verification covers independent decimal examples, retired-program/reserve-only
horizons, missing source and transfer cases, immutable archives, protected
requests, backward parity and runner/scheduler/reconcile/release regressions.
Browser and PDF checks are recorded separately from Python tests.

## Local verification - 2026-10-26

- Baseline: `9745eef`. Work was isolated on `codex/monthly-report-accuracy`.
- Tier 3: independent parity/coverage and failure-mode/quant reviews completed;
  findings about printing the preview and stopping after all programs close
  were fixed and verified. Thirty-six complete-output comparisons to baseline
  passed where the optional containing transfer scope is unnecessary.
- Focused broad regression: **1,462 passed, 4 failed** across client accounting,
  holdings, monthly reports, Performance, IBKR imports and the required live
  runner/scheduler/reconcile/release suites. All four failures also reproduce
  unchanged on the original baseline: two `test_client_operations.py` diagnostic
  health expectations (September fixtures evaluated against October's real
  clock) and both `test_dashboard_v4_monthly_golden.py` golden-page
  tests. The golden mismatches are confined to `/` and `/overview/money?period=All`
  and have identical hashes on baseline and this branch. Expectations were not
  changed to conceal them. The existing `system_data.py:535` escape warning also
  remains. This is not a full-research-suite pass claim.
- Edge browser checks passed for single program, two programs, saved archive,
  VAULT and all programs retired. Every resulting example PDF has two pages and
  no horizontal overflow; the final VAULT PDF pages were rasterized and visually
  inspected. The print interaction was instrumented to verify that the frozen
  frame is printed and the older preview is not; no physical printer was used.
- Archive route tests verify authorization, stale/draft refusal, immutable
  reads, corruption handling, embedded assets and absence of action tokens.

No broker actions, production writes, release/configuration changes, push or
deployment were performed. The four baseline failures remain separately visible.

## Presentation update - 2026-10-27

At the owner's request, the report no longer renders the "About these numbers"
section or its methodology, fee and preparation paragraphs. The accounting
contract remains documented here. Draft labels and short unavailable-holdings
and capital-history notices stay beside the relevant content. Report identity
remains in document metadata; frozen archive evidence retains the preparation
timestamp, source hashes and method version. Previously issued copies remain
unchanged.

Validation for this presentation update: 116 report/archive tests passed after
adapting their HTML selectors, and all 144 required live regression tests passed.
Five browser/PDF variants retain two pages, omit the removed section and print
the frozen copy. Independent parity/coverage and failure-mode reviews found no
remaining blocker. Tier 3 by path; accounting, execution and storage formats
are unchanged. This update remains local and does not rewrite older archives.

## Owner update - 2026-10-27: programs only, editable programs

- The headline, allocation, month bridge and posted activity cover the programs
  only. Other accounts are named in the line under the bridge ("not included").
  Reason: VAULT is mostly shekel cash, so its USD change is the exchange rate.
- Months start on the first Pod day and, once every Pod has closed, end with
  the last one. A client with only other accounts gets no report.
- The Programs panel can change or clear a saved program. Saved reports keep
  the names they were issued with.
- Engine, archive, holdings rules and the Performance page are unchanged.

Validation: 2,785 Live Ops V4, report and client TWR tests pass. The two golden
Overview tests still fail as before; they also fail at `a4e810c`. Real Flex
numbers for September and October are unchanged. A two-Pod sample with a Vault
prints on two pages with no Vault amount, and a program change through the
panel was checked in the demo browser.
