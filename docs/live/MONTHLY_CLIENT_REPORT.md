# Monthly client report

The report at `/monthly-report` uses saved broker evidence. It does not place
orders, move capital, import broker data or deploy anything. Recommended cadence:
once per completed calendar month, after statements reconcile. An unfinished
month is a printable draft. One Pod prints on one A4 page; more Pods add a Details
page; larger books may continue onto additional pages without hiding rows.

## Meaning and scope

- Headline NAV, dollar profit and TWR cover the **programs only**: the strategy
  Pods, the same scope as the Performance page. Program rows cover their own
  Pods. Other accounts such as VAULT are named on page 1 but never counted:
  IBKR translates their shekel cash to USD, so their USD change is mostly the
  exchange rate, not a result (owner decision 2026-10-27).
- "Since start" is the first Pod day. Once every Pod has closed, the months end
  with the last Pod day, and that month is final once statements reach it. A
  Pod missing a required day still blocks finality; it is not silently dropped.
- Values use the approved USD-base statement contract. A statement actually
  denominated in ILS is unsupported. Showing VAULT in shekels would need IBKR's
  cash report by currency in the Flex query, which is not configured.
- `Net invested` (`Net capital` until 2026-11-01; that is also a regulatory term for
  brokers) includes the initial measured value and
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

Holdings require same-close IBKR Open Positions in the same saved import as the
NAV row. That statement's NAV minus its positions prints as `Cash and accruals`,
or `T-bills and cash` when T-bill funds share the class; it is not presented as
proven cash (owner decision 2026-11-01). The fallback saved portfolio capture
must still reconcile to NAV within one cent. Missing or stale holdings are left
out with one plain sentence on the page and the technical reason on the operator's
screen and in the saved evidence; complete NAV-based performance can still be
issued. Any short or a negative remainder of 0.1% or more of the Pod disables the
holdings donut and lists signed holdings in Details. Current-month capital
movements use cents; more than 12 rows are folded into an explicitly labeled
earlier-movements row within that month.

## Issue, store and correct

1. Open Performance, then Monthly report. It opens on the current month to the
   latest saved day, marked as a draft; select the completed month to issue it.
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
- The report opens on the newest month, to the latest saved day. While that
  month is in progress it is a marked draft; completed months are in the list.
- The month bridge prints cents with gain/loss as the balancing line, so the
  printed lines always add up; a cent of rounding goes into gain/loss, so it
  can differ by a cent from Performance. Program cards say "since Sep 24,
  2026", and the allocation legend puts the masked account under the Pod number.
- Engine, archive, holdings rules and the Performance page are unchanged.

Validation: 2,785 Live Ops V4, report and client TWR tests pass. The two golden
Overview tests still fail as before; they also fail at `a4e810c`. Real Flex
numbers for September and October are unchanged. A two-Pod sample with a Vault
prints on two pages with no Vault amount, and a program change through the
panel was checked in the demo browser.

## Owner update - 2026-11-01: the Dusk layout

Chosen by the owner over several mockup rounds (band, then Dusk). Leading values:
elegance, simplicity, ease of reading, and the bottom line first. Accounting,
engine, archive and the Performance page are unchanged; only the page changed.

- Page 1: the picture band, a dark panel with the name and the bottom line (value,
  the month, since start, net invested, one more row when money moved), the return
  chart, holdings. Removed: the Pod donut, card frames, the Opening/Closing strip
  (now `Capital account` in Details) and filler sentences.
- One Pod = one page. More Pods, or signed holdings to list, add a Details page:
  months with Pod rows, capital account, money in and out, activity.
- The money-moved row comes from the ledger behind `Net invested`, so the row and
  that figure agree. A Pod entering on a month's first session counts in that
  month; a Pod closing on a month's last day counts in the next, because it was
  still in that month's value. The client's very first day is not money moved.
- The table under the panel goes one level down: programs when there are two or
  more, otherwise Pods still holding money plus any that closed this month, so the
  rows add up to the month; otherwise nothing. Pods print in number order, by number
  and masked account; the IBKR alias stays in the operator's Programs panel. The chart
  draws one line per program and takes the room the rest of the page leaves, 120 to
  208 px with the months on the page and up to 280 with them on Details. A crowded
  page then takes room from the picture, from 176 down to 96 px, so it stays on one
  sheet. Past that a page continues on a further sheet; a program's name stays with
  its holdings.
- The panel's top right carries the LS monogram, the name and `Quantitative
  Strategies`. Avoid regulated words here (fund, capital, asset or portfolio
  management, advice) until the lawyer has reviewed the wording.
- `Other fees` (IBKR `otherFees`) is listed beside commissions. It is already inside
  the gain/loss; the line only shows it. The activity caption says commissions and
  fees are charged by Interactive Brokers, so a client does not read them as the
  manager's charge (owner request 2026-11-01). It stays off when no cost is printed.
- A long program name is cut in the chart label only; the table shows it in full.
- Four type sizes only: 48, 22, 14 and 12 px (40 for a value over ten characters).
  The font is Hanken Grotesk, bundled in `static/fonts` and embedded in saved copies.
- A name stored in capitals reads as a name; any other spelling stays as typed.
- `TWR` is named once, as a legend on the chart (owner decision 2026-11-01). The month
  table's caption is `Return (%)`. Holdings carries no date of its own; the panel's
  `Value on <day>` gives it.
- A month not final yet says so in its date, `October 2026 · to Oct 30`, with no bar on
  paper (owner decision 2026-11-01). The operator's screen keeps the reason, the button
  stays `Print draft` and it cannot be saved as final. The amber bar prints only for
  data that does not tie, in any month.
- Accounts held apart (any Flex account with no strategy, whatever its IBKR alias) stay
  out of every number and get one quiet line under the table, naming no account: their
  cash in each currency, from IBKR's Cash Report in the same statement as their NAV, for
  example `Held separately, not included: ILS 4,100,000 and USD 3,000` (owner decision
  2026-11-01). With no Cash Report, with positions in an account, or when the cash does
  not reach its NAV, the line gives no amount: `An account held separately is not
  included.`
- A month that passed with no number in its row is lightly shaded; months still to
  come stay white (owner decision 2026-11-01).
- `Return by month` has one row per year, newest first, with Jan to Dec and the
  year's linked return, so it serves any number of years; a first year's return is
  the return since start. Pods are listed under the report's year. With one Pod and
  more than three years the table moves to a Details page.
- Activity lines that round to zero are hidden. Six or more holdings classes on one
  program use two columns.
- Holdings show tickers with no leverage multiple (owner decision 2026-11-01). Weights
  are by value, so a leveraged fund's market exposure is larger than its weight.
- Without the since-start report, `Since start` shows a dash and the chart is left
  out, for every program; a program's own month is never shown in their place.

### Pictures

The report finds `static/report_art/<YYYY-MM>.jpg|webp|png` by itself: the newest
picture not after the month, otherwise the oldest, so a report never stops for a
missing picture. Saved reports embed theirs and never change.

1. Make the pictures once a year in ChatGPT or Gemini from one fixed prompt, so
   they read as a series. Save them in a folder, named by month (`2026-11.png`).
2. Run `python scripts/prepare_report_art.py <folder>`. It crops each to the band
   (1588 x 352), saves a JPEG of about 200 KB and refuses a source under 1200 px
   wide. A full-size picture makes a 4.5 MB PDF; the band makes about 300 KB.
3. Commit the files. Each client's next pull brings them; no restart is needed for
   a picture alone. A set of 12 avoids pulling `main` onto live machines monthly.

Fixed prompt, changing only the scene:

```text
Pointillist painting made of thousands of small pastel dots, calm and quiet, at dusk.
Scene: [THE MONTH'S SCENE, for example a harbor in early winter].
One warm street lamp near the left side. Pink and peach sky, amber lights, deep teal water along the bottom edge.
Wide 16:9 image, at least 1600 pixels wide. Keep everything important in the middle 40% of the height, because the top and bottom will be cropped.
No people in close-up, no text, no logos, no signature, nothing about money or markets.
```

### Validation

Tier 3 by path; no order, sizing, price, state or schema path changed, and only
`app.py` imports `monthly_report`. A separate review compared the committed module
with the new one over 17 synthetic books: every money, bridge, flow, program, holding,
month and chart-end value was identical. Its findings were fixed and re-checked; the
new money-moved row tied to `Net invested` within $0.11 across 70 books. The year
table came after that review and is covered by tests and the page checks below.

2,833 Live Ops V4, report, client reporting and required live tests pass. The two
golden Overview tests still fail, on `/` and `/overview/money?period=All` only, and
they fail the same way without this change. Through the real route, 25 page shapes
were measured and printed in Edge: one Pod is one sheet and more Pods two; three or
four programs continue on a third. The two real IBKR exports printed at 316-360 KB,
and September matched the figures verified on 2026-10-26. A saved copy from the demo
is self-contained (320 KB, no links out, no scripts). Not run on a client's VPS yet.

The later pass (Other fees, Pod order, chart and picture budget, alias, monogram, one
`TWR`) had separate parity, failure-modes and coverage reviews. Parity: on 12 demo
reports and 5 sample books only the print order moved, and an `otherFees` charge lowers
the gain and the closing value by exactly its amount, once. Failure modes found two
pages spilling onto a sheet that held only the footer; both are fixed and in the page
checks. 2,848 tests pass, with the same two golden Overview failures. Through the real
route, 38 page shapes were measured and printed in Edge: one Pod is one sheet, more Pods
two, three programs two, four programs three. Sixteen crowded pages built by hand and
printed in Chrome hold their sheets up to eight Pods. These still run onto a further
sheet: nine Pods with a draft bar, two programs of eight long rows with a draft bar,
three programs with long ticker lists, four programs, and a three-year page carrying
every optional line.

The last round (`TWR` as a legend, no holdings date, no bar on a month not final yet)
changed display only and had no separate reviews. 2,850 tests pass, with the same two
golden Overview failures. The 38 page shapes hold their sheets; four programs now fit
on two, with no room to spare.

The account-held-apart line was checked on a real export with Cash Report: the reader
gave the balance by currency on 22 of 22 days, cash matched the account's NAV within
its accruals, and the report's other figures matched the printed copy from the VPS.
