# Alpha / Ops dashboard design refresh

Local implementation, September 2026. User scope: retain the new workspace,
restore useful allocations and operational evidence from the old screens, and
make navigation, typography, charts and current flow position simple to read.

## September 12: minimal financial view and cash allocation

Overview shows account value, P&L and TWR. Strategy results and Performance show
each strategy's starting and ending values for the selected period. Repeated
source/date captions, the pending-import line, chart-basis captions and the JSON
footer are removed. The date inputs still state the selected period; missing
figures and collapsed data issues remain. The return-method description is on
the TWR label's tooltip. Calculations, financial defaults and exports are unchanged.

Portfolio allocation uses one ring, with each strategy's invested and cash
segments adjacent. Cash uses a lighter shade of the strategy color. Percentages
use total portfolio NAV as the denominator; the center shows aggregate cash.
Repeated strategy names retain visible account identities.

```text
[Selected-date IBKR NAV] + [Same-date saved broker EOD cash]
                          |
              [Account / Pod / date / NAV match]
                          |
         [Invested segment + cash segment in one ring]
```

For strategy i: `cash_share_i = cash_i / sum(NAV_i)` and
`invested_share_i = (NAV_i - cash_i) / sum(NAV_i)`. Matching requires finite
values, `0 <= cash_i <= NAV_i`, and broker EOD equity within $0.01 of canonical
NAV. Missing, negative, over-NAV or ambiguous cash leaves that strategy's full
NAV slice unsplit with cash unavailable; total cash is withheld until every
strategy is verified. Missing cash is never zero-filled.

Local cash comes only from the configured Pod database opened with `mode=ro`,
filtered by Pod, account, user, broker source and EOD stage. The selected ET date
must match exactly, and timestamp evidence must be timezone-aware, no later
than the read and after the exchange close plus the existing 10-minute buffer.
The latest valid capture on that date is used; duplicate latest timestamps are
ambiguous. Malformed old timestamps are skipped individually. Snapshot clients
remain on their scoped saved source; synthetic demo cash is separate from
ChangeInNAV. No cash read occurs for financial exports or other financial views.

Live-impact checklist: order timing, sizing, reference-price semantics,
state/pickle/SQLite schemas, config formats, logs and released YAMLs are
unchanged. Optional missing/locked cash sources preserve financial NAV. The
existing broker capture defaults a missing TotalCashValue tag to zero and does
not retain tag-presence provenance; this display change does not alter capture.
The UI uses stored cash facts and cannot independently establish raw-tag presence.
No broker, scheduler, release or deployment action is part of this change.

Verification for September 12 (Tier 3; tests/browser tools/docs are Tier 0):

- Broad client/dashboard/live regression run: 979 passed; one old exact-markup
  assertion was updated for the tooltip attribute and passed on rerun.
- Final TWR and cash regression suites: 69 passed, including report-hash parity,
  real-provider same-date cash and unchanged stored files.
- Demo and both real-provider synthetic browser suites passed all seven views
  at 1440/768/390 pixels, plus Advanced, cash extremes/unknown HTML states and
  large mobile values. A refreshed loopback demo returns 200 with 30% cash and
  without the removed source/footer lines.
- Parity/quant, failure-mode and coverage reviews have no remaining findings.
  Fixed malformed old timestamps, summary/history duplicate-check inconsistency,
  repeated-name identity and mobile KPI width concerns.
- Scoped triage and `git diff --check` passed. Unrelated research files remain
  untouched. Cash availability remains limited by saved broker evidence.

Each VPS serves exactly one client. Entry opens that client's Overview directly;
there is no client directory or switcher. Remote viewing uses the same workspace.
The dashboard rejects a reporting registry containing multiple clients without
selecting one or loading their financial/operational evidence. The shared reporting
validator and account/Pod routing are unchanged. The CLI demo has one client with
four separate synthetic strategy accounts, including in Advanced views.

## Design and requirement evidence

| Requirement | Implemented behavior | Verification |
|---|---|---|
| Consistent minimal type | Segoe UI/system sans for text, figures and SVG; restrained size hierarchy | Browser computed-font assertions; desktop/mobile screenshots |
| Portfolio allocation on Overview | Same-date account NAV donut, account identities and percentages | Exact scope/duplicate/missing/retired-account unit tests; rendered 2- and 4-account examples |
| Per-strategy allocation | Priced holdings composition with signed values, source dates and unpriced exclusions | Long/short/missing/nonfinite tests; real-format synthetic holdings render |
| Daily dollars beside charts | Exact-date P&L readout, pointer/keyboard inspection, including losses | SOD/missing P&L regression; browser checks -$25, +$60 and unavailable baseline |
| Monthly green/red | Subtle gain/loss backgrounds and colored values | Positive and negative months visually inspected on long-history Performance |
| Visible execution flow | All saved stages, focus from explicit action/blocker, timestamps, previous-cycle labels | ACK blocker, stale/keyless, missed/expired and previous-cycle cases |
| Detailed operational evidence | Open stages for targets, orders, ACKs, fills and recorded reconciliation; freshness beside flow | Allowlist/identity/status/count-race tests; browser opens actual synthetic ACK records |
| Better graphs | Restrained blue line/area, light grid, readable aligned axes and live readout | Axis alignment, gap markers, keyboard interaction and responsive browser checks |
| Consistent Advanced styling | Advanced base loads the same typography and style layer | Advanced routes and diagnostic tabs checked at all widths |

Stage details are collapsed by default; the highlighted stage is visible before
opening them. Schedules follow the primary financial/flow content and remain
available if financial-source loading fails.

```text
[Scoped saved account / Pod facts]
                 |
       [Display validation]
           /           \
[Overview + charts]   [Strategy flow]
                          |
                [Stage details + freshness]
```

## Data contracts

Portfolio allocation is `weight_i = account_NAV_i,D / sum(account_NAV_D)` at the
displayed closing date D. Every account must have exactly one finite,
nonnegative value; their sum must match the displayed NAV within $0.01.
NAV-only account scope remains independent of performance-window completeness.
No earlier exited strategy NAV, missing cash estimate, or global VPS total is
substituted into a selected client's allocation.

Holdings composition is `abs(shares_i * reference_price_i) / sum(abs(priced_values))`.
This is explicitly a reference composition of priced holdings, excluding cash;
it is not a current NAV weight. Signed values, position/price timestamps and
unpriced holdings remain visible. Target weights are displayed separately as
saved decision targets, never mistaken for realized allocation.

Daily dollar annotations use the existing broker/accounting P&L for that exact
date. No NAV differencing, gap filling, return averaging or SOD future-day
annotation is introduced. Existing line gaps and unavailable returns remain.

Flow focus follows saved required action and explicit scheduler intent, with
blockers taking precedence. It never means that the first unknown stage is
currently executing. Previous-cycle evidence is labeled. A recorded fill is
not relabeled complete execution. Stale assessments receive no current focus.

Detailed tables are read only for a uniquely matched local Pod/account/LIVE
identity. Snapshot clients never fall back to local detail. Only allowlisted
fields reach templates. Plan identity/status and ACK/fill counts must agree
with the saved assessment. Reconciliation tables come from the persisted
snapshot bound to Pod, VPlan, status and timestamp, not a later cached-position
comparison. A missing or locked detail source preserves the summary and flow.

## Verification and live-impact checklist

Tier 3, with Tier 0 test/tooling/documentation surfaces. Read-only reviews:
parity/quant-pitfalls, failure modes, coverage. Findings fixed include NAV-only
allocation omission, invalid numeric holdings, flow blocker precedence,
previous-cycle wording, missing fallback schedules, optional SQLite failures,
summary/detail races and reconciliation provenance.

- 932 tests passed across client/dashboard, live dashboard, runner, scheduler,
  reconcile and release-manifest suites before final detail-provenance fixes.
- 153 focused tests passed after those fixes, including the live dashboard and
  client projection/integration regressions.
- After the single-client correction: 938 tests passed across the same full
  client/dashboard/live suites; 62 client-view tests passed after the final
  malformed-configuration handling fix (invalid UTF-8 and non-finite JSON).
- Browser checks passed at 1440/768/390: all seven client views, Advanced
  routes/diagnostics, local-only assets, single-client landing, rejection of
  another client's URL, and PDF export. The refreshed local preview was checked
  at `/clients`: it redirects to Overview without a Clients tab or switcher.
- Real-format synthetic browser checks verify unchanged fixture files, no
  outbound connections and rejected operational mutations.
- Both complete and incomplete-source/new-strategy browser scenarios passed.
- `git diff --check` and `uv run python scripts/review/triage.py` passed.

Live-impact checks: order timing (including next-open), sizing/amount/target
semantics, reference-price selection, saved-state/pickle/SQLite/config formats,
logging fields and released YAML routes are unchanged. The only addition to
the existing detail reader is a read-only query for the saved reconciliation.
Optional read errors fail locally without process restarts or broker calls.
No strategy signals, adjustments, costs, data sources or accounting formulas
change. No deployment, release, broker action or scheduler operation occurred.
The dashboard intentionally rejects a registry containing multiple clients;
the shared registry format and reporting calculations are unchanged. The
one-client correction passed parity, failure-mode and coverage reviews. Its
only review finding was malformed config handling, fixed and regression-tested.

Screenshots are generated by the checked-in browser scripts under
`.codex_tmp/client-ui`, `.codex_tmp/local-expanded-ui` and
`.codex_tmp/local-workspace-ui`. Preview data is synthetic. Local visual and
integration verification does not establish production source completeness.
