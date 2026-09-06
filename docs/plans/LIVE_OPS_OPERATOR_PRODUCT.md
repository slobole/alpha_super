# LIVE OPS operator product

Status: **local implementation complete**, 2026-09-05.
Branch: `codex/live-ops-operator`. Packaged for owner-requested local commit and
independent review on 2026-09-06; no push or deployment is included.

Current requirement-by-requirement evidence, approved decisions and runtime limits:
[`LIVE_OPS_ACCEPTANCE.md`](LIVE_OPS_ACCEPTANCE.md).

## Outcome and boundaries

An operator-only, client-scoped product: a clear overview, consistent broker-backed
accounting, readable strategies and exposure, useful activity, diagnostic tools,
guarded existing operational actions, and date-selectable investor report exports.
Investors receive exported reports only; this is not an investor portal.

Implementation and preview are local. No deployment, broker connection, scheduler
operation, data sync, release changes, or production mutation is authorized by this
implementation task. Existing order, sizing, signal and execution timing stay fixed.
One live strategy remains one isolated broker account/ledger.

## Acceptance checklist

These checks refer to local implementation and synthetic/mock verification, not
deployed runtime health or real-client source validation.

- [x] Dedicated branch/worktree preserves unrelated owner changes.
- [x] Persistent client identity, environment, reporting currency and explicit
      operational-check versus financial-data timestamps.
- [x] One consistent operational verdict; expected idle is not missing evidence,
      and a weekend does not conceal a missing last-required EOD.
- [x] One selected financial period drives overview, performance, strategy rows,
      charts and reports. Current operational status is not shifted by that filter.
- [x] Effective-dated reporting scope is independent of scheduler enabled flags;
      adding/retiring a strategy does not rewrite the client's earlier history.
- [x] Broker NAV and explicit flow coverage support the dollar bridge:
      closing NAV = opening NAV + broker capital movements + linking adjustments
      + mandate entry/exit capital + investment P&L.
      Missing flow evidence is unknown, not zero. Internal transfers and asset
      transfers have explicit scope rules; fee basis is disclosed.
- [x] Official per-account TWR stays authoritative. No shadow/adjusted-base
      composite is relabelled exact client/fund TWR. Any client return must carry
      a validated source/method, complete coverage and matching economic scope.
- [x] Same-period market benchmark uses explicit total-return data, complete
      session coverage and source hashes; differences are percentage points,
      not dollar profit, consolidated TWR or risk-adjusted alpha.
- [x] Quiet, legible, consistent styling; readable strategy names, aligned figures,
      semantic text plus color, responsive layout and offline static assets.
- [x] Overview answers action needed, money, material changes and next event.
- [x] Exposure discloses valuation dates/coverage and distinguishes notional weight
      from embedded leveraged ETF exposure; no misleading mixed-date freshness.
- [x] Diagnostics show saved status, lifecycle, source/provenance and filtered logs;
      normal reads neither mutate trading state nor connect to the broker.
- [x] Controlled command catalog separates copy from execute, discloses effects
      and exact target, retains authorization gates and prevents preview replay.
- [x] Read-only local preview rejects operational POSTs server-side and never
      enables notifications, broker operations or production data writes.
- [x] Date-selectable report preview and download use the same accounting result;
      missing evidence produces a visible draft, with no confidential operator
      paths, credentials, commands or raw account identifiers in investor output.
- [x] Generated report content is versioned/source-stamped so later data revisions
      cannot silently change an already downloaded report.
- [x] Multi-strategy client configuration and isolated client navigation are
      supported without pooling unrelated clients' performance or execution.
- [x] Existing parity/LIVE-vs-reference detail remains available with explicit
      dates, timing, accounting/data lineage and reconstruction caveats.
- [x] Full focused regression tests, Tier-3 triage/live-impact checklist, independent
      parity/failure-mode/coverage/quant reviews, browser checks and report rendering.

## Accounting guardrails

Daily account NAV/TWR cannot prove exact consolidated TWR when offsetting intraday
flows are possible. A daily net-flow value of zero is not gross-flow completeness.
Do not invent broker XML field names, flow timing, management fees, inception dates,
benchmark prices or missing valuation rows. Retain raw source provenance, validate
the broker field contract and fail clearly on incomplete or inconsistent coverage.
Any approximation must have its own explicit method label, never an official label.

## Delivery sequence

1. Establish source/period/client reporting contract with pure calculations and tests.
2. Unify operational evidence and build read-only diagnostics and action safeguards.
3. Integrate overview, performance, exposure/activity and coherent local styling.
4. Add frozen report exports and a safe populated preview fixture.
5. Review, test, inspect the rendered product and close every acceptance item.

## Historical implementation milestones

The entries below record earlier checkpoints. Their pending/unapproved statuses
and test counts are superseded by the final approved delivery section at the end
and the current acceptance audit linked above.

- 2026-09-05: Created isolated worktree from `59ba59e`; owner working tree untouched.
  Read doctrine and current sources. Confirmed status/next_due/doctor are not
  blanket read-only commands; current performance binding discovery can also
  initialize a state database and must be made read-only for dashboard use.

- 2026-09-05: Implemented pure raw-Flex client accounting with explicit effective
  ownership, independent MTM bridge, missing-flow guards, D+1 metrics and stable
  provenance. Official account returns, selected-period drawdown and calendar
  months are distinct from unavailable consolidated TWR. Corrected cross-period
  NAV continuity, non-session membership and account-vs-book coverage.
- 2026-09-05: Added authenticated client directory and seven offline/responsive
  tabs. Root routes to clients when configured. Current operations are matched
  by account+Pod+LIVE, separate from financial dates; global cached summaries are
  not mutated or reused for client totals. Populated synthetic 2-/4-strategy demo.
- 2026-09-05: Added client strategy lifecycle, bounded attributable activity,
  saved diagnostics JSON and per-position reference timestamp exposure. Calendar
  missed-cycle findings now agree with the scoped headline. EOD evidence checks
  account identity and actual session close/buffer, not a stage label alone.
- 2026-09-05: PDF + frozen public JSON export uses one accounting result with a
  stale-preview hash gate, draft/demo labels and source/version stamps. Issued
  snapshot identity includes issuance time. PDF sample visually checked earlier;
  regenerate/re-render after the latest identity change before final delivery.
- 2026-09-05: 172 current focused regressions passed; independent accounting
  reviewer additionally ran 20 pure probes. Browser QA passed all seven tabs,
  exact date form, operator auth and PDF download at 390/768/1440 with external
  assets blocked. Latest overview/calendar/table adjustments require final
  screenshot inspection. Full acceptance checklist remains open, not inferred
  from these focused tests.
- 2026-09-05: One-use advanced action/manual-ticket authorization patch was
  rejected by safety review as outside read-only authority. Verified no part was
  applied; explicit owner approval requested. Continue unaffected UI/reporting
  work. No broker call, deployment, production write, commit or push performed.

- 2026-09-05: Closed calendar no-data/exception, missing-stage false-green and
  red-downgrade cases with real-builder regressions. New-mandate operations are
  independent of D+1 financial availability. Financial read failures preserve
  client/status/navigation. Norgate next-cycle subdetails are visible. Manual and
  systemd launch instructions now both use `--read-only`.
- 2026-09-05: Added explicit saved SPY/$SPXTR total-return benchmarks to each
  account's exact interval, shared result/hash and PDF v2. Previous-session
  baseline, holidays/weekend account costs, missing sessions, same-date source
  revisions, compact/future dates and extreme-value overflow have regressions.
  No additional account/advisory fee or investor tax is deducted from benchmark
  returns. Three-page synthetic PDF includes actual benchmark close dates.
- 2026-09-05: Broad local suite passed 569 tests; latest narrowed suite passed
  191 tests including release-manifest compatibility after additional source/PDF
  checks. Browser seven-tab x three-width matrix passes, including report768.
  Independent calendar reviewer passed 10 targeted cases; quant reviewer found
  no remaining blocker in the bounded benchmark integration after fixes.
  Superseded by the final verification entry below after heading/preview changes.
- Remaining product acceptance: selected-client/date-safe legacy comparison
  presentation, overview activity summary, historical/remote event coverage and
  investor language acceptance. Real-source ownership/fee/finalized-input checks
  are onboarding gates, not proof supplied by a synthetic demo. Owner choice is
  pending on final multi-account NAV/P&L reports without aggregate TWR; existing
  behavior remains DRAFT. One-use operational confirmation remains unapplied
  pending explicit approval. Persistent server report archive/auto-distribution
  are optional future extensions, not invented completion blockers.
- 2026-09-05 final local verification for this milestone: **610 tests passed**
  across client accounting/benchmark/views/operations/PDF, authentication/RO,
  dashboard health/routes/manual-ticket/notifications/calendar, IBKR performance,
  runner/scheduler/reconcile and release manifests. Seven tabs at 1440/768/390,
  report-preview benchmark visibility, exact-period form and PDF download passed.
  All three newly rendered PDF pages were inspected; method heading now stays
  with its text and no clipping/overlap was observed in the English demo. This
  closes this milestone's tests/render checks, not the entire product checklist.

## Earlier live-impact boundary (before final owner approval)

- Acceptance-audit increment: API-key field variants and Basic-auth text are now
  redacted from saved diagnostics/activity; synthetic regressions also prove
  idempotence and preservation of nonsecret evidence. Advanced diagnostics label
  the configured identity as **release owner**, not an inferred client mandate.
  The current broad suite passed **741 tests in 40.15 seconds**; this supersedes
  the 725-test result below. Full acceptance remains open, not inferred from tests.

### Earlier completed implementation and verification

- Historical Activity now follows selected-period ownership, including retired
  strategies and reused routes. Reads are deduplicated per Pod; current health
  and financial reports remain independent. Optional remote saved event exports
  work without local fallback. Explicit LIVE/account/Pod aliases, aware canonical
  occurrence times, both inclusive ET ownership boundaries and export cutoffs are
  enforced. Partial coverage, source failures, empty evidence and display caps
  are visible. Overview shows three material recorded events with the same dates.
- Date errors retain client identity/navigation and an editable form with HTTP
  400. Compact dates are rejected rather than silently normalized. Weekend-only
  empty financial selections describe missing valuation rows, not absent ownership.
- Performance now has a per-strategy saved LIVE/reference evidence panel. An
  optional server-configured `reference_summary_path` pins the historical JSON;
  no discovery, generation, pickle, current-route inference or broker call occurs.
  Hashes identify the exact bytes read. Missing historical account identity means
  metadata-only. Wrong identity, different intervals or missing/mismatched actual
  EOD date/source withhold recorded values. Even eligible values are not presented
  as aligned returns, P&L, slippage, statistical tracking error or certified replay.
- Independent parity/quant, failure-modes and coverage/presentation reviews found
  three follow-up issues: conflicting Pod alias, path-like allowed metadata, and
  a time formatter that hid the year/assumed a timezone. All were fixed with
  regressions; full original comparison timestamps and full-year ET event times
  are retained. The owned-weekend empty-state issue was also fixed and tested.
- Final local regression: **725 tests passed in 41.54 seconds** across all client
  modules, investor reports, operator access/tools/health, dashboard routes,
  calendars/charts/notifications/manual-order boundaries, broker performance,
  runner/scheduler/year simulation/reconcile and release manifests. This supersedes
  earlier milestone counts; counts are not additive.
- Browser matrix passed all seven tabs at 1440/768/390, including expanded
  comparison evidence, event excerpts, exact-period navigation, invalid-date
  correction, other-client isolation, offline assets and authenticated PDF download.
  Desktop Overview/Performance and mobile Activity were visually inspected.
- Triage remains Tier 3. No order timing, sizing, execution-reference semantics,
  broker adapter, executor, release YAML, SQLite schema or persisted trading state
  was changed by this increment. The one-use operational authorization patch is
  still unapplied. No VPS deployment, broker connection, commit or push performed.

Remaining acceptance decisions: the owner has not yet approved changes to the
advanced operational confirmation path. Final multi-account report policy remains
unresolved: current PDF output is DRAFT without an exact consolidated return;
allowing a final NAV/flows/P&L report with only separate official account returns
requires that product decision. Current investor output is English; RTL/language
expansion and persistent archive/automatic distribution are not implemented.
Real source onboarding must establish account ownership, fee basis, finalized
IBKR evidence and any pinned comparison lineage; synthetic QA does not prove these.
The full product goal is not marked complete by this verification milestone.

- Tier 3 includes live-consumed UI/authentication/accounting readers, plus
  presentation, dependency metadata and local demo/test tooling.
- No signal/order timing, sizing (`amount`/`target`), reference fill/mark source,
  broker adapter, scheduler, reconciliation executor or released YAML changed.
- Existing SQLite schemas/state/pickle formats are unchanged. Reporting opens
  existing sources read-only; the optional client registry is separate.
- Existing dashboard/log field names remain. EOD display trust is stricter about
  account identity and actual close/buffer, not about order execution semantics.
- Startup deliberately requires an operator credential. Demo skips config.env,
  uses synthetic sources and forces read-only. Service template is read-only.
- Missing/invalid source, lock/path error or calendar proof cannot mean green;
  benchmark failure cannot replace account returns or initiate data retrieval.
- Windows/browser/PDF checks are local; no VPS deployment or real-money action
  has been performed. Replay-sensitive existing commands were not modified.

## Final approved delivery — 2026-09-05

The owner approved the two remaining decisions, and both are implemented locally:

1. Verified multi-account NAV/capital/P&L reports may be FINAL with separate
   official account TWR, even without consolidated TWR. The latter remains
   explicitly Not reported. Missing required account/flow/TWR evidence stays
   DRAFT; synthetic output stays DEMONSTRATION. FINAL is not external audit or
   certification. Accounting mathematics is unchanged; PDF renderer is v3.
2. Advanced operations use one-use, 120-second approvals bound to the exact
   target/action/account/release/state. Actual worker-start evidence and release
   are checked; direct submit/reconcile plan intent is bound and manual inputs
   are frozen. Cancellation, expiry, replay and changed evidence fail closed.

```text
[Read-only evidence and preview]
                |
                v
[Exact target / action / account / release / state]
                |
         approve within 120 seconds
                v
[Consume once; revalidate at worker start]
                |
                v
[Actual release / direct-action plans / frozen manual ticket]
                |
                v
[Existing executor: original timing / selection / CAS guards]
```

Supported authenticated CLI startup now defaults to read-only. Actions require
explicit `--enable-actions`, incompatible with demo mode. The service template
stays read-only. Legacy direct `create_app()` defaults are unchanged and are not
the supported secure deployment entrypoint.

The six-command catalog separates Copy from Preview; Copy runs nothing. Copied
CLI commands use current state outside UI approval, and even CLI status writes
metadata. Saved status remains the read-only option; there is no generic UI shell.

Approvals are single-process and in-memory, not a global scheduler lock or an
exactly-once broker guarantee. Restart rejects unknown approvals. Tick authorizes
a cycle that can create plans, not a fixed order list. Errors/timeouts consume
approval permanently; inspect job/broker evidence before a fresh attempt.

### Final verification and scope (refreshed 2026-09-06)

- **811 tests passed in 71.12 seconds** across client/accounting/reporting,
  dashboard/auth/confirmation/manual inputs, broker performance, runner,
  scheduler/year simulation, reconcile and release compatibility.
- Independent accounting/parity/quant, failure-mode and UI/coverage reviews
  closed their findings; details are recorded in the acceptance audit.
- Seven client tabs passed browser QA at 1440/768/390 with external assets
  blocked, exact dates, client isolation, visible recovery and PDF export.
- Synthetic real-HTMX QA passed preview/confirm, replay, cancellation, network
  failure recovery and copying without execution. Fixture processes close on exit.
- Both regenerated renderer-v4 English PDF pages were inspected without clipping or
  overlap. `output/pdf/live-ops-investor-demo.pdf` is synthetic, not a statement.
- Tier 3: optional runner authorization changed. Signal/order timing, sizing,
  reference prices, order construction, released YAML and persisted trading
  schemas did not. CLI/scheduler calls have no dashboard context and keep their
  original selection, claim and execution behavior.

Local acceptance is complete; no broker call, VPS change, deployment or push
was performed. Real-client source/ownership/fee onboarding and separately
authorized deployment/observation remain outside this implementation. Future
month-end vendor, VPS or broker availability is not guaranteed. RTL output,
persistent server archives, auto-distribution and newly certified replays remain
optional future work.

The 2026-09-06 owner-requested PDF simplification uses short metric labels and a
conditional account-source sentence instead of the long methods/source-hash
section. Full method, fee basis and provenance remain in frozen JSON. The PDF
keeps its complete document ID and draft/demo/unavailable indicators. No financial
calculation, finality gate or operational behavior changed.
