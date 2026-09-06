# LIVE OPS product acceptance audit

Date: 2026-09-05. Branch: `codex/live-ops-operator`, isolated worktree based on
`59ba59e`. Packaged for owner-requested local commit and independent review.
This is **not a production sign-off**. Verification refreshed 2026-09-06 after
the PDF simplification, compact dashboard and visible Pod flow changes.

## Verdict

The approved **local implementation is complete** against the checklist below.
The owner approved both remaining decisions: verified multi-account reports may
be FINAL without consolidated TWR, and the advanced authorization path may be
implemented and tested locally. Both are implemented and verified. Supported CLI
startup is read-only by default; operational actions require explicit opt-in.

No evidence from a real client was used to claim readiness. Real-source onboarding
and a separately authorized deployment review remain prerequisites for using this
branch on a VPS. No broker operation, sync, deployment or push was performed.

## Requirement-by-requirement evidence

The numbers follow the acceptance checklist in `LIVE_OPS_OPERATOR_PRODUCT.md`.
PASS below means local product acceptance with synthetic/mock verification, not
live runtime proof or permission to deploy or operate a broker.

| # | Requirement | Evidence inspected | Verdict |
|---|---|---|---|
| 1 | Isolated branch preserves owner work | `git worktree list`: owner `main` and separate `codex/live-ops-operator`, both based on `59ba59e`; all task edits target the separate worktree | PASS |
| 2 | Persistent client/environment/currency and separate timestamps | `client_layout.html`, `client_views.py`, `_client_operations_summary.html`; `test_client_routes_keep_scope_and_dates_and_are_operator_only` | PASS |
| 3 | Consistent truthful operational verdict | `client_operations.py`, `health.py`; tests for missing Friday EOD, early close, wrong account, future evidence, unknown calendar and preserved red actions | PASS |
| 4 | One financial period; current health stays current | `test_all_views_export_same_period_result_hash`, `test_financial_filter_never_changes_current_operations`, incomplete current-period and invalid-date recovery regressions | PASS |
| 5 | Effective ownership includes retired history | `client_reporting.py` ownership and scope-capital logic; retirement/entry tests; `test_retired_reassigned_pod_reads_once_without_current_release_or_health` | PASS |
| 6 | Dollar bridge separates capital from profit | Independent MTM components reconcile within $0.01; tests for deposits, linking adjustments, internal transfers, unknown fields, missing zeros and nonfinite components | PASS, given reviewed source/profile |
| 7 | Official account TWR; no invented fund return | Geometric broker-return linking; `test_two_accounts_have_dollar_bridge_but_no_invented_combined_twr`, equal-opposite transfer regression | PASS |
| 8 | Exact-period total-return benchmark | `client_benchmark.py`; previous-close baseline, required-session coverage, hash validation, adjustment rejection and weekend/holiday account-cost tests | PASS, given configured source |
| 9 | Readable, responsive, offline styling | `custom.css`, local fonts, friendly registry names; browser matrix at 1440/768/390 with nonlocal assets blocked and no page overflow | PASS |
| 10 | Overview answers status, money, changes, next event | Scoped status, money cards, three material events and trading calendar; excerpt cap and exact Activity-link date tests; inspected screenshot | PASS |
| 11 | Honest exposure dates/coverage | `build_reference_exposure_list`; distinct position/reference timestamps, finite-value checks, no mixed-date NAV weights or invented ETF leverage | PASS |
| 12 | Saved diagnostics and filtered logs without operations | Five diagnostic views, in-memory JSON, explicit source timestamps; read-only route tests verify no jobs/journal/notification writes; credential redaction regressions | PASS |
| 13 | Controlled command catalog, copy/execute and replay prevention | Six fixed, quoted commands; Copy is separate from Preview. Atomic single-use 120-second approval, fresh worker-start target/state checks, actual release and direct-action intent binding; frozen manual ticket. HTTP/concurrency and real HTMX tests | PASS within documented single-process boundary |
| 14 | Read-only preview blocks side effects server-side | `test_read_only_rejects_authenticated_mutation_before_provider_access`, preview/token/manual/trade-sheet rejection, notification suppression; `--demo` skips real config/provider | PASS |
| 15 | Matching report preview/export, safe DRAFT and confidentiality | Shared result and source-revision 409 gate; public-field allowlists. Approved FINAL policy requires complete verified account/flow evidence and per-account TWR; missing aggregate TWR is explicitly Not reported, never estimated. Builder integration regressions | PASS |
| 16 | Frozen source-stamped report | Accounting hash, issuance-sensitive document hash, renderer version; `test_pdf_bundle_freezes_public_snapshot_and_matching_pdf` | PASS |
| 17 | Multi-client/strategy isolation | Separate registry ownership; account-overlap rejection; local/remote source and other-client isolation tests; two-/four-strategy browser demo | PASS |
| 18 | Saved LIVE/reference detail with honest boundaries | Explicit historical JSON pins; Pod/account/LIVE, date and actual EOD-mark gates; byte hashes; full timestamps; no P&L/slippage/TE/replay certification | PASS as saved diagnostics, not certified replay |
| 19 | Tier-3 verification and complete acceptance | 829 tests, Tier-3 triage, independent parity/quant, failure-mode and coverage reviews; seven-tab responsive and HTMX QA, plus both rendered PDF pages inspected | PASS for local implementation |

The full dollar bridge is:

`Closing NAV = Opening NAV + broker capital movements + linking adjustments + mandate entry/exit capital + investment P&L`.

Daily net flows alone cannot establish exact multi-account TWR when offsetting
intraday flows are possible. The approved policy changes report eligibility, not
accounting mathematics: verified displayed facts can be issued FINAL while the
consolidated return remains absent. FINAL is not external audit or certification.

## Review fixes — 2026-09-06

Local follow-up to `529668a`; accounting methodology is unchanged.

- Navigation carries only explicit dates/presets. Operational defaults can
  include today without silently selecting today in D+1 financial views.
- Web/PDF losses use `-$10.00`; PDF renderer-v5 displays a readable UTC issue
  time while preserving full timestamp/hash provenance. The synthetic sample
  was regenerated and both pages inspected, plus a losing-account test PDF.
- Both shells display the actual access mode. Ambiguous enabled targets produce
  controlled 409 responses before command generation/export; auth/read-only
  guards remain first. Health-probe and independent-alert instructions clarified.
- Advanced VPS/diagnostics share the local offline styling. Friendly labels
  require unique current local LIVE Pod/account ownership; routing IDs remain
  unchanged. All stages stay visible on mobile. Warning rows are not painted
  red by the attention group's container; long event names wrap.
- Small bounded in-memory decode caches retain current-byte SHA256 validation,
  fresh read-only SQL, source revisions/tombstones and detached Flex attributes.
  No error/finality/report cache. Benchmark reads/hashes still occur each time;
  long XML histories can exceed the cache, so speedup is not guaranteed.

Verification: **864 passed in 72.42s**, then **132 passed** after the last
attention-panel styling change. The offline browser matrix passed all seven
client tabs plus `/vps`, `/pods/live` and all five advanced diagnostic tabs at
1440/768/390. It checks local CSS/computed warning color, all stages, client
isolation and authenticated PDF download. Separate HTMX confirmation, replay,
cancellation, network-failure recovery and copy-without-execution QA passed.
Three independent read-only reviewers covered parity/quant, failure modes and
UI/coverage; no blocker remains. Tier-3 triage and diff whitespace checks pass.

Commit closure recheck (2026-09-06): **859 tests passed in 40.90s** across
client, dashboard, IBKR performance, investor report and live runner/scheduler/
reconcile/release suites. The seven-client-tab and seven-advanced-route offline
browser matrix passed again at 1440/768/390; desktop and mobile Pod flow images
were inspected. Fresh parity, failure-mode and coverage reviews found only an
ignored Tailwind MIT license, now explicitly included with the vendored CSS.
The action-mode runbook now explicitly requires `--enable-actions`.

Live-impact checklist: next-open timing, sizing/amount/target semantics,
reference prices, schemas/state/config/pickle formats, consumed logging fields
and released YAML/routes are unchanged. Source reads still fail closed on
Windows file errors/replacements; no new service/process lifecycle is introduced.
Quant semantics (PIT, timing, adjustments, costs, sample periods and official
account return linking) are unchanged. No strategy/performance claim is added.
This follow-up is packaged as a local review-fix commit only; no push,
deployment, broker or VPS operation is part of this closure.
Real-client source validation and separately authorized rollout remain required.
Consolidated client TWR remains a separate accounting phase. Further cache-I/O
optimization, read-only trade-sheet export and optional report-layout suggestions
are not claimed as implemented by this review-fix commit.

## Pre-review verification (`529668a`)

- Main suite: **829 passed in 48.58 seconds** in the final pre-commit run,
  including the optional-comparison exclusion and diff-only fallback case.
  The 16 focused PDF tests also pass. Scope includes client accounting,
  benchmarks, views, activity, comparison, reports, access, tools, health,
  dashboard routes/calendars/charts/notifications, mocked manual execution,
  IBKR performance, runner, scheduler/year simulation, reconciliation and releases.
- Latest independent review: accounting reviewer ran 13 actual-builder scenarios;
  UI/coverage reviewer ran 35 targeted checks and two HTTP failure probes.
  Failure-modes reviewer inspected the final authorization, release-history and
  CLI changes. These checks are not added to the main suite count.
- Browser: seven tabs at 1440/768/390; expanded comparison, selected-period
  navigation, invalid-date recovery, event limits, client isolation, offline
  assets and authenticated PDF download passed. A second synthetic browser suite
  exercised real HTMX preview/confirm, replay, cancellation, network failure,
  visible recovery and copying without execution. Both suites close their own
  fixture servers and browsers; no persistent preview process is claimed running.
- Visual evidence under `.codex_tmp/client-ui/`; sample PDF at
  `output/pdf/live-ops-investor-demo.pdf`. Images/PDF are synthetic, not investor
  statements. Both regenerated renderer-v4 PDF pages were visually inspected;
  no clipping or overlap was observed in the English sample.
- The owner-requested shorter PDF removes long qualifications and printed source
  hash lists, while retaining a conditional account-source sentence, full report
  ID, dates and draft/demo/unavailable labels. Full method/fee/lineage fields
  remain in frozen JSON. Calculations and finality gates are unchanged.
- `scripts/review/triage.py` reports Tier 3; `git diff --check` passes.

### Compact dashboard revision

- Presentation only: removed repeated introductions and footer qualifications;
  placed methodology, balance bridges and routine event messages behind Details.
  Report download is near the top; strategy performance no longer repeats the
  same bottom summary table. Investor PDF renderer-v4 is unchanged.
- Current saved-check time, distinct failures, source warnings, missing values
  and partial-history labels remain visible. WARN/error activity opens expanded.
  Data completeness requires both valuation and capital coverage; chart extrema
  are labeled Range, and historical windows use the neutral Trading schedule.
- Seven tabs passed browser QA at 1440/768/390, including client/date isolation,
  closed disclosures, expanded comparison, event disclosure, offline assets and
  authenticated PDF download. All seven desktop pages and mobile Overview were
  visually inspected. Screenshots use synthetic demo data only.
- Parity, failure-modes and coverage reviewers closed their findings. Coverage
  independently passed five regression tests and seven template probes; these
  are not added to the 823 main-suite count.
- Incremental live-impact check: no change to order timing, sizing, reference
  prices, calculation/source contracts, schemas, state/config formats, released
  YAML, logging fields, authorization, process lifecycle or Windows file I/O.
  Tier 3 applies because live operators consume these templates; tests/tooling
  are lower-tier surfaces. No deployment, push or VPS operation occurred.
  Real-client onboarding and separately authorized deployment remain outstanding.

### Visible Pod flow follow-up

- Strategies now shows saved DB, Decision, VPlan, ACK, Fill, Reconcile and EOD
  stages without expanding Details, followed by reconciliation time and next
  action. This is saved evidence, not inferred live progress: recorded fills
  are not relabeled complete, previous-cycle evidence is marked, and missing
  evidence is gray. Optional comparison stays under Details, not in execution.
- Seven client tabs passed browser QA at 1440/768/390 with every demo Pod stage
  visible and no horizontal page overflow. Client directory screenshots were
  also captured; desktop/mobile Strategies and desktop directory were inspected.
- Parity, failure-modes and coverage reviews completed. Coverage independently
  passed six final flow cases after closing the optional-comparison finding.
  Tier 3, with tests/browser tooling also touched. The live-impact checklist is
  unchanged: no order timing, sizing, reference prices, schemas/state/config,
  released YAML, consumed logging, authorization or Windows process/I/O changes.
  No production connection, push or deployment occurred.

## Live-impact checklist and assumptions

- Signal/next-open order timing, sizing (`amount`, `target`, value/percent),
  execution-reference prices and order construction are unchanged. Runner entry
  points now assert optional dashboard authorization before release metadata
  writes, broker-adapter resolution and direct-action plan execution. The context
  is absent for CLI/scheduler calls, preserving their existing selection/CAS and
  execution behavior. Automatic submit selection is not replaced by explicit-ID
  selection, which would bypass the existing auto-submit filter.
- No migration to trading state, pickle, SQLite schema or released YAML occurred.
  The new operator registry is separate; reporting and preview evidence are read
  in read-only transactions. Existing released configs retain their intended route.
- Existing dashboard/log fields remain. Display checks are stricter about proof;
  they do not alter orders or reconcile executions.
- Windows paths, source-read failures and missing/old evidence have explicit
  unknown/error states. Those states must not be interpreted as a broker outage
  or as green simply because a page rendered.
- Supported startup is the documented credential-required CLI. Direct factory
  use without an operator credential intentionally retains legacy non-client
  route compatibility and is **not** covered by the operator-only startup claim.
- The CLI defaults to read-only; `--enable-actions` is explicit and incompatible
  with demo mode. Confirmation state is in-memory and single-process: restarts,
  expiry, cancellation, replay and changed target/state fail closed. This is not
  a distributed scheduler lock or exactly-once broker guarantee. Tick authorizes
  a lifecycle cycle that can create new plans; it is not a fixed-order preview.
  Errors/timeouts consume the approval and require evidence review before retry.
  Copied CLI commands do not inherit UI approval. Even CLI status writes metadata.
- D+1 gating and content hashes do not authenticate arbitrary XML as a finalized
  official statement. Operator-owned paths must contain trusted, reviewed broker
  imports; account ownership, fee basis and economic-field profiles need actual
  onboarding evidence. A hash proves byte identity, not economic authenticity.
- English investor output is implemented and visually checked. RTL, persistent
  server report archives, automatic distribution and newly generated certified
  replays are not silently added completion requirements.

## Approved decisions and review findings closed

1. FINAL may contain verified NAV/capital/P&L and separate official account TWR,
   without consolidated TWR. Incomplete required evidence stays DRAFT; demo stays
   DEMONSTRATION. No calculation or source-authority promotion changed.
2. Local authorization hardening is implemented: exact target/action/account/
   release/state context, one-use approval, execution-bound checks and frozen
   manual inputs. No real operation was used to verify these paths.

Findings fixed include cross-account plan evidence, nonfinite manual limits,
worker/action mismatch, SQLite evidence failures, sequential/concurrent replay,
uncertain manual outcomes, duplicate request keys, historical-release upgrades
and active-ID collisions. Focused regressions cover each boundary.

Remaining work outside this implementation scope: real-client source/ownership/
fee onboarding, separately authorized deployment and post-deployment observation.
No promise is made that a future month-end cannot encounter an external failure.
