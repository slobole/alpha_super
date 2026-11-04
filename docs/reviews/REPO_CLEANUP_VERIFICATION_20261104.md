# Repository cleanup: verification repairs, 2026-11-04

Status: verification repairs complete; merged into local main with owner approval.
Base: `08f1c616203f44a5da877a3c8b71fc7e0d1ece31`.
Branch: `codex/cleanup-verification`.

## Scope and preservation

This implements the approved synchronization and verification/test cleanup phase.
It does not redesign MOC, add a virtual ledger, retire Dashboard V3, change
strategies, enable releases, deploy, or connect to a VPS/broker.

Local main was fast-forwarded from `fc0870b` to `08f1c61`. All 25 pre-existing
tracked changes and 1,353 non-ignored untracked files were preserved. Only two
files overlapped incoming work: `tests/test_bench.py` and
`docs/strategies/book-strategy-caveats.md`; both sets of additions were retained.
An external byte/hash manifest and a retained recovery stash protect that work.
The final preservation check confirmed all 25 tracked states (including one
existing deletion) and all 1,353 untracked files, with no missing or altered work.
Recovery folder: `C:\Users\User\AppData\Local\Temp\alpha-cleanup-preserve-20261104-qlufwj13`;
retained stash: `700b0cb18dbd182f36eb20aaa2fb5dabf508ee0c`.
The owner subsequently authorized the local merge. Main was fast-forwarded to
`413599d`; no push or deployment occurred. All 25 pre-existing tracked states
and 1,353 untracked files were preserved again. The sole overlap,
`tests/test_bench.py`, merged cleanly; its two parametrized owner cases remain
uncommitted with the other pre-existing work.

Post-merge verification in a disposable clone, including the tracked local
working changes: **134 Bench/offline-guard tests passed**. Independent source
review confirmed the combined Bench file is exactly the cleanup version plus
the owner's original test addition. This focused check does not qualify all
uncommitted research. Merge recovery/evidence directory:
`C:\Users\User\AppData\Local\Temp\alpha-cleanup-merge-20261104-l4j0ua82`.

## Repairs

- Triage includes staged, unstaged and untracked paths, preserves both sides of
  renames, and handles NUL-delimited Unicode filenames. Operational scripts,
  dashboard tests, registry/build files and portfolio definitions receive
  explicit review tiers. Unknown paths require manual assessment (exit 2),
  rather than silently receiving Tier 0.
- Two frozen YAML specifications and the Bash proof wrapper have targeted LF
  checkout rules. The specifications' approved Git blob hashes are unchanged.
- The proof wrapper stops on failed commands and preserves diagnostics. Nested
  snapshot children require a successful exit and one valid result; errors are
  saved before the parent exits nonzero. Empty/different NDX/TAA row comparisons
  fail. Existing accepted ticker-name differences and weight policy are unchanged.
- Fifteen local state/status tests inject a missing vendor heartbeat at the
  reader boundary; the real missing-data gate remains active. Five portfolio
  construction tests inject synthetic benchmark prices while validating the
  `$SPXTR` / `TOTALRETURN` request outside production exception handling.
- Bench's default fixture has empty private results. Evidence cases seed actual
  artifact-reader inputs. A numeric Sharpe assertion now reads the complete row.
- Synthetic client operations use a fixed clock; mode-severity checks use fixed
  disk health. Dashboard notifications/journals and cross-process lock tests use
  private paths. Measured-Pod tests explicitly isolate their intended scope while
  preserving source rows and full-client ownership integration tests.
- Doctor refusal tests inject a connection-refused error at the HTTP boundary,
  then verify the actual health URL and timeout outside production catch blocks.
  The MTUM bridge bounds test uses synthetic execution/signal frames and still
  verifies both configured signal sources and the exact available end date.
- The PTA causality fixture now uses real XNYS sessions: March 28, 2024 was the
  month-end close; March 29 was Good Friday. Changing only the final March price
  must change March's return without changing March's prior-month PTA. A mutation
  experiment confirmed the test fails if the production one-month lag is removed.
- Twelve additional genuine Norgate tests are individually marked for the data
  lane: two control-book, three Tactical FI and seven ALFRED/PIT checks. Their
  pinned hashes, tolerances and assertions are preserved. The forward-snapshot
  test instead uses the existing governed local-history loader and stays offline.
- Four V4 overview goldens reflect the previously approved `a4e810c` labels.
  Reversing only those labels/source prefixes recovered every old hash. No
  numeric, date, account-scope or other HTML differences were accepted.
- Default pytest blocks network access from configuration onward, including
  imports and early fixtures. A test can use its own loopback listener.
  Real-data/saved-artifact checks require `--run-data-integration`; skips remain
  missing evidence. See [test lanes](../../tests/README.md).

No tests were deleted merely because they were old or failing. No production
`alpha/`, strategy, data, portfolio or live-release files changed. The HPI handoff
only gains the current explicit test-lane flag; its historical evidence is intact.

## Verification evidence

The test candidate contains committed `08f1c61` plus this cleanup package.
Preserved uncommitted research work in main is outside this candidate and is not
qualified by these green results.

The first complete run took about 51 minutes and reported **10,303 passed,
13 failed, 4 errors, 76 skipped**, plus 73 passing subtests. Every failure/error
was investigated: five were corrected local test fixtures; twelve were real
vendor-data checks moved to the explicit data lane, not counted as passes.

All nine affected test modules were rerun: **105 passed, 12 skipped**, zero
failures/errors. The final triage/parity-lane delta was also rerun: **37 passed,
12 skipped**. Combining results by unique test identity gives **10,309 passed,
88 skipped, no unresolved failure/error**; the extra pass is a new triage case.
This is full-run plus bounded-retest evidence, not a second complete full-suite
run. Production code and the shared test guard did not change after the full
run started. The 73 passing subtests are additional and not added to that count.

The intentionally broken PTA lag experiment failed as expected; its disposable
source was restored byte-for-byte. That expected failure is not included in the
candidate's results. No failing test was deleted or assigned an expected-failure
marker to make the suite green.

Local evidence directory:
`C:\Users\User\AppData\Local\Temp\alpha-cleanup-verify-20261104-yszsrsd0`.
It contains `logs/full.xml`, `logs/final_delta.xml`, `logs/post_full.xml`, the
per-case deduplication in `final_case_evidence.json`, the mutation experiment,
source hashes and preservation checks. The initial run's warnings remain in its
log; this cleanup does not claim a warning-free suite.

Completed focused checks (overlapping counts, not additive):

- 710 pinned/new LIVE and strategy contract tests passed.
- 303 Bench/portfolio/dashboard/runner/scheduler tests passed.
- 98 accounting, client-operation and ownership-scope tests passed.
- 230 final tooling, guard, proof, portfolio, Bench and golden checks passed.
- 84 synthetic Scout/data-lane checks passed; 54 real-data cases skipped.
- 277 JavaScript tests passed across nine files.
- Fresh `core.autocrlf=true` checkout retained exact bytes of both frozen YAMLs
  and the LF shell wrapper; Bash syntax passed.
- Original main preservation hashes passed; whitespace checks passed.

Frozen YAML SHA-256 values:

- `clean_sheet_portfolio_foundry_v4.yaml`:
  `b99434ba6d03cad2ec2c99137aa45d6b1ff538c27c379d4c5d0245bd64ac7fab`
- `ladder4_candidate_value_add_h0_h8.yaml`:
  `9d3613a92535a9ed5510c46d71ae5316553dcb1d20d6a9ca4235fdba7211bf0c`

## Independent review and LIVE impact

Tier 3, with Tier 2 shared verification and Tier 1 research-proof surfaces.
The final helper also requests manual classification for four newly touched
paths. That assessment is complete: the two Norgate doctor test files are Tier 3
operator surfaces; the ALFRED/PIT and forward-snapshot test files are Tier 1
research surfaces. The package retains Tier 3 and all required review roles.
Unknown-path refusal was retained; no generic allowlist was added to silence it.

Three independent reviewers covered quant pitfalls, parity, failure modes and
coverage. Findings fixed: early-network guard gap, swallowed nested child
failures, overly broad text-file classification, and PM assertions inside a
production exception boundary. Final fixture/marker changes were reviewed again; no material review finding
remains open. Reviews are source evidence; test execution is recorded separately.

| Contract | Result |
| --- | --- |
| Order timing and next-open semantics | Unchanged |
| Sizing, targets, percentages and capital allocation | Unchanged |
| Reference price sources and adjustment semantics | Unchanged |
| State, SQLite, pickle and release config formats | Unchanged |
| Fields consumed by dashboard/logging | Unchanged |
| Windows runtime locks, restarts and idempotency | Unchanged; test paths isolated |
| Enabled releases and intended order routes | Unchanged |

Accounting remains `P_D = E_D - B_D - F_D`, `r_D = P_D / B_D`, with geometrically
linked TWR. D+1, baseline dates, missing-day refusal and retirement assertions
remain. No lookahead, PIT universe, train/test, parameter search, corporate-action,
cost, sample-size or economic-performance contract is changed by this phase.

## Limits and next cleanup phase

The socket guard is not an OS sandbox and does not constrain arbitrary child
executables. Broad verification ran in an independent disposable checkout with
an additional write/network guard, rather than against the main Scout ledger.
Real vendor-data parity, broker execution and deployed VPS health were not
qualified. Proof command completion alone does not certify all research parity;
matching refusal rows and weight/name diagnostics still require interpretation.

V4 still imports active V3 action authorization, local-workspace and diagnostic
helpers; deleting V3 requires a separate dependency change.

Dashboard V3 retirement, dead-code deletion, broad file moves, G-037 document
reconciliation and separate disabled-feature input-validation findings remain
outside this package. Any test deletion must name the retired behavior and prove
that no remaining consumer or unique safety assertion depends on it.
