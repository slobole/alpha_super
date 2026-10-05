# MR capsule WIRED — independent review (Claude, 2026-10-05)

Subject: Codex's uncommitted WIRED implementation of the six capsule identities
`strategies.mr_capsule.strategy_mr_{dv2_vix_gated,hpi_vote_vix_gated}_{cash,bil,spmo}`,
recorded in [MR_CAPSULE_WIRING_REVIEW_20261005.md](MR_CAPSULE_WIRING_REVIEW_20261005.md)
(49 owned paths in `results/research/mr_capsule_review_20261005/owned_paths.json`).
Base: `main` at 5c0d48d. Read-only review; no repository code was changed by it.

Risk tier: **Tier 3** (live execution, sizing, state store, release validation), with Tier 2
(engine hook, snapshot store/exporter) and Tier 1 (capsule strategy modules) surfaces.
Reviewers: live failure-modes + existing-pod regression, quant pitfalls + live/backtest parity,
test coverage + commit hygiene (three independent read-only agents), plus the checks below.

## Verdict

| Question | Answer |
|---|---|
| Existing live pods (NDX, TAA, DV2, HPI, CORE5) unchanged? | **Yes.** Every new branch is gated by the capsule import prefix, the capsule data profiles or `sizing_contract_str == mr_capsule_close_targets_v1`. The SQLite `ADD COLUMN` migration was probed forward, rolled back to HEAD code and forward again. All live modules import cleanly. |
| Live decision = research rules? | **Yes, with bounded exceptions:** the inherited HPI open marker (G-033), stock shares from the pre-submit quote instead of Close_T, small broker-vs-engine NAV differences, and whole-cycle skips when a guard trips. No parity P1. |
| Research results unchanged by the final code? | **Yes.** Fresh engine reruns of DV2/BIL and HPI/BIL on the final code match the 2026-10-04 build to the cent: maximum NAV difference below 1e-9, and all 10,638 / 6,915 transactions identical. |
| Ready to commit to `main`? | **Not as is.** Two repository tests fail (B1), and the commit must stage the untracked files explicitly (B2). |
| Ready for PAPER? | **No.** Fix items P1–P6 first. |

## Evidence

| Check | Result |
|---|---|
| Full suite (7,146 collected) | 7,105 passed, 34 skipped, **7 failed**: 5 pre-existing (also fail on clean HEAD 5c0d48d: two `test_client_operations`, two `test_scout_gate` HPI saved-run gates, the PTA Good Friday test), and **2 caused by this change** (B1). |
| New/changed capsule test files | 389 passed, run file by file. |
| Mutation probes | Caught: capsule sizing switched to broker NAV × budget; pre-submit revalidation removed; abandonment ignoring claim/ACK state; engine hook default flipped. **Not caught:** migration `ALTER` removed; the four `BEGIN IMMEDIATE` statements removed; same-session rebuild guard removed. |
| Final-code engine reruns | DV2/BIL $2,941,577.27 and HPI/BIL $1,403,311.66, identical to the build. |
| PM readiness, `_cash` entry points | Run separately from this review; results are recorded in the review summary. |

## Before commit (blocking)

- **B1. Missing analysis hooks for WIRED modules.** Adding the capsule identities to
  `SUPPORTED_STRATEGY_IMPORT_TUPLE` makes `tests/test_run_capacity_analysis.py::test_deployment_wired_strategy_modules_expose_capacity_builders`
  and `tests/test_run_strategy_analysis.py::test_wired_strategy_modules_expose_all_analysis_hooks` fail. All six
  modules lack `build_capacity_analysis_inputs` and `build_execution_timing_analysis_inputs`. These are the
  capacity/timing hooks the original handoff deferred. Both tests pass on clean HEAD.
- **B2. Stage the untracked files.** `alpha/live/release_manifest.py` and `alpha/live/strategy_host.py` import
  `alpha.live.mr_capsule_adapter` at module level. A commit without the new untracked files would make every
  pod on the VPS, NDX and TAA included, fail at import. Stage exactly the 49 owned paths. Keep the unrelated
  working-tree changes (knowledge base, `portfolios/fund_*.yaml`, other research folders, `output/`, `work/`) out of this commit.

## Before PAPER

- **P1. Cash guard trips on routine postings.** `validate_mr_capsule_execution_contract`
  (`alpha/live/execution_engine.py`, the ±$0.01 cash equality) runs at VPlan build and again pre-submit.
  - **Trigger.** IBKR posts overnight: BIL's monthly distribution, stock dividends on pay dates, monthly interest and fees. Estimated 20–30 sessions a year. Each trip abandons the whole auction (entries, exits, BIL funding).
  - **Impact.** Roughly 5–10 lost auctions a year, concentrated in gate-open stress periods, which is where the edge is.
  - **Why the guard is too tight.** No order quantity depends on submit-time cash. Targets are frozen at Close_T NAV, positions must match exactly, and open orders already block.
  - **Fix.** Use an asymmetric NAV-relative tolerance: accept credits up to e.g. 2% of NAV and debits up to 0.25% of NAV (inside the 1% cash buffer). Log the drift. Keep the exact-position and no-open-order checks.
- **P2. Unresolved-cycle lockout without a resolve path.** New decisions wait for the prior cycle (`build_decision_plans`, `mr_capsule_prior_cycle_unresolved`). Completion requires full signed fills for every request (`post_execution_reconcile`).
  - **Triggers.** Any of these freezes the pod indefinitely, with no exits and no BIL sales:
    - a rejected, partial or unfilled MOO (halt at the open, late submission, margin rejection);
    - a crash between claim and transmission;
    - fills not captured after an outage (`reqExecutions` covers the current session only);
    - a foreign open order.
  - **Recovery today.** The runner has no resolve command; the only way out is a manual SQLite edit.
  - **Fix.**
    - (a) Resolve automatically from the final broker order status, matching research semantics: an unfilled entry is cancelled; an unfilled exit becomes a pending exit carried to the next decision (G-033).
    - (b) Add an audited operator resolve command: refresh broker evidence, write an operator reconciliation, and set a terminal flag honoured by the prior-cycle check.
    - (c) Alert when a cycle stays unresolved for more than one session.
- **P3. A blocked decision also blocks exits.** `build_mr_capsule_decision_plan` raises before any intent exists in several cases:
  - a held name with no exact-session close (OTC delisting, long halt; the HPI profile is unpadded);
  - a spin-off into a non-index symbol, or fractional spin-off shares;
  - any new VIX gap.

  **Fix.** Add an exit-only degraded mode (exits and BIL sales for priceable names, no entries) or an operator position exclusion, plus a runbook entry.
- **P4. Broker client id.** All six templates use `client_id_int: 31`. Two capsule pods on one gateway would collide at the pre-submit read. Give each template its own id.
- **P5. Margin is attested, not checked.** `margin_account_confirmed_bool` is a YAML flag. Add a pre-submit check that AvailableFunds covers net buys, because the same auction's BIL sale funds the stock buys.
- **P6. Tests the mutation probes showed missing:**
  - **Migration pin.** Reopen an old-schema DB and assert `target_share_json_str` appears in `PRAGMA table_info`. Insert and read back a capsule plan `{"BIL": 400}` and a DV2 plan. The current test passes even without the `ALTER`.
  - **Real race test.** Use a two-connection file DB. One store claims inside an open `BEGIN IMMEDIATE`; the other's abandonment must block or fail and never overwrite `submitting`. As defence in depth, add `AND status_str = 'ready'` to the abandonment `UPDATE` and check the rowcount.
  - **Same-session no-retry guard.** After abandonment or completion for T, `build_decision_plans` before the T+1 close must skip.
  - **Adapter fail-closed guards, without mocked snapshot metadata.** Cover:
    - fractional or negative positions;
    - a foreign identity or foreign positions;
    - the wrong parking ETF for the mode;
    - a snapshot date, profile or manifest mismatch;
    - a missing T-1 row;
    - non-positive NAV.
  - **Cash entry points.** Add the two `_cash` entry points to `ENTRY_POINT_LIST` in `tests/test_strategy_mr_capsule_pods.py`.
  - **Uncaught ValueError.** Catch the `ValueError` that `abandon_unsubmitted_mr_capsule_cycle` raises on a reused pod id that still holds an old non-capsule decision (`expire_stale_decision_plans`).

## Notes

- **Docs to update.**
  - `docs/live/release_templates/README.md` (Wired table) and `docs/strategies/index.md` (WIRED count, caveats).
  - Runbook: dry-run-validate a capsule YAML before copying it into the live releases root, because a failing YAML refuses the whole root, NDX/TAA included.
  - Runbook: restart the `serve` processes after the pull.
  - Check class-share tickers (e.g. BRK.B) for IBKR qualification in PAPER.
- **Qualification.** Include Unadjusted Close in future `qualify_wiring` comparisons. DV2 drops symbols with missing values there, so it is a decision input. It matches today.
- **Parity holds per snapshot.** PAPER comparisons must replay the stored decision-day snapshot by manifest hash. A later research rerun re-rounds float32 history after corporate actions and can flip borderline threshold decisions.
- **Owner decision: LIVE gating.** The adapter accepts an enabled `mode: live` with only the margin flag. The CORE5 precedent rejects physical LIVE pending forward qualification. Pin the chosen policy with a test.
