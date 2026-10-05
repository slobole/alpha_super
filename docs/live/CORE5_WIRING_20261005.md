# CORE5 wiring and deployment prerequisites — 2026-10-05

CORE5 has a gated production execution route and is classified **WIRED**. The
release template is disabled, automatic submission is off, and its account and
qualification fields are unfilled. No actual release, account allocation, VPS,
broker connection, or order was created by this work on `main`.

WIRED is a code integration result. Physical deployment still requires the
specific account's forward execution, margin, borrowing and operator evidence.
The local tests do not supply that evidence.

## Frozen strategy contract

The investment rules and research parameters are unchanged; zero variants were
searched. See [the complete signal formulas](CORE5_ADAPTER_QUALIFICATION.md#frozen-rules-and-formulas).

- Universe: SPY, IEF, GLD, DBC and UUP, each a fixed 20% sleeve; inactive sleeves
  hold BIL. No relative ranking or redistribution among active risk assets.
- TOTALRETURN closes drive trailing 126-observation drawdown severity ranks,
  squared-rank adaptive smoothing between 50/200-day speeds, and an SMA10 filter.
  A sleeve is long only when SMA10 exceeds the adaptive average.
- DBC alone may short when SMA10 is strictly below the adaptive average:
  `weight = -min(0.10, 0.025 / (sample_std(last 63 daily returns) * sqrt(252)))`.
  The long/BIL book stays at 100%; short proceeds remain cash. Subsequent drift
  can exceed the 10% rebalance target. Equality creates neither a long nor a short.
- Evaluate every exchange session. Rebalance only on initialization, a change
  in any long state, or the actual XNYS month end. Volatility/short-state changes
  alone do not trigger a rebalance. This is not a monthly-only deployment.
- Dedicated `norgate_eod_core5` snapshot for the exact decision date, validated
  manifest/hash and full signal history. CAPITALSPECIAL supplies traded prices;
  TOTALRETURN is confined to signal/benchmark roles.
- `NAV_T = EOD cash + sum(signed held shares * Close_T)`.
  `target_i = trunc(NAV_T * weight_i / Close_T_i)`; `delta_i = target_i - held_i`.
  Truncation is toward zero for shorts. No later quote or NAV resizes these shares.
- Research still charges 2.5 bps slippage, $0.005/share with $1/order minimum,
  and DBC borrow `abs(shares) * ceil(1.02 * close) * 0.01 * calendar_days / 360`.
  The operational adapter uses actual account cash, without charging this
  synthetic borrow fee again. Real fee, dividend and financing timing can differ.

```text
[Exact EOD data + strict dedicated account snapshot at Close_T]
                             |
[Existing signals / daily trigger / frozen whole-share target]
                             |
       *** CRITICAL *** no T+1 price can change quantities
                             |
[VPlan + current account + margin previews + DBC availability]
                             |
[Before next open minus 2 minutes: atomic submission claim]
                             |
[MKT/OPG requests -> per-request fills -> signed reconciliation]
                             |
[Atomic strategy-state commit; no-order days commit directly]
```

## What changed

The existing CORE5 adapter, data profile, frozen sizing, two-leg DBC reversals
and reconciliation were reused. The old unconditional rejection of `mode=live`
was replaced only after implementing the following checks:

1. **Strict account input:** exact routed account, explicit unambiguous USD
   cash/NAV, completed position download, whole USD stock positions from the
   six tradables, only DBC negative. Missing cash cannot become zero. All-client
   open orders are inspected at EOD, VPlan construction, submission and reconcile.
2. **Dedicated account:** no enabled second pod may share CORE5's account in
   the same mode, regardless of manifest order. Budget fraction must be 1.0.
3. **Current funding:** require standard margin (`STKNOPT`/`STKMRGN`) and available
   funds/excess liquidity. Preview every order leg using `whatIfOrder`, including
   reducing sells. Portfolio margin (`PMRGN`/`GPMRGN`) is rejected: independent
   previews cannot bound joint stress/hedge interactions. A reversal's second
   preview uses cumulative same-symbol shares against the actual broker position.
   Actual transmitted legs remain separate and unchanged.
4. **Conservative capacity:** sum positive initial/maintenance margin changes;
   give no credit for pending sells; add the sum of positive preview equity
   debits (including previewed costs) to both requirements. Each must fit the lower of the account's
   starting and final corresponding capacity. Recheck readiness, margin type,
   positions and all-client orders after previews.
5. **Incremental short availability:** when the final DBC short increases,
   current `shortableShares` must cover that increase. Missing/invalid data or
   insufficient shares blocks the entire plan; no silent long-only fallback.
   Only the temporary market-data subscription is cancelled.
6. **Timing:** normal scheduled submission stays 6m30s before next open.
   CORE5 blocks at or after open minus two minutes, both before and after
   preflight, and immediately before each socket dispatch after connection and
   contract qualification. The local deadline leaves broker MKT/OPG unchanged.
   A mid-batch timeout retains the submission claim for reconciliation of any
   earlier legs. This conservative cutoff includes the Nasdaq 09:28 MOO deadline
   described by [IBKR](https://www.interactivebrokers.com/en/trading/ordertypes.php?menu=B).
7. **Restart/concurrency:** a failing/late preflight may block only an untouched
   ready plan. It cannot overwrite another process's submission claim. After a
   crash following the claim, scheduler and expiry paths retain the pending cycle
   and route it to reconciliation.
8. **Honest reference status:** generic automatic return comparison refuses
   CORE5 with `core5_accounting_bridge_required`; that generic reference removes
   dividends but retains synthetic borrow. Conditional decision/order comparison
   remains available through the existing saved-price oracle.

## Deployment record and physical prerequisites

Use the [disabled template](release_templates/pod_taa_adaptive_macro_core5_daily_moo.yaml.example).
An enabled live release requires `params.core5_live_qualification_dict` containing:

- The exact `release_id_str` and `account_route_str` being enabled.
- `evidence_reference_str` pointing to the operator-reviewed evidence record.
- Timezone-aware `approved_at_str` and `expires_at_str`; approval must be active
  before new decisions/submissions. Qualification is rechecked before submission.
  Expired/revoked qualification does not prevent loading the release or reconciling
  orders that were already sent.
- Explicit true values for `margin_account_confirmed_bool`,
  `dbc_borrow_and_recall_policy_confirmed_bool`, `forward_execution_qualified_bool`
  and `operator_approved_bool`. Strings and numeric substitutes are rejected.

These are operator attestations, not automatically verified evidence contents.
Do not populate them from the synthetic tests. The evidence should cover the
intended account/route, natural forward cycles, DBC entry/cover/reversal and
reject/reconnect behavior, observed costs, actual borrow-rate/recall procedures
and the approved deployment. No account has been qualified by this change.

Previews cannot reserve borrow or margin, guarantee basket margin or auction
acceptance, or prevent conditions changing after the check. Portfolio-margin
accounts remain unsupported; real fills require account qualification. Borrow rate is
reported as unknown by the preflight; no new economic rate cap is invented.
Existing shorts can be recalled between cycles. Broker statement reconciliation
and the operator's recall procedure remain necessary (G-030).

Incubation uses a virtual `SIM_` account and its existing simulated ledger;
PAPER uses a distinct paper account and the physical broker preflight. Neither
incubation P&L nor PAPER fills prove physical account return/fill parity.

## Missed session or interrupted-cycle recovery

A blocked/expired or missed decision session deliberately parks CORE5. Preserve
the database, snapshot hashes and order/fill identities. Inspect the pending
DecisionPlan/VPlan and actual account orders/fills/positions before changing any
state. A `submitting` cycle must reach broker reconciliation, including the case
where the process died before saving an ACK. Never reset the database, silently
skip daily memory, replay an old MOO after its window, or initialize over holdings.

If no orders were sent, a separately reviewed recovery must restore continuous
daily strategy memory from the saved snapshots and unchanged account evidence.
If execution was partial or uncertain, resolve the orders and reconcile first.
This work adds no automatic recovery trades or discretionary replacement order.

## Verification and limits

Tier 3, with shared execution/state and quant-sensitive input dependencies.
Independent reviews covered parity, quant pitfalls, failure modes and coverage.
The final parity/quant and coverage reviews found no remaining actionable
findings after the funding, recovery and dispatch-cutoff fixes. Reviews do not
replace the tests below; these results are local code evidence, not broker approval.

- Baseline CORE5 data/strategy/adapter/invariance: 128 passed before changes.
- New strict broker transport suite: 88 passed using fake IBKR transports.
- New qualified-live/paper lifecycle, gates, cutoff and restart suite: 54 passed.
- Shared-route regression: 302 passed, comprising 256 adjacent MR execution,
  account-isolation, scheduler, concurrency and recovery tests plus 46 release
  manifest tests. [JUnit result](../../results/research/strategy/strategy_taa_adaptive_macro_core5/wiring_qualification/2026-10-05_shared_regression.xml).
- Fresh capital, benchmark and determinism checks passed against the saved
  2026-09-11 snapshot: terminal-capital ratio 2.013 when initial capital doubles;
  benchmark CAGR matches `$SPXTR`; identical reruns match exactly.
  [Saved result](../../results/research/pm_readiness/2026-10-05_162506/pm_readiness.json).
- Final-source saved-price comparison passed 51 decisions: 5 rebalances,
  46 no-order days and 3 month ends. All recorded source hashes matched the
  final files. This sample contains no DBC reversals; both reversal directions
  are covered by the synthetic lifecycle/socket tests above.
  [Parity result](../../results/research/strategy/strategy_taa_adaptive_macro_core5/wiring_qualification/2026-10-05_adapter_parity_final/qualification.json).
  The final sample overlaps an earlier 174-decision check and is not 51 new
  independent observations. Saved-price reconstruction is not historical
  decision-time replay, forward evidence or a new alpha validation.
- Broad consolidated regression: **892 passed**, including the new suites,
  strategy/snapshot/invariance checks, registry, incubation, socket, runner,
  scheduler, order clerk, reconciliation, SQLite state, dashboard, strategy host,
  reference comparison and BENCH. One unrelated Jupyter deprecation warning.
  [Final JUnit result](../../results/research/strategy/strategy_taa_adaptive_macro_core5/wiring_qualification/2026-10-05_regression_verified.xml).
  Together with the adjacent MR suite, **1,148 distinct test cases passed**
  (the 46 release-manifest tests overlap). Earlier failures were corrected
  expectations for the added local submission deadline and disabled live
  template; the final consolidated run has no failures.
- `git diff --check` passed. The working branch remains `main`; no commit,
  release activation, broker/VPS call or production deployment was performed.

The quant review preserved the existing decision boundary, universe, price
adjustments, signal history and costs. No model fit, selection, new strategy
variant or parameter search was introduced. These engineering checks make no
new performance, sample-size, regime-robustness or multiple-comparison claim.

| Live-impact check | Result |
|---|---|
| Timing | Close_T to next-open MOO retained; added earlier fail-closed deadline. |
| Sizing | Same signed whole shares, target-vs-delta semantics and separate reversal legs. |
| References | CAPITALSPECIAL Close_T retained; current quotes cannot resize. |
| State/config | No schema/pickle migration; additive qualification params and atomic status method. |
| Logs/dashboard | Existing fields retained; funding evidence and explicit block reasons added. |
| Windows/restart | Temporary SQLite tests cover competing claims and reopen-after-claim. |
| Releases | Disabled example changed; no actual runtime YAML/account/service enabled or edited. |

Unrelated uncommitted MR capsule work on `main` was preserved. No strategy rule,
backtest formula, cost assumption, research allocation or production account was
changed.
