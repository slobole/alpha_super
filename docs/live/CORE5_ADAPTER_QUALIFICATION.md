# Adaptive Macro CORE5: local execution adapter qualification

## Verdict and scope

Step 3 wires CORE5 into the existing local decision, execution-plan and state
lifecycle. It supports incubation and PAPER qualification. Physical `mode=live`
is deliberately rejected until forward execution and account borrow/margin
qualification are complete. This work creates no active release, account,
capital allocation, VPS service or broker orders.

The approved strategy and DBC short are retained. This is an integration test,
with **zero new strategy variants** and no new profitability claim. Data
qualification is recorded separately in [Step 2](CORE5_DATA_QUALIFICATION.md).

## What happens during a day

```text
[Exchange session T closes]
              |
              v
[Fresh EOD account state + exact CORE5 snapshot for T]
              |
              v
[Existing signals; initialization / long-state flip / month-end?]
       | no                                  | yes
       v                                     v
[Keep actual shares]                 [Freeze target shares at Close_T]
       |                                     |
[Commit daily memory]                        v
                                 [VPlan: check unchanged holdings]
                                             |
                        *** CRITICAL *** no T+1 price resizing
                                             |
                                 [MOO at next exchange open]
                                             |
                                 [Every request filled + no open orders]
                                             |
                                 [Reconcile positions; atomic state commit]
```

The scheduler first captures the EOD account state, then waits for the exact
T snapshot. The existing readiness buffer is 10 minutes after the exchange
close; actual data publication can take longer. Holidays and early closes use
XNYS. Scheduled MOO submission remains 6 minutes 30 seconds before the next
exchange open, normally 09:23:30 New York for a 09:30 open. Auto-submit remains
disabled in the example template; merely preparing it does not submit orders.

Initialization is itself a rebalance trigger. A clean new account can therefore
enter positions at the first eligible next-session open after its initial
decision. It does not have to wait for month-end or a future signal change.
No valid EOD state or snapshot means no decision. An unresolved prior cycle
prevents a new one; missed execution windows are not replayed automatically.

## Frozen rules and formulas

Five fixed 20% sleeves: SPY, IEF, GLD, DBC and UUP. Each inactive sleeve goes
to BIL. There is no ranking or redistribution to the active risk assets.

For each asset, use its total-return close P through T, with all available
history from the fixed 1990-01-01 request and its actual inception:

```text
H_T = maximum(P_s for s <= T)
severity_T = 1 - P_T / H_T
rank_T = (count(severity < severity_T) + (count(equal) + 1)/2) / 126
          [inclusive trailing 126 observations; equal count includes T]
a_T = rank_T^2 * 2/51 + (1 - rank_T^2) * 2/201
AMA_T = a_T * P_T + (1-a_T) * AMA_(T-1)
        [seed AMA with P at the first valid rank; never reset at trading start]
SMA10_T = mean of the latest 10 closes through T
long_state_T = 1 if SMA10_T > AMA_T, otherwise 0
vol63_T = sample_std(P_t/P_(t-1)-1 over 63 returns through T) * sqrt(252)
DBC short target = -min(0.10, 0.025 / vol63_T)
                   only if SMA10_T < AMA_T and vol63_T is finite and positive
```

The long/BIL book targets 100% of NAV. DBC short proceeds remain cash and do
not enlarge the long sleeves. Maximum short target is 10% at a rebalance;
subsequent market movement can increase actual exposure. No continuous 10%
exposure clamp is added.

A rebalance happens on initialization, a change in any of the five long
states versus the preceding exchange session, or the actual last exchange
session of the month. A short-state-only or volatility-only change does not
trigger a rebalance. Equality means neither a fresh long nor a fresh short;
an existing short can remain until another rebalance trigger. The operational
adapter corrects the research endpoint artifact: the final available price row
is not automatically month-end. The historical strategy itself is unchanged.

Sizing uses CAPITALSPECIAL closes and signed whole-share positions:

```text
NAV_Close_T = observed EOD cash + sum(held_shares_i * Close_T_i)
target_shares_i = int(NAV_Close_T * target_weight_i / Close_T_i)
order_delta_i = target_shares_i - actual_shares_i
```

`int` truncates toward zero, including shorts. Current quotes and subsequent
account NAV do not resize these frozen shares. Quotes remain visible as
execution-notional estimates. DBC direction changes preserve separate
close-old and open-new requests, each with its own identity and fills.
Orders are submitted in that order within DBC; actual auction fills need not
arrive in that order and must be verified in broker qualification.

## Account state and failure handling

- One dedicated account route, with `pod_budget_fraction_float=1.0`: all account
  equity belongs to CORE5. A shared account or a fractional account budget is
  rejected. The $100,000 template bootstrap is a virtual diagnostic amount,
  not an approved real allocation and not an EOD-state substitute.
- Require finite whole shares of the six tradables; only DBC may be negative.
  A new strategy state over unexplained existing holdings is rejected.
- Daily memory stores version 1, last committed signal date, last five long
  states, last target weights and last rebalance date in existing JSON fields.
- Recompute only prices through T. Compare the prior session's long states to
  committed memory; a changed historical signal or a skipped decision session
  requires review. This comparison is not a complete historical-vintage audit.
- Frozen metadata retains T's snapshot hash, source profile, EOD cash/time,
  close prices, NAV, target shares, prior memory and trigger flags.
- Check holdings again before VPlan construction and submission. Changed
  holdings or outstanding orders block submission. No automatic resize/retry.
- A no-order day commits memory without a VPlan or broker call. A restart
  recovers a pending no-order commit, even after the submission cutoff.
- An executed cycle requires matching final positions, signed fills for every
  request and an empty open-order list. A matching final DBC net position alone
  cannot prove both legs executed. Partial/rejected/ambiguous cycles retain
  prior strategy memory. Unknown multi-leg order identity requires resolution.
- Strategy memory and completed statuses commit in one SQLite transaction.
  Observed account cash, holdings and timestamps retain their source identity.
  Existing schema and generic fill-reader result shape remain compatible.

## Costs and what the test can prove

Research assumptions remain 2.5 basis points slippage, $0.005/share commission
with a $1 minimum per order, and the fixed 1% annual DBC borrow baseline:

```text
research borrow debit = abs(DBC shares) * ceil(1.02 * DBC close)
                       * 0.01 * calendar_days_to_next_session / 360
```

The adapter uses reported account cash. It does not deduct a second synthetic
research borrow charge. Actual broker dividends, commissions and borrow can
therefore differ from the research model. The present incubation ledger does
not model all these cash flows; its P&L must not be called account-return parity.

The local oracle runs the actual historical engine and captures its cash and
positions before each next-day dividend/fill/borrow cycle. The adapter separately
computes features from each raw prefix through T. Compare triggers, weights,
integer targets and ordered legs within each asset, then verify engine fills
and commit adapter memory. Repeating with quotes/NAV at 80% and 120% of the
reference verifies that later prices cannot change frozen quantities.

This proves conditional decision/order parity when both paths receive the same
account cash and holdings. It does not prove real broker cash, fill, fee, borrow,
recall, margin or short-sale acceptance. Current-vintage saved prices are not
historical decision-time snapshots. The fixed ETF universe is literal to this
strategy; no new point-in-time stock selection, fitting or parameter search was
introduced. No regime robustness, sample-size or alpha claim follows from an
operational parity pass.

## Local evidence and reproduction

**Recorded on 2026-09-15: passed.** The final regression run passed **351 tests**
in 111.43 seconds. The only warning was an unrelated Jupyter path deprecation.
`git diff --check` passed; mandatory triage returned Tier 3. Three read-only
reviewers covered parity, failure modes, coverage and quant pitfalls, with no
remaining actionable finding after the fixes.

| Saved-data reconstruction | Decisions | Rebalances | No-order days |
|---|---:|---:|---:|
| Original full window: decisions 2025-12-31 through 2026-09-10; executions 2026-01-02 through 2026-09-11 | 174 | 22 | 152 |
| Strengthened checker, subset: decisions 2026-06-30 through 2026-09-10; executions 2026-07-01 through 2026-09-11 | 51 | 5 | 46 |

Both runs passed. The second is a subset, not 51 additional independent days.
It adds explicit execution-time/order-flag assertions and calendar/model hashes
after review. The first evidence file retains its original checker hashes.
Neither real-data window held a DBC short or included a DBC sign flip; both
flip directions, borrow and rejected/partial legs are covered by the synthetic
engine and simulated-broker tests, not by real-broker evidence.

Evidence: [original reconstruction](../../results/research/strategy/strategy_taa_adaptive_macro_core5/adapter_qualification/2026-09-15_step3/qualification.json),
[strengthened reconstruction](../../results/research/strategy/strategy_taa_adaptive_macro_core5/adapter_qualification/2026-09-15_step3_final/qualification.json)
and [final verification](../../results/research/strategy/strategy_taa_adaptive_macro_core5/adapter_qualification/2026-09-15_step3_final/verification.json).
Final checker source hashes were reverified after execution.

The automated tests cover a synthetic multi-day engine path, dividends, borrow,
gapped opens, both DBC direction changes, state rollback/restart, no-order days,
late/mismatched data, source clock latency, partial/rejected fills, sparse order
identity and dashboard totals. Existing runner, scheduler, state, manifest,
strategy-host, broker-clerk, reconciliation and dashboard regressions also run.

Run an isolated comparison on the saved Step 2 price file:

```powershell
.\.venv\Scripts\python.exe -B scripts/review/verify_core5_adapter_parity.py --prices results/research/strategy/strategy_taa_adaptive_macro_core5/snapshot_data_qualification/2026-09-15_step2_final/snapshot_prices.parquet --start-date 2026-01-02 --output-dir results/research/strategy/strategy_taa_adaptive_macro_core5/adapter_qualification/new_run
```

The output directory must be new. The script records source/input hashes,
daily decisions, counts and failure/pass status. It reads no broker or VPS.

## Required live-impact review

Tier 3; lower-tier shared-data/strategy loading from Step 2 is also present.
Read-only reviewers cover parity, failure modes, coverage and quant pitfalls.

| Surface | Result |
|---|---|
| Timing | New CORE5 path preserves Close_T to next-open MOO; EOD prerequisite added only for CORE5. |
| Sizing | CORE5 intentionally bypasses generic current-quote floor sizing to reproduce research close-based truncation. Other strategies retain their sizing. |
| References | CAPITALSPECIAL Close_T is frozen sizing evidence; current quote is an execution estimate. No silent open/close substitution. |
| State/config | Existing JSON and SQLite schema; additive CORE5 fields; disabled example only. `live` remains blocked. |
| Logs/dashboard | Existing fields retained; two-leg DBC totals corrected to aggregate current/target/delta. |
| Windows/restart | Temporary SQLite rollback/reopen tests; source timestamps preserved; no new service, locking scheme or path dependency. Real VPS remains unqualified. |
| Released routes | No deployed YAML changed. CORE5 accepts only the dedicated data profile, XNYS, daily EOD clock, next-open MOO and full account budget. |

## Next deployment gate

Use the disabled [CORE5 template](release_templates/pod_taa_adaptive_macro_core5_daily_moo.yaml.example)
to prepare a dedicated virtual forward ledger on the intended machine, then
observe natural daily cycles. Qualify the separate PAPER broker connection and
short order lifecycle, including rejects/reconnects and the two DBC legs. Actual
borrow availability/rate, margin, account routing and measured execution costs
must be resolved before enabling a small real-capital trial. No such forward
run or real account qualification was performed by this local step.
