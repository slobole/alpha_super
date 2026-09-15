# Adaptive Macro CORE5: local data qualification

## Scope and conclusion

Step 2 prepares a dedicated snapshot profile, `norgate_eod_core5`, and a
reproducible local comparison with direct Norgate data. It does not activate a
live strategy, create a release/account route, or operate a broker or VPS.

The approved DBC short rule is unchanged. Its 10% limit is a target at rebalance;
market moves can subsequently increase its weight. No strategy variants were
searched in this step.

## Recorded local result: 2026-09-15

**Passed**, with the endpoint frozen at **2026-09-11**. The combined input
history spans 1990-01-02 through that endpoint; individual ETFs retain their
later inception dates.

| Exact comparison | Count |
|---|---:|
| Full input history | 9,241 rows, 59 columns |
| Computed feature columns | 126 |
| Daily target rows | 4,786 |
| Rebalance rows | 546 |
| Transactions | 2,679 |

All price values/provenance, features, target weights, borrow charges and NAV
matched exactly. Transactions matched after the documented absolute order-ID
rebase. The direct and snapshot input Parquet files also have identical SHA256
hashes. The diagnostic starting capital was $100,000, using unchanged strategy
parameters and borrow assumptions.

Evidence: [qualification.json](../../results/research/strategy/strategy_taa_adaptive_macro_core5/snapshot_data_qualification/2026-09-15_step2_final/qualification.json).
Its source and artifact hashes were rechecked after completion; every recorded
source file predated the run. Earlier failed diagnostic runs remain alongside
the final evidence. This endpoint is a frozen comparison date, not a statement
that a current live decision has sufficiently fresh data.

## Data contract

| Purpose | Symbols | Adjustment |
|---|---|---|
| Execution prices and native dividends | SPY, IEF, GLD, DBC, UUP, BIL | CAPITALSPECIAL |
| Signal closes | SPY, IEF, GLD, DBC, UUP | TOTALRETURN |
| Benchmark and calendar helper | $SPX, $SPXTR | TOTALRETURN |

- Request history from **1990-01-01**, retaining each ETF's actual available
  start. No synthetic history is invented before inception.
- Preserve `ALLMARKETDAYS` historical padding and the exact source tail.
  The adaptive average and running high depend on earlier history.
- Require schema v2 and native ETF dividend fields. Preserve the source field
  list, numeric dtypes and date-index name. Dtype restoration must be lossless.
- Require the complete adjustment/symbol pairs and coverage through the
  snapshot session. Reject future dates, duplicate rows, missing interior
  observations, invalid OHLC/dividends, and disagreement with declared coverage.
- Additionally read each underlying symbol at the export endpoint with
  `PaddingType.NONE`. A padded or infinite endpoint cannot certify freshness.
- Verify file hashes on each CORE5 manifest validation, including after a prior
  successful read. Validate existing and staged exports before accepting them.
- Both CORE5 price reads must retain the same snapshot directory and manifest
  hash. Publication during the combined read fails and requires a fresh retry.
- CORE5's `$SPX` benchmark label reads `$SPXTR`, matching the direct loader.
  Existing profiles retain their current benchmark selection behavior.

The fixed start request and source coverage declarations do not independently
certify Norgate's historical database. The local qualification adds an exact
comparison of all available direct and exported rows. It uses the currently
installed data vintage, not saved historical decision-time snapshots.

## Unchanged strategy and timing

Five independent sleeves each target 20%. An active sleeve holds its ETF;
otherwise it holds BIL. At a rebalance, DBC receives the following short target
only when its 10-day average is strictly below its adaptive average and its
63-day volatility is valid and positive. Otherwise its short target is zero:

```text
short_weight_T = -min(0.10, 0.025 / annualized_volatility_63_T)
```

The long/BIL targets still sum to 100%; short proceeds remain cash. Existing
borrow assumptions, signals and sizing formulas are unchanged. Initialization,
a long-state change, or month-end triggers the existing rebalance logic.

```text
[Full Norgate history through Close_T]
                 |
                 v
[Validated CORE5 snapshot: one data vintage]
                 |
                 v
[Existing signals and Close_T target-share calculation]
                 |
       *** CRITICAL *** only information through T
                 |
                 v
[Existing historical engine: execution at Open_(T+1)]
```

This diagram describes the qualified historical engine path. The subsequent
[Step 3 adapter](CORE5_ADAPTER_QUALIFICATION.md) uses the exchange calendar for
month-end; the final available price row is not automatically month-end.

## Reproduce locally

Use an installed local Norgate database and an unused output directory:

```powershell
.\.venv\Scripts\python.exe -B scripts/review/verify_core5_snapshot_parity.py --snapshot-date 2026-09-11 --output-dir results/research/strategy/strategy_taa_adaptive_macro_core5/snapshot_data_qualification/new_run
```

The command reads local Norgate and writes an isolated snapshot and evidence
under that new directory. It restores temporary process environment settings.
It does not update Norgate, publish to clients, or run the live scheduler.

The qualification compares:

1. Entire price frames, unavailable prefixes, fields, dtypes and provenance.
2. All computed features, without intersecting dates or filling mismatches.
3. Daily targets, rebalance targets, borrow charges, transactions and NAV.

The engine assigns order IDs from a process-global counter. Transaction
comparison subtracts each run's first ID; relative ID gaps/grouping and every
other transaction field must match exactly. The engine counter is not changed.

`qualification.json` records the frozen end date, source/package versions,
manifest and artifact hashes, counts, pass/failure and limitations. Failed runs
are retained. Successful local data parity is not new evidence of profitability
or actual fill/borrow availability.

## Remaining live integration

The following was the follow-up scope at Step 2 completion. Local adapter,
state and lifecycle work is now recorded in [Step 3](CORE5_ADAPTER_QUALIFICATION.md).
Forward execution and real account qualification remain outstanding.

- Wire a dedicated strategy adapter and durable strategy state.
- Use the exchange session calendar for month-end and next-open timing.
- Require the snapshot date needed by the actual decision cycle. Generic
  `end_date` slicing is not a freshness gate; use the existing readiness gate
  or `minimum_snapshot_date_str` in the integration.
- Preserve Close_T sizing and next-open execution, including short proceeds,
  borrow handling, rejected/partial orders and restart/reconciliation behavior.
- Verify the complete decision, order, fill and state lifecycle in forward
  testing before considering live activation.

## Change impact review

Tier 3 data integration, with shared data validation and a strategy loader also
touched. Parity, failure-modes, coverage and quant-pitfalls reviews apply.

The final local regression run passed **193 tests** across the CORE5 profile,
existing snapshot store/API/sync, server/client diagnostics, CORE5 strategy and
borrow study, release manifests, and scheduler utilities. Three read-only
review agents covered parity, failure modes, and combined coverage/quant
pitfalls. Their identified issues were fixed and regression-tested.
The only test warning was an unrelated Jupyter path deprecation.

| Required live-impact check | Result |
|---|---|
| Order timing | No order/scheduler timing logic changed; next-open semantics retained. |
| Sizing math | No amount, target, percent/value or share-rounding changes. |
| Reference prices | No close/open substitution; CORE5 benchmark backing corrected to direct-loader semantics. |
| State/config compatibility | Existing profiles and schema-v2 consumers retain their behavior; new metadata is CORE5-specific. No SQL/pickle migration. |
| Dashboard/log fields | No existing log fields removed or changed. |
| Windows/restarts | Uses existing staged export/promotion; no new services, background processes or lock scheme. Tests cover failed export and repeated validation; VPS behavior remains unverified. |
| Released routes | No released YAML changed. At Step 2 CORE5 was not in the supported strategy allowlist; Step 3 adds incubation/PAPER support and keeps physical LIVE blocked. |
