# Momentum capsule legs wiring - 2026-11-03

The three legs of the closed momentum capsule are WIRED as separate account routes. Templates are disabled. This is code readiness, not deployment, broker-fill evidence or an allocation decision. Nothing here changes NDX or TAA. Closed design and evidence: [research record](../research/RESIDUAL_MOMENTUM_20261031.md).

## Contract and scope

- Leg 1: `strategies.momentum.strategy_mo_capsule_leg1_natr_ndx:MomentumCapsuleLeg1Strategy`. Nasdaq-100, 10 names, ROC12 / NATR20.
- Leg 2: `strategies.momentum.strategy_mo_capsule_leg2_residual_ndx:MomentumCapsuleLeg2Strategy`. Nasdaq-100, 10 names, market and sector residual, mean / std.
- Leg 3: `strategies.momentum.strategy_mo_capsule_leg3_residual_r1000:MomentumCapsuleLeg3Strategy`. Russell 1000, 20 names, market and industry-peer residual, cumulative / volatility.
- One account per leg. An enabled leg cannot share an account with another enabled pod. `pod_budget_fraction_float` is the fraction of that account, in (0, 1].
- The books in `portfolios/momentum_capsule_three_legs.yaml` and `portfolios/momentum_capsule_two_legs.yaml` are research aggregations with an annual reset. Wiring the accounts does not implement transfers between them. Two legs run without the third by not enabling it.
- The owner has not chosen a portfolio. These routes do not imply one.

```text
[Swap session T: Norgate PIT prices, membership, gate and sector series + frozen current GICS]
        | exact T snapshot, validated hashes, complete data
        v
[The backtest's own loaders, panel and path, replayed from the first decision through T]
        | *** CRITICAL *** the last row: 0.5 * latest T0 book + 0.5 * latest T10 book, frozen at Close_T
        v
[DecisionPlan + snapshot and label provenance in the account's local state]
        | next session, pre-open broker NAV and reference quotes
        v
[Existing full-target VPlan: whole-share targets -> delta from actual holdings]
        | MOO submitted at open minus 6m30s
        v
[Next opening auction -> broker reconciliation -> persisted completion]
```

## One form for backtest and live

A leg decides on two sessions of every month: T0, the last session, and T10, the session 10 sessions before it. Each day keeps its own book, and that book's rank buffer reads only what that book held:

```text
target_{i,T} = 0.5 * w_i(latest T0 book) + 0.5 * w_i(latest T10 book)
```

A name of the other book that stopped trading since its decision carries no weight at T: its share stays in cash, as in the research form.

Book rules are in the docstring of `strategies/momentum/momentum_capsule_legs.py`, the one source for the backtest and for live. No parameter was searched for this wiring. Release validation freezes every rule; only `capital_base_float` may appear. The design itself came from 21 research rounds and about 3,000 series over a history that has been examined many times; three legs fell short of the frozen rule twice (4 and 5 reads of 9, bar 6) and were kept by the owner's decision. No deflated Sharpe exists, and no forward record has started.

**Why one account differs from research.** Research ran each day as a separate book with an annual reset. The live route has one target book per account, so the one-account form trades the other book's names back to their weights at each swap. Measured in `scripts/research/residual_momentum_20261031/r34_one_account_form.py`:

| Book | Form | CAGR | Sharpe | Max DD | Correlation |
|---|---|---|---|---|---|
| Leg 1 | Two books | 13.02% | 0.6876 | -26.42% | |
| Leg 1 | One account | 13.02% | 0.6886 | -25.59% | 0.9996 |
| Leg 2 | Two books | 12.99% | 0.7290 | -23.76% | |
| Leg 2 | One account | 12.89% | 0.7242 | -24.04% | 0.9996 |
| Leg 3 | Two books | 11.40% | 0.7429 | -22.58% | |
| Leg 3 | One account | 11.35% | 0.7400 | -22.48% | 0.9998 |
| Three legs, days 0 + 10 | Two books | 12.59% | 0.7604 | -22.87% | |
| Three legs, days 0 + 10 | One account | 12.54% | 0.7580 | -22.96% | |
| Three legs, median of 11 day pairs | Two books | | 0.7230 | | |
| Three legs, median of 11 day pairs | One account | | 0.7212 | | |

**Why no strategy memory is stored.** The path depends only on prices, membership and labels, never on the account. A decision replays it and takes the last row, so a missed cycle or a restart cannot leave a stored buffer out of step. The price: a vendor's correction of old data, or a relabelled sector, can change a past pick and so today's target. Live always equals today's backtest, not yesterday's.

## Shares and timing

For pre-open reference quote P and actual broker account NAV:

```text
account_budget = broker_NAV * pod_budget_fraction
live_target_shares_i = floor(weight_i,T * account_budget / P_i,preopen)
order_delta_i = live_target_shares_i - actual_broker_holding_i
```

Names omitted from the target go to zero. The route reuses the existing full-target execution path with `next_open_moo`: submission at 09:23:30 America/New_York, fill at the next session's open. A missed cycle is rejected, not turned into a late trade. Shared `execution_engine.py` sizing and broker price sourcing are unchanged. The plan records `momentum_legs_sizing_contract_str=full_target_preopen_v1` and `research_quantity_basis_str=Close_T`. The engine fixes quantities with Close_T NAV and prices, so identical decisions do not promise identical shares or P&L.

`month_swap_snapshot_ready` is a new clock, reserved for these releases. On any other session a release waits with `not_month_swap_session`.

## Frozen data contract

Two new schema-v2 profiles: `norgate_eod_ndx_momentum_legs_pit` for legs 1 and 2, `norgate_eod_r1000_momentum_legs_pit` for leg 3. Contract: `data/momentum_legs_snapshot_contract.py`.

- Everything a backtest reads, 1998 through T: every name with a member session since 1999-06, delisted names too; exact membership; CAPITALSPECIAL OHLC, Turnover, Unadjusted Close with ALLMARKETDAYS padding; `$SPXTR` TOTALRETURN; `SPY`, `$NDX`, `$VXN` or `$RUI`; the 11 GICS sector indexes.
- The gate, market, sector and benchmark series need one unpadded close at T equal to the exported one. A provider update during export is rejected.
- A current member without an unpadded close at T is a halt: it is listed in `unobserved_member_list` and keeps the carried close the backtest also reads. More than 2% of the members means the stocks have not updated, and the export fails. VXN keeps its as-of rule.
- GICS labels are captured on the producer: sectors, and for Russell 1000 industry groups. A provider error aborts the export. The map, hash, date and UTC capture time travel in the manifest; a plan records the hash and the target names' sectors. The client never calls Norgate.
- **A label reads UNKNOWN only after three reads say so.** Norgate answered None for a classified name while another process read it: `DIS`, once in about 6,000 reads, 2026-11-03. One such run moved 12 books over the path, including the latest target. The same exposure exists in the older sector-map readers (E2 export and backtests); they were not changed here.
- A stock with no bar at T also reads as a non-member at T, so a partial update shrinks the member list instead of filling that list. A member count at T under 98% of the prior session's therefore fails the export (worst real one-day drops: Nasdaq-100 1.9%, Russell 1000 0.4%).
- A wrong map, hash or date, a missing anchor, a hole inside a history, a missing helper or benchmark, a missing Turnover at T, more than 5% of current members without a sector, a name whose sector and industry group disagree, or a snapshot other than exact T stops a new decision.
- So does unavailable evidence at T: a missing gate, VXN or market series, no members, fewer than 80% of members scored with the gate open, or a book that is not full with the gate open. At every saved decision since 2000 a gate-open book was full in every leg and at least 96% of members were scored. A valid closed gate remains an all-cash target.
- A refused decision is logged as `momentum_leg_decision_blocked` and does not stop other pods in the same serve. It is not replayed every scheduler tick on the same manifest, because a Russell 1000 replay peaks at 7.8 GB; a new manifest or a new process decides again. Running out of memory is refused the same way and never remembered.
- File hashes are checked on every read; the content check runs once per exact manifest per process. Identity is checked before and after the decision. Prices are not kept in memory between decisions.

GICS labels are current at export, not historical (G-025). The share-class tickers among current Russell 1000 members (`BF.A`, `HEI.A`, `LEN.B`, `UHAL.B`) were added beside `BRK.B` and `BF.B` in the IBKR alias list. A new dotted ticker fails at contract qualification, closed, until it is added.

## Backtest in Bench

Bench runs the same form at 2.5 bps a side, $0.005 a share with a $1 minimum, 0% on cash:

| Book | CAGR | Sharpe | Max DD | Beta | Turnover a year |
|---|---|---|---|---|---|
| Leg 1 | 13.22% | 0.81 | -26.3% | 0.44 | 620% |
| Leg 2 | 13.30% | 0.86 | -23.8% | 0.42 | 702% |
| Leg 3 | 11.89% | 0.92 | -22.3% | 0.37 | 989% |
| Three legs, reset yearly | 12.93% | 0.91 | -22.8% | 0.41 | |
| Two legs, reset yearly | 13.36% | 0.86 | -23.3% | 0.43 | |

These are not the research numbers: research charges 5 bps plus a causal half-spread, earns T-bills on cash and reports Sharpe over T-bills. Daily correlation with research is 0.9995, 0.9995 and 0.9996. They are also day pair 0 + 10, the best of the 11 pairs in research (0.7604, median 0.723; the deepest pair falls -30.5%, or -30.8% in one account).

Plan on the research record's 11% / 0.65 / -26% for three legs and 11.3% / 0.63 / -25% for two, not on these rows.

**Capacity at the open, as wired.** Bench Capacity, every order in the opening auction, Sharpe after its central impact:

| Leg | Window | No impact | $1M | $5M | $10M |
|---|---|---|---|---|---|
| Leg 1 | Last 5 years | 0.58 | 0.60 | 0.58 | 0.56 |
| Leg 2 | Last 5 years | 0.61 | 0.63 | 0.60 | 0.57 |
| Leg 3 | Last 5 years | 0.54 | 0.61 | 0.56 | 0.51 |
| Leg 1 | 2000-2026 | 0.80 | 0.75 | 0.61 | 0.40 |
| Leg 2 | 2000-2026 | 0.86 | 0.77 | 0.53 | |
| Leg 3 | 2000-2026 | 0.90 | 0.80 | 0.40 | |

At today's volumes the open holds about $5M a leg with little cost; the full-history rows are lower because volumes were thinner. Bench gives no Recommended figure for the last 5 years: its rolling three-year Sharpe floor fails there at every size, which is the recent weakness, not liquidity. On full history it recommends $100K, $100K and $50K. Research's larger figures ($59M, $50M, $20M) assume orders worked through the day, which this route does not do. That plan leans on one era. Three legs by era, days 0 + 10, Sharpe over T-bills: 2000-07 0.54, 2008-12 0.78, 2013-21 1.07, 2022-26 0.43.

**Alpha.** Bench's regression uses a zero rate over `$SPX`, before fees, daily, Newey-West:

| Book | Full history | From 2017-07 | From 2022 |
|---|---|---|---|
| Leg 1 | 9.6%, t 3.48 | 6.5%, t 1.23 | 5.7%, t 0.74 |
| Leg 2 | 9.6%, t 3.78 | 4.8%, t 1.07 | 4.5%, t 0.69 |
| Leg 3 | 8.5%, t 4.10 | 7.1%, t 1.87 | 3.5%, t 0.71 |
| Three legs | 9.2%, t 4.12 | 6.0%, t 1.52 | 4.5%, t 0.82 |
| Two legs | 9.6%, t 3.78 | 5.7%, t 1.21 | 5.2%, t 0.77 |

The full-history t comes from the years before 2017. Research's weekly factor reads, three legs, say less:

| Over | Before fees, full | Before fees, from 2017-07 | After 2 and 20, full | After 2 and 20, from 2017-07 |
|---|---|---|---|---|
| SPY | 7.6%, t 3.50 | 4.2%, t 1.09 | 3.8%, t 1.72 | 0.0%, t 0.01 |
| IWB behind the same gate | 3.7%, t 2.57 | 2.8%, t 0.95 | -0.1%, t -0.05 | -1.4%, t -0.45 |
| ETF mix + QQQ trend | 3.8%, t 1.83 | 0.1%, t 0.02 | -0.3%, t -0.12 | -4.2%, t -1.23 |

No measurable alpha since 2017 is the honest reading.

## Risks open before paper

**Opposite orders between legs.** The legs share names and swap on the same days. When accounts count as related, IBKR cancels the later of two opposite orders in one stock at one auction ([record](../plans/IBKR_CROSS_CANCEL_20261102.md)). The backtest fills both. From the engine runs at $1M a leg, swap days since 2021:

| Pair | Swap days with an opposite pair | Names on such a day | Smaller order, median | Larger order, median |
|---|---|---|---|---|
| Leg 1 / Leg 2 | 67% | 2.3 | 0.33% of NAV | 4.35% of NAV |
| Leg 1 / Leg 3 | 52% | 1.6 | 0.29% | 2.12% |
| Leg 2 / Leg 3 | 62% | 1.8 | 0.23% | 1.92% |
| Any two legs | 85% | 3.0 | | |

These pairs hold 14.6%, 13.5% and 3.5% of each leg's traded money. Either order can be the one cancelled.

**A cancelled order freezes the leg.** The reconcile fails, the plan stays `submitted` and the scheduler reports `execution_exception_parked`. At the next swap the data gate is open but no decision is built, and every later swap is skipped until a person completes the trade in Missing trades and the reconcile passes. There is no resolve command, so a BUY in a name that has stopped trading needs a database edit. If these accounts count as related, most swaps would need a person. The options are the owner's (G-038): ask IBKR whether they count as related, or run the whole capsule as one account, where the orders net and cannot cross. Do not enable two or more legs under one related login before this is settled.

**One unquotable name stops a swap.** A target name without a reference quote or a contract gives `missing_live_price`: the plan is blocked with zero orders and the book waits for the next swap, unless a person acts. The cause is a name whose last bar is T. Saved targets held such a name in 2, 2 and 9 of 627 decisions per leg. It is the existing full-target rule, shared with NDX, and was not changed. A held name that IBKR no longer qualifies repeats it at every swap.

**One serve per pod.** A serve without `--pod-id` posts all profiles together and replays leg 3 on each due tick. Run each leg in its own serve.

**Memory.** A leg 3 decision peaks at 7.8 GB for about 80 seconds on the local machine; legs 1 and 2 at 1.6 GB for 10 seconds. The content check adds about 3.8 GB for 25 seconds, twice per download and once in the dashboard, and is not part of that peak. T0 is also a month-end for monthly pods, so with MR pods the peak can approach 20 GB if everything overlaps. Check this on the VPS before enabling leg 3.

**Disk.** A Russell 1000 snapshot is 291 MB. Clients do not prune old snapshots: about 14 GB a year for the two swap sessions, with T-1 sometimes downloaded first.

**Same evening.** All real proofs used exports days after T. At 21:37 ET on 2026-11-02 the gate, market, VXN, benchmark and 11 sector series all had that session; the earliest hour is unknown. A plan records VXN's value and its date.

**Fill timing.** Bench Timing, entries and exits filled at the same time:

| Leg | Next open (as wired) | Next close (a cancelled order completed by MOC) |
|---|---|---|
| Leg 1 | 13.22%, 0.807, -26.3% | 13.02%, 0.795, -25.9% |
| Leg 2 | 13.30%, 0.864, -23.8% | 13.11%, 0.852, -22.7% |
| Leg 3 | 11.89%, 0.918, -22.3% | 11.54%, 0.892, -21.8% |

A fill at the next close costs 0.2 to 0.35 points a year and 0.01 to 0.03 of Sharpe, so completing a cancelled order by MOC is cheap. The next-open cells equal Vanilla exactly. Timing used to stop at the first held name with no next open (`AZN-202601`, listing ended 2026-01-30); by the owner's decision of 2026-11-03 it now sells such a name at its last close and drops the order, as Vanilla does. Strategies with their own missing-price rule (MR capsule parking, HPI) still stop there, because timing cannot replay their rule. Stress is not available for these strategies in Bench, as for E2.

**Producer time.** A Russell 1000 export took about 300 seconds, and the client's request timeout is 120 seconds. At a swap session the first request can time out, so the sync shows a failure and retries after its 60-second cooldown. Without a guard the retry would have removed the running export's staging folder and started again, for ever. These two profiles therefore export one at a time on the producer: a retry waits for the running export and finds its finished snapshot. A test holds that; it was not run against a real server.

**Display.** The V4 Overview shows "Twice a month" and a waiting pod as idle. The next-step line shows the next EOD capture, not the next swap session, because the trading-window forecast knows only the monthly and daily clocks.

## Operator handoff

Templates, `enabled_bool=false`, `auto_submit_enabled_bool=false`, separate placeholder accounts and client ids 53, 54, 55:

- [Leg 1](release_templates/pod_mo_capsule_leg1_natr_ndx_month_swap.yaml.example)
- [Leg 2](release_templates/pod_mo_capsule_leg2_residual_ndx_month_swap.yaml.example)
- [Leg 3](release_templates/pod_mo_capsule_leg3_residual_r1000_month_swap.yaml.example)

This work does not edit client release files or schedule a deployment. A separately authorized rollout must deploy compatible readers and exporter, let the producer serve the new profiles, sync them to the intended clients, and assign each leg its real account.

Rollback: disable the release, and **move every leg YAML out of the releases root before reverting code**. Older code refuses a root that holds an unknown strategy, even a disabled one, and every serve there stops, NDX and TAA included. No data migration is needed.

## Verification and live impact

Tier 3, also touching Tier 2 snapshot transport and Tier 1 strategy modules. Four independent reviews (parity, failure modes, coverage, quant pitfalls) found no look-ahead and no blocker in the decision logic. Their findings fixed here: a partial stock update read as a smaller list, a missing Turnover or label map read as a smaller book, a NaN weight read as a sale, a refused leg stopping other pods in the same serve, a refused decision replayed every minute, leg 3's missing evidence, and 43 tests that close gaps a mutant survived. Their findings left to the owner are under Risks.

| Check | Result |
|---|---|
| Each book against the closed design, decision by decision | Legs 1, 2: 627 of 627. Leg 3: 622 of 627; the other 5 differ only by Norgate's rename of one ticker (RAMP, RAMP-202610); weights within 3.3e-7 |
| Whole path, snapshot against direct Norgate | T0 2026-10-30: 629 of 629 decisions identical. T10 2026-10-16: 628 of 628. Gap 0, all three legs |
| Live host decision against the backtest path at T | T0 and T10: same names, gap 0, all three legs |
| Rows through T only, real data | 40 of 40 sampled decisions, gap 0, each leg |
| One-account engine runs, repeated | Leg 3 transactions byte-identical on two clean runs |
| NDX and TAA live decisions, base 6a12cda against this branch | 32 of 32 identical |
| Dashboard goldens | Same digests on base and branch |
| Swap calendar against the strategy calendar | Every session since 2007 |
| Tests | 6,305 passed, 12 skipped, 7 failed. The same 7 fail on the clean base: a vendor fingerprint, one route, two goldens, three Bench tests that need saved runs |

Real snapshots were exported to a temporary root with the actual exporter and validated by the actual reader: NDX 48 MB in about 20 seconds, Russell 1000 291 MB in about 300 seconds. Scripts: `scripts/research/momentum_capsule_wiring_20261103/`; results stay under `results/research/momentum_capsule_wiring_20261103/`. No real client root, release file, broker or scheduled task was touched.

Live-impact checklist:

- Timing: existing next-open MOO path; exact swap-session snapshot required. Existing clocks do not read the new calendar; a test holds that.
- Sizing: unchanged full-target formula; no new `amount` or `target` semantics.
- Reference price: unchanged broker pre-submit source; no close or open substitution.
- Persistence: no SQLite or config schema migration. `strategy_state_dict` is empty for these pods.
- Logging: existing fields kept; provenance is additive.
- Windows and restarts: isolated staging and the existing local store; tests cover reopen, retry and duplicate prevention.
- Releases: three new examples parse disabled. Existing routes and templates are regression-tested. Shared edits are additive: a new clock value in the existing clock sets, a new reason code in the existing wait sets, a new branch in the decision-block handler, new aliases in the IBKR share-class list, new symbols in the non-distributing index set. The export lock and the three-read label rule apply only to the two new profiles.

Residual risks are the current-GICS gap, the Close_T and pre-open quantity difference, the items above, and unqualified real broker fills. No production or broker operation is part of this change.
