# Month-End Rebalancing Flow

This implements the frozen C_main rules from Pakal's `rj_macro_etf_flow_edges_study`, dated 2026-09-11. It is a research/BENCH strategy. The saved historical target weights reproduce the supplied reference; executable account returns use the house share ledger and costs.

## Rule and timing

SPY and TLT are traded; IEF is signal-only. For a month with N NYSE sessions, session s has `dtme = N - s + 1`. Half-days count. At the seventh-last session's close, compute SPY and IEF total returns since the previous month-end:

```text
R_SPY = TR_SPY(dtme7) / TR_SPY(previous_month_end) - 1
R_IEF = TR_IEF(dtme7) / TR_IEF(previous_month_end) - 1
bond_weight = 0.40*(1 + R_IEF) / [0.60*(1 + R_SPY) + 0.40*(1 + R_IEF)]
pressure_bps = 10000*(0.40 - bond_weight)
F = count(prior_month_pressures <= current_pressure) / count(prior_months)
bucket = min(5, floor(5*F) + 1)
```

With fewer than 24 prior pressures, the bucket is missing. History always starts in August 2002 using the July 2002 month-end close. Changing the scoring start must not shorten that history. The first finite bucket is August 2004. August 2026 uses 288 prior months; the spec's 284 refers to scored months, not the complete expanding history.

| Bucket | Final five sessions of this month | First five sessions of next month |
| --- | --- | --- |
| 1 | SPY +100% | Flat |
| 2, 3, or missing | TLT +100% | TLT -100% |
| 4, 5 | TLT +100% | SPY +50%, TLT -50% |

```text
[dtme 7 close: SPY/IEF pressure + prior-only bucket]
             | *** CRITICAL*** one full session signal lag
             v
[dtme 6 MOC: enter FINAL] --hold--> [month-end MOC: reverse to EARLY]
                                             |
                                             v
                                [next month session 5 MOC: exit]
                                             |
                                             v
                                       [stay flat]
```

All three target decisions use the same frozen monthly bucket. A hold-day target on d describes exposure from close(d-1) through close(d). An entry day's return is excluded and an exit day's return is included. The exit occurs at session 5 close; session 6 is flat.

The schedule uses the XNYS calendar through the complete next month, independent of the available price endpoint. It includes historical extraordinary closures; it is not a reconstruction of when each closure was announced. Missing price sessions fail rather than masquerade as exchange closures. The local Norgate loader explicitly requests `PaddingType.NONE`; imported account frames must declare unpadded observation and price-adjustment provenance. Snapshot mode is unsupported until equivalent observation provenance is available. Months with fewer than nine sessions are skipped. Unknown future closures require a refreshed calendar and an explicit operational policy; this implementation makes no order-delay or live replay claim.

## Execution and standard costs

The strategy sizes whole shares using only the preceding session's closing NAV and CAPITALSPECIAL prices:

```text
target_shares_i = trunc(target_weight_i * NAV_previous_close / Close_i_previous_close)
fill_price_i = Close_i_fill_session * (1 + sign(order_shares_i)*0.00025)
commission_i = max(1 USD, 0.005 USD * abs(order_shares_i))
```

The strategy-local MOC adapter supplies the closing auction price to the existing execution/commission ledger after quantities are fixed. It copies the pricing frame; it does not change the shared engine or pass the execution close into sizing. Original open prices remain available for audit. Only SPY/TLT market orders are accepted. Invalid closing prices fail before cash or fills change.

At reversal, the existing trade is closed and the new side is opened in the same auction with distinct trade IDs. Each order pays its own commission minimum. Thus a +100% to -100% reversal executes approximately 200% of NAV in sell notional. Holdings are fixed between scheduled auctions.

Borrow uses the existing conservative CORE5 convention, applied to short TLT:

```text
collateral = abs(post_MOC_short_TLT_shares) * ceil(1.02 * TLT_close)
borrow_fee = collateral * 0.01 * calendar_days_to_next_XNYS_session / 360
```

The fixed 1% annual rate is always active. Fees debit cash and NAV at the interval's start close and have a separate ledger. Weekends and holidays count; no next interval is charged after the MOC cover or beyond the requested backtest endpoint. This is a trade-date planning proxy, not historical IBKR borrow billing or evidence of availability/recalls. Positive cash and short proceeds earn 0%; proceeds do not expand target sizing.

Execution and marking use Norgate CAPITALSPECIAL prices plus the native dividend cash ledger. Norgate stamps `Dividend` at the entitlement close; the following ex-date credits or debits the prior-close shares before the day's MOC fills. Long dividends receive the existing 25% withholding treatment; shorts owe the full gross amount, including when covered at that ex-date's close. Signal-only SPY/IEF series and the `$SPX` performance benchmark (backed by `$SPXTR`) use TOTALRETURN. Dividends are never added a second time to a TOTALRETURN execution series.

## What matches, and why account returns differ

The source's fixed-holdings prose conflicts with its daily constant-weight formula. The independent fixture test reproduces that reference formula only for verification:

```text
reference_return_d = sum_i h_i,d*(TR_Close_i,d/TR_Close_i,d-1 - 1)
                     - 0.0005*sum_i abs(h_i,d - h_i,d-1)
```

This is separate from the executable cash-plus-fixed-shares account. Differences are explicit:

- Prior-close share sizing replaces unknowable exact auction-close share sizing; integer rounding and price gaps change realized weights.
- Fixed shares replace daily constant-weight compounding. A 100% gross target is not a hard cap on realized exposure, especially for shorts. Cash can become negative after gaps and costs; financing is not modeled.
- House commissions, 2.5 bps slippage and 1% borrow replace the reference's 5 bps per-side notional fee without borrow. Native costs book on fill day; source costs book on the following hold day.
- Cash dividends with long withholding replace gross TOTALRETURN reinvestment. Shorts still pay full dividends.
- A new account starts flat and waits for the next scheduled auction. The reference starts January 2003 with a short carried from the December 2002 auction and its reversal cost. A later requested scoring start is also a fresh account, not a warm-state portfolio replay.

Therefore the source's Sharpe 1.04, CAGR 10.6% and drawdown -14.2% are reference observations, not expected executable metrics or promotion gates.

## Evidence and interpretation

Reduced verification fixtures live in `tests/fixtures/month_end_rebalancing_flow`. Their manifest hashes the supplied spec, input panel and expected tables. Tests verify all 284 monthly pressures/buckets, every supplied hold target and all supplied reference returns. Prefix recomputation and future-price perturbation check causality; synthetic account tests check MOC timing, prior-close sizing, reversal IDs, fixed holdings, fees, dividends, weekend borrow, missing data and determinism.

This task implements one frozen candidate; it performs no parameter search. The earlier research already examined 4 literal strategies, 10 conditional legs plus 4 combinations, 8 configurations across 4 candidates, 5 windows across 4 candidates, 17 placebos across 2 candidates, and 3 engines across 4 costs. These overlapping families are not an independent-trial total. MOC was added after inspecting other timing results; the confirmation history is already seen. The source's local multiple-comparison adjustment does not correct the complete adaptive search. C_main is a local reconstruction, not a reproduction of the paywalled author's final strategy.

The named ETF universe has no constituent-survivorship selection. Inception and complete endpoint history are enforced. No normalization, training split or target-return feature is introduced. The bucket-1 SPY leg has crisis-rebound exposure; the earlier study's shorter-window confirmation was materially weaker. These facts limit alpha claims despite exact target parity. Borrow access, changing rates, financing, auction impact, partial fills and unscheduled-closure notice timing remain unverified.

## Running and artifacts

```powershell
uv run python strategies/run_strategy.py strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow
uv run python scripts/research/run_strategy_analysis.py strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow --analysis vanilla --analysis capacity --analysis timing --analysis risk --analysis stress --keep-going
uv run python -m pytest tests/test_strategy_month_end_rebalancing_flow.py -q
```

BENCH discovers all five analyses. Maturity is PM_READY after the 2026-09-13 capital, total-return benchmark and determinism gate passed, enabling the requested research portfolio integration. The readiness evidence is under `results/research/pm_readiness/2026-09-13_125726`; all three checks passed, with capital scale 2.001 and zero benchmark CAGR gap. This is engine compatibility, not execution or alpha approval. Vanilla and Risk use the same account path. Capacity reruns each AUM at the full-history and trailing-five-year windows, keeps all July 2002+ signal warmup, and explicitly applies the MOC impact model above the existing 2.5 bps baseline. Input prices are cached per endpoint only within the capacity process and copied for each run.

Execution Timing evaluates the 4-by-4 entry/exit matrix: same-day open, same-day MOC, next-day open and next-day close, relative to the scheduled auction session. MOC/MOC remains the default. A strategy-local adapter preserves preceding-close NAV when the timing engine posts the current dividend earlier than Vanilla; explicit entry/exit roles make the new short and the old long closure move independently in reversal cells. Each cell accrues its own borrow and dividends from actual net positions. Alternative cells are sensitivity diagnostics; the strategy rule is not reselected from the matrix. Mixed timing can temporarily change exposure before the opposing leg fills.

Stress starts fresh accounts at the standard pre-event launch offsets, retaining full pressure history while trading only the scenario calendar. Missing pre-inception crises are recorded as unavailable. The separate CrisisAnalyzer registration uses full-history replay to preserve pre-crisis positions. These are different initial-state experiments. No Portfolio Manager promotion follows from analyzer completion.

Each saved run includes the normal report, account metadata, transactions and dividend ledger, plus monthly signals, MOC schedule, prior-close decisions, held shares, borrow ledger, source hold targets and this contract. Headline NAV returns include dividends and borrow; the engine's inherited trade-level P&L and win statistics exclude these separate cash flows. Use account NAV for all-in strategy performance. The implementation does not modify LIVE, broker, scheduler, release, or portfolio allocation state.
