# QQQ Golden Cross: first historical baseline

## Summary and verdict

The fixed 50/200 Golden Cross beat matched QQQ buy-and-hold over this full
historical window, with a smaller maximum drawdown. **The headline advantage
depends heavily on the initial cash position:** Golden Cross did not buy until
27 January 2003, after 775 cash sessions. It therefore missed the dotcom decline
without demonstrating an entry-and-exit decision around that bubble.

The later periods give a more restrained result: Golden Cross materially
underperformed in 2010-2019 and delivered nearly the same return with lower
volatility and drawdown from 2020 onward. **Keep it as a research baseline for a
simple trend filter; consistent return superiority and an investable edge are
not established.** There were only 16 completed trades.

Results include the stated trading costs and dividend withholding. Financing
of negative cash is unmodeled, so these are not fully financed account returns.

## Frozen experiment

| Item | Setting |
|---|---|
| Asset / variant count | QQQ / one fixed SMA50-SMA200 variant |
| Data | Direct Norgate, CAPITALSPECIAL, PaddingType.NONE |
| Available observations | 10 March 1999 through 14 September 2026; 6,921 sessions |
| Decision/evaluation window | 22 December 1999 through 14 September 2026; 6,721 daily NAV rows |
| Initial capital | $100,000 per separate account |
| Start rule | 200 warmup sessions; observation 201 is the first decision close |
| Golden Cross entry | Flat and prior SMA50 <= SMA200, current SMA50 > SMA200 |
| Golden Cross exit | Long and prior SMA50 >= SMA200, current SMA50 < SMA200 |
| Equality | Keep the current position |
| Quantity / fill | floor(previous-close NAV / previous close), filled at next open |
| Holding rule | Fixed shares between crossings; no daily resizing |
| Buy-and-hold | Buy once at the next open after the shared first decision close |
| Terminal valuation | Last observed close; no forced sale for either account |
| Costs | 0.025% slippage per executed side; $0.005/share, minimum $1/order |
| Dividends | Mandatory cash ledger, 25% withholding, no automatic reinvestment |
| Cash / financing | Cash yield 0%; negative cash allowed by engine and financing unmodeled |

Norgate dates the Dividend field at entitlement session T; eligible shares
receive net cash before the following session's open. This is economic accrual,
not exact broker pay-date settlement. No capital-gains tax is modeled.
Golden Cross can redeploy accumulated cash on a later entry; buy-and-hold keeps
its original shares and accumulates dividends as cash.

The start and end rules were frozen before performance was viewed, with the
start determined by data availability and the required warmup. No alternative
parameters, starting positions, assets or cost settings were searched.

## Full-window comparison

| Metric | Golden Cross | Matched buy-and-hold |
|---|---:|---:|
| Ending account value | $1,585,722.69 | $820,244.28 |
| Compound annual return | 10.89% | 8.19% |
| Annualized volatility | 16.11% | 26.62% |
| Sharpe, all trading days, zero risk-free rate | 0.724 | 0.429 |
| Maximum drawdown | -28.57% | -83.46% |
| CAGR / absolute maximum drawdown | 0.381 | 0.098 |
| Sessions holding QQQ at the close | 70.11% | 99.99% |
| Average invested value / NAV | 69.94% | 98.64% |
| First purchase | 2003-01-27 | 1999-12-23 |
| Initial cash sessions | 775 | 1 |
| Entries / completed round trips | 17 / 16 | 1 / 0 |
| Shares still held at the end | 2,236 | 1,126 |
| Annualized two-sided turnover | 123.06% | 3.77% |
| Total commissions paid | $487.66 | $5.63 |
| Modeled slippage paid | $2,516.54 | $25.18 |
| Net dividend cash received | $42,056.89 | $22,466.24 |

Trading costs above are sums of paid dollars, not the compounded loss relative
to a hypothetical frictionless rerun. Both accounts retain a terminal open position.

![Matched account values and drawdowns](../../results/research/qqq_golden_cross_baseline_20260915/comparison.png)

## Fixed historical blocks

These are descriptive slices of the same continuous accounts. Positions and
capital were not reset at block boundaries. Each block includes the return
from the preceding session's closing NAV; block drawdowns include that
boundary value.

| Block | GC annual return | Buy/hold annual return | GC max drawdown | Buy/hold max drawdown | GC Sharpe | Buy/hold Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 1999-12-22 to 2009-12-31 | 4.68% | -6.41% | -24.01% | -83.46% | 0.427 | -0.016 |
| 2010-01-04 to 2019-12-31 | 11.94% | 17.04% | -21.62% | -22.12% | 0.820 | 1.021 |
| 2020-01-02 to 2026-09-14 | 19.20% | 19.51% | -28.57% | -34.29% | 0.935 | 0.865 |

The full-window advantage should not be summarized as reliable crash timing.
The cold start explains why this particular account avoided the early collapse.
The 2010s provide an unfavorable counterexample to consistent superiority.

## Negative cash: unresolved execution assumption

| Diagnostic | Golden Cross | Matched buy-and-hold |
|---|---:|---:|
| Negative-cash sessions | 1,782 | 2,261 |
| Separate negative-cash episodes | 11 | 1 |
| Lowest cash balance | -$9,514.96 | -$758.63 |
| Lowest cash / NAV | -1.23% | -3.49% |
| Average balance while negative | -$2,088.56 | -$519.06 |
| Average cash / NAV while negative | -0.34% | -1.23% |

Dollar and percentage minima need not occur on the same date. The 100% target
does not prevent opening gaps and costs from overdrawing the account. Holding
the same share count can preserve that deficit for a long time. No cash buffer,
financing debit or order adjustment was silently introduced. These assumptions
must be resolved before a deployment or allocation decision.

## Verification

- Four new synthetic tests passed: matched buy-and-hold timing, scalar-account
  agreement under gaps/costs/dividends, historical-block boundary returns, and
  flat start in an already bullish trend.
- For each account, a separate scalar implementation rebuilt signals, order
  quantities, fill prices, fees, dividend credits, shares, cash and NAV from the
  frozen raw bars. It used ordinary trailing sums rather than the strategy's
  pandas rolling signals.
- All 6,721 daily account rows matched. Maximum absolute NAV discrepancy:
  $0.000000000117 for Golden Cross and $0.0000000000073 for buy-and-hold.
- All 33 Golden Cross fills and the single buy-and-hold fill matched.
- The engine's prefix signal audit passed for both runs.
- Research change classification: Tier 1. An independent read-only quant
  reviewer assessed the method, source and saved results.
- Earlier implementation verification passed 64 tests; those results are from
  the preceding implementation phase, not a newly repeated full suite.

The comparison corrects two reporting traps locally: exposure comes from daily
holdings (so the open buy-and-hold position is counted), and headline Sharpe
includes cash days rather than only active days. No shared engine behavior changed.

## Interpretation limits and next useful test

This is a descriptive full-history result, not untouched out-of-sample
validation. The user selected QQQ; there is no claim of cross-asset generality,
and prior external research on Golden Cross is unknown. Thousands of daily
observations do not replace the small number of independent trend episodes.

The model uses fixed slippage and split-adjusted share units, so historical
per-share commissions are approximate. Opening-auction capacity, partial
fills, cash-constrained execution and broker settlement have not been verified.
Snapshot input support and LIVE wiring are outside this implementation.

A useful next separately frozen study would measure sensitivity to starting
dates and the initial holding state, with the same 50/200 rule. That would
quantify the cold-start contribution. This report does not execute that study
or recommend capital allocation.

## Metric definitions and evidence

CAGR = (ending NAV / boundary NAV)^(365.25 / elapsed calendar days) - 1.
Daily return = NAV_t / NAV_(t-1) - 1.
Volatility = sample standard deviation of daily returns * sqrt(252).
Sharpe = mean daily return * 252 / annualized volatility, with zero risk-free rate.
Maximum drawdown = min(NAV / running peak NAV - 1), including boundary NAV.
The initial zero-return row is excluded from full-window Sharpe and volatility.
Two-sided turnover = sum(abs(fill value) / preceding close NAV) / elapsed years.
Exposure is measured at daily closes, not as exact intraday invested hours.

- [Frozen plan](../../results/research/qqq_golden_cross_baseline_20260915/plan.json)
- [Data manifest and input hash](../../results/research/qqq_golden_cross_baseline_20260915/data_manifest.json)
- [Frozen raw bars](../../results/research/qqq_golden_cross_baseline_20260915/prices.parquet)
- [Complete comparison metrics](../../results/research/qqq_golden_cross_baseline_20260915/comparison.csv)
- [Fixed historical blocks](../../results/research/qqq_golden_cross_baseline_20260915/subperiods.csv)
- [Independent-account verification](../../results/research/qqq_golden_cross_baseline_20260915/verification.json)
- [Environment versions](../../results/research/qqq_golden_cross_baseline_20260915/environment.json)
- [Study runner](../../scripts/research/run_qqq_golden_cross_baseline.py)
- [Synthetic tests](../../tests/test_qqq_golden_cross_baseline.py)

The artifact directory also contains both daily ledgers, both fill ledgers,
both dividend ledgers, and both independent daily reconstructions. The runner
refuses to overwrite completed results. To reproduce, point its OUTPUT_PATH
at a fresh directory containing copies of plan.json, data_manifest.json and
prices.parquet; source and input hashes are checked before execution.
