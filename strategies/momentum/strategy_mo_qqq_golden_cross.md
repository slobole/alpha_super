# QQQ Golden Cross

## Status and purpose

Implemented for research only. The [first historical baseline](../../docs/research/QQQ_GOLDEN_CROSS_BASELINE_20260915.md)
was completed on 15 September 2026. One fixed 50/200 variant; no parameter search
or claim of independently validated investment edge. The hypothesis is to participate in sustained QQQ uptrends and leave after
a downward crossover. Lag and repeated losing trades in sideways markets are
expected weaknesses.

## Frozen rules

- Asset: QQQ only. Long or cash.
- Signal prices: observed daily Norgate CAPITALSPECIAL closes.
- SMA_n(T) = sum of the last n closes through T / n, for n=50 and n=200.
- Let d_T = SMA50(T) - SMA200(T).
- When flat, buy if d_(T-1) <= 0 and d_T > 0.
- When long, sell all shares if d_(T-1) >= 0 and d_T < 0.
- Equality itself keeps the position. Equality-to-positive/negative qualifies
  as the corresponding crossing; position guards prevent stacking entries.
- Start in cash. Warmup history creates indicators, never initial holdings.
  A crossing before the selected start date is not carried into the run.
- First possible signal requires 201 valid session closes. The runner requires
  200 warmup sessions before the start, allowing a signal on the start's close.
- Keep shares unchanged between crossings; no daily rebalancing, shorting,
  stops, additional filters, or parameter tuning.

```text
Observed Close_T -> SMA50/200 and crossing -> fixed share order -> Open_(T+1)
                    *** CRITICAL: nothing after Close_T informs the order.
```

## Sizing and accounting

On entry, target shares = floor(NAV_T / Close_T). Exit targets zero shares.
This is a 100% allocation target with whole-share rounding, not guaranteed
cash-constrained execution. Gaps, slippage and commission can create negative
cash; financing is not modeled. Preserve the engine's negative-cash diagnostics
and resolve this limitation before any deployment decision (gap G-023).

Defaults follow the existing engine: 0.025% slippage per side, $0.005 per share
commission with a $1 minimum per order, 25% withholding on positive dividends,
and 0% interest on cash. The dividend ledger is mandatory; it credits eligible
prior-close holdings on the ex-date before the open, with no automatic share
purchase. Norgate's Dividend field is dated on entitlement session T; the engine
posts it before the next session's open (T+1). This models economic accrual,
not broker pay-date settlement (G-024).

The run requires explicit history start, test start, end and initial capital.
Cost defaults can be supplied explicitly for the agreed experiment. The first
frozen study used 22 December 1999 through 14 September 2026 and $100,000 per
account, with a matched buy-and-hold account under the same costs and dividend
accounting. No actual fund allocation was made. The strategy itself attaches
no automatic price-only benchmark.

## Data integrity and scope

The loader requests direct Norgate with PaddingType.NONE. Snapshot mode is
unsupported until equivalent observed-price provenance is available. Inputs
must have unique, increasing daily dates covering every XNYS session in their
range, positive finite OHLC and volume, finite dividends, and the declared
price basis and padding policy. Missing data stops the run; no fill or row
dropping is used. The fixed ETF universe is not a reconstruction of today's
Nasdaq constituents, and selecting QQQ is not evidence of broad applicability.

BENCH discovers the module as RESEARCH. The standard run_variant hook requires
the dates and capital above. No engine, broker, LIVE adapter, release, scheduler
or portfolio allocation changes are part of this implementation.
