# Sector Dispersion IBS Family-Universe Preregistration

## TL;DR

This is a research-only, six-row universe robustness study for the existing
Sector Dispersion IBS strategy. The signal, execution timing, position sizing,
costs, and readiness rules are frozen before any result is inspected.

The study asks whether the same simple long-only mean-reversion rule survives
longer samples and coherent alternative ETF families. It does not search for a
better threshold, lookback, sizing rule, or winning subset.

## Frozen Research Question

Does the Sector Dispersion IBS rule remain economically useful when it is run
with the same mechanics across six predeclared sector-ETF universes?

The search space contains exactly six universe rows. No parameter variants are
part of this study.

## Frozen Strategy Rules

For ETF `i` on completed daily bar `T`:

```text
IBS_i,T = (Close_i,T - Low_i,T) / (High_i,T - Low_i,T)

Range_i,T = ln(High_i,T / Low_i,T)

RelativeRange_i,T
    = Range_i,T / StdDev(Range_i,T-1, ..., Range_i,T-21)
```

- Enter when flat and `IBS_i,T < 0.10` and `RelativeRange_i,T > 1.0`.
- Exit when long and `IBS_i,T > 0.90` and `RelativeRange_i,T > 1.0`.
- A decision from completed bar `T` fills at `Open_T+1`.
- Each ETF is an independent long-only state machine.
- Existing positions are not resized while held.
- There is no stop, profit target, maximum holding period, or short position.
- Target position weight is fixed at `1 / N`, where `N` is the frozen number of
  ETFs in that universe. Unused capacity remains cash.
- Strategic leverage is `1.0`, not the article's `1.5`.

### Lookahead boundary

`Range_i,T` may appear only in the numerator. The scale in the denominator must
use the 21 completed ranges from `T-1` through `T-21`. Orders generated from
that completed signal must not fill before `Open_T+1`.

## Frozen Universes

| Priority | ID | Symbols | N | Raw common Norgate start | Research role |
|---:|---|---|---:|---|---|
| 1 | `spdr_11` | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, XLC | 11 | 2018-06-19 | Article-universe translation |
| 2 | `spdr_9` | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY | 9 | 1998-12-22 | Clean long-history SPDR test |
| 3 | `vanguard_11` | VAW, VDE, VFH, VIS, VGT, VDC, VPU, VHT, VCR, VOX, VNQ | 11 | 2004-09-29 | Broad-US family robustness |
| 4 | `spdr_proxy_11` | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, VOX, IYR | 11 | 2004-09-29 | Proxy test, not historical SPDR reconstruction |
| 5 | `ishares_us_11` | IYM, IYE, IYF, IYJ, IYW, IYK, IDU, IYH, IYC, IYZ, IYR | 11 | 2000-07-14 | Long alternative US classification |
| 6 | `ishares_global_11` | MXI, IXC, IXG, EXI, IXN, KXI, JXI, IXJ, RXI, IXP, RWO | 11 | 2008-05-13 | Global robustness |

The raw dates above were audited against the local Norgate database on
2026-07-20. They are not the execution dates. Each run begins only after every
ETF has 22 consecutive valid positive-range OHLC observations, so the 21-day
lagged range scale is fully causal for the complete fixed basket.

The study endpoint is frozen at `2026-07-17`, the latest common local Norgate
session observed during preregistration.

## Data And Return Contract

- Tradable ETF OHLC uses Norgate `CAPITALSPECIAL` adjustment so execution
  prices, historical shares, and per-share commissions remain economically
  interpretable.
- Norgate cash distributions are credited to positions held across the ETF's
  ex-dividend open. A buy at that open is not credited; a position held before
  that open is credited even if sold at the open.
- The market benchmark is Norgate `$SPX` with `TOTALRETURN` adjustment, matching
  the repository benchmark convention.
- Each universe also receives a frictionless daily equal-weight total-return
  benchmark built from its own capital-adjusted closes plus cash dividends.
- Idle strategy cash earns zero interest.

## Frozen Costs And Sizing

- Capital base: `$100,000`.
- Signed open-price slippage: `0.00025` per side, or 2.5 basis points.
- Commission: `$0.00525` per share with no hard minimum.
- Target shares at signal time:

```text
target_shares_i,T
    = previous_total_value_T * (1 / N) / Close_i,T
```

The target is filled at `Open_T+1`. Overnight gaps and trading costs can make
realized weights differ from `1 / N`. The study must report maximum gross
exposure, minimum cash, and negative-cash days. These are implementation-drift
diagnostics, not a hidden permission to add strategic leverage.

## Frozen Comparisons

Each universe is evaluated over:

1. Its own full causal history.
2. The common overlap beginning on the latest effective start across all six
   universes.
3. Fixed calendar diagnostics where data exist:
   - 1999-2007
   - 2008-2017
   - 2018-2021
   - 2022-2026-07-17
4. Calendar-year returns.

The fixed subperiods are diagnostics, not train/test or genuine out-of-sample
claims. The article and its results were known before this study.

## Required Outputs

The common comparison must include:

- effective start and end dates
- annualized return
- annualized volatility
- Sharpe ratio at a 0% risk-free rate
- maximum drawdown
- MAR ratio
- annualized turnover
- annualized cost drag
- trade count
- average and maximum gross exposure
- average and maximum active positions
- minimum cash and negative-cash day count
- dividend cash credited
- equal-weight benchmark return, Sharpe, and drawdown
- strategy-minus-equal-weight annualized return
- data-quality and forced-liquidation counts

Saved evidence must include the universe manifest, metadata, daily equity,
daily exposure, transactions, completed trades, dividend credits, annual
returns, subperiod metrics, and one compact comparison table.

## Interpretation And Rejection Rules

- A higher CAGR alone is not sufficient.
- A credible result should remain directionally useful across multiple coherent
  families and should not depend only on 2020 or one ETF.
- High turnover, large cost drag, frequent realized exposure above 100%, or
  material negative cash weakens the result.
- The six rows are correlated tests. Any statistical significance claim must
  show the full six-test search count and a multiple-comparison adjustment.
- `spdr_proxy_11` must never be described as historical `XLC`/`XLRE` data.
- `ishares_us_11` contains `IYZ`, whose historical telecom exposure is not the
  modern Communication Services sector.
- `ishares_global_11` is a separate mechanism test because foreign underlying
  markets may be closed while the US-listed ETF continues trading.

## Known Model Gaps

| Issue | Expected bias direction | Impact | Mitigation |
|---|---|---|---|
| Daily completed OHLC replaces the article's 15:45 snapshot | Unknown | High for article parity | Label as the next-open translation, never an exact replication |
| Next-open fills replace MOC | Unknown | High | Preserve consistently across all six rows and report the gap |
| Simple 2.5 bps slippage and complete fills | Usually optimistic | Medium | Keep costs explicit and reject marginal edges |
| Zero cash yield | Pessimistic for this cash-heavy strategy | Medium over long samples | Report exposure and do not add cash yield after seeing results |
| ETF classification and mandate drift | Unknown | Medium to High | Use coherent family labels and discuss `VOX`, `IYZ`, and global ETFs explicitly |
| Close-based sizing followed by next-open gaps | Can create small unintended leverage or underinvestment | Medium | Report realized exposure and cash diagnostics |

## Data-Quality Amendment Before Result Inspection

The first full run stopped before producing the comparison table because the
combined Norgate calendar contains four complete no-print sessions for `RXI`:

- 2025-07-07
- 2025-07-14
- 2025-07-22
- 2025-08-12

This amendment was frozen on 2026-07-20 after identifying the data failure and
before inspecting any universe performance metrics.

For an isolated session where all four OHLC fields are missing after an ETF's
inception:

- carry the most recent prior close into Open, High, Low, and Close for
  valuation only;
- set the dividend to zero if it is missing;
- mark the row as a stale no-print session;
- cancel any order for that ETF that would otherwise fill on the stale open;
- allow no entry or exit signal from the stale row because its range is zero;
- retain the normal session for every other ETF in the basket.

The study still fails loud if a row is partially populated or internally
inconsistent, if no prior close exists, if stale sessions are consecutive, or
if one ETF has more than five stale sessions. The stale rows and canceled
orders must be saved and counted in the final comparison.

## Completion Gate

The research is complete only when all six frozen rows either run successfully
or fail with a specific saved data-quality reason, focused tests pass, artifacts
are reproducible, and the final verdict explicitly names what survived and what
was rejected.

## Post-Run Independent Review Clarification

This clarification was added after the completed run; it does not change the
frozen rules or any result. Because a stale no-print row has `High == Low`, its
log range is invalid rather than zero in the signal calculation. The lagged
21-valid-observation denominator therefore suppresses that ETF's signals until
21 subsequent valid ranges are available. The four overlapping `RXI` gaps make
its relative-range signal unavailable continuously from 2025-07-07 through
2025-09-11; it becomes eligible again on 2025-09-12. `RXI` was flat before the
first gap and no stale-open order was canceled, so this creates missed-signal
risk rather than a stale fill or a delayed exit in the saved run.

In the comparison artifact, the zero unresolved data-quality count means zero
fatal issues remaining after the declared stale-session policy. It must not be
read as zero observed stale sessions; those four sessions have their own
explicit count and ledger.
