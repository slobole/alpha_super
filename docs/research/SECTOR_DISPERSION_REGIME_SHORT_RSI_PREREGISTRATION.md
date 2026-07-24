# Sector Dispersion Regime, Short, And RSI Exit Preregistration

## Scope

This is a research-only extension of the completed daily-bar Sector Dispersion
IBS family study. It does not change the shared strategy, BENCH discovery,
live/release wiring, portfolio allocations, or pod weights.

The study answers six user-proposed changes with one frozen baseline. No result
may be inspected before the variant table, formulas, costs, universes, and
rejection rules below are saved.

## Inherited Baseline

For ETF `i` on completed session `T`:

```text
IBS_i,T = (Close_i,T - Low_i,T) / (High_i,T - Low_i,T)

Range_i,T = ln(High_i,T / Low_i,T)

RelativeRange_i,T = Range_i,T
                    / StdDev(Range_i,T-1, ..., Range_i,T-21)
```

Baseline long entry:

```text
IBS_i,T < 0.10 AND RelativeRange_i,T > 1.0
```

Baseline long exit:

```text
IBS_i,T > 0.90 AND RelativeRange_i,T > 1.0
```

Every decision is made after the completed daily bar `T` and fills at
`Open_T+1`. Positions are independent, long-only or short-only according to the
variant, and are not resized while held.

## Frozen Signal Definitions

The market regime uses `SPY` capital-price history, not the `$SPX` total-return
benchmark series:

```text
SMA_L,T = mean(SPY Close_T-L+1, ..., SPY Close_T)

Bull_L,T = SPY Close_T > SMA_L,T

Bear_L,T = SPY Close_T < SMA_L,T
```

Including `Close_T` in `SMA_T` is causal because the decision occurs after the
completed close and the order fills at `Open_T+1`.

Per-ETF RSI uses the conventional Wilder/TA-Lib 0-100 scale:

```text
RSI2_i,T = RSI(ETF Close, length=2) evaluated through Close_T
```

The user-specified normalized threshold `0.9` is implemented as conventional
`RSI2 > 90`; the mirrored short threshold is `RSI2 < 10`.

Market conditions are **entry gates only**. A regime flip does not force an
existing position to exit. This isolates entry selection from exit behavior.

## Frozen Variant Table

| Priority | Variant | Side | New-entry rule | Exit / cover rule | Direct control |
|---:|---|---|---|---|---|
| 1 | `B0` | Long | Baseline long entry | Baseline long exit | None |
| 2 | `L200` | Long | Baseline entry AND `Bull_200` | Baseline long exit | `B0` |
| 3 | `S0` | Short | Baseline long-exit signal | Baseline long-entry signal | Zero-return cash |
| 4 | `S200` | Short | `S0` entry AND `Bear_200` | `S0` cover | `S0` |
| 5 | `S100` | Short | `S0` entry AND `Bear_100` | `S0` cover | `S0` |
| 6 | `L200_RSI` | Long | Same as `L200` | Baseline long exit AND `RSI2 > 90` | `L200` |
| 7 | `S200_RSI` | Short | Same as `S200` | `S0` cover AND `RSI2 < 10` | `S200` |

This is `7 x 3 = 21` strategy rows, of which 18 are new controlled rows and
three are baseline references. No parameter grid will be added after result
inspection.

## Frozen Universes

| Universe | Symbols | Role |
|---|---|---|
| `spdr_9` | `XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY` | Primary long history from 1998/1999 |
| `vanguard_11` | `VAW,VDE,VFH,VIS,VGT,VDC,VPU,VHT,VCR,VOX,VNQ` | Independent broad-US family robustness |
| `spdr_11` | `XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLRE,XLC` | Current complete SPDR implementation from 2018 |

Each ETF receives a fixed absolute target of `1/N` of prior portfolio equity.
Long and short sleeves are tested separately; this study does not create a
combined long/short portfolio.

Short-sale proceeds remain as cash and are not reinvested. The intended maximum
gross exposure is 100%, but realized gap-driven drift above 100% must be saved
and reported rather than described as strict broker-level no leverage.

## Frozen Costs And Corporate Actions

- Initial capital: `$100,000`.
- Slippage: `2.5 bps` per side, unchanged from the baseline.
- Commission: `$0.00525/share`, no minimum, unchanged from the baseline.
- Idle cash and short-sale proceeds earn `0%`.
- ETF trade prices use Norgate `CAPITALSPECIAL`.
- `$SPX` performance benchmark uses Norgate `TOTALRETURN`.
- `SPY` regime history uses Norgate `CAPITALSPECIAL`.
- Long ETF distributions are credited to pre-open held shares.
- Short ETF distributions are debited from pre-open held shares.
- Short borrow fee: fixed `1.00%` annualized on absolute short market value,
  accrued by calendar day using the previous valid close.
- Locates, recalls, hard-to-borrow rejection, and time-varying borrow fees are
  not modeled; short results remain optimistic despite the fixed borrow debit.
- Fixed study endpoint: `2026-07-17`.

## Saved Evidence And Metrics

Every row must save daily equity, realized weights, transactions, completed
trades, dividend cash flows, borrow fees, summary metrics, annual returns, and
subperiod metrics. The comparison must report:

- CAGR, volatility, Sharpe, maximum drawdown, and MAR;
- correlation and beta to `$SPX` total return;
- turnover, modeled cost drag, trade count, win rate, holding period, and worst
  trade;
- average and maximum gross exposure plus average net exposure;
- dividend cash flow and borrow cost;
- full-history and common-2018-overlap results;
- direct-control deltas for return, Sharpe, and drawdown;
- Newey-West/HAC t-statistics for the direct-control daily return difference;
- Bonferroni-adjusted p-values across the 18 new universe/variant comparisons.

Fixed subperiods remain `1999-2007`, `2008-2017`, `2018-2021`, and
`2022-2026` where the relevant universe has observations.

## Decision And Rejection Rules

- `L200` is useful only if it improves Sharpe or MAR in at least two of three
  universes without relying on one crisis and without a large CAGR sacrifice.
- `S0`, `S200`, or `S100` is not a viable standalone sleeve unless net CAGR is
  positive in at least two of three universes after dividends, baseline costs,
  and the fixed borrow fee.
- An RSI exit is retained only if it improves Sharpe or MAR in at least two of
  three universes and does not create materially worse holding duration or
  short-tail loss.
- A row is not promoted because it has the single highest CAGR.
- The six new hypotheses are correlated and follow substantial earlier sector-
  dispersion research. Bonferroni output is diagnostic, not proof of an
  independent discovery.
- The 2018-current SPDR row remains the canonical implementation universe; the
  older families validate or reject mechanisms but do not reconstruct modern
  `XLC`/`XLRE` history.
- Any result dependent on optimistic short execution, missing locate behavior,
  or one event is research-only and non-deployable.

## Prior Research Lineage

An earlier `short_sleeve_study` tested 14 mirrored-short rows on different
baskets. Its standalone shorts were negative, it used `$SPX` total-return
SMA200, forced covers when the regime gate turned off, and omitted dividend and
borrow debits. Those inspected results mean the present short study is not a
clean independent out-of-sample discovery. The current work is a stricter
implementation and robustness/falsification exercise.
