# Sector Dispersion Four-Positioning Preregistration

## Scope

This is a research-only extension of the completed daily-bar Sector Dispersion
IBS study. It changes position sizing only. It does not change the IBS entry or
exit signal, the `Close_T -> Open_T+1` timing, the ETF universes, live/release
wiring, BENCH discovery, portfolio allocations, or pod weights.

The four positioning proposals and all evaluation rules below are frozen before
their results are inspected. The existing tracked
`run_sector_dispersion_position_sizing_study.py` is a different asset-SMA/VIX
experiment and is not evidence for this study.

## Inherited B0 Signal

For ETF `i` on completed session `T`:

```text
IBS_i,T = (Close_i,T - Low_i,T) / (High_i,T - Low_i,T)

Range_i,T = ln(High_i,T / Low_i,T)

RelativeRange_i,T = Range_i,T
                    / StdDev(Range_i,T-1, ..., Range_i,T-21)
```

Entry decision:

```text
IBS_i,T < 0.10 AND RelativeRange_i,T > 1.0
```

Exit decision:

```text
IBS_i,T > 0.90 AND RelativeRange_i,T > 1.0
```

Every decision uses the completed daily bar `T` and fills at `Open_T+1`.
Positions are long-only, independent, and are not resized while held.

## Frozen Positioning Features

### Inverse 20-day volatility

For every ETF in the complete universe:

```text
r_i,T = Close_i,T / Close_i,T-1 - 1

sigma_i,T = StdDev(r_i,T-19, ..., r_i,T) * sqrt(252)

InvVolWeight_i,T = (1 / sigma_i,T)
                   / Sum_j-in-full-universe(1 / sigma_j,T)
```

The denominator always includes the full `N`-ETF universe, not only active
signals. Therefore unused slots remain cash and a small number of active signals
cannot be renormalized to 100% exposure. If any full-universe volatility needed
for the row is missing, non-positive, or non-finite, inverse-vol variants open
no new position on that decision date.

The 20 returns are computed only through completed `Close_T`; they size an order
filled at `Open_T+1`. There is no full-sample scaling.

### Soft SPY SMA200 multiplier

Using `SPY` capital-price history:

```text
SMA200_T = mean(SPY Close_T-199, ..., SPY Close_T)

MarketScale_T = 1.00, if SPY Close_T > SMA200_T
                0.50, otherwise
```

The completed `Close_T` is included because the decision occurs after the close
and fills next session. Before a valid SMA200 exists, soft-regime variants open
no new position. The scale is fixed at entry; a later regime flip does not resize
or close an existing position.

## Frozen Variant Table

`B0_REF` is a reference row, not one of the four new proposals.

| Priority | Variant | Entry target weight for ETF `i` | Strict open cash cap | Primary control |
|---:|---|---|---|---|
| 0 | `B0_REF` | `1/N` | No; inherited behavior | Prior B0 parity |
| 1 | `P0_STRICT` | `1/N` | Yes | `B0_REF` |
| 2 | `P1_INVOL20` | `InvVolWeight_i,T` | Yes | `P0_STRICT` |
| 3 | `P2_SOFT200` | `MarketScale_T / N` | Yes | `P0_STRICT` |
| 4 | `P3_INVOL20_SOFT200` | `InvVolWeight_i,T * MarketScale_T` | Yes | `P0_STRICT`; also compare with `P1_INVOL20` and `P2_SOFT200` |

This produces `5 x 3 = 15` strategy rows: three reference rows plus 12 new
proposal/universe rows. The inspected paired-comparison family contains 18
tests: six frozen comparisons in each universe (`P0-B0`, `P1-P0`, `P2-P0`,
`P3-P0`, `P3-P1`, and `P3-P2`). No additional sizing window, SMA length,
multiplier, cap, or combination will be added after result inspection.

## Strict No-Borrowing Execution Rule

For `P0-P3`, the strategy still creates target shares after `Close_T`. At the
actual next open it then enforces cash feasibility using the real open, modeled
slippage, and modeled commissions:

```text
1. Cancel orders for a stale/no-print open.
2. Credit dividends earned by positions held before the open.
3. Calculate cash after all same-open exits, including costs.
4. Calculate the total cash required by all new-entry buys.
5. If required cash exceeds available cash, multiply every new-entry share
   delta by one common kappa in [0, 1].
6. Choose the largest kappa for which final cash is non-negative.
7. Execute exits and clipped entries at Open_T+1.
```

Same-open sale proceeds may fund buys. No shorting is present. A successful
strict row must have:

```text
minimum cash >= -$0.01
maximum gross exposure <= 100.0001%
negative-cash day count = 0
```

The original `B0_REF` deliberately retains the prior behavior so the effect of
the new hard constraint can be measured rather than hidden.

## Frozen Universes

| Universe | Symbols | Role |
|---|---|---|
| `spdr_9` | `XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY` | Long history from 1998/1999 |
| `vanguard_11` | `VAW,VDE,VFH,VIS,VGT,VDC,VPU,VHT,VCR,VOX,VNQ` | Broad-US family robustness |
| `spdr_11` | `XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLRE,XLC` | Current complete implementation from 2018 |

The common-overlap diagnostic begins when all three universes are simultaneously
available. The fixed endpoint remains `2026-07-17`.

## Costs And Corporate Actions

- Initial capital: `$100,000`.
- Slippage: `2.5 bps` per side, unchanged.
- Commission: `$0.00525/share`, no minimum, unchanged.
- Idle cash earns `0%`.
- ETF trade and signal prices use Norgate `CAPITALSPECIAL`.
- `$SPX` performance benchmark uses Norgate `TOTALRETURN`.
- `SPY` regime history uses Norgate `CAPITALSPECIAL`.
- ETF distributions are credited to shares held before the current open.
- Fractional shares remain enabled.

## Required Evidence

Every row must save daily equity, cash, realized weights, transactions,
completed trades, dividends, entry-sizing decisions, cash-cap events, benchmark
returns, and summary metrics. The study must report:

- CAGR, volatility, Sharpe, maximum drawdown, and MAR;
- correlation and beta to `$SPX` total return;
- turnover, modeled cost drag, trades, holding period, and closed-trade tails;
- average and maximum gross exposure, minimum cash, and negative-cash days;
- average entry target weight, largest entry weight, and concentration;
- full-history and common-overlap results;
- fixed subperiod and annual results;
- direct-control deltas and Newey-West/HAC paired-return tests;
- Bonferroni p-values across the 18 frozen comparisons;
- a diagnostic that scales `P0_STRICT` returns by the ex-post average-gross
  ratio of each variant. This is an analysis-only exposure comparator, not a
  tradable strategy and not evidence by itself.

## Decision Rules

- `P0_STRICT` succeeds mechanically only if every strict invariant passes. It
  becomes the clean control even if its performance is almost identical to
  `B0_REF`.
- `P1_INVOL20` or `P2_SOFT200` is useful only if it improves Sharpe or MAR in at
  least two of three universes, does not worsen maximum drawdown in at least two,
  and avoids a full/common CAGR sacrifice greater than two percentage points in
  at least two universes.
- `P3_INVOL20_SOFT200` is retained only if it improves Sharpe or MAR against at
  least one of its two component variants in at least two universes without a
  material new drawdown, concentration, or return penalty.
- A lower drawdown caused only by lower average exposure is not called improved
  alpha.
- No row is promoted from a single universe, a single crisis, or a p-value
  alone.
- If inverse-vol sizing mainly increases low-volatility sector concentration,
  that concentration must be reported rather than described as free risk
  reduction.
- `SPDR11` remains the implementation universe. The older families validate or
  reject the mechanism; they do not reconstruct modern `XLC`/`XLRE` history.

## Research Lineage And Limitations

This is not independent out-of-sample discovery. The B0/L200 study, several
earlier sector-dispersion universe studies, and an unrelated asset-SMA/VIX
position-sizing study have already been inspected. The four present rules were
proposed before this run, but the broader family search is large.

Opening fills remain a complete-fill model. Bid-ask history, opening-auction
participation, partial fills, capacity, and tax effects are not modeled. A
successful research row would still require execution-cost and capacity work
before any deployment claim.
