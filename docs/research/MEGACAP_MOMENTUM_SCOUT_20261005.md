# S&P 500 mega-cap momentum with volatility normalisation (Scout, 2026-10-05)

**Owner question (2026-10-05):** leadership changes, so why not hold each month the momentum leaders among the
biggest S&P 500 stocks? After a first screen showed drawdowns of 77% to 88%: why so deep, and does
volatility-normalised sizing fix it?

**Status: WATCHLIST / diagnostic. Not a candidate. S5 (MCPT) was not run; nothing is quoted outside the seal.**

- Registration: `sp500_megacap_momentum_voltarget_20261005`, ledger row 38, family
  `equity_cross_sectional_momentum`, 96 configurations, `prior_trials_int` 29 (the Pakal screen
  `pakal-research/reports/megacap_momentum_scout`, which saw 1991 to 2026-09). The family's vault is contaminated.
- Script: `scripts/research/megacap_momentum_20261005/run.py`. Results: `results/scout/megacap_momentum_20261005/`
  (`grid.csv`, `summary.json`, `daily_returns.parquet`). Panel `S&P 500` snapshot `4ecbebd98bf5e8af`, sealed.
- Window: 1999-02-01 to 2022-12-30. Scout weights engine, next open, house parity costs, USD 100K, whole shares,
  idle cash at T-bills. Sharpe at a zero risk-free rate.
- Rule and formulas: the script's docstring. "Biggest" = 252-session median dollar Turnover (the panel has no
  point-in-time market capitalisation), so the pool leans to heavily traded, volatile names.

## 1. Why the drawdown was so deep

The raw book is fully invested, equal weight, in the most-traded names, with no brake.

| 1999-02 to 2022-12, no overlay | CAGR | Vol | Sharpe | Max DD | Peak to trough |
|---|---|---|---|---|---|
| Top 10 by 12-1 among the 50 most traded | 7.2% | 28.7% | 0.39 | −76% | 2000-03-27 to 2002-10-07 |
| All 50 most traded, equal weight | 3.8% | 24.1% | 0.28 | −78% | 2000-09-01 to 2009-03-09 |
| SPY | 6.6% | 19.8% | 0.42 | −55% | 2007-10-09 to 2009-03-09 |

In March 2000 the most-traded names with the best 12-month return were the dot-com leaders. The book held ten of
them at 10% each through a 2.5-year bear market. The drawdown is the pool and the missing brake, not the ranking:
the pool itself lost 78%.

## 2. What each overlay does (mean of the 24 configurations in each cell)

| Volatility target | SPY 200-day gate | Sharpe | CAGR | Vol | Max DD | Mean exposure |
|---|---|---|---|---|---|---|
| none | off | 0.43 | 8.2% | 27.1% | −71% | 99% |
| 15% | off | 0.55 | 7.8% | 16.1% | −41% | 73% |
| none | on | 0.70 | 12.9% | 20.3% | −36% | 73% |
| 15% | on | 0.75 | 9.5% | 13.3% | −21% | 57% |

- **Inverse-volatility weights inside the book do almost nothing:** mean Sharpe 0.61 against 0.61 for equal
  weight, mean Max DD −42% against −43%. Ten momentum leaders fall together; reweighting them does not help.
- **The portfolio volatility target halves the drawdown** (−71% to −41%) at a cost of about 0.4 points of CAGR.
- **The market gate does more than the target** (−71% to −36%) and also raised the return in this window.
- **Ranking on momentum per unit of volatility beats raw momentum** in every cell: mean Sharpe 0.64 against 0.57,
  mean P(> matched control) 0.83 against 0.66.

## 3. Does the stock selection survive once the control gets the same overlays?

The matched control is the whole pool under the same weights, target and gate.

- 87 of 96 configurations have a higher Sharpe than their matched control; 42 of 96 reach P ≥ 0.80.
- Plateau configuration (highest neighbourhood median, 0.82): top 50, N = 5, momentum per unit of volatility, equal
  weight, 15% target, gate on. It is also the grid peak and sits on the edge of the grid (the smallest N).

| | CAGR | Vol | Sharpe | Max DD | Dot-com bear | GFC | COVID crash | 2022 bear |
|---|---|---|---|---|---|---|---|---|
| Plateau configuration | 11.6% | 13.8% | 0.87 | −20% | +5% | +4% | −11% | −10% |
| Same, N = 10 | 11.2% | 13.4% | 0.86 | −21% | | | | |
| Matched control (pool of 50) | 7.1% | 11.7% | 0.64 | −20% | | | | |
| SPY | 6.6% | 19.8% | 0.42 | −55% | | | | |

P(plateau > matched control) = 0.95. The three registered kill criteria are not triggered.

**By era, the selection is a first-decade result:**

| Plateau configuration | Sharpe | CAGR | Control Sharpe | SPY Sharpe / CAGR | P(> control) |
|---|---|---|---|---|---|
| 1999 to 2008 | 1.16 | 14.4% | 0.64 | 0.02 / −1.8% | 1.00 |
| 2009 to 2022 | 0.70 | 9.7% | 0.65 | 0.76 / 13.0% | 0.62 |
| 2013 to 2022 | 0.58 | 7.8% | 0.70 | 0.76 / 12.5% | 0.30 |

Since 2009 the book is the gated pool with no measurable selection edge, and it trails SPY by about 3 points a
year at two thirds of SPY's volatility. This is the pattern of the NDX pod (alpha 1% to 2% a year since 2013,
[momentum decision](MOMENTUM_DECISION_20261004.md)) and of the Pakal S&P 500 momentum port.

## 4. Verdict and limits

- **Drawdown question: answered.** A 15% volatility target plus the SPY 200-day gate takes the book from −76% to
  about −20%. The overlays work on the pool with or without the ranking.
- **Selection: not established.** It passes the registered in-sample rule on the strength of 1999 to 2008. The
  2009 to 2022 evidence is a coin flip. With 96 configurations here, 29 earlier ones, and a peak on the grid edge,
  the 0.87 is an upper bound, not a planning number.
- **Not done:** S5 MCPT on the point-in-time panel, luck band over rebalance offsets, S6 book value, double-cost
  stress, the post-2022 period (sealed; the Pakal screen saw it for the raw book only).
- **Decision implied by the momentum decision record:** the momentum slot is closed and belongs to the NDX family.
  This line adds a second US large-cap momentum sleeve with the same decay. No further variants; forward tracking
  from the registration timestamp is the only clean evidence.
- Causality: weights of 628 decisions recomputed on panels truncated at four dates match exactly.

## 5. Follow-up (2026-10-05): per-stock CORE5 adaptive filter and a VIX-level gate

Owner question: why not hold a stock the way CORE5 holds an asset (its own adaptive trend filter), and only when
VIX is below 20 (as an example)?

- Registration `sp500_megacap_momentum_stock_filter_vix_20261005` (ledger row 39, child of row 38), 72
  configurations plus 24 controls. Script `run_filters.py`, results `results/scout/megacap_momentum_20261005/filters/`.
- Fixed from the parent: the 50 most-traded members, ranking on momentum per unit of volatility. Fixed slots of
  1 / N; an empty slot stays in cash. Same window, engine and costs.
- **Data fix before reporting:** the first run read VIX from 2011 only, which switched every VIX gate off before
  2011. VIX is now loaded from 1998 and the script refuses gaps. The numbers below are from the corrected run.

**Stock filter on the ranked book (mean over N = 5, 10, 20):**

| Market gate, 15% target | No stock filter | Close > SMA100 | CORE5 adaptive | P(adaptive > none) |
|---|---|---|---|---|
| None | 0.61 / −40% | 0.56 / −42% | 0.54 / −45% | 0.09 |
| SPY 200-day | 0.80 / −20% | 0.80 / −22% | 0.79 / −23% | 0.31 |

(Sharpe / Max DD.) The filter adds nothing once the book is ranked on momentum: the top-ranked names are already
above their own trend, so the filter rarely removes one. This matches the Nasdaq-100 bake-off (38 filters,
Romano-Wolf p 0.19). Registered rule: **the adaptive filter is dropped** (mean P 0.26 against the 0.80 bar).

**The owner's idea without the ranking (every pool name held only while its own filter is on, slot 1 / 50):**

| No volatility target | Sharpe | CAGR | Max DD | Mean exposure | P(> no filter) |
|---|---|---|---|---|---|
| No filter, no gate | 0.28 | 3.8% | −78% | 98% | |
| CORE5 adaptive, no gate | 0.41 | 4.7% | −48% | 68% | 0.91 |
| Close > SMA100, no gate | 0.43 | 4.1% | −41% | 54% | 0.89 |
| CORE5 adaptive + SPY 200-day gate | 0.68 | 7.1% | −16% | 57% | 0.82 |
| No filter + SPY 200-day gate | 0.63 | 8.1% | −27% | 73% | |
| SPY buy and hold | 0.42 | 6.6% | −55% | 100% | |

Here the stock filter does its job as a brake: it cuts the drawdown of an unranked basket (−78% to −48%; with the
market gate −27% to −16%). It does so by holding less (57% to 68% invested), and the return stays at 5% to 7% a
year. The plain SMA100 does the same as the adaptive filter.

**VIX-level gate against the SPY 200-day gate (ranked book, no stock filter, mean over N):**

| Gate | Months on | Sharpe, no target | Max DD | Sharpe, 15% target | Max DD | P(> SPY gate) |
|---|---|---|---|---|---|---|
| SPY 200-day | 74% | 0.78 | −31% | 0.80 | −20% | |
| VIX < 20 | 57% | 0.49 | −32% | 0.58 | −22% | 0.02 to 0.03 |
| VIX < 25 | 76% | 0.43 | −65% | 0.56 | −34% | 0.01 |

**Both VIX gates are dropped.** The VIX level is not a trend signal. It stayed in the low twenties for much of the
2000 to 2002 decline, so VIX < 25 kept the book invested through the dot-com bear (−65%). It was above 20 during
the 2003 and 2009 recoveries, so VIX < 20 kept the book in cash while prices rose (CAGR 6% against 14%).

**Conclusion:** the parent overlay (SPY 200-day gate plus the 15% target) stays the best of the tested brakes. The
verdict in section 4 is unchanged. Causality: 374 decisions recomputed on truncated panels match exactly.

## 6. Owner clarification (2026-10-05): no momentum ranking; the N biggest names, each behind its own adaptive gate

The owner's actual idea: hold the N biggest stocks, with no ranking, and let each stock's CORE5 adaptive trend
filter decide whether it is held or its slot sits in cash.

- Registration `sp500_biggest_n_adaptive_entry_gate_20261005` (ledger row 40, family
  `time_series_trend_and_breakout`), 96 configurations. Script `run_biggest.py`, results
  `results/scout/megacap_momentum_20261005/biggest/`. Same panel, window, engine and costs. Monthly decision.

**The idea as stated (no market gate, no volatility target; a name with its filter off leaves its slot in cash):**

| N biggest | No filter | CORE5 adaptive gate | Close > SMA100 gate | P(adaptive > none) | P(adaptive > SPY) |
|---|---|---|---|---|---|
| 5 | 0.33 / 5.7% / −85% | 0.37 / 5.6% / −63% | 0.44 / 6.4% / −48% | 0.61 | 0.36 |
| 10 | 0.25 / 3.1% / −88% | 0.41 / 6.0% / −55% | 0.43 / 5.6% / −49% | 0.92 | 0.46 |
| 20 | 0.21 / 2.0% / −87% | 0.40 / 5.1% / −55% | 0.44 / 4.9% / −43% | 0.96 | 0.39 |
| 50 | 0.28 / 3.8% / −78% | 0.41 / 4.7% / −48% | 0.43 / 4.1% / −41% | 0.91 | 0.44 |
| SPY | 0.42 / 6.6% / −55% | | | | |

(Sharpe / CAGR / Max DD, 1999-02 to 2022-12.)

- **The gate clearly helps the basket:** Sharpe 0.21-0.33 to 0.37-0.41, drawdown −78..−88% to −48..−63%.
- **It does not beat SPY:** about the same Sharpe, return and drawdown as buying the index (P 0.36 to 0.46).
- **The adaptive part earns nothing over a plain average:** Close > SMA100 is as good or better in every row
  (P(adaptive > SMA100) 0.13 to 0.42).
- **Refilling an empty slot with the next-biggest name whose filter is on is harmful:** the book stays fully
  invested and the drawdown returns to −77..−93%. The benefit of the gate is the cash, not the swap.
- **By era** (adaptive gate, N = 10): 1999-2008 Sharpe −0.10, CAGR −3.5% (SPY 0.02, −1.8%); 2009-2022 Sharpe
  0.77, CAGR 13.2% (SPY 0.76, 13.0%). The stock-level gate did not protect in the dot-com decade: decisions are
  monthly, the filter lags, and the most-traded names of 2000 fell fast and whipsawed.

**With the market-level brakes added (SPY 200-day gate + 15% target):** no filter 0.61 to 0.73, adaptive gate 0.64
to 0.70, drawdown −16% to −22% against −20% to −26%. P(adaptive > none) 0.34 to 0.68: once the market gate is
on, the stock gate adds a little drawdown protection and no Sharpe.

**Registered verdict: not a candidate.** Mean P(adaptive > no filter) is 0.42 over the grid (0.80 needed; it is
0.61 to 0.96 only in the plain "cash" cells), mean P(> SPY) is 0.58, and the adaptive filter does not beat the
SMA100 benchmark filter. Reading: any stock-level trend filter turns a dangerous concentrated basket into
something index-like; it is a brake, not an edge. Causality: 320 decisions recomputed on a truncated panel match.
"Biggest" is still the most-traded proxy (no point-in-time market capitalisation).
