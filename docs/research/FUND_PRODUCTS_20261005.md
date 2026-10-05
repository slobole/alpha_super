# Fund products, final pass: GROWTH from three capsules, one two-pod monthly book, DEFENSIVE verified (2026-10-05)

Research record. Simulation only; not an allocation approval and nothing here is wired to LIVE.

Frozen plan: `scripts/research/fund_products_20261005/SPEC_FROZEN.md` (commit 70f5c22, reviewed by three independent reviewers before the freeze; its amendment log, written after results, is at its end). Code: the same folder. Outputs: `results/research/portfolio/fund_products_20261005/` (git-ignored). Report page (Hebrew, English statistics): the "fund products" artifact, version 5.2.

Owner decisions of 2026-10-05, all taken after the results and applied here: GR1 = TAA 3x 40 / 30 / 30 (amendment O1); GR2 = the same weights with TAA 3x 1N (O2); the monthly book is one book of two pods, TAA 3x 1N 60 / CORE5 40 (O3; Monthly Plus was dropped); GR3 = TAA 3x 1N 60 / 20 / 20 and the AGGRESSIVE rung was approved; the backtest is the headline, and the earlier "planning" and "floor" columns are replaced by one conservative case and one stress case; the Defensive-to-Growth blend is shown as its own dial.

## Summary

- **DEFENSIVE needs no update.** The stored inputs are byte-identical, the defensive legs' code is unchanged since the sleeve runs, and a re-run of the A6-d rules from an empty cache reproduced all 1,011 fields with zero differences. Only two side rows depended on the old growth book or on raw stock mean reversion; their capsule versions are shown beside them and replace nothing.
- **GROWTH is rebuilt from three capsules**, with no weight search: TAA 3x 40%, the momentum capsule (E2 with the sector cap) 30% and the MR capsule (DV2-G + HPI-G, BIL parking) 30%. The registered default was equal capital (one third each); on 2026-10-05, after the results, the owner chose the 40 / 30 / 30 dial point, a mild tilt to the TAA engine (TAA 3x is the pod that trades live today). Against equal capital (17.9% / 1.26 / -12.8%) it is a tie by the 80% rule on Sharpe (higher on 70% of paired paths, below the 80% bar) with a higher CAGR on 93% of paths; the price is more dependence on TAA. Backtest (the headline): CAGR 18.3%, excess Sharpe 1.27, max DD -13.0%. Conservative case (every engine keeps three quarters of its historical excess return, at model costs): CAGR 13.5%, excess Sharpe 0.95.
- **One ladder for more return.** GR2 is GR1 with the TAA engine's 1N variant at the same 40 / 30 / 30 weights: CAGR 21.1%, excess Sharpe 1.21, max DD -16.3% (higher excess Sharpe than GR1 on only 10% of paired paths, higher CAGR on 100%: it buys return, not Sharpe). GR3 raises the 1N variant to 60%: CAGR 23.4%, excess Sharpe 1.19, max DD -18.0%. GR3 adds 2.3 pp over GR2 and is offered by the menu rule (1.5 pp). It is the most return on the ladder and a one-engine product. GR3 passes only the AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%), which is new in this study and was approved by the owner on 2026-10-05. This is more TQQQ, not more alpha: TAA carries 46% / 56% / 77% of the risk in GR1 / GR2 / GR3.
- **One monthly book of two pods** (owner decision; designed after the results from an exploratory grid): Monthly = TAA 3x 1N 60 / CORE5 40: CAGR 19.8%, excess Sharpe 1.15, max DD -16.2%. Monthly trading only; TAA 3x 1N is wired (it has a live route but no pod is running; the pod trading live today, in the owner's account, is TAA 3x) and the only missing wiring is CORE5. It is the book that can run first and that scales. The Monthly book passes GROWTH PLUS and not GROWTH (29.6% of paths beyond -20%), so it carries more risk than GR1, not less. Its rung holds down to 0.80 of the edge (GR1 0.70); in the conservative case its breach figure at -25% is 18.2% against the 15% cap. Monthly Plus (65 / 35) was dropped. The four-pod monthly books of 2026-10-01 are kept as reference rows only.
- **GR1 against the Monthly book.** 18.3% / vol 12.9% / max DD -13.0% / excess Sharpe 1.27 against 19.8% / 15.6% / -16.2% / 1.15. GR1 has the higher excess Sharpe on 90% / 73% / 47% of paired bootstrap paths at 0 / +5 / +10 bps per side (above the 80% bar at model costs, below it at +5 bps), and by block on 56% (2008-2012, proxy era), 99% (2012-2021), 34% (2022-2026) and 27% (last three years). 2012-2021: excess Sharpe 1.56 vs 1.28; since 2022: 1.07 vs 1.17 and CAGR 17.7% vs 22.7% (a tie by the 80% rule, Monthly ahead on most paths). GR1 passes all seven pre-registered checks against it in the main frame.
- **Why GR1 stays the target: robustness to one engine failing.** A two-pod monthly book has one return engine and a defensive leg, and no backup. With the TAA engine's whole excess return removed GR1 keeps an excess Sharpe of 0.69; the Monthly book keeps only 0.14: that is the honest price of the two-pod book. The price of GR1 is daily trading, 5 research pods instead of two, an unmeasured execution cost and a low capacity.
- **Blend dial.** A client between the two base products holds the defensive launch and GR1 in fixed shares; the table below shows 0 / 20 / 40 / 60 / 80 / 100% GR1 with drawdowns (daily correlation of the two 0.73; not full diversification, both lean on the TAA engine).
- **Capacity (house model, pre-TCA).** With everything at the open, the only route the backtest models, the capsule products are about $2.5M and Monthly $5M. The difference is what can be worked: the monthly book's orders are monthly and can be worked over days ($235.9M (BTAL wall)), the MR capsule's daily stock orders cannot. With the MR stocks at the close auction the capsule products reach about $10M: an upper bound, because the engine does not model that route and the DV2 timing study rejected a same-day close.
- **Before client money**, in this order: wire CORE5 (it opens the defensive launch and the Monthly book); run the MR capsule in paper to measure slippage (the gate: at most 4 bps per side over at least 200 stock fills); the momentum capsule is optional, because GR1 with the live NDX rule in its place is almost the same book (correlation 0.99). If the MR gate fails, growth stays the Monthly book and a new frozen plan is needed. No growth book fits today's account.

## What the products are

Plain idea: three return engines with different mechanisms, each selected on the same history, so no weight may depend on an estimated return. Equal capital needs no estimate and was the registered default; the owner's 40 / 30 / 30 keeps every engine at 30% or more and gives the oldest engine a mild tilt. It is two risk clusters, not three independent bets: TAA (whose return leg is TQQQ) and the momentum capsule (Nasdaq-100 stocks) are both long Nasdaq when risk is on; the MR capsule buys S&P 500 dips in stress. The three capsule products are one ladder: GR1 (TAA 3x 40 / 30 / 30), GR2 (the same weights, a pure switch from TAA 3x to TAA 3x 1N), GR3 (TAA 3x 1N 60 / 20 / 20); the satellites always split the rest equally. The Monthly book holds one return engine (TAA 3x 1N) and one defensive leg (CORE5). Leverage on GR1 (fixed 1.40x) is shown as an alternative route to GR3's level of return for the fund stage; it is not a product.

| Product | Capital weights | Target rung | Strictest rung passed | YAML |
|---|---|---|---|---|
| GR1 Growth | TAA 3x 40%, momentum capsule 30%, MR capsule 30% (owner decision 2026-10-05) | GROWTH | GROWTH | `portfolios/fund_growth.yaml` |
| GR2 Growth Plus | TAA 3x 1N 40%, momentum capsule 30%, MR capsule 30% (owner decision 2026-10-05: the same weights, 1N variant) | GROWTH PLUS | GROWTH PLUS | `portfolios/fund_growth_plus.yaml` |
| GR3 Aggressive | TAA 3x 1N 60%, momentum capsule 20%, MR capsule 20% (owner decision 2026-10-05) | AGGRESSIVE | AGGRESSIVE | `portfolios/fund_growth_aggressive.yaml` |
| Monthly (runs first, scales) | TAA 3x 1N 60 / CORE5 40 (owner decision 2026-10-05) | none registered (designed after results) | GROWTH PLUS (holds down to 0.80 of the edge) | `portfolios/fund_growth_monthly.yaml` |
| Reference: Monthly of 2026-10-01 (four pods, not offered) | TAA 3x 1N 38.4 / NDX-VXN 25.6 / CORE5 18 / BTAL_QQQ 18 | GROWTH | GROWTH | `portfolios/fund_growth_monthly_20261001.yaml` |
| Reference: Monthly, more return, of 2026-10-01 (four pods, not offered) | TAA 3x 1N 57.4 / NDX-VXN 24.6 / CORE5 9 / BTAL_QQQ 9 | GROWTH PLUS | GROWTH PLUS | `portfolios/fund_growth_plus_monthly_20261001.yaml` |

Internal weights: momentum capsule = `ndx_atr_cap` 50 / `ndx_natr_cap` 50; MR capsule = `dv2_g` 50 / `hpi_g` 50 (one shared VIX stress gate, idle cash in BIL). Book model: pods compound independently and are reset to target weights at the first session of each calendar year (a transfer with no trade and no cost). Rungs: GROWTH = max DD >= -17% and breach figure at -20% <= 15%; GROWTH PLUS = -22% / -25%; AGGRESSIVE = -27% / -30% (new in this study, not yet formally confirmed by the owner). A rung passes only if the historical max DD and the bootstrap cap (10-seed mean and worst seed) hold both in the main frame and at +5 bps. No product needed added cash. In the JSON files the internal keys were not renamed: `S9 incumbent launch` holds the Monthly book, `old growth plus` holds the dropped 65 / 35 book (not a product), and the 2026-10-01 books are `S13 old monthly (2026-10-01)` and `old monthly plus (2026-10-01)`.

## Results (LONG window 2008-03-04 to 2026-08-19, gross, fair cash)

The backtest is the headline. The engine already charges IBKR commissions and 2.5 bps of slippage per side, and the headline frame credits idle cash, so the cards and tables of the report show the backtest only. What stays true: no figure is out of sample, every engine was selected on this history, and the product weights and the monthly book were chosen after the results. GR1's excess Sharpe was 1.56 in 2012-2021 and 1.07 since 2022. One conservative case is shown beside the backtest: every engine keeps three quarters of its historical excess return, at model costs. The earlier "planning" column also added 5 bps per side; for TAA and momentum (monthly orders) that double-counted a cost the engine already charges, and it is a real question only for the MR capsule's daily stock orders, which is what the paper-trading gate measures. The stress case is half the excess return and +5 bps per side. Both are conventions, not forecasts.

| Book | CAGR | Excess Sharpe | Vol | Max DD | Breach figure at its rung | Conservative case: CAGR / excess Sharpe | Stress case: CAGR / excess Sharpe | Net of 2/20 CAGR | CAGR at +5 bps | CAGR, engine cash 0% |
|---|---|---|---|---|---|---|---|---|---|---|
| GR1 Growth | 18.3% | 1.27 | 12.9% | -13.0% | at -20%: 2.4% | 13.5% / 0.95 | 8.3% / 0.59 | 12.8% | 17.0% | 18.1% |
| GR2 Growth Plus | 21.1% | 1.21 | 15.7% | -16.3% | at -25%: 3.5% | 15.4% / 0.91 | 9.4% / 0.57 | 15.1% | 19.9% | 20.9% |
| GR3 Aggressive | 23.4% | 1.19 | 17.9% | -18.0% | at -25%: 11.6%; at -30%: 2.0% | 16.9% / 0.89 | 10.4% / 0.57 | 16.9% | 22.5% | 23.3% |
| Monthly (TAA 3x 1N 60 / CORE5 40) | 19.8% | 1.15 | 15.6% | -16.2% | at -25%: 5.6% | 14.4% / 0.86 | 9.2% / 0.56 | 14.0% | 19.4% | 19.7% |
| Reference: Monthly of 2026-10-01 (four pods) | 18.1% | 1.14 | 14.3% | -13.1% | at -20%: 11.4% | 13.3% / 0.86 | 8.5% / 0.56 | 12.7% | 17.7% | 18.0% |
| Reference: Monthly, more return, of 2026-10-01 (four pods) | 21.4% | 1.13 | 17.4% | -16.9% | at -25%: 11.1% | n/a | n/a | 15.4% | 21.1% | 21.3% |
| Equal capital, 1/3 each (registered default) | 17.9% | 1.26 | 12.7% | -12.8% | at -20%: 2.8% | 13.2% / 0.94 | 8.1% / 0.58 | 12.5% | 16.6% | 17.8% |
| GR1 with the MR 30% in BIL | 13.5% | 1.12 | 10.7% | -11.8% | at -20%: 0.6% | 10.0% / 0.83 | 6.6% / 0.54 | 9.1% | 13.2% | 13.4% |

The Monthly book passes GROWTH PLUS and not GROWTH (29.6% of paths beyond -20%), so it carries more risk than GR1, not less. Its rung holds down to 0.80 of the edge (GR1 0.70); in the conservative case its breach figure at -25% is 18.2% against the 15% cap.

The breach figure is a yardstick on one convention (share of stationary-bootstrap paths with a deeper drawdown; mean block 63, full backtest edge), not the chance of that drawdown. For GR1 at -20% it is 2.4% by the rule, 9.3% at block 21, 44.7% with independent days, and 10.2% in the conservative case. GR1 still passes its rung down to 0.70 of the edge. GR2 holds GROWTH PLUS (-25%) down to 0.75 of the edge (conservative case 12.3% against the 15% cap). GR3 is read on the rung it actually passes, AGGRESSIVE (limit -30%): its figure is 2.0%, it holds down to 0.70 of the edge, the conservative case gives 9.6% against the 15% cap, and the figure is 7.8% at block 21 and 35.0% with independent days; at -25% (the GROWTH PLUS limit, which it does not pass) it is 11.6%, and its max DD is -18.0%. GR3 passes only the AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%), which is new in this study and was approved by the owner on 2026-10-05. TAA carries 77% of GR3's risk; the moderate alternative is GR2 (the same 1N variant at 40%, TAA share of risk 56%).

## The blend dial between Defensive and Growth

Two base products and one dial: a client between them holds the defensive launch (CORE5 54 / BTAL_QQQ 36 / cash 10) and GR1 in fixed shares, reset annually, and picks the point by the drawdown they can live with. Descriptive, in sample. The right-hand columns are GR1 diluted with BIL to about the same volatility: the test of whether the defensive book is worth more than cash.

| GR1 share | CAGR | Excess Sharpe | Vol | Max DD | Breach at -10% / -15% / -20% | GFC window | 2022 bear window | About the same volatility with BIL: vol / CAGR / excess Sharpe / max DD |
|---|---|---|---|---|---|---|---|---|
| 0% (defensive launch) | 8.2% | 1.12 | 6.1% | -5.6% | 5.1% / 0.0% / 0.0% | 2.9% | 1.9% | n/a |
| 20% | 10.3% | 1.25 | 7.0% | -6.0% | 7.0% / 0.1% / 0.0% | 1.4% | 0.5% | GR1 + 45% BIL: 7.3% / 10.8% / 1.26 / -7.1% |
| 40% | 12.3% | 1.29 | 8.3% | -7.1% | 19.1% / 0.3% / 0.0% | 0.0% | -1.0% | GR1 + 36% BIL: 8.4% / 12.3% / 1.26 / -8.3% |
| 60% | 14.4% | 1.30 | 9.7% | -8.5% | 48.8% / 1.9% / 0.1% | -1.4% | -2.5% | GR1 + 24% BIL: 9.9% / 14.3% / 1.26 / -9.9% |
| 80% | 16.3% | 1.29 | 11.3% | -10.8% | 89.8% / 9.1% / 0.4% | -2.7% | -4.0% | GR1 + 12% BIL: 11.4% / 16.3% / 1.27 / -11.4% |
| 100% (GR1) | 18.3% | 1.27 | 12.9% | -13.0% | 99.8% / 28.4% / 2.4% | -4.1% | -5.4% | n/a |

Reading: 3 of the 4 interior points have a higher excess Sharpe than both ends (1.25 to 1.30). Against cash dilution at about the same volatility (the match is approximate: the diluted books carry 1 to 4% more volatility) the blend's CAGR differs by -0.47 to +0.05 pp, its excess Sharpe by -0.01 to +0.03, and its max DD is shallower by about 0.5 to 1.4 pp on the one historical path; at 20% GR1 cash dilution is slightly better. A small advantage. It is not full diversification: GR1 and the defensive launch correlate 0.73 on all days and 0.74 on the worst 5% of S&P 500 days, because BTAL_QQQ in the defensive book runs on the same Defense First engine as TAA.

## The monthly book: why two pods, and why 60 / 40 (exploratory, run after the results)

The owner asked for the monthly book to be reduced to TAA 3x 1N and CORE5 and chose the 60 / 40 ratio. `monthly.py` ran a grid after the results (2 TAA variants x 4 TAA:CORE5 ratios x momentum 0 / 15 / 30%, a TAA 3x 1N / CORE5 ladder, two levered rows and four reference books). It is exploratory and not pre-registered; differences between neighbouring rows are small. The ladder rows show the trade-off behind the ratio: each step adds return and deepens the drawdown.

| Monthly book | CAGR | Excess Sharpe | Vol | Max DD | Breach at -20% / -25% | 2022 bear window | CAGR at +5 bps | GROWTH rung | GROWTH PLUS rung |
|---|---|---|---|---|---|---|---|---|---|
| TAA 3x 1N 40% / CORE5 60% | 15.6% | 1.18 | 11.8% | -11.6% | 2.8% / 0.2% | -5.3% | 15.2% | yes | yes |
| TAA 3x 1N 50% / CORE5 50% | 17.7% | 1.17 | 13.7% | -13.9% | 11.2% / 1.4% | -7.5% | 17.3% | yes | yes |
| TAA 3x 1N 60% / CORE5 40% | 19.8% | 1.15 | 15.6% | -16.2% | 29.6% / 5.6% | -9.7% | 19.4% | no | yes |
| TAA 3x 1N 65% / CORE5 35% | 20.8% | 1.15 | 16.6% | -17.4% | 41.8% / 9.6% | -10.7% | 20.4% | no | yes |
| TAA 3x 1N 70% / CORE5 30% | 21.8% | 1.14 | 17.5% | -18.7% | 54.6% / 15.2% | -11.8% | 21.5% | no | no |
| TAA 3x 1N 50 / CORE5 50 x1.25 (levered) | 21.2% | 1.15 | 16.9% | -17.5% | 44.5% / 10.5% | -9.9% | 20.7% | no | yes |
| TAA 3x 1N 40 / CORE5 60 x1.50 (levered) | 21.5% | 1.14 | 17.2% | -17.7% | 49.5% / 12.5% | -9.1% | 21.0% | no | no |
| Reference: old monthly (2026-10-01) | 18.1% | 1.14 | 14.3% | -13.1% | 11.4% / 1.2% | -9.5% | 17.7% | yes | yes |
| Reference: old monthly plus (2026-10-01) | 21.4% | 1.13 | 17.4% | -16.9% | 46.8% / 11.1% | -13.2% | 21.1% | no | yes |
| Reference: TAA 3x 57 / momentum 43, no CORE5 | 18.5% | 1.12 | 15.0% | -16.9% | 17.5% / 2.1% | -8.6% | 18.1% | no | yes |
| Reference: TAA 3x 40 / momentum 30 / CORE5 30 | 15.1% | 1.16 | 11.7% | -11.6% | 1.3% / 0.1% | -5.0% | 14.7% | yes | yes |

- BTAL_QQQ is nearly the same engine as TAA 3x 1N (daily correlation 0.85), so it adds no diversification; CORE5 correlates 0.48 with TAA 3x 1N.
- The momentum capsule replaces NDX-VXN everywhere in the fund, and in a monthly TAA + CORE5 book it does not earn a weight: over the 16 grid books with momentum, the share of paired paths with a higher excess Sharpe than the same book without it is 29% to 75%, never 80% (below 50% in 6 of the 8 books with the 1N variant); CAGR is equal or lower in 12 of 16. What momentum changes is the historical max DD (+1.0 to +4.5 pp, positive = shallower), and the 2022 window is worse in 16 of 16.
- CORE5 is needed: TAA 3x 1N alone has a max DD of -26.7% and a breach figure at -20% of 98.5% and passes no rung. TAA 3x 57 / momentum 43 without CORE5 also fails GROWTH, on the breach figure (17.5% against the 15% cap; max DD -16.9%).
- The new Monthly against the four-pod book of 2026-10-01: CAGR 19.8%, excess Sharpe 1.15, max DD -16.2% against CAGR 18.1%, excess Sharpe 1.14, max DD -13.1%; the new book has the higher excess Sharpe on 57% of paired paths (below the bar: a tie) and the old book the higher CAGR on 5%. Nearly the same book with two pods instead of four.
- The Monthly book passes GROWTH PLUS and not GROWTH (29.6% of paths beyond -20%), so it carries more risk than GR1, not less. Its rung holds down to 0.80 of the edge (GR1 0.70); in the conservative case its breach figure at -25% is 18.2% against the 15% cap.

## Evidence for the structure

**If one engine stops working** (its whole excess return removed, its volatility kept):

| Book | Backtest excess Sharpe | TAA dead | TAA and BTAL_QQQ dead | Momentum dead | MR dead | All at 3/4 (conservative case) | All at 1/2 |
|---|---|---|---|---|---|---|---|
| GR1 Growth | 1.27 | 0.69 | 0.69 | 0.97 | 0.93 | 0.95 | 0.63 |
| Equal capital, 1/3 each | 1.26 | 0.77 | 0.77 | 0.93 | 0.87 | 0.94 | 0.62 |
| GR2 Growth Plus | 1.21 | 0.58 | 0.58 | 0.98 | 0.94 | 0.91 | 0.60 |
| GR3 Aggressive | 1.19 | 0.34 | 0.34 | 1.05 | 1.03 | 0.89 | 0.59 |
| Monthly | 1.15 | 0.14 | 0.14 | 1.15 | 1.15 | 0.86 | 0.57 |
| Reference: Monthly of 2026-10-01 (four pods) | 1.14 | 0.45 | 0.31 | 0.93 | 1.14 | 0.86 | 0.57 |
| TAA 3x 50 / 25 / 25 | 1.27 | 0.56 | 0.56 | 1.03 | 0.99 | 0.95 | 0.63 |

The 40 / 30 / 30 book is not claimed to be minimax: the best worst case among the neighbours and challengers is 0.81 (TAA 3x 30 / 35 / 35) against 0.69 for GR1. The new Monthly holds one TAA pod and no BTAL_QQQ, so for it the column "TAA and BTAL_QQQ dead" equals "TAA dead"; in the four-pod reference book BTAL_QQQ runs on the same Defense First engine, so there that column removes both.

**Does each capsule earn its slot** (GR1 with one capsule's slot replaced by BIL; share of paired bootstrap paths on which GR1 has the higher excess Sharpe):

| Replaced | CAGR | Excess Sharpe | GR1 higher Sharpe at 0 / +5 / +10 bps | GR1 higher CAGR |
|---|---|---|---|---|
| Momentum -> BIL | 14.3% | 1.32 | 24% / 33% / 42% | 100% |
| MR -> BIL | 13.5% | 1.12 | 99% / 91% / 69% | 100% |
| TAA -> BIL | 10.2% | 1.08 | 97% / 99% / 100% | 100% |
| Momentum -> QQQ total return | 19.4% | 1.22 | 70% / 63% / 54% | 17% |

The MR capsule earns its slot; GR1 and GR1 without MR have the same excess Sharpe at about 13 bps of extra cost per side (the same CAGR at about 25 bps). The momentum capsule adds return, not risk-adjusted return: with BIL in its slot the book's excess Sharpe is higher on 76% of paths (below the 80% bar) and its CAGR is 4.0 pp lower. QQQ in the slot: CAGR 19.4%, max DD -20.9%, strictest rung passed GROWTH PLUS.

**Challengers.** 14 alternative structures were tested against GR1 with seven checks (rung, 80% paired share on 20,000 paths, breach, 2012+ window, +5 bps, both halves). Passed: none. GR1 with a defensive quarter (75 / 25): higher excess Sharpe on 85% of paths, failed checks: first half; its price is 2.2 pp less CAGR. More MR and less momentum (50 / 15 / 35; 86%) failed: breach, first half. No momentum (TAA 3x 50 / MR 50; 79%) failed: rung, 80% share, breach, first half. Equal capital does not pass against 40 / 30 / 30 (30% of paths): two nearly identical books. By the pre-registered rule no challenger replaces a product in this study; a pass would only have meant a forward-tracking candidate, and no pass means no evidence against the default, not confirmation. In the other direction GR1 passes every check against the Monthly book in the main frame (90% of paths on excess Sharpe, 19% on CAGR), .

**Dependence.** Daily correlations: TAA 3x-momentum 0.52, TAA 3x-MR 0.28, momentum-MR 0.39. Diversification ratio by half: 1.28 and 1.28 (largest correlation move 0.03). The capsules do fall together in the tail: P(B in its worst 5% | A in its worst 5%) is 36% / 21% / 28% on daily returns against 5% under independence. Effective number of bets: 2.25 (GR1), 2.02 (GR2), 2.02 (GR3).

**Exposure look-through (2012-10-02 on; the proxy era has no stored TQQQ series).** TQQQ weight inside the TAA 3x pod: mean 26%, 90th percentile 67%, max 101% (1N variant: mean 40%). Nasdaq-100 look-through notional (3 x TQQQ + momentum stocks) per unit of product NAV, mean / peak: GR1 0.55x / 1.51x, GR2 0.72x / 1.50x, GR3 0.88x / 2.01x at target weights. With the pod weights the books actually carried between annual resets the peaks were 1.69x / 1.73x / 2.20x (in 2013-12); since 2015 at most 1.22x / 1.46x / 1.83x. A one-day Nasdaq-100 fall of 10% at peak exposure costs about -18.0% (GR1) and -22.0% (GR3) by arithmetic (TQQQ = 3 x the index, momentum and MR stocks beta 1). The sample has no such day and no 2000-02 type Nasdaq bear; the bootstrap cannot produce one. On an average day about 26% of GR1 is in BIL or cash.

**Margin instead of the ladder** (debt as a negative-weight pod at DTB3 + 1.5%, reset annually; no margin calls or gaps modelled; L is the ratio of the unlevered volatilities, so the realised volatility of the levered book is slightly below the target's; tolerance 0.3 pp CAGR, 0.5 pp drawdown or breach). GR1 x 1.22 against GR2: CAGR 21.50% vs 21.09%, max DD -16.0% vs -16.3%, breach at -25% 1.6% vs 3.5%: the levered book is ahead beyond the tolerance; at +5 bps 20.00% vs 19.86%, at a 2.5% spread 21.26%. GR1 x 1.39 against GR3: 23.95% / -18.2% vs 23.40% / -18.0%: the levered book is ahead beyond the tolerance; at +5 bps 22.24% vs 22.48% (leverage also levers the MR capsule's costs). GR2 x 1.14 against GR3: CAGR 23.50% vs 23.40%, max DD -18.4% vs -18.0%: a tie inside the tolerance. No winner is declared: that is a judgement, not a tolerance result. Leverage raises the peak Nasdaq exposure and divides the capacity by L. With pods in separate Reg-T accounts only the ladder is practical; a fund with one cross-margined account can use either.

## Fee income (hedge-fund style schedules)

Model: mgmt accrued daily on NAV; performance fee on gains above the high-water mark, crystallised at calendar year end, no hurdle. AUM is held at $1M (income is linear in size). No fund expenses (administration, audit, legal) are included; at this size they can take most of the income. Each cell: average income a year on $1M in the backtest, then in the conservative case (3/4 of the excess return).

| Book | 2 / 20 | 1.5 / 20 | 1 / 15 | 1 / 10 | 1 / 5 |
|---|---|---|---|---|---|
| Defensive launch | $32K, $29K | $28K, $25K | $21K, $18K | $17K, $15K | $14K, $13K |
| Blend Growth 40 / Defensive 60 | $41K, $35K | $37K, $31K | $28K, $23K | $22K, $19K | $16K, $15K |
| Growth | $55K, $45K | $51K, $41K | $38K, $30K | $29K, $24K | $20K, $17K |
| Growth Plus | $61K, $49K | $57K, $45K | $43K, $33K | $32K, $26K | $21K, $18K |
| Aggressive | $67K, $53K | $63K, $49K | $47K, $36K | $35K, $28K | $23K, $19K |
| Growth x1.40 (leverage) | $68K, $54K | $64K, $50K | $47K, $37K | $35K, $28K | $23K, $20K |
| Monthly | $59K, $47K | $54K, $43K | $40K, $32K | $31K, $25K | $21K, $18K |

What the client keeps for Growth (CAGR / excess Sharpe after the fee; backtest, then conservative case):

| Schedule | Backtest | Conservative case | Share of the return above T-bills the fee takes (backtest / conservative) |
|---|---|---|---|
| 2 / 20 | 12.8% / 0.88 | 9.1% / 0.63 | 32% / 36% |
| 1.5 / 20 | 13.3% / 0.91 | 9.5% / 0.66 | 29% / 32% |
| 1 / 15 | 14.6% / 1.01 | 10.6% / 0.74 | 22% / 24% |
| 1 / 10 | 15.4% / 1.07 | 11.2% / 0.78 | 17% / 19% |
| 1 / 5 | 16.3% / 1.13 | 11.8% / 0.83 | 12% / 14% |

Recommendation as written on the page: 1 / 15 for the growth products and 1 / 10 for the defensive product and the blends, with a high-water mark and no hurdle. Reasons: the fee should take at most about a quarter of the return above T-bills (about a third in the conservative case) (Growth at 1 / 15: 22% and 24%); at 2 / 20 the Growth client keeps an excess Sharpe of 0.88 (0.63 conservative) against 0.61 for the S&P 500 and 0.77 for QQQ, hard to sell without a live record; the defensive product cannot carry a high fee (at 2 / 20 the fee takes 48% of its return above T-bills, at 1 / 10 26%); the lever is AUM, not the fee percentage. Two caveats: no fund expenses are included, and a lawyer comes first (who may charge a performance fee, and to whom, depends on licensing and investor type).

The report page no longer shows a header line or net-of-fees rows and columns; fee effects are in its fees section only. The net of 2/20 column in the results table above is kept in this record for reference.

**Leverage at a fixed 1.40x as an alternative to GR3** (not a product; fund stage only, one cross-margined account). GR1 x 1.40: CAGR 24.10%, excess Sharpe 1.24, max DD -18.4%, breach at -25% / -30% 6.1% / 0.8%, 2022 window -8.6%, at +5 bps 22.37%, at a 2.5% financing spread 23.67%, TAA-dead excess Sharpe 0.66, peak Nasdaq look-through 2.11x. GR3: CAGR 23.40%, excess Sharpe 1.19, max DD -18.0%, breach at -30% 2.0%, 2022 window -13.7%, at +5 bps 22.48%, TAA-dead 0.34, peak look-through 2.01x. Verdict at the tolerance (0.3 pp CAGR, 0.5 pp drawdown or breach): the levered book is ahead beyond the tolerance. It keeps the three-engine balance instead of concentrating in TAA; its costs are financing at DTB3 + 1.5%, no margin calls or gap days in the model, the MR capsule's trading cost levered too, and capacity divided by 1.4.

## Capacity and ease (pre-TCA)

| Book | House model: open (modelled) | close auction | worked + blocks (upper bound) | Per-leg P99 order = 5% of a median day (binding leg) | Largest same-day order, all pods = 5% | BTAL 10%-ownership wall | Pods: in research / in the live build | Not wired | Minimum clean size |
|---|---|---|---|---|---|---|---|---|---|
| GR1 Growth | $2.5M | $5M | $10M | $2.5M (TAA 3x, BTAL) | $2M | $235.9M | 5 / 4 | DV2-G, HPI-G, NDX ATR cap, NDX NATR cap | $667K |
| GR2 Growth Plus | $2.5M | $5M | $10M | $2.5M (TAA 3x 1N, BTAL) | $1.8M | $353.8M | 5 / 4 | DV2-G, HPI-G, NDX ATR cap, NDX NATR cap | $667K |
| GR3 Aggressive | $2.5M | $2.5M | $25M | $1.7M (TAA 3x 1N, BTAL) | $1.3M | $235.9M | 5 / 4 | DV2-G, HPI-G, NDX ATR cap, NDX NATR cap | $1M |
| Monthly | $5M | $5M | $235.9M (BTAL wall) | $1.7M (TAA 3x 1N, BTAL) | $1.3M | $235.9M | 2 / 2 | CORE5 | $75K |
| Reference: Monthly of 2026-10-01 (four pods) | $2.5M | $5M | $250M or more | $2.6M (TAA 3x 1N, BTAL) | $1.4M | $253.4M | 4 / 4 | CORE5 | $391K |

The house model's opening-auction limits are strict (0.05% of volume) and uncalibrated. The worked route works the monthly ETF orders over days (BTAL as blocks) and sends the MR stock orders to the close auction, which the engine does not model and the DV2 timing study rejected, so for the capsule products it is an upper bound; there the binding leg is the MR capsule (FOX, NWS). The capsule products are labelled CAPACITY-LIMITED by the pre-registered AUM rule (worked route below $25M); the Monthly book is not. The MR pods' BIL parking orders are left out of the gates (amendment C1); the BTAL wall is computed on peak BTAL weights (the TAA pods hold BTAL). The Monthly book needs only CORE5 wired. The MR capsule's stock turnover is about 41 times the pod a year.

## DEFENSIVE

Kept as published on 2026-10-01 (launch CORE5 54 / BTAL_QQQ 36 / cash 10 and the other slots). Two rows are shown beside target-stage versions that replace nothing:

- More return: the published 2026-10-01 row (growth slice = the four-pod monthly book of that date, kept as published) CAGR 9.36%, max DD -6.1%; the same scan with GR1 as the growth slice picks 30% growth inside the invested part and 20% cash (24.0% of capital in growth): CAGR 9.77%, excess Sharpe 1.27, max DD -5.5%. The same code with the old growth book reproduced the stored slot exactly. The GR1 pick is a boundary point: it clears the -5% crisis floor by 0.05 pp; the next passing points are within 0.2 pp of CAGR, which is noise.
- Gated upgrade: 60/40 at 90% + MR capsule 10% (cash 5%) beats the launch on 100% / 98% / 89% of paths at 0 / +5 / +10 bps, against 98% / 90% / 68% for the stored raw-HPI row. It stays behind the MR slippage gate.

## Data verification

- Stored sleeves re-run at HEAD from a clean worktree: taa3x, taa3x_1n, core5, btal_qqq, ndx_vxn, etf_dv2, eom_flow, downshock byte-identical to the stored files; dv2 and hpi_vote differ by at most 5.5e-5 a day on about 540 days in 2024-2026 (Norgate revision; CAGR -0.0005 / -0.0014 pp), inside the threshold.
- The E2 book is bit-identical to the stored 2026-10-04 run; the MR capsule's stored $100K runs reproduce byte for byte; the study's $1M capsule runs are within 0.05 pp of CAGR and correlate 0.9996.
- The runs are causal in the end date (two end-date tests). The main checkout's uncommitted edits (another session's live wiring) are backtest-neutral and no stored series came from that tree.
- The predecessor study's four-pod monthly book of 2026-10-01 reproduces exactly in the new pipeline (breach figure at -20% 11.41% in both). `a6d.py` reproduces `a6d.json` (1,011 fields, 0 different) and `report_a6.py` reproduces `report_a6.json` (14,899 fields, 0 different); `a6.py` differs from `a6.json` in 7 crisis-drawdown fields of the old growth rows (up to 0.36 pp) because `a6.py` was edited after `a6.json` was written on 2026-10-01; no defensive field differs.
- Engine confirmation (each product YAML through the PortfolioManager, engine cash 0%, 2013-01-02 on, against the research book model): GR1: CAGR 20.73% vs 20.73%, correlation 1.0000, accepted; GR2: CAGR 23.76% vs 23.76%, correlation 1.0000, accepted; GR3: CAGR 26.58% vs 26.59%, correlation 1.0000, accepted; Monthly: CAGR 22.64% vs 22.64%, correlation 1.0000, accepted.
- After the owner's decisions of 2026-10-05 the whole pipeline was re-run and the page and this record were checked by two independent reviewers (numbers of Growth Plus and of the two monthly books of that version rebuilt with separate code; all matched). The later decisions (GR3 60 / 20 / 20, Monthly 60 / 40, leverage 1.40 as an alternative) were re-run and checked by the author only.

## Important to know (direction and size)

| Convention or caveat | Direction | Size |
|---|---|---|
| Engine, Bench and YAML books pay 0% on idle cash; the headline frame credits DTB3 - 0.5% and charges DTB3 + 1.5% on negative cash | engine conservative | CAGR gap, percentage points a year: GR1 0.13%; TAA 3x alone 0.03% (about 2% cash), momentum capsule 0.29%, MR capsule 0.10% |
| The products are not fully invested | note | about 26% of GR1 is BIL or cash on an average day |
| Three cash treatments live side by side: the MR capsule's BIL is a real position (25% withholding, trade costs); TAA and momentum idle cash is credited DTB3 - 0.5% without cost; the cash leg of a book is BIL total return | conservative for MR | about 0.6 pp of capsule CAGR; GR1 0.17% under the symmetric treatment |
| TAA before 2012-10-02 is a synthetic TQQQ / BTAL proxy (a quarter of the sample, the only 2008) | unknown | GR1 on 2012+ only: CAGR 20.3%, excess Sharpe 1.40 |
| TAA commissions on split-adjusted TQQQ shares | conservative | about 0.3 pp a year per TAA pod on average 2012-2026 (0.38 pp in the 1N variant), near zero now |
| Negative cash is not financed in the engine | optimistic, small | MR pods 167 and 168 days, to -9.6% of the pod; momentum pods about 1792 days, to -1.9%; TAA pods about 1,500 days (2012 on), to -1.8%; charged DTB3 + 1.5% in the headline frame |
| Two of the three engines are long Nasdaq when risk is on (TAA through TQQQ, momentum through Nasdaq-100 stocks); no 2000-02 type bear in the sample | optimistic | peak look-through 1.51x (GR1), 2.01x (GR3) at target weights; 1.69x and 2.20x as carried; a 10% one-day Nasdaq fall at peak exposure costs -18.0% and -22.0% |
| Every engine was selected on this history; nothing is out of sample (MR: about 110 variants; momentum: about 45 trials plus a 151-configuration grid; the products: 3 of 14 books seen before the freeze) | optimistic | the backtest is the headline (GR1 CAGR 18.3%, excess Sharpe 1.27); conservative case, 3/4 of the excess return at model costs: 13.5% / 0.95; GR1's excess Sharpe was 1.56 in 2012-2021 and 1.07 since 2022 |
| The product weights were chosen by the owner after the results, from the pre-registered dial points (GR1 and GR2 at 40 / 30 / 30; the registered default was equal capital; the 40 / 30 / 30 point was not in the grid seen before the freeze) | optimistic | GR1 against equal capital: a tie by the 80% rule on Sharpe (70% of paired paths) |
| The Monthly book (TAA 3x 1N 60 / CORE5 40) was designed after the results from an exploratory grid that was not pre-registered; the owner chose the ratio | optimistic | differences between neighbouring grid rows are small; one return engine: with TAA dead the Monthly book keeps an excess Sharpe of 0.14 (GR1 0.69) |
| Breach figures are a block-63, full-edge convention | optimistic | GR1 at -20%: 2.4% by the rule, 10.2% in the conservative case |
| MR capsule cost sensitivity; no live fills; the slippage gate (4 bps per side over 200 fills) is unmet; 2020-21 are about a third of its log-wealth; stock turnover about 41 times the pod a year | optimistic until measured | capsule alone 16.8% -> 13.7% at +5 bps; GR1 18.3% -> 17.0% |
| Momentum capsule: today's GICS labels; selection unproven against QQQ at the same exposure; in a drawdown of about 18% from its June 2026 peak (at 2026-10-02) | unknown | in the slot test BIL in its place has the higher excess Sharpe on 76% of paths and 4.0 pp less CAGR: it adds return, not risk-adjusted return |
| Annual reset is a cost-free transfer on one date | optimistic, small | GR1 CAGR over the twelve start months 18.2% to 18.6% |
| Capacity is pre-TCA and route-dependent | unknown | GR1 $2.5M at the open (modelled), up to $10M if the MR stocks trade at the close (upper bound, not modelled); binding legs: TAA / BTAL on one-day routes, MR stocks (FOX, NWS) on the worked route; Monthly $5M at the open, $235.9M (BTAL wall) worked |
| The Monthly book's rung and its margin | optimistic | The Monthly book passes GROWTH PLUS and not GROWTH (29.6% of paths beyond -20%), so it carries more risk than GR1, not less. Its rung holds down to 0.80 of the edge (GR1 0.70); in the conservative case its breach figure at -25% is 18.2% against the 15% cap. |
| Leverage on GR1 (fixed 1.40x) is an alternative, not a product | optimistic | financing at DTB3 + 1.5%, no margin calls or gap days modelled, leverage also levers the MR capsule's trading cost and divides capacity by 1.4; not available while each pod sits in its own Reg-T account |
| Minimum clean size | note | GR1 about $667K, Monthly about $75K |
| Wiring | note | wired: TAA 3x, TAA 3x 1N, BTAL_QQQ, NDX-VXN (of these only TAA 3x has a pod trading live today; TAA 3x 1N has a live route but is not running); not wired: CORE5 (PM_READY) and the four momentum and MR capsule pods (PM_READY, no live route at 5c0d48d; the MR capsule needs a new order shape and two margin accounts); the Monthly book needs only CORE5 |
| Margin rows | optimistic | DTB3 + 1.5%, no margin calls, no gap; volatility match is approximate; capacity divides by L |
| Gross against net of 2/20 | note | GR1 18.3% gross, 12.8% net; Monthly 19.8% gross, 14.0% net; no fund expenses |
| Sharpe basis | note | GR1 excess Sharpe 1.27 daily, 1.48 monthly, 1.37 at a zero rate |
| The weeks after the window are not out of sample | note | 2026-08-20 to 2026-10-02: GR1 1.1%, Monthly 3.9% |
| The engine BIL pod is not BIL total return | note | 1.08% vs 1.61% a year from 2012 (withholding, no reinvestment) |

## Owner decisions

Decided:

1. DECIDED 2026-10-05: GR1 = TAA 3x 40 / momentum 30 / MR 30 is the growth product to build toward (`fund_growth.yaml`). The reason is engine diversification, not a proven Sharpe gain.
2. DECIDED 2026-10-05: GR2 (Growth Plus) = the same 40 / 30 / 30 with the TAA 3x 1N variant (`fund_growth_plus.yaml`).
2b. DECIDED 2026-10-05: GR3 (Aggressive) = TAA 3x 1N 60 / momentum 20 / MR 20 (`fund_growth_aggressive.yaml`; it was 50 / 25 / 25). The ladder is TAA 3x 40, TAA 3x 1N 40, TAA 3x 1N 60.
3. DECIDED 2026-10-05: one monthly book of two pods: Monthly = TAA 3x 1N 60 / CORE5 40 (`fund_growth_monthly.yaml`). Monthly Plus was dropped (`fund_growth_monthly_plus.yaml` deleted). It runs first and is the scalable alternative; if the MR slippage gate fails, growth stays the Monthly book. The four-pod books of 2026-10-01 are reference rows.
3b. DECIDED 2026-10-05: the AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%) is approved; GR3 passes only that rung.
3c. Leverage on GR1 at a fixed 1.40x is shown as an alternative route to GR3's level of return for the fund stage (one cross-margined account); it is not a product on the menu and no decision is asked.
4. DECIDED 2026-10-05: the backtest is the headline; one conservative case (3/4 of the excess return, model costs) and one stress case (1/2, +5 bps) are shown in the "how much to believe" section only.

Open:

2. Run the MR capsule in paper or small live and measure slippage: the gate is at most 4 bps per side over at least 200 stock fills.
3. Wiring order: CORE5 first (it opens the defensive launch and the Monthly book), then the MR capsule (in progress; paper first), then, optionally, one momentum strategy that averages the two books; the live NDX rule is an acceptable stand-in for the momentum capsule.
4. MR execution route: at the open GR1 is a product of about $2.5M; a close-auction route would lift it to about $10M at most and has no supporting evidence yet.
5. Commit and merge of this study (it lives on a worktree branch).

## Independent review

Version 5.2 (this record): after the owner's decisions of 2026-10-05 (Growth 40 / 30 / 30, Growth Plus 40 / 30 / 30 with the 1N variant, two-pod monthly books, the backtest as the headline; later the same day: GR3 = 60 / 20 / 20, one Monthly book at 60 / 40, the AGGRESSIVE rung approved) the study was re-run and the page and record then checked by two independent reviewers: all numbers of Growth Plus and of the two monthly books of that version were rebuilt with separate code and matched; the one gap they found, the monthly books' thin rung margin, is now stated on the page and here, and about twenty wording and labelling items were corrected.

History. The three rounds below were run on the equal-capital version; after the owner's 40 / 30 / 30 decision for GR1 the whole study was re-run, the engine confirmation repeated and the updated page and record reviewed by two more independent reviewers (the GR1 numbers were rebuilt with separate code; leftover equal-capital wording was corrected). The equal-capital outputs are kept in `results/research/portfolio/fund_products_20261005/report_equal_capital_snapshot/`. Three rounds by independent agents with their own code. (1) Before the freeze, three reviewers blocked the first draft of the plan (incomplete disclosure of what had been seen, the MR gate dropped, undefined rules); it was rewritten and then frozen. (2) Eight data audits: every stored sleeve re-run at HEAD, input mapping, freshness, causality. (3) Five reviewers of the results: two rebuilt every headline number from the raw path and transaction files (272 book comparisons, largest difference below 1e-6); a timing and look-ahead audit found none (the VIX gate is aligned to the prior close, 100% of DV2-G buy fills fall on gate-open sessions; the new sleeves re-run to 2019-12-31 match the full runs to 1e-11). No headline number was wrong. What changed after the review, all toward less optimism (figures of the equal-capital version; "monthly book" there means the four-pod book of 2026-10-01): the GR1-over-monthly share is shown at +5 and +10 bps and by block and the summary was rewritten; capacity was corrected (close-auction capacity $10M to $5M, per-leg participation $34.5M to $3.0M); the carried Nasdaq peak is shown beside the target-weight peak; GR3 is read at -25% as well as -30%; the four-pod monthly book's dead-engine figure includes BTAL_QQQ (0.31); the margin reading, the crisis-window labels and the amendment log were fixed. The timing reviewer's own estimate of a halved Nasdaq premium at time-average exposure (equal-capital GR1 11.9% / 0.85; not recomputed for 40 / 30 / 30) is quoted beside the study's regression-beta row; it is harsher than the conservative case used here (equal capital at 3/4 of the excess return: 13.2% / 0.94).

## Computed, in the JSON files, not on the report page

Correlation matrices for blocks B and C, each half and the five crises (`battery.json` dependence); rolling-correlation P10 and P90; the ten worst 21-session windows of GR2 and GR3; breach figures at -10, -15, -17, -22, -27 and -35% with worst-seed values (`study.json` tails; GR1 at -17%: 11.3%, worst seed 12.6%); gap tables of the dial-map and margin books (`exposure.json`); gross CAGR percentiles (`battery.json` bootstrap); the full monthly-book grid with momentum at 15% (`monthly.json`); the old "planning" rows (3/4 of the excess return and +5 bps) are still in `battery.json` edge_decay.headline.

## Files

Capsule products: `portfolios/fund_growth.yaml` (GR1), `fund_growth_plus.yaml` (GR2), `fund_growth_aggressive.yaml` (GR3). Monthly book: `portfolios/fund_growth_monthly.yaml` (Monthly, TAA 3x 1N 60 / CORE5 40); `fund_growth_monthly_plus.yaml` was deleted with the Monthly Plus book. The four-pod books of 2026-10-01 are kept under dated names: `portfolios/fund_growth_monthly_20261001.yaml` and `portfolios/fund_growth_plus_monthly_20261001.yaml`. Deleted: `fund_growth_mr.yaml` (its HPI-RSI pod was demoted). This work lives on a worktree branch; nothing was changed in the main checkout or pushed.

## Forward review triggers (fixed in the plan)

A trigger opens a review; it is not an automatic exit: a capsule's live or paper drawdown beyond its planning maximum (momentum -30%, MR -21%, TAA -26%); MR slippage above 4 bps per side over 200 fills; a capsule behind BIL over a rolling three years.

## Reproduce

From the worktree root, with `PYTHONDONTWRITEBYTECODE=1`: `python scripts/research/fund_products_20261005/build_sources.py` (six engine runs), then in that folder `study.py`, `battery.py`, `exposure.py`, `capacity.py`, `defensive.py`, `versus.py`, `pm_confirm.py write`, the PortfolioManager runs, `pm_confirm.py compare`, `monthly.py`, `build_report.py`, `build_record.py`. The study imports the shelf-rebuild library from the main checkout (`scripts/research/shelf_rebuild_20260929/lib.py`, untracked there; its hash is in the study ledger).

