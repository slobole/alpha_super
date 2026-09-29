# Mean reversion beyond DV2 — verdict (2026-09-26)

Research-only. Nothing in live trading, releases, schedulers, brokers or registry tiers changed.
Owner request (Hebrew, 2026-09-26): find a "real" mean-reversion strategy worth trading, with a reason behind it,
given that DV2 behaves like crisis convexity rather than ordinary mean reversion and HPI is highly correlated with it;
other universes than the S&P 500 are welcome. Frozen plan and every amendment:
[`scripts/research/mr_beyond_dv2_20260926/SPEC_FROZEN.md`](../../scripts/research/mr_beyond_dv2_20260926/SPEC_FROZEN.md).
An independent quant-pitfalls review ran before this version; its corrections are included.

## Verdict

**No new mean-reversion pod passes, and none is recommended. The owner's diagnosis of DV2 is right, and it is not
specific to DV2: in liquid US markets, short-horizon mean reversion is a premium for supplying liquidity, paid mainly
when liquidity is scarce.**

1. **The crisis convexity is structural.** Six pre-registered families were tested: stock reversal, forced index
   flows, cross-asset ETFs, ETF relative value, calendar flows and Treasury auctions. They cover 5 point-in-time
   stock universes, 60+ ETFs and about 4,750 cells/configurations. No continuous mean-reversion rule has a calm-market
   edge that beats costs.

   In the S&P 500 the DV2 signal earned +38 bps per trade over the average stock in the 1990s, +28 in the 2000s,
   and 0-4 bps in every block since 2010. Price-reversal signals earn 21-33 bps per trade in high-VIX periods and
   −4 to +5 bps in low- and mid-VIX periods, within 2010-2026 alone. Measuring dips against the stock's sector ETF,
   moving to mid/small caps, or using ETF pairs does not change this. It is the pattern Nagel (2012, "Evaporating
   Liquidity") documented: reversal profits are pay for bearing liquidity risk, and they rise with the VIX.
2. **Mean reversion that is paid regularly would have to come from flows that arrive on a schedule.** This study
   found such effects, but none is ready to trade:
   - **S&P 500 post-inclusion fade.** Stocks new to the S&P family lose vs SPY in the 5 sessions after they enter the
     S&P 500: −2.1% in the 1990s (t −3.2), −1.7% in 2000-2026 (t −4.4), −1.6% for tradable names only (t −4.1). This
     is the one robust new flow effect, but it needs a short side.
   - **Turn of year.** Nasdaq-100 deletions and December's losers rebound in early January. Both were strong before
     2013 and have faded since: the tax-loss rebound is +0.4-0.6% (t < 1.1), and the deletion rebound's mean fell from
     6.9% to 2.8%.
   - **Month-end Treasury cycle.** TLT tends to rise into month-end and fall in the first days of the month. A
     no-signal rule (long TLT the last 5 sessions, short the first 5) earns net Sharpe 0.74, with 0.80 in 2003-2014
     and 0.68 in 2015-2026.
3. **The existing month-end rebalancing flow pod (EOM flow) is a forward-test candidate, not a recommendation to
   trade.**
   - **What it shows, in-sample, at base cost:** correlation 0.14 with DV2 and HPI, beta 0.13, calm-market Sharpe
     1.17, Sharpe 1.11. Adding it at 10% raises the growth book's Sharpe from 1.51 to 1.57.
   - **What this study did not run on it:** the +5 bps/side stress, G4-G7, or a deflated Sharpe, which the new
     candidates had to pass.
   - **Its timing and signal:** its close-auction timing was chosen after other timings were seen, and next-open
     versions reached only Sharpe 0.42-0.76. 39% of its daily variance is the no-signal month-end Treasury cycle
     (correlation 0.63). It does keep 6.5%/yr of return that the cycle doesn't explain (t 4.0, in-sample).

   Its already-planned small paper test is the right gate.
4. **HPI is a second copy of DV2's premium** (daily correlation 0.73 with DV2-ADV). Only one stock mean-reversion pod
   is needed to hold that premium. The book with HPI swapped for EOM (Sharpe 1.59, Calmar 2.08) is an in-sample,
   post-hoc illustration. It is not a tested change and should wait for EOM's forward evidence.

## Why DV2 and HPI look like crisis convexity

Stock-level reversal is the market paying whoever absorbs other people's urgent trades. Stat-arb funds and market
makers compete for that pay, so in calm markets almost nothing is left for a daily-bar trader filling at the next
open. When volatility jumps, their risk limits bind, fewer of them absorb the flow, and the reward for absorbing it
becomes large. A long-only dip buyer also carries market beta, which in calm years is most of its return. DV2, HPI and
QPI are three entry rules for the same trade, which is why they correlate 0.74-0.87.

Bottom decile of each signal among the liquid half of the S&P 500 (the DV2 floor), bps per trade, excess over the
average eligible stock, next-open entry and exit. VIX terciles are computed over 1990-2026 (< 15.2, 15.2-20.8,
> 20.8); the last three columns repeat the split within 2010-2026 only.

| Signal (5-day hold, hedge) | 1991-99 | 2000-09 | 2010-14 | 2015-19 | 2020-22 | 2023-26 | VIX low / mid / high | 2010-26: low / mid / high |
|---|---|---|---|---|---|---|---|---|
| DV2(126), none | 37.6 | 28.3 | 0.2 | 2.0 | 4.4 | 2.4 | 5.5 / 5.3 / 23.9 | 2.2 / −1.7 / 6.9 |
| 5-day return, SPY | 38.9 | 29.9 | 2.3 | 4.3 | 17.2 | 14.0 | 4.9 / 0.0 / 42.5 | 1.5 / −2.7 / 33.2 |
| 5-day residual vs SPY, SPY | 31.6 | 27.3 | −0.7 | 1.6 | 14.6 | 6.4 | 2.4 / 0.6 / 34.5 | 0.3 / −4.0 / 22.0 |
| 5-day residual vs sector ETF, sector | — | 17.9 | 0.7 | 2.4 | 20.1 | 5.8 | 3.7 / −1.1 / 27.5 | 1.2 / −2.7 / 24.8 |
| 5-day intraday-only return, SPY | 38.5 | 22.5 | 4.1 | 4.8 | 16.8 | 10.3 | 6.3 / 5.7 / 27.4 | 5.4 / 1.3 / 21.2 |

The round-trip cost hurdle is 8-10 bps (engine costs; 10 bps when hedged).

## How it was tested

- **Pre-registration.** Question, mechanisms, universes, signals, gates and cost hurdles were frozen before any
  result. Seven dated amendments and a results log follow. Families E and F were added after A, C and D failed; they
  are disclosed as such and their trials counted. The study files are not yet committed, so the amendment order is
  documented only in the file.
- **Data.** Norgate US equities incl. delisted, point-in-time membership (S&P 500 from 1990, Russell from 1990-07,
  Nasdaq-100 from 1993-10). Signals, fills and marks use CAPITALSPECIAL; dividends come from the Dividend field;
  liquidity is Unadjusted Close × Volume. There are 77 ETFs and $VIX. Treasury auction dates come from the US
  Treasury Fiscal Data API (2,788 note and bond auctions since 1979) and FRED yields cover the holdout.
- **Windows.** Main window 2000-01-03 → 2026-08-19 in blocks: 2000-09, calm C1 2010-14, calm C2 2015-19, stress
  2020-22, calm C3 2023-26. Holdouts are 1991-1999 (stocks) and 1983-2002 (Treasury yields).
- **Costs.** Engine default (2.5 bps per side plus $0.005/share, $1 minimum). ETF borrow is 0.5%/yr. Longs receive
  75% of dividends and shorts pay 100%.
- **Causality.** Every feature on row t uses data through Close_t. Beta is measured through t−1 for day t's residual.
  Sector assignment comes from the last month-end strictly before t. Fills are at Open_{t+1}.
  `tests/test_research_mr_beyond_dv2_features.py` perturbs every price after t and checks that features up to t do
  not change. A spot check on AAPL matched an independent pandas computation to float precision.
- **Screens.** A cell was LIVE only if the t-statistic of (edge − cost hurdle) was ≥ 2 in the calm pool and the edge
  was positive in each calm block (the strict reading, amendments A1/A3). The Newey-West t uses Bartlett weights with
  h−1 lags; uniform weights would give 5-15% lower t. This makes the negative verdict more conservative and inflates
  the near misses slightly.
- **Search size.** Family A 4,080 cells, B 224 event cells plus 56 pods, C 30, D 22, E 312, F 14 plus 2 holdout
  tests and 2 post-review controls: about 4,750 in total. Nothing is promoted; the positive effects above are
  labelled forward hypotheses.

## Results by family

### A. Stock reversal, including residual vs sector ETF, hedged — fails everywhere

Five universes were tested: S&P 500 (selection), Nasdaq-100, S&P 400, S&P 600, and Russell 1000 excluding same-day
S&P 500 members. Each ran 14 signals × 4 holds × 3 hedges × 2 liquidity tiers × 2 bucket sizes: **0 of 3,360 cells
pass.** The best calm-pool net t was 1.43, in S&P 600 small caps, whose real trading costs exceed the modeled 10 bps.
Pod-relevant hedged returns in calm years are ≤ ~12 bps per trade everywhere. Nasdaq-100 shows ~40 bps SPY-hedged,
but that is the 2010-2026 tech drift: only about 1-4 bps of it is excess over other Nasdaq-100 names.

The conditioning splits are diagnostic only:
- Dips on low abnormal volume reverted more than high-volume dips (S&P 500, 21-day sector residual, 5-day hold,
  calm: +13.4 vs −1.5 bps), in the direction Medhat & Schmeling (2022) predict, but barely above costs.
- Losers in rising sectors did not revert more.

### B. Forced-flow rebound after index exits — real only in the Nasdaq-100, and fading

There were 41,704 membership changes in six indices. Before the tradability filter, exits to no index showed very
large rebounds; the S&P 500 group averaged +17% over 20 days. Those rebounds came from distressed names under $5
(35-50% of events; GGP at $0.35 in 2008). After the filter added in amendment A5 (raw close > $5, ADV63 > $5M), only
the Nasdaq-100 keeps a robust effect:
- 243 tradable exits in 2000-2026, mean +4.8% and median +2.8% SPY-hedged over 20 sessions, t 3.6, 60% positive.
- Two thirds are in December (mean +4.9%, median +3.3%); the other months give mean +4.8%, median +1.8%, t 1.7.
- The mean fell from +6.9% (2000-2012) to +2.8% (2013-2026).

Migrations into sibling indices (S&P 500 → 400, Russell 2000 → 1000) show no rebound.

| Pod (20 slots, 20-day hold, unhedged) | CAGR | Sharpe | Max DD | Invested | Corr DV2 |
|---|---|---|---|---|---|
| Nasdaq-100 exits (frozen pick) | 2.5% | 0.79 | −7.2% | 26% of days | 0.09 |
| S&P 500 / 400 / 600 / Russell 1000 exits | 0.0-0.7% | 0.02-0.23 | | | |
| All groups pooled (10 slots) | 8.2% | 0.75 | −27% | 53% | 0.13 |

The pick passes G1 (different), G2 (calm) and G8 (book: Sharpe 1.506 → 1.520, Calmar 1.918 → 1.937), but misses G3
(Sharpe 0.79 < 0.8). It fails G4: the same rule on the other index groups has Sharpe ≤ 0.23. **Rejected as a pod.**

The additions control is the stronger flow effect. After inclusion, stocks new to the S&P family lose vs SPY over 5
sessions: −2.1% in 1991-99 (t −3.2), −1.7% in 2000-26 (t −4.4), −1.6% for tradable names only (t −4.1). Migrations up
from the S&P 400 lose only −0.35%.

A labelled exploratory long/short pod ran long tradable deletions against short S&P 500 and Nasdaq-100 additions,
SPY-hedged, with 20 slots and a 20-day hold. It returned 4.3% CAGR, Sharpe 0.70, max DD −12.5%, correlation −0.09
with DV2. It needs single-stock shorts (gap G-007) and cannot be promoted from this history.

### C. Cross-asset ETFs (bonds, commodities, currencies, real estate) — fails

- **Commodities and currencies** do not mean-revert at these horizons; most cells are negative, consistent with
  trend behavior.
- **Bonds** revert slightly (+10 bps per 5-day trade, 2010-26, t 1.7).
- **Equity ETFs** revert more (3-day z < −1.5: +26 bps over 5 days, t 1.9), but concentrated in high-VIX days
  (+66 vs −6 bps).

No class passes.

### D. ETF relative value (industry vs sector, country vs region) — fails

All pairs pooled: +17 bps (5 days) and +25 bps (10 days) per trade, gross t 2.8-3.0. Net of the 12 bps hedged hurdle
the t is 0.9-1.5, and the edge lives in high VIX (+66 bps vs −2 in the mid tercile). Health care pairs are the best
group (+50 bps over 10 days, net t 1.9) and still fail.

### E. Calendar flows in stocks — fails

- **Month-end losers** do not rebound more than losers on ordinary days; they do slightly worse (−2 to −10 bps). The
  window-dressing story finds no support.
- **December year-to-date losers,** bought at the first January open, beat SPY by +161 bps over 5 sessions (26 years,
  t 2.05, net t 1.92). This is the best of 12 cells for this effect. It was +259 bps in 2000-2012 but only +63 bps
  (t 1.1) in 2013-2026. The Nasdaq-100 shows the same pattern: +159 bps overall, +43 bps (t 0.6) since 2013.

### F. Treasury auction supply — fails as mean reversion

After 20/30-year bond auctions (TLT) and 10-year note auctions (IEF), no post-auction window is positive enough; all
are about 0 bps and inside the placebo band. The pre-auction control window looked striking at first: TLT −33 bps
over Close_{A−5} → Close_{A−1}, t −3.2, beyond all 200 placebo runs. But the placebo matched only the month, not the
position in it, and auctions fall on trading day 7-9 of the month, exactly where TLT's month-end cycle is negative.

Each auction's pre-window was then compared with non-auction windows ending on the same trading day of the month
(`stage1_f.py dom`):
- **Long bonds:** −20 bps, t −1.9. It appears only in 2015-2026 (−26 bps, t −2.2), not in 2003-2014 (−3 bps).
- **10-year notes:** −0.4 bps, t −0.1.

The untouched 1983-2002 yield holdout, as declared in A7, gives +2.5 bps (t 1.5) for both maturities. The first draft
quoted a drift-adjusted +2.9/+3.0 bps (t 1.7/1.8), which A7 did not declare. **So the auction effect is weak and
mostly the month-end calendar. It is dropped as a separate finding.**

## EOM flow: what it is and why it is only a forward-test candidate

Month-end rebalancing flow (`strategies/taa_beyond_6040/strategy_taa_month_end_rebalancing_flow.py`, PM_READY
research, gap G-032) works in five steps:
1. At the close 7 sessions before month-end, it measures how far a 60/40 SPY/IEF portfolio has drifted.
2. It ranks that drift against prior months only.
3. From that close through month-end it holds a scheduled close-auction position, which can include a TLT short
   (1% borrow).
4. At month-end it flips to SPY or TLT.
5. It exits at the 5th session of the next month.

Orders use prior-close data only. The stated mechanism: balanced funds and pensions must sell the winner and buy the
loser into month-end, and the pressure reverses.

In-sample, same gates, inventory run 2003-01-24 → 2026-08-19 at base cost:

| | EOM flow | DV2 (wired) | HPI vote | Industry-ETF DV2 / sector dispersion (for reference) |
|---|---|---|---|---|
| CAGR / Sharpe / max DD | 11.2% / 1.11 / −13.8% | 22.2% / 1.03 / −30.9% | 17.0% / 1.09 / −17.7% | 5-8% / 0.9 / −10 to −20% |
| Sharpe by block, 2000s / C1 / C2 / 2020-22 / C3 | 1.00 / 1.56 / 0.97 / 1.11 / 0.83 | 1.09 / 0.87 / 0.79 / 1.28 / 1.07 | 1.19 / 0.82 / 0.68 / 1.78 / 1.02 | |
| Correlation with DV2-ADV / HPI | 0.14 / 0.14 | 0.87 / 0.74 | 0.74 / 1 | 0.48-0.56 |
| Beta to S&P 500 TR | 0.13 | 0.77 | 0.57 | 0.11-0.35 |

Why this is not enough:
- **Unequal testing.** This study did not put EOM through the +5 bps/side stress, the out-of-sample and plateau
  gates G4-G5, the luck test G6, the capacity gate G7, or a deflated Sharpe. Every new candidate had to pass those.
- **Timing chosen on the data.** Its close-auction timing was chosen after other timings had been seen (see
  `docs/research/month_end_rebalancing_flow.md`). Next-open versions in earlier studies reached Sharpe 0.42-0.76,
  and the live close-auction path is not built.
- **Partly a calendar effect.** 39% of its daily variance is the no-signal month-end Treasury cycle (correlation
  0.63). The cycle alone: net Sharpe 0.74, CAGR 7.1%. EOM keeps 6.5%/yr beyond it (t 4.0, in-sample).

Growth book (TAA 32 / NDX 32 / DV2-ADV 9 / industry-ETF DV2 9 / HPI 18, annual reset, 2012-10-02 → 2026-08-19;
drawdown including the 2008 proxy in brackets). Every row is in-sample and at base cost:

| Book | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---|---|
| G3 (TAA 50 / NDX 50) | 23.2% | 1.42 | −14.8% | 1.57 |
| Current G3+MR (DV2-ADV 9 + ETF 9 + HPI 18) | 20.7% | 1.51 | −10.8% (−11.0%) | 1.92 |
| + EOM flow 10% | 19.8% | 1.57 | −9.7% (−10.2%) | 2.05 |
| + Nasdaq-100 exits 10% (frozen pick) | 19.0% | 1.52 | −9.8% | 1.94 |
| + exploratory flow long/short 10% | 19.0% | 1.53 | −9.7% | 1.96 |
| HPI 18 → EOM flow 18 (post-hoc illustration) | 19.5% | 1.59 | −9.4% | 2.08 |

## What this does not show

- **Seen history.** 2000-2026 has been studied heavily for mean reversion in this repo. The 1990s block was printed
  in the family A, B and E tables, so only the 1983-2002 yield holdout was untouched.
- **Survivorship bias in membership.** The engine membership loader drops the last 5 membership rows of past index
  members. Recomputed with raw membership, the effect is negligible and of mixed sign: at most 1.2 bps per block.
- **Costs.** Costs are the house model. Small-cap and event-stock costs are likely higher, which favors the negative
  verdict.
- **Timing.** All mean-reversion entries are next-open. The DV2 study showed that the part of the edge between the
  close and the next open is not reachable with a realistic 15:45 decision.
- **Hedge proxy.** Before SPY (1991-92) the hedge uses the $SPX price index, whose Opens equal the prior Close. The
  SPY-hedged holdout rows mistime the hedge slightly, which mostly cancels in the excess.
- **ETF lists** are today's funds (survivorship).

## Next steps (need your approval)

1. **Keep DV2-ADV as the single crisis-liquidity stock pod.** Do not expect regular returns from any stock
   mean-reversion variant. HPI adds little diversification.
2. **EOM flow:** run its planned small paper test with scheduled close-auction orders, measuring fills against the
   official close. Before any allocation change, also run it through the gates this study skipped: +5 bps/side, a
   calendar-control test (does its signal beat the no-signal month-end Treasury cycle out of sample?), and G4-G7.
3. **If you are open to a short side:** a separately pre-registered study of the S&P 500 post-inclusion fade is the
   best new lead. It needs announcement dates, borrow cost and capacity.

## Reproduce

Scripts in `scripts/research/mr_beyond_dv2_20260926/`:
- `cache_build.py` builds the Russell 1000, Nasdaq-100 and ETF caches (the S&P caches are reused from the DV2 study).
- `features.py` computes the causal features.
- `stage1_a.py` runs the stock map.
- `stage1_b.py` builds the index events (`events`) and their screen statistics (`stats`).
- `stage1_cd.py` runs the ETF class and pair maps.
- `stage1_e.py` runs the calendar map.
- `stage1_f.py` covers the auctions: `download`, `screen`, `placebo`, `holdout`, and `dom` (day-of-month control).
- `sim_events.py` + `stage2_b.py` run the event pods.
- `stage3_book.py` runs the book test.
- `report_tables.py` produces every remaining number in this report: the family-A tally, the signal table, the event
  diagnostics, and the EOM gates with the calendar control.
- `pod_stats.py` holds the metrics.

Outputs are in `results/research/mr_beyond_dv2_20260926/` (gitignored). Test:
`tests/test_research_mr_beyond_dv2_features.py`.
