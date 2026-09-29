# "Buy IPOs at all-time highs" and the same rule after stock splits - verdict (2026-09-28)

Research only. Nothing in live trading, releases, schedulers, brokers or registry tiers changed.
- **Owner request** (Hebrew, 2026-09-28): a quant study of the strategy in a pasted talk, in two variants:
  1. IPOs, as in the talk;
  2. stocks that did a split.
- **Frozen plan:** [`IPO_SPLIT_ATH_PREREG_20260928.md`](IPO_SPLIT_ATH_PREREG_20260928.md).
- **Amendments:** [`..._AMENDMENTS.md`](IPO_SPLIT_ATH_PREREG_20260928_AMENDMENTS.md).
- **Code:** `scripts/research/ipo_split_ath_20260928/`. **Results and charts:** `results/research/ipo_split_ath_20260928/`.

## Verdict

**Neither variant is a pod for this book, and the talk's performance is not reproducible without hindsight.**

1. **IPO variant.**
   - **The edge existed in 1993-1999 and has not been there since.** After an IPO closes at an all-time high:
     - 1990s: it beat SPY by +4.1% over the next 20 sessions (clustered t 3.4 versus other all-time highs);
     - 2000-2011: -0.6%; 2012-2021: -1.5%; 2022-2026: -6.9%.
   - **Over the full period the pre-registered monthly-clustered test is negative (t -1.4).** Each month counts once
     in it.
     - Weighting months by their event counts gives t +1.5. That is still below 2.
     - The talk's style of test, which treats every event as independent, gives t +9.0 on the same data.
     - The gap comes from two things: 1990s events dominate the pool, and same-day events are not independent.
   - **Pod** (the talk's final setup: 20 slots, +20% target, 10% trailing stop):
     - 1993-2026: CAGR 2.4%, Sharpe 0.28, max DD -38%;
     - 1993-99: +16.8% a year; since 2001: about -0.5% a year.
2. **Split variant.**
   - **No edge in any period.** Stocks at an all-time high within 90 sessions after a forward split do slightly worse
     than other all-time-high stocks (t -2.0). They also do worse than post-split stocks that are not at a high
     (t -4.1).
   - **Pod:** CAGR -0.8%, Sharpe -0.03, max DD -59% (in 2000-2011).
3. **Book test.**
   - In the book's quarter slot (TAA 0.5, NDX 0.25, slot 0.25), T-bills give Sharpe 1.365 for 2012-26. IPO gives
     1.279 and split 1.342.
   - None of the 30 settings tried beats T-bills in that slot (Reality Check p 0.998).
   - Both fail the frozen rule on R1, R3 and R4.
4. **Why the talk looks so good: probably look-ahead in its universe.**
   - The talk used Sharadar and kept only "mid, large and mega caps". Sharadar's market-cap tier is based on each
     company's most recent market cap. Used as a historical filter, it keeps only the IPOs that later grew large.
   - We rebuilt that filter from each company's last known market cap (a deliberate look-ahead diagnostic, added after
     the verdict). The same rule then gives, for 2001-2023:
     - **17.0% a year, Sharpe 1.35, max DD -15%, 43% winners, +3.2% per trade, 63% of capital invested (the 1993-2026
       average);**
     - the talk reports about 18%, 1.4, under 20%, 67%, 44% and +3.7%.
   - The same run also reproduces the talk's best month, October 2021: the Trump SPAC (DWAC).
   - With the point-in-time universe the same years give -0.5% a year.
   - This is strong circumstantial evidence, not proof. We do not have the talk's code.

![Equity since 2001](../../results/research/ipo_split_ath_20260928/charts/equity_2001_2026.png)

## What was tested

**The rule, as described in the talk.** Carlos's interpretation of Parker's words; the missing details were fixed
before any result.
- **Universe.** US operating companies, current and delisted (21,259 stocks from Norgate). A stock is eligible on a
  day when both hold:
  - its 20-day median dollar volume ranks in the top 1,000;
  - its price is at least $5.

  Norgate has no historical market cap, so this is the honest stand-in for "mid, large and mega cap".
- **Variant 1, IPO.**
  - The stock was first listed 1-89 sessions ago. Its listing date is after 1990, when Norgate's history starts.
  - It was not an uplisting from OTC.
  - It was not a SPAC at listing. Including SPACs changes nothing: +2.3% versus +2.4% a year.
- **Variant 2, split.**
  - A forward split took effect 0-89 sessions ago. Splits are detected from the adjustment factor and must be a clean
    ratio such as 2:1 or 3:2.
  - The stock has at least one year of history.
  - The detector finds 8,954 splits, including every well-known 2020-2024 split.
- **Signal.** Today's close is at or above every earlier close since listing. Carlos's first-day exclusion applies.
- **Trading.**
  - Buy at the next open, largest dollar volume first, into free slots.
  - Budget per slot: min(pod value / N, cash / free slots).
  - Profit target: a sell limit at entry +P.
  - Trailing stop: a sell stop at (1 - L) x the highest close since entry, placed each evening for the next day.
  - If a gap opens through a level, the fill is at the open. If one day touches both levels, the stop is assumed
    first.
  - No other exits.
- **Costs.** The engine's costs: 2.5 bps slippage per side, $0.005 a share with a $1 minimum, dividends at 75%. Idle
  cash earns T-bills (BIL).
- **Settings.**
  - The talk's final setup (N 20, +20%, -10%) is the only candidate; it was not chosen from our data.
  - All 15 combinations of N in {10, 20, 40} and the talk's five target/stop pairs are reported as the search.

**Checks passed before any result was read:**
- 30 unit tests.
- A download-date test: features rebuilt from data "as of" 2001, 2010 and 2020 are identical (685 checks, 0
  failures).
- A random-rescaling test (300 checks, 0 failures).
- Engine parity: the repo's real engine, replaying the rule for 2015-2019, gives the same 578 transactions and the same
  daily NAV (correlation 1.000000).
- One real bug was found this way and fixed before any result. Norgate gives active stocks no "last traded" date, so
  the first build silently dropped every active stock.

## Stage A - is there an edge? (event study)

Next-open entry. Excess = the stock's 20-session return minus SPY's over the same dates. Clustered t: each month's
average gap between the population and the comparison group, Newey-West t over months.

| Group | Events | 1993-99 | 2000-11 | 2012-21 | 2022-26 | Full |
|---|---|---|---|---|---|---|
| IPO at all-time high, 20-day excess vs SPY | 14,933 | +4.06% | -0.58% | -1.46% | -6.93% | +1.98% |
| Split at all-time high | 18,816 | +0.57% | -0.11% | +0.21% | +0.05% | +0.32% |
| Other stocks at all-time high (baseline) | 279,730 | +0.28% | +0.50% | -0.14% | +0.01% | +0.16% |
| Any IPO day, 1-89 (second baseline) | 98,374 | +3.77% | -1.88% | -1.34% | -4.66% | +1.13% |
| **IPO-ATH minus baseline: clustered t** | | **+3.41** | -1.60 | -0.57 | -1.95 | **-1.37** |
| Naive t (the talk's test) | | +15.2 | -2.4 | -2.6 | -3.5 | **+9.0** |
| **Split-ATH minus baseline: clustered t** | | -0.39 | -2.59 | -1.51 | +0.11 | **-1.98** |

- **The full-period IPO mean (+1.98%) looks positive only because 62% of all events fall in 1993-99.**
  - Month by month, the IPO group trails the baseline in 56% of months.
  - Weighting each month by its event count (review check, not pre-registered): D +1.08%, clustered t 1.47.
  - The top 1% of 1993-99 events supply a third of that block's excess.
- **Being at a high adds nothing to being young.** IPO-at-high versus any IPO day: t -2.2.
- The pre-registered test for "edge present" required t > 2 on the full period and a positive gap in at least 3 of the
  4 blocks. IPO: t -1.37, 1 of 4 blocks. Split: t -1.98, 1 of 4 blocks.

![Excess by year](../../results/research/ipo_split_ath_20260928/charts/stage_a_excess_by_year.png)

## Stage B - the pods

The talk's setup, with idle cash in T-bills. CAGR / Sharpe / max DD per block.

| Pod | 1993-99 | 2000-11 | 2012-21 | 2022-26 | 1993-2026 | Invested | Trades/yr | Win rate | Avg win / loss |
|---|---|---|---|---|---|---|---|---|---|
| IPO (candidate) | 16.8% / 1.00 / -25% | -2.2% / -0.25 / -29% | 0.9% / 0.16 / -32% | -2.1% / -0.29 / -11% | 2.4% / 0.28 / -38% | 32% | 128 | 36% | +16.7% / -8.6% |
| Split (candidate) | 0.5% / 0.11 / -30% | -5.4% / -0.45 / -59% | 2.0% / 0.61 / -5% | 3.8% / 1.71 / -2% | -0.8% / -0.03 / -59% | 35% | 93 | 34% | +13.5% / -7.2% |

**Frozen rule** (the book slot {TAA 0.5, NDX L 0.25, X 0.25} against T-bills in the slot):

| | 2008-11 | 2012-21 | 2022-26 | 2012-26 Sharpe / DD | R1 | R2 | R3 (+5 bps) | R4 (top-500 universe) | R5 capacity | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| T-bills (control) | 0.728 | 1.334 | 1.437 | 1.365 / -11.7% | | | | | | |
| IPO | 0.661 | 1.281 | 1.281 | 1.279 / -12.9% | no | yes | no | no | yes | **no** |
| Split | 0.626 | 1.311 | 1.416 | 1.342 / -12.0% | no | yes | no | no | yes | **no** |

![Grid](../../results/research/ipo_split_ath_20260928/charts/grid_book_vs_tbills.png)

- **Search.** All 30 cells lose to T-bills in the slot. The 40-slot cells lose least only because they sit mostly
  in cash, which is itself T-bills.
- **Multiplicity.**
  - Reality Check p 0.998.
  - Paired bootstrap versus T-bills: IPO -0.086 (90% CI -0.147..-0.028); split -0.024 (-0.048..-0.000).
  - The deflated Sharpe probability is about 0 for both. Their Sharpe over T-bills since 2007 is negative.
- **Capacity is not the problem.** R5 passes: the largest 2021-26 order reaches 5% of daily volume only at about $40M
  of pod capital. At the owner's size, $30k, the IPO pod is about the same (+2.3% a year); the split pod is worse
  (-1.4%) because of the $1 minimum commission.

**Labels (not candidates).** They show which details matter, but none rescues the verdict:
- **E2, exit at the next open after a close through a level.** This is what our live stack can do today.
  - IPO: 9.2% a year, Sharpe 0.81, 1993-2026. But 2022-26 is -4.3% a year, and in the book it loses to T-bills in
    2008-11 (0.675) and 2022-26 (1.232).
  - E2 beats E1 (intraday stops) because IPO stocks' wide daily ranges hit a 10% intraday stop far more often than
    a close-based one.
- **Buying at the signal's own close** (not tradable without an order placed before the close is known):
  - IPO: 9.6% a year, Sharpe 0.94. Part of the post-high move happens overnight.
  - Its 2012-26 book Sharpe of 1.365 only ties T-bills. It still loses in 2008-11 (0.705 vs 0.728) and 2022-26
    (1.326 vs 1.437).
  - 2022-26 standalone is still about 0%.
- **+20 bps per side** for IPO spreads: IPO -0.1% a year.
- **Delisting haircut and SPACs included:** no change.

## Why the talk and this study disagree

| | Talk (Sharadar, about 2001-23) | This study, point-in-time (2001-23) | This study, latest-market-cap filter (2001-23) |
|---|---|---|---|
| 20-day return after an IPO high | +4% | -0.6% | +2.3% |
| Win rate (20 days) | 57% | 46% | 55% |
| Pod CAGR / Sharpe | ~18% / 1.4 | -0.5% / -0.03 (E1); 3.9% / 0.54 (E2) | 9.2% / 1.04 (E1); **17.0% / 1.35 (E2)** |
| Capital invested | 67% | 32% | 57% (E1) / 63% (E2) |

- **The filter "was mid, large or mega cap when last seen" is a winner filter.** Of about 12,700 IPOs, only 20% end
  up at $2B or more. By construction, those are mostly the ones that rose.
- **The diagnostic changes two things at once.** It adds the latest-cap filter, and it drops our liquidity and $5
  filters.
  - The independent reviewer separated the two with point-in-time runs only (2001-23, A0, sweep):
    - all IPO highs above $5: 1.2% / 0.16 / DD -69% (E1); 7.3% / 0.56 / -61% (E2);
    - top 3,000 by liquidity: 3.9% / 0.36 / -52% (E1); 9.6% / 0.75 / -49% (E2).
  - Widening the universe lifts returns somewhat. Only the hindsight filter brings the drawdown from about -50% to
    -60% down to the talk's -15%.
  - 2022-26 is negative in every point-in-time version.
  - 714 delisted IPOs have no share data in Norgate, so they drop out of the hindsight set. That flatters it.
- **The talk's exits seem close-based.** Its numbers match our E2 exit form better than intraday stops.
- **Caveats.**
  - This remains a diagnostic. It does not prove what the talk's code did.
  - The talk's other choices (no price filter, SPACs included, no stated costs) push in the same direction.
- **Parker's own remark supports the conclusion.** He said the strategy stopped working after 2022; in point-in-time
  data it had stopped long before.

## Limitations

- **Size proxy.** Liquidity rank replaces market cap. IPOs trade heavily in their first weeks, so the proxy admits
  smaller IPOs than a market-cap filter would. The stricter top-500 universe (R4) is worse, not better.
- **New listings.** Spin-offs and re-listings count as new listings, as in Parker's "recently listed". SPAC
  listings are excluded, and including them changes nothing.
- **Daily bars.** They cannot order a stop and a target inside one day. Both-touched days assume the stop first
  (conservative); only 60 of 4,304 exits. E1 needs GTC exit orders at the broker, which our live stack does not place
  today.
- **Why E1 and E2 differ, per the review.**
  - 48% of E1 stop-outs closed back above the stop that day.
  - E1's fixed +20% limit cuts off the momentum tail: the next open averages +3.0% above the target.
- **Minor, immaterial here.**
  - Stage A dividends are summed over bars j+1..j+h, one bar early.
  - An age-1 IPO's liquidity is one day's turnover, which is disclosed and matters only for capacity.
  - The split detector can mistake a spin-off at a clean ratio such as 1.25 for a split. This is rare.
- **Split dates.** Splits are dated at the ex-date. The announcement date, where the classic post-split drift
  literature starts, is not in Norgate.
- **Sweep.** Standalone results before mid-2007 have no T-bill sweep on idle cash, which understates those years
  slightly. Book tests start in 2008 and are unaffected.
- **One history.** The only positive period, 1993-99, is one IPO boom.

## Recommendation

- Do not trade either variant, and do not paper-trade them. The frozen rule gives no forward test to a candidate that
  fails.
- No further IPO or split variant is tried in this session, as frozen.
- **Reusable lessons:**
  1. Any backtest that filters by a vendor's "current" size tier is suspect.
  2. For volatile names, intraday stops and close-based stops are different strategies.
  3. The Stage A clustering shows how event-pooled t-tests overstate significance.
