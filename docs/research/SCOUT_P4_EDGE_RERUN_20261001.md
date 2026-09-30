# The owner's QPI and DV2 edge notebooks, re-run through Scout S1-S3

Date: 2026-10-01. Script: `scripts/research/scout_p4_edge_rerun_20261001/run.py`. Results:
`results/scout/p4_edge_rerun/edge_rerun.json` (not in git). Panel: S&P 500, point-in-time, snapshot `4ecbebd98bf5e8af`,
1998-01-02 to 2022-12-30 (sealed at the vault; 2023 onward was not looked at). Research only. Reviewed by an
independent read-only agent; its corrections are applied below.

## Verdict

| Study | Notebook method | Scout S3 (date-level Newey-West, point-in-time members) | Verdict |
|---|---|---|---|
| **QPI pullback**: uptrend, 3-day return < 0, QPI(3, 5y) < 15, hold 5 sessions | +0.22% per trade over non-events, Welch t = **10.0** (77,755 events) | −0.05% per date, t = **−1.3**; per event +0.05%, t = 1.8 (51,263 events, 4,279 dates) | **WATCHLIST by the rule, no edge of its own**: QPI is the 3-day return in disguise |
| **DV2 oversold**: uptrend, 126-day return > 0, DV2(126) < 10, hold 5 sessions | +0.31% per trade, Welch t = **27.0** (220,585 events) | +0.09% per date, t = **3.3**; per event +0.10%, t = 5.5 (139,670 events, 5,626 dates) | **WATCHLIST**: real stock-picking information, but it lives in stress and has faded since 2008 |

What S3 measures: the excess return of an event over the average stock in the same regime on the same date. That is
the information the signal adds when choosing among stocks that are already eligible. A long-only pod also earns the
regime's own return. That part is judged in S6 (value to the book), not here.

Two estimators are reported. The date-level one averages the events of each date and tests the date series; P2
calibrated it at a 5% false-positive rate. The per-event one weights every event equally, with the same date-level
error; it is more powerful but not yet calibrated (P4b). A hard fail needs the sign to be wrong under both, or the
date-level t to be −2 or lower.

## Why the notebooks looked so much better

The "notebook method" rows re-run the notebooks' statistics on the sealed Scout panel. That panel holds the symbols
that were index members at some point from 1998 on (1,172). The notebooks used their "Current & Past" list (1,303)
and unpadded data through 2026-03, and printed t = 8.4 (QPI) and 26.7 (DV2). The small differences do not change
anything below.

| Step | QPI | DV2 |
|---|---|---|
| Notebook method (every symbol on every date, events vs non-events, per-event t) | t = 10.0 | t = 27.0 |
| Scout statistics, membership filter off (date-level t) | t = 2.3 | t = 5.8 |
| Scout with point-in-time membership (the full correction) | t = −1.3 | t = 3.3 |

1. **Membership.** The notebooks count a stock on dates when it was not in the S&P 500: 34% of QPI events and 37%
   of DV2 events. Where those QPI events fall (5-day raw forward return):

   | Group | Share of non-member events | 5-day raw forward return |
   |---|---|---|
   | Before the stock's first inclusion | 54% | +68 bp |
   | After its last removal | 29% | +112 bp |
   | Gaps between membership spells | 3.5% | +99 bp |
   | Never a member before 2023 | 13% | +47 bp |
   | Member events, for comparison | — | +41 bp |

   The short-term rebound is much stronger in smaller, more volatile names outside the index, whether on their way
   in or on their way out. Those names are not in the tradeable universe on that date, and they account for most of
   QPI's apparent edge.
2. **Unit of inference.** The notebooks treat each of tens of thousands of events as independent, but events cluster
   on the same stress days and overlap over 5-day holds. The switch to date-level inference alone divides the
   t-statistics by 2-3 (the table's middle row also changes the comparison group). P2 showed that per-event tests
   reject a true null 19% of the time on such panels.
3. **Comparison group.** The notebooks compare events with non-events on raw forward returns. Scout compares with
   the same-date regime average, which removes the market move of the day.

## QPI pullback in detail

- **The two estimators disagree in sign, and neither is significant** (date-level −5.1 bp, t −1.3; per event
  +4.8 bp, t 1.8). By the D22 rule that is not evidence against, so the verdict is WATCHLIST, not REJECTED. In
  substance there is nothing to watch:
  - **Eras:** −10.8 bp (t −1.4) in 1998-2007, −9.7 bp (−1.4) in 2008-2015, +3.2 bp (0.6) in 2016-2022.
  - **Deciles of QPI inside the regime** (ranked per date): no pattern. The lowest decile is −2.3 bp and the best
    decile is the third (+8.1 bp).
  - **Station 2:** QPI has a Spearman correlation of **0.94 with the plain 3-day return**. Inside this regime it is
    almost the same number. The threshold also drifts: QPI < 15 selects 5% of eligible observations in 2003-2007
    but 13.5% in 2018-2022, so it is not the same rule across eras.
- **Decision:** QPI is not studied again as its own family. Any short-term reversal idea is registered under the
  3-day-return family, with these runs recorded as prior trials.
- **Station 1:** QPI, the 3-day return, SMA200 and liquidity pass the truncation and future-split tests. No leak.
- This agrees with QPI's demotion to research on 2026-09-29.

## DV2 oversold in detail

- **The deciles are monotone, which is the signature of real information** (ranked per date, so eras are not mixed):

  | DV2 decile | Median DV2 | Mean excess | t |
  |---|---|---|---|
  | 1 (most oversold) | 10 | +11.0 bp | 5.3 |
  | 2 | 23 | +7.8 bp | 4.6 |
  | 3 | 32 | +5.5 bp | 4.1 |
  | … | … | … | … |
  | 9 | 80 | −9.3 bp | −5.9 |
  | 10 | 92 | −11.5 bp | −5.9 |

  All ten deciles fall in order (Spearman −1.0). This is stronger evidence than the event study alone.
- **The edge has faded:**

  | Era | Mean excess per date | t | Per event |
  |---|---|---|---|
  | 1998-2007 | +19.0 bp | 4.0 | +19.6 bp |
  | 2008-2015 | +5.3 bp | 1.1 | +4.6 bp |
  | 2016-2022 | +0.6 bp | 0.1 | +2.6 bp |

  Positive in 68% of years.
- **It lives in stress:**
  - by volatility regime: calm +1.7 bp (t 0.5), normal +5.4 bp (1.2), stressed **+22.6 bp (3.4)**;
  - without the three crisis windows (2000-02, 2008-09, 2020): t = 1.4.
  - Trimming the most extreme 1% of event-dates on both sides leaves t = 4.1. The edge is not a few outlier days.
    (The first draft trimmed only the best 1% and got t = 0.6. That test is biased by construction: it gives
    t ≈ −3 on pure noise, so it was replaced.)
  - This matches the earlier finding that mean reversion is a premium for supplying liquidity in stress
    (2026-09-26).
- **It decays within two sessions:**

  | Entry delay | Mean excess | t |
  |---|---|---|
  | 0 | 9.3 bp | 3.3 |
  | 1 session | 5.5 bp | 2.1 |
  | 2 sessions | 1.9 bp | 0.7 |

  A real, short-lived effect: it decays gradually rather than all at once, which argues against a data artefact.
- **Liquidity:** none in the bottom turnover tercile (−0.7 bp), and present in the middle and top terciles (+11.6 bp
  and +12.6 bp, t ≈ 3). Tradeable names carry the edge.
- **Costs:** about 9 bp of excess per trade against about 10 bp round trip in the middle tercile, so a cost coverage
  of 0.9 against the 2.0 bar. This is the one soft check it fails.
- **Station 2:** threshold stable (DV2 < 10 selects 7.8-8.8% in every block); correlation with the 3-day return
  0.56. DV2 is its own indicator.
- **Station 1:** no leak.

What this means for DV2 the strategy:
- Its stock-picking edge beyond an uptrending stock is real, but too small to pay for trading on its own.
- The edge is concentrated in stress episodes and nearly gone since 2016.
- The pod's backtested profits depend heavily on the regime's own return and on stress episodes. S6 (value to the
  book) and the forward shadow test are where that gets judged.
- It remains WATCHLIST, not REJECTED: the evidence is against a large edge today, not against any edge.

## Method and limits

- **Horizon and entry:** 5 sessions, entry at the next open, exit at the fifth close, as in the notebooks.
- **Excess:** measured against the equal-weight eligible (regime and member) mean of the same date.
- **Significance:** Newey-West lag 4 on the date series.
- **Costs:** 2.5, 5 or 10 bp per side by turnover tercile (design 6.2). The tradeable tercile is the median tercile
  of the events.
- **Membership integrity:** of 355 stocks that stopped trading while in or near the index, 31 (8.7%) left the
  index a few sessions before their last bar, as real removals before an acquisition do. A tail-trimmed mask scores
  100%.
- **Data:** CAPITALSPECIAL prices padded to all market days (the engine's loader). Halted days are carried at the
  last price.
- **Diagnostic cut points:** the volatility terciles and indicator deciles use full-sample cut points. They
  describe the history; they are not trading rules.
- **Not yet in S3:** the replication check in a sibling universe (Nasdaq-100 panel, P4b), bootstrap intervals,
  placebo dates, and up/down-market splits.
- **Not ledger-registered:** these are methodology re-runs of pre-ledger notebooks. When DV2 is re-audited as a
  family (P6), the registration records these runs as prior trials.
- **The vault is untouched:** everything ends on 2022-12-30.
