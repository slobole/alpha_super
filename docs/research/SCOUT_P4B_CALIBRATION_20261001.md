# Scout P4b: the per-asset null, S3 calibration, Masters' S2 battery, replication, and Z9

Date: 2026-10-01. Protocol: `scripts/research/scout_p4b_calibration_20261001/PROTOCOL.md`.
- **Commits:** frozen in df76c34, before any result. Amendment 1 (planted-edge sizes) and amendment 2 (score and
  panel realism, after an independent review) were both recorded before any Part B result.
- **Results:** `results/scout/p4b_calibration/` (not in git). Research only; nothing here touches live trading.

## Verdict

| Item | Result | Decision |
|---|---|---|
| Per-asset MCPT null (S5 gate for stock families) | 8.0% false passes for momentum, 2.0% for reversal (bar 8.0%); power 67% and 40% | **Adopted** (N1, membership strata): stock families can now pass S5. Momentum sits exactly at the bar, so a pass with 0.025 < p ≤ 0.05 is flagged *marginal* on the card |
| S3 per-event estimator | 4.0% false positives on the standard panel, **9.5%** on the panel with sector factors and a persistent signal (bar 7%) | Stays informational: it can save an idea from a hard fail, never cause one |
| S3 shift placebo | 4.5% and 4.3% false positives; power 97-100% | Adopted as S3's placebo diagnostic |
| MCPT on Zorro Z9 (known dead) | p = 0.11 on its 2017 list, 0.15 on the neutral list | **The gate stops it** |
| Masters' S2 battery | Built; the break test was recalibrated after review | Diagnostic (WARN only) |
| Nasdaq-100 panel and the S3 replication check | Built; DV2 replicates, QPI does not | Soft check in S3 |

## Part B: the per-asset null

**The problem P4b solves.** S5's only gate is the MCPT: re-run the whole search on histories with the order of the
days shuffled, and ask how often a winner this good appears by chance. A stock universe changes over time (listings,
delistings, index membership), so shuffling whole days scatters those changes and breaks the test (P2's review: 10.7%
false passes). Until now stock families could not pass S5.

**The null (`alpha.stats.mcpt.live_span_source_index_mat`, `alpha.scout.null.permuted_panel`):**
- one global permutation of the dates;
- each stock follows it wherever the source date is one of its own listed dates and in the same membership state;
- its remaining dates fill the other slots, in the order of one global random key.

What this keeps and what it destroys:
- **Kept:** every stock's listing dates, membership dates and its own set of daily bars per membership state; stocks
  live on the same dates move together; liquidity stays on its real dates.
- **Destroyed:** the order of the bars, which is the only thing a timing or selection rule can exploit.

**The score.** For these panels it is the Sharpe of the daily active return over the baseline (equal-weight
members), not "strategy Sharpe minus baseline Sharpe".
- **Why:** the null keeps co-movement only where stocks overlap. On the S&P 500 a stock is a member for about a
  third of its listed life, so the null market is calmer than the real one. With market drift the calmer baseline
  gains Sharpe in the null, and the old score passed noise about 9% of the time (independent review).
- **The fix works:** drift and common moves cancel inside the active return. A paired check at 8 bp a day of drift
  showed no shift toward false passes (−0.07 ± 0.05 and −0.05 ± 0.06).

**Calibration panel.** 150 stocks × 10 years, built to resemble the S&P 500:
- staggered listings and delistings;
- a market, sector and idiosyncratic factor structure;
- market drift of +0.04% a day;
- 50 index members chosen by size every quarter, so member tenure is 0.41 (S&P 500: 0.37);
- the null keeps 77-81% of the equal-weight member volatility (S&P 500: 71%).

**Searches.** Two families with plateau selection:
- short-term reversal: 27 configurations;
- cross-sectional momentum: 9 configurations.

Each history ran 200 permutations, on 200 null seeds and 100 planted-edge seeds per family.

| Family | Null | False passes, p ≤ 0.05 (bar 8.0%) | Power (bar 2 × false passes) | Median real score | Median null score |
|---|---|---|---|---|---|
| Momentum | N1 (membership strata) | **8.0%** | **67%** | 0.21 (null seeds) | 0.17 |
| Momentum | N2 (no strata) | 7.5% | 72% | | 0.17 |
| Reversal | N1 | **2.0%** | **40%** | 0.29 (null seeds) | 0.29 |
| Reversal | N2 | 2.0% | 34% | | 0.35 |

**Decision, by the frozen rule:** N1 passes for both families (false passes ≤ 8.0%, power ≥ twice the false
passes), so it is S5's null for point-in-time panels.

**Caveats on the decision:**
- **Momentum is at the bar, not comfortably under it.** 16 of 200 null histories passed, where 5% would be 10.
  On null seeds its real scores sit slightly above the null's (0.21 vs 0.17); a residual of the weaker co-movement
  is the likely cause.
- **Interim safeguard:** until a stock family has been run for real, a cross-sectional ranking family whose MCPT
  p is between 0.025 and 0.05 passes with a *marginal* flag on its research card.
- **Reversal is conservative** (2%). With moderate edges its power is 40%, so the gate will miss some real but
  small reversal edges. These go to WATCHLIST, not REJECTED (D22).
- **N2 versus N1:** similar. N1 is kept because the protocol named it first, and because it keeps a stock's calmer
  index years apart from its volatile pre-inclusion years.

**The run, for the record:**
- The first Part B launch ran an earlier version of the leftover fill and was stopped unread.
- The second was stopped after the review (amendment 2), also unread.
- The numbers above come from the third run, on the code that ships.

## Part A: S3 estimators

Two P2 panels (200 stocks × 10 years): the standard one, and a dependent one with sector factors and a signal that
persists. 400 null seeds and 200 planted-edge seeds each.

| Test | Standard panel: false positives / power | Dependent panel: false positives / power |
|---|---|---|
| Date-level Newey-West t (the S3 test, reference) | 4.5% / 94% | 6.3% / 100% |
| Per-event estimator (ratio of sums, date-level error) | 4.0% / 99.5% | **9.5%** / 100% |
| Shift placebo (event mask shifted ≥ 1 year) | 4.5% / 97% | 4.3% / 100% |

- **Per-event estimator:** fails the 7% bar on the dependent panel. It stays informational: it can keep an idea
  out of a hard fail (the sign under at least one estimator), but never causes one.
- **Placebo:** holds size on both panels and is added to S3 as a diagnostic.
  - On real panels the shifted events are re-intersected with eligibility (regime and membership), which thins
    them. So its size there is not calibrated; it is probably conservative.

## Part C: MCPT on Zorro Z9

The Z9 search (24 configurations: momentum × rebalance × crash filter × weighting) was re-run on 1,000 date-shuffled
histories of its own ETF lists, in sample, before its publication (2017-10). Score: the best configuration's Sharpe
minus the list's equal weight.

| List | In sample | Best Sharpe | Equal weight | MCPT p |
|---|---|---|---|---|
| 2017 list (XBI ITB SMH XLV VOO AGG HYG IGSB TLT) | 2012-12 to 2017-09 | 1.39 | 1.37 | **0.11** |
| Neutral 14-ETF list | 2005-11 to 2017-09 | 0.74 | 0.62 | **0.15** |

- **Verdict:** the gate stops Z9 before its publication date, as it should for a system that later failed out of
  sample.
- **Replica:** a vectorised, gross, close-to-close copy of the Pakal audit's rules. It tracks the audit's baseline
  run at 0.97 daily correlation (Sharpe 1.06 vs 1.11 on the 2017 list; 0.74 vs 0.64 on the neutral list, over a
  slightly different window).
- **What the p-values judge:** they test this replica. The replica's simplifications (common start, gross, cash at
  zero) apply equally to the real and the shuffled histories.

## Masters' S2 battery

Built into S2 as diagnostics. It covers:
- tails (range/IQR and the share beyond 3 IQR);
- relative entropy of a 20-bin histogram;
- mutual information with the forward excess, against 20 within-date shuffles;
- a single mean-break test on the monthly median.

Threshold optimisation is deliberately left out: it is a search, and it belongs to S4/S5, where trials are
counted.

**Recalibration after review.** The break test first used a fixed Newey-West lag, and it warned on 51-77% of
break-free series with monthly autocorrelation 0.9-0.95. With Andrews' (1991) automatic lag:
- false warnings are 0-3.8% for autocorrelation 0 to 0.95;
- power is 96% for a half-standard-deviation shift at mid-sample.

**On the real indicators:**

| | QPI S&P 500 | QPI NDX | DV2 S&P 500 | DV2 NDX |
|---|---|---|---|---|
| Tails beyond 3 IQR | 0% | 0% | 0% | 0% |
| Relative entropy | 0.99 | 0.99 | 1.00 | 1.00 |
| Mutual information vs shuffle maximum (10⁻⁴ bits) | 18.8 vs 12.3 | 11.6 vs 9.0 | 5.3 vs 3.5 | 8.6 vs 4.9 |
| Level break (sup-Wald, bar 8.68) | 6.3 | 8.6 | 1.6 | 1.7 |
| Novelty warning | 3-day return (0.94) | 3-day return (0.92) | none | none |

**Reading:**
- **Shape:** both indicators are well shaped (bounded ranks, full use of the range, no tails).
- **QPI:** carries information about the next five days (mutual information above the shuffles), but S2 already
  names where it comes from: the 3-day return.
- **QPI's drift:** its threshold drift found in P4 is a change in what "< 15" selects, not a break in the median
  level. The battery does not flag it, and the P4 threshold-share table remains the evidence.

## Nasdaq-100 panel and replication

- **The panel:** snapshot `e2692505d969ef1f`, 1998-2022, sealed at the vault.
- **Membership check:** 17 of 52 stocks that stopped trading left the index before their last bar.
  - The first version of the check (fail above a 20% share) flagged this.
  - The gaps are scattered and mostly one session, which is how Nasdaq removes a stock ahead of an acquisition. A
    hindsight trim instead gives every such stock the same gap.
  - The check now looks for that fingerprint: at least 5 stocks sharing one gap of 2 or more sessions, covering at
    least half the trimmed stocks. It passes both real indexes and fails 5-session-trimmed copies of both.

| Study | S&P 500 (date-level) | Nasdaq-100 (date-level) | Replicates? |
|---|---|---|---|
| DV2 oversold | +9.3 bp, t 3.3; placebo p 0.02 | +17.1 bp, t 3.3; placebo p 0.095 | **Yes** |
| QPI pullback | −5.1 bp, t −1.3 | −13.3 bp, t −1.9 (per event −0.07): **REJECTED (hard fail)** | No |

**DV2 on the Nasdaq-100 is stronger than on the S&P 500:**
- **Cost coverage:** 1.7 (S&P 500: 0.9), still under the 2.0 bar.
- **Outside the crisis windows:** t 2.2 (S&P 500: 1.4).
- **Fade:** smaller, 27 → 13 → 10 bp across the eras (S&P 500: 19 → 5 → 0.6).
- **Deciles:** monotone except near the top (Spearman −0.89).
- **Liquidity:** the middle and top terciles carry it again.

This is input for the DV2 re-audition (P6), not a strategy result.

QPI fails with the wrong sign under both estimators on the Nasdaq-100, which confirms the P4 decision.

## Changes from the independent review

- **HIGH:** the per-asset null biased a "Sharpe minus Sharpe" score toward false passes. Fixed by scoring the
  daily active return; amendment 2 (Part B above).
- **Liquidity in the null:** Volume and Turnover moved with the bars and scrambled liquidity across eras. They now
  stay on their real dates.
- **The break test:** over-warned on persistent series. Fixed with Andrews' automatic lag.
- **The placebo's calibration scope:** stated in S3 (re-intersected with eligibility on real panels).
- **Smaller fixes:**
  - `add_replication` reads the study's own sign and never duplicates its check;
  - strata labels must be finite integers;
  - High and Low are clipped to bracket Open and Close;
  - the baseline in Part B earns from t+2;
  - the planted momentum drift uses only listed returns.
- **New tests:**
  - a leftover-order leak test, checked against the mutant the reviewer used;
  - cross-asset alignment;
  - the Low bound;
  - liquidity kept on its real dates.

## Limits

- **Validity rests on calibration.** The per-asset null is not exact in finite samples: each asset's map is
  uniform, but the joint law across assets is not a group. Its validity comes from Part B, on a synthetic panel
  built to resemble the S&P 500 in tenure and drift.
- **Weaker co-movement in the null.** The null market is calmer than the real one (it keeps 71% of the S&P 500's
  equal-weight volatility). Families that time the market inside a stock panel are not what Part B calibrated.
  They need their own check before relying on the gate.
- **Speed.** One permuted S&P 500 panel takes about 2-15 s depending on machine load, so a 1,000-permutation MCPT
  on a stock family takes hours before any search cost.
