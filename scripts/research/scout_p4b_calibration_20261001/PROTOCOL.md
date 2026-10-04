# Scout P4b calibration: pre-registered protocol

Written 2026-10-01 before any result below was computed. Decisions follow from the rules here, not from the results.

## Part A: S3 estimators

**Panels:** the two P2 S3 generators, 200 stocks x 2,520 sessions, 5-session hold:
- **standard:** market factor, events clustered after market down days (`run_s3_inference.py`);
- **dependent:** adds 10 sector factors and a persistent signal, repeat probability 0.6 (`run_s3_inference_exploratory.py`).

Each panel is run on 400 null seeds and 200 seeds with a planted edge (+0.2% per event).

**Tests:**
1. **Date-level Newey-West t,** lag 4. This is the reference: P2 measured it at 5%.
2. **Per-event estimator:** the ratio of sums with a date-level Newey-West error, as in `s3_edge._per_event_nw`.
3. **Shift placebo:**
   - The event mask is shifted in time by a random circular offset of 252 to T − 252 sessions, 200 draws.
   - Statistic: the date-level mean excess. One-sided p = (1 + #{shifted ≥ observed}) / 201.
   - The shift keeps the signal's persistence and clustering and breaks its link with returns.

**Decisions** (rejection rate on the null seeds, at nominal 5%):
- **Per-event estimator:** two-sided rejection ≤ 7.0% on BOTH panels makes it calibrated. It may then cause a hard
  fail like the date-level test (t ≤ −2). Otherwise it stays informational, as now.
- **Shift placebo:** one-sided rejection ≤ 7.0% on both panels makes it the S3 placebo diagnostic. Otherwise it is
  dropped.

The 7.0% bound is 5% plus two binomial standard errors at 400 seeds (1.1% each).

## Part B: the per-asset MCPT null (S5 gate for point-in-time panels)

**Panel:** 150 stocks x 2,520 sessions.
- **Listing:** 100 stocks listed from day 0; 50 list on a uniform random day in [0, 2,000). 30 random stocks delist
  on a uniform random day at least 500 sessions after listing.
- **Returns:** GARCH Student-t market times beta ~ U(0.5, 1.5), 5 sector factors (GARCH, 12% a year), and
  idiosyncratic Student-t(5) at 2% a day.
- **Membership:** the 90 largest listed stocks by cap (price times a lognormal share count), reconstituted every
  63 sessions and fixed in between, like an index.

**Families, each searched as a regular grid with plateau selection (`alpha.stats.selection.plateau_choice`):**
- **Reversal:** each day, among members, buy the stocks whose L-day return is in the bottom q of members, and hold
  h sessions (overlapping cohorts, equal weight). Grid: L ∈ {2, 3, 5} × q ∈ {0.05, 0.10, 0.20} × h ∈ {3, 5, 10}.
- **Momentum:** every 21 sessions, hold the top K members by L-day return, skipping the last 5 sessions. Grid:
  L ∈ {63, 126, 252} × K ∈ {5, 10, 20}.
- **Score:** the Sharpe of the selected configuration minus the Sharpe of the equal-weight member portfolio on the
  same history (the baseline-relative score of A5). Next-day execution, no costs.

**Planted edges:**
- **Reversal:** a member whose 3-day return is in the bottom 5% of members earns +0.05% a day for the next 5
  sessions.
- **Momentum:** each stock's daily drift is 0.10 times its own mean return over the last 126 sessions, lagged one
  session.

**Nulls compared:**
- **N1 (adopted if it passes):** `mcpt_live_spans` with the membership flag as stratum.
- **N2:** the same without strata.
- The date-row shuffle cannot run on these panels, which is the point of P4b.

**Runs:** 200 null seeds and 100 edge seeds per family, 200 permutations each.

**Decisions:**
- **N1 becomes the S5 null for point-in-time panels** if its false-pass rate at p ≤ 0.05 is ≤ 8.0% for both
  families and its power is ≥ 2 × its false-pass rate for both planted edges. 8.0% is 5% plus two binomial standard
  errors at 200 seeds.
- **If N1 fails and N2 passes,** N2 is adopted.
- **If both fail,** stock families stay unable to pass S5 (WATCHLIST), and the failure is reported.

## Part C: MCPT on Zorro Z9 (a known-dead case, in sample only)

**Search:** the Z9 grid on its own list: momentum ∈ {mean200, roc252, roc126} × rebalance ∈ {25, 35} × crash filter
∈ {off, sma200} × weighting ∈ {equal, momentum} (24 configurations), on the L2017 list and on the neutral N14 list.

**Implementation:**
- Vectorised close-to-close replica of the Pakal audit rules, gross.
- The decision at the close of T is held from the close of T+1.
- Cash earns zero.
- SPY is a column of the permuted matrix, so the crash filter moves with its dates.

**Scope and score:**
- **In sample:** from the first date every list symbol and SPY has its lookback, to 2017-09-29.
- **Score:** the selected Sharpe (maximum; the grid is categorical) minus the Sharpe of the list's equal-weight
  portfolio.
- **Null:** plain date-row shuffle (the list has a common span), 1,000 permutations.
- **Sanity check:** the replica's baseline configuration is compared with the audit's Sharpe over the same
  window. The case is reported whatever the p-value.

**Expected:** Z9 is known dead after publication, so a pass here counts against the gate's power to stop hindsight
lists, not against its size. It is reported, not a decision input.

## Amendment 1 (2026-10-01, before any Part B calibration result)

A smoke run of the Part B searches on a few seeds, without MCPT, showed that the planted edges as registered do not
measure power:
- **Reversal:** too strong. At +0.05% a day the search scored about 1.0 Sharpe above its baseline, so power would be
  about 100%.
- **Momentum:** too weak. At a coefficient of 0.10 the score was barely above the null seeds'.

The edges were re-sized by scanning 6-8 seeds of the searches alone, to about 0.4-0.5 Sharpe above the null, the
range P2 used:
- **Reversal:** +0.02% a day for 5 sessions.
- **Momentum:** a coefficient of 0.15.

This changes only the power runs. The null seeds, the nulls, the families, the decision rules and the Part A
results are unchanged.

## Amendment 2 (2026-10-01, after an independent code review, before any Part B result)

**Why.** The review showed that the per-asset null keeps co-movement only where two stocks are live, and in the
same membership state, on both dates.
- On the S&P 500 a stock is a member for about a third of its listed life. One null draw cut the equal-weight
  member market's volatility from 1.34% to 0.94%.
- With market drift, the calmer null baseline gains Sharpe, so a score of the form "selected Sharpe minus baseline
  Sharpe" is shifted against the null by about 0.3 of its standard deviation. That is roughly 9% false passes at a
  nominal 5%.
- The Part B panel as registered could not see this: it had zero drift and a member tenure of 0.75.

The first Part B launch was stopped with no results read.

**Changes:**
1. **Score:** the Sharpe of the daily active return of the configuration over the baseline, the equal-weight members.
   Plateau selection runs on these active Sharpes. Drift and common moves cancel inside the active return. A paired
   check at 8 bp a day of drift moved the observed-minus-null gap by −0.07 (se 0.05) for momentum and −0.05 (0.06)
   for reversal: no bias toward false passes.
2. **Panel realism:**
   - market drift of +0.04% a day on every stock;
   - 50 index members out of 150 stocks. Member tenure is now 0.41 (S&P 500: 0.37), and the null keeps 77-81% of
     the equal-weight member volatility (S&P 500: 71%).
3. **Timing:** the baseline decides at the close of t and earns from t+2, like the strategies.
4. **Planted momentum drift:** uses only listed returns.
5. **`permuted_panel`:**
   - Volume and Turnover stay on their real dates. Liquidity is structural, like membership.
   - High and Low are clipped so that they bracket Open and Close exactly.

The decision rules are unchanged.
