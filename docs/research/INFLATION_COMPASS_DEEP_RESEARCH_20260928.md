# Inflation Compass deep research — verdict (2026-09-28)

Research-only. Nothing in live trading, releases, schedulers, broker routes, the engine or portfolio YAMLs changed.
Owner request (2026-09-28): deep quant research on Varadi's Inflation Compass (CSS Analytics, 2026-07-27) and
Allocate Smartly's review — pitfalls, is there a real edge, sensitivity, other mechanisms/assets, readiness, verdict.
Frozen plan and dated amendments: [`scripts/research/inflation_compass_deep_20260928/SPEC_FROZEN.md`](../../scripts/research/inflation_compass_deep_20260928/SPEC_FROZEN.md).
Results: `results/research/inflation_compass_deep_20260928/`. Everything uses the look-ahead-fixed module
(T5YIE dated before T; see [LEAKAGE_HUNT_BOOKS_20260927](LEAKAGE_HUNT_BOOKS_20260927.md) item 2).

## Verdict

**There is something real behind it, but much less than advertised, and it does not earn its slot in the fund
books. Demote it from the menu books to a forward shadow, and if it is kept anywhere, use QQQ instead of XLK.**

1. **A real signal in-sample, but it lives in two episodes.** Timing energy vs tech on breakeven inflation is not
   random on 2003–2026: the actual inflation switch beats all 257 shifted and 1,000 reshuffled placebo switches
   (p < 0.001); after an "inflation on" month-end, XLE beats XLK by +3.1% the next month (t = 3.3 overall, but
   t = 2.1 / 0.7 / 1.4 in the three periods); alpha against a static mix with the same average holdings is +11%/yr
   (t = 4.0). The mechanism is plausible: breakevens move with oil and commodity prices (correlation 0.64 with
   60-day DBC returns). **But** the inflation switch added +12 to +31 pp a year in 2003–07 and +66 pp in 2022, and
   roughly nothing or less in 2008–2021 (2008–21 Sharpe 0.83 vs 0.99 for the same rule with the inflation switch
   removed). Without 2003–07 and 2022 it trails a plain SPY trend switch (0.90 vs 0.97).
2. **The published version is a lucky peak.** Its Sharpe (1.085) sits at the 98.7th percentile of 700 nearby
   parameter sets (median 0.92) and is the best of 21 rebalance days around month-end (median 0.96, CAGR
   14.9–21.1%). Correcting parameter luck and rebalance-day luck together (880 runs) gives a median Sharpe of
   **0.87** (5–95%: 0.69–1.04). Swapping T5YIE for the Fed's own 5-year breakeven series (correlation 0.98) costs
   about 0.09 Sharpe. The honest in-sample expectation is **Sharpe ≈ 0.85, CAGR ≈ 16%, daily max drawdown ≈ −25 to
   −30%**, and less out of sample (holdout 0.74), not the article's 1.41 / 23.5% / −16%.
3. **It failed the untouched pre-2003 holdout.** 1983–2002 on proxies: Sharpe 0.74 vs 0.96 for a plain SPY
   200-day trend switch (CAGR 13.2% vs 13.2%, drawdown −49% vs −33%), with a −43% loss in 2000–02 from holding
   utilities. Caveat that matters: the proxy inflation series was above 2% in every month, so the holdout could not
   test the breakeven gate. An exploratory rerun of 1999–2002 with real breakevens still lost to the trend switch
   (0.31 vs 0.48).
4. **In the books it is roughly a coin flip against T-bills in the same slot.** It adds 1.5–4.7 pp of CAGR in
   the five fund-menu books that hold it (8–25%), paid for with drawdown. On plain Sharpe (zero rate), T-bills in
   its slot give a higher Calmar in all five books; on excess-of-T-bill returns, Compass wins on Sharpe in all five
   but passes the frozen "better Sharpe and Calmar" rule in only two (growth, low-touch balanced), by thin
   margins. It does not clearly earn its slot, and that is with the lucky-peak sleeve.
5. **One improvement passes every main-window test: QQQ instead of XLK** (Sharpe +0.05, bootstrap p = 0.031,
   better in all three periods and with doubled costs). That p-value is not corrected for the 7 candidates
   (Bonferroni ≈ 0.007) and QQQ came from Allocate Smartly, who saw the same data; it cannot be tested before 1999.
   So it is a shadow candidate, not a promotion. Averaging parameters or rebalance days (the "Enhanced" idea) makes
   the backtest worse (−0.08 to −0.14 Sharpe) — which is exactly what a lucky-peak baseline predicts.

Recommended actions (owner decisions): (a) take Compass out of the menu books or cut it to a small satellite
(≤ 5%) until a forward shadow exists; (b) if a version is shadowed, shadow the QQQ variant with its full-history
numbers quoted beside the 21-day and parameter-grid ranges; (c) update the book numbers — the menu figures that
include Compass quote the lucky-peak sleeve.

## Implementation of the QQQ shadow candidate (2026-09-28, owner request)

`strategies/taa_df/strategy_taa_inflation_compass_qqq.py` implements candidate C2: the same module and data path
with QQQ in the growth-up / inflation-off cell (the parent gained a `goldilocks_asset_str` config field, default
XLK, with no change to its results: 21.072% / 1.0850 reproduced). Engine run 2003-05-01 → 2026-08-19: CAGR
22.07%, Sharpe 1.138, max DD −23.4% (research replica: 22.06% / 1.138). Registered PM_READY after the capital,
benchmark and determinism checks. PM_READY is plumbing only: the verdict above (lucky-peak parameters, holdout
fail, coin flip against T-bills in the books, C2's p-value uncorrected) applies unchanged. The Stress analyzer is
not registered for it.

## How it was tested

- **Instrument.** A fast share-accounting replica of the Vanilla engine (next-open fills sized from the prior
  close, 5 bps per side, 25% dividend withholding like the engine). Against the real engine over 2003-05-01 →
  2026-08-19: CAGR −0.003 pp, Sharpe −0.0001, daily correlation 0.9999998. The vectorized signal equals the fixed
  module on all 282 decisions. The engine reproduces the audit exactly (21.072% / 1.0850).
- **Pre-registration.** Questions, grid, candidates, promotion rules, periods (P1 2003–12, P2 2013–19, P3
  2020–26) and holdout frozen before any study result; three dated amendments, A3 explicitly post-hoc.
- **Search size.** 762 configurations in this study + 64 disclosed elsewhere (article grid 15, Allocate Smartly 4,
  the earlier Pakal study 44, module 1) = 826 trials for deflation. The deflated Sharpe is ≈ 1.0 for every
  variant, but it is **uninformative** here: the grid cells are highly correlated, so the null bar is tiny, and
  even SPY buy & hold (0.97) and the static mix (0.99) would pass it. The decisions below rest on the placebo,
  grid, offset, bootstrap, holdout and book tests instead.

## 1. Why the article shows Sharpe 1.41 and we show 1.085

Same strategy, stepwise (2003-05 → 2026-06, daily Sharpe; monthly-sampled Sharpe and drawdown in brackets):

| Step | CAGR | Sharpe | Max DD (daily) | [monthly Sharpe / DD] |
|---|---|---|---|---|
| Ours: next open, published T5YIE, 5 bps, 25% withholding | 21.30% | 1.099 | −24.3% | [1.22 / −17.1%] |
| No withholding | 21.87% | 1.123 | −24.2% | [1.25] |
| No costs | 22.26% | 1.140 | −24.0% | [1.27] |
| Same-date T5YIE (the old leak) | 22.70% | 1.159 | −24.0% | [1.30] |
| Fill at the same month-end close (the article's setup) | 23.30% | 1.186 | −23.6% | [1.34 / −16.2%] |
| Article | 23.5% | 1.41 | −23.6% (stated) | [— / −16.2%] |

The article's CAGR and −16.2% drawdown are reproduced; its Sharpe is closest to our month-end-sampled 1.34
(the earlier Pakal replication also got 1.36). About 0.12 of the gap is monthly-vs-daily measurement, about 0.09
is withholding + costs + the leak + same-close fills, and about 0.07 is unexplained (data source / method).

## 2. Is there an edge?

Main window 2003-05 → 2026-08, all next-open with 5 bps:

| Strategy | CAGR | Sharpe | Max DD |
|---|---|---|---|
| **Compass (fixed)** | **21.07%** | **1.08** | **−24.3%** |
| SPY buy & hold | 11.06% | 0.66 | −55.5% |
| QQQ / XLK buy & hold | 15.6% | 0.76–0.78 | −53% |
| 60/40 SPY/IEF | 7.92% | 0.78 | −33.1% |
| SPY > SMA200 else IEF | 10.13% | 0.81 | −31.1% |
| Growth axis only (XLK or XLP/IEF) | 13.26% | 0.81 | −33.8% |
| Inflation axis only (XLE or XLK, no growth filter) | 20.55% | 0.90 | −52.6% |
| Static mix with the same average holdings | 13.02% | 0.74 | −46.9% |
| Inflation = sector-basket slope only (no T5YIE) | 13.74% | 0.74 | −31.4% |
| Inflation = T5YIE level and 60-day change only | 18.26% | 1.00 | −23.3% |
| Inflation = T5YIE > 2% only | 19.27% | 0.99 | −29.7% |

- Time in regime: tech 48%, energy 34%, staples/bonds 12%, utilities 6%. The strategy is mainly an energy-vs-tech
  switch with a trend filter.
- The value is in T5YIE, not in the sector basket: the basket slope alone is worth nothing (0.74), while
  "T5YIE above 2%" alone reaches 0.99. The article's "remove T5YIE → 0.82" is confirmed in direction.
- Map permutations: the literal map is #1 of 24 on Sharpe. This is not evidence: an author who built the map on
  2003–2026 would produce exactly that.
- Where the inflation switch earns (Compass minus the growth-only rule, per year): +12, +28, +31, +24, +22 pp in
  2003–07; +66 pp in 2022; +9 and +14 pp in 2025–26; between −8 and +3 pp in every other year. 2008–2021 Sharpe:
  Compass 0.83, growth-only 0.99, SPY trend 0.92. Excluding 2003–07 and 2022: 0.90 / 0.98 / 0.97.
- Placebo inflation switches (same frequency and spell lengths): median Sharpe 0.63–0.66, 95th percentile
  0.86–0.90, actual 1.085 → p < 0.001 both ways.
- Next-month XLE−XLK after "inflation on" (growth-up months): +3.1%/month, t = 3.3 overall; P1 t = 2.1, P2 t = 0.7
  (inflation was on in only 9% of 2013–19), P3 t = 1.4. The utilities-vs-staples/bonds cell shows nothing
  (t = 0.7, 51 months).
- Is it just an oil proxy? Partly (exploratory): the 60-day T5YIE change correlates 0.64 with DBC and 0.54 with
  XLE. Replacing it with DBC momentum gives 0.96 vs 0.995 on the same window; replacing it with XLE-minus-XLK
  momentum gives 0.90. T5YIE carries a little more than price momentum.

## 3. Sensitivity (how fragile)

- 700-cell grid (threshold 1.8–2.5 × breakeven window 20–120 × basket window 20–120 × SMA 100–250): Sharpe
  median 0.92, 5–95% range 0.77–1.05, CAGR median 16.6%; baseline at the 98.7th percentile.
- One axis at a time, the defaults are sharp local peaks: basket window 40/60/80 → Sharpe 0.96 / 1.085 / 0.98;
  breakeven window 40/60/80 → 1.02 / 1.085 / 1.03; threshold 1.8/2.0/2.2 → 1.02 / 1.085 / 1.02; SMA 150 is better
  than 200 (1.125). This contradicts the article's "middle of a plateau" claim.
- Rebalance day: month-end is the best of 21 days (−10…+10 sessions). Median 0.96, range 0.81–1.085; CAGR
  14.9–21.1%. An equal split across all 21 days gives Sharpe 1.00, CAGR 18.4%, max DD −27.7%.
- Both kinds of luck together (80 random grid cells × 11 rebalance days, 880 runs): median Sharpe 0.87,
  5–95% range 0.69–1.04; median 0.93 at month-end and 0.86 on other days.
- One exact two-decimal tie (2023-04-28) moves CAGR by 0.9 pp (anchor dated vs lagged one session).
- Execution: next close 1.053, second open 1.034 (−0.03 to −0.05). Instruments: QQQ for XLK 1.138; TLT for IEF
  1.088; SHY 1.071; XLP only 1.032 (DD −34%); IEF only 1.079.

## 4. Candidates (declared before results)

| Candidate | CAGR | Sharpe | Max DD | ΔSharpe | P(ΔSharpe ≤ 0) | ΔP1 / ΔP2 / ΔP3 | +10 bps | Verdict |
|---|---|---|---|---|---|---|---|---|
| Baseline | 21.07% | 1.085 | −24.3% | — | — | — | — | — |
| C1 ensemble of 27 cells | 18.25% | 0.996 | −24.1% | −0.089 | 0.97 | −0.07 / −0.08 / −0.14 | −0.09 | fails non-inferiority |
| **C2 QQQ for XLK** | **22.06%** | **1.138** | **−23.4%** | **+0.053** | **0.031** | +0.04 / +0.06 / +0.07 | +0.05 | passes main tests; holdout untestable → shadow |
| C3 stagflation 50% XLU + 50% DBC | 20.74% | 1.078 | −24.3% | −0.007 | 0.59 | | | no |
| C4 four rebalance tranches | 18.39% | 1.003 | −26.0% | −0.080 | 0.92 | | | fails non-inferiority |
| C5 hysteresis 2.1/1.9 | 21.60% | 1.106 | −24.3% | +0.022 | 0.26 | | | not significant |
| C6 volatility target 15% | 15.98% | 1.085 | −18.5% | 0.000 | 0.49 | | | risk dial, not an edge |
| C7 ensemble + tranches | 16.61% | 0.941 | −26.3% | −0.142 | 0.98 | | | fails |

The "Enhanced" ideas (C1, C4, C7) remove specification luck and therefore land on the typical outcome.
C2 is the natural fix for XLK losing Alphabet and Meta to XLC in 2018; QQQ trades ~15× XLK's dollar volume.

## 5. Holdout (used once) — FAIL

Primary, 1983–2002, all proxies (French 12 industries, $SPXTR/$SPX, synthetic 7–10y Treasury; Cleveland Fed
5-year expected inflation usable from the end of the following month); next-close fills, 5 bps, no withholding.

| | 1983–2002 | 1983–90 | 1991–98 | 1999–2002 |
|---|---|---|---|---|
| Compass | 13.2% / 0.74 / −48.6% | 0.996 | 0.83 | 0.27 |
| SPY buy & hold | 11.7% / 0.74 / −47.4% | 0.79 | 1.50 | −0.21 |
| SPY > SMA200 else bonds | 13.2% / 0.96 / −33.2% | 0.78 | 1.38 | 0.58 |
| Equal-weight 8 sectors | 14.0% / 0.98 / −31.2% | 1.03 | 1.63 | 0.10 |

(Cells: CAGR / Sharpe / max DD, or Sharpe for sub-periods.) The pass rule (beat both SPY and the trend switch)
fails. Limits, stated beside the result: the proxy inflation series was above 2% in all 240 months (the gate never
switched off, so "inflation on" 70% of the time vs 42% live), its changes correlate 0.03 with T5YIE, and SPY is
price-only before 1988. Real sector SPDRs 2000–02: Compass 0.16 Sharpe, −43% (utilities in the 2002 power-sector
crash). Exploratory (post-hoc, A3), with the Fed GSW 5-year breakeven: 1999-11..2002 Sharpe 0.31, −35% (the trend
switch's 0.48 covers 2000–02, a slightly different window; GSW's extra months were in XLK and flatter it);
2003–2026 with GSW instead of T5YIE: 18.3% / 0.96 against 20.4% / 1.05 for T5YIE under the same next-close fills.

## 6. Book role (2012-10 → 2026-08, independent pods, annual reset, HPI and Compass fixes applied)

| Book (Compass weight) | As is | Slot → T-bills | Slot → other sleeves | QQQ variant |
|---|---|---|---|---|
| Aggressive (25%) | 17.45% / 1.25 / Calmar 1.46 | 12.73% / 1.20 / 1.57 | 16.17% / 1.16 / 1.53 | 17.62% / 1.26 / 1.50 |
| Growth (11%) | 14.75% / 1.36 / 1.34 | 12.67% / 1.37 / 1.36 | 13.98% / 1.35 / 1.34 | 14.83% / 1.37 / 1.36 |
| Balanced (8%) | 12.43% / 1.52 / 1.62 | 10.90% / 1.55 / 1.70 | 11.68% / 1.53 / 1.67 | 12.48% / 1.53 / 1.64 |
| Low-touch growth (18%) | 14.24% / 1.32 / 1.57 | 10.82% / 1.30 / 1.76 | 12.73% / 1.27 / 1.71 | 14.36% / 1.33 / 1.63 |
| Low-touch balanced (12%) | 11.41% / 1.41 / 1.79 | 9.11% / 1.43 / 1.87 | 10.08% / 1.40 / 1.83 | 11.49% / 1.43 / 1.86 |

(Cells: CAGR / Sharpe / Calmar, zero-rate convention.) Frozen rule: Compass earns its slot only if "as is" beats
both alternatives on Sharpe **and** Calmar. Zero-rate convention: it does in none of the five. Excess-of-T-bill
convention (the spec asked for both; the zero-rate Sharpe favours a T-bill substitute by construction): Compass
wins on Sharpe in all five (e.g. aggressive 1.12 vs 1.03) and passes both tests only in growth (excess Calmar
1.151 vs 1.138 / 1.137) and low-touch balanced (1.102 vs 1.085 / 1.087). It fails in aggressive, balanced and
low-touch growth. The QQQ variant is at least as good as "as is" in every book on both conventions. It is a return engine, not a diversifier (correlation 0.4–0.55 with most growth sleeves; negative
only with the tail hedges). The typical-rebalance-day version lowers every book's Calmar further.

## 7. Readiness gaps (if it is ever run with money)

- Decision cutoff after the ~17:00 ET FRED/H.15 release; rule for a late or missing T5YIE (not defined).
- Negative cash from sizing on Close_T and filling at Open_(T+1) (G-028) — needs a cash-constrained or financed
  contract.
- Single-ETF concentration: one sector holds 100% for months; the utilities cell has no evidence (6% of months,
  t = 0.7) and failed in 2002.
- Capacity is not the binding constraint: the least liquid holding (XLU) trades ~$0.9B/day, so worked orders can
  move a $10M+ sleeve per switch; the opening-auction route is much smaller and the Capacity analyzer's ETF proxy
  (which rejects even $50k) is not credible for these ETFs.
- Only ~2 months of post-publication data exist. A forward shadow of at least 12–24 months, frozen, is the only
  clean evidence left.

## Pitfalls checklist

Look-ahead: fixed (T5YIE published T+1) and re-verified; every new path lags data explicitly. Survivorship: none in
the ETFs, but the sector choice and the map are ex-post. Data mining: author grid + ours + Pakal = 826 trials;
the published point is a peak. In-sample contamination: 2003–2026 was seen by the author before publication; the
only untouched data (pre-2003) failed. Regime dependence: 2022 (+35.5% here vs SPY −18%) and the 2003–08 energy boom
carry much of the edge; 2013–19 is weak (P2 Sharpe 0.97, below SPY's 1.06). Sample size: 282 decisions but only
76 holding changes, 26 of them into energy. Costs: minor (−0.02 Sharpe per 5 bps). Live/backtest: next open vs article's same
close costs 0.02–0.05.

## Verification fields

- Tier: 1 (research scripts and a research doc; no strategy, engine, live or release code changed in this phase).
- Agents used: one holdout-data builder (data only, no rule computed), one independent quant-pitfalls reviewer.
- Reviewer result: no look-ahead in any new path; claims 1, 2 and 6 supported. Findings fixed in this report after
  re-computation (`p7_review_checks.json`, `p5_books.csv` excess columns): the edge's concentration in 2003–07 and
  2022; the joint-luck expectation (0.87, not 0.95–1.0); the deflated Sharpe declared uninformative; C2's p-value
  declared uncorrected; the book result stated on both Sharpe conventions; the holdout/GSW window mismatch and
  the like-for-like GSW reference. Not fixed (minor, disclosed): tranche sub-books start on slightly different
  dates and are not rebalanced; the Cleveland Fed series is a full-sample model fit (biases the holdout towards
  passing, so the fail stands); dividend credit on an ex-date trade is one day late in both engine and replica.
- Tests run: replica vs engine validation (above); signal equality 282/282.
- Residual risk: the holdout cannot test the breakeven gate; pre-2003 inflation data are modeled; ~0.07 of the
  article's Sharpe unexplained; one Norgate vintage; the in-sample edge rests on two episodes.
