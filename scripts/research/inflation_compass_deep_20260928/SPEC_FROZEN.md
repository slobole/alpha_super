# Inflation Compass deep research - frozen specification

Written 2026-09-28, after the T5YIE publication-lag fix and before any result of this study was computed.
Owner request (2026-09-28, Hebrew): deep quant research on the Inflation Compass - original article
(CSS Analytics, 2026-07-27) and Allocate Smartly's review; quant pitfalls, is there a real edge, sensitivity,
other mechanisms/assets to improve Sharpe or CAGR, bring it to readiness, and a clear verdict.
Research-only: no live, release, scheduler, broker, engine or portfolio YAML changes.
Later changes are appended as dated amendments; nothing above the amendment log is edited after results.

## Known before freezing (disclosed so it is not mistaken for a finding)

- Causal module (fixed 2026-09-28): 2003-05-01..2026-08-19 CAGR 21.07%, Sharpe 1.085, MaxDD -24.3% (daily);
  the over-lagged anchor gives 20.17% / 1.046 via one exact tie (2023-04-28). Full history to 2026-09-25 through
  Bench: 20.87% / 1.075.
- Article (Varadi): 2003-2026 CAGR 23.5%, Sharpe 1.41, MaxDD -16.2% (month-end sampled; -23.6% daily), trades at
  the month-end close; his grid: threshold 1.8/2.0/2.2, windows 40/60/80, SMA 100-252; "remove T5YIE -> Sharpe
  0.82"; one-day delay 18.7% / 1.15. Allocate Smartly: 1990+ test (pre-2003 via proxies), prefers an ensembled
  "Enhanced" version (QQQ for XLK, PDBC with XLU in stagflation).
- Pakal study (knowledge base): 22.44% causal CAGR with same-date T5YIE; failed its frozen 2013-2019 Sharpe gate.
- The whole 2003-2026 T5YIE era was seen by the rule's author before publication. The only fully untouched
  data are (a) pre-2003 proxy history (below), (b) anything after 2026-07-27 (too short to test).

## Question

1. Why does our causal next-open implementation show Sharpe ~1.08 when the article shows 1.41?
2. Is there a real, economically explainable edge beyond (a) owning tech/energy in a 2003-2026 bull market and
   (b) a plain SPY 200-day trend filter? Specifically: does the inflation axis add value?
3. How fragile is it (parameters, ties, rebalance day, execution timing)?
4. Can a small, pre-declared set of changes improve it robustly?
5. What role, if any, should it have in the fund-menu books, versus T-bills in the same slot?
Final output: one verdict - keep as is / adopt a named change / demote / drop - and readiness gaps.

## Data and fixed settings

- Norgate ETFs: signals TOTALRETURN closes, fills/marks CAPITALSPECIAL with dividends (module contract).
- T5YIE current-vintage FRED (0 ALFRED revisions since 2014), published = observation dated < T; anchor dated T-L.
- DTB3 for T-bill cash (lagged one session), used for cash sleeves and excess-return statistics only.
- Main window: executions 2003-05-01 -> 2026-08-19 (audit window, reproduction anchor 21.072% / 1.0850).
  $100k, 5 bps slippage per side, zero commission (module contract). Stress: +10 bps per side.
- Periods: P1 2003-05..2012-12, P2 2013-01..2019-12 (the Pakal gate period), P3 2020-01..2026-08.
- Sharpe on daily returns, zero risk-free rate (house rule); excess-of-T-bill Sharpe also reported.

## Instrument (Phase 0)

A fast vectorized replica of the Vanilla engine for monthly target-weight rotation: close-to-close returns on
held shares, switch at Open_(T+1) with 5 bps per side on traded value, whole-share rounding ignored.
Acceptance: on the fixed module's weights it must reproduce the engine's NAV path with CAGR within 0.10 pp and
Sharpe within 0.01, daily-return correlation >= 0.999; otherwise the residual is explained. Finalists are re-run
through the real engine.

## Phase 1 - reconcile with the article

Decompose the 1.41 vs 1.08 gap one step at a time: (a) same-day close execution with same-date T5YIE (the
article's setup), (b) same-day close with published T5YIE, (c) next open (ours), (d) month-end-sampled vs daily
Sharpe/MaxDD, (e) costs on/off, (f) TOTALRETURN vs CAPITALSPECIAL marks. Report each step's delta.

## Phase 2 - is there an edge (mechanism tests)

- Benchmarks: SPY, QQQ, XLK buy & hold; equal-weight 9 sectors; 60/40 SPY/IEF; SPY>SMA200 else IEF; XLK>SMA200
  else IEF; QQQ>SMA200 else IEF; the static mix with the rule's average weights (time-in-regime weights,
  monthly rebalanced).
- Axis decomposition: growth-only (inflation fixed off -> XLK / 50 XLP+50 IEF), inflation-only (growth fixed
  on -> XLE/XLK), full rule.
- Map permutation: all 24 assignments of {XLE, XLK, XLU, XLP+IEF} to the four regimes; rank of the literal map.
- Placebo inflation axis: replace inflation_on with (i) circular shifts of the monthly inflation_on series by
  12..(N-12) months, (ii) 1,000 random regime series with the same on-frequency and the same run-length
  distribution (block shuffle of spells). p-value = share of placebos with Sharpe >= actual.
- Forward-return test: next-month XLE-XLK return spread on inflation_on (growth-up months only), OLS with HAC
  (lag 3) t-stat, by period.
- Timing vs static exposure: regress rule excess returns on the static-mix excess returns (and on XLK, XLE, SPY);
  report alpha and HAC t.
- Pre-2003 holdout (used once, baseline + finalists only, after finalists are chosen):
  * H1 1999-01..2002-12: sector SPDRs + real ETFs where available; inflation from Cleveland Fed 5-year expected
    inflation (FRED EXPINF5YR, monthly; the value for month m is used only from the end of month m+1).
  * H2 1983..1998: Kenneth French 12-industry daily value-weighted returns (Enrgy->XLE, HiTec->XLK, Utils->XLU,
    NoDur->XLP, Manuf->XLI, Money->XLF, Chems->XLB, Hlth->XLV), S&P 500 ($SPX) for growth, a constant-maturity
    7-10y Treasury return proxy from DGS10 for IEF, EXPINF5YR for inflation as in H1.
  * Literal 2% threshold only (declared now); a note says whether the gate is ever off.
  Holdout pass: Sharpe > SPY buy & hold Sharpe and > SPY>SMA200 timing Sharpe over H1+H2.

## Phase 3 - sensitivity (reported as distributions, not for selection)

- Threshold {1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5}; breakeven lookback {20, 40, 60, 80, 120}; slope lookback
  {20, 40, 60, 80, 120}; SMA {100, 150, 200, 250}; full joint grid 7x5x5x4 = 700 cells.
- Tie rule (> vs >=), anchor dated vs lagged one session.
- Rebalance-day offset: decision at month-end -10..+10 sessions (timing luck).
- Execution: next open (base), next close, second open.
- Instruments: QQQ for XLK; TLT/IEF/SHY for IEF; XLP alone, IEF alone.

## Phase 4 - improvement candidates (declared now; nothing else will be promoted)

- C1 Ensemble: equal-weight average of the targets of the 27 cells threshold {1.8,2.0,2.2} x breakeven lookback
  {40,60,80} x slope lookback {40,60,80} (SMA 200 fixed); fractional weights.
- C2 QQQ instead of XLK.
- C3 Stagflation cell 50% XLU + 50% DBC (DBC from 2006-02; before that XLU 100%).
- C4 Tranching: four equal sub-books deciding at month-end, -5, -10, -15 sessions; each holds one month.
- C5 Hysteresis on the 2% level: switch on at > 2.1, off at < 1.9.
- C6 Volatility target 15% annual on 63-day realized vol of the held sleeve, max 100%, rest in T-bills
  (cash earns lagged DTB3).
- C7 = C1 + C4 (both are specification-risk reducers).
Promotion ("better"): (i) paired stationary block bootstrap (mean block 21 days, 5,000 draws)
P(dSharpe <= 0) < 0.05 over the main window; (ii) dSharpe > 0 in P1, P2, P3; (iii) still better with +10 bps per
side; (iv) deflated Sharpe >= 0.95 with the full trial count; (v) holdout Sharpe not worse than baseline by > 0.10.
"Robustness replacement" (C1, C4, C7): non-inferior if dSharpe >= -0.05, bootstrap 5th pct >= -0.15, MaxDD not
worse by > 3 pp, each period within -0.10; preferred if non-inferior, because it removes knife-edge dependence.
Trial count for deflation: this study's configurations + the article's disclosed grid (15) + Allocate Smartly's
variants (~4) + Pakal study variants (from its ledger) + this repo's module variants; reported explicitly.

## Phase 5 - book role

Using the fund-menu sleeve inventory (growth_shelf_v2 `shelf_books`, independent pods, annual reset,
2012-10-02..2026-08-19) with the corrected Compass sleeve: for each menu book that holds Compass, compare
(a) as is, (b) Compass weight moved to T-bills, (c) moved pro rata to the book's other sleeves, (d) the promoted
Compass variant if any. Report CAGR, Sharpe, MaxDD, Calmar. Compass earns its slot only if (a) beats (b) and (c)
on Sharpe and Calmar.

## Phase 6 - readiness

List the gaps to PAPER: release-time cutoff (T5YIE ~17:00 ET), negative cash (G-028), live data route, FRED
failure rule, forward shadow length. Capacity is checked only as a sanity bound (sector ETF opening auctions).

## Amendment log

### A1 - 2026-09-28, after Phase 1-3 results (before Phase 4 results and before any holdout computation)

- Instrument note: the engine withholds 25% of long dividends by default; the replica does the same
  (validation: CAGR -0.003 pp, Sharpe -0.0001, daily correlation 0.9999998 vs the engine).
- Holdout execution: the Kenneth French data have no open prices, so every holdout run (baseline and finalists)
  executes at the NEXT CLOSE after the decision (lag one session, close fill), with the same 5 bps per side.
  The main-window next-close baseline (20.40% / 1.053) is the like-for-like reference. No dividend withholding
  can be applied to FF/synthetic total returns; this is stated beside the holdout numbers.
- Post-hoc mechanism probe, labelled EXPLORATORY (not a candidate, not promotable): is the T5YIE signal mainly an
  oil/commodity momentum proxy? (i) correlation of 60-session T5YIE change with 60-session DBC and XLE returns;
  (ii) the rule with the T5YIE 60-session change replaced by DBC 60-session return > 0 (from 2006) and by
  XLE-minus-XLK 60-session return > 0; the 2% level gate kept.
- Timing-luck view: the average NAV of the 21 rebalance-day offsets (-10..+10) is reported as the expected
  result for an arbitrary rebalance day. It is a diagnostic, not a candidate.

### A2 - 2026-09-28, after the holdout data description and before reading any Phase 4 result

- Holdout data facts (description only, no rule computed): EXPINF5YR is above 2.0 in all 240 months 1983-2002
  (min 2.014), so the level gate never binds; its monthly changes correlate 0.03 with T5YIE changes (2003+).
  The holdout therefore tests the growth leg and the "rising expected inflation OR basket slope" leg with a
  different inflation measure; it cannot test the T5YIE mechanism. This is stated beside every holdout number.
- Primary holdout = 1983-01..2002-12 on proxies throughout (French 12 industries, $SPXTR spliced with $SPX price
  before 1988 = benchmark understated by ~3.8%/yr in 1983-87, synthetic 7-10y bond). Secondary = 1999-06..2002-12
  with the real sector SPDRs. The pass test in the spec (Sharpe > SPY buy & hold and > SPY>SMA200 timing) is
  applied to the primary holdout.
- Candidates that cannot be expressed on the proxy data (C2 QQQ, C3 DBC, C5 hysteresis with a gate that never
  binds) are marked "not testable in holdout"; if one is otherwise promotable it is reported as "passes except
  holdout: untestable" and not promoted.

### A3 - 2026-09-28, AFTER the holdout result (post-hoc, EXPLORATORY; cannot change any frozen verdict)

The frozen holdout failed (primary 1983-2002 Sharpe 0.742 vs SPY>SMA200 0.957; 2002 utilities drawdown while the
proxy gate could never switch off). One exploratory check only: 1999-2002 with the real sector SPDRs and the
Federal Reserve's reconstructed 5-year TIPS breakeven (Gurkaynak-Sack-Wright, feds200805 BKEVEN05, daily from
1999), used with a one-session lag like T5YIE. This asks whether the real level gate would have avoided the 2002
utilities position. The GSW curve is a later fitted reconstruction, not real-time data; the result is reported as
exploratory and does not count as a holdout pass.
