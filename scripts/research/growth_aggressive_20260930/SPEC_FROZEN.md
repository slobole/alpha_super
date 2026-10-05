# GROWTH and AGGRESSIVE shelf for a 2/20 fund: investor net return under drawdown rungs (frozen plan)

Written 2026-09-30, before any book of this study was built or read. Later changes go to the amendment log at the
end, dated, with the reason and whether they came before or after a result. Nothing above the log is edited after
the first book is built. The SHA-256 of this file is recorded in
`results/research/portfolio/growth_aggressive_20260930/experiment_ledger.jsonl` (this worktree) at freeze time and
with every event.

## 0. What is already known (this is not a blind test)

Every sleeve's 2000-2026 history has been studied many times. Known before this freeze (shelf rebuild 2026-09-29,
house 0% idle cash): G3 (taa3x 50 / ndx_vxn 50) LONG 17.6% CAGR, -15.3% max DD, bootstrap P(DD < -20%) 13.5%;
G3 + DV2 18 + HPI 18 (core 64%) 18.1% at 6% breach; G3 core + capsule (DV2 12 / HPI 12 / industry-ETF DV2 12)
16.6% / -11.4% at 2% breach; TAA3x-1N + NDX-VXN 20.9% at 60% breach; NDX-ATR lost 47% in 2000 (NDX-VXN 29%),
before the LONG window; Inflation Compass parameters are a lucky peak; the HPI backtest refills an exited slot at
the same open while the live host waits a day (-2.6 pp a year on the sleeve; the owner chose to align live to the
backtest on 2026-09-28, not yet landed). The bootstrap and PBO below measure selection luck inside this study's own
search only; they cannot remove what earlier looks at the same data already did.

## 1. Owner goal (2026-09-30, agreed)

Products for a future 2/20 fund. Objective: the investor's NET return after fees. Two rungs:

| rung | historical LONG max DD (incl. 2008 proxy) | bootstrap breach limit | max breach probability |
|---|---|---|---|
| GROWTH (main product) | >= -17% | -20% | 15% |
| AGGRESSIVE | >= -22% | -25% | 15% |

Champion: G3. A pick replaces it only if it beats it on >= 80% of paired bootstrap paths without raising the
breach risk. Benchmarks: S&P 500 TR, 60/40, `portfolios/ladder_4_growth.yaml`. Two lines: MAIN (daily single-stock
mean reversion allowed) and LOW-TOUCH (monthly-trading pods only). TQQQ inside TAA allowed; portfolio-level margin
leverage is NOT selectable (descriptive "what if" only). Report capacity at $10M / $25M / $50M and the manager's fee
income at each.

## 2. Inputs (reused, unchanged)

- Sleeve runs: `results/research/portfolio/shelf_rebuild_20260929/sources` in the main checkout (HEAD f9ad358, $1M
  per sleeve, end 2026-08-19), loaded by `scripts/research/shelf_rebuild_20260929/lib.load_inputs` (main checkout,
  read only) with its A2 industry-ETF DV2 fill and the validated 2008 synthetic TQQQ/BTAL proxy (`splice_scaled`)
  before 2012-10-02 for taa3x, taa3x_1n, taa2x_1n, btal_qqq. No sleeve is re-run.
- MAIN frame = FAIR CASH (`cash_long`): positive idle cash earns max(DTB3 - 0.5%, 0), negative cash pays
  DTB3 + 1.5%, from each run's prior-day cash / NAV (`lib.cash_realism_add`). EXACT fair frame = `cash_exact`.
- T-bills = BIL TOTALRETURN (the T-bill pod, the slot-test replacement, the hurdle for "excess").
- Windows (lib): LONG 2008-03-04 -> 2026-08-19; EXACT 2012-10-02 -> end; blocks A (2008-03-04..2012-10-01),
  B (2012-10-02..2021-12-31), C (2022-01-03..end), RECENT (2023-08-21..end); crises = lib.CRISIS_DICT, every S&P 500
  TR decline >= 10% inside LONG, and the six worst stock-bond co-fall windows (lib.cofall_windows).

## 3. Fee model (2/20, the objective's currency)

Per session: pre-fee NAV compounds the book's gross return; management fee = pre-fee NAV x 0.02 / 252 is taken;
performance fee = 20% x max(pre-fee NAV - HWM, 0) is ACCRUED in the investor NAV every session; at the last session
of each calendar year the accrued fee is paid, and if it was positive the HWM resets to the post-payment NAV. No
hurdle, no loss carry beyond the HWM. (Identical to `growth_shelf_20260924/growth_dossier.fee_path`.) Investor net
NAV = pre-fee NAV - accrued performance fee. Net CAGR in calendar time from the book's NAV base. On bootstrap paths
a "year" is each consecutive block of 252 sessions from the path start. A window measured on its own (blocks,
RECENT) is a new investor: fee path restarted at the window start with HWM = 1.

## 4. Book model and family

Pod model as `lib.book_returns`: pods compound independently, reset to fixed target weights after the last close
of each calendar year (cost-free reallocation, disclosed). EQ weights only.

Book = core + satellite.
- Core legs: TAA leg in {taa3x, taa3x_1n, taa2x_1n} x NDX leg in {ndx_vxn, ndx_atr, ndx_natr20} x third leg in
  {none, compass_qqq}; core legs equal (1/2 each, or 1/3 each with Compass). 18 cores.
- Satellite type (15): MR with DV2 variant v in {dv2, dv2_adv, dv2_floor}: `v` alone, `pair` (v + hpi_vote),
  `capsule` (v + hpi_vote + etf_dv2) = 9; `hpi` (hpi_vote), `etf` (etf_dv2), `hpi_etf` (hpi_vote + etf_dv2) = 3;
  defensive slices `def3` (core5 + btal_qqq + etf_dv2, equal; the defensive-v2 candidate core) and `def2`
  (core5 + btal_qqq, equal) = 2; `tbill` = 1.
- Satellite share sigma in {0.18, 0.36} of the book, split equally inside the satellite; the core takes 1 - sigma.
  0.36 is the owner-approved capsule size (2026-09-26); 0.18 its half.
- MAIN family = 18 cores x (no satellite + 15 types x 2 shares) = 558 books.
- LOW-TOUCH family = the MAIN books whose every pod trades monthly (satellite in {none, def2, tbill}) = 90 books.
- Not searched (disclosed): TAA:NDX ratios other than 50:50, satellites combined with each other, share grids.
- Champion G3 = taa3x 50 / ndx_vxn 50 (in both families).

## 5. Gates (per rung R: hist limit H_R, breach limit B_R; GROWTH -17% / -20%, AGGRESSIVE -22% / -25%)

- R1: LONG max drawdown >= H_R for BOTH the gross book and the investor net NAV.
- R2: paired stationary bootstrap of the LONG daily returns (2,000 paths, mean block 63 sessions, seed 20260929,
  every book on the same resampled session indices): share of paths with max drawdown worse than B_R <= 15%, for
  BOTH gross and net.
- R3: investor net excess CAGR over T-bills > 0 in blocks B, C and RECENT (A is reported).
- R4: T-bill slot test, LONG, with the objective's metric (investor net CAGR): replacing any one non-T-bill pod's
  returns with T-bill returns (same weight) must lower the book's net CAGR.
- R5: a book with compass_qqq must beat its no-Compass twin (same TAA/NDX legs, satellite and share) on >= 90% of
  the bootstrap paths by net CAGR (strict twin test; the Compass parameters are a known lucky peak).
- LOW-TOUCH: family restriction above.

## 6. Objective, tie band, tie-break, champion test

- Objective: LONG investor net CAGR (fair cash).
- Tie band: among gate passers of rung R and line L, the top by objective; a passer is tied with it when the top
  beats it on fewer than 90% of the bootstrap paths (net CAGR per path).
- Tie-break inside the band, in order: (1) every pod passes the slot test on RECENT (net CAGR, new investor);
  (2) no shadow sleeve (dv2_adv, dv2_floor, ndx_natr20, etf_dv2); (3) lower bootstrap P(max DD < B_R), gross;
  (4) fewer pods; (5) higher objective. The first book is the rung-line pick P(R, L).
  Reason for (3) before pod count: the owner's objective is net return under a breach-probability constraint, and
  robustness comes first; among books the return data cannot separate, the one least likely to break the investor's
  limit is preferred. Ease is represented by the separate LOW-TOUCH line.
- Champion test, GROWTH (each line): P(GROWTH, L) replaces G3 only if (a) it beats G3 on >= 80% of the bootstrap
  paths by net CAGR and (b) its P(gross max DD < -20%) <= G3's (same frame, same paths). Otherwise the GROWTH
  product of that line is G3. If G3 fails a GROWTH gate in this study's frame, that is reported, and the product is
  P(GROWTH, L) labelled "by default".
- Champion test, AGGRESSIVE (each line): P(AGGRESSIVE, L) is offered as a separate product only if it beats that
  line's GROWTH product on >= 80% of the bootstrap paths by net CAGR (its own breach limit is gate R2). Otherwise
  there is no separate AGGRESSIVE product in that line (the GROWTH product serves both), and the AGGRESSIVE pick is
  shown as a labelled candidate. "Without raising the breach risk" for AGGRESSIVE therefore means within its own
  rung limit (G3's P(< -25%) is not a sensible ceiling for a riskier rung); this reading is a design choice.
- The top-objective gate passer of every rung-line and its champion-test figures are reported beside the pick.

## 7. Statistics

- PBO (CSCV, 16 contiguous blocks of the LONG sessions, 12,870 half/half splits) per rung-line over that rung-line's
  gate passers, for the rule "argmax of CAGR" (gross CAGR, a monotone proxy of net CAGR under 2/20 without the HWM
  path effect; labelled). Reported with the median out-of-sample rank. For the fixed books (each pick, the top
  passer, G3) the out-of-sample relative rank in every split and the share of splits in the top half.
- Every figure for the picks, G3, and the benchmarks: section-5 metrics of the shelf rebuild (`lib.full_metrics`)
  gross, LONG and EXACT; investor net CAGR, max DD, Sharpe, worst year; year-by-year gross and net; crises and
  co-falls; blocks; ease fields (`lib.ops_fields`).

## 8. Sensitivities (each re-runs the COMPLETE selection: gates, bootstrap on its own frame, tie band, tie-break,
champion tests; never used to select)

- S1 house cash: idle cash 0% (`long`), closer to a small account.
- S2 unscaled BTAL proxy: `long_unscaled` plus the fair-cash add of the main frame (the scaled proxy runs' cash add
  is applied to the unscaled proxy dates; approximation, labelled).
- S3 +5 bps per side on every traded dollar (`stressed_long` plus the fair-cash add).
- S4 industry-ETF DV2 idle (0%) before its first engine trade on 2010-01-13 instead of the A2 research fill.
- S5 HPI live gap: hpi_vote returns minus 2.6% a year (daily 0.026/252), the measured live-slot gap if the
  align-live fix does not land.
- S6 EXACT window (2012-10-02 -> end, `cash_exact`), gates on EXACT (R3 blocks B, C, RECENT unchanged).
- S7 bootstrap mean block 126 sessions (R2, R5, tie band and champion tests all on the new paths).
- S8 bootstrap mean block 21 sessions.

## 9. Descriptive, not selectable

- Margin "what if": G3 and every product at 1.25x and 1.5x gross exposure, rebalanced daily, borrowing at
  DTB3 + 1.5% (lagged DTB3), gross and net. Neither the engine nor the live host supports it; the investor
  one-pager promises no margin.
- Capacity: the growth-shelf route model (`growth_shelf_v2_20260926/shelf_books`: house auction limits for stocks,
  worked ETF orders, urgent ETF orders worked within one day) on the fills of RECENT, product weights fixed with an
  annual reset: cost per year and gate status at $10M, $25M and $50M for the MOO (today), MOC and worked+blocks
  routes, and the recommended AUM per route (cost <= 25% of the EXACT excess over T-bills and every gate passing).
- Manager fee income: management and performance fee as % of average AUM per year (LONG, EXACT, RECENT, and a
  "live 30% below backtest" case scaling daily gross returns by 0.7), times $10M / $25M / $50M, after the route's
  trading cost at that AUM.
- Fee mapping check: net CAGR against gross CAGR over the family (owner's rough mapping 17.6 -> 12.5, 20 -> 14.5,
  23 -> 16.5).
- Breach frontier: every family book's net CAGR against its P(breach) for both rungs.

## 10. Result versus judgement

Gates, objective, tie band, tie-break and champion tests produce the products mechanically. Anything the report adds
(for example a preference between lines, or offering an AGGRESSIVE candidate that failed its champion test) is
labelled as judgement. Every book tested is saved.

## 11. Review

Before the report is finalised an independent quant-pitfalls review agent (Tier 1) reads this spec, the code and the
results; its findings and the responses are added to the amendment log.

## Amendment log

- A1 (2026-09-30, AFTER the main-frame selection; post-result, descriptive only, nothing re-selected): the mechanical
  GROWTH pick of the MAIN line (G3 core + DV2 18 / HPI 18) fails champion test (a) (beats G3 on 62.6% of paths) and
  the band's top book fails (b) (breach 14.65% vs G3's 12.75%), while 3 other GROWTH passers meet both (a) and (b).
  Added to the report as a labelled descriptive table: every gate passer of each rung-line that meets both champion
  conditions, with its tier and ease fields. Choosing one of them would be a search over the champion test itself
  (selection on the same bootstrap paths), so any preference for them is judgement. Also added: G3's margin to each
  GROWTH gate (it passes R1 net by 0.8 pp and R2 net by 0.45 pp), because the champion's own robustness decides
  whether "keep G3" is a safe default.
- A2 (2026-09-30, AFTER all results and the Tier-1 review; post-result checks, nothing re-selected): the review
  (`results/.../growth_aggressive_20260930/REVIEW_TIER1.md`) found that G3's net P(DD < -20%) of 14.55% sits inside the
  Monte Carlo error of the 15% limit (SE 0.79 pp at 2,000 paths) and that G3 fails R2 on 6 of 7 other seeds, so the
  mechanical "G3 kept" result depends on the frozen seed. Added, labelled post-result: `multiseed.py` recomputes R2
  (gross and net, both rungs) and the paired beat-G3 share for the key books on 10 seeds (the frozen seed plus 9) with
  the same block length; the report shows the seed mean and range beside every R2 figure and treats R2 as an ordinal
  screen (it moves 7.7% / 12.8% / 34.7% for G3 at blocks 126 / 63 / 21). Also from the review: the reality-check
  p-values of the beat-G3 shares after the search are reported (bootstrap beat shares are not confidence); the
  fee-income "30% below backtest" case now subtracts 30% of the gross CAGR as an even daily drag instead of scaling
  returns by 0.7 (which also shrank volatility); ladder_4 (yaml `rebalance: null`) is shown with the study's annual
  reset and labelled so; margin what-if and annual reallocation are cost-free (disclosed). The recommendation that
  follows (step below the limit, LOW-TOUCH shelf) is judgement and is labelled so.
