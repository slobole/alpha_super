# Client barbell: the risky sleeve and defensive core of a split client account (frozen plan)

Written 2026-09-30 before any client-account book of this study was built or read. Designed from three independent
designs (allocator, statistician, practitioner lenses) and a gap critic (workflow wf_1531ff99-354). Later changes go
to the amendment log, dated, with reason and whether before or after a result. SHA-256 in
`results/research/portfolio/barbell_20260930/experiment_ledger.jsonl` at freeze time and with every event.

## 0. Not a blind test

Every sleeve, both fund verdict products (GROWTH, AGGRESSIVE, EA2), the defensive champion and the defensive
candidates were chosen on this same 2008-2026 history. This study is a second-level selection on top of them. All
evidence is in-sample; the champion test, reality check, PBO, forward split and year-drop checks measure only this
study's own search.

## 1. Owner question (Hebrew, 2026-09-30, paraphrased)

"I want to understand how I reach a higher CAGR. I can always split money between defensive and aggressive. Live
example: a first client with USD 1.5M; USD 0.5M goes to AGGRESSIVE (or GROWTH, to decide), USD 1.0M to the defensive.
I want a real, robust quant check." The client's risk is the whole account, not a sleeve.

## 2. Inputs

As the growth study (`../growth_aggressive_20260930/ga_lib.frames`): shelf-rebuild sleeve runs (HEAD f9ad358, $1M,
to 2026-08-19), LONG 2008-03-04 -> 2026-08-19 with the validated splice_scaled TQQQ/BTAL proxy before 2012-10-02 and
the A2 DV2-IND fill, EXACT 2012-10-02 on, FAIR CASH main frame (idle cash max(DTB3 - 0.5%, 0), negative cash
DTB3 + 1.5%), BIL TR as T-bills. Blocks A/B/C/RECENT, crises, co-falls as lib.

## 3. Account model

- Risky sleeve R: a fixed-weight book (annual pod reset) from the growth-study grammar (`select_ext.family` weights).
- Defensive core D: fixed (EQ/FIXED) or inverse-vol (IV, weights from the 252 sessions before each reset) book.
- Account = s x R + (1 - s) x D. Primary policy P0: after the last close of each calendar year both sleeves are reset
  to s / 1 - s, costing 5 bps per side on the dollars moved between sleeves (2 x 5 bps x |V_R - s V|). Pods inside a
  sleeve keep their own annual reset (cost-free, house convention). With EQ cores P0 equals one flat look-through book
  with weights s wR + (1 - s) wD (unit test to 1e-12 with zero transfer cost).
- s = 1/3 for the client picks (Part A); s in {0, 0.05, ..., 1} for the frontier (Part B).
- Other policies (reported, Part A picks and champions): P3 quarterly reset; P2 band (month-end check, reset when the
  risky share leaves s +- 5 pp, and +- 10 pp); P1 drift (never rebalanced) only on rolling month-end entry cohorts
  with 3y and 5y horizons (over 18 years drift is meaningless), with the risky-share distribution at horizon end.
- Minimum pod size USD 25K at USD 1.5M (look-through weight >= 1.67%); every book in the family satisfies it (checked).

## 4. Families

Risky sleeves (non-shadow only):
- Cores: TAA leg {taa3x, taa3x_1n, taa2x_1n} x NDX leg {ndx_vxn, ndx_atr} x TAA:NDX {50:50, 60:40, 70:30} (18), plus
  TAA-only {taa3x, taa3x_1n, taa2x_1n} and NDX-only {ndx_vxn, ndx_atr} (5) = 23 cores.
- LOW-TOUCH satellites: none, def2@18, def2@36, tbill@18, tbill@36 -> R_LT = 115 sleeves (contains G3, the GROWTH
  verdict 38.4/25.6/18/18 and the AGGRESSIVE verdict 57.4/24.6/9/9).
- MAIN adds wired daily satellites dv2@18, dv2@36, pair_dv2@18, pair_dv2@36 (dv2 + hpi_vote), hpi@18, hpi@36 ->
  R_MAIN = 253.
- Excluded (disclosed): NDX-RM (post hoc; fails its twin test without 2020 and 2026), NDX-NATR20, dv2_adv, dv2_floor,
  DV2-IND as a risky satellite (shadow), Compass (failed every twin test).

Defensive cores:
- FUNDED (deployable once CORE5 is wired): D0 CORE5 60 / BTAL_QQQ 40 FIXED (defensive champion); D1 CORE5 + BTAL_QQQ
  EQ; D2 CORE5 + BTAL_QQQ IV; D6 CORE5 alone; D7 BTAL_QQQ alone; D8 BIL; D9 BTAL_QQQ 50 / BIL 50.
- TARGET (not deployable now; reported, never selected for this client): D3 CORE5 + BTAL_QQQ + DV2-IND IV (owner's
  lean); D4 CORE5 + BTAL_QQQ + EOM IV; D5 CORE5 + BTAL_QQQ + EOM + DV2-IND EQ.

Lines (each a complete selection):
- LT-FUNDED (PRIMARY): R_LT x FUNDED = 805 client books at s = 1/3.
- MAIN-FUNDED: R_MAIN x FUNDED = 1,771.
- TARGET (reported): R_LT x {D3, D4, D5}.
- NOW (fallback, reported; only pods WIRED today): R from cores with taa3x/taa3x_1n and ndx_vxn/ndx_atr, TAA-only
  taa3x/taa3x_1n, NDX-only, satellites {none, tbill@18, tbill@36}; D in {D7, D8, D9}.

## 5. Objective modes

- GROSS (primary; no fee is charged today and this client's terms are open): LONG client CAGR in calendar time.
- NET-F1 (complete second selection): 2/20 on the TOTAL account, one HWM, `ga_lib.fee_nav` (daily accrual, paid at
  the last session of each year); risk gates on both gross and net.
- Reported fee structures for picks and champions: F2 2/20 per sleeve (separate HWMs, sleeves mixed at net NAV;
  netting cost F1 - F2), F3 1/10 on the total account, and manager income on USD 1.5M (LONG, EXACT, RECENT, and a
  "live 30% below backtest" drift haircut).

## 6. Client risk rungs

- RM (PRIMARY, "more CAGR without more risk"): LONG hist max DD >= CH's; EXACT hist max DD >= CH's; 10-seed mean
  DDaR10 >= CH's 10-seed DDaR10 (DDaR10 = 10th percentile of bootstrap path max DD).
- Decision table (absolute limits the client could sign): B in {-10%, -12.5%, -15%, -17.5%, -20%}; build rule LONG
  hist max DD >= B + 3 pp; 10-seed mean P(max DD < B) <= 10% (15% column reported).
- In NET mode the risk gates apply to both gross and net NAV.

## 7. Bootstrap (one engine)

Paired stationary bootstrap, 2,000 paths, mean block 63 sessions, seeds 20260929 + k (k = 0..9). Rows of
[all risky sleeves, all defensive cores] are resampled jointly (same session indices), and the between-sleeve policy is
applied on each path (reset every 252 sessions from the path start; a "year" = 252 sessions for CAGR and fees).
Computation rule: every book on the frozen seed; 10 seeds for every book whose frozen-seed gate metric is within
3 pp of its threshold or that passes on the frozen seed, and for champions, picks and tie bands; gates use the 10-seed
mean where computed, else the frozen seed (a clear fail by > 3 pp).

## 8. Gates, tie band, tie-break, champions

Gates per line, mode and rung: (G1) the rung's risk gates; (G2) client excess over BIL > 0 in blocks B, C and RECENT
(mode currency); (G3) slot test in the objective's metric: for each non-BIL pod of the look-through book, replacing
that pod's returns with BIL wherever it appears (both sleeves) must lower client LONG CAGR; (G4) client look-through
TAA share (taa3x + taa3x_1n + taa2x_1n) <= 25% (owner concentration policy at the account level; an uncapped line is
reported and never selects); (G5) line membership.

Tie band: the top gate passer by objective; a passer is tied when the top beats it on < 90% of frozen-seed paths.
Tie-break: (1) better 10-seed tail (RM: higher DDaR10; absolute rungs: lower breach); (2) lower taa_df signal-family
share (TAA legs + BTAL_QQQ; lineage: all in `strategies.taa_df`, verified in code before the run); (3) every pod passes
the slot test on RECENT; (4) fewer pods; (5) higher objective.

Champion CH = 1/3 AGGRESSIVE verdict + 2/3 D0, P0 (the owner's stated plan). CH-G = 1/3 GROWTH verdict + 2/3 D0 (the
pre-specified comparison). A pick replaces CH only if it (a) beats CH on >= 80% of frozen-seed paired paths (10-seed
range shown), (b) has a 10-seed tail no worse than CH's for that rung, and (c) beats CH's historical CAGR on EXACT and
in block C. Otherwise CH stays.

## 9. Pre-specified comparisons

- GROWTH vs AGGRESSIVE for the USD 0.5M: CH-G vs CH at s = 1/3 with every FUNDED core and D3: CAGR (gross, net),
  hist DD, 10-seed DDaR10 and breach at every B, paired beat share; and at iso-risk: the s at which the GROWTH sleeve
  matches CH's 10-seed DDaR10 (bisection), and its CAGR there ("hotter sleeve vs bigger split").
- Redundant-defence test: the AGGRESSIVE and GROWTH verdict cores without their def2 slice, at the s that matches CH's
  DDaR10, against CH.

## 10. Part B: how to reach a higher CAGR (descriptive)

- Frontier: risky sleeves {G3, GROWTH verdict, AGGRESSIVE verdict, TAA3x-1N + NDX-VXN at 50/60/70 no satellite,
  TAA3x + NDX-VXN 70:30, TAA3x-1N only, TAA3x only, TAA2x-1N + NDX-VXN 70:30} x cores {D0, D2, D8, D3, D5} x s grid;
  frozen-seed DDaR10, hist DD, CAGR; 10 seeds on the upper envelope and its neighbours. Envelope per client-risk
  budget, slope in pp CAGR per pp DDaR10, FUNDED and TARGET envelopes separately.
- Lever table from CH, each alone then combined: split (s 0.40, 0.50), sleeve, ballast (D0 -> D2, D3, D5, D8), fee
  (none / F1 / F2 / F3), rebalancing policy, TAA cap.
- Drawdown governor (descriptive, never selects): at month-end closes, when the account DD from its peak is below
  -X, the risky share is cut to s/2 until the DD is above -X/2; X in {6%, 8%}; for CH and the picks.

## 11. Robustness (each reported; frames re-run the complete selection of the PRIMARY line in both modes)

Frames: S1 house 0% cash; S2 unscaled BTAL proxy; S3 +5 bps per side; S5 HPI live gap (MAIN line); S6 EXACT window
(complete re-selection); S7 block 126; S8 block 21. For the picks and champions: 10 seeds; reset-date phase (12
month-end reset phases at pod and sleeve level); forward split (select on 2008-03..2017-06, evaluate on 2017-07..end,
and the reverse); year-drop jackknife (18 years); PBO (CSCV 16 blocks, argmax of client CAGR over passers) and fixed-
book out-of-sample ranks; White reality check of the pick vs CH over the passers (labelled, not a gate); factor alpha of
the client account (QQQ alone; QQQ/IEF/GLD/DBC/UUP; plus the QQQ 200-day rule; weekly, Newey-West 4 lags; gross and
net; LONG and EXACT halves); crises and co-falls; crisis correlation between R and D (S&P 500 worst 5% days); rolling
63-day beta of the account to QQQ (max and 99th percentile, the aligned-Nasdaq proxy); horizon breach on 3y and 5y
bootstrap paths; first-year loss risk (P(DD < -10% within 252 sessions)); rolling month-end entries (1y/3y/5y outcome
distributions), also conditioned on NDX-VXN under water by more than 10% at entry (as now).

## 12. Readiness, interim, capacity (descriptive)

Wiring CORE5 (and TAA3x-1N / TAA2x-1N if picked) with one clean paper cycle is a precondition before client money
enters a FUNDED core. Interim rule while CORE5 is unwired: the NOW line's best core for the USD 1.0M, and the cost of
a 1-, 3- or 6-month delay. Aggregate capacity: the number of such clients at which the defensive core's big ETF orders
(DBC/UUP/BTAL) need routing (earlier result: MOO fine to ~$1.5-7M). Currency: an ILS-based client bears USD/ILS risk
(flag). Taxes on rebalancing gains and VAT on fees are not modelled.

## 13. Result versus judgement; review

Gates, objective, tie band, tie-break and champion tests produce the picks mechanically; anything else is labelled
judgement. Before the report: an independent verification workflow (independent replication of key numbers, multi-lens
adversarial review, completeness critic); findings go to the amendment log.

## Amendment log

- P1 (2026-09-30, BEFORE any result; engineering only): the frozen seed's bootstrap (tie band, champion test, beat
  shares) runs in float64; seeds 2-10 (tail metrics DDaR10 and breach, beat-share ranges) run the same engine in float32,
  which is twice as fast (measured on 2,530 books x 200 paths: CAGR error <= 8e-6, max-DD error <= 6e-5, beat shares
  within 0.5 pp). The main frame runs as two processes (the LT-based lines, and MAIN-FUNDED) writing to separate folders.
  Reason: under SPEC 7 almost every client book is a contender (at s = 1/3 nearly all pass the -20% rung), so the
  10-seed stage covers ~2,000 books per frame.
