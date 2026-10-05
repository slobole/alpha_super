# Defensive shelf v2: robust main and low-touch lines (frozen plan, 2026-09-29)

Written before any book of this study was built. Later changes go to the amendment log at the end, dated, with
whether they came before or after a result. SHA-256 of this file is written to the study ledger.

## Owner request (Hebrew, 2026-09-29, paraphrased)

Rebuild the defensive shelf taking the best of both earlier studies: the main and low-touch lines of the fund menu
(2026-09-23) and the breach-probability lens and robustness checks of the shelf rebuild (2026-09-29). Treat idle
cash fairly. Consider every defensive candidate, for example CORE5 + BTAL_QQQ + industry-ETF DV2, against
CORE5 + BTAL_QQQ. Robustness is what matters. Report in English. The owner delegates the quant design.

## Known before this freeze

Shelf rebuild 2026-09-29 (same sleeve runs): D* = CORE5 + EOM (house cash), D' = CORE5 + BTAL_QQQ IV; the 0% idle-cash
convention biased slot tests against mostly-cash sleeves; under fair cash the rule picked CORE5 + BTAL_QQQ + EOM +
DOWNSHOCK IV. The strategy inventory (same day, fair cash) showed EOM, industry-ETF DV2, DISP and DOWNSHOCK with
significant alpha and low factor R², Trinity mostly beta, Tactical FI without significant alpha. The 2008-2026 history
has been examined many times; this is not a blind test.

## Inputs

The shelf rebuild's sleeve runs (27 at HEAD f9ad358, $1M, to 2026-08-19), its 2008 proxy for the BTAL TAA sleeves
and its A2 fill for industry-ETF DV2, all read through `shelf_rebuild_20260929/lib.py`. **Fair cash everywhere:**
every sleeve return includes the cash-realism add (positive idle cash at max(DTB3 - 0.5%, 0), negative at DTB3 + 1.5%,
prior close cash and prior DTB3); T-bills = BIL TR. Windows: LONG 2008-03-04 -> 2026-08-19 (primary), EXACT
2012-10-02 -> 2026-08-19, blocks A/B/C and RECENT as in the shelf rebuild. Book model: pods, annual reset (EQ or IV,
as in the shelf rebuild).

## Families

Growth slice option S in {none, G3 at 10%, G3 at 20%}, G3 = TAA3x 50 / NDX-VXN 50 (the live pair), taken out of the
book pro rata. Defensive pods are chosen from a pool, 1 to 4 of them, weighted EQ or IV among themselves.

- LOW-TOUCH line: pool {core5, btal_qqq, tactical_fi}; only monthly-MOO pods; subsets of size 1-3.
- MAIN line: pool {core5, btal_qqq, tactical_fi, trinity, eom_flow, downshock, disp, etf_dv2}; subsets of size 1-4.
Every book = defensive part (EQ or IV) scaled to 1 - S, plus S of G3 (TAA3x and NDX-VXN each S/2).

Benchmark (owner, 2026-09-29): CORE5 alone. Champion: CORE5 60 / BTAL_QQQ 40, fixed weights, annual reset (the
owner's "very strong, very simple" version); it is also a member of both families. References (not selectable): the
fund-menu YAMLs fund_menu_defensive and fund_menu_low_touch_defensive as they are now.

## Gates (all must hold on the main inputs)

- R1 historical: LONG max drawdown >= -8% (a buffer inside the owner's -10%).
- R2 breach: the probability that max drawdown is worse than -10% on 2,000 stationary-bootstrap paths of the book's
  LONG daily returns (mean block 63 sessions, seed 20260929) is <= 10%.
- R3 every period: excess CAGR over T-bills > 0 in blocks A, B, C and RECENT.
- R4 every pod earns its slot: replacing any pod's capital with T-bills (same weight rule) lowers LONG excess Calmar.
- R5 robust to inputs: R1, R3 and R4 also hold under every sensitivity frame (unscaled BTAL proxy; +5 bps per side;
  Tactical FI in its frozen FRED mode; industry-ETF DV2 idle before 2010). R2 is not re-bootstrapped per frame.

## Objective, ties, tie-break

Objective: LONG excess CAGR over T-bills (fair cash). Tie band: gate-passing books that the top gate-passer beats on
fewer than 90% of paired bootstrap paths (same 2,000 paths). Tie-break, in order: (1) fewer pods (G3 counts as two);
(2) lower weight in pods without a live route today (shadow sleeves, eom_flow, trinity daily bands count here;
WIRED and PM_READY monthly pods do not); (3) lower breach probability (R2); (4) fewer trading days per year (EXACT);
(5) higher objective. One pick per line: MAIN* and LOW*.

Champion rule (robustness first): a line's pick replaces the 60/40 champion only if it beats the champion's
objective on at least 80% of the paired bootstrap paths AND its breach probability (R2) is not higher than the
champion's by more than 2 points. Otherwise the recommendation for that line is the champion, and the pick is
reported as the best challenger.

## Ease and AUM (reported for the picks, the champion and the benchmark)

Pods, live status of each pod, trading days per year, daily or MOC needs; capacity with the growth-shelf route model
(shelf_rebuild part_m.capacity: recommended AUM for MOO, MOC, worked + blocks, first failing gate, cost at $25M);
small-account friction at $30K and $100K (new runs for pods not already covered).

## Reported robustness (not used to select)

For every tie-band book and the picks: EXACT-window metrics; RECENT metrics; crisis and stock-bond co-fall windows;
the full rule re-run under each sensitivity frame (does the pick change?); CSCV out-of-sample rank of the picks and
the PBO of argmax-by-objective within each line; the weight plateau (the pick with each pod's weight moved by +-10
points, renormalised); house-cash (0%) numbers beside the fair-cash ones; small-account friction from the shelf
rebuild's runs where the pods coincide.

## Amendment log

- A1 (2026-09-30, AFTER the results; owner-directed): the owner said the defensive shelf is about Sharpe and
  protection, not return ("8% is fine"). The objective changes from LONG excess CAGR to LONG Sharpe (fair cash, zero
  risk-free rate, the house rule); the tie band, the tie-break and the champion rule use Sharpe on the same 2,000
  bootstrap paths. Gates R1-R5 are unchanged. The excess-CAGR selection stays in the record as the first run.
- A2 (2026-09-30, BEFORE any of these checks ran; owner asked to finish v2): the reported-robustness section is run
  for the Sharpe selection (A1), in `sharpe_checks.py`, plus three checks added now. None of them re-selects the
  recommendation automatically; each is reported beside the A1 picks.
  1. House cash (0% on idle cash, the small-account case): the full rule (gates R1-R5 on the house-cash versions of
     all five frames, R2 on its own bootstrap, Sharpe objective, tie band, tie-break, champion rule) for both lines.
  2. Excess Sharpe: the full A1 rule with Sharpe of (book - BIL) as the objective, under fair and house cash. Reason:
     with fair cash a mostly-cash sleeve earns about the T-bill rate at near-zero volatility, which raises a
     zero-rate Sharpe without any skill; excess Sharpe removes that.
  3. Sharpe under each sensitivity frame (same method as the first run's sensitivity picks), CSCV (PBO of
     argmax-Sharpe, fixed-book OOS rank of the picks, 16 blocks), weight plateau (+-10 points per pod,
     renormalised, fixed weights), sub-period and crisis/co-fall tables, tie-band composition (which pods recur),
     ease and capacity (part_m.capacity) and small-account friction for the pick pods (`friction_v2.py`).
- A3 (2026-09-30, AFTER the A2 results and the independent review; labelled post-result): follow-ups in
  `review_followups.py`, reported beside the picks, none re-selecting automatically.
  1. Tie-break (2) classification: the code counts only etf_dv2, eom_flow and trinity as "without a live route"; the
     SPEC exempts only WIRED/PM_READY *monthly* pods, so the literal reading also counts the daily PM_READY pods
     DISP and DOWNSHOCK. Every selection is shown under both readings.
  2. EOM stress frames (the A2 frames barely touch EOM): +10 and +20 bps per side on EOM's own fills (close
     auction), and the rule on the EXACT window only (2012-10 on, dropping EOM's strong 2008-11 years). Each frame
     re-evaluates R1, R3, R4 and R2 (own bootstrap) and the tie band on its own bootstrap.
  3. Forward split (true out of sample): the rule (R1-R4, Sharpe tie band, tie-break, champion rule) on
     2008-03-04..2017-06-30 only, the chosen book scored on 2017-07-01..2026-08-19; and the reverse.
  4. The review's runner-up CORE5 + BTAL_QQQ + EOM [IV] is added to every report table.
