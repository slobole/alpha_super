# Defensive shelf v2: robustness checks and the third-slot question

Dates: 2026-09-29 (plan, first run) to 2026-09-30 (A1-A3). Owner request (Hebrew): rebuild the defensive shelf with
fair cash, robustness first, report in English; the owner delegated the quant design. Report page: artifact
"Defensive Shelf v2" (https://claude.ai/artifact/NRuDbY9RUi1srPeNYim6ku, private; copy in
`results/research/portfolio/defensive_v2_20260929/report/defensive_v2.html`). This file is the record.

- Plan: `scripts/research/defensive_v2_20260929/SPEC_FROZEN.md` (frozen before any run). A1 (after results, owner):
  objective LONG Sharpe instead of excess CAGR. A2 (before these checks ran): robustness checks. A3 (after the A2
  results and the independent review, labelled): review follow-ups and the owner's DV2-IND candidate.
- Code: `defensive_v2.py` (first run), `reselect_sharpe.py` (A1), `sharpe_checks.py` (A2), `friction_v2.py`,
  `review_followups.py`, `eom_cost_breakeven.py`, `candidate_dossier.py` (A3), `report_data.py`, `build_report.py`,
  `report_template.html`. Tests: `tests/test_research_defensive_v2_20260929.py` (4).
- Inputs: the shelf rebuild's 27 sleeve runs at HEAD f9ad358 ($1M, to 2026-08-19), its 2008 BTAL proxy and A2
  DV2-IND fill; fair cash (idle cash at max(DTB3 - 0.5%, 0)); BIL TR as T-bills. Families: MAIN 949 books (1-4 of 8
  defensive sleeves, EQ or IV, with 0/10/20% of G3), LOW-TOUCH 34 books.

## Results

- LOW-TOUCH (CORE5, BTAL_QQQ, TFI only): no book passes the gates. The binding gate is R2: CORE5 alone and the
  60/40 champion (CORE5 60 / BTAL_QQQ 40) break -10% on 13-14% of 2,000 bootstrap paths (limit 10%).
- A1 pick (fair Sharpe): CORE5 + EOM + DISP [IV], Sharpe 1.72 vs 1.31 for the champion, max DD -6.1%.
- A2: the pick holds with 0% idle cash, with excess Sharpe (book - BIL) as the objective, and in all four input
  sensitivity frames; the no-EOM line is unstable (its pick changes each frame; under excess Sharpe the champion is
  kept). Zero-rate Sharpe does favour cash-heavy sleeves slightly (0.07 of the 0.41 gap); excess Sharpe keeps the
  ranking.
- Independent review (Tier 1): numbers reproduced, no look-ahead. HIGH: the A1 pick won a 2.2-point tie-break that
  counted DISP (daily, PM_READY) as live; the SPEC's wording counts it as not live, which picks
  CORE5 + BTAL_QQQ + EOM [IV]. MEDIUM: the A2 frames barely touch EOM; the fixed-book CSCV rank is not out of sample;
  the champion fails R2. LOW: R2 fail count, partial-year labels, ledger (all fixed).
- A3: under the SPEC-literal reading the pick is CORE5 + BTAL_QQQ + EOM [IV] in 3 of 4 fair/house x Sharpe/excess
  runs. With +10 bps per side on EOM's fills, without 2008-11, or selecting on 2017-26 only, the rule moves away from
  EOM books. Forward split: selected on 2008-17 the rule picks BTAL_QQQ + EOM [IV] (2017-26 Sharpe 1.88 vs 1.74
  champion); selected on 2017-26 it picks CORE5 + DISP [IV], which scores 0.80 in 2008-17 (champion 0.88).
- EOM: 45x annual turnover, so each bp per side costs ~0.45%/yr on the sleeve; the EOM book stops beating the
  champion at ~20 bps per side extra. Rolling 3-year excess over BIL: ~20%/yr (2011), 5-15% (2017-2025), +2.1% by
  Aug 2026 (-1.6% with 0% cash); in the last three years T-bills in EOM's place would have helped both EOM books tested.
- DV2-IND (owner candidate, CORE5 + BTAL_QQQ + DV2-IND [IV]): Sharpe 1.51 LONG, 1.06 in 2008-17, 1.91 in 2017-26,
  2.35 in the last three years (best of the candidates); fails R4 over the LONG window only because of 2008-12.
  Turnover 19x; breakeven ~17 bps per side.
- Four-pod CORE5 + BTAL_QQQ + EOM + DV2-IND [EQ]: Sharpe 1.84 (0% cash 1.75), max DD -5.3%, P(-10%) ~0%, 2008-17 1.55,
  2017-26 2.14, last 3y 2.39 (CAGR 13.9% vs 16.5% for the champion); plateau 1.75-1.87; with +10 bps on both EOM and
  DV2-IND 1.56, +20 bps 1.28 (champion 1.31). Fails R4 by a hair on CORE5 (1.68 vs 1.71). Capacity $10M open auction,
  $100M worked; friction 1.0-2.2%/yr at $30K, 0.2-0.6% at $100K.

## Recommendation (judgement, labelled in the report)

Target the four-pod book as the defensive core because it does not depend on either edge alone. Keep the 60/40 live
until EOM (month-end close auction + TLT short, gap G-032) and DV2-IND (shadow, daily open) are wired and their
real fill costs are measured against the ~15-20 bps limits; if EOM is still asleep then, use
CORE5 + BTAL_QQQ + DV2-IND. Without EOM or DV2-IND, keep the 60/40.

## Open owner decisions

Approve the four-pod target (or one of the three-pod books); wire CORE5; order the EOM and DV2-IND engineering and
forward shadows; low-touch line: accept a 13-14% breach probability or relax R2 to 15%; update the portfolio map's
Defensive and Balanced rungs once the core is chosen.
