# Amendments to TREND_BREAKOUT_PREREG_20260927.md

The frozen PREREG file is left byte-identical (SHA-256 `240753210944e8c8b16066a7a10046a9a52e0d28fd014569367ec20595ef23ae`,
recorded in `results/research/trend_breakout_20260927/prereg_freeze.json` at 2026-09-27 14:21:15 +03:00, before any cache
rebuild, parity run or grid cell). Everything below was written on 2026-09-27. No item changes a rule or a verdict.
Items N1-N6 and BUG1 were logged by the research agent in `amendments_and_bugs.json` during the run; items R1-R8 come
from the lead's check and the independent quant-pitfalls review after the results.

## Logged during the run (research agent)

- **N1 / N6 - engine artifact ("phantom fills").** In historical-share mode the engine maps raw shares to ledger units
  with U_t / Close_t. Norgate's adjusted closes carry per-bar rounding, so that ratio drifts by about 1e-8 between days,
  and a month-end re-size to an UNCHANGED raw share count becomes a ~1e-5-raw-share order that the engine fills at the
  $1 minimum commission. The replica reproduces the engine exactly, so both show it (41 such fills over 26 years in the
  17 NDX family-A cells, about $41; the live pod's own ledger has one). V1 (section 7) was therefore judged on NAV with
  those orders cancelled in both runs (`invariance_v1_phantom_free.json`: 2.4e-15 NDX, 1.7e-15 SP500); positions and
  decisions were identical without the adjustment. This is an engine bug outside this study's scope.
- **N2** - family C: sample standard deviation (ddof 1); a score needs every residual of its window and the regression
  at least 0.8 x W valid pairs.
- **N3** - a stop is evaluated only when close, highest-close-since-entry and (for CH) ATR20 are finite; entries and
  refills are skipped when the fill-day close is missing (0 occurred).
- **N4** - episode P&L and "terminal P&L share" = P&L of episodes closed by terminal liquidation / (final NAV - $100,000);
  each book window is its own pod-model run from target weights.
- **N5** - family B has 16 distinct cells, not 17 (B0 belongs to both stages): 36 configurations and 56 book trials
  instead of 37 / 58. The DSR keeps the frozen N = 293 as a lower bound.
- **BUG1** - the G2 parity runner could not look up the off-grid check cell B/N250/k8/K10/R2 (KeyError after cells 1-3
  had passed); fixed with a key parser and cell 4 re-run alone before the gate was declared passed (15:04:28) and before
  any grid cell ran (15:04:52).

## Added after the results (lead and independent review)

- **R1 - V2 diagnostic re-run.** The first V2 run reported a highest-close-since-entry relative difference of 119 and an
  infinite refill-score difference because the diagnostic compared quantities in mixed units; the check code (not any
  strategy code) was fixed and re-run (`log_invariance_v2_rerun.txt`): 0 mismatches, feature differences at Norgate's
  factor rounding (5.5e-6) or below. Not logged by the agent; recorded here.
- **R2 - code edited while the grids ran.** `simulate.py` (15:10) and `run_study.py` (15:06) were edited during or
  after the grid runs. The independent reviewer re-ran the current code and reproduced the saved B returns
  bit-for-bit (SP500 R2 cells, NDX B0 and R2 cells).
- **R3 - unsaved post-hoc code.** The `posthoc_sensitivities` block of `results.json` (both-legs haircut and similar)
  was written by code that is not saved; its numbers reproduce. The lead's own post-hoc diagnostics are saved:
  `posthoc_terminal_diagnostic.py` and `posthoc_spy_controls.py`.
- **R4 - R5 capacity.** R5 was evaluated on the candidate centre's capacity, not on a neighbourhood median; both
  neighbourhood cells pass ($50.7M and $101M for the largest order at 5% of ADV20, 2021-26).
- **R5 - terminal-economics flag not emitted.** PREREG section 5 asks to flag a candidate whose terminal P&L share
  exceeds 5%. The passing B2 centre is at 4.69% with the N4 denominator (NAV P&L) and 6.05% with the episode-P&L
  denominator, so it straddles the threshold: report it as "borderline terminal-economics dependent". All 43 terminal
  liquidations of the centre are completed acquisitions (for example WYE, BNI, GENZ, CELG, AET, RHT, TIF, ALXN, EA; no
  price series resumed; none distress-like), so the base case (last close ~ deal consideration) is the fairer reading
  than the flat 25% haircut.
- **R6 - optional SP500 engine replay.** Section 6 made a SP500 engine replay optional; it was not run in the gate, so
  the passing leg had been replayed by the engine only on NDX (B0 and an R2 cell). The lead ran the engine replay of the
  B2 centre on SP500 after the results: identical to the replica (max daily difference 6.7e-16, same final NAV,
  identical positions on all 6,697 sessions; `parity_g2_SP500.json`).
- **R7 - wording.** "Stops never beat L" holds standalone (16 of 16 below L at the month-end schedule, Reality Check
  p = 0.99). Inside the book 2 of 16 cells are marginally above G3 (PT-25 CASH +0.006, PT-20 REFILL +0.003), and PT-20
  CASH beats the same-offset G3 on 18 of 21 rebalance offsets (median +0.016). Correct wording: "no stop beats L
  standalone; loose percentage stops are roughly neutral inside the book".
- **R8 - multiplicity labels.** The book-level DSR tests Sharpe > 0 (G3 itself scores 0.99998) and says nothing about
  the incremental claim; the paired bootstrap and the Reality Check are the relevant labels. N = 293 is a lower bound:
  other searches aimed at lifting G3 (the MR search's 3,360 cells, DV2, the growth shelves) are not in it.

## Post-hoc controls (not pre-registered; interpretation only)

- **Interior neighbour.** The B2 pass does not depend on the K = 10 edge: the interior K = 20 / R2 neighbourhood also
  passes R1-R5 (margins +0.016 / +0.035 / +0.040, DD +2.72 / -1.38 pp, R4 NDX 0.570 vs B0 0.531). The NDX R4 knife-edge
  of the centre (0.53066 vs 0.53057) comes from the two-cell edge median; on NDX the K ordering reverses.
- **S&P 500 controls in the same quarter slot** (`posthoc_spy_controls.py`): passive SPY (price only, no costs)
  -0.095 / +0.032 / +0.045 by block, G-FULL +0.036; SPY gated by SMA200 -0.042 / -0.009 / +0.053, G-FULL +0.010;
  B2 centre +0.036 / +0.059 / +0.040, G-FULL +0.052. Most of the book gain is adding S&P 500 exposure; B2's distinct
  value over passive SPY sits mainly in 2008-11 (the synthetic-proxy block).
- **After the study END** (2026-08-20..2026-09-25, outside every block): B2 centre -5.6%, SPY +0.3%, L -0.8%.
