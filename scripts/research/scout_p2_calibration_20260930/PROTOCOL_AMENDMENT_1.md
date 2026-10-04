# Scout P2 calibration — amendment 1 (frozen before its results)

Date: 2026-09-30. Written after the first run and its independent review, and committed before any amendment-1
result exists. It does not change what the first run found; it answers the review's objections with a new,
pre-registered run.

## Why

The review of the first run found:
1. **The gate choice sat on a coin flip.** DSR's false-pass rate sat at the 5% admissibility cut (10/200), and the
   cut equalled DSR's own nominal level, so any well-calibrated test lands on the boundary. With 2,200 seeds it was
   5.4%.
2. **DSR's deflation vanishes when the clustered N_eff collapses** (N_eff = 1 on every Zorro grid). False pass was
   22% at N_eff = 2.
3. **The edge labels were optimistic.** They were calibrated on one 400-year path; the true Sharpes were about
   0.74 / 0.45 / 0.27, not 0.8 / 0.5 / 0.3.
4. **MCPT was frozen at p ≤ 0.01 while DSR ran at a 5% level,** which is not a like-for-like comparison.

## Changes to the method

- **DSR benchmark:** `alpha.stats.psr_dsr.null_selected_sharpe_benchmark`, the expected Sharpe of the configuration
  that the plateau rule would select when no configuration has an edge. It is simulated from the family's own
  correlation matrix (20,000 draws). PSR against that benchmark ≥ 0.95 passes ("DSR-corr"). The old clustered DSR is
  still computed, for comparison only.
- **MCPT** is judged at p ≤ 0.05 (200 permutations), plain and stratified.
- **GARCH recursion on r²_{t−1},** as the original protocol says. The first run used the noise part only; the
  difference matters only in the edge cases.
- **Edge strength calibration:** `a` is set so that the MEAN Sharpe of the true configuration over 8 independent
  100-year paths hits 0.3, 0.5 and 0.8.

## Families (two, both with plateau selection)

| Family | Grid | Why |
|---|---|---|
| A (as before) | L ∈ {5, 10, 20, 40, 60, 120, 250} × θ ∈ {0, 0.25, 0.5} | a normal, varied grid (N_eff about 3-4) |
| B (near-duplicates) | L ∈ {16, 18, 20, 22, 24} × θ ∈ {0, 0.05, 0.1} | configurations nearly identical, so the clustered N_eff is 1-2 by construction |

## Cases and sample sizes (per family)

`noise_garch`: 2,000 seeds. `edge_030`, `edge_050`, `edge_080`: 500 seeds each.

## Candidate gate sets

1. DSR-corr
2. MCPT plain (p ≤ 0.05)
3. MCPT stratified (p ≤ 0.05)
4. DSR-corr and MCPT plain
5. any 2 of {DSR-corr, MCPT plain, walk-forward}

## Decision rules (frozen)

1. **MCPT null:** plain, unless plain's false-pass rate exceeds 6.0% in either family. The stratified null is known to
   absorb volatility-timing edges (P0 review), so it needs a reason to be chosen.
2. **Admissible set:** in BOTH families, the false-pass rate on `noise_garch` is ≤ 6.0% (a nominal-5% test with 2,000
   seeds has a standard error of about 0.5%) and the false-reject rate on `edge_080` is ≤ 30%.
3. **Choice:** among admissible sets, the highest mean power on `edge_050` across both families. Any set within
   3 points of the best counts as tied, and the tie goes to fewer tests. A remaining tie goes to the set whose size
   does not depend on estimating the number of trials (MCPT before DSR-corr).
4. **Fallback:** if nothing is admissible, report it and pick the lowest false-reject on `edge_080` among sets with
   false pass ≤ 6.0% in both families.
5. **Walk-forward and PBO** stay diagnostics unless a set containing walk-forward wins under rule 3.

## Part C re-run

Same dead cases as before, with DSR-corr (the plateau rule is replaced by the maximum, as in the original studies,
and the benchmark uses the same maximum rule). MCPT stays impossible there.
