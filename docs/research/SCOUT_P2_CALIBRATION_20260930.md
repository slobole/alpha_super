# Scout P2 — calibration of the overfitting gates

Date: 2026-09-30. Research only: nothing here touches live trading, strategies or the engine.

Two pre-registered runs:
1. The first run (`PROTOCOL.md`, frozen in commit `5660232`).
2. Amendment 1 (`PROTOCOL_AMENDMENT_1.md`, frozen in `52e961b`). It was written after an independent review of the
   first run, and before any of its own results.

Code in `scripts/research/scout_p2_calibration_20260930/`. Results in `results/scout/p2_calibration/` (not in git).

## Verdict

- **S5 gate: the Monte Carlo permutation test of the whole search (MCPT), plain date shuffle, p ≤ 0.05.**
  - This is what amendment 1's frozen rule chose.
  - False pass on realistic noise: 4.4% on a varied grid, 4.5% on a grid of near-duplicate configurations (2,000
    histories each).
  - Power at Sharpe 0.52: 62% and 75%. At Sharpe 0.79: 93% and 95%.
  - Its size is exact by construction and does not depend on estimating how many independent trials were run.
- **DSR with a correlation-aware benchmark ("DSR-corr") becomes a WARN-level check, not a gate.**
  - It is sound (2.2% and 3.5% false pass) but more conservative (power 53% and 73% at Sharpe 0.52).
  - It is kept because it is the only test that also counts the family's earlier trials from the ledger. MCPT only
    sees the grid it re-runs.
  - A miss needs a written owner note before promotion. It does not stop the family.
- **The DSR in the original design was flawed.** It deflated by a clustered count of effective trials. When the
  configurations are near-duplicates that count collapses to 1, and the deflation vanishes: 11.8% false pass instead
  of 5%. On the real Zorro Z9 grid it passed a 24-way selection with no deflation at all.
- **Walk-forward (21-25% false pass on noise) and PBO (it cannot recognise a real edge shared by neighbouring
  configurations) are printed diagnostics only.**
- **S3's significance test is the date-level Newey-West t.**
  - With realistic dependence (sector factors, a signal that persists day to day), the naive per-event t-test rejects
    a true null 19% of the time.
  - The within-date permutation the design had planned for S3 rejects it 11% of the time.
  - The date-level Newey-West t held 5% everywhere.
- **The first run's conclusion ("DSR alone") is withdrawn.** It rested on 10 of 200 noise seeds sitting exactly at the
  5% cut, on the flawed DSR above, and on edge labels that were too high.

## Amendment 1 — the decisive run (14,000 search evaluations: 7,000 histories × 2 families)

### Set-up

- **Data:** one asset, 20 years of daily returns, zero drift, GARCH(1,1) volatility on r²ₜ₋₁ with Student-t(5) shocks.
- **Planted edge:** momentum in returns. Its strength is calibrated over 8 independent 100-year paths, so the true
  configuration's Sharpe in the simulated histories is 0.31, 0.52 or 0.79.
- **Family A (varied):** 7 lookbacks × 3 thresholds. The median clustered N_eff is 3.
- **Family B (near-duplicates):** lookbacks 16-24 × thresholds 0-0.1. N_eff is 1.
- **Selection:** plateau choice in both families.
- **Sample sizes:** 2,000 noise histories and 500 per edge strength, per family.

### Results

| Test | Family | noise (false pass) | Sharpe 0.31 | Sharpe 0.52 | Sharpe 0.79 |
|---|---|---|---|---|---|
| **MCPT plain, p ≤ 0.05 — chosen** | A | **4.4%** | 34% | **62%** | **93%** |
| | B | **4.5%** | 38% | **75%** | **95%** |
| MCPT stratified, p ≤ 0.05 | A | 4.1% | 34% | 63% | 92% |
| | B | 4.4% | 41% | 73% | 95% |
| DSR-corr ≥ 0.95 | A | 2.2% | 25% | 53% | 89% |
| | B | 3.5% | 37% | 73% | 94% |
| DSR-corr and MCPT | A | 2.0% | 24% | 53% | 89% |
| | B | 3.3% | 35% | 72% | 94% |
| any 2 of DSR-corr, MCPT, walk-forward | A | 3.9% | 29% | 58% | 92% |
| | B | 4.0% | 38% | 74% | 95% |
| old DSR (clustered N_eff) | A | 5.7% | 36% | 66% | 92% |
| | B | **11.8%** | 57% | 86% | 99% |
| MCPT plain, p ≤ 0.01 | A | 1.3% | 13% | 37% | 80% |
| | B | 0.7% | 19% | 49% | 84% |
| walk-forward | A | **20.6%** | 39% | 58% | 79% |
| | B | **24.5%** | 54% | 71% | 86% |
| PBO ≤ 0.20 | A | 10.0% | 13% | 19% | **42%** |
| | B | 11.7% | 10% | 11% | **11%** |
| naive t ≥ 2 on the selected configuration | A | 10.2% | 49% | 80% | 96% |
| | B | 5.5% | 44% | 77% | 96% |

### How the frozen rules decided

- **Rule 1 (MCPT null):** plain MCPT false pass was 4.4% and 4.5%, below the 6% bar in both families. The plain null
  is kept; stratification is known to absorb volatility-timing edges.
- **Rules 2-3 (gate set):** all five candidate sets were admissible (false pass ≤ 6% and false reject at Sharpe 0.79
  ≤ 30%, in both families). MCPT plain had the highest mean power at Sharpe 0.52 (68.8%). The next best was MCPT
  stratified at 67.8%, a tie within 3 points. The tie went to the test that does not depend on N_eff, and then to the
  higher power.

### What the other numbers show

- **The old DSR fails exactly where the review predicted.**
  - False pass by clustered N_eff on noise: 22% at N_eff = 2, 12% at N_eff = 1 (family B), 7% at 3, 2% at 4.
  - Luck makes configurations look alike, and the clustered count shrinks precisely when luck looks like an edge.
- **The correlation-aware benchmark fixes that.**
  - In family B it deflates by a median of 0.13 annual Sharpe; the old rule applied 0.
  - Its false pass is 3.5% or less in both families.
- **The walk-forward rule has no power to reject noise.** About a fifth of pure-noise searches pass it.
- **PBO measures rank stability, not edge.** When many neighbouring configurations share the same real edge, which one
  wins out of sample is itself noise, so PBO stays near 0.5.

## Part B — S3 inference unit (event panels)

### Frozen panel (200 histories per case)

- 200 stocks × 2,520 sessions: a GARCH market factor with heterogeneous betas and idiosyncratic Student-t noise.
- Oversold-style events that cluster after market down days, about 10,000 events per history.
- 5-session holds, measured as excess over the equal-weight panel.

| Test at nominal 5% | false positive | power (+0.2% per event) |
|---|---|---|
| naive t on pooled events | 3.5% | 99.5% |
| date-level Newey-West t (lag 4) | 4.0% | 93.5% |
| within-date permutation | 4.5% | 99.5% |

The frozen panel did not reproduce the problem. Subtracting the same-date panel return removes the common move, and
nothing else linked the events.

### Exploratory panel (designed after the frozen result; 100 histories per case)

The same panel plus the two dependence sources that real oversold signals have:
- **10 sector factors.**
- **A persistent signal:** an event repeats on the same stock the next day with probability 0.6.

| Test at nominal 5% | false positive | power | spread of t under the null (should be 1.00) |
|---|---|---|---|
| naive t on pooled events | **19%** | 100% | 1.50 |
| **date-level Newey-West t (lag 4)** | **5%** | 100% | 1.03 |
| within-date permutation | **11%** | 100% | — |

- **Naive t:** with real-world dependence it over-rejects almost fourfold.
- **Within-date permutation:** it over-rejects too. Moving an event to a random stock on the same date breaks both the
  persistence of the signal and its concentration in a sector.
- **Date-level Newey-West t:** the only test calibrated in both panels.

A persistence-preserving permutation (for example, circular shifts of each stock's event series) is to be calibrated
in P4. So are the real-panel re-runs of the Pakal DV2 and QPI studies, where all three tests can be compared on real
data.

## Part C — known-dead real cases (in-sample only)

Selection = highest in-sample Sharpe, as in the original studies. The DSR-corr benchmark uses the same maximum rule
and the grid's own correlation matrix.

| Case | variants | mean pairwise correlation | chosen in-sample Sharpe | old DSR (N_eff) | DSR-corr (benchmark Sharpe) | out-of-sample Sharpe of the choice |
|---|---|---|---|---|---|---|
| Z8 L2016, excess over 1/N, to 2016-06 | 16 | 0.97 | −0.04 | 0.45 fail (1) | 0.34 fail (0.10) | −0.74 |
| Z9 L2017 (hindsight list), excess over 1/N, to 2017-09 | 24 | 0.73 | 0.98 | **0.98 pass** (1) | **0.89 fail** (0.41) | 0.11 |
| Z9 N14 (neutral list), excess over 1/N, to 2017-09 | 24 | 0.87 | 0.16 | 0.70 fail (1) | 0.47 fail (0.18) | −0.16 |
| Alpha101 long-short, **gross**, 2000-2011 | 97 | 0.17 | 2.01 | 0.98 pass (51) | 1.00 pass (0.65) | 0.82 gross |

- **Z8 and Z9 on the neutral list** never beat equal weight in sample. Any test rejects them.
- **Z9 on its 2017 list** was passed by the old DSR because the 24 variants counted as one trial. The correlation-aware
  benchmark (0.41 annual Sharpe of pure selection luck) rejects it. Choosing the list after its sectors had won is a
  further, unrecorded search that no statistic on the grid can see; that is S0's job.
- **Alpha101's gross edge was real in 2000-2011 and stayed positive gross afterwards (0.82).** It died of costs: the
  median break-even fell to about 1 bp per side. S5 is right to pass it gross. S4's cost stress is the guard.
- MCPT could not be run here, because re-running these searches needs their full simulators.

## The first run, for the record

The first run chose "DSR alone" (old DSR 5.0% false pass, 55% / 87% power). Three reasons to withdraw it:
- **Sampling error:** with 200 seeds the result sat exactly on the admissibility cut, which also equalled DSR's own
  nominal level. With 2,200 seeds its false pass was 5.4%.
- **The flawed DSR:** see above.
- **Optimistic edge labels:** the edges were calibrated on one path and were really 0.27 / 0.45 / 0.74.

Its other findings held up in amendment 1: walk-forward too lenient, PBO blind to shared edges, MCPT at p ≤ 0.01 too
strict. A float comparison bug in its admissibility check marked one set inadmissible at exactly 30% false reject; it
did not change that run's decision.

## What changes in Scout (amendment A5)

1. **S3 significance:** the date-level Newey-West t, whose p-value feeds the ledger-level Benjamini–Yekutieli FDR.
   The within-date permutation is a diagnostic until P4 calibrates a persistence-preserving version.
2. **S5 gate:** MCPT, plain date shuffle, p ≤ 0.05, re-running the full search including plateau selection.
3. **S5 WARN:** DSR-corr ≥ 0.95, with the benchmark simulated from the family's correlation matrix and the ledger's
   earlier trials of the same family added as independent draws. A miss needs a written owner note.
4. **S5 printed diagnostics:** walk-forward efficiency and the 8-design strip, PBO, and the old DSR (for reference
   only).
5. **Every S5 miss is a soft fail (WATCHLIST, D22).**
6. **S0 must record how a universe or asset list was chosen and whether results had been seen** (Z9 L2017). This is
   implemented with the S0/S1 station work in P4.

## Limits

- **One kind of synthetic edge** (time-series momentum on one asset), in two grid shapes. Cross-sectional and
  volatility-regime families may behave differently. P4 and P5 re-check on the first real families.
- **No costs in the synthetic runs.** Costs are S4's job.
- **MCPT was not run on the real dead cases.**
- **200 permutations per MCPT** give a p-value resolution of 0.005: enough for the 0.05 cut.
