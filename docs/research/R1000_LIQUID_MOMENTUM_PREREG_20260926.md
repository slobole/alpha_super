# PREREG (frozen) - Russell 1000 scale-free momentum leg with a liquidity filter

Frozen 2026-09-26, before any liquidity-filtered run was computed. Freeze evidence: SHA-256 and timestamp in
`results/research/ndx_param_robustness_20260926/r1000_liquid_prereg_freeze.json`. Research only; nothing live changes.
Follows the NDX parameter-robustness study (`NDX_PARAM_ROBUSTNESS_REPORT_20260926.md`) and uses its code and data.

## 0. What is already known (disclosed; this weakens the evidence)

- The exploratory walk-forward run (`explore_ndx_param_robustness_other_universes.py`) picked, on 2000-2011 standalone
  Sharpe only, the Russell 1000 configuration **W** below. Its UNFILTERED results for 2012-2026 were then seen: G3
  Sharpe 0.92 / 1.26 / 1.35 by block (L: 0.70 / 1.30 / 1.29), G3 2012-26 1.29 at 15.7%/yr, max DD -11.8%.
- The same run showed a capacity problem: the largest order reached 1% of 20-day dollar volume at a pod size of about
  $0.3M (2021-26). That is why this check exists.
- The universe itself was chosen after seeing results. Only a forward shadow line is clean evidence.

## 1. Configurations (all frozen now; no tuning afterwards)

**W** (the walk-forward pick, unchanged): PIT Russell 1000; score (ME(1)/ME(12) - 1) / NATR20 (12-1 momentum over ATR
as a fraction of price); N = 30; weights proportional to 1/sigma63 with the equal-weight budget; stock filter
Close > SMA200; regime SPY > SMA200 else cash; rank buffer 2 (a held name stays while ranked within 32); month-end
decision, next-open fills; exposure x clip(18 / VXN, 0.125, 1). Engine semantics and costs as in the NDX study.

**A0-R** (reference): the NDX live design with the scale-free score (ROC12/NATR20, N 10 EW, SMA100, SPY, VXN 22/0.25)
on Russell 1000.

## 2. Liquidity filter (added to eligibility; everything else unchanged)

ADV20_i(t) = median of Close_CS x Volume_CS over the 20 sessions ending at the decision close (20 valid sessions).
Close_CS x Volume_CS equals the actual traded dollars (checked: NVDA, AAPL, SIRI), so the filter is download-date
invariant. A name without ADV20 is ineligible under any filter.

- **Primary, REL25:** exclude PIT members whose ADV20 is below the 25th percentile of ADV20 among that day's PIT
  members.
- Sensitivity REL50: below the 50th percentile excluded.
- Sensitivity ABS5M: ADV20 below $5,000,000 excluded (the MOSAIC convention; nominal dollars).

Runs: W and A0-R, each unfiltered, REL25, REL50 and ABS5M; engine costs and +5 bps per side.

## 3. Measurements

As in the NDX study: standalone blocks P1 2000-11, P2 2012-21, P3 2022-26, FULL; G3 = 0.5 TAA + 0.5 leg daily, blocks
G-P1 2008-11 (TAA proxy), G-P2 2012-10..2021, G-P3 2022-26, G-FULL 2012-10..2026-07-24, G-LONG 2008-26. Turnover.
Capacity proxy (pod AUM at which the 95th-percentile and the largest order reach 1% and 5% of ADV20; full history
and 2021-26). Share of the eligible pool removed by the filter, and overlap with the unfiltered list.
Descriptive only: a three-leg book 0.5 TAA + 0.25 NDX L + 0.25 W-REL25 (daily rebalanced), and the correlation of
W-REL25 with NDX L.

## 4. Decision rule (frozen)

**W-REL25 replaces the NDX leg in G3** only if all hold:
- R1: G3 Sharpe strictly above L's (NDX live) in each of G-P1, G-P2 and G-P3;
- R2: G3 max DD not worse than L's by more than 2.0 pp on G-FULL and on G-LONG;
- R3: R1 and R2 also hold with +5 bps per side for both;
- R5 (tradability): in 2021-26 the largest order stays below 5% of ADV20 up to a pod size of at least $5M.
If it fails, keep L. If it passes R1-R3 and R5, recommend a forward shadow line first (not a switch), because of
section 0; the report will say so. REL50 and ABS5M are sensitivities: if the verdict differs between them and REL25,
the report flags the result as filter-dependent.
Confidence label: paired stationary bootstrap (2,000 draws, mean block 21, seed 20260926) of G-FULL Sharpe W-REL25
minus L, one-sided p.
