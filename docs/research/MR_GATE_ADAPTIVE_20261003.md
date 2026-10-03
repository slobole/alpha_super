# Adaptive stress measures for the DV2 gate — verdict (2026-10-03)

Research-only. Frozen plan: [`scripts/research/mr_gate_adaptive_20261003/SPEC_FROZEN.md`](../../scripts/research/mr_gate_adaptive_20261003/SPEC_FROZEN.md)
(amendment 1 before any run: holdout 1995–99 for all; C3 memory semantics). Ideas from CSS Analytics (D. Varadi):
VIX percentile from the Ehlers-alpha post (2019), adaptive volatility (2017), grid-robustness testing (2017).

## Verdict
**No adaptive measure beats C3 (VIX > 20, gate open at least 10 sessions). All 19 variants fail the rule; C3 stays.**

| Gate | Book 2012–26 | 2008–26 | 2008–11 | 2012–21 | 2022–26 | +5 bps 2012–26 | DV2 1995–99 | Switches/yr |
|---|---|---|---|---|---|---|---|---|
| **C3: VIX > 20, ≥ 10 sessions** | **1.558** | **1.383** | 0.825 | **1.598** | 1.475 | **1.511** | 2.16 | 8.4 |
| ANY_OFF | 1.522 | 1.353 | 0.812 | 1.558 | 1.447 | 1.480 | 2.06 | 13.5 |
| VIX percentile (8 yr), D < 0.3 | 1.491 | 1.345 | 0.884 | 1.458 | 1.569 | 1.454 | 1.96 | 3.5 |
| VIX percentile, D < 0.4 | 1.434 | 1.281 | 0.799 | 1.453 | 1.397 | 1.391 | 1.95 | 3.4 |
| VIX percentile, D < 0.5 | 1.473 | 1.309 | 0.803 | 1.510 | 1.396 | 1.422 | 1.91 | 3.6 |
| Adaptive vol > 14 / 16 / 18 / 20% | 1.46 / 1.48 / 1.46 / 1.43 | 1.29–1.33 | 0.84–0.87 | 1.41–1.49 | 1.44–1.47 | 1.41–1.45 | 1.66 / 1.76 / 1.10 / 0.46 | 2.6–3.6 |
| Realized vol 20d > 14 / 16 / 18 / 20%, memory | 1.49 / 1.52 / 1.46 / 1.45 | 1.29–1.34 | 0.78–0.82 | 1.45–1.55 | 1.38–1.50 | 1.42–1.48 | 1.88 / 1.98 / 1.95 / 1.41 | 4.3–5.4 |
| Vote of 2 of 3 (VIX 20, PR 0.4, AV 16%) | 1.477 | 1.309 | 0.786 | 1.494 | 1.444 | 1.436 | 2.09 | 4.5 |
| Ungated DV2 | 1.430 | 1.280 | 0.809 | 1.449 | 1.391 | 1.339 | 2.30 | |
| T-bills in the slot | 1.365 | 1.226 | 0.728 | 1.334 | 1.437 | 1.357 | | |

Memory variants of PR and AV change little (within ±0.03).

## Findings
- **Absolute implied volatility is the right measure.** VIX is forward-looking and prices dealers' risk appetite,
  which is the liquidity premium MR earns. Realized measures look backward. Percentiles normalize away the absolute
  risk level that matters: after 2008 the 8-year window treats VIX 20 as "calm".
- **VIX percentile is not on a plateau.** D < 0.3 / 0.4 / 0.5 gives 1.49 / 1.43 / 1.47 (non-monotone). D < 0.3 is
  strong in 2008–11 and 2022–26 (1.57, beats T-bills even at +5 bps) but weak in 2012–21.
- **Adaptive volatility does not beat plain realized volatility here** (better at 18% only). The post's claim was for
  volatility targeting, a different use; as a stress gate it adds nothing. Its 1995–99 holdout collapses at high
  thresholds (gate rarely open in the 1990s).
- **Combining measures does not help** (vote 1.477).

## Caveats
Results reuse 2000–2026 data seen in earlier gate studies; the holdout is 1995–99 only. 19 variants. C3's own edge
over ANY_OFF stays modest (+0.036); its robust advantages are simplicity and fewer switches.

Outputs: `results/research/mr_gate_adaptive_20261003/` (gitignored). Script: `run_adaptive.py`.
