# How to define the DV2 stress gate — verdict (2026-10-03)

Research-only. Nothing live changed. Frozen plan: [`scripts/research/mr_gate_definition_20261003/SPEC_FROZEN.md`](../../scripts/research/mr_gate_definition_20261003/SPEC_FROZEN.md).
Follows [stress-gated MR](MR_STRESS_REGIME_20261002.md).

## Verdict

**Replace ANY_OFF with "VIX only, with memory":** the gate opens when VIX closes above 20 and then stays open for at
least 10 sessions (C3). Simpler (one input, no SMA200), fewer switches, and a little better everywhere.

| Book {TAA .5, L .25, X .25} | 2008–11 | 2012–21 | 2022–26 | 2012–26 | 2008–26 | +5 bps 2012–26 | +5 bps 2022–26 | Switches/yr |
|---|---|---|---|---|---|---|---|---|
| T-bills in the slot | 0.728 | 1.334 | 1.437 | 1.365 | 1.226 | 1.357 | 1.428 | |
| DV2 ungated | 0.809 | 1.449 | 1.391 | 1.430 | 1.280 | 1.339 | 1.299 | |
| ANY_OFF (current) | 0.812 | 1.558 | 1.447 | 1.522 | 1.353 | 1.480 | 1.400 | 13.5 |
| **VIX > 20, open ≥ 10 sessions (C3)** | **0.825** | **1.598** | **1.475** | **1.558** | **1.383** | **1.511** | **1.420** | **8.4** |
| VIX > 20, close < 18 (C2) | 0.802 | 1.573 | 1.462 | 1.537 | 1.362 | 1.494 | 1.411 | 6.5 |

Frozen rule: only C2 and C3 pass (C3 preferred: best, and the preference for the simplest does not apply within 0.01).
QPI cross-check, standalone 2004–26: ungated 1.049, C3 1.086, C2 1.062, ANY_OFF 1.109. DV2 1991–99 holdout: C3 1.89,
ANY_OFF 1.85, ungated 2.19. Against T-bills in 2022–26 at +5 bps it is still a tie (1.420 vs 1.428).

## What the families show

- **Threshold:** every VIX threshold 16–25 beats ungated DV2 (plateau in direction), but the level matters. Book
  2012–26: 16 → 1.443, 18 → 1.487, 20 → 1.528, 22 → 1.537, 25 → 1.482. 20–22 is the sweet spot; 16 is almost ungated.
- **SMA200 adds nothing in the book** (VIX 20: 1.528, with SMA200: 1.522). It lifts the standalone pod (1.05 vs 0.98),
  because SMA200 keeps the gate open into bear-market rallies.
- **Memory is what helps.** Stress clusters: one VIX close under 20 rarely ends a stress episode. Minimum-open grid
  (exploratory, after the result), book 2012–26:

| Open above | min 5 | min 10 | min 15 | min 20 |
|---|---|---|---|---|
| VIX 18 | 1.462 | 1.455 | 1.474 | 1.493 |
| VIX 20 | 1.520 | **1.558** | 1.569 | 1.566 |
| VIX 22 | 1.565 | 1.542 | 1.543 | 1.565 |

  C3 sits inside a plateau (VIX 20–22, 10–20 sessions: 1.54–1.57), not on a peak.
- **Graded sizing hurts** (1.47–1.49): it keeps capital in calm markets, which is what the gate removes.
- **VIX term structure (VIX/VIX3M > 1) alone:** best in 2022–26 (1.561) but poor in 2008–21 and open only 12% of days.
  With VIX > 20 it adds nothing. **Realized volatility > 15%** is worse than VIX (1.445).

## Caveats
- 18 frozen variants plus a 12-cell exploratory grid. C3's gain over ANY_OFF (+0.036) is modest; the robust gains are
  fewer switches and dropping the SMA200 input.
- Few independent stress episodes; the gain over ungated DV2 rests on 2000–02, 2008–09, 2020 and 2022.

Outputs: `results/research/mr_gate_definition_20261003/` (gitignored). Script: `run_gates.py`.
