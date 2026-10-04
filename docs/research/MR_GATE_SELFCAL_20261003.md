# Self-calibrating DV2 stress gate — verdict (2026-10-03)

Research-only. Frozen plan: [`scripts/research/mr_gate_selfcal_20261003/SPEC_FROZEN.md`](../../scripts/research/mr_gate_selfcal_20261003/SPEC_FROZEN.md).
Question: can the gate learn its own threshold and memory from history instead of hand-picked 20 / 10?

## Verdict
No self-calibrating candidate passes the non-inferiority rule; C3 (VIX > 20, >= 10 sessions) stays. But the study
answers where the numbers come from:
- **The threshold is learnable and stable.** The expanding mean of VIX (all history up to each day) has stayed
  between 19.0 and 20.6 since 2000 (19.4 today). "20" is the self-learned value, not a guess. Exploratory (after the
  result): learned threshold + fixed 10-session memory gives 1.555 vs C3 1.558 in the book.
- **The memory is not learnable without a hidden choice.** A VIX half-life depends on what "normal" it reverts to:
  18 sessions versus a 1-year mean, 31 versus 3 years, 35 versus the expanding mean. S1's expanding estimate sat at
  the 40-session cap from 1998, kept the gate open 61% of days and failed (G-FULL 1.539, 2008–11 0.771, 1995–99 1.95).
- **Learning from strategy returns is the worst option.** The walk-forward pick (S3) chose 18 / 5 for 20 straight
  years and scored lowest (1.460, 2022–26 1.356, 1995–99 1.75). This is the overfitting the owner wants to avoid.

| Gate | Book 2012–26 | 2008–26 | 2008–11 | 2022–26 | +5 bps 2012–26 | DV2 1995–99 | Open | Switches/yr |
|---|---|---|---|---|---|---|---|---|
| C3: VIX > 20, 10 sessions | 1.558 | 1.383 | 0.825 | 1.475 | 1.511 | 2.16 | 46% | 8.4 |
| VIX > 20, 15 sessions | 1.569 | 1.376 | 0.786 | 1.470 | 1.517 | 2.10 | 50% | 7.5 |
| S1: learned mean + learned half-life | 1.539 | 1.354 | 0.771 | 1.490 | 1.474 | 1.95 | 61% | 4.6 |
| S2: learned median + learned half-life | 1.488 | 1.323 | 0.797 | 1.414 | 1.419 | 2.06 | 69% | 4.6 |
| S3: walk-forward on DV2 returns | 1.460 | 1.313 | 0.840 | 1.356 | 1.408 | 1.75 | 54% | 9.3 |
| Learned mean + 10 sessions (exploratory) | 1.555 | 1.375 | 0.815 | 1.466 | 1.506 | 1.98 | 47% | 7.8 |
| Learned mean + 15 sessions (exploratory) | 1.561 | 1.371 | 0.781 | 1.467 | 1.508 | 2.03 | 51% | 7.0 |

## Recommended definition
Threshold = expanding mean of VIX (self-calibrating; ~19.4 now, equivalent to 20). Memory = a fixed 10–15 sessions,
justified by the typical stress episode (median 11 sessions to a 20-day calm) and the 10–20 plateau, not fitted.
The 1995–99 holdout is lower with the learned threshold (1.98 vs 2.16) because the early expanding mean was 16–17.

Outputs: `results/research/mr_gate_selfcal_20261003/` (gitignored).
