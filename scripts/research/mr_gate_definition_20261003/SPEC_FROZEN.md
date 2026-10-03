# How to define the DV2 stress gate (frozen 2026-10-03, before any run of these variants)

Follows MR_STRESS_REGIME_20261002 (ANY_OFF = new entries only if VIX > 20 or S&P 500 < SMA200).
Known before freezing: ANY 1.522 / V20 1.528 / MKT 1.421 book G-FULL; ANY switches 13.5 times a year, 54% of its
spells last <= 3 days; open days: VIX-only 34%, SMA200-only 10%, both 57%.

## Question
Which gate definition is robust: VIX only or VIX-or-trend; which threshold; does hysteresis or graded sizing help;
do other stress measures do better? Goal: the simplest definition that is at least as good as ANY_OFF and sits on a
plateau, not a peak.

## Fixed setup (as the stress study)
DV2 wired replica (generic runner), next-open entries, exits never gated, idle cash at BIL (DTB3 before 2007-06),
engine costs and +5 bps stress. Book {TAA 0.5, NDX-L 0.25, X 0.25}, official pod model, blocks G-P1/G-P2/G-P3,
G-FULL, G-LONG. Inputs known at close t decide entries filled at open t+1.

## Variants (all reported)
- A. VIX only, OFF, threshold k in {16, 18, 20, 22, 25}.
- B. ANY (VIX > k or SMA200), k in {16, 18, 20, 22, 25}.
- C. Hysteresis, VIX only: open when VIX > 20 and close when VIX < 17 (C1) or < 18 (C2); C3 = open above 20,
  stay open at least 10 sessions.
- D. Graded, VIX only: slot budget = clip((VIX - a) / (b - a), 0, 1), no entry at 0; (a, b) in {(14, 25), (16, 22),
  (12, 30)}.
- E. Term structure: E1 = VIX / VIX3M > 1; E2 = E1 or VIX > 20 (from 2002; VIX3M starts 2002-01-02).
- F. Realized: S&P 500 20-day realized volatility (annualized) > 15%.
Reference: ANY_OFF (= B at k = 20). Controls: ungated DV2, T-bills (C_BIL).

## Decision rule
- Plateau: family A (and B) is "robust" if every k in 16–25 beats ungated DV2 in G-FULL and G-LONG.
- A variant replaces ANY_OFF only if ALL hold:
  1. book Sharpe >= ANY_OFF in G-FULL and G-LONG, and >= ANY_OFF - 0.03 in each of G-P1, G-P2, G-P3;
  2. the same with +5 bps per side;
  3. its parameter neighbours (A/B: adjacent k; C: C1 vs C2; D: the other (a, b)) are >= ANY_OFF - 0.03 in G-FULL.
- Among passing variants prefer the simplest (fewest inputs, no hysteresis/grading) if within 0.01 of the best
  G-FULL Sharpe. If none passes, ANY_OFF stays.
- Reported, not gating: standalone Sharpe / CAGR / DD 2000–26 (2002– for E), 1991–99 holdout (VIX-based only),
  switches per year, share of days open, QPI cross-check for the finalists.
