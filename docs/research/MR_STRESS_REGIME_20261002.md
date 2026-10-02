# Stress-gated mean reversion — verdict (2026-10-02)

Research-only. Nothing live changed. Frozen plan: [`scripts/research/mr_stress_regime_20261002/SPEC_FROZEN.md`](../../scripts/research/mr_stress_regime_20261002/SPEC_FROZEN.md).
Question: should DV2 (and QPI) take new entries only, or mostly, in stressed markets?

## Verdict

**No variant passes the frozen rule. All eight fail R3. Among the variants that pass R1, R2 and R4, the margin is
real against ungated DV2 but not against T-bills in 2022–26.**

- **Against ungated DV2:** the VIX gate wins in every block, at engine and at stress costs.
  - ANY_OFF = new entries only when VIX > 20 or the S&P 500 < SMA200.
  - Book G-FULL Sharpe: 1.52 vs 1.43 ungated; stressed 1.48 vs 1.34.
- **Against T-bills in the slot:** a tie in 2022–26. At engine costs, ANY_OFF is 1.447 vs C_BIL 1.437. With +5 bps it
  is 1.400 vs 1.428, so R3 fails. V20_HALF and ANY_HALF fail the same way.
- **Practical reading:** if the book holds a DV2 / MR slot, the gated version is better than the ungated one and
  should replace it in any forward shadow. Whether to hold the slot at all instead of T-bills is still a tie
  since 2022.

## Book: {TAA 0.5, NDX-L 0.25, X 0.25}, official pod model, idle cash at BIL

| X in the slot | 2008–11 | 2012–21 | 2022–26 | 2012–26 | 2008–26 | DD 2012–26 | +5 bps 2022–26 |
|---|---|---|---|---|---|---|---|
| T-bills (C_BIL) | 0.728 | 1.334 | 1.437 | 1.365 | 1.226 | −11.7% | 1.428 |
| DV2 ungated (C_DV2) | 0.809 | 1.449 | 1.391 | 1.430 | 1.280 | −13.0% | 1.299 |
| **DV2 ANY_OFF** | **0.812** | **1.558** | **1.447** | **1.522** | **1.353** | −12.3% | 1.400 |
| DV2 V20_OFF | 0.786 | 1.558 | 1.465 | 1.528 | 1.351 | −12.3% | 1.418 |
| DV2 ANY_HALF | 0.840 | 1.488 | 1.457 | 1.478 | 1.324 | −12.5% | 1.386 |
| DV2 V20_HALF | 0.843 | 1.474 | 1.466 | 1.471 | 1.320 | −12.5% | 1.396 |
| DV2 MKT_OFF | 0.875 | 1.417 | 1.430 | 1.421 | 1.298 | −11.7% | 1.395 |

Rule outcome: R1 (beat both controls in each block) passes for V20_HALF, ANY_OFF and ANY_HALF. R2 (DD) passes
for all but VREL_OFF. **R3 (stress) fails for all eight.** R4 (QPI cross-check) passes for 4 of 8.
V20_OFF fails R1 only in 2008–11 (0.786 vs 0.809).

## Standalone (idle cash at T-bills), 2000–2026

| | DV2 CAGR | DV2 Sharpe | DV2 Max DD | DV2 1991–99 | QPI Sharpe 2004–26 | QPI 1995–2003 |
|---|---|---|---|---|---|---|
| Ungated | 22.6% | 1.04 | −30.9% | 2.19 | 1.05 | 1.63 |
| ANY_OFF | 19.5% | 1.05 | −28.8% | 1.85 | 1.11 | 1.53 |
| ANY_HALF | 19.6% | 1.07 | −28.5% | 2.16 | 1.09 | 1.56 |
| V20_HALF | 18.9% | 1.05 | −28.6% | 2.14 | 1.06 | 1.56 |
| MKT_OFF | 12.2% | 0.82 | −26.4% | 1.12 | 0.82 | 1.18 |

Gates are open 38% (V20), 42% (ANY), 28% (MKT) of days. On its own the gate barely changes Sharpe. Its value is in the
book: calm-market DV2 is mostly S&P 500 beta, and the gate swaps it for T-bills next to two equity-heavy pods.
The 1990s holdout loses with OFF gates (edge was large in calm markets then); HALF keeps it.

## Mechanism (per trade, market-adjusted, ungated DV2, by VIX > 20 at entry)

| | 1990s | 2000–09 | 2010–19 | 2020–26 |
|---|---|---|---|---|
| Calm (VIX ≤ 20) | 65 bps | 38 bps | 10 bps | 6 bps |
| Stress (VIX > 20) | 75 bps | 76 bps | 17 bps | 55 bps |

The stress premium is large in 2000–09 and 2020–26 and small in the 1990s and 2010–19. So it rests on a few crisis
episodes (2000–02, 2008–09, 2020, 2022). QPI shows the same pattern.

## Caveats
- Few independent stress episodes; the gate thresholds were fixed (VIX 20, SMA200), not tuned.
- Eight variants per strategy; ANY and V20 overlap heavily (open 42% vs 38% of days).
- TAA and NDX-L sleeves end 2026-08-19. Engine costs plus a +5 bps stress; small-account minimum commissions are
  not modelled.

Outputs: `results/research/mr_stress_regime_20261002/` (gitignored). Script: `scripts/research/mr_stress_regime_20261002/run_stress.py`.
