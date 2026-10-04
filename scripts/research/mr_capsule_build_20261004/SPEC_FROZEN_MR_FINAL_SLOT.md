# MR slot, final decision: where does DV2 with the liquidity floor and ADV rank belong? (frozen 2026-10-04)

Written before any run. Owner request, 2026-10-04: "close the MR event". The question: does DV2-LF-ADV
(`strategies/dv2/strategy_mr_dv2_liquidity_floor_adv_rank.py`) improve the book's MR slot? It earns about the same
alpha in calm and in stress markets (`adv_rank_calm.py`), so it may fill the calm periods with real alpha where the
gated pods sit in cash.

## Candidates for the MR slot

Each slot is reset to its weights once a year, and idle cash is swept at the T-bill rate.

| Id | Slot |
|---|---|
| M0 | Status quo: 0.5 DV2-G + 0.5 HPI-G (the MR capsule) |
| M1 | 0.5 ADV-U + 0.5 HPI-G |
| M2 | Thirds: DV2-G + HPI-G + ADV-U |
| M3 | 0.5 ADV-G + 0.5 HPI-G |
| M4 | ADV-U alone |

Legs:
- DV2-G and ADV-G/U use the engine-parity replica with engine costs. ADV63 comes from native Turnover, and the floor is applied explicitly.
- HPI-G uses the real-engine research run (`components.parquet`).
- G means the MR capsule gate; U means ungated.

## Book and windows

- **Book:** TAA 0.5 + NDX 0.25 + MR slot 0.25, the research book. Full window 2008-03-04 → 2026-08-19.
- **Blocks:** 2008–11, 2012–21, 2022–26.
- **Cost check:** the MR legs at +5 bps per side (HPI-G from its stress run).

## Decision rule (fixed now)

A challenger replaces M0 only if **all** of these hold:
1. Book Sharpe 2008–26 above M0, at engine costs **and** at +5 bps.
2. Better than M0 in at least 2 of the 3 blocks (engine costs).
3. Paired block bootstrap (20 days, 2,000 draws): P(challenger Sharpe > M0) ≥ 0.80.
4. Book max drawdown 2008–26 no worse than M0 by more than 1.0 pp.

If several pass, take the highest engine-cost book Sharpe. Within 0.01, take the one with fewer pods. If none passes,
**M0 stays and the MR slot is closed as is.** SPMO parking is out of scope: it is a separate money dial.

## Caveats on record

- All legs were selected in sample: the gate on DV2, and the ADV rank on the DV2 deep-research grid. The comparison is still like for like.
- Replica, not engine. The engine confirms only the winner.
- Research conventions: T-bill sweep and no withholding, the same for every candidate.
