# MR capsule: DV2 + HPI with the stress gate (frozen 2026-10-03, before any capsule result)

Owner (2026-10-03): the MR capsule is DV2 + HPI (main). Goal: verify it is robust and lifts Sharpe.
- Optional A: the industry-ETF DV2 pod (lower CAGR, not growth).
- Optional B: the limit-entry execution (Scout k 0.5), because it needs a live order-type change.

## Known before freezing (disclosed)
- **Gate (DV2-G):** VIX > expanding mean of VIX since 1990 (min 500 sessions) opens it; it stays open >= 15 sessions
  from the opening.
- **Gate tested on:** DV2 (replica, ~60 variants), QPI (cross-check) and HPI vote (real engine, engine costs and +5 bps).
- **Single-pod book results (G-FULL):**

  | Pod | Engine costs | +5 bps |
  |---|---|---|
  | DV2-G | 1.561 | 1.508 |
  | HPI-G | 1.501 | 1.465 |
  | HPI ungated | 1.435 | 1.375 |

- **Correlation:** stock MR pods correlate 0.74–0.87 daily (2026-09-26).
- **Scout limit study (2004–2022, S&P 500 DV2):** limit k 0.5 + limit exit raises net Sharpe under spread cost
  models (AR 0.26 → ~0.70, pooled 0.77 → ~0.82); k 0.5 is a lucky point.

## Components (daily returns; idle cash parked; window 2004-01-05 → 2026-08-19 for books, → latest standalone)
- **DV2-G / DV2-U:** generic replica of wired DV2 (engine parity 1.0687 vs 1.0693), gated / ungated.
- **HPI-G / HPI-U:** the real-engine runs of strategy_mr_hpi_sp500_2_3_5_vote (check B / C), gated / ungated.
- **Costs:** engine costs (2.5 bps per side + $0.005/share, $1 min) and +5 bps per side (DV2 replica slippage;
  HPI engine slippage 0.00075).

## Capsule construction
- **Pod model:** each component keeps its own sub-account; weights reset to target each year-end (the official
  book code).
- **Main weights:** 50/50 capital. Reported for the plateau: DV2 share in {0, 0.25, 0.5, 0.75, 1}, and inverse
  volatility from the trailing 252 sessions at each year-end (no look-ahead).
- **Parking** (one shared gate, so both pods are idle together):
  - T-bills (main);
  - CORE5 pod (from 2008-03);
  - SPMO with an 8% volatility target (from 2015-11);
  - T-bills levered to the ungated capsule's volatility (financing T-bill + 1.5%). The scale is set in sample:
    reported only.

## Measurements
- **Standalone:** CAGR, Sharpe, Sortino, max DD, Calmar, worst year, crises 2007–2025; engine and +5 bps.
- **Diversification:** daily correlation DV2 vs HPI (gated and ungated); position overlap (names held by both pods on
  the same day); correlation of their drawdowns.
- **Book** {TAA .5, NDX-L .25, capsule .25} against:
  - the T-bills slot;
  - DV2-G alone;
  - HPI-G alone;
  - the ungated capsule.

  Blocks G-P1 / G-P2 / G-P3, G-FULL, G-LONG, at engine and +5 bps.
- **Paired block bootstrap** (20-day blocks, 2,000 draws) on the G-FULL book: capsule vs each single component.
- **Pre-2004 proxy (reported only):** 1995–2003 with DV2 replica + QPI replica as the HPI stand-in, both gated.

## Decision rule (main)
The capsule (DV2-G + HPI-G, 50/50, T-bills parking) is the MR slot if ALL hold:
1. Book Sharpe > T-bills slot in each of G-P1, G-P2 and G-P3, at engine costs and at +5 bps.
2. Book Sharpe >= max(DV2-G alone, HPI-G alone) - 0.02 in G-FULL and G-LONG, at both cost levels. This is a
   non-inferiority test: the capsule is for robustness and capacity, not to beat its best part.
3. Book Sharpe > the ungated capsule in G-FULL and G-LONG, at both cost levels.
4. Every DV2 share in {0.25, 0.5, 0.75} gives a G-FULL book Sharpe within 0.03 of 50/50.
5. Book max DD (G-LONG) is not worse than the T-bills slot's by more than 2 pp.

**Parking:** T-bills stays the default unless an alternative meets all of these, at both cost levels:
- beats it by >= 0.02 in G-FULL and G-LONG;
- is within -0.03 in each block.

SPMO is judged on 2015-11+ only.

## Optional A: industry-ETF DV2
- **Pod:** 19 ETFs, ADV > $50M, wired rules, replica; validated against the stored engine sleeve etf_ind_fix
  (daily correlation and Sharpe 2008+).
- **Variants:** gated and ungated.
- **Capsule3:** equal thirds.
- **Adoption:** adopted over the main capsule only if its book Sharpe >= main capsule in G-FULL and G-LONG at both
  costs, and >= -0.03 in each block. CAGR is reported.

## Optional B: limit entry (report only)
- **Code:** Scout limit book (alpha.scout universes, sealed window 2004–2022: the Scout vault 2023+ stays untouched).
- **Pod:** S&P 500 DV2 only; there is no HPI version of this code.
- **Grid:** entry {market-on-open, limit k 0.5 with limit exit} × gate {off, on} × cost {gross, AR, pooled}.
- **Questions:**
  - Does the gate's gain survive under spread cost models?
  - Does the limit add on top of the gate?
- **No adoption here:** any live change needs paper fills first (Scout recommendation, owner decision).
