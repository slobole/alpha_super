# Final checks on the DV2-G gate (frozen 2026-10-03, before any run)

Reference gate (DV2-G): VIX > expanding mean of VIX since 1990 (min 500 sessions) opens the gate; it stays open
>= 15 sessions from the opening; closes on the first close at or below the threshold after that.

## Check A — long rolling mean instead of the expanding mean (owner: "be more dynamic")
Threshold = rolling mean of VIX over W years (W x 252 sessions; min 500 sessions, so it equals the expanding mean
until W years exist): W in {10, 15, 20}. Memory 15. Same DV2 replica setup, book, blocks, +5 bps, 1995–99 holdout
as the gate studies. Note: 15y exists from 2005, 20y from 2010; before that the threshold is the expanding mean.
Rule: W replaces the expanding mean only if book Sharpe >= reference in G-FULL and G-LONG, >= reference - 0.03
in each block, the same at +5 bps, and its neighbour W values are >= reference - 0.03 in G-FULL. Otherwise the
expanding mean stays (it is the simpler one).

## Check B — the same gate on HPI 2/3/5 vote (independent strategy, real engine)
Arms: HPI vote as is (Turnover rank) vs HPI vote with the DV2-G gate (no new entries when the gate is closed on the
decision date; exits unchanged). Real engine 2004-01-01 -> latest, capital 100,000, engine costs. Idle cash swept at
BIL (DTB3 before 2007-06) for both, from the engine's cash column. Book {TAA .5, L .25, X .25} with X = HPI.
Pass: gated book Sharpe >= ungated HPI in G-FULL and G-LONG and >= ungated - 0.03 in each block, and gated
standalone Sharpe 2004–26 >= ungated - 0.05. Reported: CAGR, max DD, trades, exposure, crises.

## Results (2026-10-03)

### A — rolling mean: fails. The more dynamic, the worse.
Book 2012–26:
| Threshold | Sharpe |
|---|---|
| Expanding | 1.561 |
| 20y | 1.547 |
| 15y | 1.522 |
| 10y | 1.474 |

2022–26:
| Threshold | Sharpe |
|---|---|
| Expanding | 1.467 |
| 20y | 1.463 |
| 15y | 1.376 |
| 10y | 1.263 |

A shorter window lowered the threshold after 2018 (10y: 17.5–18.6), so the gate opened in calmer markets.

### B — HPI vote, real engine: fails formally, by 0.004 in 2008–11. Same pattern as DV2.
| HPI vote | Ungated | Gated |
|---|---|---|
| Book 2008–11 | 0.843 | 0.809 (needs >= 0.813) |
| Book 2012–21 | 1.418 | 1.498 |
| Book 2022–26 | 1.472 | 1.508 |
| Book 2012–26 | 1.435 | 1.501 |
| Book 2008–26 | 1.306 | 1.343 |
| Standalone Sharpe 2004–26 | 1.077 | 1.047 |
| Standalone CAGR | 16.9% | 13.6% |
| Max DD | −17.6% | −16.5% |
| Standalone 2004–14 / 2015–26 | 1.01 / 1.14 | 0.80 / 1.26 |
| Exposure | 70% | 32% |
| Entries per year | 274 | 132 |

Crises:
| Crisis | Gated | Ungated |
|---|---|---|
| 2015–16 | +7.7% | −3.5% |
| Volmageddon | −0.8% | −9.8% |
| Q4 2018 | −2.5% | −13.7% |
| 2025 | −7.9% | −12.7% |
| GFC | +7.9% | +11.2% |

T-bills slot: 0.728 / 1.334 / 1.437.

## Check C (added 2026-10-03, before any run): HPI vote at +5 bps per side
Same two engine arms with slippage 0.00075 (2.5 bps engine + 5 bps stress), the book with the stored stress NDX-L.
Reported against each other and against the T-bill slot at stress costs (C_BIL_stress_L: 0.724 / 1.326 / 1.428).
Reading: gated is preferred in the book if its stressed book Sharpe >= ungated in G-FULL and G-LONG and >= ungated
- 0.03 in each block.

### C — result: gated preferred at +5 bps
| HPI vote | Ungated | Gated |
|---|---|---|
| Standalone Sharpe | 0.900 | 0.945 |
| Standalone CAGR | 13.7% | 12.1% |
| Max DD | −18.4% | −17.0% |
| Book 2008–11 | 0.798 | 0.772 (within −0.03) |
| Book 2012–21 | 1.361 | 1.466 |
| Book 2022–26 | 1.407 | 1.464 |
| Book 2012–26 | 1.375 | 1.465 |
| Book 2008–26 | 1.249 | 1.306 |

T-bills slot at stress: 0.724 / 1.326 / 1.428.
- Ungated HPI loses to T-bills in 2022–26 at stress (1.407).
- Gated HPI beats T-bills in all three blocks at stress. This is the first MR slot in these studies to do so.
