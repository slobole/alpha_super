# MR capsule: DV2 + HPI with the stress gate — verdict (2026-10-03)

Research-only. Nothing live changed. Frozen plan: [`scripts/research/mr_capsule_20261003/SPEC_FROZEN.md`](../../scripts/research/mr_capsule_20261003/SPEC_FROZEN.md).
Gate: VIX above its expanding mean (since 1990) opens it; it stays open at least 15 sessions
([gate studies](MR_GATE_SELFCAL_20261003.md)). Components: DV2 (engine-parity replica) and HPI 2/3/5 vote (real engine),
2004–2026, idle cash at T-bills unless stated.

## Verdict

**Use the capsule (DV2-G + HPI-G, 50/50, idle cash in T-bills) as the MR slot.** It misses the frozen rule by 0.001
(R2), so this is a judgment, stated plainly.
- It passes 4 of 5 rules:
  - beats the T-bill slot in every block at engine and at +5 bps costs (R1);
  - beats the ungated capsule (R3; bootstrap P 0.97 / 0.99);
  - sits on a weight plateau (R4);
  - its drawdown is within limits (R5).
- **R2 (non-inferior to the best single pod) misses by 0.001.** Book 2012–26 Sharpe is 1.540 against DV2-G alone
  1.561 (tolerance 0.02). The bootstrap calls it a tie: P(capsule better) is 0.24 / 0.32 against DV2-G and
  0.90 / 0.84 against HPI-G.
- **Why the capsule anyway: robustness.**
  - DV2-G alone fails the T-bill test at +5 bps in 2022–26 (1.405 vs 1.428). The capsule passes it (1.443).
  - The capsule's worst block is better (2008–11: 0.799 vs 0.781; at +5 bps 0.753 vs 0.727).
  - Its standalone drawdown is much smaller (−21% vs −29% for DV2-G).
  - It has two implementations of the idea instead of one, and twice the capacity.

## Book: TAA 0.5 + NDX 0.25 + MR slot 0.25 (Sharpe)

| MR slot | 2008–11 | 2012–21 | 2022–26 | 2012–26 | 2008–26 | Max DD 08–26 | CAGR 08–26 |
|---|---|---|---|---|---|---|---|
| T-bills | 0.728 | 1.334 | 1.437 | 1.365 | 1.226 | −14.3% | 14.6% |
| DV2-G alone | 0.781 | **1.607** | 1.467 | **1.561** | 1.371 | −14.8% | 19.4% |
| HPI-G alone | 0.809 | 1.498 | **1.508** | 1.501 | 1.343 | −15.6% | 18.2% |
| Capsule ungated | **0.832** | 1.443 | 1.443 | 1.442 | 1.302 | −15.1% | 19.0% |
| **Capsule (DV2-G + HPI-G)** | 0.799 | 1.561 | 1.498 | 1.540 | 1.365 | −15.1% | 18.8% |
| Capsule, idle cash in CORE5 | 0.841 | 1.551 | 1.512 | 1.539 | **1.376** | **−14.0%** | **19.5%** |
| Capsule, levered T-bills (1.20×) | 0.793 | 1.589 | 1.489 | 1.557 | 1.373 | −15.6% | 19.6% |
| Capsule + industry ETF (thirds) | 0.779 | 1.519 | 1.495 | 1.511 | 1.342 | −15.0% | 17.7% |

At +5 bps per side:

| MR slot | 2008–11 | 2012–21 | 2022–26 | 2012–26 | 2008–26 |
|---|---|---|---|---|---|
| T-bills | 0.724 | 1.326 | 1.428 | 1.357 | 1.218 |
| DV2-G alone | 0.727 | 1.557 | 1.405 | 1.508 | 1.317 |
| HPI-G alone | 0.772 | 1.466 | 1.464 | 1.465 | 1.306 |
| Capsule ungated | 0.776 | 1.368 | 1.363 | 1.366 | 1.230 |
| **Capsule** | 0.753 | 1.519 | 1.443 | 1.495 | 1.319 |

Since 2015-11:

| Slot | Book Sharpe, engine | Book Sharpe, +5 bps |
|---|---|---|
| Capsule | 1.593 | 1.541 |
| Capsule, idle cash in SPMO | 1.609 | 1.559 |
| Capsule, idle cash in CORE5 | 1.606 | 1.555 |
| T-bills | 1.438 | 1.429 |

## Standalone, 2004–2026 (engine costs)

| | CAGR | Sharpe | Max DD | Worst year |
|---|---|---|---|---|
| Capsule (DV2-G + HPI-G) | 15.6% | **1.10** | −21.3% | +0.7% |
| Capsule ungated | **18.5%** | 1.09 | −23.1% | +3.8% |
| DV2-G alone | 17.4% | 1.02 | −28.8% | −0.1% |
| HPI-G alone | 13.6% | 1.05 | −16.5% | +0.7% |
| Capsule + industry ETF | 12.1% | **1.16** | **−14.8%** | +0.7% |
| Capsule, levered T-bills | 17.9% | 1.06 | −25.2% | +0.4% |
| Capsule since 2015-11 | 20.3% | 1.31 | −21.3% | |
| Capsule, idle cash in SPMO, since 2015-11 | 25.9% | 1.45 | −24.8% | |

## Construction questions

- **Weights:** a plateau. DV2 share 25% / 50% / 75% gives a book Sharpe of 1.523 / 1.540 / 1.553; inverse
  volatility gives 1.541. Equal weight stays.
- **Diversification is limited.**
  - Daily correlation 0.77 gated (0.74 ungated); drawdown correlation 0.81.
  - The pods hold 0.5 names in common on average (about 14% of positions), on 61% of the days both are invested.
  - The capsule is one family with two implementations, not two independent bets.
- **Parking:** T-bills stays the default. CORE5 gives the best long-window book (1.376, lowest DD) but is +0.011, under
  the +0.02 bar. SPMO since 2015 is +0.016, also under the bar.
- **Optional A, industry ETF:** not adopted. In the book it is lower (1.511 vs 1.540; CAGR 17.7% vs 18.8%), although
  standalone it has the best Sharpe and drawdown (correlation 0.47–0.48 with the stock pods).
- **Pre-2004 proxy** (DV2 + QPI as an HPI stand-in, 1995–2003, reported only):

  | Slot | Sharpe |
  |---|---|
  | Capsule, gated | 1.67 |
  | Capsule, ungated | 1.83 |
  | DV2-G | 1.63 |
  | QPI-G | 1.47 |

  The capsule beats its parts. The gate costs in the 1990s, as known.

## Optional B: limit entry × gate (Scout limit book, S&P 500 DV2, 2004–2022 sealed window)

Book TAA 0.5 + NDX 0.25 + DV2 0.25 (Sharpe), idle cash at T-bills (approximate sweep), Scout spread cost models:

| DV2 execution | Cost model | 2008–11 | 2012–22 | 2008–22 |
|---|---|---|---|---|
| T-bills slot | | 0.728 | 1.217 | 1.089 |
| Market-on-open, no gate | AR / pooled | 0.590 / 0.712 | 1.087 / 1.278 | 0.939 / 1.103 |
| Limit k 0.5, no gate | AR / pooled | **0.843 / 0.871** | 1.221 / 1.257 | 1.114 / 1.146 |
| Market-on-open + gate | AR / pooled | 0.562 / 0.656 | 1.309 / **1.427** | 1.078 / 1.184 |
| **Limit k 0.5 + gate** | AR / pooled | 0.769 / 0.791 | **1.353** / 1.374 | **1.177 / 1.197** |

The two are complementary in the book.
- The gate helps from 2012 on.
- The limit helps most in 2008–11.
- Together they give the best 2008–22 result under both cost models.

Standalone, the limit dominates and the gate adds nothing:
- AR: 0.78 with the limit, 0.77 with limit + gate;
- pooled: 0.89 with the limit, 0.84 with limit + gate.

Caveats: DV2 only (no HPI version), the 2023+ vault is unused, fills are optimistic (strict fills cut the limit's gain,
Scout), and the Scout book has no dividends. **Not adopted here:** a live order-type change needs paper fills first.

## Open decisions (owner)
1. Capsule as the MR slot despite the 0.001 R2 miss (recommended).
2. Idle cash: T-bills (default) or CORE5 (better long-window book and drawdown, not over the bar).
3. Limit execution: a paper shadow for DV2 (Scout recommendation); the evidence now says it complements the gate.

Outputs: `results/research/mr_capsule_20261003/` (gitignored). Scripts: `components.py`, `evaluate.py`, `limit_axis.py`.

## Owner decision (2026-10-03): parking — one pod in SPMO, the other in T-bills

**Assignment chosen: DV2 parks in SPMO (8% volatility target), HPI parks in T-bills.**

Since 2015-11 (SPMO), both cost levels:

| Assignment | Capsule $100K | Capsule Sharpe | Book $100K | Book Sharpe | Book 2015–20 / 2021–26 | Book at +5 bps |
|---|---|---|---|---|---|---|
| **DV2 → SPMO, HPI → T-bills** | **$979K** | **1.423** | **$858K** | **1.608** | **1.697 / 1.568** | **1.557** |
| DV2 → T-bills, HPI → SPMO | $944K | 1.394 | $848K | 1.599 | 1.679 / 1.565 | 1.548 |

- **SPMO record:** the chosen assignment is ahead on every line, in both halves, at both cost levels, and in the crises
  (2022: +2.0% vs +0.4%; Volmageddon −1.5% vs −2.2%).
- **Mechanism:** idle cash is about 88–90% for both pods while the gate is closed. While it is open (stress), DV2 is
  25% idle and HPI 35% idle. With SPMO in DV2, the momentum exposure sits mostly in calm markets, where it works, and
  less in stress, where momentum falls with the market.
- **Long-window proxies (2004–26):** with the SPY or QQQ 8% volatility target as the equity parking, the two
  assignments tie in book Sharpe (1.349 vs 1.348; 1.359 vs 1.361). So nothing argues against the choice.
- **The gap is small** (book +0.009 Sharpe, about +$10K per $100K over 11 years). The choice is made on consistency
  and mechanism, not size.
- **Implementation note:** the simulation re-weights SPMO daily (weight = min(1, 8% / 20-day realised volatility),
  rest T-bills). Live, a weekly re-weight with a tolerance band is the practical version; this was not tested.
