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

## Final parking decision (owner, 2026-10-04) — supersedes the 2026-10-03 assignment above

**Both pods park in SPMO, but only while the gate is closed; while it is open, idle cash sits in T-bills. The SPMO
weight is re-set weekly.** (Exploratory follow-ups `spmo_rebalance.py` and `spmo_gateoff.py`; SPMO trades charged
2.5 bps per side.)

### Re-weight frequency
Weekly ≈ daily.

| Re-weight | Capsule $100K | Sharpe |
|---|---|---|
| Daily | $963K | 1.413 |
| Weekly | $956K | 1.408 |

- Real SPMO since 2015-11, earlier assignment.
- Crisis onsets differ by ≤ 0.6 pp.
- A 10-point band adds nothing.
- Monthly is worse at Q4 2018.

### The gate rule
Most SPMO trading came from the pods' own entries and exits. With SPMO held only while the gate is closed:
- turnover halves;
- the 2008 cost almost disappears;
- book Sharpe is unchanged.

Real SPMO, 2015-11 → 2026, weekly:

| Parking | Capsule $100K | Capsule Sharpe | Capsule max DD | Book $100K | Book Sharpe | Q4 2018 | 2022 | 2025 |
|---|---|---|---|---|---|---|---|---|
| T-bills both (reference) | $743K | 1.311 | −21.3% | $803K | 1.593 | −6.0% | +3.0% | −10.0% |
| DV2 SPMO / HPI T-bills | $956K | 1.408 | −23.2% | $853K | 1.603 | −9.5% | +1.9% | −12.2% |
| DV2 SPMO / HPI T-bills + rule | $909K | 1.403 | −22.5% | $842K | 1.603 | −8.2% | +2.7% | −11.7% |
| SPMO both, no rule | $1.16M | 1.413 | −25.2% | $894K | 1.598 | −13.0% | −0.7% | −14.2% |
| **SPMO both + rule (chosen)** | **$1.07M** | **1.427** | −23.4% | **$877K** | **1.603** | −10.5% | +2.3% | −13.2% |

2008 (PDP / synthetic proxies), whole GFC window (2007-10 → 2009-03):

| Parking | Return |
|---|---|
| T-bills both | +7.2% |
| SPMO both, no rule | −4.6% / −1.0% |
| **SPMO both + rule** | **+5.5% / +5.5%** |

Calendar 2008: T-bills both +14.3%; SPMO both + rule +12.8% / +12.9%.

Long history (2008–26, proxy before 2015-11):

| Parking | Book Sharpe | Book $100K | Book max DD |
|---|---|---|---|
| T-bills both | 1.365 | $2.39M | −15.1% |
| SPMO both + rule | 1.348–1.352 | $2.63–2.65M | −15.2% |

On the long history SPMO both + rule makes more money at a slightly lower Sharpe. Since 2015 it is higher on both.

### Final capsule specification (for the build)
- **Gate (shared):**
  - After each close, VIX > expanding mean of all VIX closes since 1990-01-02 (min 500) opens the gate.
  - Once open, it stays open at least 15 sessions from the opening.
  - It closes on the first close at or below the threshold after that.
- **Pods:**
  - DV2-G: wired DV2 rules.
  - HPI-G: HPI 2/3/5 vote rules.
  - Both: new entries only while the gate is open; exits always by the pod's own rule.
- **Idle cash of each pod:**
  - Gate open: T-bills (BIL).
  - Gate closed: SPMO at weight min(1, 8% / 20-day realised volatility), the rest BIL.
  - Re-weight every Friday close; set it to the target at the close the gate shuts, and to 0 at the close it opens.
  - Orders fill at the next open.
  - New stock entries happen only while the gate is open, when SPMO is 0. So entries are funded from cash or BIL and
    never force SPMO sales.
- **Capsule:** 50/50 capital, two separate sub-accounts, reset to 50/50 once a year.
- **Untested at build time (to verify):**
  - engine parity of DV2-G (replica only so far);
  - the BIL trades of the T-bill leg;
  - small-account minimum commissions.

## Build record (2026-10-04): the capsule in the real engine

Research-only. Nothing live changed.
- **Code:** `strategies/mr_capsule/` (shared gate, parking, two pods).
- **Tests:** `tests/test_strategy_mr_capsule_gate.py`, `tests/test_strategy_mr_capsule_pods.py` (36).
- **Checks:** `scripts/research/mr_capsule_build_20261004/run_engine.py` and `compare.py`; outputs go to `results/research/mr_capsule_build_20261004/` (gitignored).
- **Status:** both pods are RESEARCH (absent from the registry). They have no live route and no portfolio-manager route.

### Parity with the research record

| Check | Result |
|---|---|
| HPI-G, idle cash at 0%, vs the research engine run (check B) | Identical: 5,998 of 5,998 trade events; NAV max difference 0.0 over 5,724 sessions |
| DV2-G, idle cash at 0%, vs the research replica (first engine run of DV2-G) | 98.7% of trade events in common (9,822); daily return corr 0.9988; 2004–26 CAGR 15.91% vs 16.01%, Sharpe 0.951 vs 0.955, max DD −29.0% both |
| Gate vs the research gate | Identical on real VIX 2004–2026 (independent review); identical on a synthetic path with 20+ openings and at the 15-session boundary (tests) |
| Stock trades with parking on | Same as the cash-only runs: parking never changes a stock decision |

### Build amendments
These are implementation changes, not new research.
- **B1, SPMO tradability guard.**
  - SPMO gets a weight only if it traded on each of the last 20 sessions.
  - In 2015–2017 SPMO had 22, 151 and 87 sessions without a trade, and a median daily turnover of USD 0–2K. Its prices there are stale.
  - From 2018 it trades every session (median USD 0.3M a day in 2018–21, USD 250M in 2026), and the guard is inactive.
  - So idle cash stays in BIL until 2018.
- **B2, weekly BIL sweep.**
  - BIL is bought only on re-target closes (week end, gate switch). It is sold whenever the day's orders need the cash.
  - Re-targeting BIL on every close traded it 72 times a year in DV2-G, about 20× the pod one-way.
  - The sweep halves the orders to 39 a year (11× the pod) and lifts DV2-G from 16.61% / 0.970 to 16.84% / 0.981 (2004–26).
- **Padding.** The parking ETFs keep Norgate's market-day padding in both pods, as in DV2's own frame. HPI's removed-name rule therefore never force-sells a held ETF on a session without a trade.
- **Statistics.** Trade statistics and exposure time count stock trades only. NAV and costs include the parking.

### Engine results: capsule 50/50, annual reset

CAGR / Sharpe / Max DD.

| Idle cash | 2004–2026 | 2007-06 → 2026 | 2018-02 → 2026 |
|---|---|---|---|
| SPMO spec (gate closed SPMO + BIL; open BIL) | 15.01% / 1.034 / −23.5% | 16.90% / 1.076 / −23.5% | 22.83% / 1.276 / −23.5% |
| BIL only | 14.31% / 1.019 / −21.4% | 16.07% / 1.058 / −21.4% | 22.20% / 1.307 / −21.4% |
| Cash at 0% | 14.18% / 1.010 / −21.6% | 15.91% / 1.049 / −21.6% | 21.54% / 1.274 / −21.6% |
| Research record, SPMO spec | 17.45% / 1.167 / −24.4% | 19.14% / 1.186 / −24.4% | 23.74% / 1.321 / −24.4% |

From 2018-02, where both sides use the same data, the engine is 0.9 pp/yr and 0.045 Sharpe below the research record. The research did not charge the following frictions:
- trading costs on about 16× the pod a year of BIL and SPMO (about 0.5 pp);
- the 25% withholding on BIL (about 0.3 pp, conservative);
- BIL's expense ratio;
- the 1% cash buffer.

Before 2018 the gap also includes 0% cash before BIL (2004–07) and the B1 guard.

Capsule in the crises (engine):

| Window | SPMO spec | BIL only |
|---|---|---|
| GFC 2007-10 → 2009-03 | +7.3% | +7.3% |
| Aug 2011 | −10.8% | −10.8% |
| Q4 2018 | −12.8% | −6.3% |
| COVID 2020 | −23.5% | −21.4% |
| 2022 | +14.3% | +15.6% |
| Feb–Apr 2025 | −13.2% | −10.2% |

### SPMO vs BIL: the parking decision needs revisiting

Capsule Sharpe / book Sharpe. The book is TAA 0.5 + NDX 0.25 + capsule 0.25.

| Window | Research T-bills | Research SPMO | Engine BIL | Engine SPMO |
|---|---|---|---|---|
| 2015-11 → 2018-01 (SPMO mostly untraded) | 1.124 / 2.254 | 2.318 / 2.540 | 1.075 / 2.243 | 1.548 / 2.312 |
| 2018-02 → 2026-08 | 1.388 / 1.498 | 1.330 / 1.461 | 1.316 / 1.475 | 1.278 / 1.440 |
| 2008-03 → 2026-08 (book CAGR) | 1.365 (18.79%) | 1.378 (19.34%) | 1.351 (18.57%) | 1.343 (18.78%) |

**The research case for SPMO came from 2015-11 to 2018-01.** That case was "since 2015 it is higher on both money and Sharpe". But in those years SPMO did not trade on most days.

**From 2018-02, research and engine agree:**
- SPMO adds a little money: capsule +0.1 to +0.6 pp/yr, book +0.05 to +0.15 pp/yr.
- It lowers Sharpe: capsule −0.04 to −0.06, book −0.035.
- It deepens drawdowns: capsule −23.5% vs −21.4%.

**Claude's recommendation, updated: BIL only.** The owner decides. The build keeps the owner's 2026-10-04 choice (SPMO) as the default; `spmo_parking_enabled_bool=False` runs BIL only.

### Multiple testing

DSR of the engine capsule on daily excess returns over T-bills. N = 110 trials, counted as independent; the null sampling variance is used.

| Window | DSR | Sharpe vs deflated benchmark |
|---|---|---|
| 2004–26 | 0.966 | 0.92 vs 0.53 |
| From 2007-06 | 0.962 | — |
| From 2015-11 | 0.88 | shorter sample |
| From 2018-02 | 0.78 | shorter sample |

What counts against the record:
- about 100 gate variants and about 10 parking forks;
- SPMO missed its pre-registered +0.02 bar (+0.016) and was adopted by owner decision;
- "SPMO only while the gate is closed" was chosen on the same 2015–26 window;
- the 2022–26 block overlaps Scout's 2023+ vault.

Forward paper is the only clean evidence left.

### Independent review (quant-pitfalls agent, 2026-10-04)

**No issues found:**
- no lookahead or leakage;
- no drift from the parents: an AST diff of the copied `iterate` bodies, plus parity with 10 full slots;
- the gate is identical on real VIX.

**Fixed:**
- BIL churn (B2);
- a vacuous gate test, now replaced, with a boundary test added;
- no-volume ETF sessions (B1 and padding);
- the signal-audit state: the SPMO weight is now computed per decision from the engine's data;
- stock-only trade statistics;
- test gaps:
  - full-slot parity with the parking symbols in the frame;
  - a test that the weekly re-weight happens;
  - a test of `append_parking_prices`;
  - the BIL-only mode.

**On record, not fixed:**
- **Negative cash.** It comes from the parents' 10 × 10% sizing, not from parking:
  - DV2-G: 136 sessions with parking vs 133 without, minimum −8.8% of NAV;
  - HPI-G: 128 vs 127.
  - It is not financed (G-023).
- **Slippage.** The engine charges 2.5 bps on BIL, against BIL's spread of about 1 bp. This is conservative.
- **Small-pod commissions.** About 52–58 parking orders a year at the USD 1 minimum cost about 0.35–0.4 pp/yr at USD 15K per pod and about 0.1 at USD 50K.

### Before any paper run (owner decisions)
- **Parking:** SPMO spec, or BIL only (recommended).
- **Accounts:**
  - a margin account per pod, because entries on a gate-opening day are funded by the same auction's sales;
  - one account per pod, because both pods hold BIL.
- **Data:** the snapshot profiles would need SPMO, BIL and $VIX from 1990.
- **Portfolio manager:** no manager-format book until the pods are PM_READY. The 50/50 annual reset is computed in `compare.py`.
