# DV2 deep research — verdict (2026-09-25)

Research-only. Nothing in live trading, releases, schedulers or broker routes changed.
Owner request: "run everything, final verdict at the end" (/goal). Frozen plan and every later amendment:
[`scripts/research/dv2_deep_20260925/SPEC_FROZEN.md`](../../scripts/research/dv2_deep_20260925/SPEC_FROZEN.md).

## Verdict

**Keep DV2 in the growth book, change one thing, and do not move it to the close auction.**

1. **Change the ranking from NATR14 to 63-day dollar volume (ADV63), keeping the liquidity floor** (finalist F1).
   It is the only change that is better in-sample (Sharpe 1.18 vs 1.07, p = 0.04, max DD −21% vs −24%), passes
   the untouched 1991–1999 holdout (Sharpe 2.37 vs 2.34), survives +5 bps/side (0.94 vs 0.86), and lifts the
   growth book's stressed Calmar above G3 (1.62 vs 1.54). It missed the strict "better" rule only because
   2021–2026 is flat (−0.015 Sharpe). NATR ranking is no better than a random ranking. Adopt through a shadow
   period, not directly.
2. **Keep everything else as it is**: DV2(126) < 10, Close > SMA200, 126-day return > 5%, the Close > yesterday's
   High exit, 10 slots, next-open execution. None of the alternatives beat them.
3. **Do not trade DV2 at the same-day close.** With real 15:45 data (Alpaca SIP, 2016–2026), deciding at 15:45
   and filling at the close is worse than today's next-open fill for every version (Sharpe −0.10 to −0.27).
   Exits at the close are worse even with the final close known.
4. **Add an industry-ETF DV2 pod as a forward-test candidate** (same frozen rules, ADV > $50M). Alone it is small
   (CAGR 5–6%, Sharpe 0.87–0.92, DD −10 to −12%), but splitting DV2's book weight 50/50 with it gives the best
   book tested: Calmar 1.85 (1.72 stressed), max DD −11% incl. 2008, and twice the close-auction capacity.
5. **Know what DV2 is today.** Its alpha against the S&P 500 was large before 2010 and is close to zero in calm
   markets since (2010–14 +1%/yr, 2015–19 +4%, 2023–26 +3%, all t < 1). It earns in stressed markets (2020–22
   +19%/yr, t = 1.9). In calm years it behaves like ~0.85 S&P 500 beta. It is worth keeping as a crisis-convex
   diversifier, not as a stand-alone alpha engine.
6. **Capacity is the binding constraint for a fund.** DV2 alone scales to about $5M in the house model at the
   close auction (open auction: ~$1M). A stricter liquidity floor buys capacity with edge
   (top-quartile ADV: $10M, CAGR 17% instead of 22%).

## How it was tested

- **Instrument.** A fast replica of the Vanilla engine for this rule family, validated trade for trade: WIRED DV2
  (25,086 fills), liquidity floor (23,184), ADV rank (25,636), and all five finalists through the real engine
  (F1 23,286; F2 32,163; F3 20,556; F4 22,862; ETF 4,652 fills — zero differences; F4 needed the median
  definition of the rewritten floor module, see "Out-of-scope findings"). ~5 s per 26-year run vs ~15 min.
- **Pre-registration.** Grid, promotion rules, holdout and MOC gates frozen before any result; five dated
  amendments, each written before the results they concern.
- **Search size.** 98 pre-declared configurations, their +5 bps/side reruns, 200 random-rank runs (luck band),
  23 universe/ETF runs, 4 post-hoc liquidity runs (labelled), MOC and hybrid runs. Deflated Sharpe with 148 trials
  (including ~50 prior DV2 trials): about 1.0 for the baseline — the DV2 edge itself is not a multiple-testing
  artifact. Promotions were decided by paired block bootstraps and per-period rules.
- **Data.** Norgate US incl. delisted, point-in-time S&P 500 from 1990. Main window 2000-01-03 → 2026-08-19,
  $1M, engine costs (2.5 bps/side + $0.005/share). Periods P1 2000–14, P2 2015–20, P3 2021–26. Locked holdout
  1991–1999, used once.

## Your questions, answered

### DV2 vs DV3 / DV4 / longer, and the 126 window

Sharpe on the floor baseline (rows = DV smoothing k, columns = rank window):

| k | 63 | 126 | 252 |
|---|---|---|---|
| 1 | 0.96 | 1.02 | 1.02 |
| **2** | 0.99 | **1.07** | 1.14 |
| 3 | 0.95 | 1.06 | 1.05 |
| 4 | 0.91 | 0.92 | 0.94 |
| 5 | 0.93 | 0.94 | 0.92 |
| 10 | 0.91 | 0.87 | 0.81 |

DV2 and DV3 form a small plateau; k ≥ 4 is clearly worse. The smoothing length barely changes turnover (90–93×/yr),
so the cost argument for longer k does not hold. A 252-day rank window (Varadi's original) is +0.075 Sharpe,
positive in every period, but not significant (p = 0.09): no reason to change. Threshold: 10 is best (5: 1.03,
15: 0.97, 20: 0.87).

### The 5% momentum threshold

| 6-month return threshold | 63-day | 126-day | 252-day |
|---|---|---|---|
| > 0% | 0.95 | 0.99 | 0.96 |
| > 5% | 0.97 | **1.07** | 0.98 |
| > 10% | 1.00 | 1.10 | 1.02 |

The 5% was chosen after seeing results, but it sits on a slope, not a spike: 0% is significantly worse (paired
bootstrap ~98% that 5% > 0%) and 10% is about the same. Keep 5% and record it as an in-house choice. The
threshold-free vote "2 of 3/6/12-month returns positive" is significantly worse (0.96). Dropping the filter: 0.92.
SMA filter: 150–250 is a plateau (none: 1.04).

### Ensembles instead of one parameter

| Candidate | CAGR | Sharpe | Max DD | Turnover | Book Calmar (stressed) |
|---|---|---|---|---|---|
| Floor baseline | 22.2% | 1.07 | −24.4% | 93× | 1.69 (1.54) |
| Vote: ≥5 of 9 DV percentiles < 10 | 20.5% | 1.07 | −21.9% | 83× | 1.82 (1.66) |
| Average of 9 percentiles < 10 | 17.1% | 0.98 | −24.4% | 73× | — |

The vote ensemble matches the baseline with 10% less trading, a shallower drawdown and the best single-pod book
Calmar. It missed the pre-declared non-inferiority bar by 0.003 on the bootstrap 5th percentile, and at 15:45 it
degrades the most (Sharpe 0.74). A legitimate alternative, not an improvement.

### Ranking

A random choice among the same candidates gives Sharpe 1.02–1.16 (90% band, median 1.09). NATR14 scores 1.07 —
the 33rd percentile: no better than random. ADV63 scores 1.18 (97.5th percentile) with the smallest drawdown.
DV2-lowest-first: 1.14.

### Other universes

| Universe (frozen rules, real exit) | CAGR | Sharpe | Max DD | Note |
|---|---|---|---|---|
| S&P 500 floor | 22.2% | 1.07 | −24% | reference |
| S&P MidCap 400 floor | 11.5% | 0.60 | −46% | edge mostly pre-2015 |
| S&P SmallCap 600 floor | 15.6% | 0.66 | −65% | P2 0.39, P3 0.25 |
| ETF industries (19) | 6.0% | 0.87 | −12% | all periods positive; corr 0.48 |
| ETF industries, ADV > $50M | 5.2% | 0.92 | −10% | post-hoc liquidity screen |
| All 60 ETFs | 6.1% | 0.66 | −18% | |
| Sector SPDRs / countries / broad | 1.7–1.9% | 0.33–0.54 | | fail the ETF gate |

The July Russell transfer failure holds with the real exit: the edge is specific to the S&P 500 (and the 1990s).
Industry ETFs are the one new universe that works.

### Exits

| Exit (same entries) | CAGR | Sharpe | Stressed Sharpe |
|---|---|---|---|
| **Close > yesterday's High (current)** | **22.2%** | **1.07** | **0.86** |
| or Close < SMA200 | 21.5% | 1.05 | 0.82 |
| or 10-day time limit | 20.9% | 1.01 | 0.80 |
| IBS > 0.9 or RSI2 > 90 (Murphy's Law) | 19.9% | 0.97 | 0.81 |
| Close > SMA5 | 19.8% | 0.96 | 0.71 |
| DV2 > 50 | 18.5% | 0.95 | 0.68 |
| First up-close | 16.8% | 0.90 | 0.48 |

The current exit wins. The article's IBS/RSI2 exit fails on DV2 again, as it did in March.

## Where the money comes from

- **Overnight.** About 17%/yr of the return accrues overnight and 2–4% during the day.
- **Stress.** Return beyond an exposure-matched S&P 500 is concentrated in stressed years (2000 +105%, 2002 +38%,
  2008 +49%, 2020 +56%, 2022 +19%) and near zero or negative in calm bull years (2023 −5%, 2024 +3%, 2025 −2%).
  By regime: 32%/yr when the S&P 500 is below its 200-day average vs 19% above; 8%/yr in the calmest volatility
  tercile.
- **Decay.** Sharpe 2.1–2.7 in 1991–1999, 0.9–1.5 in 2000–2009, 0.8–1.3 since (floor, ADV rank, WIRED).
- **No news effect.** Dips on volume spikes (a news proxy) revert as well as quiet dips (market-adjusted
  +0.34% vs +0.28% per trade), so a "fundamental repricing" filter has no support here.
- **Holdout.** Untouched 1991–1999: every finalist is profitable, Sharpe 2.3–2.5; still 1.0–1.3 at
  25 bps/side. The mechanism is real; it has decayed.

## Same-day close (MOC)

| Sharpe, 2016-01 → 2026-09 | Next open (today) | Close, final close known | 15:45 decision, exact |
|---|---|---|---|
| Floor | 1.01 | 1.17 | 0.92 |
| Floor + ADV rank | 1.06 | 1.09 | 0.87 |
| Vote ensemble | 1.01 | 1.08 | 0.74 |
| 252 window | 1.04 | 1.13 | 0.80 |
| WIRED | 1.07 | 1.04 | 0.80 |

Data: Alpaca SIP 15-minute bars for every name DV2 could act on (2,697 sessions, 96% coverage). Alpaca's official
close equals Norgate's within 10 bps 99.3% of the time. Missing names keep the final-close state, which flatters
the 15:45 result.

- **Why it fails.** Trades that only fire at 15:45 average +0.16–0.24%; the trades they miss average +0.46–0.51%.
  The best dislocations form in the last 15 minutes and the closing auction.
- **The model failed its validation.** The resampling model overstated the 2016+ result by ~4pp CAGR, so it was
  not used for 2000–2015.
- **Entry vs exit (exploratory).** Exits at the close are worse even with the final close known (the post-bounce
  overnight is lost). Entries at the close with exits at the next open are mixed across versions (+0.25, 0.00,
  −0.08 Sharpe) — not robust.

## Book level and capacity

Growth book G3+MR (TAA 32 / NDX 32 / DV2 18 / HPI 18, annual reset, 2012-10 → 2026-08, drawdown incl. the 2008
proxy):

| Book | CAGR | Sharpe | Max DD | Calmar | Calmar stressed | Book MOC capacity |
|---|---|---|---|---|---|---|
| G3 (TAA/NDX 50/50) | 23.2% | 1.42 | −14.8% | 1.57 | 1.54 | $2.5M |
| + MR, WIRED DV2 | 22.2% | 1.49 | −14.0% | 1.58 | 1.43 | $5M |
| + MR, floor | 21.9% | 1.48 | −13.0% | 1.69 | 1.54 | $5M |
| **+ MR, floor + ADV rank** | 21.8% | 1.49 | −12.2% | 1.78 | **1.62** | $5M |
| + MR, vote ensemble | 21.6% | 1.48 | −11.9% | 1.82 | 1.66 | $5M |
| **+ MR, ADV-rank DV2 9 + ETF-industries 9** | 20.8% | **1.51** | **−11.3%** | **1.85** | **1.72** | **$10M** |

All books pass the growth rules. The book's capacity limit comes from HPI (NWS) and TAA (BTAL), not DV2. DV2 alone:

| DV2 version | Close-auction recommended size | First failure | Modelled cost at $25M |
|---|---|---|---|
| WIRED | $1M | NWS | 7.3%/yr |
| Floor / ADV rank / ensemble / 252 | $5M | $10M (TPL, EQT, PCAR) | 4.4–5.1%/yr |
| Top-quartile ADV floor + ADV rank | $10M | $25M | 3.3%/yr |
| ETF industries, ADV > $50M | $2.5M | — | house ETF model; likely conservative |

## What this does not show

- All 2000–2026 results are on seen history. Only 1991–1999 was untouched, and it predates decimalization.
- Costs are the engine's plus a +5 bps/side stress. There is no measured auction impact; capacity uses the house
  model.
- The ETF list is today's ETFs (survivorship). Industry ETFs rarely close, but some did. The $50M ADV screen and
  the stricter stock floors are post-hoc and need forward testing.
- MOC hybrids and the stricter floors are exploratory (post-result) and were not promoted.
- The replica matches the engine exactly; neither models partial fills or queue position.

## Out-of-scope findings

- `strategies/dv2/strategy_mr_dv2_liquidity_floor.py` was rewritten by another agent at 18:46 today into a
  standalone class whose floor median now covers every member with ADV (previously members with complete DV2 rows).
  The effect is ≤ 0.01 Sharpe (F0 1.078 vs 1.070; F1 1.177 vs 1.179), but the definition in production should be
  chosen deliberately. Its docstring says Bench displays it as WIRED.
- Four March DV2 variant files (`ibs_rsi_exit`, `nasdaq100`, `r3000`, `price_adv`) have been 3-line self-importing
  wrappers since `bf1a334` (2026-04-04); their code exists only in git history.
- Norgate quirks: VLO's Dividend field is all zero when a request starts in 1989 (populated from 1998); a load
  ending on a date can miss a dividend dated the day before (APO, 2026-08-18). Caches here take every field from
  1998-start loads from 1998 on.
- The pakal DV2 studies used the source's 0% momentum filter, not the live 5%.

## Next steps (need your approval)

1. Shadow-run F1 (floor + ADV rank) next to the WIRED DV2 for 3–6 months on live data; switch if it tracks.
2. Forward-test the industry-ETF DV2 pod (ADV > $50M) as a paper pod.
3. Keep DV2 at the next open. If MOC matters for capacity, a paper test of MOC entries with MOO exits is the only
   variant worth trying.
4. Size DV2 for the fund knowing its standalone capacity (~$5M at the close in the house model) and that its
   calm-market alpha is near zero.

## Reproduce

Scripts: `scripts/research/dv2_deep_20260925/` — `data_cache.py`, `replica.py`, `validate_replica.py`,
`phase3_grid.py`, `analyze_grid.py`, `phase2_mechanism.py`, `moc_layer1.py`, `phase4_universes.py`,
`phase5_finalists.py`, `phase5_book.py`, `phase5_capacity_alone.py`, `phase5_followup_liquidity.py`,
`engine_finalists.py`, `compare_engine.py`, `moc_needed.py`, `alpaca_fetch.py`, `moc_layers23.py`, `moc_hybrid.py`.
Outputs: `results/research/dv2_deep_20260925/` (gitignored).

## Addendum 2026-09-26: forward-ready modules and book statistics

- `strategies/dv2/strategy_mr_dv2_liquidity_floor_adv_rank.py` (floor module + ADV63 rank) and
  `strategies/dv2/strategy_mr_dv2_industry_etf.py` (19 industry ETFs, ADV63 > $50M, WIRED DV2 rules, default
  start 2012-01-03). Both carry the Bench display-only WIRED badge like the floor module; registry tier RESEARCH,
  not in the live release manifest or the portfolio allowlist. Tests: `tests/test_strategy_mr_dv2_adv_rank_and_industry_etf.py`.
- Real-engine runs equal the replica trade for trade (ADV rank 2000-2026: 23,286 fills, CAGR 22.2%, Sharpe 1.18,
  max DD -21%; industry ETF 2012-2026: 2,962 fills, CAGR 8.2%, Sharpe 1.24, max DD -10.4%, vol 6.6%).
- Growth book 2012-10 -> 2026-08 (annual reset): G3+MR with DV2-ADV 9 + ETF 9: CAGR 20.7%, vol 13.1%, Sharpe 1.51,
  Sortino 1.99, max DD -10.8% (-11.0% incl. 2008 proxy), Calmar 1.92, worst year -1.5%, beta 0.52
  (`scripts/research/dv2_deep_20260925/phase6_book_stats.py`).
- Downshock filter on DV2-ADV (move < -0.5 x prior ATR): Sharpe 1.16 vs 1.18, max DD -19% vs -21%, 2021-26 Sharpe
  1.10 vs 0.86, 1990s 2.10 vs 2.37 - exploratory, not adopted.
- Every stock mean-reversion pod (DV2, HPI vote, HPI IBS/RSI, QPI) has the same profile: beta 0.56-0.77, pairwise
  correlation 0.74-0.87 with DV2, alpha concentrated in 2020-22 and pre-2010. The ETF mean-reversion pods (sector
  dispersion, industry-ETF DV2, VOX/IYR downshock) have beta 0.11-0.43 and steadier, smaller calm-market alpha.

## Addendum 2026-09-26 (2): mean-reversion shorts

Frozen plan: `scripts/research/dv2_deep_20260925/SPEC_SHORTS.md`; script `shorts_study.py`; replica short mechanics
equal the real engine trade for trade (SH1: 19,355 fills). S&P 500 PIT, liquidity floor, 10 slots, next open,
engine costs, borrow 0.5%/yr (stress 3%/yr), full dividend paid on shorts, no hard-to-borrow model (upper bound).

| Short pod, 2000-2026 | CAGR | Sharpe | Max DD | P3 2021-26 CAGR |
|---|---|---|---|---|
| SH1 mirror (DV2 > 90, below SMA200, R126 < -5%) | -2.2% | 0.08 | -92% | -16.8% |
| SH2 mirror, ADV rank | -2.1% | 0.03 | -87% | -13.1% |
| SH3 DV2 > 90, no trend filter | -6.8% | -0.07 | -97% | -27.9% |
| SH4 DV2 > 90 in an uptrend | -6.5% | -0.22 | -96% | -23.9% |

None is viable. In the capsule (DV2 50 / HPI 50), 20% SH1 instead of 20% cash cut the drawdown (-13% vs -19%) and
helped before 2021 but lowered the 2021-26 Sharpe (1.03 vs 1.27): not useful by the frozen rule. A causal
S&P futures beta hedge of the capsule: CAGR 10.3%, Sharpe 0.93, DD -16%, correlation to S&P -0.12. Earlier
archived runs agree (QPI short -0.1%/yr, DD -77%; DV2 short VIX filter +0.5%, DD -52%). Verdict: no
mean-reversion short leg.
