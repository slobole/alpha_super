# Scout P5: the two live pods re-audited through S4-S6

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Scripts:** `scripts/research/scout_p5_reaudition_20261002/`; calibration in `scout_p5_calibration_20261002/`.
- **Cards:** `results/scout/cards/TAA_3x_reaudition_20261002.html` and `NDX_VXN_reaudition_20261002.html`. These are
  not in git; the numbers below come from them.
- **Data window:** in sample = up to the vault seal (2022-12-30). The years since were seen before, so they are
  shown separately and do not count as a clean test.
- **Review:** an independent read-only review found one bug that changed a conclusion (NDX alpha). It is fixed and
  everything was re-run.

## Verdict

| | TAA 3x | NDX VXN |
|---|---|---|
| **Scout grade** | **CANDIDATE (S3 pending)** | **WATCHLIST** |
| In-sample Sharpe, live decision day / median over 16 decision days | 1.18 / **0.92** | 0.70 / **0.59** |
| S4 plateau (live configuration's rank) | 0.94, live = plateau choice (1 of 20) | 0.90, plateau prefers ROC 6 (live ROC 12 is 11 of 36) |
| S4 twice the costs + 10 bp | Sharpe 1.12 | Sharpe 0.64 |
| **S5 MCPT (the gate)** | **p 0.010 PASS** | **stock selection p 0.18, timing overlay p 0.09: FAIL** |
| S5 DSR, live (warning) | p 0.008 | p 0.062 (WARN) |
| S6 net alpha, fullest model | **+13.2% a year, t 2.71** | +3.7% a year, t 1.25 (FAIL) |
| **S6 T-bill slot** | **P 0.92**, book 1.20 vs 0.93 | **P 0.47**, book 1.20 vs 1.21 (FAIL) |
| Capacity at 1% of ADV (last 3 years) | **$0.8M** (BTAL binds) | $18.9M |

**What the grades mean** (section 10: a re-audition never demotes a pod by itself; that stays the owner's decision):
- **TAA 3x passes every Scout test** it can take today. It is the first strategy through the full S4-S6
  pipeline.
- **NDX VXN fails the gate and adds nothing to this book.**
  - Neither its stock picking nor its market-timing overlay can be told apart from what its own search would find
    in data without structure.
  - Inside the live book (2012-2022), replacing it with T-bills leaves the book's Sharpe unchanged (1.21 vs 1.20).
  - It is not evidence against the pod (D22): it is WATCHLIST, tracked forward.

## TAA 3x

- **Planning number.** The live month-end decision is the best of 16 decision days (Sharpe 1.18). The median is
  0.92 and the worst is 0.79.
  - Moving the decision one session earlier drops the Sharpe to 0.87. The review checked that this jump is real,
    not a code artefact.
  - Plan with about 0.9, not 1.2.
- **Grid:** live is the plateau's centre, ratio 0.94, so the parameters are not a lone lucky peak.
- **MCPT p = 0.010.** The search beats 99% of its shuffled histories. The score nets out a volatility-targeted
  equal weight of the same six ETFs, so the VIX gate's generic volatility timing is not counted as edge (A9).
- **Factor alpha, net:**

  | Factors | Net alpha / yr | t | Gross / yr | t |
  |---|---|---|---|---|
  | QQQ | +14.7% | 3.26 | +15.3% | 3.41 |
  | ETF mix (SPY QQQ IEF GLD) | +15.0% | 3.26 | +15.6% | 3.41 |
  | + trend | +14.3% | 3.04 | +14.9% | 3.18 |
  | + NDX VXN | +13.2% | 2.71 | +13.8% | 2.84 |

- **Crises** (2018 Q4, 2020, 2022): −2.7%, −0.4%, −3.8%, against SPY −19%, −33%, −24%.
- **The weak points:**
  - **Capacity.** At 1% of ADV the binding asset is BTAL, so the pod holds about $0.8M on today's volumes
    ($0.25M over the full history). This is a simplified estimate, not the capacity v2 study. It is fine for the
    $18K live slot; for a fund it needs routing, or a substitute for BTAL.
  - **History:** only 10 years in sample (2012-2022).
  - **Live record:** 124 sessions since adoption. Showing even that the Sharpe is above zero would take about
    24 months of live record.
- **Since 2023 (seen, not clean):** Sharpe 1.79, +29% a year.

## NDX VXN

- **Gate.**
  - **Stock selection:** the real score is 0.52; the null's 95th percentile is 0.67.
  - **Overlay (SPY regime × VXN scale):** 0.39 against 0.46.
  - The selection null is the per-asset null of P4b, and the overlay null is a date shuffle with the A9 score.
- **Grid.** The surface is flat: Sharpe 0.60-0.85 across 36 configurations. The plateau prefers a 6-month ROC; the
  live 12-month rule ranks 11th. PBO 0.77 (diagnostic) says the in-sample best usually does not stay best.
- **Alpha depends on the factor set and the period:**

  | Factors | Net alpha / yr | t | Months |
  |---|---|---|---|
  | QQQ | +7.7% | 3.32 | 276 (2000 on) |
  | ETF mix | +6.6% | 2.53 | 218 (2004-11 on, when GLD starts) |
  | + trend | +6.1% | 2.44 | 218 |
  | + TAA 3x | +3.7% | 1.25 | 123 (2012-10 on) |

  On its own it beats QQQ and the ETF factors. Inside the book, over the years both pods trade, TAA 3x already
  carries what it adds.
- **Crises:**
  - The regime filter worked in 2008: GFC −3.5% against SPY −55%.
  - In fast crashes it holds less well: 2020 −11.9%, 2022 −12.3%, 2011 −11%.
- **Live record:** 95 sessions since adoption, Sharpe 0.48. About 68 months would be needed to show that the
  Sharpe is above zero.

## Findings about the method

- **A9: the MCPT score for volatility-timed families.** It was decided by a pre-registered calibration
  (`scout_p5_calibration_20261002/PROTOCOL.md`, amendment 1 before any result):

  | Score | False passes, TAA-like / trend | Power |
  |---|---|---|
  | Sharpe minus equal-weight Sharpe (A5 as written) | 2.5% / **8.5%** | — |
  | Sharpe minus volatility-targeted equal-weight Sharpe | 1.0% / 7.0% | 14% |
  | Active return over equal weight | **8.5%** / 6.5% | — |
  | **Active return over volatility-targeted equal weight (adopted)** | **2.0% / 7.0%** | **21%** |

  - **Power on TAA-like families is low** (6%). The leveraged fallback dominates their risk, so ranking the
    defensive assets barely moves the Sharpe.
  - **TAA 3x's p = 0.010 therefore reflects a large effect, not a sensitive test.**
- **Fast replicas** (`alpha/scout/searches.py`) run the MCPT. Against the engine:
  - daily correlation is 0.98-0.995;
  - the ranking of configurations agrees at 0.96 (TAA) and 0.83-0.87 (NDX).
  - The NDX replica's plateau pick differs from the engine's (ROC 15 vs 6). The FAIL holds for every
    configuration: none scores above 0.52 against a null 95th percentile of 0.67.

## Review fixes applied before the final run

- **HIGH:** months before a factor existed were counted as 0% returns, which inflated NDX's fullest-model alpha
  from t 1.17 to t 2.94. Empty months are now NaN, and the card states each model's months.
- **Luck band:** limited to offsets 0-15. Offsets 19-20 skipped months.
- **Card headline:** it now quotes the median decision day, the number to plan with.
- **DSR check:** it reports the live configuration.
- **One common window:** every configuration is compared over the same in-sample dates, from the date all of them
  have started.
- **Diversification:** each comparison is aligned on its own overlap, so NDX's 2008 and 2011 crises are back.
- **Trend factor:** a one-month over-lag removed, and it is now built on excess returns.
- **Smaller items:**
  - the capacity label names the asset at the quoted percentile;
  - the grade reads "CANDIDATE (S3 pending)";
  - the minimum-record text says what it measures.
- **New tests:**
  - a factor that starts late;
  - the grade mapping;
  - the TAA replica against the engine (Norgate).

## Not done in P5, and why

- **S3 for the re-auditions:** class W needs per-asset predictive regressions, and class X an event definition.
  This comes in P6 with the other pods.
- **The G3 reference book:** only the live book was used for the T-bill slot.
- **Fama-French factors:** not stored offline.
- **Capacity:** the full capacity v2 study was not run.
- **Ledger:** the two RETRO registrations are written (`research_ledger/scout_ledger.jsonl`, rows 1-2). Station
  verdict rows will follow once the owner has read the cards.
