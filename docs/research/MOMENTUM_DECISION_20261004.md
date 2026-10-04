# Momentum family: map, evidence and decision (2026-10-04)

**Owner request (2026-10-04):** put everything about the momentum strategies in order (what exists, what was chosen,
the statistics) and decide on the momentum strategy.

**Evidence pack:** `scripts/research/scout_robustness_20261002/ndx_momentum_decision.py`, results in
`results/scout/robustness/ndx_momentum_decision.json`. The controls were registered in the Scout ledger before the
run (`ndx_momentum_decision_controls_20261004`). No new candidate was tried and nothing was re-selected.
All history through 2026-10 has been examined; nothing below is an untouched holdout.

## Decision (quant architect, owner-delegated; the live items need the owner's sign-off)

1. **One momentum slot, one family: monthly Nasdaq-100 stock momentum.** Top 10 by 12-month return per unit of
   volatility, Close > SMA100, SPY SMA200 regime gate, VXN exposure scale. No other momentum or trend line in the
   repository earns a slot (section 1).
2. **The design depends on the pod size:**

| Pod size | Design | Why |
|---|---|---|
| Below about USD 100K | **L**, the live rule (dollar-ATR ranking, 10 names) | The only variant that fits whole shares: 6% of the intended exposure is lost to rounding at USD 12K, against 22% for the blend |
| USD 100K and above, and every fund book | **E2 + 40% sector cap** (book `ndx_e2_sector_cap_5050`) | Two rankings, about 15 names, no sector above 40%; Max DD −21.5% vs −29.0% for L |

3. **Live account: no change.** L stays. A switch buys nothing measurable at the book level: in the live 60/40 book
   the variants are the same strategy (Sharpe 1.35 to 1.37 since 2012-11; section 5).
4. **Planning numbers for the slot:** Sharpe 0.75, CAGR 12%, Max DD −30%. These are the luck-band medians
   (0.74 to 0.78) and the worst variant's drawdown, not the 0.80 to 0.87 headline.
5. **What the slot is: timed Nasdaq exposure with a stock-picking tilt.** Registered verdict on the selection layer:
   **UNCLEAR**. It is clearly better than random picks (99.5th percentile of 200 books) and than equal weight
   (P 0.98), but not proven better than QQQ held at the same exposure (P 0.78; Sharpe +0.08; section 3).
6. **Closed:** the NDX parameter, filter and ranking search (about 45 trials, closed 2026-10-03) and every other
   momentum line in section 1. No shadow-to-decide (a 0.1 Sharpe gap at correlation 0.93 needs decades). Research
   time goes to book-level diversifiers (MR capsule, EOM), where the book's Sharpe actually moves.
7. **Open for the owner, two items only:**
   - approve the size rule in item 2;
   - whether to commission the live build of E2 + cap now. It is needed only when a pod of USD 100K or more is
     planned. Today the pair is two PM-ready strategies; a live pod needs one strategy that averages the two books.

## 1. What exists

59 strategy files under `strategies/momentum/` and about 60 momentum or trend studies in the knowledge base.

| Family | Files | Maturity | Best honest evidence | Status |
|---|---|---|---|---|
| **NDX monthly stock momentum** | 16 | 2 WIRED, 2 PM_READY, 12 research | Sharpe 0.80 to 0.87 over 2000-09 to 2026-10; luck median 0.74 to 0.81 | **The house momentum slot** |
| Stock momentum on other universes (Russell 1000 / 3000, S&P 500) | 25 | Research (MOSAIC demoted 2026-09-28) | MOSAIC 0.62 after the leak fix; Russell 1000 liquid leg ties the book (1.28 vs 1.30); S&P 500 breakout loses to T-bills in the same slot; smooth trend, AMAF, Paper B: Sharpe 0.1 to 0.5 | Closed, none passes |
| ETF and index trend or timing (sector ETFs, SPHB/SPLV canary, golden cross, adaptive momentum, MTUM timing) | 18 | Research | Sector ETF momentum: CAGR 7%, Sharpe 0.72; canary: 40 to 81 trades, short history, never validated; adaptive momentum already lives inside CORE5 | Not a momentum pod; closed |
| External claims audited in the knowledge base | about 60 studies | n/a | None promoted: rejected, diagnostic or forward hypothesis | Components only: volatility scaling, the IQR dispersion switch, low-vol + momentum rotation "D", NDX residual momentum (forward logs, no allocation) |

The NDX family, file by file:

| Strategy file (`strategy_mo_...`) | Maturity | Role |
|---|---|---|
| `atr_normalized_ndx_vxn_scaled` | WIRED | **LIVE** pod (about USD 12K): L |
| `atr_normalized_ndx` | WIRED | L without the VXN scale (Max DD −47%); no pod runs it today |
| `atr_normalized_ndx_vxn_scaled_sector_cap`, `natr20_ndx_vxn_scaled_sector_cap` | PM_READY | The two books of E2 + sector cap (built 2026-10-04) |
| `natr20_ndx`, `natr20_ndx_vxn_scaled` | Research | The scale-free ranking; half of E2 |
| `..._corr_penalty`, `..._roc_variants`, `..._weekly`, `..._weekly_kama`, `..._short`, `..._paper_b`, `..._paper_b_short`, `ev_lrb_252_ndx`, `radge_ndx`, `amaf_nasdaq100` | Research | Tried, none replaced L. Their saved runs predate the corporate-action fix of 2026-09-27 and are not comparable |

How the choice was made (all recorded):

| Date | Study | Outcome |
|---|---|---|
| 2026-07-31 | Correlation penalty on NDX; MOSAIC on Russell 1000 | No effect inside NDX; MOSAIC promoted, then demoted 2026-09-28 (0.62 after the leak fix) |
| 2026-09-26 | NDX parameter robustness (151 configurations, Reality Check p 0.61) | Keep L |
| 2026-09-26 / 27 | Russell 1000 liquid leg; trend, breakout and stops; momentum universe search | No replacement; stops do not help |
| 2026-10-02 / 03 | Scout A15: rankings (dollar ATR, NATR20, ROC, linear trend), 38 trend filters (Romano-Wolf p 0.19), plateau ensembles, candidate C | Filter stays SMA100; search closed; design of record C, then E2 |
| 2026-10-04 | Sector cap on E2 (registered rule passed); PM-ready pair and book; this decision pack | Design of record E2 + 40% sector cap; size rule; planning numbers |

## 2. The four NDX finalists, one method

2000-09-01 to 2026-10-02, Scout weights engine, engine costs, idle cash at T-bills.

| | CAGR | Vol | Sharpe | Sharpe (0% cash) | Max DD | Sharpe to 2022 | Sharpe 2023 on | Luck band min / median / max | Names |
|---|---|---|---|---|---|---|---|---|---|
| **L live** (dollar ATR) | 12.6% | 16.5% | 0.80 | 0.77 | −29.0% | 0.73 | 1.14 | 0.71 / 0.74 / 0.81 | 10 |
| NATR20 | 15.2% | 18.8% | 0.85 | 0.82 | −29.3% | 0.84 | 0.95 | 0.71 / 0.81 / 0.88 | 10 |
| E2 (50/50 of the two) | 14.0% | 17.1% | 0.85 | 0.82 | −24.8% | 0.81 | 1.07 | 0.75 / 0.80 / 0.85 | 15 |
| **E2 + 40% sector cap** | 13.6% | 16.1% | 0.87 | 0.84 | **−21.5%** | 0.85 | 1.01 | 0.74 / 0.78 / 0.87 | 15 |

- **They are one strategy in four coats:** daily correlations 0.88 to 0.98. P(E2 + cap > L) is 0.88 on Sharpe over
  the full period, and L led after 2022.
- **Recent years (calendar returns):**

| | 2022 | 2023 | 2024 | 2025 | 2026 to date |
|---|---|---|---|---|---|
| L live | −13.4% | 9.9% | 15.2% | 37.5% | 29.7% |
| E2 | −16.7% | 13.0% | 19.9% | 25.1% | 35.3% |
| E2 + sector cap | −17.0% | 13.0% | 21.3% | 24.1% | 15.0% |
| QQQ buy and hold | −32.6% | 54.9% | 25.6% | 20.8% | 22.4% |

- **The cap's price is visible in 2026:** +15.0% against +35.3% without it (the semiconductor run of May and June).
  Its benefit was the July crash (−11.5% against −17.3%) and the smaller full-period drawdown.
- **Crashes (window return):** dot-com from 2000-09 / GFC / 2011 / Q4 2018 / COVID / 2022 / 2025:
  L +0.7, −2.1, −11.0, −3.3, −11.9, −11.4, −13.1; E2 + cap +2.3, +1.8, −5.3, −6.3, −11.9, −15.0, −16.2 (%).

## 3. Does the stock selection earn its place?

The same gates and the same exposure, with the selection removed.

| 2000-09 to 2026-10 | CAGR | Vol | Sharpe | Max DD | Sharpe to 2022 | Sharpe 2023 on |
|---|---|---|---|---|---|---|
| **QQQ at L's own daily exposure** (idle part at T-bills) | 10.4% | 13.8% | 0.79 | −30.1% | 0.73 | 1.07 |
| Every eligible member, equal weight (65 names on average) | 8.8% | 13.4% | 0.69 | −22.3% | 0.66 | 0.85 |
| 200 random-pick books, turnover-matched (5th / median / 95th percentile) | 7.2% / 9.5% / 11.6% | n/a | 0.53 / 0.67 / 0.80 | −30% / −26% / −22% | median 0.66 | median 0.71 |
| QQQ buy and hold | 8.7% | 25.6% | 0.45 | −80.4% | 0.32 | 1.51 |
| QQQ 200-day rule, else T-bills | 9.5% | 15.1% | 0.68 | −35.8% | 0.54 | 1.41 |

| Finalist | Percentile among random books | vs equal weight: P, Sharpe gap | vs QQQ at the same exposure: P, Sharpe gap |
|---|---|---|---|
| L live | 94% | 0.87, +0.10 | 0.54, +0.01 |
| NATR20 | 98% | 0.94, +0.15 | 0.69, +0.06 |
| E2 | 98% | 0.97, +0.16 | 0.72, +0.06 |
| E2 + sector cap | 99.5% | 0.98, +0.18 | 0.78, +0.08 |

- **Registered verdict for the design of record: UNCLEAR.** The rule asked for P of at least 0.80 with a gap of at
  least 0.05 against both controls and the 90th percentile of the random books. It clears the random books and equal
  weight, and misses QQQ at the same exposure by a hair (P 0.78).
- **Reading:** the ranking carries real information against the neutral benchmarks (random picks, equal weight).
  Cap-weighted QQQ is a harder benchmark because the mega-caps won the period.
- **Most of the Sharpe comes from the gates.** QQQ behind the same regime gate and VXN scale earns 0.79.
  The selection adds about 3 points of CAGR a year (12.6% to 13.6% against 10.4%) at higher volatility: about 0.5 to
  2 points a year at equal risk.
- **The selection's gain came in two stretches:** 2000-09 to 2013-09 (CAGR 11.0% for L and 13.6% for E2 + cap,
  against 8.2% for the control) and 2021-10 on (16.6% and 11.9% against 8.3%). From 2013-10 to 2021-09 it added
  nothing (12.7% and 14.6% against 15.5%). The table is in section 8.

## 4. Factor alpha

Weekly excess returns, Newey-West 4 lags (the standing method). Annual alpha and t-statistic.

| | vs QQQ, full | vs QQQ + QQQ 200-day rule, full | vs ETF mix + rule, 2007 on | First half (to 2013-09), QQQ + rule | Second half (2013-09 on), QQQ + rule |
|---|---|---|---|---|---|
| L live | 8.1% (3.1) | 4.9% (2.2) | 4.8% (1.9) | 7.5% (2.4) | 2.2% (0.7) |
| NATR20 | 10.2% (3.6) | 6.7% (2.8) | 6.0% (2.2) | 10.1% (3.4) | 2.2% (0.6) |
| E2 | 9.1% (3.6) | 5.8% (2.7) | 5.4% (2.3) | 8.8% (3.1) | 2.3% (0.7) |
| E2 + sector cap | 8.9% (3.7) | 5.7% (2.8) | 4.9% (2.2) | 9.7% (3.6) | 1.2% (0.4) |

ETF mix = QQQ, IEF, GLD, DBC, UUP. QQQ beta is 0.34 to 0.39 (0.11 to 0.13 once the 200-day rule is in the model).

- **Full period: significant alpha** of about 5% to 6% a year after QQQ and a simple QQQ trend rule (t 2.2 to 2.8).
- **Since 2013: 1% to 2% a year, not significant.** A simple QQQ trend rule explains the pod in the last 13 years.

## 5. In the book (60% TAA 3x / 40% NDX leg, the live weights, monthly rebalance)

| NDX leg | 2012-11 on (real TAA): CAGR / Sharpe / Max DD | 2008-03 on (TAA proxy before 2012-11): CAGR / Sharpe / Max DD |
|---|---|---|
| L live | 21.5% / 1.37 / −14.9% | 18.6% / 1.22 / −17.4% |
| NATR20 | 22.8% / 1.34 / −17.0% | 20.0% / 1.23 / −17.6% |
| E2 | 22.2% / 1.37 / −15.9% | 19.3% / 1.23 / −17.5% |
| E2 + sector cap | 21.4% / 1.35 / −15.9% | 18.8% / 1.22 / −17.5% |
| QQQ at L's exposure | 20.3% / 1.33 / −13.5% | 17.8% / 1.19 / −17.8% |
| TAA 3x alone (no NDX leg) | 24.6% / 1.36 / −17.7% | 21.3% / 1.19 / −26.4% |

- **The variant does not matter in the book.** P(book with E2 + cap > book with L) is 0.35 since 2012 and 0.57 since
  2008: a coin flip.
- **The NDX leg's job is drawdown, not Sharpe.** It takes the book's worst drawdown from −26.4% to −17.4% (2008 on).
  Since 2012 TAA alone has the same Sharpe as the book.
- **Monthly correlation with TAA 3x:** L 0.34 to 0.40, E2 + cap 0.44 to 0.48, QQQ at the same exposure 0.50 to 0.52.
- **The ceiling is structural:** TAA's TQQQ leg already carries the Nasdaq factor, so a better long-only Nasdaq sleeve
  cannot lift the book much. A non-Nasdaq third leg can (MR capsule thirds book: 1.39 against 1.21, 2008 on).

## 6. Size, capacity, execution

Whole shares on the 2023+ decisions (the live engine floors to whole shares): share of target positions below one
share / share of the intended exposure left in cash.

| Pod size | L live | E2 and E2 + sector cap |
|---|---|---|
| USD 12K | 1% / 6% | 16% / 22% |
| USD 25K | 0% / 3% | 3% / 10% |
| USD 50K | 0% / 1% | 2% / 6% |
| USD 100K | 0% / 1% | 1% / 3% |
| USD 250K | 0% / 0% | 0% / 1% |

- **Size rule:** the blend is workable from USD 50K (it then loses what L loses today at USD 12K) and clean from
  USD 100K.
- **Capacity at the opening auction** (house MOO model, pre-TCA, recent five years, orders up to 0.05% of ADV at the
  95th percentile): the dollar-ATR book to about USD 0.5M and the NATR20 book to about USD 1M.
- **Beyond that, trade at the next day's close:** the timing matrix shows a cost of 0.01 to 0.02 Sharpe
  (0.76 to 0.74 and 0.81 to 0.80), and the closing auction is far deeper.
- **The house capacity rule fails the pair on the recent five years for a different reason:** at 0% cash the two
  capped books earned 10.0% and 9.1% a year against 14.1% for the S&P 500 total return (2021-10 to 2026-10).

## 7. Caveats on record

| Caveat | Direction | Size |
|---|---|---|
| The design was chosen on 2000-2026 data after about 45 NDX trials | Optimistic | Plan on 0.75, not 0.80 to 0.87 |
| Month-end is a lucky rebalance day | Optimistic | Luck-band median is 0.04 to 0.09 below the month-end Sharpe |
| Sector labels are today's GICS applied to all history (G-025) | Optimistic, mild | Not measurable without point-in-time labels |
| The cap costs return in technology booms | Design | 2023-on CAGR 19.7% vs 24.9%; 2026 to date 15.0% vs 35.3% |
| L's dollar-ATR score favours low-priced shares | Design | Median pick price USD 68 vs USD 198 for NATR20 |
| The selection is not proven against QQQ at the same exposure | Unproven | P 0.54 (L) to 0.78 (E2 + cap) |
| Alpha since 2013 is not significant | Decay | 1% to 2% a year, t below 1, after QQQ and its 200-day rule |
| Idle cash earns 0% in the engine (mean invested 68%) | Conservative | About 0.03 Sharpe and 0.7 points of CAGR |
| No live route for E2 + cap | Status | Needs a single-account strategy and the WIRED checks |

## 8. Owner question (2026-10-04): "so is it better to simply buy QQQ?"

Script `ndx_momentum_qqq_question.py`, results `results/scout/robustness/ndx_momentum_qqq_question.json`.
Descriptive, on the series of the registered pack; no candidate was added.

Two different things are called QQQ here:

- **QQQ buy and hold:** always fully invested.
- **QQQ gated:** the registered control of section 3. QQQ is held only while the pod's gates are open, at the pod's
  own exposure (69% invested on average).

The leg alone:

| 2000-09 to 2026-10 | CAGR | Vol | Sharpe | Max DD | Year 2022 | CAGR 2000-09 to 2013-09 | CAGR 2013-10 to 2021-09 | CAGR 2021-10 on |
|---|---|---|---|---|---|---|---|---|
| QQQ buy and hold | 8.7% | 25.6% | 0.45 | −80.4% | −32.6% | −1.4% | 22.0% | 16.7% |
| QQQ gated | 10.4% | 13.8% | 0.79 | −30.1% | −26.0% | 8.2% | 15.5% | 8.3% |
| L live | 12.6% | 16.5% | 0.80 | −29.0% | −13.4% | 11.0% | 12.7% | 16.6% |
| Capsule (E2 + sector cap) | 13.6% | 16.1% | 0.87 | −21.5% | −17.0% | 13.6% | 14.6% | 11.9% |

Inside the live book (60% TAA 3x / 40% leg, monthly rebalance):

| Leg | 2012-11 on: CAGR / Sharpe / Max DD | Book's year 2022 | 2008-03 on (TAA proxy before 2012-11): CAGR / Sharpe / Max DD |
|---|---|---|---|
| L live | 21.5% / 1.37 / −14.9% | −4.2% | 18.6% / 1.22 / −17.4% |
| Capsule | 21.4% / 1.35 / −15.9% | −5.7% | 18.8% / 1.22 / −17.5% |
| QQQ gated | 20.3% / 1.33 / −13.5% | −9.8% | 17.8% / 1.19 / −17.8% |
| QQQ buy and hold | 23.3% / 1.37 / −17.8% | −12.7% | 20.3% / 1.21 / −24.3% |

- **QQQ buy and hold is not a substitute for the pod: it has no brake.**
  - Alone it lost 80% in 2000-2002.
  - In the book it gives the same Sharpe as L (P 0.48 to 0.50) and 1.7 points more CAGR, but a Max DD of −24.3%
    instead of −17.4%, and a 2022 three times worse.
  - The book test starts in 2008, so it does not contain the dot-com bear. There a 40% QQQ leg alone would have cost
    the book about 32 points.
- **QQQ gated is a legitimate simpler alternative, not a better one.**
  - Alone: the same Sharpe as L, with 2.2 points less CAGR.
  - In the book: 0.03 to 0.04 less Sharpe, 0.8 to 1.2 points less CAGR, and 2022 at −9.8% instead of −4.2%.
  - P(L book > gated QQQ book) is 0.64 to 0.70 (capsule: 0.64 to 0.69). That leans to the pod; it is not proof.
  - In the last five years it earned 8.3% a year against 16.6% for L.
- **The stock pod diversifies the book better,** because TAA 3x already holds levered QQQ (TQQQ): daily correlation
  with TAA is 0.49 to 0.51 for L and 0.57 to 0.59 for gated QQQ.
- **Where gated QQQ wins: friction and capacity.** It pays almost nothing at any size and has no capacity limit.
  A pod started on 2023-01-03 (whole shares, USD 1 minimum fee):

| Start size | L live: CAGR | Capsule: CAGR |
|---|---|---|
| USD 12K | 21.3% | 15.9% |
| USD 25K | 23.4% | 17.6% |
| USD 50K | 24.0% | 18.6% |
| USD 100K | 24.2% | 19.3% |
| USD 1M | 24.4% | 19.7% |

  - L at USD 12K lost 3.1 points a year to rounding and minimum fees (the fees are USD 497 over 3.75 years). The loss
    is this large because returns were high; the readiness audit measured 1.0 to 1.4 points over the long run.
  - Even so, L at USD 12K (21.3%) stayed ahead of gated QQQ over the same window (17.7%).
- **Decision: unchanged.** The pod stays. Gated QQQ is recorded as the fallback for an account too small for stocks
  and for capacity beyond the stock pod. Adopting it as a pod would be a new strategy, with its own registration and
  live build.

## 9. Files

- QQQ question: `scripts/research/scout_robustness_20261002/ndx_momentum_qqq_question.py`, results
  `results/scout/robustness/ndx_momentum_qqq_question.json`.
- Decision pack: `scripts/research/scout_robustness_20261002/ndx_momentum_decision.py`, tests
  `tests/test_ndx_momentum_decision_helpers.py`, results `results/scout/robustness/ndx_momentum_decision.json`.
- Earlier record: [Scout A15 robustness report](SCOUT_ROBUSTNESS_20261002.md) (rankings, filters, ensembles, cap),
  [NDX parameter robustness](NDX_PARAM_ROBUSTNESS_REPORT_20260926.md),
  [Russell 1000 liquid momentum](R1000_LIQUID_MOMENTUM_REPORT_20260926.md),
  [trend and breakout](TREND_BREAKOUT_REPORT_20260927.md).
- Strategies: `strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py` (live),
  `strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap.py`, `strategy_mo_natr20_ndx_vxn_scaled_sector_cap.py`;
  book `portfolios/ndx_e2_sector_cap_5050.yaml`.
