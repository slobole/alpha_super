# Scout A15: robustness diagnostics (contribution, convexity, ablation, random parameters)

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Source of the ideas:** a concept review of github.com/Lucas-Joly-GH/trends-research (a futures thesis): its tests
  T23 (progressive parsimony), T24 (random-parameter Monte Carlo), T30 (instrument leave-one-out) and its
  Henriksson-Merton / Treynor-Mazuy battery. The owner chose three of them on 2026-10-02 and asked for test 2 to be a
  distribution check ("no asset or stock made all the return") rather than a literal leave-one-out.
- **Code:** `alpha/scout/stations/robustness.py`; card section `alpha/scout/card.py` (`robustness_html`); per-asset P&L
  output `asset_pnl_df` in the weights engine; ablation switches in the TAA, NDX and CORE5 specs (defaults = the
  engine rule, so the identity gate is unchanged).
- **Plans:** `scripts/research/scout_robustness_20261002/plans.py`, committed (1b041a0) before the full run.
- **Runner:** `scripts/research/scout_robustness_20261002/run.py`. Output: `results/scout/robustness/<pod>/` and a new
  section on every re-audition card in `results/scout/cards/` (not in git).

**None of the four diagnostics gates or changes a grade** (D20: a test gates only after a calibration says it adds
something). Every Sharpe below is the S4 definition (zero risk-free rate, house parity costs), in sample (to the vault
seal 2022-12-30), with idle cash credited at the T-bill rate unless stated.

## Verdict

| Pod | Window | Where the return came from | Shape vs SPY | Components that earn their place | Live Sharpe vs random draws (median) |
|---|---|---|---|---|---|
| **TAA 3x** (LIVE) | 2012-11 to 2022 | CONCENTRATED: TQQQ 84%; without it 0.48 (from 1.22) | LINEAR (down 0.24, up 0.71) | fallback, rank weights | **1.22 vs 0.94** (top 1%) |
| **NDX VXN** (LIVE) | 2000-09 to 2022 | SPREAD: top stock 8%, top 10 47% | LINEAR | SPY regime filter | 0.73 vs 0.61 (top 15%) |
| NDX NATR20 VXN | 2000-09 to 2022 | SPREAD: top stock 8% | LINEAR | SPY regime filter | 0.84 vs 0.68 (top 6%) |
| **CORE5** | 2008-04 to 2022 | SPREAD: SPY 39%, DBC 27% | LINEAR, nearly convex (down 0.00, up 0.16, g t 1.9) | all four | **0.95 vs 0.80** (top 1%) |
| BTAL_QQQ (TAA linearity 1/N QQQ) | 2012-11 to 2022 | CONCENTRATED: QQQ 73%; without it 0.50 | LINEAR (down 0.10, up 0.32) | fallback only | 1.09 vs 0.95 (top 13%) |

## What this means

**1. No pod lives on one stock; the TAA pods live on one ETF, by design.**
- **Stock pods (all SPREAD).** Take out the top 1% of names:
  - DV2 (857 names, the best 3% of return): Sharpe 0.94 → 0.78.
  - HPI vote (755 names, 2%): 1.06 → 0.95.
  - NDX VXN (188 names, 8%): 0.69 → 0.63.
  - The best names are the expected ones (NDX: MU, NVDA, AMAT, AAPL); none carries the book.
- **TAA family (all six variants CONCENTRATED).** The fallback ETF is 73-93% of the summed return. Without it Sharpe
  falls to 0.34-0.50. This is the A10 finding seen from the other side: the pod is a timed leveraged Nasdaq position
  with a defensive shelf, not a rotation across five assets.
- **CORE5 is the only ETF pod spread across sleeves:** SPY 32 pp, DBC 27 pp, GLD 11 pp, IEF 8 pp of return. Without
  SPY it keeps 0.68.

**2. No pod is significantly convex; the defensive pods protect by a low down-beta, not by a convex payoff.**
- **No convexity.** Every pod reads LINEAR (Henriksson-Merton g t < 2 against SPY and QQQ).
- **The closest is CORE5 vs SPY:** down beta 0.00, up beta 0.16, g t 1.9, Treynor-Mazuy t 2.2.
- **Down months (pod vs SPY):**

| Pod | Pod | SPY |
|---|---|---|
| CORE5 | −0.16% | −4.27% |
| BTAL_QQQ | −0.44% | −3.91% |
| TFI | +0.16% | |
| EOM | +0.96% | |
| TAA 3x | −1.01% | |
| NDX VXN | −1.74% | |
| DV2 | −1.79% | |

**3. Ablation: what each component is worth (single drops; P = paired bootstrap probability that the live Sharpe is
higher).**
- **TAA 3x (live 1.22, CAGR 22.8%, Max DD −18%):**

| Without | Sharpe | CAGR | Max DD | P | Verdict |
|---|---|---|---|---|---|
| fallback | 0.50 | 3.4% | −12% | 0.99 | earns its place |
| rank weights | 1.13 | 27.2% | −26% | 0.82 | earns its place |
| VIX gate | 1.13 | | −21% | 0.79 | unclear |
| defensive assets | 1.14 | | | 0.76 | unclear |
| DTB3 hurdle | 1.21 | | | 0.85 | small |
| 3x leverage (QQQ instead) | | 10.2% | −10% | | |

  - Rank weights buy drawdown, not return.
  - The leverage row shows sizing: Sharpe barely moves, CAGR and drawdown do.
- **NDX VXN (live 0.73):**
  - **The dollar-ATR normalisation hurts:** ranking on 12-month ROC alone gives 0.82 (P 0.23). This is the share-price
    bias found in A10.
  - **The VXN scaling shows no evidence:** 0.73 either way; CAGR is 11.8% without it vs 10.7% with it.
  - **The SPY regime filter earns its place:** 0.73 → 0.61. Removed together with VXN, Max DD goes from −30% to −64%.
  - **The stock trend filter is small.**
- **NDX NATR20 VXN (live 0.84):**
  - The regime filter earns its place (0.84 → 0.71).
  - VXN scaling and the stock trend filter show no evidence (0.84 and 0.87 without).
  - NATR20 vs plain ROC is unclear (0.84 vs 0.82).
- **CORE5 (live 0.95, Max DD −7%):** every component earns its place.

| Without | Sharpe |
|---|---|
| DBC short | 0.87 |
| adaptive speed (at the same average speed) | 0.71 |
| price smoothing | 0.55 |
| the trend rule (static 20% sleeves) | 0.53 (Max DD −20%) |

  It is the cleanest rule of the five.
- **BTAL_QQQ (live 1.09):**
  - BTAL adds nothing measurable here: 1.09 with its slot in cash, P 0.53.
  - The VIX gate is unclear (1.02, P 0.75).
  - The QQQ fallback is the pod (0.56 without).

**4. Random parameters: no edge hinges on the values, but the live values are lucky picks.**
- **Every pod is ROBUST TO VALUES:**
  - All 1,000 draws (200 per pod) have a positive Sharpe.
  - Every median is at least 70% of the live Sharpe.
- **But the live configuration sits high:**

| Pod | Live | Median draw | Live rank |
|---|---|---|---|
| TAA 3x | 1.22 | 0.94 | top 1% |
| CORE5 | 0.95 | 0.80 | top 1% |
| NDX NATR20 VXN | 0.84 | 0.68 | top 6% |
| BTAL_QQQ | 1.09 | 0.95 | top 13% |
| NDX VXN | 0.73 | 0.61 | top 15% |

- **Planning number:** use the median draw, not the live value. Expect about 15-25% less Sharpe forward. The futures
  thesis read its own top-1.8% result as robustness; by its own docstring it means the chosen values matter.

## Decisions for the owner (research findings; no live change is made or implied)

1. **NDX ranking.** A10 and this ablation agree: the live dollar-ATR score is worse than both NATR20 and plain ROC.
   This strengthens the NATR20 shadow case.
2. **VXN scaling.** It shows no evidence in either NDX variant. It is a candidate for removal at the next NDX review,
   on a forward test, not on this in-sample result.
3. **BTAL inside BTAL_QQQ.** It adds nothing measurable in 2012-2022. Keep it only if its role is a crisis hedge
   beyond this window (2011 was its one big year, just before the window starts).
4. **Expectations.** Book and fund planning should use the random-draw medians above, not the live backtest Sharpe.

## Method notes and caveats

- **Contribution.**
  - Exact per-asset dollar P&L from the engine, divided by the previous close's value; the contributions of a session
    add up to its return (tested on every engine path).
  - The strip test removes the top assets' contributions and leaves their capital idle. It is first order (no
    re-compounding), and the removed assets are picked after the fact.
  - Contribution uses the engine's 0% cash.
- **Timing.**
  - Monthly excess returns, Newey-West lag 3.
  - Fewer than 24 months, or fewer than 6 down or 6 up months, gives INSUFFICIENT.
  - Daily-reset 3x funds are mechanically convex in their underlying, so CONVEX for a TAA variant would not prove timing
    skill. None reached it anyway.
- **Ablation.**
  - Switches are added to the specs. The defaults are the engine rule, and every switch reads only data at the
    decision date (quant review).
  - P ≥ 0.80 is a weak bar, and there are 21 single-drop tests here: expect one or two false "earns its place".
  - The 0.05 Sharpe materiality bar (SMALL) was added after the first TAA smoke run, where the DTB3 hurdle cost 0.02 at
    P 0.86. It is disclosed in the code.
- **Cash.**
  - Idle cash is credited at the T-bill rate, as a return-level credit, not re-invested.
  - The engine pays 0%, which made every cash-heavy ablation look worse (the owner's cash-convention rule).
  - The 0%-cash Sharpe is printed beside each row. It never changes a verdict here by more than 0.07.
- **Windows.**
  - Each window starts at the longest warm-up any random draw can need.
  - The first plan started later (TAA 2013, NDX 2002, CORE5 2009). That cut out 2000-01 and 2008, and the quant review
    caught it.
  - The windows were moved before the full run, but after TAA 3x, BTAL_QQQ and NDX VXN had run once. Their results
    barely changed.
- **Random boxes.**
  - Fixed in `plans.py` before the run and wider than the S4 grids.
  - The verdict depends on the box chosen.
  - No draw failed.
- **Other families.** The other 19 re-audited families got contribution and timing only, on the live configuration,
  from their first trade, with 0% cash.
- **Reviews.**
  - Quant-pitfalls: windows, zero-filled QQQ months, CORE5 average speed, BTAL isolation, leverage as sizing, cash
    convention. All fixed before the full run.
  - Coverage: one-asset and no-trade books, timing with no down months, all draws failed, card rendering, ledger
    identity per asset. All fixed and tested.
  - Parity: see the A15 amendment.

## Follow-up (2026-10-02): NDX ranking, plateaus and trend filters

Scripts: `ndx_ranking_compare.py` and `ndx_plateau_filter.py` in the same folder. Results: `results/scout/robustness/ndx_*.json`.
The plateau and filter study was registered in the ledger before any number was computed
(`ndx_natr20_vxn_plateau_filter_20261002`, 77 configurations, a sensitivity map; nothing is promoted from it).

**Rankings (live NDX VXN rule otherwise; in sample 2000-09 to 2022).**

| Ranking | Sharpe | CAGR | Max DD |
|---|---|---|---|
| Dollar ATR (live) | 0.73 | 10.7% | −29.0% |
| NATR20 | 0.84 | 13.7% | −23.5% |
| ROC only | 0.82 | 14.7% | −29.0% |

- **NATR20 wins robustly in sample:**
  - It beats dollar ATR on 83% of 200 identical random parameter sets, and ROC alone on 88%.
  - It is higher in every rebalance offset.
- **But dollar ATR fell less in the fast crashes:**

| Crash | Dollar ATR | NATR20 |
|---|---|---|
| Q4 2018 | −3.3% | −9.5% |
| 2022 | −11.4% | −17.1% |
| 2025 | −13.1% | −19.2% |

- **Dollar ATR also led from 2023 on** (seen period, Sharpe 1.14 vs 0.95).
- **Month-end luck:** in all three rankings, the month-end decision day sits near the best of 16 offsets.
- **Recommendation:** keep NATR20 in shadow beside the live pod; do not switch on in-sample evidence alone.

**Plateaus (NATR20 VXN).**
- **The ROC x filter surface is flat:**
  - Plateau at ROC 12 / SMA150 (neighbourhood median 0.87, peak 0.91, ratio 0.96).
  - ROC 3 is the only weak row (0.67-0.72).
  - Short filters (SMA20/50) are the weak columns.
- **ROC x top count:** plateau at ROC 9 / top 15 (median 0.85, ratio 0.94).
- **The live ROC 12 / SMA100 / top 10** is inside the plateau, not on a peak.

**Trend-filter variants (17; ROC 12, top 10).**
- They range from 0.75 (Close more than 10% above SMA100) to 0.93 (Close more than 10% above SMA200). The live filter
  is at 0.84 and no filter at all gives 0.87.
- **Longer or looser filters land at 0.86-0.93:** SMA150-250, SMA21 > SMA100, SMA50 > SMA100, the distance from SMA200.
- **Short filters land at 0.81-0.83:** SMA20/50, 10/50, 21/50.
- **Not significant after the search:** the top three are best-of-17 picks, and their paired P of 0.98-0.99 is not
  corrected for that search. The differences (about +0.05 to +0.09) are within the noise a 17-variant search
  produces.
- **What the evidence supports:** the filter matters little, and short filters hurt. It does not support a specific
  replacement.
