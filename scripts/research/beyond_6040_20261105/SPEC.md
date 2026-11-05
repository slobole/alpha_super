# Beyond 60/40 — frozen research spec (2026-11-05)

Research-only. Nothing here authorises trading, capital or a LIVE route.

## Question

Is the inverse-volatility stocks / long-bond / gold portfolio with an 8% volatility
target (Beyond Passive, "Beyond 60/40") robust enough to become a house strategy,
and which changes make it *more* robust rather than merely better in-sample?

## Data

| Series | Source | Span |
|---|---|---|
| Stocks | French US market (to 1987) → $SPXTR (to 2004-11) → SPY TR | 1968-01 → |
| Long bond | Synthetic 25y par Treasury off DGS20/DGS30 (to 2004-11) → TLT TR | 1968-01 → |
| Gold | Frankfurt fixing (to 1982) → XAUUSD (to 2004-11) → GLD TR | 1968-01 → |
| 10y bond | Synthetic 10y par Treasury off DGS10 → IEF TR (2002-07) | 1968-01 → |
| World ex-US stocks | French Developed ex-US market (USD) → EFA TR (2001-08) | 1990-07 → |
| Commodities | DBC TR | 2006-02 → |
| TIPS | TIP TR | 2003-12 → |
| Cash | Fed funds (DFF); T-bills (DTB3); 0% convention | 1968-01 → |
| CPI | CPIAUCSL (monthly, lagged one month for availability) | 1968-01 → |
| USD/ILS | Norgate forex spot | 1991-01 → |
| Israel short rate | OECD immediate rate (FRED IRSTCI01ILM156N, monthly) | 1992-12 → |

Panel reused from `pakal/pakal-research/reports/beyond_passive_poor_mans_trend_study/data/panel.parquet`
(already audited: on-or-before alignment, no backward fill). Extras added here with the same rule.

## Vault

All development uses sessions **before 2023-01-01**. The vault (2023-01-01 → latest) is
opened once, for the final chosen variant and the fixed benchmarks only (Gate G6).

## Timing and costs

- Signals from closes through day t-1; position held through day t (one-session lag,
  equivalent to our Close_T decision → next-session execution). Replication check also at lag 2
  (the author's convention).
- Cost: 5 bp per unit of risky-asset weight traded (central); 0 and 10 bp as sensitivity.
- Idle cash earns Fed funds (central). 0% and T-bill conventions reported beside every headline.
- Borrowing (only in leverage arm): Fed funds + 1.5%.

## Baseline B0 (the author's rule)

Monthly (last session) inverse-63-day-vol weights across stocks / long bond / gold.
Daily overlay: exposure = 1 if 63-day realised portfolio vol ≤ 8.5%, else min(1, 8% / vol).
No leverage, no short, no band.

Benchmarks: 60/40 (stocks/long bond, monthly), equal-thirds (monthly), stocks only, cash.

## Gates to become a strategy (pre-vault data 1970-01 → 2022-12)

- **G1 Return**: excess Sharpe (over Fed funds) ≥ 60/40 + 0.15 on full history, and ≥ 60/40 in
  at least 5 of the 6 blocks {1970s, 1980s, 1990s, 2000s, 2010s, 2005-01→2022-12 (ETF era)};
  never below 60/40 by more than 0.10 in any block.
- **G2 Drawdown**: max DD ≤ 2/3 of 60/40's, full history and ETF era.
- **G3 Plateau**: the chosen parameter cell is within 0.05 excess Sharpe of the median of its
  neighbourhood (±1 step in every dimension) and ≥ 80% of the whole grid beats 60/40 on excess
  Sharpe. The chosen cell is the plateau centre, never the argmax.
- **G4 Walk-forward**: parameters re-chosen every 5 years from the expanding past only;
  walk-forward excess Sharpe ≥ fixed-parameter excess Sharpe − 0.10.
- **G5 Cash convention**: G1 and G2 re-evaluated with 0% idle cash; a failure there is reported
  as a qualifier, not a veto (the long history has a 5% average cash rate that will not repeat).
- **G6 Vault**: on 2023-01 → latest the realised excess return lies inside the 5–95% band of a
  63-day stationary block bootstrap of the pre-vault variant, and realised max DD ≤ 1.25 × the
  pre-vault worst-decade max DD.

## Adoption rule for every change to B0

A change is adopted only if, written down before the run, it has an economic reason, and:
(a) worst-block excess Sharpe improves; (b) full-history excess Sharpe falls by at most 0.05;
(c) the 90% block-bootstrap interval of the Sharpe difference has lower bound > −0.05;
(d) turnover does not more than double.
Insurance-type changes (trend tilt, exposure cap) may fail (a) if they cut max DD by ≥ 25%
with a premium ≤ 0.5% CAGR per year.

## Arms

| Arm | Hypothesis written before the run |
|---|---|
| A1 Plateau | vol window {21,42,63,126,252} × target {6,7,8,9,10}% × trigger buffer {0,0.5,1.0}pp × rebalance {W,M,Q} × band {0,5%}. Expect a broad plateau; 63/8/0.5/M is near its centre. |
| A2 Weighting | IVOL vs equal vs ERC vs MinVar vs IVOL/EW blend. Expect near-identical Sharpe; MinVar concentrates. |
| A3 Portfolio-vol estimator | rolling 63 vs EWMA(0.94) vs max(21,63) vs 126. Expect no winner beyond noise (PAKAL found the same). |
| A4 Trend tilt | per-asset ensemble trend (21/63/126/252) → multiplier 0.5+0.5s; half tilt; 10-month SMA on/off. Expect insurance: pays in positive-correlation + falling-stock regimes, costs ≤ 0.5%/yr elsewhere. |
| A5 Equity leg | US vs 50/50 US/developed-ex-US vs VT. Expect small Sharpe change, lower single-country risk; USD investor sees no gain in-sample. |
| A6 Bond leg | 25y vs 10y vs 50/50. Expect 10y lower vol and higher IVOL weight; similar Sharpe. |
| A7 Inflation leg | gold vs gold/commodities 50/50 vs gold/TIPS (ETF era). Expect commodities worse Sharpe, slightly better 2022. |
| A8 Leverage | exposure cap 1.0 / 1.25 / 1.5 with financing, with and without 1.5x total / 0.7 per-asset cap. For the Growth shelf only. Expect Sharpe to fall with leverage after financing. |
| A9 Cash convention | 0% / Fed funds / T-bills side table. |
| A10 ILS investor | NAV in ILS unhedged vs fully hedged (carry = IL rate − FF) vs active trend hedge. 1993 →. Expect unhedged DD materially deeper. |
| A11 Regimes | blocks by decade, stock-bond correlation sign, inflation rising/falling (CPI vs trailing 10y). |
| A12 Book fit | correlation and marginal Sharpe against CORE5, TAA 1N, MR capsule, Flow Vote; replaces or joins? One pod per engine. |
| A13 Statistics | 63-day stationary block bootstrap (2,000 draws), walk-forward, worst 12m / 36m, time under water, CVaR 5% daily and monthly. |

## Reporting

Stats in English; short Hebrew explanations. Every headline shows Fed-funds cash and 0% cash.
Sharpe is excess over Fed funds unless marked "raw" (the author's figures are raw).
